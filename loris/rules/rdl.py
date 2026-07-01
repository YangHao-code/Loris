"""RDL — Rule for Document Labeling — and the RDLSet container.

A rule has the form ``X -> p0``: a conjunction of body predicates ``X`` and a
consequence label ``p0`` with an op (``add`` / ``remove`` / ``replace``). An
:class:`RDLSet` is the ordered collection produced by rule discovery and
consumed by Chase inference; it knows how to predict, serialize, and persist.

Extracted from ``rule_discovery/loris_rule_discovery.py`` (Phase 3) — the
data structures only, independent of any particular learner. The old
``RuleLearner`` is intentionally left behind (to be deleted with its pipelines).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import f1_score

from loris.document import Document
from loris.predicates import (
    Predicate,
    TextualPredicate,
    MatchPredicate,
    FreqPredicate,
    CooccurPredicate,
    BeforePredicate,
    MLPredicate,
    MLThresholdPredicate,
    LabelPredicate,
    predicate_to_dict,
    predicate_from_dict,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RDL:
    """
    Rule for Document Labeling: X → p0

    X = body 中所有谓词的合取式（conjunction），当文档满足所有前提时规则触发。
    p0 = consequence 目标标签。

    Parameters
    ----------
    body : Tuple[Predicate, ...]
        合取式谓词集（规则前提 X）。
    consequence : str
        结论标签 p0。
    score : float
        该规则在验证集上的 F1 增益。
    coverage : float
        该规则在验证集上的覆盖率（触发比例）。
    trial_number : int
        产生该规则的 Optuna trial 编号。
    """

    body: Tuple[Predicate, ...]
    consequence: str
    consequence_op: str = "add"  # "add" | "remove" | "replace"
    score: float = 0.0
    coverage: float = 0.0
    trial_number: int = -1
    val_stats: Dict[str, Any] = field(default_factory=dict)

    def fires(self, doc: Document) -> bool:
        """判断规则是否在文档上触发（所有前提谓词均满足）。"""
        if not self.body:
            return True
        # 使用代理文档副本，避免 LabelPredicate(op="minus") 副作用污染
        # 原始 Document 是 frozen dataclass，不能直接赋值 doc.lbl
        proxy = Document(
            cnt=doc.cnt,
            lbl=set(doc.lbl) if doc.lbl else set(),
            mtd=doc.mtd,
            ttl=doc.ttl,
        )
        return all(p(proxy) for p in self.body)

    def __repr__(self) -> str:
        body_str = " ∧ ".join(repr(p) for p in self.body) if self.body else "⊤"
        op_sym = {"add": "+", "remove": "-", "replace": "=", "equal": "=="}.get(self.consequence_op, "+")
        return (
            f"RDL({body_str} → {op_sym}{self.consequence!r}, "
            f"score={self.score:.4f}, coverage={self.coverage:.4f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 安全的字典。"""
        return {
            "body": [predicate_to_dict(p) for p in self.body],
            "consequence": self.consequence,
            "consequence_op": self.consequence_op,
            "score": self.score,
            "coverage": self.coverage,
            "trial_number": self.trial_number,
            "val_stats": self.val_stats,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RDL":
        """从字典反序列化。"""
        body = tuple(predicate_from_dict(pd) for pd in d["body"])
        return cls(
            body=body,
            consequence=d["consequence"],
            consequence_op=d.get("consequence_op", "add"),
            score=d.get("score", 0.0),
            coverage=d.get("coverage", 0.0),
            trial_number=d.get("trial_number", -1),
            val_stats=d.get("val_stats", {}),
        )


# ===========================================================================
# RDLSet — 规则集合容器
# ===========================================================================

class RDLSet:
    """
    Collection of discovered RDLs for multi-label document classification.

    多标签语义：一个文档可以被多条规则触发，每条规则分配一个标签，
    最终取所有被触发规则的标签并集作为该文档的多标签预测结果。

    Parameters
    ----------
    rules : List[RDL]
        发现的规则列表。
    label_names : List[str]
        标签名称列表（决定输出矩阵的列顺序）。
    """

    def __init__(self, rules: List[RDL], label_names: List[str]) -> None:
        self.rules = list(rules)
        self.label_names = list(label_names)
        self._label2idx = {name: i for i, name in enumerate(self.label_names)}

    # [CORE LOGIC] 双集合 Chase 语义：ADD→pos, REMOVE→neg, 返回 pos & ~neg
    def predict(self, docs: List[Document], propagate_labels: bool = False) -> np.ndarray:
        """
        Apply all discovered rules to produce multi-hot predictions.

        Uses dual-set semantics (Church-Rosser): ADD rules write to pos set,
        REMOVE rules write to neg set. Final prediction = pos & ~neg.
        Rule application order does not affect results.
        """
        n_labels = len(self.label_names)
        pos = np.zeros((len(docs), n_labels), dtype=bool)
        neg = np.zeros((len(docs), n_labels), dtype=bool)

        for rule in self.rules:
            label_idx = self._label2idx.get(rule.consequence)
            if label_idx is None:
                continue
            # B7 fix: only add/remove are valid per-doc fast-path ops. equal /
            # replace / subset are cross-document consequences that must be applied
            # by the chase; skip them here rather than mis-applying them as "add".
            if rule.consequence_op not in ("add", "remove"):
                continue
            is_remove = rule.consequence_op == "remove"
            for i, doc in enumerate(docs):
                if rule.fires(doc):
                    if is_remove:
                        neg[i, label_idx] = True
                        if propagate_labels:
                            doc.lbl.discard(rule.consequence)
                    else:
                        pos[i, label_idx] = True
                        if propagate_labels:
                            doc.lbl.add(rule.consequence)

        return (pos & ~neg).astype(np.float32)

    def predict_on_base(
        self,
        docs: List[Document],
        base_predictions: np.ndarray,
        propagate_labels: bool = False,
    ) -> np.ndarray:
        """Apply rules on top of base predictions using dual-set semantics."""
        n_labels = len(self.label_names)
        pos = (np.asarray(base_predictions, dtype=np.float32) > 0)
        neg = np.zeros((len(docs), n_labels), dtype=bool)
        for rule in self.rules:
            label_idx = self._label2idx.get(rule.consequence)
            if label_idx is None:
                continue
            # B7 fix: only add/remove on the per-doc fast path (see predict()).
            if rule.consequence_op not in ("add", "remove"):
                continue
            is_remove = rule.consequence_op == "remove"
            for i, doc in enumerate(docs):
                if rule.fires(doc):
                    if is_remove:
                        neg[i, label_idx] = True
                        if propagate_labels:
                            doc.lbl.discard(rule.consequence)
                    else:
                        pos[i, label_idx] = True
                        if propagate_labels:
                            doc.lbl.add(rule.consequence)
        return (pos & ~neg).astype(np.float32)

    def chase_predict(
        self,
        docs: List[Document],
        base_predictions: Optional[np.ndarray] = None,
        max_rounds: int = 100,
        time_limit_sec: Optional[float] = None,
        conflict_mode: str = "negative_wins",
        enable_transitivity: bool = False,
        sim_graphs: Optional[Dict] = None,
        sim_decay: float = 1.0,
        sim_conf_threshold: float = 0.0,
        virtual_attrs: Optional[Dict] = None,
    ) -> "ChaseResult":
        """Apply rules using multi-label chase semantics.

        Instead of a single-pass rule application, this method iteratively
        applies rules until a fixpoint is reached, with formal conflict
        detection and optional transitivity propagation.

        Parameters
        ----------
        docs : List[Document]
            Documents to classify.
        base_predictions : np.ndarray, optional
            (n_docs, n_labels) base model predictions to seed from.
        max_rounds : int
            Maximum chase iterations.
        time_limit_sec : float or None
            Wall-clock time limit.
        conflict_mode : str
            ``"halt"`` | ``"negative_wins"`` | ``"positive_wins"``.
        enable_transitivity : bool
            Whether to propagate labels via subset relations. Defaults to
            ``False`` since B-5 (the prediction-bitmap ``sub``/``sup`` source was
            the B1 bug and is removed; a legitimate co-membership source is added
            in C-9).
        sim_graphs : dict, optional
            {threshold: csr_matrix} for SimPredicate evaluation.

        Returns
        -------
        ChaseResult
        """
        from chase_inference.multi_chase import MultiChase
        chase = MultiChase.from_rdlset(
            self,
            max_rounds=max_rounds,
            time_limit_sec=time_limit_sec,
            conflict_mode=conflict_mode,
            enable_transitivity=enable_transitivity,
            sim_graphs=sim_graphs,
            sim_decay=sim_decay,
            sim_conf_threshold=sim_conf_threshold,
            virtual_attrs=virtual_attrs,
        )
        return chase.run(docs, base_predictions=base_predictions)

    def evaluate_on_base(
        self,
        docs: List[Document],
        y_true: np.ndarray,
        base_predictions: np.ndarray,
        propagate_labels: bool = False,
    ) -> Dict[str, float]:
        """Evaluate model+rules combined predictions."""
        combined = self.predict_on_base(docs, base_predictions, propagate_labels=propagate_labels)
        micro_f1 = float(f1_score(y_true, combined, average="micro", zero_division=0))
        macro_f1 = float(f1_score(y_true, combined, average="macro", zero_division=0))
        return {"micro_f1": micro_f1, "macro_f1": macro_f1}

    def evaluate(
        self,
        docs: List[Document],
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """在文档集上评估规则集的分类效果。"""
        y_pred = self.predict(docs)
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return {"micro_f1": micro_f1, "macro_f1": macro_f1}

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 安全的字典。"""
        return {
            "label_names": self.label_names,
            "n_rules": len(self.rules),
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RDLSet":
        """从字典反序列化。"""
        rules = [RDL.from_dict(rd) for rd in d["rules"]]
        return cls(rules=rules, label_names=d["label_names"])

    def save(self, path: str) -> None:
        """持久化到 JSON 文件。"""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("RDLSet saved %d rules to %s.", len(self.rules), dest)

    @classmethod
    def load(cls, path: str) -> "RDLSet":
        """从 JSON 文件加载。"""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def __len__(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return (
            f"RDLSet(n_rules={len(self.rules)}, "
            f"labels={self.label_names})"
        )


# ===========================================================================
# 谓词重建辅助函数
# ===========================================================================



def _is_redundant(candidate: RDL, existing_rules: List[RDL]) -> bool:
    """
    检查候选规则是否与已有规则冗余。
    冗余定义：相同后件标签 + body 谓词集合重叠超过 80%。
    """
    candidate_body_set = set(candidate.body)
    if not candidate_body_set:
        # 空规则体与任何已有规则不冗余（但应避免多条空规则）
        return any(not r.body and r.consequence == candidate.consequence for r in existing_rules)

    for rule in existing_rules:
        if rule.consequence != candidate.consequence:
            continue
        # An ADD and a REMOVE (or replace) rule on the same label have opposite
        # effect — they are NOT redundant even with high body overlap.
        if rule.consequence_op != candidate.consequence_op:
            continue
        rule_body_set = set(rule.body)
        if not rule_body_set:
            continue
        union_size = len(candidate_body_set | rule_body_set)
        intersection_size = len(candidate_body_set & rule_body_set)
        if union_size > 0 and intersection_size / union_size > 0.8:
            return True
    return False


# ===========================================================================
# RuleLearner — 元学习规则发现主类
# ===========================================================================

