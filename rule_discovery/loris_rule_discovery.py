"""
rule_discovery/loris_rule_discovery.py
--------------------------------------
Accuracy-Guided Rule Discovery via Meta-learning (基于元学习的规则发现)

本模块实现了 LORIS 框架中最核心的规则搜索引擎。给定一个文档簇（Cluster）、
提取好的候选文本模式（Patterns）和候选 ML 模型集合，通过贝叶斯优化（Optuna）
搜索出能最大化 Validation F1 Score 的逻辑规则集合 Σ。

规则形式：X → p0
    - X = 合取式谓词集（规则前提）
    - p0 = 结论标签（规则应当分配的目标标签）

核心思路沿用 OHunt 参考代码中的 Optuna + JournalStorage 搜索模式：
    1. 定义配置空间（Configuration Space）
    2. 构建 Surrogate Model（TPE 采样器）
    3. 通过 Acquisition Function 迭代寻找最优参数

Design notes
------------
* 搜索空间 Θ 包含三类参数：结构选择（二值开关）、谓词参数微调、标签联合搜索
* 采用贪心序列覆盖策略：每轮发现一条最优规则，更新预测后再搜索下一条
* 所有规则由 frozen dataclass RDL 表示，支持 JSON 序列化
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import f1_score

import optuna

# 抑制 Optuna 的冗余日志输出
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pattern_extraction.document import Document
from pattern_extraction.predicates import (
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
    _get_ml_model,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# RDL — Rule for Document Labeling（文档标注规则）
# ===========================================================================

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
        op_sym = {"add": "+", "remove": "-", "replace": "="}.get(self.consequence_op, "+")
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
            for i, doc in enumerate(docs):
                if rule.fires(doc):
                    if rule.consequence_op == "remove":
                        neg[i, label_idx] = True
                    else:  # "add"
                        pos[i, label_idx] = True
                    if propagate_labels:
                        if rule.consequence_op == "add":
                            doc.lbl.add(rule.consequence)
                        elif rule.consequence_op == "remove":
                            doc.lbl.discard(rule.consequence)

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
            for i, doc in enumerate(docs):
                if rule.fires(doc):
                    if rule.consequence_op == "remove":
                        neg[i, label_idx] = True
                    else:  # "add"
                        pos[i, label_idx] = True
                    if propagate_labels:
                        if rule.consequence_op == "add":
                            doc.lbl.add(rule.consequence)
                        elif rule.consequence_op == "remove":
                            doc.lbl.discard(rule.consequence)
        return (pos & ~neg).astype(np.float32)

    def chase_predict(
        self,
        docs: List[Document],
        base_predictions: Optional[np.ndarray] = None,
        max_rounds: int = 100,
        time_limit_sec: Optional[float] = None,
        conflict_mode: str = "halt",
        enable_transitivity: bool = True,
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
            Whether to propagate labels via subset relations.
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

def _reconstruct_predicate(
    trial: optuna.Trial,
    idx: int,
    original: Predicate,
) -> Predicate:
    """
    # [CORE LOGIC] 谓词参数微调：根据 Optuna 建议的超参数重构谓词对象
    对于被选中的谓词，搜索其最佳连续/离散参数，创建新的冻结实例。
    因为所有谓词都是 frozen dataclass，不能原地修改，必须重建。

    Parameters
    ----------
    trial : optuna.Trial
        当前 Optuna trial 对象。
    idx : int
        谓词在候选列表中的索引。
    original : Predicate
        原始谓词对象。

    Returns
    -------
    Predicate
        参数微调后的新谓词实例。
    """
    # FreqPredicate：微调频次阈值 η 和比较算子 op
    if isinstance(original, FreqPredicate):
        eta = trial.suggest_int(f"eta_{idx}", low=1, high=20)
        op = trial.suggest_categorical(f"op_{idx}", [">=", "<=", "=="]) # , ">", "<"
        # 如果启用了语义匹配，额外微调 threshold
        if original.sim:
            threshold = trial.suggest_float(f"threshold_{idx}", low=0.3, high=0.95)
        else:
            threshold = original.threshold
        return FreqPredicate(
            attr=original.attr, r=original.r,
            op=op, eta=float(eta),
            sim=original.sim, threshold=threshold,
        )

    # MatchPredicate：如果启用语义匹配则微调 threshold
    if isinstance(original, MatchPredicate) and original.sim:
        threshold = trial.suggest_float(f"threshold_{idx}", low=0.3, high=0.95)
        return MatchPredicate(
            attr=original.attr, r=original.r,
            sim=original.sim, threshold=threshold,
        )

    # CooccurPredicate：如果启用语义匹配则微调 threshold
    if isinstance(original, CooccurPredicate) and original.sim:
        threshold = trial.suggest_float(f"threshold_{idx}", low=0.3, high=0.95)
        return CooccurPredicate(
            attr=original.attr, r1=original.r1, r2=original.r2,
            sim=original.sim, threshold=threshold,
        )

    # BeforePredicate：如果启用语义匹配则微调 threshold
    if isinstance(original, BeforePredicate) and original.sim:
        threshold = trial.suggest_float(f"threshold_{idx}", low=0.3, high=0.95)
        return BeforePredicate(
            attr=original.attr, r1=original.r1, r2=original.r2,
            sim=original.sim, threshold=threshold,
        )

    # 其他谓词类型不需要参数微调，原样返回
    return original


# ===========================================================================
# Singleton scoring for predicate shortlisting
# ===========================================================================

def _score_singletons(
    candidate_predicates: List[Predicate],
    label_list: List[str],
    val_docs: List[Document],
    val_labels: np.ndarray,
    existing_predictions: np.ndarray,
    baseline_f1: float,
    top_per_type: int = 5,
    metric_mode: str = "global_macro",
    cluster_label_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, List[int]], np.ndarray, np.ndarray]:
    """
    Per-type singleton scoring: return {type_name: [top-K indices]} for each
    predicate type.  Each predicate is scored by its best singleton F1-gain
    across all labels, then the top-K per type are kept.

    Returns
    -------
    type_groups : Dict[str, List[int]]
    fire_masks : np.ndarray, shape (n_candidates, n_val), dtype bool
        Precomputed predicate fire masks for caching.
    pred_label_phi : np.ndarray, shape (n_candidates, n_labels), dtype float32
        Matthews correlation between each predicate and each label.
    """
    n_cands = len(candidate_predicates)
    n_val = len(val_docs)
    n_labels = len(label_list)

    # Group predicates by type
    type_map: Dict[str, List[int]] = {}
    for i, pred in enumerate(candidate_predicates):
        tname = type(pred).__name__
        type_map.setdefault(tname, []).append(i)

    if n_cands == 0:
        return type_map, np.zeros((0, n_val), dtype=bool), np.zeros((0, n_labels), dtype=np.float32)

    # Pre-compute fire masks for all predicates
    fire_masks = np.zeros((n_cands, n_val), dtype=bool)
    for i, pred in enumerate(candidate_predicates):
        for j, doc in enumerate(val_docs):
            try:
                fire_masks[i, j] = bool(pred(doc))
            except Exception:
                pass

    # Vectorised phi computation
    pred_label_phi = _compute_pred_label_phi(fire_masks, val_labels)

    # Score each predicate: best singleton F1-gain across all labels × all ops
    scores = np.full(n_cands, -np.inf)
    for i in range(n_cands):
        fires = fire_masks[i]
        if fires.sum() == 0:
            scores[i] = -1.0
            continue
        best_gain = -1.0
        for l_idx in range(n_labels):
            for op in ("add", "remove"):
                new_preds = existing_predictions.copy()
                if op == "add":
                    new_preds[fires, l_idx] = 1.0
                elif op == "remove":
                    new_preds[fires, l_idx] = 0.0
                if metric_mode == "cluster_local" and cluster_label_indices is not None:
                    all_new = f1_score(val_labels, new_preds, average=None, zero_division=0)
                    all_old = f1_score(val_labels, existing_predictions, average=None, zero_division=0)
                    cluster_g = float(all_new[cluster_label_indices].mean() - all_old[cluster_label_indices].mean())
                    global_g = float(f1_score(val_labels, new_preds, average="macro", zero_division=0)) - baseline_f1
                    gain = 0.9 * cluster_g + 0.1 * global_g
                else:
                    gain = float(f1_score(val_labels, new_preds, average="macro", zero_division=0)) - baseline_f1
                if gain > best_gain:
                    best_gain = gain
        scores[i] = best_gain

    # Per-type top-K selection
    result: Dict[str, List[int]] = {}
    for tname, indices in type_map.items():
        ranked = sorted(indices, key=lambda i: -scores[i])
        top = [i for i in ranked[:top_per_type] if scores[i] > -1.0]
        if top:
            result[tname] = top

    return result, fire_masks, pred_label_phi


def _compute_pred_label_phi(
    fire_masks: np.ndarray,
    val_labels: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Matthews phi correlation between predicates and labels.

    Parameters
    ----------
    fire_masks : (n_preds, n_docs) bool
    val_labels : (n_docs, n_labels) float32

    Returns
    -------
    phi : (n_preds, n_labels) float32
    """
    F = fire_masks.astype(np.float32)   # (n_preds, n_docs)
    L = val_labels.astype(np.float32)   # (n_docs, n_labels)
    n = F.shape[1]

    tp = F @ L                          # (n_preds, n_labels)
    fp = F @ (1 - L)
    label_sum = L.sum(axis=0, keepdims=True)  # (1, n_labels)
    fn = label_sum - tp
    tn = n - tp - fp - fn

    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, num / den, 0.0).astype(np.float32)


# ===========================================================================
# 目标函数（评估单条规则配置）
# ===========================================================================

def evaluate_configuration(
    trial: optuna.Trial,
    candidate_predicates: List[Predicate],
    candidate_ml_models: List[str],
    label_list: List[str],
    val_docs: List[Document],
    val_labels: np.ndarray,
    baseline_f1: float,
    existing_predictions: np.ndarray,
    min_coverage: float = 0.05,
    type_groups: Optional[Dict[str, List[int]]] = None,
    rule_min_precision: float = 0.0,
    metric_mode: str = "global_macro",
    cluster_label_indices: Optional[List[int]] = None,
    precomputed_fire_masks: Optional[np.ndarray] = None,
    pred_label_phi: Optional[np.ndarray] = None,
    ml_proba_cache: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """
    OHunt-style objective: per-type binary switches + within-type categorical.

    Parameters
    ----------
    type_groups : Dict[str, List[int]]
        {predicate_type_name: [indices into candidate_predicates]},
        produced by _score_singletons().
    precomputed_fire_masks : np.ndarray, optional
        Shape (n_candidates, n_val) bool.  If provided, textual predicate fire
        results are looked up instead of re-evaluated per trial.
    pred_label_phi : np.ndarray, optional
        Shape (n_candidates, n_labels).  If provided, used to guide
        consequence label selection.
    ml_proba_cache : Dict[str, np.ndarray], optional
        {model_name: proba_matrix} where proba_matrix is (n_val, n_labels).
        Pre-computed model probabilities for threshold-based ML predicates.
    """
    try:
        body: List[Predicate] = []
        body_textual_indices: List[int] = []  # indices into candidate_predicates

        # ================================================================
        # 0. Body size budget — TPE learns to prefer small conjunctions
        # ================================================================
        max_body_size = trial.suggest_int("max_body_size", 1, 3)

        # ================================================================
        # 1. 文本谓词：每种类型一个二值开关 + 类内 categorical 选择
        # ================================================================
        if type_groups:
            for type_name, type_indices in sorted(type_groups.items()):
                if len(body) >= max_body_size:
                    break
                need = trial.suggest_int(f"need_{type_name}", low=0, high=1)
                if need == 1:
                    idx = trial.suggest_categorical(f"{type_name}_pred", type_indices)
                    tuned_pred = _reconstruct_predicate(trial, idx, candidate_predicates[idx])
                    body.append(tuned_pred)
                    body_textual_indices.append(idx)

            # Optional second MatchPredicate for conjunction rules
            match_indices = type_groups.get("MatchPredicate", [])
            if len(match_indices) >= 2 and len(body) < max_body_size:
                need2 = trial.suggest_int("need_MatchPredicate_2", low=0, high=1)
                if need2 == 1:
                    idx2 = trial.suggest_categorical("MatchPredicate_pred_2", match_indices)
                    tuned2 = _reconstruct_predicate(trial, idx2 + 10000, candidate_predicates[idx2])
                    if tuned2 not in body:
                        body.append(tuned2)
                        body_textual_indices.append(idx2)

        # ================================================================
        # 2. ML 谓词（阈值模式）
        # ================================================================
        ml_body_info: List[Tuple[str, str, float]] = []  # (model_name, label, threshold)
        for j, model_name in enumerate(candidate_ml_models):
            if len(body) >= max_body_size:
                break
            need_ml = trial.suggest_int(f"need_ml_{j}", low=0, high=1)
            if need_ml == 1:
                ml_label = trial.suggest_categorical(f"ml_label_{j}", label_list)
                ml_thresh = trial.suggest_float(
                    f"ml_thresh_{j}", low=0.2, high=0.8, step=0.1
                )
                body.append(MLThresholdPredicate(
                    model_name=model_name, label=ml_label, threshold=ml_thresh,
                ))
                ml_body_info.append((model_name, ml_label, ml_thresh))

        # ================================================================
        # 3. Label 谓词
        # ================================================================
        if len(body) < max_body_size:
            need_label_contains = trial.suggest_int("need_label_contains", low=0, high=1)
            if need_label_contains == 1:
                label_contains_tau = trial.suggest_categorical("label_contains_tau", label_list)
                body.append(LabelPredicate(label=label_contains_tau, op="contains"))

        if len(body) < max_body_size:
            need_label_eq = trial.suggest_int("need_label_eq", low=0, high=1)
            if need_label_eq == 1:
                label_eq_tau = trial.suggest_categorical("label_eq_tau", label_list)
                body.append(LabelPredicate(label=label_eq_tau, op="eq"))

        if len(body) < max_body_size:
            need_label_minus = trial.suggest_int("need_label_minus", low=0, high=1)
            if need_label_minus == 1:
                label_minus_tau = trial.suggest_categorical("label_minus_tau", label_list)
                body.append(LabelPredicate(label=label_minus_tau, op="minus"))

        # ================================================================
        # 3b. 纯 LabelPredicate body 标记
        # ================================================================
        _label_only_body = body and all(isinstance(p, LabelPredicate) for p in body)

        # ================================================================
        # 4. 规则后件标签 + 操作类型
        # ================================================================
        consequence_op = trial.suggest_categorical("consequence_op", ["add", "remove", "replace"])
        consequence_label = trial.suggest_categorical("consequence_label", label_list)

        # ================================================================
        # 5. 覆盖率计算 (cached fire masks)
        # ================================================================
        n_val = len(val_docs)

        if precomputed_fire_masks is not None and body_textual_indices:
            # Start with textual predicates from cache
            fires = np.ones(n_val, dtype=bool)
            for pidx in body_textual_indices:
                fires &= precomputed_fire_masks[pidx]
        elif not body:
            fires = np.ones(n_val, dtype=bool)
        else:
            fires = np.ones(n_val, dtype=bool)

        # ML threshold predicates from proba cache
        for model_name, ml_label, ml_thresh in ml_body_info:
            if ml_proba_cache is not None and model_name in ml_proba_cache:
                ml_label_idx = label_list.index(ml_label)
                fires &= (ml_proba_cache[model_name][:, ml_label_idx] >= ml_thresh)
            else:
                # fallback: evaluate per-doc
                pred = MLThresholdPredicate(model_name=model_name, label=ml_label, threshold=ml_thresh)
                for i in range(n_val):
                    if fires[i]:
                        fires[i] = bool(pred(val_docs[i]))

        # LabelPredicates: must evaluate per-doc (depend on doc.lbl)
        label_preds_in_body = [p for p in body if isinstance(p, LabelPredicate)]
        if label_preds_in_body:
            for i in range(n_val):
                if fires[i]:
                    proxy = Document(
                        cnt=val_docs[i].cnt,
                        lbl=set(val_docs[i].lbl) if val_docs[i].lbl else set(),
                        mtd=val_docs[i].mtd,
                        ttl=val_docs[i].ttl,
                    )
                    fires[i] = all(p(proxy) for p in label_preds_in_body)

        # Handle case where body has textual preds but no cache
        if not body_textual_indices and not ml_body_info and not label_preds_in_body:
            # empty body — fires = all True (already set)
            pass
        elif not body_textual_indices and precomputed_fire_masks is None and body:
            # fallback: full per-doc evaluation
            fires = np.zeros(n_val, dtype=bool)
            for i, doc in enumerate(val_docs):
                proxy = Document(
                    cnt=doc.cnt,
                    lbl=set(doc.lbl) if doc.lbl else set(),
                    mtd=doc.mtd,
                    ttl=doc.ttl,
                )
                fires[i] = all(p(proxy) for p in body)

        coverage = float(fires.sum()) / n_val if n_val > 0 else 0.0

        # ================================================================
        # 6. 自适应覆盖率检查
        # ================================================================
        label_idx = label_list.index(consequence_label)
        label_prevalence = float(val_labels[:, label_idx].sum()) / n_val if n_val > 0 else 0.0
        adaptive_min_cov = min(min_coverage, max(label_prevalence * 0.3, 0.001))
        if coverage < adaptive_min_cov:
            if trial.number < 5:
                logger.info("Trial %d PRUNED@coverage: cov=%.4f < %.4f, body_size=%d, n_fire=%d",
                            trial.number, coverage, adaptive_min_cov, len(body), int(fires.sum()))
            # Graded penalty: closer to threshold → less negative → TPE learns
            # coverage=0 → -1.0, coverage=threshold → -0.5
            return -1.0 + 0.5 * (coverage / adaptive_min_cov) if adaptive_min_cov > 0 else -1.0

        # ================================================================
        # 7. 增量 F1
        # ================================================================
        new_predictions = existing_predictions.copy()
        if consequence_op == "add":
            new_predictions[fires, label_idx] = 1.0
        elif consequence_op == "remove":
            new_predictions[fires, label_idx] = 0.0
        elif consequence_op == "replace":
            new_predictions[fires, :] = 0.0
            new_predictions[fires, label_idx] = 1.0

        # ================================================================
        # 7b. Correction precision gate
        # ================================================================
        if rule_min_precision > 0.0:
            effective_min_prec = rule_min_precision
            if _label_only_body:
                effective_min_prec = min(rule_min_precision * 1.2, 1.0)
            old_hit = (existing_predictions[fires, label_idx] == val_labels[fires, label_idx])
            new_hit = (new_predictions[fires, label_idx] == val_labels[fires, label_idx])
            n_improved = int((new_hit & ~old_hit).sum())
            n_worsened = int((old_hit & ~new_hit).sum())
            n_changes = n_improved + n_worsened
            if n_changes > 0:
                corr_prec = n_improved / n_changes
                # When n_changes is small, the precision estimate is unreliable.
                # Skip gate and let f1_gain decide; still prune if all changes are wrong.
                if n_changes < 10:
                    if n_improved == 0:
                        if trial.number < 5:
                            logger.info("Trial %d PRUNED@corr_prec: 0/%d corrections wrong (small sample)",
                                        trial.number, n_changes)
                        # Graded: coverage gives partial credit
                        return -0.5 + 0.3 * coverage
                elif corr_prec < effective_min_prec:
                    if trial.number < 5:
                        logger.info("Trial %d PRUNED@corr_prec: %.3f < %.3f (label_only=%s), n_changes=%d",
                                    trial.number, corr_prec, effective_min_prec, _label_only_body, n_changes)
                    # Graded: closer precision → less negative
                    return -0.5 + 0.3 * (corr_prec / effective_min_prec)

        if metric_mode == "cluster_local" and cluster_label_indices is not None:
            all_new_f1s = f1_score(val_labels, new_predictions, average=None, zero_division=0)
            all_old_f1s = f1_score(val_labels, existing_predictions, average=None, zero_division=0)
            cluster_gain = float(all_new_f1s[cluster_label_indices].mean()
                                 - all_old_f1s[cluster_label_indices].mean())
            global_gain = float(
                f1_score(val_labels, new_predictions, average="macro", zero_division=0)
            ) - baseline_f1
            f1_gain = 0.9 * cluster_gain + 0.1 * global_gain
        else:
            new_macro_f1 = float(
                f1_score(val_labels, new_predictions, average="macro", zero_division=0)
            )
            f1_gain = new_macro_f1 - baseline_f1
        if trial.number < 5:
            logger.info("Trial %d RESULT: f1_gain=%.6f, coverage=%.4f, body_size=%d, label=%s, op=%s",
                        trial.number, f1_gain, coverage, len(body), consequence_label, consequence_op)
        return f1_gain

    except Exception as e:
        logger.warning("Error during trial %d: %s", trial.number, e)
        raise e


# ===========================================================================
# 从 trial.params 重构 RDL
# ===========================================================================

def _trial_to_rdl(
    trial: optuna.trial.FrozenTrial,
    candidate_predicates: List[Predicate],
    candidate_ml_models: List[str],
    label_list: List[str],
    val_docs: List[Document],
    val_labels: np.ndarray,
    existing_predictions: np.ndarray,
    type_groups: Optional[Dict[str, List[int]]] = None,
) -> RDL:
    """
    从 trial.params 字典重构完整的 RDL 对象（OHunt 风格类型开关）。
    """
    params = trial.params
    body: List[Predicate] = []

    # 重建文本谓词（与 evaluate_configuration 的类型开关对称）
    n_cands = len(candidate_predicates)
    if type_groups:
        for type_name, type_indices in sorted(type_groups.items()):
            if params.get(f"need_{type_name}", 0) != 1:
                continue
            idx = params.get(f"{type_name}_pred")
            if idx is None:
                continue
            i = int(idx)
            if i < 0 or i >= n_cands:
                continue
            pred = candidate_predicates[i]
            # 重建参数微调后的谓词
            if isinstance(pred, FreqPredicate):
                eta = params.get(f"eta_{i}", pred.eta)
                op = params.get(f"op_{i}", pred.op)
                threshold = params.get(f"threshold_{i}", pred.threshold) if pred.sim else pred.threshold
                body.append(FreqPredicate(
                    attr=pred.attr, r=pred.r,
                    op=op, eta=float(eta),
                    sim=pred.sim, threshold=threshold,
                ))
            elif isinstance(pred, MatchPredicate) and pred.sim:
                threshold = params.get(f"threshold_{i}", pred.threshold)
                body.append(MatchPredicate(
                    attr=pred.attr, r=pred.r,
                    sim=pred.sim, threshold=threshold,
                ))
            elif isinstance(pred, CooccurPredicate) and pred.sim:
                threshold = params.get(f"threshold_{i}", pred.threshold)
                body.append(CooccurPredicate(
                    attr=pred.attr, r1=pred.r1, r2=pred.r2,
                    sim=pred.sim, threshold=threshold,
                ))
            elif isinstance(pred, BeforePredicate) and pred.sim:
                threshold = params.get(f"threshold_{i}", pred.threshold)
                body.append(BeforePredicate(
                    attr=pred.attr, r1=pred.r1, r2=pred.r2,
                    sim=pred.sim, threshold=threshold,
                ))
            else:
                body.append(pred)

    # 重建第二个 MatchPredicate（可选的合取槽位）
    if type_groups and params.get("need_MatchPredicate_2", 0) == 1:
        idx2 = params.get("MatchPredicate_pred_2")
        if idx2 is not None:
            i2 = int(idx2)
            if 0 <= i2 < n_cands:
                pred2 = candidate_predicates[i2]
                if isinstance(pred2, MatchPredicate) and pred2.sim:
                    threshold2 = params.get(f"threshold_{i2 + 10000}", pred2.threshold)
                    tuned2 = MatchPredicate(
                        attr=pred2.attr, r=pred2.r,
                        sim=pred2.sim, threshold=threshold2,
                    )
                else:
                    tuned2 = pred2
                if tuned2 not in body:
                    body.append(tuned2)

    # 重建 ML 谓词 (threshold mode)
    for j, model_name in enumerate(candidate_ml_models):
        if params.get(f"need_ml_{j}", 0) == 1:
            ml_label = params.get(f"ml_label_{j}", label_list[0])
            ml_thresh = params.get(f"ml_thresh_{j}", 0.5)
            body.append(MLThresholdPredicate(
                model_name=model_name, label=ml_label, threshold=float(ml_thresh),
            ))

    # 重建 LabelPredicate
    if params.get("need_label_contains", 0) == 1:
        tau = params.get("label_contains_tau", label_list[0])
        body.append(LabelPredicate(label=tau, op="contains"))
    if params.get("need_label_eq", 0) == 1:
        tau = params.get("label_eq_tau", label_list[0])
        body.append(LabelPredicate(label=tau, op="eq"))
    if params.get("need_label_minus", 0) == 1:
        tau = params.get("label_minus_tau", label_list[0])
        body.append(LabelPredicate(label=tau, op="minus"))

    consequence = params.get("consequence_label", label_list[0])
    consequence_op = params.get("consequence_op", "add")

    # 计算覆盖率（使用代理文档副本避免 frozen dataclass 赋值问题）
    n_val = len(val_docs)
    fires_count = 0
    for doc in val_docs:
        if not body:
            fires_count += 1
        else:
            proxy = Document(
                cnt=doc.cnt,
                lbl=set(doc.lbl) if doc.lbl else set(),
                mtd=doc.mtd,
                ttl=doc.ttl,
            )
            if all(p(proxy) for p in body):
                fires_count += 1
    coverage = fires_count / n_val if n_val > 0 else 0.0

    return RDL(
        body=tuple(body),
        consequence=consequence,
        consequence_op=consequence_op,
        score=trial.value if trial.value is not None else 0.0,
        coverage=coverage,
        trial_number=trial.number,
    )


# ===========================================================================
# 规则去重
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

class RuleLearner:
    """
    Accuracy-Guided Rule Discovery via Meta-learning.
    基于 Optuna 贝叶斯优化搜索最优 RDL 规则集合。

    沿用参考代码的 study 创建 + optimize + JournalStorage 持久化模式。

    Parameters
    ----------
    candidate_predicates : List[Predicate]
        候选文本谓词列表（来自 PatternStore 或手动构建）。
    candidate_ml_models : List[str]
        候选 ML 模型名称列表（已通过 register_ml_model 注册）。
    label_list : List[str]
        全部标签名称列表。
    val_docs : List[Document]
        验证集文档。
    val_labels : np.ndarray
        验证集真实标签（multi-hot 矩阵, shape=(n_val, n_labels)）。
    max_trials : int
        每轮搜索的最大 trial 数量。
    top_n : int
        最多发现的规则数量（贪心序列覆盖轮数）。
    min_coverage : float
        规则最小覆盖率阈值。
    seed : int
        随机种子。
    storage_path : str or None
        Optuna JournalStorage 文件路径前缀。为 None 则使用内存存储。
    verbose : bool
        是否打印详细日志。
    """

    def __init__(
        self,
        candidate_predicates: List[Predicate],
        candidate_ml_models: List[str],
        label_list: List[str],
        val_docs: List[Document],
        val_labels: np.ndarray,
        max_trials: int = 200,
        top_n: int = 10,
        min_coverage: float = 0.05,
        seed: int = 42,
        storage_path: Optional[str] = None,
        verbose: bool = True,
        base_predictions: Optional[np.ndarray] = None,
        base_f1: Optional[float] = None,
        top_per_type: int = 5,
        rule_min_precision: float = 0.0,
        accept_docs: Optional[List[Document]] = None,
        accept_labels: Optional[np.ndarray] = None,
        accept_predictions: Optional[np.ndarray] = None,
        accept_f1: Optional[float] = None,
        metric_mode: str = "global_macro",
        cluster_label_indices: Optional[List[int]] = None,
    ) -> None:
        self.candidate_predicates = list(candidate_predicates)
        self.candidate_ml_models = list(candidate_ml_models)
        self.label_list = list(label_list)
        self.val_docs = list(val_docs)
        self.val_labels = np.asarray(val_labels, dtype=np.float32)
        self.max_trials = max_trials
        self.top_n = top_n
        self.min_coverage = min_coverage
        self.seed = seed
        self.storage_path = storage_path
        self.verbose = verbose
        self.base_predictions = (
            np.asarray(base_predictions, dtype=np.float32)
            if base_predictions is not None
            else None
        )
        self.base_f1 = base_f1
        self.top_per_type = top_per_type
        self.rule_min_precision = rule_min_precision
        self.accept_docs = list(accept_docs) if accept_docs is not None else None
        self.accept_labels = (
            np.asarray(accept_labels, dtype=np.float32)
            if accept_labels is not None else None
        )
        self.accept_predictions = (
            np.asarray(accept_predictions, dtype=np.float32)
            if accept_predictions is not None else None
        )
        self.accept_f1 = accept_f1
        self.metric_mode = metric_mode
        self.cluster_label_indices = cluster_label_indices

    # [CORE LOGIC] 主发现循环：创建 Optuna 研究 → 优化 → 提取最优规则集
    def discover(self) -> RDLSet:
        """
        Run the full rule discovery pipeline.

        采用贪心序列覆盖策略：每轮发现一条最优规则，更新现有预测后
        再搜索下一条，直到达到 top_n 轮或无法继续提升 F1。

        Returns
        -------
        RDLSet
            发现的最优规则集合。
        """
        discovered_rules: List[RDL] = []
        # Save original doc labels for potential retry-loop resets
        original_labels = [set(doc.lbl) for doc in self.val_docs]
        if self.base_predictions is not None:
            existing_predictions = self.base_predictions.copy()
        else:
            existing_predictions = np.zeros_like(self.val_labels, dtype=np.float32)
        if self.base_f1 is not None:
            baseline_f1 = self.base_f1
        else:
            baseline_f1 = float(
                f1_score(self.val_labels, existing_predictions,
                         average="macro", zero_division=0)
            )

        # Acceptance data state (for train/val split: BO on train, accept on val)
        _has_accept = self.accept_docs is not None
        if _has_accept:
            accept_original_labels = [set(doc.lbl) for doc in self.accept_docs]
            accept_preds = (
                self.accept_predictions.copy()
                if self.accept_predictions is not None
                else np.zeros_like(self.accept_labels, dtype=np.float32)
            )
            accept_baseline = (
                self.accept_f1
                if self.accept_f1 is not None
                else float(f1_score(self.accept_labels, accept_preds,
                                    average="macro", zero_division=0))
            )

        for round_idx in range(self.top_n):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  规则发现第 {round_idx + 1}/{self.top_n} 轮")
                print(f"  当前基线 Macro-F1: {baseline_f1:.4f}")
                print(f"{'='*60}")

            # Per-round singleton scoring → per-type top-K
            type_groups, fire_masks, pred_label_phi = _score_singletons(
                self.candidate_predicates, self.label_list,
                self.val_docs, self.val_labels,
                existing_predictions, baseline_f1,
                top_per_type=self.top_per_type,
                metric_mode=self.metric_mode,
                cluster_label_indices=self.cluster_label_indices,
            )
            total_shortlisted = sum(len(v) for v in type_groups.values())
            if self.verbose:
                parts = [f"{t}={len(idxs)}" for t, idxs in sorted(type_groups.items())]
                print(f"  Singleton scoring: {total_shortlisted} predicates "
                      f"({', '.join(parts)})")

            # [CORE LOGIC] 创建 Optuna 贝叶斯优化研究，支持日志持久化
            study = self._create_study(round_idx)

            # Pre-compute ML proba cache for threshold-based evaluation
            _ml_proba_cache: Dict[str, np.ndarray] = {}
            for model_name in self.candidate_ml_models:
                try:
                    model = _get_ml_model(model_name)
                    if hasattr(model, '_clf') and hasattr(model._clf, 'predict_proba'):
                        _texts = [doc.cnt for doc in self.val_docs]
                        _ml_proba_cache[model_name] = model._clf.predict_proba(_texts).astype(np.float32)
                    elif hasattr(model, 'predict_proba_single'):
                        # Slower fallback: one-by-one
                        _proba = np.array(
                            [model.predict_proba_single(doc.cnt) for doc in self.val_docs],
                            dtype=np.float32,
                        )
                        _ml_proba_cache[model_name] = _proba
                except Exception as _e:
                    logger.warning("Could not cache proba for %s: %s", model_name, _e)

            # 构建 objective 闭包，捕获当前轮次的状态
            _candidate_predicates = self.candidate_predicates
            _candidate_ml_models = self.candidate_ml_models
            _label_list = self.label_list
            _val_docs = self.val_docs
            _val_labels = self.val_labels
            _baseline_f1 = baseline_f1
            _existing_predictions = existing_predictions.copy()
            _min_coverage = self.min_coverage
            _type_groups = type_groups
            _fire_masks = fire_masks
            _pred_label_phi = pred_label_phi

            _rule_min_precision = self.rule_min_precision

            def objective(trial):
                return evaluate_configuration(
                    trial=trial,
                    candidate_predicates=_candidate_predicates,
                    candidate_ml_models=_candidate_ml_models,
                    label_list=_label_list,
                    val_docs=_val_docs,
                    val_labels=_val_labels,
                    baseline_f1=_baseline_f1,
                    existing_predictions=_existing_predictions,
                    min_coverage=_min_coverage,
                    type_groups=_type_groups,
                    rule_min_precision=_rule_min_precision,
                    metric_mode=self.metric_mode,
                    cluster_label_indices=self.cluster_label_indices,
                    precomputed_fire_masks=_fire_masks,
                    pred_label_phi=_pred_label_phi,
                    ml_proba_cache=_ml_proba_cache,
                )

            # 执行优化（沿用参考代码的 study.optimize 模式）
            study.optimize(objective, n_trials=self.max_trials)

            # Debug: summarize trial outcomes
            all_values = [t.value for t in study.trials if t.value is not None]
            if all_values:
                pos_vals = [v for v in all_values if v > 0]
                neg_vals = [v for v in all_values if v == -1.0]
                zero_vals = [v for v in all_values if v == 0.0]
                logger.info(
                    "Optuna round %d summary: %d trials, %d positive (max=%.4f), "
                    "%d zero, %d rejected (-1.0), best=%.4f",
                    round_idx, len(all_values), len(pos_vals),
                    max(pos_vals) if pos_vals else 0.0,
                    len(zero_vals), len(neg_vals),
                    max(all_values),
                )

            # 提取最优 trial
            best = study.best_trial
            if self.verbose:
                print(f"  最优 trial #{best.number}: F1 增益 = {best.value:.4f}")
                print(f"  最优超参数: {best.params}")

            if best.value is None or best.value <= 0:
                if self.verbose:
                    print("  无法继续提升 F1，停止搜索。")
                break

            # 从 trial 重构 RDL 对象
            rule = _trial_to_rdl(
                trial=best,
                candidate_predicates=self.candidate_predicates,
                candidate_ml_models=self.candidate_ml_models,
                label_list=self.label_list,
                val_docs=self.val_docs,
                val_labels=self.val_labels,
                existing_predictions=existing_predictions,
                type_groups=type_groups,
            )

            # 去重检查
            if _is_redundant(rule, discovered_rules):
                if self.verbose:
                    print(f"  规则冗余，跳过。")
                continue

            # Validate on acceptance data (if separate from BO data)
            label_idx = self.label_list.index(rule.consequence)
            if _has_accept:
                a_fires = np.array(
                    [rule.fires(doc) for doc in self.accept_docs], dtype=bool
                )
                test_a = accept_preds.copy()
                if rule.consequence_op == "add":
                    test_a[a_fires, label_idx] = 1.0
                elif rule.consequence_op == "remove":
                    test_a[a_fires, label_idx] = 0.0
                elif rule.consequence_op == "replace":
                    test_a[a_fires, :] = 0.0
                    test_a[a_fires, label_idx] = 1.0
                new_accept_f1 = float(
                    f1_score(self.accept_labels, test_a,
                             average="macro", zero_division=0)
                )
                if new_accept_f1 <= accept_baseline:
                    if self.verbose:
                        print(f"  规则在验证集上无提升 "
                              f"({new_accept_f1:.4f} <= {accept_baseline:.4f})，跳过。")
                    continue
                # Update accept state
                accept_preds = test_a
                accept_baseline = new_accept_f1
                for i, doc in enumerate(self.accept_docs):
                    if a_fires[i]:
                        if rule.consequence_op == "add":
                            doc.lbl.add(rule.consequence)
                        elif rule.consequence_op == "remove":
                            doc.lbl.discard(rule.consequence)
                        elif rule.consequence_op == "replace":
                            doc.lbl.clear()
                            doc.lbl.add(rule.consequence)

            discovered_rules.append(rule)
            if self.verbose:
                print(f"  发现规则: {rule}")

            # 更新 BO 现有预测、基线和文档标签（label propagation）
            for i, doc in enumerate(self.val_docs):
                if rule.fires(doc):
                    if rule.consequence_op == "add":
                        existing_predictions[i, label_idx] = 1.0
                        doc.lbl.add(rule.consequence)
                    elif rule.consequence_op == "remove":
                        existing_predictions[i, label_idx] = 0.0
                        doc.lbl.discard(rule.consequence)
                    elif rule.consequence_op == "replace":
                        existing_predictions[i, :] = 0.0
                        existing_predictions[i, label_idx] = 1.0
                        doc.lbl.clear()
                        doc.lbl.add(rule.consequence)
            baseline_f1 = float(
                f1_score(self.val_labels, existing_predictions, average="macro", zero_division=0)
            )

            if self.verbose:
                if _has_accept:
                    print(f"  更新后 BO 基线: {baseline_f1:.4f}  "
                          f"Accept 基线: {accept_baseline:.4f}")
                else:
                    print(f"  更新后基线 Macro-F1: {baseline_f1:.4f}")

        # Restore original labels
        if _has_accept:
            for i, doc in enumerate(self.accept_docs):
                doc.lbl.clear()
                doc.lbl.update(accept_original_labels[i])

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  规则发现完成！共发现 {len(discovered_rules)} 条规则")
            print(f"  最终 BO Macro-F1: {baseline_f1:.4f}")
            if _has_accept:
                print(f"  最终 Accept Macro-F1: {accept_baseline:.4f}")
            print(f"{'='*60}")

        return RDLSet(rules=discovered_rules, label_names=self.label_list)

    # ------------------------------------------------------------------
    # Batch 模式：仅运行 BO 收集 trials（不做贪心选择）
    # ------------------------------------------------------------------

    def run_bo(
        self,
    ) -> List[Tuple["optuna.trial.FrozenTrial", Dict[str, List[int]]]]:
        """
        Run a single Bayesian optimisation study and return **all** completed
        trials together with the singleton type-groups context.

        This is the building-block for the per-cluster batch pipeline
        described in the paper pseudocode.  No greedy selection or label
        propagation is performed here — that happens in :meth:`batch_select`.

        Returns
        -------
        list of (FrozenTrial, type_groups)
            Only trials with ``state == COMPLETE`` and positive F1 gain.
        """
        # Initialise baseline (same logic as discover())
        if self.base_predictions is not None:
            existing_predictions = self.base_predictions.copy()
        else:
            existing_predictions = np.zeros_like(self.val_labels, dtype=np.float32)
        if self.base_f1 is not None:
            baseline_f1 = self.base_f1
        else:
            baseline_f1 = float(
                f1_score(self.val_labels, existing_predictions,
                         average="macro", zero_division=0)
            )

        # Singleton scoring (once)
        type_groups, fire_masks, pred_label_phi = _score_singletons(
            self.candidate_predicates, self.label_list,
            self.val_docs, self.val_labels,
            existing_predictions, baseline_f1,
            top_per_type=self.top_per_type,
            metric_mode=self.metric_mode,
            cluster_label_indices=self.cluster_label_indices,
        )
        if self.verbose:
            total = sum(len(v) for v in type_groups.values())
            parts = [f"{t}={len(idxs)}" for t, idxs in sorted(type_groups.items())]
            print(f"  [run_bo] Singleton scoring: {total} predicates "
                  f"({', '.join(parts)})")

        # Pre-compute ML proba cache
        _ml_proba_cache: Dict[str, np.ndarray] = {}
        for model_name in self.candidate_ml_models:
            try:
                model = _get_ml_model(model_name)
                if hasattr(model, '_clf') and hasattr(model._clf, 'predict_proba'):
                    _texts = [doc.cnt for doc in self.val_docs]
                    _ml_proba_cache[model_name] = model._clf.predict_proba(_texts).astype(np.float32)
                elif hasattr(model, 'predict_proba_single'):
                    _proba = np.array(
                        [model.predict_proba_single(doc.cnt) for doc in self.val_docs],
                        dtype=np.float32,
                    )
                    _ml_proba_cache[model_name] = _proba
            except Exception as _e:
                logger.warning("Could not cache proba for %s: %s", model_name, _e)

        # Build objective closure
        _cp = self.candidate_predicates
        _cm = self.candidate_ml_models
        _ll = self.label_list
        _vd = self.val_docs
        _vl = self.val_labels
        _bf = baseline_f1
        _ep = existing_predictions.copy()
        _mc = self.min_coverage
        _tg = type_groups

        _rmp = self.rule_min_precision

        def objective(trial):
            return evaluate_configuration(
                trial=trial,
                candidate_predicates=_cp,
                candidate_ml_models=_cm,
                label_list=_ll,
                val_docs=_vd,
                val_labels=_vl,
                baseline_f1=_bf,
                existing_predictions=_ep,
                min_coverage=_mc,
                type_groups=_tg,
                rule_min_precision=_rmp,
                metric_mode=self.metric_mode,
                cluster_label_indices=self.cluster_label_indices,
                precomputed_fire_masks=fire_masks,
                pred_label_phi=pred_label_phi,
                ml_proba_cache=_ml_proba_cache,
            )

        study = self._create_study(round_idx=0)
        study.optimize(objective, n_trials=self.max_trials)

        # Collect all completed trials with positive gain
        completed = [
            (t, type_groups)
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and t.value > 0
        ]

        # Diagnostic summary: breakdown of trial outcomes
        all_vals = [t.value for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None]
        n_neg1 = sum(1 for v in all_vals if v == -1.0)
        n_zero = sum(1 for v in all_vals if v == 0.0)
        n_pos = sum(1 for v in all_vals if v > 0)
        n_neg_other = len(all_vals) - n_neg1 - n_zero - n_pos
        logger.info("  [BO summary] %d trials: %d pruned(-1), %d zero-gain, "
                    "%d negative-gain, %d positive-gain",
                    len(all_vals), n_neg1, n_zero, n_neg_other, n_pos)
        if n_pos > 0:
            best_gain = max(v for v in all_vals if v > 0)
            logger.info("  [BO summary] best gain=%.6f", best_gain)
        if n_neg1 == len(all_vals) and len(all_vals) > 0:
            # All trials pruned — log body_size distribution for diagnosis
            body_sizes = [t.params.get("max_body_size", "?") for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]
            logger.warning("  [BO summary] ALL trials pruned! body_sizes=%s",
                          dict(Counter(body_sizes)) if body_sizes else "N/A")

        return completed

    # ------------------------------------------------------------------
    # Batch 全局选择：贪心添加规则到 Σ
    # ------------------------------------------------------------------

    @staticmethod
    def batch_select(
        all_trials: List[
            Tuple[
                "optuna.trial.FrozenTrial",
                Dict[str, List[int]],   # type_groups
                List[Predicate],        # cluster predicates
                List[str],              # cluster ML model names
            ]
        ],
        label_list: List[str],
        val_docs: List[Document],
        val_labels: np.ndarray,
        base_predictions: Optional[np.ndarray] = None,
        base_f1: Optional[float] = None,
        sort_by_gain: bool = True,
        verbose: bool = True,
        rule_min_precision: float = 0.0,
        accept_docs: Optional[List[Document]] = None,
        accept_labels: Optional[np.ndarray] = None,
        accept_predictions: Optional[np.ndarray] = None,
        accept_f1: Optional[float] = None,
    ) -> "RDLSet":
        """
        Global batch validation (paper Algorithm lines 11-13).

        Iterate collected trials from per-cluster BO runs and greedily add
        each rule to Σ **only if it improves** the overall macro-F1 on the
        full validation set.

        Parameters
        ----------
        all_trials
            List of ``(trial, type_groups, cluster_predicates,
            cluster_ml_models)`` tuples collected from per-cluster
            :meth:`run_bo` calls.
        sort_by_gain : bool
            If ``True`` (default), sort trials by F1 gain descending before
            selection.  If ``False``, iterate in original (Optuna) order.
        accept_docs / accept_labels / accept_predictions / accept_f1
            Optional separate dataset for rule acceptance validation
            (BO data is used for candidate scoring, accept data for final
            acceptance).  When None, acceptance uses the same BO data.
        """
        val_labels = np.asarray(val_labels, dtype=np.float32)
        if base_predictions is not None:
            existing_predictions = np.asarray(base_predictions, dtype=np.float32).copy()
        else:
            existing_predictions = np.zeros_like(val_labels, dtype=np.float32)
        if base_f1 is not None:
            baseline_f1 = float(base_f1)
        else:
            baseline_f1 = float(
                f1_score(val_labels, existing_predictions,
                         average="macro", zero_division=0)
            )

        # Accept data state
        _has_accept = accept_docs is not None
        if _has_accept:
            accept_labels_arr = np.asarray(accept_labels, dtype=np.float32)
            accept_preds = (
                np.asarray(accept_predictions, dtype=np.float32).copy()
                if accept_predictions is not None
                else np.zeros_like(accept_labels_arr, dtype=np.float32)
            )
            accept_baseline = (
                float(accept_f1)
                if accept_f1 is not None
                else float(f1_score(accept_labels_arr, accept_preds,
                                    average="macro", zero_division=0))
            )
            accept_original_labels = [set(doc.lbl) for doc in accept_docs]

        if sort_by_gain:
            all_trials = sorted(all_trials, key=lambda x: -(x[0].value or 0.0))

        # Preserve original labels for restoration at the end
        original_labels = [set(doc.lbl) for doc in val_docs]
        discovered_rules: List[RDL] = []

        for trial, type_groups, cluster_preds, cluster_ml in all_trials:
            rule = _trial_to_rdl(
                trial=trial,
                candidate_predicates=cluster_preds,
                candidate_ml_models=cluster_ml,
                label_list=label_list,
                val_docs=val_docs,
                val_labels=val_labels,
                existing_predictions=existing_predictions,
                type_groups=type_groups,
            )

            # Redundancy check
            if _is_redundant(rule, discovered_rules):
                continue

            # Simulate adding rule: compute new F1
            label_idx = label_list.index(rule.consequence)
            test_preds = existing_predictions.copy()
            fires = np.zeros(len(val_docs), dtype=bool)
            for i, doc in enumerate(val_docs):
                fires[i] = rule.fires(doc)

            if rule.consequence_op == "add":
                test_preds[fires, label_idx] = 1.0
            elif rule.consequence_op == "remove":
                test_preds[fires, label_idx] = 0.0
            elif rule.consequence_op == "replace":
                test_preds[fires, :] = 0.0
                test_preds[fires, label_idx] = 1.0

            # Correction precision gate (column-level: 只看 consequence label)
            #   纯 LabelPredicate body 使用更高门槛 (×1.5)
            if rule_min_precision > 0.0:
                _lbl_only = rule.body and all(isinstance(p, LabelPredicate) for p in rule.body)
                eff_prec = min(rule_min_precision * 1.2, 1.0) if _lbl_only else rule_min_precision
                old_hit = (existing_predictions[fires, label_idx] == val_labels[fires, label_idx])
                new_hit = (test_preds[fires, label_idx] == val_labels[fires, label_idx])
                n_improved = int((new_hit & ~old_hit).sum())
                n_worsened = int((old_hit & ~new_hit).sum())
                n_changes = n_improved + n_worsened
                if n_changes > 0:
                    corr_prec = n_improved / n_changes
                    if n_changes < 10:
                        if n_improved == 0:
                            continue
                    elif corr_prec < eff_prec:
                        continue

            new_f1 = float(
                f1_score(val_labels, test_preds, average="macro", zero_division=0)
            )

            if new_f1 > baseline_f1:
                # Validate on accept data (if separate from BO data)
                if _has_accept:
                    a_fires = np.array(
                        [rule.fires(doc) for doc in accept_docs], dtype=bool
                    )
                    test_a = accept_preds.copy()
                    if rule.consequence_op == "add":
                        test_a[a_fires, label_idx] = 1.0
                    elif rule.consequence_op == "remove":
                        test_a[a_fires, label_idx] = 0.0
                    elif rule.consequence_op == "replace":
                        test_a[a_fires, :] = 0.0
                        test_a[a_fires, label_idx] = 1.0
                    new_accept_f1 = float(
                        f1_score(accept_labels_arr, test_a,
                                 average="macro", zero_division=0)
                    )
                    if new_accept_f1 <= accept_baseline:
                        continue
                    # Update accept state
                    accept_preds = test_a
                    accept_baseline = new_accept_f1
                    for i, doc in enumerate(accept_docs):
                        if a_fires[i]:
                            if rule.consequence_op == "add":
                                doc.lbl.add(rule.consequence)
                            elif rule.consequence_op == "remove":
                                doc.lbl.discard(rule.consequence)
                            elif rule.consequence_op == "replace":
                                doc.lbl.clear()
                                doc.lbl.add(rule.consequence)

                discovered_rules.append(rule)
                existing_predictions = test_preds
                baseline_f1 = new_f1
                # BO label propagation
                for i, doc in enumerate(val_docs):
                    if fires[i]:
                        if rule.consequence_op == "add":
                            doc.lbl.add(rule.consequence)
                        elif rule.consequence_op == "remove":
                            doc.lbl.discard(rule.consequence)
                        elif rule.consequence_op == "replace":
                            doc.lbl.clear()
                            doc.lbl.add(rule.consequence)
                if verbose:
                    msg = f"  [batch_select] +rule {rule}  BO-F1={baseline_f1:.4f}"
                    if _has_accept:
                        msg += f"  Accept-F1={accept_baseline:.4f}"
                    print(msg)

        # Restore original labels
        for i, doc in enumerate(val_docs):
            doc.lbl.clear()
            doc.lbl.update(original_labels[i])
        if _has_accept:
            for i, doc in enumerate(accept_docs):
                doc.lbl.clear()
                doc.lbl.update(accept_original_labels[i])

        if verbose:
            print(f"  [batch_select] Finished: {len(discovered_rules)} rules, "
                  f"final BO-F1={baseline_f1:.4f}")

        return RDLSet(rules=discovered_rules, label_names=label_list)

    def _create_study(self, round_idx: int) -> optuna.Study:
        """
        # [CORE LOGIC] 创建 Optuna 贝叶斯优化研究
        沿用参考代码的 JournalFileOpenLock + JournalStorage 模式。

        Parameters
        ----------
        round_idx : int
            当前贪心轮次索引。

        Returns
        -------
        optuna.Study
        """
        sampler = optuna.samplers.TPESampler(seed=self.seed + round_idx)

        if self.storage_path:
            file_path = f"{self.storage_path}_round{round_idx}.log"
            lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(
                    file_path, lock_obj=lock_obj
                ),
            )
        else:
            storage = None

        study = optuna.create_study(
            storage=storage,
            direction="maximize",
            sampler=sampler,
            study_name=f"loris_rule_discovery_round{round_idx}",
        )
        return study


# ===========================================================================
# Phase 3 migration: RDL / RDLSet / _is_redundant now live in loris.rules.rdl.
# Re-bind the module-level names to the migrated definitions so that external
# importers of this legacy module (and the dead RuleLearner above) share the
# single authoritative implementation. This shadows the local class defs.
# ===========================================================================
from loris.rules.rdl import RDL, RDLSet, _is_redundant  # noqa: E402,F811

# ===========================================================================
# __main__ 测试块
# ===========================================================================

if __name__ == "__main__":
    """
    模拟一次完整的规则发现流程。
    使用合成数据测试 RuleLearner 的基本功能。
    """
    print("=" * 60)
    print("  LORIS Rule Discovery — 模拟测试")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 创建合成 Document 数据
    # ------------------------------------------------------------------
    # 已知模式：
    #   - 含 "bank" 或 "stock" → finance
    #   - 含 "hospital" 或 "doctor" → health
    #   - 含 "software" 或 "algorithm" → tech
    docs_data = [
        # finance
        {"cnt": "The bank reported record profits this quarter.", "labels": {"finance"}, "lbl": {"finance"}},
        {"cnt": "Stock market surged after the announcement.", "labels": {"finance"}, "lbl": {"finance"}},
        {"cnt": "The central bank adjusted interest rates.", "labels": {"finance"}, "lbl": {"finance"}},
        {"cnt": "Investors are watching the stock exchange closely.", "labels": {"finance"}, "lbl": {"finance"}},
        {"cnt": "The bank offers new savings accounts.", "labels": {"finance"}, "lbl": {"finance"}},
        # health
        {"cnt": "The hospital announced a new treatment program.", "labels": {"health"}, "lbl": {"health"}},
        {"cnt": "The doctor recommended regular exercise.", "labels": {"health"}, "lbl": {"health"}},
        {"cnt": "Hospital staff were praised for their dedication.", "labels": {"health"}, "lbl": {"health"}},
        {"cnt": "A new doctor joined the medical team.", "labels": {"health"}, "lbl": {"health"}},
        {"cnt": "The hospital received a major funding grant.", "labels": {"health"}, "lbl": {"health"}},
        # tech
        {"cnt": "The software update includes new security features.", "labels": {"tech"}, "lbl": {"tech"}},
        {"cnt": "Our algorithm achieves state-of-the-art performance.", "labels": {"tech"}, "lbl": {"tech"}},
        {"cnt": "New software tools for developers were released.", "labels": {"tech"}, "lbl": {"tech"}},
        {"cnt": "The algorithm was trained on a large dataset.", "labels": {"tech"}, "lbl": {"tech"}},
        {"cnt": "Software engineering best practices were discussed.", "labels": {"tech"}, "lbl": {"tech"}},
        # 多标签
        {"cnt": "The bank invested in new software platforms.", "labels": {"finance", "tech"}, "lbl": {"finance", "tech"}},
        {"cnt": "Hospital management software was upgraded.", "labels": {"health", "tech"}, "lbl": {"health", "tech"}},
        {"cnt": "The doctor reviewed stock options for retirement.", "labels": {"health", "finance"}, "lbl": {"health", "finance"}},
    ]

    label_list = ["finance", "health", "tech"]
    label2idx = {l: i for i, l in enumerate(label_list)}

    val_docs: List[Document] = []
    val_labels_list: List[List[float]] = []
    for d in docs_data:
        # 创建带有标签集的 Document，使 LabelPredicate 能正常工作
        val_docs.append(Document(cnt=d["cnt"], lbl=set(d.get("lbl", set()))))
        row = [0.0] * len(label_list)
        for lbl in d["labels"]:
            row[label2idx[lbl]] = 1.0
        val_labels_list.append(row)
    val_labels = np.array(val_labels_list, dtype=np.float32)

    print(f"\n验证集: {len(val_docs)} 篇文档, {len(label_list)} 个标签")
    print(f"标签列表: {label_list}")
    print(f"标签分布: {val_labels.sum(axis=0).astype(int).tolist()}")

    # ------------------------------------------------------------------
    # 2. 构建候选 Predicate 列表
    # ------------------------------------------------------------------
    # 注意：搜索空间大小 = 2^(n_predicates + 3_label_ops) × |label_list|^k
    # 候选谓词数量要适当控制，避免搜索空间过大导致 trial 不足以找到好配置
    candidate_predicates: List[Predicate] = [
        # finance 相关
        MatchPredicate(attr="cnt", r="bank"),
        MatchPredicate(attr="cnt", r="stock"),
        # health 相关
        MatchPredicate(attr="cnt", r="hospital"),
        MatchPredicate(attr="cnt", r="doctor"),
        # tech 相关
        MatchPredicate(attr="cnt", r="software"),
        MatchPredicate(attr="cnt", r="algorithm"),
    ]

    print(f"候选谓词数量: {len(candidate_predicates)}")
    for i, p in enumerate(candidate_predicates):
        print(f"  [{i}] {p}")

    # ------------------------------------------------------------------
    # 3. 实例化 RuleLearner 并运行
    # ------------------------------------------------------------------
    learner = RuleLearner(
        candidate_predicates=candidate_predicates,
        candidate_ml_models=[],  # 本测试不使用 ML 谓词
        label_list=label_list,
        val_docs=val_docs,
        val_labels=val_labels,
        max_trials=100,          # 足够的 trial 数量以探索搜索空间
        top_n=5,
        min_coverage=0.01,       # 小数据集适当降低覆盖率阈值
        seed=42,
        storage_path=None,       # 内存存储，不写入文件
        verbose=True,
    )

    print("\n开始规则发现...")
    rdl_set = learner.discover()

    # ------------------------------------------------------------------
    # 4. 查看发现的规则
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  发现的规则")
    print(f"{'='*60}")
    for i, rule in enumerate(rdl_set.rules):
        print(f"\n  规则 #{i+1}:")
        print(f"    Body: {' ∧ '.join(repr(p) for p in rule.body) if rule.body else '⊤'}")
        print(f"    Consequence: {rule.consequence}")
        print(f"    F1 增益: {rule.score:.4f}")
        print(f"    覆盖率: {rule.coverage:.4f}")

    # ------------------------------------------------------------------
    # 5. 评估规则集
    # ------------------------------------------------------------------
    metrics = rdl_set.evaluate(val_docs, val_labels)
    print(f"\n{'='*60}")
    print("  规则集评估结果")
    print(f"{'='*60}")
    print(f"  Micro-F1: {metrics['micro_f1']:.4f}")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")

    # ------------------------------------------------------------------
    # 6. 序列化/反序列化测试
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  序列化/反序列化测试")
    print(f"{'='*60}")
    rdl_dict = rdl_set.to_dict()
    rdl_set_restored = RDLSet.from_dict(rdl_dict)
    metrics_restored = rdl_set_restored.evaluate(val_docs, val_labels)
    print(f"  原始 Macro-F1:   {metrics['macro_f1']:.4f}")
    print(f"  反序列化 Macro-F1: {metrics_restored['macro_f1']:.4f}")
    assert abs(metrics["macro_f1"] - metrics_restored["macro_f1"]) < 1e-6, \
        "序列化/反序列化结果不一致！"
    print("  ✓ 序列化/反序列化一致性测试通过")

    print(f"\n{'='*60}")
    print("  测试完成！")
    print(f"{'='*60}")
