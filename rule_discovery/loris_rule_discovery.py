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
    LabelPredicate,
    predicate_to_dict,
    predicate_from_dict,
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
    score: float = 0.0
    coverage: float = 0.0
    trial_number: int = -1

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
        return (
            f"RDL({body_str} → {self.consequence!r}, "
            f"score={self.score:.4f}, coverage={self.coverage:.4f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 安全的字典。"""
        return {
            "body": [predicate_to_dict(p) for p in self.body],
            "consequence": self.consequence,
            "score": self.score,
            "coverage": self.coverage,
            "trial_number": self.trial_number,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RDL":
        """从字典反序列化。"""
        body = tuple(predicate_from_dict(pd) for pd in d["body"])
        return cls(
            body=body,
            consequence=d["consequence"],
            score=d.get("score", 0.0),
            coverage=d.get("coverage", 0.0),
            trial_number=d.get("trial_number", -1),
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

    # [CORE LOGIC] 多标签预测：多条规则各自独立触发，取并集构成多标签输出
    def predict(self, docs: List[Document]) -> np.ndarray:
        """
        Apply all discovered rules to produce multi-hot predictions.

        Parameters
        ----------
        docs : List[Document]
            待预测文档列表。

        Returns
        -------
        np.ndarray, shape (n_docs, n_labels), dtype float32
            Multi-hot 预测矩阵。
        """
        n_labels = len(self.label_names)
        predictions = np.zeros((len(docs), n_labels), dtype=np.float32)

        for rule in self.rules:
            label_idx = self._label2idx.get(rule.consequence)
            if label_idx is None:
                continue
            for i, doc in enumerate(docs):
                if rule.fires(doc):
                    predictions[i, label_idx] = 1.0

        return predictions

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
        op = trial.suggest_categorical(f"op_{idx}", [">=", "<=", "==", ">", "<"])
        # 如果启用了语义匹配，额外微调 threshold
        if original.sim:
            threshold = trial.suggest_float(f"threshold_{idx}", low=0.7, high=0.99)
        else:
            threshold = original.threshold
        return FreqPredicate(
            attr=original.attr, r=original.r,
            op=op, eta=float(eta),
            sim=original.sim, threshold=threshold,
        )

    # MatchPredicate：如果启用语义匹配则微调 threshold
    if isinstance(original, MatchPredicate) and original.sim:
        threshold = trial.suggest_float(f"threshold_{idx}", low=0.7, high=0.99)
        return MatchPredicate(
            attr=original.attr, r=original.r,
            sim=original.sim, threshold=threshold,
        )

    # CooccurPredicate：如果启用语义匹配则微调 threshold
    if isinstance(original, CooccurPredicate) and original.sim:
        threshold = trial.suggest_float(f"threshold_{idx}", low=0.7, high=0.99)
        return CooccurPredicate(
            attr=original.attr, r1=original.r1, r2=original.r2,
            sim=original.sim, threshold=threshold,
        )

    # BeforePredicate：如果启用语义匹配则微调 threshold
    if isinstance(original, BeforePredicate) and original.sim:
        threshold = trial.suggest_float(f"threshold_{idx}", low=0.7, high=0.99)
        return BeforePredicate(
            attr=original.attr, r1=original.r1, r2=original.r2,
            sim=original.sim, threshold=threshold,
        )

    # 其他谓词类型不需要参数微调，原样返回
    return original


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
) -> float:
    """
    # [CORE LOGIC] 目标函数：从 trial 采样配置 → 构建 RDL → 评估 F1 增益
    沿用参考代码的 objective 函数模式，在函数内部完成所有参数采样。

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial 对象，用于采样搜索空间参数。
    candidate_predicates : List[Predicate]
        候选文本谓词列表（来自 PatternStore）。
    candidate_ml_models : List[str]
        候选 ML 模型名称列表。
    label_list : List[str]
        全部标签名称列表。
    val_docs : List[Document]
        验证集文档。
    val_labels : np.ndarray
        验证集真实标签（multi-hot 矩阵）。
    baseline_f1 : float
        当前基线 Macro-F1（在现有预测之上）。
    existing_predictions : np.ndarray
        现有规则的预测结果（多轮贪心累积）。
    min_coverage : float
        最小覆盖率阈值。

    Returns
    -------
    float
        F1 增益（正值为提升，负值为惩罚）。
    """
    try:
        body: List[Predicate] = []

        # ================================================================
        # 1. 结构选择 + 参数微调：文本谓词（稀疏索引采样）
        # ================================================================
        # [CORE LOGIC] 关键修复：不再为每个谓词单独设二值开关（会导致平均选中
        # n/2 个谓词，AND 合取后覆盖率趋近于 0）。改用两阶段稀疏采样：
        #   阶段 1: 采样规则体谓词数量 body_size ∈ {1, 2, 3, 4}
        #   阶段 2: 采样 body_size 个谓词在候选列表中的绝对索引
        # 这将 Optuna 的搜索空间从 O(2^N) 压缩至 O(N^4)，使 TPE 能有效学习。
        n_cands = len(candidate_predicates)
        if n_cands > 0:
            body_size = trial.suggest_int("body_size", 1, min(4, n_cands))
            selected_indices: set = set()
            for k in range(body_size):
                idx = trial.suggest_int(f"body_pred_idx_{k}", 0, n_cands - 1)
                selected_indices.add(idx)
            for idx in sorted(selected_indices):
                tuned_pred = _reconstruct_predicate(trial, idx, candidate_predicates[idx])
                body.append(tuned_pred)

        # ================================================================
        # 2. 结构选择：ML 谓词
        # ================================================================
        # [CORE LOGIC] ML 谓词的开关 + 目标标签 τ 联合搜索
        for j, model_name in enumerate(candidate_ml_models):
            need_ml = trial.suggest_int(f"need_ml_{j}", low=0, high=1)
            if need_ml == 1:
                # [CORE LOGIC] 搜索 ML 谓词最匹配的标签 τ
                ml_tau = trial.suggest_categorical(f"ml_tau_{j}", label_list)
                body.append(MLPredicate(model_name=model_name, label=ml_tau))

        # ================================================================
        # 3. 结构选择：LabelPredicate（三种标签谓词操作）
        # ================================================================
        # [CORE LOGIC] 标签谓词 x.lbl ⊗ y.lbl | τ ∈ x.lbl | x.lbl \ τ

        # 3a. contains: τ ∈ x.lbl（检查文档是否已包含某标签）
        need_label_contains = trial.suggest_int("need_label_contains", low=0, high=1)
        if need_label_contains == 1:
            label_contains_tau = trial.suggest_categorical(
                "label_contains_tau", label_list
            )
            body.append(LabelPredicate(label=label_contains_tau, op="contains"))

        # 3b. eq: x.lbl == {τ}（检查文档标签集是否恰好等于某标签）
        need_label_eq = trial.suggest_int("need_label_eq", low=0, high=1)
        if need_label_eq == 1:
            label_eq_tau = trial.suggest_categorical("label_eq_tau", label_list)
            body.append(LabelPredicate(label=label_eq_tau, op="eq"))

        # 3c. minus: x.lbl \ τ（从文档标签集中移除某标签）
        need_label_minus = trial.suggest_int("need_label_minus", low=0, high=1)
        if need_label_minus == 1:
            label_minus_tau = trial.suggest_categorical("label_minus_tau", label_list)
            body.append(LabelPredicate(label=label_minus_tau, op="minus"))

        # ================================================================
        # 4. 标签联合搜索：规则后件标签 p0
        # ================================================================
        # [CORE LOGIC] 规则结论标签作为搜索参数，让优化器自动发现最佳标签分配
        consequence_label = trial.suggest_categorical("consequence_label", label_list)

        # ================================================================
        # 5. 应用规则到验证集，计算覆盖率
        # ================================================================
        # [CORE LOGIC] 规则触发判定：文档满足所有体谓词时规则激活
        # 注意：LabelPredicate(op="minus") 有副作用（会修改 doc.lbl），
        # 因此需要在评估前保存标签集，评估后恢复，防止污染后续 trial。
        n_val = len(val_docs)
        fires = np.zeros(n_val, dtype=bool)
        for i, doc in enumerate(val_docs):
            if not body:
                fires[i] = True  # 空规则体 → 全部触发
            else:
                # 使用代理文档副本，避免 LabelPredicate(op="minus") 副作用
                # 污染原始 val_docs（Document 是 frozen dataclass，不能赋值 lbl）
                proxy = Document(
                    cnt=doc.cnt,
                    lbl=set(doc.lbl) if doc.lbl else set(),
                    mtd=doc.mtd,
                    ttl=doc.ttl,
                )
                fires[i] = all(p(proxy) for p in body)

        coverage = float(fires.sum()) / n_val if n_val > 0 else 0.0

        # ================================================================
        # 6. 覆盖率检查
        # ================================================================
        if coverage < min_coverage:
            return -1.0  # 覆盖率不足，返回惩罚分数

        # ================================================================
        # 7. 计算增量 F1
        # ================================================================
        # [CORE LOGIC] 增量 F1 评估：衡量新规则对现有预测的边际贡献
        label_idx = label_list.index(consequence_label)
        new_predictions = existing_predictions.copy()
        new_predictions[fires, label_idx] = 1.0

        new_macro_f1 = float(
            f1_score(val_labels, new_predictions, average="macro", zero_division=0)
        )
        f1_gain = new_macro_f1 - baseline_f1
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
) -> RDL:
    """
    # [CORE LOGIC] 从 trial.params 字典重构完整的 RDL 对象
    读取已完成 trial 中保存的参数值，重建规则体和后件标签。

    Parameters
    ----------
    trial : optuna.trial.FrozenTrial
        已完成的 trial 对象。
    其余参数用于重建谓词实例和计算覆盖率。

    Returns
    -------
    RDL
        重构的规则对象。
    """
    params = trial.params
    body: List[Predicate] = []

    # 重建文本谓词（与 evaluate_configuration 的稀疏索引采样对称）
    n_cands = len(candidate_predicates)
    body_size = int(params.get("body_size", 0))
    selected_indices: set = set()
    for k in range(body_size):
        idx = params.get(f"body_pred_idx_{k}")
        if idx is not None:
            selected_indices.add(int(idx))
    for idx in sorted(selected_indices):
        pred = candidate_predicates[idx]
        # 重建参数微调后的谓词
        if isinstance(pred, FreqPredicate):
            eta = params.get(f"eta_{idx}", pred.eta)
            op = params.get(f"op_{idx}", pred.op)
            threshold = params.get(f"threshold_{idx}", pred.threshold) if pred.sim else pred.threshold
            body.append(FreqPredicate(
                attr=pred.attr, r=pred.r,
                op=op, eta=float(eta),
                sim=pred.sim, threshold=threshold,
            ))
        elif isinstance(pred, MatchPredicate) and pred.sim:
            threshold = params.get(f"threshold_{idx}", pred.threshold)
            body.append(MatchPredicate(
                attr=pred.attr, r=pred.r,
                sim=pred.sim, threshold=threshold,
            ))
        elif isinstance(pred, CooccurPredicate) and pred.sim:
            threshold = params.get(f"threshold_{idx}", pred.threshold)
            body.append(CooccurPredicate(
                attr=pred.attr, r1=pred.r1, r2=pred.r2,
                sim=pred.sim, threshold=threshold,
            ))
        elif isinstance(pred, BeforePredicate) and pred.sim:
            threshold = params.get(f"threshold_{idx}", pred.threshold)
            body.append(BeforePredicate(
                attr=pred.attr, r1=pred.r1, r2=pred.r2,
                sim=pred.sim, threshold=threshold,
            ))
        else:
            body.append(pred)

    # 重建 ML 谓词
    for j, model_name in enumerate(candidate_ml_models):
        if params.get(f"need_ml_{j}", 0) == 1:
            ml_tau = params.get(f"ml_tau_{j}", label_list[0])
            body.append(MLPredicate(model_name=model_name, label=ml_tau))

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
        existing_predictions = np.zeros_like(self.val_labels, dtype=np.float32)
        baseline_f1 = 0.0

        for round_idx in range(self.top_n):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  规则发现第 {round_idx + 1}/{self.top_n} 轮")
                print(f"  当前基线 Macro-F1: {baseline_f1:.4f}")
                print(f"{'='*60}")

            # [CORE LOGIC] 创建 Optuna 贝叶斯优化研究，支持日志持久化
            study = self._create_study(round_idx)

            # 构建 objective 闭包，捕获当前轮次的状态
            _candidate_predicates = self.candidate_predicates
            _candidate_ml_models = self.candidate_ml_models
            _label_list = self.label_list
            _val_docs = self.val_docs
            _val_labels = self.val_labels
            _baseline_f1 = baseline_f1
            _existing_predictions = existing_predictions.copy()
            _min_coverage = self.min_coverage

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
                )

            # 执行优化（沿用参考代码的 study.optimize 模式）
            study.optimize(objective, n_trials=self.max_trials)

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
            )

            # 去重检查
            if _is_redundant(rule, discovered_rules):
                if self.verbose:
                    print(f"  规则冗余，跳过。")
                continue

            discovered_rules.append(rule)
            if self.verbose:
                print(f"  发现规则: {rule}")

            # 更新现有预测和基线
            label_idx = self.label_list.index(rule.consequence)
            for i, doc in enumerate(self.val_docs):
                if rule.fires(doc):
                    existing_predictions[i, label_idx] = 1.0
            baseline_f1 = float(
                f1_score(self.val_labels, existing_predictions, average="macro", zero_division=0)
            )

            if self.verbose:
                print(f"  更新后基线 Macro-F1: {baseline_f1:.4f}")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  规则发现完成！共发现 {len(discovered_rules)} 条规则")
            print(f"  最终 Macro-F1: {baseline_f1:.4f}")
            print(f"{'='*60}")

        return RDLSet(rules=discovered_rules, label_names=self.label_list)

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
