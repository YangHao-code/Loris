"""
chase_rule_discovery
--------------------
Chase-driven rule discovery engine (adapted from hybrid_rule_discovery).

Key differences from HybridRuleLearner:
  - Add-only rules (no remove) — Chase conflict-free
  - Dual-track BO: Track 1 (seed rules, ML+Text only) → Track 2 (propagation, SimPredicate+LabelPredicate)
  - SimPredicate(threshold) support via precomputed sim graphs and SpMV
  - neighbor_label_masks for efficient BO evaluation of pairwise rules
"""

from __future__ import annotations

import logging
import operator as _op
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from loris.document import Document
import scipy.sparse as sp

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
    SimPredicate,
    GroupPredicate,
    predicate_to_dict,
    predicate_from_dict,
)
from loris.predicates._core import _get_ml_model
from loris.rules.sim_graph import precompute_neighbor_label_masks, precompute_neighbor_label_counts
from loris.rules.rdl import (
    RDL,
    RDLSet,
    _is_redundant,
)

logger = logging.getLogger(__name__)

# ===========================================================================
# 常量
# ===========================================================================

# FreqPredicate 离散化网格
FREQ_ETA_BINS: List[float] = [2.0, 5.0, 10.0, 20.0]
FREQ_OP_BINS: List[str] = [">=", "<="]

# MLThresholdPredicate 离散化网格
ML_THRESHOLD_BINS: List[float] = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]

# 贪心评分默认权重
DEFAULT_GREEDY_WEIGHTS: Dict[str, float] = {
    "precision": 0.5,
    "info_gain": 0.2,
    "coverage": 0.2,
    "diversity": 0.1,
}

ERROR_AWARE_WEIGHTS: Dict[str, float] = {
    "info_gain": 0.3,
    "precision": 0.6,
    "coverage": 0.05,
    "diversity": 0.05,
}

# 比较算子映射 (用于 FreqPredicate 快速掩码生成)
_COMPARE_OPS = {"==": _op.eq, ">=": _op.ge, "<=": _op.le,
                "!=": _op.ne, ">": _op.gt, "<": _op.lt}

# 文本谓词类型的贪心填充顺序（先锚定高精度上下文）
_TEXT_TYPE_ORDER = ["MatchPredicate", "FreqPredicate",
                    "CooccurPredicate", "BeforePredicate"]


# ===========================================================================
# 诊断计数器：追踪 trial 被拒绝的原因分布
# ===========================================================================
_DIAG_COUNTERS: Dict[str, int] = {}

def _diag_inc(key: str) -> None:
    _DIAG_COUNTERS[key] = _DIAG_COUNTERS.get(key, 0) + 1

def _diag_reset() -> None:
    _DIAG_COUNTERS.clear()

def _diag_report() -> str:
    if not _DIAG_COUNTERS:
        return "  (no diagnostic data)"
    lines = []
    total = sum(_DIAG_COUNTERS.values())
    for k, v in sorted(_DIAG_COUNTERS.items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v} ({100*v/total:.1f}%)")
    return "\n".join(lines)


# ===========================================================================
# 快速 macro-F1 (替代 sklearn.metrics.f1_score, 纯 numpy 向量化)
# ===========================================================================

def _adaptive_corr_prec(base_min: float, label_support: int) -> float:
    """Per-label adaptive corr_prec threshold based on support size."""
    if label_support > 1000:
        return base_min
    elif label_support > 100:
        return min(base_min, 0.70)
    else:
        return min(base_min, 0.60)


def _fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """纯 numpy macro-F1，跳过 sklearn 的类型检查/验证开销。

    Parameters
    ----------
    y_true, y_pred : ndarray of shape (n_samples, n_labels), binary {0, 1}

    Returns
    -------
    float : macro-averaged F1 (与 sklearn zero_division=0 行为一致)
    """
    tp = (y_true * y_pred).sum(axis=0).astype(np.float64)
    fp = ((1 - y_true) * y_pred).sum(axis=0).astype(np.float64)
    fn = (y_true * (1 - y_pred)).sum(axis=0).astype(np.float64)
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    # 只对"存在样本"的标签求平均 (与 zero_division=0 一致)
    mask = (tp + fp + fn) > 0
    return float(f1[mask].mean()) if mask.any() else 0.0


def _fast_per_label_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """纯 numpy per-label F1 (替代 f1_score(..., average=None))。

    Returns
    -------
    ndarray of shape (n_labels,)
    """
    tp = (y_true * y_pred).sum(axis=0).astype(np.float64)
    fp = ((1 - y_true) * y_pred).sum(axis=0).astype(np.float64)
    fn = (y_true * (1 - y_pred)).sum(axis=0).astype(np.float64)
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    return np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)


# ===========================================================================
# 预计算函数
# ===========================================================================

def precompute_fire_masks(
    candidates: List[Predicate],
    docs: List[Document],
) -> np.ndarray:
    """预计算所有候选谓词在文档集上的触发掩码。

    返回 bool 矩阵 shape = (n_candidates, n_docs)。
    使用 joblib 并行化以加速大规模评估。
    """
    from joblib import Parallel, delayed

    n_cands = len(candidates)
    n_docs = len(docs)

    if n_cands == 0:
        return np.zeros((0, n_docs), dtype=bool)

    t0 = time.time()

    def _eval_pred(pred):
        row = np.zeros(n_docs, dtype=bool)
        for j, doc in enumerate(docs):
            try:
                row[j] = bool(pred(doc))
            except Exception:
                row[j] = False
        return row

    rows = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(_eval_pred)(p) for p in candidates
    )
    masks = np.stack(rows)

    logger.info("预计算 fire masks 完成: %d 谓词 × %d 文档 (%.1fs)",
                n_cands, n_docs, time.time() - t0)
    return masks


def precompute_freq_counts(
    candidates: List[Predicate],
    docs: List[Document],
) -> Dict[int, np.ndarray]:
    """预计算 FreqPredicate 候选的原始匹配计数。

    返回 {候选索引: count_array}，count_array shape = (n_docs,) int32。
    贪心搜索时，只需 `counts >= eta` 这样的向量化比较即可生成掩码，
    无需重新执行正则匹配或语义嵌入计算。
    """
    result: Dict[int, np.ndarray] = {}
    n_docs = len(docs)

    for idx, pred in enumerate(candidates):
        if not isinstance(pred, FreqPredicate):
            continue
        counts = np.zeros(n_docs, dtype=np.int32)
        for j, doc in enumerate(docs):
            try:
                text = doc.get_attr(pred.attr)
                if pred.sim:
                    # 语义匹配模式：按句切分后统计相似度超标的句子数
                    from loris.predicates._core import (
                        _split_sentences, _get_embedding, _cosine_similarity,
                    )
                    sents = _split_sentences(text)
                    emb_pat = _get_embedding(pred.r.sim_text or pred.r.raw)
                    counts[j] = sum(
                        1 for s in sents
                        if _cosine_similarity(emb_pat, _get_embedding(s)) >= pred.threshold
                    )
                else:
                    counts[j] = len(pred.r.findall(text))
            except Exception:
                counts[j] = 0
        result[idx] = counts

    logger.info("预计算 freq counts 完成: %d 个 FreqPredicate", len(result))
    return result


def precompute_ml_proba(
    ml_model_names: List[str],
    docs: List[Document],
) -> Dict[str, np.ndarray]:
    """预缓存 ML 模型在文档集上的概率矩阵。

    返回 {model_name: proba_matrix}，proba_matrix shape = (n_docs, n_classes) float32。
    贪心搜索时只需 `proba[:, label_idx] >= threshold` 即可生成掩码。
    """
    cache: Dict[str, np.ndarray] = {}
    texts = [doc.cnt for doc in docs]

    for model_name in ml_model_names:
        try:
            model = _get_ml_model(model_name)
            if hasattr(model, "_clf") and hasattr(model._clf, "predict_proba"):
                cache[model_name] = model._clf.predict_proba(texts).astype(np.float32)
            elif hasattr(model, "predict_proba_single"):
                proba = np.array(
                    [model.predict_proba_single(doc.cnt) for doc in docs],
                    dtype=np.float32,
                )
                cache[model_name] = proba
        except Exception as e:
            logger.warning("无法缓存模型 %s 的概率: %s", model_name, e)

    logger.info("预计算 ML proba 完成: %d 个模型", len(cache))
    return cache


# ===========================================================================
# 辅助函数
# ===========================================================================

def _group_candidates_by_type(
    candidates: List[Predicate],
) -> Dict[str, List[int]]:
    """将候选谓词索引按类型名称分组。"""
    pools: Dict[str, List[int]] = {}
    for i, pred in enumerate(candidates):
        type_name = type(pred).__name__
        pools.setdefault(type_name, []).append(i)
    return pools


def _make_proxy(doc: Document) -> Document:
    """创建文档代理副本，避免 LabelPredicate 副作用。"""
    return Document(
        cnt=doc.cnt,
        lbl=set(doc.lbl) if doc.lbl else set(),
        mtd=doc.mtd,
        ttl=doc.ttl,
    )


# ===========================================================================
# 贪心评分函数
# ===========================================================================

def _score_candidate(
    cand_mask: np.ndarray,
    conjunction_mask: np.ndarray,
    selected_masks: List[np.ndarray],
    labels: np.ndarray,
    label_idx: int,
    consequence_op: str,
    existing_preds: np.ndarray,
    weights: Dict[str, float],
) -> float:
    """对一个候选谓词打分，用于贪心选择。

    三项加权评分（全部基于 numpy bool 数组位运算，无字符串匹配）：
      1. 信息增益 (info_gain)：加入该谓词后标签分布熵的减少量
      2. Precision (precision)：合取触发文档中 consequence 标签正确率
      3. 多样性 (diversity)：1 - 与已选谓词掩码的平均 Jaccard 相似度

    Parameters
    ----------
    cand_mask : (n_docs,) bool — 候选谓词的触发掩码
    conjunction_mask : (n_docs,) bool — 当前已选谓词合取后的掩码
    selected_masks : 已选谓词的各自掩码列表
    labels : (n_docs, n_labels) float32 — 真实标签
    label_idx : consequence 标签在 labels 中的列索引
    consequence_op : "add" | "remove"
    existing_preds : (n_docs, n_labels) float32 — 当前预测
    weights : 三项权重字典

    Returns
    -------
    float : 综合评分，越高越好；覆盖为 0 时返回 -1.0
    """
    # 合取后的触发掩码 (bitwise AND)
    combined = conjunction_mask & cand_mask
    n_fire = int(combined.sum())
    if n_fire == 0:
        return -1.0

    n_parent = int(conjunction_mask.sum())
    if n_parent == 0:
        return -1.0

    # ------------------------------------------------------------------
    # 1. 纠错信息增益 (Correction Information Gain)
    #    只在模型**会被改变**的文档上计算熵，衡量候选谓词区分
    #    "纠正"与"恶化"的能力，而非区分真实标签本身。
    # ------------------------------------------------------------------
    eps = 1e-12

    # 构造纠错指示变量：该文档被规则改变后是否变好
    # correction_target=1 表示改变是正确的(improved)，=0 表示改变是错误的(worsened)
    parent_preds = existing_preds[conjunction_mask, label_idx]
    parent_labels = labels[conjunction_mask, label_idx]
    if consequence_op == "remove":
        # 只有 old_pred==1 的文档会被改变（改为 0）
        parent_changeable = (parent_preds == 1.0)
        # 改变后变好：truth==0 (FP→TN)；变差：truth==1 (TP→FN)
        parent_correct = parent_changeable & (parent_labels == 0.0)
    else:  # "add"
        parent_changeable = (parent_preds == 0.0)
        parent_correct = parent_changeable & (parent_labels == 1.0)

    n_parent_changeable = int(parent_changeable.sum())
    if n_parent_changeable == 0:
        # 父节点中没有可改变的文档 → info_gain 无意义
        norm_ig = 0.0
    else:
        # 父节点熵：在可改变文档中，纠正率的二元熵
        p_parent_corr = float(parent_correct.sum()) / n_parent_changeable
        h_parent = -(
            p_parent_corr * np.log2(p_parent_corr + eps)
            + (1 - p_parent_corr) * np.log2(1 - p_parent_corr + eps)
        )

        # 左子节点 (combined fires 中可改变的文档)
        comb_preds = existing_preds[combined, label_idx]
        comb_labels = labels[combined, label_idx]
        if consequence_op == "remove":
            comb_changeable = (comb_preds == 1.0)
            comb_correct = comb_changeable & (comb_labels == 0.0)
        else:
            comb_changeable = (comb_preds == 0.0)
            comb_correct = comb_changeable & (comb_labels == 1.0)
        n_left_ch = int(comb_changeable.sum())

        if n_left_ch > 0:
            p_left = float(comb_correct.sum()) / n_left_ch
            h_left = -(
                p_left * np.log2(p_left + eps)
                + (1 - p_left) * np.log2(1 - p_left + eps)
            )
        else:
            h_left = 0.0

        # 右子节点 (conjunction fires but candidate does not)
        right_mask = conjunction_mask & ~cand_mask
        right_preds = existing_preds[right_mask, label_idx]
        right_labels = labels[right_mask, label_idx]
        if consequence_op == "remove":
            right_changeable = (right_preds == 1.0)
            right_correct = right_changeable & (right_labels == 0.0)
        else:
            right_changeable = (right_preds == 0.0)
            right_correct = right_changeable & (right_labels == 1.0)
        n_right_ch = int(right_changeable.sum())

        if n_right_ch > 0:
            p_right = float(right_correct.sum()) / n_right_ch
            h_right = -(
                p_right * np.log2(p_right + eps)
                + (1 - p_right) * np.log2(1 - p_right + eps)
            )
        else:
            h_right = 0.0

        # 加权子熵（按可改变文档数加权）
        total_ch = n_left_ch + n_right_ch
        if total_ch > 0:
            h_child = (n_left_ch / total_ch) * h_left + (n_right_ch / total_ch) * h_right
        else:
            h_child = 0.0
        info_gain = max(h_parent - h_child, 0.0)
        # 归一化到 [0, 1]（二元熵最大为 1.0）
        norm_ig = min(info_gain / 1.0, 1.0)

    # ------------------------------------------------------------------
    # 2. Correction Precision（只看实际会被改变的文档中的纠正率）
    # ------------------------------------------------------------------
    old_preds = existing_preds[combined, label_idx]
    true_lbls = labels[combined, label_idx]

    if consequence_op == "remove":
        # 只有 model 当前预测为 1 的文档会被改为 0
        will_change = (old_preds == 1.0)
        improved = float((will_change & (true_lbls == 0.0)).sum())  # FP→TN
        worsened = float((will_change & (true_lbls == 1.0)).sum())  # TP→FN
    else:  # "add"
        # 只有 model 当前预测为 0 的文档会被改为 1
        will_change = (old_preds == 0.0)
        improved = float((will_change & (true_lbls == 1.0)).sum())  # FN→TP
        worsened = float((will_change & (true_lbls == 0.0)).sum())  # TN→FP

    n_changes = improved + worsened
    precision = improved / n_changes if n_changes > 0 else 0.0

    # ------------------------------------------------------------------
    # 3. 多样性 (1 - mean Jaccard overlap)
    # ------------------------------------------------------------------
    if selected_masks:
        overlaps = []
        for prev_mask in selected_masks:
            intersection = float((cand_mask & prev_mask).sum())
            union = float((cand_mask | prev_mask).sum())
            jaccard = intersection / union if union > 0 else 0.0
            overlaps.append(jaccard)
        avg_overlap = float(np.mean(overlaps))
    else:
        avg_overlap = 0.0

    diversity = 1.0 - avg_overlap

    # ------------------------------------------------------------------
    # 4. 覆盖率 (合取后仍覆盖多少文档，log 归一化)
    # ------------------------------------------------------------------
    n_total = len(conjunction_mask)
    coverage_ratio = np.log1p(n_fire) / np.log1p(n_total) if n_total > 0 else 0.0

    # ------------------------------------------------------------------
    # 综合评分
    # ------------------------------------------------------------------
    w = weights if weights else DEFAULT_GREEDY_WEIGHTS
    score = (
        w.get("info_gain", 0.2) * norm_ig
        + w.get("precision", 0.5) * precision
        + w.get("coverage", 0.2) * coverage_ratio
        + w.get("diversity", 0.1) * diversity
    )
    return score


# ===========================================================================
# 贪心实例化
# ===========================================================================

def greedy_instantiate(
    structure: Dict[str, int],
    candidate_pools: Dict[str, List[int]],
    candidates: List[Predicate],
    fire_masks: np.ndarray,
    freq_counts: Dict[int, np.ndarray],
    ml_proba_cache: Dict[str, np.ndarray],
    labels: np.ndarray,
    label_idx: int,
    consequence_op: str,
    existing_preds: np.ndarray,
    docs: List[Document],
    label_list: List[str],
    ml_model_names: List[str],
    weights: Optional[Dict[str, float]] = None,
    label_pred_masks: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
) -> Tuple[List[Predicate], List[np.ndarray], List[int]]:
    """在 Optuna 给定的宏观结构下，贪心选出最优的具体谓词实例。

    填充顺序：Text → ML → Label。
    Text 在全量空间搜索，低精度时自然 skip 不收缩 mask；ML 在宽空间搜索获得充足 fires。
    所有评分计算均基于预计算的布尔掩码位运算，严禁循环内做字符串匹配或模型推理。

    Parameters
    ----------
    label_pred_masks : optional precomputed {(label_name, op): mask}
        If provided, LabelPredicate evaluation uses precomputed masks
        instead of re-evaluating per doc.

    Returns
    -------
    (selected_predicates, selected_masks, selected_indices)
        selected_indices — index of each selected predicate in ``candidates``
        (textual/ML predicates), or -1 for LabelPredicates (not in candidates).
    """
    w = weights or DEFAULT_GREEDY_WEIGHTS
    n_docs = len(docs)
    n_labels = labels.shape[1] if labels.ndim > 1 else 1

    selected_preds: List[Predicate] = []
    selected_masks: List[np.ndarray] = []
    selected_indices: List[int] = []
    # 合取掩码：初始全 True，每选一个谓词后 AND 上该谓词的掩码
    conjunction_mask = np.ones(n_docs, dtype=bool)

    # ==============================================================
    # Phase 1: 文本谓词（按 _TEXT_TYPE_ORDER 由宽到窄框选）
    #   Match → Freq → Cooccur → Before
    #   NOTE: Text first — on BGC, text predicates usually score<=0
    #   and get skipped, so conjunction_mask stays ~100%.
    #   This ensures ML phase searches the full document space.
    # ==============================================================
    for type_name in _TEXT_TYPE_ORDER:
        n_slots = structure.get(type_name, 0)
        pool_indices = candidate_pools.get(type_name, [])
        if n_slots == 0 or not pool_indices:
            continue

        for _slot in range(n_slots):
            if conjunction_mask.sum() == 0:
                break  # 合取覆盖归零，无需继续

            best_pred: Optional[Predicate] = None
            best_score = -float("inf")
            best_mask: Optional[np.ndarray] = None
            best_idx: int = -1

            if type_name == "FreqPredicate":
                # ---------------------------------------------------------
                # FreqPredicate 特殊处理：遍历候选 × eta × op 网格
                # 使用预计算 counts 做向量化比较，不重新执行正则
                # ---------------------------------------------------------
                for idx in pool_indices:
                    orig = candidates[idx]
                    if not isinstance(orig, FreqPredicate):
                        continue
                    counts = freq_counts.get(idx)
                    if counts is None:
                        continue

                    for eta in FREQ_ETA_BINS:
                        for op_str in FREQ_OP_BINS:
                            # 向量化生成掩码（核心加速：无正则匹配）
                            if op_str == ">=":
                                variant_mask = counts >= eta
                            elif op_str == "<=":
                                variant_mask = counts <= eta
                            else:
                                variant_mask = _COMPARE_OPS[op_str](counts, eta)

                            # 提前退出：合取后覆盖 < 2 的候选不可能有意义
                            if int((conjunction_mask & variant_mask).sum()) < 2:
                                continue

                            score = _score_candidate(
                                variant_mask, conjunction_mask, selected_masks,
                                labels, label_idx, consequence_op,
                                existing_preds, w,
                            )
                            if score > best_score:
                                best_score = score
                                best_mask = variant_mask.copy()
                                best_idx = idx
                                # 构造变体谓词（仅记录参数，不做匹配）
                                best_pred = FreqPredicate(
                                    attr=orig.attr,
                                    r=orig.r,
                                    op=op_str,
                                    eta=float(eta),
                                    sim=orig.sim,
                                    threshold=orig.threshold,
                                )
            else:
                # ---------------------------------------------------------
                # Match / Cooccur / Before：直接使用预计算 fire_mask
                # ---------------------------------------------------------
                for idx in pool_indices:
                    cand_mask = fire_masks[idx]
                    # 提前退出：合取后覆盖 < 2 的候选不可能有意义
                    if int((conjunction_mask & cand_mask).sum()) < 2:
                        continue
                    score = _score_candidate(
                        cand_mask, conjunction_mask, selected_masks,
                        labels, label_idx, consequence_op,
                        existing_preds, w,
                    )
                    if score > best_score:
                        best_score = score
                        best_pred = candidates[idx]
                        best_mask = cand_mask
                        best_idx = idx

            # 选中最佳谓词
            if best_pred is not None and best_score > 0 and best_mask is not None:
                selected_preds.append(best_pred)
                selected_masks.append(best_mask)
                selected_indices.append(best_idx)
                conjunction_mask = conjunction_mask & best_mask
            else:
                break  # 该类型无正向增益，跳过剩余槽位

    # ==============================================================
    # Phase 2: ML 谓词（模型置信区域）
    #   Searches all K models × labels × thresholds.
    #   After text phase (which usually skips on BGC),
    #   conjunction_mask is still ~100%, giving ML full coverage.
    # ==============================================================
    n_ml_slots = structure.get("MLThresholdPredicate", 0)
    for _slot in range(n_ml_slots):
        if conjunction_mask.sum() == 0:
            break

        best_pred = None
        best_score = -float("inf")
        best_mask = None

        for model_name in ml_model_names:
            if model_name not in ml_proba_cache:
                continue
            proba = ml_proba_cache[model_name]  # (n_docs, n_model_labels)
            n_model_labels = proba.shape[1] if proba.ndim > 1 else 1

            # 遍历所有标签 × threshold 网格
            for l_idx in range(min(n_model_labels, len(label_list))):
                for threshold in ML_THRESHOLD_BINS:
                    # 向量化掩码生成（核心加速：无模型推理）
                    if proba.ndim > 1:
                        ml_mask = proba[:, l_idx] >= threshold
                    else:
                        ml_mask = proba >= threshold

                    score = _score_candidate(
                        ml_mask, conjunction_mask, selected_masks,
                        labels, label_idx, consequence_op,
                        existing_preds, w,
                    )
                    if score > best_score:
                        best_score = score
                        best_mask = ml_mask.copy()
                        best_pred = MLThresholdPredicate(
                            model_name=model_name,
                            label=label_list[l_idx] if l_idx < len(label_list) else str(l_idx),
                            threshold=threshold,
                        )

        if best_pred is not None and best_score > 0 and best_mask is not None:
            selected_preds.append(best_pred)
            selected_masks.append(best_mask)
            selected_indices.append(-1)  # ML predicates not in candidates list
            conjunction_mask = conjunction_mask & best_mask
        else:
            break

    # ==============================================================
    # Phase 3: Label 谓词
    # ==============================================================
    label_types = [
        ("LabelPredicate_contains", "contains"),
        ("LabelPredicate_eq", "eq"),
    ]
    for struct_key, lbl_op in label_types:
        n_slots = structure.get(struct_key, 0)
        if n_slots == 0:
            continue

        for _slot in range(n_slots):
            if conjunction_mask.sum() == 0:
                break

            best_pred = None
            best_score = -float("inf")
            best_mask = None

            for label_name in label_list:
                lp = LabelPredicate(label=label_name, op=lbl_op)
                # 优先使用预计算掩码，避免每 trial 重复逐文档 eval
                _lp_key = (label_name, lbl_op)
                if label_pred_masks and _lp_key in label_pred_masks:
                    lp_mask = label_pred_masks[_lp_key]
                else:
                    # fallback: 逐文档评估（向后兼容）
                    lp_mask = np.zeros(n_docs, dtype=bool)
                    for j, doc in enumerate(docs):
                        try:
                            proxy = _make_proxy(doc)
                            lp_mask[j] = bool(lp(proxy))
                        except Exception:
                            lp_mask[j] = False

                score = _score_candidate(
                    lp_mask, conjunction_mask, selected_masks,
                    labels, label_idx, consequence_op,
                    existing_preds, w,
                )
                if score > best_score:
                    best_score = score
                    best_pred = lp
                    best_mask = lp_mask

            if best_pred is not None and best_score > 0 and best_mask is not None:
                selected_preds.append(best_pred)
                selected_masks.append(best_mask)
                selected_indices.append(-1)  # LabelPredicate not in candidates
                conjunction_mask = conjunction_mask & best_mask
            else:
                break

    return selected_preds, selected_masks, selected_indices


def beam_instantiate(
    structure: Dict[str, int],
    candidate_pools: Dict[str, List[int]],
    candidates: List[Predicate],
    fire_masks: np.ndarray,
    freq_counts: Dict[int, np.ndarray],
    ml_proba_cache: Dict[str, np.ndarray],
    labels: np.ndarray,
    label_idx: int,
    consequence_op: str,
    existing_preds: np.ndarray,
    docs: List[Document],
    label_list: List[str],
    ml_model_names: List[str],
    weights: Optional[Dict[str, float]] = None,
    label_pred_masks: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    beam_width: int = 3,
) -> Tuple[List[Predicate], List[np.ndarray], List[int]]:
    """Beam search 版谓词实例化，替代 greedy_instantiate。

    保持 Text → ML → Label 填充顺序，但每步保留 top-K 候选路径，
    能发现"单独弱但组合强"的谓词对。

    Parameters
    ----------
    beam_width : int
        每步保留的最佳路径数（默认 3）。

    Returns
    -------
    (selected_predicates, selected_masks, selected_indices)
    """
    w = weights or DEFAULT_GREEDY_WEIGHTS
    n_docs = len(docs)

    # Beam entry: (total_score, preds, masks, indices, conjunction_mask)
    BeamEntry = Tuple[float, List[Predicate], List[np.ndarray], List[int], np.ndarray]
    initial_beam: BeamEntry = (0.0, [], [], [], np.ones(n_docs, dtype=bool))
    beams: List[BeamEntry] = [initial_beam]

    # ── Build phase list: each entry = (phase_type, phase_info) ──
    # phase_type: "text_freq", "text_other", "ml", "label"
    phases: List[Tuple[str, Any]] = []
    for type_name in _TEXT_TYPE_ORDER:
        n_slots = structure.get(type_name, 0)
        pool_indices = candidate_pools.get(type_name, [])
        if n_slots == 0 or not pool_indices:
            continue
        for _ in range(n_slots):
            if type_name == "FreqPredicate":
                phases.append(("text_freq", (type_name, pool_indices)))
            else:
                phases.append(("text_other", (type_name, pool_indices)))

    n_ml_slots = structure.get("MLThresholdPredicate", 0)
    for _ in range(n_ml_slots):
        phases.append(("ml", None))

    label_types = [
        ("LabelPredicate_contains", "contains"),
        ("LabelPredicate_eq", "eq"),
    ]
    for struct_key, lbl_op in label_types:
        n_slots = structure.get(struct_key, 0)
        for _ in range(n_slots):
            phases.append(("label", lbl_op))

    # ── Iterate phases ──
    for phase_type, phase_info in phases:
        next_beams: List[BeamEntry] = []

        for (beam_score, preds, masks, indices, conj) in beams:
            if conj.sum() == 0:
                # Dead beam — carry forward unchanged
                next_beams.append((beam_score, preds, masks, indices, conj))
                continue

            # Collect top candidates for this beam
            expansions: List[BeamEntry] = []

            if phase_type == "text_freq":
                _type_name, pool_indices = phase_info
                for idx in pool_indices:
                    orig = candidates[idx]
                    if not isinstance(orig, FreqPredicate):
                        continue
                    counts = freq_counts.get(idx)
                    if counts is None:
                        continue
                    for eta in FREQ_ETA_BINS:
                        for op_str in FREQ_OP_BINS:
                            if op_str == ">=":
                                variant_mask = counts >= eta
                            elif op_str == "<=":
                                variant_mask = counts <= eta
                            else:
                                variant_mask = _COMPARE_OPS[op_str](counts, eta)
                            if int((conj & variant_mask).sum()) < 2:
                                continue
                            score = _score_candidate(
                                variant_mask, conj, masks,
                                labels, label_idx, consequence_op,
                                existing_preds, w,
                            )
                            if score > 0:
                                new_pred = FreqPredicate(
                                    attr=orig.attr, r=orig.r,
                                    op=op_str, eta=float(eta),
                                    sim=orig.sim, threshold=orig.threshold,
                                )
                                expansions.append((
                                    beam_score + score,
                                    preds + [new_pred],
                                    masks + [variant_mask.copy()],
                                    indices + [idx],
                                    conj & variant_mask,
                                ))

            elif phase_type == "text_other":
                _type_name, pool_indices = phase_info
                for idx in pool_indices:
                    cand_mask = fire_masks[idx]
                    if int((conj & cand_mask).sum()) < 2:
                        continue
                    score = _score_candidate(
                        cand_mask, conj, masks,
                        labels, label_idx, consequence_op,
                        existing_preds, w,
                    )
                    if score > 0:
                        expansions.append((
                            beam_score + score,
                            preds + [candidates[idx]],
                            masks + [cand_mask],
                            indices + [idx],
                            conj & cand_mask,
                        ))

            elif phase_type == "ml":
                for model_name in ml_model_names:
                    if model_name not in ml_proba_cache:
                        continue
                    proba = ml_proba_cache[model_name]
                    n_model_labels = proba.shape[1] if proba.ndim > 1 else 1
                    for l_idx in range(min(n_model_labels, len(label_list))):
                        for threshold in ML_THRESHOLD_BINS:
                            if proba.ndim > 1:
                                ml_mask = proba[:, l_idx] >= threshold
                            else:
                                ml_mask = proba >= threshold
                            if int((conj & ml_mask).sum()) < 2:
                                continue
                            score = _score_candidate(
                                ml_mask, conj, masks,
                                labels, label_idx, consequence_op,
                                existing_preds, w,
                            )
                            if score > 0:
                                expansions.append((
                                    beam_score + score,
                                    preds + [MLThresholdPredicate(
                                        model_name=model_name,
                                        label=label_list[l_idx] if l_idx < len(label_list) else str(l_idx),
                                        threshold=threshold,
                                    )],
                                    masks + [ml_mask.copy()],
                                    indices + [-1],
                                    conj & ml_mask,
                                ))

            elif phase_type == "label":
                lbl_op = phase_info
                for label_name in label_list:
                    lp = LabelPredicate(label=label_name, op=lbl_op)
                    _lp_key = (label_name, lbl_op)
                    if label_pred_masks and _lp_key in label_pred_masks:
                        lp_mask = label_pred_masks[_lp_key]
                    else:
                        lp_mask = np.zeros(n_docs, dtype=bool)
                        for j, doc in enumerate(docs):
                            try:
                                proxy = _make_proxy(doc)
                                lp_mask[j] = bool(lp(proxy))
                            except Exception:
                                lp_mask[j] = False
                    if int((conj & lp_mask).sum()) < 2:
                        continue
                    score = _score_candidate(
                        lp_mask, conj, masks,
                        labels, label_idx, consequence_op,
                        existing_preds, w,
                    )
                    if score > 0:
                        expansions.append((
                            beam_score + score,
                            preds + [lp],
                            masks + [lp_mask],
                            indices + [-1],
                            conj & lp_mask,
                        ))

            # Also keep the "skip this slot" option (beam unchanged)
            expansions.append((beam_score, preds, masks, indices, conj))

            next_beams.extend(expansions)

        # Keep top beam_width entries
        next_beams.sort(key=lambda x: -x[0])
        beams = next_beams[:beam_width]

    # Return the best beam
    if not beams:
        return [], [], []
    best = beams[0]
    return best[1], best[2], best[3]


# ===========================================================================
# 合取掩码计算
# ===========================================================================

def _compute_conjunction_mask(
    body: List[Predicate],
    body_masks: List[np.ndarray],
    docs: List[Document],
) -> np.ndarray:
    """从 body 谓词列表及其掩码列表计算最终合取掩码。

    如果 body_masks 已经提供且与 body 长度一致，直接 AND；
    否则回退到逐文档评估（用于 batch_select 重构场景）。
    """
    n_docs = len(docs)
    if body_masks and len(body_masks) == len(body):
        mask = np.ones(n_docs, dtype=bool)
        for m in body_masks:
            mask &= m
        return mask

    # 回退：逐文档评估
    mask = np.zeros(n_docs, dtype=bool)
    for j, doc in enumerate(docs):
        proxy = _make_proxy(doc)
        mask[j] = all(p(proxy) for p in body)
    return mask


# ===========================================================================
# Optuna 目标函数 (Level 1)
# ===========================================================================

def _wilson_lb(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound — sample-size-aware precision (A2/LBoost).
    100%-on-1-fire scores far below 95%-on-50; downweights rare patterns."""
    if n <= 0:
        return 0.0
    p = k / n
    z2 = z * z
    return (p + z2 / (2 * n) - z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)) / (1 + z2 / n)


def evaluate_chase_configuration(
    trial: optuna.Trial,
    candidates: List[Predicate],
    candidate_pools: Dict[str, List[int]],
    fire_masks: np.ndarray,
    freq_counts: Dict[int, np.ndarray],
    ml_model_names: List[str],
    ml_proba_cache: Dict[str, np.ndarray],
    label_list: List[str],
    val_docs: List[Document],
    val_labels: np.ndarray,
    existing_predictions: np.ndarray,
    baseline_f1: float,
    min_coverage: float = 0.05,
    rule_min_precision: float = 0.0,
    greedy_weights: Optional[Dict[str, float]] = None,
    metric_mode: str = "global_macro",
    cluster_label_indices: Optional[List[int]] = None,
    # ── split mode: 谓词实例化用 fit_* (train), 评估仍用 val_* ──
    fit_docs: Optional[List[Document]] = None,
    fit_labels: Optional[np.ndarray] = None,
    fit_fire_masks: Optional[np.ndarray] = None,
    fit_freq_counts: Optional[Dict] = None,
    fit_ml_proba_cache: Optional[Dict] = None,
    fit_existing_preds: Optional[np.ndarray] = None,
    val_label_pred_masks: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    fit_label_pred_masks: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    min_rule_fires: int = 1,
    min_n_changes: int = 1,
    min_corr_prec: float = 0.0,
    _body_cache: Optional[Dict[int, Tuple]] = None,
    _structure_cache: Optional[Dict[Tuple, Tuple]] = None,
    # ── label sampling ──
    add_label_names: Optional[List[str]] = None,
    # ── Chase-specific ──
    track: str = "base",
    sim_graphs: Optional[Dict] = None,
    neighbor_label_masks: Optional[Dict] = None,
    neighbor_label_counts: Optional[Dict] = None,
    label_state: Optional[np.ndarray] = None,
    sim_cascade_rounds: int = 3,
    sim_min_precision: float = 0.75,
    sim_self_loop_min_prec: float = -1.0,  # -1 = 使用 max(sim_min_precision, 0.85)
    label_prec_objective: bool = False,  # True = label 规则也用 precision 导向目标
    prop_admit_on_precision: bool = False,  # LBoost-style: admit precise sim/label propagation rules even without staged F1-gain (value shows under RILL seeds)
    # ── REMOVE rules (staged pipeline) ──
    consequence_op_mode: str = "add",  # "add" / "remove" / "both"
    remove_label_names: Optional[List[str]] = None,
    # ── Weak-label rescue (weighted sampling + relaxed constraints) ──
    label_weights: Optional[Dict[str, float]] = None,
    weak_labels: Optional[Set[str]] = None,
) -> float:
    """Optuna 目标函数：建议宏观结构 → 贪心实例化 → 评估 F1 增益。

    搜索空间仅 ~12 个参数（每种谓词类型数量 + consequence），
    具体谓词选择由 greedy_instantiate 在预计算掩码上完成。

    Split mode (fit_docs is not None):
        greedy_instantiate 在 fit_* (训练集) 上选择谓词，
        F1 评估在 val_* (验证集) 上进行。

    Returns
    -------
    float : F1 增益（正值表示规则有效），负值表示被剪枝
    """
    _split = fit_docs is not None
    n_val = len(val_docs)
    n_labels = len(label_list)

    # ==============================================================
    # Step 1: Suggest 宏观结构 — Chase: track-aware + add-only
    # ==============================================================
    structure: Dict[str, int] = {}

    # 文本谓词：Match/Freq [0,2], Cooccur/Before [0,1]
    _PAIR_TYPES = {"CooccurPredicate", "BeforePredicate"}
    for type_name in _TEXT_TYPE_ORDER:
        pool = candidate_pools.get(type_name, [])
        cap = 1 if type_name in _PAIR_TYPES else 2
        hi = min(cap, len(pool)) if pool else 0
        if hi > 0:
            structure[type_name] = trial.suggest_int(f"n_{type_name}", 0, hi)
        else:
            structure[type_name] = 0

    # ML 谓词: [0, 2] (or [0, 3] when weak labels exist)
    if ml_model_names and ml_proba_cache:
        _ml_cap = 3 if weak_labels else 2
        structure["MLThresholdPredicate"] = trial.suggest_int("n_ml", 0, _ml_cap)
    else:
        structure["MLThresholdPredicate"] = 0

    # ── Track-dependent predicate control (Action 3) ──
    _has_sim = None  # only meaningful for track=="propagation"
    if track == "base":
        # Track 1: no label/sim predicates
        structure["LabelPredicate_contains"] = 0
        structure["LabelPredicate_eq"] = 0
        structure["LabelPredicate_minus"] = 0
    elif track == "propagation":
        # Track 2: BO chooses has_sim
        #   has_sim=1 → sim(x,y) ∧ label(τ∈y.lbl) ∧ [text/ML] → add σ to x
        #   has_sim=0 → label(τ∈x.lbl) ∧ [text/ML]           → add σ to x
        _has_sim = trial.suggest_int("has_sim", 0, 1)
        structure["LabelPredicate_contains"] = 1  # always 1 label predicate
        structure["LabelPredicate_eq"] = 0
        structure["LabelPredicate_minus"] = 0
    elif track == "remove":
        # Stage 2: REMOVE rules — force ML predicate, no label/sim
        if not ml_model_names or not ml_proba_cache:
            return -1.0
        structure["MLThresholdPredicate"] = max(structure.get("MLThresholdPredicate", 0), 1)
        structure["LabelPredicate_contains"] = 0
        structure["LabelPredicate_eq"] = 0
        structure["LabelPredicate_minus"] = 0
    else:
        # fallback (original behavior)
        structure["LabelPredicate_contains"] = trial.suggest_int("n_label_contains", 0, 2)
        structure["LabelPredicate_eq"] = 0
        structure["LabelPredicate_minus"] = 0

    total_slots = sum(structure.values())

    # Track "propagation" will add SimPredicate (has_sim=1) or LabelPredicate (has_sim=0)
    if track == "propagation":
        if _has_sim == 1:
            # SimPredicate + LabelPredicate guaranteed → always have predicates
            pass
        else:
            # has_sim=0: need at least one text or ML predicate alongside label(x)
            n_text = sum(structure.get(t, 0) for t in _TEXT_TYPE_ORDER)
            n_ml = structure.get("MLThresholdPredicate", 0)
            if n_text + n_ml == 0:
                _diag_inc("no_text_or_ml_predicate")
                return -1.0
    elif total_slots == 0:
        _diag_inc("empty_body")
        return -1.0

    # Track "base" or "remove": require at least one text or ML predicate
    if track in ("base", "remove"):
        n_text = sum(structure.get(t, 0) for t in _TEXT_TYPE_ORDER)
        n_ml = structure.get("MLThresholdPredicate", 0)
        if n_text + n_ml == 0:
            _diag_inc("no_text_or_ml_predicate")
            return -1.0

    # ── Consequence — Action 4: parameterized op (add/remove/both) ──
    # B10 fix: make the consequence op + label real Optuna params so the TPE
    # sampler optimises WHICH label/op to fix — previously they were RNG'd on
    # trial.number (pure random search over the single most impactful decision,
    # wasting trials on un-improvable labels). Static categorical choices (from
    # fixed inputs) keep the search space constant so multivariate TPE can model
    # them. Also makes params['consequence_label'/'consequence_op'] exist, fixing
    # the latent _trial_to_rdl KeyError on a _body_cache miss.
    if consequence_op_mode == "both":
        consequence_op = trial.suggest_categorical("consequence_op", ["add", "remove"])
    elif consequence_op_mode == "remove":
        consequence_op = trial.suggest_categorical("consequence_op", ["remove"])
    else:
        consequence_op = trial.suggest_categorical("consequence_op", ["add"])

    _cand_labels: List[str] = []
    if add_label_names:
        _cand_labels += list(add_label_names)
    if remove_label_names:
        _cand_labels += list(remove_label_names)
    if cluster_label_indices is not None and len(cluster_label_indices) > 0:
        _cand_labels += [label_list[i] for i in cluster_label_indices]
    if not _cand_labels:
        _cand_labels = list(label_list)
    _cand_labels = sorted(dict.fromkeys(_cand_labels))      # dedup, deterministic
    consequence_label = trial.suggest_categorical("consequence_label", _cand_labels)
    label_idx = label_list.index(consequence_label)

    _rng = np.random.RandomState(trial.number + 100003)     # kept for feature dropout

    # Landmine disarm (audit L1): the optional structure-cache writes below read
    # `_cache_key`, which was never assigned → a latent NameError the moment a
    # caller passes a non-None `_structure_cache`. Define it here. Keyed by the
    # structure + consequence so the cache is also CORRECT if ever enabled.
    _cache_key = (tuple(sorted(structure.items())), label_idx, consequence_op)

    # ── Feature Dropout: 每 trial 随机子采样 80% 谓词候选，打破 greedy 确定性 ──
    _dropout_pools = {}
    for _type, _indices in candidate_pools.items():
        if len(_indices) <= 3:
            _dropout_pools[_type] = _indices
        else:
            _k = max(3, int(len(_indices) * 0.8))
            _dropout_pools[_type] = sorted(_rng.choice(_indices, size=_k, replace=False).tolist())

    # ==============================================================
    # Step 2: 贪心实例化（Level 2 — 位运算加速）
    # 如果 split mode，在 fit (train) 数据上实例化谓词
    # ==============================================================
    if _split:
        body, _fit_body_masks, _sel_indices = beam_instantiate(
            structure=structure,
            candidate_pools=_dropout_pools,
            candidates=candidates,
            fire_masks=fit_fire_masks,
            freq_counts=fit_freq_counts,
            ml_proba_cache=fit_ml_proba_cache or ml_proba_cache,
            labels=fit_labels,
            label_idx=label_idx,
            consequence_op=consequence_op,
            existing_preds=fit_existing_preds if fit_existing_preds is not None else existing_predictions,
            docs=fit_docs,
            label_list=label_list,
            ml_model_names=ml_model_names,
            weights=greedy_weights,
            label_pred_masks=fit_label_pred_masks,
        )
    else:
        body, _body_masks, _sel_indices = beam_instantiate(
            structure=structure,
            candidate_pools=_dropout_pools,
            candidates=candidates,
            fire_masks=fire_masks,
            freq_counts=freq_counts,
            ml_proba_cache=ml_proba_cache,
            labels=val_labels,
            label_idx=label_idx,
            consequence_op=consequence_op,
            existing_preds=existing_predictions,
            docs=val_docs,
            label_list=label_list,
            ml_model_names=ml_model_names,
            weights=greedy_weights,
            label_pred_masks=val_label_pred_masks,
        )

    # ── Track "propagation": inject SimPredicate (has_sim=1) or label(x) mask (has_sim=0) ──
    _sim_fire_mask = None  # will be set if track==propagation and has_sim=1
    _label_x_fire_mask = None  # will be set if track==propagation and has_sim=0
    _is_cross_label = False  # cross-label bonus only for Track 2 propagation rules
    if track == "propagation" and _has_sim == 1 and sim_graphs and neighbor_label_masks:
        # ── has_sim=1: sim(x,y) ∧ label(τ∈y.lbl) ∧ [text/ML] → add σ ──
        available_thresholds = sorted(sim_graphs.keys())
        sim_thresh = trial.suggest_categorical(
            "sim_threshold", [str(t) for t in available_thresholds],
        )
        sim_thresh = float(sim_thresh)
        sim_pred = SimPredicate(threshold=sim_thresh)

        # The LabelPredicate in body was already selected by greedy (Phase 3).
        # Find it to compute the SpMV mask.
        lp_in_body = [p for p in body if isinstance(p, LabelPredicate)]
        if not lp_in_body:
            # Safety: suggest a label for propagation
            lp_label = trial.suggest_categorical("lp_label", label_list)
            lp = LabelPredicate(label=lp_label, op="contains")
            body.append(lp)
            lp_in_body = [lp]

        # Insert SimPredicate at front of body
        body.insert(0, sim_pred)

        # Compute the pairwise fire mask via neighbor counts (if available)
        # or fall back to boolean masks
        _sim_min_neighbors = trial.suggest_int("sim_min_neighbors", 1, 5) \
            if neighbor_label_counts else 1
        _sim_fire_mask = np.ones(n_val, dtype=bool)
        for lp in lp_in_body:
            key = (lp.label, sim_thresh)
            if neighbor_label_counts and key in neighbor_label_counts:
                _sim_fire_mask &= (neighbor_label_counts[key] >= _sim_min_neighbors)
            elif key in neighbor_label_masks:
                _sim_fire_mask &= neighbor_label_masks[key]
            else:
                _sim_fire_mask[:] = False
                break

        # ── Cross-label bonus flag ──
        _is_cross_label = not (len(lp_in_body) == 1
                               and lp_in_body[0].label == consequence_label)

    elif track == "propagation" and _has_sim == 0 and label_state is not None:
        # ── has_sim=0: label(τ∈x.lbl) ∧ [text/ML] → add σ ──
        # LabelPredicate checks x's own labels via label_state
        lp_in_body = [p for p in body if isinstance(p, LabelPredicate)]
        if not lp_in_body:
            lp_label = trial.suggest_categorical("lp_label", label_list)
            lp = LabelPredicate(label=lp_label, op="contains")
            body.append(lp)
            lp_in_body = [lp]

        # Compute label(x) fire mask from label_state
        _label_x_fire_mask = np.ones(n_val, dtype=bool)
        for lp in lp_in_body:
            if lp.label in label_list:
                lp_idx = label_list.index(lp.label)
                _label_x_fire_mask &= (label_state[:, lp_idx] > 0)
            else:
                _label_x_fire_mask[:] = False
                break

        # Cross-label bonus
        _is_cross_label = not (len(lp_in_body) == 1
                               and lp_in_body[0].label == consequence_label)

    if not body:
        _diag_inc("greedy_empty")
        if _structure_cache is not None and _cache_key is not None:
            _structure_cache[_cache_key] = (-1.0, None)
        return -1.0  # 贪心未选出任何谓词

    # ==============================================================
    # Step 3: 计算合取触发掩码（bitwise AND）
    # 评估始终在 val_docs 上
    # ==============================================================
    if _sim_fire_mask is not None:
        # Track "propagation" has_sim=1: combine sim fire mask with text predicate masks
        # Text predicates on x are from greedy; sim/label are from SpMV
        text_fire = np.ones(n_val, dtype=bool)
        for pred in body:
            if isinstance(pred, (SimPredicate, LabelPredicate)):
                continue
            idx_in_greedy = [i for i, p in enumerate(body) if p is pred]
            if idx_in_greedy:
                gi = idx_in_greedy[0]
                # Adjust for inserted SimPredicate at front
                greedy_idx = gi - 1  # SimPredicate was inserted at 0
                if _split:
                    if greedy_idx >= 0 and greedy_idx < len(_sel_indices):
                        sidx = _sel_indices[greedy_idx]
                        if sidx >= 0:
                            text_fire &= fire_masks[sidx]
                            continue
                else:
                    if greedy_idx >= 0 and greedy_idx < len(_body_masks):
                        text_fire &= _body_masks[greedy_idx]
                        continue
            # Fallback: per-doc eval
            text_fire &= np.array([bool(pred(_make_proxy(d))) for d in val_docs])
        fires = text_fire & _sim_fire_mask

        # ── Cascade simulation: simulate multi-round sim propagation ──
        if sim_cascade_rounds > 1 and lp_in_body and sim_graphs:
            cascaded_preds = existing_predictions.copy()
            if consequence_op == "add":
                cascaded_preds[fires, label_idx] = 1.0
            for _cr in range(sim_cascade_rounds - 1):
                new_sim_mask = np.ones(n_val, dtype=bool)
                for lp in lp_in_body:
                    lp_idx = label_list.index(lp.label) if lp.label in label_list else -1
                    if lp_idx < 0:
                        new_sim_mask[:] = False
                        break
                    col = (cascaded_preds[:, lp_idx] > 0).astype(np.float32)
                    has_neighbor = np.asarray(
                        (sim_graphs[sim_thresh] @ col) > 0
                    ).ravel()
                    new_sim_mask &= has_neighbor
                new_fires = text_fire & new_sim_mask
                if new_fires.sum() == fires.sum():
                    break
                _delta = int(new_fires.sum() - fires.sum())
                _delta_correct = int((val_labels[new_fires, label_idx] == 1).sum() -
                                     (val_labels[fires, label_idx] == 1).sum())
                _diag_inc(f"sim_cascade_round_{_cr+2}(fires={int(new_fires.sum())},delta={_delta},delta_correct={_delta_correct})")
                fires = new_fires
                if consequence_op == "add":
                    cascaded_preds[fires, label_idx] = 1.0
    elif _label_x_fire_mask is not None:
        # Track "propagation" has_sim=0: combine label(x) mask with text/ML predicate masks
        # No SimPredicate inserted → greedy indices are aligned with body indices
        text_fire = np.ones(n_val, dtype=bool)
        for pred in body:
            if isinstance(pred, LabelPredicate):
                continue  # handled by _label_x_fire_mask
            idx_in_greedy = [i for i, p in enumerate(body) if p is pred]
            if idx_in_greedy:
                gi = idx_in_greedy[0]
                if _split:
                    if gi < len(_sel_indices):
                        sidx = _sel_indices[gi]
                        if sidx >= 0:
                            text_fire &= fire_masks[sidx]
                            continue
                else:
                    if gi < len(_body_masks):
                        text_fire &= _body_masks[gi]
                        continue
            text_fire &= np.array([bool(pred(_make_proxy(d))) for d in val_docs])
        fires = text_fire & _label_x_fire_mask
    elif _split:
        # 利用 _sel_indices 从 val fire_masks 重建 val body masks，走 fast path
        _val_body_masks = []
        for pred, sidx in zip(body, _sel_indices):
            if sidx >= 0:
                _val_body_masks.append(fire_masks[sidx])
            elif isinstance(pred, LabelPredicate) and val_label_pred_masks:
                _lk = (pred.label, pred.op)
                if _lk in val_label_pred_masks:
                    _val_body_masks.append(val_label_pred_masks[_lk])
                else:
                    _val_body_masks.append(
                        np.array([bool(pred(_make_proxy(d))) for d in val_docs])
                    )
            else:
                _val_body_masks.append(
                    np.array([bool(pred(_make_proxy(d))) for d in val_docs])
                )
        fires = _compute_conjunction_mask(body, _val_body_masks, val_docs)
    else:
        fires = _compute_conjunction_mask(body, _body_masks, val_docs)
    n_fire = int(fires.sum())
    coverage = n_fire / n_val if n_val > 0 else 0.0

    # ==============================================================
    # Step 4: Soft fires penalty (替代 hard rejection，给 TPE 连续梯度)
    # ==============================================================
    if n_fire == 0:
        _diag_inc("zero_fires")
        if _structure_cache is not None and _cache_key is not None:
            _structure_cache[_cache_key] = (-1.0, None)
        return -1.0  # 完全不触发 → 无信息
    # fires_factor: Sigmoid 软阈值 — 远低于 min_rule_fires 时 ≈0，远高于时 ≈1
    # 给 TPE 平滑的连续梯度，而非线性衰减
    import math as _math
    # 自适应 min_rule_fires: 稀有 label 放宽阈值
    _label_pos = int(val_labels[:, label_idx].sum())
    _adaptive_mrf = max(3, min(min_rule_fires, int(_label_pos * 0.1)))
    # 3+ predicate rules: raise floor to guard against overfitting
    n_body = len(body)
    if n_body >= 3:
        _adaptive_mrf = max(_adaptive_mrf, 3 if n_body == 3 else 4)
    _mrf = max(_adaptive_mrf, 1)
    _k = 4.0 / _mrf  # 过渡陡度：在 ±min_fires 范围内完成 0→1 过渡
    fires_factor = 1.0 / (1.0 + _math.exp(-_k * (n_fire - _mrf)))

    # ==============================================================
    # Step 5: 模拟应用规则，计算新预测 (在 val 上)
    # ==============================================================
    new_predictions = existing_predictions.copy()
    if consequence_op == "add":
        new_predictions[fires, label_idx] = 1.0
    elif consequence_op == "remove":
        new_predictions[fires, label_idx] = 0.0
    # ==============================================================
    # Step 6: Correction precision 评估
    # ==============================================================
    corr_prec = 0.0
    n_changes = 0
    n_improved = 0
    n_worsened = 0
    changes_factor = 1.0
    if n_fire > 0:
        old_hit = existing_predictions[fires, label_idx] == val_labels[fires, label_idx]
        new_hit = new_predictions[fires, label_idx] == val_labels[fires, label_idx]
        n_improved = int((new_hit & ~old_hit).sum())
        n_worsened = int((old_hit & ~new_hit).sum())
        n_changes = n_improved + n_worsened

        if n_changes == 0:
            _diag_inc(f"no_changes(fires={n_fire})")
            _soft = -0.05 * min(n_fire / 10.0, 1.0)
            if _structure_cache is not None and _cache_key is not None:
                _structure_cache[_cache_key] = (_soft, None)
            return _soft

        # Soft changes penalty (替代 hard min_n_changes rejection)
        changes_factor = min(1.0, n_changes / max(min_n_changes, 1))

        corr_prec = n_improved / (n_improved + n_worsened + 1)  # Laplace+1
        # 硬性拒绝：完全无正纠正
        if n_improved == 0:
            _diag_inc(f"zero_improved(worsened={n_worsened})")
            if _structure_cache is not None and _cache_key is not None:
                _structure_cache[_cache_key] = (-0.5, None)
            return -0.5
        # 硬性精度门槛（按 label support 自适应；REMOVE 规则不降级）
        if consequence_op == "remove":
            _eff_min_prec = min_corr_prec
        else:
            _eff_min_prec = _adaptive_corr_prec(min_corr_prec, _label_pos)
        if corr_prec < _eff_min_prec:
            _diag_inc(f"low_corr_prec({corr_prec:.2f}<{_eff_min_prec:.2f})")
            if _structure_cache is not None and _cache_key is not None:
                _structure_cache[_cache_key] = (-0.5, None)
            return -0.5

    # ==============================================================
    # Step 7: Target-label reward
    # ==============================================================
    # 复杂度惩罚：body > 2 个谓词开始衰减 (Occam's razor)
    # 弱 label 使用更宽松的系数，允许更长的 conjunction
    n_body = len(body)
    _is_weak = bool(weak_labels and consequence_label in weak_labels)
    _len_base = 0.93 if _is_weak else 0.85
    length_penalty = _len_base ** max(0, n_body - 1)

    # 跨标签 bonus：Track 2 中跨标签传播规则比同标签更稀有更有价值
    cross_label_bonus = 1.3 if _is_cross_label else 1.0

    if consequence_op == "remove":
        # === REMOVE 规则专用目标：precision 导向 ===
        # n_improved = FP→TN (成功删除 FP)
        # n_worsened = TP→FN (误删 TP, 代价极高)
        remove_raw_prec = n_improved / max(n_improved + n_worsened, 1)
        _remove_min_prec = max(min_corr_prec, 0.65 if _is_weak else 0.85)
        if n_improved < 2 or remove_raw_prec < _remove_min_prec:
            _diag_inc(f"remove_low_prec(raw={remove_raw_prec:.3f},thr={_remove_min_prec},fires={n_fire},imp={n_improved},wor={n_worsened})")
            if _structure_cache is not None and _cache_key is not None:
                _structure_cache[_cache_key] = (-0.5, None)
            return -0.5

        f1_gain = ((remove_raw_prec ** 3)
                   * np.sqrt(max(n_improved, 1))
                   * length_penalty * fires_factor * changes_factor)

        _diag_inc(f"remove_positive(gain={f1_gain:.4f},raw_prec={remove_raw_prec:.3f},fires={n_fire},imp={n_improved},wor={n_worsened},body={n_body})")

    elif _sim_fire_mask is not None:
        # === Sim 规则：F1 增益目标 + 精度保护 ===
        sim_raw_prec = n_improved / max(n_improved + n_worsened, 1)
        if not _is_cross_label:
            _sim_prec_thr = max(sim_min_precision, 0.85) if sim_self_loop_min_prec < 0 else sim_self_loop_min_prec
        else:
            _sim_prec_thr = sim_min_precision
        if n_improved < 2 or sim_raw_prec < _sim_prec_thr:
            _diag_inc(f"sim_low_prec(raw={sim_raw_prec:.3f},lap={corr_prec:.3f},thr={_sim_prec_thr},self_loop={not _is_cross_label},fires={n_fire},imp={n_improved},wor={n_worsened})")
            if _structure_cache is not None and _cache_key is not None:
                _structure_cache[_cache_key] = (-0.5, None)
            return -0.5

        all_new_f1s = _fast_per_label_f1(val_labels, new_predictions)
        all_old_f1s = _fast_per_label_f1(val_labels, existing_predictions)
        target_gain = float(all_new_f1s[label_idx] - all_old_f1s[label_idx])

        if target_gain <= 0:
            _diag_inc(f"sim_neg_f1_gain({target_gain:.6f},raw_prec={sim_raw_prec:.3f},fires={n_fire},imp={n_improved},wor={n_worsened})")
            if not prop_admit_on_precision:
                return target_gain
            # LBoost-style admission: a high-precision propagation rule that does
            # NOT move the zero-budget staged F1 (the strong base already covers
            # those docs) is still admitted on confidence × coverage — its value
            # shows under the RILL human-seed sweep, not the staged metric. The
            # tiny scale keeps it strictly below any true-F1-gain rule.
            # A2 (Wilson): require a sound small-sample confidence lower bound
            # (blocks 100%-on-1-fire artifacts).
            if _wilson_lb(n_improved, n_fire) < 0.60:
                return target_gain
            return 1e-3 * (corr_prec ** 2.0) * float(np.log1p(n_improved))

        precision_multiplier = corr_prec ** 2.0
        coverage_bonus = np.log1p(n_improved) / 3.0

        # A1 (LBoost confidence): corr_prec is fire-count-invariant, so reward
        # well-supported rules — 95% precision on 20 fires outranks 95% on 2.
        conf_mult = min(1.0, (n_improved / max(n_fire, 1)) / 0.75)
        f1_gain = (target_gain * precision_multiplier
                   * (1.0 + coverage_bonus) * length_penalty
                   * fires_factor * changes_factor * cross_label_bonus
                   * conf_mult)

        _diag_inc(f"sim_positive(gain={f1_gain:.4f},tgt={target_gain:.4f},raw_prec={sim_raw_prec:.3f},self_loop={not _is_cross_label},fires={n_fire},ff={fires_factor:.2f},imp={n_improved},wor={n_worsened},body={n_body})")

    elif label_prec_objective and _label_x_fire_mask is not None:
        # === Label 规则：F1 增益目标 + 精度保护 ===
        label_raw_prec = n_improved / max(n_improved + n_worsened, 1)
        if n_improved < 2 or label_raw_prec < sim_min_precision:
            _diag_inc(f"label_low_prec(raw={label_raw_prec:.3f},thr={sim_min_precision},fires={n_fire},imp={n_improved},wor={n_worsened})")
            if _structure_cache is not None and _cache_key is not None:
                _structure_cache[_cache_key] = (-0.5, None)
            return -0.5

        all_new_f1s = _fast_per_label_f1(val_labels, new_predictions)
        all_old_f1s = _fast_per_label_f1(val_labels, existing_predictions)
        target_gain = float(all_new_f1s[label_idx] - all_old_f1s[label_idx])

        if target_gain <= 0:
            _diag_inc(f"label_neg_f1_gain({target_gain:.6f},raw_prec={label_raw_prec:.3f},fires={n_fire},imp={n_improved},wor={n_worsened})")
            if not prop_admit_on_precision:
                return target_gain
            # LBoost-style: admit precise label-propagation rules even without
            # staged F1-gain (see sim path above); value shows under RILL seeds.
            if _wilson_lb(n_improved, n_fire) < 0.60:
                return target_gain
            return 1e-3 * (corr_prec ** 2.0) * float(np.log1p(n_improved))

        precision_multiplier = corr_prec ** 2.0
        coverage_bonus = np.log1p(n_improved) / 3.0

        # A1 (LBoost confidence): reward well-supported propagation rules.
        conf_mult = min(1.0, (n_improved / max(n_fire, 1)) / 0.75)
        f1_gain = (target_gain * precision_multiplier
                   * (1.0 + coverage_bonus) * length_penalty
                   * fires_factor * changes_factor * cross_label_bonus
                   * conf_mult)

        _diag_inc(f"label_positive(gain={f1_gain:.4f},tgt={target_gain:.4f},raw_prec={label_raw_prec:.3f},fires={n_fire},ff={fires_factor:.2f},imp={n_improved},wor={n_worsened},body={n_body})")
    else:
        # === 非 sim 规则：F1 增益导向 ===
        all_new_f1s = _fast_per_label_f1(val_labels, new_predictions)
        all_old_f1s = _fast_per_label_f1(val_labels, existing_predictions)
        target_gain = float(all_new_f1s[label_idx] - all_old_f1s[label_idx])

        if target_gain <= 0:
            _diag_inc(f"neg_f1_gain({target_gain:.6f},fires={n_fire},chg={n_changes},imp={n_improved},wor={n_worsened})")
            return target_gain

        precision_multiplier = corr_prec ** 1.5
        coverage_bonus = np.log1p(n_improved) / 3.0

        f1_gain = (target_gain * precision_multiplier
                   * (1.0 + coverage_bonus) * length_penalty
                   * fires_factor * changes_factor * cross_label_bonus)

        _diag_inc(f"positive(gain={f1_gain:.4f},tgt={target_gain:.4f},prec={corr_prec:.2f},fires={n_fire},ff={fires_factor:.2f},chg={n_changes},imp={n_improved},wor={n_worsened},body={n_body})")

    # 缓存正增益 trial 的 body，供 run_bo()/discover() 直接构建 RDL（避免重跑 greedy）
    _body_info = None
    if f1_gain > 0:
        _body_info = (tuple(body), consequence_label, consequence_op, coverage)
        if _body_cache is not None:
            _body_cache[trial.number] = _body_info

    # 存入结构缓存，下次相同 (structure, label_idx) 直接返回
    if _structure_cache is not None and _cache_key is not None:
        _structure_cache[_cache_key] = (f1_gain, _body_info)

    return f1_gain


# ===========================================================================
# ChaseRuleLearner — 混合架构规则搜索引擎
# ===========================================================================

class ChaseRuleLearner:
    """混合架构规则搜索引擎。

    与原版 RuleLearner 接口完全兼容，可在 pipeline 中直接替换。

    核心改进：
    - Optuna 搜索空间从 50+ 参数降至 ~12 个（仅决定宏观结构）
    - 具体谓词选择由贪心算法在预计算布尔掩码上完成
    - FreqPredicate 参数通过离散网格 + 向量化计数比较加速
    - ML 概率矩阵预缓存，threshold 离散化网格评估

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
    greedy_weights : Dict[str, float] or None
        贪心评分权重 {info_gain, precision, diversity}。
        默认 {0.4, 0.3, 0.3}。
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
        greedy_weights: Optional[Dict[str, float]] = None,
        # ── split mode: 谓词实例化用 fit (train) 数据 ──
        fit_docs: Optional[List[Document]] = None,
        fit_labels: Optional[np.ndarray] = None,
        fit_base_predictions: Optional[np.ndarray] = None,
        min_rule_fires: int = 1,
        min_n_changes: int = 1,
        min_corr_prec: float = 0.0,
        precomputed_val_fire_masks: Optional[np.ndarray] = None,
        # ── Chase-specific parameters ──
        sim_graphs: Optional[Dict[float, Any]] = None,
        track: str = "base",  # "base" | "propagation"
        neighbor_label_masks: Optional[Dict] = None,
        sim_cascade_rounds: int = 3,
        sim_min_precision: float = 0.75,
        sim_self_loop_min_prec: float = -1.0,
        label_prec_objective: bool = False,
        prop_admit_on_precision: bool = False,
        consequence_op_mode: str = "add",  # "add" / "remove" / "both"
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
            if accept_labels is not None
            else None
        )
        self.accept_predictions = (
            np.asarray(accept_predictions, dtype=np.float32)
            if accept_predictions is not None
            else None
        )
        self.accept_f1 = accept_f1
        self.metric_mode = metric_mode
        self.cluster_label_indices = cluster_label_indices
        self.greedy_weights = greedy_weights
        self.min_rule_fires = min_rule_fires
        self.min_n_changes = min_n_changes
        self.min_corr_prec = min_corr_prec

        # Chase-specific
        self.sim_graphs = sim_graphs or {}
        self.track = track
        self.neighbor_label_masks = neighbor_label_masks or {}
        self.neighbor_label_counts: Optional[Dict] = None
        self.sim_cascade_rounds = sim_cascade_rounds
        self.sim_min_precision = sim_min_precision
        self.sim_self_loop_min_prec = sim_self_loop_min_prec
        self.label_prec_objective = label_prec_objective
        self.prop_admit_on_precision = prop_admit_on_precision
        self.consequence_op_mode = consequence_op_mode

        # Weak-label rescue
        self.label_weights: Optional[Dict[str, float]] = None
        self.weak_labels: Optional[Set[str]] = None

        # split mode: 谓词实例化用 fit (train) 数据
        self.fit_docs = list(fit_docs) if fit_docs is not None else None
        self.fit_labels = (
            np.asarray(fit_labels, dtype=np.float32)
            if fit_labels is not None else None
        )
        self.fit_base_predictions = (
            np.asarray(fit_base_predictions, dtype=np.float32)
            if fit_base_predictions is not None else None
        )

        # ── error-driven filter (staged mode) ──
        self.error_driven_filter: bool = False

        # 预计算缓存（在 discover() / run_bo() 中填充）
        self._precomputed_val_fire_masks = precomputed_val_fire_masks
        self._candidate_pools: Dict[str, List[int]] = {}
        self._fire_masks: Optional[np.ndarray] = None
        self._freq_counts: Dict[int, np.ndarray] = {}
        self._ml_proba_cache: Dict[str, np.ndarray] = {}
        # fit (train) 数据的预计算缓存
        self._fit_fire_masks: Optional[np.ndarray] = None
        self._fit_freq_counts: Dict[int, np.ndarray] = {}
        self._fit_ml_proba: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 预计算阶段
    # ------------------------------------------------------------------

    def _precompute(self) -> None:
        """一次性预计算所有缓存（在 discover/run_bo 循环外调用）。"""
        if self._fire_masks is not None:
            if self.verbose:
                print("  [Hybrid] 预计算缓存已存在，跳过")
            return

        if self.verbose:
            print("  [Hybrid] 预计算阶段...")

        # 1. 按类型分组候选谓词
        self._candidate_pools = _group_candidates_by_type(self.candidate_predicates)
        if self.verbose:
            parts = [f"{t}={len(idxs)}" for t, idxs in sorted(self._candidate_pools.items())]
            print(f"  [Hybrid] 候选谓词池: {', '.join(parts)}")

        # 2. 预计算触发掩码 (bool 矩阵) — 如有预计算结果则直接复用
        if self._precomputed_val_fire_masks is not None:
            self._fire_masks = self._precomputed_val_fire_masks
            logger.info("复用预计算 val fire masks: %d 谓词 × %d 文档",
                        self._fire_masks.shape[0], self._fire_masks.shape[1])
        else:
            self._fire_masks = precompute_fire_masks(self.candidate_predicates, self.val_docs)

        # 3. 预计算 FreqPredicate 原始计数
        self._freq_counts = precompute_freq_counts(self.candidate_predicates, self.val_docs)

        # 4. 预缓存 ML 概率矩阵
        self._ml_proba_cache = precompute_ml_proba(self.candidate_ml_models, self.val_docs)

        if self.verbose:
            print(f"  [Chase] 预计算完成: fire_masks={self._fire_masks.shape}, "
                  f"freq_counts={len(self._freq_counts)}, "
                  f"ml_models={len(self._ml_proba_cache)}")

        # 5. 预计算 LabelPredicate 掩码 (避免每个 trial 重复 eval)
        self._label_pred_masks: Dict[Tuple[str, str], np.ndarray] = {}
        # eq 仍预计算以备兼容；minus 已彻底移除——LabelPredicate(op="minus")
        # 在 _core.py 中不可构造（raise），保留它只会在 eq 启用时崩溃 (Phase-5).
        _lp_ops = ["contains"]
        if self._candidate_pools.get("LabelPredicate_eq", []):
            _lp_ops.append("eq")
        for label_name in self.label_list:
            for op in _lp_ops:
                lp = LabelPredicate(label=label_name, op=op)
                mask = np.zeros(len(self.val_docs), dtype=bool)
                for j, doc in enumerate(self.val_docs):
                    proxy = _make_proxy(doc)
                    mask[j] = bool(lp(proxy))
                self._label_pred_masks[(label_name, op)] = mask
        if self.verbose:
            print(f"  [Hybrid] LabelPredicate masks 预计算完成: "
                  f"{len(self._label_pred_masks)} 个 (label, op) 组合")

        # 6. split mode: 对 fit (train) 数据也预计算
        if self.fit_docs is not None:
            if self.verbose:
                print(f"  [Hybrid] Split mode — 预计算 fit (train) 数据 ({len(self.fit_docs)} docs)...")
            self._fit_fire_masks = precompute_fire_masks(self.candidate_predicates, self.fit_docs)
            self._fit_freq_counts = precompute_freq_counts(self.candidate_predicates, self.fit_docs)
            self._fit_ml_proba = precompute_ml_proba(self.candidate_ml_models, self.fit_docs)
            # LabelPredicate masks on fit (train) docs
            self._fit_label_pred_masks: Dict[Tuple[str, str], np.ndarray] = {}
            for label_name in self.label_list:
                for op in _lp_ops:
                    lp = LabelPredicate(label=label_name, op=op)
                    mask = np.zeros(len(self.fit_docs), dtype=bool)
                    for j, doc in enumerate(self.fit_docs):
                        proxy = _make_proxy(doc)
                        mask[j] = bool(lp(proxy))
                    self._fit_label_pred_masks[(label_name, op)] = mask
            if self.verbose:
                print(f"  [Chase] Fit 预计算完成: fire_masks={self._fit_fire_masks.shape}")

        # 7. Chase: 预计算 neighbor_label_masks (Track 2 propagation)
        if self.sim_graphs and not self.neighbor_label_masks:
            label_state = (
                self.base_predictions if self.base_predictions is not None
                else self.val_labels
            )
            self.neighbor_label_masks = precompute_neighbor_label_masks(
                self.sim_graphs, label_state > 0, self.label_list,
            )
            if self.verbose:
                print(f"  [Chase] neighbor_label_masks 预计算完成: "
                      f"{len(self.neighbor_label_masks)} 个 (label, threshold) 组合")

    # ------------------------------------------------------------------
    # 从 trial 重构 RDL
    # ------------------------------------------------------------------

    def _trial_to_rdl(
        self,
        trial: optuna.trial.FrozenTrial,
        existing_predictions: np.ndarray,
    ) -> RDL:
        """从 Optuna trial 参数重构 RDL 对象。

        因为 greedy_instantiate 是确定性的（给定相同输入产生相同输出），
        所以这里重新运行贪心获得与评估时完全一致的 body。
        """
        params = trial.params

        # 还原宏观结构
        structure: Dict[str, int] = {}
        for type_name in _TEXT_TYPE_ORDER:
            structure[type_name] = params.get(f"n_{type_name}", 0)
        structure["MLThresholdPredicate"] = params.get("n_ml", 0)
        structure["LabelPredicate_contains"] = params.get("n_label_contains", 0)
        structure["LabelPredicate_eq"] = params.get("n_label_eq", 0)
        structure["LabelPredicate_minus"] = params.get("n_label_minus", 0)

        consequence_label = params["consequence_label"]
        consequence_op = params["consequence_op"]
        label_idx = self.label_list.index(consequence_label)

        # 重新运行贪心（确定性）
        body, body_masks, _ = beam_instantiate(
            structure=structure,
            candidate_pools=self._candidate_pools,
            candidates=self.candidate_predicates,
            fire_masks=self._fire_masks,
            freq_counts=self._freq_counts,
            ml_proba_cache=self._ml_proba_cache,
            labels=self.val_labels,
            label_idx=label_idx,
            consequence_op=consequence_op,
            existing_preds=existing_predictions,
            docs=self.val_docs,
            label_list=self.label_list,
            ml_model_names=self.candidate_ml_models,
            weights=self.greedy_weights,
            label_pred_masks=self._label_pred_masks,
        )

        # Track "propagation" has_sim=1: inject SimPredicate (same as BO objective)
        if self.track == "propagation" and params.get("has_sim", 0) == 1:
            sim_thresh = float(params.get("sim_threshold", 0.0))
            sim_pred = SimPredicate(threshold=sim_thresh)
            body.insert(0, sim_pred)
            body_masks.insert(0, np.ones(len(self.val_docs), dtype=bool))  # placeholder
            # Ensure LabelPredicate in body
            lp_in_body = [p for p in body if isinstance(p, LabelPredicate)]
            if not lp_in_body:
                lp_label = params.get("lp_label", self.label_list[0] if self.label_list else "")
                lp = LabelPredicate(label=lp_label, op="contains")
                body.append(lp)
                body_masks.append(np.ones(len(self.val_docs), dtype=bool))

        # 去重 body 中的重复谓词（如 label('X') 出现两次）
        seen: set = set()
        deduped_body = []
        deduped_masks = []
        for pred, mask in zip(body, body_masks):
            key = str(pred)
            if key not in seen:
                seen.add(key)
                deduped_body.append(pred)
                deduped_masks.append(mask)
        body, body_masks = deduped_body, deduped_masks

        # 计算覆盖率
        fires = _compute_conjunction_mask(body, body_masks, self.val_docs)
        coverage = float(fires.sum()) / len(self.val_docs) if self.val_docs else 0.0

        return RDL(
            body=tuple(body),
            consequence=consequence_label,
            consequence_op=consequence_op,
            score=trial.value if trial.value is not None else 0.0,
            coverage=coverage,
            trial_number=trial.number,
        )

    # ------------------------------------------------------------------
    # Optuna Study 创建
    # ------------------------------------------------------------------

    def _create_study(self, round_idx: int) -> optuna.Study:
        """创建 Optuna 贝叶斯优化研究（TPE sampler + 可选 JournalStorage）。"""
        sampler = optuna.samplers.TPESampler(
            seed=self.seed + round_idx,
            n_startup_trials=30,
            multivariate=True,
        )

        if self.storage_path:
            file_path = f"{self.storage_path}_hybrid_round{round_idx}.log"
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
            study_name=f"hybrid_rule_discovery_round{round_idx}",
        )
        return study

    def update_base_predictions(self, new_preds: np.ndarray, new_f1: float):
        """更新 base predictions 用于残差学习（复用已缓存的 fire masks）。"""
        self.base_predictions = np.asarray(new_preds, dtype=np.float32)
        self.base_f1 = new_f1
        if self.fit_base_predictions is not None:
            self.fit_base_predictions = None

    # ------------------------------------------------------------------
    # Stage 1: inject per-label optimal ML baseline rules (standalone)
    # ------------------------------------------------------------------

    @staticmethod
    def inject_ml_baseline_rules(
        ml_model_names: List[str],
        ml_proba_cache: Dict[str, np.ndarray],
        label_names: List[str],
        val_labels: np.ndarray,
        threshold_grid: Tuple[float, ...] = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8),
        allow_multi_model: bool = True,
        weak_label_f1_threshold: float = 0.65,
        weak_label_max_rules: int = 3,
        fine_search: bool = True,
    ) -> Tuple[List["RDL"], List[Dict]]:
        """为每个 label 注入 F1 最优的 ML baseline 规则。

        For weak labels (best F1 < weak_label_f1_threshold), injects up to
        weak_label_max_rules diverse rules to create an OR ensemble.

        Returns (rules, trials_log).
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        injected: List[RDL] = []
        trials_log: List[Dict] = []

        for lidx, label in enumerate(label_names):
            # Collect ALL candidate rules for this label
            _candidates: List[Dict] = []

            # --- single model search ---
            for model_name in ml_model_names:
                if model_name not in ml_proba_cache:
                    continue
                proba = ml_proba_cache[model_name]
                try:
                    model_obj = _get_ml_model(model_name)
                    ml_lidx = model_obj.label_index(label)
                except (RuntimeError, KeyError):
                    continue
                if ml_lidx < 0:
                    continue
                proba_col = proba[:, ml_lidx]
                for t in threshold_grid:
                    preds_t = (proba_col >= t).astype(np.int32)
                    f1_t = f1_score(val_labels[:, lidx], preds_t, zero_division=0)
                    prec_t = precision_score(val_labels[:, lidx], preds_t, zero_division=0)
                    rec_t = recall_score(val_labels[:, lidx], preds_t, zero_division=0)
                    trials_log.append({
                        "label": label, "type": "single",
                        "model": model_name, "threshold": t,
                        "f1": round(float(f1_t), 4),
                        "precision": round(float(prec_t), 4),
                        "recall": round(float(rec_t), 4),
                        "n_pos": int(preds_t.sum()),
                    })
                    if f1_t > 0:
                        _candidates.append({
                            "f1": float(f1_t),
                            "precision": float(prec_t),
                            "recall": float(rec_t),
                            "model": model_name,
                            "threshold": t,
                            "body": (MLThresholdPredicate(
                                model_name=model_name, label=label, threshold=t),),
                        })

                # --- fine search around best threshold ---
                if fine_search and _candidates:
                    _model_cands = [c for c in _candidates if c["model"] == model_name]
                    if _model_cands:
                        _best_t = max(_model_cands, key=lambda c: c["f1"])["threshold"]
                        for t_fine in np.arange(
                            max(0.05, _best_t - 0.05),
                            min(0.95, _best_t + 0.06),
                            0.01,
                        ):
                            t_fine = round(float(t_fine), 2)
                            if t_fine in threshold_grid:
                                continue
                            preds_t = (proba_col >= t_fine).astype(np.int32)
                            f1_t = f1_score(val_labels[:, lidx], preds_t, zero_division=0)
                            prec_t = precision_score(val_labels[:, lidx], preds_t, zero_division=0)
                            rec_t = recall_score(val_labels[:, lidx], preds_t, zero_division=0)
                            trials_log.append({
                                "label": label, "type": "single_fine",
                                "model": model_name, "threshold": t_fine,
                                "f1": round(float(f1_t), 4),
                                "precision": round(float(prec_t), 4),
                                "recall": round(float(rec_t), 4),
                                "n_pos": int(preds_t.sum()),
                            })
                            if f1_t > 0:
                                _candidates.append({
                                    "f1": float(f1_t),
                                    "precision": float(prec_t),
                                    "recall": float(rec_t),
                                    "model": model_name,
                                    "threshold": t_fine,
                                    "body": (MLThresholdPredicate(
                                        model_name=model_name, label=label, threshold=t_fine),),
                                })

            # --- multi-model AND search (2 models) ---
            if allow_multi_model and len(ml_model_names) >= 2:
                coarse_grid = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
                for i_a, m_a in enumerate(ml_model_names):
                    for m_b in ml_model_names[i_a + 1:]:
                        if m_a not in ml_proba_cache or m_b not in ml_proba_cache:
                            continue
                        try:
                            obj_a = _get_ml_model(m_a)
                            obj_b = _get_ml_model(m_b)
                            lidx_a = obj_a.label_index(label)
                            lidx_b = obj_b.label_index(label)
                        except (RuntimeError, KeyError):
                            continue
                        if lidx_a < 0 or lidx_b < 0:
                            continue
                        proba_a = ml_proba_cache[m_a][:, lidx_a]
                        proba_b = ml_proba_cache[m_b][:, lidx_b]
                        for t_a in coarse_grid:
                            for t_b in coarse_grid:
                                preds_ab = ((proba_a >= t_a) & (proba_b >= t_b)).astype(np.int32)
                                f1_ab = f1_score(val_labels[:, lidx], preds_ab, zero_division=0)
                                trials_log.append({
                                    "label": label, "type": "multi",
                                    "model": f"{m_a}+{m_b}",
                                    "threshold": f"{t_a}+{t_b}",
                                    "f1": round(float(f1_ab), 4),
                                    "n_pos": int(preds_ab.sum()),
                                })
                                if f1_ab > 0:
                                    _candidates.append({
                                        "f1": float(f1_ab),
                                        "precision": 0.0,
                                        "recall": 0.0,
                                        "model": f"{m_a}+{m_b}",
                                        "threshold": f"{t_a}+{t_b}",
                                        "body": (
                                            MLThresholdPredicate(model_name=m_a, label=label, threshold=t_a),
                                            MLThresholdPredicate(model_name=m_b, label=label, threshold=t_b),
                                        ),
                                    })

            if not _candidates:
                continue

            # Sort by F1 descending
            _candidates.sort(key=lambda x: -x["f1"])
            best_f1 = _candidates[0]["f1"]

            if best_f1 < weak_label_f1_threshold and weak_label_max_rules > 1:
                # Weak label: inject top-K diverse rules
                _selected = [_candidates[0]]
                for cand in _candidates[1:]:
                    if len(_selected) >= weak_label_max_rules:
                        break
                    # Diversity: different model OR threshold diff > 0.1
                    _dominated = False
                    for sel in _selected:
                        if (cand["model"] == sel["model"]
                                and isinstance(cand["threshold"], (int, float))
                                and isinstance(sel["threshold"], (int, float))
                                and abs(cand["threshold"] - sel["threshold"]) <= 0.1):
                            _dominated = True
                            break
                    if not _dominated:
                        _selected.append(cand)

                # Also inject a precision-optimized rule if available
                _prec_cands = [c for c in _candidates
                               if c["precision"] >= 0.7 and c not in _selected]
                if _prec_cands:
                    _prec_cands.sort(key=lambda x: -x["recall"])
                    if len(_selected) < weak_label_max_rules + 1:
                        _selected.append(_prec_cands[0])

                for sel in _selected:
                    injected.append(RDL(
                        body=sel["body"],
                        consequence=label,
                        consequence_op="add",
                        score=sel["f1"],
                        coverage=0.0,
                        trial_number=-1,
                        val_stats={"injected_baseline": True, "best_f1": sel["f1"],
                                   "weak_label": True},
                    ))
            else:
                # Strong label: single best rule (existing behavior)
                injected.append(RDL(
                    body=_candidates[0]["body"],
                    consequence=label,
                    consequence_op="add",
                    score=best_f1,
                    coverage=0.0,
                    trial_number=-1,
                    val_stats={"injected_baseline": True, "best_f1": float(best_f1)},
                ))

        logger.info("[inject_ml_baseline_rules] %d rules from %d labels, "
                    "%d trial combos searched",
                    len(injected), len(label_names), len(trials_log))
        return injected, trials_log

    # ------------------------------------------------------------------
    # [CORE] 主发现循环：贪心序列覆盖
    # ------------------------------------------------------------------

    def discover(self) -> RDLSet:
        """运行完整的混合规则发现 pipeline。

        采用贪心序列覆盖策略：每轮发现一条最优规则，更新现有预测后
        再搜索下一条，直到达到 top_n 轮或无法继续提升 F1。

        Returns
        -------
        RDLSet
            发现的最优规则集合。
        """
        # ==============================================================
        # 预计算阶段（循环外一次性执行）
        # ==============================================================
        self._precompute()

        discovered_rules: List[RDL] = []

        if self.base_predictions is not None:
            existing_predictions = self.base_predictions.copy()
        else:
            existing_predictions = np.zeros_like(self.val_labels, dtype=np.float32)
        if self.base_f1 is not None:
            baseline_f1 = self.base_f1
        else:
            baseline_f1 = float(
                _fast_macro_f1(self.val_labels, existing_predictions)
            )

        # ==============================================================
        # 贪心序列覆盖循环
        # ==============================================================
        for round_idx in range(self.top_n):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  [Hybrid] 规则发现第 {round_idx + 1}/{self.top_n} 轮")
                print(f"  当前基线 Macro-F1: {baseline_f1:.4f}")
                print(f"{'='*60}")

            # 创建 Optuna Study
            study = self._create_study(round_idx)

            # 构建 objective 闭包，捕获当前轮次状态
            _ep = existing_predictions.copy()
            _bf = baseline_f1
            _candidates = self.candidate_predicates
            _pools = self._candidate_pools
            _fm = self._fire_masks
            _fc = self._freq_counts
            _mlp = self._ml_proba_cache
            _ml_names = self.candidate_ml_models
            _ll = self.label_list
            _vd = self.val_docs
            _vl = self.val_labels
            _mc = self.min_coverage
            _rmp = self.rule_min_precision
            _gw = self.greedy_weights
            _mm = self.metric_mode
            _cli = self.cluster_label_indices

            # ── 4a+4b: 排除不可行标签 + (label, op) CDF 采样 ──
            _active_labels = _cli if _cli is not None else list(range(len(self.label_list)))
            _min_feasible = max(4, int(np.ceil(self.min_corr_prec * 2 / (1 - self.min_corr_prec)))) if self.min_corr_prec < 1.0 else 4
            _add_label_names: Optional[List[str]] = None
            for _li in _active_labels:
                _fn = int(((_vl[:, _li] == 1) & (_ep[:, _li] == 0)).sum())  # add 可修
                if _fn >= _min_feasible:
                    if _add_label_names is None:
                        _add_label_names = []
                    _add_label_names.append(_ll[_li])
            if self.verbose and _cli is not None:
                logger.info("  Label-op pruning: %d active → %d add-feasible labels (min_errors=%d)",
                            len(_active_labels), len(_add_label_names) if _add_label_names else 0,
                            _min_feasible)

            # split mode 闭包变量
            _fit_kw: Dict = {}
            _vlpm = self._label_pred_masks   # val label pred masks (always set)
            _flpm: Optional[Dict] = None     # fit label pred masks (split mode only)
            if self.fit_docs is not None:
                _fit_kw = dict(
                    fit_docs=self.fit_docs,
                    fit_labels=self.fit_labels,
                    fit_fire_masks=self._fit_fire_masks,
                    fit_freq_counts=self._fit_freq_counts,
                    fit_ml_proba_cache=self._fit_ml_proba,
                    fit_existing_preds=self.fit_base_predictions,
                )
                _flpm = getattr(self, '_fit_label_pred_masks', None)

            # 缓存 trial body，避免 study.optimize 之后重跑 greedy_instantiate
            _body_cache: Dict[int, Tuple] = {}

            def objective(trial):
                return evaluate_chase_configuration(
                    trial=trial,
                    candidates=_candidates,
                    candidate_pools=_pools,
                    fire_masks=_fm,
                    freq_counts=_fc,
                    ml_model_names=_ml_names,
                    ml_proba_cache=_mlp,
                    label_list=_ll,
                    val_docs=_vd,
                    val_labels=_vl,
                    existing_predictions=_ep,
                    baseline_f1=_bf,
                    min_coverage=_mc,
                    rule_min_precision=_rmp,
                    greedy_weights=_gw,
                    metric_mode=_mm,
                    cluster_label_indices=_cli,
                    val_label_pred_masks=_vlpm,
                    fit_label_pred_masks=_flpm,
                    min_rule_fires=self.min_rule_fires,
                    min_n_changes=self.min_n_changes,
                    min_corr_prec=self.min_corr_prec,
                    _body_cache=_body_cache,
                    _structure_cache=None,
                    add_label_names=_add_label_names,
                    track=self.track,
                    sim_graphs=self.sim_graphs,
                    neighbor_label_masks=self.neighbor_label_masks,
                    neighbor_label_counts=self.neighbor_label_counts,
                    label_state=_ep,  # existing_predictions as label_state for has_sim=0
                    sim_cascade_rounds=self.sim_cascade_rounds,
                    sim_min_precision=self.sim_min_precision,
                    sim_self_loop_min_prec=self.sim_self_loop_min_prec,
                    label_prec_objective=self.label_prec_objective,
                    prop_admit_on_precision=self.prop_admit_on_precision,
                    **_fit_kw,
                )

            # 执行优化
            study.optimize(objective, n_trials=self.max_trials)

            # 诊断日志
            logger.info("[Chase] round %d: %d total trials (feature dropout, no structure cache)",
                        round_idx, len(study.trials))
            all_values = [t.value for t in study.trials if t.value is not None]
            if all_values:
                pos_vals = [v for v in all_values if v > 0]
                logger.info(
                    "[Chase] round %d: %d trials, %d positive (max=%.4f), best=%.4f",
                    round_idx, len(all_values), len(pos_vals),
                    max(pos_vals) if pos_vals else 0.0,
                    max(all_values),
                )

            # 提取最优 trial
            best = study.best_trial
            if self.verbose:
                print(f"  最优 trial #{best.number}: F1 增益 = {best.value:.4f}")
                print(f"  宏观结构: {best.params}")

            if best.value is None or best.value <= 0:
                if self.verbose:
                    print("  无法继续提升 F1，停止搜索。")
                break

            # 从缓存构建 RDL（零成本），fallback 到 _trial_to_rdl（重跑 greedy）
            cached = _body_cache.get(best.number)
            if cached is not None:
                body_tuple, cons_label, cons_op, cov = cached
                rule = RDL(
                    body=body_tuple,
                    consequence=cons_label,
                    consequence_op=cons_op,
                    score=best.value,
                    coverage=cov,
                    trial_number=best.number,
                )
            else:
                rule = self._trial_to_rdl(best, existing_predictions)

            # 去重检查
            if _is_redundant(rule, discovered_rules):
                if self.verbose:
                    print("  规则冗余，跳过。")
                continue

            discovered_rules.append(rule)
            if self.verbose:
                print(f"  发现规则: {rule}  (gain={best.value:.4f})")

            # 注意：不更新 existing_predictions 和 doc.lbl
            # 所有规则基于相同 baseline 独立评估，允许发现更多高精度规则

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  [Hybrid] 规则发现完成！共发现 {len(discovered_rules)} 条规则")
            print(f"  基线 Macro-F1: {baseline_f1:.4f}")
            print(f"{'='*60}")

        return RDLSet(rules=discovered_rules, label_names=self.label_list)

    # ------------------------------------------------------------------
    # Batch 模式：仅运行 BO 收集 trials（不做贪心选择）
    # ------------------------------------------------------------------

    def run_bo(
        self,
    ) -> List[Tuple[optuna.trial.FrozenTrial, Dict[str, List[int]],
                     List[Predicate], List[str]]]:
        """运行单次 BO 研究，返回所有正增益 trials。

        用于 per-cluster batch pipeline：各集群分别运行 run_bo，
        然后汇总到 batch_select 做全局贪心筛选。

        Returns
        -------
        list of (FrozenTrial, candidate_pools, candidates, ml_model_names)
            仅包含 state == COMPLETE 且 F1 gain > 0 的 trials。
        """
        # 预计算
        self._precompute()

        # ── error-driven candidate filtering (scoped to this run_bo call) ──
        _saved_candidates = None
        _saved_fire_masks = None
        _saved_freq_counts = None
        _saved_pools = None
        if self.error_driven_filter and self.base_predictions is not None:
            _ep_for_filter = self.base_predictions
            _cons_op = "remove" if self.consequence_op_mode == "remove" else "add"
            _kept_indices = _error_driven_filter(
                candidate_predicates=self.candidate_predicates,
                fire_masks=self._fire_masks,
                existing_predictions=_ep_for_filter,
                val_labels=self.val_labels,
                label_list=self.label_list,
                consequence_op=_cons_op,
            )
            if _kept_indices and len(_kept_indices) < len(self.candidate_predicates):
                _saved_candidates = self.candidate_predicates
                _saved_fire_masks = self._fire_masks
                _saved_freq_counts = self._freq_counts
                _saved_pools = self._candidate_pools
                _n_before = len(self.candidate_predicates)
                _kept_set = set(_kept_indices)
                _old_to_new = {}
                _new_preds = []
                for new_i, old_i in enumerate(sorted(_kept_set)):
                    _old_to_new[old_i] = new_i
                    _new_preds.append(self.candidate_predicates[old_i])
                self.candidate_predicates = _new_preds
                self._fire_masks = self._fire_masks[sorted(_kept_set)]
                new_freq = {}
                for old_i, arr in self._freq_counts.items():
                    if old_i in _old_to_new:
                        new_freq[_old_to_new[old_i]] = arr
                self._freq_counts = new_freq
                self._candidate_pools = _group_candidates_by_type(self.candidate_predicates)
                logger.info("[ErrorFilter] %d → %d candidates (%s)",
                            _n_before, len(self.candidate_predicates), _cons_op)
            else:
                logger.info("[ErrorFilter] no filtering applied (kept=%s, total=%d)",
                            len(_kept_indices) if _kept_indices else 0,
                            len(self.candidate_predicates))

        # 初始化基线
        if self.base_predictions is not None:
            existing_predictions = self.base_predictions.copy()
        else:
            existing_predictions = np.zeros_like(self.val_labels, dtype=np.float32)
        if self.base_f1 is not None:
            baseline_f1 = self.base_f1
        else:
            baseline_f1 = float(
                _fast_macro_f1(self.val_labels, existing_predictions)
            )

        if self.verbose:
            print(f"  [Hybrid run_bo] 基线 F1: {baseline_f1:.4f}")

        # 构建 objective 闭包
        _ep = existing_predictions.copy()
        _bf = baseline_f1

        # ── 排除不可行标签 → add-only label 列表 ──
        _active_labels = self.cluster_label_indices if self.cluster_label_indices is not None else list(range(len(self.label_list)))
        _min_feasible = max(4, int(np.ceil(self.min_corr_prec * 2 / (1 - self.min_corr_prec)))) if self.min_corr_prec < 1.0 else 4
        _add_label_names: Optional[List[str]] = None
        for _li in _active_labels:
            _fn = int(((self.val_labels[:, _li] == 1) & (_ep[:, _li] == 0)).sum())
            if _fn >= _min_feasible:
                if _add_label_names is None:
                    _add_label_names = []
                _add_label_names.append(self.label_list[_li])
        if self.verbose:
            logger.info("  Label-op pruning: %d active → %d add-feasible labels (min_errors=%d)",
                        len(_active_labels), len(_add_label_names) if _add_label_names else 0,
                        _min_feasible)

        # ── remove-feasible 标签列表（FP ≥ min_feasible）──
        _remove_label_names: Optional[List[str]] = None
        if self.consequence_op_mode in ("remove", "both"):
            _remove_label_names = []
            for _li in _active_labels:
                _fp = int(((self.val_labels[:, _li] == 0) & (_ep[:, _li] == 1)).sum())
                if _fp >= _min_feasible:
                    _remove_label_names.append(self.label_list[_li])
            if not _remove_label_names:
                _remove_label_names = None
            if self.verbose:
                logger.info("  Remove-feasible labels: %d (min_errors=%d)",
                            len(_remove_label_names) if _remove_label_names else 0,
                            _min_feasible)

        # split mode 闭包变量
        _fit_kwargs: Dict = {}
        if self.fit_docs is not None:
            _fit_kwargs = dict(
                fit_docs=self.fit_docs,
                fit_labels=self.fit_labels,
                fit_fire_masks=self._fit_fire_masks,
                fit_freq_counts=self._fit_freq_counts,
                fit_ml_proba_cache=self._fit_ml_proba,
                fit_existing_preds=self.fit_base_predictions,
            )

        # 缓存 trial body，避免 study.optimize 之后重跑 greedy_instantiate
        _body_cache: Dict[int, Tuple] = {}

        def objective(trial):
            return evaluate_chase_configuration(
                trial=trial,
                candidates=self.candidate_predicates,
                candidate_pools=self._candidate_pools,
                fire_masks=self._fire_masks,
                freq_counts=self._freq_counts,
                ml_model_names=self.candidate_ml_models,
                ml_proba_cache=self._ml_proba_cache,
                label_list=self.label_list,
                val_docs=self.val_docs,
                val_labels=self.val_labels,
                existing_predictions=_ep,
                baseline_f1=_bf,
                min_coverage=self.min_coverage,
                rule_min_precision=self.rule_min_precision,
                greedy_weights=self.greedy_weights,
                metric_mode=self.metric_mode,
                cluster_label_indices=self.cluster_label_indices,
                min_rule_fires=self.min_rule_fires,
                min_n_changes=self.min_n_changes,
                min_corr_prec=self.min_corr_prec,
                _body_cache=_body_cache,
                _structure_cache=None,
                add_label_names=_add_label_names,
                track=self.track,
                sim_graphs=self.sim_graphs,
                neighbor_label_masks=self.neighbor_label_masks,
                neighbor_label_counts=self.neighbor_label_counts,
                label_state=_ep,  # existing_predictions as label_state for has_sim=0
                sim_cascade_rounds=self.sim_cascade_rounds,
                sim_min_precision=self.sim_min_precision,
                sim_self_loop_min_prec=self.sim_self_loop_min_prec,
                label_prec_objective=self.label_prec_objective,
                prop_admit_on_precision=self.prop_admit_on_precision,
                consequence_op_mode=self.consequence_op_mode,
                remove_label_names=_remove_label_names,
                label_weights=self.label_weights,
                weak_labels=self.weak_labels,
                **_fit_kwargs,
            )

        _diag_reset()
        _bo_t0 = time.time()
        _bo_pos = 0

        def _bo_progress_callback(study, trial):
            nonlocal _bo_pos
            if trial.value is not None and trial.value > 0:
                _bo_pos += 1
            n_done = len(study.trials)
            elapsed = time.time() - _bo_t0
            avg = elapsed / n_done if n_done else 0
            eta = avg * (self.max_trials - n_done)
            if n_done % 20 == 0 or n_done == self.max_trials:
                logger.info(
                    "[BO] %d/%d trials (%.1fs avg, ETA %.0fs) | "
                    "positive=%d | best=%.4f | val=%.4f",
                    n_done, self.max_trials, avg, eta,
                    _bo_pos,
                    study.best_value if study.best_trial else 0.0,
                    trial.value if trial.value is not None else -1.0,
                )

        study = self._create_study(round_idx=0)
        study.optimize(objective, n_trials=self.max_trials,
                       callbacks=[_bo_progress_callback])

        logger.info("[run_bo] %d total trials (feature dropout, no structure cache)",
                    len(study.trials))

        # 收集正增益 trials，优先从 _body_cache 构建 RDL（零成本），
        # fallback 到 _trial_to_rdl（重跑 greedy_instantiate）
        completed = []
        for t in study.trials:
            if (t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None and t.value > 0):
                cached = _body_cache.get(t.number)
                if cached is not None:
                    body_tuple, cons_label, cons_op, cov = cached
                    local_rule = RDL(
                        body=body_tuple,
                        consequence=cons_label,
                        consequence_op=cons_op,
                        score=t.value,
                        coverage=cov,
                        trial_number=t.number,
                    )
                else:
                    logger.warning("[run_bo] Trial #%d cache miss, "
                                   "fallback to _trial_to_rdl", t.number)
                    local_rule = self._trial_to_rdl(t, existing_predictions)
                completed.append((t, local_rule))

        # 诊断日志
        all_vals = [t.value for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None]
        n_pos = sum(1 for v in all_vals if v > 0)
        n_neg = sum(1 for v in all_vals if v <= 0)
        logger.info(
            "[Hybrid BO] %d trials: %d positive, %d non-positive",
            len(all_vals), n_pos, n_neg,
        )
        logger.info("[Hybrid BO] Trial rejection breakdown:\n%s", _diag_report())
        if self.verbose:
            print(f"  [Hybrid run_bo] {len(all_vals)} trials, {n_pos} positive-gain")
            print(_diag_report())

        # ── BO 诊断：参数重要性 + 收敛趋势 ──
        # NOTE: get_param_importances (fANOVA) is extremely slow with many
        # categorical params; skip by default unless explicitly requested.
        importance = {}
        if os.environ.get("LORIS_PARAM_IMPORTANCE", "0") == "1":
            try:
                importance = optuna.importance.get_param_importances(study)
                logger.info("[BO Diagnostics] Parameter importance:\n%s",
                            "\n".join(f"  {k}: {v:.4f}" for k, v in importance.items()))
            except Exception:
                pass

        # 统计正增益规则中的谓词类型分布
        body_type_counts: Dict[str, int] = {}
        for _t, _rdl in completed:
            for p in _rdl.body:
                tname = type(p).__name__
                body_type_counts[tname] = body_type_counts.get(tname, 0) + 1
        if body_type_counts:
            logger.info("[BO Diagnostics] Predicate type distribution in positive rules:\n%s",
                        "\n".join(f"  {k}: {v}" for k, v in
                                  sorted(body_type_counts.items(), key=lambda x: -x[1])))

        # 统计最常被选中的 consequence labels
        label_counts: Dict[str, int] = {}
        for _t, _rdl in completed:
            label_counts[_rdl.consequence] = label_counts.get(_rdl.consequence, 0) + 1
        if label_counts:
            top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:10]
            logger.info("[BO Diagnostics] Top consequence labels:\n%s",
                        "\n".join(f"  {lbl}: {cnt}" for lbl, cnt in top_labels))

        # 构建诊断数据供外部保存
        self._bo_diagnostics = {
            "n_trials": len(all_vals),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "param_importance": {k: round(v, 4) for k, v in importance.items()},
            "body_type_distribution": body_type_counts,
            "consequence_label_distribution": label_counts,
            "rejection_breakdown": dict(_DIAG_COUNTERS),
            "value_progression": [
                {"trial": i, "value": v}
                for i, v in enumerate(all_vals)
            ],
        }

        # ── restore candidates if error-driven filter was applied ──
        if _saved_candidates is not None:
            self.candidate_predicates = _saved_candidates
            self._fire_masks = _saved_fire_masks
            self._freq_counts = _saved_freq_counts
            self._candidate_pools = _saved_pools

        return completed
    # ------------------------------------------------------------------

    @staticmethod
    def batch_select(
        all_trials: List[Tuple[optuna.trial.FrozenTrial, "RDL"]],
        label_list: List[str],
        val_docs: List[Document],
        val_labels: np.ndarray,
        base_predictions: Optional[np.ndarray] = None,
        base_f1: Optional[float] = None,
        sort_by_gain: bool = True,
        verbose: bool = True,
        selection_mode: str = "independent",
        min_rule_fires: int = 1,
        min_n_changes: int = 1,
        min_corr_prec: float = 0.5,
        prop_admit_on_precision: bool = False,  # LBoost-style: keep precise sim/group propagation rules even if they show low corr-prec / zero-improved on the strong select-base (value at test-with-seeds / RILL)
        # ── accept cross-check: 训练集泛化验证 ──
        accept_docs: Optional[List[Document]] = None,
        accept_labels: Optional[np.ndarray] = None,
        accept_base_predictions: Optional[np.ndarray] = None,
        # ── Chase: sim graph support ──
        sim_graphs: Optional[Dict] = None,
        label_state: Optional[np.ndarray] = None,
        # ── Chase: group propagation support (multi-value csr membership) ──
        virtual_attrs: Optional[Dict[str, sp.csr_matrix]] = None,
        # ── ML baseline injection ──
        inject_ml_baseline: bool = False,
        # ── 预计算 fire masks（避免重复计算） ──
        precomputed_pred_masks: Optional[np.ndarray] = None,
        precomputed_pred_to_idx: Optional[Dict[int, int]] = None,
        precomputed_ml_proba: Optional[Dict[str, np.ndarray]] = None,
    ) -> RDLSet:
        """全局 batch 验证：从各集群收集的预构建 RDL 规则中筛选高精度规则。

        规则已由 run_bo() 在局部 cluster 数据上通过 _trial_to_rdl() 预构建，
        本方法仅在全局数据上**评估**（不再重新实例化），消除 Frankenstein rules。

        采用两阶段流程：
        1) 遍历所有 (trial, rule) 对，用 rule.fires() 在全局 val_docs 上触发，
           独立模拟每条规则并计算 corr_prec / F1 gain；
        2) 按 corr_prec 降序 → F1 gain 降序排列，贪心筛选进最终 RDLSet。
        """
        if selection_mode not in ("independent", "greedy"):
            raise ValueError(f"Unknown selection_mode: {selection_mode!r}")

        val_labels = np.asarray(val_labels, dtype=np.float32)
        if base_predictions is not None:
            existing_predictions = np.asarray(base_predictions, dtype=np.float32).copy()
        else:
            existing_predictions = np.zeros_like(val_labels, dtype=np.float32)
        if base_f1 is not None:
            baseline_f1 = float(base_f1)
        else:
            baseline_f1 = _fast_macro_f1(val_labels, existing_predictions)

        # accept cross-check 数据预处理
        if accept_labels is not None:
            accept_labels = np.asarray(accept_labels, dtype=np.float32)
        if accept_base_predictions is not None:
            accept_base_predictions = np.asarray(accept_base_predictions, dtype=np.float32)

        # ================================================================
        # 预计算：收集所有唯一谓词，一次性计算 fire masks（避免重复评估）
        # ================================================================
        import time as _time
        _t0_precompute = _time.time()

        # 收集所有规则中的唯一谓词
        unique_preds: List[Predicate] = []
        _pred_id_set: set = set()
        for _trial, _rule in all_trials:
            for p in _rule.body:
                pid = id(p)
                if pid not in _pred_id_set:
                    _pred_id_set.add(pid)
                    unique_preds.append(p)

        _ml_preds = [p for p in unique_preds if isinstance(p, MLThresholdPredicate)]
        _non_ml_preds = [p for p in unique_preds if not isinstance(p, MLThresholdPredicate)]

        # 使用预计算 masks（如果提供）
        if (precomputed_pred_masks is not None
                and precomputed_pred_to_idx is not None
                and precomputed_ml_proba is not None):
            # 从预计算的全量 masks 中提取当前 trials 用到的子集
            # 非 ML 谓词：从 precomputed 中查找
            _non_ml_masks_rows = []
            _non_ml_mapped = []
            for _p in _non_ml_preds:
                _pidx = precomputed_pred_to_idx.get(id(_p))
                if _pidx is not None:
                    _non_ml_masks_rows.append(precomputed_pred_masks[_pidx])
                    _non_ml_mapped.append(True)
                else:
                    _non_ml_masks_rows.append(None)
                    _non_ml_mapped.append(False)
            _n_missing = sum(1 for m in _non_ml_mapped if not m)
            if _n_missing > 0:
                logger.info("[batch_select] %d/%d non-ML preds not in precomputed cache, "
                            "computing fallback...", _n_missing, len(_non_ml_preds))
                _missing = [p for p, m in zip(_non_ml_preds, _non_ml_mapped) if not m]
                _missing_masks = precompute_fire_masks(_missing, val_docs)
                _mi = 0
                for _j in range(len(_non_ml_preds)):
                    if not _non_ml_mapped[_j]:
                        _non_ml_masks_rows[_j] = _missing_masks[_mi]
                        _mi += 1
            if _non_ml_masks_rows:
                _non_ml_masks = np.stack(_non_ml_masks_rows)
            else:
                _non_ml_masks = np.zeros((0, len(val_docs)), dtype=bool)

            # ML 谓词：从 precomputed_ml_proba 按阈值切分
            _ml_proba = precomputed_ml_proba
            _ml_masks = np.zeros((len(_ml_preds), len(val_docs)), dtype=bool)
            for _i, _p in enumerate(_ml_preds):
                if _p.model_name in _ml_proba:
                    _model = _get_ml_model(_p.model_name)
                    _lidx = _model.label_index(_p.label)
                    _ml_masks[_i] = _ml_proba[_p.model_name][:, _lidx] >= _p.threshold
            _ml_model_names = sorted(set(p.model_name for p in _ml_preds))

            logger.info("[batch_select] Using precomputed masks: %d non-ML + %d ML "
                        "(%d fallback)", len(_non_ml_preds), len(_ml_preds), _n_missing)
        else:
            # 原始路径：从头计算
            logger.info("[batch_select] Precomputing fire masks: %d non-ML + %d ML predicates "
                        "on %d val_docs ...", len(_non_ml_preds), len(_ml_preds), len(val_docs))
            _non_ml_masks = precompute_fire_masks(_non_ml_preds, val_docs)

            _ml_model_names = sorted(set(p.model_name for p in _ml_preds))
            _ml_proba = precompute_ml_proba(_ml_model_names, val_docs) if _ml_preds else {}
            _ml_masks = np.zeros((len(_ml_preds), len(val_docs)), dtype=bool)
            for _i, _p in enumerate(_ml_preds):
                if _p.model_name in _ml_proba:
                    _model = _get_ml_model(_p.model_name)
                    _lidx = _model.label_index(_p.label)
                    _ml_masks[_i] = _ml_proba[_p.model_name][:, _lidx] >= _p.threshold

        # 合并 mask 矩阵，建立统一索引
        if _ml_preds and _non_ml_preds:
            val_pred_masks = np.concatenate([_non_ml_masks, _ml_masks], axis=0)
        elif _ml_preds:
            val_pred_masks = _ml_masks
        else:
            val_pred_masks = _non_ml_masks

        _pred_to_idx: Dict[int, int] = {}
        for _i, _p in enumerate(_non_ml_preds):
            _pred_to_idx[id(_p)] = _i
        for _i, _p in enumerate(_ml_preds):
            _pred_to_idx[id(_p)] = len(_non_ml_preds) + _i

        # accept_docs 同理
        acc_pred_masks = None
        if accept_docs is not None:
            logger.info("[batch_select] Precomputing fire masks: %d non-ML + %d ML predicates "
                        "on %d accept_docs ...", len(_non_ml_preds), len(_ml_preds), len(accept_docs))
            _acc_non_ml = precompute_fire_masks(_non_ml_preds, accept_docs)
            _acc_ml_proba = precompute_ml_proba(_ml_model_names, accept_docs) if _ml_preds else {}
            _acc_ml_masks = np.zeros((len(_ml_preds), len(accept_docs)), dtype=bool)
            for _i, _p in enumerate(_ml_preds):
                if _p.model_name in _acc_ml_proba:
                    _model = _get_ml_model(_p.model_name)
                    _lidx = _model.label_index(_p.label)
                    _acc_ml_masks[_i] = _acc_ml_proba[_p.model_name][:, _lidx] >= _p.threshold
            if _ml_preds and _non_ml_preds:
                acc_pred_masks = np.concatenate([_acc_non_ml, _acc_ml_masks], axis=0)
            elif _ml_preds:
                acc_pred_masks = _acc_ml_masks
            else:
                acc_pred_masks = _acc_non_ml

        _t1_precompute = _time.time()
        logger.info("[batch_select] Fire mask precomputation done in %.1fs",
                    _t1_precompute - _t0_precompute)

        # ================================================================
        # Pass 1: 遍历所有预构建规则，在全局 val_docs 上独立评估
        # ================================================================
        _CandidateInfo = Tuple  # (corr_prec, trial_value, rule, fires, label_idx, new_f1, n_improved, n_worsened)
        scored_candidates: List[_CandidateInfo] = []

        # ── rejection 诊断统计 ──
        _rej = {"low_fires": 0, "low_changes": 0, "zero_improved": 0,
                "low_corr_prec": 0, "accept_zero_fires": 0, "accept_low_prec": 0,
                "low_label_gain": 0, "passed": 0}
        _rej_details: List[dict] = []  # top rejected rules for diagnosis

        for trial, rule in all_trials:
            consequence_label = rule.consequence
            consequence_op = rule.consequence_op
            if consequence_label not in label_list:
                continue
            label_idx = label_list.index(consequence_label)

            # 用预计算的 fire masks AND-reduce 构建规则触发掩码
            fires = np.ones(len(val_docs), dtype=bool)
            for pred in rule.body:
                if isinstance(pred, SimPredicate):
                    # SimPredicate: use SpMV with label predicates in body
                    if sim_graphs and label_state is not None:
                        adj = sim_graphs.get(pred.threshold)
                        lps = [p for p in rule.body if isinstance(p, LabelPredicate)]
                        if adj is not None and lps:
                            for lp in lps:
                                if lp.label in label_list:
                                    lp_idx = label_list.index(lp.label)
                                    sim_mask = np.asarray(
                                        (adj @ label_state[:, lp_idx].astype(np.float32)) > 0
                                    ).ravel()
                                    fires &= sim_mask
                    continue  # LabelPredicate handled above via SpMV
                if isinstance(pred, GroupPredicate):
                    # GroupPredicate: attribute equality propagation
                    if virtual_attrs is not None and label_state is not None:
                        group_ids = virtual_attrs.get(pred.attr_name)
                        lps = [p for p in rule.body if isinstance(p, LabelPredicate)]
                        if group_ids is not None and lps:
                            from loris.rules.group_propagation import compute_group_fire_mask
                            for lp in lps:
                                if lp.label in label_list:
                                    lp_idx = label_list.index(lp.label)
                                    group_mask = compute_group_fire_mask(
                                        group_ids, lp_idx, label_state
                                    )
                                    fires &= group_mask
                    continue  # LabelPredicate handled above via group membership
                if isinstance(pred, LabelPredicate):
                    has_sim = any(isinstance(p, (SimPredicate, GroupPredicate)) for p in rule.body)
                    if has_sim:
                        # has_sim=1: label checks y's labels via SpMV/group (handled above)
                        continue
                    else:
                        # has_sim=0: label checks x's own labels via label_state
                        if label_state is not None and pred.label in label_list:
                            lp_idx = label_list.index(pred.label)
                            fires &= (label_state[:, lp_idx] > 0)
                        else:
                            fires[:] = False
                        continue
                pidx = _pred_to_idx.get(id(pred))
                if pidx is not None:
                    fires &= val_pred_masks[pidx]
                else:
                    # fallback（理论上不会走到这里）
                    fires &= np.array([bool(pred(_make_proxy(d))) for d in val_docs], dtype=bool)

            n_fires_val = int(fires.sum())
            # 自适应 min_rule_fires: 稀有 label 放宽阈值
            _label_pos_bs = int(val_labels[:, label_idx].sum())
            _adaptive_mrf_bs = max(3, min(min_rule_fires, int(_label_pos_bs * 0.1)))
            if n_fires_val < _adaptive_mrf_bs:
                _rej["low_fires"] += 1
                continue
            coverage = float(n_fires_val) / len(val_docs) if val_docs else 0.0

            # 更新 rule 的 coverage（局部 cluster 上算的 coverage 可能不同）
            rule = RDL(
                body=rule.body,
                consequence=rule.consequence,
                consequence_op=rule.consequence_op,
                score=rule.score,
                coverage=coverage,
                trial_number=rule.trial_number,
            )

            # 模拟添加规则（基于原始 baseline，独立评估）
            test_preds = existing_predictions.copy()
            if consequence_op == "add":
                test_preds[fires, label_idx] = 1.0
            elif consequence_op == "remove":
                test_preds[fires, label_idx] = 0.0

            # 计算 correction precision
            old_hit = (existing_predictions[fires, label_idx] == val_labels[fires, label_idx])
            new_hit = (test_preds[fires, label_idx] == val_labels[fires, label_idx])
            n_improved = int((new_hit & ~old_hit).sum())
            n_worsened = int((old_hit & ~new_hit).sum())
            n_changes = n_improved + n_worsened
            corr_prec = n_improved / (n_improved + n_worsened + 1) if n_changes > 0 else 0.0  # Laplace+1

            # 硬性过滤：最低改变数
            if n_changes < min_n_changes:
                _rej["low_changes"] += 1
                if len(_rej_details) < 10:
                    _rej_details.append({"reason": "low_changes", "label": consequence_label,
                                         "fires": n_fires_val, "chg": n_changes, "imp": n_improved, "wor": n_worsened})
                continue

            # LBoost-style loose admission for PROPAGATION rules (sim/group): on a
            # strong select-base they would flip already-correct labels → low
            # precision / zero-improved HERE, but their value is at test-time with
            # human seeds (RILL), not the zero-budget select metric. Admit them when
            # --prop_admit_on_precision (they still must FIRE — low_fires unchanged).
            _prop_loose = prop_admit_on_precision and any(
                isinstance(p, (SimPredicate, GroupPredicate)) for p in rule.body)

            # 硬性过滤：完全无正纠正 → 跳过
            if n_changes > 0 and n_improved == 0 and not _prop_loose:
                _rej["zero_improved"] += 1
                continue

            # 硬性过滤：corr_prec 低于门槛（REMOVE 规则不降级）→ 跳过
            if consequence_op == "remove":
                _eff_min_prec = min_corr_prec
            else:
                _eff_min_prec = _adaptive_corr_prec(min_corr_prec, _label_pos_bs)
            if n_changes > 0 and corr_prec < _eff_min_prec and not _prop_loose:
                _rej["low_corr_prec"] += 1
                if len(_rej_details) < 10:
                    _rej_details.append({"reason": "low_corr_prec", "label": consequence_label,
                                         "fires": n_fires_val, "chg": n_changes, "prec": round(corr_prec, 3),
                                         "imp": n_improved, "wor": n_worsened,
                                         "eff_min_prec": round(_eff_min_prec, 2)})
                continue

            # Accept cross-check: 验证规则在训练集上也能泛化
            if accept_docs is not None and accept_labels is not None:
                _acc_fires = np.ones(len(accept_docs), dtype=bool)
                for pred in rule.body:
                    pidx = _pred_to_idx.get(id(pred))
                    if pidx is not None and acc_pred_masks is not None:
                        _acc_fires &= acc_pred_masks[pidx]
                    else:
                        _acc_fires &= np.array(
                            [bool(pred(_make_proxy(d))) for d in accept_docs], dtype=bool)
                n_acc = int(_acc_fires.sum())
                if n_acc > 0:
                    _a_old = accept_base_predictions[_acc_fires, label_idx]
                    _a_true = accept_labels[_acc_fires, label_idx]
                    _a_new_v = 1.0 if consequence_op == "add" else 0.0
                    _a_imp = int(((np.full_like(_a_old, _a_new_v) == _a_true) & (_a_old != _a_true)).sum())
                    _a_wor = int(((_a_old == _a_true) & (np.full_like(_a_old, _a_new_v) != _a_true)).sum())
                    if _a_imp == 0 or (_a_imp + _a_wor > 0 and _a_imp / (_a_imp + _a_wor) < _eff_min_prec):
                        _rej["accept_low_prec"] += 1
                        if len(_rej_details) < 10:
                            _a_prec = _a_imp / (_a_imp + _a_wor) if (_a_imp + _a_wor) > 0 else 0.0
                            _rej_details.append({"reason": "accept_low_prec", "label": consequence_label,
                                                 "val_fires": n_fires_val, "val_prec": round(corr_prec, 3),
                                                 "train_fires": n_acc, "train_imp": _a_imp, "train_wor": _a_wor,
                                                 "train_prec": round(_a_prec, 3)})
                        continue  # fails training set generalization
                else:
                    _rej["accept_zero_fires"] += 1
                    if len(_rej_details) < 10:
                        _rej_details.append({"reason": "accept_zero_fires", "label": consequence_label,
                                             "val_fires": n_fires_val, "train_fires": 0})
                    continue  # 0 fires on training set → overfit to val

            # 独立评估 F1 增益（per-label，与 Optuna reward 对齐）
            all_new_f1s = _fast_per_label_f1(val_labels, test_preds)
            all_old_f1s = _fast_per_label_f1(val_labels, existing_predictions)
            target_label_gain = float(all_new_f1s[label_idx] - all_old_f1s[label_idx])

            # 必须在目标标签上带来 F1 提升（最低正增益门槛）
            _min_label_gain = 0.001
            if target_label_gain < _min_label_gain:
                _rej["low_label_gain"] += 1
                if len(_rej_details) < 10:
                    _rej_details.append({"reason": "low_label_gain", "label": consequence_label,
                                         "fires": n_fires_val, "prec": round(corr_prec, 3),
                                         "imp": n_improved, "wor": n_worsened,
                                         "label_gain": round(target_label_gain, 6)})
                continue

            _rej["passed"] += 1

            # macro-F1 仅用于展示/日志
            new_f1 = _fast_macro_f1(val_labels, test_preds)

            scored_candidates.append(
                (corr_prec, (trial.value if trial is not None else rule.score) or 0.0, rule, fires, label_idx, new_f1,
                 n_improved, n_worsened)
            )

        # ── rejection 诊断日志输出 ──
        _total = sum(_rej.values())
        logger.info("[batch_select] Pass 1 rejection breakdown (%d total trials):", _total)
        for _reason, _cnt in sorted(_rej.items(), key=lambda x: -x[1]):
            if _cnt > 0:
                logger.info("  %-20s %4d  (%.1f%%)", _reason, _cnt, 100.0 * _cnt / max(_total, 1))
        if _rej_details:
            logger.info("[batch_select] Top rejected rule details:")
            for _d in _rej_details[:10]:
                logger.info("  %s", _d)

        # ================================================================
        # Pass 2: 按 BO objective score↓ 排序，贪心筛选
        # (不按 corr_prec 排序，避免小样本霸权问题)
        # ================================================================
        if sort_by_gain:
            scored_candidates.sort(key=lambda x: -x[1])

        discovered_rules: List[RDL] = []
        discovered_fires: List[Tuple[np.ndarray, int]] = []  # (fires_mask, label_idx)
        # 在 greedy 模式下用于跟踪累积状态（per-label F1）
        greedy_preds = existing_predictions.copy() if selection_mode == "greedy" else None
        greedy_label_f1s = _fast_per_label_f1(val_labels, existing_predictions) if selection_mode == "greedy" else None

        for (corr_prec, trial_val, rule, fires, label_idx, indep_f1,
             n_improved, n_worsened) in scored_candidates:

            # 冗余检查 1: body 结构重叠
            if _is_redundant(rule, discovered_rules):
                continue
            # 冗余检查 2: fires IoU > 0.95 且同一 consequence label → 跳过
            _fires_dup = False
            for prev_fires, prev_lidx in discovered_fires:
                if prev_lidx != label_idx:
                    continue
                intersection = (fires & prev_fires).sum()
                union = (fires | prev_fires).sum()
                if union > 0 and intersection / union > 0.95:
                    _fires_dup = True
                    break
            if _fires_dup:
                continue

            if selection_mode == "greedy":
                # 重新模拟：基于已累积的预测
                test_preds = greedy_preds.copy()
                if rule.consequence_op == "add":
                    test_preds[fires, label_idx] = 1.0
                elif rule.consequence_op == "remove":
                    test_preds[fires, label_idx] = 0.0
                # per-label F1 gate（与 Optuna reward 对齐）
                all_eff_f1s = _fast_per_label_f1(val_labels, test_preds)
                eff_label_gain = float(all_eff_f1s[label_idx] - greedy_label_f1s[label_idx])
                if eff_label_gain <= 0:
                    continue  # 在累积状态下目标标签不再有增益
                # 基于累积状态重算 corr_prec（Pass 1 的值是 stale 的）
                old_hit = greedy_preds[fires, label_idx] == val_labels[fires, label_idx]
                new_hit = test_preds[fires, label_idx] == val_labels[fires, label_idx]
                n_improved = int((new_hit & ~old_hit).sum())
                n_worsened = int((old_hit & ~new_hit).sum())
                n_changes_live = n_improved + n_worsened
                if n_changes_live > 0 and n_improved == 0:
                    continue  # 累积状态下全是负面影响
                _n_fires_live = int(fires.sum())
                corr_prec = n_improved / (n_improved + n_worsened + 1) if n_changes_live > 0 else 0.0  # Laplace+1
                # 接受规则，更新累积状态
                greedy_preds = test_preds
                greedy_label_f1s = all_eff_f1s
                eff_f1 = _fast_macro_f1(val_labels, test_preds)  # for display
                display_f1 = eff_f1
            else:
                display_f1 = indep_f1

            # 填充 val_stats 并接受规则
            enriched_rule = RDL(
                body=rule.body,
                consequence=rule.consequence,
                consequence_op=rule.consequence_op,
                score=rule.score,
                coverage=rule.coverage,
                trial_number=rule.trial_number,
                val_stats={
                    "n_fires": int(fires.sum()),
                    "n_redundant": int(fires.sum()) - n_improved - n_worsened,
                    "n_improved": n_improved,
                    "n_worsened": n_worsened,
                    "corr_prec": round(corr_prec, 4),
                    "f1_gain": round(display_f1 - baseline_f1, 4),
                    "display_f1": round(display_f1, 4),
                },
            )
            discovered_rules.append(enriched_rule)
            discovered_fires.append((fires, label_idx))
            if verbose:
                n_changes = n_improved + n_worsened
                print(f"  [batch_select] +rule {rule}  "
                      f"F1={display_f1:.4f} (gain={display_f1 - baseline_f1:.4f})  "
                      f"corr_prec={corr_prec:.2f}  "
                      f"n_fire={int(fires.sum())}  "
                      f"improved={n_improved} worsened={n_worsened}")

        if verbose:
            print(f"  [batch_select] 完成: {len(discovered_rules)} 条规则 "
                  f"(mode={selection_mode}, min_corr_prec={min_corr_prec})")

        # ================================================================
        # Post-processing: inject per-label optimal baseline ML rules
        # ================================================================
        if inject_ml_baseline and _ml_proba and _ml_model_names:
            from sklearn.metrics import f1_score as _f1_score

            existing_ml = {}
            for r in discovered_rules:
                if (len(r.body) == 1
                        and isinstance(r.body[0], MLThresholdPredicate)):
                    existing_ml.setdefault(r.consequence, []).append(
                        r.body[0].threshold)

            _thresh_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            _injected = []
            for _lidx, _label in enumerate(label_list):
                model_results = []  # (f1, thresh, model_name)
                for _mn in _ml_model_names:
                    if _mn not in _ml_proba:
                        continue
                    try:
                        _model_obj = _get_ml_model(_mn)
                    except RuntimeError:
                        continue
                    _ml_lidx = _model_obj.label_index(_label)
                    if _ml_lidx < 0:
                        continue
                    _proba_col = _ml_proba[_mn][:, _ml_lidx]
                    best_f1_for_model = 0.0
                    best_thresh_for_model = 0.5
                    for _t in _thresh_grid:
                        _preds_t = (_proba_col >= _t).astype(np.int32)
                        _f1_t = _f1_score(
                            val_labels[:, _lidx], _preds_t, zero_division=0)
                        if _f1_t > best_f1_for_model:
                            best_f1_for_model = _f1_t
                            best_thresh_for_model = _t
                    if best_f1_for_model > 0:
                        model_results.append(
                            (best_f1_for_model, best_thresh_for_model, _mn))
                model_results.sort(key=lambda x: -x[0])
                for f1_val, thresh, model_name in model_results[:2]:
                    if (_label in existing_ml
                            and min(existing_ml[_label]) <= thresh):
                        continue
                    _baseline_rule = RDL(
                        body=(MLThresholdPredicate(
                            model_name=model_name,
                            label=_label,
                            threshold=thresh),),
                        consequence=_label,
                        consequence_op="add",
                        score=0.0,
                        coverage=0.0,
                        trial_number=-1,
                        val_stats={"injected_baseline": True,
                                   "best_f1": float(f1_val)},
                    )
                    _injected.append(_baseline_rule)
                    discovered_rules.append(_baseline_rule)

            # Clean up redundant pure ML rules per label (keep top-2 lowest threshold)
            for _label in label_list:
                _ml_rules = [
                    (i, r) for i, r in enumerate(discovered_rules)
                    if (r is not None
                        and r.consequence == _label
                        and len(r.body) == 1
                        and isinstance(r.body[0], MLThresholdPredicate))
                ]
                if len(_ml_rules) <= 2:
                    continue
                _ml_rules.sort(key=lambda x: x[1].body[0].threshold)
                for _idx, _ in _ml_rules[2:]:
                    discovered_rules[_idx] = None
            discovered_rules = [r for r in discovered_rules if r is not None]

            logger.info("[batch_select] Injected %d baseline ML rules, "
                        "final rule count: %d", len(_injected),
                        len(discovered_rules))

        return RDLSet(rules=discovered_rules, label_names=label_list)


# ======================================================================
# Error-driven candidate filtering
# ======================================================================

def _error_driven_filter(
    candidate_predicates: list,
    fire_masks: np.ndarray,
    existing_predictions: np.ndarray,
    val_labels: np.ndarray,
    label_list: List[str],
    consequence_op: str,
    min_discrimination: float = 0.03,
    top_k_per_label: int = 30,
) -> List[int]:
    """Multi-strategy candidate predicate filter for ML errors.

    Uses three complementary strategies (union of results):
      S1 — single-variable discrimination (fire-rate gap)
      S2 — decision-tree feature importance (captures combinations)
      S3 — pairwise interaction boost (pairs whose AND improves disc)

    Returns sorted list of predicate indices to keep.
    """
    from sklearn.tree import DecisionTreeClassifier

    n_cands, n_docs = fire_masks.shape
    n_labels = len(label_list)
    keep_s1: set = set()
    keep_s2: set = set()
    keep_s3: set = set()

    _k1 = top_k_per_label
    _k2 = max(top_k_per_label // 2, 10)
    _k3 = max(top_k_per_label // 3, 5)

    for lidx in range(n_labels):
        if consequence_op == "remove":
            errors = (existing_predictions[:, lidx] == 1) & (val_labels[:, lidx] == 0)
        else:
            errors = (existing_predictions[:, lidx] == 0) & (val_labels[:, lidx] == 1)

        n_err = int(errors.sum())
        if n_err < 2:
            continue
        if (n_docs - n_err) < 1:
            continue

        err_rates = fire_masks[:, errors].mean(axis=1)
        non_rates = fire_masks[:, ~errors].mean(axis=1)
        disc = np.abs(err_rates - non_rates)

        # ── S1: single-variable discrimination ──
        above_min = np.where(disc >= min_discrimination)[0]
        if len(above_min) == 0:
            s1_idx = np.argsort(-disc)[:max(_k1 // 3, 5)]
        elif len(above_min) <= _k1:
            s1_idx = above_min
        else:
            sub_disc = disc[above_min]
            s1_idx = above_min[np.argsort(-sub_disc)[:_k1]]
        keep_s1.update(s1_idx.tolist())

        # ── S2: decision-tree feature importance ──
        y = errors.astype(np.int32)
        if y.sum() >= 3 and (len(y) - y.sum()) >= 3:
            X = fire_masks.T.astype(np.float32)
            try:
                tree = DecisionTreeClassifier(
                    max_depth=4, min_samples_leaf=max(2, n_err // 10),
                    class_weight="balanced", random_state=lidx,
                )
                tree.fit(X, y)
                fi = tree.feature_importances_
                s2_idx = np.argsort(-fi)[:_k2]
                s2_idx = s2_idx[fi[s2_idx] > 0]
                keep_s2.update(s2_idx.tolist())
            except Exception:
                pass

        # ── S3: pairwise interaction boost ──
        s1_top = np.argsort(-disc)[:min(40, n_cands)]
        best_pairs = []
        for i_pos in range(len(s1_top)):
            for j_pos in range(i_pos + 1, len(s1_top)):
                pi, pj = s1_top[i_pos], s1_top[j_pos]
                combo = fire_masks[pi].astype(bool) & fire_masks[pj].astype(bool)
                c_err = combo[errors].mean() if errors.any() else 0.0
                c_non = combo[~errors].mean() if (~errors).any() else 0.0
                c_disc = abs(c_err - c_non)
                boost = c_disc - max(disc[pi], disc[pj])
                if boost > 0.01:
                    best_pairs.append((boost, pi, pj))
        best_pairs.sort(key=lambda x: -x[0])
        for _, pi, pj in best_pairs[:_k3]:
            keep_s3.add(pi)
            keep_s3.add(pj)

    all_kept = keep_s1 | keep_s2 | keep_s3
    logger.info("[ErrorFilter] S1=%d, S2=%d, S3=%d → union=%d / %d (%s)",
                len(keep_s1), len(keep_s2), len(keep_s3),
                len(all_kept), n_cands, consequence_op)

    if not all_kept:
        return list(range(n_cands))

    return sorted(all_kept)


# ======================================================================
# Decision-tree warm-up rule seeding
# ======================================================================

def _tree_seeded_rules(
    fire_masks: np.ndarray,
    candidate_predicates: list,
    ml_proba_cache: Dict[str, np.ndarray],
    existing_predictions: np.ndarray,
    val_labels: np.ndarray,
    label_list: List[str],
    consequence_op: str,
    max_depth: int = 3,
    n_trees: int = 5,
    min_improved: int = 2,
    min_precision: float = 0.6,
) -> List["RDL"]:
    """Use shallow decision trees to discover high-discrimination rules.

    Trains trees to predict ML errors (FP for remove, FN for add) using
    predicate fire masks as features.  Extracts leaf-to-root paths and
    converts them into RDL rules.
    """
    from sklearn.tree import DecisionTreeClassifier

    n_cands, n_docs = fire_masks.shape
    X = fire_masks.T.astype(np.float32)

    ml_cols = []
    ml_col_info = []
    for model_name, proba in ml_proba_cache.items():
        for ml_lidx in range(proba.shape[1]):
            ml_cols.append(proba[:, ml_lidx])
            ml_col_info.append((model_name, ml_lidx))
    if ml_cols:
        ml_matrix = np.column_stack(ml_cols).astype(np.float32)
        X_full = np.hstack([X, ml_matrix])
    else:
        X_full = X
        ml_matrix = None

    n_text_features = n_cands
    rules: List["RDL"] = []

    for lidx, label in enumerate(label_list):
        if consequence_op == "remove":
            y = ((existing_predictions[:, lidx] == 1) &
                 (val_labels[:, lidx] == 0)).astype(np.int32)
        else:
            y = ((existing_predictions[:, lidx] == 0) &
                 (val_labels[:, lidx] == 1)).astype(np.int32)

        n_pos = int(y.sum())
        if n_pos < min_improved:
            continue

        for tree_i in range(n_trees):
            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=max(2, min_improved),
                class_weight="balanced",
                random_state=42 + tree_i + lidx * 1000,
            )
            clf.fit(X_full, y)

            tree = clf.tree_
            for leaf_id in range(tree.node_count):
                if tree.children_left[leaf_id] != tree.children_right[leaf_id]:
                    continue
                leaf_class = int(np.argmax(tree.value[leaf_id]))
                if leaf_class != 1:
                    continue
                n_leaf = int(tree.value[leaf_id].sum())
                if n_leaf < min_improved:
                    continue

                path_features = []
                path_thresholds = []
                path_directions = []
                node = leaf_id
                while node != 0:
                    parent = _find_parent(tree, node)
                    if parent < 0:
                        break
                    feat = tree.feature[parent]
                    thresh = tree.threshold[parent]
                    go_left = (tree.children_left[parent] == node)
                    path_features.append(feat)
                    path_thresholds.append(thresh)
                    path_directions.append(go_left)
                    node = parent

                body = []
                for feat, thresh, go_left in zip(
                    path_features, path_thresholds, path_directions
                ):
                    if feat < n_text_features:
                        if go_left and thresh < 0.5:
                            continue
                        if not go_left and thresh >= 0.5:
                            body.append(candidate_predicates[feat])
                        elif go_left and thresh >= 0.5:
                            body.append(candidate_predicates[feat])
                    else:
                        ml_offset = feat - n_text_features
                        if ml_offset < len(ml_col_info):
                            m_name, m_lidx = ml_col_info[ml_offset]
                            try:
                                model_obj = _get_ml_model(m_name)
                                _ll = getattr(model_obj, 'label_list', None) or getattr(model_obj, '_label_names', None)
                                if _ll is None:
                                    continue
                                ml_label = _ll[m_lidx]
                            except (RuntimeError, KeyError, IndexError, AttributeError):
                                continue
                            if go_left:
                                body.append(MLThresholdPredicate(
                                    model_name=m_name, label=ml_label,
                                    threshold=round(float(thresh), 4)))
                            else:
                                body.append(MLThresholdPredicate(
                                    model_name=m_name, label=ml_label,
                                    threshold=round(float(thresh) + 0.01, 4)))

                if not body:
                    continue

                fires = np.ones(n_docs, dtype=bool)
                for pred in body:
                    if isinstance(pred, MLThresholdPredicate):
                        m = pred.model_name
                        if m in ml_proba_cache:
                            try:
                                mobj = _get_ml_model(m)
                                mi = mobj.label_index(pred.label)
                                fires &= ml_proba_cache[m][:, mi] >= pred.threshold
                            except (RuntimeError, KeyError):
                                fires[:] = False
                                break
                    else:
                        pidx = None
                        for ci, cp in enumerate(candidate_predicates):
                            if cp is pred:
                                pidx = ci
                                break
                        if pidx is not None and pidx < fire_masks.shape[0]:
                            fires &= fire_masks[pidx]
                        else:
                            fires[:] = False
                            break

                n_fire = int(fires.sum())
                if n_fire == 0:
                    continue

                old_correct = (existing_predictions[fires, lidx] ==
                               val_labels[fires, lidx])
                new_preds = existing_predictions.copy()
                if consequence_op == "add":
                    new_preds[fires, lidx] = 1.0
                else:
                    new_preds[fires, lidx] = 0.0
                new_correct = new_preds[fires, lidx] == val_labels[fires, lidx]

                n_imp = int((new_correct & ~old_correct).sum())
                n_wor = int((old_correct & ~new_correct).sum())

                if n_imp < min_improved:
                    continue
                prec = n_imp / max(n_imp + n_wor, 1)
                if prec < min_precision:
                    continue

                rule = RDL(
                    body=tuple(body),
                    consequence=label,
                    consequence_op=consequence_op,
                    score=float(prec * np.sqrt(n_imp)),
                    coverage=n_fire / n_docs,
                    trial_number=-2,
                )
                rules.append(rule)

    seen = set()
    unique = []
    for r in rules:
        key = (r.consequence, r.consequence_op, frozenset(id(p) for p in r.body))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _find_parent(tree, node_id: int) -> int:
    """Find parent of a node in sklearn decision tree."""
    for i in range(tree.node_count):
        if tree.children_left[i] == node_id or tree.children_right[i] == node_id:
            return i
    return -1


# ======================================================================
# Staged pipeline helpers
# ======================================================================

def _vectorized_staged_predict(
    rules: List["RDL"],
    label_names: List[str],
    n_docs: int,
    ml_proba_cache: Dict[str, np.ndarray],
    text_fire_masks: Optional[np.ndarray] = None,
    text_pred_to_idx: Optional[Dict[int, int]] = None,
    base: Optional[np.ndarray] = None,
) -> np.ndarray:
    """用预计算的 fire masks 和 ML proba 向量化构造预测矩阵。

    双集合语义（Church-Rosser）：ADD 写入 pos 集合，REMOVE 写入 neg 集合，
    最终 predictions = pos & ~neg。规则应用顺序不影响结果。
    """
    n_labels = len(label_names)
    _label2idx = {name: i for i, name in enumerate(label_names)}

    if base is not None:
        pos = (np.asarray(base, dtype=np.float32) > 0)
    else:
        pos = np.zeros((n_docs, n_labels), dtype=bool)
    neg = np.zeros((n_docs, n_labels), dtype=bool)

    for rule in rules:
        lidx = _label2idx.get(rule.consequence)
        if lidx is None:
            continue

        fires = np.ones(n_docs, dtype=bool)
        _skip = False
        for pred in rule.body:
            if isinstance(pred, MLThresholdPredicate):
                if pred.model_name not in ml_proba_cache:
                    _skip = True
                    break
                try:
                    model_obj = _get_ml_model(pred.model_name)
                    ml_lidx = model_obj.label_index(pred.label)
                except (RuntimeError, KeyError):
                    _skip = True
                    break
                if ml_lidx < 0:
                    _skip = True
                    break
                fires &= (ml_proba_cache[pred.model_name][:, ml_lidx] >= pred.threshold)
            elif isinstance(pred, (SimPredicate, LabelPredicate)):
                _skip = True
                break
            else:
                if text_pred_to_idx is not None and text_fire_masks is not None:
                    pidx = text_pred_to_idx.get(id(pred))
                    if pidx is not None:
                        fires &= text_fire_masks[pidx]
                    else:
                        _skip = True
                        break
                else:
                    _skip = True
                    break

        if _skip:
            continue

        if rule.consequence_op == "remove":
            neg[fires, lidx] = True
        else:  # "add"
            pos[fires, lidx] = True

    return (pos & ~neg).astype(np.float32)

def _save_stage_logs(exp_dir: str, stage_data: Dict[str, Any]) -> None:
    import json as _json
    log_path = os.path.join(exp_dir, "staged_pipeline_logs.json")
    with open(log_path, "w") as f:
        _json.dump(stage_data, f, indent=2, default=str)
    logger.info("Staged pipeline logs saved to %s", log_path)


def _extract_trial_stats(rules) -> List[Dict]:
    stats = []
    for rule in rules:
        if rule is None:
            continue
        entry = {
            "label": rule.consequence,
            "consequence_op": rule.consequence_op,
            "body_len": len(rule.body),
            "body_types": [type(p).__name__ for p in rule.body],
            "body_str": str(rule.body),
            "score": round(float(rule.score), 4),
            "trial_number": rule.trial_number,
        }
        if hasattr(rule, "val_stats") and rule.val_stats:
            entry.update(rule.val_stats)
        stats.append(entry)
    return stats
