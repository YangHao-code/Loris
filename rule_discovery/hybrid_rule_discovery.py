"""
hybrid_rule_discovery
---------------------
混合架构规则搜索引擎 (Hybrid Rule Discovery)。

核心思想：将规则的「结构搜索」和「内容实例化」解耦——
  Level 1 (Optuna BO)：仅决定每种谓词大类的数量（~12 参数），
  Level 2 (Greedy)   ：在预计算布尔掩码上用信息增益 + Precision + 低重叠惩罚
                       贪心选出具体谓词实例。

相比原版 `loris_rule_discovery.RuleLearner` 搜索空间从 50+ 降至 ~12，
同时通过预计算 fire masks 和 freq counts 避免循环内字符串匹配与模型推理。
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
from rule_discovery.loris_rule_discovery import (
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
ML_THRESHOLD_BINS: List[float] = [0.4, 0.5, 0.6, 0.8]

# 贪心评分默认权重
DEFAULT_GREEDY_WEIGHTS: Dict[str, float] = {
    "precision": 0.5,
    "info_gain": 0.2,
    "coverage": 0.2,
    "diversity": 0.1,
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
                    from pattern_extraction.predicates import (
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
        ("LabelPredicate_minus", "minus"),
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

def evaluate_hybrid_configuration(
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
    # ── CDF-based (label, op) sampling ──
    label_op_cdf: Optional[np.ndarray] = None,
    label_op_list: Optional[List[Tuple[int, str]]] = None,
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
    # Step 1: Suggest 宏观结构（~12 个 Optuna 参数）
    # ==============================================================
    structure: Dict[str, int] = {}

    # 文本谓词：Match/Freq [0,2], Cooccur/Before [0,1] (pair 谓词自身已是 conjunction)
    _PAIR_TYPES = {"CooccurPredicate", "BeforePredicate"}
    for type_name in _TEXT_TYPE_ORDER:
        pool = candidate_pools.get(type_name, [])
        cap = 1 if type_name in _PAIR_TYPES else 2
        hi = min(cap, len(pool)) if pool else 0
        if hi > 0:
            structure[type_name] = trial.suggest_int(f"n_{type_name}", 0, hi)
        else:
            structure[type_name] = 0

    # ML 谓词 [0, 2]
    if ml_model_names and ml_proba_cache:
        structure["MLThresholdPredicate"] = trial.suggest_int("n_ml", 0, 2)
    else:
        structure["MLThresholdPredicate"] = 0

    # Label 谓词：每种 op [0, 2]
    structure["LabelPredicate_contains"] = trial.suggest_int("n_label_contains", 0, 2)
    structure["LabelPredicate_eq"] = 0       # disabled: only keep contains
    structure["LabelPredicate_minus"] = 0    # disabled: only keep contains

    total_slots = sum(structure.values())
    if total_slots == 0:
        _diag_inc("empty_body")
        return -1.0  # 空 body，直接剪枝

    # 强制至少一个文本谓词（不允许纯 label / ML 规则）
    n_text = sum(structure.get(t, 0) for t in _TEXT_TYPE_ORDER)
    if n_text == 0:
        _diag_inc("no_text_predicate")
        return -1.0

    # Consequence — CDF trick: 单个 suggest_float 同时决定 (label, op)
    if label_op_cdf is not None and label_op_list is not None and len(label_op_list) > 0:
        _pointer = trial.suggest_float("label_op_pointer", 0.0, 1.0)
        _local_idx = min(int(np.searchsorted(label_op_cdf, _pointer)),
                         len(label_op_list) - 1)
        label_idx, consequence_op = label_op_list[_local_idx]
        consequence_label = label_list[label_idx]
    elif cluster_label_indices is not None and len(cluster_label_indices) > 0:
        available_labels = [label_list[i] for i in cluster_label_indices]
        consequence_label = trial.suggest_categorical("consequence_label", available_labels)
        label_idx = label_list.index(consequence_label)
        consequence_op = trial.suggest_categorical("consequence_op", ["add", "remove"])
    else:
        consequence_label = trial.suggest_categorical("consequence_label", label_list)
        label_idx = label_list.index(consequence_label)
        consequence_op = trial.suggest_categorical("consequence_op", ["add", "remove"])

    # ==============================================================
    # Step 2: 贪心实例化（Level 2 — 位运算加速）
    # 如果 split mode，在 fit (train) 数据上实例化谓词
    # ==============================================================
    if _split:
        body, _fit_body_masks, _sel_indices = greedy_instantiate(
            structure=structure,
            candidate_pools=candidate_pools,
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
        body, _body_masks, _sel_indices = greedy_instantiate(
            structure=structure,
            candidate_pools=candidate_pools,
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

    if not body:
        _diag_inc("greedy_empty")
        return -1.0  # 贪心未选出任何谓词

    # ==============================================================
    # Step 3: 计算合取触发掩码（bitwise AND）
    # 评估始终在 val_docs 上
    # ==============================================================
    if _split:
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
        return -1.0  # 完全不触发 → 无信息
    # fires_factor: Sigmoid 软阈值 — 远低于 min_rule_fires 时 ≈0，远高于时 ≈1
    # 给 TPE 平滑的连续梯度，而非线性衰减
    import math as _math
    _mrf = max(min_rule_fires, 1)
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
            return -1.0  # fire 了但没改变任何预测 → 无意义

        # Soft changes penalty (替代 hard min_n_changes rejection)
        changes_factor = min(1.0, n_changes / max(min_n_changes, 1))

        corr_prec = n_improved / (n_improved + n_worsened + 2)  # Laplace smoothing
        # 硬性拒绝：完全无正纠正
        if n_improved == 0:
            _diag_inc(f"zero_improved(worsened={n_worsened})")
            return -0.5
        # 硬性精度门槛：排除改错比改对多的规则
        if corr_prec < min_corr_prec:
            _diag_inc(f"low_corr_prec({corr_prec:.2f}<{min_corr_prec})")
            return -0.5

    # ==============================================================
    # Step 7: Target-label reward (解决梯度稀释问题)
    # ==============================================================
    # 1. 核心信号：仅关注目标标签的独立 F1 增益 (不被 31 个无关标签稀释)
    all_new_f1s = _fast_per_label_f1(val_labels, new_predictions)
    all_old_f1s = _fast_per_label_f1(val_labels, existing_predictions)
    target_gain = float(all_new_f1s[label_idx] - all_old_f1s[label_idx])

    if target_gain <= 0:
        _diag_inc(f"neg_f1_gain({target_gain:.6f},fires={n_fire},chg={n_changes},imp={n_improved},wor={n_worsened})")
        return target_gain  # 负增益直接返回，加速剪枝

    # 2. 质量乘子：corr_prec² — 改错的越少，reward 放大越多
    precision_multiplier = corr_prec ** 1.5

    # 3. 广度奖励：同等精度下，改变更多样本的规则更有价值
    coverage_bonus = np.log1p(n_changes) / 3.0

    # 4. 复杂度惩罚：body > 2 个谓词开始衰减 (Occam's razor)
    n_body = len(body)
    length_penalty = 0.85 ** max(0, n_body - 1)

    # 终极融合：fires_factor 和 changes_factor 提供连续梯度
    f1_gain = (target_gain * precision_multiplier
               * (1.0 + coverage_bonus) * length_penalty
               * fires_factor * changes_factor)

    _diag_inc(f"positive(gain={f1_gain:.4f},tgt={target_gain:.4f},prec={corr_prec:.2f},fires={n_fire},ff={fires_factor:.2f},chg={n_changes},imp={n_improved},wor={n_worsened},body={n_body})")

    # 缓存正增益 trial 的 body，供 run_bo()/discover() 直接构建 RDL（避免重跑 greedy）
    if _body_cache is not None and f1_gain > 0:
        _body_cache[trial.number] = (
            tuple(body), consequence_label, consequence_op, coverage,
        )

    return f1_gain


# ===========================================================================
# HybridRuleLearner — 混合架构规则搜索引擎
# ===========================================================================

class HybridRuleLearner:
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
            print(f"  [Hybrid] 预计算完成: fire_masks={self._fire_masks.shape}, "
                  f"freq_counts={len(self._freq_counts)}, "
                  f"ml_models={len(self._ml_proba_cache)}")

        # 5. 预计算 LabelPredicate 掩码 (避免每个 trial 重复 eval)
        self._label_pred_masks: Dict[Tuple[str, str], np.ndarray] = {}
        _lp_ops = ["contains"]  # eq/minus 已在 Mod 5 中禁用，仍预计算以备兼容
        if any(self._candidate_pools.get(f"LabelPredicate_{op}", [])
               for op in ["eq", "minus"]):
            _lp_ops.extend(["eq", "minus"])
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
                print(f"  [Hybrid] Fit 预计算完成: fire_masks={self._fit_fire_masks.shape}")

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
        body, body_masks, _ = greedy_instantiate(
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
            _label_op_list: List[Tuple[int, str]] = []
            _weights_list: List[float] = []
            for _li in _active_labels:
                _fn = int(((_vl[:, _li] == 1) & (_ep[:, _li] == 0)).sum())  # add 可修
                _fp = int(((_vl[:, _li] == 0) & (_ep[:, _li] == 1)).sum())  # remove 可修
                if _fn >= _min_feasible:
                    _label_op_list.append((_li, "add"))
                    _weights_list.append(float(_fn))
                if _fp >= _min_feasible:
                    _label_op_list.append((_li, "remove"))
                    _weights_list.append(float(_fp))
            if _label_op_list:
                _w = np.array(_weights_list, dtype=float)
                _w /= _w.sum()
                _label_op_cdf = np.cumsum(_w)
                _label_op_cdf[-1] = 1.0
            else:
                _label_op_cdf = None
                _label_op_list = None
            if self.verbose and _cli is not None:
                _n_add = sum(1 for _, op in (_label_op_list or []) if op == "add")
                _n_rem = sum(1 for _, op in (_label_op_list or []) if op == "remove")
                logger.info("  Label-op pruning: %d active → %d entries (%d add, %d remove, min_errors=%d)",
                            len(_active_labels), len(_label_op_list) if _label_op_list else 0,
                            _n_add, _n_rem, _min_feasible)

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
                return evaluate_hybrid_configuration(
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
                    label_op_cdf=_label_op_cdf,
                    label_op_list=_label_op_list,
                    **_fit_kw,
                )

            # 执行优化
            study.optimize(objective, n_trials=self.max_trials)

            # 诊断日志
            all_values = [t.value for t in study.trials if t.value is not None]
            if all_values:
                pos_vals = [v for v in all_values if v > 0]
                logger.info(
                    "[Hybrid] round %d: %d trials, %d positive (max=%.4f), best=%.4f",
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

        # ── 4a+4b: 排除不可行标签 + (label, op) CDF 采样 ──
        _active_labels = self.cluster_label_indices if self.cluster_label_indices is not None else list(range(len(self.label_list)))
        _min_feasible = max(4, int(np.ceil(self.min_corr_prec * 2 / (1 - self.min_corr_prec)))) if self.min_corr_prec < 1.0 else 4
        _label_op_list: List[Tuple[int, str]] = []
        _weights_list: List[float] = []
        for _li in _active_labels:
            _fn = int(((self.val_labels[:, _li] == 1) & (_ep[:, _li] == 0)).sum())
            _fp = int(((self.val_labels[:, _li] == 0) & (_ep[:, _li] == 1)).sum())
            if _fn >= _min_feasible:
                _label_op_list.append((_li, "add"))
                _weights_list.append(float(_fn))
            if _fp >= _min_feasible:
                _label_op_list.append((_li, "remove"))
                _weights_list.append(float(_fp))
        if _label_op_list:
            _w = np.array(_weights_list, dtype=float)
            _w /= _w.sum()
            _label_op_cdf = np.cumsum(_w)
            _label_op_cdf[-1] = 1.0
        else:
            _label_op_cdf = None
            _label_op_list = None
        if self.verbose:
            _n_add = sum(1 for _, op in (_label_op_list or []) if op == "add")
            _n_rem = sum(1 for _, op in (_label_op_list or []) if op == "remove")
            logger.info("  Label-op pruning: %d active → %d entries (%d add, %d remove, min_errors=%d)",
                        len(_active_labels), len(_label_op_list) if _label_op_list else 0,
                        _n_add, _n_rem, _min_feasible)

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
            return evaluate_hybrid_configuration(
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
                _body_cache=_body_cache,
                label_op_cdf=_label_op_cdf,
                label_op_list=_label_op_list,
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

        return completed

    # ------------------------------------------------------------------
    # Batch 全局筛选：贪心添加规则到 RDLSet
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
        # ── accept cross-check: 训练集泛化验证 ──
        accept_docs: Optional[List[Document]] = None,
        accept_labels: Optional[np.ndarray] = None,
        accept_base_predictions: Optional[np.ndarray] = None,
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

        # 使用已有的 joblib 并行函数预计算 val_docs 上的 fire masks
        logger.info("[batch_select] Precomputing fire masks for %d unique predicates "
                    "on %d val_docs ...", len(unique_preds), len(val_docs))
        val_pred_masks = precompute_fire_masks(unique_preds, val_docs)
        # 建立 pred id → mask index 映射
        _pred_to_idx = {id(p): i for i, p in enumerate(unique_preds)}

        # accept_docs 同理
        acc_pred_masks = None
        if accept_docs is not None:
            logger.info("[batch_select] Precomputing fire masks for %d unique predicates "
                        "on %d accept_docs ...", len(unique_preds), len(accept_docs))
            acc_pred_masks = precompute_fire_masks(unique_preds, accept_docs)

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
                "neg_label_gain": 0, "passed": 0}
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
                pidx = _pred_to_idx.get(id(pred))
                if pidx is not None:
                    fires &= val_pred_masks[pidx]
                else:
                    # fallback（理论上不会走到这里）
                    fires &= np.array([bool(pred(_make_proxy(d))) for d in val_docs], dtype=bool)

            n_fires_val = int(fires.sum())
            if n_fires_val < min_rule_fires:
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
            corr_prec = n_improved / (n_improved + n_worsened + 2) if n_changes > 0 else 0.0  # Laplace

            # 硬性过滤：最低改变数
            if n_changes < min_n_changes:
                _rej["low_changes"] += 1
                if len(_rej_details) < 10:
                    _rej_details.append({"reason": "low_changes", "label": consequence_label,
                                         "fires": n_fires_val, "chg": n_changes, "imp": n_improved, "wor": n_worsened})
                continue

            # 硬性过滤：完全无正纠正 → 跳过
            if n_changes > 0 and n_improved == 0:
                _rej["zero_improved"] += 1
                continue

            # 硬性过滤：corr_prec 低于门槛 → 跳过
            if n_changes > 0 and corr_prec < min_corr_prec:
                _rej["low_corr_prec"] += 1
                if len(_rej_details) < 10:
                    _rej_details.append({"reason": "low_corr_prec", "label": consequence_label,
                                         "fires": n_fires_val, "chg": n_changes, "prec": round(corr_prec, 3),
                                         "imp": n_improved, "wor": n_worsened})
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
                    if _a_imp == 0 or (_a_imp + _a_wor > 0 and _a_imp / (_a_imp + _a_wor) < min_corr_prec):
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

            # 必须在目标标签上带来 F1 提升（允许微小噪声级负增益）
            _eps = min(0.001, 1.0 / max(len(val_docs), 1))
            if target_label_gain < -_eps:
                _rej["neg_label_gain"] += 1
                if len(_rej_details) < 10:
                    _rej_details.append({"reason": "neg_label_gain", "label": consequence_label,
                                         "fires": n_fires_val, "prec": round(corr_prec, 3),
                                         "imp": n_improved, "wor": n_worsened,
                                         "label_gain": round(target_label_gain, 6)})
                continue

            _rej["passed"] += 1

            # macro-F1 仅用于展示/日志
            new_f1 = _fast_macro_f1(val_labels, test_preds)

            scored_candidates.append(
                (corr_prec, trial.value or 0.0, rule, fires, label_idx, new_f1,
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
        # Pass 2: 按 corr_prec↓, F1 gain↓ 排序，贪心筛选
        # ================================================================
        if sort_by_gain:
            scored_candidates.sort(key=lambda x: (-x[0], -x[1]))

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
                corr_prec = n_improved / (n_improved + n_worsened + 2) if n_changes_live > 0 else 0.0  # Laplace
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

        return RDLSet(rules=discovered_rules, label_names=label_list)
