"""
Group-based propagation rule discovery (Track 2 v7).

Exhaustively enumerates (virtual_attribute, label) combinations and evaluates
correction precision. Optionally rescues borderline rules with text predicates.

Output format is compatible with batch_select: List[Tuple[MockTrial, RDL]].
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from loris.document import Document
from loris.predicates import (
    GroupPredicate,
    LabelPredicate,
    MatchPredicate,
    Predicate,
)
from loris.predicates._core import _Pattern
from loris.rules.rdl import RDL

logger = logging.getLogger(__name__)


@dataclass
class MockTrial:
    """Minimal trial-like object compatible with batch_select interface."""
    number: int = 0
    values: List[float] = field(default_factory=lambda: [0.0])
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """Single-objective score, mirroring optuna FrozenTrial.value.

        batch_select reads ``trial.value`` (singular); optuna's real trials
        expose it, but MockTrial only stored ``values`` (plural). Without this
        the Track2 group-propagation merge raises AttributeError. Returns the
        first objective (or 0.0 if empty).
        """
        return self.values[0] if self.values else 0.0



def _fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fast macro-F1 computation."""
    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0).astype(float)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0).astype(float)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0).astype(float)
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(f1.mean())


def compute_group_fire_mask(
    group_ids: np.ndarray,
    label_idx: int,
    label_state: np.ndarray,
) -> np.ndarray:
    """Compute propagation fire mask for a (group_attr, label) pair.

    mask[i] = True iff:
      - group_ids[i] >= 0 (not degenerate)
      - there exists j != i with group_ids[j] == group_ids[i] and label_state[j, label_idx] == 1

    This is the core "x.attr == y.attr ∧ label(A∈y.lbl)" evaluation.
    """
    n = len(group_ids)
    mask = np.zeros(n, dtype=bool)

    unique_groups = np.unique(group_ids[group_ids >= 0])
    for gid in unique_groups:
        members = np.where(group_ids == gid)[0]
        has_label = label_state[members, label_idx].astype(bool)
        if has_label.any():
            mask[members] = True
            # docs that already have the label via label_state still get mask=True
            # (batch_select handles the "only fire if prediction differs" logic)

    return mask


def evaluate_group_rule(
    fire_mask: np.ndarray,
    label_idx: int,
    val_labels: np.ndarray,
    existing_predictions: np.ndarray,
    min_fires: int = 3,
    min_corr_prec: float = 0.60,
) -> Optional[Dict[str, Any]]:
    """Evaluate a single group propagation rule.

    Returns metrics dict if rule passes thresholds, else None.
    """
    # Only fire on docs where prediction would change (add: pred=0 → pred=1)
    actionable = fire_mask & (existing_predictions[:, label_idx] == 0)
    n_fires = int(actionable.sum())
    if n_fires < min_fires:
        return None

    # Correction precision
    truth_at_fires = val_labels[actionable, label_idx]
    n_improved = int((truth_at_fires == 1).sum())
    n_worsened = int((truth_at_fires == 0).sum())
    if n_improved == 0:
        return None
    corr_prec = n_improved / (n_improved + n_worsened) if (n_improved + n_worsened) > 0 else 0.0

    # F1 gain
    test_preds = existing_predictions.copy()
    test_preds[actionable, label_idx] = 1.0
    old_f1 = _fast_macro_f1(val_labels, existing_predictions)
    new_f1 = _fast_macro_f1(val_labels, test_preds)
    f1_gain = new_f1 - old_f1

    return {
        "n_fires": n_fires,
        "n_improved": n_improved,
        "n_worsened": n_worsened,
        "corr_prec": corr_prec,
        "f1_gain": f1_gain,
    }


def _extract_rescue_predicates(
    fire_mask: np.ndarray,
    label_idx: int,
    val_labels: np.ndarray,
    existing_predictions: np.ndarray,
    val_docs: List[Document],
    top_k: int = 5,
) -> List[Tuple[MatchPredicate, bool]]:
    """Extract discriminative text predicates from TP/FP analysis.

    Returns [(predicate, is_negated), ...] where is_negated=True means NOT match.
    - Positive predicates: words frequent in TP but rare in FP
    - Negative predicates: words frequent in FP but rare in TP
    """
    actionable = fire_mask & (existing_predictions[:, label_idx] == 0)
    truth = val_labels[actionable, label_idx]
    tp_indices = np.where(actionable)[0][truth == 1]
    fp_indices = np.where(actionable)[0][truth == 0]

    if len(tp_indices) < 2 or len(fp_indices) < 2:
        return []

    word_re = re.compile(r'\b[a-z]{3,15}\b')
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                 'have', 'been', 'from', 'this', 'that', 'with', 'they',
                 'will', 'each', 'make', 'like', 'than', 'them', 'its',
                 'over', 'such', 'into', 'also', 'some', 'could', 'which',
                 'when', 'what', 'their', 'there', 'would', 'about', 'these',
                 'other', 'were', 'more', 'then', 'very', 'after', 'should'}

    def _word_freq(indices):
        counter = Counter()
        for idx in indices:
            words = set(word_re.findall(val_docs[idx].cnt.lower()))
            words -= stopwords
            counter.update(words)
        return counter

    tp_freq = _word_freq(tp_indices)
    fp_freq = _word_freq(fp_indices)
    n_tp = max(len(tp_indices), 1)
    n_fp = max(len(fp_indices), 1)

    results: List[Tuple[MatchPredicate, bool]] = []

    # Positive predicates: high in TP, low in FP
    pos_scores = []
    for word, tp_count in tp_freq.most_common(100):
        tp_rate = tp_count / n_tp
        fp_rate = fp_freq.get(word, 0) / n_fp
        if tp_rate > 0.3 and fp_rate < 0.15:
            score = tp_rate - fp_rate
            pos_scores.append((word, score))
    pos_scores.sort(key=lambda x: -x[1])
    for word, _ in pos_scores[:top_k]:
        pred = MatchPredicate(attr="cnt", r=_Pattern(raw=rf'\b{word}\b', flags=re.IGNORECASE))
        results.append((pred, False))

    # Negative predicates: high in FP, low in TP
    neg_scores = []
    for word, fp_count in fp_freq.most_common(100):
        fp_rate = fp_count / n_fp
        tp_rate = tp_freq.get(word, 0) / n_tp
        if fp_rate > 0.3 and tp_rate < 0.15:
            score = fp_rate - tp_rate
            neg_scores.append((word, score))
    neg_scores.sort(key=lambda x: -x[1])
    for word, _ in neg_scores[:top_k]:
        pred = MatchPredicate(attr="cnt", r=_Pattern(raw=rf'\b{word}\b', flags=re.IGNORECASE))
        results.append((pred, True))

    return results


def discover_group_rules(
    virtual_attrs: Dict[str, np.ndarray],
    label_state: np.ndarray,
    label_names: List[str],
    val_labels: np.ndarray,
    existing_predictions: np.ndarray,
    val_docs: List[Document],
    min_fires: int = 3,
    min_corr_prec: float = 0.60,
    min_f1_gain: float = 0.0005,
    rescue_prec_range: Tuple[float, float] = (0.45, 0.60),
    max_rescue_preds: int = 10,
) -> List[Tuple[MockTrial, RDL]]:
    """Exhaustively enumerate (attr, label) group propagation rules.

    Phase 1: Pure group+label rules with corr_prec >= min_corr_prec pass directly.
    Phase 2: Rules with corr_prec in rescue_prec_range get text predicate rescue.

    Returns list compatible with batch_select: [(MockTrial, RDL), ...]
    """
    val_labels = np.asarray(val_labels, dtype=np.float32)
    existing_predictions = np.asarray(existing_predictions, dtype=np.float32)

    results: List[Tuple[MockTrial, RDL]] = []
    borderline: List[Tuple[str, int, np.ndarray, Dict]] = []
    trial_counter = 0

    logger.info("=== Group Rule Discovery: %d attrs × %d labels = %d candidates ===",
                len(virtual_attrs), len(label_names),
                len(virtual_attrs) * len(label_names))

    # Phase 1: Pure group+label rules
    for attr_name, group_ids in virtual_attrs.items():
        n_valid = int((group_ids >= 0).sum())
        if n_valid < min_fires:
            continue

        for label_idx, label_name in enumerate(label_names):
            fire_mask = compute_group_fire_mask(group_ids, label_idx, label_state)
            metrics = evaluate_group_rule(
                fire_mask, label_idx, val_labels, existing_predictions,
                min_fires=min_fires, min_corr_prec=0.0,
            )
            if metrics is None:
                continue

            corr_prec = metrics["corr_prec"]
            f1_gain = metrics["f1_gain"]

            if corr_prec >= min_corr_prec and f1_gain >= min_f1_gain:
                n_groups = len(np.unique(group_ids[group_ids >= 0]))
                body = (
                    GroupPredicate(attr_name=attr_name, group_count=n_groups),
                    LabelPredicate(label=label_name, op="contains"),
                )
                rule = RDL(
                    body=body,
                    consequence=label_name,
                    consequence_op="add",
                    score=f1_gain,
                    coverage=metrics["n_fires"] / len(val_docs),
                    trial_number=trial_counter,
                    val_stats=metrics,
                )
                trial = MockTrial(
                    number=trial_counter,
                    values=[f1_gain],
                    params={"attr_name": attr_name, "label": label_name,
                            "corr_prec": corr_prec},
                )
                results.append((trial, rule))
                logger.info("  PASS: %s × %s → corr_prec=%.3f, fires=%d, f1_gain=%.4f",
                           attr_name, label_name, corr_prec, metrics["n_fires"], f1_gain)
                trial_counter += 1

            elif rescue_prec_range[0] <= corr_prec < rescue_prec_range[1]:
                borderline.append((attr_name, label_idx, fire_mask, metrics))

    logger.info("Phase 1: %d rules passed, %d borderline for rescue", len(results), len(borderline))

    # Phase 2: Rescue borderline rules with text predicates
    n_rescued = 0
    for attr_name, label_idx, fire_mask, metrics in borderline:
        label_name = label_names[label_idx]
        rescue_preds = _extract_rescue_predicates(
            fire_mask, label_idx, val_labels, existing_predictions,
            val_docs, top_k=max_rescue_preds // 2,
        )
        if not rescue_preds:
            continue

        best_rule = None
        best_prec = 0.0
        best_metrics = None
        best_pred_info = None

        for pred, is_negated in rescue_preds:
            # Compute text predicate mask
            text_mask = np.array(
                [bool(pred(Document(cnt=d.cnt))) for d in val_docs], dtype=bool
            )
            if is_negated:
                text_mask = ~text_mask

            # Combined fire mask
            combined = fire_mask & text_mask
            new_metrics = evaluate_group_rule(
                combined, label_idx, val_labels, existing_predictions,
                min_fires=min_fires, min_corr_prec=min_corr_prec,
            )
            if new_metrics is None:
                continue
            if new_metrics["corr_prec"] > best_prec:
                best_prec = new_metrics["corr_prec"]
                best_metrics = new_metrics
                best_pred_info = (pred, is_negated)

        if best_pred_info is not None and best_metrics is not None:
            pred, is_negated = best_pred_info
            n_groups = len(np.unique(virtual_attrs[attr_name][virtual_attrs[attr_name] >= 0]))
            body_preds: List[Predicate] = [
                GroupPredicate(attr_name=attr_name, group_count=n_groups),
                LabelPredicate(label=label_name, op="contains"),
            ]
            if is_negated:
                body_preds.append(MatchPredicate(
                    attr="cnt", r=pred.r, sim=False, threshold=0.85
                ))
                # Store negation info in val_stats
                neg_info = f"NOT {pred.r.raw}"
            else:
                body_preds.append(pred)
                neg_info = pred.r.raw

            rule = RDL(
                body=tuple(body_preds),
                consequence=label_name,
                consequence_op="add",
                score=best_metrics["f1_gain"],
                coverage=best_metrics["n_fires"] / len(val_docs),
                trial_number=trial_counter,
                val_stats={**best_metrics, "rescue_pred": neg_info,
                           "is_negated": is_negated},
            )
            trial = MockTrial(
                number=trial_counter,
                values=[best_metrics["f1_gain"]],
                params={"attr_name": attr_name, "label": label_name,
                        "corr_prec": best_prec, "rescue_pred": neg_info},
            )
            results.append((trial, rule))
            n_rescued += 1
            logger.info("  RESCUE: %s × %s + %s → corr_prec=%.3f, fires=%d",
                       attr_name, label_name, neg_info, best_prec, best_metrics["n_fires"])
            trial_counter += 1

    logger.info("Phase 2: %d rules rescued. Total: %d rules discovered.", n_rescued, len(results))
    return results


def simulate_cross_attr_cascade(
    rules: List[Tuple[str, int, np.ndarray]],
    virtual_attrs: Dict[str, np.ndarray],
    label_state: np.ndarray,
    existing_predictions: np.ndarray,
    val_labels: np.ndarray,
    label_names: List[str],
    max_rounds: int = 3,
) -> Tuple[np.ndarray, int]:
    """Simulate cross-attribute cascade propagation.

    Within a single attribute, propagation completes in one round (group is a clique).
    But across attributes, cascade is real:
      Round 1: cluster_200 rule gives doc_x label_A
      Round 2: emb_top1 rule sees doc_x has label_A, propagates to emb_top1 group

    Returns (final_predictions, total_new_labels_added).
    """
    preds = existing_predictions.copy()
    state = label_state.copy()
    total_added = 0

    for round_idx in range(max_rounds):
        round_added = 0
        for attr_name, label_idx, _ in rules:
            group_ids = virtual_attrs[attr_name]
            fire_mask = compute_group_fire_mask(group_ids, label_idx, state)
            actionable = fire_mask & (preds[:, label_idx] == 0)
            new_labels = actionable.sum()
            if new_labels > 0:
                preds[actionable, label_idx] = 1.0
                state[actionable, label_idx] = 1
                round_added += int(new_labels)

        total_added += round_added
        if round_added == 0:
            logger.info("  Cascade converged at round %d (total added: %d)",
                       round_idx + 1, total_added)
            break
        logger.info("  Cascade round %d: +%d labels", round_idx + 1, round_added)

    return preds, total_added
