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
import scipy.sparse as sp

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


def _per_label_f1(y_true: np.ndarray, y_pred: np.ndarray, j: int) -> float:
    """F1 of a single target label column ``j``.

    B12 fix: comparison/group rules touch ONE label, but the admission gate used
    the global macro mean over all labels — a single-label gain of e.g. +0.3 on a
    tail label becomes +0.3/30 ≈ +0.01 in the mean and is rejected by min_f1_gain.
    Scoring the rule on its own label's F1 is what the paper's per-label,
    accuracy-driven selection (§5.2) intends.
    """
    yt = y_true[:, j]
    yp = y_pred[:, j]
    tp = float(((yt == 1) & (yp == 1)).sum())
    fp = float(((yt == 0) & (yp == 1)).sum())
    fn = float(((yt == 1) & (yp == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def compute_group_fire_mask(
    membership: sp.csr_matrix,
    label_idx: int,
    label_state: np.ndarray,
) -> np.ndarray:
    """Compute the comparison-predicate fire mask for an (attr, label) pair.

    mask[i] = True iff doc i shares >=1 value of this attribute type with some
    doc j that has the label (`x.A=y.A ∧ label∈y.lbl`). Computed via sparse
    matrix-vector products, never materializing the n_docs×n_docs co-membership
    matrix::

        col  = membershipᵀ @ y           # (n_values,) qualifying docs per value
        fire = (membership @ col) > 0     # (n_docs,) shares a value with a qualifier

    SELF-INCLUSION: this is intentionally self-inclusive — a doc that itself has
    the label contributes to its own value column, so a lone qualifier still
    fires. This matches the legacy single-value behaviour (one-hot membership
    reproduces the old per-group loop bit-for-bit). The "∃j≠i" intent is
    enforced downstream by every consumer AND-ing with ~already-has-consequence
    (multi_chase, discovery.batch_select, orchestrator, evaluate_group_rule), so
    no explicit self-subtraction is applied here. Empty rows (a doc with no
    surviving value for this attr) never fire.
    """
    y = np.asarray(label_state[:, label_idx], dtype=np.float32).ravel()
    col = membership.T.dot(y)
    fire = np.asarray(membership.dot(col)).ravel() > 0
    return fire


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

    # F1 gain — B12 fix: per-TARGET-label F1 delta, not the global macro mean.
    # test_preds differs from existing_predictions only in column label_idx, so the
    # per-label F1 of every other label is unchanged; the global-macro delta just
    # divided this rule's real gain by n_labels and the gate rejected it.
    test_preds = existing_predictions.copy()
    test_preds[actionable, label_idx] = 1.0
    old_f1 = _per_label_f1(val_labels, existing_predictions, label_idx)
    new_f1 = _per_label_f1(val_labels, test_preds, label_idx)
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
    base_label_prec: Optional[np.ndarray] = None,
    asym_margin: float = 0.05,
    narrow_prec_floor: Optional[float] = None,
) -> List[Tuple[MockTrial, RDL]]:
    """Exhaustively enumerate (attr, label) group propagation rules.

    Phase 1: Pure group+label rules with corr_prec >= eff_gate pass directly.
    Phase 2: Rules with corr_prec in [narrow_floor, eff_gate) get text-predicate
             narrowing (comparison ∧ text → add τ) — strengthened to fire on ALL
             below-gate-but-has-signal candidates, not only a thin band.

    Asymmetric per-label gate (strategy 3): when ``base_label_prec`` is given,
    a rule for label L is admitted at ``max(min_corr_prec, base_prec_L+asym_margin)``
    — tighter for already-precise head labels (protect them), at the floor for weak
    tail labels (let recall rules through). ``base_label_prec=None`` ⇒ flat
    ``min_corr_prec`` (bit-for-bit the old behaviour; golden-neutral).

    Returns list compatible with batch_select: [(MockTrial, RDL), ...]
    """
    val_labels = np.asarray(val_labels, dtype=np.float32)
    existing_predictions = np.asarray(existing_predictions, dtype=np.float32)

    def _eff_gate(label_idx: int) -> float:
        if base_label_prec is None:
            return min_corr_prec
        return max(min_corr_prec, float(base_label_prec[label_idx]) + asym_margin)

    # lower bound for attempting text-narrowing of a below-gate candidate
    _narrow_floor = rescue_prec_range[0] if narrow_prec_floor is None else narrow_prec_floor

    results: List[Tuple[MockTrial, RDL]] = []
    borderline: List[Tuple[str, int, np.ndarray, Dict]] = []
    trial_counter = 0

    logger.info("=== Group Rule Discovery: %d attrs × %d labels = %d candidates ===",
                len(virtual_attrs), len(label_names),
                len(virtual_attrs) * len(label_names))

    # Phase 1: Pure group+label rules
    for attr_name, membership in virtual_attrs.items():
        # number of docs that still have >=1 surviving value for this attribute
        n_valid = int((membership.getnnz(axis=1) > 0).sum())
        if n_valid < min_fires:
            continue

        for label_idx, label_name in enumerate(label_names):
            fire_mask = compute_group_fire_mask(membership, label_idx, label_state)
            metrics = evaluate_group_rule(
                fire_mask, label_idx, val_labels, existing_predictions,
                min_fires=min_fires, min_corr_prec=0.0,
            )
            if metrics is None:
                continue

            corr_prec = metrics["corr_prec"]
            f1_gain = metrics["f1_gain"]

            _gate = _eff_gate(label_idx)
            if corr_prec >= _gate and f1_gain >= min_f1_gain:
                n_values = membership.shape[1]
                body = (
                    GroupPredicate(attr_name=attr_name, group_count=n_values),
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

            else:
                # Upper bound of the text-narrowing band. When base_label_prec is
                # None (default) this is the ORIGINAL rescue_prec_range[1] so the
                # band is bit-for-bit the old [rescue_lo, rescue_hi); only the
                # asymmetric mode widens it up to the (possibly higher) gate.
                _narrow_upper = rescue_prec_range[1] if base_label_prec is None else _gate
                if _narrow_floor <= corr_prec < _narrow_upper:
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
            # Compute text predicate mask; for a negated candidate the rule fires
            # where the phrase is ABSENT (B5: now emitted as a real negated
            # MatchPredicate below, so validation and application agree).
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

        _accept = best_pred_info is not None and best_metrics is not None
        # New strengthened mode (base_label_prec given): the wider narrowing band
        # admits more candidates, so gate the narrowed rule on the asymmetric
        # precision. When base_label_prec is None we keep the exact old acceptance
        # (no extra gate) → bit-for-bit golden-neutral.
        if _accept and base_label_prec is not None:
            _accept = best_prec >= _eff_gate(label_idx)
        if _accept:
            pred, is_negated = best_pred_info
            n_values = virtual_attrs[attr_name].shape[1]
            body_preds: List[Predicate] = [
                GroupPredicate(attr_name=attr_name, group_count=n_values),
                LabelPredicate(label=label_name, op="contains"),
            ]
            if is_negated:
                # B5 complete: a real negated MatchPredicate (fires when the phrase
                # is ABSENT) — matches the ~text_mask used to validate it.
                body_preds.append(MatchPredicate(
                    attr="cnt", r=pred.r, sim=False, threshold=0.85, negate=True
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


def discover_equal_rules(
    virtual_attrs: Dict[str, sp.csr_matrix],
    val_labels: np.ndarray,
    existing_predictions: np.ndarray,
    min_fires: int = 3,
    min_corr_prec: float = 0.60,
    min_f1_gain: float = 0.0005,
    max_rounds: int = 5,
) -> List[Tuple[MockTrial, RDL]]:
    """Discover comparison-consequence rules ``x.lbl = y.lbl``, one per attribute.

    An equal-rule on attribute A copies the whole positive-label set among docs
    sharing >=1 value of A (paper §6.1 chase step, ⊗="="), realized via the
    SpMV label-set closure ``pos |= (M @ (Mᵀ @ pos)) > 0`` iterated to fixpoint.

    Admission is **accuracy-guided** (paper §5.2: statistical support/confidence
    are inadequate; optimize labeling accuracy directly): simulate the closure on
    the validation set and keep attributes whose closure improves macro-F1 with
    correction precision >= min_corr_prec. The candidate space is one rule per
    attribute (~tens), so no statistical anti-explosion pre-filter is needed.

    Returns [(MockTrial, RDL), ...] with body=(GroupPredicate(attr),),
    consequence="" and consequence_op="equal". These bypass ``batch_select``
    (built for single-label add/remove rules) and are appended to the rule set
    directly by the caller; the chase / orchestrator fast-path apply them via the
    same SpMV closure (`MultiChase._eval_equal_rules`).
    """
    val_labels = np.asarray(val_labels, dtype=np.float32)
    existing = np.asarray(existing_predictions, dtype=np.float32)
    base_pos = existing > 0
    old_f1 = _fast_macro_f1(val_labels, existing)

    results: List[Tuple[MockTrial, RDL]] = []
    trial_counter = 0
    logger.info("=== Equal Rule Discovery: %d candidate attributes ===",
                len(virtual_attrs))

    for attr_name, membership in virtual_attrs.items():
        n_valid = int((membership.getnnz(axis=1) > 0).sum())
        if n_valid < min_fires:
            continue

        # Simulate the equal-rule's SpMV label-set closure to fixpoint (monotone).
        pos = base_pos.copy()
        for _ in range(max_rounds):
            grown = (np.asarray(membership.dot(
                membership.T.dot(pos.astype(np.float32)))) > 0) | pos
            if np.array_equal(grown, pos):
                break
            pos = grown

        newly = pos & ~base_pos  # (n_docs, n_labels) labels this rule would add
        n_fires = int(newly.sum())
        if n_fires < min_fires:
            continue

        # Correction precision over the newly-added (doc, label) positions.
        added_truth = val_labels[newly]
        n_improved = int((added_truth == 1).sum())
        n_worsened = int((added_truth == 0).sum())
        if n_improved == 0:
            continue
        corr_prec = n_improved / (n_improved + n_worsened)
        if corr_prec < min_corr_prec:
            continue

        f1_gain = _fast_macro_f1(val_labels, pos.astype(np.float32)) - old_f1
        if f1_gain < min_f1_gain:
            continue

        n_values = membership.shape[1]
        metrics = {
            "n_fires": n_fires, "n_improved": n_improved,
            "n_worsened": n_worsened, "corr_prec": round(corr_prec, 4),
            "f1_gain": round(float(f1_gain), 4),
        }
        rule = RDL(
            body=(GroupPredicate(attr_name=attr_name, group_count=n_values),),
            consequence="",
            consequence_op="equal",
            score=float(f1_gain),
            coverage=n_fires / max(val_labels.shape[0], 1),
            trial_number=trial_counter,
            val_stats=metrics,
        )
        results.append((
            MockTrial(number=trial_counter, values=[float(f1_gain)],
                      params={"attr": attr_name, "op": "equal"}),
            rule,
        ))
        trial_counter += 1
        logger.info("  equal-rule attr=%s: n_fires=%d corr_prec=%.3f f1_gain=%.4f",
                    attr_name, n_fires, corr_prec, f1_gain)

    logger.info("=== Equal Rule Discovery: %d equal-rules admitted ===", len(results))
    return results

# NOTE (paper-vs-code audit, 2026-06-04): simulate_cross_attr_cascade was REMOVED
# — it was dead in the active pipeline (only re-exported by the legacy shim) and
# its docstring example joins on fabricated attrs (cluster_200 / emb_top1). The
# real cross-attribute cascade-to-fixpoint is the chase (multi_chase), not this
# standalone simulator.
