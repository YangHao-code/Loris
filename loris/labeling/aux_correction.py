"""Fit a sparse, label-independent corrective suffix using explicit dev rows.

The fitting arrays never become Chase facts. Fitting is a conservative screening
step; acceptance of a complete program still requires its actual calibrated
Chase/output score. These diagnostics are not an independent test guarantee.
"""

import math
import numpy as np

from .aux_extension import static_correction_body


def _threshold(value, name):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= 1):
        raise ValueError(name + " must be a finite number in [0,1]")


def validate_specification(value, extension, labels=None):
    required = {"complement_alias", "positive_active", "negative_rule_indices",
                "negative_default_threshold", "negative_thresholds", "min_gap"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Aux correction requires its complete declared specification")
    alias = value["complement_alias"]
    if (not isinstance(alias, str) or not alias or alias in
            {extension["alias"], extension["source_alias"],
             extension.get("initial_alias", extension["source_alias"])}):
        raise ValueError("Aux complement must have an independent alias")
    active = value["positive_active"]
    if (not isinstance(active, list) or any(not isinstance(x, str) or not x for x in active)
            or len(active) != len(set(active))):
        raise ValueError("positive_active must contain unique label names")
    indices = value["negative_rule_indices"]
    if (not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices)
            or len(indices) != len(set(indices)) or indices != sorted(indices)):
        raise ValueError("negative_rule_indices must be sorted unique nonnegative integers")
    low = value["negative_thresholds"]
    if not isinstance(low, dict) or any(not isinstance(k, str) or not k for k in low):
        raise ValueError("negative_thresholds must be keyed by label name")
    _threshold(value["negative_default_threshold"], "negative default threshold")
    _threshold(value["min_gap"], "minimum positive/negative gap")
    if value["min_gap"] <= 0:
        raise ValueError("minimum positive/negative gap must be positive")
    for p in low.values():
        _threshold(p, "negative threshold")
    if labels is not None and ((set(active) | set(low)) - set(labels)):
        raise ValueError("Aux correction label is outside the complete label axis")
    # Check EVERY overridden channel, not only currently active positive rules:
    # later reactivation/recalibration must not introduce an overlapping region.
    check_labels = set(low) | set(extension["thresholds"]) | set(active)
    pairs = [(extension["default_threshold"], value["negative_default_threshold"])]
    pairs += [(extension["thresholds"].get(label, extension["default_threshold"]),
               low.get(label, value["negative_default_threshold"])) for label in check_labels]
    if any(float(hi) - float(lo) + 1e-12 < value["min_gap"] for hi, lo in pairs):
        raise ValueError("Aux positive/negative thresholds violate the minimum gap")
    return value


def _f1(truth, prediction):
    tp = int(np.count_nonzero(truth & prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    fn = int(np.count_nonzero(truth & ~prediction))
    return 2. * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.


def fit_correction(*, labels, rules, truth, probabilities, rule_fires, fit_mask,
                   complement_alias, baseline_positive=None, positive_grid=(.5, .6, .7, .8, .9, .95),
                   negative_grid=(.02, .05, .1, .15), positive_precision=.95,
                   negative_precision=.9, rule_precision_ceiling=.8, min_support=10,
                   min_f1_gain=.001, default_threshold=.95, negative_default_threshold=.05,
                   min_gap=.05):
    """Return extension updates and a row-bound diagnostic receipt.

    Positive gates are evaluated only on NEW activations. Negative gates are
    evaluated within static contexts of low-precision old rules. Gold is used
    solely on explicitly selected fit rows; callers evaluate the resulting
    fixed suffix on disjoint rows and, separately, on a frozen test.

    ``rule_fires`` is [old rule, row]. It must come from the original frozen
    closure, not from the adaptive model's own new-rule predictions.
    """
    labels, rules = tuple(labels), tuple(rules)
    y, p, fires = np.asarray(truth), np.asarray(probabilities), np.asarray(rule_fires)
    mask = np.asarray(fit_mask)
    if (y.ndim != 2 or p.shape != y.shape or y.shape[1] != len(labels)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
            or ((p < 0) | (p > 1)).any()):
        raise ValueError("fit truth/probabilities must share a finite row/label axis")
    if (fires.shape != (len(rules), len(y)) or not np.isin(fires, [0, 1]).all()
            or mask.dtype != bool or mask.shape != (len(y),) or not mask.any()):
        raise ValueError("fit rule_fires and explicit boolean fit_mask must match row axes")
    if len(set(labels)) != len(labels):
        raise ValueError("fit label axis must be unique")
    if type(min_support) is not int or min_support < 1:
        raise ValueError("minimum support must be a positive integer")
    for value, name in ((positive_precision, "positive precision"),
                        (negative_precision, "negative precision"),
                        (rule_precision_ceiling, "rule precision ceiling"),
                        (default_threshold, "positive default"),
                        (negative_default_threshold, "negative default"),
                        (min_gap, "minimum gap")):
        _threshold(value, name)
    if not np.isfinite(min_f1_gain) or min_f1_gain < 0:
        raise ValueError("minimum F1 gain must be nonnegative")
    for grid in (positive_grid, negative_grid):
        if not grid:
            raise ValueError("threshold grids cannot be empty")
        for threshold in grid:
            _threshold(threshold, "grid threshold")
    y, p, fires = y.astype(bool, copy=False)[mask], p[mask], fires.astype(bool, copy=False)[:, mask]
    index = {label: i for i, label in enumerate(labels)}
    if baseline_positive is None:
        base = np.zeros_like(y)
        for i, rule in enumerate(rules):
            if getattr(rule.head, "polarity", None) == "pos":
                if rule.head.label not in index:
                    raise ValueError("old rule head outside fit label axis")
                base[:, index[rule.head.label]] |= fires[i]
    else:
        supplied = np.asarray(baseline_positive)
        if supplied.shape != np.asarray(truth).shape or not np.isin(supplied, [0, 1]).all():
            raise ValueError("baseline positive matrix must match full fit input axis")
        base = supplied.astype(bool, copy=False)[mask]
    eligible = {}
    rule_receipts = []
    for i, rule in enumerate(rules):
        if not static_correction_body(rule):
            continue
        j = index[rule.head.label]
        support = int(fires[i].sum())
        precision = float(y[fires[i], j].mean()) if support else None
        accepted = support >= min_support and precision < rule_precision_ceiling
        rule_receipts.append(dict(rule_index=i, fit_support=support, fit_precision=precision,
                                  low_quality_static_context=bool(accepted)))
        if accepted:
            eligible.setdefault(j, []).append(i)
    positive_thresholds, negative_thresholds, active, selected_indices, diagnostics = {}, {}, [], [], []
    for j, label in enumerate(labels):
        baseline_score = _f1(y[:, j], base[:, j])
        pos_best = None
        for hi in sorted(set(map(float, positive_grid))):
            extra = (p[:, j] >= hi) & ~base[:, j]
            support = int(extra.sum())
            precision = float(y[extra, j].mean()) if support else None
            gain = _f1(y[:, j], base[:, j] | extra) - baseline_score
            if support >= min_support and precision >= positive_precision and gain >= min_f1_gain:
                # Higher gain, then higher precision, then conservative threshold.
                key = (gain, precision, hi)
                if pos_best is None or key > pos_best[0]:
                    pos_best = (key, hi, support, precision, gain)
        hi = pos_best[1] if pos_best else float(default_threshold)
        low_best = None
        ids = eligible.get(j, [])
        if ids:
            context = np.logical_or.reduce(fires[ids], axis=0) & base[:, j]
            for lo in sorted(set(map(float, negative_grid))):
                if hi - lo + 1e-12 < min_gap:
                    continue
                removed = context & (p[:, j] <= lo)
                support = int(removed.sum())
                precision = float((~y[removed, j]).mean()) if support else None
                gain = _f1(y[:, j], base[:, j] & ~removed) - baseline_score
                if support >= min_support and precision >= negative_precision and gain >= min_f1_gain:
                    key = (gain, precision, -lo)
                    if low_best is None or key > low_best[0]:
                        low_best = (key, lo, support, precision, gain)
        if pos_best:
            active.append(label)
            positive_thresholds[label] = hi
        if low_best:
            negative_thresholds[label] = low_best[1]
            selected_indices.extend(ids)
        diagnostics.append(dict(label=label, baseline_f1=baseline_score,
            positive=(dict(threshold=hi, new_support=pos_best[2], precision=pos_best[3],
                           screening_f1_gain=pos_best[4]) if pos_best else None),
            negative=(dict(threshold=low_best[1], context_support=low_best[2],
                           precision=low_best[3], screening_f1_gain=low_best[4],
                           rule_indices=ids) if low_best else None)))
    correction = dict(complement_alias=complement_alias, positive_active=active,
        negative_rule_indices=sorted(selected_indices),
        negative_default_threshold=float(negative_default_threshold),
        negative_thresholds=negative_thresholds, min_gap=float(min_gap))
    update = dict(default_threshold=float(default_threshold), thresholds=positive_thresholds,
                  correction=correction)
    # Catch invalid default/fitted gaps before anything can execute the program.
    validate_specification(correction, dict(alias="__fit_positive__", source_alias="__fit_source__",
                                            **update), labels)
    receipt = dict(fit_rows=int(mask.sum()), confirmation_rows=int((~mask).sum()),
        fit_row_indices=np.flatnonzero(mask).tolist(),
        truth_entered_chase=False, threshold_selection_scope="explicit_fit_rows_only",
        screening_is_actual_chase=False, actual_chase_confirmation_required=True,
        positive_precision_floor=float(positive_precision),
        negative_precision_floor=float(negative_precision), minimum_support=min_support,
        rule_precision_ceiling=float(rule_precision_ceiling), minimum_f1_gain=float(min_f1_gain),
        rules=rule_receipts, labels=diagnostics)
    return update, receipt
