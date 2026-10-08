"""Lowest-priority default rule for logically undecided cells (2026-09-30).

Final output for a cell (x, tau):
  1. a logical positive fact is kept;
  2. a logical negative fact (or validated Gamma negative) is never filled;
  3. a conflict_unknown cell keeps its existing model arbitration;
  4. otherwise the cell is undecided and becomes positive iff the dynamic Aux
     probability reaches the label threshold tau_l.

Step 4 is a stratified default rule applied after the round closure. Its
conclusions do not trigger other rules, are never weak supervision and never
enter Gamma or query selection. Thresholds are either declared in the manifest
(``calibration: fixed``) or recalibrated for every committed model on the
independent calibration documents (``calibration: independent_calibration``),
maximizing the per-label F1 of the complete final output on those documents.
"""
import numpy as np

SCHEMA = "loris_undecided_cell_model_fallback_v1"
CALIBRATIONS = ("fixed", "independent_calibration")
# 1.0 lets calibration choose "never fill" for a label (a probability of
# exactly 1 is the only value that can still reach it).
DEFAULT_GRID = [round(0.05 * i, 2) for i in range(1, 21)]


def configuration(manifest):
    value = manifest.get("fallback")
    if value is None:
        return None
    required = {"enabled", "alias", "calibration", "default_threshold", "thresholds"}
    optional = {"grid", "minimum_positive_support"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - optional:
        raise ValueError("fallback requires enabled/alias/calibration/default_threshold/thresholds")
    if type(value["enabled"]) is not bool:
        raise ValueError("fallback.enabled must be a bool")
    expected_alias = manifest.get("aux_extension", {}).get(
        "source_alias", manifest["dynamic"]["alias"])
    if value["alias"] != expected_alias:
        raise ValueError("fallback must use the declared dynamic Aux model")
    if value["calibration"] not in CALIBRATIONS:
        raise ValueError("fallback.calibration must be fixed or independent_calibration")
    thresholds = value["thresholds"]
    if not isinstance(thresholds, dict) or any(not isinstance(k, str) or not k for k in thresholds):
        raise ValueError("fallback thresholds must map label names to numbers")
    grid = value.get("grid", DEFAULT_GRID)
    for number in [value["default_threshold"], *thresholds.values(), *grid]:
        if type(number) not in (int, float) or not np.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("fallback thresholds and grid must be finite numbers in [0,1]")
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("fallback grid must be nonempty and unique")
    support = value.get("minimum_positive_support", 3)
    if type(support) is not int or support < 1:
        raise ValueError("fallback minimum_positive_support must be a positive integer")
    return dict(value, grid=sorted(float(x) for x in grid), minimum_positive_support=support)


RECORD_BINDING_SCHEMA = "loris_fallback_calibration_binding_v1"


def record_binding(manifest, labels, calibration_binding, source):
    """What a FALLBACK_CALIBRATION.json record depends on (2026-10-01).

    The declared fallback, arbitration and commit-gate configuration, the
    label axis, the independent calibration binding (calibration rows,
    answers, model tables and model binding) and the source binding of the
    reliability artifact (initial model, program, Gamma and rows). A record
    is reused only by a run whose binding is identical.
    """
    loop = manifest.get("loop", {})
    from loris._engine.typed_session import digest

    binding = dict(
        schema=RECORD_BINDING_SCHEMA,
        labels=list(labels),
        fallback=manifest.get("fallback"),
        arbitration=manifest.get("arbitration"),
        commit_gate=loop.get("commit_gate", "none"),
        dynamic_alias=manifest["dynamic"]["alias"],
        calibration_binding_id=digest(calibration_binding),
        source_binding_id=digest(source),
    )
    if loop.get("commit_gate_metric_policy", "mean_f1") == "aux_loris_joint":
        binding["joint_gate_configuration"] = dict(
            policy="aux_loris_joint",
            auxiliary_grid=list(loop.get("commit_gate_aux_grid", DEFAULT_GRID)),
            partition={key: list(values) for key, values in loop.get("joint_gate_partition", {}).items()},
            completeness_tolerance=float(loop.get("commit_gate_comp_tolerance", 0.0)),
            false_negative_tolerance=int(loop.get("commit_gate_fn_tolerance", 0)),
        )
    return binding


def check_record(record, expected_binding, path):
    if not isinstance(record, dict) or record.get("binding") != expected_binding:
        raise ValueError("FALLBACK_CALIBRATION.json does not match this run's fallback/arbitration/"
                         "commit-gate configuration, labels, model or calibration data; recalibrate: "
                         + str(path))


def declared_thresholds(config, labels):
    if set(config["thresholds"]) - set(labels):
        raise ValueError("fallback threshold label outside the complete label axis")
    return np.array([float(config["thresholds"].get(label, config["default_threshold"]))
                     for label in labels], dtype=np.float64)


def arbitration_thresholds(manifest, labels):
    arbitration = manifest.get("arbitration") or {}
    default = float(arbitration.get("default_threshold", 0.5))
    overrides = arbitration.get("thresholds", {})
    return np.array([float(overrides.get(label, default)) for label in labels], dtype=np.float64)


def undecided(positive, negative, unknown, gamma_negative, arbitration=None):
    decided = positive | negative | unknown | gamma_negative
    if arbitration is not None:
        decided = decided | arbitration[0] | arbitration[1]
    return ~decided


def fill(positive, negative, unknown, gamma_negative, probabilities, thresholds, arbitration=None):
    p = np.asarray(probabilities, np.float64)
    if p.shape != positive.shape or thresholds.shape != (p.shape[1],):
        raise ValueError("fallback probability/threshold axes differ")
    return undecided(positive, negative, unknown, gamma_negative, arbitration) & (p >= thresholds[None, :])


def _f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return np.divide(2.0 * tp, denominator, out=np.zeros(np.shape(tp)), where=denominator != 0)


def output_quality(truth_positive, observed, output, rows):
    """Score explicit observed cells; completeness uses fully checked multilabel rows."""
    truth = np.asarray(truth_positive, bool)
    observed = np.asarray(observed, bool)
    output = np.asarray(output, bool)
    rows = np.asarray(rows, bool)
    if truth.ndim != 2 or observed.shape != truth.shape or output.shape != truth.shape or rows.shape != (len(truth),):
        raise ValueError("output quality requires matching row and label axes")
    # Gate denominators are usually a small held-out slice of a large corpus.
    # Select rows before dense cell operations, rather than repeatedly copying
    # all full-train anchor cells that are never scored.
    observed = observed[rows]
    truth = truth[rows] & observed
    output = output[rows] & observed
    tp = (output & truth).sum(0)
    fp = (output & ~truth).sum(0)
    fn = truth.sum(0) - tp
    micro = float(_f1(tp.sum(), fp.sum(), fn.sum()))
    macro = float(_f1(tp, fp, fn).mean())
    multi = observed.all(1) & (truth.sum(1) > 1)
    comp = float(((output & truth)[multi].sum(1) / truth[multi].sum(1)).mean()) if multi.any() else None
    return dict(macro_f1=macro, micro_f1=micro, mean_f1=(macro + micro) / 2,
                TP=int(tp.sum()), FP=int(fp.sum()), FN=int(fn.sum()),
                multi_label_completeness=comp, completeness_documents=int(multi.sum()),
                rows=int(rows.sum()), positive_cells=int(truth.sum()))


def auxiliary_quality(probabilities, positive, negative, *, fit_probabilities,
                      fit_positive, fit_negative, grid=DEFAULT_GRID):
    """Fit cutoffs on A, then evaluate F1 and ranking on distinct B rows."""
    p, y, n = np.asarray(probabilities, np.float64), np.asarray(positive, bool), np.asarray(negative, bool)
    fit_p, fit_y, fit_n = (np.asarray(fit_probabilities, np.float64),
                           np.asarray(fit_positive, bool), np.asarray(fit_negative, bool))
    grid = np.asarray(grid, np.float64)
    if (p.ndim != 2 or y.shape != p.shape or n.shape != p.shape or np.any(y & n)
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or grid.ndim != 1 or not len(grid) or len(np.unique(grid)) != len(grid)
            or not np.isfinite(grid).all() or np.any((grid < 0) | (grid > 1))):
        raise ValueError("Aux quality requires explicit signed answers and valid probabilities/grid")
    if (fit_p.ndim != 2 or fit_p.shape[1] != p.shape[1] or not len(fit_p)
            or fit_y.shape != fit_p.shape or fit_n.shape != fit_p.shape or np.any(fit_y & fit_n)
            or not np.isfinite(fit_p).all() or np.any((fit_p < 0) | (fit_p > 1))):
        raise ValueError("Aux threshold fitting requires valid explicit A arrays")
    grid = np.sort(grid)
    observed = y | n
    thresholds, ap = [], []
    for col in range(p.shape[1]):
        valid = observed[:, col]
        scores, labels = p[valid, col], y[valid, col]
        fit_valid = (fit_y | fit_n)[:, col]
        fit_scores, fit_labels = fit_p[fit_valid, col], fit_y[fit_valid, col]
        candidate = fit_scores[:, None] >= grid[None, :]
        tp = (candidate & fit_labels[:, None]).sum(0)
        fp = (candidate & ~fit_labels[:, None]).sum(0)
        fn = int(fit_labels.sum()) - tp
        f1 = _f1(tp, fp, fn)
        thresholds.append(float(grid[np.flatnonzero(f1 >= f1.max())[-1]]))
        if not labels.any():
            ap.append(0.0)
            continue
        order = np.argsort(-scores, kind="stable")
        score_sorted, label_sorted = scores[order], labels[order]
        ends = np.r_[np.flatnonzero(np.diff(score_sorted)), len(scores) - 1]
        cumulative = np.cumsum(label_sorted)[ends]
        precision = cumulative / (ends + 1)
        recall_increments = np.diff(np.r_[0, cumulative]) / int(labels.sum())
        ap.append(float(np.sum(precision * recall_increments)))
    thresholds = np.asarray(thresholds, np.float64)
    result = output_quality(y, observed, p >= thresholds[None, :], np.ones(len(p), bool))
    result.update(macro_average_precision=float(np.mean(ap)), thresholds=thresholds.tolist(),
                  threshold_policy="fit_A_score_B_per_label_F1_grid_ties_larger",
                  fit_rows=len(fit_p), calibration_answers_used_for_threshold_fitting=True,
                  labels_entered_model_fitting=False)
    return result


def joint_gate_decision(parent, candidate, *, completeness_tolerance=0.0, false_negative_tolerance=0):
    """No mean-score compensation: both Aux and deployed output must not regress."""
    eps = 1e-12
    failures, gains = [], []
    if (type(false_negative_tolerance) is not int or false_negative_tolerance < 0
            or type(completeness_tolerance) not in (int, float)
            or not np.isfinite(completeness_tolerance) or not 0 <= completeness_tolerance <= 1):
        raise ValueError("joint gate tolerances must be explicit nonnegative values")
    for family in ("auxiliary", "loris"):
        for field in ("rows", "positive_cells"):
            if parent[family][field] != candidate[family][field]:
                raise ValueError("joint gate scored answer domain changed")
    for family in ("auxiliary", "loris"):
        for metric in ("macro_f1", "micro_f1"):
            before, after = float(parent[family][metric]), float(candidate[family][metric])
            if not np.isfinite([before, after]).all():
                raise ValueError("joint gate scores must be finite")
            if after < before - eps:
                failures.append(f"{family}.{metric}_decreased")
            gains.append(after - before)
    before_ap, after_ap = parent["auxiliary"]["macro_average_precision"], candidate["auxiliary"]["macro_average_precision"]
    if not np.isfinite([before_ap, after_ap]).all():
        raise ValueError("joint gate average precision must be finite")
    if after_ap < before_ap - eps:
        failures.append("auxiliary.macro_average_precision_decreased")
    for family in ("loris",):
        before, after = parent[family], candidate[family]
        if int(after["FN"]) > int(before["FN"]) + false_negative_tolerance:
            failures.append(f"{family}.false_negatives_increased")
        if before["completeness_documents"] != after["completeness_documents"]:
            raise ValueError("joint gate completeness scoring domain changed")
        a, b = before["multi_label_completeness"], after["multi_label_completeness"]
        if (a is None) != (b is None):
            raise ValueError("joint gate completeness support changed")
        if a is not None and (not np.isfinite([a, b]).all() or b < a - completeness_tolerance - eps):
            failures.append(f"{family}.completeness_decreased")
    if max(gains) <= eps:
        failures.append("no_strict_F1_gain_keep_parent")
    return dict(accepted=not failures, failures=failures, numeric_tolerance=eps,
                F1_deltas=gains, theta_and_kappa_atomic=True,
                completeness_tolerance=float(completeness_tolerance),
                false_negative_tolerance=int(false_negative_tolerance))


def calibrate(*, positive, negative, unknown, probabilities, arbitration_kappa, truth_positive,
              truth_negative, rows, config, labels):
    """Per-label F1-optimal fallback thresholds on independent calibration rows.

    The complete final output is evaluated: closure positives, arbitration of
    conflict_unknown cells with the declared arbitration thresholds, and the
    fallback on undecided cells. Ties keep the larger (more conservative)
    threshold. Labels with fewer than minimum_positive_support calibration
    positives share one pooled threshold chosen by their summed F1 counts.
    Returns thresholds, gate scores (macro/micro/mean F1 of the final output)
    and an audit record.
    """
    p = np.asarray(probabilities, np.float64)
    rows = np.asarray(rows, bool)
    truth_positive = np.asarray(truth_positive, bool) & rows[:, None]
    observed = (np.asarray(truth_positive, bool) | np.asarray(truth_negative, bool)) & rows[:, None]
    arbitrated = unknown & (p >= arbitration_kappa[None, :])
    base = (positive & ~unknown) | arbitrated
    candidates = undecided(positive, negative, unknown, np.zeros_like(positive)) & observed
    grid = np.asarray(config["grid"], np.float64)
    k = p.shape[1]
    base_tp = (base & truth_positive & observed).sum(0)
    base_fp = (base & ~truth_positive & observed).sum(0)
    positives = truth_positive.sum(0)
    tp = np.empty((len(grid), k))
    fp = np.empty((len(grid), k))
    for index, threshold in enumerate(grid):
        filled = candidates & (p >= threshold)
        tp[index] = base_tp + (filled & truth_positive).sum(0)
        fp[index] = base_fp + (filled & ~truth_positive).sum(0)
    fn = positives[None, :] - tp
    scores = _f1(tp, fp, fn)
    thresholds = np.empty(k)
    supported = positives >= config["minimum_positive_support"]
    for label in np.flatnonzero(supported):
        best = scores[:, label].max()
        thresholds[label] = grid[np.flatnonzero(scores[:, label] >= best)[-1]]
    pooled = None
    if (~supported).any():
        low = ~supported
        pooled_scores = _f1(tp[:, low].sum(1), fp[:, low].sum(1), fn[:, low].sum(1))
        pooled = float(grid[np.flatnonzero(pooled_scores >= pooled_scores.max())[-1]])
        thresholds[low] = pooled
    final = base | (candidates & (p >= thresholds[None, :]))
    f_tp = (final & truth_positive).sum(0)
    f_fp = (final & ~truth_positive & observed).sum(0)
    f_fn = positives - f_tp
    per_label = _f1(f_tp, f_fp, f_fn)
    total = 2 * f_tp.sum() + f_fp.sum() + f_fn.sum()
    micro = float(2 * f_tp.sum() / total) if total else 0.0
    macro = float(per_label.mean())
    gate = output_quality(truth_positive, observed, final, rows)
    audit = dict(schema=SCHEMA, grid=[float(x) for x in grid],
                 minimum_positive_support=config["minimum_positive_support"],
                 supported_labels=int(supported.sum()), pooled_threshold=pooled,
                 tie_break="larger_threshold", arbitration_kappa_used_for_unknown=True,
                 filled_cells=int((candidates & (p >= thresholds[None, :])).sum()))
    return {label: float(t) for label, t in zip(labels, thresholds)}, gate, audit


def score_output(*, positive, negative, unknown, probabilities, arbitration_kappa, truth_positive,
                 truth_negative, rows, thresholds=None):
    """Macro/micro/mean F1 of the final output on rows (fixed thresholds or no fallback)."""
    p = np.asarray(probabilities, np.float64)
    rows = np.asarray(rows, bool)
    observed = (np.asarray(truth_positive, bool) | np.asarray(truth_negative, bool)) & rows[:, None]
    truth = np.asarray(truth_positive, bool) & observed
    output = (positive & ~unknown) | (unknown & (p >= arbitration_kappa[None, :]))
    if thresholds is not None:
        output = output | (undecided(positive, negative, unknown, np.zeros_like(positive))
                           & (p >= thresholds[None, :]))
    output &= observed
    tp = (output & truth).sum(0)
    fp = (output & ~truth).sum(0)
    fn = truth.sum(0) - tp
    per_label = _f1(tp, fp, fn)
    total = 2 * tp.sum() + fp.sum() + fn.sum()
    micro = float(2 * tp.sum() / total) if total else 0.0
    macro = float(per_label.mean())
    return output_quality(truth, observed, output, rows)
