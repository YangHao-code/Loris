"""Fast document review selection from inference state, never evaluation truth.

One positive label does not establish that a multi-label document is complete.
The review domain includes sparse output, unresolved conflicts, and disagreement
with the currently selected Aux. A small random audit also covers documents on
which both inference sources agree, without claiming that agreement is correct.
"""
import math

import numpy as np


def configuration(manifest):
    """Validate the optional query.review configuration and return defaults."""
    query = manifest.get("query", {})
    value = query.get("review", {})
    if not isinstance(value, dict):
        raise ValueError("query.review must be an object")
    unknown = set(value) - {"cardinality_ratio", "audit_fraction",
                            "aux_default_threshold", "aux_thresholds",
                            "eligibility_mode", "near_boundary_ratio"}
    if unknown:
        raise ValueError("unsupported query.review options: %s" % ", ".join(sorted(unknown)))
    result = dict(cardinality_ratio=value.get("cardinality_ratio", .7),
                  audit_fraction=value.get("audit_fraction", .125),
                  aux_default_threshold=value.get("aux_default_threshold", .5))
    for name in result:
        raw = result[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("query.review.%s must be a number" % name)
        result[name] = float(raw)
        if not math.isfinite(result[name]):
            raise ValueError("query.review.%s must be finite" % name)
    if result["cardinality_ratio"] <= 0:
        raise ValueError("query.review.cardinality_ratio must be positive")
    if not 0 <= result["audit_fraction"] <= 1:
        raise ValueError("query.review.audit_fraction must be in [0, 1]")
    if not 0 <= result["aux_default_threshold"] <= 1:
        raise ValueError("query.review.aux_default_threshold must be in [0, 1]")
    thresholds = value.get("aux_thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("query.review.aux_thresholds must be a label-to-threshold object")
    converted = {}
    for label, threshold in thresholds.items():
        if not isinstance(label, str) or not label:
            raise ValueError("query.review.aux_thresholds needs nonempty label names")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("query.review.aux_thresholds must be finite in [0, 1]")
        converted[label] = float(threshold)
    result["aux_thresholds"] = converted
    if "eligibility_mode" in value or "near_boundary_ratio" in value:
        mode = value.get("eligibility_mode", "legacy")
        if mode not in ("legacy", "evidence"):
            raise ValueError("query.review.eligibility_mode must be legacy or evidence")
        if "near_boundary_ratio" in value and mode != "evidence":
            raise ValueError("near_boundary_ratio requires evidence eligibility")
        boundary = value.get("near_boundary_ratio", .9)
        if (type(boundary) not in (int, float) or not math.isfinite(boundary)
                or not 0 < boundary <= 1):
            raise ValueError("near_boundary_ratio must be finite in (0,1]")
        result.update(eligibility_mode=mode, near_boundary_ratio=float(boundary))
    return result


def _boolean_rows(value, length, name):
    value = np.asarray(value)
    if value.dtype != np.bool_ or value.shape != (length,):
        raise ValueError("%s must be a boolean document mask" % name)
    return value


def train_label_cardinality(gamma_positive, gamma_negative, training_rows):
    """Estimate cardinality using only fully answered, designated training rows.

    Partial rows are excluded: treating their unobserved labels as negatives
    would underestimate the reference cardinality. Query/test roles must never
    be included in ``training_rows``, even after a complete answer is acquired.
    """
    pos, neg = np.asarray(gamma_positive, bool), np.asarray(gamma_negative, bool)
    if pos.ndim != 2 or pos.shape != neg.shape or np.any(pos & neg):
        raise ValueError("training Gamma must have compatible disjoint sign matrices")
    training_rows = _boolean_rows(training_rows, len(pos), "training_rows")
    complete = training_rows & (pos | neg).all(axis=1)
    if not complete.any():
        raise ValueError("review cardinality requires complete training Gamma rows")
    counts = pos[complete].sum(axis=1)
    return dict(mean=float(counts.mean()), rows=int(complete.sum()),
                designated_training_rows=int(training_rows.sum()),
                incomplete_training_rows=int((training_rows & ~complete).sum()),
                positive_labels=int(counts.sum()), minimum=int(counts.min()),
                maximum=int(counts.max()), source="complete_training_Gamma_only")


def review_candidates(*, positive, negative, unknown, gamma_positive,
                      gamma_negative, query_rows, aux_probabilities,
                      aux_thresholds, train_mean, ratio=0.7,
                      eligibility_mode="legacy", near_boundary_ratio=.9):
    """Build a review plan using model/rule disagreement and label count.

    ``positive`` is the available document output after conflict arbitration,
    rather than historical positive supports which may currently be unknown.
    ``unknown`` retains the conflict information for prioritizing a review.
    A cardinality threshold is strictly ``count < ratio * training mean``;
    therefore counts ``0 .. ceil(threshold)-1`` satisfy it, with no rounding
    of the threshold before comparison.
    """
    pos = np.asarray(positive, bool)
    if pos.ndim != 2 or pos.shape[1] == 0:
        raise ValueError("review selection needs a document-by-label matrix")
    n, k = pos.shape
    matrices = dict(negative=negative, unknown=unknown,
                    gamma_positive=gamma_positive, gamma_negative=gamma_negative)
    converted = {}
    for name, value in matrices.items():
        value = np.asarray(value, bool)
        if value.shape != pos.shape:
            raise ValueError("%s has incompatible review axes" % name)
        converted[name] = value
    gp, gn = converted["gamma_positive"], converted["gamma_negative"]
    if np.any(gp & gn):
        raise ValueError("trusted Gamma has contradictory signs")
    query_rows = _boolean_rows(query_rows, n, "query_rows")
    probability = np.asarray(aux_probabilities)
    thresholds = np.asarray(aux_thresholds)
    if probability.shape != pos.shape or thresholds.shape != (k,):
        raise ValueError("Aux probabilities/thresholds have incompatible review axes")
    if (not np.isfinite(probability).all() or not np.isfinite(thresholds).all()
            or np.any((probability < 0) | (probability > 1))
            or np.any((thresholds < 0) | (thresholds > 1))):
        raise ValueError("Aux probabilities and thresholds must be finite in [0, 1]")
    train_mean, ratio = float(train_mean), float(ratio)
    if not math.isfinite(train_mean) or train_mean < 0:
        raise ValueError("train_mean must be finite and nonnegative")
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("review cardinality ratio must be finite and positive")

    if eligibility_mode not in ("legacy", "evidence"):
        raise ValueError("unsupported review eligibility mode")
    if (type(near_boundary_ratio) not in (int, float) or not math.isfinite(near_boundary_ratio)
            or not 0 < near_boundary_ratio <= 1):
        raise ValueError("near_boundary_ratio must be finite in (0,1]")
    if eligibility_mode == "evidence":
        return _evidence_candidates(pos, converted["unknown"], gp, gn, query_rows,
                                    probability, thresholds, train_mean, near_boundary_ratio)

    unanswered = query_rows & ~(gp | gn).all(axis=1)
    cardinality = pos.sum(axis=1)
    target = ratio * train_mean
    sparse = cardinality < target
    uncovered = cardinality == 0
    conflicts = converted["unknown"].sum(axis=1)
    disagreements = np.count_nonzero(pos != (probability >= thresholds), axis=1)
    eligible = unanswered & (sparse | uncovered | (conflicts > 0) | (disagreements > 0))
    deficit = np.maximum(target - cardinality, 0) / max(target, 1.0)
    scale = max(train_mean, 1.0)
    # Use the most uncertain O(cardinality) labels rather than averaging all
    # labels, which would make many confident negative cells dominate the score.
    uncertain_labels = min(k, max(1, int(math.ceil(train_mean))))
    closeness = 1.0 - np.abs(probability - thresholds)
    top = np.partition(closeness, k - uncertain_labels, axis=1)[:, -uncertain_labels:]
    uncertainty = top.mean(axis=1)
    scores = (2.0 * (conflicts > 0) + conflicts / scale
              + disagreements / scale + deficit + 0.1 * uncertainty)
    scores = np.asarray(scores, np.float64)
    scores[~unanswered] = -np.inf
    return dict(eligible=eligible, unanswered=unanswered, scores=scores,
                reasons=dict(low_cardinality=sparse & unanswered,
                             no_positive_label=uncovered & unanswered,
                             conflict=(conflicts > 0) & unanswered,
                             disagreement=(disagreements > 0) & unanswered),
                components=dict(cardinality=cardinality, deficit=deficit,
                                disagreements=disagreements, conflicts=conflicts,
                                uncertainty=uncertainty),
                parameters=dict(cardinality_ratio=ratio, train_mean=train_mean,
                                strict_count_threshold=target,
                                maximum_sparse_count=int(math.ceil(target)) - 1,
                                uncertainty_labels=uncertain_labels,
                                truth_used=False))


def _evidence_candidates(pos, conflicts, gp, gn, query_rows, probability, thresholds,
                         train_mean, boundary_ratio):
    """Query-only opt-in. A label count never establishes a missing label."""
    n,k=pos.shape
    unanswered=query_rows & ~(gp|gn).all(1)
    untrusted=~(gp|gn)
    cardinality=pos.sum(1)
    uncovered=cardinality==0
    low=cardinality < train_mean
    proposed=probability >= thresholds
    missing=(~pos) & proposed & untrusted
    possible_fp=pos & ~proposed & untrusted
    # Relative proximity handles small ASL thresholds: .95*kappa is near,
    # while a probability far below a tiny kappa is not. No threshold changes.
    near=(~pos) & untrusted & (thresholds > 0) & (probability < thresholds)
    near &= probability >= boundary_ratio * thresholds
    near_missing=low & near.any(1)
    conflict_count=conflicts.sum(1)
    disagreement_count=(missing|possible_fp).sum(1)
    eligible=unanswered & (uncovered | (conflict_count>0) | (disagreement_count>0) | near_missing)
    deficit=np.maximum(train_mean-cardinality,0)/max(train_mean,1.)
    scores=(2.*conflict_count+disagreement_count+near_missing.astype(float)+uncovered.astype(float)+.05*deficit)
    scores=np.asarray(scores,np.float64);scores[~unanswered]=-np.inf
    return dict(eligible=eligible,unanswered=unanswered,scores=scores,
        reasons=dict(low_cardinality=low & unanswered,
                     no_positive_label=(cardinality==0) & unanswered,
                     conflict=(conflict_count>0) & unanswered,
                     disagreement=(disagreement_count>0) & unanswered,
                     proposed_missing_label=missing.any(1) & unanswered,
                     possible_false_positive=possible_fp.any(1) & unanswered,
                     low_cardinality_near_boundary=near_missing & unanswered),
        components=dict(cardinality=cardinality,deficit=deficit,
                        disagreements=disagreement_count,conflicts=conflict_count,
                        near_boundary_labels=near.sum(1)),
        parameters=dict(eligibility_mode="evidence",train_mean=train_mean,
                        near_boundary_ratio=float(boundary_ratio),
                        cardinality_ranking_weight=.05,cardinality_alone_is_eligibility=False,
                        deployed_predicate_thresholds_are_precision_certificates=False,
                        truth_used=False))


def rank_review_candidates(plan, *, limit, seed, audit_fraction=0.125):
    """Return a deterministic scored batch plus an unbiased random audit.

    Audit rows are sampled from all unanswered query documents, including
    documents outside ``needs_review``. They are not counted twice. A seed is
    passed by the caller so successive batches can use seed + query count.
    """
    if isinstance(limit, bool) or int(limit) != limit or limit < 0:
        raise ValueError("review limit must be a nonnegative integer")
    if not math.isfinite(audit_fraction) or not 0 <= audit_fraction <= 1:
        raise ValueError("review audit fraction must be finite in [0, 1]")
    eligible = np.asarray(plan["eligible"], bool)
    unanswered = np.asarray(plan["unanswered"], bool)
    scores = np.asarray(plan["scores"], float)
    if eligible.ndim != 1 or scores.shape != eligible.shape or unanswered.shape != eligible.shape:
        raise ValueError("review plan row axes differ")
    if np.any(eligible & ~unanswered) or not np.isfinite(scores[eligible]).all():
        raise ValueError("review plan has an invalid eligible row/score")
    limit = min(int(limit), int(unanswered.sum()))
    reserve = min(limit, int(math.ceil(limit * audit_fraction)))
    rng = np.random.RandomState(seed)
    audit_rows = rng.permutation(np.flatnonzero(unanswered))[:reserve]
    remaining = eligible.copy()
    remaining[audit_rows] = False
    candidates = np.flatnonzero(remaining)
    order = np.lexsort((candidates, -scores[candidates]))
    ranked = candidates[order][:limit - len(audit_rows)]
    # If the risk gates are exhausted, continue the allocated document budget
    # as random auditing. Agreement/sufficient cardinality must not establish
    # completeness, and the reserved audit alone would underfill a small pool.
    selected_mask = np.zeros(len(unanswered), bool)
    selected_mask[ranked] = True
    selected_mask[audit_rows] = True
    fill = limit - len(ranked) - len(audit_rows)
    if fill and audit_fraction > 0:
        extra_audit = rng.permutation(np.flatnonzero(unanswered & ~selected_mask))[:fill]
        audit_rows = np.r_[audit_rows, extra_audit].astype(np.int64)
    chosen = np.r_[ranked, audit_rows].astype(np.int64)
    return chosen.tolist(), dict(
        criterion="needs_review_matrix",
        parameters=dict(plan["parameters"], audit_fraction=float(audit_fraction)),
        eligible_documents=int(eligible.sum()),
        unanswered_query_documents=int(unanswered.sum()),
        selected_documents=int(len(chosen)), ranked_rows=ranked.tolist(),
        random_audit_rows=audit_rows.tolist(), seed=int(seed),
        reserved_audit_documents=reserve,
        reasons={name: int(mask.sum()) for name, mask in plan["reasons"].items()},
        selection_uses_target_truth=False,
        actual_queried_accuracy_is_not_assumed=True)
