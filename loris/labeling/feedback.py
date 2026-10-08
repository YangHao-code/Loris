"""Signed logical-A supervision and checked-assignment threshold calibration.

The weight equations are the author's explicit supplement to the paper.
Rule calibration labels never enter this module. The main weighting combines
the maximum q of immediate label/propagation sources with the existing model
disagreement discount. Pure q and complete max-min proofs remain ablations.
"""

from dataclasses import dataclass
import numpy as np

from loris._engine.typed_session import array_identity, digest


WEIGHT_ALIASES = {
    "reliability_only": "complete_provenance",
    "provenance": "complete_provenance_discount",
}
DIRECT_RULE_WEIGHTINGS = frozenset(("direct_rule_precision", "direct_rule_precision_discount"))
DISCOUNT_WEIGHTINGS = frozenset(("direct_rule_precision_discount", "complete_provenance_discount"))
RELIABILITY_WEIGHTINGS = DIRECT_RULE_WEIGHTINGS | frozenset(
    ("complete_provenance", "complete_provenance_discount")
)
WEIGHTINGS = (
    "direct_rule_precision_discount",
    "direct_rule_precision",
    "uniform",
    "confidence_only",
    "complete_provenance",
    "complete_provenance_discount",
    *WEIGHT_ALIASES,
)
RECEIVER_POLICIES = ("same_model", "affected_channels_to_aux")


def canonical_weighting(value):
    if value not in WEIGHTINGS:
        raise ValueError("explicit supported feedback weighting required")
    return WEIGHT_ALIASES.get(value, value)


@dataclass
class Batch:
    targets: np.ndarray
    observed: np.ndarray
    weights: np.ndarray
    true_mask: np.ndarray
    weak_mask: np.ndarray
    audit: dict
    feedback_state: object = None


def cumulative_verified_rows(gamma, training_rows, acquired_rows, pending_rows):
    """Required optimizer rows are acquired complete answers plus current partial answers.

    Gamma is the sole target source. The caller supplies the acquired-answer
    ledger; no unknown or unqueried row is invented as a negative example.
    """
    rows = np.asarray(training_rows)
    if rows.dtype != np.bool_ or rows.shape != (len(gamma.pos),):
        raise ValueError("cumulative verified rows need an explicit training domain")
    acquired = np.asarray(sorted(set(acquired_rows)), dtype=int)
    pending = np.asarray(sorted(set(pending_rows)), dtype=int)
    for values in (acquired, pending):
        if np.any(values < 0) or np.any(values >= len(rows)):
            raise ValueError("verified answer row is outside Gamma")
    # The acquired ledger is tiny compared with full-train Gamma. Avoid an
    # extra N x labels temporary solely to check at most a few hundred rows.
    acquired_known = gamma.pos[acquired] | gamma.neg[acquired]
    pending_known = gamma.pos[pending] | gamma.neg[pending]
    complete = acquired[rows[acquired] & acquired_known.all(1)]
    current = pending[rows[pending] & pending_known.any(1)]
    return complete, np.union1d(complete, current)


def limit_training_rows(batch, maximum, fresh_rows, previous, *, seed):
    """Explicit approximate row budget, shared by every training strategy.

    Always keep every active newly verified document. Sample the other active
    rows uniformly with the declared deterministic seed. Untrained changed
    weak facts do not advance the retained feedback history.
    """
    if maximum is None:
        return batch
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_training_rows must be a positive integer or null")
    active = np.flatnonzero(batch.observed.any(axis=1))
    fresh = np.intersect1d(active, np.asarray(fresh_rows, dtype=int))
    if len(fresh) > maximum:
        raise ValueError("training row budget cannot discard newly verified documents")
    selected = active
    if len(active) > maximum:
        optional = np.setdiff1d(active, fresh)
        rng = np.random.RandomState(seed)
        selected = np.sort(np.r_[fresh, rng.choice(optional, maximum-len(fresh), replace=False)])
    keep = np.zeros(batch.observed.shape[0], bool)
    keep[selected] = True
    removed_weak = batch.weak_mask & ~keep[:, None]
    removed_true = batch.true_mask & ~keep[:, None]
    if batch.feedback_state is not None:
        if previous is None:
            batch.feedback_state[removed_weak] = 0
        else:
            batch.feedback_state[removed_weak] = previous[removed_weak]
    for array in (batch.targets, batch.weights, batch.observed, batch.true_mask, batch.weak_mask):
        array[~keep] = 0
    if "old_trusted_replay_rows" in batch.audit:
        batch.audit["old_trusted_replay_rows"] = [
            int(i) for i in batch.audit["old_trusted_replay_rows"] if keep[int(i)]
        ]
    if "additional_trusted_replay_cells" in batch.audit:
        old_selected = np.setdiff1d(selected, fresh)
        batch.audit["additional_trusted_replay_cells"] = int(batch.true_mask[old_selected].sum())
    batch.audit.update(
        training_row_budget=dict(maximum=maximum, available=len(active), selected=len(selected),
            selected_rows=selected.tolist(), required_newly_verified_rows=fresh.tolist(), seed=int(seed),
            sampling="uniform_without_replacement_over_other_active_documents",
            approximation="eligible feedback/replay rows omitted from this optimizer pass",
            omitted_weak_cells=int(removed_weak.sum()), omitted_trusted_cells=int(removed_true.sum()),
            untrained_changed_weak_facts_advance_feedback_history=False),
        true_cells=int(batch.true_mask.sum()), weak_cells=int(batch.weak_mask.sum()),
        true_mass=float(batch.weights[batch.true_mask].sum()),
        weak_mass=float(batch.weights[batch.weak_mask].sum()),
        weak_positive=int((batch.weak_mask & (batch.targets == 1)).sum()),
        weak_negative=int((batch.weak_mask & (batch.targets == 0)).sum()),
    )
    return batch


def incremental_batch(batch, gamma, rows, fresh_rows, replay_fraction, previous):
    """Keep changed signed rule targets, fresh answers, and sampled old answers."""
    state = np.where(batch.weak_mask, np.where(batch.targets == 1, 1, -1), 0).astype(np.int8)
    unchanged = batch.weak_mask & (state == previous) if previous is not None else np.zeros_like(batch.weak_mask)
    batch.weak_mask &= ~unchanged
    fresh = np.zeros(len(rows), dtype=bool)
    fresh[fresh_rows] = True
    fresh &= rows
    known = gamma.pos | gamma.neg
    old_rows = np.flatnonzero(rows & ~fresh & known.any(axis=1))
    count = int(np.ceil(len(old_rows) * replay_fraction))
    selected = old_rows[(np.arange(count) * len(old_rows) // count)] if count else old_rows[:0]
    replay = np.zeros(len(rows), dtype=bool)
    replay[selected] = True
    batch.true_mask[:] = known & (fresh | replay)[:, None]
    batch.weak_mask &= ~batch.true_mask
    batch.targets[batch.true_mask] = gamma.pos[batch.true_mask]
    batch.weights[batch.true_mask] = 1.0
    batch.observed[:] = batch.true_mask | batch.weak_mask
    batch.weights[~batch.observed] = 0
    batch.targets[~batch.observed] = 0
    batch.feedback_state = state
    batch.audit.update(
        rule_feedback_scope="new_or_changed",
        rule_feedback_change_basis="signed_target_or_reappearing_eligibility; current_weights_on_selected_cells",
        unchanged_rule_feedback_cells_skipped=int(unchanged.sum()),
        old_trusted_rows_available=len(old_rows),
        old_trusted_replay_rows=selected.tolist(),
        fresh_verified_batch_rows=np.flatnonzero(fresh).tolist(),
        fresh_verified_batch_cells=int((known & fresh[:, None]).sum()),
        additional_trusted_replay_cells=int((known & replay[:, None]).sum()),
        trusted_correction_cells=0,
        trusted_correction_policy="fresh_answers_and_sampled_replay_only",
        true_cells=int(batch.true_mask.sum()),
        weak_cells=int(batch.weak_mask.sum()),
        true_mass=float(batch.weights[batch.true_mask].sum()),
        weak_mass=float(batch.weights[batch.weak_mask].sum()),
        weak_positive=int((batch.weak_mask & (batch.targets == 1)).sum()),
        weak_negative=int((batch.weak_mask & (batch.targets == 0)).sum()),
    )
    return batch


def assignments(
    gamma,
    closure,
    independent,
    probabilities,
    channels,
    rows,
    *,
    strategy,
    weighting,
    reliability=None,
    reliability_scores=None,
    reliability_audit=None,
    require_independent=False,
    disagreement_discount=0.5,
    replay=True,
    self_train_confidence=0.95,
    probabilities_by_channel=None,
    channel_polarities=None,
    receiver_policy="same_model",
    receiver_occurrences=None,
    replay_fraction=1.0,
    weak_mass_ratio=None,
    weak_rows_per_label=None,
    reliability_floor=None,
    selection_seed=0,
):
    """One fact per pair, even when several original eta occurrences disagree.

    2026-09-30 weak-supervision controls (None keeps the original behaviour):
    reliability_floor drops weak cells whose support score (max q of the
    supporting rules) is below the floor; weak_rows_per_label keeps at most
    that many weak cells per label (deterministic seeded sample); and
    weak_mass_ratio caps each label's weak mass at ratio x its verified mass
    by uniformly rescaling that label's weak weights.

    `channels` maps actual (rule-index, atom-index) occurrences to (label,kappa).
    `receiver_occurrences` identifies those belonging to the trainable receiver.
    The main policy uses only their disagreements; transferring disagreements
    from dependent ensembles is an explicitly named additional strategy.
    The caller checks the returned weak roots in Gamma union W before training.
    Self-training is an explicitly different baseline, not a logical correction.
    """
    if closure.status != "fixpoint" or (
        independent is not None and independent.status != "fixpoint"
    ):
        raise ValueError("feedback requires complete successful closures")
    p = np.asarray(probabilities, float)
    gp, gn = gamma.pos, gamma.neg
    pos, neg = closure.positive_facts, closure.negative_facts
    if p.shape != gp.shape or rows.dtype != np.bool_ or rows.shape != (len(p),):
        raise ValueError("feedback full row/label axes differ")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid receiver probabilities")
    requested_weighting = weighting
    weighting = canonical_weighting(weighting)
    if strategy not in ("paper", "supervised", "self_train", "frozen"):
        raise ValueError("explicit supported feedback policy required")
    if type(replay_fraction) not in (int, float) or not np.isfinite(replay_fraction) or not 0 <= replay_fraction <= 1:
        raise ValueError("replay_fraction must be finite in [0,1]")
    if strategy == "self_train" and weighting in RELIABILITY_WEIGHTINGS:
        raise ValueError(
            "SelfTrain predictions cannot use reliability of the independent logical source"
        )
    known = gp | gn
    observed_logic = pos | neg
    # Quarantine only hides Gamma from the current inference state. The stored
    # verified labels remain authorized supervision, never replaced by an
    # arbitration prediction. For ordinary closures these arrays equal logic.
    decision_observed = observed_logic | known
    decision_positive = np.where(known, gp, pos)
    disagree = np.zeros_like(known)
    channel_tables = _channel_tables(p, channels, probabilities_by_channel)
    polarities = _channel_polarities(channels, channel_polarities)
    if receiver_policy not in RECEIVER_POLICIES:
        raise ValueError("unsupported correction receiver policy")
    if receiver_occurrences is None:
        if receiver_policy == "same_model" and probabilities_by_channel is not None:
            raise ValueError(
                "same_model needs explicit receiver_occurrences for per-channel tables"
            )
        receiver_occurrences = set(channels)
    else:
        receiver_occurrences = set(receiver_occurrences)
    if not receiver_occurrences <= set(channels):
        raise ValueError("receiver occurrences must belong to the actual dynamic channels")
    if receiver_policy == "same_model":
        checked = set()
        complement = None
        for occurrence in receiver_occurrences:
            table, polarity = channel_tables[occurrence], polarities[occurrence]
            key = (id(table), polarity)
            if key in checked:
                continue
            checked.add(key)
            if polarity == -1 and complement is None:
                complement = 1.0 - p
            expected = p if polarity == 1 else complement
            if not np.array_equal(table, expected):
                raise ValueError(
                    "receiver occurrence probability table differs from receiver probabilities"
                )
    eligible_occurrences = (
        receiver_occurrences if receiver_policy == "same_model" else set(channels)
    )
    all_disagree = np.zeros_like(known)
    for occurrence, (label, threshold) in channels.items():
        prediction = channel_tables[occurrence][:, label] >= threshold
        if polarities[occurrence] == -1:
            prediction = ~prediction
        occurrence_disagrees = decision_observed[:, label] & (
            decision_positive[:, label] != prediction
        )
        all_disagree[:, label] |= occurrence_disagrees
        if occurrence in eligible_occurrences:
            disagree[:, label] |= occurrence_disagrees
    # Independent-Chase existence includes alternate derivations even when the
    # full closure's first witness reads this receiver. Shared historical model
    # ancestry is not claimed to be statistical independence.
    independent_mask = (
        np.zeros_like(known)
        if independent is None
        else ((pos & independent.positive_facts) | (neg & independent.negative_facts))
    )
    candidate = disagree & ~known & rows[:, None]
    weak = (
        (candidate & independent_mask if require_independent else candidate.copy())
        if strategy == "paper"
        else np.zeros_like(known)
    )
    targets = pos.astype(np.float32)
    if strategy == "self_train":
        # SelfTrain uses its own high-confidence predictions on declared
        # adaptable channels, including labels already resolved by logic.
        active = np.zeros(p.shape[1], bool)
        for label, _ in channels.values():
            active[label] = True
        if not np.isfinite(self_train_confidence) or not 0.5 < self_train_confidence <= 1:
            raise ValueError("SelfTrain needs explicit high-confidence threshold > .5")
        confident = (p >= self_train_confidence) | (p <= 1 - self_train_confidence)
        weak = ~known & rows[:, None] & active[None, :] & confident
        targets = (p >= 0.5).astype(np.float32)
    if strategy in ("frozen", "supervised"):
        weak[:] = False
    confidence = np.where(targets == 1, p, 1 - p)
    if (
        isinstance(disagreement_discount, bool)
        or not np.isfinite(disagreement_discount)
        or not 0 <= disagreement_discount <= 1
    ):
        raise ValueError("disagreement discount must be a finite coefficient in [0,1]")
    support_scores = np.ones_like(p)
    use_reliability = strategy == "paper" and weighting in RELIABILITY_WEIGHTINGS
    if use_reliability:
        from .reliability import SCHEMA, validated_rule_scores

        if reliability is None or reliability.get("schema") != SCHEMA:
            raise ValueError(
                "reliability weighting needs an independent per-rule calibration artifact; historical fit pooled priors are not accepted"
            )
        validated_rule_scores(reliability)
        if reliability_scores is None:
            raise ValueError(
                "all current label and relation propagation source scores are required; first-witness scores are insufficient"
                if weighting in DIRECT_RULE_WEIGHTINGS
                else "complete max-min derivation scores are required; first-witness and pooled label scores are insufficient"
            )
        support_scores = np.asarray(reliability_scores, float)
        if (
            support_scores.shape != p.shape
            or not np.isfinite(support_scores).all()
            or np.any((support_scores < 0) | (support_scores > 1))
        ):
            raise ValueError("support scores must be finite [N,L] values in [0,1]")
    unsupported_weak = int((weak & (support_scores == 0)).sum()) if use_reliability else 0
    below_floor = 0
    if reliability_floor is not None:
        if not use_reliability:
            raise ValueError("reliability_floor requires rule reliability weighting")
        low = weak & (support_scores < reliability_floor)
        below_floor = int(low.sum())
        weak &= ~low
    if weighting in DIRECT_RULE_WEIGHTINGS and unsupported_weak:
        raise ValueError(
            "source-weight invariant failed: logical corrections have no current Label, "
            "relation propagation, or verified-root support"
        )
    disagreement = np.abs(p - targets)
    raw_weight = (
        1.0
        if weighting == "uniform"
        else (
            confidence
            if weighting == "confidence_only"
            else (
                support_scores
                if weighting in ("direct_rule_precision", "complete_provenance")
                else support_scores * (1.0 - disagreement_discount * disagreement)
            )
        )
    )
    weights = np.where(weak, raw_weight, 0.0).astype(np.float32)
    weak &= weights > 0
    trusted_corrections = (
        known & disagree & rows[:, None] if strategy == "paper" else np.zeros_like(known)
    )
    if replay and strategy != "frozen":
        if replay_fraction >= 1:
            replay_rows = np.ones(len(gp), dtype=bool)
        elif replay_fraction == 0:
            replay_rows = np.zeros(len(gp), dtype=bool)
        else:
            eligible = np.flatnonzero(rows & known.any(axis=1))
            count = int(np.ceil(len(eligible) * replay_fraction))
            selected = eligible[np.arange(count) * len(eligible) // count] if count else eligible[:0]
            replay_rows = np.zeros(len(gp), dtype=bool)
            replay_rows[selected] = True
        true = known & rows[:, None] & replay_rows[:, None]
    else:
        true = trusted_corrections.copy()
    true |= trusted_corrections
    targets[true] = gp[true]
    weights[true] = 1.0
    weak &= ~true
    capped_rows = rescaled_labels = 0
    if weak_rows_per_label is not None and strategy == "paper":
        rng = np.random.RandomState(selection_seed)
        for label in range(weak.shape[1]):
            rows_l = np.flatnonzero(weak[:, label])
            if len(rows_l) > weak_rows_per_label:
                keep = np.zeros(len(rows_l), bool)
                keep[rng.choice(len(rows_l), weak_rows_per_label, replace=False)] = True
                weak[rows_l[~keep], label] = False
                capped_rows += int((~keep).sum())
    weights[~(weak | true)] = 0.0
    if weak_mass_ratio is not None and strategy == "paper":
        for label in range(weak.shape[1]):
            weak_mass = float(weights[weak[:, label], label].sum(dtype=np.float64))
            limit = weak_mass_ratio * float(weights[true[:, label], label].sum(dtype=np.float64))
            if weak_mass > limit and weak_mass > 0:
                weights[weak[:, label], label] *= np.float32(limit / weak_mass)
                rescaled_labels += 1
        weak &= weights > 0
    observed = weak | true
    targets[~observed] = 0  # storage placeholder only; weight is exactly zero
    weights[~observed] = 0
    audit = dict(
        strategy=strategy,
        weighting=weighting,
        requested_weighting=requested_weighting,
        weighting_alias=requested_weighting if requested_weighting != weighting else None,
        weak_target=(
            "logical_A"
            if strategy == "paper"
            else "own_prediction_baseline" if strategy == "self_train" else "none"
        ),
        candidate_disagreements=int(candidate.sum()),
        receiver_policy=receiver_policy,
        receiver_occurrences=[list(o) for o in sorted(receiver_occurrences)],
        correction_eligible_occurrences=[list(o) for o in sorted(eligible_occurrences)],
        other_model_only_disagreements=int(
            (all_disagree & ~disagree & ~known & rows[:, None]).sum()
        ),
        independent_supported=int((candidate & independent_mask).sum()),
        only_receiver_dependent_or_no_independent_support=int(
            (candidate & ~independent_mask).sum()
        ),
        unsupported_reliability=unsupported_weak,
        weak_below_reliability_floor=below_floor,
        reliability_floor=reliability_floor,
        weak_rows_per_label=weak_rows_per_label,
        weak_cells_removed_by_row_cap=capped_rows,
        weak_mass_ratio=weak_mass_ratio,
        labels_with_rescaled_weak_mass=rescaled_labels,
        true_cells=int(true.sum()),
        weak_cells=int(weak.sum()),
        trusted_correction_cells=int(trusted_corrections.sum()),
        additional_trusted_replay_cells=int((true & ~trusted_corrections).sum()),
        trusted_replay_fraction=float(replay_fraction if replay and strategy != "frozen" else 0.0),
        trusted_weight_policy="verified_Gamma_targets_unit_weight_operational_choice",
        weak_positive=int((weak & (targets == 1)).sum()),
        weak_negative=int((weak & (targets == 0)).sum()),
        true_mass=float(weights[true].sum()),
        weak_mass=float(weights[weak].sum()),
        weak_promoted_to_Gamma=False,
        unknown_as_negative=False,
        confidence_policy="pre_update_receiver_probability_of_A",
        disagreement_discount=disagreement_discount,
        receiver_source_filter="exclude_receiver" if require_independent else "none",
        disagreement_tables=(
            "declared_occurrence_model" if probabilities_by_channel is not None else "receiver"
        ),
        weight_formula={
            "uniform": "1",
            "confidence_only": "P_receiver(A)",
            "direct_rule_precision": "max_direct_support_rules(q_rule)",
            "direct_rule_precision_discount": "max_direct_support_rules(q_rule)*(1-disagreement_discount*abs(p_before_update-A))",
            "complete_provenance": "rho=max_complete_derivations(min_rule_q)",
            "complete_provenance_discount": "rho*(1-disagreement_discount*abs(p_before_update-A))",
        }[weighting],
        confidence_applied=weighting == "confidence_only" or weighting in DISCOUNT_WEIGHTINGS,
        reliability_applied=use_reliability,
        quality_is_not_proved_by_provenance=True,
        reliability_unit="rule_distinct_checkable_conclusion" if use_reliability else None,
        reliability_prior_policy=reliability.get("transfer_policy") if use_reliability else None,
        reliability_derivation_audit=reliability_audit if use_reliability else None,
        reliability_base_binding=reliability.get("binding") if use_reliability else None,
        reliability_estimator=reliability.get("estimator") if use_reliability else None,
        reliability_schema=reliability.get("schema") if use_reliability else None,
        normalization="common_observed_mass_per_actual_optimizer_accumulation_window",
        separate_true_and_weak_unit_mass_normalization=False,
        evidence_scope=(
            "complete_receiver_excluded_Chase" if require_independent else "complete_current_Chase"
        ),
        parameter_or_first_witness_changes_are_not_new_evidence=True,
    )
    if strategy == "self_train":
        audit.update(
            self_train_confidence=self_train_confidence,
            self_train_mask="not trusted Gamma; allowed training rows; declared dynamic labels; own high-confidence predictions",
        )
    return Batch(targets, observed, weights, true, weak, audit)


def batch_id(batch):
    return digest(
        {
            k: array_identity(getattr(batch, k))
            for k in ("targets", "observed", "weights", "true_mask", "weak_mask")
        }
    )


def _channel_tables(probabilities, channels, probabilities_by_channel):
    if probabilities_by_channel is None:
        return {occurrence: probabilities for occurrence in channels}
    if set(probabilities_by_channel) != set(channels):
        raise ValueError(
            "per-occurrence probability mapping must match every actual dynamic channel"
        )
    tables, checked = {}, {}
    for occurrence, value in probabilities_by_channel.items():
        key = id(value)
        if key in checked:
            tables[occurrence] = checked[key]
            continue
        p = np.asarray(value, dtype=np.float64)
        if p.shape != probabilities.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("per-occurrence probability table differs from complete checked axes")
        checked[key] = p
        tables[occurrence] = p
    return tables


def _channel_polarities(channels, values):
    if values is None:
        return {key: 1 for key in channels}
    if set(values) - set(channels) or any(type(v) is not int or v not in (-1, 1)
                                       for v in values.values()):
        raise ValueError("channel polarities must identify actual channels with +1 or -1")
    return {key: values.get(key, 1) for key in channels}


def calibrate(
    probabilities,
    batch,
    channels,
    *,
    grid,
    minimum_support_each_sign,
    probabilities_by_channel=None,
    channel_polarities=None,
    cells="all",
    max_step=None,
):
    """Weighted binary accuracy on EXACT checked train assignments; strict gain.

    2026-09-30 options (defaults keep the original behaviour):
    cells="trusted_only" scores only verified (true_mask) cells, never weak
    logical corrections, so kappa is not pulled toward the model's own
    disagreements; max_step bounds |kappa_new - kappa_old| per round.

    Each original occurrence retains its own old kappa on ties/no support.
    No dev metric and no arbitrary 0.5 reset. The grid is fixed before fitting;
    the old value is always retained as the zero-change option.
    """
    # ReferenceChase._tables promotes stored probabilities before evaluating
    # ML >= eta. Preserve that boundary behavior (float32(.95) < float(.95)).
    p = np.asarray(probabilities, dtype=np.float64)
    grid = tuple(grid)
    if (
        not grid
        or len(set(grid)) != len(grid)
        or any(isinstance(t, bool) or not np.isfinite(t) or not 0 <= t <= 1 for t in grid)
        or type(minimum_support_each_sign) is not int
        or minimum_support_each_sign < 1
    ):
        raise ValueError("finite unique threshold grid and positive support per sign required")
    if cells not in ("all", "trusted_only") or (
            max_step is not None and (type(max_step) not in (int, float)
                                      or not np.isfinite(max_step) or max_step <= 0)):
        raise ValueError("calibration cells/max_step are invalid")
    if p.shape != batch.targets.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("calibration probabilities differ from checked assignment axes")
    channel_tables = _channel_tables(p, channels, probabilities_by_channel)
    polarities = _channel_polarities(channels, channel_polarities)
    checked_id = batch_id(batch)
    result, records = {}, []
    for occurrence, (label, old) in channels.items():
        valid = batch.observed[:, label] & (batch.weights[:, label] > 0)
        if cells == "trusted_only":
            valid &= batch.true_mask[:, label]
        y = batch.targets[valid, label].astype(bool)
        w = batch.weights[valid, label].astype(float)
        counts = [int((~y).sum()), int(y.sum())]
        item = dict(
            occurrence=list(occurrence),
            label_index=label,
            previous=old,
            signed_support=counts,
            checked_assignments_id=checked_id,
            channel_polarity=polarities[occurrence],
            criterion="weighted_binary_accuracy",
            comparator=">=",
            probability_comparison_dtype="float64_like_ReferenceChase",
        )
        if min(counts) < minimum_support_each_sign:
            result[occurrence] = (label, old)
            records.append(
                dict(item, selected=old, status="unsupported_preserve_previous", scores=[])
            )
            continue
        values = sorted(set(grid + (old,)))
        if max_step is not None:
            values = [t for t in values if abs(t - old) <= max_step + 1e-12]
        channel_p = channel_tables[occurrence]
        event_target = y if polarities[occurrence] == 1 else ~y
        scores = {
            t: float(np.sum(w * ((channel_p[valid, label] >= t) == event_target)) / w.sum()) for t in values
        }
        best = max(scores.values())
        selected = old if best <= scores[old] else max(t for t in values if scores[t] == best)
        result[occurrence] = (label, selected)
        records.append(
            dict(
                item,
                selected=selected,
                status="strict_gain" if selected != old else "no_gain",
                scores=[dict(kappa=t, score=scores[t]) for t in values],
            )
        )
    return result, dict(
        checked_assignments_id=checked_id,
        occurrences=records,
        no_unqueried_truth=True,
        comparison="same_train_assignments_not_heldout_selection",
        cells=cells,
        max_step=max_step,
        probability_tables=(
            "declared_occurrence_model" if probabilities_by_channel is not None else "receiver"
        ),
    )
