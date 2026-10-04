"""Keep classical closures distinct from certified quarantined fixed points."""

from pathlib import Path

import numpy as np

from loris.discovery.manifest import bound, file_record


QUARANTINED_STATES = ("quarantined_fixpoint", "retained_quarantined_fixpoint")


def final_state_kind(record):
    classical = record.get("current_Gamma_has_valid_closure") is True
    quarantined = record.get("current_Gamma_has_quarantined_fixpoint") is True
    retained = record.get("current_history_has_quarantined_fixpoint") is True
    if (
        retained
        and not classical
        and not quarantined
        and record.get("effective_execution") in ("retained_incremental", "retained_full_scan")
    ):
        return "retained_quarantined_fixpoint"
    if classical and not quarantined and not retained:
        return "classical_fixpoint"
    if (
        quarantined
        and not classical
        and not retained
        and record.get("effective_execution") == "full_recompute_quarantine"
    ):
        return "quarantined_fixpoint"
    return None


def completion_state_evidence(record, manifest, run):
    kind = final_state_kind(record)
    policy = manifest.get("loop", {}).get("conflict_policy", "halt")
    if kind is None or (policy == "unknown_arbitrate") != (kind in QUARANTINED_STATES):
        raise ValueError(
            "there is no valid completed closure or declared quarantined fixpoint to evaluate"
        )
    if kind in QUARANTINED_STATES and (
        manifest.get("loop", {}).get("unknown_semantics", "full_rebuild") == "retain_consequences"
    ) != (kind == "retained_quarantined_fixpoint"):
        raise ValueError("completed unknown semantics differ from the declared protocol")
    return dict(
        final_state_kind=kind,
        current_Gamma_has_valid_closure=kind == "classical_fixpoint",
        current_Gamma_has_quarantined_fixpoint=kind == "quarantined_fixpoint",
        current_history_has_quarantined_fixpoint=kind == "retained_quarantined_fixpoint",
        unknown_semantics=manifest.get("loop", {}).get("unknown_semantics", "full_rebuild"),
        conflict_policy=policy,
        active_quarantine_pairs=record.get("active_quarantine_pairs", 0),
        effective_execution=record.get("effective_execution"),
        full_recompute_reason=record.get("full_recompute_reason"),
        model_arbitration_rules=(
            file_record(Path(run) / "MODEL_ARBITRATION_RULES.json")
            if kind in QUARANTINED_STATES
            else None
        ),
    )


def evaluation_has_valid_state(record):
    kind = final_state_kind(record)
    if kind in QUARANTINED_STATES:
        evidence = record.get("model_arbitration_rules")
        if not isinstance(evidence, dict):
            return False
        bound(evidence, Path("."))
    return kind is not None


def comparable_endpoint(record):
    """Saturation permits a comparison, never a claim that all labels resolved."""
    status = record.get("run_status", record.get("status"))
    if "task_complete" in record and record["task_complete"] is not (
        status == "all_documents_resolved"
    ):
        return False
    if status in ("all_documents_resolved", "label_budget_exhausted"):
        return True
    return (
        status == "conflict_unknown_saturated"
        and final_state_kind(record) in QUARANTINED_STATES
        and record.get("task_complete", False) is False
        and record.get("active_quarantine_pairs", 0) > 0
    )


def final_arrays_evidence(facts, context, completion):
    """Check the already-loaded output, its axes and the existing state identity.

    This does not re-read or checksum every input/model artifact. Old strict
    two-array outputs remain readable, with their limited evidence stated.
    """
    shape = (len(context["row_ids"]), len(context["labels"]))
    pos, neg = facts["positive"], facts["negative"]
    if (
        pos.shape != shape
        or neg.shape != shape
        or pos.dtype != np.bool_
        or neg.dtype != np.bool_
        or np.any(pos & neg)
    ):
        raise ValueError("invalid final signed closure")
    axes = {name for name in ("row_ids", "labels") if name in facts.files}
    if axes and (
        axes != {"row_ids", "labels"}
        or facts["row_ids"].tolist() != list(context["row_ids"])
        or facts["labels"].tolist() != list(context["labels"])
    ):
        raise ValueError("final prediction row/label axis differs from application context")
    fields = (
        "logical_positive",
        "logical_negative",
        "conflict_unknown",
        "remaining_unknown",
        "arbitration_positive",
        "arbitration_negative",
    )
    present = {name for name in fields if name in facts.files}
    quarantined = final_state_kind(completion) in QUARANTINED_STATES
    # Undecided-cell Aux fallback (2026-09-30): an output-only positive layer.
    fallback = None
    if "fallback_positive" in facts.files:
        fallback = facts["fallback_positive"]
        if fallback.shape != shape or fallback.dtype != np.bool_ or not present:
            raise ValueError("fallback positives require complete final masks on the same axes")
    if present != set(fields) and (present or quarantined):
        raise ValueError("final evaluation requires all logical/unknown/arbitration masks")
    if present:
        if any(facts[name].shape != shape or facts[name].dtype != np.bool_ for name in fields):
            raise ValueError("final logical/unknown/arbitration mask axes or types differ")
        lp, ln, unknown, remaining, ap, an = (facts[name] for name in fields)
        fp = np.zeros(shape, bool) if fallback is None else fallback
        if (
            np.any(lp & ln)
            or np.any(ap & an)
            or np.any((lp | ln) & unknown)
            or np.any((ap | an) & ~unknown)
            or np.any(fp & (lp | ln | unknown))
            or not np.array_equal(pos, lp | ap | fp)
            or not np.array_equal(neg, ln | an)
            or not np.array_equal(remaining, unknown & ~(ap | an))
        ):
            raise ValueError("final output/logical/arbitration/unknown masks are inconsistent")
        if not quarantined and np.any(unknown | ap | an):
            raise ValueError("classical final state cannot contain quarantined/arbitrated pairs")
    else:
        lp, ln, unknown = pos, neg, np.zeros(shape, bool)
    if int(completion.get("active_quarantine_pairs", 0)) != int(unknown.sum()):
        raise ValueError("completion quarantine count differs from final masks")
    pending = ~lp.any(axis=1) | unknown.any(axis=1)
    all_resolved = not pending.any()
    claimed = completion.get("task_complete", False) is True
    status_complete = completion.get("status") == "all_documents_resolved"
    if claimed != status_complete or (status_complete and not all_resolved):
        raise ValueError("completion all-resolved claim contradicts final label state")
    if completion.get("status") == "conflict_unknown_saturated" and not (
        quarantined and unknown.any() and not claimed
    ):
        raise ValueError("conflict saturation requires actual quarantined unresolved pairs")
    checked = []
    state = completion.get("state", {})
    if state:
        from loris._engine.typed_session import array_identity

        arrays = dict(positive=lp, negative=ln)
        if present:
            arrays.update({name: facts[name] for name in fields[2:]})
        for name, values in arrays.items():
            if name in state:
                if array_identity(values) != state[name]:
                    raise ValueError("final array differs from recorded state: " + name)
                checked.append(name)
    return dict(
        prediction_axes="explicit" if axes else "legacy_context_order_only",
        masks="complete_signed_views" if present else "legacy_signed_output_only",
        fallback_positive_cells=None if fallback is None else int(fallback.sum()),
        existing_state_arrays_checked=checked,
        unresolved_documents=int(pending.sum()),
        all_documents_resolved_from_arrays=bool(all_resolved),
        new_file_hash_protocol=False,
    )
