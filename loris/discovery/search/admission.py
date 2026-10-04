"""Paper Algorithm 1 descending-score admission, plus historical proposal order.

Sorting occurs only after the complete cluster BO. Raw proposals and Optuna
observations keep their original order. Every changed prefix is re-evaluated.
"""

import json
import math
from pathlib import Path

from loris._engine.rlearner import admit_cluster, digest, immutable_json, rule_record
from loris.rule_forms import (
    audit_program_rule_forms,
    validate_program_rule_forms,
    trial_rule_form_counts,
)

ORDERS = ("proposal_order", "standalone_score_desc", "random_consistent")


def order_trials(records, order):
    if order not in ORDERS:
        raise ValueError("unsupported admission_order")
    if any(r["number"] != i for i, r in enumerate(records)):
        raise ValueError("raw trial order must be complete and unchanged")
    if any(
        type(r["standalone"]["score"]) not in (int, float)
        or not math.isfinite(r["standalone"]["score"])
        for r in records
    ):
        raise ValueError(
            "BO optimizer scores must be finite; failed candidates retain their sentinel status"
        )
    return (
        list(range(len(records)))
        if order in ("proposal_order", "random_consistent")
        else sorted(range(len(records)), key=lambda i: (-records[i]["standalone"]["score"], i))
    )


def admit_ordered_cluster(
    *, order, directory, records, selected, evaluator, rule_parser, on_progress=None,
    policy="strict", score_batch=None, batch_size=8
):
    from .parallel_admission import POLICIES, admit_batches
    if policy not in POLICIES:
        raise ValueError("unsupported admission policy")
    validate_program_rule_forms(selected)
    candidate_forms = audit_program_rule_forms(r["rule"] for r in records if r["rule"] is not None)
    immutable_json(
        Path(directory) / "RULE_FORM_AUDIT.json",
        dict(candidate_forms, trial_rejections=trial_rule_form_counts(records)),
    )
    validate_program_rule_forms(r["rule"] for r in records if r["rule"] is not None)
    numbers = order_trials(records, order)
    if order == "proposal_order" and policy == "strict":
        return admit_cluster(
            directory=directory,
            records=records,
            selected=selected,
            evaluator=evaluator,
            rule_parser=rule_parser,
            on_progress=on_progress,
        )
    directory = Path(directory)
    complete = json.loads((directory.parent / "BO_COMPLETE.json").read_text())
    protocol = json.loads((directory.parent / "protocol.json").read_text())
    if (
        complete["protocol_id"] != digest(protocol)
        or protocol["evaluator"] != evaluator.binding
        or protocol["whole_program_prefix_used_in_BO"]
        or protocol["historical_candidates_loaded"]
        or complete["complete_trials"] != len(records)
        or complete["all_trial_records_sha256"] != digest(records)
    ):
        raise ValueError("ordered admission requires the unchanged completed fresh BO records")
    if policy == "batch_prefilter":
        if score_batch is None:
            raise ValueError("batch prefilter requires an explicit parallel scorer")
        from .source_cache import cached_source_extraction
        with cached_source_extraction():
            return admit_batches(order=order, directory=directory, records=records,
                numbers=numbers, selected=selected, evaluator=evaluator,
                rule_parser=rule_parser, score_batch=score_batch, batch_size=batch_size,
                on_progress=on_progress)
    immutable_json(
        directory / "ORDER.json",
        dict(
            order=order,
            trial_numbers=numbers,
            tie_break="original trial number within the unchanged cluster order",
            raw_records_sha256=digest(records),
            changed_prefix_requires_rescoring=True,
            strict_gain_epsilon=0.0,
            score_guided=order != "random_consistent",
        ),
    )
    selected = list(selected)
    immutable_json(
        directory / "before.json",
        dict(rules=[rule_record(r) for r in selected], evaluation=evaluator.evaluate(selected)),
    )
    for position, number in enumerate(numbers):
        record = records[number]
        raw = record["rule"]
        if raw is None:
            decision = dict(line=12, accepted=False, reason=record["no_body_reason"])
        else:
            rule = rule_parser(raw)
            # Algorithm 1 requires strictly positive marginal F1. A numerical
            # gain floor belongs only to the explicit historical refilter.
            if order == "random_consistent":
                baseline = evaluator.evaluate(selected)
                if baseline["status"] != "fixpoint":
                    raise ValueError("random admission requires a consistent selected prefix")
                proposed = evaluator.evaluate(tuple(selected) + (rule,))
                duplicate = raw in [rule_record(r) for r in selected]
                accepted = proposed["status"] == "fixpoint" and not duplicate
                delta = (
                    proposed["positive_metrics"]["macro_f1"]
                    - baseline["positive_metrics"]["macro_f1"]
                    if proposed["status"] == "fixpoint"
                    else None
                )
                decision = dict(
                    line=12,
                    accepted=accepted,
                    criterion="consistent_distinct_rule",
                    delta_macro_f1=delta,
                    baseline=baseline,
                    evaluation=proposed,
                    score_used_for_selection=False,
                    reason=(
                        "duplicate"
                        if duplicate
                        else ("consistent_random_rule" if accepted else "conflict")
                    ),
                )
            else:
                decision = evaluator.admit(selected, rule, epsilon=0.0)
        receipt = dict(
            trial_number=number,
            admission_position=position,
            candidate=raw,
            standalone=record["standalone"],
            decision=decision,
            selected_before=len(selected),
        )
        if decision["accepted"]:
            selected.append(rule)
        receipt.update(
            selected_after=len(selected), selected_program_key=evaluator.program_key(selected)
        )
        immutable_json(directory / ("candidate_%05d.json" % number), receipt)
        if on_progress:
            on_progress(
                dict(
                    phase="line12_whole_program_admission",
                    processed=position + 1,
                    source_trial=number,
                    admission_order=order,
                    budget=len(records),
                    selected=len(selected),
                    actual_chase_calls=evaluator.actual_calls,
                    cache_hits=evaluator.cache_hits,
                )
            )
    immutable_json(
        directory / "after.json",
        dict(
            rules=[rule_record(r) for r in selected],
            evaluation=evaluator.evaluate(selected),
            all_candidates_processed=len(records),
        ),
    )
    return selected
