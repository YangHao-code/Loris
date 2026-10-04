#!/usr/bin/env python
"""Require matched protocols/events/final states before reporting NoInc speedup."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from loris.experiments.states import completion_state_evidence, comparable_endpoint


def compare(workspace):
    work = Path(workspace)
    configs = [
        json.loads((work / "effective" / ("labeling_" + v + ".json")).read_text())
        for v in ("main", "noinc")
    ]
    modes = [c["loop"].pop("execution") for c in configs]
    if modes != ["incremental", "full_recompute"] or configs[0] != configs[1]:
        raise ValueError("NoInc comparison must differ only in closure execution")
    runs = [work / "labeling" / v for v in ("main", "noinc")]
    completions = [json.loads((p / "COMPLETE.json").read_text()) for p in runs]
    states = [
        completion_state_evidence(c, config, run)
        for c, config, run in zip(completions, configs, runs)
    ]
    if completions[0]["status"] != completions[1]["status"] or not all(
        comparable_endpoint(c) for c in completions
    ):
        raise ValueError("NoInc requires matched normal endpoints, not resource or failed runs")
    for name in ("FINAL_P.npz", "TRUSTED_GAMMA_FINAL.npz"):
        with np.load(runs[0] / name, allow_pickle=False) as a, np.load(
            runs[1] / name, allow_pickle=False
        ) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError("NoInc semantic mismatch: " + name)
    sequences = []
    for run in runs:
        seq = []
        for line in (run / "EVENTS.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["kind"] in ("label_request", "answer_accepted", "source_review_approved"):
                seq.append(
                    {
                        k: v
                        for k, v in event.items()
                        if k in ("kind", "row", "row_id", "reason", "rule_index", "Gamma")
                    }
                )
        sequences.append(seq)
    if sequences[0] != sequences[1]:
        raise ValueError("NoInc annotation/source-review event sequences differ")
    for key in (
        "actual_optimizer_updates",
        "committed_model_updates",
        "actual_label_requests",
        "actual_source_reviews",
    ):
        if completions[0].get(key) != completions[1].get(key):
            raise ValueError("NoInc update or supervision counts differ: " + key)
    times = [c["timings"]["chase_seconds"] for c in completions]
    unknown = configs[0]["loop"].get("conflict_policy", "halt") == "unknown_arbitrate"
    retained = (
        unknown
        and configs[0]["loop"].get("unknown_semantics", "full_rebuild") == "retain_consequences"
    )
    quarantined_recompute = unknown and not retained
    if retained:
        # History is part of the new state. Equal final P alone can conceal
        # different retained ancestors and therefore different next updates.
        histories = [c.get("state", {}).get("retained_history") for c in completions]
        if histories[0] is None or histories[0] != histories[1]:
            raise ValueError("NoInc retained history differs or is not recorded")
    result = dict(
        status="equivalent",
        same_final_signed_P=True,
        same_Gamma=True,
        same_external_events=True,
        same_declared_update_counts=True,
        endpoint_status=completions[0]["status"],
        all_documents_resolved=completions[0]["status"] == "all_documents_resolved",
        final_state_kind=states[0]["final_state_kind"],
        requested_incremental_chase_seconds=times[0],
        incremental_chase_seconds=None if quarantined_recompute else times[0],
        full_recompute_chase_seconds=times[1],
        chase_speedup=times[1] / times[0] if times[0] > 0 and not quarantined_recompute else None,
        incremental_speedup_claimed=not quarantined_recompute,
        execution_interpretation=(
            "both_full_recompute_under_unknown_arbitrate"
            if quarantined_recompute
            else (
                "retained_history_incremental_vs_full_rule_scan"
                if retained
                else "legacy_halt_incremental_vs_full"
            )
        ),
        scope="matched final-state/event equivalence; this is not a proof for every possible dataset",
    )
    path = work / "NOINC_EQUIVALENCE.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    print(json.dumps(compare(p.parse_args().workspace), indent=2))
