#!/usr/bin/env python
"""Aggregate comparable completed runs; retain incomplete runs as diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from loris.experiments.states import comparable_endpoint, evaluation_has_valid_state, final_state_kind


def aggregate(workspaces, output):
    workspaces = [Path(p).resolve(strict=True) for p in workspaces]
    if len(set(workspaces)) != len(workspaces):
        raise ValueError("a workspace cannot count as two repetitions")
    results = {}
    for work in workspaces:
        asset = json.loads((work / "ASSETS.json").read_text())
        for p in sorted((work / "evaluation").glob("*/RESULTS.json")):
            item = json.loads(p.read_text())
            key = (
                item["variant"],
                item["evaluation_role"],
                tuple(item["labels"]),
                item["evaluation_manifest"]["sha256"],
                item.get("discovery_proposal_strategy", "unknown"),
                item.get("discovery_admission_order", "unknown"),
                item.get("supervision_protocol", "unknown_legacy"),
                item.get("weighting", "unknown_legacy"),
                final_state_kind(item),
            )
            results.setdefault(key, []).append((str(work), asset, item))
    if not results:
        raise ValueError("no actual evaluation results")
    rows = []
    for key, group in results.items():
        if len(group) != len(workspaces):
            raise ValueError(
                "a variant/evaluation axis is missing or differs across repetitions: "
                + str(key[:2])
            )
        completed = [
            item
            for _, _, item in group
            if item.get("task_complete") is True
            and item["run_status"] == "all_documents_resolved"
            and evaluation_has_valid_state(item)
        ]
        budgets = {item.get("declared_label_budget") for _, _, item in group}
        statuses = {item["run_status"] for _, _, item in group}
        endpoint = (
            len(budgets) == 1
            and None not in budgets
            and len(statuses) == 1
            and statuses <= {"label_budget_exhausted", "conflict_unknown_saturated"}
            and all(
                comparable_endpoint(item) and evaluation_has_valid_state(item)
                for _, _, item in group
            )
        )
        scope = (
            "all_resolved"
            if len(completed) == len(group)
            else (
                "frozen_zero_budget_endpoint"
                if endpoint and budgets == {0} and key[0] == "frozen_zero_budget"
                else "quarantine_saturated_endpoint" if endpoint and statuses == {"conflict_unknown_saturated"}
                else "budget_endpoint" if endpoint else "diagnostic_only"
            )
        )
        row = dict(
            variant=key[0],
            evaluation_role=key[1],
            runs=len(group),
            complete_runs=len(completed),
            discovery_proposal_strategy=key[4],
            discovery_admission_order=key[5],
            supervision_protocol=key[6],
            weighting=key[7],
            final_state_kind=key[8],
            all_resolved_runs=len(completed),
            statistics_scope=scope,
            declared_label_budgets=sorted(budgets, key=str),
            all_resolved_main_result_claimed=(key[0] == "main" and scope == "all_resolved"),
            workspace_statuses=[
                dict(
                    workspace=w, status=i["run_status"], task_complete=i.get("task_complete", False)
                )
                for w, _, i in group
            ],
            statistics=None,
        )
        if scope != "diagnostic_only":
            row["statistics"] = {}
            for metric in ("macro_f1", "micro_f1", "coverage", "multi_label_completeness"):
                values = [item["metrics"][metric] for _, _, item in group]
                row["statistics"][metric] = (
                    None
                    if any(v is None for v in values)
                    else dict(
                        mean=statistics.mean(values),
                        sample_std=statistics.stdev(values) if len(values) > 1 else None,
                        values=values,
                    )
                )
        rows.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        rows=rows,
        independent_seed_training_requires_author_provenance=True,
        incomplete_runs_never_count_as_completed_main_results=True,
        test_based_configuration_selection=False,
        matched_budget_does_not_certify_matched_supervision=True,
    )
    (output / "REPEATED_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspaces", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aggregate(args.workspaces, args.output)
