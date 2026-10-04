"""Endpoint accounting and actual incremental/recompute experiment controls."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from loris.experiments.workflow import write
from loris.discovery.manifest import file_record


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts/experiments" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repetitions(tmp_path, status="label_budget_exhausted", budget=5, variant="main"):
    works = []
    for i in range(3):
        work = tmp_path / str(i)
        write(work / "ASSETS.json", {})
        write(
            work / "evaluation" / variant / "RESULTS.json",
            dict(
                variant=variant,
                evaluation_role="dev",
                labels=["A"],
                evaluation_manifest={"sha256": "one-scoring-protocol"},
                run_status=status,
                task_complete=status == "all_documents_resolved",
                current_Gamma_has_valid_closure=True,
                declared_label_budget=budget,
                metrics=dict(
                    macro_f1=0.5 + i * 0.1,
                    micro_f1=0.6,
                    coverage=0.7,
                    multi_label_completeness=None,
                ),
            ),
        )
        works.append(work)
    return works


def test_repeated_budget_endpoint_never_claims_complete_main(tmp_path):
    module = script("aggregate_repeats")
    works = repetitions(tmp_path)
    row = module.aggregate(works, tmp_path / "summary")["rows"][0]
    assert row["statistics_scope"] == "budget_endpoint"
    assert row["statistics"]["macro_f1"]["mean"] == pytest.approx(0.6)
    assert row["all_resolved_runs"] == 0 and not row["all_resolved_main_result_claimed"]


@pytest.mark.parametrize(
    "status,budget",
    [("resource_limit", 5), ("pending_external_answer", 5), ("label_budget_exhausted", 6)],
)
def test_failed_pending_or_mismatched_budget_cannot_enter_repeated_statistics(
    tmp_path, status, budget
):
    works = repetitions(tmp_path)
    path = works[-1] / "evaluation/main/RESULTS.json"
    data = json.loads(path.read_text())
    data.update(run_status=status, declared_label_budget=budget)
    write(path, data)
    row = script("aggregate_repeats").aggregate(works, tmp_path / "summary")["rows"][0]
    assert row["statistics"] is None and row["statistics_scope"] == "diagnostic_only"


def noinc_pair(tmp_path):
    for variant, mode in [("main", "incremental"), ("noinc", "full_recompute")]:
        write(
            tmp_path / "effective" / ("labeling_" + variant + ".json"),
            dict(loop=dict(execution=mode, label_budget=0)),
        )
        run = tmp_path / "labeling" / variant
        write(
            run / "COMPLETE.json",
            dict(
                status="label_budget_exhausted",
                current_Gamma_has_valid_closure=True,
                timings=dict(chase_seconds=1 if variant == "main" else 2),
                actual_optimizer_updates=0,
                committed_model_updates=0,
                actual_label_requests=0,
                actual_source_reviews=0,
            ),
        )
        (run / "EVENTS.jsonl").write_text(json.dumps(dict(kind="Chase", phase="initial")) + "\n")
        for name in ("FINAL_P.npz", "TRUSTED_GAMMA_FINAL.npz"):
            np.savez(run / name, positive=np.ones((1, 1), bool), negative=np.zeros((1, 1), bool))


def test_noinc_requires_actual_final_fact_and_gamma_equality(tmp_path):
    noinc_pair(tmp_path)
    np.savez(
        tmp_path / "labeling/noinc/TRUSTED_GAMMA_FINAL.npz",
        positive=np.zeros((1, 1), bool),
        negative=np.zeros((1, 1), bool),
    )
    with pytest.raises(ValueError, match="semantic mismatch"):
        script("check_noinc_equivalence").compare(tmp_path)
    assert not (tmp_path / "NOINC_EQUIVALENCE.json").exists()


def test_noinc_equivalent_reports_measured_chase_speedup(tmp_path):
    noinc_pair(tmp_path)
    assert script("check_noinc_equivalence").compare(tmp_path)["chase_speedup"] == 2


def test_unknown_arbitration_recompute_never_claims_incremental_speedup(tmp_path):
    noinc_pair(tmp_path)
    for variant in ("main", "noinc"):
        path = tmp_path / "effective" / ("labeling_" + variant + ".json")
        data = json.loads(path.read_text())
        data["loop"]["conflict_policy"] = "unknown_arbitrate"
        write(path, data)
        run = tmp_path / "labeling" / variant
        complete = json.loads((run / "COMPLETE.json").read_text())
        complete.update(
            current_Gamma_has_valid_closure=False,
            current_Gamma_has_quarantined_fixpoint=True,
            effective_execution="full_recompute_quarantine",
            active_quarantine_pairs=1,
        )
        write(run / "COMPLETE.json", complete)
        write(run / "MODEL_ARBITRATION_RULES.json", dict(test_fixture_only=True))
    result = script("check_noinc_equivalence").compare(tmp_path)
    assert result["chase_speedup"] is None
    assert result["incremental_chase_seconds"] is None
    assert not result["incremental_speedup_claimed"]
    assert result["requested_incremental_chase_seconds"] == 1


def test_repeated_quarantined_results_are_evaluated_without_claiming_classical_closure(tmp_path):
    works = repetitions(tmp_path)
    for work in works:
        rules = work / "MODEL_ARBITRATION_RULES.json"
        write(rules, dict(test_fixture_only=True))
        path = work / "evaluation/main/RESULTS.json"
        data = json.loads(path.read_text())
        data.update(
            current_Gamma_has_valid_closure=False,
            current_Gamma_has_quarantined_fixpoint=True,
            effective_execution="full_recompute_quarantine",
            model_arbitration_rules=file_record(rules),
        )
        write(path, data)
    row = script("aggregate_repeats").aggregate(works, tmp_path / "report")["rows"][0]
    assert row["final_state_kind"] == "quarantined_fixpoint"
    assert row["statistics_scope"] == "budget_endpoint"
    assert row["statistics"]["macro_f1"]["mean"] == pytest.approx(0.6)


def test_scaling_rejects_hidden_second_axis_change_before_execution(tmp_path):
    assets = tmp_path / "assets.json"
    write(assets, {})
    points = [
        dict(
            id="small",
            document_ids=["d1"],
            labels=["A"],
            rule_ids=["r"],
            assets=file_record(assets),
        ),
        dict(
            id="large",
            document_ids=["d1", "d2"],
            labels=["A", "B"],
            rule_ids=["r"],
            assets=file_record(assets),
        ),
    ]
    path = tmp_path / "sweep.json"
    write(
        path,
        dict(schema="loris_prepared_scaling_sweep_v1", dimension="document_ids", points=points),
    )
    with pytest.raises(ValueError, match="only the declared scaling dimension"):
        script("run_scaling").main(path, tmp_path / "output", True)
    assert not (tmp_path / "output").exists()
