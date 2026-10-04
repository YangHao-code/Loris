"""Paper split protection, real saved-view consistency and offline round scoring."""

import csv
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from loris._engine.rdl_reference import RefGamma
from loris._engine.typed_session import array_identity
from loris.discovery.manifest import file_record
from loris.experiments import workflow
from tests.test_release_workflow import annotation_evaluation_fixture, quarantined_evaluation_fixture, dataset
from tests.test_revision2_experiment_protocols import completed_fixed_gamma, script
from tests.test_experiment_controls import noinc_pair, repetitions


def replace_context(work, transform):
    path = work / "effective/labeling_main.json"
    app = workflow.read(path)
    context_path = Path(app["context"]["path"])
    context = pickle.loads(context_path.read_bytes())
    transform(context)
    context_path.write_bytes(pickle.dumps(context))
    app["context"] = file_record(context_path)
    workflow.write(path, app)
    workflow.write(work / "labeling/main/INPUT_RECEIPT.json", dict(manifest=file_record(path)))


@pytest.mark.parametrize("polarity", ["pos", "neg"])
def test_test_evaluation_rejects_initial_Gamma_even_with_no_initial_fit_overlap(tmp_path, polarity):
    work = annotation_evaluation_fixture(tmp_path)
    def contaminate(context):
        getattr(context["gamma"], polarity)[2, 0] = True
    replace_context(work, contaminate)
    with pytest.raises(ValueError, match="initial Gamma"):
        workflow.evaluate_final(work, "main", work / "evaluation/main")


@pytest.mark.parametrize("root_id", ["train0", "test0"])
def test_paper_initial_Gamma_gate_uses_real_split_ID_not_context_role(tmp_path, root_id):
    from loris.document import Document

    source = dataset(tmp_path)
    data = workflow.read(source)
    data["application_splits"] = ["train", "test"]
    workflow.write(source, data)
    # Reordered input and deliberately misleading context roles.
    ids = ["test0", "train0"]
    positive = np.zeros((2, 1), bool)
    positive[ids.index(root_id), 0] = True
    gamma = RefGamma(positive, np.zeros_like(positive))
    context_path = tmp_path / "context.pkl"
    context_path.write_bytes(pickle.dumps(dict(row_ids=ids, row_roles=["train", "train"],
        labels=["A"], gamma=gamma, docs=[Document(cnt="test"), Document(cnt="train")])))
    app = tmp_path / "app.json"
    workflow.write(app, dict(context=file_record(context_path)))
    work = tmp_path / "work"
    workflow.write(work / "ASSETS.json", dict(asset_root=str(tmp_path), repo_root=str(tmp_path),
        artifacts=dict(dataset_audit=file_record(source), labeling_manifest=file_record(app))))
    if root_id == "test0":
        with pytest.raises(ValueError, match="actual train IDs"):
            workflow.audit_application(work, workflow.audit_dataset(source))
    else:
        result = workflow.audit_application(work, workflow.audit_dataset(source))
        assert result["application_context"]["initial_Gamma_documents"] == 1


def final_arrays(work):
    path = work / "labeling/main/FINAL_P.npz"
    with np.load(path, allow_pickle=False) as data:
        values = {k: data[k].copy() for k in data.files}
    values.update(row_ids=np.array(["train0", "test0", "test1", "test2"]), labels=np.array(["A"]))
    return path, values


@pytest.mark.parametrize("change", ["replace_output", "row_order", "arbitration_outside_unknown", "remaining_unknown", "false_completion", "recorded_state"])
def test_final_views_and_existing_state_reject_inconsistent_artifacts(tmp_path, change):
    work = quarantined_evaluation_fixture(tmp_path)
    path, values = final_arrays(work)
    completion_path = work / "labeling/main/COMPLETE.json"
    completion = workflow.read(completion_path)
    if change == "replace_output":
        values["positive"][:] = False
    elif change == "row_order":
        values["row_ids"] = values["row_ids"][::-1]
    elif change == "arbitration_outside_unknown":
        values["arbitration_positive"][1] = True
    elif change == "remaining_unknown":
        values["remaining_unknown"][2] = True
    elif change == "false_completion":
        completion.update(status="all_documents_resolved", task_complete=True)
    else:
        completion["state"] = {"positive": array_identity(np.zeros((4, 1), bool))}
    workflow.write(completion_path, completion)
    np.savez(path, **values)
    with pytest.raises(ValueError):
        workflow.evaluate_final(work, "main", work / "evaluation/main")


def offline_round_fixture(tmp_path, missing=False):
    work = quarantined_evaluation_fixture(tmp_path)
    path, values = final_arrays(work)
    np.savez(path, **values)
    run = work / "labeling/main"
    (run / "rounds").mkdir()
    completion = workflow.read(run / "COMPLETE.json")
    completion["round_history"] = []
    rows = np.array([], dtype=int) if missing else np.array([3, 1, 2])
    row_ids = np.array(["train0", "test0", "test1", "test2"])[rows]
    for r in range(2):
        logical = np.array([[False], [True], [False]]) if r == 0 else np.array([[True], [True], [False]])
        output = np.array([[True], [True], [False]]) if r == 0 else np.ones((3, 1), bool)
        decisions = np.full((3,), r > 0, bool)
        if missing:
            logical, output, decisions = logical[:0], output[:0], decisions[:0]
        np.savez(run / "rounds" / ("%05d.npz" % r), rows=rows, row_ids=row_ids, labels=np.array(["A"]),
                 logical=logical, output_positive=output, aux_raw=decisions[:, None], **{"0_1": decisions})
        completion["round_history"].append(dict(round=r, version=str(r), file="rounds/%05d.npz" % r,
            occurrence_labels={"0_1": 0}, decision_change=dict(fraction=float(r), changed=3*r, total=3)))
    workflow.write(run / "COMPLETE.json", completion)
    return work


def test_independent_truth_scores_rounds_and_keeps_logic_separate_from_output(tmp_path):
    work = offline_round_fixture(tmp_path)
    result = workflow.evaluate_final(work, "main", work / "evaluation/main", require_round_metrics=True)
    rounds = result["round_evaluation"]["rounds"]
    assert result["round_evaluation"]["status"] == "complete"
    assert [r["scored_documents"] for r in rounds] == [3, 3]
    assert [r["logical_metrics"]["macro_f1"] for r in rounds] == pytest.approx([0.5, 0.8])
    assert [r["output_metrics"]["macro_f1"] for r in rounds] == pytest.approx([0.8, 1.0])
    assert [r["dynamic_predicate_F1"]["0_1"]["macro_f1"] for r in rounds] == [0.0, 1.0]
    assert result["task_complete"] is False
    assert result["annotation_evaluation"]["human_equivalent_automatic_percent"] is None
    workflow.collect(work, work / "report")
    with (work / "report/ROUNDS.csv").open() as f:
        table = list(csv.DictReader(f))
    assert [float(r["output_macro_f1"]) for r in table] == pytest.approx([0.8, 1.0])
    assert [float(r["logical_macro_f1"]) for r in table] == pytest.approx([0.5, 0.8])
    assert (work / "report/DYNAMIC_PREDICATES.csv").is_file()


def test_missing_saved_rows_never_become_a_complete_RQ2_curve(tmp_path):
    work = offline_round_fixture(tmp_path, missing=True)
    result = workflow.evaluate_final(work, "main", work / "evaluation/main")
    assert result["round_evaluation"]["status"] == "incomplete_saved_rows"
    assert result["round_evaluation"]["rounds"][0]["scored_documents"] == 0
    with pytest.raises(ValueError, match="RQ2 requires complete"):
        workflow.evaluate_final(work, "main", work / "evaluation/strict", require_round_metrics=True)


def test_fixed_Gamma_accepts_valid_unresolved_quarantine_saturation(tmp_path):
    completed_fixed_gamma(tmp_path, "unknown_arbitrate")
    for variant in ("frozen", "main"):
        p = tmp_path / "labeling" / variant / "COMPLETE.json"
        data = workflow.read(p)
        data.update(status="conflict_unknown_saturated", task_complete=False, active_quarantine_pairs=1)
        workflow.write(p, data)
    assert script("check_fixed_gamma").compare(tmp_path, ["frozen", "main"])["same_stored_Gamma"]


def test_noinc_accepts_saturation_equality_without_completion_or_speedup_claim(tmp_path):
    noinc_pair(tmp_path)
    for variant in ("main", "noinc"):
        p = tmp_path / "effective" / ("labeling_" + variant + ".json")
        data = workflow.read(p); data["loop"]["conflict_policy"] = "unknown_arbitrate"; workflow.write(p, data)
        run = tmp_path / "labeling" / variant
        data = workflow.read(run / "COMPLETE.json")
        data.update(status="conflict_unknown_saturated", task_complete=False,
                    current_Gamma_has_valid_closure=False, current_Gamma_has_quarantined_fixpoint=True,
                    active_quarantine_pairs=1, effective_execution="full_recompute_quarantine")
        workflow.write(run / "COMPLETE.json", data)
        workflow.write(run / "MODEL_ARBITRATION_RULES.json", dict(test_fixture_only=True))
    result = script("check_noinc_equivalence").compare(tmp_path)
    assert not result["all_documents_resolved"] and result["chase_speedup"] is None


def test_repeated_saturated_outputs_are_explicit_endpoints_not_RQ3_success(tmp_path):
    works = repetitions(tmp_path, status="conflict_unknown_saturated", budget=0)
    for work in works:
        p = work / "evaluation/main/RESULTS.json"
        data = workflow.read(p)
        rules = work / "MODEL_ARBITRATION_RULES.json"
        workflow.write(rules, dict(test_fixture_only=True))
        data.update(current_Gamma_has_valid_closure=False, current_Gamma_has_quarantined_fixpoint=True,
                    active_quarantine_pairs=1, effective_execution="full_recompute_quarantine",
                    model_arbitration_rules=file_record(rules), status="evaluated")
        workflow.write(p, data)
    row = script("aggregate_repeats").aggregate(works, tmp_path / "summary")["rows"][0]
    assert row["statistics_scope"] == "quarantine_saturated_endpoint"
    assert row["complete_runs"] == 0 and not row["all_resolved_main_result_claimed"]


def test_plot_reader_accepts_workflow_round_names_and_excludes_incomplete_rows(tmp_path):
    path = tmp_path / "rounds.csv"
    path.write_text("variant,round,output_macro_f1,logical_macro_f1,scoring_status\nmain,0,0.8,0.5,complete\nmain,1,1,0.8,incomplete_saved_rows\n")
    module_path = Path(__file__).resolve().parents[1] / "scripts/plot_results.py"
    spec = importlib.util.spec_from_file_location("plot_results", module_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows = module.load_rows(path, "fixture", "output_macro_f1")
    assert len(rows) == 1 and rows[0]["logical_macro_f1"] == "0.5"


@pytest.mark.parametrize("change", [None, "on_logical_positive", "on_unknown", "missing_from_output"])
def test_final_evaluation_accepts_only_consistent_undecided_cell_fallback(tmp_path, change):
    work = quarantined_evaluation_fixture(tmp_path)
    path, values = final_arrays(work)
    values["logical_positive"][3] = False          # undecided cell, filled by the Aux fallback
    fallback = np.zeros_like(values["positive"])
    fallback[3] = True
    if change == "on_logical_positive":
        fallback[1] = True
    elif change == "on_unknown":
        fallback[2] = True
    elif change == "missing_from_output":
        values["positive"][3] = False
    values["fallback_positive"] = fallback
    np.savez(path, **values)
    if change is not None:
        with pytest.raises(ValueError):
            workflow.evaluate_final(work, "main", work / "evaluation/main")
        return
    result = workflow.evaluate_final(work, "main", work / "evaluation/main")
    accounting = result["fallback_accounting"]
    assert accounting["final_prediction_metrics_include_fallback"] is True
    assert accounting["filled_cells"] == 1
    assert result["final_array_validation"]["fallback_positive_cells"] == 1
    assert accounting["rule_output_metrics"] != result["metrics"]
