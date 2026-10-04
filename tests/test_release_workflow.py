import copy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from loris.discovery.manifest import file_record
from loris.experiments.registry import check_complete_program
from loris.experiments.workflow import (
    audit_dataset,
    bind_assets,
    require_audit,
    write,
    evaluate_final,
)
from loris._engine.rdl_reference import Doc, ML, Label, Rule, RefGamma


def complete_program(tmp_path):
    program = tmp_path / "SELECTED.json"
    data = dict(
        rules=[Rule((Doc("x"), ML("x", "m", "A", 0.5, True)), Label("x", "A")).to_dict()],
        all_candidates_sequentially_evaluated=True,
        complete_BO_trials=2,
        clusters=[dict(cluster="0", complete_BO_trials=2)],
        evaluation=dict(status="fixpoint"),
    )
    write(program, data)
    receipt = tmp_path / "COMPLETE.json"
    write(
        receipt,
        dict(
            status="complete",
            selected_rules=1,
            complete_BO_trials=2,
            selection=file_record(program),
        ),
    )
    return program, receipt


def dataset(tmp_path, duplicate=False):
    splits = {}
    for i, role in enumerate(("train", "dev", "test")):
        doc = tmp_path / (role + ".jsonl")
        write(doc, dict(row_id=role + "0", text="same" if duplicate else role))
        # jsonl is exactly one JSON object per line.
        doc.write_text(
            json.dumps(dict(row_id=role + "0", text="same" if duplicate else role)) + "\n"
        )
        truth = tmp_path / (role + ".npz")
        np.savez(
            truth,
            row_ids=np.array([role + "0"]),
            labels=np.array(["A"]),
            targets=np.ones((1, 1), bool),
        )
        splits[role] = dict(documents=file_record(doc), truth=file_record(truth))
    path = tmp_path / "data.json"
    write(
        path,
        dict(
            schema="loris_release_dataset_audit_v1",
            dataset="toy",
            scope="complete fixture",
            source_provenance="test fixture",
            labels=["A"],
            expected_counts=dict(train=1, dev=1, test=1),
            duplicate_policy="require_disjoint",
            splits=splits,
        ),
    )
    return path


def test_complete_cache_rejects_partial_and_wrong_receipt(tmp_path):
    program, receipt = complete_program(tmp_path)
    assert check_complete_program(program, receipt)["rules"] == 1
    original = json.loads(program.read_text())
    value = copy.deepcopy(original)
    value["clusters"][0]["complete_BO_trials"] = 1
    write(program, value)
    with pytest.raises(ValueError, match="cluster BO counts"):
        check_complete_program(program, receipt)
    write(program, original)
    value = json.loads(receipt.read_text())
    value["selection"]["sha256"] = "0" * 64
    write(receipt, value)
    with pytest.raises(ValueError, match="another ordered program"):
        check_complete_program(program, receipt)


def test_audit_detects_content_overlap_even_unique_ids(tmp_path):
    path = dataset(tmp_path, True)
    with pytest.raises(ValueError, match="duplicate text"):
        audit_dataset(path)
    d = json.loads(path.read_text())
    d["duplicate_policy"] = "report_official"
    write(path, d)
    result = audit_dataset(path)
    assert result["exact_text_overlaps"] == {"train/dev": 1, "train/test": 1, "dev/test": 1}


def test_audit_rejects_misaligned_label_ids(tmp_path):
    path = dataset(tmp_path)
    np.savez(
        tmp_path / "test.npz",
        row_ids=np.array(["other"]),
        labels=np.array(["A"]),
        targets=np.ones((1, 1), bool),
    )
    data = json.loads(path.read_text())
    data["splits"]["test"]["truth"] = file_record(tmp_path / "test.npz")
    write(path, data)
    with pytest.raises(ValueError, match="truth document IDs"):
        audit_dataset(path)


def test_bound_workspace_rechecks_nested_data_tampering(tmp_path):
    data = dataset(tmp_path)
    program, receipt = complete_program(tmp_path)
    source = tmp_path / "assets.json"
    write(
        source,
        dict(
            schema="loris_release_assets_v1",
            dataset="toy",
            scope="extra_dataset",
            rules_mode="reuse",
            notes="test",
            artifacts=dict(
                dataset_audit={"path": "${ASSET_ROOT}/data.json"},
                cached_program=file_record(program),
                cached_completion=file_record(receipt),
            ),
        ),
    )
    work = tmp_path / "work"
    bind_assets(source, work, asset_root=tmp_path)
    audit_dataset(data, work / "DATA_AUDIT.json")
    require_audit(work)
    (tmp_path / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="identity"):
        require_audit(work)


def test_final_evaluation_uses_final_facts_and_marks_resource_cap(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    context = tmp_path / "context.pkl"
    with context.open("wb") as f:
        pickle.dump(
            dict(
                row_ids=["train0", "test0"],
                gamma=RefGamma.empty(2, 1),
                row_roles=["source_fit", "test"],
                labels=["A"],
                fit_indices=[0],
            ),
            f,
        )
    truth = tmp_path / "truth.npz"
    np.savez(
        truth, row_ids=np.array(["test0"]), labels=np.array(["A"]), targets=np.ones((1, 1), bool)
    )
    evaluation = tmp_path / "evaluation.json"
    write(
        evaluation, dict(schema="loris_final_evaluation_v1", role="test", truth=file_record(truth))
    )
    write(
        work / "ASSETS.json",
        dict(
            artifacts=dict(evaluation=file_record(evaluation)),
            asset_root=str(tmp_path),
            repo_root=str(tmp_path),
        ),
    )
    app = work / "effective/labeling_main.json"
    write(app, dict(context=file_record(context), domains=dict(training_roles=["source_fit"])))
    run = work / "labeling/main"
    run.mkdir(parents=True)
    write(run / "INPUT_RECEIPT.json", dict(manifest=file_record(app)))
    write(
        run / "COMPLETE.json",
        dict(
            status="resource_limit",
            current_Gamma_has_valid_closure=True,
            task_complete=False,
            acquired_documents=[],
        ),
    )
    np.savez(run / "FINAL_P.npz", positive=np.ones((2, 1), bool), negative=np.zeros((2, 1), bool))
    result = evaluate_final(work, "main", work / "evaluation/main")
    assert result["metrics"]["macro_f1"] == 1
    assert result["task_complete"] is False
    assert result["run_status"] == "resource_limit"


def test_test_evaluation_rejects_directly_queried_rows(tmp_path):
    # Reuse a valid setup, then check explicit query-induced scope switching.
    test_final_evaluation_uses_final_facts_and_marks_resource_cap(tmp_path)
    work = tmp_path / "work"
    run = work / "labeling/main"
    completion = json.loads((run / "COMPLETE.json").read_text())
    completion["acquired_documents"] = [1]
    write(run / "COMPLETE.json", completion)
    with pytest.raises(ValueError, match="interactive_test"):
        evaluate_final(work, "main", work / "evaluation/queried")


def test_missing_text_requires_explicit_protocol_and_reports_count(tmp_path):
    path = dataset(tmp_path)
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(dict(row_id="train0", text="   ")) + "\n")
    data = json.loads(path.read_text())
    data["splits"]["train"]["documents"] = file_record(source)
    write(path, data)
    with pytest.raises(ValueError, match="empty/blank document texts"):
        audit_dataset(path)
    data["text_policy"] = dict(
        empty_text="report_missing",
        missing_text_reason="Official source explicitly preserves a missing-body record.",
    )
    write(path, data)
    result = audit_dataset(path)
    assert result["splits"]["train"]["empty_text_rows"] == 1
    assert result["full_original_text_present"] is False


def test_feature_id_vectors_are_not_ordered_natural_language(tmp_path):
    path = dataset(tmp_path)
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(dict(row_id="train0", text="w12 w314 w5")) + "\n")
    data = json.loads(path.read_text())
    data["splits"]["train"]["documents"] = file_record(source)
    write(path, data)
    with pytest.raises(ValueError, match="feature-ID documents"):
        audit_dataset(path)


def test_observed_mask_must_cover_actual_label_table(tmp_path):
    path = dataset(tmp_path)
    source = tmp_path / "train.npz"
    np.savez(
        source,
        row_ids=np.array(["train0"]),
        labels=np.array(["A"]),
        targets=np.ones((1, 1), bool),
        observed=np.ones((1,), bool),
    )
    data = json.loads(path.read_text())
    data["splits"]["train"]["truth"] = file_record(source)
    write(path, data)
    with pytest.raises(ValueError, match="complete label table"):
        audit_dataset(path)


def test_label_axis_cannot_be_a_string(tmp_path):
    path = dataset(tmp_path)
    data = json.loads(path.read_text())
    data["labels"] = "A"
    write(path, data)
    with pytest.raises(ValueError, match="complete unique label axis"):
        audit_dataset(path)


def test_ordered_subset_ablation_requires_complete_parent_and_exact_positions(tmp_path):
    from loris.experiments.registry import check_execution_program

    parent, completion = complete_program(tmp_path)
    selected = tmp_path / "subset.json"
    write(selected, dict(rules=json.loads(parent.read_text())["rules"]))
    receipt = tmp_path / "subset_receipt.json"
    write(
        receipt,
        dict(
            schema="loris_nested_program_subset_v1",
            status="complete",
            selection=file_record(selected),
            parent_program=file_record(parent),
            parent_completion=file_record(completion),
            parent_indices=[0],
            scope="fixed_program_subset_ablation",
        ),
    )
    assert check_execution_program(selected, receipt)["new_discovery_result_claimed"] is False
    data = json.loads(selected.read_text())
    data["rules"][0]["head"]["label"] = "B"
    write(selected, data)
    d = json.loads(receipt.read_text())
    d["selection"] = file_record(selected)
    write(receipt, d)
    with pytest.raises(ValueError, match="certified complete parent positions"):
        check_execution_program(selected, receipt)


def test_application_audit_allows_order_changes_but_rejects_hidden_subsets(tmp_path):
    from loris.document import Document
    from loris.experiments.workflow import audit_application

    source = dataset(tmp_path)
    data = json.loads(source.read_text())
    data["application_splits"] = ["train", "test"]
    write(source, data)
    context = tmp_path / "context.pkl"
    value = dict(
        row_ids=["test0", "train0"],
        gamma=RefGamma.empty(2, 1),
        labels=["A"],
        docs=[Document(cnt="test"), Document(cnt="train")],
    )
    with context.open("wb") as f:
        pickle.dump(value, f)
    app = tmp_path / "app.json"
    write(app, dict(context=file_record(context)))
    work = tmp_path / "work"
    write(
        work / "ASSETS.json",
        dict(
            asset_root=str(tmp_path),
            repo_root=str(tmp_path),
            artifacts=dict(dataset_audit=file_record(source), labeling_manifest=file_record(app)),
        ),
    )
    result = audit_application(work, audit_dataset(source))
    assert result["application_context"]["rows"] == 2
    assert result["application_context"]["all_texts_match_by_id"]
    value["row_ids"] = value["row_ids"][:1]
    value["docs"] = value["docs"][:1]
    with context.open("wb") as f:
        pickle.dump(value, f)
    write(app, dict(context=file_record(context)))
    assets = json.loads((work / "ASSETS.json").read_text())
    assets["artifacts"]["labeling_manifest"] = file_record(app)
    write(work / "ASSETS.json", assets)
    with pytest.raises(ValueError, match="exactly all declared split"):
        audit_application(work, audit_dataset(source))


def annotation_evaluation_fixture(tmp_path, status="all_documents_resolved"):
    work = tmp_path / "work"
    work.mkdir()
    context = tmp_path / "context.pkl"
    with context.open("wb") as f:
        pickle.dump(
            dict(
                row_ids=["train0", "test0", "test1", "test2"],
                gamma=RefGamma.empty(4, 1),
                row_roles=["source_fit", "test", "test", "test"],
                labels=["A"],
                fit_indices=[0],
            ),
            f,
        )
    truth = tmp_path / "truth.npz"
    np.savez(
        truth,
        row_ids=np.array(["test0", "test1", "test2"]),
        labels=np.array(["A"]),
        targets=np.ones((3, 1), bool),
    )
    evaluation = tmp_path / "evaluation.json"
    write(
        evaluation,
        dict(schema="loris_final_evaluation_v1", role="interactive_test", truth=file_record(truth)),
    )
    write(
        work / "ASSETS.json",
        dict(
            artifacts=dict(evaluation=file_record(evaluation)),
            asset_root=str(tmp_path),
            repo_root=str(tmp_path),
        ),
    )
    app = work / "effective/labeling_main.json"
    write(
        app,
        dict(
            context=file_record(context),
            domains=dict(training_roles=["source_fit"]),
            loop=dict(label_budget=10),
        ),
    )
    run = work / "labeling/main"
    run.mkdir(parents=True)
    write(run / "INPUT_RECEIPT.json", dict(manifest=file_record(app)))
    write(
        run / "COMPLETE.json",
        dict(
            status=status,
            current_Gamma_has_valid_closure=True,
            task_complete=status == "all_documents_resolved",
            acquired_documents=[1, 2, 3],
            actual_label_requests=8,
            annotation_documents=dict(human=[0, 1], simulated_human=[2], llm=[3], simulated_llm=[]),
            annotation_sources=dict(
                human_answers=3,
                simulated_human_answers=4,
                llm_proposals=9,
                simulated_llm_proposals=2,
                llm_accepted=1,
            ),
            inference_counts=dict(full=1, chaseInc=3, chaseDelta=2, calibration=2),
            timings=dict(chase_seconds=2.0, calibration_seconds=7.5),
            rule_reliability_update="current_model",
        ),
    )
    np.savez(run / "FINAL_P.npz", positive=np.ones((4, 1), bool), negative=np.zeros((4, 1), bool))
    return work


def test_human_cost_uses_accepted_evaluation_rows_not_provider_calls(tmp_path):
    from loris.experiments.workflow import collect

    work = annotation_evaluation_fixture(tmp_path)
    result = evaluate_final(work, "main", work / "evaluation/main")
    a = result["annotation_evaluation"]
    assert a["accepted_documents"] == dict(human=1, simulated_human=1, llm=1, simulated_llm=0)
    assert a["human_equivalent_unique_documents"] == 2 and a["human_source_mode"] == "mixed"
    assert a["human_equivalent_automatic_percent"] == pytest.approx(100.0 / 3)
    collect(work, work / "report")
    row = json.loads((work / "report/SUMMARY.json").read_text())["rows"][0]
    assert (
        row["provider_calls_human_answers"] == 3 and row["evaluation_accepted_human_documents"] == 1
    )
    assert row["calls_calibration"] == 2 and row["calibration_seconds"] == 7.5


def quarantined_evaluation_fixture(tmp_path):
    work = annotation_evaluation_fixture(tmp_path)
    app = work / "effective/labeling_main.json"
    data = json.loads(app.read_text())
    data["loop"]["conflict_policy"] = "unknown_arbitrate"
    write(app, data)
    run = work / "labeling/main"
    write(run / "INPUT_RECEIPT.json", dict(manifest=file_record(app)))
    complete = json.loads((run / "COMPLETE.json").read_text())
    complete.update(
        status="conflict_unknown_saturated",
        task_complete=False,
        current_Gamma_has_valid_closure=False,
        current_Gamma_has_quarantined_fixpoint=True,
        active_quarantine_pairs=1,
        effective_execution="full_recompute_quarantine",
        full_recompute_reason="nonmonotone_conflict_quarantine_and_temporary_model_seeds",
    )
    write(run / "COMPLETE.json", complete)
    write(run / "MODEL_ARBITRATION_RULES.json", dict(test_fixture_only=True))
    pos = np.ones((4, 1), bool)
    zero = np.zeros_like(pos)
    logical = pos.copy()
    logical[2] = False
    conflict = zero.copy()
    conflict[2] = True
    np.savez(
        run / "FINAL_P.npz",
        positive=pos,
        negative=zero,
        logical_positive=logical,
        logical_negative=zero,
        conflict_unknown=conflict,
        remaining_unknown=zero,
        arbitration_positive=conflict,
        arbitration_negative=zero,
    )
    return work


def test_quarantined_final_state_evaluation_includes_arbitration_and_reports_unknown_masks(
    tmp_path,
):
    from loris.experiments.workflow import collect

    work = quarantined_evaluation_fixture(tmp_path)
    result = evaluate_final(work, "main", work / "evaluation/main")
    assert result["current_Gamma_has_valid_closure"] is False
    assert result["current_Gamma_has_quarantined_fixpoint"] is True
    assert result["final_state_kind"] == "quarantined_fixpoint"
    assert result["metrics"]["macro_f1"] == 1
    assert result["conflict_accounting"]["final_logical_metrics"]["macro_f1"] == pytest.approx(0.8)
    assert result["conflict_accounting"]["counts"]["conflict_unknown"] == 1
    collect(work, work / "report")
    row = json.loads((work / "report/SUMMARY.json").read_text())["rows"][0]
    assert row["current_Gamma_has_valid_closure"] is False
    assert row["evaluation_arbitration_positive_cells"] == 1
    assert row["effective_execution"] == "full_recompute_quarantine"


def test_quarantined_evaluation_requires_explicit_arbitration_rule_evidence(tmp_path):
    work = quarantined_evaluation_fixture(tmp_path)
    (work / "labeling/main/MODEL_ARBITRATION_RULES.json").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_final(work, "main", work / "evaluation/main")


def test_budget_endpoint_with_annotation_evidence_cannot_claim_automatic_resolution_percentage(
    tmp_path,
):
    work = annotation_evaluation_fixture(tmp_path, "label_budget_exhausted")
    result = evaluate_final(work, "main", work / "evaluation/main")
    assert result["annotation_evaluation"]["human_equivalent_unique_documents"] == 2
    assert result["annotation_evaluation"]["human_equivalent_automatic_percent"] is None


def test_old_completion_without_source_rows_keeps_human_metrics_unknown(tmp_path):
    work = annotation_evaluation_fixture(tmp_path)
    completion = json.loads((work / "labeling/main/COMPLETE.json").read_text())
    del completion["annotation_documents"]
    write(work / "labeling/main/COMPLETE.json", completion)
    result = evaluate_final(work, "main", work / "evaluation/main")
    a = result["annotation_evaluation"]
    assert a["human_equivalent_unique_documents"] is None
    assert a["human_equivalent_automatic_percent"] is None
    assert a["status"] == "unknown_missing_document_source_evidence"


def enable_online_target_training(work):
    app = work / "effective/labeling_main.json"
    data = json.loads(app.read_text())
    data["domains"]["training_roles"].append("test")
    write(app, data)
    write(work / "labeling/main/INPUT_RECEIPT.json", dict(manifest=file_record(app)))


def test_interactive_evaluation_allows_declared_online_feedback_and_reports_actual_trace(tmp_path):
    work = annotation_evaluation_fixture(tmp_path)
    enable_online_target_training(work)
    run = work / "labeling/main"
    complete = json.loads((run / "COMPLETE.json").read_text())
    complete.update(training_attempts=1, actual_optimizer_updates=2, committed_model_updates=1)
    write(run / "COMPLETE.json", complete)
    write(run / "update_00001/ACTUAL_TRAINING_TRACE.json", dict(sampled_rows=[0, 1, 1, 3]))
    result = evaluate_final(work, "main", work / "evaluation/main")
    adapt = result["online_adaptation"]
    assert adapt["declared_online_evaluation_documents"] == 3
    assert adapt["actual_sampled_evaluation_documents"] == 2
    assert adapt["actual_optimizer_updates"] == 2 and adapt["committed_model_updates"] == 1
    assert adapt["training_trace_complete"]


def test_pure_test_rejects_target_online_adaptation_permission(tmp_path):
    work = annotation_evaluation_fixture(tmp_path)
    enable_online_target_training(work)
    source = tmp_path / "evaluation.json"
    protocol = json.loads(source.read_text())
    protocol["role"] = "test"
    write(source, protocol)
    assets = json.loads((work / "ASSETS.json").read_text())
    assets["artifacts"]["evaluation"] = file_record(source)
    write(work / "ASSETS.json", assets)
    with pytest.raises(ValueError, match="online adaptation requires interactive_test"):
        evaluate_final(work, "main", work / "evaluation/main")


def test_interactive_target_still_rejects_initial_supervised_fit_contamination(tmp_path):
    work = annotation_evaluation_fixture(tmp_path)
    enable_online_target_training(work)
    source = tmp_path / "context.pkl"
    with source.open("rb") as f:
        context = pickle.load(f)
    context["fit_indices"] = [0, 1]
    with source.open("wb") as f:
        pickle.dump(context, f)
    app = work / "effective/labeling_main.json"
    data = json.loads(app.read_text())
    data["context"] = file_record(source)
    write(app, data)
    write(work / "labeling/main/INPUT_RECEIPT.json", dict(manifest=file_record(app)))
    with pytest.raises(ValueError, match="initial supervised model-fit"):
        evaluate_final(work, "main", work / "evaluation/main")
