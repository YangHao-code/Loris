"""Guard causal RQ2 controls and the corrected default experiment weighting."""

import copy
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from loris.chase.state import GammaSeed
from loris._engine.rdl_reference import AttrEqual, Doc, Label, Rule
from loris.discovery.manifest import file_record
from loris._engine.typed_session import seed_identity
from loris.experiments import workflow


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts/experiments" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_fixture(tmp_path, monkeypatch):
    # Configuration-only fixture. This file is not a calibrated reliability claim.
    workflow.write(
        tmp_path / "DATA_AUDIT.json",
        dict(application_context=dict(status="verified_complete_declared_split_axis")),
    )
    workflow.write(tmp_path / "calibration/RELIABILITY.json", dict(test_fixture_only=True))
    workflow.write(tmp_path / "calibration-halt/RELIABILITY.json", dict(test_fixture_only=True))
    program = tmp_path / "SELECTED.json"
    workflow.write(program, dict(rules=[]))
    source = dict(
        strategy="paper",
        weighting="uniform",
        source_reliability=None,
        dynamic=dict(alias="aux"),
        feedback=dict(trusted_replay=True, candidate_check="nocheck"),
        loop=dict(
            label_budget=7,
            source_review_budget=2,
            query_batch_size=1,
            conflict_policy="review_instances",
        ),
        oracle=dict(
            kind="directory_llm_then_human",
            directory="same_answers",
            allow_known_source_review=True,
        ),
        query=dict(strategy="rdg", seed=42),
    )
    monkeypatch.setattr(workflow, "manifest", lambda *args: copy.deepcopy(source))
    monkeypatch.setattr(workflow, "program", lambda *args: program)
    return source


def test_frozen_preserves_declared_external_permissions_and_zero_budget_is_named(
    tmp_path, monkeypatch
):
    source = manifest_fixture(tmp_path, monkeypatch)
    frozen = workflow.label_manifest(tmp_path, "frozen")
    assert frozen["loop"]["label_budget"] == 7
    assert frozen["loop"]["source_review_budget"] == 2
    assert frozen["oracle"] == source["oracle"]
    zero = workflow.label_manifest(tmp_path, "frozen_zero_budget")
    assert zero["loop"]["label_budget"] == zero["loop"]["source_review_budget"] == 0
    assert zero["oracle"]["kind"] == "disabled"
    assert source["loop"]["label_budget"] == 7


def test_main_default_and_old_proof_heuristics_are_distinct_choices(tmp_path, monkeypatch):
    manifest_fixture(tmp_path, monkeypatch)
    main = workflow.label_manifest(tmp_path, "main")
    assert main["weighting"] == "direct_rule_precision_discount"
    assert main["feedback"]["disagreement_discount"] == 0.5
    assert main["feedback"]["receiver_policy"] == "same_model"
    assert main["loop"]["conflict_policy"] == "unknown_arbitrate"
    assert main["arbitration"] == dict(alias="aux", default_threshold=0.5, thresholds={})
    for variant in (
        "direct_rule_precision",
        "direct_rule_precision_discount",
        "complete_provenance",
        "complete_provenance_discount",
    ):
        result = workflow.label_manifest(tmp_path, variant)
        assert result["weighting"] == variant and result["source_reliability"] is not None
    assert workflow.label_manifest(tmp_path, "uniform")["source_reliability"] is None


def test_main_discount_coefficient_respects_explicit_configuration(tmp_path, monkeypatch):
    source = manifest_fixture(tmp_path, monkeypatch)
    source["feedback"]["disagreement_discount"] = 0.25
    data = workflow.label_manifest(tmp_path, "main")
    assert data["weighting"] == "direct_rule_precision_discount"
    assert data["feedback"]["disagreement_discount"] == 0.25


def test_main_cannot_silently_reuse_zero_query_protocol(tmp_path, monkeypatch):
    source = manifest_fixture(tmp_path, monkeypatch)
    source["loop"]["label_budget"] = None
    assert workflow.label_manifest(tmp_path, "main")["loop"]["label_budget"] is None
    source["loop"]["label_budget"] = 0
    with pytest.raises(ValueError, match="explicit fixed_gamma"):
        workflow.label_manifest(tmp_path, "main")
    assert workflow.label_manifest(tmp_path, "main", "fixed_gamma")["loop"]["label_budget"] == 0
    source["loop"]["label_budget"] = None
    source["oracle"]["kind"] = "disabled"
    with pytest.raises(ValueError, match="external answer provider"):
        workflow.label_manifest(tmp_path, "main")


def test_receiver_transfer_is_a_named_single_control_ablation(tmp_path, monkeypatch):
    source = manifest_fixture(tmp_path, monkeypatch)
    source["feedback"]["receiver_policy"] = "affected_channels_to_aux"
    main = workflow.label_manifest(tmp_path, "main", "fixed_gamma")
    transfer = workflow.label_manifest(tmp_path, "affected_channels_to_aux", "fixed_gamma")
    assert main["feedback"]["receiver_policy"] == "same_model"
    assert transfer["feedback"]["receiver_policy"] == "affected_channels_to_aux"
    transfer["feedback"]["receiver_policy"] = "same_model"
    assert transfer == main
    assert source["feedback"]["receiver_policy"] == "affected_channels_to_aux"


def test_fixed_gamma_intervention_applies_to_all_core_strategies(tmp_path, monkeypatch):
    manifest_fixture(tmp_path, monkeypatch)
    for variant in ("frozen", "supervised", "self_train", "main"):
        data = workflow.label_manifest(tmp_path, variant, "fixed_gamma")
        assert data["loop"]["label_budget"] == data["loop"]["source_review_budget"] == 0
        assert data["loop"]["conflict_policy"] == "unknown_arbitrate"
        assert data["oracle"]["kind"] == "disabled"
    with pytest.raises(ValueError, match="annotation-policy"):
        workflow.label_manifest(tmp_path, "random", "fixed_gamma")


def test_arbitration_thresholds_are_shared_with_frozen_and_halt_is_named(tmp_path, monkeypatch):
    source = manifest_fixture(tmp_path, monkeypatch)
    source["arbitration"] = dict(alias="aux", default_threshold=0.6, thresholds={"A": 0.7})
    for variant in ("main", "frozen", "frozen_zero_budget", "noinc", "nocheck"):
        data = workflow.label_manifest(tmp_path, variant, "fixed_gamma")
        assert data["arbitration"] == source["arbitration"]
        assert data["loop"]["conflict_policy"] == "unknown_arbitrate"
    halt = workflow.label_manifest(tmp_path, "halt_conflicts", "fixed_gamma")
    main = workflow.label_manifest(tmp_path, "main", "fixed_gamma")
    assert halt["source_reliability"]["path"].endswith("/calibration-halt/RELIABILITY.json")
    assert main["source_reliability"]["path"].endswith("/calibration/RELIABILITY.json")
    halt["loop"]["conflict_policy"] = "unknown_arbitrate"
    halt["source_reliability"] = main["source_reliability"]
    assert halt == main


def completed_fixed_gamma(tmp_path, conflict_policy="halt"):
    gamma = GammaSeed(np.array([[True], [False]]), np.zeros((2, 1), bool))
    context = tmp_path / "context.pkl"
    with context.open("wb") as f:
        pickle.dump(dict(gamma=gamma), f)
    checkpoint = tmp_path / "M0.pt"
    checkpoint.write_bytes(b"fixture checkpoint identity, not a trained model")
    program = tmp_path / "SELECTED.json"
    workflow.write(program, dict(rules=[]))
    for variant in ("frozen", "main"):
        app = tmp_path / "effective" / ("labeling_" + variant + ".json")
        workflow.write(
            app,
            dict(
                context=file_record(context),
                program=file_record(program),
                dynamic=dict(alias="aux", checkpoint=file_record(checkpoint)),
                arbitration=dict(alias="aux", default_threshold=0.5, thresholds={}),
                domains=dict(training_roles=["source_fit"]),
                strategy="frozen" if variant == "frozen" else "paper",
                weighting="uniform" if variant == "frozen" else "direct_rule_precision_discount",
                loop=dict(label_budget=0, source_review_budget=0, conflict_policy=conflict_policy),
                oracle=dict(kind="disabled"),
            ),
        )
        workflow.write(
            app.with_suffix(".experiment.json"),
            dict(supervision_protocol="fixed_gamma", effective_manifest=file_record(app)),
        )
        run = tmp_path / "labeling" / variant
        workflow.write(run / "EFFECTIVE_MANIFEST.json", json.loads(app.read_text()))
        if conflict_policy == "unknown_arbitrate":
            workflow.write(run / "MODEL_ARBITRATION_RULES.json", dict(test_fixture_only=True))
        workflow.write(
            run / "COMPLETE.json",
            dict(
                status="label_budget_exhausted",
                current_Gamma_has_valid_closure=conflict_policy == "halt",
                current_Gamma_has_quarantined_fixpoint=conflict_policy == "unknown_arbitrate",
                effective_execution=(
                    "full_recompute_quarantine"
                    if conflict_policy == "unknown_arbitrate"
                    else "incremental"
                ),
                active_quarantine_pairs=0,
                actual_label_requests=0,
                actual_source_reviews=0,
                actual_optimizer_updates=0,
                committed_model_updates=0,
                checkpoint=file_record(checkpoint),
                Gamma=seed_identity(gamma),
                version="fixed_fixture_omega_kappa_version",
                round_history=[dict(version="fixed_fixture_omega_kappa_version")],
            ),
        )
        workflow.write(run / "ROOTS_INITIAL.json", seed_identity(gamma))
        np.savez(run / "TRUSTED_GAMMA_FINAL.npz", positive=gamma.pos, negative=gamma.neg)
    return gamma


def test_fixed_gamma_certificate_checks_actual_final_roots(tmp_path):
    gamma = completed_fixed_gamma(tmp_path)
    changed = gamma.pos.copy()
    changed[1, 0] = True
    np.savez(
        tmp_path / "labeling/main/TRUSTED_GAMMA_FINAL.npz", positive=changed, negative=gamma.neg
    )
    with pytest.raises(ValueError, match="actual final Gamma changed"):
        script("check_fixed_gamma").compare(tmp_path, ["frozen", "main"])
    assert not (tmp_path / "FIXED_GAMMA_COMPARISON.json").exists()


def test_fixed_gamma_certificate_records_verified_outputs(tmp_path):
    completed_fixed_gamma(tmp_path)
    result = script("check_fixed_gamma").compare(tmp_path, ["frozen", "main"])
    assert result["same_initial_and_final_Gamma"] and result["no_new_external_supervision"]
    assert len(result["variants"]) == 2
    assert all(item["final_Gamma"]["sha256"] for item in result["variants"])


def test_fixed_gamma_certifies_stored_roots_with_unknown_arbitration(tmp_path):
    completed_fixed_gamma(tmp_path, "unknown_arbitrate")
    result = script("check_fixed_gamma").compare(tmp_path, ["frozen", "main"])
    assert result["same_stored_Gamma"]
    assert not result["same_effective_quarantine_view_claimed"]
    assert all(record["arbitration_rules"]["sha256"] for record in result["variants"])


def test_fixed_gamma_rejects_changed_actual_arbitration_thresholds(tmp_path):
    completed_fixed_gamma(tmp_path, "unknown_arbitrate")
    path = tmp_path / "labeling/frozen/EFFECTIVE_MANIFEST.json"
    data = json.loads(path.read_text())
    data["arbitration"]["default_threshold"] = 0.9
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="arbitration configuration"):
        script("check_fixed_gamma").compare(tmp_path, ["frozen", "main"])


def test_fixed_gamma_certificate_rejects_changed_actual_run_permissions(tmp_path):
    completed_fixed_gamma(tmp_path)
    actual = tmp_path / "labeling/main/EFFECTIVE_MANIFEST.json"
    data = json.loads(actual.read_text())
    data["loop"]["label_budget"] = 7
    actual.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="supervision intervention"):
        script("check_fixed_gamma").compare(tmp_path, ["frozen", "main"])


def test_fixed_gamma_certificate_rejects_frozen_threshold_version_change(tmp_path):
    completed_fixed_gamma(tmp_path)
    path = tmp_path / "labeling/frozen/COMPLETE.json"
    data = json.loads(path.read_text())
    data["version"] = "changed_kappa_with_same_checkpoint"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="threshold version"):
        script("check_fixed_gamma").compare(tmp_path, ["frozen", "main"])


def test_fixed_gamma_certificate_rejects_relation_root_change(tmp_path):
    completed_fixed_gamma(tmp_path)
    path = tmp_path / "labeling/main/COMPLETE.json"
    data = json.loads(path.read_text())
    data["Gamma"]["subset"] = [[0, 1]]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="including relation roots"):
        script("check_fixed_gamma").compare(tmp_path, ["frozen", "main"])


def test_repeat_aggregation_does_not_mix_weighting_or_supervision_protocol(tmp_path):
    workspaces = []
    for i, protocol in enumerate(("fixed_gamma", "as_declared")):
        work = tmp_path / str(i)
        workflow.write(work / "ASSETS.json", {})
        workflow.write(
            work / "evaluation/main/RESULTS.json",
            dict(
                variant="main",
                evaluation_role="dev",
                labels=["A"],
                evaluation_manifest=dict(sha256="same_axis"),
                run_status="label_budget_exhausted",
                task_complete=False,
                current_Gamma_has_valid_closure=True,
                declared_label_budget=0,
                weighting="direct_rule_precision_discount",
                supervision_protocol=protocol,
                metrics=dict(macro_f1=0.5, micro_f1=0.5, coverage=1, multi_label_completeness=None),
            ),
        )
        workspaces.append(work)
    with pytest.raises(ValueError, match="missing or differs"):
        script("aggregate_repeats").aggregate(workspaces, tmp_path / "report")


def test_historical_bgc_is_auditable_but_current_rule_form_blocks_execution(tmp_path):
    from loris.experiments.registry import (
        check_complete_program,
        check_execution_program,
        check_rules,
    )

    # Keep this regression self-contained: the historical registry is not part
    # of the clean submission package. The generated program has one legacy
    # cross-document single-label rule at the same historical position (165).
    valid = Rule((Doc("x"),), Label("x", "A")).to_dict()
    invalid = Rule((Doc("x"), Doc("y"), AttrEqual("x", "cnt", "y", "cnt")),
                   Label("y", "A")).to_dict()
    program = tmp_path / "bgc_SELECTED.json"
    program.write_text(json.dumps({
        "rules": [valid] * 165 + [invalid] + [valid] * 11,
        "all_candidates_sequentially_evaluated": True,
        "complete_BO_trials": 1,
        "clusters": [{"cluster": 0, "complete_BO_trials": 1}],
        "evaluation": {"status": "fixpoint"},
    }))
    from loris.discovery.manifest import file_record
    completion = tmp_path / "bgc_COMPLETE.json"
    completion.write_text(json.dumps({
        "status": "complete", "selected_rules": 177, "complete_BO_trials": 1,
        "selection": {"sha256": file_record(program)["sha256"]},
    }))
    index = tmp_path / "INDEX.json"
    index.write_text(json.dumps({"entries": [{
        "dataset": "bgc", "rules": 177,
        "local_program": {"path": program.name, "sha256": file_record(program)["sha256"]},
    }]}))
    historical = check_complete_program(
        program, completion
    )
    assert historical["rules"] == 177
    assert historical["current_rule_form_audit"]["invalid_count"] == 1
    assert historical["current_rule_form_audit"]["violations"][0]["rule_index"] == 165
    with pytest.raises(ValueError, match="165"):
        check_execution_program(program, completion)
    audit = check_rules(index)
    assert sum(row["current_rule_form_audit"]["invalid_count"] for row in audit["entries"]) == 1
    valid_program = tmp_path / "valid_SELECTED.json"
    valid_program.write_text(json.dumps({
        "rules": [valid], "all_candidates_sequentially_evaluated": True,
        "complete_BO_trials": 1, "clusters": [{"cluster": 0, "complete_BO_trials": 1}],
        "evaluation": {"status": "fixpoint"},
    }))
    valid_completion = tmp_path / "valid_COMPLETE.json"
    valid_completion.write_text(json.dumps({
        "status": "complete", "selected_rules": 1, "complete_BO_trials": 1,
        "selection": {"sha256": file_record(valid_program)["sha256"]},
    }))
    result = check_execution_program(valid_program, valid_completion)
    assert result["current_rule_form_audit"]["invalid_count"] == 0
