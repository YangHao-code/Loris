"""Real calibration and runtime loading for all three conflict protocols."""

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from loris.discovery.manifest import file_record
from loris.experiments import workflow
from loris.labeling import cli
from loris.labeling.reliability import (
    QUARANTINE_CHASE_SEMANTICS,
    RETAINED_CHASE_SEMANTICS,
    STRICT_CHASE_SEMANTICS,
)
from tests.test_fit_reliability import calibration_fixture


def workflow_fixture(tmp_path, *, calibration_conflict=False):
    """Bound four-row synthetic data; real Chase, no training or empirical claim."""
    assets = tmp_path / "assets"
    assets.mkdir()
    args, _, protocol = calibration_fixture(assets)
    manifest = workflow.read(args.manifest)
    program_path = Path(manifest["program"]["path"])
    rules = workflow.read(program_path)
    if calibration_conflict:
        from loris._engine.rdl_reference import Doc, Label, Rule, Text
        from loris.predicates import MatchPredicate

        rules.append(
            Rule(
                (
                    Doc("x"),
                    Text("x", MatchPredicate("cnt", "red")),
                    Text("x", MatchPredicate("cnt", "independent")),
                ),
                Label("x", "A", "neg"),
            ).to_dict()
        )
    workflow.write(
        program_path,
        dict(
            rules=rules,
            all_candidates_sequentially_evaluated=True,
            complete_BO_trials=2,
            clusters=[dict(cluster="fixture", complete_BO_trials=2)],
            evaluation=dict(status="fixpoint"),
            notes="Synthetic completed-program input fixture; no BO execution claim.",
        ),
    )
    completion = assets / "program_completion.json"
    workflow.write(
        completion,
        dict(
            status="complete",
            selected_rules=len(rules),
            complete_BO_trials=2,
            selection=file_record(program_path),
        ),
    )
    manifest["program"] = file_record(program_path)
    workflow.write(args.manifest, manifest)
    # An explicitly marked source plan must also produce an independently run
    # strict plan; rewriting an already-estimated artifact would be invalid.
    protocol["model_binding"]["chase_semantics"] = QUARANTINE_CHASE_SEMANTICS
    workflow.write(args.protocol, protocol)
    with Path(manifest["context"]["path"]).open("rb") as stream:
        context = pickle.load(stream)
    splits = {}
    for role, rows in (("train", [0, 1]), ("dev", [2]), ("test", [3])):
        documents = assets / (role + ".jsonl")
        documents.write_text(
            "".join(
                json.dumps(dict(row_id=context["row_ids"][r], text=context["docs"][r].cnt)) + "\n"
                for r in rows
            )
        )
        truth = assets / (role + ".npz")
        np.savez(
            truth,
            row_ids=np.array([context["row_ids"][r] for r in rows]),
            labels=np.array(context["labels"]),
            targets=context["gamma"].pos[rows],
        )
        splits[role] = dict(documents=file_record(documents), truth=file_record(truth))
    dataset = assets / "dataset.json"
    workflow.write(
        dataset,
        dict(
            schema="loris_release_dataset_audit_v1",
            dataset="synthetic_four_rows",
            scope="complete synthetic fixture",
            source_provenance="test fixture, not a dataset result",
            labels=context["labels"],
            expected_counts=dict(train=2, dev=1, test=1),
            application_splits=["train", "dev", "test"],
            duplicate_policy="report_official",
            splits=splits,
        ),
    )
    asset_map = assets / "assets.json"
    workflow.write(
        asset_map,
        dict(
            schema=workflow.SCHEMA,
            dataset="synthetic_four_rows",
            scope="extra_dataset",
            rules_mode="reuse",
            notes="Synthetic integration inputs; actual calibration, no training.",
            artifacts=dict(
                dataset_audit=file_record(dataset),
                labeling_manifest=file_record(args.manifest),
                reliability_protocol=file_record(args.protocol),
                cached_program=file_record(program_path),
                cached_completion=file_record(completion),
            ),
        ),
    )
    work = tmp_path / "work"
    workflow.bind_assets(asset_map, work)
    assert workflow.main(["audit", "--workspace", str(work)])["status"] == "verified"
    return work, Path(args.protocol)


def exercise_both_calibrations(tmp_path):
    work, source_protocol = workflow_fixture(tmp_path)
    original_protocol = source_protocol.read_bytes()
    dry = workflow.main(
        ["calibrate", "--workspace", str(work), "--conflict-policy", "halt", "--dry-run"]
    )
    assert dry["status"] == "dry_run"
    assert dry["command"][-1] == str(work / "calibration-halt")
    assert not (work / "effective").exists()

    assert workflow.main(["calibrate", "--workspace", str(work)])["status"] == "complete"
    main_artifact = file_record(work / "calibration/RELIABILITY.json")
    # A completed main calibration is never an implicit strict fallback.
    with pytest.raises(FileNotFoundError):
        workflow.label_manifest(work, "halt_conflicts")
    assert (
        workflow.main(["calibrate", "--workspace", str(work), "--conflict-policy", "halt"])[
            "status"
        ]
        == "complete"
    )
    assert file_record(work / "calibration/RELIABILITY.json") == main_artifact
    with pytest.raises(FileNotFoundError):
        workflow.label_manifest(work, "unknown_full_rebuild")
    assert (
        workflow.main(
            ["calibrate", "--workspace", str(work), "--unknown-semantics", "full_rebuild"]
        )["status"]
        == "complete"
    )
    assert file_record(work / "calibration/RELIABILITY.json") == main_artifact
    assert source_protocol.read_bytes() == original_protocol

    effective = []
    results = []
    for variant, directory, semantics, inference in (
        ("main", "calibration", RETAINED_CHASE_SEMANTICS, "retention_without_arbitration"),
        (
            "unknown_full_rebuild",
            "calibration-full-rebuild",
            QUARANTINE_CHASE_SEMANTICS,
            "quarantine_without_arbitration",
        ),
        ("halt_conflicts", "calibration-halt", STRICT_CHASE_SEMANTICS, "strict_chase"),
    ):
        manifest = workflow.label_manifest(work, variant, "fixed_gamma")
        path = work / "effective" / ("labeling_" + variant + ".json")
        workflow.write(path, manifest)
        loaded, _, _, artifact, _ = cli.load(path)
        assert loaded["source_reliability"] == file_record(work / directory / "RELIABILITY.json")
        assert artifact["chase_semantics"] == semantics
        assert [(r["support"], r["correct"], r["q"]) for r in artifact["rule_scores"]] == [
            (0, 0, 0.5),
            (18, 17, 0.9),
        ]
        receipt = workflow.read(work / directory / "RESULTS.json")
        assert receipt["calibration_inference"] == inference
        assert receipt["answers_entered_inference"] is False
        assert receipt["model_updates"] == receipt["rule_learning_calls"] == 0
        effective.append(workflow.read(work / directory / "PROTOCOL.json"))
        results.append(
            dict(
                variant=variant,
                source_reliability=loaded["source_reliability"],
                chase_semantics=semantics,
                calibration_inference=inference,
                support=[r["support"] for r in artifact["rule_scores"]],
                correct=[r["correct"] for r in artifact["rule_scores"]],
                q=[r["q"] for r in artifact["rule_scores"]],
            )
        )
        for other in ("calibration", "calibration-full-rebuild", "calibration-halt"):
            if other != directory:
                with pytest.raises(ValueError, match="chase_semantics mismatch"):
                    cli.load(path, source_reliability=work / other / "RELIABILITY.json")
    for p in effective:
        p["model_binding"].pop("chase_semantics")
    assert effective[0] == effective[1] == effective[2]
    assert (work / "receipts/calibrate_calibration.json").is_file()
    assert (work / "receipts/calibrate_calibration_halt.json").is_file()
    assert (work / "receipts/calibrate_calibration_full_rebuild.json").is_file()
    return dict(
        scope="actual Chase calibration and runtime loading on synthetic inputs; no model training",
        outputs=results,
        source_protocol_unchanged=True,
        independent_inputs_identical=True,
        cross_semantics_load_rejected_both_directions=True,
    )


def test_halt_control_prepares_runs_and_loads_its_own_calibration(tmp_path):
    exercise_both_calibrations(tmp_path)


def test_conflicting_strict_calibration_cannot_reuse_successful_quarantine_scores(tmp_path):
    work, _ = workflow_fixture(tmp_path, calibration_conflict=True)
    assert workflow.main(["calibrate", "--workspace", str(work)])["status"] == "complete"
    artifact = workflow.read(work / "calibration/RELIABILITY.json")
    assert artifact["chase_semantics"] == RETAINED_CHASE_SEMANTICS
    assert [r["support"] for r in artifact["rule_scores"]] == [0, 18, 18]
    assert workflow.read(work / "calibration/RESULTS.json")["conflict_unknown_pairs"] == 18
    assert (
        workflow.main(
            ["calibrate", "--workspace", str(work), "--unknown-semantics", "full_rebuild"]
        )["status"]
        == "complete"
    )
    old_artifact = workflow.read(work / "calibration-full-rebuild/RELIABILITY.json")
    assert old_artifact["chase_semantics"] == QUARANTINE_CHASE_SEMANTICS
    assert [r["support"] for r in old_artifact["rule_scores"]] == [0, 0, 0]
    with pytest.raises(RuntimeError, match="experiment stage failed"):
        workflow.main(["calibrate", "--workspace", str(work), "--conflict-policy", "halt"])
    assert (
        workflow.read(work / "calibration-halt/RESULTS.json")["status"]
        == "conflict_no_reliability_estimate"
    )
    assert not (work / "calibration-halt/RELIABILITY.json").exists()
    with pytest.raises(FileNotFoundError):
        workflow.label_manifest(work, "halt_conflicts")
