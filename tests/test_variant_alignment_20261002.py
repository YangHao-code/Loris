"""Comparison variants keep the main protocol's commit gate and per-model fallback calibration.

2026-10-02: NoWeight (uniform), confidence_only, the q-weighting aliases, SelfTrain, Supervised
and frozen_prior differ from the main protocol only in the mechanism they study; every variant
loads its calibration through the complete artifact binding (program, semantics, model).
"""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from loris.labeling import cli
from loris.labeling.runner import LabelingRun
from tests.test_current_model_reliability import scripted_trainer
from tests.test_review2_fixes_20261001 import calibrated

TEMPLATE = Path(__file__).resolve().parents[1] / "configs/templates/labeling.template.json"


def stubbed_label_manifest(monkeypatch):
    from loris.experiments import workflow

    template = json.loads(TEMPLATE.read_text())
    template["oracle"] = dict(kind="directory_llm_then_human", directory="answers",
                              allow_known_source_review=False)
    monkeypatch.setattr(workflow, "read", lambda path: dict(application_context=dict(
        status="verified_complete_declared_split_axis")))
    monkeypatch.setattr(workflow, "manifest", lambda workspace, name: copy.deepcopy(template))
    monkeypatch.setattr(workflow, "program", lambda workspace: "program.json")
    monkeypatch.setattr(workflow, "file_record", lambda path: dict(path=str(path), sha256="0" * 64))
    return workflow


def test_every_comparison_variant_keeps_gate_fallback_and_binds_its_calibration(monkeypatch, tmp_path):
    workflow = stubbed_label_manifest(monkeypatch)
    for variant in workflow.VARIANTS:
        data = workflow.label_manifest(tmp_path, variant)
        directory = Path(data["source_reliability"]["path"]).parent.name
        if variant == "original_semantics":
            assert "fallback" not in data and data["loop"]["commit_gate"] == "none"
            assert directory == "calibration-original"
            continue
        assert data["loop"]["commit_gate"] == "independent_calibration", variant
        assert data["fallback"]["calibration"] == "independent_calibration", variant
        expected = {"halt_conflicts": "calibration-halt",
                    "unknown_full_rebuild": "calibration-full-rebuild"}.get(variant, "calibration")
        assert directory == expected, variant
    for variant in workflow.CALIBRATION_VARIANTS:
        assert workflow.label_manifest(tmp_path, variant)["source_reliability"] is None


def load_variant(root, args, strategy, weighting, mode="current_model", change=None, artifact=True):
    manifest = json.loads(Path(args.manifest).read_text())
    manifest.update(strategy=strategy, weighting=weighting)
    manifest["feedback"]["rule_reliability_update"] = mode
    if strategy == "supervised":
        manifest["feedback"].update(trusted_replay=True, trusted_replay_fraction=1.0, rule_feedback_scope="all")
    if change is not None:
        change(manifest)
    path = Path(tempfile.mkdtemp(dir=root)) / "manifest.json"
    path.write_text(json.dumps(manifest))
    reliability = root / "calibrated/RELIABILITY.json" if artifact else None
    return cli.load(path, source_reliability=reliability) if artifact else cli.load(path, ignore_reliability=True)


def run_variant(root, loaded):
    effective, ctx, rules, artifact, _ = loaded
    trainer, seen = scripted_trainer(ctx)
    out = Path(tempfile.mkdtemp(dir=root)) / "run"
    run = LabelingRun(context=ctx, rules=rules, manifest=effective, output=out, trainer=trainer,
                      reliability=artifact)
    return run, run.run(), seen, out


@pytest.mark.parametrize("strategy, weighting", [
    ("paper", "uniform"), ("paper", "confidence_only"), ("paper", "provenance"),
    ("self_train", "uniform"), ("supervised", "uniform"),
])
def test_comparison_strategies_recalibrate_every_candidate_and_apply_the_gate(tmp_path, strategy, weighting):
    args = calibrated(tmp_path)
    run, result, seen, out = run_variant(tmp_path, load_variant(tmp_path, args, strategy, weighting))
    assert seen and result["inference_counts"]["calibration"] == len(seen)
    assert len(list(out.rglob("GATE.json"))) == len(seen)
    assert all("independent_calibration" in kwargs["prediction_sets"] for kwargs in seen)


def test_frozen_variant_starts_from_the_bound_m0_record_and_never_refreshes(tmp_path):
    args = calibrated(tmp_path)
    record = json.loads((tmp_path / "calibrated/FALLBACK_CALIBRATION.json").read_text())
    run, result, seen, _ = run_variant(tmp_path, load_variant(tmp_path, args, "frozen", "uniform"))
    assert not seen and result["inference_counts"]["calibration"] == 0
    assert run.fallback_thresholds.tolist() == [record["thresholds"][label] for label in run.labels]
    assert run.best_gate_score == record["gate"]["mean_f1"]


def test_uniform_training_weights_do_not_depend_on_the_bound_calibration(tmp_path):
    args = calibrated(tmp_path)

    def plain(manifest):
        manifest.pop("fallback")
        manifest["loop"]["commit_gate"] = "none"
        manifest["source_reliability"] = None

    _, _, with_artifact, _ = run_variant(tmp_path, load_variant(tmp_path, args, "paper", "uniform"))
    _, _, without, _ = run_variant(tmp_path, load_variant(tmp_path, args, "paper", "uniform", change=plain,
                                                          artifact=False))
    first, other = with_artifact[0]["batch"], without[0]["batch"]
    for name in ("targets", "observed", "weights", "true_mask", "weak_mask"):
        np.testing.assert_array_equal(getattr(first, name), getattr(other, name))


def test_frozen_prior_keeps_its_prior_q_after_a_committed_update(tmp_path):
    from loris.labeling.reliability import FROZEN_POLICY

    args = calibrated(tmp_path)

    def lenient(manifest):
        manifest["loop"]["commit_gate_tolerance"] = 1.0  # let the scripted candidate commit

    frozen_run, frozen, _, _ = run_variant(tmp_path, load_variant(
        tmp_path, args, "paper", "direct_rule_precision_discount", "frozen_prior", lenient))
    current_run, current, _, _ = run_variant(tmp_path, load_variant(
        tmp_path, args, "paper", "direct_rule_precision_discount", "current_model", lenient))
    assert frozen["committed_model_updates"] >= 1 and current["committed_model_updates"] >= 1
    source = json.loads((tmp_path / "calibrated/RELIABILITY.json").read_text())
    assert frozen_run.reliability["transfer_policy"] == FROZEN_POLICY
    assert frozen_run.reliability["rule_scores"] == source["rule_scores"]
    assert frozen["final_reliability_artifact"]["path"] == str((tmp_path / "calibrated/RELIABILITY.json").resolve())
    assert current["final_reliability_artifact"] != frozen["final_reliability_artifact"]
    assert frozen["inference_counts"]["calibration"] >= 1  # refreshed for the fallback/gate only


@pytest.mark.parametrize("strategy, weighting", [
    ("frozen", "uniform"), ("paper", "uniform"), ("self_train", "uniform"), ("supervised", "uniform"),
])
def test_a_program_changed_after_calibration_is_rejected(tmp_path, strategy, weighting):
    args = calibrated(tmp_path)

    def changed_rules(manifest):
        from loris.discovery.manifest import file_record

        rules = json.loads(Path(manifest["program"]["path"]).read_text())
        rules[0]["body"][1]["eta"] = 0.7
        path = Path(tempfile.mkdtemp(dir=tmp_path)) / "program.json"
        path.write_text(json.dumps(rules))
        manifest["program"] = file_record(path)

    load_variant(tmp_path, args, strategy, weighting)  # the calibrated program loads
    with pytest.raises(ValueError, match="binding mismatch"):
        load_variant(tmp_path, args, strategy, weighting, change=changed_rules)
