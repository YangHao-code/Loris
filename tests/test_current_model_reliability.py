"""Current-checkpoint q calibration, atomic commit and frozen-prior ablation."""

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from loris.discovery.manifest import file_record
from loris.labeling import cli
from loris.labeling.reliability import (
    main_calibrate,
    refresh_current,
    calibration_texts,
    TRANSFER_POLICY,
    FROZEN_POLICY,
    validate_artifact,
)
from loris.labeling.runner import LabelingRun, with_channels
from loris._engine.typed_session import digest, array_identity
from loris._engine.rdl_reference import Doc, ML, Label, Rule
from tests.test_fit_reliability import calibration_fixture


def prediction_record(folder, texts, labels, checkpoint, value=0.99):
    p = np.full((len(texts), len(labels)), 0.1, np.float32)
    p[:, 0] = value
    path = folder / "independent.npz"
    np.savez(path, probabilities=p)
    return dict(
        file=file_record(path),
        rows=len(texts),
        labels=list(labels),
        texts_id=digest(texts),
        checkpoint_sha256=checkpoint["sha256"],
        supervision_used=False,
        prediction_seconds=0.0,
    )


def initial(tmp_path, conflict_rule=False):
    args, calibration, _ = calibration_fixture(tmp_path)
    if conflict_rule:
        manifest = json.loads(Path(args.manifest).read_text())
        path = Path(manifest["program"]["path"])
        rules = json.loads(path.read_text())
        rules.append(Rule((Doc("x"), ML("x", "aux", "A", 0.95)), Label("x", "A", "neg")).to_dict())
        path.write_text(json.dumps(rules))
        manifest["program"] = file_record(path)
        Path(args.manifest).write_text(json.dumps(manifest))
    assert main_calibrate(args) == 0
    path = tmp_path / "calibrated/RELIABILITY.json"
    effective, context, rules, artifact, _ = cli.load(
        args.manifest, weighting="provenance", source_reliability=path
    )
    return effective, context, rules, artifact, calibration


def test_current_prediction_and_thresholds_recalibrate_q_static_model_stays_initial(tmp_path):
    manifest, ctx, rules, artifact, calibration = initial(tmp_path)
    texts = calibration_texts(artifact)
    checkpoint = tmp_path / "new.pt"
    checkpoint.write_bytes(b"new frozen model")
    new_manifest = copy.deepcopy(manifest)
    new_manifest["dynamic"]["checkpoint"] = file_record(checkpoint)
    ctx = dict(ctx, model_probabilities=dict(ctx["model_probabilities"]))
    ctx["model_probabilities"]["aux"] = np.full_like(ctx["model_probabilities"]["aux"], 0.9)
    record = prediction_record(
        tmp_path, texts, ctx["labels"], new_manifest["dynamic"]["checkpoint"]
    )
    updated, ref = refresh_current(
        artifact, ctx, rules, new_manifest, record, tmp_path / "refresh1"
    )
    assert updated["rule_scores"][0]["support"] == 20
    assert updated["rule_scores"][0]["q"] == pytest.approx(18 / 22)
    assert updated["transfer_policy"] == TRANSFER_POLICY
    assert updated["calibration_binding"]["static_model_tables"]["aux"] == array_identity(
        calibration["model_probabilities"]["aux"]
    )
    validate_artifact(updated, ctx, rules, new_manifest)
    threshold_rules = with_channels(rules, {(0, 1): (0, 1.0)})
    refreshed, _ = refresh_current(
        updated, ctx, threshold_rules, new_manifest, record, tmp_path / "refresh2"
    )
    assert refreshed["rule_scores"][0]["support"] == 0 and refreshed["rule_scores"][0]["q"] == 0.5
    assert (
        refreshed["calibration_binding"]["static_model_tables"]
        == updated["calibration_binding"]["static_model_tables"]
    )
    with open(refreshed["protocol"]["path"]) as f:
        protocol = json.load(f)
    with open(protocol["context"]["path"], "rb") as f:
        independent = pickle.load(f)
    assert not independent["gamma"].pos.any() and not independent["gamma"].neg.any()
    assert "truth" not in independent and "source_gold" not in independent


def scripted_trainer(ctx, missing_predictions=False):
    seen = []

    def trainer(**kwargs):
        folder = Path(kwargs["output"])
        folder.mkdir()
        checkpoint = folder / "checkpoint"
        checkpoint.write_text(folder.name)
        p = ctx["model_probabilities"]["aux"].copy()
        p[1, 0] = 0.9
        fit = dict(optimizer_updates=1, checkpoint=file_record(checkpoint))
        seen.append(kwargs)
        if "prediction_sets" in kwargs and not missing_predictions:
            assert set(kwargs["prediction_sets"]) == {"independent_calibration"}
            texts = kwargs["prediction_sets"]["independent_calibration"]
            assert all(isinstance(t, str) for t in texts)
            fit["prediction_sets"] = {
                "independent_calibration": prediction_record(
                    folder, texts, ctx["labels"], fit["checkpoint"]
                )
            }
        return p, fit

    return trainer, seen


def test_new_model_and_new_q_commit_together_and_repeated_B_does_not_refresh(tmp_path):
    manifest, ctx, rules, artifact, _ = initial(tmp_path)
    trainer, seen = scripted_trainer(ctx)
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=trainer,
        reliability=artifact,
    )
    result = run.run()
    assert result["committed_model_updates"] == 1 and result["actual_optimizer_updates"] == 2
    assert result["inference_counts"]["calibration"] == 1
    assert result["final_reliability_artifact"] != manifest["source_reliability"]
    assert run.reliability["rule_scores"][0]["q"] == pytest.approx(18 / 22)
    assert run.reliability["binding"]["checkpoint_sha256"] == result["checkpoint"]["sha256"]
    assert len(seen) == 2 and all("prediction_sets" in call for call in seen)
    assert len(list((tmp_path / "run").glob("update_*/reliability/RELIABILITY.json"))) == 1


@pytest.mark.parametrize("missing,conflict", [(True, False), (False, True)])
def test_missing_or_conflicting_current_calibration_cannot_install_model_or_reuse_q(
    tmp_path, missing, conflict
):
    manifest, ctx, rules, artifact, _ = initial(tmp_path, conflict_rule=conflict)
    trainer, _ = scripted_trainer(ctx, missing_predictions=missing)
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=trainer,
        reliability=artifact,
    )
    result = run.run()
    assert result["committed_model_updates"] == 0 and result["actual_optimizer_updates"] == 1
    assert result["checkpoint"] == manifest["dynamic"]["checkpoint"]
    assert result["final_reliability_artifact"] == manifest["source_reliability"]
    assert run.reliability["rule_scores"] == artifact["rule_scores"]
    assert result["status"] == "update_failed_retaining_zero_update_state"
    assert any(
        "calibration" in e.get("reason", "")
        for e in run.events
        if e["kind"] == "update_not_committed"
    )


def test_frozen_prior_is_named_ablation_and_has_no_hidden_current_model_claim(tmp_path):
    manifest, ctx, rules, artifact, _ = initial(tmp_path)
    manifest["feedback"]["rule_reliability_update"] = "frozen_prior"
    trainer, seen = scripted_trainer(ctx)
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=trainer,
        reliability=artifact,
    )
    result = run.run()
    assert result["committed_model_updates"] == 1 and result["inference_counts"]["calibration"] == 0
    assert (
        run.reliability["rule_scores"][0]["q"] == 0.5
        and run.reliability["transfer_policy"] == FROZEN_POLICY
    )
    assert all("prediction_sets" not in call for call in seen)
    assert result["rule_reliability_update"] == "frozen_prior"


def test_renamed_duplicate_calibration_document_is_rejected(tmp_path):
    args, calibration, protocol = calibration_fixture(tmp_path)
    calibration["docs"][0].cnt = "  red  "
    with open(protocol["context"]["path"], "wb") as f:
        pickle.dump(calibration, f)
    protocol["context"] = file_record(protocol["context"]["path"])
    Path(args.protocol).write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="content overlaps"):
        main_calibrate(args)
    assert not (tmp_path / "calibrated/RELIABILITY.json").exists()


def test_failed_refresh_preserves_acquired_Gamma_and_rebuilds_old_model_closure(tmp_path):
    args, _, _ = calibration_fixture(tmp_path)
    source = json.loads(Path(args.manifest).read_text())
    program = Path(source["program"]["path"])
    records = json.loads(program.read_text())
    records.append(Rule((Doc("x"), ML("x", "aux", "B", 0.8)), Label("x", "A", "neg")).to_dict())
    program.write_text(json.dumps(records))
    source["program"] = file_record(program)
    Path(args.manifest).write_text(json.dumps(source))
    assert main_calibrate(args) == 0
    manifest, ctx, rules, artifact, _ = cli.load(
        args.manifest,
        weighting="provenance",
        source_reliability=tmp_path / "calibrated/RELIABILITY.json",
    )
    answers = tmp_path / "answers"
    answers.mkdir()
    (answers / "00000002.json").write_text(
        json.dumps(dict(row_id="r2", labels=["A", "B"], targets=[1, 0]))
    )
    manifest["oracle"].update(
        kind="directory_simulated_human", directory=str(answers), allow_known_source_review=True
    )
    manifest["loop"].update(
        label_budget=1,
        source_review_budget=2,
        max_chase_calls=100,
        conflict_policy="review_instances",
    )

    def trainer(**kwargs):
        folder = Path(kwargs["output"])
        folder.mkdir()
        checkpoint = folder / "checkpoint"
        checkpoint.write_text("candidate")
        p = ctx["model_probabilities"]["aux"].copy()
        p[2] = 0.9
        # Deliberate missing independent predictions: fail after candidate
        # conflict has obtained a real answer and resolved its wrong instance.
        return p, dict(optimizer_updates=1, checkpoint=file_record(checkpoint))

    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=trainer,
        reliability=artifact,
    )
    report = run.run()
    assert (
        report["committed_model_updates"] == 0
        and report["checkpoint"] == manifest["dynamic"]["checkpoint"]
    )
    assert report["actual_label_requests"] == 1 and run.gamma.pos[2, 0] and run.gamma.neg[2, 1]
    assert report["current_Gamma_has_valid_closure"]
    assert report["final_reliability_artifact"] == manifest["source_reliability"]
    assert report["status"] == "update_failed_retaining_zero_update_state"
    assert any(
        e["kind"] == "Chase"
        and e["phase"] == "rollback_model_keep_new_answers_after_failed_refresh"
        for e in run.events
    )
