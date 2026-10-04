"""The new calibration denominator is actual fired conclusions, including U."""

import json
import pickle
from pathlib import Path

import numpy as np

from loris._engine.rdl_reference import Doc, Label, Rule
from loris.labeling.reliability import RETAINED_CHASE_SEMANTICS, _estimate, validate_artifact
from tests.test_fit_reliability import calibration_fixture
from tests.test_quarantine_reliability import label_rule


def test_actual_retained_calibration_does_not_hide_conflicting_rule_predictions(tmp_path):
    args, calibration, protocol = calibration_fixture(tmp_path)
    manifest = json.loads(Path(args.manifest).read_text())
    manifest["loop"].update(
        conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences"
    )
    with Path(manifest["context"]["path"]).open("rb") as stream:
        target = pickle.load(stream)
    rules = (
        label_rule("red", "B"),
        label_rule("red", "B", "neg"),
        Rule((Doc("x"),), Label("x", "A")),
    )
    assert (
        _estimate(
            manifest,
            target,
            rules,
            protocol,
            calibration,
            calibration["gamma"].pos,
            calibration["gamma"].neg,
            np.ones(20, bool),
            Path(args.output),
            Path(protocol["context"]["path"]),
            Path(protocol["answers"]["path"]),
            {},
        )
        == 0
    )
    artifact = json.loads((Path(args.output) / "RELIABILITY.json").read_text())
    assert artifact["chase_semantics"] == RETAINED_CHASE_SEMANTICS
    scores = artifact["rule_scores"]
    assert [r["support"] for r in scores] == [18, 18, 20]
    # Both signs were produced before isolation: exactly one is correct for
    # every independently observed B, not two q=.5 priors after discarding U.
    assert scores[0]["correct"] + scores[1]["correct"] == 18
    assert scores[2]["correct"] == 17
    receipt = json.loads((Path(args.output) / "RESULTS.json").read_text())
    assert receipt["conflict_unknown_pairs"] == 18
    assert receipt["calibration_inference"] == "retention_without_arbitration"
    assert receipt["answers_entered_inference"] is False
    validate_artifact(artifact, target, rules, manifest)
