"""Independent calibration: hidden answers, distinct rule heads, bound receipts."""

import copy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from loris.chase import GammaSeed
from loris.discovery.manifest import file_record
from loris.document import Document
from loris.labeling import cli
from loris.labeling.reliability import (
    PROTOCOL_SCHEMA,
    rule_score,
    inference_context,
    count_rule_conclusions,
    main_calibrate,
    validate_artifact,
)
from loris._engine.rdl_reference import Doc, Label, Text, Rule, RefGamma, LabelCompare
from loris._storage.discovery_factory import make_discovery_runner
from loris.predicates import MatchPredicate
from tests.test_labeling_reliability_weights import runtime_manifest


def calibration_fixture(tmp_path):
    manifest, m = runtime_manifest(tmp_path)
    n = 20
    gp = np.zeros((n, 2), bool)
    gp[:17, 0] = True
    gp[:15, 1] = True
    gn = ~gp
    calibration = dict(
        docs=[
            Document(cnt=("red" if i < 18 else "blue") + " independent-" + str(i), ttl="", mtd="")
            for i in range(n)
        ],
        labels=["A", "B"],
        row_ids=["calibration-%d" % i for i in range(n)],
        row_roles=["calibration"] * n,
        gamma=RefGamma(gp, gn, ((0, 1),), ()),
        model_probabilities={
            "aux": np.full((n, 2), 0.1, np.float32),
            "fixed": np.full((n, 2), 0.7, np.float32),
        },
        model_decisions={},
        truth="must_not_enter_inference",
        source_gold="must_not_enter_inference",
    )
    path = tmp_path / "independent_context.pkl"
    with path.open("wb") as f:
        pickle.dump(calibration, f)
    answers = tmp_path / "independent_answers.npz"
    np.savez(
        answers,
        positive=gp,
        negative=gn,
        row_ids=np.array(calibration["row_ids"]),
        labels=np.array(calibration["labels"]),
    )
    protocol = dict(
        schema=PROTOCOL_SCHEMA,
        context=file_record(path),
        answers=file_record(answers),
        calibration_row_ids=calibration["row_ids"],
        model_fit_row_ids=["r0"],
        rule_fit_row_ids=["r0"],
        model_binding=dict(
            checkpoint_sha256=m["dynamic"]["checkpoint"]["sha256"],
            probability_lineage=m["dynamic"]["probability_lineage"],
        ),
        independence_statement="Synthetic independent fixture; no model training and no real-data quality claim.",
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    args = SimpleNamespace(
        manifest=str(manifest), protocol=str(protocol_path), output=str(tmp_path / "calibrated")
    )
    return args, calibration, protocol


def test_answers_and_all_scored_row_roots_are_removed_before_inference(tmp_path):
    _, context, _ = calibration_fixture(tmp_path)
    selected = np.ones(20, bool)
    selected[-1] = False
    projected = inference_context(context, selected)
    assert not projected["gamma"].pos[selected].any() and not projected["gamma"].neg[selected].any()
    assert not projected["gamma"].subset
    assert projected["gamma"].neg[-1].all()
    assert set(projected).isdisjoint({"truth", "source_gold", "scope_decisions"})
    assert not projected["docs"][0].lbl
    assert context["gamma"].pos.any() and context["gamma"].neg.any()


def test_distinct_conclusions_per_rule_counts_every_alternative_not_first_witness(tmp_path):
    _, context, _ = calibration_fixture(tmp_path)
    selected = np.ones(20, bool)
    projected = inference_context(context, selected)
    text = Text("x", MatchPredicate("cnt", "red"))
    rules = (
        Rule((Doc("x"), text), Label("x", "A")),
        Rule((Doc("x"), Doc("y"), text), Label("x", "A")),
        Rule((Doc("x"), text), Label("x", "B")),
    )
    runner = make_discovery_runner(projected, tmp_path / "count_cache")
    try:
        closure = runner(rules, projected["gamma"])
        counts = count_rule_conclusions(
            runner, closure, rules, context["gamma"].pos, context["gamma"].neg, selected
        )
    finally:
        runner.close()
    assert [(x["support"], x["correct"], x["q"]) for x in counts] == [
        (18, 17, 0.9),
        (18, 17, 0.9),
        (18, 15, 0.8),
    ]
    assert counts[1]["matching_instances"] == 18 * 20
    assert closure.proof_dag["pos:0:0"]["rule_id"] == 0


def test_comparison_rule_with_no_labels_propagated_has_zero_target_support(tmp_path):
    _, context, _ = calibration_fixture(tmp_path)
    context["docs"] = context["docs"][:2]
    for name in ("row_ids", "row_roles"):
        context[name] = context[name][:2]
    context["gamma"] = RefGamma.empty(2, 2)
    context["model_probabilities"] = {a: p[:2] for a, p in context["model_probabilities"].items()}
    rules = (Rule((Doc("x"), Doc("y")), LabelCompare("x", "eq", "y")),)
    runner = make_discovery_runner(context, tmp_path / "relation_cache")
    try:
        state = runner(rules, context["gamma"])
        pos = np.array([[True, False], [True, False]])
        neg = ~pos
        counts = count_rule_conclusions(runner, state, rules, pos, neg, np.ones(2, bool))
        # Establishing relations alone creates no Label facts. Whole-set
        # comparison agreement is no longer the denominator for relation q.
        assert counts[0]["support"] == 0 and counts[0]["correct"] == 0 and counts[0]["q"] == 0.5
        assert counts[0]["conclusion_unit"] == "propagated_distinct_target_Label"
    finally:
        runner.close()


def test_actual_calibration_cli_artifact_and_initial_binding(tmp_path):
    args, _, _ = calibration_fixture(tmp_path)
    assert (
        cli.main(
            [
                "calibrate-reliability",
                "--manifest",
                args.manifest,
                "--protocol",
                args.protocol,
                "--output",
                args.output,
            ]
        )
        == 0
    )
    path = tmp_path / "calibrated/RELIABILITY.json"
    effective, context, rules, artifact, _ = cli.load(
        args.manifest, weighting="provenance", source_reliability=path
    )
    assert [r["q"] for r in artifact["rule_scores"]] == [0.5, 0.9]
    assert [r["support"] for r in artifact["rule_scores"]] == [0, 18]
    validate_artifact(artifact, context, rules, effective)
    receipt = json.loads((tmp_path / "calibrated/RESULTS.json").read_text())
    assert receipt["answers_entered_inference"] is False and receipt["model_updates"] == 0
    with pytest.raises(FileExistsError):
        main_calibrate(args)


@pytest.mark.parametrize(
    "change",
    ["model_overlap", "rule_overlap", "application_overlap", "model_binding", "answer_axes"],
)
def test_calibration_rejects_fit_exposure_and_misaligned_inputs(tmp_path, change):
    args, context, p = calibration_fixture(tmp_path)
    if change == "model_overlap":
        p["model_fit_row_ids"].append(context["row_ids"][0])
    elif change == "rule_overlap":
        p["rule_fit_row_ids"].append(context["row_ids"][0])
    elif change == "application_overlap":
        p["calibration_row_ids"][0] = "r0"
    elif change == "model_binding":
        p["model_binding"]["checkpoint_sha256"] = "different"
    else:
        np.savez(
            p["answers"]["path"],
            positive=context["gamma"].pos,
            negative=context["gamma"].neg,
            row_ids=np.array(context["row_ids"][::-1]),
            labels=np.array(context["labels"]),
        )
        p["answers"] = file_record(p["answers"]["path"])
    with open(args.protocol, "w") as f:
        json.dump(p, f)
    with pytest.raises(ValueError):
        main_calibrate(args)
    assert not (tmp_path / "calibrated/RELIABILITY.json").exists()


@pytest.mark.parametrize("change", ["q", "context", "program", "checkpoint", "receipt", "protocol"])
def test_runtime_artifact_rejects_tampered_counts_or_semantic_bindings(tmp_path, change):
    args, _, _ = calibration_fixture(tmp_path)
    assert main_calibrate(args) == 0
    path = tmp_path / "calibrated/RELIABILITY.json"
    a = json.loads(path.read_text())
    effective, context, rules, _, _ = cli.load(args.manifest)
    if change == "q":
        a["rule_scores"][1]["q"] = 1.0
    elif change == "context":
        context["model_probabilities"]["fixed"][0, 0] = 0.1
    elif change == "program":
        rules = rules[::-1]
    elif change == "checkpoint":
        effective["dynamic"]["checkpoint"]["sha256"] = "other"
    elif change == "receipt":
        record = a["estimation_receipt"]
        value = json.loads(open(record["path"]).read())
        value["status"] = "failed"
        with open(record["path"], "w") as f:
            json.dump(value, f)
        a["estimation_receipt"] = file_record(record["path"])
    else:
        record = a["protocol"]
        value = json.loads(open(record["path"]).read())
        value["model_fit_row_ids"].append("calibration-0")
        with open(record["path"], "w") as f:
            json.dump(value, f)
        a["protocol"] = file_record(record["path"])
    with pytest.raises(ValueError):
        validate_artifact(a, context, rules, effective)


def test_partial_conflicting_closure_produces_no_reliability_artifact(tmp_path):
    args, _, _ = calibration_fixture(tmp_path)
    manifest = json.loads(open(args.manifest).read())
    program = manifest["program"]["path"]
    rules = json.loads(open(program).read())
    opposite = copy.deepcopy(rules[1])
    opposite["head"]["polarity"] = "neg"
    rules.append(opposite)
    with open(program, "w") as f:
        json.dump(rules, f)
    manifest["program"] = file_record(program)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f)
    assert main_calibrate(args) == 2
    assert not (tmp_path / "calibrated/RELIABILITY.json").exists()
    assert (
        json.loads((tmp_path / "calibrated/RESULTS.json").read_text())["status"]
        == "conflict_no_reliability_estimate"
    )


def test_real_calibration_artifact_to_current_all_proof_runner_feedback(tmp_path):
    from loris.labeling.runner import LabelingRun
    from tests.test_labeling import synthetic_same_trainer

    args, _, _ = calibration_fixture(tmp_path)
    assert main_calibrate(args) == 0
    path = tmp_path / "calibrated/RELIABILITY.json"
    effective, context, rules, artifact, _ = cli.load(
        args.manifest, weighting="provenance", source_reliability=path
    )
    run = LabelingRun(
        context=context,
        rules=rules,
        manifest=effective,
        output=tmp_path / "run",
        trainer=synthetic_same_trainer,
        reliability=artifact,
    )
    try:
        result, session, _ = run.chase(run.version, phase="calibrated_feedback_test")
        batch = run.feedback(result, run.version)
        assert batch.weak_mask[1, 0] and batch.weights[1, 0] == pytest.approx(0.54)
        assert np.array_equal(batch.weights[0], [1, 1])
        assert batch.audit["reliability_derivation_audit"]["all_derivations"]
        assert batch.audit["reliability_derivation_audit"]["closure_calls"] == 3
        assert batch.audit["reliability_artifact"] == file_record(path)
    finally:
        for session in run.sessions:
            session.close()
