"""V2 isolation, real typed Chase, calibration and external-query integration."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, ML, Rule
from loris._engine.typed_session import array_identity
from loris.discovery.manifest import file_record
from loris.labeling import cli
from loris.labeling.aux_extension import (
    configuration, attach_context, extend_program, partition_audit, prepare,
)
from loris.labeling.runner import LabelingRun, Version, with_channels
from loris.labeling.reliability import main_calibrate, refresh_current, calibration_texts
from tests.test_labeling import fixture, synthetic_same_trainer
from tests.test_fit_reliability import calibration_fixture
from tests.test_current_model_reliability import prediction_record


def isolated():
    context, rules, manifest = fixture()
    manifest["dynamic"]["alias"] = "adaptive"
    manifest["aux_extension"] = dict(source_alias="aux", alias="adaptive",
                                     default_threshold=.5, thresholds={"B": .8})
    return attach_context(context, manifest), extend_program(rules, manifest, context["labels"]), manifest


def close(run):
    for session in run.sessions:
        session.close()


def test_old_rule_bytes_and_old_alias_are_frozen_after_new_model_and_threshold_change(tmp_path):
    context, old, manifest = fixture()
    # Old Aux A would imply B if it changed; it must keep its original .2 table.
    old = (Rule((Doc("x"), ML("x", "aux", "A", .7)), Label("x", "B")),)
    before = [r.to_dict() for r in old]
    manifest["dynamic"]["alias"] = "adaptive"
    manifest["aux_extension"] = dict(source_alias="aux", alias="adaptive",
                                     default_threshold=.5, thresholds={"B": .8})
    ctx = attach_context(context, manifest)
    rules = extend_program(old, manifest, ctx["labels"])
    run = LabelingRun(context=ctx, rules=rules, manifest=manifest,
                      output=tmp_path / "run", trainer=synthetic_same_trainer)
    try:
        assert run.tables["aux"] is run.tables["adaptive"]  # One immutable M0 copy.
        assert run.receiver_occurrences == {(1, 1), (2, 1)}
        probabilities = run.version.probabilities.copy()
        probabilities[:, 0] = .9
        channels = dict(run.channels)
        channels[(1, 1)] = (0, .85)
        candidate = Version(probabilities, {"path": "synthetic", "sha256": "new"}, channels, "new")
        actual, _, _ = run.chase(candidate, phase="new_isolated_model")
        assert actual.positive_facts[2, 0] and not actual.positive_facts[2, 1]
        assert np.array_equal(run.tables_for(candidate)["aux"], context["model_probabilities"]["aux"])
        assert [r.to_dict() for r in with_channels(rules, channels)[:1]] == before
        assert [r.to_dict() for r in old] == before
        assert run.model_predicate_map()["occurrences"][0]["interpretation"] == "initial_table"
    finally:
        close(run)


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), True, "0.5"])
def test_invalid_thresholds_fail_closed(value):
    _, _, manifest = isolated()
    manifest["aux_extension"]["thresholds"]["B"] = value
    with pytest.raises(ValueError, match="thresholds"):
        configuration(manifest, ["A", "B"])


def test_alias_collisions_unknown_labels_and_old_ensemble_refresh_rejected():
    context, old, manifest = fixture()
    manifest["aux_extension"] = dict(source_alias="aux", alias="aux",
                                     default_threshold=.5, thresholds={})
    with pytest.raises(ValueError, match="distinct"):
        configuration(manifest)
    ctx, rules, manifest = isolated()
    with pytest.raises(ValueError, match="collides"):
        extend_program(rules, manifest, ctx["labels"])
    with pytest.raises(ValueError, match="already exists"):
        attach_context(ctx, manifest)
    manifest["aux_extension"]["thresholds"] = {"nonexistent": .8}
    with pytest.raises(ValueError, match="label axis"):
        configuration(manifest, ctx["labels"])
    manifest["aux_extension"]["thresholds"] = {}
    manifest["dynamic"]["ensemble_dependencies"] = [{"alias": "old_ensemble"}]
    with pytest.raises(ValueError, match="original ensemble"):
        configuration(manifest)


def test_prepare_and_actual_calibration_keep_source_artifacts_unchanged(tmp_path):
    args, _, _ = calibration_fixture(tmp_path)
    original = json.loads(Path(args.manifest).read_text())
    paths = [Path(args.manifest), Path(args.protocol), Path(original["program"]["path"]),
             Path(original["context"]["path"])]
    bytes_before = {p: p.read_bytes() for p in paths}
    dest = tmp_path / "v2"
    prepared = prepare(args.manifest, dest, protocol_path=args.protocol,
                       thresholds={"A": .75}, label_budget=8, query_batch_size=2)
    manifest_path = prepared["labeling_manifest"]["path"]
    m, ctx, rules, _, receipt = cli.load(manifest_path)
    assert len(rules) == 4 and receipt["dynamic_occurrences"] == 2
    assert rules[-2].body[1].eta == .75 and rules[-1].body[1].eta == .5
    assert m["loop"]["query_pool"] == "all_unanswered" and m["loop"]["query_batch_size"] == 2
    assert m["source_reliability"] is None
    args2 = SimpleNamespace(manifest=manifest_path,
        protocol=prepared["reliability_protocol"]["path"], output=str(dest / "calibration"))
    assert main_calibrate(args2) == 0
    m, ctx, rules, artifact, _ = cli.load(manifest_path, weighting="provenance",
                                         source_reliability=dest / "calibration/RELIABILITY.json")
    assert len(artifact["rule_scores"]) == 4
    assert artifact["rule_scores"][0]["support"] == artifact["rule_scores"][2]["support"] == 0
    assert all(p.read_bytes() == raw for p, raw in bytes_before.items())
    with pytest.raises(FileExistsError):
        prepare(args.manifest, dest)


def test_recalibration_updates_only_new_alias_over_two_candidates(tmp_path):
    args, calibration, _ = calibration_fixture(tmp_path)
    prepared = prepare(args.manifest, tmp_path / "v2", protocol_path=args.protocol)
    mp = prepared["labeling_manifest"]["path"]
    assert main_calibrate(SimpleNamespace(manifest=mp,
        protocol=prepared["reliability_protocol"]["path"], output=str(tmp_path / "cal"))) == 0
    m, ctx, rules, artifact, _ = cli.load(mp, weighting="provenance",
                                        source_reliability=tmp_path / "cal/RELIABILITY.json")
    frozen = array_identity(calibration["model_probabilities"]["aux"])
    for i, value in enumerate((.99, .01)):
        checkpoint = tmp_path / ("checkpoint%d" % i)
        checkpoint.write_text("candidate%d" % i)
        m["dynamic"]["checkpoint"] = file_record(checkpoint)
        ctx["model_probabilities"][m["dynamic"]["alias"]] = np.full((4, 2), value)
        folder = tmp_path / ("prediction%d" % i);folder.mkdir()
        prediction = prediction_record(folder, calibration_texts(artifact), ctx["labels"],
                                       m["dynamic"]["checkpoint"], value=value)
        artifact, _ = refresh_current(artifact, ctx, rules, m, prediction, tmp_path / ("refresh%d" % i))
        # Original dynamic=True Aux rule remains inactive after BOTH refreshes.
        assert artifact["rule_scores"][0]["support"] == 0
        assert artifact["rule_scores"][2]["support"] == (20 if value > .5 else 0)
        assert artifact["calibration_binding"]["model_tables"]["aux"] == frozen
        assert artifact["calibration_binding"]["static_model_tables"]["aux"] == frozen


def test_all_unanswered_queries_already_covered_rows_with_explicit_human_source(tmp_path):
    ctx, rules, m = isolated()
    # Every row already has a positive rule label; the old uncovered pool stops.
    rules = (Rule((Doc("x"),), Label("x", "A")),) + rules
    m["strategy"] = "frozen"
    m["loop"].update(label_budget=2, query_batch_size=1, query_pool="all_unanswered",
                     query_cadence="each_closure")
    answers = tmp_path / "answers";answers.mkdir()
    for i in (1, 2):
        (answers / ("%08d.human.json" % i)).write_text(json.dumps(dict(
            row_id="r%d" % i, labels=["A", "B"], targets=[1, 0], provider="simulated_human",
            provenance="Synthetic unit fixture, not a real human or empirical dataset.")))
    m["oracle"].update(kind="directory_human", directory=str(answers))
    run = LabelingRun(context=ctx, rules=rules, manifest=m, output=tmp_path / "run",
                      trainer=synthetic_same_trainer)
    report = run.run()
    assert report["actual_label_requests"] == 2
    assert report["annotation_sources"]["human_answers"] == 0
    assert report["annotation_sources"]["simulated_human_answers"] == 2
    assert report["aux_extension"]["added_rules"] == 2


def test_v1_manifests_keep_their_exact_program_and_model_axis():
    context, rules, m = fixture()
    assert attach_context(context, m) is context
    assert extend_program(rules, m, context["labels"]) == rules
    assert partition_audit(rules, m, context["labels"]) is None


@pytest.mark.parametrize("change_threshold", [False, True])
def test_incremental_aux_retraction_preserves_frozen_rule_support(tmp_path, change_threshold):
    ctx, rules, m = isolated()
    m["loop"].update(conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences",
                     gamma_conflict_policy="validated_gamma_wins")
    m["arbitration"] = dict(alias="aux", default_threshold=.5, thresholds={})
    run = LabelingRun(context=ctx, rules=rules, manifest=m, output=tmp_path / "run",
                      trainer=synthetic_same_trainer)
    try:
        base, session, _ = run.chase(run.version, phase="initial")
        high = run.version.probabilities.copy();high[:, 0] = .9
        v1 = Version(high, {"path": "fixture", "sha256": "new1"}, dict(run.channels), "v1")
        a, sa, _ = run.chase(v1, phase="high", previous=(base, session), update_kind="chaseDelta")
        assert a.positive_facts[2, 0]
        low = high.copy();channels = dict(v1.channels)
        if change_threshold:channels[(len(rules) - 2, 1)] = (0, .95)
        else:low[:, 0] = .1
        v2 = Version(low, {"path": "fixture", "sha256": "new2"}, channels, "v2")
        b, sb, _ = run.chase(v2, phase="low", previous=(a, sa), update_kind="chaseDelta")
        assert not b.positive_facts[2, 0]  # New Aux's obsolete conclusion retracts.
        assert b.positive_facts[1, 0]  # Original text rule is still independent support.
        assert run.tables_for(v2)["aux"] is run.tables["aux"]
        assert all(rule.alias == "aux" for rule in run.arbitration_rules)
        assert [r.to_dict() for r in with_channels(rules, channels)[:-2]] == [r.to_dict() for r in rules[:-2]]
    finally:
        close(run)


def test_prepare_freezes_actual_v1_fallback_and_keeps_original_arbitration(tmp_path):
    args, _, _ = calibration_fixture(tmp_path)
    m = json.loads(Path(args.manifest).read_text())
    m["loop"].update(conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences",
                     gamma_conflict_policy="validated_gamma_wins", commit_gate="independent_calibration")
    m["arbitration"] = dict(alias="aux", default_threshold=.5, thresholds={"B": .8})
    m["fallback"] = dict(enabled=True, alias="aux", calibration="independent_calibration",
                         default_threshold=1., thresholds={})
    Path(args.manifest).write_text(json.dumps(m))
    with pytest.raises(ValueError, match="source fallback"):
        prepare(args.manifest, tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()
    assert main_calibrate(args) == 0
    m["source_reliability"] = file_record(Path(args.output) / "RELIABILITY.json")
    Path(args.manifest).write_text(json.dumps(m))
    prepared = prepare(args.manifest, tmp_path / "v2", protocol_path=args.protocol)
    v2 = json.loads(Path(prepared["labeling_manifest"]["path"]).read_text())
    old_thresholds = json.loads((Path(args.output) / "FALLBACK_CALIBRATION.json").read_text())["thresholds"]
    assert v2["arbitration"] == m["arbitration"]
    assert v2["fallback"]["alias"] == "aux"
    assert v2["fallback"]["thresholds"] == old_thresholds
    assert v2["fallback"]["calibration"] == "fixed"
    assert v2["loop"]["commit_gate"] == "independent_calibration"
    assert main_calibrate(SimpleNamespace(manifest=prepared["labeling_manifest"]["path"],
        protocol=prepared["reliability_protocol"]["path"], output=str(tmp_path / "v2cal"))) == 0
    effective, ctx, rules, artifact, _ = cli.load(prepared["labeling_manifest"]["path"],
                                                 source_reliability=tmp_path / "v2cal/RELIABILITY.json")
    run = LabelingRun(context=ctx, rules=rules, manifest=effective, reliability=artifact,
                      output=tmp_path / "run", trainer=synthetic_same_trainer)
    try:
        assert run.fallback["alias"] == "aux"
        assert run.best_gate_score is not None
        assert run.fallback_thresholds.tolist() == [old_thresholds[x] for x in ctx["labels"]]
        high = Version(np.full((4, 2), .99), {"path": "fixture", "sha256": "changed"},
                       run.version.channels, "changed")
        view = SimpleNamespace(status="fixpoint", positive_facts=np.zeros((4, 2), bool),
                               negative_facts=np.zeros((4, 2), bool))
        run._refresh_fallback(view, run.version)
        before = view.fallback_positive.copy()
        run._refresh_fallback(view, high)
        np.testing.assert_array_equal(before, view.fallback_positive)
    finally:
        close(run)


def test_real_torch_training_receives_only_the_new_aux_channel(tmp_path):
    from tests.test_labeling_actual_aux import assets, Tokenizer, Encoder
    from loris.labeling.trainer import train_actual

    source, parent, config = assets(tmp_path)
    ctx, rules, m = isolated()
    m["dynamic"].update(checkpoint=file_record(parent), backend_source=source, configuration=config)
    m["loop"].update(query_pool="all_unanswered", max_inner_rounds=2)
    seen = []
    def actual(**kwargs):
        seen.append(kwargs)
        return train_actual(**kwargs, tokenizer_factory=lambda _: Tokenizer(),
                            model_factory=lambda _, k: Encoder(k))
    run = LabelingRun(context=ctx, rules=rules, manifest=m, output=tmp_path / "run", trainer=actual)
    frozen = run.tables["aux"].copy()
    report = run.run()
    assert report["actual_optimizer_updates"] >= 1 and seen
    assert all(i >= len(rules) - 2 for i, _ in run.receiver_occurrences)
    assert np.array_equal(run.tables["aux"], frozen)
    assert run.tables["aux"].flags.writeable is False
    assert report["aux_extension"]["old_model_tables_refreshed"] is False
    trace = json.loads((tmp_path / "run/update_00001/ACTUAL_TRAINING_TRACE.json").read_text())
    assert trace["windows"] and report["annotation_sources"]["human_answers"] == 0
