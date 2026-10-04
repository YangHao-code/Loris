"""Real complete-Chase contract tests; synthetic trainers are named test seams."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from loris.chase import GammaSeed
from loris.document import Document
from loris.predicates import MatchPredicate
from loris._engine.rdl_reference import Doc, Label, ML, Text, Rule
from loris.labeling.feedback import Batch, assignments, calibrate
from loris.labeling.runner import LabelingRun, Version, rule_channels
from loris.labeling.cli import validate


def fixture():
    p = np.array([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2]], np.float32)
    gp = np.zeros((4, 2), bool)
    gn = gp.copy()
    gp[0, 0] = True
    gn[0, 1] = True
    ctx = dict(
        docs=[Document(cnt=t, ttl="", mtd="") for t in ("red", "red", "blue", "blank")],
        labels=["A", "B"],
        row_ids=["r0", "r1", "r2", "r3"],
        row_roles=["known", "train", "train", "dev"],
        gamma=GammaSeed(gp, gn),
        model_probabilities={"aux": p, "fixed": np.full_like(p, 0.7)},
        model_decisions={},
        receiver_indices=np.array([3]),
        truth=np.array([[0, 0]], bool),
    )
    rules = (
        Rule((Doc("x"), ML("x", "aux", "A", 0.8)), Label("x", "A")),
        Rule((Doc("x"), Text("x", MatchPredicate("cnt", "red"))), Label("x", "A")),
    )
    m = dict(
        schema="loris_latest_paper_labeling_v1",
        context={"path": "fixture", "sha256": "fixture"},
        program={"path": "fixture", "sha256": "fixture"},
        dynamic=dict(
            alias="aux",
            checkpoint={"path": "fixture", "sha256": "fixture"},
            backend_source="fixture",
            configuration={},
            device="cpu",
            gpu_lock=None,
            ensemble_dependencies=[],
            probability_lineage="synthetic_fixture",
        ),
        domains=dict(
            training_roles=["known", "train"], query_roles=["train"], scoring_roles=["dev"]
        ),
        loop=dict(
            max_inner_rounds=3,
            query_batch_size=1,
            label_budget=0,
            source_review_budget=0,
            saturation_rounds=1,
            max_chase_calls=50,
            max_rdg_estimates=30,
            max_update_seconds=60,
        ),
        feedback=dict(
            trusted_replay=True,
            normalization="common_mass_per_accumulation_window",
            channel_policy="any_actual_occurrence_disagrees_deduplicate_pair",
            candidate_check="checked",
            self_train_confidence=0.95,
        ),
        query=dict(strategy="rdg", seed=42),
        calibration=dict(grid=[0.5, 0.8, 0.95], minimum_support_each_sign=1),
        oracle=dict(kind="disabled", directory=None, allow_known_source_review=False),
        strategy="paper",
        weighting="uniform",
        source_reliability=None,
        notes="Synthetic mechanism fixture, never an empirical dataset result",
    )
    return ctx, rules, m


def synthetic_same_trainer(**kwargs):
    # Deliberately explicit fixture; public CLI always binds actual backend.
    Path(kwargs["output"]).mkdir()
    p = np.full(kwargs["batch"].targets.shape, 0.2, np.float32)
    return p, dict(
        optimizer_updates=1, checkpoint={"path": "synthetic", "sha256": Path(kwargs["output"]).name}
    )


def test_real_complete_chase_A_target_unknown_not_negative_and_last_stability(tmp_path):
    ctx, rules, m = fixture()
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=m,
        output=tmp_path / "run",
        trainer=synthetic_same_trainer,
    )
    report = run.run()
    assert report["status"] == "label_budget_exhausted" and report["committed_model_updates"] == 0
    assert report["training_attempts"] == 1
    assert report["checkpoint"]["sha256"] == "fixture"
    assert any(e.get("candidate_not_installed") for e in run.events)
    event = next(e for e in run.events if e["kind"] == "supervision")
    assert event["weak_positive"] == 1 and event["weak_negative"] == 0
    assert event["true_mass"] == 2 and event["weak_mass"] == 1
    assert event["feedback_check"] == "complete_Gamma_union_W_success"
    assert event["logical_A_check_redundant_for_already_entailed_A"]
    with np.load(tmp_path / "run/supervision_00000/ARRAYS.npz") as z:
        assert z["targets"][1, 0] == 1 and z["observed"][1, 0]
        assert not z["observed"][2].any() and not z["observed"][3].any()
    assert np.array_equal(run.gamma.pos, ctx["gamma"].pos) and np.array_equal(
        run.gamma.neg, ctx["gamma"].neg
    )
    assert all(e["rounds"] >= 1 for e in run.events if e["kind"] == "Chase")


def test_alternate_independent_support_not_lost_to_first_Aux_witness(tmp_path):
    ctx, rules, m = fixture()
    ctx["model_probabilities"]["aux"][1, 0] = 0.9
    # Receiver occurrence with high eta disagrees; its first .5 rule generates
    # the first witness, but the text rule is a real alternative support.
    rules = (Rule((Doc("x"), ML("x", "aux", "A", 0.5)), Label("x", "A")),) + rules
    run = LabelingRun(
        context=ctx, rules=rules, manifest=m, output=tmp_path / "r", trainer=synthetic_same_trainer
    )
    result, _, _ = run.chase(run.version, phase="test")
    ind = run.independent(run.version)
    assert result.proof_dag["pos:1:0"]["rule_id"] == 0
    # Make the receiver's high threshold a true disagreement while retaining
    # actual .5 firing; no source filtering is inferred from first proof.
    channels = dict(run.version.channels)
    channels[(1, 1)] = (0, 0.95)
    b = assignments(
        run.gamma,
        result,
        ind,
        run.version.probabilities,
        channels,
        run.training_rows,
        strategy="paper",
        weighting="uniform",
    )
    assert b.weak_mask[1, 0]
    for s in run.sessions:
        s.close()


def test_calibration_keeps_occurrence_eta_and_unknowns_do_not_supply_support():
    y = np.array([[1, 0], [0, 0], [0, 0]], np.float32)
    mask = np.array([[1, 0], [1, 0], [0, 0]], bool)
    b = Batch(y, mask, mask.astype(np.float32), mask, np.zeros_like(mask), {})
    p = np.array([[0.7, 0.9], [0.6, 0.9], [0.999, 0.999]])
    channels = {(0, 1): (0, 0.8), (1, 1): (0, 0.65), (2, 1): (1, 0.2)}
    selected, receipt = calibrate(p, b, channels, grid=[0.65, 0.8], minimum_support_each_sign=1)
    assert selected[(0, 1)] == (0, 0.65) and selected[(1, 1)] == (0, 0.65)
    assert selected[(2, 1)] == (1, 0.2)
    assert receipt["occurrences"][-1]["status"] == "unsupported_preserve_previous"


def test_failed_candidate_keeps_current_Gamma_and_valid_model(tmp_path):
    ctx, rules, m = fixture()
    rules = rules + (Rule((Doc("x"), ML("x", "aux", "B", 0.8)), Label("x", "B")),)

    def bad(**kw):
        Path(kw["output"]).mkdir()
        return np.full((4, 2), 0.95, np.float32), dict(
            optimizer_updates=1, checkpoint={"path": "bad", "sha256": "bad"}
        )

    run = LabelingRun(context=ctx, rules=rules, manifest=m, output=tmp_path / "r", trainer=bad)
    report = run.run()
    assert report["status"] == "candidate_conflict_retaining_current_Gamma_state"
    assert report["current_Gamma_has_valid_closure"] and report["committed_model_updates"] == 0
    assert report["checkpoint"]["sha256"] == "fixture" and run.gamma.neg[0, 1]


def test_unchanged_refinement_dispatches_without_waiting_for_round_cap(tmp_path):
    ctx, rules, m = fixture()
    m["loop"].update(max_inner_rounds=1, label_budget=1)
    run = LabelingRun(
        context=ctx, rules=rules, manifest=m, output=tmp_path / "r", trainer=synthetic_same_trainer
    )
    r = run.run()
    assert r["status"] == "pending_external_answer" and r["committed_model_updates"] == 0


def test_last_budget_answer_gets_training_and_answers_survive_failure(tmp_path):
    ctx, rules, m = fixture()
    m["loop"]["label_budget"] = 1
    oracle = tmp_path / "answers"
    oracle.mkdir()
    (oracle / "00000002.json").write_text(
        json.dumps(dict(row_id="r2", labels=["A", "B"], targets=[1, 0]))
    )
    m["oracle"].update(kind="directory_simulated_human", directory=str(oracle))
    seen = []

    def train(**kw):
        seen.append(int(kw["batch"].true_mask.sum()))
        if kw["batch"].true_mask[2].any():
            raise RuntimeError("controlled post-answer training failure")
        return synthetic_same_trainer(**kw)

    run = LabelingRun(context=ctx, rules=rules, manifest=m, output=tmp_path / "r", trainer=train)
    r = run.run()
    assert seen == [2, 4]
    assert r["current_Gamma_has_valid_closure"] and r["actual_label_requests"] == 1
    assert run.gamma.pos[2, 0] and run.gamma.neg[2, 1] and 2 in run.answered


def test_actual_known_conflict_reviews_source_without_query(tmp_path):
    ctx, rules, m = fixture()
    m["strategy"] = "frozen"
    m["oracle"]["allow_known_source_review"] = True
    m["loop"].update(source_review_budget=2, conflict_policy="review_instances")
    ctx["model_probabilities"]["aux"][0, 1] = 0.9
    rules = rules + (Rule((Doc("x"), ML("x", "aux", "B", 0.8)), Label("x", "B")),)
    run = LabelingRun(
        context=ctx, rules=rules, manifest=m, output=tmp_path / "r", trainer=synthetic_same_trainer
    )
    r = run.run()
    assert r["current_Gamma_has_valid_closure"]
    assert (
        r["actual_label_requests"] == 0 and r["actual_source_reviews"] == 1 and run.gamma.neg[0, 1]
    )


def test_provenance_rejects_historical_pooled_fit_counts(tmp_path):
    ctx, rules, m = fixture()
    run = LabelingRun(
        context=ctx, rules=rules, manifest=m, output=tmp_path / "r", trainer=synthetic_same_trainer
    )
    result, _, _ = run.chase(run.version, phase="test")
    ind = run.independent(run.version)
    with pytest.raises(ValueError, match="historical fit pooled"):
        assignments(
            run.gamma,
            result,
            ind,
            run.version.probabilities,
            run.channels,
            run.training_rows,
            strategy="paper",
            weighting="provenance",
            reliability={"support": [[0, 0], [0, 0]], "correct": [[0, 0], [0, 0]]},
        )
    for s in run.sessions:
        s.close()


def test_manifest_invalid_provider_and_budget_are_not_silent_defaults():
    _, _, m = fixture()
    validate(m)
    m["oracle"]["kind"] = "fake_llm"
    with pytest.raises(ValueError, match="LLM"):
        validate(m)
    _, _, m = fixture()
    m["loop"]["max_inner_rounds"] = True
    with pytest.raises(ValueError, match="budget"):
        validate(m)


def test_static_receiver_source_is_not_independent(tmp_path):
    ctx, rules, m = fixture()
    ctx["model_decisions"]["aux"] = np.ones((4, 2), bool)
    rules = (rules[0], Rule((Doc("x"), ML("x", "aux", "A", 0.1, dynamic=False)), Label("x", "A")))
    run = LabelingRun(
        context=ctx, rules=rules, manifest=m, output=tmp_path / "r", trainer=synthetic_same_trainer
    )
    result, _, _ = run.chase(run.version, phase="test")
    ind = run.independent(run.version)
    b = assignments(
        run.gamma,
        result,
        ind,
        run.version.probabilities,
        run.channels,
        run.training_rows,
        strategy="paper",
        weighting="uniform",
        require_independent=True,
    )
    assert not b.weak_mask.any() and not ind.positive_facts[1, 0]
    for s in run.sessions:
        s.close()


def test_mismatched_initial_ensemble_is_not_zero_update(tmp_path):
    ctx, rules, m = fixture()
    ctx["model_probabilities"]["vote"] = np.ones((4, 2), np.float32)
    m["dynamic"]["ensemble_dependencies"] = [
        dict(
            alias="vote",
            dependencies=["aux", "fixed"],
            operation="binary_vote_mean",
            vote_threshold=0.5,
        )
    ]
    with pytest.raises(ValueError, match="M0"):
        LabelingRun(
            context=ctx,
            rules=rules,
            manifest=m,
            output=tmp_path / "r",
            trainer=synthetic_same_trainer,
        )
    m["dynamic"]["ensemble_dependencies"][0]["alias"] = "aux"
    with pytest.raises(ValueError, match="Aux"):
        LabelingRun(
            context=ctx,
            rules=rules,
            manifest=m,
            output=tmp_path / "s",
            trainer=synthetic_same_trainer,
        )


def test_predicate_delta_does_not_confuse_unchanged_P_with_saturation(tmp_path):
    ctx, rules, m = fixture()
    m["loop"].update(max_inner_rounds=1, label_budget=1, saturation_rounds=1)
    # This Aux conjunction never fires. Its predicate decision still changes.
    rules = (
        Rule(
            (Doc("x"), ML("x", "aux", "A", 0.8), Text("x", MatchPredicate("cnt", "absent"))),
            Label("x", "A"),
        ),
        rules[1],
    )

    def trained(**kw):
        Path(kw["output"]).mkdir()
        return np.full((4, 2), 0.95, np.float32), dict(
            optimizer_updates=1, checkpoint={"path": "new", "sha256": "new"}
        )

    run = LabelingRun(context=ctx, rules=rules, manifest=m, output=tmp_path / "r", trainer=trained)
    r = run.run()
    assert r["status"] == "inner_round_cap_not_saturated" and r["actual_label_requests"] == 0
    e = next(e for e in run.events if e["kind"] == "model_update_committed")
    assert e["unchanged_complete_facts"] and not e["all_actual_dynamic_occurrences_unchanged"]


def test_supervised_once_and_NoCheck_still_checks_candidate_submission(tmp_path):
    ctx, rules, m = fixture()
    m["strategy"] = "supervised"
    run = LabelingRun(
        context=ctx, rules=rules, manifest=m, output=tmp_path / "a", trainer=synthetic_same_trainer
    )
    assert run.run()["training_attempts"] == 1
    ctx, rules, m = fixture()
    m["feedback"]["candidate_check"] = "nocheck"
    rules = rules + (Rule((Doc("x"), ML("x", "aux", "B", 0.8)), Label("x", "B")),)

    def bad(**kw):
        Path(kw["output"]).mkdir()
        return np.full((4, 2), 0.95, np.float32), dict(
            optimizer_updates=1, checkpoint={"path": "new", "sha256": "new"}
        )

    run = LabelingRun(context=ctx, rules=rules, manifest=m, output=tmp_path / "b", trainer=bad)
    r = run.run()
    assert r["committed_model_updates"] == 0 and r["current_Gamma_has_valid_closure"]
    assert any(e.get("feedback_check") == "NoCheck_ablation_skipped" for e in run.events)


def test_model_candidate_actual_conflict_enters_approved_source_review(tmp_path):
    ctx, rules, m = fixture()
    m["strategy"] = "supervised"
    m["oracle"]["allow_known_source_review"] = True
    m["loop"].update(source_review_budget=1, conflict_policy="review_instances")
    rules = rules + (Rule((Doc("x"), ML("x", "aux", "B", 0.8)), Label("x", "B")),)

    def changed(**kw):
        Path(kw["output"]).mkdir()
        return np.full((4, 2), 0.95, np.float32), dict(
            optimizer_updates=1, checkpoint={"path": "new", "sha256": "new"}
        )

    run = LabelingRun(context=ctx, rules=rules, manifest=m, output=tmp_path / "r", trainer=changed)
    r = run.run()
    assert (
        r["committed_model_updates"] == 1
        and r["actual_source_reviews"] == 1
        and r["actual_label_requests"] == 0
    )
    assert run.gamma.neg[0, 1] and r["current_Gamma_has_valid_closure"]


def test_failed_fit_costs_preserved(tmp_path):
    ctx, rules, m = fixture()

    def failed(**kw):
        out = Path(kw["output"])
        out.mkdir()
        (out / "FAILED.json").write_text(json.dumps(dict(actual_optimizer_updates=3)))
        raise RuntimeError("real step failure fixture")

    run = LabelingRun(context=ctx, rules=rules, manifest=m, output=tmp_path / "r", trainer=failed)
    r = run.run()
    assert r["actual_optimizer_updates"] == 3 and r["committed_model_updates"] == 0
