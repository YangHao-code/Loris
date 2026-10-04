"""Independent integration tests of the explicit unknown/arbitration protocol.

Synthetic rule examples test real LabelingRun execution; the final test runs
the real CPU backend and observes gradients. These are not dataset experiments.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, ML, RefGamma, Rule, Text
from loris.discovery.manifest import file_record
from loris.document import Document
from loris.labeling.runner import LabelingRun, Version
from loris.labeling.trainer import train_actual
from loris.predicates import MatchPredicate
from tests.test_labeling import fixture


def at(row):
    return Text("x", MatchPredicate("cnt", "^node%d$" % row))


def rule(label, *body, sign="pos"):
    return Rule((Doc("x"), at(0)) + body, Label("x", label, sign))


def setup(p=0.2, labels=("A", "B", "C", "D"), known=False):
    ctx, _, manifest = fixture()
    pos = np.zeros((2, len(labels)), bool)
    if known:
        pos[0, 0] = True
    ctx.update(
        docs=[Document(cnt="node%d" % i, ttl="", mtd="") for i in range(2)],
        labels=list(labels),
        row_ids=["r0", "r1"],
        row_roles=["train", "train"],
        gamma=RefGamma(pos, np.zeros_like(pos)),
        model_probabilities={"aux": np.full(pos.shape, p, np.float32)},
        receiver_indices=np.array([], dtype=int),
        truth=np.empty((0, len(labels)), bool),
    )
    manifest["loop"].update(conflict_policy="unknown_arbitrate", label_budget=0)
    manifest["arbitration"] = dict(alias="aux", default_threshold=0.5, thresholds={})
    manifest["feedback"].update(trusted_replay=False, receiver_policy="same_model")
    manifest["domains"].update(training_roles=["train"], query_roles=["train"], scoring_roles=[])
    manifest["weighting"] = "uniform"
    return ctx, manifest


def forbidden_training(**kwargs):
    pytest.fail("model arbitration must not create its own logical weak training targets")


def close(run):
    for runner in run.sessions:
        runner.close()


def static_conflict():
    return (rule("A"), rule("A", sign="neg"), rule("A", ML("x", "aux", "A", 0.8)))


def test_Gamma_conflict_is_isolated_but_saved_truth_and_explicit_arbitration_evidence_survive(
    tmp_path,
):
    ctx, manifest = setup(known=True)
    manifest["strategy"] = "frozen"
    rules = (rule("A", sign="neg"), rule("A", ML("x", "aux", "A", 0.8)))
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=forbidden_training,
    )
    report = run.run()
    assert report["status"] == "conflict_unknown_saturated"
    assert report["current_Gamma_has_quarantined_fixpoint"]
    assert not report["current_Gamma_has_valid_closure"]
    with np.load(run.output / "TRUSTED_GAMMA_FINAL.npz") as saved:
        np.testing.assert_array_equal(saved["positive"], ctx["gamma"].pos)
        np.testing.assert_array_equal(saved["negative"], ctx["gamma"].neg)
    with np.load(run.output / "FINAL_P.npz") as saved:
        assert saved["conflict_unknown"][0, 0]
        assert not saved["logical_positive"][0, 0] and not saved["logical_negative"][0, 0]
        assert saved["arbitration_negative"][0, 0] and not saved["remaining_unknown"][0, 0]
    evidence = json.loads((run.output / "chase_00000/ARBITRATION.json").read_text())
    assert evidence["stored_Gamma_unchanged"] and evidence["predictions_are_temporary"]
    assert len(evidence["instances"]) == 1
    item = evidence["instances"][0]
    assert item["rule_id"].startswith("arbitration:") and item["guard"] == "conflict_unknown"
    assert item["polarity"] == "neg" and item["probability"] == pytest.approx(0.2)
    assert item["kappa"] == 0.5 and item["temporary"]
    assert len(json.loads((run.output / "MODEL_ARBITRATION_RULES.json").read_text())) == 2 * len(
        ctx["labels"]
    )


def test_persistent_nonGamma_static_conflict_is_not_weak_supervision_or_an_infinite_loop(tmp_path):
    ctx, manifest = setup()
    run = LabelingRun(
        context=ctx,
        rules=static_conflict(),
        manifest=manifest,
        output=tmp_path / "run",
        trainer=forbidden_training,
    )
    report = run.run()
    assert report["status"] == "conflict_unknown_saturated"
    assert report["training_attempts"] == report["actual_optimizer_updates"] == 0
    assert report["actual_Chase_calls"] == 1
    assert not run.gamma.pos.any() and not run.gamma.neg.any()
    with np.load(run.output / "supervision_00000/ARRAYS.npz") as batch:
        assert not batch["weak_mask"].any() and not batch["observed"].any()


def temporary_case(tmp_path, candidate_check="checked", extra_conflict=False):
    ctx, manifest = setup()
    ctx["model_probabilities"]["aux"][0, 0] = 0.9
    manifest["feedback"]["candidate_check"] = candidate_check
    conditional = ML("x", "aux", "A", 0.8)
    rules = [
        rule("A", conditional),
        rule("A", conditional, sign="neg"),
        rule("B", Label("x", "A")),
        rule("B", ML("x", "aux", "B", 0.8)),
    ]
    if extra_conflict:
        rules += [rule("D"), rule("D", sign="neg"), rule("C"), rule("C", ML("x", "aux", "C", 0.8))]
    run = LabelingRun(
        context=ctx,
        rules=tuple(rules),
        manifest=manifest,
        output=tmp_path / "run",
        trainer=forbidden_training,
    )
    before, session, _ = run.chase(run.version, phase="initial_zero_update")
    assert before.conflict_unknown[0, 0] and before.arbitration_positive[0, 0]
    p = run.version.probabilities.copy()
    p[0, 0] = 0.6
    candidate = Version(
        p,
        dict(path="test-candidate", sha256="test-candidate"),
        run.version.channels,
        "synthetic-new-frozen-model",
    )
    after, new_session, _ = run.chase(
        candidate,
        phase="candidate_updated_model",
        previous=(before, session),
        update_kind="chaseDelta",
    )
    return run, candidate, before, after, new_session


def test_new_model_rearbitrates_old_unknown_and_temporary_only_descendants_do_not_return_as_logic(
    tmp_path,
):
    run, version, before, after, session = temporary_case(tmp_path)
    try:
        assert not after.conflict_unknown[0, 0]
        assert after.temporary_seed[0][0, 0] and not after.temporary_seed[1][0, 0]
        assert after.positive_facts[0, 0] and after.positive_facts[0, 1]
        assert after.proof_dag["pos:0:0"]["origin"] == "temporary_seed"
        ledger = json.loads((run.output / "chase_00001/SEED_ARBITRATION.json").read_text())
        assert ledger["persistent_Gamma_changed"] is False
        assert ledger["source"]["current_model_version"] == version.identity
        sources = ledger["source"]["instances"]
        assert len(sources) == 1 and sources[0]["rule_id"].startswith("arbitration:")
        assert sources[0]["row"] == 0 and sources[0]["label"] == "A"
        assert sources[0]["guard"] == "conflict_unknown" and sources[0]["temporary"]
        assert sources[0]["probability"] == pytest.approx(0.6) and sources[0]["kappa"] == 0.5
        assert sources[0]["polarity"] == "pos" and sources[0]["model_version"] == version.identity
        assert sources[0]["rule_id"] in {r.rule_id for r in run.arbitration_rules}
        # Current U is resolved, so its separate final arbitration can be empty;
        # the prior-U -> new-model seed still needs its complete rule evidence.
        final_arbitration = json.loads((run.output / "chase_00001/ARBITRATION.json").read_text())
        assert not final_arbitration["instances"]
        assert after.feedback_view is not None
        assert not after.feedback_view.positive_facts[0, :2].any()
        batch = run.feedback(after, version, session=session)
        assert not batch.weak_mask.any() and not batch.observed.any()
        assert batch.audit["arbitration_only_descendants_excluded"]
        assert not run.gamma.pos.any() and not run.gamma.neg.any()
        (run.output / "TEST_FEEDBACK_VIEW_OBSERVATION.json").write_text(
            json.dumps(
                dict(
                    scope="synthetic integration-test observation",
                    current_model_version=version.identity,
                    previous_unknown=before.conflict_unknown.tolist(),
                    current_unknown=after.conflict_unknown.tolist(),
                    temporary_positive=after.temporary_seed[0].tolist(),
                    logical_positive=after.positive_facts.tolist(),
                    feedback_positive=after.feedback_view.positive_facts.tolist(),
                    weak_mask=batch.weak_mask.tolist(),
                    observed=batch.observed.tolist(),
                    stored_Gamma_positive=run.gamma.pos.tolist(),
                    stored_Gamma_negative=run.gamma.neg.tolist(),
                ),
                indent=2,
            )
        )
    finally:
        close(run)


def test_checked_guard_fixes_unknown_and_temporary_roots_and_matches_NoCheck_supervision(tmp_path):
    outputs = []
    for policy in ("checked", "nocheck"):
        run, version, before, after, session = temporary_case(
            tmp_path / policy, policy, extra_conflict=True
        )
        try:
            assert after.conflict_unknown[0, 3] and after.positive_facts[0, 1]
            batch = run.feedback(after, version, session=session)
            assert batch.weak_mask[0, 2] and int(batch.weak_mask.sum()) == 1
            assert not batch.weak_mask[0, 1]  # B exists only via the temporary model seed.
            assert batch.audit["rejected_by_consistency_check"] == 0
            assert (
                batch.audit["candidate_check_role"]
                == "fixed_quarantine_and_temporary_seed_Gamma_union_W_guard"
            )
            outputs.append((batch, run.actual_chases))
        finally:
            close(run)
    for name in ("targets", "observed", "weights", "true_mask", "weak_mask"):
        np.testing.assert_array_equal(getattr(outputs[0][0], name), getattr(outputs[1][0], name))
    assert outputs[0][1] > outputs[1][1]


def test_arbitration_threshold_flip_is_part_of_model_state_and_commits_even_when_Sigma_decisions_do_not_change(
    tmp_path,
):
    ctx, manifest = setup(known=True)
    rules = (rule("A", sign="neg"), rule("A", ML("x", "aux", "A", 0.8)))
    calls = []

    def trainer(**kwargs):
        calls.append(kwargs["batch"])
        output = Path(kwargs["output"])
        output.mkdir()
        p = np.full((2, 4), 0.2, np.float32)
        p[0, 0] = 0.6
        return p, dict(
            optimizer_updates=1, checkpoint=dict(path="synthetic-model", sha256="synthetic-.6")
        )

    run = LabelingRun(
        context=ctx, rules=rules, manifest=manifest, output=tmp_path / "run", trainer=trainer
    )
    p = run.version.probabilities.copy()
    p[0, 0] = 0.6
    candidate = Version(
        p, dict(path="synthetic-model", sha256="synthetic-.6"), run.version.channels, "different"
    )
    assert all(
        np.array_equal(
            run.channel_tables(run.version)[o] >= threshold,
            run.channel_tables(candidate)[o] >= threshold,
        )
        for o, (_, threshold) in run.channels.items()
    )
    assert run.predicate_id(run.version) != run.predicate_id(candidate)
    report = run.run()
    assert report["committed_model_updates"] == 1
    assert report["status"] == "conflict_unknown_saturated"
    assert len(calls) == 2  # First commits the new arbitration state; the repeat stops.
    assert all(batch.true_mask[0, 0] and batch.targets[0, 0] == 1 for batch in calls)
    assert all(not batch.weak_mask.any() for batch in calls)
    with np.load(run.output / "FINAL_P.npz") as saved:
        assert saved["conflict_unknown"][0, 0] and saved["arbitration_positive"][0, 0]
        assert not saved["arbitration_negative"][0, 0]


def test_independent_strict_relation_contradiction_is_still_pending(tmp_path):
    ctx, manifest = setup()
    ctx["gamma"] = RefGamma(ctx["gamma"].pos, ctx["gamma"].neg, strict_subset=((0, 0),))
    run = LabelingRun(
        context=ctx,
        rules=static_conflict(),
        manifest=manifest,
        output=tmp_path / "run",
        trainer=forbidden_training,
    )
    report = run.run()
    assert report["status"] == "pending_initial_conflict"
    assert report["training_attempts"] == 0
    assert not report["current_Gamma_has_quarantined_fixpoint"]
    assert any(e["kind"] == "Chase" and e["status"] == "conflict" for e in run.events)
    assert not (run.output / "FINAL_P.npz").exists()


def test_REAL_Gamma_target_survives_quarantine_and_opposite_arbitration_without_replay(tmp_path):
    import torch
    from tests.test_labeling_actual_aux import assets, Encoder, Tokenizer

    source, parent, configuration = assets(tmp_path)
    ctx, manifest = setup(known=True)
    model = Encoder(4)
    with torch.no_grad():
        model.classifier_weight.zero_()
        model.classifier_bias.fill_(math.log(0.2 / 0.8))
    initial_bias = float(model.classifier_bias[0].detach())
    torch.save(
        dict(
            state_dict=model.state_dict(),
            label_names=ctx["labels"],
            model_name=configuration["model_name"],
            head=configuration["head"],
        ),
        str(parent),
    )
    configuration.update(prefer_ema=False, batch_size=1, grad_accum_steps=1)
    manifest["dynamic"].update(
        checkpoint=file_record(parent), backend_source=source, configuration=configuration
    )
    calls, gradients = [], []

    def factory(_, count):
        current = Encoder(count)
        current.classifier_bias.register_hook(
            lambda g: gradients.append(g.detach().cpu().numpy().copy())
        )
        return current

    def actual(**kwargs):
        calls.append(kwargs["batch"])
        return train_actual(
            **kwargs, tokenizer_factory=lambda _: Tokenizer(), model_factory=factory
        )

    rules = (rule("A", sign="neg"), rule("A", ML("x", "aux", "A", 0.8)))
    run = LabelingRun(
        context=ctx, rules=rules, manifest=manifest, output=tmp_path / "run", trainer=actual
    )
    report = run.run()
    assert len(calls) == report["actual_optimizer_updates"] == 1
    batch = calls[0]
    assert batch.true_mask[0, 0] and batch.observed[0, 0] and batch.targets[0, 0] == 1
    assert batch.weights[0, 0] == 1 and not batch.weak_mask.any()
    assert int(batch.observed.sum()) == 1
    assert gradients and gradients[0][0] == pytest.approx(-0.8, abs=1e-5)
    assert np.allclose(gradients[0][1:], 0)
    trace = json.loads((run.output / "update_00001/ACTUAL_TRAINING_TRACE.json").read_text())
    assert trace["sampled_rows"] == [0] and trace["successful_optimizer_updates"] == 1
    assert trace["windows"][0]["true_mass"] == 1 and trace["windows"][0]["weak_mass"] == 0
    receipt = json.loads((run.output / "update_00001/COMPLETE.json").read_text())
    selected = torch.load(receipt["checkpoint"]["path"], map_location="cpu")
    assert float(selected["state_dict"]["classifier_bias"][0]) > initial_bias
    (run.output / "TEST_REAL_GRADIENT_OBSERVATION.json").write_text(
        json.dumps(
            dict(
                scope="synthetic real CPU optimization, not an empirical dataset experiment",
                torch_version=torch.__version__,
                initial_Aux_probability=0.2,
                stored_Gamma_target=1,
                arbitration_target=0,
                trusted_replay=False,
                observed_bias_gradients=[g.tolist() for g in gradients],
                observed_training_targets=batch.targets.tolist(),
                observed_mask=batch.observed.tolist(),
                true_mask=batch.true_mask.tolist(),
                weak_mask=batch.weak_mask.tolist(),
                initial_bias=initial_bias,
                selected_bias=float(selected["state_dict"]["classifier_bias"][0]),
                checkpoint=receipt["checkpoint"],
                actual_optimizer_updates=report["actual_optimizer_updates"],
                committed_model_updates=report["committed_model_updates"],
                status=report["status"],
            ),
            indent=2,
        )
    )
    with np.load(run.output / "TRUSTED_GAMMA_FINAL.npz") as saved:
        assert saved["positive"][0, 0] and not saved["negative"].any()
    with np.load(run.output / "FINAL_P.npz") as saved:
        assert saved["arbitration_negative"][0, 0]  # It never became the training target.
