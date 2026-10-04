"""Author weight equations, complete derivations, and strict artifact handoff."""

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
from loris.labeling.feedback import assignments
from loris.labeling.reliability import (
    SCHEMA,
    UNIT,
    ESTIMATOR,
    TRANSFER_POLICY,
    rule_score,
    lineage_scores,
)
from loris._engine.rdl_reference import (
    Doc,
    Label,
    LabelCompare,
    Rule,
    Text,
    ReferenceChase,
    RefGamma,
)
from loris._engine.typed_session import digest
from loris.predicates import MatchPredicate
from tests.test_labeling import fixture


def artifact_for(rules=(), counts=()):
    return dict(
        schema=SCHEMA,
        source_unit=UNIT,
        estimator=copy.deepcopy(ESTIMATOR),
        transfer_policy=TRANSFER_POLICY,
        rule_scores=[
            dict(
                rule_id=i,
                rule_sha256=digest(rule.to_dict()),
                support=n,
                correct=c,
                q=float(rule_score(n, c)),
            )
            for i, (rule, (n, c)) in enumerate(zip(rules, counts))
        ],
        binding={"fixture": "independent_calibration"},
    )


def evidence():
    gp = np.zeros((5, 2), bool)
    gn = gp.copy()
    gp[0, 0] = True
    gn[0, 1] = True
    pos = gp.copy()
    neg = gn.copy()
    pos[1, 0] = True
    neg[2, 0] = True
    pos[3, 1] = True
    closure = SimpleNamespace(status="fixpoint", positive_facts=pos, negative_facts=neg)
    p = np.full((5, 2), 0.2)
    p[2, 0] = 0.8
    p[3, 1] = 0.1
    p[4] = 0.99
    rho = np.zeros_like(p)
    rho[0] = 1
    rho[1, 0] = 0.8
    rho[2, 0] = 0.8
    rho[3, 1] = 0.5
    return dict(
        gamma=GammaSeed(gp, gn),
        closure=closure,
        independent=closure,
        probabilities=p,
        channels={(0, 1): (0, 0.5), (1, 1): (1, 0.5)},
        rows=np.ones(5, bool),
        strategy="paper",
        reliability=artifact_for(),
        reliability_scores=rho,
    )


@pytest.mark.parametrize(
    "policy,positive,negative,prior",
    [
        ("uniform", 1.0, 1.0, 1.0),
        ("confidence_only", 0.2, 0.2, 0.1),
        ("reliability_only", 0.8, 0.8, 0.5),
        ("provenance", 0.48, 0.48, 0.275),
    ],
)
def test_author_weight_examples_both_signs_true_roots_and_unknowns(
    policy, positive, negative, prior
):
    data = evidence()
    batch = assignments(**data, weighting=policy)
    assert np.allclose(batch.weights[0], [1, 1])
    assert batch.weights[1, 0] == pytest.approx(positive)
    assert batch.weights[2, 0] == pytest.approx(negative)
    assert batch.weights[3, 1] == pytest.approx(prior)
    assert not batch.observed[4].any() and not batch.weights[4].any()
    assert batch.targets[1, 0] == 1 and batch.targets[2, 0] == 0
    assert batch.audit["quality_is_not_proved_by_provenance"]
    assert batch.audit["disagreement_discount"] == 0.5
    assert batch.audit["unsupported_reliability"] == 0
    if policy in ("reliability_only", "provenance"):
        assert batch.audit["reliability_base_binding"] == data["reliability"]["binding"]
        assert batch.audit["reliability_prior_policy"] == TRANSFER_POLICY


def test_discount_is_explicit_ablation_and_full_proof_scores_required():
    data = evidence()
    assert assignments(**data, weighting="provenance", disagreement_discount=0).weights[
        1, 0
    ] == pytest.approx(0.8)
    for discount in (-1, 2, float("nan"), True):
        with pytest.raises(ValueError, match="discount"):
            assignments(**data, weighting="provenance", disagreement_discount=discount)
    data["reliability_scores"] = None
    with pytest.raises(ValueError, match="complete max-min"):
        assignments(**data, weighting="provenance")
    data = evidence()
    data["reliability"] = {"schema": "loris_fit_reliability_v1", "rho": [[1, 1], [1, 1]]}
    with pytest.raises(ValueError, match="historical fit pooled"):
        assignments(**data, weighting="provenance")


def test_receiver_exclusion_is_explicit_ablation_and_selftrain_has_no_rule_rho():
    data = evidence()
    data["independent"] = None
    assert assignments(**data, weighting="uniform").weak_mask.any()
    assert not assignments(**data, weighting="uniform", require_independent=True).weak_mask.any()
    for mode in ("provenance", "reliability_only"):
        data = evidence()
        data["strategy"] = "self_train"
        with pytest.raises(ValueError, match="SelfTrain predictions"):
            assignments(**data, weighting=mode)


def test_max_min_alternate_derivation_cycle_and_zero_support_prior():
    labels = ["A", "B", "C", "D"]
    docs = [Document(cnt="red", ttl="", mtd="")]
    text = Text("x", MatchPredicate("cnt", "red"))
    rules = (
        Rule((Doc("x"), text), Label("x", "B")),  # First witness is the less reliable direct rule.
        Rule((Doc("x"), text), Label("x", "A")),
        Rule((Doc("x"), Label("x", "A")), Label("x", "B")),
        Rule((Doc("x"), Label("x", "B")), Label("x", "C")),
        Rule((Doc("x"), Label("x", "C")), Label("x", "B")),
        Rule((Doc("x"), Label("x", "D")), Label("x", "D")),
    )
    counts = [(8, 5), (18, 17), (18, 15), (18, 18), (18, 18), (98, 98)]
    artifact = artifact_for(rules, counts)
    gamma = RefGamma.empty(1, 4)
    reference = ReferenceChase(labels)

    def chase(program, seed):
        return reference.run(docs, program, seed)

    full = chase(rules, gamma)
    assert full.proof_dag["pos:0:1"]["rule_id"] == 0
    rho, audit = lineage_scores(chase, rules, gamma, artifact)
    assert np.allclose(rho, [[0.9, 0.8, 0.8, 0.0]])
    assert audit["all_derivations"] and not audit["first_witness_only"]
    without_chain, _ = lineage_scores(chase, rules, gamma, artifact, excluded_rule_ids=(2,))
    assert without_chain[0, 1] == pytest.approx(0.6)
    assert rule_score(0, 0) == 0.5


def test_builtin_comparison_propagation_keeps_empty_proof_weight_one():
    labels = ["A"]
    docs = [Document(cnt="", ttl="", mtd="") for _ in range(2)]
    gp = np.array([[True], [False]])
    gamma = RefGamma(gp, np.zeros_like(gp), ((0, 1),), ())
    reference = ReferenceChase(labels)
    rho, audit = lineage_scores(
        lambda rules, seed: reference.run(docs, rules, seed), (), gamma, artifact_for()
    )
    assert np.array_equal(rho, np.ones((2, 1))) and audit["closure_calls"] == 1


def runtime_manifest(tmp_path):
    ctx, rules, data = fixture()
    data["loop"]["saturation_rounds"] = 1
    context = tmp_path / "context.pkl"
    with context.open("wb") as f:
        pickle.dump(ctx, f)
    program = tmp_path / "program.json"
    program.write_text(json.dumps([r.to_dict() for r in rules]))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"No model loading in input-only test")
    backend = tmp_path / "backend"
    (backend / "loris_ssl").mkdir(parents=True)
    (backend / "loris_ssl/chase_backend.py").write_text("# Input-only fixture\n")
    data["context"] = file_record(context)
    data["program"] = file_record(program)
    data["dynamic"]["checkpoint"] = file_record(checkpoint)
    data["dynamic"]["backend_source"] = str(backend)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    return manifest, data


def test_cli_requires_independent_artifact_and_explicit_protocol(tmp_path):
    manifest, data = runtime_manifest(tmp_path)
    artifact = tmp_path / "rho.json"
    artifact.write_text(json.dumps({"schema": "loris_fit_reliability_v1"}))
    with pytest.raises(ValueError, match="historical fit pooled"):
        cli.load(manifest, weighting="provenance", source_reliability=artifact)
    data["weighting"] = "reliability_only"
    with pytest.raises(ValueError, match="independent per-rule"):
        cli.validate(data)
    for args in (
        ["calibrate-reliability", "--manifest", "m", "--output", "o"],
        ["fit-reliability"],
    ):
        with pytest.raises(SystemExit):
            cli.main(args)
    data["weighting"] = "uniform"
    data["feedback"].update(source_filter="exclude_receiver", disagreement_discount=0.25)
    cli.validate(data)


def test_explicit_ensemble_transfer_uses_own_prediction_but_weight_uses_aux():
    data = evidence()
    data["receiver_policy"] = "affected_channels_to_aux"
    tables = {occ: np.ones_like(data["probabilities"]) for occ in data["channels"]}
    # Aux disagrees with A=1, but the actual ensemble occurrence agrees: no correction.
    b = assignments(**data, weighting="provenance", probabilities_by_channel=tables)
    assert not b.weak_mask[1, 0]
    tables[(0, 1)][:, 0] = 0.1
    # Ensemble disagreement identifies the correction; its weight still uses pre-update Aux p=.2.
    b = assignments(**data, weighting="provenance", probabilities_by_channel=tables)
    assert b.weak_mask[1, 0] and b.weights[1, 0] == pytest.approx(0.48)
    with pytest.raises(ValueError, match="every actual dynamic channel"):
        assignments(**data, weighting="uniform", probabilities_by_channel={})


def test_threshold_calibration_compares_each_declared_model_table():
    from loris.labeling.feedback import Batch, calibrate

    targets = np.array([[1.0], [0.0]], np.float32)
    known = np.ones((2, 1), bool)
    batch = Batch(targets, known, np.ones_like(targets), known, ~known, {})
    aux = np.array([[0.9], [0.1]])
    ensemble = np.array([[0.6], [0.4]])
    channels = {(0, 1): (0, 0.8)}
    selected, receipt = calibrate(
        aux,
        batch,
        channels,
        grid=[0.5, 0.8],
        minimum_support_each_sign=1,
        probabilities_by_channel={(0, 1): ensemble},
    )
    assert selected[(0, 1)] == (0, 0.5)
    assert receipt["probability_tables"] == "declared_occurrence_model"


def test_selftrain_uses_own_confident_prediction_even_for_logically_resolved_label():
    data = evidence()
    data["strategy"] = "self_train"
    data["probabilities"][
        1, 0
    ] = 0.01  # Logic says positive; own confident pseudo label is negative.
    data["probabilities"][
        2, 0
    ] = 0.99  # Logic says negative; own confident pseudo label is positive.
    data["probabilities"][0] = 0.99  # Trusted Gamma must remain untouched.
    batch = assignments(**data, weighting="uniform")
    assert batch.weak_mask[1, 0] and batch.targets[1, 0] == 0
    assert batch.weak_mask[2, 0] and batch.targets[2, 0] == 1
    assert np.array_equal(batch.targets[0], [1, 0]) and not batch.weak_mask[0].any()
    assert not batch.weak_mask[3].any()  # .1 is below the configured confidence cutoff.


def test_disagreeing_Gamma_corrections_survive_disabled_additional_replay():
    data = evidence()
    batch = assignments(**data, weighting="provenance", replay=False)
    # Trusted positive A disagrees with .2 prediction, so remains a correction.
    assert batch.true_mask[0, 0] and batch.targets[0, 0] == 1 and batch.weights[0, 0] == 1
    # Trusted negative B agrees, and extra replay was disabled.
    assert not batch.true_mask[0, 1] and not batch.observed[0, 1]
    assert batch.weak_mask[1, 0] and batch.weights[1, 0] == pytest.approx(0.48)
    assert not batch.observed[4].any()  # Unknowns never become verified corrections.
    assert batch.audit["trusted_correction_cells"] == 1
    assert batch.audit["additional_trusted_replay_cells"] == 0
    replay = assignments(**data, weighting="provenance", replay=True)
    assert replay.true_mask[0].all() and replay.audit["additional_trusted_replay_cells"] == 1


def test_trusted_replay_fraction_is_deterministic_and_keeps_corrections():
    data = evidence()
    data["strategy"] = "supervised"
    positive = data["gamma"].pos.copy()
    positive[3, 1] = True
    data["gamma"] = GammaSeed(positive, data["gamma"].neg.copy())
    full = assignments(**data, weighting="uniform", replay=True)
    partial = assignments(**data, weighting="uniform", replay=True, replay_fraction=0.5)
    assert full.true_mask[0, 0] and full.true_mask[3, 1]
    assert partial.true_mask[0, 0] and not partial.true_mask[3, 1]
    assert partial.audit["trusted_replay_fraction"] == 0.5
    assert partial.audit["additional_trusted_replay_cells"] == 2
