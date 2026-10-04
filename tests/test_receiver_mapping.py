"""Receiver ownership is explicit, and selected corrections drive REAL gradients.

Synthetic CPU software fixtures only. The optional real backend tests use an
Aux checkpoint whose initial probabilities exactly match the declared table.
Uniform weights isolate receiver routing from rule precision calibration.
"""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, ML, Rule, Text, RefGamma
from loris.discovery.manifest import file_record
from loris.document import Document
from loris.labeling import cli
from loris.labeling.feedback import assignments
from loris.labeling.runner import LabelingRun
from loris.labeling.trainer import train_actual
from loris.predicates import MatchPredicate
from tests.test_labeling import fixture
from tests.test_labeling_reliability_weights import runtime_manifest


def signed_inputs():
    positive = np.array([[True], [False], [False]])
    negative = np.array([[False], [True], [False]])
    probabilities = np.array([[0.8], [0.2], [0.8]])
    return dict(
        gamma=RefGamma.empty(3, 1),
        closure=SimpleNamespace(
            status="fixpoint", positive_facts=positive, negative_facts=negative
        ),
        independent=None,
        probabilities=probabilities,
        channels={(0, 1): (0, 0.5), (1, 1): (0, 0.5)},
        rows=np.ones(3, bool),
        strategy="paper",
        weighting="uniform",
        replay=False,
        probabilities_by_channel={
            (0, 1): probabilities.copy(),
            (1, 1): np.array([[1 / 3], [0.8], [1 / 3]]),
        },
    )


def test_same_model_default_ignores_ensemble_disagreement_when_Aux_agrees():
    data = signed_inputs()
    same = assignments(**data, receiver_occurrences={(0, 1)})
    assert not same.weak_mask.any() and not same.observed.any()
    transfer = assignments(
        **data, receiver_occurrences={(0, 1)}, receiver_policy="affected_channels_to_aux"
    )
    assert np.array_equal(transfer.weak_mask[:, 0], [True, True, False])
    assert transfer.targets[0, 0] == 1 and transfer.targets[1, 0] == 0
    assert transfer.weights[0, 0] == transfer.weights[1, 0] == 1


def test_multiple_actual_Aux_thresholds_are_used_without_synthetic_threshold():
    data = signed_inputs()
    data["channels"][(2, 3)] = (0, 0.9)
    data["probabilities_by_channel"][(2, 3)] = data["probabilities"].copy()
    same = assignments(**data, receiver_occurrences={(0, 1), (2, 3)}, receiver_policy="same_model")
    # Aux .8 agrees at its actual .5 occurrence but disagrees at its actual .9.
    assert np.array_equal(same.weak_mask[:, 0], [True, False, False])
    assert same.targets[0, 0] == 1
    del data["channels"][(2, 3)]
    del data["probabilities_by_channel"][(2, 3)]
    removed = assignments(**data, receiver_occurrences={(0, 1)})
    assert not removed.weak_mask.any()


def test_negative_correction_uses_own_Aux_decision_and_unknown_stays_unobserved():
    data = signed_inputs()
    data["probabilities"][1, 0] = 0.8
    data["probabilities_by_channel"][(0, 1)] = data["probabilities"].copy()
    same = assignments(**data, receiver_occurrences={(0, 1)})
    assert np.array_equal(same.weak_mask[:, 0], [False, True, False])
    assert same.targets[1, 0] == 0 and same.weights[1, 0] == 1
    assert not same.observed[2, 0]


def test_only_ensemble_occurrences_do_not_invent_an_Aux_predicate():
    data = signed_inputs()
    data["channels"] = {(1, 1): (0, 0.5)}
    data["probabilities_by_channel"] = {(1, 1): data["probabilities_by_channel"][(1, 1)]}
    same = assignments(**data, receiver_occurrences=set())
    assert not same.weak_mask.any() and not same.observed.any()
    transfer = assignments(
        **data, receiver_occurrences=set(), receiver_policy="affected_channels_to_aux"
    )
    assert transfer.weak_mask[:2].all() and not transfer.weak_mask[2].any()


def test_foreign_probability_tables_require_explicit_occurrence_ownership():
    data = signed_inputs()
    with pytest.raises(ValueError):
        assignments(**data)  # mixed model tables without receiver_occurrences
    with pytest.raises(ValueError):
        assignments(**data, receiver_occurrences={(99, 99)})
    # Without per-model tables the low-level API explicitly means all given
    # occurrences belong to the receiver; it does not guess aliases by values.
    data["probabilities_by_channel"] = None
    assert not assignments(**data).weak_mask.any()


def test_public_cli_default_and_explicit_receiver_extension(tmp_path):
    manifest, _ = runtime_manifest(tmp_path)
    current, _, _, _, _ = cli.load(manifest)
    assert current["feedback"].get("receiver_policy", "same_model") == "same_model"
    transfer, _, _, _, _ = cli.load(manifest, receiver_policy="affected_channels_to_aux")
    assert transfer["feedback"]["receiver_policy"] == "affected_channels_to_aux"
    assert cli.main(["check", "--manifest", str(manifest), "--receiver-policy", "same_model"]) == 0
    invalid = copy.deepcopy(current)
    invalid["feedback"]["receiver_policy"] = "auto_guess"
    with pytest.raises(ValueError):
        cli.validate(invalid)


def real_case(tmp_path, *, aux_probability, receiver_policy, include_aux=True):
    import torch
    from tests.test_labeling_actual_aux import assets, Encoder, Tokenizer

    source, parent, configuration = assets(tmp_path)
    ctx, _, manifest = fixture()
    ctx.update(
        docs=[Document(cnt="red", ttl="", mtd=""), Document(cnt="blue", ttl="", mtd="")],
        labels=["A"],
        row_ids=["r0", "r1"],
        row_roles=["train", "train"],
        gamma=RefGamma.empty(2, 1),
        receiver_indices=np.array([], dtype=int),
        truth=np.empty((0, 1), bool),
        model_decisions={},
        model_probabilities={
            "aux": np.full((2, 1), aux_probability, np.float32),
            "fixed1": np.full((2, 1), 0.1, np.float32),
            "fixed2": np.full((2, 1), 0.2, np.float32),
        },
    )
    ensemble_value = 1 / 3 if aux_probability >= 0.5 else 0.0
    ctx["model_probabilities"]["ensemble"] = np.full((2, 1), ensemble_value, np.float32)
    red = Text("x", MatchPredicate("cnt", "red"))
    rules = [Rule((Doc("x"), red), Label("x", "A"))]
    if include_aux:
        rules.append(Rule((Doc("x"), red, ML("x", "aux", "A", 0.5)), Label("x", "A")))
    rules.append(Rule((Doc("x"), red, ML("x", "ensemble", "A", 0.5)), Label("x", "A")))
    model = Encoder(1)
    with torch.no_grad():
        model.classifier_weight.zero_()
        model.classifier_bias.fill_(math.log(aux_probability / (1 - aux_probability)))
    initial_bias = float(model.classifier_bias.detach()[0])
    assert float(torch.sigmoid(model.classifier_bias.detach())[0]) == pytest.approx(aux_probability)
    torch.save(
        dict(
            state_dict=model.state_dict(),
            label_names=["A"],
            model_name=configuration["model_name"],
            head=configuration["head"],
        ),
        str(parent),
    )
    configuration.update(prefer_ema=False, batch_size=1, grad_accum_steps=1)
    manifest["dynamic"].update(
        checkpoint=file_record(parent),
        backend_source=source,
        configuration=configuration,
        ensemble_dependencies=[
            dict(
                alias="ensemble",
                dependencies=["aux", "fixed1", "fixed2"],
                operation="binary_vote_mean",
                vote_threshold=0.5,
            )
        ],
    )
    manifest["domains"].update(training_roles=["train"], query_roles=["train"], scoring_roles=[])
    manifest["feedback"].update(trusted_replay=False, receiver_policy=receiver_policy)
    manifest["weighting"] = "uniform"
    calls = []
    gradients = []

    def factory(_, count):
        model = Encoder(count)
        model.classifier_bias.register_hook(
            lambda gradient: gradients.append(gradient.detach().cpu().numpy().copy())
        )
        return model

    def actual(**kwargs):
        call = dict(batch=kwargs["batch"], output=Path(kwargs["output"]))
        calls.append(call)
        p, receipt = train_actual(
            **kwargs, tokenizer_factory=lambda _: Tokenizer(), model_factory=factory
        )
        call["receipt"] = receipt
        return p, receipt

    run = LabelingRun(
        context=ctx, rules=tuple(rules), manifest=manifest, output=tmp_path / "loop", trainer=actual
    )
    report = run.run()
    return dict(
        run=run,
        report=report,
        calls=calls,
        gradients=gradients,
        initial_bias=initial_bias,
        aux_probability=aux_probability,
        ensemble_probability=ensemble_value,
    )


@pytest.mark.parametrize(
    "aux_probability,policy,include_aux,should_train",
    [
        (0.8, "same_model", True, False),
        (0.8, "affected_channels_to_aux", True, True),
        (0.2, "same_model", True, True),
        (0.8, "same_model", False, False),
    ],
)
def test_REAL_Aux_gradient_matches_declared_receiver_mapping(
    tmp_path, aux_probability, policy, include_aux, should_train
):
    import torch

    result = real_case(
        tmp_path, aux_probability=aux_probability, receiver_policy=policy, include_aux=include_aux
    )
    report = result["report"]
    assert report["current_Gamma_has_valid_closure"]
    if not should_train:
        assert report["training_attempts"] == 0 and report["actual_optimizer_updates"] == 0
        assert not result["calls"] and not result["gradients"]
        assert not list((tmp_path / "loop").glob("update_*"))
        return
    assert len(result["calls"]) == 1 and report["actual_optimizer_updates"] == 1
    call = result["calls"][0]
    batch = call["batch"]
    assert np.array_equal(batch.observed, [[True], [False]])
    assert batch.weak_mask[0, 0] and batch.targets[0, 0] == 1 and batch.weights[0, 0] == 1
    assert result["gradients"] and result["gradients"][0][0] < 0
    assert abs(float(result["gradients"][0][0])) > 1e-6
    payload = torch.load(call["receipt"]["checkpoint"]["path"], map_location="cpu")
    assert float(payload["state_dict"]["classifier_bias"][0]) > result["initial_bias"]
    assert call["receipt"]["parameter_state_changed"]
    trace = json.loads((call["output"] / "ACTUAL_TRAINING_TRACE.json").read_text())
    assert trace["sampled_rows"] == [0] and trace["successful_optimizer_updates"] == 1
    assert trace["windows"][0]["true_mass"] == 0 and trace["windows"][0]["weak_mass"] == 1
