"""Small actual fit-selection/BO/Chase controls; no production experiments."""

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from loris.discovery import model_selection
from loris.discovery.__main__ import main
from loris.discovery.archive import archive_index, status
from loris.discovery.manifest import file_record, load_inputs
from loris.discovery.runner import run_discovery
from loris.discovery.search.domain import build_config_space
from loris.discovery.search.admission import order_trials
from loris._engine.rdl_reference import Doc, Label, LabelCompare, Text, AttrEqual
from loris._engine.rlearner import array_identity, digest
from loris._engine.rlearner_config import ModelSpec, ParameterDomain, InvalidConfiguration
from loris.discovery.upstream import DEFAULT_SELECTOR_PROTOCOL
from tests.test_discovery_runner import inputs, semantic_trials


def selection_fixture(tmp_path):
    kw = inputs()
    ctx = kw["context"]
    perfect = ctx["model_probabilities"]["m"].copy()
    aliases = ["z_best", "a_bad", "m_second"]
    ctx["model_probabilities"] = {
        aliases[0]: perfect,
        aliases[1]: 1 - perfect,
        aliases[2]: np.zeros_like(perfect) + 0.1,
    }
    ctx["model_probabilities"][aliases[2]][:, 0] = perfect[:, 0]
    ctx["model_decisions"] = {a: p >= 0.5 for a, p in ctx["model_probabilities"].items()}
    fit = ctx["fit_indices"]
    arrays = dict(
        features=np.ones((len(fit), 4), np.float32),
        truth=ctx["source_gold"]["fit"],
        cluster_assignments=np.zeros(len(fit), np.int64),
        decisions=np.stack([ctx["model_decisions"][a][fit] for a in aliases]),
    )
    interpretation = dict(row_ids=ctx["row_ids"], probabilities={}, decisions={})
    for i, alias in enumerate(aliases):
        for kind, prefix in [("probabilities", "p"), ("decisions", "d")]:
            key = prefix + str(i)
            arrays[key] = ctx["model_" + kind][alias]
            interpretation[kind][alias] = key
    tmp_path.mkdir(parents=True, exist_ok=True)
    array_path = tmp_path / "INPUT.npz"
    np.savez_compressed(array_path, **arrays)
    path = tmp_path / "select.manifest.json"
    definition = dict(
        arrays=file_record(array_path),
        row_ids=[ctx["row_ids"][i] for i in fit],
        aliases=aliases,
        labels=ctx["labels"],
        model_interpretation=interpretation,
        protocol=dict(
            DEFAULT_SELECTOR_PROTOCOL, k=2, epochs=1, num_samples=2, hidden_dim=4, device="cpu"
        ),
        input_binding=dict(
            role="fit",
            expected_fit_rows=len(fit),
            feature_provenance={"fixture": "features"},
            decision_provenance={"fixture": "frozen probability >= .5"},
        ),
    )
    path.write_text(json.dumps(definition))
    return path, ctx, kw["patterns"]["0"], arrays


def discovery_manifest(tmp_path, strategy="indiv"):
    manifest, context, patterns, arrays = selection_fixture(tmp_path)
    output = tmp_path / "selection"
    model_selection.select_models(manifest, output, strategy=strategy)
    context_path = tmp_path / "context.pkl"
    context_path.write_bytes(pickle.dumps(context))
    pattern_path = tmp_path / "patterns.json"
    pattern_path.write_text(json.dumps({"0": [p.to_dict() for p in patterns]}))
    fit = context["fit_indices"]
    identity = dict(
        fit_row_ids=[context["row_ids"][i] for i in fit],
        labels=context["labels"],
        positive=array_identity(context["gamma"].pos[fit]),
        negative=array_identity(context["gamma"].neg[fit]),
        clusters=array_identity(arrays["cluster_assignments"]),
        texts={
            a: digest([context["docs"][int(i)].get_attr(a) for i in fit]) for a in ("cnt", "ttl")
        },
    )
    (tmp_path / "FIT_IDENTITY.json").write_text(json.dumps(identity))
    supply = tmp_path / "SUPPLY.json"
    supply.write_text(json.dumps(dict(status="complete", patterns=file_record(pattern_path))))
    auxiliary = tmp_path / "EXPOSURE.json"
    auxiliary.write_text(json.dumps({"fixture": True, "all_permitted_labels_disclosed": True}))
    result = dict(
        schema="loris_rich_discovery_manifest_v1",
        dataset=dict(
            name="synthetic6",
            split_protocol=file_record(auxiliary),
            scope="train_dev_only",
            roots_contract="complete_signed_known_train",
        ),
        context=file_record(context_path),
        model_selection=file_record(output / "SELECTED.json"),
        pattern_supply=file_record(supply),
        supervision_exposure=file_record(auxiliary),
        relation_attributes=["mtd.group"],
        oracle_policy="accuracy_then_vote_bce",
        run=dict(
            seed=42, budget_per_cluster=4, workers=1, initializer_slots=2, label_rotation="axis"
        ),
    )
    path = tmp_path / "discover.manifest.json"
    path.write_text(json.dumps(result))
    return path


@pytest.mark.parametrize("strategy", ["indiv", "random", "all"])
def test_fit_controls_authenticate_real_arrays_without_training(tmp_path, strategy):
    manifest, context, _, _ = selection_fixture(tmp_path)
    out = tmp_path / "chosen"
    assert (
        main(
            [
                "select-models",
                "--manifest",
                str(manifest),
                "--output",
                str(out),
                "--strategy",
                strategy,
            ]
        )
        == 0
    )
    verified = model_selection.verify_selection_output(out / "SELECTED.json")
    assert (
        model_selection.verify_selection_output(out / "COMPLETE.json")["result"]
        == verified["result"]
    )
    selected = verified["result"]
    assert selected["completed_optimizer_steps"] == 0 and selected["new_BO"] == 0
    assert selected["effective_k"] == (3 if strategy == "all" else 2)
    if strategy == "indiv":
        assert selected["selected_aliases_by_cluster"]["0"] == ["z_best", "m_second"]
    assert np.array_equal(
        verified["model_interpretation"]["probabilities"]["z_best"],
        context["model_probabilities"]["z_best"],
    )
    assert not (out / "CHECKPOINT.pt").exists()
    with pytest.raises(FileExistsError):
        model_selection.select_models(manifest, out, strategy=strategy)
    raw = json.loads((out / "SELECTED.json").read_text())
    raw["selected_aliases_by_cluster"]["0"] = ["a_bad"]
    (out / "SELECTED.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        model_selection.verify_selection_output(out / "SELECTED.json")


def test_random_is_seeded_and_cluster_local_indiv_never_reads_control_truth():
    arrays = dict(
        assignments=np.array([0, 0, 1, 1]), truth=np.array([[1, 0], [0, 1], [1, 0], [0, 1]], bool)
    )
    truth = arrays["truth"]
    arrays["decisions"] = np.stack([truth.copy(), ~truth, np.zeros_like(truth)])
    arrays["decisions"][0, 2:] = ~truth[2:]
    arrays["decisions"][1, 2:] = truth[2:]
    value = model_selection._control(arrays, ["z", "a", "b"], ["A", "B"], 1, 42, "indiv")
    assert value["selected_aliases_by_cluster"] == {"0": ["z"], "1": ["a"]}
    assert model_selection._control(
        arrays, ["z", "a", "b"], ["A", "B"], 2, 42, "random"
    ) == model_selection._control(arrays, ["z", "a", "b"], ["A", "B"], 2, 42, "random")


def test_learned_switch_runs_actual_existing_cpu_selector(tmp_path):
    manifest, _, _, _ = selection_fixture(tmp_path)
    out = tmp_path / "learned"
    assert (
        main(
            [
                "select-models",
                "--strategy",
                "learned",
                "--manifest",
                str(manifest),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    verified = model_selection.verify_selection_output(out / "SELECTED.json")
    assert verified["result"]["completed_optimizer_steps"] == 1
    assert (
        model_selection.verify_selection_output(out / "COMPLETE.json")["result"]
        == verified["result"]
    )
    assert (out / "PROTOCOL.json").exists()


def test_manifest_strategy_override_and_domain_flags_drive_real_discovery(tmp_path, capsys):
    path = discovery_manifest(tmp_path)
    a = load_inputs(path)
    assert a["si"] == {"0": ["z_best", "m_second"]}
    assert a["admission_order"] == "standalone_score_desc"
    b = load_inputs(path, {"selection_strategy": "all", "candidate_domain": "no_text"})
    assert len(b["si"]["0"]) == 3 and b["candidate_domain"] == "no_text"
    assert (
        b["input_binding"]["effective_manifest_id"] != a["input_binding"]["effective_manifest_id"]
    )
    with pytest.raises(ValueError, match="actually trained"):
        load_inputs(path, {"selection_strategy": "learned"})
    out = tmp_path / "actual"
    assert (
        main(
            [
                "learn-rules",
                "--manifest",
                str(path),
                "--output",
                str(out),
                "--model-selection",
                "indiv",
                "--candidate-domain",
                "local",
                "--admission-order",
                "standalone_score_desc",
            ]
        )
        == 0
    )
    assert (out / "COMPLETE.json").exists()
    capsys.readouterr()
    records = [
        json.loads(p.read_text())
        for p in sorted((out / "learning/cluster_00").glob("trial_*.json"))
    ]
    for r in records:
        if r["rule"] is not None:
            from loris._engine.rdl_reference import Rule

            rule = Rule.from_dict(r["rule"])
            assert isinstance(rule.head, Label)
            assert not any(isinstance(atom, (Label, LabelCompare, AttrEqual)) for atom in rule.body)
            assert all(getattr(atom, "var", "x") == "x" for atom in rule.body)
    assert status(out)["status"] == "complete_verified_identity"
    assert archive_index(out, tmp_path / "archive")["clusters"][0][
        "all_sequential_admissions_verified"
    ]


@pytest.mark.parametrize("domain", ["no_text", "local"])
def test_domains_remove_actual_proposal_axes_including_initializer(tmp_path, domain):
    kw = inputs()
    space = build_config_space(
        cluster_id="0",
        labels=kw["context"]["labels"],
        patterns=kw["patterns"]["0"],
        models=(ModelSpec("m", True, ParameterDomain(low=0.0, high=1.0)),),
        relations=kw["relations"],
        candidate_domain=domain,
    )
    if domain == "no_text":
        assert not space.patterns and all(f.kind != "pattern" for f in space.features)
        assert any(f.kind == "label" for f in space.features)
    else:
        assert space.variables == ("x",) and space.head_kinds == ("add", "remove")
        assert not space.relations and all(f.kind != "label" for f in space.features)
    for params in space.initial_configurations(35, seed=42):
        receipt = space.configuration_receipt(params)
        if not receipt["admissible"]:
            # Author-form-invalid startup proposals are recorded as budgeted
            # invalid trials, rather than silently repaired or resampled.
            assert receipt["invalid_reason"].startswith("author_rule_form:")
            continue
        rule = space.instantiate(params)
        if domain == "no_text":
            assert not any(isinstance(a, Text) for a in rule.body)
        else:
            assert not any(isinstance(a, (Label, LabelCompare, AttrEqual)) for a in rule.body)
    params = space.empty_parameters("add", label="A", variable="x")
    params["forbidden.include"] = 1
    with pytest.raises(InvalidConfiguration):
        space.instantiate(params)
    out = tmp_path / "run"
    # Go beyond the 30 startup trials and execute every structured recipe and
    # the broad TPE branch on the actually restricted feature axes.
    done = run_discovery(output=out, candidate_domain=domain, **dict(kw, budget_per_cluster=43))
    assert done["complete_BO_trials"] == 43 and done["status"] == "complete"
    from loris._engine.rdl_reference import Rule

    for path in sorted((out / "learning/cluster_00").glob("trial_*.json")):
        record = json.loads(path.read_text())
        if record["rule"] is None:
            continue
        rule = Rule.from_dict(record["rule"])
        if domain == "no_text":
            assert not any(isinstance(a, Text) for a in rule.body)
        else:
            assert not any(isinstance(a, (Label, LabelCompare, AttrEqual)) for a in rule.body)


def test_score_order_reorders_admission_not_bo_and_is_resumable(tmp_path):
    kw = inputs()
    raw = tmp_path / "raw"
    sorted_run = tmp_path / "sorted"
    run_discovery(output=raw, admission_order="proposal_order", **kw)
    done = run_discovery(output=sorted_run, admission_order="standalone_score_desc", **kw)
    assert semantic_trials(raw) == semantic_trials(sorted_run)
    records = [
        json.loads(p.read_text())
        for p in sorted((sorted_run / "learning/cluster_00").glob("trial_*.json"))
    ]
    order = json.loads((sorted_run / "learning/cluster_00/admission/ORDER.json").read_text())
    assert order["trial_numbers"] == order_trials(records, "standalone_score_desc")
    assert order["strict_gain_epsilon"] == 0.0
    assert (
        run_discovery(output=sorted_run, resume=True, admission_order="standalone_score_desc", **kw)
        == done
    )
    assert archive_index(sorted_run, tmp_path / "archive")["completion"]["BO_trials"] == 4
    path = sorted_run / "learning/cluster_00/admission/ORDER.json"
    order["trial_numbers"] = list(reversed(order["trial_numbers"]))
    path.write_text(json.dumps(order))
    with pytest.raises(ValueError):
        status(sorted_run)
    with pytest.raises(ValueError):
        run_discovery(output=sorted_run, resume=True, admission_order="standalone_score_desc", **kw)


def test_order_exact_descending_stable_ties_including_failed_sentinels():
    records = [
        dict(number=i, standalone=dict(score=score))
        for i, score in enumerate([0.1, -1, 0.7, 0.7, 0, -1])
    ]
    assert order_trials(records, "standalone_score_desc") == [2, 3, 0, 4, 1, 5]
    assert order_trials(records, "proposal_order") == list(range(6))
