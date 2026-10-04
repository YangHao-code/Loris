"""Real compact checkpoints survive current module names and resume faithfully."""

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from loris._engine.compact_storage import ParameterCodec
from loris._engine.rdl_reference import Label, Rule
from loris._engine.rlearner import TwoFilterEvaluator
from loris._engine.rlearner_config import ConfigurationSpace, ModelSpec, ParameterDomain
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._learning import structured_bo, structured_bo_storage
from loris.discovery.search import storage as current_storage
from tests.test_discovery_runner import inputs


class UnseededSpace(ConfigurationSpace):
    """Exercise actual future sampler draws, without queued warm-start vectors."""

    def initial_configurations(self, **kwargs):
        return []

    def iter_initial_configurations(self, **kwargs):
        return iter(())


def structured_inputs():
    values = inputs()
    context = values["context"]
    space = UnseededSpace(
        cluster_id=0, patterns=values["patterns"]["0"],
        models=[ModelSpec("m", True, ParameterDomain(values=(.5, .8)))],
        labels=context["labels"], variables=("x",), relations=(),
    )
    fit = context["fit_indices"]
    prior = structured_bo.build_fit_witness_prior(
        space, docs=[context["docs"][int(i)] for i in fit],
        row_ids=[context["row_ids"][int(i)] for i in fit],
        row_roles=["source_fit"] * len(fit), labels=context["labels"],
        gamma_pos=context["gamma"].pos[fit], gamma_neg=context["gamma"].neg[fit],
        model_probabilities={"m": context["model_probabilities"]["m"][fit]},
    )
    return context, space, prior


def make_sampler(space, prior, seed=17):
    return structured_bo.StructuredWitnessTPESampler(
        tuple(f.include_key for f in space.features), seed=seed, prior=prior,
        feature_families={f.include_key: f.kind for f in space.features},
    )


def test_structured_checkpoint_uses_imported_sources_and_rejects_old_binding(tmp_path):
    _, space, prior = structured_inputs()
    sampler = make_sampler(space, prior)
    store = structured_bo_storage.SamplerCheckpointStore(
        tmp_path / "checkpoints", ParameterCodec.from_space(space)
    )
    record = store.save(sampler, -1)
    binding = record["structured_sampler_envelope"]
    assert binding["schema"] == "paper36_structured_witness_checkpoint_envelope_v1"
    assert binding["class"] == "loris._learning.structured_bo.StructuredWitnessTPESampler"
    expected = {str(Path(inspect.getfile(cls)).resolve()) for cls in (
        structured_bo_storage.OriginalCheckpointStore,
        structured_bo_storage.JointBinaryTPESampler,
    )}
    assert expected <= set(binding["sources"])
    restored = store.restore(record)
    assert type(restored) is type(sampler)
    assert restored.prior.binding_id == sampler.prior.binding_id
    assert np.array_equal(restored.rng.random(12), sampler.rng.random(12))
    old = copy.deepcopy(record)
    old["structured_sampler_envelope"]["sources"] = {
        path.replace("/_engine/", "/paper36/"): sha
        for path, sha in binding["sources"].items()
    }
    with pytest.raises(ValueError, match="source/class binding differs"):
        store.restore(old)


@pytest.mark.parametrize("kind", ["RichCombinationTPESampler", "RandomJointSampler"])
def test_current_sampler_binding_still_uses_actual_source_modules(kind):
    binding = current_storage._binding(kind)
    assert binding["class"].endswith("." + kind)
    assert len(binding["sources"]) == 7
    assert all(Path(path).is_file() for path in binding["sources"])
    assert not any("/paper36/" in path for path in binding["sources"])


def test_structured_compact_interruption_resumes_actual_proposals_and_chase(tmp_path):
    context, space, prior = structured_inputs()
    runner = IndexedChaseRunner(context, tmp_path / "cache")
    evaluator = TwoFilterEvaluator(
        runner, labels=context["labels"], receiver_indices=context["receiver_indices"],
        truth=context["truth"], gamma=context["gamma"], context_id=context["context_id"],
        line7_mode="positive_macro_f1",
        head_info=lambda rule: (rule.head.label, rule.head.polarity == "neg")
        if isinstance(rule.head, Label) else None,
    )
    factory = lambda seed: make_sampler(space, prior, seed)
    binding = dict(factory(17).protocol,
                   execution_specialization=structured_bo._specialized_cluster_runner()[1])
    kwargs = dict(space=space, evaluator=evaluator, seed=17, budget=6,
                  rule_parser=Rule.from_dict, sampler_factory=factory, sampler_binding=binding,
                  storage_mode="compact_full_indicators_v1")
    interrupted, fresh = tmp_path / "interrupted", tmp_path / "fresh"

    def stop(record):
        if record["complete"] == 2:
            raise RuntimeError("controlled structured interruption")

    try:
        with pytest.raises(RuntimeError, match="controlled structured interruption"):
            structured_bo.run_structured_cluster_search(
                directory=interrupted, on_progress=stop, **kwargs
            )
        assert json.loads((interrupted / "sampler.json").read_text())["completed_trial"] == 1
        resumed = structured_bo.run_structured_cluster_search(directory=interrupted, **kwargs)
        uninterrupted = structured_bo.run_structured_cluster_search(directory=fresh, **kwargs)
        projection = lambda records: [
            (r["params"], r["rule"], r["standalone"]["score"]) for r in records
        ]
        assert projection(resumed) == projection(uninterrupted)
        assert len(resumed) == 6 and evaluator.actual_calls > 0
        assert any(r["rule"] is not None for r in resumed)
        assert json.loads((interrupted / "initial_configurations.json").read_text()) == []
        codec = ParameterCodec.from_space(space)
        restored = []
        for directory in (interrupted, fresh):
            record = json.loads((directory / "sampler.json").read_text())
            assert record["completed_trial"] == 5 and record["observation_count"] == 6
            restored.append(structured_bo_storage.SamplerCheckpointStore(
                directory / "sampler_states", codec
            ).restore(record))
        assert np.array_equal(restored[0].rng.random(12), restored[1].rng.random(12))
        for number in range(6):
            left, right = restored[0].observations[number], restored[1].observations[number]
            assert left[0] == right[0] and np.array_equal(left[1], right[1])
    finally:
        evaluator.close()
        runner.close()
        prior.close()
