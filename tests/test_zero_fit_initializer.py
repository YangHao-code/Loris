"""Zero fit seeds skip the fit grid, without changing ordinary BO starts."""

import hashlib
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from loris._engine.compact_storage import ParameterCodec
from loris._engine.rlearner_config import ModelSpec, ParameterDomain, PatternSpec
from loris.discovery import runner as portable
from loris.discovery.search import initialization
from loris.discovery.search.configuration import build_config_space
from loris.discovery.search.sampler import build_fit_witness_prior
from loris.document import Document
from loris.predicates import MatchPredicate
from tests.test_discovery_runner import inputs, semantic_trials


@pytest.fixture
def fit_case():
    labels = tuple("L%02d" % i for i in range(35))
    answers = np.array([[1] * len(labels), [0] * len(labels)], bool)
    space = build_config_space(
        cluster_id="1",
        labels=labels,
        patterns=(
            PatternSpec("red", MatchPredicate("cnt", "red")),
            PatternSpec("blue", MatchPredicate("cnt", "blue")),
        ),
        models=tuple(
            ModelSpec(alias, True, ParameterDomain(low=0.0, high=1.0)) for alias in ("m", "n")
        ),
    )
    prior = build_fit_witness_prior(
        space,
        docs=[Document(cnt="red"), Document(cnt="blue")],
        row_ids=[0, 1],
        row_roles=["source_fit"] * 2,
        labels=labels,
        gamma_pos=answers,
        gamma_neg=~answers,
        model_probabilities={
            "m": np.where(answers, 0.9, 0.1).astype(float),
            "n": np.where(answers, 0.8, 0.2).astype(float),
        },
        model_decisions={"m": answers, "n": answers},
        attribute_values={},
        attribute_presence={},
    )
    try:
        yield space, prior
    finally:
        prior.close()


@pytest.mark.parametrize("rotation", ["axis", "seeded_cluster_rotation"])
def test_zero_slots_skip_masks_and_dense_candidates_keep_axis_and_resume(
    tmp_path, monkeypatch, fit_case, rotation
):
    space, prior = fit_case
    original_manifest = space.manifest()
    options = dict(initializer_slots=0, label_rotation=rotation, cluster_position=3,
                   initializer_candidate_archive="evaluated_with_analytic_skips")
    output = tmp_path / "zero"

    def forbidden(*args, **kwargs):
        pytest.fail("zero startup slots must not evaluate or materialize fit candidates")

    with monkeypatch.context() as patch:
        patch.setattr(initialization, "pattern_mask", forbidden)
        patch.setattr(type(space), "empty_parameters", forbidden)
        patch.setattr(type(space), "instantiate", forbidden)
        patch.setattr(ParameterCodec, "encode", forbidden)
        initialized = initialization.build_fit_initialized_space(
            space, prior, output_dir=output, **options
        )

    assert space.manifest() == original_manifest
    assert initialized.patterns == space.patterns and initialized.models == space.models
    assert initialized.features == space.features and initialized.labels == space.labels
    assert gzip.decompress((output / "EVALUATED_FIT_CANDIDATES.jsonl.gz").read_bytes()) == b""
    axis = json.loads((output / "PARAMETER_AXIS.json").read_text())
    assert axis["include_keys"] == list(ParameterCodec.from_space(space).include_keys)
    seeds = json.loads((output / "SEEDS.json").read_text())
    assert seeds["seeds"] == seeds["selected_label_indices"] == []
    assert seeds["eligible_but_outside_startup_budget"] == []
    assert seeds["labels_without_seed"] == []  # Eligibility was not evaluated.
    assert seeds["labels_not_evaluated"] == list(space.labels)
    expected_order = list(range(len(space.labels)))
    if rotation == "seeded_cluster_rotation":
        expected_order = np.random.RandomState(42).permutation(len(space.labels)).tolist()
    assert seeds["label_order"] == expected_order
    for name in ("PROTOCOL.json", "SEEDS.json", "COMPLETE.json"):
        record = json.loads((output / name).read_text())
        assert record["fit_grid_skipped_because_zero_slots"] is True
        assert record["fit_grid_candidate_count"] == 0
        assert "without enumerating" in record["fit_grid_skip_reason"]
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["seed_count"] == 0 and all(n == 0 for n in complete["counts"].values())
    assert complete["BO_calls"] == complete["Chase_calls"] == complete["direct_rules_imported"] == 0
    assert {Path(r["path"]).name for r in complete["files"]} == {
        "PROTOCOL.json",
        "PARAMETER_AXIS.json",
        "EVALUATED_FIT_CANDIDATES.jsonl.gz",
        "ANALYTICALLY_SKIPPED.json",
        "SEEDS.json",
    }
    before = {p: p.read_bytes() for p in output.iterdir()}
    resumed = initialization.build_fit_initialized_space(space, prior, output_dir=output, **options)
    for seed in (7, 42):
        fallback = space.initial_configurations(36, seed=seed)
        assert initialized.initial_configurations(36, seed=seed) == fallback
        assert resumed.initial_configurations(36, seed=seed) == fallback
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(ValueError, match="options differ"):
        initialization.build_fit_initialized_space(
            space, prior, output_dir=output, initializer_slots=1
        )
    (output / "EVALUATED_FIT_CANDIDATES.jsonl.gz").write_text("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        initialization.build_fit_initialized_space(space, prior, output_dir=output, **options)


def test_zero_slots_real_bo_resume_after_startup_preserves_trials(tmp_path, monkeypatch):
    kw = dict(inputs(), initializer_slots=0, budget_per_cluster=36,
              initializer_candidate_archive="all")
    output = tmp_path / "interrupted"
    actual = portable.run_rich_cluster_search

    def interrupted(**args):
        callback = args["on_progress"]

        def progress(record):
            callback(record)
            if record["complete"] == 31:
                raise RuntimeError("interrupted after first post-startup trial")

        return actual(**dict(args, on_progress=progress))

    monkeypatch.setattr(portable, "run_rich_cluster_search", interrupted)
    with pytest.raises(RuntimeError, match="interrupted after first post-startup trial"):
        portable.run_discovery(output=output, **kw)
    assert len(semantic_trials(output)) == 31
    monkeypatch.setattr(portable, "run_rich_cluster_search", actual)
    resumed = portable.run_discovery(output=output, resume=True, **kw)
    fresh_output = tmp_path / "fresh"
    fresh = portable.run_discovery(output=fresh_output, **kw)
    assert resumed["complete_BO_trials"] == fresh["complete_BO_trials"] == 36
    assert len(resumed["failed_worker_attempts"]) == 1
    assert semantic_trials(output) == semantic_trials(fresh_output)
    assert resumed["final_positive_metrics"] == fresh["final_positive_metrics"]
    with np.load(output / "FACTS.npz") as a, np.load(fresh_output / "FACTS.npz") as b:
        assert all(np.array_equal(a[key], b[key]) for key in ("pos", "neg"))
    assert portable.run_discovery(output=output, resume=True, **kw) == resumed


@pytest.fixture
def revision8_initializer(monkeypatch):
    """Authentic old initializer with the same current transitive dependencies."""
    source_root = os.environ.get("LORIS_REVISION8_SOURCE")
    if not source_root:
        pytest.skip("set LORIS_REVISION8_SOURCE to the frozen revision8 source tree")
    source = Path(source_root) / "loris/discovery/search/initialization.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "c8bcfd13aec374c09d018b0385a76625f56684c814c246d8a19918f097a36eba"
    )
    name = "loris.discovery.search._revision8_initializer_test"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.golden
@pytest.mark.parametrize("slots", [0, 2, 30])
@pytest.mark.parametrize("rotation", ["axis", "seeded_cluster_rotation"])
def test_frozen_initializer_fallback_and_nonzero_candidate_receipts_match(
    tmp_path, fit_case, revision8_initializer, slots, rotation
):
    space, prior = fit_case
    options = dict(initializer_slots=slots, label_rotation=rotation, rotation_seed=42, cluster_position=1,
                   initializer_candidate_archive="all")
    current = initialization.build_fit_initialized_space(
        space, prior, output_dir=tmp_path / "current", **options
    )
    previous = revision8_initializer.build_fit_initialized_space(
        space, prior, output_dir=tmp_path / "previous", **options
    )
    for seed in (7, 42):
        assert current.initial_configurations(36, seed=seed) == previous.initial_configurations(
            36, seed=seed
        )
    old_complete = json.loads((tmp_path / "previous/COMPLETE.json").read_text())
    new_complete = json.loads((tmp_path / "current/COMPLETE.json").read_text())
    assert old_complete["counts"]["total"] == 245
    assert old_complete["seed_count"] == new_complete["seed_count"] == slots
    if slots:
        assert gzip.decompress((tmp_path / "current/ALL_FIT_CANDIDATES.jsonl.gz").read_bytes()) == (
            tmp_path / "previous/ALL_FIT_CANDIDATES.jsonl"
        ).read_bytes()
        for name in ("SEEDS.json", "PARAMETER_AXIS.json"):
            assert (tmp_path / "current" / name).read_bytes() == (
                tmp_path / "previous" / name
            ).read_bytes()
        assert old_complete["counts"] == new_complete["counts"]
        assert "fit_grid_skipped_because_zero_slots" not in new_complete
        old_protocol = json.loads((tmp_path / "previous/PROTOCOL.json").read_text())
        new_protocol = json.loads((tmp_path / "current/PROTOCOL.json").read_text())
        old_protocol.pop("source")
        new_protocol.pop("source")
        assert new_protocol.pop("candidate_encoding")["compression"] == "gzip"
        assert new_protocol.pop("candidate_rows") == "ALL_FIT_CANDIDATES.jsonl.gz"
        assert new_protocol.pop("candidate_archive") == "all"
        assert new_protocol.pop("analytic_skip_scope") is None
        assert new_protocol["initialization_options"].pop("initializer_candidate_archive") == "all"
        assert old_protocol == new_protocol
    else:
        assert new_complete["counts"]["total"] == 0
        assert gzip.decompress((tmp_path / "current/ALL_FIT_CANDIDATES.jsonl.gz").read_bytes()) == b""


@pytest.mark.golden
@pytest.mark.parametrize("slots", [0, 2, 30])
def test_frozen_initializer_real_bo_trials_unchanged(
    tmp_path, monkeypatch, revision8_initializer, slots
):
    kw = dict(inputs(), initializer_slots=slots, budget_per_cluster=36,
              initializer_candidate_archive="all")

    def legacy_build(*args, **kwargs):
        # The frozen author implementation predates the storage policy option.
        # Only its original complete archive is a valid reference here.
        assert kwargs.pop("initializer_candidate_archive") == "all"
        return revision8_initializer.build_fit_initialized_space(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            portable,
            "build_fit_initialized_space",
            legacy_build,
        )
        patch.setattr(portable, "FitInitializedSpace", revision8_initializer.FitInitializedSpace)
        previous = portable.run_discovery(output=tmp_path / "previous", **kw)
    current = portable.run_discovery(output=tmp_path / "current", **kw)
    old_trials = semantic_trials(tmp_path / "previous")
    assert len(old_trials) == 36
    assert semantic_trials(tmp_path / "current") == old_trials
    assert current["final_positive_metrics"] == previous["final_positive_metrics"]
    assert current["selected_rules"] == previous["selected_rules"]
    with np.load(tmp_path / "previous/FACTS.npz") as a, np.load(
        tmp_path / "current/FACTS.npz"
    ) as b:
        assert all(np.array_equal(a[key], b[key]) for key in ("pos", "neg"))
