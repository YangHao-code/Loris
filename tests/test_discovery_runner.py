"""Small real BO/Chase transactions; never train or read production labels."""

import ast
import copy
import gzip
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from loris.chase import GammaSeed
from loris.document import Document
from loris.predicates import MatchPredicate
from loris._engine.rdl_reference import AttrEqual, LabelCompare
from loris._engine.rlearner_config import PatternSpec
from loris.discovery import runner as portable


def historical_root(required):
    root = Path(os.environ.get("LORIS_HISTORY_ROOT", Path(__file__).resolve().parents[1])).resolve()
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        pytest.skip(
            "historical comparison only: missing authenticated source assets under "
            + str(root)
            + ": "
            + ", ".join(missing)
        )
    return root


def inputs(clusters=1):
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], bool)
    fit = np.array([0, 1])
    control = np.array([4, 5])
    p = np.where(y, 0.9, 0.1).astype(float)
    gp = np.zeros_like(y)
    gn = np.zeros_like(y)
    gp[fit] = y[fit]
    gn[fit] = ~y[fit]
    context = dict(
        docs=[
            Document(cnt="red" if row[0] else "blue", ttl=str(i), mtd="") for i, row in enumerate(y)
        ],
        labels=["A", "B"],
        row_ids=list(range(6)),
        row_roles=["source_fit"] * 2 + ["unlabeled_train"] * 2 + ["val_ctrl"] * 2,
        gamma=GammaSeed(gp, gn),
        fit_indices=fit,
        receiver_indices=control,
        truth=y[control],
        source_gold={"fit": y[fit], "val_ctrl": y[control]},
        model_probabilities={"m": p},
        model_decisions={"m": p >= 0.5},
        attribute_values={"mtd.group": ["r", "b", "r", "b", "r", "b"]},
        attribute_presence={"mtd.group": np.ones(6, bool)},
        context_id="portable_synthetic6",
    )
    si = {str(i): ["m"] for i in range(clusters)}
    patterns = {c: [PatternSpec("red", MatchPredicate("cnt", "red"))] for c in si}
    return dict(
        context=context,
        si=si,
        patterns=patterns,
        relations=(AttrEqual("x", "mtd.group", "y", "mtd.group"), LabelCompare("x", "subset", "y")),
        input_binding={"synthetic_fixture": "portable_synthetic6"},
        seed=42,
        budget_per_cluster=4,
        workers=1,
        initializer_slots=2,
        label_rotation="axis",
    )


def semantic_trials(folder, cluster=0):
    records = [
        json.loads(p.read_text())
        for p in sorted((folder / "learning" / ("cluster_%02d" % cluster)).glob("trial_*.json"))
    ]
    return [(r["params"], r["rule"], r["standalone"]["score"]) for r in records]


def test_actual_discovery_full_archive_proof_and_readonly_complete_resume(tmp_path):
    out = tmp_path / "discovery"
    kw = inputs()
    receipt = portable.run_discovery(output=out, **kw)
    assert receipt["status"] == "complete" and receipt["complete_BO_trials"] == 4
    assert receipt["actual_total_Chase_calls_this_attempt"] == sum(
        receipt[k]
        for k in (
            "actual_parent_scoring_Chase_calls_this_attempt",
            "actual_worker_scoring_Chase_calls_this_attempt",
            "actual_final_proof_Chase_calls_this_attempt",
        )
    )
    assert receipt["actual_final_proof_Chase_calls_this_attempt"] == 1
    assert receipt["selected_rules"] == 2 and receipt["final_positive_metrics"]["macro_f1"] == 1.0
    folder = out / "learning/cluster_00"
    assert (
        json.loads((folder / "admission/ORDER.json").read_text())["order"]
        == "standalone_score_desc"
    )
    assert len(list(folder.glob("trial_*.json"))) == 4
    assert len(list((folder / "admission").glob("candidate_*.json"))) == 4
    assert json.loads((out / "PROOF_REPLAY.json").read_text())[
        "complete_facts_conflicts_rounds_scores_equal"
    ]
    with gzip.open(out / "PROVENANCE.jsonl.gz", "rt") as stream:
        proof = [json.loads(line) for line in stream]
    assert (sum(r["type"] == "proof" for r in proof) +
            sum(r["record"]["expanded_root_count"] for r in proof if r["type"] == "implicit_seed_roots")) == proof[0]["identity"]["count"]
    assert proof[0]["schema"] == "loris_complete_final_provenance_v3"
    from loris._engine.relation_rows import RelationRows
    from loris._engine.typed_runtime import RelationFacts

    for header in (r for r in proof if r["type"] == "relation"):
        overrides = [r for r in proof if r["type"] == "relation_row" and r["name"] == header["name"]]
        assert len(overrides) == header["stored_rows"]
        restored = RelationRows.from_dict(dict(
            n=header["identity"]["n"], default_diagonal=header["default_diagonal"],
            overrides=[[r["row"], r["encoding"], r["value"]] for r in overrides]))
        assert RelationFacts(restored).identity() == header["identity"]
    before = {
        p: p.read_bytes()
        for p in (out / "COMPLETE.json", folder / "WORKER_ATTEMPTS.jsonl", folder / "optuna.sqlite")
    }
    assert portable.run_discovery(output=out, resume=True, **kw) == receipt
    assert all(p.read_bytes() == b for p, b in before.items())
    with pytest.raises(FileExistsError):
        portable.run_discovery(output=out, **kw)
    with pytest.raises(ValueError):
        portable.run_discovery(output=out, resume=True, **dict(kw, seed=43))
    admission = folder / "admission/candidate_00000.json"
    original = admission.read_bytes()
    damaged = json.loads(original)
    damaged["selected_after"] += 1
    admission.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="proposal/admission"):
        portable.run_discovery(output=out, resume=True, **kw)
    admission.write_bytes(original)
    (out / "FACTS.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="facts"):
        portable.run_discovery(output=out, resume=True, **kw)


def test_interrupted_bo_resume_preserves_real_next_proposals(tmp_path, monkeypatch):
    kw = inputs()
    out = tmp_path / "interrupted"
    actual = portable.run_rich_cluster_search

    def interrupt(**args):
        callback = args["on_progress"]

        def progress(record):
            callback(record)
            if record["complete"] == 2:
                raise RuntimeError("controlled interruption")

        return actual(**dict(args, on_progress=progress))

    monkeypatch.setattr(portable, "run_rich_cluster_search", interrupt)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        portable.run_discovery(output=out, **kw)
    assert len(list((out / "learning/cluster_00").glob("trial_*.json"))) == 2
    monkeypatch.setattr(portable, "run_rich_cluster_search", actual)
    a = portable.run_discovery(output=out, resume=True, **kw)
    assert len(a["failed_worker_attempts"]) == 1
    assert (
        a["actual_worker_scoring_Chase_calls_all_recorded_attempts"]
        > a["actual_worker_scoring_Chase_calls_this_attempt"]
    )
    b = portable.run_discovery(output=tmp_path / "fresh", **kw)
    assert semantic_trials(out) == semantic_trials(tmp_path / "fresh")
    assert a["final_positive_metrics"] == b["final_positive_metrics"]
    with np.load(out / "FACTS.npz") as x, np.load(tmp_path / "fresh/FACTS.npz") as y:
        assert all(np.array_equal(x[k], y[k]) for k in ("pos", "neg"))


def test_parallel_worker_semantics_match_serial(tmp_path):
    kw = inputs(clusters=2)
    a = portable.run_discovery(output=tmp_path / "serial", **kw)
    b = portable.run_discovery(output=tmp_path / "parallel", **dict(kw, workers=2))
    assert a["final_positive_metrics"] == b["final_positive_metrics"]
    for i in range(2):
        assert semantic_trials(tmp_path / "serial", i) == semantic_trials(tmp_path / "parallel", i)


@pytest.mark.parametrize(
    "damage",
    [
        "test_role",
        "outside_fit_root",
        "hidden_truth",
        "incomplete_fit",
        "wrong_control",
        "negative_seed",
    ],
)
def test_api_enforces_same_permission_and_axis_contract(tmp_path, damage):
    kw = inputs(clusters=2)
    ctx = kw["context"]
    if damage == "test_role":
        ctx["row_roles"][2] = "test"
    elif damage == "outside_fit_root":
        p = ctx["gamma"].pos.copy()
        p[2, 0] = True
        ctx["gamma"] = GammaSeed(p, ctx["gamma"].neg)
    elif damage == "hidden_truth":
        ctx["source_gold"]["hidden_train"] = np.ones((2, 2), bool)
    elif damage == "incomplete_fit":
        n = ctx["gamma"].neg.copy()
        n[0, 1] = False
        ctx["gamma"] = GammaSeed(ctx["gamma"].pos, n)
    elif damage == "wrong_control":
        ctx["truth"] = ~ctx["truth"]
    elif damage == "negative_seed":
        kw["seed"] = -1
    with pytest.raises(ValueError):
        portable.run_discovery(output=tmp_path / "invalid", **kw)
    assert not (tmp_path / "invalid").exists()


# These are the six exact import paths changed when the maintained modules
# moved to their release packages. No names, constants, expressions, function
# bodies, docstrings, or other AST nodes are normalized by this comparison.
HISTORICAL_IMPORT_RENAMES = {
    "loris.paper36.rdl_reference": "loris._engine.rdl_reference",
    "loris.paper36.rlearner_config": "loris._engine.rlearner_config",
    "loris.paper36_connected_20260910.configuration": "loris._search_space.configuration",
    "loris.paper36.sparse_bo": "loris._engine.sparse_bo",
    "loris.paper36.primitive_masks": "loris._engine.primitive_masks",
    "loris.paper36_aligned_20260911.structured_bo": "loris._learning.structured_bo",
}


@pytest.mark.golden
def test_migrated_sampler_matches_historical_ast_except_declared_import_renames():
    # The current grammar deliberately changed after the author's 2026-09-22
    # cross-rule clarification; only the sampler source retains this guarantee.
    pairs = [
        (
            "loris/discovery/search/sampler.py",
            "experiments/rich_rule_discovery_bgc_20260916/search/sampler.py",
        )
    ]
    root = historical_root([old for _, old in pairs])
    current = Path(__file__).resolve().parents[1]
    normalized = set()
    for new, old in pairs:
        historical = ast.parse((root / old).read_text())
        for node in ast.walk(historical):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module in HISTORICAL_IMPORT_RENAMES
            ):
                normalized.add(node.module)
                node.module = HISTORICAL_IMPORT_RENAMES[node.module]
        assert ast.dump(ast.parse((current / new).read_text())) == ast.dump(historical)
    assert normalized == set(HISTORICAL_IMPORT_RENAMES) - {
        "loris.paper36_connected_20260910.configuration"
    }


def initializer_case():
    from loris._engine.rlearner_config import ModelSpec, ParameterDomain
    from loris.discovery.search.configuration import build_config_space
    from loris.discovery.search.sampler import build_fit_witness_prior

    labels = tuple("L%02d" % i for i in range(35))
    y = np.array([[1] * 35, [0] * 35], bool)
    space = build_config_space(
        cluster_id="1",
        labels=labels,
        patterns=(),
        models=(ModelSpec("m", True, ParameterDomain(low=0.0, high=1.0)),),
    )
    prior = build_fit_witness_prior(
        space,
        docs=[Document(cnt="red"), Document(cnt="blue")],
        row_ids=[0, 1],
        row_roles=["source_fit"] * 2,
        labels=labels,
        gamma_pos=y,
        gamma_neg=~y,
        model_probabilities={"m": np.where(y, 0.9, 0.1).astype(float)},
        model_decisions={"m": y},
        attribute_values={},
        attribute_presence={},
    )
    return labels, space, prior


def test_initializer_axis_and_rotation_without_restricting_domain(tmp_path):
    from loris.discovery.search.initialization import build_fit_initialized_space

    labels, space, prior = initializer_case()
    try:
        for name, rotation in [("axis", "axis"), ("rotated", "seeded_cluster_rotation")]:
            a = build_fit_initialized_space(
                space,
                prior,
                output_dir=tmp_path / name,
                initializer_slots=30,
                label_rotation=rotation,
                rotation_seed=42,
                cluster_position=1,
            )
            assert len(a.initial_configurations(35, seed=42)) == 35
            assert a.labels == labels
        rotating = json.loads((tmp_path / "rotated/SEEDS.json").read_text())
        assert len(rotating["selected_label_indices"]) == 30
        assert any(i >= 30 for i in rotating["selected_label_indices"])
        assert len(rotating["eligible_but_outside_startup_budget"]) == 5
        with pytest.raises(ValueError, match="options differ"):
            build_fit_initialized_space(
                space, prior, output_dir=tmp_path / "axis", initializer_slots=1
            )
        partial = tmp_path / "partial"
        partial.mkdir()
        with pytest.raises(FileExistsError):
            build_fit_initialized_space(space, prior, output_dir=partial)
    finally:
        prior.close()


@pytest.mark.golden
def test_initializer_matches_authenticated_historical_orders(tmp_path, monkeypatch):
    paths = [
        "experiments/rich_rule_discovery_bgc_20260916/search/initialization.py",
        "experiments/rich_rule_discovery_bgc_20260916/search/sampler.py",
        "experiments/formal_evaluation_20260917/reuters_single_seed/initialization.py",
    ]
    root = historical_root(paths)
    # Execute the authentic initializer and sampler sources unchanged. Only
    # their imported dependencies receive test-local release package aliases;
    # monkeypatch removes these aliases after this test. This is a comparison
    # of original seed-selection logic with the same current space/prior, not
    # a claim that every transitive historical runtime module is unchanged.
    aliases = dict(
        HISTORICAL_IMPORT_RENAMES,
        **{
            "loris.paper36.compact_storage": "loris._engine.compact_storage",
            "loris.paper36.rlearner": "loris._engine.rlearner",
            "loris.paper36": "loris._engine",
            "loris.paper36_connected_20260910": "loris._search_space",
            "loris.paper36_aligned_20260911": "loris._learning",
        },
    )
    for historical, current in aliases.items():
        monkeypatch.setitem(sys.modules, historical, importlib.import_module(current))
    monkeypatch.syspath_prepend(str(root))
    import experiments.rich_rule_discovery_bgc_20260916.search.initialization as bgc
    import experiments.formal_evaluation_20260917.reuters_single_seed.initialization as reuters

    assert Path(bgc.__file__).resolve() == root / paths[0]
    assert Path(reuters.__file__).resolve() == root / paths[2]
    from loris.discovery.search.initialization import build_fit_initialized_space

    _, space, prior = initializer_case()
    try:
        for name, old, rotation in [
            ("axis", bgc.build_fit_initialized_space, "axis"),
            ("rotated", reuters.build_fit_initialized_space, "seeded_cluster_rotation"),
        ]:
            actual = build_fit_initialized_space(
                space,
                prior,
                output_dir=tmp_path / name,
                initializer_slots=30,
                label_rotation=rotation,
                rotation_seed=42,
                cluster_position=1,
            )
            historical = old(space, prior, output_dir=tmp_path / (name + "_old"))
            assert actual.initial_configurations(35, seed=42) == historical.initial_configurations(
                35, seed=42
            )
    finally:
        prior.close()
