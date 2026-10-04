"""Actual finite expanded BO and unsliced fit-only predicate supplies."""

import json
import sqlite3

import numpy as np
import pytest

from loris.discovery import patterns
from loris.discovery.__main__ import parser
from loris.discovery.archive import archive_index
from loris.discovery.manifest import file_record, load_inputs
from loris.discovery.runner import run_discovery
from loris.discovery.search.domain import variable_names
from loris.document import Document
from tests.test_discovery_patterns import example
from tests.test_discovery_variants import discovery_manifest


def test_three_variable_public_inputs_reach_real_bo_and_complete_archives(tmp_path):
    manifest = discovery_manifest(tmp_path / "inputs")
    overrides = dict(
        max_variables=3,
        context_probes=79,
        acquisition_candidates=97,
        expected_body_size=5.0,
        initializer_slots=0,
        budget_per_cluster=36,
    )
    inputs = load_inputs(manifest, overrides)
    assert {a.left_var for a in inputs["relations"]} == {"x", "y", "z"}
    assert {a.right_var for a in inputs["relations"]} == {"x", "y", "z"}
    assert len(inputs["relations"]) == 21  # Three AttrEqual pairs and six per comparison kind.
    out = tmp_path / "three_variables"
    complete = run_discovery(output=out, **inputs)
    assert complete["status"] == "complete" and complete["complete_BO_trials"] == 36
    protocol = json.loads((out / "learning/protocol.json").read_text())
    assert protocol["spaces"][0]["variables"] == ["x", "y", "z"]
    for key, value in overrides.items():
        assert protocol["options"][key] == value
    records = [
        json.loads(p.read_text())
        for p in sorted((out / "learning/cluster_00").glob("trial_*.json"))
    ]
    assert len(records) == 36
    worker_protocol = json.loads((out / "learning/cluster_00/protocol.json").read_text())
    assert worker_protocol["sampler"]["context_probes"] == 79
    assert worker_protocol["sampler"]["base_TPE"]["acquisition_candidates"] == 97
    assert worker_protocol["sampler"]["base_TPE"]["prior_expected_size_requested"] == 5.0
    assert records[30]["sampler_diagnostic"]["base"]["acquisition_candidates"] == 97
    assert records[30]["sampler_diagnostic"]["branch"] == "original_full_axis_TPE"
    assert any(
        record["rule"]
        and {a["var"] for a in record["rule"]["body"] if a["type"] == "Doc"} == {"x", "y", "z"}
        for record in records
    )
    index = archive_index(out, tmp_path / "archive")
    assert index["completion"]["BO_trials"] == 36
    assert index["clusters"][0]["all_sequential_admissions_verified"]
    assert index["study_snapshots"][0]["table_rows"]["trials"] == 36
    # These options are part of the transaction, not unbound launcher hints.
    for key, value in (
        ("max_variables", 2),
        ("context_probes", 80),
        ("acquisition_candidates", 98),
        ("expected_body_size", 6.0),
    ):
        with pytest.raises(ValueError):
            run_discovery(output=out, resume=True, **dict(inputs, **{key: value}))


@pytest.mark.parametrize("value", [True, 1, 0, -1, 2.5, "3"])
def test_variable_axis_rejects_ambiguous_or_insufficient_values(value):
    with pytest.raises(ValueError, match="max_variables"):
        variable_names(value)


def test_more_variables_have_unique_stable_names_and_local_remains_local(tmp_path):
    from loris.discovery.search.domain import build_config_space
    from loris._engine.rlearner_config import paper_relations

    assert variable_names(5) == ("x", "y", "z", "v3", "v4")
    space = build_config_space(
        cluster_id="0",
        labels=("A",),
        patterns=(),
        models=(),
        variables=variable_names(3),
        relations=paper_relations(variable_names(3), ["cnt"]),
        candidate_domain="local",
    )
    assert space.variables == ("x",) and not space.relations


def test_all_inventory_is_one_actual_extraction_without_quota_or_mask_merging(tmp_path):
    data = example()
    data.update(
        body_quota=None,
        title_quota=None,
        tfidf_terms=None,
        deduplicate_masks=False,
        exclude_constant_masks=False,
    )
    data["docs"] = [
        Document(cnt=d.cnt + " sharedmarker", ttl=d.ttl, mtd=d.mtd) for d in data["docs"]
    ]
    out = tmp_path / "all_inventory"
    complete = patterns.build_supply(output=out, **data)
    protocol = json.loads((out / "PROTOCOL.json").read_text())
    assert protocol["max_per_cluster"] is None
    assert protocol["tfidf_terms_per_cluster"] is None
    assert protocol["fit_mask_deduplication"] is False
    assert protocol["support"]["constant_masks_excluded"] is False
    with sqlite3.connect(out / "candidates.sqlite") as db:
        assert (
            db.execute(
                "SELECT count(*) FROM candidates WHERE reason IN ('fit_mask_diversity_prefilter','outside_declared_diversity_budget','constant_mask')"
            ).fetchone()[0]
            == 0
        )
        assert (
            db.execute(
                "SELECT count(*) FROM candidates WHERE reason='selected' AND support=10"
            ).fetchone()[0]
            > 0
        )
        assert (
            db.execute(
                "SELECT count(*) FROM candidates WHERE reason='selected' AND support=0"
            ).fetchone()[0]
            == 0
        )
        assert (
            db.execute(
                "SELECT count(*) FROM frequency_partitions WHERE support=10 AND reason='eligible'"
            ).fetchone()[0]
            > 0
        )
    summary = json.loads((out / "SUMMARY.json").read_text())
    assert all(
        group["selected"] == group["eligible_before_diversity"] for group in summary["groups"]
    )
    with np.load(out / "cluster_00/FIT_MASKS.npz", allow_pickle=False) as saved:
        masks = saved["masks"]
        # Distinct predicates with identical observed masks remain available.
        assert len({np.packbits(m).tobytes() for m in masks}) < len(masks)
    for key in (
        "candidate_archive",
        "body_raw",
        "title_raw",
        "body_extraction",
        "title_extraction",
        "additional_anchors",
        "tfidf_terms",
    ):
        assert complete[key] == file_record(complete[key]["path"])
    assert complete["BO_calls"] == complete["Chase_calls"] == 0


def test_supply_cli_all_controls_are_explicit_and_old_defaults_remain(monkeypatch):
    captured = []
    monkeypatch.setattr(patterns, "build_from_manifest", lambda **kw: captured.append(kw) or {})
    required = [
        "--supervision-manifest",
        "s",
        "--cluster-arrays",
        "a",
        "--clustering",
        "c",
        "--output",
        "o",
    ]
    assert patterns.main(required) == 0
    assert captured[-1]["body_quota"] == 48 and captured[-1]["title_quota"] == 16
    assert captured[-1]["tfidf_terms"] == 128
    assert captured[-1]["deduplicate_masks"] and captured[-1]["exclude_constant_masks"]
    assert (
        patterns.main(
            required
            + [
                "--body-quota",
                "all",
                "--title-quota",
                "all",
                "--tfidf-terms",
                "all",
                "--no-deduplicate-masks",
                "--no-exclude-constant-masks",
            ]
        )
        == 0
    )
    assert all(captured[-1][key] is None for key in ("body_quota", "title_quota", "tfidf_terms"))
    assert captured[-1]["deduplicate_masks"] is captured[-1]["exclude_constant_masks"] is False
    args = parser().parse_args(
        [
            "check",
            "--manifest",
            "x",
            "--max-variables",
            "3",
            "--context-probes",
            "128",
            "--acquisition-candidates",
            "256",
            "--expected-body-size",
            "4",
        ]
    )
    assert (
        args.max_variables,
        args.context_probes,
        args.acquisition_candidates,
        args.expected_body_size,
    ) == (3, 128, 256, 4.0)


def test_new_supply_controls_fail_before_creating_output(tmp_path):
    for override in (
        {"deduplicate_masks": 0},
        {"exclude_constant_masks": None},
        {"screening": "mcc", "deduplicate_masks": False},
        {"tfidf_terms": 0},
        {"body_quota": 0},
        {"title_quota": -1},
    ):
        with pytest.raises(ValueError):
            patterns.build_supply(output=tmp_path / "invalid", **dict(example(), **override))
    assert not (tmp_path / "invalid").exists()
