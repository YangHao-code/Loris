"""Raw provenance survives while only explicit semantic metadata enters rules."""

import copy
import json

import numpy as np
import pytest

from loris.discovery import upstream
from loris.discovery.manifest import load_inputs
from loris._engine.rdl_reference import Doc, Label, AttrEqual, Rule, RefGamma, ReferenceChase
from tests.test_discovery_variants import discovery_manifest
from tests.test_discovery_upstream import fixture_assets


def test_projected_metadata_excludes_administrative_fields_without_changing_records():
    records = [
        dict(
            text="document",
            metadata=dict(
                title="Real title", author="Alice", source_split="train", source_record="17"
            ),
        )
    ]
    before = copy.deepcopy(records)
    views = upstream.document_views(
        records, np.array([1]), semantic_metadata_fields=["title", "author"]
    )
    assert json.loads(views["docs"][0].mtd) == dict(title="Real title", author="Alice")
    assert views["docs"][0].ttl == "Real title"
    assert set(views["attribute_values"]) == {"mtd.title", "mtd.author", "mtd.semantic_cluster_id"}
    assert records == before
    legacy = upstream.document_views(records)
    assert json.loads(legacy["docs"][0].mtd)["source_split"] == "train"
    records[0]["metadata"]["source_record"] = 17
    projected = upstream.document_views(records, semantic_metadata_fields=["title", "author"])
    assert projected["docs"][0].mtd == views["docs"][0].mtd
    assert records[0]["metadata"]["source_record"] == 17
    with pytest.raises(ValueError, match="metadata values"):
        upstream.document_views(records)  # The legacy all-field contract stays strict.


@pytest.mark.parametrize("attribute", ["ttl", "mtd"])
def test_empty_semantic_view_does_not_create_empty_equalities_or_truth(attribute):
    records = [
        dict(text="red", metadata=dict(source_split="train", source_record=str(i)))
        for i in range(2)
    ]
    views = upstream.document_views(records, np.array([0, 0]), semantic_metadata_fields=[])
    assert all(doc.ttl == doc.mtd == "" for doc in views["docs"])
    assert not views["attribute_presence"][attribute].any()
    pos = np.array([[1], [0]], bool)
    gamma = RefGamma(pos, np.zeros_like(pos))
    rule = Rule(
        (Doc("x"), Doc("y"), Label("x", "A"), AttrEqual("x", attribute, "y", attribute)),
        Label("y", "A"),
    )
    chase = ReferenceChase(
        ["A"],
        attribute_values=views["attribute_values"],
        attribute_presence=views["attribute_presence"],
    )
    result = chase.run(views["docs"], [rule], gamma)
    np.testing.assert_array_equal(result.positive_facts, pos)


def test_context_receipt_binds_projection_and_changes_identity(tmp_path, monkeypatch):
    manifest, features, handoff, _, _ = fixture_assets(tmp_path, monkeypatch)
    first = upstream.prepare_context(manifest, handoff, features, tmp_path / "legacy")
    second = upstream.prepare_context(
        manifest, handoff, features, tmp_path / "semantic", semantic_metadata_fields=[]
    )
    assert first["semantic_metadata_fields"] is None
    assert second["semantic_metadata_fields"] == []
    assert first["context"]["sha256"] != second["context"]["sha256"]
    from loris.discovery.patterns import build_from_manifest

    supply = tmp_path / "projected_supply"
    build_from_manifest(
        supervision_manifest=manifest,
        cluster_arrays=features.parent / "CLUSTER_ARRAYS.npz",
        clustering=features.parent / "CLUSTERING.json",
        output=supply,
        semantic_metadata_fields=[],
        min_support=1,
        body_quota=None,
        title_quota=None,
        tfidf_terms=None,
        deduplicate_masks=False,
        exclude_constant_masks=False,
    )
    assert json.loads((supply / "FIT_IDENTITY.json").read_text())["semantic_metadata_fields"] == []
    policy = json.loads((supply / "PROTOCOL.json").read_text())
    assert policy["inputs"]["semantic_metadata_fields"] == []
    summary = json.loads((supply / "SUMMARY.json").read_text())
    assert all(row["raw"] == 0 for row in summary["groups"] if row["attr"] == "ttl")


def test_supply_and_context_projection_mismatch_is_rejected_even_when_texts_match(tmp_path):
    manifest = discovery_manifest(tmp_path)
    identity = tmp_path / "FIT_IDENTITY.json"
    data = json.loads(identity.read_text())
    data["semantic_metadata_fields"] = []
    identity.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different fit inputs"):
        load_inputs(manifest)


@pytest.mark.parametrize("fields", [["missing"], ["title", "title"], "title", [True]])
def test_bad_projection_is_not_inferred_or_coerced(fields):
    with pytest.raises(ValueError, match="semantic_metadata_fields"):
        upstream.document_views(
            [dict(text="x", metadata=dict(title="title"))], semantic_metadata_fields=fields
        )
