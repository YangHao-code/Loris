import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from loris.discovery.manifest import file_record
from loris.discovery.supervision import load_supervision
from loris.discovery import upstream
from tests.test_discovery_supervision import prepared, prepare


def fixture_assets(tmp_path, monkeypatch):
    source = prepared(tmp_path / "source")
    supervised = tmp_path / "supervision"
    prepare(source, supervised)
    manifest = supervised / "SUPERVISION_MANIFEST.json"
    data = load_supervision(manifest)
    snapshot = tmp_path / "synthetic_feature_model"
    snapshot.mkdir()
    (snapshot / "modules.json").write_text('{"fixture_only":true}')
    random = np.random.RandomState(3)
    matrix = random.normal(size=(len(data["docs"]), 384)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    monkeypatch.setattr(upstream, "_encode_documents", lambda *args, **kwargs: matrix.copy())
    features = tmp_path / "features"
    upstream.prepare_features(manifest, features, backbone=str(snapshot), device="cpu", n_clusters=2)
    pool = tmp_path / "pool"
    pool.mkdir()
    channels = {}
    # Intentionally opposite lexical order: fixed combo tie ordering must survive
    # JSON sort_keys and must not be inferred from dictionary insertion order.
    order = ["z_model", "a_model", "m_model", "b_model"]
    for alias in order:
        path = pool / (alias + ".npz")
        scores = random.uniform(size=(len(data["docs"]), len(data["labels"]))).astype(np.float32)
        np.savez_compressed(path, scores=scores, labels=np.asarray(data["labels"]),
                            row_ids=np.asarray(data["row_ids"]), global_rows=np.arange(len(data["docs"])))
        channels[alias] = {"probabilities": file_record(path)}
    handoff = {"status": "complete", "supervision_manifest": file_record(manifest), "channels": channels,
               "row_ids": data["row_ids"], "labels": data["labels"], "row_roles": data["row_roles"],
               "model_order": order}
    hand = pool / "HANDOFF.json"
    hand.write_text(json.dumps(handoff, sort_keys=True))
    return manifest, features / "FEATURES_COMPLETE.json", hand, data, order


def test_documents_preserve_missing_metadata_without_fabrication():
    records = [{"text": "abstract", "metadata": {}}, {"text": "body", "metadata": {"title": "Title", "author": ""}}]
    result = upstream.document_views(records, np.asarray([0, 1]))
    assert result["docs"][0].cnt == "abstract"
    assert result["docs"][0].ttl == ""
    assert result["docs"][0].mtd == ""
    assert result["docs"][1].ttl == "Title"
    assert result["attribute_values"]["mtd.author"] == ["", ""]
    assert not result["attribute_presence"]["mtd.author"].any()
    assert result["attribute_values"]["mtd.semantic_cluster_id"] == ["fit_kmeans:0", "fit_kmeans:1"]
    assert "semantic_cluster_id" not in records[0]["metadata"]
    with pytest.raises(ValueError, match="overwrite"):
        upstream.document_views([{"text": "x", "metadata": {"semantic_cluster_id": "original"}}], np.asarray([0]))


def test_fit_clusters_never_fit_dev_features():
    features = np.asarray([[0., 0.], [0., .1], [10., 10.], [10., 10.1], [1e6, 1e6]])
    first, labels = upstream.fit_clusters(features, np.arange(4), n_clusters=2)
    features[-1] = [-1e8, 0]
    second, changed = upstream.fit_clusters(features, np.arange(4), n_clusters=2)
    assert np.array_equal(first.cluster_centers_, second.cluster_centers_)
    assert np.array_equal(labels[:4], changed[:4])


def test_real_axis_context_selector_flow_and_fixed_model_order(tmp_path, monkeypatch):
    manifest, features, hand, data, order = fixture_assets(tmp_path, monkeypatch)
    context = tmp_path / "context"
    done = upstream.prepare_context(manifest, hand, features, context)
    assert done["model_order"] == order
    with (context / "discovery_context.pkl").open("rb") as f:
        ctx = pickle.load(f)
    assert list(ctx["model_probabilities"]) == order
    assert not ctx["gamma"].pos[data["receiver_indices"]].any()
    assert not ctx["gamma"].neg[data["receiver_indices"]].any()
    assert set(ctx["source_gold"]) == {"fit", "val_ctrl"}
    assert all(not d.lbl for d in ctx["docs"])
    selector = tmp_path / "selector"
    protocol = dict(upstream.DEFAULT_SELECTOR_PROTOCOL, epochs=1, num_samples=4, device="cpu")
    upstream.prepare_selector(manifest, context / "COMPLETE.json", features, selector, protocol=protocol)
    result = json.loads((selector / "select_models.manifest.json").read_text())
    assert result["aliases"] == order
    with np.load(selector / "SELECTOR_INPUTS.npz", allow_pickle=False) as z:
        assert np.array_equal(z["truth"], data["fit_targets"])
        assert z["features"].shape == (3, 384)
        assert z["decisions"].shape == (4, 3, 2)
        assert not any("dev" in key or "test" in key for key in z.files)
    # The maintained selector-only boundary validates without loading labeling.
    from loris.discovery.selector_backend import validate_selector_manifest
    from loris.discovery.__main__ import main
    validate_selector_manifest(result)
    assert main(["train-selector", "--manifest", str(selector / "select_models.manifest.json"),
                 "--output", str(tmp_path / "tiny_synthetic_selector")]) == 0
    assert (tmp_path / "tiny_synthetic_selector/COMPLETE.json").is_file()


def test_probability_axes_or_order_cannot_be_renamed_into_compatibility(tmp_path, monkeypatch):
    manifest, features, hand, data, order = fixture_assets(tmp_path, monkeypatch)
    spec = json.loads(hand.read_text())
    spec["model_order"] = [order[0]] * len(order)
    hand.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="fixed model_order"):
        upstream.prepare_context(manifest, hand, features, tmp_path / "bad_order")
    spec["model_order"] = order
    probability = Path(spec["channels"][order[0]]["probabilities"]["path"])
    with np.load(probability, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["row_ids"] = arrays["row_ids"][::-1]
    np.savez_compressed(probability, **arrays)
    spec["channels"][order[0]]["probabilities"] = file_record(probability)
    hand.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="probability identity"):
        upstream.prepare_context(manifest, hand, features, tmp_path / "bad_probability")


def test_feature_stage_does_not_open_dev_scoring(tmp_path, monkeypatch):
    source = prepared(tmp_path / "source")
    sup = tmp_path / "supervision"
    prepare(source, sup)
    (sup / "scoring/dev_truth.npz").unlink()
    snapshot = tmp_path / "synthetic_only"
    snapshot.mkdir()
    (snapshot / "modules.json").write_text("fixture")
    arr = np.random.RandomState(4).normal(size=(20,384)).astype(np.float32)
    monkeypatch.setattr(upstream, "_encode_documents", lambda *args, **kwargs: arr)
    done = upstream.prepare_features(sup / "SUPERVISION_MANIFEST.json", tmp_path / "features",
                                     backbone=snapshot, device="cpu", n_clusters=2)
    assert done["dev_targets_read"] is False
    assert done["cost"]["peak_gpu_allocated_bytes"] is None


def test_missing_backbone_produces_failure_receipt_before_encoding(tmp_path, monkeypatch):
    source = prepared(tmp_path / "source")
    sup = tmp_path / "supervision"
    prepare(source, sup)
    snapshot = tmp_path / "empty_model_directory"
    snapshot.mkdir()
    monkeypatch.setattr(upstream, "_encode_documents", lambda *a, **k: pytest.fail("must not encode"))
    out = tmp_path / "failed_features"
    with pytest.raises(ValueError, match="modules.json"):
        upstream.prepare_features(sup / "SUPERVISION_MANIFEST.json", out, backbone=snapshot, device="cpu", n_clusters=2)
    assert json.loads((out / "FAILED.json").read_text())["status"] == "failed"
    assert not (out / "FEATURES_COMPLETE.json").exists()


def test_explicit_title_field_has_same_pattern_and_context_semantics(tmp_path, monkeypatch):
    records = [{"text": "title plus abstract", "metadata": {"Title": "Real title", "pmid": "123"}}]
    assert upstream.records_to_documents(records)[0].ttl == ""
    assert upstream.records_to_documents(records, title_field="Title")[0].ttl == "Real title"
    assert upstream.document_views(records, title_field="Title")["docs"][0].ttl == "Real title"
    manifest, features, hand, data, order = fixture_assets(tmp_path, monkeypatch)
    default_dir, explicit_dir = tmp_path / "default_context", tmp_path / "explicit_context"
    default = upstream.prepare_context(manifest, hand, features, default_dir)
    explicit = upstream.prepare_context(manifest, hand, features, explicit_dir, title_field="Title")
    assert explicit["title_field"] == "Title" and default["title_field"] == "title"
    with (default_dir / "discovery_context.pkl").open("rb") as f:
        first = pickle.load(f)
    with (explicit_dir / "discovery_context.pkl").open("rb") as f:
        second = pickle.load(f)
    assert first["context_id"] != second["context_id"]
    assert second["receipt"]["binding"]["text_interpretation"]["title_field"] == "Title"


def test_named_backbone_uses_standard_huggingface_cache_offline(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants
    source = prepared(tmp_path / "source")
    sup = tmp_path / "supervision"
    prepare(source, sup)
    snapshot = tmp_path / "fixture_model"
    snapshot.mkdir()
    (snapshot / "modules.json").write_text("fixture")
    calls = []
    def resolve(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", resolve)
    array = np.random.RandomState(4).normal(size=(20,384)).astype(np.float32)
    monkeypatch.setattr(upstream, "_encode_documents", lambda *a, **k: array)
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(key, raising=False)
    upstream.prepare_features(sup / "SUPERVISION_MANIFEST.json", tmp_path / "default_cache", device="cpu", n_clusters=2)
    assert calls[-1]["cache_dir"] == constants.HF_HUB_CACHE
    monkeypatch.setenv("HF_HOME", str(tmp_path / "custom_home"))
    upstream.prepare_features(sup / "SUPERVISION_MANIFEST.json", tmp_path / "home_cache", device="cpu", n_clusters=2)
    assert calls[-1]["cache_dir"] == str(tmp_path / "custom_home/hub")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit_hub"))
    upstream.prepare_features(sup / "SUPERVISION_MANIFEST.json", tmp_path / "explicit_cache", device="cpu", n_clusters=2)
    assert calls[-1]["cache_dir"] == str(tmp_path / "explicit_hub")
    assert all(call["local_files_only"] is True for call in calls)
