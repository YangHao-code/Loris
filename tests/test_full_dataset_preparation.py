"""Boundary contracts for native data preparation, without downloading data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_full_datasets.py"
SPEC = importlib.util.spec_from_file_location("loris_native_preparer", SCRIPT)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


def test_latest_id_selection_never_unions_labels_and_uses_date_then_version():
    latest = native.LatestVersions()
    latest.add("paper", "2024-01-01", 8, ("title", "old abstract", "cs.AI"), ("a", 1))
    latest.add("paper", "2025-01-01", 1, ("title", "new abstract", "math.PR"), ("b", 2))
    latest.add("paper", "2025-01-01", 2, ("title", "corrected", "cs.LG"), ("c", 3))
    assert latest.locations() == {"paper": ("c", 3)}
    assert latest.duplicates == {"paper": 2}


def test_latest_payload_tie_is_rejected_but_obsolete_tie_does_not_block_new_version():
    latest = native.LatestVersions()
    latest.add("paper", "2024-01-01", 1, ("a", "b", "c"), ("a", 0))
    latest.add("paper", "2024-01-01", 1, ("a", "different", "c"), ("b", 0))
    with pytest.raises(ValueError, match="tied at latest"):
        latest.locations()
    latest.add("paper", "2024-01-02", 1, ("a", "corrected", "c"), ("c", 0))
    assert latest.locations() == {"paper": ("c", 0)}


def test_official_membership_is_preserved_and_train_derived_dev_groups_text():
    for role in native.SPLITS:
        for seed in ("first", "second"):
            assert native.assign_role("A document", role, seed=seed) == role
    first = native.assign_role(" A  DOCUMENT ", "official_train_needs_dev")
    second = native.assign_role("a document", "official_train_needs_dev")
    assert first == second
    assert first in {"train", "dev"}
    assert native.assign_role("a", "not_modapte") == "not_modapte"


def test_complete_empty_gold_and_unsupported_axis_survive_public_materializer(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("original untouched")
    records = [
        native.record("train-id", "training text", ["A"], {"title": "train"}, {"row": 0}, "train"),
        native.record("dev-id", "development text", [], {"title": "dev"}, {"row": 1}, "dev"),
        native.record("test-id", "test text", ["B"], {"title": "test"}, {"row": 2}, "test"),
    ]
    output = tmp_path / "prepared"
    manifest = native.materialize(
        lambda: iter(records),
        ["A", "B", "never-seen"],
        [source],
        ["title"],
        {"official_splits": True},
        {},
        output,
        "native-example",
    )
    assert manifest["counts"] == {"train": 1, "dev": 1, "test": 1}
    with np.load(output / "truth/dev.npz", allow_pickle=False) as truth:
        assert truth["row_ids"].tolist() == ["dev-id"]
        assert truth["labels"].tolist() == ["A", "B", "never-seen"]
        assert truth["targets"].tolist() == [[False, False, False]]
        assert truth["observed"].all()
    document = json.loads((output / "documents/dev.jsonl").read_text())
    assert set(document) == {"row_id", "text", "metadata", "source"}
    assert "labels" not in document
    assert source.read_text() == "original untouched"
    with pytest.raises(FileExistsError, match="new output"):
        native.materialize(
            lambda: iter(records), ["A"], [source], ["title"], {}, {}, output, "another"
        )


def test_duplicate_retained_id_fails_without_touching_source(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("raw")
    records = [
        native.record("same", "x", [], {}, {}, "train"),
        native.record("same", "y", ["A"], {}, {}, "test"),
    ]
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="duplicate retained"):
        native.materialize(lambda: iter(records), ["A"], [source], [], {}, {}, output, "bad")
    assert (output / "FAILED.json").is_file()
    assert not (output / "MANIFEST.json").exists()
    assert source.read_text() == "raw"


def test_missing_arxiv_shard_is_rejected_before_any_source_is_read():
    with pytest.raises(ValueError, match="incomplete"):
        native.validate_shards([Path("train-00000-of-00002.parquet")])
    native.validate_shards(
        [Path("train-00000-of-00002.parquet"), Path("train-00001-of-00002.parquet")]
    )


def test_actual_parquet_conflicting_latest_ids_leave_explicit_failure(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "raw"
    source.mkdir()
    path = source / "train-00000-of-00001.parquet"
    rows = [
        {
            "id": "one",
            "title": "title",
            "abstract": "first",
            "categories": "A",
            "update_date": "2025-01-01",
            "versions": [{"version": "v1", "created": "2025-01-01"}],
        },
        {
            "id": "one",
            "title": "title",
            "abstract": "different",
            "categories": "B",
            "update_date": "2025-01-01",
            "versions": [{"version": "v1", "created": "2025-01-01"}],
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    before = path.read_bytes()
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="payload ties"):
        native.main(["--dataset", "arxiv", "--source", str(source), "--output", str(output)])
    assert json.loads((output / "DUPLICATE_CONFLICTS.json").read_text()) == {
        "one": [[path.name, 0], [path.name, 1]]
    }
    assert (output / "FAILED.json").is_file()
    assert not (output / "MANIFEST.json").exists()
    assert path.read_bytes() == before
