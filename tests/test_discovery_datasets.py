import csv
import json

import numpy as np
import pytest

from loris.discovery.datasets import SOURCE_SCHEMA, assign_splits, import_aapd, prepare_dataset
from loris.discovery.manifest import file_record


def test_hash_split_is_label_blind_stable_and_majority_test():
    rows = ["doc:" + str(i) for i in range(100)]
    first = assign_splits(rows)
    second = assign_splits(list(reversed(rows)))
    assert first["counts"] == {"train": 10, "dev": 10, "test": 80}
    assert {s: set(v) for s, v in first["membership"].items()} == {
        s: set(v) for s, v in second["membership"].items()}
    assert set().union(*map(set, first["membership"].values())) == set(rows)
    assert not first["labels_used_for_split"]


def test_official_split_wins_and_missing_dev_stays_inside_train():
    train, test = ["tr" + str(i) for i in range(20)], ["te" + str(i) for i in range(80)]
    result = assign_splits(train + test, official_splits={"train": train, "test": test})
    assert result["membership"]["test"] == test
    assert set(result["membership"]["train"] + result["membership"]["dev"]) == set(train)
    assert result["counts"] == {"train": 18, "dev": 2, "test": 80}
    explicit = {"train": train[:15], "dev": train[15:], "test": test}
    assert assign_splits(train + test, official_splits=explicit)["membership"] == explicit


def test_groups_stay_together_and_missing_metadata_does_not_connect_rows():
    rows = [str(i) for i in range(20)]
    groups = {row: str(i // 2) for i, row in enumerate(rows)}
    result = assign_splits(rows, group_ids=groups)
    roles = {row: split for split, values in result["membership"].items() for row in values}
    for i in range(0, 20, 2):
        assert roles[str(i)] == roles[str(i + 1)]
    assert result["group_counts"] == {"train": 1, "dev": 1, "test": 8}
    empty = assign_splits(rows, group_ids={row: "" for row in rows})
    assert sum(empty["group_counts"].values()) == 20
    official = {"train": ["a"], "dev": ["c"], "test": ["b"]}
    fixed = assign_splits(["a", "b", "c"], official_splits=official,
                          group_ids={"a": "author", "b": "author"})
    assert fixed["membership"] == official
    assert fixed["group_overlap_count"] == 1
    assert fixed["group_overlaps"][0]["rows_by_split"] == {"train": ["a"], "test": ["b"]}


def test_invalid_partition_cannot_silently_drop_or_duplicate_documents():
    with pytest.raises(ValueError, match="unique"):
        assign_splits(["a", "a", "b"])
    with pytest.raises(ValueError, match="exhaust"):
        assign_splits(["a", "b", "c"], official_splits={"train": ["a"], "test": ["b"]})
    with pytest.raises(ValueError, match="at least three"):
        assign_splits(["a", "b"])
    with pytest.raises(ValueError, match="summing"):
        assign_splits(["a", "b", "c"], fractions=(0.1, 0.1, 0.1))


def source(tmp_path, *, mode="declared", train_targets=((1, 0), (0, 0)), dev_targets=((0, 1),)):
    rows = {"train": train_targets, "dev": dev_targets, "test": ((0, 1),)}
    refs = {}
    for split, target_rows in rows.items():
        path = tmp_path / (split + ".csv")
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["text", "author", "a", "b"])
            for i, y in enumerate(target_rows):
                writer.writerow(["" if i == 0 else "real text", "", *y])
        refs[split] = file_record(path)
    return {"schema": SOURCE_SCHEMA, "dataset": "fixture", "version": "v1", "sources": refs,
            "columns": {"text": ["text"], "id": None, "labels": ["a", "b"], "metadata": ["author"]},
            "label_axis": {"mode": mode, "labels": ["a", "b"] if mode == "declared" else None,
                           "provenance": "fixture released axis"},
            "split": {"seed": 42, "fractions": [0.1, 0.1, 0.8], "dev_fraction": 0.1, "group_column": None},
            "provenance": {"source": "synthetic test fixture, never reported as actual data"}}


def test_import_keeps_unsupported_axis_and_empty_text_without_roots(tmp_path):
    spec = source(tmp_path)
    output = tmp_path / "prepared"
    result = prepare_dataset(spec, output)
    assert result["counts"] == {"train": 2, "dev": 1, "test": 1}
    assert result["labels"] == ["a", "b"]
    assert result["support"]["unsupported_axis_labels"] == ["b"]
    assert result["rows_dropped"] == 0
    assert result["text_missing_rows"] == 3
    assert result["Gamma_created"] is False
    doc = json.loads((output / "documents/dev.jsonl").read_text())
    assert "targets" not in doc and doc["text"] == ""
    with np.load(output / "truth/dev.npz", allow_pickle=False) as array:
        assert array["observed"].all()
        assert array["targets"].tolist() == [[False, True]]
    with pytest.raises(FileExistsError):
        prepare_dataset(spec, output)


def test_axis_inference_uses_train_not_dev_or_test(tmp_path):
    result = prepare_dataset(source(tmp_path, mode="train_supported"), tmp_path / "prepared")
    assert result["labels"] == ["a"]
    assert result["support"]["excluded_candidate_labels"] == ["b"]
    assert result["support"]["dev_test_used_to_choose_labels"] is False


def test_source_identity_and_unknown_answers_are_not_silently_reinterpreted(tmp_path):
    spec = source(tmp_path)
    spec["sources"]["train"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA mismatch"):
        prepare_dataset(spec, tmp_path / "bad_sha")
    spec = source(tmp_path, train_targets=((1, ""), (0, 0)))
    with pytest.raises(ValueError, match="unknown is not negative"):
        prepare_dataset(spec, tmp_path / "unknown")
    assert not (tmp_path / "unknown").exists()


def test_truth_columns_cannot_be_group_or_text_inputs(tmp_path):
    spec = source(tmp_path)
    spec["columns"]["metadata"] = ["b"]
    spec["split"]["group_column"] = "b"
    with pytest.raises(ValueError, match="label fields cannot"):
        prepare_dataset(spec, tmp_path / "leaky")


def test_aapd_adapter_rejects_truncated_or_old_top30_assets(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("abstract," + ",".join("l" + str(i) for i in range(30)) + "\n")
    with pytest.raises(ValueError, match="exactly54"):
        import_aapd(tmp_path, tmp_path / "fake_full")
