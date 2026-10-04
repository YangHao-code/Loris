import csv
import json
from pathlib import Path

import numpy as np
import pytest

from loris.discovery.datasets import prepare_dataset
from loris.discovery.manifest import file_record
from loris.discovery.supervision import (load_fit_supervision, load_supervision,
                                        prepare_supervision, select_known_rows)


def prepared(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    records = {}
    for split, count in (("train", 17), ("dev", 3), ("test", 2)):
        path = tmp_path / (split + ".csv")
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["text", "real_group", "a", "b"])
            for i in range(count):
                writer.writerow([split + " text " + str(i), "" if i % 2 else "shared", i % 2, (i + 1) % 2])
        records[split] = file_record(path)
    spec = {"schema": "loris_dataset_source_v1", "dataset": "toy", "version": "1", "sources": records,
            "columns": {"text": ["text"], "id": None, "labels": ["a", "b"], "metadata": ["real_group"]},
            "label_axis": {"mode": "declared", "labels": ["a", "b"], "provenance": "fixture"},
            "split": {"seed": 42, "fractions": [.1, .1, .8], "dev_fraction": .1, "group_column": None},
            "provenance": {"source": "synthetic only"}}
    directory = tmp_path / "prepared"
    prepare_dataset(spec, directory)
    return directory / "MANIFEST.json"


def prepare(manifest, output):
    # Historical partial-supervision cases remain explicit ablations.
    return prepare_supervision(manifest, output, historical_exposure={"scope": "synthetic only"},
                               fraction_numerator=7204, fraction_denominator=58715)


def test_membership_is_label_blind_and_row_order_independent():
    ids = ["doc:" + str(i) for i in range(53840)]
    indices = select_known_rows(ids, "aapd", fraction_numerator=7204, fraction_denominator=58715)
    assert len(indices) == 6606
    reversed_ids = list(reversed(ids))
    other = select_known_rows(reversed_ids, "aapd", fraction_numerator=7204, fraction_denominator=58715)
    assert {ids[i] for i in indices} == {reversed_ids[i] for i in other}
    for kwargs in ({"seed": True}, {"fraction_numerator": 0}, {"fraction_denominator": 0},
                   {"fraction_numerator": 2, "fraction_denominator": 1}):
        with pytest.raises(ValueError):
            select_known_rows(ids, "aapd", **kwargs)


def test_default_uses_every_train_answer_without_dev_or_test_roots(tmp_path):
    source = prepared(tmp_path / "source")
    out = tmp_path / "full_train"
    # No test access, even when every training answer is known.
    (source.parent / "truth/test.npz").unlink()
    (source.parent / "documents/test.jsonl").unlink()
    result = prepare_supervision(source, out, historical_exposure={"scope": "fixture"})
    assert result["counts"]["known_train"] == result["counts"]["train"] == 17
    assert result["counts"]["unknown_train"] == 0
    data = load_supervision(out / "SUPERVISION_MANIFEST.json")
    assert data["fit_indices"].tolist() == list(range(17))
    assert (data["gamma_pos"] | data["gamma_neg"])[:17].all()
    assert not (data["gamma_pos"] | data["gamma_neg"])[17:].any()


def test_complete_signed_known_only_no_test_access(tmp_path):
    source = prepared(tmp_path / "source")
    # Missing source test files do not prevent supervision preparation.
    # Opening/hashing any of them would fail, proving the allow-list boundary.
    for path in [source.parent / "truth/test.npz", source.parent / "documents/test.jsonl",
                 tmp_path / "source/test.csv"]:
        path.unlink()
    out = tmp_path / "known"
    result = prepare(source, out)
    assert result["counts"] == {"train": 17, "dev": 3, "discovery": 20, "known_train": 3,
                                "unknown_train": 14, "excluded_test": 2, "labels": 2}
    assert not any("test" in key for key in result["source_files_read"])
    loaded = load_supervision(out / "SUPERVISION_MANIFEST.json")
    assert loaded["fit_targets"].shape == (3, 2)
    assert loaded["dev_targets"].shape == (3, 2)
    known = np.zeros(20, bool)
    known[loaded["fit_indices"]] = True
    assert (loaded["gamma_pos"] | loaded["gamma_neg"])[known].all()
    assert not (loaded["gamma_pos"] | loaded["gamma_neg"])[~known].any()
    assert not (loaded["gamma_pos"] & loaded["gamma_neg"]).any()
    assert loaded["row_roles"][-3:] == ["val_ctrl"] * 3
    assert all(not set(row) & {"labels", "targets", "observed"} for row in loaded["docs"])


def test_runtime_reader_never_reopens_unmasked_sources_and_fit_never_reads_dev_gold(tmp_path):
    source = prepared(tmp_path / "source")
    out = tmp_path / "known"
    prepare(source, out)
    expected = load_supervision(out / "SUPERVISION_MANIFEST.json")
    (source.parent / "truth/train.npz").unlink()
    (source.parent / "truth/dev.npz").unlink()
    (out / "scoring/dev_truth.npz").unlink()
    actual = load_fit_supervision(out / "SUPERVISION_MANIFEST.json")
    assert np.array_equal(actual["fit_truth"], expected["fit_targets"])
    assert len(actual["docs"]) == 3
    assert all(row["role"] == "source_fit" for row in actual["docs"])
    assert "scoring/dev_truth.npz" not in actual["files_read"]
    assert "dev_targets" not in actual
    with pytest.raises(FileNotFoundError):
        load_supervision(out / "SUPERVISION_MANIFEST.json")


def test_unknown_source_target_changes_cannot_change_exported_answers(tmp_path):
    source = prepared(tmp_path / "source")
    first = tmp_path / "known1"
    prepare(source, first)
    expected = load_supervision(first / "SUPERVISION_MANIFEST.json")
    spec = json.loads(source.read_text())
    truth = Path(spec["artifacts"]["truth/train.npz"]["path"])
    with np.load(truth, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    unknown = np.ones(len(arrays["targets"]), bool)
    unknown[expected["fit_indices"]] = False
    arrays["targets"][unknown] = ~arrays["targets"][unknown]
    np.savez_compressed(truth, **arrays)
    spec["artifacts"]["truth/train.npz"] = file_record(truth)
    source.write_text(json.dumps(spec))
    second = tmp_path / "known2"
    prepare(source, second)
    actual = load_supervision(second / "SUPERVISION_MANIFEST.json")
    assert np.array_equal(actual["fit_indices"], expected["fit_indices"])
    assert np.array_equal(actual["gamma_pos"], expected["gamma_pos"])
    assert np.array_equal(actual["gamma_neg"], expected["gamma_neg"])
    with np.load(second / "roots/train_known_masked.npz") as archive:
        assert not archive["targets"][unknown].any()
        assert not archive["observed"][unknown].any()


def test_malformed_mask_and_identity_tampering_rejected(tmp_path):
    source = prepared(tmp_path / "source")
    out = tmp_path / "known"
    prepare(source, out)
    manifest = out / "SUPERVISION_MANIFEST.json"
    masked = out / "roots/train_known_masked.npz"
    with np.load(masked, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["targets"][~arrays["known_rows"]] = True
    np.savez_compressed(masked, **arrays)
    with pytest.raises(ValueError, match="identity differs"):
        load_supervision(manifest)
    spec = json.loads(manifest.read_text())
    spec["artifacts"]["roots/train_known_masked.npz"] = file_record(masked)
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="unknown targets must be masked"):
        load_supervision(manifest)


def test_no_overwrite_or_implicit_historical_clean_claim(tmp_path):
    source = prepared(tmp_path / "source")
    out = tmp_path / "known"
    with pytest.raises(ValueError, match="historical_exposure"):
        prepare_supervision(source, out, historical_exposure={})
    assert not out.exists()
    prepare(source, out)
    with pytest.raises(FileExistsError):
        prepare(source, out)
