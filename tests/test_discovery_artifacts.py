"""Artifact integrity and public command boundaries, without production data."""
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pytest

from loris.discovery.archive import archive_index, snapshot_sqlite, status
from loris.discovery.manifest import load_manifest, validate_context
from loris.discovery.__main__ import main
from loris.discovery.runner import run_discovery
from tests.test_discovery_runner import inputs


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    out = tmp_path_factory.mktemp("archive_source") / "run"
    run_discovery(output=out, **inputs())
    return out


def test_archive_preserves_all_trials_and_complete_admissions(completed, tmp_path):
    result = archive_index(completed, tmp_path / "index")
    assert result["completion"]["BO_trials"] == 4
    assert result["clusters"][0]["all_sequential_admissions_verified"]
    assert result["study_snapshots"][0]["table_rows"]["trials"] == 4
    assert not result["archive_is_self_contained"]
    assert not result["all_rejected_candidate_DAGs_materialized"]
    paths = {r["relative_path"] for r in result["files"]}
    assert "learning/cluster_00/trial_00003.json" in paths
    assert "learning/cluster_00/admission/candidate_00003.json" in paths
    assert "PROVENANCE.jsonl.gz" in paths


@pytest.mark.parametrize("damage", ["trial", "missing_admission", "wrong_decision", "facts", "proof", "proof_records"])
def test_tampering_is_not_a_complete_archive(completed, tmp_path, damage):
    run = tmp_path / "run"
    shutil.copytree(completed, run)
    # Receipts intentionally bind absolute immutable artifact paths. Rebind the
    # fixture before tampering so this test targets copied artifacts only.
    from loris.discovery.manifest import file_record
    proof = json.loads((run / "PROOF_REPLAY.json").read_text())
    proof["facts"] = file_record(run / "FACTS.npz")
    proof["proof_records"] = file_record(run / "PROVENANCE.jsonl.gz")
    (run / "PROOF_REPLAY.json").write_text(json.dumps(proof))
    receipt = json.loads((run / "COMPLETE.json").read_text())
    receipt["selection"] = file_record(run / "learning/SELECTED.json")
    receipt["proof"] = file_record(run / "PROOF_REPLAY.json")
    (run / "COMPLETE.json").write_text(json.dumps(receipt))
    cluster = run / "learning/cluster_00"
    if damage == "trial":
        p = cluster / "trial_00000.json"
        obj = json.loads(p.read_text()); obj["params"]["tamper"] = True
        p.write_text(json.dumps(obj))
    elif damage == "missing_admission":
        (cluster / "admission/candidate_00000.json").unlink()
    elif damage == "wrong_decision":
        p = cluster / "admission/candidate_00000.json"
        obj = json.loads(p.read_text()); obj["decision"]["accepted"] = not obj["decision"]["accepted"]
        p.write_text(json.dumps(obj))
    elif damage == "facts":
        (run / "FACTS.npz").write_bytes(b"invalid")
    elif damage == "proof_records":
        (run / "PROVENANCE.jsonl.gz").write_bytes(b"invalid")
    else:
        (run / "PROOF_REPLAY.json").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        archive_index(run, tmp_path / "index")


def test_sqlite_snapshot_includes_committed_wal(tmp_path):
    source = tmp_path / "live.sqlite"
    writer = sqlite3.connect(str(source))
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE trials (id INTEGER)")
        writer.execute("INSERT INTO trials VALUES (17)")
        writer.commit()
        assert Path(str(source) + "-wal").stat().st_size > 0
        result = snapshot_sqlite(source, tmp_path / "backup.sqlite")
        with sqlite3.connect(result["snapshot"]["path"]) as saved:
            assert saved.execute("SELECT id FROM trials").fetchall() == [(17,)]
    finally:
        writer.close()


def test_builtin_document_presence_is_valid_but_unbound_metadata_is_not():
    context = inputs()["context"]
    context["attribute_presence"]["cnt"] = np.ones(6, bool)
    assert validate_context(context)["documents"] == 6
    context["attribute_presence"]["fabricated"] = np.ones(6, bool)
    with pytest.raises(ValueError, match="presence"):
        validate_context(context)


def test_cli_reports_failure_without_creating_output(tmp_path, capsys):
    missing = tmp_path / "no_manifest.json"
    out = tmp_path / "run"
    assert main(["learn-rules", "--manifest", str(missing), "--output", str(out)]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "failed"
    assert not out.exists()


def test_manifest_rejects_unimplemented_options_before_loading_pickle(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"schema": "loris_rich_discovery_manifest_v1", "margin": 0.5}))
    with pytest.raises(ValueError, match="unknown or missing"):
        load_manifest(bad)
