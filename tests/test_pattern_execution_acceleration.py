"""Exercise actual parsing, screening, persisted masks and reuse boundaries."""
import json
import sqlite3

import numpy as np
import pytest

from loris.document import Document
from loris.discovery import patterns


def data():
    truth = np.array([[True, False], [False, True]] * 6)
    return dict(
        docs=[Document(cnt=("The red fox jumps over 12 logs." if v[0] else
                            "Blue birds fly across 34 rivers."), ttl="") for v in truth],
        fit_indices=np.arange(12), truth=truth, negative=~truth,
        labels=["red", "blue"], row_ids=[str(i) for i in range(12)],
        metadata=[{"group": str(i % 3)} for i in range(12)],
        cluster_assignments=np.repeat([0, 1], 6), input_binding={"fixture": "known_fit"},
        min_support=1, body_quota=3, title_quota=0, tfidf_terms=8,
    )


def database(path):
    with sqlite3.connect(str(path)) as db:
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {n: db.execute('SELECT * FROM "' + n + '" ORDER BY rowid').fetchall() for n in names}


def same_outputs(left, right):
    for name in ("body_raw.sqlite", "title_raw.sqlite", "candidates.sqlite"):
        assert database(left / name) == database(right / name)
    assert json.loads((left / "PATTERNS_BY_CLUSTER.json").read_text()) == json.loads(
        (right / "PATTERNS_BY_CLUSTER.json").read_text())
    for a in sorted(left.glob("cluster_*/FIT_MASKS.npz")):
        with np.load(a, allow_pickle=False) as x, np.load(right / a.relative_to(left), allow_pickle=False) as y:
            assert x.files == y.files
            for name in x.files:
                assert np.array_equal(x[name], y[name])


def test_actual_parallel_parse_and_reused_mcc_have_same_inventory(tmp_path, monkeypatch):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    fresh, reused = tmp_path / "mcc_fresh", tmp_path / "mcc_reused"
    patterns.build_supply(output=serial, parse_workers=1, **data())
    patterns.build_supply(output=parallel, parse_workers=2, **data())
    same_outputs(serial, parallel)
    patterns.build_supply(output=fresh, screening="mcc", **data())
    import spacy
    monkeypatch.setattr(spacy, "load", lambda *a, **k: pytest.fail("reuse must not load the parser"))
    patterns.build_supply(output=reused, screening="mcc", reuse_extraction=serial, **data())
    same_outputs(fresh, reused)
    assert json.loads((reused / "EXTRACTION_REUSE.json").read_text())["semantic_parse_calls"] == 0


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_parser_workers_do_not_start_extraction(tmp_path, workers):
    with pytest.raises(ValueError, match="parse_workers"):
        patterns.build_supply(output=tmp_path / "invalid", parse_workers=workers, **data())
    assert not (tmp_path / "invalid").exists()


def test_reuse_rejects_changed_text_settings_and_tampered_raw(tmp_path):
    source = tmp_path / "source"
    patterns.build_supply(output=source, **data())
    changed = data()
    changed["docs"][0] = Document(cnt="different text")
    with pytest.raises(ValueError, match="identity differs"):
        patterns.build_supply(output=tmp_path / "changed", reuse_extraction=source, **changed)
    changed = data()
    changed["tfidf_terms"] = 4
    with pytest.raises(ValueError, match="configuration differs"):
        patterns.build_supply(output=tmp_path / "settings", reuse_extraction=source, **changed)
    with (source / "additional_anchors.jsonl.gz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError):
        patterns.build_supply(output=tmp_path / "tampered", reuse_extraction=source, **data())
