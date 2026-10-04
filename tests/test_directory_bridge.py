import json
import os
from pathlib import Path

import pytest

from loris.labeling import external
from loris.labeling.directory_bridge import DirectoryBridge, write


def test_relative_root_is_bound_before_working_directory_changes(tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(tmp_path)
    bridge = DirectoryBridge(
        "bridge",
        lambda row: "document-%d" % row,
        timeout=1,
        sleep=lambda _: write(
            root / "ready/00000003.llm.json",
            {"row_id": "doc:3", "kind": "llm", "status": "answer_ready"},
        ),
    )
    monkeypatch.chdir(other)
    bridge.request(3, "doc:3", ["A", "B"], "llm")
    assert json.loads((root / "requests/00000003.llm.json").read_text()) == {
        "row": 3,
        "row_id": "doc:3",
        "labels": ["A", "B"],
        "text": "document-3",
    }


def test_bridge_rejects_identity_changes_and_service_errors(tmp_path):
    bridge = DirectoryBridge(tmp_path / "bridge", lambda row: "text")
    request = tmp_path / "bridge/requests/00000001.llm.json"
    request.parent.mkdir(parents=True)
    request.write_text(json.dumps({"row": 1, "row_id": "wrong", "labels": ["A"], "text": "text"}))
    with pytest.raises(ValueError):
        bridge.request(1, "doc:1", ["A"], "llm")


def test_install_requires_the_bound_answers_directory(tmp_path):
    bridge = DirectoryBridge(tmp_path / "bridge", lambda row: "text")
    original = external.external_answer
    bridge.install()
    try:
        with pytest.raises(ValueError):
            external.external_answer(1, "doc:1", ["A"], tmp_path / "other", lambda answer: True)
    finally:
        external.external_answer = original
