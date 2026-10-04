"""Directory transport for externally supplied labeling answers.

The bridge writes identity-bound requests and waits for a credential-free
service receipt. It does not make network calls, inspect API keys, or decide
whether an answer is logically consistent; ``external`` remains responsible
for that decision.
"""

import json
import math
import os
from pathlib import Path
import tempfile
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent),
        prefix="." + path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class DirectoryBridge:
    def __init__(self, root, text_for_row, timeout=900, sleep=time.sleep):
        if not callable(text_for_row) or not callable(sleep):
            raise ValueError("bridge callbacks must be callable")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("bridge timeout must be a finite nonnegative number")
        # Resolve once. A relative manifest directory must not change meaning
        # if the labeling process changes its working directory later.
        self.root = Path(root).expanduser().resolve()
        self.text_for_row = text_for_row
        self.timeout = timeout
        self.sleep = sleep

    def _path(self, *parts):
        path = (self.root.joinpath(*parts)).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError("bridge path escapes its root")
        return path

    def request(self, row, row_id, labels, kind):
        if type(row) is not int or row < 0 or not isinstance(row_id, str) or not row_id:
            raise ValueError("bridge request requires an explicit row and document ID")
        labels = list(labels)
        if not labels or len(set(labels)) != len(labels) or any(not isinstance(x, str) or not x for x in labels):
            raise ValueError("bridge request requires a unique nonempty label axis")
        if kind not in ("llm", "human"):
            raise ValueError("unsupported bridge request kind")
        stem = "%08d.%s" % (row, kind)
        request = self._path("requests", stem + ".json")
        ready = self._path("ready", stem + ".json")
        value = dict(row=row, row_id=row_id, labels=labels)
        if kind == "llm":
            text = self.text_for_row(row)
            if not isinstance(text, str):
                raise ValueError("bridge document text callback must return text")
            value["text"] = text
        if request.exists() or request.is_symlink():
            if request.is_symlink() or json.loads(request.read_text(encoding="utf-8")) != value:
                raise ValueError("existing bridge request has different document identity")
        else:
            write(request, value)
        started = time.monotonic()
        while not ready.is_file():
            if time.monotonic() - started > self.timeout:
                raise TimeoutError("external answer service did not return; labeling is pending")
            self.sleep(.5)
        status = json.loads(ready.read_text(encoding="utf-8"))
        if status.get("row_id") != row_id or status.get("kind") != kind:
            raise ValueError("external service receipt identity differs")
        if status.get("status") == "service_error":
            raise RuntimeError("external service error; see credential-free receipt")

    def install(self):
        from . import external

        original = external.external_answer
        expected = self._path("answers")

        def answer(row, row_id, labels, directory, validate_llm):
            if Path(directory).expanduser().resolve() != expected:
                raise ValueError("bridge answer directory differs")
            self.request(row, row_id, labels, "llm")
            value, receipt = original(row, row_id, labels, expected, validate_llm)
            if value is not None:
                return value, receipt
            self.request(row, row_id, labels, "human")
            return external._human_response(row, row_id, labels, expected, receipt)

        external.external_answer = answer
        return original
