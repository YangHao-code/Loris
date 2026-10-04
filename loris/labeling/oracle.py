"""Privileged simulated oracle process: disclose exactly one requested row.

The child may decode the source NPZ internally, as an oracle holding the answer
bank. The learner receives only one complete answer and its source identity.
This is simulation, not a claim that the provider never accessed other bytes.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from loris.discovery.manifest import file_record, bound


def prepare(
    *,
    context_path,
    truth_path,
    truth_key,
    axis_path,
    axis_key,
    roles,
    source_rows,
    source_offset,
    output,
):
    context_path = Path(context_path).resolve(strict=True)
    with context_path.open("rb") as f:
        ctx = pickle.load(f)
    labels = json.loads(Path(axis_path).read_text())[axis_key]
    if labels != list(ctx["labels"]):
        raise ValueError("oracle source and context label axes differ")
    if (
        type(source_rows) is not int
        or source_rows < 1
        or type(source_offset) is not int
        or source_offset < 0
    ):
        raise ValueError("explicit source extent required")
    raw = np.asarray(ctx["raw_indices"])
    if raw.shape != (len(ctx["docs"]),):
        raise ValueError("context lacks complete declared raw indices")
    rows = {}
    for i, role in enumerate(ctx["row_roles"]):
        if role not in roles:
            continue
        source = int(raw[i]) - source_offset
        if not 0 <= source < source_rows:
            raise ValueError("eligible raw source index outside declared official split")
        rows[str(i)] = dict(row_id=ctx["row_ids"][i], source_row=source, role=role)
    if len({v["source_row"] for v in rows.values()}) != len(rows):
        raise ValueError("oracle source mapping must be one-to-one")
    # Hash source bytes for identity, do NOT decode any truth target here.
    manifest = dict(
        schema="loris_isolated_npz_simulated_human_v1",
        context=file_record(context_path),
        truth=file_record(truth_path),
        truth_key=truth_key,
        source_shape=[source_rows, len(labels)],
        label_axis=file_record(axis_path),
        label_axis_key=axis_key,
        labels=labels,
        rows=rows,
        roles=roles,
        source_offset=source_offset,
        raw_index_policy="context.raw_indices minus explicit official split offset",
        truth_decoded_during_prepare=False,
        queries_during_prepare=0,
        provider="development_truth_simulated_human",
        learner_access="one answer via separate process stdout after charged request",
        exposure="Oracle child owns full answer bank; only the selected row is returned. Not an operating-system sandbox.",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return dict(
        manifest=file_record(output),
        eligible_provider_rows=len(rows),
        truth_decoded=False,
        queries=0,
    )


def answer(manifest_path, request):
    manifest_path = Path(manifest_path).resolve(strict=True)
    data = json.loads(manifest_path.read_text())
    if data["schema"] != "loris_isolated_npz_simulated_human_v1":
        raise ValueError("wrong oracle schema")
    if set(request) != {"row", "row_id", "labels"} or type(request["row"]) is not int:
        raise ValueError("one explicit oracle request required")
    entry = data["rows"].get(str(request["row"]))
    if entry is None or entry["row_id"] != request["row_id"] or request["labels"] != data["labels"]:
        raise ValueError("row not permitted or row/label identity mismatch")
    axis = bound(data["label_axis"], manifest_path.parent)
    if json.loads(axis.read_text())[data["label_axis_key"]] != data["labels"]:
        raise ValueError("source label axis changed")
    truth_path = bound(data["truth"], manifest_path.parent)
    with np.load(truth_path, allow_pickle=False) as z:
        y = z[data["truth_key"]]
        if list(y.shape) != data["source_shape"]:
            raise ValueError("truth source shape differs")
        selected = np.array(y[entry["source_row"]], copy=True)
    if not np.isin(selected, (0, 1)).all():
        raise ValueError("complete binary answer required")
    return dict(
        row_id=request["row_id"],
        labels=data["labels"],
        targets=selected.astype(int).tolist(),
        provider="development_truth_simulated_human_isolated_process",
        source=data["truth"],
        source_row=entry["source_row"],
        disclosed_answer_rows=1,
        unqueried_answers_returned=0,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    a = p.parse_args(argv)
    request = json.load(sys.stdin)
    print(json.dumps(answer(a.manifest, request), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
