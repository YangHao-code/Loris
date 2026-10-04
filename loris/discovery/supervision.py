"""Freeze complete known-train answers; readers never open sealed source truth.

The preparation step is the authorized answer provider: it opens the prepared
train truth container after freezing a label-blind row list. It exports only
those answers, with false observed masks for all other train cells. Runtime
readers authenticate these masked outputs, not the sealed source truth files.
"""

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

from .manifest import bound, file_record

SCHEMA = "loris_initial_supervision_v1"
DEFAULT_NUMERATOR = 1
DEFAULT_DENOMINATOR = 1


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_known_rows(
    row_ids,
    dataset,
    *,
    seed=42,
    fraction_numerator=DEFAULT_NUMERATOR,
    fraction_denominator=DEFAULT_DENOMINATOR,
):
    """Return source-order indices, with membership determined only by row IDs."""
    rows = list(row_ids)
    if (
        not rows
        or any(not isinstance(x, str) or not x for x in rows)
        or len(set(rows)) != len(rows)
    ):
        raise ValueError("nonempty unique train row IDs required")
    if not isinstance(dataset, str) or not dataset or "|" in dataset:
        raise ValueError("nonempty dataset name without hash delimiter required")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if (
        type(fraction_numerator) is not int
        or type(fraction_denominator) is not int
        or not 0 < fraction_numerator <= fraction_denominator
    ):
        raise ValueError("integer fraction must satisfy 0 < numerator <= denominator")
    count = (len(rows) * fraction_numerator + fraction_denominator - 1) // fraction_denominator
    if count == len(rows):
        return np.arange(len(rows), dtype=np.int64)

    def rank(row):
        message = "loris-formal-20260917|{}|known|{}|{}".format(dataset, seed, row)
        return hashlib.sha256(message.encode("utf-8")).hexdigest(), row

    ranked = sorted(range(len(rows)), key=lambda i: rank(rows[i]))
    return np.asarray(sorted(ranked[:count]), dtype=np.int64)


def _truth(path, expected_rows, labels):
    with np.load(path, allow_pickle=False) as source:
        if not {"row_ids", "targets", "observed"} <= set(source.files):
            raise ValueError("truth must have row_ids, targets and observed")
        if source["row_ids"].tolist() != expected_rows:
            raise ValueError("truth row axis differs from declared split")
        y, mask = source["targets"], source["observed"]
        if (
            y.dtype != bool
            or mask.dtype != bool
            or y.shape != (len(expected_rows), len(labels))
            or mask.shape != y.shape
            or not mask.all()
        ):
            raise ValueError("source answers must be complete boolean targets and masks")
        if "labels" in source.files and source["labels"].tolist() != labels:
            raise ValueError("truth label axis differs")
        return y.copy()


def _documents(path, expected_rows, metadata_fields):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if (
                not isinstance(item, dict)
                or set(item) != {"row_id", "text", "metadata", "source"}
                or not isinstance(item["text"], str)
                or not isinstance(item["metadata"], dict)
                or set(item["metadata"]) != set(metadata_fields)
            ):
                raise ValueError(
                    "prepared text/metadata contract differs; labels cannot enter documents"
                )
            records.append(item)
    if [row["row_id"] for row in records] != expected_rows:
        raise ValueError("document axis differs from declared split")
    return records


def prepare_supervision(
    prepared_manifest,
    output_dir,
    *,
    historical_exposure,
    seed=42,
    fraction_numerator=DEFAULT_NUMERATOR,
    fraction_denominator=DEFAULT_DENOMINATOR,
    contract_sources=(),
):
    """Prepare label-blind complete initial answers and isolated dev scoring.

    historical_exposure is an explicit JSON object describing prior model,
    checkpoint/calibration and test exposure. It is disclosure, not a grant to
    reuse historical model weights. No test document or truth file is opened.
    """
    start = time.monotonic()
    manifest_path = Path(prepared_manifest).resolve(strict=True)
    prepared = _read(manifest_path)
    if (
        prepared.get("schema") != "loris_dataset_prepared_v1"
        or prepared.get("status") != "complete"
    ):
        raise ValueError("complete prepared dataset manifest required")
    if not isinstance(historical_exposure, dict) or not historical_exposure:
        raise ValueError("explicit nonempty historical_exposure disclosure required")
    exposure = json.loads(json.dumps(historical_exposure, allow_nan=False))
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("new supervision namespace required: " + str(output))
    base = manifest_path.parent
    artifacts = prepared["artifacts"]
    # Explicit allow-list: never validate/hash all artifacts, because that would
    # open test truth and re-expose source targets unnecessarily.
    input_paths = {
        name: bound(artifacts[name], base)
        for name in ("AXES.json", "documents/train.jsonl", "documents/dev.jsonl")
    }
    axes = _read(input_paths["AXES.json"])
    labels = prepared["labels"]
    if (
        not labels
        or len(set(labels)) != len(labels)
        or axes["labels"] != labels
        or len(axes["row_ids"]) != len(axes["row_roles"])
    ):
        raise ValueError("prepared label/row axes differ")
    all_rows, roles = axes["row_ids"], axes["row_roles"]
    if len(set(all_rows)) != len(all_rows) or not set(roles) <= {"train", "dev", "test"}:
        raise ValueError("prepared row membership is invalid")
    split_rows = {
        s: [r for r, role in zip(all_rows, roles) if role == s] for s in ("train", "dev", "test")
    }
    if any(len(split_rows[s]) != prepared["counts"][s] for s in split_rows):
        raise ValueError("prepared split count differs")
    train_rows, dev_rows = split_rows["train"], split_rows["dev"]
    if not dev_rows:
        raise ValueError("declared dev scoring split required")
    fit = select_known_rows(
        train_rows,
        prepared["dataset"],
        seed=seed,
        fraction_numerator=fraction_numerator,
        fraction_denominator=fraction_denominator,
    )
    known = np.zeros(len(train_rows), dtype=bool)
    known[fit] = True
    train_docs = _documents(
        input_paths["documents/train.jsonl"], train_rows, prepared["metadata_fields"]
    )
    dev_docs = _documents(input_paths["documents/dev.jsonl"], dev_rows, prepared["metadata_fields"])
    contracts = [file_record(Path(p)) for p in contract_sources]
    # Freeze identity/membership before opening the answer container.
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("documents", "roots", "scoring"):
        (output / folder).mkdir()
    selection = {
        "schema": "loris_known_train_selection_v1",
        "dataset": prepared["dataset"],
        "seed": seed,
        "fraction_numerator": fraction_numerator,
        "fraction_denominator": fraction_denominator,
        "count_rule": "integer ceil(train_rows * numerator / denominator)",
        "hash": "SHA256 UTF-8; ascending hexadecimal, then row_id",
        "hash_preimage": "loris-formal-20260917|{dataset}|known|{seed}|{row_id}",
        "fit_indices": fit.tolist(),
        "fit_row_ids": [train_rows[i] for i in fit],
        "membership_uses_labels": False,
        "membership_uses_text_or_metadata": False,
        "source_order_retained": True,
        "prepared_manifest": file_record(manifest_path),
    }
    _write(output / "SELECTION.json", selection)
    source_records = {name: file_record(path) for name, path in input_paths.items()}
    try:
        for name in ("truth/train.npz", "truth/dev.npz"):
            input_paths[name] = bound(artifacts[name], base)
            source_records[name] = file_record(input_paths[name])
        # NPZ decoding materializes the source train target matrix transiently.
        # No unknown target is inspected, scored, selected on, or exported.
        complete_train = _truth(input_paths["truth/train.npz"], train_rows, labels)
        targets = np.zeros(complete_train.shape, dtype=bool)
        targets[fit] = complete_train[fit]
        del complete_train
        observed = np.repeat(known[:, None], len(labels), axis=1)
        dev_targets = _truth(input_paths["truth/dev.npz"], dev_rows, labels)
        np.savez_compressed(
            output / "roots/train_known_masked.npz",
            targets=targets,
            observed=observed,
            known_rows=known,
            row_ids=np.asarray(train_rows),
            labels=np.asarray(labels),
            pos=targets & observed,
            neg=(~targets) & observed,
        )
        np.savez_compressed(
            output / "scoring/dev_truth.npz",
            targets=dev_targets,
            observed=np.ones_like(dev_targets),
            row_ids=np.asarray(dev_rows),
            labels=np.asarray(labels),
        )
        row_ids = train_rows + dev_rows
        row_roles = ["source_fit" if value else "unlabeled_train" for value in known] + [
            "val_ctrl"
        ] * len(dev_rows)
        with (output / "documents/train_dev.jsonl").open("x", encoding="utf-8") as stream:
            for index, row in enumerate(train_docs + dev_docs):
                doc = dict(
                    row,
                    row_index=index,
                    split="train" if index < len(train_rows) else "dev",
                    role=row_roles[index],
                )
                stream.write(
                    json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
        _write(
            output / "AXIS.json",
            {
                "labels": labels,
                "row_ids": row_ids,
                "row_roles": row_roles,
                "fit_indices": fit.tolist(),
                "train_indices": list(range(len(train_rows))),
                "receiver_indices": list(range(len(train_rows), len(row_ids))),
                "metadata_fields": prepared["metadata_fields"],
            },
        )
        signed_cells = int(observed.sum())
        _write(
            output / "EXPOSURE.json",
            {
                "schema": "loris_initial_exposure_v1",
                "historical": exposure,
                "new_run": {
                    "known_complete_train_answers": len(fit),
                    "known_signed_cells": signed_cells,
                    "known_positive_cells": int(targets.sum()),
                    "known_negative_cells": signed_cells - int(targets.sum()),
                    "unknown_train_answers_in_training": 0,
                    "dev_labeled_rows_permitted_for_selection": len(dev_rows),
                    "test_training_scoring_or_query_access": False,
                    "total_supervision_not_equal_to_Gamma_claim": True,
                },
                "preparation_access": {
                    "full_train_truth_container_opened": True,
                    "source_train_targets_transiently_decoded": True,
                    "unknown_targets_used_for_membership_or_statistics": False,
                    "unknown_targets_exported": False,
                    "test_truth_or_documents_opened": False,
                    "known_list_frozen_before_answer_container_open": True,
                    "permission_enforcement": "explicit reader contract and separate files; not OS isolation",
                },
                "known_positive_support": targets[fit].sum(0).tolist(),
                "known_negative_support": ((~targets[fit]) & observed[fit]).sum(0).tolist(),
                "labels_without_known_positive": [
                    labels[j] for j in np.flatnonzero(~targets[fit].any(0))
                ],
            },
        )
        result = {
            "schema": SCHEMA,
            "status": "complete",
            "dataset": prepared["dataset"],
            "prepared_manifest": file_record(manifest_path),
            "labels": labels,
            "counts": {
                "train": len(train_rows),
                "dev": len(dev_rows),
                "discovery": len(row_ids),
                "known_train": len(fit),
                "unknown_train": len(train_rows) - len(fit),
                "excluded_test": len(split_rows["test"]),
                "labels": len(labels),
            },
            "sampling": {
                k: selection[k]
                for k in ("seed", "fraction_numerator", "fraction_denominator", "hash_preimage")
            },
            "contract_sources": contracts,
            "producer_source": file_record(__file__),
            "source_files_read": source_records,
            "artifacts": {
                name: file_record(output / name)
                for name in (
                    "SELECTION.json",
                    "AXIS.json",
                    "EXPOSURE.json",
                    "documents/train_dev.jsonl",
                    "roots/train_known_masked.npz",
                    "scoring/dev_truth.npz",
                )
            },
            "domains": {
                "train": "known complete answers only; unknown cells unobserved",
                "discovery": "all train+dev text; complete signed known-train Gamma only",
                "dev": "declared model/calibration/BO/version selection; no Gamma or gradient targets",
                "test": "no text, truth, relations, scoring, training or queries in this preparation",
            },
            "environment": {"python": sys.version, "numpy": np.__version__},
            "seconds": time.monotonic() - start,
            "training_calls": 0,
            "BO_calls": 0,
            "Chase_calls": 0,
        }
        _write(output / "SUPERVISION_MANIFEST.json", result)
        return result
    except Exception as error:
        _write(
            output / "FAILED.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "reason": str(error),
                "seconds": time.monotonic() - start,
            },
        )
        raise


def load_supervision(manifest, *, include_dev_targets=True):
    """Read authenticated masked outputs, never the source train/test truth.

    With include_dev_targets=False the dev scoring artifact is neither opened
    nor hashed. This variant still returns the complete text/role axis. For a
    fit-only text/label view, use load_fit_supervision instead.
    """
    path = Path(manifest).resolve(strict=True)
    value = _read(path)
    if value.get("schema") != SCHEMA or value.get("status") != "complete":
        raise ValueError("complete initial supervision manifest required")
    files = value["artifacts"]
    names = [
        "SELECTION.json",
        "AXIS.json",
        "EXPOSURE.json",
        "documents/train_dev.jsonl",
        "roots/train_known_masked.npz",
    ]
    if include_dev_targets:
        names.append("scoring/dev_truth.npz")
    paths = {name: bound(files[name], path.parent) for name in names}
    axis = _read(paths["AXIS.json"])
    labels, rows, roles = axis["labels"], axis["row_ids"], axis["row_roles"]
    counts = value["counts"]
    ntrain, ndev, k = counts["train"], counts["dev"], counts["labels"]
    if (
        labels != value["labels"]
        or len(labels) != k
        or len(set(labels)) != k
        or len(rows) != ntrain + ndev
        or len(set(rows)) != len(rows)
        or len(roles) != len(rows)
        or ntrain <= 0
        or ndev <= 0
    ):
        raise ValueError("supervision row/label axis is invalid")
    expected_fit = select_known_rows(
        rows[:ntrain],
        value["dataset"],
        seed=value["sampling"]["seed"],
        fraction_numerator=value["sampling"]["fraction_numerator"],
        fraction_denominator=value["sampling"]["fraction_denominator"],
    )
    fit = np.asarray(axis["fit_indices"], dtype=np.int64)
    train = np.arange(ntrain, dtype=np.int64)
    control = np.arange(ntrain, len(rows), dtype=np.int64)
    exclusion = None
    excluded = np.asarray([], dtype=np.int64)
    if 'dev_exclusion' in value:
        exclusion = _read(bound(value['dev_exclusion'], path.parent))
        entries = exclusion.get('excluded', [])
        excluded = np.asarray([entry['global_row'] for entry in entries], dtype=np.int64)
        if (
            exclusion.get('schema') != 'loris_train_duplicate_dev_exclusion_v1'
            or exclusion.get('dataset') != value['dataset']
            or len(set(excluded.tolist())) != len(excluded)
            or np.any(excluded < ntrain)
            or np.any(excluded >= len(rows))
            or [rows[index] for index in excluded] != [entry['row_id'] for entry in entries]
            or counts.get('excluded_dev') != len(excluded)
            or counts.get('scored_dev') != ndev - len(excluded)
        ):
            raise ValueError('Invalid explicit duplicate-dev exclusion axis')
        control = control[~np.isin(control, excluded)]
    expected_roles = ["unlabeled_train"] * ntrain + ["val_ctrl"] * ndev
    for index in excluded:
        expected_roles[int(index)] = 'excluded_dev'
    for index in expected_fit:
        expected_roles[int(index)] = "source_fit"
    selection = _read(paths["SELECTION.json"])
    if (
        not np.array_equal(fit, expected_fit)
        or roles != expected_roles
        or axis["train_indices"] != train.tolist()
        or axis["receiver_indices"] != control.tolist()
        or selection["fit_indices"] != fit.tolist()
        or selection["fit_row_ids"] != [rows[i] for i in fit]
        or len(fit) != counts["known_train"]
        or ntrain - len(fit) != counts["unknown_train"]
    ):
        raise ValueError("known selection/roles differ from frozen label-blind sampling")
    with np.load(paths["roots/train_known_masked.npz"], allow_pickle=False) as source:
        targets, observed, known = source["targets"], source["observed"], source["known_rows"]
        if (
            source["labels"].tolist() != labels
            or source["row_ids"].tolist() != rows[:ntrain]
            or targets.shape != (ntrain, k)
            or targets.dtype != bool
            or observed.shape != targets.shape
            or observed.dtype != bool
            or known.shape != (ntrain,)
            or known.dtype != bool
            or not np.array_equal(np.flatnonzero(known), fit)
            or not observed[known].all()
            or observed[~known].any()
            or targets[~known].any()
            or source["pos"].dtype != bool
            or source["neg"].dtype != bool
            or not np.array_equal(source["pos"], targets & observed)
            or not np.array_equal(source["neg"], (~targets) & observed)
        ):
            raise ValueError(
                "unknown targets must be masked; signed known answers must be complete"
            )
        fit_targets = targets[fit].copy()
        gp = np.zeros((len(rows), k), dtype=bool)
        gn = gp.copy()
        gp[:ntrain], gn[:ntrain] = source["pos"], source["neg"]
    docs = []
    with paths["documents/train_dev.jsonl"].open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            doc = json.loads(line)
            if (
                index >= len(rows)
                or set(doc)
                != {"row_id", "text", "metadata", "source", "row_index", "split", "role"}
                or doc["row_index"] != index
                or doc["row_id"] != rows[index]
                or doc["role"] != roles[index]
                or doc["split"] != ("train" if index < ntrain else "dev")
                or not isinstance(doc["text"], str)
                or not isinstance(doc["metadata"], dict)
                or set(doc["metadata"]) != set(axis["metadata_fields"])
            ):
                raise ValueError("document/role axis differs or carries forbidden target fields")
            docs.append(doc)
    if len(docs) != len(rows):
        raise ValueError("incomplete discovery text axis")
    if exclusion is not None:
        def text_hash(text):
            normalized = ' '.join(unicodedata.normalize('NFKC', text).split())
            return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

        train_hashes = {}
        for index in train:
            train_hashes.setdefault(text_hash(docs[index]['text']), []).append(rows[index])
        actual = []
        for index in range(ntrain, len(rows)):
            digest = text_hash(docs[index]['text'])
            if digest in train_hashes:
                actual.append(dict(global_row=index, row_id=rows[index],
                    normalized_text_sha256=digest, train_row_ids=train_hashes[digest]))
        if actual != exclusion['excluded']:
            raise ValueError('Duplicate-dev exclusion does not exactly match all train text groups')
    dev_targets = (
        _truth(paths["scoring/dev_truth.npz"], [rows[index] for index in control], labels)
        if include_dev_targets
        else None
    )
    identity = file_record(path)
    return {
        "docs": docs,
        "labels": labels,
        "row_ids": rows,
        "row_roles": roles,
        "fit_indices": fit,
        "receiver_indices": control,
        "excluded_dev_indices": excluded,
        "train_indices": train,
        "fit_targets": fit_targets,
        "fit_truth": fit_targets,
        "fit_row_ids": [rows[i] for i in fit],
        "dev_targets": dev_targets,
        "gamma_pos": gp,
        "gamma_neg": gn,
        "supervision_identity": identity,
        "manifest_record": identity,
        "manifest": value,
        "files_read": {name: files[name] for name in names},
    }


def load_fit_supervision(manifest):
    """Return only fit text/labels; dev scoring is not opened or hashed."""
    data = load_supervision(manifest, include_dev_targets=False)
    return {
        "docs": [data["docs"][i] for i in data["fit_indices"]],
        "fit_indices": data["fit_indices"],
        "labels": data["labels"],
        "fit_targets": data["fit_targets"],
        "fit_truth": data["fit_targets"],
        "fit_row_ids": data["fit_row_ids"],
        "row_ids": data["fit_row_ids"],
        "metadata": [data["docs"][i]["metadata"] for i in data["fit_indices"]],
        "supervision_identity": data["supervision_identity"],
        "manifest_record": data["manifest_record"],
        "manifest": data["manifest"],
        "files_read": data["files_read"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--historical-exposure",
        required=True,
        help="explicit prior-exposure JSON, not a training permission",
    )
    parser.add_argument("--contract-source", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction-numerator", type=int, default=DEFAULT_NUMERATOR)
    parser.add_argument("--fraction-denominator", type=int, default=DEFAULT_DENOMINATOR)
    args = parser.parse_args(argv)
    result = prepare_supervision(
        args.prepared_manifest,
        args.output,
        historical_exposure=_read(args.historical_exposure),
        contract_sources=args.contract_source,
        seed=args.seed,
        fraction_numerator=args.fraction_numerator,
        fraction_denominator=args.fraction_denominator,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "manifest": file_record(Path(args.output) / "SUPERVISION_MANIFEST.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
