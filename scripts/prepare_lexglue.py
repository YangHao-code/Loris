#!/usr/bin/env python3
"""Fetch verified official LexGLUE tasks only with --download; prepare is optional.

Default execution prints a plan without networking or changing files. Raw split
files preserve all official rows/labels. Preparation creates separate text and
truth stores, no Gamma, no training, no label-dependent split selection.
"""
import argparse
import csv
import hashlib
import json
import numbers
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen

SPECS = {
    "ecthr_a": {
        "dataset": "lexglue_ecthr_a",
        "revision": "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c",
        "labels": ["2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1"],
        "counts": {"train": 9000, "test": 1000, "validation": 1000},
        "files": [
            {
                "path": "ecthr_a/test-00000-of-00001.parquet",
                "bytes": 5677114,
                "sha256_from_author_lfs_pointer": "e306383d3ffd3baec21ba7419c03a9d2273f1c0946cf2d5512cf98c81d1a3f8b",
                "content_hash_locally_verified": False,
                "url": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/c23fdff1a6bf74e0e1a71cb86f1e781d37da888c/ecthr_a/test-00000-of-00001.parquet",
            },
            {
                "path": "ecthr_a/train-00000-of-00001.parquet",
                "bytes": 42415324,
                "sha256_from_author_lfs_pointer": "2f263758eb80ec1c532e1c4bde6cbfb2a8ff41378ebec5999aca23929c39cdf4",
                "content_hash_locally_verified": False,
                "url": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/c23fdff1a6bf74e0e1a71cb86f1e781d37da888c/ecthr_a/train-00000-of-00001.parquet",
            },
            {
                "path": "ecthr_a/validation-00000-of-00001.parquet",
                "bytes": 5260148,
                "sha256_from_author_lfs_pointer": "d74de774cf566f7eea52563b54dec5721de6b7f5a34438c698aa48d7ff9de205",
                "content_hash_locally_verified": False,
                "url": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/c23fdff1a6bf74e0e1a71cb86f1e781d37da888c/ecthr_a/validation-00000-of-00001.parquet",
            },
        ],
        "license": "cc-by-4.0 (author repository metadata; licensing prose incomplete)",
        "scope": "complete published LexGLUE task, not all underlying ontology labels",
        "total_bytes": 53352586,
    },
    "eurlex": {
        "dataset": "lexglue_eurlex",
        "revision": "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c",
        "labels": [
            "100163",
            "100168",
            "100169",
            "100170",
            "100171",
            "100172",
            "100173",
            "100174",
            "100175",
            "100176",
            "100177",
            "100179",
            "100180",
            "100183",
            "100184",
            "100185",
            "100186",
            "100187",
            "100189",
            "100190",
            "100191",
            "100192",
            "100193",
            "100194",
            "100195",
            "100196",
            "100197",
            "100198",
            "100199",
            "100200",
            "100201",
            "100202",
            "100204",
            "100205",
            "100206",
            "100207",
            "100212",
            "100214",
            "100215",
            "100220",
            "100221",
            "100222",
            "100223",
            "100224",
            "100226",
            "100227",
            "100229",
            "100230",
            "100231",
            "100232",
            "100233",
            "100234",
            "100235",
            "100237",
            "100238",
            "100239",
            "100240",
            "100241",
            "100242",
            "100243",
            "100244",
            "100245",
            "100246",
            "100247",
            "100248",
            "100249",
            "100250",
            "100252",
            "100253",
            "100254",
            "100255",
            "100256",
            "100257",
            "100258",
            "100259",
            "100260",
            "100261",
            "100262",
            "100263",
            "100264",
            "100265",
            "100266",
            "100268",
            "100269",
            "100270",
            "100271",
            "100272",
            "100273",
            "100274",
            "100275",
            "100276",
            "100277",
            "100278",
            "100279",
            "100280",
            "100281",
            "100282",
            "100283",
            "100284",
            "100285",
        ],
        "counts": {"train": 55000, "test": 5000, "validation": 5000},
        "files": [
            {
                "path": "eurlex/test-00000-of-00001.parquet",
                "bytes": 24273776,
                "sha256_from_author_lfs_pointer": "0c2ce5e5bc26c907607613aa1d7e4209553dbc3b85ec6ddcff1c745e2c7fdb9d",
                "content_hash_locally_verified": False,
                "url": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/c23fdff1a6bf74e0e1a71cb86f1e781d37da888c/eurlex/test-00000-of-00001.parquet",
            },
            {
                "path": "eurlex/train-00000-of-00001.parquet",
                "bytes": 166694290,
                "sha256_from_author_lfs_pointer": "c0370abfe067fadec918490a01ba43ffc5bfdc6155283baffa5c120cf7754d3a",
                "content_hash_locally_verified": False,
                "url": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/c23fdff1a6bf74e0e1a71cb86f1e781d37da888c/eurlex/train-00000-of-00001.parquet",
            },
            {
                "path": "eurlex/validation-00000-of-00001.parquet",
                "bytes": 17059983,
                "sha256_from_author_lfs_pointer": "b7f15ce05817231e33b5e0d63f3c07972a0490e1caa2bbfed4d6316204c893c8",
                "content_hash_locally_verified": False,
                "url": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/c23fdff1a6bf74e0e1a71cb86f1e781d37da888c/eurlex/validation-00000-of-00001.parquet",
            },
        ],
        "license": "cc-by-4.0 (author repository metadata; licensing prose incomplete)",
        "scope": "complete published LexGLUE task, not all underlying ontology labels",
        "total_bytes": 208028049,
    },
}


def file_record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def split_name(remote_path):
    name = Path(remote_path).name.split("-")[0]
    return "dev" if name == "validation" else name


def local_file(raw, task, remote):
    role = split_name(remote["path"])
    purpose = {"train": "train_query_oracle", "dev": "dev_evaluator", "test": "test_evaluator"}[
        role
    ]
    return Path(raw) / task / purpose / Path(remote["path"]).name


def verified_file(raw, task, remote):
    path = local_file(raw, task, remote)
    rec = file_record(path)
    if (
        path.stat().st_size != remote["bytes"]
        or rec["sha256"] != remote["sha256_from_author_lfs_pointer"]
    ):
        raise ValueError("Official byte identity mismatch: " + str(path))
    return rec


def download_file(raw, task, remote):
    path = local_file(raw, task, remote)
    if path.exists():
        return verified_file(raw, task, remote)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    digest, count = hashlib.sha256(), 0
    # No implicit partial resume or overwrite of old files.
    with partial.open("xb") as stream:
        os.chmod(partial, 0o600)
        with urlopen(remote["url"], timeout=60) as response:
            if response.status != 200:
                raise ValueError("Full response required, got " + str(response.status))
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > remote["bytes"]:
                    raise ValueError("Remote response exceeds frozen byte size")
                digest.update(block)
                stream.write(block)
    if count != remote["bytes"] or digest.hexdigest() != remote["sha256_from_author_lfs_pointer"]:
        raise ValueError("Downloaded identity mismatch; retained .partial for diagnosis")
    os.link(partial, path)  # atomic no-overwrite publication
    partial.unlink()
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def text_and_binary_labels(row, task, labels):
    """Use true input paragraphs in their released order; never fabricate title."""
    text, ids = row["text"], row["labels"]
    if task == "ecthr_a":
        if not isinstance(text, list) or any(not isinstance(x, str) for x in text):
            raise ValueError("ECtHR text must be an ordered paragraph list")
        text = "\n\n".join(text)
    elif not isinstance(text, str):
        raise ValueError("EUR-LEX text must be a string")
    if (
        not isinstance(ids, list)
        or len(set(ids)) != len(ids)
        or any(
            isinstance(x, bool)
            or not isinstance(x, numbers.Integral)
            or not 0 <= int(x) < len(labels)
            for x in ids
        )
    ):
        raise ValueError("Invalid or duplicated official label index")
    present = set(int(x) for x in ids)
    return text, [int(i in present) for i in range(len(labels))]


def prepare(raw, output, task):
    import pyarrow.parquet as pq

    csv.field_size_limit(sys.maxsize)
    from loris.discovery.datasets import prepare_dataset

    spec = SPECS[task]
    output = Path(output).resolve() / task
    if output.exists():
        raise FileExistsError("New prepared namespace required: " + str(output))
    refs = {split_name(x["path"]): verified_file(raw, task, x) for x in spec["files"]}
    output.mkdir(parents=True)
    source = output / "source_truth_csv"
    source.mkdir(mode=0o700)
    sources = {}
    for role in ("train", "dev", "test"):
        path = source / (role + ".csv")
        rows = 0
        with path.open("x", encoding="utf-8", newline="") as stream:
            os.chmod(path, 0o600)
            writer = csv.writer(stream)
            writer.writerow(["source_row_id", "text"] + spec["labels"])
            parquet = pq.ParquetFile(refs[role]["path"])
            if set(parquet.schema_arrow.names) != {"text", "labels"}:
                raise ValueError("Unexpected fields in frozen official Parquet")
            for batch in parquet.iter_batches(batch_size=256, columns=["text", "labels"]):
                for row in batch.to_pylist():
                    text, target = text_and_binary_labels(row, task, spec["labels"])
                    writer.writerow([role + ":" + str(rows), text] + target)
                    rows += 1
        official_role = "validation" if role == "dev" else role
        if rows != spec["counts"][official_role]:
            raise ValueError("Frozen official split count mismatch: " + role)
        sources[role] = file_record(path)
    manifest = {
        "schema": "loris_dataset_source_v1",
        "dataset": spec["dataset"],
        "version": spec["revision"],
        "sources": sources,
        "columns": {
            "text": ["text"],
            "id": "source_row_id",
            "labels": spec["labels"],
            "metadata": [],
        },
        "label_axis": {
            "mode": "declared",
            "labels": spec["labels"],
            "provenance": "Fixed author LexGLUE ClassLabel axis; not inferred from test positives",
        },
        "split": {
            "seed": 42,
            "fractions": [0.1, 0.1, 0.8],
            "dev_fraction": 0.1,
            "group_column": None,
        },
        "provenance": {
            "official_dataset": "https://huggingface.co/datasets/coastalcph/lex_glue",
            "revision": spec["revision"],
            "raw_files": refs,
            "license": spec["license"],
            "scope": spec["scope"],
            "rows_dropped": 0,
            "metadata_fabricated": False,
            "text_transform": "ECtHR ordered paragraphs joined by two newlines; EURLEX unchanged",
            "test_truth_access": "import/query-oracle preparation only; never fitting/selection",
            "training_permission": "No answers acquired, no Gamma created by this preparation",
            "converter_source": file_record(__file__),
        },
    }
    write_json(output / "SOURCE_MANIFEST.json", manifest)
    result = prepare_dataset(output / "SOURCE_MANIFEST.json", output / "prepared")
    write_json(
        output / "PREPARED_RECEIPT.json",
        {
            "status": "complete",
            "task": task,
            "counts": result["counts"],
            "labels": result["labels"],
            "Gamma_created": False,
            "training_calls": 0,
            "BO_calls": 0,
            "test_truth_used_for_selection_or_training": False,
            "manifest": file_record(output / "prepared" / "MANIFEST.json"),
        },
    )
    return str(output / "prepared" / "MANIFEST.json")


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", choices=("ecthr_a", "eurlex", "both"), default="both")
    cli.add_argument("--raw-dir", type=Path)
    cli.add_argument(
        "--download", action="store_true", help="Explicitly fetch complete official split files"
    )
    cli.add_argument(
        "--prepare-dir", type=Path, help="Offline verified conversion into a new namespace"
    )
    args = cli.parse_args(argv)
    tasks = list(SPECS) if args.task == "both" else [args.task]
    if (args.download or args.prepare_dir) and args.raw_dir is None:
        cli.error("--raw-dir is required for download or preparation")
    if not args.download and args.prepare_dir is None:
        print(
            json.dumps(
                {"status": "plan_only_no_network", "tasks": {k: SPECS[k] for k in tasks}}, indent=2
            )
        )
        return 0
    started = time.monotonic()
    for task in tasks:
        if args.download:
            initially_missing = [
                x for x in SPECS[task]["files"] if not local_file(args.raw_dir, task, x).exists()
            ]
            download_started = time.monotonic()
            refs = {
                split_name(x["path"]): download_file(args.raw_dir, task, x)
                for x in SPECS[task]["files"]
            }
            receipt = args.raw_dir / task / "RESOURCE_MANIFEST.json"
            record = {
                "schema": "loris_official_lexglue_resource_v1",
                "status": "complete",
                "task": task,
                "specification": SPECS[task],
                "local_files": refs,
                "verified_corpus_bytes": sum(x["bytes"] for x in SPECS[task]["files"]),
                "downloaded_bytes_this_creation": sum(x["bytes"] for x in initially_missing),
                "download_and_hash_seconds": time.monotonic() - download_started,
                "labels_inspected": False,
                "training_calls": 0,
                "Gamma_created": False,
                "test_policy": "Held separately as test_evaluator; no discovery/training permission",
                "separation_enforcement": "Directory and loader contract; not an OS-user security boundary",
                "script_source": file_record(__file__),
            }
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old["specification"] != record["specification"] or old["local_files"] != refs:
                    raise ValueError("Existing immutable resource receipt differs")
            else:
                write_json(receipt, record)
            print(
                json.dumps(
                    {"task": task, "download": "verified", "manifest": str(receipt.resolve())}
                ),
                flush=True,
            )
        if args.prepare_dir:
            print(
                json.dumps(
                    {
                        "task": task,
                        "prepared_manifest": prepare(args.raw_dir, args.prepare_dir, task),
                    }
                ),
                flush=True,
            )
    print(json.dumps({"status": "complete", "seconds": time.monotonic() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
