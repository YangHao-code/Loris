"""Source-bound dataset import; this module never trains or creates trusted roots.

Official split membership takes priority. Only datasets without a released split
use the declared hash split. Truth files are evaluator/query stores, not a grant
of training permission: a later supervision manifest must select known answers.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .manifest import file_record


SOURCE_SCHEMA = "loris_dataset_source_v1"
OUTPUT_SCHEMA = "loris_dataset_prepared_v1"
SPLITS = ("train", "dev", "test")


def _json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _units(row_ids, group_ids):
    groups = {} if group_ids is None else dict(group_ids)
    if set(groups) - set(row_ids):
        raise ValueError("group_ids contains rows outside the document axis")
    result = {}
    for row in row_ids:
        group = groups.get(row)
        if group is not None and not isinstance(group, str):
            raise ValueError("group IDs must be strings or null")
        # Missing metadata cannot connect unrelated documents.
        key = ("group", group) if group is not None and group.strip() else ("row", row)
        result.setdefault(key, []).append(row)
    return result


def _ranked(units, seed, domain):
    return sorted(
        units,
        key=lambda key: (
            hashlib.sha256(_json_bytes(["loris_dataset_split_v1", domain, seed, key])).hexdigest(),
            key,
        ),
    )


def assign_splits(
    row_ids,
    *,
    official_splits=None,
    group_ids=None,
    seed=42,
    fractions=(0.1, 0.1, 0.8),
    dev_fraction=0.1,
):
    """Return row-ID membership without examining labels or document contents.

    With official train/test, only the absent dev is carved from official train.
    Otherwise exact unit counts use floor(train*N), floor(dev*N), with at least
    one unit per split. A unit is one row or one supplied relation group. Group
    fractions are by groups; actual document proportions are reported separately.
    Official groups spanning splits are reported rather than moved or dropped.
    """
    row_ids = list(row_ids)
    row_set = set(row_ids)
    if (
        not row_ids
        or any(not isinstance(x, str) or not x for x in row_ids)
        or len(set(row_ids)) != len(row_ids)
    ):
        raise ValueError("a nonempty unique string row axis is required")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    units = _units(row_ids, group_ids)
    if official_splits is not None:
        if not isinstance(official_splits, dict) or not {"train", "test"} <= set(official_splits):
            raise ValueError("official splits require train and test")
        if set(official_splits) - set(SPLITS):
            raise ValueError("unknown official split")
        membership = {}
        for split, values in official_splits.items():
            if not values:
                raise ValueError("official split must not be empty: " + split)
            for row in values:
                if row not in row_set or row in membership:
                    raise ValueError("official row is missing or duplicated")
                membership[row] = split
        if set(membership) != set(row_ids):
            raise ValueError("official split membership must exhaust the row axis")
        policy = "official_membership_preserved"
        if "dev" not in official_splits:
            if (
                isinstance(dev_fraction, bool)
                or not isinstance(dev_fraction, (int, float))
                or not math.isfinite(dev_fraction)
                or not 0 < dev_fraction < 1
            ):
                raise ValueError("dev_fraction must be strictly between zero and one")
            train_units = {
                key: [row for row in values if membership[row] == "train"]
                for key, values in units.items()
            }
            train_units = {key: values for key, values in train_units.items() if values}
            if len(train_units) < 2:
                raise ValueError("at least two official train units needed to create dev")
            count = min(len(train_units) - 1, max(1, math.ceil(len(train_units) * dev_fraction)))
            for key in _ranked(train_units, seed, "official_train_dev")[:count]:
                for row in train_units[key]:
                    membership[row] = "dev"
            policy = "official_test_preserved_dev_from_official_train"
    else:
        if (
            len(fractions) != 3
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
                for v in fractions
            )
            or not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("three positive fractions summing to one required")
        if len(units) < 3:
            raise ValueError("at least three independent units required for train/dev/test")
        ranked = _ranked(units, seed, "custom_train_dev_test")
        train_count = max(1, math.floor(len(ranked) * fractions[0]))
        dev_count = max(1, math.floor(len(ranked) * fractions[1]))
        if train_count + dev_count >= len(ranked):
            raise ValueError("fractions leave no test unit")
        membership = {}
        for i, key in enumerate(ranked):
            split = "train" if i < train_count else "dev" if i < train_count + dev_count else "test"
            for row in units[key]:
                membership[row] = split
        policy = "custom_hash_split"
    # Output order follows immutable source order, not label-dependent sorting.
    result = {split: [row for row in row_ids if membership[row] == split] for split in SPLITS}
    group_overlaps = [
        {
            "group_id": key[1],
            "rows_by_split": {
                s: [row for row in values if membership[row] == s]
                for s in SPLITS
                if any(membership[row] == s for row in values)
            },
        }
        for key, values in units.items()
        if len({membership[row] for row in values}) > 1
    ]
    return {
        "membership": result,
        "policy": policy,
        "seed": seed,
        "fraction_unit": "relation_group" if group_ids is not None else "document",
        "requested_fractions_without_official_split": list(fractions),
        "dev_fraction_if_official_dev_absent": dev_fraction,
        "counts": {s: len(result[s]) for s in SPLITS},
        "group_counts": {
            s: sum(any(membership[row] == s for row in v) for v in units.values()) for s in SPLITS
        },
        "group_overlap_count": len(group_overlaps),
        "group_overlaps": group_overlaps,
        "official_membership_never_changed_to_isolate_groups": True,
        "actual_document_fractions": {s: len(result[s]) / len(row_ids) for s in SPLITS},
        "labels_used_for_split": False,
        "missing_groups_are_singletons": True,
    }


def _names(values, name, nonempty=True):
    if (
        not isinstance(values, list)
        or (nonempty and not values)
        or any(not isinstance(x, str) or not x for x in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(name + " must be an ordered list of unique nonempty names")
    return values


def prepare_dataset(source_manifest, output_dir):
    """Import a declared wide binary CSV dataset into separate text/truth stores.

    The source manifest can be a JSON filename or a dictionary. Its sources are
    SHA-bound train[/dev]/test files or one 'all' file. label_axis.mode is either
    'declared' (retain the supplied full axis) or 'train_supported' (infer only
    from positives in the final train split). Unknown cells are not accepted as
    complete source answers. Missing text is retained and counted, not removed.
    """
    start = time.monotonic()
    if isinstance(source_manifest, (str, Path)):
        manifest_path = Path(source_manifest).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent
        source_identity = file_record(manifest_path)
    else:
        manifest = json.loads(json.dumps(source_manifest, allow_nan=False))
        base = Path.cwd()
        source_identity = {
            "inline_manifest_sha256": hashlib.sha256(_json_bytes(manifest)).hexdigest()
        }
    required = {
        "schema",
        "dataset",
        "version",
        "sources",
        "columns",
        "label_axis",
        "split",
        "provenance",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != required
        or manifest["schema"] != SOURCE_SCHEMA
    ):
        raise ValueError("explicit dataset source manifest fields required")
    for key in ("dataset", "version"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ValueError("dataset and version must be nonempty names")
    sources = manifest["sources"]
    if not isinstance(sources, dict) or (
        set(sources) != {"all"}
        and (not {"train", "test"} <= set(sources) or set(sources) - set(SPLITS))
    ):
        raise ValueError("sources must be all, train/test, or train/dev/test")
    columns = manifest["columns"]
    if not isinstance(columns, dict) or set(columns) != {"text", "id", "labels", "metadata"}:
        raise ValueError("text, id, labels and metadata columns must be explicit")
    text_columns = _names(columns["text"], "text columns")
    label_columns = _names(columns["labels"], "label columns")
    meta_columns = _names(columns["metadata"], "metadata columns", nonempty=False)
    id_column = columns["id"]
    if id_column is not None and (not isinstance(id_column, str) or not id_column):
        raise ValueError("id column must be a name or null")
    if set(label_columns) & (
        set(text_columns) | set(meta_columns) | ({id_column} if id_column else set())
    ):
        raise ValueError("label fields cannot be text, identity or metadata inputs")
    axis_spec = manifest["label_axis"]
    if not isinstance(axis_spec, dict) or set(axis_spec) != {"mode", "labels", "provenance"}:
        raise ValueError("label axis must declare mode, labels and provenance")
    if axis_spec["mode"] == "declared":
        _names(axis_spec["labels"], "declared axis")
        if not set(axis_spec["labels"]) <= set(label_columns):
            raise ValueError("declared label axis not present in source columns")
    elif axis_spec["mode"] != "train_supported" or axis_spec["labels"] is not None:
        raise ValueError("train_supported must not supply labels; unknown axis mode")
    split_spec = manifest["split"]
    if not isinstance(split_spec, dict) or set(split_spec) != {
        "seed",
        "fractions",
        "dev_fraction",
        "group_column",
    }:
        raise ValueError("split settings must be fully explicit")
    group_column = split_spec["group_column"]
    if group_column is not None and group_column not in meta_columns:
        raise ValueError("group_column must be a declared non-label metadata column")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("new output namespace required: " + str(output))
    rows, targets, official, records, groups = [], [], {}, {}, {}
    needed = (
        set(text_columns)
        | set(label_columns)
        | set(meta_columns)
        | ({id_column} if id_column else set())
    )
    for role in SPLITS if "all" not in sources else ("all",):
        if role not in sources:
            continue
        reference = sources[role]
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("source references require exactly path and sha256")
        path = (base / reference["path"]).resolve(strict=True)
        record = file_record(path)
        if record["sha256"] != reference["sha256"]:
            raise ValueError("source SHA mismatch: " + str(path))
        records[role] = record
        official[role] = []
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if (
                reader.fieldnames is None
                or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not needed <= set(reader.fieldnames)
            ):
                raise ValueError("duplicate/missing CSV columns: " + str(path))
            for number, value in enumerate(reader):
                if None in value or any(value.get(key) is None for key in needed):
                    raise ValueError("malformed CSV row")
                local_id = value[id_column] if id_column else role + ":" + str(number)
                if not local_id or not local_id.strip():
                    raise ValueError("empty source document ID")
                row_id = manifest["dataset"] + ":" + local_id
                y = []
                for label in label_columns:
                    cell = value[label].strip()
                    if cell not in {"0", "1"}:
                        raise ValueError(
                            "source labels must be complete binary 0/1; unknown is not negative"
                        )
                    y.append(cell == "1")
                text = "\n".join(value[key].strip() for key in text_columns if value[key].strip())
                rows.append(
                    {
                        "row_id": row_id,
                        "text": text,
                        "metadata": {key: value[key] for key in meta_columns},
                        "source": {"split_file": role, "csv_row": number},
                    }
                )
                targets.append(y)
                official[role].append(row_id)
                if group_column is not None:
                    groups[row_id] = value[group_column]
    row_ids = [x["row_id"] for x in rows]
    partition = assign_splits(
        row_ids,
        official_splits=None if "all" in sources else official,
        group_ids=groups if group_column is not None else None,
        seed=split_spec["seed"],
        fractions=split_spec["fractions"],
        dev_fraction=split_spec["dev_fraction"],
    )
    roles = {row: role for role, values in partition["membership"].items() for row in values}
    full_y = np.asarray(targets, dtype=bool)
    train_idx = np.asarray(
        [i for i, row in enumerate(row_ids) if roles[row] == "train"], dtype=np.int64
    )
    support = full_y[train_idx].sum(axis=0)
    axis = (
        axis_spec["labels"]
        if axis_spec["mode"] == "declared"
        else [label for label, count in zip(label_columns, support) if count > 0]
    )
    if not axis:
        raise ValueError("no supported label exists in train; cannot define an empty task")
    selected = [label_columns.index(label) for label in axis]
    y = full_y[:, selected]
    unsupported = [label for label, count in zip(label_columns, support) if count == 0]
    support_report = {
        "train_positive_support": {label: int(support[i]) for i, label in enumerate(label_columns)},
        "unsupported_in_train": unsupported,
        "unsupported_axis_labels": [label for label in axis if label in unsupported],
        "excluded_candidate_labels": [label for label in label_columns if label not in axis],
        "dev_test_used_to_choose_labels": False,
    }
    output.mkdir(parents=True)
    try:
        for name in ("documents", "truth"):
            (output / name).mkdir()
        for role in SPLITS:
            indices = np.asarray(
                [i for i, row in enumerate(row_ids) if roles[row] == role], dtype=np.int64
            )
            with (output / "documents" / (role + ".jsonl")).open("w", encoding="utf-8") as stream:
                for i in indices:
                    stream.write(json.dumps(rows[int(i)], ensure_ascii=False) + "\n")
            np.savez_compressed(
                output / "truth" / (role + ".npz"),
                row_ids=np.asarray([row_ids[int(i)] for i in indices], dtype=str),
                targets=y[indices],
                observed=np.ones((len(indices), len(axis)), dtype=bool),
            )
        _write_json(output / "SOURCE_MANIFEST.json", manifest)
        _write_json(output / "SPLITS.json", partition)
        _write_json(
            output / "AXES.json",
            {"labels": axis, "row_ids": row_ids, "row_roles": [roles[row] for row in row_ids]},
        )
        _write_json(output / "SUPPORT.json", support_report)
        _write_json(
            output / "truth" / "ACCESS_POLICY.json",
            {
                "purpose": "immutable evaluation/query truth; no implicit training authorization",
                "train": "only explicitly acquired known answers may enter Gamma or training",
                "dev": "declared selection/calibration/BO scoring only",
                "test": "separate truth store; no discovery or model selection access",
                "enforcement": "loader contract, not OS-level access separation",
                "known_answers_selected": 0,
                "roots_created": 0,
            },
        )
        artifacts = {
            str(p.relative_to(output)): file_record(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        result = {
            "schema": OUTPUT_SCHEMA,
            "status": "complete",
            "dataset": manifest["dataset"],
            "version": manifest["version"],
            "source_manifest": source_identity,
            "importer_source": file_record(__file__),
            "environment": {"python": sys.version, "numpy": np.__version__},
            "sources": records,
            "split_policy": partition["policy"],
            "counts": partition["counts"],
            "labels": axis,
            "support": support_report,
            "provenance": manifest["provenance"],
            "artifacts": artifacts,
            "text_missing_rows": sum(not x["text"] for x in rows),
            "metadata_fields": meta_columns,
            "training_calls": 0,
            "BO_calls": 0,
            "Chase_calls": 0,
            "Gamma_created": False,
            "rows_dropped": 0,
            "seconds": time.monotonic() - start,
        }
        _write_json(output / "MANIFEST.json", result)
        return result
    except Exception as error:
        _write_json(output / "FAILED.json", {"status": "failed", "error": repr(error)})
        raise


def import_aapd(source_dir, output_dir):
    """Import the real 53,840/1,000/1,000 AAPD CSV release without re-splitting.

    The local research mirror has `abstract` plus 54 binary label columns. This
    imports its published split; it does not claim raw author-release identity.
    """
    source_dir = Path(source_dir).resolve(strict=True)
    with (source_dir / "train.csv").open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    if header[0] != "abstract" or len(header) != 55 or len(set(header)) != 55:
        raise ValueError("AAPD expects abstract plus exactly54 unique declared label columns")
    manifest = {
        "schema": SOURCE_SCHEMA,
        "dataset": "aapd",
        "version": "published_csv54_53840_1000_1000",
        "sources": {s: file_record(source_dir / (s + ".csv")) for s in SPLITS},
        "columns": {"text": ["abstract"], "id": None, "labels": header[1:], "metadata": []},
        "label_axis": {
            "mode": "declared",
            "labels": header[1:],
            "provenance": "published54-column train header; not inferred from test positives",
        },
        "split": {
            "seed": 42,
            "fractions": [0.1, 0.1, 0.8],
            "dev_fraction": 0.1,
            "group_column": None,
        },
        "provenance": {
            "source": "https://github.com/hf-lis/Multi-Label-Datasets/tree/master/AAPD",
            "upstream_declared_by_readme": "https://git.uwaterloo.ca/jimmylin/hedwig-data/-/tree/master/datasets/AAPD",
            "scope": "published research mirror; raw original release not byte-certified",
            "real_metadata_missing": ["paper_id", "title", "authors"],
        },
    }
    # Validate release counts before writing, not after a mislabeled import.
    expected = {"train": 53840, "dev": 1000, "test": 1000}
    for role in SPLITS:
        with (source_dir / (role + ".csv")).open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader) != header or sum(1 for _ in reader) != expected[role]:
                raise ValueError("AAPD published row/column axes differ: " + role)
    return prepare_dataset(manifest, output_dir)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    generic = sub.add_parser(
        "prepare", help="import a SHA-bound source manifest into a new namespace"
    )
    generic.add_argument("--manifest", required=True)
    generic.add_argument("--output", required=True)
    aapd = sub.add_parser(
        "import-aapd", help="import actual published54-label three-split AAPD CSVs"
    )
    aapd.add_argument("--source", required=True)
    aapd.add_argument("--output", required=True)
    args = cli.parse_args(argv)
    try:
        result = (
            prepare_dataset(args.manifest, args.output)
            if args.command == "prepare"
            else import_aapd(args.source, args.output)
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("status", "dataset", "counts", "split_policy", "seconds")
                },
                indent=2,
            )
        )
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
