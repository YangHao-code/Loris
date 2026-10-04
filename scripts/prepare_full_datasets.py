#!/usr/bin/env python3
"""Prepare complete native corpora without changing originals or fitting models.

Examples::

    python scripts/prepare_full_datasets.py --dataset aapd --source DATA/AAPD --output OUT/aapd54
    python scripts/prepare_full_datasets.py --dataset bgc --source DATA/bgc/raw --output OUT/bgc146
    python scripts/prepare_full_datasets.py --dataset arxiv --source DATA/arxiv_parquet --output OUT/arxiv176

Published partitions are retained. Reuters development rows come only from
official training. Unsplit arXiv/PubMed sources receive a declared, deterministic
split by text group. arXiv retains the latest record per ID, never a union of
labels. Test labels are not used to rank/filter classes or tune any parameter.
"""

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


SPLITS = ("train", "dev", "test")
DEFAULT_SEED = "loris-native-20260923-v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def text_group(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def split_value(group_id, seed=DEFAULT_SEED):
    digest = hashlib.sha256((str(seed) + "\0" + group_id).encode()).hexdigest()
    return int(digest[:16], 16) / 2**64


def assign_role(
    text, official=None, *, seed=DEFAULT_SEED, dev_fraction=0.1, fractions=(0.8, 0.1, 0.1)
):
    """Keep official membership; only a train-only development split is derived."""
    if official in SPLITS:
        return official
    value = split_value(text_group(text), seed)
    if official == "official_train_needs_dev":
        return "dev" if value < dev_fraction else "train"
    if official == "not_modapte":
        return official
    return "train" if value < fractions[0] else ("dev" if value < sum(fractions[:2]) else "test")


class LatestVersions:
    """Choose max date/version; reject different payloads tied at the maximum.

    Ties at an obsolete rank do not invalidate a later unambiguous record.
    Payloads are compared as title/abstract/category strings, not unioned.
    """

    def __init__(self):
        self.best = {}
        self.conflicts = {}
        self.duplicates = Counter()

    def add(self, row_id, date, version, payload, location):
        rank = (str(date or ""), int(version))
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).digest()
        previous = self.best.get(row_id)
        if previous is not None:
            self.duplicates[row_id] += 1
        if previous is None or rank > previous[0]:
            self.best[row_id] = (rank, digest, location)
            self.conflicts.pop(row_id, None)
        elif rank == previous[0] and digest != previous[1]:
            self.conflicts.setdefault(row_id, [previous[2]]).append(location)
        elif rank == previous[0] and location < previous[2]:
            self.best[row_id] = (rank, digest, location)

    def locations(self):
        if self.conflicts:
            raise ValueError(
                "different arXiv payloads tied at latest date/version: "
                + ", ".join(sorted(self.conflicts))
            )
        return {row_id: value[2] for row_id, value in self.best.items()}


def xml_value(block, tag):
    match = re.search(r"<" + tag + r">(.*?)</" + tag + r">", block, re.S)
    return html.unescape(match.group(1)) if match else ""


def record(row_id, text, labels, metadata, source, role):
    return {
        "row_id": row_id,
        "text": text,
        "labels": list(labels),
        "metadata": metadata,
        "source": source,
        "role": role,
    }


def validate_shards(files):
    """Check every member of one explicitly numbered snapshot is present."""
    parsed = [re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.parquet", path.name) for path in files]
    if not files or any(match is None for match in parsed):
        raise ValueError(
            "arXiv requires a complete numbered Parquet snapshot (part-N-of-M.parquet)"
        )
    families = {(match[1], int(match[3])) for match in parsed}
    if len(families) != 1:
        raise ValueError("mixed arXiv snapshot shard families")
    _, total = next(iter(families))
    if total < 1 or len(files) != total or {int(match[2]) for match in parsed} != set(range(total)):
        raise ValueError("incomplete arXiv snapshot shards")


def prepare_readers(dataset, source, seed, fractions, dev_fraction):
    """Return a repeatable reader plus an entire native axis and source evidence."""
    source = Path(source).resolve(strict=True)
    extra = {}
    semantic = []
    if dataset == "aapd":
        files = [source / (role + ".csv") for role in SPLITS]
        with files[0].open() as stream:
            header = next(csv.reader(stream))
        if header[0] != "abstract" or len(header) != 55:
            raise ValueError("AAPD requires abstract plus all 54 native labels")
        labels = header[1:]

        def reader():
            expected = {"train": 53840, "dev": 1000, "test": 1000}
            for role, path in zip(SPLITS, files):
                count = 0
                with path.open() as stream:
                    rows = csv.DictReader(stream)
                    if rows.fieldnames != header:
                        raise ValueError("AAPD source axes differ")
                    for index, row in enumerate(rows):
                        if any(row[label] not in ("0", "1") for label in labels):
                            raise ValueError("AAPD labels must be complete binary values")
                        count += 1
                        yield record(
                            "aapd:" + role + ":" + str(index),
                            row["abstract"],
                            [label for label in labels if row[label] == "1"],
                            {},
                            {"file": path.name, "record": index, "split_file": role},
                            role,
                        )
                if count != expected[role]:
                    raise ValueError("incomplete AAPD published split: " + role)

        policy = {
            "official_splits": "published mirror partitions retained; original byte identity not asserted",
            "method": "retain 53840/1000/1000",
            "source": "https://aclanthology.org/C18-1330.pdf",
        }
    elif dataset == "bgc":
        files = [source / ("BlurbGenreCollection_EN_" + role + ".txt") for role in SPLITS]
        labels = sorted(
            {
                html.unescape(label)
                for path in files
                for label in re.findall(r"<d\d+>(.*?)</d\d+>", path.read_text(), re.S)
            }
        )
        if len(labels) != 146:
            raise ValueError("original BGC corpus must expose all 146 assigned labels")
        semantic = ["title", "author", "published", "page_num", "isbn", "url"]

        def reader():
            expected = {"train": 58715, "dev": 14785, "test": 18394}
            for role, path in zip(SPLITS, files):
                blocks = re.findall(r"<book\b[^>]*>(.*?)</book>", path.read_text(), re.S)
                if len(blocks) != expected[role]:
                    raise ValueError("incomplete original BGC partition: " + role)
                for index, block in enumerate(blocks):
                    metadata = {field: xml_value(block, field).strip() for field in semantic}
                    gold = sorted(
                        {
                            html.unescape(label)
                            for label in re.findall(r"<d\d+>(.*?)</d\d+>", block, re.S)
                        }
                    )
                    yield record(
                        "bgc:" + role + ":" + str(index),
                        xml_value(block, "body"),
                        gold,
                        metadata,
                        {"file": path.name, "record": index, "split_file": role},
                        role,
                    )

        hierarchy = source / "hierarchy.txt"
        if hierarchy.exists():
            edges = [line.split("\t") for line in hierarchy.read_text().splitlines()]
            nodes = sorted({node.strip() for edge in edges for node in edge})
            extra["taxonomy"] = {
                "edges": edges,
                "nodes": nodes,
                "declared_but_unobserved": sorted(set(nodes) - set(labels)),
                "active_axis": "all corpus-assigned labels; unassigned taxonomy nodes retained separately",
            }
            files.append(hierarchy)
        policy = {
            "official_splits": True,
            "method": "retain all shipped original rows and assigned labels; ISBN is metadata, split plus ordinal is row identity",
            "source": "https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html",
        }
    elif dataset == "reuters":
        files = sorted(source.glob("reut2-*.sgm"))
        axis_file = source / "all-topics-strings.lc.txt"
        labels = [line.strip() for line in axis_file.read_text().splitlines()]
        if len(files) != 22 or len(labels) != 135 or len(set(labels)) != 135:
            raise ValueError("Reuters requires all 22 SGML files and 135 declared topic names")
        semantic = ["title", "dateline", "date"]

        def reader():
            counts = Counter()
            for path in files:
                if path.suffix != ".sgm":
                    continue
                for index, (attrs, block) in enumerate(
                    re.findall(
                        r"<REUTERS\s+([^>]+)>(.*?)</REUTERS>",
                        path.read_text(encoding="latin1"),
                        re.S,
                    )
                ):
                    attrs = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
                    match = re.search(r"<TEXT\b[^>]*>(.*?)</TEXT>", block, re.S)
                    text = " ".join(
                        html.unescape(
                            re.sub(r"<[^>]+>", " ", match.group(1) if match else "")
                        ).split()
                    )
                    gold = re.findall(r"<D>(.*?)</D>", xml_value(block, "TOPICS"))
                    official = "not_modapte"
                    if attrs.get("TOPICS") == "YES" and attrs.get("LEWISSPLIT") == "TRAIN":
                        official = "official_train_needs_dev"
                    elif attrs.get("TOPICS") == "YES" and attrs.get("LEWISSPLIT") == "TEST":
                        official = "test"
                    counts[official] += 1
                    role = assign_role(text, official, seed=seed, dev_fraction=dev_fraction)
                    metadata = {
                        field: xml_value(block, field.upper()).strip() for field in semantic
                    }
                    yield record(
                        "reuters:" + attrs["NEWID"],
                        text,
                        gold,
                        metadata,
                        {
                            "file": path.name,
                            "record": index,
                            "NEWID": attrs["NEWID"],
                            "LEWISSPLIT": attrs.get("LEWISSPLIT"),
                            "TOPICS_attribute": attrs.get("TOPICS"),
                        },
                        role,
                    )
            if counts != Counter(official_train_needs_dev=9603, test=3299, not_modapte=8676):
                raise ValueError("incomplete Reuters native/ModApte identity counts")

        files.append(axis_file)
        policy = {
            "official_test": True,
            "official_train": 9603,
            "official_test_rows": 3299,
            "method": "split only official train into train/dev by normalized text group; retain non-ModApte originals separately",
            "dev_fraction": dev_fraction,
            "seed": seed,
            "source": "https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection",
        }
    elif dataset in ("ecthr_a", "eurlex"):
        import pyarrow.parquet as pq

        labels = [str(index) for index in range(10 if dataset == "ecthr_a" else 100)]
        files = sorted(source.rglob("*.parquet"))
        by_role = {}
        for path in files:
            role = path.name.split("-")[0]
            role = "dev" if role == "validation" else role
            if role not in SPLITS or role in by_role:
                raise ValueError("supply exactly one original LexGLUE Parquet per split")
            by_role[role] = path
        if set(by_role) != set(SPLITS):
            raise ValueError("all three LexGLUE splits required")

        def reader():
            expected = dict(
                zip(SPLITS, (9000, 1000, 1000) if dataset == "ecthr_a" else (55000, 5000, 5000))
            )
            for role in SPLITS:
                index = 0
                for batch in pq.ParquetFile(by_role[role]).iter_batches(batch_size=4096):
                    for row in batch.to_pylist():
                        text = (
                            "\n".join(row["text"]) if isinstance(row["text"], list) else row["text"]
                        )
                        yield record(
                            "lexglue:" + dataset + ":" + role + ":" + str(index),
                            text,
                            [str(value) for value in row["labels"]],
                            {},
                            {"file": by_role[role].name, "record": index, "split_file": role},
                            role,
                        )
                        index += 1
                if index != expected[role]:
                    raise ValueError("incomplete official LexGLUE split")

        policy = {
            "official_splits": True,
            "method": "retain all official rows, including empty gold sets; join factual paragraphs with newline",
            "source": "https://huggingface.co/datasets/coastalcph/lex_glue",
        }
    elif dataset == "pubmed":
        files = [source]
        labels = list("ABCDEFGHIJLMNZ")
        semantic = ["title"]
        parents, pmids, texts, group_ids = [], {}, {}, {}

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left, right):
            left, right = find(left), find(right)
            parents[max(left, right)] = min(left, right)

        with source.open() as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                parents.append(index)
                text = row["Title"].strip() + "\n\n" + row["abstractText"].strip()
                key, pmid = text_group(text), row["pmid"].strip()
                if key in texts:
                    union(index, texts[key])
                if pmid and pmid in pmids:
                    union(index, pmids[pmid])
                texts[key] = index
                if pmid:
                    pmids[pmid] = index
                group_ids[index] = key
        if len(parents) != 50000:
            raise ValueError("local PubMed derivative must contain its complete 50000 rows")
        group_keys = {}
        for index, key in group_ids.items():
            root = find(index)
            group_keys[root] = min(group_keys.get(root, key), key)

        def reader():
            with source.open() as stream:
                for index, row in enumerate(csv.DictReader(stream)):
                    if any(row[label] not in ("0", "1", "0.0", "1.0") for label in labels):
                        raise ValueError("complete binary PubMed labels required")
                    value = split_value(group_keys[find(index)], seed)
                    role = (
                        "train"
                        if value < fractions[0]
                        else ("dev" if value < sum(fractions[:2]) else "test")
                    )
                    text = row["Title"].strip() + "\n\n" + row["abstractText"].strip()
                    yield record(
                        "pubmed14:" + str(index),
                        text,
                        [label for label in labels if row[label] in ("1", "1.0")],
                        {"title": row["Title"]},
                        {"file": source.name, "record": index, "pmid": row["pmid"]},
                        role,
                    )

        policy = {
            "official_splits": False,
            "method": "group connected components sharing PMID or normalized text; retain every original row",
            "seed": seed,
            "fractions": list(fractions),
            "scope": "local 50000-row/14-column derivative, not all PubMed",
        }
    elif dataset == "arxiv":
        import pyarrow.parquet as pq

        files = sorted(source.glob("*.parquet"))
        validate_shards(files)
        semantic = ["title", "authors", "first_submitted"]
        latest, catalog = LatestVersions(), set()
        for path in files:
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(
                columns=["id", "title", "abstract", "categories", "update_date", "versions"]
            ):
                for row in batch.to_pylist():
                    versions = row["versions"] or []
                    version = max(
                        [int(value["version"].lstrip("v")) for value in versions], default=0
                    )
                    latest.add(
                        row["id"],
                        row["update_date"],
                        version,
                        (row["title"], row["abstract"], row["categories"]),
                        (path.name, index),
                    )
                    catalog.update(row["categories"].split())
                    index += 1
        if latest.conflicts:
            error = ValueError(
                "arXiv latest-version payload ties need separate review: "
                + json.dumps(latest.conflicts)
            )
            error.conflict_records = latest.conflicts
            raise error
        chosen = latest.locations()
        labels = sorted(catalog)
        extra["deduplication"] = {
            "policy": "max update_date then max numerical version; identical ties use first source location; different tied latest payloads reject; never union labels",
            "duplicate_ids": len(latest.duplicates),
            "extra_rows_removed": sum(latest.duplicates.values()),
            "selected_duplicate_locations": {key: chosen[key] for key in latest.duplicates},
            "source_records_retained_in_originals": True,
        }

        def reader():
            for path in files:
                index = 0
                for batch in pq.ParquetFile(path).iter_batches(
                    columns=["id", "title", "abstract", "categories", "authors", "versions"]
                ):
                    for row in batch.to_pylist():
                        location = (path.name, index)
                        index += 1
                        if chosen[row["id"]] != location:
                            continue
                        text = row["title"].strip() + "\n\n" + row["abstract"].strip()
                        versions = row["versions"] or []
                        metadata = {
                            "title": row["title"],
                            "authors": row["authors"] or "",
                            "first_submitted": versions[0]["created"] if versions else "",
                        }
                        yield record(
                            "arxiv:" + row["id"],
                            text,
                            row["categories"].split(),
                            metadata,
                            {"file": path.name, "record": location[1], "arxiv_id": row["id"]},
                            assign_role(text, seed=seed, fractions=fractions),
                        )

        policy = {
            "official_splits": False,
            "method": "all local snapshot IDs, latest version per ID, normalized text groups assigned by deterministic hash",
            "seed": seed,
            "fractions": list(fractions),
            "source": "https://info.arxiv.org/help/bulk_data.html",
            "paper_exact_snapshot": False,
        }
    else:
        raise ValueError("unknown dataset")
    return reader, labels, files, semantic, policy, extra


def materialize(reader, labels, sources, semantic, policy, extra, output, name):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("new output directory required: " + str(output))
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("complete unique native label axis required")
    output.mkdir(parents=True)
    (output / "documents").mkdir()
    (output / "truth").mkdir()
    paths, gold, ids, stats, supports, text_sets = {}, {}, {}, {}, {}, {}
    axis, seen = {label: index for index, label in enumerate(labels)}, set()
    try:
        for row in reader():
            role = row["role"]
            if role not in SPLITS + ("not_modapte",):
                raise ValueError("undeclared split")
            if row["row_id"] in seen:
                raise ValueError("duplicate retained document ID")
            seen.add(row["row_id"])
            if set(row["metadata"]) != set(semantic):
                raise ValueError("only declared genuine semantic metadata may enter documents")
            if not isinstance(row["text"], str) or (
                not row["text"].strip() and role != "not_modapte"
            ):
                raise ValueError("missing source text in experimental split")
            if role not in paths:
                paths[role] = (output / "documents" / (role + ".jsonl")).open("w")
                gold[role] = (output / "truth" / (role + ".targets.bin")).open("wb")
                ids[role], stats[role] = [], Counter()
                supports[role], text_sets[role] = np.zeros(len(labels), dtype=np.int64), set()
            target = np.zeros(len(labels), dtype=bool)
            for label in row["labels"]:
                if label not in axis:
                    raise ValueError("gold label outside complete declared axis")
                target[axis[label]] = True
            gold[role].write(target.tobytes())
            supports[role] += target
            ids[role].append(row["row_id"])
            stats[role]["rows"] += 1
            stats[role]["empty_text_rows"] += not row["text"].strip()
            stats[role]["zero_positive_label_rows"] += not target.any()
            digest = hashlib.sha256(row["text"].encode()).digest()
            stats[role]["duplicate_exact_text_rows"] += digest in text_sets[role]
            text_sets[role].add(digest)
            document = {key: row[key] for key in ("row_id", "text", "metadata", "source")}
            paths[role].write(json.dumps(document, ensure_ascii=False) + "\n")
        for stream in list(paths.values()) + list(gold.values()):
            stream.close()
        if not set(SPLITS) <= set(ids):
            raise ValueError("nonempty train/dev/test required")
        for role in ids:
            binary = output / "truth" / (role + ".targets.bin")
            targets = np.memmap(binary, mode="r", dtype=bool, shape=(len(ids[role]), len(labels)))
            np.savez_compressed(
                output / "truth" / (role + ".npz"),
                row_ids=np.asarray(ids[role], dtype=str),
                labels=np.asarray(labels, dtype=str),
                targets=targets,
                observed=np.ones_like(targets),
            )
            del targets
            binary.unlink()
        all_ids = [value for role in SPLITS for value in ids[role]]
        all_roles = [role for role in SPLITS for _ in ids[role]]
        write_json(output / "labels.json", labels)
        write_json(
            output / "AXES.json", {"labels": labels, "row_ids": all_ids, "row_roles": all_roles}
        )
        write_json(
            output / "SPLITS.json",
            {
                "policy": policy,
                "counts": {role: len(ids[role]) for role in SPLITS},
                "membership": {role: ids[role] for role in SPLITS},
            },
        )
        support = {
            "train_positive_support": dict(zip(labels, supports["train"].tolist())),
            "unsupported_axis_labels": [
                label for label, count in zip(labels, supports["train"]) if not count
            ],
            "excluded_candidate_labels": [],
            "dev_test_used_to_choose_labels": False,
            "native_catalog_from_all_original_label_names": True,
            "frequency_or_score_label_selection": False,
        }
        write_json(output / "SUPPORT.json", support)
        write_json(
            output / "SOURCE_MANIFEST.json",
            {
                "schema": "loris_native_source_manifest_v1",
                "dataset": name,
                "sources": [identity(path) for path in sources],
                "source_files_modified": False,
                "split_policy": policy,
                "native_axis": labels,
                "extra": extra,
            },
        )
        write_json(
            output / "SEMANTIC_METADATA.json",
            {
                "fields": semantic,
                "title_field": "title" if "title" in semantic else None,
                "source_bookkeeping_is_not_semantic": True,
                "target_labels_excluded": True,
            },
        )
        write_json(
            output / "truth" / "ACCESS_POLICY.json",
            {
                "purpose": "separate immutable gold for explicit roots, queries and evaluation",
                "test": "not permitted for model/rule selection",
                "enforcement": "loader contracts, not OS sandbox",
                "known_answers_selected": 0,
            },
        )
        artifacts = {
            str(path.relative_to(output)): identity(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema": "loris_dataset_prepared_v1",
            "status": "complete",
            "dataset": name,
            "version": "native_v1",
            "labels": labels,
            "counts": {role: len(ids[role]) for role in SPLITS},
            "metadata_fields": semantic,
            "artifacts": artifacts,
            "split_policy": policy,
            "support": support,
            "provenance": {
                "native_complete_task": True,
                "paper_exact_subset": False,
                "source_files_modified": False,
            },
            "training_calls": 0,
            "BO_calls": 0,
            "Chase_calls": 0,
            "Gamma_created": False,
        }
        write_json(output / "MANIFEST.json", manifest)
        audit = {
            "schema": "loris_release_dataset_audit_v1",
            "dataset": name,
            "scope": "complete native source task, separately named from paper projections",
            "source_provenance": policy,
            "labels": labels,
            "expected_counts": manifest["counts"],
            "duplicate_policy": "report_official",
            "application_splits": ["train", "test"],
            "text_policy": {"empty_text": "reject", "missing_text_reason": ""},
            "splits": {
                role: {
                    "documents": artifacts["documents/" + role + ".jsonl"],
                    "truth": artifacts["truth/" + role + ".npz"],
                }
                for role in SPLITS
            },
        }
        write_json(output / "dataset_audit.json", audit)
        overlaps = {
            left + "/" + right: len(text_sets[left] & text_sets[right])
            for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
        }
        receipt = {
            "schema": "loris_native_preparation_v1",
            "status": "prepared_raw_and_splits",
            "dataset": name,
            "label_axis_size": len(labels),
            "splits": {
                role: {key: int(value) for key, value in stats[role].items()} for role in ids
            },
            "cross_split_exact_texts": overlaps,
            "split_policy": policy,
            "extra": extra,
            "empty_gold_rows_preserved": True,
            "models_or_rules_prepared": False,
        }
        write_json(output / "PREPARED.json", receipt)
        return manifest
    except Exception as error:
        for stream in list(paths.values()) + list(gold.values()):
            stream.close()
        write_json(
            output / "FAILED.json",
            {"status": "failed", "error": repr(error), "originals_modified": False},
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("aapd", "bgc", "reuters", "arxiv", "ecthr_a", "eurlex", "pubmed"),
        required=True,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Raw source directory; PubMed takes its original CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing paths are rejected",
    )
    parser.add_argument(
        "--name", help="Distinct native dataset identity, never a paper-subset name"
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "DEV", "TEST"),
    )
    parser.add_argument("--reuters-dev-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; source and prior artifacts are never overwritten")
    if any(value <= 0 for value in args.fractions) or abs(sum(args.fractions) - 1) > 1e-9:
        parser.error("positive train/dev/test fractions must sum to one")
    if not 0 < args.reuters_dev_fraction < 1:
        parser.error("Reuters development fraction must lie strictly between zero and one")
    csv.field_size_limit(30_000_000)
    try:
        prepared = prepare_readers(
            args.dataset, args.source, args.seed, args.fractions, args.reuters_dev_fraction
        )
    except ValueError as error:
        if hasattr(error, "conflict_records"):
            args.output.mkdir(parents=True)
            write_json(args.output / "DUPLICATE_CONFLICTS.json", error.conflict_records)
            write_json(
                args.output / "FAILED.json",
                {
                    "status": "failed",
                    "reason": "ambiguous_latest_version_payload",
                    "originals_modified": False,
                },
            )
        raise
    defaults = {
        "aapd": "aapd_native54_published",
        "bgc": "bgc_native146_official",
        "reuters": "reuters21578_native135_modapte",
        "arxiv": "arxiv_native" + str(len(prepared[1])) + "_latest_grouped",
        "ecthr_a": "lexglue_ecthr_a_native10_official",
        "eurlex": "lexglue_eurlex_native100_official",
        "pubmed": "pubmed_mesh14_local50000_grouped",
    }
    name = args.name or defaults[args.dataset]
    manifest = materialize(*prepared, args.output, name)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dataset": name,
                "counts": manifest["counts"],
                "labels": len(manifest["labels"]),
                "manifest": str(args.output / "MANIFEST.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
