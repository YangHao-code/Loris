"""Fit-only contextual candidates with paper entropy screening.

No BO, Chase, model calls, control truth or test inputs belong in this API.
Entropy uses per-label occurrence fractions exactly as in Section 5.1; these
fractions are not normalized across labels for multi-label documents. The
historical MCC/diversity screening remains an explicit ablation.
"""

import argparse
import collections
import math
import gzip
import importlib
import json
import re
import sqlite3
import shutil
import time
from dataclasses import replace
from pathlib import Path
import hashlib
import numpy as np
from loris.document import Document
from loris._engine.pattern_anchors import literal_regex
from loris._engine.pattern_discovery import CompleteRegexStore
from loris._engine.rlearner import canonical, digest, array_identity
from loris._engine.rlearner_config import PatternSpec, ParameterDomain, FREQUENCY_OPS
from loris._learning.semantic_patterns import semantic_record, _spec
from .manifest import file_record, bound
from .mcc_ranking import select_group
from . import pattern_selection

FAMILIES = ("Match", "Cooccur", "Before", "Freq")
SCREENING_POLICIES = ("entropy", "mcc", "discriminative")


def label_entropy(mask, truth):
    """H(p)=-sum P(label|p) log P(label|p), using only supplied fit rows."""
    mask, truth = np.asarray(mask), np.asarray(truth)
    if (
        mask.dtype != np.bool_
        or truth.dtype != np.bool_
        or truth.ndim != 2
        or mask.shape != (len(truth),)
        or not truth.shape[1]
    ):
        raise ValueError("Boolean fit mask and nonempty label axis required")
    if not mask.any():
        return None  # No conditional distribution, not zero-entropy evidence.
    fractions = truth[mask].mean(axis=0)
    positive = fractions[fractions > 0]
    return float(-np.sum(positive * np.log(positive)))


def entropy_limit(truth, max_entropy):
    """A configurable screening ceiling; default is the cluster baseline H.

    The paper gives H but no numeric cutoff. This data-local default is an
    operational choice, recorded in every supply protocol and cluster summary.
    """
    if max_entropy is not None and (
        isinstance(max_entropy, bool) or not np.isfinite(max_entropy) or max_entropy < 0
    ):
        raise ValueError("maximum entropy must be finite and nonnegative")
    return (
        float(max_entropy)
        if max_entropy is not None
        else label_entropy(np.ones(len(truth), dtype=bool), truth)
    )


def select_entropy_group(items, quota, *, deduplicate_masks=True):
    """Bound the already screened candidate inventory by entropy, then support.

    Equal fit masks are only a search diversity reduction, never a semantic
    equivalence claim. Final rule selection remains full validation F1.
    """
    unique, rejected = {}, []
    for item in sorted(items, key=lambda value: value["spec"].key):
        key = (hashlib.sha256(np.packbits(item["mask"], bitorder="little").tobytes()).hexdigest()
               if deduplicate_masks else item["spec"].key)
        if deduplicate_masks and key in unique:
            rejected.append((item, unique[key]["spec"].key))
        else:
            unique[key if deduplicate_masks else item["spec"].key] = item
    ranked = sorted(
        unique.values(), key=lambda value: (value["entropy"], -value["support"], value["spec"].key)
    )
    return [(item, "entropy_then_coverage", None, None) for item in ranked[:quota]], rejected


def add_paper_anchors(docs, clusters, body, out, tfidf_terms):
    """Add fit-only TF-IDF terms, number spans and short dependency verb phrases.

    Semantic noun/entity anchors and pairs have already been parsed. This
    supplement retains their raw inventory and records every new literal span.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w\w+\b")
    selected = {}
    try:
        matrix = vectorizer.fit_transform([d.cnt for d in docs])
    except ValueError as error:
        if "empty vocabulary" not in str(error):
            raise
        matrix = None
    if matrix is not None:
        names = vectorizer.get_feature_names_out()
        for cluster in sorted(set(clusters.tolist())):
            scores = np.asarray(matrix[clusters == cluster].mean(axis=0)).ravel()
            order = sorted(np.flatnonzero(scores > 0), key=lambda i: (-scores[i], names[i]))[
                :tfidf_terms
            ]
            selected[int(cluster)] = {str(names[i]) for i in order}
    counts, definitions, seen = collections.Counter(), {}, set()
    number = re.compile(r"\b\d+(?:[.,:/-]\d+)*(?:%|[A-Za-z]{1,3})?\b")
    dependencies = {"dobj", "obj", "iobj", "attr", "oprd", "aux", "auxpass", "neg", "advmod", "prt"}
    with gzip.open(out / "body_extraction.jsonl.gz", "rt") as parsed, gzip.open(
        out / "additional_anchors.jsonl.gz", "wt"
    ) as saved:
        for record in map(json.loads, parsed):
            row, cluster = record["fit_local"], record["cluster"]
            text = docs[row].cnt
            spans = []
            for match in re.finditer(r"(?u)\b\w\w+\b", text):
                if match.group().lower() in selected.get(cluster, set()):
                    spans.append((match.start(), match.end(), "tfidf_term"))
            spans.extend((m.start(), m.end(), "number_span") for m in number.finditer(text))
            tokens = record["tokens"]
            for i, token in enumerate(tokens):
                if token[2] != "VERB":
                    continue
                parts = [i] + [
                    j for j, t in enumerate(tokens) if t[4] == i and t[3] in dependencies
                ]
                if len(parts) > 1 and max(parts) - min(parts) < 8:
                    spans.append((tokens[min(parts)][5], tokens[max(parts)][6], "verb_phrase"))
            for start, end, kind in sorted(set(spans)):
                surface = text[start:end]
                raw, structure = literal_regex(surface)
                definition = dict(
                    raw=raw, flags=int(re.IGNORECASE), literal_alternatives_by_token=structure
                )
                key = digest(["cnt", raw, int(re.IGNORECASE)])
                item = (cluster, key, row)
                if item not in seen:
                    seen.add(item)
                    counts[(cluster, key)] += 1
                    definitions[(cluster, key)] = (definition, surface)
                saved.write(
                    canonical(
                        dict(
                            fit_local=row,
                            cluster=cluster,
                            start=start,
                            end=end,
                            surface=surface,
                            kind=kind,
                            key=key,
                        )
                    )
                    + "\n"
                )
    for (cluster, key), count in sorted(counts.items()):
        definition, surface = definitions[(cluster, key)]
        # max is a conservative parse-support lower bound when semantic and
        # added sources overlap. Actual coverage is measured with full masks
        # below; summing these counts would double-count documents.
        body.execute(
            "INSERT INTO anchors VALUES(?,?,?,?) ON CONFLICT(cluster,key) DO UPDATE SET parse_df=max(parse_df,excluded.parse_df)",
            (cluster, key, canonical(definition), count),
        )
        body.execute(
            "INSERT OR REPLACE INTO anchor_validity VALUES(?,?,?,?,?)",
            (cluster, key, 1, surface, canonical(re.findall(r"\w+", surface))),
        )
    body.commit()
    save(
        out / "TFIDF_TERMS.json",
        dict(
            fit_only=True,
            per_cluster={str(c): sorted(t) for c, t in selected.items()},
            max_terms_per_cluster=tfidf_terms,
            ranking="mean fit TF-IDF descending; token ascending on ties",
        ),
    )


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def correlation(mask, y):
    n = len(mask)
    s = int(mask.sum())
    tp = y[mask].sum(0).astype(float)
    pos = y.sum(0).astype(float)
    den = np.sqrt(s * (n - s) * pos * (n - pos))
    return np.divide(n * tp - s * pos, den, out=np.zeros(y.shape[1]), where=den > 0), tp.astype(int)


def title_domain(docs, clusters, stops, out):
    db = sqlite3.connect(str(out / "title_raw.sqlite"))
    db.executescript(
        "CREATE TABLE anchors(cluster INTEGER,key TEXT,definition TEXT,parse_df INTEGER,PRIMARY KEY(cluster,key)); CREATE TABLE pairs(cluster INTEGER,family TEXT,left_key TEXT,right_key TEXT,parse_df INTEGER,PRIMARY KEY(cluster,family,left_key,right_key));"
    )
    with gzip.open(out / "title_extraction.jsonl.gz", "wt") as f:
        for row, (doc, cl) in enumerate(zip(docs, clusters)):
            words = list(re.finditer(r"[^\W\d_]+", doc.ttl, flags=re.UNICODE))
            defs = {}
            singles = []
            for a, w in enumerate(words):
                if len(w.group()) < 2 or w.group().lower() in stops:
                    continue
                for width in (1, 2, 3):
                    part = words[a : a + width]
                    if len(part) != width or any(
                        not doc.ttl[u.end() : v.start()].isspace() for u, v in zip(part, part[1:])
                    ):
                        continue
                    surface = doc.ttl[part[0].start() : part[-1].end()]
                    raw, structure = literal_regex(surface)
                    d = {
                        "raw": raw,
                        "flags": int(re.IGNORECASE),
                        "literal_alternatives_by_token": structure,
                    }
                    key = digest(["ttl", raw, int(re.IGNORECASE)])
                    defs[key] = d
                    if width == 1:
                        singles.append(key)
            pairs = set()
            for left, right in zip(singles, singles[1:]):
                if left == right:
                    continue
                lo, hi = sorted((left, right))
                pairs.add(("Cooccur", lo, hi))
                pairs.add(("Before", left, right))
            for key, d in defs.items():
                db.execute(
                    "INSERT INTO anchors VALUES(?,?,?,1) ON CONFLICT(cluster,key) DO UPDATE SET parse_df=parse_df+1",
                    (int(cl), key, canonical(d)),
                )
            for family, left, right in pairs:
                db.execute(
                    "INSERT INTO pairs VALUES(?,?,?,?,1) ON CONFLICT(cluster,family,left_key,right_key) DO UPDATE SET parse_df=parse_df+1",
                    (int(cl), family, left, right),
                )
            f.write(
                canonical(
                    {"fit_local": row, "cluster": int(cl), "anchors": defs, "pairs": sorted(pairs)}
                )
                + "\n"
            )
    db.commit()
    return db


def _select_group_reference(items, quota, y):
    # Empirical identical fit masks are a SEARCH diversity prefilter, not a claim
    # that two regexes are equivalent on unseen documents.
    unique = {}
    rejected = []
    for item in sorted(items, key=lambda v: v["spec"].key):
        h = hashlib.sha256(np.packbits(item["mask"], bitorder="little").tobytes()).hexdigest()
        if h in unique:
            rejected.append((item, unique[h]["spec"].key))
        else:
            unique[h] = item
    pool = list(unique.values())
    chosen = []
    used = set()
    cover = np.zeros((2, y.shape[1], len(y)), bool)
    diversity_quota = int(np.ceil(quota * 0.75))
    for turn in range(diversity_quota):
        label = turn % y.shape[1]
        best = None
        for item in pool:
            if item["spec"].key in used:
                continue
            sign = int(item["corr"][label] >= 0)
            novel = np.count_nonzero(item["mask"] & ~cover[sign, label])
            value = abs(float(item["corr"][label])) * np.sqrt(novel / max(1, item["support"]))
            rank = (-value, -abs(float(item["corr"][label])), -item["support"], item["spec"].key)
            if best is None or rank < best[0]:
                best = (rank, item, label, sign)
        if best is None:
            break
        _, item, label, sign = best
        used.add(item["spec"].key)
        chosen.append((item, "label_balanced_complementarity", label, sign))
        cover[sign, label] |= item["mask"]
    covered = np.zeros(len(y), bool)
    for item, _, _, _ in chosen:
        covered |= item["mask"]
    while len(chosen) < quota:
        candidates = [v for v in pool if v["spec"].key not in used]
        if not candidates:
            break
        item = min(
            candidates,
            key=lambda v: (
                -np.count_nonzero(v["mask"] & ~covered),
                -float(np.max(abs(v["corr"]))),
                -v["support"],
                v["spec"].key,
            ),
        )
        used.add(item["spec"].key)
        chosen.append((item, "coverage_complementarity", None, None))
        covered |= item["mask"]
    return chosen, rejected


def body_domain(docs, fit, row_ids, clusters, stops, nlp, out, parse_workers=1):
    db = sqlite3.connect(str(out / "body_raw.sqlite"))
    db.executescript(
        "CREATE TABLE anchors(cluster INTEGER,key TEXT,definition TEXT,parse_df INTEGER,PRIMARY KEY(cluster,key));CREATE TABLE pairs(cluster INTEGER,family TEXT,left_key TEXT,right_key TEXT,parse_df INTEGER,PRIMARY KEY(cluster,family,left_key,right_key));CREATE TABLE anchor_validity(cluster INTEGER,key TEXT,valid INTEGER,surface TEXT,tokens TEXT,PRIMARY KEY(cluster,key));"
    )
    validity = {}
    nlp.max_length = max(nlp.max_length, max(len(d.cnt) for d in docs) + 1)
    with gzip.open(out / "body_extraction.jsonl.gz", "wt") as stream:
        for local, parsed in enumerate(nlp.pipe((d.cnt for d in docs), batch_size=32, n_process=parse_workers)):
            cl = int(clusters[local])
            record = semantic_record(parsed)
            record.update(
                row=int(fit[local]), row_id=row_ids[local], fit_local=local, cluster=cl, attr="cnt"
            )
            definitions = {}
            keys = {}
            pairset = set()
            for anchor in record["anchors"]:
                if record["text"][anchor["start"] : anchor["end"]] != anchor["surface"]:
                    raise ValueError("semantic span differs")
                raw, structure = literal_regex(anchor["surface"])
                definition = dict(
                    raw=raw, flags=int(re.IGNORECASE), literal_alternatives_by_token=structure
                )
                key = digest(["cnt", raw, int(re.IGNORECASE)])
                anchor["key"] = key
                keys[(anchor["token_start"], anchor["token_end"])] = key
                definitions[key] = definition
                tokens = re.findall(r"[^\W\d_]+", anchor["surface"], flags=re.UNICODE)
                valid = any(len(t) >= 2 and t.isalpha() and t.lower() not in stops for t in tokens)
                if (cl, key) in validity and validity[(cl, key)] != valid:
                    raise ValueError("canonical content validity differs")
                if (cl, key) not in validity:
                    db.execute(
                        "INSERT INTO anchor_validity VALUES (?,?,?,?,?)",
                        (cl, key, int(valid), anchor["surface"], canonical(tokens)),
                    )
                validity[(cl, key)] = valid
            for key, definition in definitions.items():
                db.execute(
                    "INSERT INTO anchors VALUES(?,?,?,1) ON CONFLICT(cluster,key) DO UPDATE SET parse_df=parse_df+1",
                    (cl, key, canonical(definition)),
                )
            for pair in record["pairs"]:
                left, right = keys[tuple(pair["left"])], keys[tuple(pair["right"])]
                if left != right:
                    lo, hi = sorted((left, right))
                    pairset.add(("Cooccur", lo, hi))
                    pairset.add(("Before", left, right))
            for family, left, right in pairset:
                db.execute(
                    "INSERT INTO pairs VALUES(?,?,?,?,1) ON CONFLICT(cluster,family,left_key,right_key) DO UPDATE SET parse_df=parse_df+1",
                    (cl, family, left, right),
                )
            stream.write(canonical(record) + "\n")
            if (local + 1) % 100 == 0:
                db.commit()
                print(
                    canonical(
                        {
                            "event": "complete_fit_semantic_parse",
                            "parsed": local + 1,
                            "total": len(docs),
                        }
                    ),
                    flush=True,
                )
    db.commit()
    return db


EXTRACTION_FUNCTIONS = ("add_paper_anchors", "title_domain", "body_domain")


def extraction_source_digest(text):
    """SHA of the exact source of the raw extraction functions in a module text."""
    import ast
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    parts = []
    for name in EXTRACTION_FUNCTIONS:
        node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
        if node is None:
            raise ValueError("extraction function missing: " + name)
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        parts.append("".join(lines[start - 1:node.end_lineno]))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _reuse_extraction(source, output, policy):
    """Reuse authenticated raw parsing, never another screening's decisions."""
    source = Path(source).resolve(strict=True)
    complete = read(source / "COMPLETE.json")
    if complete.get("status") != "complete":
        raise ValueError("extraction reuse requires a completed supply")
    protocol = read(bound(complete["protocol"], source))
    if read(source / "FIT_IDENTITY.json") != read(output / "FIT_IDENTITY.json"):
        raise ValueError("extraction reuse fit/text/cluster identity differs")
    for key in ("title_field", "spacy_version", "spacy_model", "model_files",
                "extraction", "tfidf_terms_per_cluster"):
        if protocol.get(key) != policy.get(key):
            raise ValueError("extraction reuse configuration differs: " + key)
    # Raw parsing depends only on the extraction functions, not on screening.
    # Older supplies record only the whole-file SHA: recompute the extraction
    # digest from that exact (SHA-verified) source file.
    old_digest = protocol.get("extraction_source_sha256")
    if old_digest is None:
        record = protocol.get("source") or {}
        old_path = Path(record.get("path", ""))
        if not old_path.is_file() or file_record(old_path)["sha256"] != record.get("sha256"):
            raise ValueError("extraction reuse source file is unavailable or changed")
        old_digest = extraction_source_digest(old_path.read_text())
    if old_digest != policy.get("extraction_source_sha256"):
        raise ValueError("extraction reuse configuration differs: extraction_source_sha256")
    if read(source / "STOP_WORDS.json") != read(output / "STOP_WORDS.json"):
        raise ValueError("extraction reuse stop words differ")
    files = {
        "body_raw": "body_raw.sqlite", "title_raw": "title_raw.sqlite",
        "body_extraction": "body_extraction.jsonl.gz",
        "title_extraction": "title_extraction.jsonl.gz",
        "additional_anchors": "additional_anchors.jsonl.gz",
        "tfidf_terms": "TFIDF_TERMS.json",
    }
    validated = [(bound(complete[key], source), output / name) for key, name in files.items()]
    for src, dst in validated:
        shutil.copyfile(src, dst)
    save(output / "EXTRACTION_REUSE.json", dict(
        source_complete=file_record(source / "COMPLETE.json"),
        raw_files=list(files.values()), screening_recomputed=True,
        semantic_parse_calls=0, source_supply_modified=False,
    ))


def build_supply(
    *,
    docs,
    fit_indices,
    truth,
    negative,
    labels,
    row_ids,
    metadata,
    cluster_assignments,
    input_binding,
    output,
    min_support=5,
    body_quota=48,
    title_quota=16,
    label_stride=36,
    spacy_model="en_core_web_sm",
    title_field="title",
    screening="entropy",
    max_entropy=None,
    tfidf_terms=128,
    deduplicate_masks=True,
    exclude_constant_masks=True,
    parse_workers=1,
    reuse_extraction=None,
    patterns_per_cluster=512,
    min_support_fraction=0.01,
    max_fit_fraction=0.9,
    per_label_patterns=None,
):
    """Supply finite predicates from complete fit text/answers only.

    The caller must authenticate the frozen known-train subset. This API accepts
    only that subset, never whole unknown targets or scoring rows. Missing titles
    stay empty; unused title quota is not transferred to body candidates.
    """
    start = time.monotonic()
    out = Path(output).resolve()
    fit = np.asarray(fit_indices)
    y = np.asarray(truth)
    neg = np.asarray(negative)
    clusters = np.asarray(cluster_assignments)
    if (
        not docs
        or len(docs) != len(row_ids)
        or len(docs) != len(metadata)
        or fit.shape != (len(docs),)
        or fit.dtype.kind not in "iu"
        or len(set(fit.tolist())) != len(fit)
        or (fit < 0).any()
        or len(set(row_ids)) != len(row_ids)
        or len(set(labels)) != len(labels)
        or not labels
        or y.dtype != bool
        or neg.dtype != bool
        or y.shape != (len(docs), len(labels))
        or neg.shape != y.shape
        or not np.all(y ^ neg)
    ):
        raise ValueError("complete unique known-fit text and signed label axes required")
    if any(not isinstance(d, Document) or d.lbl for d in docs):
        raise ValueError("unlabeled Document texts required; roots are separate")
    if (
        clusters.shape != (len(docs),)
        or clusters.dtype.kind not in "iu"
        or set(clusters.tolist()) != set(range(int(clusters.max()) + 1))
    ):
        raise ValueError("nonempty contiguous fit-only cluster axis required")
    n_clusters = int(clusters.max()) + 1
    if (
        any(type(v) is not int or v < 1 for v in (min_support, label_stride))
        or (body_quota is not None and (type(body_quota) is not int or body_quota < 1))
        or (title_quota is not None and (type(title_quota) is not int or title_quota < 0))
    ):
        raise ValueError(
            "positive support/label stride; positive body quota and nonnegative title quota, or None for all"
        )
    if not isinstance(input_binding, dict) or not input_binding:
        raise ValueError("frozen fit and clustering identity required")
    if not isinstance(title_field, str) or not title_field:
        raise ValueError("explicit nonempty title metadata field required")
    if screening not in SCREENING_POLICIES:
        raise ValueError("screening must be entropy or the explicit mcc ablation")
    entropy_limit(y, max_entropy)
    if screening != "entropy" and max_entropy is not None:
        raise ValueError("max_entropy applies only to entropy screening")
    discriminative = screening == "discriminative"
    if discriminative and (
        type(patterns_per_cluster) is not int or patterns_per_cluster < 1
        or not 0 <= min_support_fraction < 1 or not 0 < max_fit_fraction <= 1
        or (per_label_patterns is not None and (type(per_label_patterns) is not int or per_label_patterns < 1))
        or not deduplicate_masks or not exclude_constant_masks
    ):
        raise ValueError("discriminative screening requires a positive budget, fractions in range, mask dedup and constant exclusion")
    if type(deduplicate_masks) is not bool or type(exclude_constant_masks) is not bool:
        raise ValueError("explicit Boolean mask filtering controls required")
    if screening == "mcc" and not deduplicate_masks:
        raise ValueError(
            "disabling mask deduplication requires entropy screening; historical MCC stays unchanged"
        )
    if tfidf_terms is not None and (type(tfidf_terms) is not int or tfidf_terms < 1):
        raise ValueError("positive fit TF-IDF term budget or None for all fit vocabulary required")
    if type(parse_workers) is not int or parse_workers < 1:
        raise ValueError("positive parse_workers required")
    canonical(input_binding)
    if out.exists():
        raise FileExistsError("new primitive supply namespace required")
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS

    nlp = spacy.load(spacy_model) if reuse_extraction is None else None
    package = importlib.import_module(spacy_model)
    modelroot = Path(package.__file__).parent
    stops = set(STOP_WORDS)
    out.mkdir(parents=True, exist_ok=False)
    policy = dict(
        schema="loris_rich_fit_compound_primitive_supply_v2",
        inputs=input_binding,
        fit_documents=len(fit),
        labels=list(labels),
        clusters=n_clusters,
        title_field=title_field,
        missing_title_stays_empty=True,
        support=dict(
            min_parse_documents=min_support,
            min_actual_cluster_documents=min_support,
            constant_masks_excluded=exclude_constant_masks,
            zero_support_excluded=True,
        ),
        quotas={
            "cnt": {f: body_quota for f in FAMILIES},
            "ttl": {f: title_quota for f in FAMILIES},
        },
        unused_quota_transfer=False,
        max_per_cluster=(
            None if body_quota is None or title_quota is None else 4 * (body_quota + title_quota)
        ),
        quota_none_means="all statistically eligible inventory; no quota truncation",
        fit_mask_deduplication=deduplicate_masks,
        fit_mask_equivalence_is_semantic_equivalence=False,
        screening=screening,
        discriminative=(dict(
            patterns_per_cluster=patterns_per_cluster,
            min_cluster_support="max(min_support, ceil(min_support_fraction * cluster size))",
            min_support_fraction=min_support_fraction,
            max_fit_fraction=max_fit_fraction,
            per_label_patterns=per_label_patterns if per_label_patterns is not None else "max(2, min(32, budget // (2 * labels with cluster positives)))",
            score="max over labels with cluster positives of binary mutual information(fit-wide mask, fit label)",
            deduplication="identical fit-wide masks; first by attr cnt<ttl, family Match<Cooccur<Before<Freq, key",
            family_share_cap=0.5,
            frequency_partition="partition with maximum score among supported non-constant partitions",
            source=file_record(Path(__file__).with_name("pattern_selection.py")),
        ) if discriminative else None),
        entropy=dict(
            formula="-sum_tau P(tau|pattern)*log(P(tau|pattern))",
            label_fractions_not_renormalized=True,
            log_base="e",
            max_entropy=max_entropy,
            default_ceiling="cluster_unconditional_label_entropy",
            comparison="<=",
            cutoff_is_operational_choice_not_paper_constant=True,
        ),
        ranking=(
            "per-label then global binary mutual information within a per-cluster budget; final selection by validation F1"
            if screening == "discriminative"
            else "entropy ascending, coverage descending, stable pattern key; final selection by validation F1"
            if screening == "entropy"
            else "historical absolute signed MCC with uncovered-fit complementarity first ceil(.75*quota), then coverage"
        ),
        label_rotation=dict(
            stride=label_stride,
            formula="((cluster*8+attribute_index*4+family_index)*stride)%label_count",
        ),
        extraction="semantic noun/entity anchors, fit TF-IDF terms, number spans, verb phrases and title ngrams",
        tfidf_terms_per_cluster=tfidf_terms,
        frequency="integer 0..max full-fit count+1; six operators retained; ranking uses exact unique supported partitions",
        spacy_version=spacy.__version__,
        spacy_model=spacy_model,
        model_files=[
            file_record(p)
            for p in sorted(modelroot.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        ],
        standalone_Text_F1_gate=False,
        universal_zero_error_gate=False,
        dev_truth_consulted=False,
        dev_text_for_extraction=False,
        test_access=False,
        hidden_train_truth_access=False,
        BO_calls=0,
        Chase_calls=0,
        model_calls=0,
        source=file_record(__file__),
        extraction_source_sha256=extraction_source_digest(Path(__file__).read_text()),
        execution=dict(parse_workers=parse_workers, mcc_engine="exact_bitset_lazy_heap",
                       extraction_reuse=None if reuse_extraction is None else str(Path(reuse_extraction).resolve())),
    )
    save(out / "PROTOCOL.json", policy)
    save(out / "STOP_WORDS.json", {"spacy_version": spacy.__version__, "words": sorted(stops)})
    save(
        out / "FIT_IDENTITY.json",
        dict(
            fit_indices=fit.tolist(),
            fit_row_ids=list(row_ids),
            labels=list(labels),
            positive=array_identity(y),
            negative=array_identity(neg),
            texts={a: digest([d.get_attr(a) for d in docs]) for a in ("cnt", "ttl")},
            semantic_metadata_fields=input_binding.get("semantic_metadata_fields"),
            clusters=array_identity(clusters),
            dev_truth_consulted=False,
        ),
    )
    resources = []
    try:
        if reuse_extraction is None:
            body = body_domain(docs, fit, row_ids, clusters, stops, nlp, out, parse_workers)
            resources.append(body)
            title = title_domain(docs, clusters, stops, out)
            resources.append(title)
            add_paper_anchors(docs, clusters, body, out, tfidf_terms)
        else:
            _reuse_extraction(reuse_extraction, out, policy)
            body = sqlite3.connect(str(out / "body_raw.sqlite"))
            resources.append(body)
            title = sqlite3.connect(str(out / "title_raw.sqlite"))
            resources.append(title)
        newstore = CompleteRegexStore(docs, out / "exact_fit_regex", regex_index="literal")
        resources.append(newstore)
        memo = {}

        def counts(attr, d):
            key = digest([attr, d["raw"], d["flags"]])
            if key not in memo:
                value = newstore.counts_first(attr, d)
                if value.shape != (2, len(fit)) or value.dtype != np.int64:
                    raise ValueError("count cache axis differs")
                memo[key] = value
            return memo[key]

        db = sqlite3.connect(str(out / "candidates.sqlite"))
        resources.append(db)
        db.executescript(
            "CREATE TABLE candidates(cluster INTEGER,attr TEXT,family TEXT,key TEXT,definition TEXT,parse_df INTEGER,support INTEGER,positive_counts TEXT,mcc TEXT,rank_parameter TEXT,reason TEXT,selected_rank INTEGER,detail TEXT,PRIMARY KEY(cluster,attr,family,key)); CREATE TABLE frequency_partitions(cluster INTEGER,attr TEXT,key TEXT,op TEXT,eta INTEGER,support INTEGER,positive_counts TEXT,mcc TEXT,reason TEXT);"
        )
        db.execute(
            "CREATE TABLE entropy_screening(cluster INTEGER,attr TEXT,key TEXT,op TEXT,eta INTEGER,entropy REAL,max_entropy REAL,accepted INTEGER)"
        )
        selected_specs = {}
        metadata_fields = None
        summaries = []
        for cl in range(n_clusters):
            local = np.flatnonzero(clusters == cl)
            truth = y[local]
            chosen_all = []
            ceiling = entropy_limit(truth, max_entropy)
            cluster_support = (
                max(min_support, int(math.ceil(min_support_fraction * len(local))))
                if discriminative else min_support
            )
            cluster_labels = np.flatnonzero(truth.any(axis=0))
            cluster_pool = []

            def fit_score(fit_mask):
                if not len(cluster_labels):
                    return 0.0
                return float(pattern_selection.label_scores([fit_mask], y, cluster_labels).max())

            def fit_coverage_ok(fit_mask):
                return not discriminative or int(fit_mask.sum()) <= max_fit_fraction * len(fit)
            for attr_index, (attr, domain) in enumerate([("cnt", body), ("ttl", title)]):
                anchors = {
                    k: (json.loads(d), df)
                    for k, d, df in domain.execute(
                        "SELECT key,definition,parse_df FROM anchors WHERE cluster=? ORDER BY key",
                        (cl,),
                    )
                }
                validity = (
                    {
                        k: bool(v)
                        for k, v in body.execute(
                            "SELECT key,valid FROM anchor_validity WHERE cluster=?", (cl,)
                        )
                    }
                    if attr == "cnt"
                    else {k: True for k in anchors}
                )
                candidates = []
                for k, (d, df) in anchors.items():
                    for family in ("Match", "Freq"):
                        candidates.append((family, k, None, df))
                candidates.extend(
                    domain.execute(
                        "SELECT family,left_key,right_key,parse_df FROM pairs WHERE cluster=? ORDER BY family,left_key,right_key",
                        (cl,),
                    )
                )
                eligible = {f: [] for f in FAMILIES}
                counter = collections.Counter()
                for family, left, right, parse_df in candidates:
                    definition = anchors[left][0]
                    spec = _spec(
                        family,
                        attr,
                        definition,
                        anchors[right][0] if right else None,
                        op=">=" if family == "Freq" else None,
                        eta=2 if family == "Freq" else None,
                    )
                    reason = None
                    support = None
                    corr = None
                    tp = None
                    rank_param = None
                    mask = None
                    entropy = None
                    if not validity[left] or (right and not validity[right]):
                        reason = "invalid_content_anchor"
                    elif parse_df < cluster_support:
                        reason = "parse_support_below_minimum"
                    else:
                        c = counts(attr, definition)
                        present = newstore.presence[attr]
                        mask = (c[0] > 0) & present
                        if family in ("Cooccur", "Before"):
                            rc = counts(attr, anchors[right][0])
                            mask &= rc[0] > 0
                            if family == "Before":
                                mask &= c[1] < rc[1]
                        elif family == "Freq":
                            max_count = int(c[0].max())
                            spec = PatternSpec(
                                "FreqDomain:"
                                + digest([attr, definition, 0, max_count + 1, list(FREQUENCY_OPS)]),
                                spec.predicate,
                                ParameterDomain(low=0, high=max_count + 1, integer=True),
                                FREQUENCY_OPS,
                            )
                            best = None
                            partitions = set()
                            cv = c[0][local]
                            # All six operators reduce exactly to these unique count partitions.
                            for op, fn in [
                                (">=", np.greater_equal),
                                ("<=", np.less_equal),
                                ("==", np.equal),
                                ("!=", np.not_equal),
                            ]:
                                for eta in sorted(set([0, max_count + 1] + cv.tolist())):
                                    pm = fn(cv, eta) & present[local]
                                    bits = np.packbits(pm).tobytes()
                                    if bits in partitions:
                                        continue
                                    partitions.add(bits)
                                    ps = int(pm.sum())
                                    pc, pt = correlation(pm, truth)
                                    why = (
                                        "eligible"
                                        if cluster_support <= ps
                                        and (not exclude_constant_masks or ps < len(local))
                                        else "below_support_or_constant"
                                    )
                                    fit_partition = None
                                    if discriminative and why == "eligible":
                                        fit_partition = fn(c[0], eta) & present
                                        if not fit_coverage_ok(fit_partition):
                                            why = "fit_coverage_above_maximum"
                                    pe = label_entropy(pm, truth)
                                    if (
                                        why == "eligible"
                                        and screening == "entropy"
                                        and pe > ceiling
                                    ):
                                        why = "entropy_above_maximum"
                                    db.execute(
                                        "INSERT INTO entropy_screening VALUES(?,?,?,?,?,?,?,?)",
                                        (
                                            cl,
                                            attr,
                                            spec.key,
                                            op,
                                            int(eta),
                                            pe,
                                            ceiling,
                                            int(why == "eligible"),
                                        ),
                                    )
                                    db.execute(
                                        "INSERT INTO frequency_partitions VALUES(?,?,?,?,?,?,?,?,?)",
                                        (
                                            cl,
                                            attr,
                                            spec.key,
                                            op,
                                            int(eta),
                                            ps,
                                            canonical(pt.tolist()),
                                            canonical(pc.tolist()),
                                            why,
                                        ),
                                    )
                                    if why == "eligible":
                                        rank = (
                                            (
                                                -fit_score(fit_partition)
                                                if discriminative
                                                else pe
                                                if screening == "entropy"
                                                else -float(np.max(abs(pc)))
                                            ),
                                            -ps,
                                            op,
                                            int(eta),
                                        )
                                        if best is None or rank < best[0]:
                                            best = (rank, op, eta, pm, pc, pt)
                            if best is None:
                                reason = (
                                    "no_supported_nonconstant_frequency_partition"
                                    if exclude_constant_masks
                                    else "no_supported_frequency_partition"
                                )
                                mask = None
                            else:
                                _, op, eta, pm, corr, tp = best
                                rank_param = {"op": op, "eta": int(eta)}
                                mask = {
                                    "<=": np.less_equal,
                                    ">=": np.greater_equal,
                                    "==": np.equal,
                                    "!=": np.not_equal,
                                }[op](c[0], eta) & present
                                spec = replace(
                                    spec, predicate=replace(spec.predicate, op=op, eta=int(eta))
                                )
                        if mask is not None:
                            support = int(mask[local].sum())
                            if corr is None:
                                corr, tp = correlation(mask[local], truth)
                            entropy = label_entropy(mask[local], truth)
                            reason = (
                                "actual_support_below_minimum"
                                if support < cluster_support
                                else (
                                    "constant_mask"
                                    if exclude_constant_masks and support == len(local)
                                    else "eligible"
                                )
                            )
                            if reason == "eligible" and not fit_coverage_ok(mask):
                                reason = "fit_coverage_above_maximum"
                            if (
                                reason == "eligible"
                                and screening == "entropy"
                                and entropy > ceiling
                            ):
                                reason = "entropy_above_maximum"
                            if family != "Freq":
                                db.execute(
                                    "INSERT INTO entropy_screening VALUES(?,?,?,?,?,?,?,?)",
                                    (
                                        cl,
                                        attr,
                                        spec.key,
                                        None,
                                        None,
                                        entropy,
                                        ceiling,
                                        int(reason == "eligible"),
                                    ),
                                )
                    counter[(family, reason)] += 1
                    db.execute(
                        "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            cl,
                            attr,
                            family,
                            spec.key,
                            canonical(spec.to_dict()),
                            int(parse_df),
                            support,
                            canonical(tp.tolist()) if tp is not None else None,
                            canonical(corr.tolist()) if corr is not None else None,
                            canonical(rank_param) if rank_param else None,
                            reason,
                            None,
                            None,
                        ),
                    )
                    if reason == "eligible":
                        eligible[family].append(
                            {
                                "spec": spec,
                                "mask": mask[local].copy(),
                                "all_fit_mask": mask.copy(),
                                "support": support,
                                "corr": corr,
                                "tp": tp,
                                "entropy": entropy,
                            }
                        )
                if discriminative:
                    for family in FAMILIES:
                        cluster_pool.extend(dict(item, family=family, attr=attr)
                                            for item in eligible[family])
                        summaries.append({
                            "cluster": cl, "attr": attr, "family": family,
                            "raw": sum(n for (f, r), n in counter.items() if f == family),
                            "eligible_before_diversity": len(eligible[family]),
                            "screening": screening,
                        })
                for family_index, family in enumerate(FAMILIES if not discriminative else ()):
                    quota = body_quota if attr == "cnt" else title_quota
                    offset = ((cl * 8 + attr_index * 4 + family_index) * label_stride) % len(labels)
                    axis = np.roll(np.arange(len(labels)), -offset)
                    originals = {item["spec"].key: item for item in eligible[family]}
                    ranking = [dict(item, corr=item["corr"][axis]) for item in eligible[family]]
                    ranked_chosen, redundant = (
                        select_entropy_group(ranking, quota, deduplicate_masks=deduplicate_masks)
                        if screening == "entropy"
                        else select_group(
                            ranking, len(ranking) if quota is None else quota, truth[:, axis]
                        )
                    )
                    chosen = [
                        (
                            originals[item["spec"].key],
                            method,
                            int(axis[label]) if label is not None else None,
                            sign,
                        )
                        for item, method, label, sign in ranked_chosen
                    ]
                    for item, representative in redundant:
                        db.execute(
                            "UPDATE candidates SET reason=?,detail=? WHERE cluster=? AND attr=? AND family=? AND key=?",
                            (
                                "fit_mask_diversity_prefilter",
                                canonical(
                                    {
                                        "representative": representative,
                                        "not_semantic_equivalence": True,
                                    }
                                ),
                                cl,
                                attr,
                                family,
                                item["spec"].key,
                            ),
                        )
                    for rank, (item, method, label, sign) in enumerate(chosen):
                        if metadata_fields is None:
                            metadata_fields = tuple(sorted({key for m in metadata for key in m}))
                        detail = {
                            "selection": method,
                            "target_label_for_ranking": (
                                labels[label] if label is not None else None
                            ),
                            "association_sign": sign,
                            "actual_fit_documents": item["support"],
                            "attribute_group_support": {
                                name: {
                                    "distinct_nonempty_groups": len(
                                        {
                                            str(metadata[int(i)].get(name, ""))
                                            for i in local[item["mask"]]
                                            if metadata[int(i)].get(name) not in ("", None)
                                        }
                                    ),
                                    "missing_documents": sum(
                                        metadata[int(i)].get(name) in ("", None)
                                        for i in local[item["mask"]]
                                    ),
                                }
                                for name in metadata_fields
                            },
                            "positive_counts": item["tp"].tolist(),
                            "ranking_predicate": item["spec"].to_dict()["predicate"],
                            "no_assigned_conclusion": True,
                        }
                        detail.update(
                            entropy=item["entropy"], entropy_ceiling=ceiling, screening=screening
                        )
                        db.execute(
                            "UPDATE candidates SET reason=?,selected_rank=?,detail=? WHERE cluster=? AND attr=? AND family=? AND key=?",
                            (
                                "selected",
                                rank,
                                canonical(detail),
                                cl,
                                attr,
                                family,
                                item["spec"].key,
                            ),
                        )
                        chosen_all.append((item, detail))
                    db.execute(
                        "UPDATE candidates SET reason=? WHERE cluster=? AND attr=? AND family=? AND reason=?",
                        ("outside_declared_diversity_budget", cl, attr, family, "eligible"),
                    )
                    summaries.append(
                        {
                            "cluster": cl,
                            "attr": attr,
                            "family": family,
                            "raw": sum(n for (f, r), n in counter.items() if f == family),
                            "eligible_before_diversity": len(eligible[family]),
                            "empirical_duplicate_masks": len(redundant),
                            "selected": len(chosen),
                            "quota": quota,
                            "ranking_label_offset": offset if screening == "mcc" else None,
                            "entropy_ceiling": ceiling,
                            "screening": screening,
                        }
                    )
                db.commit()
                newstore.commit()
                print(
                    canonical(
                        {
                            "cluster": cl,
                            "attr": attr,
                            "selected": len(chosen_all),
                            "seconds": round(time.monotonic() - start, 3),
                        }
                    ),
                    flush=True,
                )
            if discriminative:
                chosen, duplicates, outside = pattern_selection.select(
                    cluster_pool, truth=y, labels=cluster_labels, budget=patterns_per_cluster,
                    per_label=per_label_patterns)
                for item, representative in duplicates:
                    db.execute(
                        "UPDATE candidates SET reason=?,detail=? WHERE cluster=? AND attr=? AND family=? AND key=?",
                        ("fit_mask_diversity_prefilter",
                         canonical({"representative": representative, "not_semantic_equivalence": True}),
                         cl, item["attr"], item["family"], item["spec"].key))
                for item in outside:
                    db.execute(
                        "UPDATE candidates SET reason=?,detail=? WHERE cluster=? AND attr=? AND family=? AND key=?",
                        ("outside_declared_diversity_budget", canonical({"score": item["score"]}),
                         cl, item["attr"], item["family"], item["spec"].key))
                if metadata_fields is None:
                    metadata_fields = tuple(sorted({key for m in metadata for key in m}))
                for rank, item in enumerate(chosen):
                    detail = {
                        "selection": item["selection"],
                        "target_label_for_ranking": (labels[item["selection_label"]]
                                                     if item["selection_label"] is not None else None),
                        "score_mutual_information": item["score"],
                        "score_label": labels[item["score_label"]] if item["score_label"] is not None else None,
                        "fit_coverage": int(item["all_fit_mask"].sum()),
                        "association_sign": None,
                        "actual_fit_documents": item["support"],
                        "attribute_group_support": {
                            name: {
                                "distinct_nonempty_groups": len({
                                    str(metadata[int(i)].get(name, "")) for i in local[item["mask"]]
                                    if metadata[int(i)].get(name) not in ("", None)}),
                                "missing_documents": sum(
                                    metadata[int(i)].get(name) in ("", None) for i in local[item["mask"]]),
                            }
                            for name in metadata_fields
                        },
                        "positive_counts": item["tp"].tolist(),
                        "ranking_predicate": item["spec"].to_dict()["predicate"],
                        "no_assigned_conclusion": True,
                        "screening": screening,
                    }
                    db.execute(
                        "UPDATE candidates SET reason=?,selected_rank=?,detail=? WHERE cluster=? AND attr=? AND family=? AND key=?",
                        ("selected", rank, canonical(detail), cl, item["attr"], item["family"], item["spec"].key))
                    chosen_all.append((item, detail))
                summaries.append({
                    "cluster": cl, "screening": screening, "eligible": len(cluster_pool),
                    "empirical_duplicate_masks": len(duplicates), "selected": len(chosen),
                    "budget": patterns_per_cluster, "cluster_support_minimum": cluster_support,
                    "labels_with_cluster_positives": len(cluster_labels),
                    "selected_by_family": dict(collections.Counter(i["family"] for i in chosen)),
                    "selected_by_attr": dict(collections.Counter(i["attr"] for i in chosen)),
                })
                db.commit()
            cldir = out / ("cluster_%02d" % cl)
            cldir.mkdir()
            save(
                cldir / "PATTERNS.json",
                {
                    "patterns": [v["spec"].to_dict() for v, d in chosen_all],
                    "details": [{**d, "key": v["spec"].key} for v, d in chosen_all],
                    "role": "compound_RDL_primitive_supply_not_SignedChase_selected_Pi",
                    "protocol": file_record(out / "PROTOCOL.json"),
                },
            )
            np.savez_compressed(
                cldir / "FIT_MASKS.npz",
                masks=np.asarray([v["all_fit_mask"] for v, d in chosen_all]),
                fit_indices=fit,
            )
            selected_specs[str(cl)] = [v["spec"].to_dict() for v, d in chosen_all]
            # Bound memory: statistics are durable in caches; next cluster may reload.
            memo.clear()

        db.commit()
        newstore.commit()
        save(out / "PATTERNS_BY_CLUSTER.json", selected_specs)
        save(
            out / "SUMMARY.json",
            {
                "groups": summaries,
                "counts": {k: len(v) for k, v in selected_specs.items()},
                "labels": labels,
                "fit_docs": len(fit),
                "seconds": time.monotonic() - start,
            },
        )
        save(
            out / "COMPLETE.json",
            {
                "status": "complete",
                "protocol": file_record(out / "PROTOCOL.json"),
                "patterns": file_record(out / "PATTERNS_BY_CLUSTER.json"),
                "candidate_archive": file_record(out / "candidates.sqlite"),
                "body_raw": file_record(out / "body_raw.sqlite"),
                "title_raw": file_record(out / "title_raw.sqlite"),
                "body_extraction": file_record(out / "body_extraction.jsonl.gz"),
                "title_extraction": file_record(out / "title_extraction.jsonl.gz"),
                "additional_anchors": file_record(out / "additional_anchors.jsonl.gz"),
                "tfidf_terms": file_record(out / "TFIDF_TERMS.json"),
                "counts": {k: len(v) for k, v in selected_specs.items()},
                "seconds": time.monotonic() - start,
                "BO_calls": 0,
                "Chase_calls": 0,
                "model_inference_or_training_calls": 0,
                "hidden_train_or_dev_or_test_truth_consulted": False,
                "source": file_record(__file__),
            },
        )
        return read(out / "COMPLETE.json")
    except BaseException as error:
        save(
            out / "FAILURE.json",
            dict(
                status="failed",
                type=type(error).__name__,
                reason=str(error),
                partial_records_preserved=True,
            ),
        )
        raise
    finally:
        for resource in reversed(resources):
            resource.close()


def build_from_manifest(
    *,
    supervision_manifest,
    cluster_arrays,
    clustering,
    output,
    title_field="title",
    semantic_metadata_fields=None,
    **options,
):
    """Authenticate the fit-only reader and clustering before pattern extraction."""
    from .supervision import load_fit_supervision
    from .upstream import records_to_documents

    data = load_fit_supervision(supervision_manifest)
    definition = read(clustering)
    if (
        definition["fit_indices"] != data["fit_indices"].tolist()
        or definition["fit_row_ids"] != data["fit_row_ids"]
        or definition.get("dev_or_test_used_to_fit_centroids") is not False
    ):
        raise ValueError("clustering must bind the same known-train fit axis")
    with np.load(cluster_arrays, allow_pickle=False) as saved:
        clusters = saved["cluster_assignments"].copy()
        if not np.array_equal(saved["fit_indices"], data["fit_indices"]):
            raise ValueError("cluster array has a different fit axis")
    if definition["assignments"] != array_identity(clusters):
        raise ValueError("clustering receipt/assignment identity differs")
    return build_supply(
        docs=records_to_documents(
            data["docs"], title_field=title_field, semantic_metadata_fields=semantic_metadata_fields
        ),
        fit_indices=data["fit_indices"],
        truth=data["fit_truth"],
        negative=~data["fit_truth"],
        labels=data["labels"],
        row_ids=data["fit_row_ids"],
        metadata=(
            data["metadata"]
            if semantic_metadata_fields is None
            else [
                {key: metadata.get(key, "") for key in semantic_metadata_fields}
                for metadata in data["metadata"]
            ]
        ),
        cluster_assignments=clusters,
        input_binding=dict(
            supervision=data["manifest_record"],
            files_read=data["files_read"],
            clustering=file_record(clustering),
            cluster_arrays=file_record(cluster_arrays),
            semantic_metadata_fields=semantic_metadata_fields,
        ),
        output=output,
        title_field=title_field,
        **options,
    )


def inventory_limit(value):
    """CLI spelling for a finite quota or its complete extracted inventory."""
    if value == "all":
        return None
    try:
        return int(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("expected an integer or 'all'") from error


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--supervision-manifest", required=True)
    cli.add_argument("--cluster-arrays", required=True)
    cli.add_argument("--clustering", required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--min-support", type=int, default=5)
    cli.add_argument(
        "--body-quota",
        type=inventory_limit,
        default=48,
        help="per family per cluster; 'all' retains the complete eligible inventory",
    )
    cli.add_argument(
        "--title-quota",
        type=inventory_limit,
        default=16,
        help="per family or 'all'; missing titles stay empty",
    )
    cli.add_argument(
        "--label-stride", type=int, default=36, help="fixed rotation stride declared before scoring"
    )
    cli.add_argument(
        "--screening",
        choices=SCREENING_POLICIES,
        default="entropy",
        help="paper entropy screen or historical MCC/diversity ablation",
    )
    cli.add_argument(
        "--max-entropy",
        type=float,
        help="entropy ceiling in nats; default is each fit cluster unconditional entropy",
    )
    cli.add_argument(
        "--tfidf-terms",
        type=inventory_limit,
        default=128,
        help="fit TF-IDF terms per cluster or 'all' for the observed fit vocabulary",
    )
    duplicates = cli.add_mutually_exclusive_group()
    duplicates.add_argument(
        "--deduplicate-masks",
        dest="deduplicate_masks",
        action="store_true",
        default=True,
        help="merge equal empirical fit masks within a family (default)",
    )
    duplicates.add_argument(
        "--no-deduplicate-masks",
        dest="deduplicate_masks",
        action="store_false",
        help="keep distinct predicates with equal fit masks; entropy screening required",
    )
    constants = cli.add_mutually_exclusive_group()
    constants.add_argument(
        "--exclude-constant-masks",
        dest="exclude_constant_masks",
        action="store_true",
        default=True,
        help="exclude all-true fit masks (default)",
    )
    constants.add_argument(
        "--no-exclude-constant-masks",
        dest="exclude_constant_masks",
        action="store_false",
        help="retain all-true masks; zero-support candidates remain unsupported",
    )
    cli.add_argument("--patterns-per-cluster", type=int, default=512,
                     help="discriminative screening: per-cluster pattern budget")
    cli.add_argument("--min-support-fraction", type=float, default=0.01,
                     help="discriminative screening: minimum cluster support as a fraction of the cluster")
    cli.add_argument("--max-fit-fraction", type=float, default=0.9,
                     help="discriminative screening: maximum fit-wide coverage fraction")
    cli.add_argument("--per-label-patterns", type=int, default=None,
                     help="discriminative screening: per-label top-k before the global fill (default automatic)")
    cli.add_argument("--spacy-model", default="en_core_web_sm")
    cli.add_argument("--parse-workers", type=int, default=1, help="ordered spaCy worker processes")
    cli.add_argument("--reuse-extraction", help="completed supply with identical fit/text/clusters/extractor; reuse raw parsing only")
    cli.add_argument(
        "--title-field",
        default="title",
        help="actual source metadata field; missing values remain empty",
    )
    cli.add_argument(
        "--semantic-metadata-fields",
        type=json.loads,
        help="same explicit JSON source metadata field list as upstream context; [] is valid",
    )
    result = build_from_manifest(**vars(cli.parse_args(argv)))
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
