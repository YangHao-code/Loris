"""Dataset suitability screen for LORIS text-predicate experiments.

For every dataset with processed ``text,<label…>`` CSVs, this reports whether the
``text`` column is genuine raw text (usable by textual predicates like
Match/Freq/Cooccur/Before) or a tokenized bag of opaque word-IDs (e.g. RCV1's
``w5215 w26534 …``, on which text predicates produce only noise), plus the basic
multi-label shape (docs, labels, cardinality/density, text length).

The goal is to "finally select" the usable datasets on evidence. RCV1 is the
worked example of a fail (tokenized → ``has_text=False``).

CLI
---
    python -m loris.analysis.dataset_screen --out experiments/analysis/dataset_screen

By default it screens every ``data/<name>/processed`` directory found under the
repo root; restrict with ``--datasets a,b,c``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]  # repo root (…/Loris)

# rcv1-style token: a 'w' followed by digits (the classic stemmed-token-id form)
_WORDID_RE = re.compile(r"^w\d+$")
_ALPHA_RE = re.compile(r"[A-Za-z]")

# Heuristic thresholds for the tokenized-vs-raw decision.
_WORDID_FRAC_FAIL = 0.30      # >30% of tokens are wNNNNN → tokenized
_MIN_DOCS = 200              # too few docs to be a serious benchmark
_MIN_LABELS = 3


def _sample_texts(df: pd.DataFrame, n: int = 3000) -> List[str]:
    col = df.columns[0]
    texts = df[col].astype(str)
    if len(texts) > n:
        texts = texts.iloc[:n]
    return texts.tolist()


def _text_diagnostics(texts: List[str]) -> Dict[str, Any]:
    wordid_hits = 0
    total_tokens = 0
    alpha_tokens = 0
    word_counts: List[int] = []
    char_counts: List[int] = []
    for t in texts:
        toks = t.split()
        word_counts.append(len(toks))
        char_counts.append(len(t))
        for tok in toks:
            total_tokens += 1
            if _WORDID_RE.match(tok):
                wordid_hits += 1
            elif _ALPHA_RE.search(tok):
                alpha_tokens += 1
    wordid_frac = wordid_hits / max(1, total_tokens)
    alpha_frac = alpha_tokens / max(1, total_tokens)
    return {
        "wordid_fraction": round(wordid_frac, 4),
        "alpha_token_fraction": round(alpha_frac, 4),
        "avg_words": round(statistics.fmean(word_counts), 1) if word_counts else 0,
        "median_words": statistics.median(word_counts) if word_counts else 0,
        "avg_chars": round(statistics.fmean(char_counts), 1) if char_counts else 0,
        "sample_text_head": (texts[0][:160] if texts else ""),
    }


def screen_csv(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)
    label_cols = list(df.columns[1:])
    n_labels = len(label_cols)
    n_docs = len(df)
    # label cardinality / density from the 0/1 matrix
    card = density = 0.0
    empty_docs = 0
    if n_labels and n_docs:
        Y = df[label_cols].fillna(0)
        # coerce to numeric 0/1
        Y = (Y.apply(pd.to_numeric, errors="coerce").fillna(0) > 0).astype(int)
        per_doc = Y.sum(axis=1)
        card = float(per_doc.mean())
        density = card / n_labels
        empty_docs = int((per_doc == 0).sum())
    diag = _text_diagnostics(_sample_texts(df))
    return {
        "path": str(path),
        "n_docs": n_docs,
        "n_labels": n_labels,
        "label_cardinality": round(card, 3),
        "label_density": round(density, 5),
        "empty_label_docs": empty_docs,
        **diag,
    }


def screen_dataset(name: str, processed_dir: Path) -> Dict[str, Any]:
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"
    entry: Dict[str, Any] = {"dataset": name, "processed_dir": str(processed_dir)}
    if not train_csv.exists():
        entry["status"] = "missing"
        return entry
    entry["train"] = screen_csv(train_csv)
    if test_csv.exists():
        entry["test"] = screen_csv(test_csv)

    tr = entry["train"]
    # Verdict: raw-text-suitable unless tokenized / too small.
    reasons: List[str] = []
    tokenized = tr["wordid_fraction"] >= _WORDID_FRAC_FAIL or tr["alpha_token_fraction"] < 0.4
    if tokenized:
        reasons.append(
            f"text looks tokenized (wordid_frac={tr['wordid_fraction']}, "
            f"alpha_frac={tr['alpha_token_fraction']})")
    if tr["n_docs"] < _MIN_DOCS:
        reasons.append(f"too few docs ({tr['n_docs']})")
    if tr["n_labels"] < _MIN_LABELS:
        reasons.append(f"too few labels ({tr['n_labels']})")
    entry["has_raw_text"] = not tokenized
    entry["text_predicate_suitable"] = len(reasons) == 0
    entry["verdict"] = "PASS" if not reasons else "FAIL"
    entry["reasons"] = reasons
    return entry


def _discover_datasets(only: Optional[List[str]]) -> List[tuple]:
    out = []
    data_root = _ROOT / "data"
    for d in sorted(data_root.iterdir()) if data_root.exists() else []:
        if not d.is_dir():
            continue
        if only and d.name not in only:
            continue
        processed = d / "processed"
        if processed.exists():
            out.append((d.name, processed))
    return out


def render_markdown(results: List[Dict[str, Any]]) -> str:
    L: List[str] = ["# LORIS dataset suitability screen\n"]
    L.append("| dataset | verdict | raw text | docs (tr/te) | labels | cardinality | "
             "avg words | wordid frac |")
    L.append("| :-- | :-- | :-- | :-- | --: | --: | --: | --: |")
    for r in results:
        if r.get("status") == "missing":
            L.append(f"| {r['dataset']} | MISSING | — | — | — | — | — | — |")
            continue
        tr = r["train"]
        te = r.get("test", {})
        L.append(
            f"| {r['dataset']} | **{r['verdict']}** | "
            f"{'yes' if r['has_raw_text'] else 'NO'} | "
            f"{tr['n_docs']}/{te.get('n_docs', '—')} | {tr['n_labels']} | "
            f"{tr['label_cardinality']} | {tr['avg_words']} | {tr['wordid_fraction']} |")
    L.append("")
    passes = [r["dataset"] for r in results if r.get("verdict") == "PASS"]
    fails = [f"{r['dataset']} ({'; '.join(r.get('reasons', []))})"
             for r in results if r.get("verdict") == "FAIL"]
    L.append("## Recommended set (text-predicate suitable)\n")
    L.append(", ".join(passes) if passes else "_none_")
    L.append("\n## Excluded\n")
    L.extend(f"- {f}" for f in fails) if fails else L.append("_none_")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Screen datasets for text-predicate suitability.")
    ap.add_argument("--datasets", default="", help="comma-separated subset; default = all found")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.json and <out>.md")
    args = ap.parse_args(argv)

    only = [s for s in args.datasets.split(",") if s.strip()] or None
    datasets = _discover_datasets(only)
    if not datasets:
        print("No processed datasets found under data/.")
        return 1

    results = [screen_dataset(name, processed) for name, processed in datasets]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    out.with_suffix(".md").write_text(render_markdown(results))

    for r in results:
        if r.get("status") == "missing":
            print(f"  {r['dataset']:16s} MISSING")
        else:
            print(f"  {r['dataset']:16s} {r['verdict']:4s} "
                  f"docs={r['train']['n_docs']:>6} labels={r['train']['n_labels']:>3} "
                  f"wordid_frac={r['train']['wordid_fraction']} "
                  f"{'' if not r['reasons'] else '— ' + '; '.join(r['reasons'])}")
    print(f"Wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
