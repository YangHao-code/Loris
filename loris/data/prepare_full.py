"""Full-scale (million-scale) dataset preparation.

Downloads the FULL versions of the candidate datasets (auth-free mirrors) and
writes them to ``data/<ds>_full/processed/{train,test}.csv`` in the standard
``text,<label…>`` multi-hot format, so they register + load like any other
LORIS dataset. Kept separate from the small validated slices in ``prepare.py``.

Sources (no credentials required):
  * arxiv_full     — HF ``ppxscal/arxiv-metadata-oai-snapshot`` (~2.7M papers)
  * hupd_full      — HF ``HUPD/hupd`` per-year tars (~4.5M patents; all years)
  * pubmed_full    — HF ``Tellurio/PubMed-MultiLabel-MeSH`` (title+abstract → 14 MeSH roots)
  * goodreads_full — UCSD Book Graph books + genres (~2.3M books)

CLI: ``python -m loris.data.prepare_full --dataset arxiv_full [--limit N] [--years all]``
Use ``--limit`` for a quick smoke before the full pull.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from loris.data.prepare import _ROOT, _save_multi_hot_csv, _download_file

_SEED = 42


def _split_save(texts, labels_list, all_labels, out_dir: Path, test_frac=0.08):
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_t, te_t, tr_l, te_l = train_test_split(
        texts, labels_list, test_size=test_frac, random_state=_SEED)
    _save_multi_hot_csv(tr_t, tr_l, all_labels, out_dir / "train.csv")
    _save_multi_hot_csv(te_t, te_l, all_labels, out_dir / "test.csv")


def _curl(url: str, dest: Path) -> Path:
    """Resumable download via curl (robust for multi-GB files; urlretrieve is not)."""
    import subprocess
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        # try to resume/complete; -C - is a no-op if already complete
        pass
    rc = subprocess.call(["curl", "-fL", "-C", "-", "--retry", "8", "--retry-delay", "5",
                          "-o", str(dest), url])
    if rc != 0 or not (dest.exists() and dest.stat().st_size > 0):
        raise RuntimeError(f"curl failed ({rc}) for {url}")
    return dest


def _top_labels(labels_list: List[List[str]], top_n: int) -> List[str]:
    freq = Counter(l for ls in labels_list for l in ls)
    return [l for l, _ in freq.most_common(top_n)]


def _restrict(texts, labels_list, keep_labels):
    keep = set(keep_labels)
    ft, fl = [], []
    for t, ls in zip(texts, labels_list):
        ls2 = [l for l in ls if l in keep]
        if t and ls2:
            ft.append(t)
            fl.append(ls2)
    return ft, fl


# ── arXiv full ────────────────────────────────────────────────────────────────
def prepare_arxiv_full(limit: int = 0, top_n: int = 100) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    repo = "ppxscal/arxiv-metadata-oai-snapshot"
    raw = _ROOT / "data" / "arxiv_full" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    shards = [s.rfilename for s in api.dataset_info(repo).siblings
              if s.rfilename.endswith(".parquet")]
    texts, labels_list = [], []
    for sh in sorted(shards):
        p = hf_hub_download(repo, sh, repo_type="dataset", local_dir=str(raw))
        df = pd.read_parquet(p)
        tcol = "title" if "title" in df.columns else None
        acol = "abstract" if "abstract" in df.columns else None
        ccol = "categories" if "categories" in df.columns else None
        for _, r in df.iterrows():
            title = str(r.get(tcol) or "").strip()
            abst = str(r.get(acol) or "").strip()
            text = (title + ". " + abst).strip() if title else abst
            cats = str(r.get(ccol) or "").split()
            if text and cats:
                texts.append(text)
                labels_list.append(cats)
            if limit and len(texts) >= limit:
                break
        os.remove(p)
        if limit and len(texts) >= limit:
            break
    keep = _top_labels(labels_list, top_n)
    texts, labels_list = _restrict(texts, labels_list, keep)
    _split_save(texts, labels_list, keep, _ROOT / "data" / "arxiv_full" / "processed")
    print(f"[arxiv_full] {len(texts)} docs × {len(keep)} labels", flush=True)


# ── HUPD full (all years) ───────────────────────────────────────────────────────
def prepare_hupd_full(years: str = "all", limit: int = 0, top_n: int = 200) -> None:
    import tarfile
    from huggingface_hub import hf_hub_download
    repo = "HUPD/hupd"
    all_years = [str(y) for y in range(2004, 2019)]
    yrs = all_years if years == "all" else [y.strip() for y in years.split(",")]
    raw = _ROOT / "data" / "hupd_full" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    texts, labels_list = [], []
    for y in yrs:
        try:
            tar_path = hf_hub_download(repo, f"data/{y}.tar.gz", repo_type="dataset",
                                       local_dir=str(raw))
        except Exception as e:
            print(f"[hupd_full] year {y} download failed: {e}", flush=True)
            continue
        n0 = len(texts)
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                if not m.name.endswith(".json"):
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                try:
                    d = json.load(fh)
                except Exception:
                    continue
                title = (d.get("title") or "").strip()
                abstract = (d.get("abstract") or "").strip()
                text = (title + ". " + abstract).strip() if title else abstract
                ipc = d.get("ipcr_labels") or []
                subs = sorted({c[:4] for c in ipc if c and len(c) >= 4})
                if text and subs:
                    texts.append(text)
                    labels_list.append(subs)
                if limit and len(texts) >= limit:
                    break
        os.remove(tar_path)  # bound disk: one year-tar at a time
        print(f"[hupd_full] year {y}: +{len(texts)-n0} patents (total {len(texts)})", flush=True)
        if limit and len(texts) >= limit:
            break
    keep = _top_labels(labels_list, top_n)
    texts, labels_list = _restrict(texts, labels_list, keep)
    _split_save(texts, labels_list, keep, _ROOT / "data" / "hupd_full" / "processed")
    print(f"[hupd_full] {len(texts)} docs × {len(keep)} IPC-subclass labels", flush=True)


# ── PubMed full (Tellurio MeSH) ─────────────────────────────────────────────────
_MESH_ROOTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "L", "M", "N", "Z"]


def prepare_pubmed_full(limit: int = 0) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    repo = "Tellurio/PubMed-MultiLabel-MeSH"
    raw = _ROOT / "data" / "pubmed_full" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    files = [s.rfilename for s in HfApi().dataset_info(repo).siblings
             if s.rfilename.endswith(".parquet")]

    def rows(df):
        texts, labels_list = [], []
        for _, r in df.iterrows():
            title = str(r.get("title") or "").strip()
            abstract = str(r.get("abstract") or "").strip()
            text = (title + ". " + abstract).strip() if title else abstract
            roots = r.get("mesh_roots")
            labs = []
            if isinstance(roots, dict):
                labs = [k for k in _MESH_ROOTS if int(roots.get(k, 0)) > 0]
            if text and labs:
                texts.append(text)
                labels_list.append(labs)
            if limit and len(texts) >= limit:
                break
        return texts, labels_list

    out_dir = _ROOT / "data" / "pubmed_full" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        fs = [f for f in files if f"/{split}-" in f or f.startswith(f"data/{split}")]
        tx, lb = [], []
        for f in fs:
            p = hf_hub_download(repo, f, repo_type="dataset", local_dir=str(raw))
            a, b = rows(pd.read_parquet(p))
            tx += a
            lb += b
        _save_multi_hot_csv(tx, lb, _MESH_ROOTS, out_dir / f"{split}.csv")
        print(f"[pubmed_full] {split}: {len(tx)} docs", flush=True)


# ── Goodreads full (UCSD) ───────────────────────────────────────────────────────
def prepare_goodreads_full(limit: int = 0, top_n: int = 20) -> None:
    base = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
    raw = _ROOT / "data" / "goodreads_full" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    books_gz = _curl(f"{base}/goodreads_books.json.gz", raw / "books.json.gz")
    genres_gz = _curl(f"{base}/goodreads_book_genres_initial.json.gz",
                      raw / "genres.json.gz")

    # book_id -> list of genre labels
    genres = {}
    with gzip.open(genres_gz, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            g = d.get("genres") or {}
            if g:
                genres[str(d.get("book_id"))] = list(g.keys())

    texts, labels_list = [], []
    with gzip.open(books_gz, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            bid = str(d.get("book_id"))
            labs = genres.get(bid)
            if not labs:
                continue
            title = str(d.get("title") or "").strip()
            desc = str(d.get("description") or "").strip()
            text = (title + ". " + desc).strip() if title else desc
            if text and labs:
                texts.append(text)
                labels_list.append(labs)
            if limit and len(texts) >= limit:
                break
    keep = _top_labels(labels_list, top_n)
    texts, labels_list = _restrict(texts, labels_list, keep)
    _split_save(texts, labels_list, keep, _ROOT / "data" / "goodreads_full" / "processed")
    print(f"[goodreads_full] {len(texts)} docs × {len(keep)} genres", flush=True)


_PREPARERS = {
    "arxiv_full": prepare_arxiv_full,
    "hupd_full": prepare_hupd_full,
    "pubmed_full": prepare_pubmed_full,
    "goodreads_full": prepare_goodreads_full,
}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Full-scale dataset preparation.")
    ap.add_argument("--dataset", required=True, choices=sorted(_PREPARERS))
    ap.add_argument("--limit", type=int, default=0, help="cap docs (0=all; for smoke tests)")
    ap.add_argument("--years", default="all", help="hupd_full only: 'all' or CSV of years")
    args = ap.parse_args(argv)
    t0 = time.time()
    fn = _PREPARERS[args.dataset]
    if args.dataset == "hupd_full":
        fn(years=args.years, limit=args.limit)
    else:
        fn(limit=args.limit)
    print(f"[{args.dataset}] done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
