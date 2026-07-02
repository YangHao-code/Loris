#!/usr/bin/env python
"""Fetch a GitHub repo's files via trees-API + raw.githubusercontent.

`git clone` / codeload are blocked on this box (throttled Azure IPs), but the
Fastly-backed endpoints work: api.github.com trees-API lists files, and
raw.githubusercontent.com serves blobs. Slow (~15s/file) but reliable, with
retries. Skips large binaries (data/checkpoints) to protect the small disk.

Usage:
  python fetch_repo.py <owner/repo> <branch> <dest_dir> [--max-mb N] [--only PREFIX,..] [--ext .py,.md,..]
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.request

UA = {"User-Agent": "Mozilla/5.0"}

def _get(url, timeout=40, retries=4, raw=False):
    last = None
    for i in range(retries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
            return r.read() if raw else json.load(r)
        except Exception as e:
            last = e
            time.sleep(2 + 3 * i)
    raise last


def _get_blob(repo, branch, path, size):
    """Fetch one file's bytes, preferring fast CDNs over throttled raw.github.

    raw.githubusercontent (Fastly) is intermittently SSL-timeout-prone from this
    box; jsDelivr and ghproxy mirror the same content and are far more reliable.
    Tries each mirror once before falling back to raw with its own retries.
    """
    mirrors = [
        f"https://cdn.jsdelivr.net/gh/{repo}@{branch}/{path}",
        f"https://ghproxy.net/https://raw.githubusercontent.com/{repo}/{branch}/{path}",
    ]
    for url in mirrors:
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=20).read()
        except Exception:
            continue
    # last resort: raw.githubusercontent with retries
    return _get(f"https://raw.githubusercontent.com/{repo}/{branch}/{path}",
                raw=True, retries=3, timeout=30)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("branch"); ap.add_argument("dest")
    ap.add_argument("--max-mb", type=float, default=5.0, help="skip blobs larger than this")
    ap.add_argument("--only", default="", help="comma list of path prefixes to keep (default all)")
    ap.add_argument("--ext", default="", help="comma list of extensions to keep (default all)")
    ap.add_argument("--skip-ext", default=".bin,.pt,.pth,.ckpt,.h5,.zip,.tar,.gz,.npy,.npz,.pkl,.parquet,.model,.safetensors,.png,.jpg,.jpeg,.gif,.pdf",
                    help="comma list of extensions to always skip")
    args = ap.parse_args()

    only = [p for p in args.only.split(",") if p]
    exts = [e for e in args.ext.split(",") if e]
    skip_exts = [e for e in args.skip_ext.split(",") if e]
    max_bytes = args.max_mb * 1024 * 1024

    tree = None
    for api in (f"https://ghproxy.net/https://api.github.com/repos/{args.repo}/git/trees/{args.branch}?recursive=1",
                f"https://api.github.com/repos/{args.repo}/git/trees/{args.branch}?recursive=1"):
        try:
            tree = _get(api, retries=2, timeout=30)
            break
        except Exception:
            continue
    if tree is None:
        print("FATAL: could not list repo tree"); return 2
    blobs = [x for x in tree.get("tree", []) if x["type"] == "blob"]
    print(f"{args.repo}@{args.branch}: {len(blobs)} blobs total", flush=True)

    kept = skipped = failed = 0
    for x in blobs:
        path, size = x["path"], x.get("size", 0)
        low = path.lower()
        if only and not any(path.startswith(p) for p in only):
            skipped += 1; continue
        if exts and not any(low.endswith(e) for e in exts):
            skipped += 1; continue
        if any(low.endswith(e) for e in skip_exts):
            skipped += 1; continue
        if size > max_bytes:
            print(f"  SKIP (big {size//1024}KB) {path}", flush=True); skipped += 1; continue
        dest = os.path.join(args.dest, path)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            kept += 1; continue
        try:
            data = _get_blob(args.repo, args.branch, path, size)
            with open(dest, "wb") as f:
                f.write(data)
            kept += 1
            if kept % 10 == 0:
                print(f"  ... {kept} files", flush=True)
        except Exception as e:
            print(f"  FAIL {path}: {repr(e)[:60]}", flush=True); failed += 1
    print(f"DONE {args.repo}: kept={kept} skipped={skipped} failed={failed}", flush=True)
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
