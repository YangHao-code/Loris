#!/usr/bin/env python
"""Phase 3 (v2) — LORIS-integrated baselines for Group A (model selection) and
Group C (pattern selection), per the paper protocol: replace one LORIS component
and report LORIS's downstream Macro-F1.

Each cell shells `python -m loris ... --selector X` (Group A) or
`... --pattern_select Y` (Group C), then parses `final_macro_f1`/`final_micro_f1`
from the run's metrics.json and writes a uniform baseline JSON under
experiments/baselines_loris/<name>__<dataset>__seed<seed>.json.

The "LORIS" reference row = `--selector router` (router ON at the same K), so the
baseline selectors are compared apples-to-apples against LORIS's own selection.

Canonical caps come from loris.baselines.common.DATASET_DEFAULTS.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from loris.baselines.common import DATASET_DEFAULTS

_ROOT = Path(__file__).resolve().parent
OUT = _ROOT / "experiments" / "baselines_loris"

# Group A: model-selection methods (the LORIS reference = 'router').
SELECTORS = ["router", "random_ms", "indiv_ms", "hybrid_llm", "caas"]
# Group C: pattern-selection methods (the LORIS reference = 'loris').
PATTERN_SELECTORS = ["loris", "filter_mi", "filter_chi2", "weshap", "localboost"]

CANON_FLAGS = [
    "--top_labels", "30", "--two_val", "--rule_strategy", "batch",
    "--cluster_model_selection", "global", "--batch_metric_mode", "global_macro",
    "--track1_baseline", "blank", "--pattern_mode", "full",
    # pattern_mode 'full' (regex matching), NOT 'sim': sim-mode screening embeds
    # every sentence per predicate-call on CPU (~34.5M cold encodes/cell on
    # reuters = days/cell, intractable cold). 'full' uses regex matching — fast.
    # Applied EQUALLY to the LORIS reference row (A_router) so the selector /
    # pattern-selector comparisons stay apples-to-apples. Disclosed in report.
    # Bound the chase BO: adaptive floor is max(max_trials, 15*n_labels)=450/cluster
    # at 30 labels otherwise. Fixed budget applied to every cell incl. LORIS ref.
    "--max_trials", "40", "--no_adaptive_trials",
]


def _caps(ds: str):
    d = DATASET_DEFAULTS[ds]
    extra = []
    if d.subset_size:
        extra += ["--subset_size", str(d.subset_size)]
    if d.max_test_docs:
        extra += ["--max_test_docs", str(d.max_test_docs)]
    return extra


def _run_loris(ds, seed, k, *, selector="router", pattern_select="loris",
               exp_dir: Path) -> dict:
    """Run one LORIS cell, return parsed metrics dict (or {} on failure)."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "loris", "--dataset", ds,
           *CANON_FLAGS, *_caps(ds),
           "--selector", selector, "--pattern_select", pattern_select,
           "--k_models", str(k), "--seed", str(seed),
           "--exp_dir", str(exp_dir)]
    env = dict(os.environ,
               HF_HOME="/root/autodl-tmp/hf_cache", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    # CUDA_VISIBLE_DEVICES inherited from the parent (set per-lane by the
    # 2-GPU launcher); default to GPU 0 if unset.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    t0 = time.time()
    log = exp_dir / "run.log"
    # No hard wall-clock kill: LORIS runs at canonical caps legitimately take a
    # long time (esp. bgc/arxiv), and a premature kill yields spurious FAILs with
    # null macro. Optional soft cap via LORIS_CELL_TIMEOUT (unset = no timeout).
    _to = os.environ.get("LORIS_CELL_TIMEOUT", "").strip()
    timeout_s = int(_to) if _to else None
    with open(log, "w") as lf:
        try:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 env=env, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            lf.write(f"\n[run_phase3_loris] TIMEOUT after {timeout_s}s\n")
            rc = 124
    sec = time.time() - t0
    # find the newest metrics.json under exp_dir
    metrics = {}
    cands = sorted(exp_dir.glob(f"{ds}_*/metrics.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        try:
            metrics = json.loads(cands[0].read_text())
        except Exception:
            metrics = {}
    # Reclaim disk: each cell leaves a ~530MB model_pool.pkl + embedding .npy
    # caches in its exp_dir. Metrics are already parsed above, so purge the heavy
    # artifacts — 45 cells x 530MB would (and did) fill the 30GB disk → ENOSPC.
    try:
        import glob as _glob
        for big in _glob.glob(str(exp_dir / "**" / "model_pool.pkl"), recursive=True):
            os.remove(big)
        for npy in _glob.glob(str(exp_dir / "**" / "*.npy"), recursive=True):
            os.remove(npy)
    except Exception:
        pass
    return {"rc": rc, "wall_sec": sec, "metrics": metrics}


def _write(name, ds, seed, k, res, group):
    OUT.mkdir(parents=True, exist_ok=True)
    m = res["metrics"]
    payload = {
        "baseline": name, "dataset": ds, "seed": seed, "group": group,
        "k_models": k,
        "macro_f1": m.get("final_macro_f1"),
        "micro_f1": m.get("final_micro_f1"),
        "loris_baseline_macro_f1": m.get("baseline_test_macro_f1"),
        "n_rules": m.get("n_rules"),
        "rc": res["rc"], "wall_sec": round(res["wall_sec"], 1),
        "protocol": "LORIS-integrated (component swap; report LORIS final_macro_f1)",
    }
    p = OUT / f"{name}__{ds}__seed{seed}.json"
    p.write_text(json.dumps(payload, indent=2))
    return p, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["A", "C", "both"], default="both")
    ap.add_argument("--datasets", default="reuters21578,aapd,arxiv,bgc,rcv1")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    ok = fail = 0

    for seed in seeds:
        for ds in datasets:
            # Group A: vary --selector (pattern_select fixed 'loris')
            if args.group in ("A", "both"):
                for sel in SELECTORS:
                    name = f"A_{sel}"
                    ed = OUT / "_runs" / f"{name}_{ds}_s{seed}"
                    res = _run_loris(ds, seed, args.k, selector=sel,
                                     exp_dir=ed)
                    p, pl = _write(name, ds, seed, args.k, res, "A:model-select")
                    status = "OK" if (res["rc"] == 0 and pl["macro_f1"] is not None) else "FAIL"
                    if status == "OK": ok += 1
                    else: fail += 1
                    print(f"[{status}] {name}/{ds} s{seed} macro={pl['macro_f1']} "
                          f"({pl['wall_sec']}s) -> {p.name}", flush=True)
            # Group C: vary --pattern_select (selector fixed 'router')
            if args.group in ("C", "both"):
                for ps in PATTERN_SELECTORS:
                    if ps == "loris" and args.group == "both":
                        continue  # LORIS ref already covered by A_router
                    name = f"C_{ps}"
                    ed = OUT / "_runs" / f"{name}_{ds}_s{seed}"
                    res = _run_loris(ds, seed, args.k, pattern_select=ps,
                                     exp_dir=ed)
                    p, pl = _write(name, ds, seed, args.k, res, "C:pattern-sel")
                    status = "OK" if (res["rc"] == 0 and pl["macro_f1"] is not None) else "FAIL"
                    if status == "OK": ok += 1
                    else: fail += 1
                    print(f"[{status}] {name}/{ds} s{seed} macro={pl['macro_f1']} "
                          f"({pl['wall_sec']}s) -> {p.name}", flush=True)

    print(f"\nDONE group={args.group} datasets={datasets} seeds={seeds}: "
          f"ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
