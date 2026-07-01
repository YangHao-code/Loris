#!/usr/bin/env python
"""Varying-K sweep for the LORIS model-selection experiment (paper Fig 5b/c).

For each K in K_GRID, run LORIS with each Group-A selector (router + the 4
baselines) and record LORIS's final_macro_f1. This shows how the selection
method's quality varies with the number of selected models K — where a learned
router's advantage over random/greedy selection is expected to emerge at small K.

Writes experiments/baselines_loris/ksweep/<selector>__<dataset>__k<K>__seed<seed>.json
so the main-matrix k=3 files (flat dir) are untouched.

Usage: python run_ksweep_loris.py --datasets aapd,reuters21578 --seeds 0 --kgrid 1,2,3,4,5
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from loris.baselines.common import DATASET_DEFAULTS

_ROOT = Path(__file__).resolve().parent
OUT = _ROOT / "experiments" / "baselines_loris" / "ksweep"
SELECTORS = ["router", "random_ms", "indiv_ms", "hybrid_llm", "caas"]
CANON_FLAGS = [
    "--top_labels", "30", "--two_val", "--rule_strategy", "batch",
    "--cluster_model_selection", "global", "--batch_metric_mode", "global_macro",
    "--track1_baseline", "blank", "--pattern_mode", "full",
    "--max_trials", "40", "--no_adaptive_trials",
]

def _caps(ds):
    d = DATASET_DEFAULTS[ds]; ex = []
    if d.subset_size: ex += ["--subset_size", str(d.subset_size)]
    if d.max_test_docs: ex += ["--max_test_docs", str(d.max_test_docs)]
    return ex

def _run(ds, seed, k, selector, exp_dir):
    exp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "loris", "--dataset", ds, *CANON_FLAGS, *_caps(ds),
           "--selector", selector, "--pattern_select", "loris",
           "--k_models", str(k), "--seed", str(seed), "--exp_dir", str(exp_dir)]
    env = dict(os.environ, HF_HOME="/root/autodl-tmp/hf_cache", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    t0 = time.time()
    with open(exp_dir / "run.log", "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    metrics = {}
    cands = sorted(exp_dir.glob(f"{ds}_*/metrics.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        try: metrics = json.loads(cands[0].read_text())
        except Exception: pass
    # reclaim disk
    import glob as _g
    for big in _g.glob(str(exp_dir/"**"/"model_pool.pkl"), recursive=True):
        try: os.remove(big)
        except Exception: pass
    for npy in _g.glob(str(exp_dir/"**"/"*.npy"), recursive=True):
        try: os.remove(npy)
        except Exception: pass
    return rc, time.time()-t0, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--kgrid", default="1,2,3,4,5")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()!=""]
    kgrid = [int(k) for k in args.kgrid.split(",") if k.strip()!=""]
    OUT.mkdir(parents=True, exist_ok=True)
    ok=fail=0
    for seed in seeds:
        for ds in datasets:
            for k in kgrid:
                for sel in SELECTORS:
                    ed = OUT / "_runs" / f"{sel}_{ds}_k{k}_s{seed}"
                    rc, sec, m = _run(ds, seed, k, sel, ed)
                    macro = m.get("final_macro_f1")
                    payload = {"baseline": sel, "dataset": ds, "seed": seed,
                               "k_models": k, "macro_f1": macro,
                               "micro_f1": m.get("final_micro_f1"),
                               "n_rules": m.get("n_rules"), "rc": rc,
                               "wall_sec": round(sec,1)}
                    p = OUT / f"{sel}__{ds}__k{k}__seed{seed}.json"
                    p.write_text(json.dumps(payload, indent=2))
                    status = "OK" if (rc==0 and macro is not None) else "FAIL"
                    ok += status=="OK"; fail += status=="FAIL"
                    print(f"[{status}] {sel}/{ds}/k{k} s{seed} macro={macro} ({round(sec)}s)", flush=True)
    print(f"\nKSWEEP DONE datasets={datasets} kgrid={kgrid} seeds={seeds}: ok={ok} fail={fail}", flush=True)
    return 0 if fail==0 else 1

if __name__ == "__main__":
    sys.exit(main())
