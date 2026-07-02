#!/usr/bin/env python
"""Exp-4 (control 2) — Varying model-pool size |M|.

For each |M| in {2..6} (nested top-|M| of the canonical pool: tfidf×3, textcnn,
bilstm, encoder_mlp) and each Group-A selector, run LORIS and record the final
macro-F1 + selection wall-clock. Shows how each selector scales with pool size —
where a learned router's advantage over random/greedy is expected to widen.

Protocol mirrors run_phase3_loris (selector swap): the 'router' cell uses the
per-cluster router; the baseline selectors use --cluster_model_selection global
+ --selector X at the same K (K = min(3, |M|)). --pool_size restricts BEFORE
training so small-|M| cells are cheap (encoder only built when |M|≥6).

→ experiments/baselines_loris/msweep/<selector>__<ds>__M<M>__seed0.json

Usage: python run_msweep_loris.py --datasets aapd,reuters21578,bgc --mgrid 2,3,4,5,6
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from exp_common import (ROOT, run_cell, write_json, parse_datasets)

OUT = ROOT / "experiments" / "baselines_loris" / "msweep"
SELECTORS = ["router", "random_ms", "indiv_ms", "hybrid_llm", "caas"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="aapd,reuters21578,bgc")
    ap.add_argument("--mgrid", default="2,3,4,5,6")
    args = ap.parse_args()
    datasets = parse_datasets(args.datasets)
    mgrid = [int(x) for x in args.mgrid.split(",") if x.strip() != ""]
    ok = fail = 0

    for ds in datasets:
        for M in mgrid:
            k = min(3, M)
            for sel in SELECTORS:
                extra = ["--pool_size", str(M), "--k_models", str(k)]
                if sel == "router":
                    extra += ["--selector", "router"]  # CANON already per-cluster router
                else:
                    extra += ["--cluster_model_selection", "global", "--selector", sel]
                ed = OUT / "_runs" / f"{sel}_{ds}_M{M}"
                # router hyperparams are pool-size dependent; don't force the tuned
                # (k could exceed |M|) — let k_models drive it.
                res = run_cell(ds, extra, ed, use_tuned_router=False)
                m = res["metrics"]
                payload = {"selector": sel, "dataset": ds, "M": M, "k_models": k,
                           "seed": 0, "macro_f1": m.get("final_macro_f1"),
                           "micro_f1": m.get("final_micro_f1"),
                           "selection_wall_sec": m.get("selection_wall_sec"),
                           "n_rules": m.get("n_rules"),
                           "pool_models_used": m.get("pool_models_used"),
                           "rc": res["rc"], "wall_sec": res["wall_sec"]}
                write_json(OUT / f"{sel}__{ds}__M{M}__seed0.json", payload)
                st = "OK" if (res["rc"] == 0 and payload["macro_f1"] is not None) else "FAIL"
                ok += st == "OK"; fail += st == "FAIL"
                print(f"[{st}] msweep/{sel}/{ds}/M{M} macro={payload['macro_f1']} "
                      f"({res['wall_sec']}s)", flush=True)

    print(f"\nMSWEEP DONE datasets={datasets} mgrid={mgrid}: ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
