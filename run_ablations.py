#!/usr/bin/env python
"""Exp-5 — Ablation study (architecture + staged pipeline + pattern selection).

Architecture ablations (new code): noS (Gumbel-Softmax instead of stochastic
smoothing), noL (task-loss-only router), noInc (non-incremental chase). Staged
ablations: leave-one-out of stages 1/2/3; pool=embedding-only; gt-leak oracle
bound. Each arm records LORIS final macro-F1 (+ chase timing for noInc).

  → experiments/ablations/<ds>.json

The noLLM ablation (LLM pre-annotator vs human-only) is quantified as a HUMAN-COST
curve by run_humancost.py (arms no_llm / llm), not here — its headline metric is
# human annotations, not macro-F1.

Pattern-selection alternatives (filter_mi / filter_chi2 / weshap / localboost) are
the Group-C gate produced by run_phase3_loris.py (component swap); gen_report reads
those. Run run_phase3_loris for any missing datasets (pubmed/hupd).

Usage: python run_ablations.py --datasets aapd,bgc --pool base
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from exp_common import (ROOT, run_cell, write_json, parse_datasets)

OUT = ROOT / "experiments" / "ablations"

# (name, extra_flags) — 'full' is the reference cell.
ARMS = [
    ("full", []),
    ("noS_gumbel", ["--no_smoothing"]),
    ("noL_taskonly", ["--task_loss_only"]),
    ("noInc", ["--disable_incremental"]),
    ("no_stage1", ["--no_stage1_fn_add"]),
    ("no_stage2", ["--no_stage2_fp_remove"]),
    ("no_stage3", ["--no_stage3_propagation"]),
    ("pool_embed", ["--model_pool", "embedding"]),
    ("gtleak", ["--prop_label_source", "gt"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="aapd,bgc")
    ap.add_argument("--pool", choices=["base", "cpu"], default="base",
                    help="base incl. RoBERTa encoder (paper); cpu = --no_encoder (fast).")
    ap.add_argument("--only", default=None, help="comma list to restrict arms.")
    args = ap.parse_args()
    datasets = parse_datasets(args.datasets)
    pool_flags = ["--no_encoder"] if args.pool == "cpu" else []
    only = {a.strip() for a in args.only.split(",")} if args.only else None
    ok = fail = 0

    for ds in datasets:
        rows = []
        for name, cf in ARMS:
            if only and name not in only:
                continue
            # pool_embed already drops tfidf; don't also force --no_encoder there.
            pf = [] if name == "pool_embed" else pool_flags
            ed = OUT / "_runs" / f"{name}_{ds}"
            res = run_cell(ds, pf + cf, ed, use_tuned_router=True)
            m = res["metrics"]
            rows.append({"arm": name, "macro_f1": m.get("final_macro_f1"),
                         "micro_f1": m.get("final_micro_f1"),
                         "n_rules": m.get("n_rules"),
                         "chase_wall_sec": m.get("chase_wall_sec"),
                         "total_wall_sec": m.get("total_wall_sec"),
                         "rc": res["rc"], "wall_sec": res["wall_sec"]})
            st = "OK" if (res["rc"] == 0 and m.get("final_macro_f1") is not None) else "FAIL"
            ok += st == "OK"; fail += st == "FAIL"
            print(f"[{st}] abl/{ds}/{name} macro={m.get('final_macro_f1')} "
                  f"chase={m.get('chase_wall_sec')}s ({res['wall_sec']}s)", flush=True)
        write_json(OUT / f"{ds}.json", {"dataset": ds, "seed": 0, "arms": rows})

    print(f"\nABLATIONS DONE datasets={datasets}: ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
