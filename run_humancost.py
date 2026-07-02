#!/usr/bin/env python
"""Exp-2 — Human cost: rules substitute for labels; the LLM pre-annotator saves
human annotations. All from scratch (--track1_baseline blank; rules LEARN from the
human labels, they never "correct" a model prediction).

Two complementary curves per dataset:

1. Training-time Γ sweep (control |Γ|): --label_budget g ∈ gamma_grid. LORIS fits
   the pool + discovers rules from ONLY g human-labeled train docs (coverage-seeded),
   then labels the whole test set. Records final macro/micro + Γ (= # human
   annotations). → experiments/humancost/gamma_<ds>.json

2. Test-time active-labeling sweep: a single run with --rill_budget_sweep, in two
   arms — --no_llm (human-only) and --use_llm_annotator (LLM pre-annotator + trust
   check + human fallback). n_human_queries per budget shows the LLM's human-cost
   saving. → experiments/humancost/rill_<ds>_<arm>.json

BESRA/CoMAL HITL curves are the standalone baselines under experiments/baselines/budget/
(run by run_phase3.py); gen_report.py plots LORIS vs those at matched annotation counts.

Usage:
  python run_humancost.py --datasets bgc,aapd --gamma_grid 0,25,50,100,200,400 \
      --rill_grid 0,25,50,100,200,400 --arms no_llm,llm
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from exp_common import (ROOT, DATASETS_7, run_cell, sweep_json, write_json, parse_datasets)

OUT = ROOT / "experiments" / "humancost"


def _pool_flags(pool: str) -> list:
    # base = encoder pool (no LoRA); cpu = --no_encoder (fast).
    return ["--no_encoder"] if pool == "cpu" else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS_7))
    ap.add_argument("--gamma_grid", default="0,25,50,100,200,400",
                    help="training-time |Γ| label budgets (empty ⇒ skip).")
    ap.add_argument("--rill_grid", default="0,25,50,100,200,400",
                    help="test-time RILL active-labeling budgets (empty ⇒ skip).")
    ap.add_argument("--arms", default="no_llm,llm",
                    help="RILL annotator arms: no_llm (human only) and/or llm (LLM+human).")
    ap.add_argument("--pool", choices=["base", "cpu"], default="base")
    args = ap.parse_args()
    datasets = parse_datasets(args.datasets)
    gamma_grid = [int(x) for x in args.gamma_grid.split(",") if x.strip() != ""]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ok = fail = 0

    for ds in datasets:
        # ── 1. training-time Γ sweep ──────────────────────────────────────────
        gamma_rows = []
        for g in gamma_grid:
            extra = _pool_flags(args.pool) + (
                ["--label_budget", str(g), "--label_budget_seeding", "coverage"]
                if g > 0 else [])
            ed = OUT / "_runs" / f"gamma_{ds}_g{g}"
            res = run_cell(ds, extra, ed, use_tuned_router=True)
            m = res["metrics"]
            row = {"gamma": g, "n_human_annotations": m.get("n_human_annotations", g),
                   "macro_f1": m.get("final_macro_f1"), "micro_f1": m.get("final_micro_f1"),
                   "baseline_macro_f1": m.get("baseline_test_macro_f1"),
                   "n_rules": m.get("n_rules"), "rc": res["rc"],
                   "wall_sec": res["wall_sec"]}
            gamma_rows.append(row)
            st = "OK" if (res["rc"] == 0 and row["macro_f1"] is not None) else "FAIL"
            ok += st == "OK"; fail += st == "FAIL"
            print(f"[{st}] gamma/{ds} Γ={g} macro={row['macro_f1']} ({res['wall_sec']}s)", flush=True)
        if gamma_rows:
            write_json(OUT / f"gamma_{ds}.json", {"dataset": ds, "seed": 0, "rows": gamma_rows})

        # ── 2. test-time RILL active-labeling sweep (LLM vs no-LLM) ───────────
        if args.rill_grid.strip():
            for arm in arms:
                extra = _pool_flags(args.pool) + [
                    "--rill_budget_sweep", args.rill_grid,
                    ("--use_llm_annotator" if arm == "llm" else "--no_llm"),
                ]
                ed = OUT / "_runs" / f"rill_{ds}_{arm}"
                res = run_cell(ds, extra, ed, use_tuned_router=True)
                sw = sweep_json(res["run_dir"])
                write_json(OUT / f"rill_{ds}_{arm}.json",
                           {"dataset": ds, "arm": arm, "seed": 0,
                            "budgets": sw.get("budgets", []),
                            "rc": res["rc"], "wall_sec": res["wall_sec"]})
                nb = len(sw.get("budgets", []))
                st = "OK" if (res["rc"] == 0 and nb > 0) else "FAIL"
                ok += st == "OK"; fail += st == "FAIL"
                print(f"[{st}] rill/{ds}/{arm} budgets={nb} ({res['wall_sec']}s)", flush=True)

    print(f"\nHUMANCOST DONE datasets={datasets}: ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
