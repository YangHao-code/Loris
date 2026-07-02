#!/usr/bin/env python
"""Exp-3 — Scalability (runtime) + noInc ablation.

Three sub-sweeps per dataset (each records total_wall_sec / chase_wall_sec /
selection_wall_sec + macro so the same runs double as an efficiency check):

  --do_dsize   |D| sweep: subset_size at 20/40/60/80/100% of the dataset base cap.
  --do_sigma   |Σ| sweep: rule-set size via --stage{1,2,3}_max_rules_per_label caps.
  --do_incremental  incremental vs --disable_incremental (noInc): same rules/labels,
                    compares chase wall-clock (the incremental optimisation's win).

→ experiments/scalability/<ds>.json  (keys: dsize, sigma, incremental)

Usage:
  python run_scalability.py --datasets bgc,aapd --do_dsize --do_sigma --do_incremental
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from exp_common import (ROOT, run_cell, write_json, parse_datasets)
from loris.baselines.common import DATASET_DEFAULTS

OUT = ROOT / "experiments" / "scalability"
DFRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
SIGMA_CAPS = [1, 2, 4, 8, 0]  # per-label rule caps (0 = uncapped)


def _timing(m: dict) -> dict:
    return {k: m.get(k) for k in ("total_wall_sec", "chase_wall_sec",
                                  "selection_wall_sec", "chase_n_rounds", "n_rules",
                                  "final_macro_f1", "final_micro_f1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="bgc,aapd")
    ap.add_argument("--do_dsize", action="store_true")
    ap.add_argument("--do_sigma", action="store_true")
    ap.add_argument("--do_incremental", action="store_true")
    ap.add_argument("--pool", choices=["base", "cpu"], default="cpu",
                    help="cpu (--no_encoder) keeps the timing sweep fast; base incl. encoder.")
    args = ap.parse_args()
    if not (args.do_dsize or args.do_sigma or args.do_incremental):
        args.do_dsize = args.do_sigma = args.do_incremental = True
    datasets = parse_datasets(args.datasets)
    pool_flags = ["--no_encoder"] if args.pool == "cpu" else []
    ok = fail = 0

    for ds in datasets:
        result = {"dataset": ds, "seed": 0}
        base = DATASET_DEFAULTS[ds].subset_size or 6000

        # ── |D| sweep ─────────────────────────────────────────────────────────
        if args.do_dsize:
            rows = []
            for f in DFRACS:
                n = max(500, int(base * f))
                ed = OUT / "_runs" / f"dsize_{ds}_{int(f*100)}"
                res = run_cell(ds, pool_flags + ["--subset_size", str(n)], ed)
                rows.append({"frac": f, "subset_size": n, **_timing(res["metrics"]),
                             "rc": res["rc"], "wall_sec": res["wall_sec"]})
                st = "OK" if res["rc"] == 0 else "FAIL"; ok += st == "OK"; fail += st == "FAIL"
                print(f"[{st}] dsize/{ds} n={n} total={res['metrics'].get('total_wall_sec')}s", flush=True)
            result["dsize"] = rows

        # ── |Σ| sweep (rule-set size) ─────────────────────────────────────────
        if args.do_sigma:
            rows = []
            for cap in SIGMA_CAPS:
                cf = ([] if cap == 0 else
                      ["--stage1_max_rules_per_label", str(cap),
                       "--stage2_max_rules_per_label", str(cap),
                       "--stage3_max_rules_per_label", str(cap)])
                ed = OUT / "_runs" / f"sigma_{ds}_cap{cap}"
                res = run_cell(ds, pool_flags + cf, ed)
                rows.append({"cap_per_label": cap, **_timing(res["metrics"]),
                             "rc": res["rc"], "wall_sec": res["wall_sec"]})
                st = "OK" if res["rc"] == 0 else "FAIL"; ok += st == "OK"; fail += st == "FAIL"
                print(f"[{st}] sigma/{ds} cap={cap} n_rules={res['metrics'].get('n_rules')} "
                      f"total={res['metrics'].get('total_wall_sec')}s", flush=True)
            result["sigma"] = rows

        # ── incremental vs noInc ──────────────────────────────────────────────
        if args.do_incremental:
            rows = []
            for label, cf in (("incremental", []), ("noInc", ["--disable_incremental"])):
                ed = OUT / "_runs" / f"inc_{ds}_{label}"
                res = run_cell(ds, pool_flags + cf, ed)
                m = res["metrics"]
                rows.append({"mode": label, "chase_wall_sec": m.get("chase_wall_sec"),
                             "chase_n_rounds": m.get("chase_n_rounds"),
                             "total_wall_sec": m.get("total_wall_sec"),
                             "final_macro_f1": m.get("final_macro_f1"),
                             "n_rules": m.get("n_rules"), "rc": res["rc"]})
                st = "OK" if res["rc"] == 0 else "FAIL"; ok += st == "OK"; fail += st == "FAIL"
                print(f"[{st}] inc/{ds}/{label} chase={m.get('chase_wall_sec')}s "
                      f"macro={m.get('final_macro_f1')}", flush=True)
            result["incremental"] = rows

        write_json(OUT / f"{ds}.json", result)

    print(f"\nSCALABILITY DONE datasets={datasets}: ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
