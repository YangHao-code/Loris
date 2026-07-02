#!/usr/bin/env python
"""Exp-1a — Main evaluation table: per-model reference accuracies + LORIS.

For each dataset runs LORIS from scratch (canonical config) with the FULL pool.
Two arms per dataset:
  * base  — 6 cheap/neural models + RoBERTa encoder   → main_7ds/base_<ds>/
  * lora  — base pool + BOTH LoRA SLMs (Mistral-7B + Llama-3-8B) → main_7ds/lora_<ds>/

The run's metrics.json carries per_model_test (each model alone = reference rows),
final_micro/macro_f1 (the LORIS block), n_rules, and the recorded LoRA PEFT config
+ trainable-%. gen_report.py reads main_7ds/ for the main table.

Usage:
  python run_main_loris.py --datasets reuters21578,aapd,... --arm both
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from exp_common import (ROOT, DATASETS_7, LORA_BOTH, run_cell, write_json, parse_datasets)

OUT = ROOT / "experiments" / "main_7ds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS_7))
    ap.add_argument("--arm", choices=["base", "lora", "both"], default="both")
    args = ap.parse_args()
    datasets = parse_datasets(args.datasets)
    arms = ["base", "lora"] if args.arm == "both" else [args.arm]
    ok = fail = 0
    for ds in datasets:
        for arm in arms:
            extra = []
            if arm == "lora":
                extra = ["--lora_model", LORA_BOTH]
            # full pool: do NOT pass --no_encoder (RoBERTa included).
            ed = OUT / f"{arm}_{ds}"
            res = run_cell(ds, extra, ed, use_tuned_router=True)
            m = res["metrics"]
            macro = m.get("final_macro_f1")
            payload = {
                "dataset": ds, "arm": arm, "seed": 0,
                "final_macro_f1": macro, "final_micro_f1": m.get("final_micro_f1"),
                "baseline_test_macro_f1": m.get("baseline_test_macro_f1"),
                "baseline_test_micro_f1": m.get("baseline_test_micro_f1"),
                "n_rules": m.get("n_rules"),
                "per_model_test": m.get("per_model_test"),
                "pool_models_used": m.get("pool_models_used"),
                "lora_peft_config": m.get("lora_peft_config"),
                "total_wall_sec": m.get("total_wall_sec"),
                "rc": res["rc"], "run_dir": res["run_dir"],
            }
            write_json(OUT / f"main_{arm}__{ds}.json", payload)
            status = "OK" if (res["rc"] == 0 and macro is not None) else "FAIL"
            ok += status == "OK"; fail += status == "FAIL"
            print(f"[{status}] main/{arm}/{ds} macro={macro} "
                  f"({res['wall_sec']}s)", flush=True)
    print(f"\nMAIN DONE datasets={datasets} arms={arms}: ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
