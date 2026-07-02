#!/usr/bin/env python
"""Assemble the LORIS main accuracy table (Paper Table 1) from main_0619 runs.

Rows = each base model (standalone test F1) + LORIS (best-of-pool baseline +
discovered rules). Cols = each dataset × {macro, micro}. LORIS row shows Δmacro
over the val-selected baseline. Reads metrics.json{per_model_test, baseline_*,
final_*}."""
import glob
import json
import os

DATASETS = ["aapd", "bgc", "rcv1", "reuters21578"]
EXP = "experiments/main_0619"
DIR = {"aapd": "main_aapd", "bgc": "main_bgc", "rcv1": "main_rcv1",
       "reuters21578": "main_reuters"}
MODEL_ORDER = ["tfidf_svm_unigram", "tfidf_svm_bigram", "tfidf_lr_bigram",
               "textcnn", "bilstm", "encoder_mlp", "lora_slm"]


def load(ds):
    g = glob.glob(f"{EXP}/{DIR[ds]}/**/metrics.json", recursive=True)
    return json.load(open(sorted(g)[0])) if g else None


def fmt(x):
    return f"{x:.4f}" if x is not None else "  –  "


if __name__ == "__main__":
    data = {ds: load(ds) for ds in DATASETS}
    done = {ds: d for ds, d in data.items() if d}
    print(f"\n## Main accuracy table — per-model standalone vs LORIS (test macro / micro)\n")
    print(f"_{len(done)}/{len(DATASETS)} datasets complete; config: full supervision, "
          f"pattern_mode full, multiattr, positive-only rescue, CPU pool (no encoder/LoRA yet)_\n")
    # header
    cols = " | ".join(f"{ds} macro | {ds} micro" for ds in DATASETS)
    print(f"| model | {cols} |")
    print("|" + "---|" * (1 + 2 * len(DATASETS)))
    # per-model rows
    for m in MODEL_ORDER:
        cells = []
        present = False
        for ds in DATASETS:
            d = data[ds]
            pm = (d or {}).get("per_model_test", {}).get(m) if d else None
            if pm:
                present = True
                cells.append(f"{fmt(pm['macro_f1'])} | {fmt(pm['micro_f1'])}")
            else:
                cells.append("  –   |   –  ")
        if present:
            print(f"| {m} | " + " | ".join(cells) + " |")
    # baseline (best-of-pool, val-selected) + LORIS rows
    for label, kmac, kmic in [("**best-of-pool baseline**", "baseline_test_macro_f1", "baseline_test_micro_f1"),
                              ("**LORIS (base+rules)**", "final_macro_f1", "final_micro_f1")]:
        cells = []
        for ds in DATASETS:
            d = data[ds]
            cells.append(f"{fmt(d.get(kmac))} | {fmt(d.get(kmic))}" if d else "  –   |   –  ")
        print(f"| {label} | " + " | ".join(cells) + " |")
    # delta row
    cells = []
    for ds in DATASETS:
        d = data[ds]
        if d:
            dm = d["final_macro_f1"] - d["baseline_test_macro_f1"]
            di = d["final_micro_f1"] - d["baseline_test_micro_f1"]
            cells.append(f"{dm:+.4f} | {di:+.4f}")
        else:
            cells.append("  –   |   –  ")
    print(f"| **Δ LORIS vs baseline** | " + " | ".join(cells) + " |")
    print(f"\nn_rules per dataset: " +
          ", ".join(f"{ds}={data[ds]['n_rules']}" if data[ds] else f"{ds}=–" for ds in DATASETS))
