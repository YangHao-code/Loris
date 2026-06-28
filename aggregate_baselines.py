"""Aggregate baseline result JSONs into a comparison matrix.

Reads ``experiments/baselines/*.json`` (written by
``loris.baselines.run_baselines``) and prints a macro/micro-F1 table grouped by
baseline × dataset, plus a CSV.  Mirrors ``aggregate_main_table.py`` so the
baseline rows can be folded into ``gen_report.py`` / ``paper/results_auto.tex``.

    python aggregate_baselines.py [--dir experiments/baselines] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

DATASETS = ["reuters21578", "aapd", "rcv1", "bgc", "arxiv"]

# group label for nicer printing
GROUP = {
    "random_ms": "A:model-select", "indiv_ms": "A:model-select",
    "hybrid_llm": "A:model-select", "caas": "A:model-select",
    "snuba": "B:end-to-end", "self_pretrain": "B:end-to-end",
    "ruleprompt": "B:end-to-end", "deberta_svm": "B:end-to-end",
    "deberta_xgboost": "B:end-to-end", "roberta_svm": "B:end-to-end",
    "roberta_xgboost": "B:end-to-end", "gpt4": "B:end-to-end",
    "besra": "B:end-to-end(HITL)", "ral": "B:end-to-end(HITL)",
    "filter_mi": "C:pattern-sel", "filter_chi2": "C:pattern-sel",
    "weshap": "C:pattern-sel", "localboost": "C:pattern-sel",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/baselines")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    # (baseline, dataset) -> list of records (avg over seeds)
    recs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        recs[(d["baseline"], d["dataset"])].append(d)

    if not recs:
        print(f"No result JSONs under {args.dir}/. Run baselines first.")
        return

    baselines = sorted({b for (b, _) in recs}, key=lambda b: (GROUP.get(b, "z"), b))
    rows = []
    for b in baselines:
        for metric in ("macro_f1", "micro_f1"):
            cells = []
            for ds in DATASETS:
                vals = [r[metric] for r in recs.get((b, ds), []) if r.get(metric) == r.get(metric)]
                cells.append(sum(vals) / len(vals) if vals else None)
            rows.append((b, GROUP.get(b, "?"), metric, cells))

    # pretty table (macro then micro per baseline)
    w = 14
    header = f"{'baseline':<18}{'group':<20}{'metric':<9}" + "".join(f"{ds:>{w}}" for ds in DATASETS)
    print(header)
    print("-" * len(header))
    for b, g, metric, cells in rows:
        line = f"{b:<18}{g:<20}{metric:<9}"
        for c in cells:
            line += f"{('%.4f' % c) if c is not None else '—':>{w}}"
        print(line)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["baseline", "group", "metric"] + DATASETS)
            for b, g, metric, cells in rows:
                wtr.writerow([b, g, metric] + [("" if c is None else f"{c:.4f}") for c in cells])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
