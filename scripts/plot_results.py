#!/usr/bin/env python3
"""Plot actual final results or complete offline round/predicate metrics."""
import argparse
import csv
from pathlib import Path


def load_rows(path, dataset, metric):
    with open(path) as stream:
        rows = [r for r in csv.DictReader(stream)
                if r.get("dataset", dataset) == dataset and r.get(metric)
                and r.get("run_status", "all_documents_resolved") in (
                    "all_documents_resolved", "label_budget_exhausted", "conflict_unknown_saturated")
                and r.get("scoring_status", "complete") in ("complete", "legacy_unspecified")]
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metric", default="macro_f1",
                   help="Final macro_f1, or an explicit round output_macro_f1/logical_macro_f1/aux_at_0_5_macro_f1")
    a = p.parse_args()
    rows = load_rows(a.results, a.dataset, a.metric)
    if not rows:
        raise SystemExit("no actual scored rows for this dataset")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.6), 4.5))
    if "round" in rows[0]:
        groups = {}
        for row in rows:
            key = row.get("variant", row.get("name", "run"))
            if row.get("occurrence"):
                key += "/" + row["occurrence"]
            groups.setdefault(key, []).append(row)
        for key, group in groups.items():
            group.sort(key=lambda r: int(r["round"]))
            ax.plot([int(r["round"]) for r in group], [float(r[a.metric]) for r in group],
                    marker="o", label=key)
        ax.set_xlabel("Recorded round")
        ax.legend()
    else:
        ax.bar(range(len(rows)), [float(r[a.metric]) for r in rows])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([
            r.get("name", r.get("variant", "run")) + (
                " [" + r["run_status"] + "]" if r.get("run_status", "all_documents_resolved") != "all_documents_resolved" else "")
            for r in rows], rotation=65, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(a.metric)
    ax.set_title(a.dataset + " — observed results; see CSV for evaluation protocol")
    fig.tight_layout()
    dest = Path(a.output)
    if dest.exists():
        raise SystemExit("output exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)


if __name__ == "__main__":
    main()
