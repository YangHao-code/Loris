#!/usr/bin/env python
"""Phase 3 orchestration driver for LORIS baselines on a SINGLE GPU.

Runs the full paper protocol for the 16 non-LLM baselines (gpt4/ruleprompt held
for the OpenAI key):

  * main matrix : every applicable baseline x every dataset x seeds {0,1,2}
  * K-sweep     : selectors (random_ms/indiv_ms/hybrid_llm/caas) k=1..6 x seeds
  * budget-sweep: HITL (besra/ral) budget={0,50,100,200,400} x seeds

Single GPU => GPU-touching datasets run sequentially in this process. rcv1 is
CPU/linear-only and is launched as a SEPARATE concurrent process by the shell
wrapper (run_phase3.sh) so it overlaps the GPU work for free.

Result filenames only encode baseline__dataset__seedN, so the sweeps (which vary
k / budget) are written to per-config out_dirs to avoid collisions:
  experiments/baselines/                        <- main matrix, FLAT (the default
                                                   aggregate_baselines.py glob)
  experiments/baselines/ksweep/k<K>/            <- selector K-sweep
  experiments/baselines/budget/b<B>/            <- HITL budget-sweep
The sweep subdirs are ignored by the default aggregator (non-recursive glob) and
analysed separately.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from loris.baselines import common as C
from loris.baselines.run_baselines import build_registry

log = logging.getLogger("phase3")

GROUP_A = ["random_ms", "indiv_ms", "hybrid_llm", "caas"]
GROUP_B_NOLLM = ["snuba", "self_pretrain", "deberta_svm", "deberta_xgboost",
                 "roberta_svm", "roberta_xgboost", "besra", "comal"]
GROUP_C = ["filter_mi", "filter_chi2", "weshap", "localboost"]
HITL = ["besra", "comal"]   # RAL slot -> CoMAL (substitute, reported under own name)
# everything except the two LLM baselines (held for the key)
ALL_NONLLM = GROUP_A + GROUP_B_NOLLM + GROUP_C

SEEDS = [0, 1, 2]
K_GRID = [1, 2, 3, 4, 5, 6]
# budget=0 would fit on an empty labeled set; start the human-cost curve at 50.
BUDGET_GRID = [50, 100, 200, 400]


def run_one(registry, name, split, *, seed, out_dir, **kwargs):
    """Run a single (baseline, split, seed) and write its JSON. Returns status."""
    fn = registry[name]
    log.info("RUN %s | %s | seed=%d | %s", name, split.dataset, seed,
             {k: v for k, v in kwargs.items()})
    t0 = time.time()
    try:
        out = fn(split, seed=seed, **kwargs)
    except NotImplementedError as exc:
        log.warning("SKIP %s on %s: %s", name, split.dataset, exc)
        return "skip"
    except Exception as exc:  # noqa
        log.exception("FAIL %s on %s seed=%d: %s", name, split.dataset, seed, exc)
        return "fail"
    sec = time.time() - t0
    path = C.write_result(out_dir, name, split.dataset, out, seed=seed,
                          n_annotations=int(out.get("n_annotations", 0)),
                          n_test=len(split.test_X), wall_sec=sec,
                          extra=out.get("extra", {}))
    log.info("OK   %s/%s seed=%d macro=%.4f micro=%.4f (%.1fs) -> %s",
             name, split.dataset, seed, out.get("macro_f1", float("nan")),
             out.get("micro_f1", float("nan")), sec, path)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True,
                    help="comma list, e.g. reuters21578,aapd,bgc,arxiv or rcv1")
    ap.add_argument("--out_root", default="experiments/baselines")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--skip_sweeps", action="store_true",
                    help="main matrix only (no K / budget sweeps)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    registry = build_registry()
    out_root = Path(args.out_root)

    summary = {"ok": 0, "skip": 0, "fail": 0}

    for ds in datasets:
        log.info("########## DATASET %s ##########", ds)
        # Load each split ONCE per seed (selectors memoize their pool on the split
        # object, so a fresh split per seed is correct and the pool fits once per
        # seed across all four selectors + the K-sweep).
        for seed in seeds:
            t_load = time.time()
            split = C.load_split(ds)
            log.info("Loaded %s seed=%d: train=%d val=%d test=%d labels=%d (%.1fs)",
                     ds, seed, len(split.train_X), len(split.val_X),
                     len(split.test_X), split.n_labels, time.time() - t_load)

            # ---- main matrix (k=3 default for selectors, budget=400 default HITL) ----
            for name in ALL_NONLLM:
                st = run_one(registry, name, split, seed=seed,
                             out_dir=out_root)
                summary[st] = summary.get(st, 0) + 1

            if args.skip_sweeps:
                continue

            # ---- K-sweep: selectors, k=1..6 (reuses the memoized pool) ----
            for k in K_GRID:
                for name in GROUP_A:
                    st = run_one(registry, name, split, seed=seed, k=k,
                                 out_dir=out_root / "ksweep" / f"k{k}")
                    summary[st] = summary.get(st, 0) + 1

            # ---- budget-sweep: HITL besra/ral, budget grid ----
            for bdg in BUDGET_GRID:
                for name in HITL:
                    st = run_one(registry, name, split, seed=seed, budget=bdg,
                                 out_dir=out_root / "budget" / f"b{bdg}")
                    summary[st] = summary.get(st, 0) + 1

    log.info("######## DONE datasets=%s : ok=%d skip=%d fail=%d ########",
             datasets, summary.get("ok", 0), summary.get("skip", 0),
             summary.get("fail", 0))
    return 0 if summary.get("fail", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
