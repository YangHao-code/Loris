"""Per-dataset router hyperparameter tuning (oracle-based, chase-free).

The dynamic router historically reused ONE hyperparameter config across every
dataset. This module tunes the router per dataset by an *intrinsic* oracle
metric on held-out val — it trains ONLY the SelectionNetwork on cached pool
predictions (no rule-discovery / chase), so a full sweep costs one pool fit plus
a handful of cheap router trainings.

Metric (shared with the run-time diagnostic in ``loris.pipeline.shared``):
build the per-doc val oracle (top-k models by label-overlap) and score the
router's raw scores by **oracle-hit@K** and **MRR** against a FIXED reference
oracle (``k_ref``) so configs with different ``k_models`` stay comparable. The
tuned config is written to ``experiments/router_tuning/router_tuned_<ds>_s<seed>.json``
and consumed by ``run_phase3_loris.py --use_tuned_router``.

CLI
---
    python -m loris.selection.router_tuning --dataset reuters21578 --seed 0 \\
        --backend perturbed --max_configs 18 --out experiments/router_tuning
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from loris.data import DATASET_REGISTRY, HParams, load_data
from loris.baselines.common import DATASET_DEFAULTS, DSConfig
from loris.pipeline.shared import (
    init_models,
    train_models,
    _train_selection_net,
    _build_multi_label_oracle_mask,
    router_oracle_metrics,
)

# Default sweep grid (per dataset).
DEFAULT_GRID = {
    "k_models": [2, 3, 4],
    "router_sigma": [0.05, 0.1, 0.25],
    "router_lr": [1e-3, 3e-3],
    "router_epochs": [30, 60],
}


def _enumerate_grid(grid: Dict[str, List]) -> List[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def evaluate_config(pool, train_X, train_y, val_X, val_y, base_hp: HParams,
                    cfg_over: dict, device, oracle_ref: np.ndarray, k_ref: int,
                    seed: int) -> dict:
    """Train the router with one config and score it against the reference oracle."""
    hp_i = replace(base_hp, **cfg_over)
    k = min(int(hp_i.k_models), len(pool))
    torch.manual_seed(seed)  # identical smoothing noise stream across configs
    net, tfidf_vec, svd = _train_selection_net(
        pool, train_X, train_y, hp_i, k, device, verbose=False)
    Xval = svd.transform(tfidf_vec.transform(val_X)).astype(np.float32)
    Xval_t = torch.from_numpy(Xval).to(device)
    with torch.no_grad():
        scores_val = net.get_scores(Xval_t).cpu().numpy()
    metrics = router_oracle_metrics(scores_val, oracle_ref, k_ref)
    return {"cfg": cfg_over, **metrics}


def tune_dataset(dataset: str, seed: int = 0, backend: str = "perturbed",
                 max_configs: int = 18, grid: Optional[dict] = None,
                 model_pool: str = "default") -> dict:
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {sorted(DATASET_REGISTRY)}")
    grid = grid or DEFAULT_GRID
    cfg = DATASET_REGISTRY[dataset]
    ds_def = DATASET_DEFAULTS.get(dataset, DSConfig())

    base_hp = HParams(
        top_labels=ds_def.top_labels,
        subset_size=ds_def.subset_size,
        val_ratio=0.40,
        seed=seed,
        router_backend=backend,
    )
    base_hp.max_test_docs = ds_def.max_test_docs

    t0 = time.time()
    print(f"[tune:{dataset}] loading data …", flush=True)
    _data = load_data(cfg, base_hp)
    train_X, val_X = _data[0], _data[1]
    train_y, val_y = _data[3], _data[4]
    label_names = _data[6]

    print(f"[tune:{dataset}] building + fitting model pool …", flush=True)
    pool = init_models(len(label_names), lora_model_name=None,
                       drop_tfidf=(model_pool == "embedding"))
    train_models(pool, train_X, train_y, val_X, val_y)
    n_models = len(pool)

    # Fixed reference oracle (comparable scoring across configs with different k).
    k_ref = min(3, n_models)
    oracle_ref = _build_multi_label_oracle_mask(pool, val_X, val_y, k_ref)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = _enumerate_grid(grid)
    if max_configs and len(configs) > max_configs:
        configs = random.Random(seed).sample(configs, max_configs)

    print(f"[tune:{dataset}] sweeping {len(configs)} configs "
          f"(pool={n_models} models, k_ref={k_ref}) …", flush=True)
    results = []
    for i, cfg_over in enumerate(configs, 1):
        try:
            r = evaluate_config(pool, train_X, train_y, val_X, val_y, base_hp,
                                cfg_over, device, oracle_ref, k_ref, seed)
        except Exception as exc:
            print(f"  [{i}/{len(configs)}] cfg {cfg_over} FAILED: {exc}", flush=True)
            continue
        results.append(r)
        print(f"  [{i}/{len(configs)}] {cfg_over} -> hit@{k_ref}={r['oracle_hit_at_k']} "
              f"MRR={r['mrr']} combined={r['combined']}", flush=True)

    if not results:
        raise RuntimeError(f"No router configs succeeded for {dataset}")

    best = max(results, key=lambda r: r["combined"])
    out = {
        "dataset": dataset,
        "seed": seed,
        "backend": backend,
        "n_models": n_models,
        "n_train": len(train_X),
        "n_val": len(val_X),
        "k_ref": k_ref,
        "metric": "0.5*oracle_hit@k_ref + 0.5*MRR",
        "best": {**best["cfg"], **{k: best[k] for k in ("oracle_hit_at_k", "mrr", "combined")}},
        "grid_results": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(f"[tune:{dataset}] BEST {out['best']}  ({out['elapsed_sec']}s)", flush=True)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Per-dataset oracle-based router tuning.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="perturbed", choices=["custom", "perturbed"])
    ap.add_argument("--max_configs", type=int, default=18)
    ap.add_argument("--model_pool", default="default", choices=["default", "embedding"])
    ap.add_argument("--out", default="experiments/router_tuning",
                    help="output directory for router_tuned_<ds>_s<seed>.json")
    args = ap.parse_args(argv)

    out = tune_dataset(args.dataset, seed=args.seed, backend=args.backend,
                       max_configs=args.max_configs, model_pool=args.model_pool)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"router_tuned_{args.dataset}_s{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
