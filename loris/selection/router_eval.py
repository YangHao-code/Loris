"""Router TEST-set oracle-gap evaluation.

Answers the question: on the *test* set, does the router's fixed selected-K
match the per-document GT-best-K, and how big is the gap?

For a dataset it (1) fits the model pool, (2) trains the router with the tuned
config (``experiments/router_tuning/router_tuned_<ds>_s<seed>.json`` if present,
else defaults), (3) computes:

  * val vs **test** oracle-hit@K / MRR of the router's raw scores against the
    per-doc oracle (top-K models by GT label-overlap) — the val→test
    generalization gap;
  * the router's FINAL fixed global selection S (the K models applied to every
    doc) and, on test, how often S matches the per-doc GT-best-K
    (``mean_overlap_hit``, ``frac_docs_matching_fixed``, ``n_distinct_perdoc_topk``);
  * a **ceiling gap**: macro-F1 of the fixed-selection ensemble vs a per-doc
    oracle ensemble (route perfectly per doc) — what fixing K costs.

CLI
---
    python -m loris.selection.router_eval --dataset reuters21578 --seed 0 \\
        --tuned experiments/router_tuning --out experiments/router_eval
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score

from loris.data import DATASET_REGISTRY, HParams, load_data
from loris.baselines.common import DATASET_DEFAULTS, DSConfig
from loris.pipeline.shared import (
    init_models, train_models, _train_selection_net,
    _build_multi_label_oracle_mask, router_oracle_metrics, FinalSelector,
)


def _model_probas(pool, names, X) -> np.ndarray:
    """(n_models, n_docs, n_labels) predict_proba for every pool model."""
    out = []
    for name in names:
        try:
            p = np.asarray(pool[name].predict_proba(X), dtype=np.float32)
        except Exception:
            p = None
        out.append(p)
    n_labels = next(p.shape[1] for p in out if p is not None)
    n_docs = len(X)
    for i, p in enumerate(out):
        if p is None:
            out[i] = np.zeros((n_docs, n_labels), dtype=np.float32)
    return np.stack(out)  # (n_models, n_docs, n_labels)


def _ensemble_macro_f1(proba_stack: np.ndarray, member_idx, y_true: np.ndarray) -> float:
    """Macro-F1 of the mean-proba ensemble over `member_idx` (a fixed list)."""
    ens = proba_stack[member_idx].mean(axis=0)           # (n_docs, n_labels)
    preds = (ens >= 0.5).astype(int)
    return float(f1_score(y_true, preds, average="macro", zero_division=0))


def _perdoc_oracle_macro_f1(proba_stack: np.ndarray, oracle_mask: np.ndarray,
                            y_true: np.ndarray) -> float:
    """Ceiling: per doc, average the proba of ITS oracle-best-K models."""
    n_models, n_docs, n_labels = proba_stack.shape
    ens = np.zeros((n_docs, n_labels), dtype=np.float32)
    for i in range(n_docs):
        sel = np.nonzero(oracle_mask[i])[0]
        if len(sel) == 0:
            continue
        ens[i] = proba_stack[sel, i, :].mean(axis=0)
    preds = (ens >= 0.5).astype(int)
    return float(f1_score(y_true, preds, average="macro", zero_division=0))


def eval_dataset(dataset: str, seed: int = 0, tuned_dir: str = "experiments/router_tuning",
                 backend: str = "perturbed", model_pool: str = "default") -> dict:
    cfg = DATASET_REGISTRY[dataset]
    ds_def = DATASET_DEFAULTS.get(dataset, DSConfig())
    base_hp = HParams(top_labels=ds_def.top_labels, subset_size=ds_def.subset_size,
                      val_ratio=0.40, seed=seed, router_backend=backend)
    base_hp.max_test_docs = ds_def.max_test_docs

    # apply tuned config if available
    tuned_path = Path(tuned_dir) / f"router_tuned_{dataset}_s{seed}.json"
    tuned = {}
    if tuned_path.exists():
        best = json.loads(tuned_path.read_text()).get("best", {})
        tuned = {k: best[k] for k in ("k_models", "router_sigma", "router_lr", "router_epochs")
                 if k in best}
        base_hp = replace(base_hp, **{k: (int(v) if k in ("k_models", "router_epochs") else float(v))
                                      for k, v in tuned.items()})

    t0 = time.time()
    print(f"[eval:{dataset}] loading data + fitting pool …", flush=True)
    _data = load_data(cfg, base_hp)
    train_X, val_X, test_X = _data[0], _data[1], _data[2]
    train_y, val_y, test_y = _data[3], _data[4], _data[5]

    pool = init_models(len(_data[6]), lora_model_name=None,
                       drop_tfidf=(model_pool == "embedding"))
    train_models(pool, train_X, train_y, val_X, val_y)
    names = list(pool.keys())
    n_models = len(names)
    k = min(int(base_hp.k_models), n_models)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    net, tfidf_vec, svd = _train_selection_net(pool, train_X, train_y, base_hp, k, device,
                                               model_names=names, verbose=False)

    def _scores(X):
        Xt = torch.from_numpy(svd.transform(tfidf_vec.transform(X)).astype(np.float32)).to(device)
        with torch.no_grad():
            return net.get_scores(Xt).cpu().numpy(), Xt

    val_scores, _ = _scores(val_X)
    test_scores, Xtest_t = _scores(test_X)

    oracle_val = _build_multi_label_oracle_mask(pool, val_X, val_y, k)
    oracle_test = _build_multi_label_oracle_mask(pool, test_X, test_y, k)
    m_val = router_oracle_metrics(val_scores, oracle_val, k)
    m_test = router_oracle_metrics(test_scores, oracle_test, k)

    # router's FIXED global selection S (applied to every test doc)
    sel_idx, _ = FinalSelector.select(net, Xtest_t, k)
    sel_set = set(int(i) for i in sel_idx)
    selected_models = [names[i] for i in sel_idx]

    # fixed S vs per-doc oracle on test
    order_test = np.argsort(-test_scores, axis=1)[:, :k]  # router per-doc top-k (pre-aggregation)
    perdoc_hits, matches = [], 0
    distinct = set()
    for i in range(len(test_X)):
        oset = set(np.nonzero(oracle_test[i])[0].tolist())
        if oset:
            perdoc_hits.append(len(sel_set & oset) / k)
        topk_i = tuple(sorted(order_test[i].tolist()))
        distinct.add(topk_i)
        if set(order_test[i].tolist()) == sel_set:
            matches += 1

    # ceiling gap (macro-F1): fixed selection vs per-doc oracle ensemble on test
    proba_test = _model_probas(pool, names, test_X)
    fixed_f1 = _ensemble_macro_f1(proba_test, list(sel_idx), test_y)
    ceiling_f1 = _perdoc_oracle_macro_f1(proba_test, oracle_test, test_y)

    out = {
        "dataset": dataset, "seed": seed, "backend": backend,
        "k": k, "n_models": n_models, "n_test": len(test_X),
        "tuned_config": tuned,
        "selected_models_fixed": selected_models,
        "val": m_val,
        "test": m_test,
        "val_test_gap": {
            "hit_at_k_drop": round(m_val["oracle_hit_at_k"] - m_test["oracle_hit_at_k"], 4),
            "mrr_drop": round(m_val["mrr"] - m_test["mrr"], 4),
        },
        "fixed_vs_perdoc_on_test": {
            "mean_overlap_hit": round(float(np.mean(perdoc_hits)) if perdoc_hits else 0.0, 4),
            "frac_docs_matching_fixed_topk": round(matches / max(1, len(test_X)), 4),
            "n_distinct_perdoc_topk_sets": len(distinct),
        },
        "ceiling_macro_f1": {
            "fixed_selection": round(fixed_f1, 4),
            "perdoc_oracle": round(ceiling_f1, 4),
            "gap": round(ceiling_f1 - fixed_f1, 4),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(f"[eval:{dataset}] val hit@{k}={m_val['oracle_hit_at_k']} MRR={m_val['mrr']} | "
          f"test hit@{k}={m_test['oracle_hit_at_k']} MRR={m_test['mrr']} | "
          f"fixed-vs-perdoc overlap={out['fixed_vs_perdoc_on_test']['mean_overlap_hit']} | "
          f"ceiling gap(macroF1)={out['ceiling_macro_f1']['gap']}", flush=True)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Router test-set oracle-gap evaluation.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="perturbed", choices=["custom", "perturbed"])
    ap.add_argument("--tuned", default="experiments/router_tuning")
    ap.add_argument("--model_pool", default="default", choices=["default", "embedding"])
    ap.add_argument("--out", default="experiments/router_eval")
    args = ap.parse_args(argv)

    out = eval_dataset(args.dataset, seed=args.seed, tuned_dir=args.tuned,
                       backend=args.backend, model_pool=args.model_pool)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"router_eval_{args.dataset}_s{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
