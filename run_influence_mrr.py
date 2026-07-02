#!/usr/bin/env python
"""Exp-1c — Influence estimation (RDG): quality (MRR / hit@k) + time saving.

Two outputs → experiments/influence/:

1. quality.json — per-dataset router influence-estimation quality (oracle hit@k +
   MRR), aggregated from experiments/router_tuning/router_tuned_<ds>_s0.json (the
   tuner's held-out metric). No new runs needed.

2. rdg_timing.json — the "RDG saves XX% vs exact" claim, as a structural
   micro-benchmark on the RuleDependencyGraph: for rule DAGs of growing size we
   time RDG bounded-BFS influence estimation (trust_bfs_depth=2, as used in the
   RILL loop) against the EXACT full transitive-closure influence, and report the
   per-doc speedup + ranking agreement (Spearman-free MRR of RDG's top pick vs the
   exact ranking). Disclosed as a structural benchmark of the estimation operator.

Usage: python run_influence_mrr.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

from exp_common import ROOT, DATASETS_7, write_json

OUT = ROOT / "experiments" / "influence"
RT = ROOT / "experiments" / "router_tuning"


def aggregate_quality() -> dict:
    rows = []
    for ds in DATASETS_7:
        p = RT / f"router_tuned_{ds}_s0.json"
        if not p.exists():
            continue
        try:
            best = json.loads(p.read_text()).get("best", {})
        except Exception:
            continue
        rows.append({"dataset": ds,
                     "hit_at_k": best.get("oracle_hit_at_k"),
                     "mrr": best.get("mrr"),
                     "k_models": best.get("k_models")})
    return {"rows": rows}


# ── RDG bounded-BFS vs exact-closure timing benchmark ────────────────────────
def _rand(n, seed):
    # tiny deterministic LCG (Math.random / np.random unavailable-safe, seedable)
    x = (seed * 2654435761) & 0xFFFFFFFF
    out = []
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out.append(x / 0x7FFFFFFF)
    return out


def _build_adjacency(n_labels, fanout, seed):
    """Random DAG over labels: rule for label i may consume labels < i (acyclic)."""
    adj = {i: set() for i in range(n_labels)}          # producer-label -> consumer-labels
    r = _rand(n_labels * fanout, seed)
    idx = 0
    for i in range(n_labels):
        for _ in range(fanout):
            j = int(r[idx] * i) if i > 0 else 0
            idx += 1
            if j < i:
                adj[j].add(i)   # labeling j influences rules producing i
    return adj


def _exact_closure(adj, start):
    """Exact influence = full transitive closure (unbounded BFS to fixpoint)."""
    seen, frontier = set([start]), [start]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v); nxt.append(v)
        frontier = nxt
    return seen


def _bounded_bfs(adj, start, depth):
    """RDG estimate = bounded-depth BFS (the RILL trust_bfs_depth truncation)."""
    seen, frontier, d = set([start]), [start], 0
    while frontier and d < depth:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v); nxt.append(v)
        frontier = nxt; d += 1
    return seen


def rdg_timing() -> dict:
    rows = []
    for n_labels in (30, 60, 120, 240):
        adj = _build_adjacency(n_labels, fanout=4, seed=n_labels)
        # exact
        t0 = time.time()
        exact = {i: len(_exact_closure(adj, i)) for i in range(n_labels)}
        t_exact = time.time() - t0
        # RDG bounded (depth 2, as in the RILL loop)
        t0 = time.time()
        est = {i: len(_bounded_bfs(adj, i, depth=2)) for i in range(n_labels)}
        t_rdg = time.time() - t0
        # ranking agreement: does RDG's argmax match exact's argmax? (MRR of the
        # exact-best label within RDG's ranking)
        exact_best = max(exact, key=exact.get)
        rdg_order = sorted(est, key=est.get, reverse=True)
        rank = rdg_order.index(exact_best) + 1
        rows.append({"n_labels": n_labels,
                     "exact_sec": round(t_exact, 6), "rdg_sec": round(t_rdg, 6),
                     "speedup": round(t_exact / max(t_rdg, 1e-9), 2),
                     "time_saved_pct": round(100 * (1 - t_rdg / max(t_exact, 1e-9)), 1),
                     "mrr_top": round(1.0 / rank, 3)})
    return {"note": "structural micro-benchmark of the RDG influence operator: "
                    "bounded-BFS (depth=2, as in RILL) vs exact transitive closure",
            "rows": rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q = aggregate_quality()
    write_json(OUT / "quality.json", q)
    print("[influence] quality rows:", len(q["rows"]))
    for r in q["rows"]:
        print(f"  {r['dataset']:14s} hit@k={r['hit_at_k']} MRR={r['mrr']}")
    t = rdg_timing()
    write_json(OUT / "rdg_timing.json", t)
    print("[influence] RDG timing:")
    for r in t["rows"]:
        print(f"  n_labels={r['n_labels']:4d}  RDG saves {r['time_saved_pct']}%  "
              f"(speedup {r['speedup']}x, MRR_top {r['mrr_top']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
