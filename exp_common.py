#!/usr/bin/env python
"""Shared infrastructure for the LORIS paper experiment drivers (Exp 1-5).

Every driver shells `python -m loris ...` per cell with the *canonical* LORIS
configuration the paper uses, parses the run's metrics.json, writes a uniform
per-cell JSON, and purges the heavy caches. Keeping this in one place guarantees
all experiments use identical params (predicates=full, from-scratch blank,
per-cluster tuned router, per-dataset caps) — the settings the user pinned.
"""
from __future__ import annotations

import glob as _glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from loris.baselines.common import DATASET_DEFAULTS

ROOT = Path(__file__).resolve().parent

# The 7 datasets for the paper (goodreads dropped; MIMIC unavailable).
DATASETS_7 = ["reuters21578", "aapd", "rcv1", "bgc", "arxiv", "pubmed", "hupd"]

# Two LoRA SLMs for the model pool. Llama-3-8B is gated (application pending), so
# this first pass uses Mistral-7B + Qwen2-7B; swap in Llama-3 once access lands.
LORA_BOTH = "mistralai/Mistral-7B-v0.1,Qwen/Qwen2-7B"

# Canonical LORIS config (the params the user emphasised). Per-cluster tuned
# router + full-mode predicates + label-from-scratch + bounded chase BO.
CANON_FLAGS = [
    "--two_val", "--rule_strategy", "batch",
    "--cluster_model_selection", "router", "--router_backend", "perturbed",
    "--batch_metric_mode", "global_macro",
    "--track1_baseline", "blank", "--pattern_mode", "full",
    "--max_trials", "40", "--no_adaptive_trials",
]


def caps(ds: str) -> list:
    """Per-dataset top_labels + subset/test caps from DATASET_DEFAULTS."""
    d = DATASET_DEFAULTS[ds]
    ex = ["--top_labels", str(d.top_labels)]
    if d.subset_size:
        ex += ["--subset_size", str(d.subset_size)]
    if d.max_test_docs:
        ex += ["--max_test_docs", str(d.max_test_docs)]
    return ex


def tuned_router_flags(ds: str, seed: int = 0) -> list:
    """Per-dataset oracle-tuned router hyperparameters (k_models/sigma/lr/epochs).

    Reads experiments/router_tuning/router_tuned_<ds>_s<seed>.json. Falls back to
    defaults (empty) if absent.
    """
    tp = ROOT / "experiments" / "router_tuning" / f"router_tuned_{ds}_s{seed}.json"
    if not tp.exists():
        return []
    try:
        best = json.loads(tp.read_text()).get("best", {})
    except Exception:
        return []
    flags = []
    if "k_models" in best:
        flags += ["--k_models", str(int(best["k_models"]))]
    if "router_sigma" in best:
        flags += ["--router_sigma", str(best["router_sigma"])]
    if "router_lr" in best:
        flags += ["--router_lr", str(best["router_lr"])]
    if "router_epochs" in best:
        flags += ["--router_epochs", str(int(best["router_epochs"]))]
    return flags


def base_env(gpu: str | None = None) -> dict:
    env = dict(os.environ,
               HF_HOME=os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache"),
               HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"),
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    else:
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _purge(exp_dir: Path) -> None:
    """Reclaim disk: drop the ~530MB model_pool.pkl + embedding .npy caches."""
    for pat in ("**/model_pool.pkl", "**/*.npy"):
        for f in _glob.glob(str(exp_dir / pat), recursive=True):
            try:
                os.remove(f)
            except Exception:
                pass


def run_cell(ds: str, extra_flags: list, exp_dir: Path,
             *, use_tuned_router: bool = True, env: dict | None = None,
             timeout_s: int | None = None) -> dict:
    """Run one LORIS cell; return {rc, wall_sec, metrics, run_dir}.

    extra_flags are appended AFTER CANON_FLAGS so argparse last-wins lets a driver
    override anything (e.g. add --disable_incremental, --label_budget, --pool_models).
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "loris", "--dataset", ds,
           *CANON_FLAGS, *caps(ds)]
    if use_tuned_router:
        cmd += tuned_router_flags(ds)
    cmd += ["--seed", "0", "--exp_dir", str(exp_dir), *extra_flags]
    if env is None:
        env = base_env()
    t0 = time.time()
    with open(exp_dir / "run.log", "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        try:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 env=env, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            lf.write(f"\n[exp_common] TIMEOUT after {timeout_s}s\n")
            rc = 124
    wall = time.time() - t0
    metrics, run_dir = {}, None
    cands = sorted(exp_dir.glob(f"{ds}_*/metrics.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        run_dir = cands[0].parent
        try:
            metrics = json.loads(cands[0].read_text())
        except Exception:
            metrics = {}
    _purge(exp_dir)
    return {"rc": rc, "wall_sec": round(wall, 1), "metrics": metrics,
            "run_dir": str(run_dir) if run_dir else None}


def sweep_json(run_dir: str | None, name: str = "rill_budget_sweep.json") -> dict:
    """Load a per-run auxiliary JSON (e.g. the RILL budget sweep) if present."""
    if not run_dir:
        return {}
    p = Path(run_dir) / name
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def parse_datasets(s: str) -> list:
    return [d.strip() for d in s.split(",") if d.strip()]
