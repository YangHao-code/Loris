"""Unified CLI to run LORIS paper baselines and write comparable result JSONs.

Run from the repo root, e.g.::

    HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1 \
      python -m loris.baselines.run_baselines --baseline indiv_ms --dataset aapd

    # everything on one dataset:
    python -m loris.baselines.run_baselines --baseline all --dataset bgc

    # GPT4 (needs OPENAI_API_KEY; subsample test to control cost):
    OPENAI_API_KEY=sk-... python -m loris.baselines.run_baselines \
      --baseline gpt4 --dataset reuters21578 --max_test 1000

Each (baseline, dataset, seed) writes ``<out_dir>/<baseline>__<dataset>__seedN.json``
with the schema in :func:`loris.baselines.common.write_result`.  Aggregate with
``aggregate_baselines.py``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

from loris.baselines import common as C

log = logging.getLogger("loris.baselines")

# module name -> imported lazily; each exposes BASELINES = {name: fn}
# Order matters: later modules OVERRIDE earlier same-named baselines. The
# ``*_orig``/``*_reef`` modules drive the AUTHORS' original source code (vendored
# under refs/) and intentionally override the first-pass ports of the same name.
# See refs/ and BASELINE_PLAN §4. RulePrompt keeps the first-pass reimpl (its
# original openprompt+torch1.12 stack is incompatible with this Blackwell GPU).
_MODULES = [
    "selectors",
    "encoder_head",
    "self_pretrain",
    "gpt4_zeroshot",
    "snuba",
    "ruleprompt",
    "hitl",
    "pattern_select",
    # ── original-source adapters (override the first-pass ports above) ──
    "snuba_reef",        # Snuba  <- HazyResearch/reef
    "weshap_orig",       # WeShap <- Gnaiqing/WeShap
    "besra_orig",        # BESRA  <- davidtw999/BESRA
    "rulecleaner_orig",  # LocalBoost slot -> RuleCleaner (JayLi2018/RuleCleanerKDD25)
    "comal_orig",        # RAL slot -> CoMAL (chengzju/CoMAL); also drops first-pass 'ral'
]


def build_registry() -> dict:
    """Merge every module's BASELINES dict; skip modules that fail to import."""
    registry: dict = {}
    for mod in _MODULES:
        try:
            m = importlib.import_module(f"loris.baselines.{mod}")
        except Exception as exc:  # a broken/optional module shouldn't kill the CLI
            log.warning("Could not import loris.baselines.%s: %s", mod, exc)
            continue
        bl = getattr(m, "BASELINES", {})
        for name, fn in bl.items():
            if name in registry:
                log.warning("Duplicate baseline name '%s' (from %s) — overwriting.", name, mod)
            registry[name] = fn
    # RAL released no usable code and was substituted by CoMAL (reported under its
    # own name). Drop the first-pass 'ral' port so the RAL slot is CoMAL only.
    if "comal" in registry:
        registry.pop("ral", None)
    return registry


# kwargs that the CLI forwards to baseline fns (only the non-None ones)
_PASSTHROUGH = ("k", "budget", "batch", "top_k", "max_test", "rounds", "iters", "conf", "seed_frac")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run LORIS paper baselines.")
    p.add_argument("--baseline", required=True,
                   help="baseline name, comma-list, or 'all' / 'list'")
    p.add_argument("--dataset", default=None,
                   help="one of reuters21578/aapd/rcv1/bgc/arxiv (omit only for --baseline list)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="experiments/baselines")
    # split overrides (default = canonical caps in common.DATASET_DEFAULTS)
    p.add_argument("--top_labels", type=int, default=None)
    p.add_argument("--subset_size", type=int, default=None)
    p.add_argument("--max_test_docs", type=int, default=None)
    # baseline-specific kwargs (forwarded when set)
    p.add_argument("--k", type=int, default=None, help="#models for selectors (Group A)")
    p.add_argument("--budget", type=int, default=None, help="annotation budget for BESRA/RAL")
    p.add_argument("--batch", type=int, default=None, help="AL batch size")
    p.add_argument("--top_k", type=int, default=None, help="#patterns for Group C")
    p.add_argument("--max_test", type=int, default=None, help="test subsample for GPT4/RulePrompt")
    p.add_argument("--rounds", type=int, default=None, help="self-training / iter rounds")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--conf", type=float, default=None, help="pseudo-label confidence")
    p.add_argument("--seed_frac", type=float, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    registry = build_registry()

    if args.baseline == "list":
        print("Available baselines:")
        for name in sorted(registry):
            print("  ", name)
        return 0

    if not args.dataset:
        p.error("--dataset is required (unless --baseline list)")

    if args.baseline == "all":
        names = sorted(registry)
    else:
        names = [b.strip() for b in args.baseline.split(",") if b.strip()]

    unknown = [n for n in names if n not in registry]
    if unknown:
        p.error(f"unknown baseline(s) {unknown}; run --baseline list")

    kwargs = {k: getattr(args, k) for k in _PASSTHROUGH if getattr(args, k) is not None}

    log.info("Loading split: %s", args.dataset)
    split = C.load_split(
        args.dataset,
        top_labels=args.top_labels,
        subset_size=args.subset_size,
        max_test_docs=args.max_test_docs,
    )
    log.info("train=%d val=%d test=%d labels=%d",
             len(split.train_X), len(split.val_X), len(split.test_X), split.n_labels)

    rc = 0
    for name in names:
        fn = registry[name]
        log.info("=== baseline: %s | dataset: %s | seed: %d ===", name, args.dataset, args.seed)
        try:
            with C.Timer() as t:
                out = fn(split, seed=args.seed, **kwargs)
        except NotImplementedError as exc:
            log.warning("SKIP %s on %s: %s", name, args.dataset, exc)
            continue
        except Exception as exc:
            log.exception("FAILED %s on %s: %s", name, args.dataset, exc)
            rc = 1
            continue
        path = C.write_result(
            args.out_dir, name, args.dataset, out,
            seed=args.seed,
            n_annotations=int(out.get("n_annotations", 0)),
            n_test=len(split.test_X),
            wall_sec=t.sec,
            extra=out.get("extra", {}),
        )
        log.info("%s/%s — macro=%.4f micro=%.4f (%.1fs) -> %s",
                 name, args.dataset, out.get("macro_f1", float("nan")),
                 out.get("micro_f1", float("nan")), t.sec, path.name)
    return rc


if __name__ == "__main__":
    sys.exit(main())
