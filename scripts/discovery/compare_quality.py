#!/usr/bin/env python3
"""Prepare isolated quality-selector comparisons, then optionally learn rules.

Default: save a readable PLAN.json and commands, without fitting or BO.
--execute: fit selectors, freeze Si, bind the SAME discovery context/patterns,
           and run the real input checks for every case.
--execute --execute-bo: additionally run rule learning at the declared budget.
"""

import argparse
import copy
import itertools
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from loris.discovery.manifest import bound, file_record, read_json, load_inputs
from loris.discovery.quality_selection import UTILITIES, SCORERS, AGGREGATIONS, select_quality


def make_plan(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("comparison output exists; preserve it and choose a new directory")
    if args.execute_bo and not args.execute:
        raise ValueError("--execute-bo requires --execute")
    if args.workers < 1 or args.bo_trials < 1 or args.k < 1:
        raise ValueError("K, worker count and per-cluster BO budget must be positive")
    for field in ("utilities", "scorers", "aggregations"):
        if len(getattr(args, field)) != len(set(getattr(args, field))):
            raise ValueError("duplicate comparison " + field)
    selector = Path(args.selector_manifest).resolve(strict=True)
    quality = Path(args.quality_inputs).resolve(strict=True)
    base_path = Path(args.base_manifest).resolve(strict=True)
    base = read_json(base_path)
    if base["run"]["initializer_slots"] > min(30, args.bo_trials):
        raise ValueError("base initializer_slots exceeds requested BO budget; choose an explicit compatible base manifest")
    # Existing model_selection is intentionally replaced below. Its historical
    # implementation does not need to be rerun to use this template.
    template = copy.deepcopy(base)
    for name in ("context", "pattern_supply", "supervision_exposure"):
        template[name] = file_record(bound(base[name], base_path.parent))
    template["dataset"]["split_protocol"] = file_record(bound(base["dataset"]["split_protocol"], base_path.parent))
    if template.get("lexical_cache_dir"):
        template["lexical_cache_dir"] = str((base_path.parent / template["lexical_cache_dir"]).resolve())
    cases = []
    for utility, scorer, aggregation in itertools.product(args.utilities, args.scorers, args.aggregations):
        name = "__".join((utility, scorer, aggregation))
        folder = out / name
        options = dict(utility=utility, scorer=scorer, aggregation=aggregation, k=args.k,
                       folds=args.folds, steps=args.steps, seed=args.seed, threads=args.threads,
                       device=args.device, max_combinations=args.max_combinations)
        selection = [sys.executable, "-m", "loris.discovery", "select-quality", "fit",
                     "--manifest", str(selector), "--quality-inputs", str(quality),
                     "--output", str(folder / "selector")]
        for key, value in options.items():
            selection += ["--" + key.replace("_", "-"), str(value)]
        check = [sys.executable, "-m", "loris.discovery", "check", "--manifest", str(folder / "discovery.json")]
        learn = [sys.executable, "-m", "loris.discovery", "learn-rules", "--manifest", str(folder / "discovery.json"),
                 "--output", str(folder / "rules")]
        cases.append(dict(name=name, options=options, selection_command=selection,
                          check_command=check, rule_command=learn))
    return out, template, dict(schema="loris_quality_comparison_plan_v1", cases=cases,
        selector_manifest=file_record(selector), quality_inputs=file_record(quality),
        base_manifest=file_record(base_path), workers=args.workers, budget_per_cluster=args.bo_trials,
        rule_seed=args.seed, execute_selectors=args.execute, execute_BO=args.execute_bo,
        distinction="new utility/aggregation research controls; not proven paper results",
        invariants="same expert pool, text clusters, pattern supply, K, labels, roles and rule budget",
        full_pool_cost="cached expert predictions have already incurred upstream inference cost",
        scope="rule discovery uses the base manifest's declared development scoring; selector fitting uses fit truth only")


def run_comparison(args):
    from loris._engine.rlearner import immutable_json
    from loris.discovery.quality_selection import load_quality_inputs

    out, template, plan = make_plan(args)
    if args.execute:
        # Reject an old 7-expert/fullD context paired with 16-expert/subset inputs
        # before doing any selector fitting. Pattern checks also run after binding.
        import pickle
        from loris.discovery.manifest import validate_context

        data, protocol, arrays, interpretation, _, _ = load_quality_inputs(args.selector_manifest, args.quality_inputs)
        with Path(template["context"]["path"]).open("rb") as stream:
            context = pickle.load(stream)
        validate_context(context)
        import numpy as np

        fit = context["fit_indices"]
        if (data["row_ids"] != [context["row_ids"][int(i)] for i in fit]
                or data["labels"] != context["labels"]
                or interpretation["row_ids"] != context["row_ids"]
                or protocol.oracle_policy != template["oracle_policy"]
                or not np.array_equal(arrays["truth"], context["source_gold"]["fit"])):
            raise ValueError("comparison context differs from selector fit/full document, label or protocol axes")
        for alias in data["aliases"]:
            for kind in ("probabilities", "decisions"):
                if alias not in context["model_" + kind] or not np.array_equal(interpretation[kind][alias], context["model_" + kind][alias]):
                    raise ValueError("comparison context lacks matching full-axis expert interpretation: " + alias)
        if args.workers > len(np.unique(arrays["assignments"])):
            raise ValueError("BO worker count exceeds the number of text clusters")
        del context, arrays, interpretation
    out.mkdir(parents=True, exist_ok=False)
    immutable_json(out / "PLAN.json", plan)
    lines = ["# Review PLAN.json before execution. Each case gets a fresh directory.",
             "# The comparison runner creates discovery.json after freezing the selector."]
    for case in plan["cases"]:
        lines += ["", "# " + case["name"], shlex.join(case["selection_command"]),
                  "# Bind selector/COMPLETE.json into discovery.json (performed by this script).",
                  shlex.join(case["check_command"]), "# Optional BO:", "# " + shlex.join(case["rule_command"])]
    (out / "COMMANDS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.execute:
        return dict(status="planned_only", cases=len(plan["cases"]), output=str(out),
                    new_fits=0, new_BO=0, next="review plan; rerun with --execute and a NEW output directory")
    summaries = []
    for case in plan["cases"]:
        folder = out / case["name"]
        selected = select_quality(args.selector_manifest, args.quality_inputs, folder / "selector", **case["options"])
        manifest = copy.deepcopy(template)
        manifest.update(model_selection=file_record(folder / "selector/COMPLETE.json"), selection_strategy="quality")
        manifest["run"].update(seed=args.seed, workers=args.workers, budget_per_cluster=args.bo_trials)
        immutable_json(folder / "discovery.json", manifest)
        inputs = load_inputs(folder / "discovery.json")
        immutable_json(folder / "CHECK.json", dict(status="checked", Si=inputs["si"],
                       summary=inputs["input_binding"]["summary"], expert_union=selected["expert_union"]))
        summary = dict(case=case["name"], status="selector_and_inputs_ready", expert_union=selected["expert_union"],
                       manifest=str(folder / "discovery.json"), rule_command=case["rule_command"])
        if args.execute_bo:
            from loris.discovery.runner import run_discovery
            receipt = run_discovery(output=folder / "rules", **inputs)
            summary.update(status="rule_discovery_complete", rule_receipt=file_record(folder / "rules/COMPLETE.json"),
                           complete_BO_trials=receipt["complete_BO_trials"])
        summaries.append(summary)
    result = dict(status="complete", stage="rule_discovery" if args.execute_bo else "selectors_and_checks",
                  cases=summaries, output=str(out))
    immutable_json(out / "COMPLETE.json", result)
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("selector-manifest", "quality-inputs", "base-manifest", "output"):
        cli.add_argument("--" + name, required=True)
    cli.add_argument("--utilities", nargs="+", choices=UTILITIES, default=list(UTILITIES))
    cli.add_argument("--scorers", nargs="+", choices=SCORERS, default=["static_cluster", "ridge", "residual_mlp"])
    cli.add_argument("--aggregations", nargs="+", choices=AGGREGATIONS, default=list(AGGREGATIONS))
    for name, default in [("k",4), ("folds",3), ("steps",80), ("seed",42), ("threads",4), ("workers",4), ("bo-trials",200), ("max-combinations",100000)]:
        cli.add_argument("--" + name, type=int, default=default)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--execute-bo", action="store_true")
    try:
        print(json.dumps(run_comparison(cli.parse_args(argv)), indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="failed", error=type(error).__name__, message=str(error))), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
