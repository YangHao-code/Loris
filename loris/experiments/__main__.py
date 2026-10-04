"""Public experiment inventory, registry, and fixed-program evaluation CLI."""

import argparse
import json
import sys
from .evaluation import VARIANTS


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "aux-tuning":
        from .aux_tuning import main as tuning_main

        return tuning_main(argv[1:])
    if argv and argv[0] == "workflow":
        try:
            from .workflow import main as workflow_main

            print(json.dumps(workflow_main(argv[1:]), ensure_ascii=False, indent=2))
            return 0
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            print(
                json.dumps(dict(status="failed", error=type(error).__name__, reason=str(error))),
                file=sys.stderr,
            )
            return 2
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("aux-tuning", help="train-only Aux epoch and pseudo-label controls", add_help=False)
    sub.add_parser(
        "workflow", help="portable asset binding and audited experiment stages", add_help=False
    )
    plan = sub.add_parser(
        "prepare", help="write a matrix and scripts without launching experiments"
    )
    plan.add_argument("--assets", required=True)
    plan.add_argument("--output", required=True)
    run = sub.add_parser("run", help="execute one explicit ready experiment ID")
    run.add_argument("--matrix", required=True)
    run.add_argument("--id", required=True)
    run.add_argument("--dry-run", action="store_true")
    collect = sub.add_parser(
        "collect", help="aggregate actual receipts; leave unexecuted scores empty"
    )
    collect.add_argument("--matrix", required=True)
    collect.add_argument("--output", required=True)
    reg = sub.add_parser(
        "export-rules", help="copy exact ordered small programs into one directory"
    )
    reg.add_argument("--source-index", required=True)
    reg.add_argument("--output", required=True)
    check = sub.add_parser("check-rules")
    check.add_argument("--index", required=True)
    base = sub.add_parser(
        "baselines", help="score all models and fixed votes; select thresholds on dev only"
    )
    base.add_argument("--manifest", required=True)
    base.add_argument("--output", required=True)
    base.add_argument(
        "--threshold-grid", type=float, nargs="+", default=[i / 20 for i in range(1, 20)]
    )
    ev = sub.add_parser("evaluate", help="whole-program evaluation; failures retain null metrics")
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--rules")
    ev.add_argument("--output", required=True)
    ev.add_argument("--variant", choices=VARIANTS, default="full")
    ev.add_argument("--model", help="exact model alias for a complete single-model RDL candidate")
    ev.add_argument(
        "--calibrate", action="store_true", help="19-point per-label dev grid; models only"
    )
    args = p.parse_args(argv)
    try:
        if args.command == "prepare":
            from .matrix import prepare

            result = prepare(args.assets, args.output)
            print(
                json.dumps(
                    dict(
                        status="prepared",
                        experiments=len(result["experiments"]),
                        output=args.output,
                    )
                )
            )
            return 0
        if args.command == "collect":
            from .reporting import collect

            result = collect(args.matrix, args.output)
        elif args.command == "run":
            from .matrix import run

            result = run(args.matrix, args.id, args.dry_run)
        elif args.command == "export-rules":
            from .registry import export_rules

            result = export_rules(args.source_index, args.output)
        elif args.command == "check-rules":
            from .registry import check_rules

            result = check_rules(args.index)
        elif args.command == "baselines":
            from .evaluation import baselines

            r = baselines(args.manifest, args.output, args.threshold_grid)
            result = dict(status="complete", records=len(r["records"]), output=args.output)
        elif args.command == "evaluate":
            if bool(args.model) == bool(args.rules):
                raise ValueError("choose exactly one of --model or --rules")
            if args.calibrate and not args.model:
                raise ValueError("--calibrate requires --model")
            if args.model and args.variant != "full":
                raise ValueError("model programs use --variant full")
            from .evaluation import evaluate_program

            r = evaluate_program(
                args.manifest,
                args.rules,
                args.output,
                args.variant,
                args.model,
                calibrate_grid=[i / 20 for i in range(1, 20)] if args.calibrate else None,
            )
            result = dict(
                status=r["record"]["status"], metrics=r["record"]["metrics"], output=args.output
            )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as e:
        print(
            json.dumps(
                dict(status="failed", error=type(e).__name__, reason=str(e)), ensure_ascii=False
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
