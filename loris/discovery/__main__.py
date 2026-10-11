"""Stable CLI for rich rule discovery. Run --help before creating a new run."""

import argparse
import json
import sys


UPSTREAM_COMMANDS = {
    "staged-rules": ("staged.cli", "calibrated model foundation, original BO and optional rule replacement"),
    "admit-pool": ("continue_admission", "continue completed BO in a separate approximate-admission branch; no new BO"),
    "prepare-supervision": (
        "supervision",
        "freeze known train answers and masked supervision; no training",
    ),
    "model-pool": ("model_pool", "freeze/run the real dataset-specific model pool; inspect --help"),
    "upstream": ("upstream", "prepare fit features, model context and selector inputs"),
    "train-selector": (
        "selector",
        "launch the unchanged selector with deterministic CUDA workspace configured",
    ),
    "select-models": (
        "model_selection",
        "learned/indiv/random/all selection on the same fit inputs; inspect --help",
    ),
    "select-quality": (
        "quality_selection",
        "experimental Brier/threshold quality scoring and fit-only candidate sets; inspect --help",
    ),
    "supply-patterns": (
        "patterns",
        "build fit-only composite primitives and preserve raw inventory",
    ),
    "prewarm-patterns": ("prewarm", "warm original lexical cache and verify full-axis fit masks"),
}


def _execution_options(parser):
    parser.add_argument(
        "--manifest", required=True, help="SHA-bound JSON inputs and explicit discovery protocol"
    )
    parser.add_argument(
        "--seed", type=int, help="override search seed; recorded in the effective run identity"
    )
    parser.add_argument(
        "--bo-trials",
        dest="budget_per_cluster",
        type=int,
        help="BO trials PER cluster, including initialization slots",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="parallel independent cluster searches; admission order stays fixed",
    )
    parser.add_argument(
        "--initialization-workers", type=int,
        help="parallel independent fit initialization; separate from BO workers",
    )
    parser.add_argument(
        "--gamma-conflict-policy", choices=("strict", "soft_fit_precision"),
        help="strict: any derived contradiction of validated fit Gamma invalidates the program; "
             "soft_fit_precision: validated Gamma wins, contradictions are blocked and counted, "
             "and a rule below the fit precision floor is scored -1 and not admitted",
    )
    parser.add_argument(
        "--fit-precision-floor", type=float,
        help="minimum agreements/(agreements+blocked) on validated fit Gamma (soft policy)",
    )
    parser.add_argument(
        "--initializer-candidate-archive",
        choices=("all", "evaluated_with_analytic_skips", "winners_and_counts"),
        help="retain all fit-grid rows, or record provably zero-positive text/model combinations as explicit analytic skips; same seeds",
    )
    parser.add_argument(
        "--initializer-slots",
        type=int,
        help="fit starts occupying existing startup slots, 0..min(30,budget)",
    )
    parser.add_argument(
        "--label-rotation",
        choices=("axis", "seeded_cluster_rotation"),
        help="explicit initialization label order; changing it requires a new run",
    )
    parser.add_argument(
        "--model-selection",
        dest="selection_strategy",
        choices=("learned", "quality", "indiv", "random", "all"),
        help="use authenticated fit arrays to replace Si; all is a full-pool capacity control",
    )
    parser.add_argument(
        "--candidate-domain",
        choices=("rich", "no_text", "local"),
        help="actual proposal capabilities; changed domains require new BO from empty Sigma",
    )
    parser.add_argument(
        "--proposal-strategy",
        choices=("accuracy_guided", "random_joint"),
        help="random_joint samples the same joint space without F1 learning; requires zero initializer slots and random_consistent admission",
    )
    parser.add_argument(
        "--admission-order",
        choices=("proposal_order", "standalone_score_desc", "random_consistent"),
        help="paper default: decreasing single-rule F1; proposal_order is a historical ablation",
    )
    parser.add_argument("--admission-policy", choices=("strict", "batch_prefilter"),
                        help="strict Algorithm 1, or explicit approximate frozen-prefix batch prefilter")
    parser.add_argument("--admission-workers", type=int, help="prefilter processes, default 8")
    parser.add_argument("--admission-batch-size", type=int, help="fixed approximation batch, default 8; independent of worker count")
    parser.add_argument(
        "--max-variables",
        type=int,
        help="finite document-variable axis, at least 2; values above 2 use full-axis TPE; local remains one variable",
    )
    parser.add_argument(
        "--context-probes", type=int, help="fit witness probes per structured proposal; positive"
    )
    parser.add_argument(
        "--acquisition-candidates", type=int, help="joint full-axis TPE acquisition draws; positive"
    )
    parser.add_argument(
        "--expected-body-size",
        type=float,
        help="positive sparse proposal prior; not a hard body-length cap",
    )


def parser():
    cli = argparse.ArgumentParser(
        description=(
            "Author-approved rich RLearner: optional ML context, text/model/label/attribute combinations, "
            "complete Chase and strict whole-program F1 admission. No online Aux or external labeling."
        )
    )
    sub = cli.add_subparsers(dest="command", required=True)
    for name, (_, description) in UPSTREAM_COMMANDS.items():
        sub.add_parser(name, help=description, add_help=False)
    prepare = sub.add_parser(
        "prepare-data",
        help="import source-bound CSV data with declared official/custom splits; no training",
    )
    prepare.add_argument(
        "--manifest", required=True, help="loris_dataset_source_v1 source manifest"
    )
    prepare.add_argument("--output", required=True, help="new output directory")
    aapd = sub.add_parser(
        "import-aapd", help="prepare the complete standard AAPD 54-label train/dev/test CSV release"
    )
    aapd.add_argument(
        "--source", required=True, help="directory containing train.csv, dev.csv, test.csv"
    )
    aapd.add_argument("--output", required=True, help="new output directory")
    check = sub.add_parser(
        "check", help="authenticate data/Si/Pi and role boundaries, without BO or training"
    )
    _execution_options(check)
    learn = sub.add_parser(
        "learn-rules", help="complete rich discovery from empty Sigma; preserve every BO trial"
    )
    _execution_options(learn)
    learn.add_argument("--output", required=True, help="isolated output directory")
    learn.add_argument(
        "--resume",
        action="store_true",
        help="resume only this package's matching input/source/configuration transaction",
    )
    refilter = sub.add_parser(
        "refilter", help="re-admit an authenticated historical proposal pool; no new BO"
    )
    refilter.add_argument(
        "--manifest", required=True, help="exactly matching data/model/Pi/Si manifest"
    )
    refilter.add_argument("--source", required=True, help="completed original discovery")
    refilter.add_argument(
        "--output", required=True, help="new independent output; no resume supported"
    )
    refilter.add_argument(
        "--min-gain",
        type=float,
        required=True,
        help="nonnegative strict F1 gain floor; zero retains the original 1e-12 tolerance",
    )
    state = sub.add_parser("status", help="read completion identity and actual stored trial counts")
    state.add_argument("--run", required=True)
    archive = sub.add_parser(
        "archive-index", help="index all completed BO records and back up each Optuna study"
    )
    archive.add_argument("--run", required=True)
    archive.add_argument(
        "--output", required=True, help="new archive directory outside the original run"
    )
    archive.add_argument(
        "--no-study-snapshot",
        action="store_true",
        help="create references only; original SQLite/WAL must remain available",
    )
    return cli


def main(argv=None):
    try:
        argv = list(sys.argv[1:] if argv is None else argv)
        if argv and argv[0] in UPSTREAM_COMMANDS:
            import importlib

            module = importlib.import_module("." + UPSTREAM_COMMANDS[argv[0]][0], __package__)
            return module.main(argv[1:])
        args = parser().parse_args(argv)
        if args.command in ("prepare-data", "import-aapd"):
            from .datasets import prepare_dataset, import_aapd

            result = (
                prepare_dataset(args.manifest, args.output)
                if args.command == "prepare-data"
                else import_aapd(args.source, args.output)
            )
        elif args.command in ("check", "learn-rules", "refilter"):
            from .manifest import load_inputs, RUN_FIELDS

            overrides = {
                key: getattr(args, key, None) for key in RUN_FIELDS | {"selection_strategy"}
            }
            if args.command == "refilter":
                # This command explicitly consumes historical proposal-order
                # archives. New-discovery defaults must not reinterpret them.
                overrides["admission_order"] = "proposal_order"
                overrides["proposal_strategy"] = "accuracy_guided"
            inputs = load_inputs(args.manifest, overrides)
            if args.command == "check":
                result = {
                    "status": "checked_inputs_not_executed",
                    "summary": inputs["input_binding"]["summary"],
                    "Si": inputs["si"],
                    "Pi_counts": inputs["input_binding"]["Pi_counts"],
                    "run": {k: inputs[k] for k in RUN_FIELDS},
                    "effective_manifest_id": inputs["input_binding"]["effective_manifest_id"],
                    "new_BO": 0,
                    "new_training": 0,
                    "new_Chase": 0,
                }
            elif args.command == "learn-rules":
                from .runner import run_discovery

                result = run_discovery(output=args.output, resume=args.resume, **inputs)
            else:
                from .refilter import refilter

                result = refilter(
                    source=args.source, output=args.output, min_gain=args.min_gain, **inputs
                )
        elif args.command == "status":
            from .archive import status

            result = status(args.run)
        else:
            from .archive import archive_index

            archive = archive_index(
                args.run, args.output, snapshot_studies=not args.no_study_snapshot
            )
            result = {
                "status": archive["status"],
                "output": args.output,
                "BO_trials": archive["completion"]["BO_trials"],
                "indexed_files": len(archive["files"]),
                "indexed_bytes": archive["indexed_bytes"],
                "study_snapshots": len(archive["study_snapshots"]),
                "archive_is_self_contained": False,
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(error).__name__, "reason": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
