#!/usr/bin/env python3
"""Plan or execute ONE explicit stage of full-inventory rule discovery.

No stage executes without --execute. Model fitting and BO have distinct stages;
the command planner itself does not import models or inspect hidden labels.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    "supervision",
    "pool-freeze",
    "pool-cpu",
    "pool-gpu",
    "pool-handoff",
    "features",
    "context",
    "selector-inputs",
    "selector",
    "patterns",
    "prewarm",
    "bind",
    "check",
    "bo",
    "status",
    "archive",
)


def record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def write_new(path, value):
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    path = Path(path)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(
                "existing stage artifact differs; choose a new task namespace: " + str(path)
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)


def configuration(path):
    path = Path(path).resolve(strict=True)
    config = json.loads(path.read_text())
    required = {
        "schema",
        "dataset",
        "prepared_manifest",
        "historical_exposure",
        "workdir",
        "seed",
        "fit_fraction",
        "pretrained_model",
        "feature_backbone",
        "backend_source",
        "gpu_lock",
        "pool_settings",
        "title_field",
        "semantic_metadata_fields",
        "relation_attributes",
        "n_clusters",
        "workers",
        "bo",
    }
    if required - set(config) or set(config) - required - {
        "selector_protocol",
        "prewarm",
        "disk_reserve_gib",
        "pattern_screening",
        "patterns_per_cluster",
        "relation_screening",
    }:
        raise ValueError("unknown or missing full discovery plan fields")
    pattern_screening(config)
    if "patterns_per_cluster" in config and (
            type(config["patterns_per_cluster"]) is not int or config["patterns_per_cluster"] < 1):
        raise ValueError("patterns_per_cluster must be a positive integer")
    if "patterns_per_cluster" in config and pattern_screening(config) != "discriminative":
        raise ValueError("patterns_per_cluster applies only to discriminative pattern screening")
    if "relation_screening" in config:
        relation = config["relation_screening"]
        # Full threshold validation happens when the discovery manifest is bound.
        if (not isinstance(relation, dict) or relation.get("policy") not in RELATION_SCREENING_POLICIES
                or set(relation) - {"policy"} - set(RELATION_SCREENING_THRESHOLDS)
                or any(type(relation[k]) not in (int, float) for k in set(relation) - {"policy"})):
            raise ValueError("relation_screening must be {\"policy\": \"none\"} or the discriminative "
                             "blocked-equality policy with numeric thresholds")
    if config["schema"] != "loris_full_inventory_discovery_plan_v1":
        raise ValueError("wrong discovery plan schema")
    for key in (
        "prepared_manifest",
        "historical_exposure",
        "workdir",
        "pretrained_model",
        "feature_backbone",
        "backend_source",
        "gpu_lock",
        "pool_settings",
        "selector_protocol",
    ):
        if key in config:
            config[key] = str((path.parent / config[key]).resolve())
    if not isinstance(config["dataset"], str) or not config["dataset"]:
        raise ValueError("explicit separately named raw-label task required")
    for key in ("seed", "n_clusters", "workers"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError("invalid " + key)
    if config["workers"] > config["n_clusters"]:
        raise ValueError("workers exceed cluster count")
    attrs = config["relation_attributes"]
    if (
        not isinstance(attrs, list)
        or any(not isinstance(a, str) or not a for a in attrs)
        or len(set(attrs)) != len(attrs)
    ):
        raise ValueError(
            "explicit unique semantic relation_attributes required; provenance fields are not inferred"
        )
    fields = config["semantic_metadata_fields"]
    if (
        not isinstance(fields, list)
        or any(not isinstance(field, str) or not field for field in fields)
        or len(set(fields)) != len(fields)
    ):
        raise ValueError("explicit unique semantic_metadata_fields, possibly [], required")
    return config


def disk_preflight(config, paths):
    """Account for known arrays plus declared headroom, not a BO-size theorem."""
    data = json.loads(Path(config["prepared_manifest"]).read_text())
    n, k = data["counts"]["train"] + data["counts"]["dev"], len(data["labels"])
    fit = (
        data["counts"]["train"]
        * config["fit_fraction"]["numerator"]
        // config["fit_fraction"]["denominator"]
    )
    models = (
        "tfidf_svm_unigram",
        "tfidf_svm_bigram",
        "tfidf_lr_bigram",
        "textcnn",
        "bilstm",
        "encoder_mlp",
        "aux_dynamic",
    )
    # .npy and context pickle store dense arrays without compression. NPZ is
    # budgeted by its raw arrays; its actual compression ratio is unknown.
    artifacts = [
        (
            paths["features"] / "features.npy",
            n * 384 * 4,
            True,
            paths["features"] / "FEATURES_COMPLETE.json",
        ),
        (
            paths["context"] / "discovery_context.pkl",
            n * k * (7 * 5 + 2),
            True,
            paths["context"] / "COMPLETE.json",
        ),
        (
            paths["selector_inputs"] / "SELECTOR_INPUTS.npz",
            n * k * 7 * 5 + fit * (384 * 4 + 8 * k),
            False,
            paths["selector_inputs"] / "COMPLETE.json",
        ),
    ]
    artifacts += [
        (
            paths["pool"] / name / "probabilities.npz",
            n * k * 4,
            False,
            paths["pool"] / name / "COMPLETE.json",
        )
        for name in models
    ]
    remaining = []
    for path, size, plain, receipt in artifacts:
        present = path.stat().st_size if path.is_file() else 0
        completed = path.is_file() and receipt.is_file()
        remaining.append(
            dict(
                path=str(path),
                raw_array_bytes=size,
                already_stored_bytes=present,
                completion_receipt_present=completed,
                remaining_raw_budget=0 if completed else max(0, size - present),
                uncompressed=plain,
            )
        )
    reserve = config.get("disk_reserve_gib", 10)
    if type(reserve) not in (int, float) or not 0 < reserve < float("inf"):
        raise ValueError("positive finite disk_reserve_gib required")
    target = paths["root"]
    while not target.exists():
        target = target.parent
    free = shutil.disk_usage(target).free
    forecast = sum(row["remaining_raw_budget"] for row in remaining)
    result = dict(
        status=(
            "sufficient_for_declared_preflight"
            if free >= forecast + int(reserve * 2**30)
            else "insufficient_declared_headroom"
        ),
        filesystem=str(target),
        free_bytes=free,
        train_dev_rows=n,
        labels=k,
        guaranteed_uncompressed_array_lower_bound=sum(
            row["remaining_raw_budget"] for row in remaining if row["uncompressed"]
        ),
        remaining_raw_array_budget=forecast,
        reserve_bytes=int(reserve * 2**30),
        artifacts=remaining,
        limits="NPZ budget uses raw bytes, not a physical lower bound. Existing completion paths only estimate remaining writes; each consuming stage authenticates receipts. Texts/checkpoints/raw NLP/BO/study snapshots are covered only by explicit headroom; actual BO storage is unbounded by this estimate.",
    )
    return result


def layout(config, *, variables=2, models="learned", budget=None, search_seed=None):
    if type(variables) is not int or variables < 2 or models not in ("learned", "all"):
        raise ValueError("two or more variables and learned/all model policy required")
    budget = config["bo"]["budget_per_cluster"] if budget is None else budget
    seed = config["seed"] if search_seed is None else search_seed
    if type(budget) is not int or budget < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("explicit positive per-cluster BO budget and uint32 search seed required")
    slots = config["bo"].get("initializer_slots", 0)
    if type(slots) is not int or not 0 <= slots <= min(30, budget):
        raise ValueError("initializer_slots must be within the existing startup and trial budget")
    root = Path(config["workdir"])
    variant = "v%d_%s_seed%d_B%d" % (variables, models, seed, budget)
    if slots != min(30, budget):
        variant += "_I%d" % slots
    return dict(
        root=root,
        supervision=root / "supervision/SUPERVISION_MANIFEST.json",
        pool=root / "model_pool",
        features=root / "features",
        context=root / "context",
        selector_inputs=root / "selector_inputs",
        selector=root / "selector",
        # Separate directories: a plan never reads the other screening's supply.
        patterns=root / ("patterns_full" if pattern_screening(config) == "entropy_full_inventory"
                         else "patterns_discriminative"),
        prewarm=root / ("prewarm_full" if pattern_screening(config) == "entropy_full_inventory"
                        else "prewarm_discriminative"),
        manifest=root / "discovery_manifests" / (variant + ".json"),
        run=root / "rules" / variant,
        archive=root / "archives" / variant,
        variables=variables,
        models=models,
        budget=budget,
        search_seed=seed,
        initializer_slots=slots,
        variant=variant,
    )


def commands(config, paths, stage, *, resume=False):
    base = [sys.executable, "-m", "loris.discovery"]

    def command(*args):
        return base + [str(value) for value in args]

    sup, root = paths["supervision"], paths["root"]
    features, context = paths["features"], paths["context"]
    if stage == "supervision":
        return [
            command(
                "prepare-supervision",
                "--prepared-manifest",
                config["prepared_manifest"],
                "--historical-exposure",
                config["historical_exposure"],
                "--output",
                sup.parent,
                "--seed",
                config["seed"],
                "--fraction-numerator",
                config["fit_fraction"]["numerator"],
                "--fraction-denominator",
                config["fit_fraction"]["denominator"],
            )
        ]
    if stage == "pool-freeze":
        return [
            command(
                "model-pool",
                "freeze",
                "--supervision-manifest",
                sup,
                "--output",
                paths["pool"],
                "--pretrained-model",
                config["pretrained_model"],
                "--backend-source",
                config["backend_source"],
                "--gpu-lock",
                config["gpu_lock"],
                "--settings",
                config["pool_settings"],
            )
        ]
    if stage in ("pool-cpu", "pool-gpu", "pool-handoff"):
        return [
            command(
                "model-pool", "run", "--output", paths["pool"], "--phase", stage.split("-", 1)[1]
            )
        ]
    if stage == "features":
        return [
            command(
                "upstream",
                "features",
                "--supervision",
                sup,
                "--output",
                features,
                "--backbone",
                config["feature_backbone"],
                "--device",
                "cuda",
                "--gpu-lock",
                config["gpu_lock"],
                "--n-clusters",
                config["n_clusters"],
                "--seed",
                config["seed"],
            )
        ]
    if stage == "context":
        return [
            command(
                "upstream",
                "context",
                "--supervision",
                sup,
                "--features",
                features / "FEATURES_COMPLETE.json",
                "--pool",
                paths["pool"] / "HANDOFF.json",
                "--title-field",
                config["title_field"],
                "--semantic-metadata-fields",
                json.dumps(config["semantic_metadata_fields"]),
                "--output",
                context,
            )
        ]
    if stage == "selector-inputs":
        args = command(
            "upstream",
            "selector",
            "--supervision",
            sup,
            "--features",
            features / "FEATURES_COMPLETE.json",
            "--context",
            context / "COMPLETE.json",
            "--output",
            paths["selector_inputs"],
        )
        if config.get("selector_protocol"):
            args += ["--protocol", config["selector_protocol"]]
        return [args]
    if stage == "selector":
        args = command(
            "train-selector",
            "--manifest",
            paths["selector_inputs"] / "select_models.manifest.json",
            "--output",
            paths["selector"],
        )
        return [args + (["--resume"] if resume else [])]
    if stage == "patterns":
        common = [
            "supply-patterns",
            "--supervision-manifest",
            sup,
            "--cluster-arrays",
            features / "CLUSTER_ARRAYS.npz",
            "--clustering",
            features / "CLUSTERING.json",
            "--output",
            paths["patterns"],
            "--title-field",
            config["title_field"],
            "--semantic-metadata-fields",
            json.dumps(config["semantic_metadata_fields"]),
            "--body-quota",
            "all",
            "--title-quota",
            "all",
            "--tfidf-terms",
            "all",
        ]
        if pattern_screening(config) == "entropy_full_inventory":
            return [
                command(
                    *common,
                    "--screening",
                    "entropy",
                    "--min-support",
                    1,
                    "--no-deduplicate-masks",
                    "--no-exclude-constant-masks",
                )
            ]
        # 2026-09-30 default: the complete extracted inventory is screened to
        # discriminative predicates (support band, fit-mask deduplication,
        # per-label mutual information within a per-cluster budget).
        return [
            command(
                *common,
                "--screening",
                "discriminative",
                "--min-support",
                3,
                "--deduplicate-masks",
                "--exclude-constant-masks",
                "--patterns-per-cluster",
                config.get("patterns_per_cluster", 512),
                "--parse-workers",
                config["workers"],
            )
        ]
    if stage == "prewarm":
        limits = config.get("prewarm", {})
        return [
            command(
                "prewarm-patterns",
                "--context-receipt",
                context / "COMPLETE.json",
                "--supply",
                paths["patterns"],
                "--output",
                paths["prewarm"],
                "--workers",
                config["workers"],
                "--memory-mib",
                limits.get("memory_mib", 128),
                "--disk-mib",
                limits.get("disk_mib", 2048),
            )
        ]
    if stage == "bind":
        return []
    if stage in ("check", "bo"):
        args = command("check", "--manifest", paths["manifest"])
        if stage == "check":
            return [args]
        learn = command("learn-rules", "--manifest", paths["manifest"], "--output", paths["run"])
        return [args, learn + (["--resume"] if resume else [])]
    if stage == "status":
        return [command("status", "--run", paths["run"])]
    if stage == "archive":
        return [command("archive-index", "--run", paths["run"], "--output", paths["archive"])]
    raise ValueError("unknown stage")


PATTERN_SCREENINGS = ("discriminative", "entropy_full_inventory")
# Mirrors loris.discovery.manifest.RELATION_SCREENING_DEFAULT (validated there).
RELATION_SCREENING_POLICIES = ("none", "discriminative_blocked_equality_v1")
RELATION_SCREENING_THRESHOLDS = ("min_groups", "min_group_fraction", "max_mean_block_partners",
                                 "min_nonsingleton_fraction")


def pattern_screening(config):
    value = config.get("pattern_screening", "discriminative")
    if value not in PATTERN_SCREENINGS:
        raise ValueError("pattern_screening must be discriminative or entropy_full_inventory")
    return value


def bind(config, paths):
    # These are locally produced trusted context pickles, authenticated before opening.
    receipt = json.loads((paths["context"] / "COMPLETE.json").read_text())
    context_file = paths["context"] / receipt["context"]["path"]
    if receipt.get("status") != "complete" or record(context_file) != receipt["context"]:
        raise ValueError("incomplete or altered local context")
    with context_file.open("rb") as stream:
        context = pickle.load(stream)
    attrs = config["relation_attributes"]
    if not set(attrs) <= {"cnt", "ttl", "mtd"} | set(context["attribute_values"]):
        raise ValueError("declared semantic attributes are absent from the actual context")
    supply = json.loads((paths["patterns"] / "COMPLETE.json").read_text())
    policy = json.loads((paths["patterns"] / "PROTOCOL.json").read_text())
    full_inventory = pattern_screening(config) == "entropy_full_inventory"
    if (
        supply.get("status") != "complete"
        or supply["protocol"] != record(paths["patterns"] / "PROTOCOL.json")
        or policy.get("max_per_cluster") is not None
        or policy.get("tfidf_terms_per_cluster") is not None
        or policy.get("fit_mask_deduplication") is not (not full_inventory)
        or policy["support"]["constant_masks_excluded"] is not (not full_inventory)
        or policy.get("screening") != ("entropy" if full_inventory else "discriminative")
    ):
        raise ValueError("plan requires its declared pattern supply (" + pattern_screening(config) + ")")
    database = paths["patterns"] / "candidates.sqlite"
    if record(database) != supply["candidate_archive"]:
        raise ValueError("candidate archive differs")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        counts = [
            dict(cluster=c, attr=a, family=f, reason=r, count=n)
            for c, a, f, r, n in connection.execute(
                "SELECT cluster,attr,family,reason,count(*) FROM candidates GROUP BY cluster,attr,family,reason ORDER BY cluster,attr,family,reason"
            )
        ]
    forbidden = {
        "outside_declared_diversity_budget",
        "fit_mask_diversity_prefilter",
        "constant_mask",
    }
    # The discriminative supply excludes these by design; only the full
    # inventory promises to keep them.
    if full_inventory and any(row["reason"] in forbidden for row in counts):
        raise ValueError("expanded inventory still has forbidden quota/mask exclusions")
    write_new(
        paths["root"] / "SUPPLY_INVENTORY_AUDIT.json",
        dict(candidate_archive=record(database), counts=counts),
    )
    run = dict(
        config["bo"],
        seed=paths["search_seed"],
        budget_per_cluster=paths["budget"],
        workers=config["workers"],
        initializer_slots=paths["initializer_slots"],
        # Independent per-cluster initializers run in parallel with the same
        # worker count as BO (the serial default took days on large supplies).
        initialization_workers=config["workers"] if paths["initializer_slots"] else 1,
        label_rotation="seeded_cluster_rotation",
        max_variables=paths["variables"],
        candidate_domain="rich",
        proposal_strategy="accuracy_guided",
        admission_order="standalone_score_desc",
    )
    manifest = dict(
        schema="loris_rich_discovery_manifest_v1",
        dataset=dict(
            name=config["dataset"],
            split_protocol=record(config["prepared_manifest"]),
            scope="train_dev_only",
            roots_contract="complete_signed_known_train",
        ),
        context=record(context_file),
        model_selection=record(paths["selector"] / "COMPLETE.json"),
        selection_strategy=paths["models"],
        pattern_supply=record(paths["patterns"] / "COMPLETE.json"),
        supervision_exposure=record(paths["supervision"].parent / "EXPOSURE.json"),
        relation_attributes=attrs,
        oracle_policy="accuracy_then_vote_bce",
        run=run,
        lexical_cache_dir=str(paths["prewarm"] / "lexical_cache"),
        notes="Finite full extracted inventory, explicit variable axis and declared budget; not exhaustive rule search. Preserve all upstream/run archives.",
    )
    if "relation_screening" in config:
        # e.g. {"policy": "none"} for the original unscreened relation domain;
        # bo.gamma_conflict_policy="strict" restores the original conflicts.
        manifest["relation_screening"] = config["relation_screening"]
    write_new(paths["manifest"], manifest)
    return dict(
        status="bound_not_executed",
        manifest=record(paths["manifest"]),
        inventory=record(paths["root"] / "SUPPLY_INVENTORY_AUDIT.json"),
    )


def execute(config_path, config, paths, stage, argv, *, resume=False):
    folder = (
        paths["root"] / "stage_attempts" / ("%s_%s_%d" % (stage, paths["variant"], time.time_ns()))
    )
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_new(
        folder / "STARTED.json",
        dict(
            stage=stage,
            variant=paths["variant"],
            resume=resume,
            config=record(config_path),
            launcher=record(__file__),
            commands=argv,
        ),
    )
    try:
        if stage in (
            "supervision",
            "pool-cpu",
            "pool-gpu",
            "features",
            "context",
            "selector-inputs",
            "selector",
            "patterns",
            "prewarm",
            "bo",
        ):
            capacity = disk_preflight(config, paths)
            write_new(folder / "DISK_PREFLIGHT.json", capacity)
            if capacity["status"] != "sufficient_for_declared_preflight":
                raise RuntimeError(
                    "insufficient free disk for the declared remaining arrays plus reserve; retain raw assets and choose sufficient persistent storage"
                )
        if stage == "bind":
            result = bind(config, paths)
        else:
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            environment["PYTHONPATH"] = str(ROOT) + (
                os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
            )
            for index, command in enumerate(argv):
                with (folder / ("command_%02d.log" % index)).open("x") as log:
                    with subprocess.Popen(
                        command,
                        cwd=str(ROOT),
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1,
                    ) as child:
                        for line in child.stdout:
                            log.write(line)
                            log.flush()
                            print(line, end="", flush=True)
                        code = child.wait()
                    if code:
                        raise RuntimeError(
                            "stage command failed with exit %d; retained log %s" % (code, log.name)
                        )
            result = dict(status="stage_completed", stage=stage)
        write_new(folder / "RESULT.json", dict(result, seconds=time.monotonic() - started))
        return result
    except BaseException as error:
        write_new(
            folder / "RESULT.json",
            dict(
                status="failed",
                error_type=type(error).__name__,
                reason=str(error),
                seconds=time.monotonic() - started,
            ),
        )
        raise


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", required=True)
    cli.add_argument("--stage", choices=STAGES, required=True)
    cli.add_argument("--variables", type=int, default=2)
    cli.add_argument("--models", choices=("learned", "all"), default="learned")
    cli.add_argument("--bo-trials", type=int)
    cli.add_argument("--search-seed", type=int)
    cli.add_argument(
        "--resume", action="store_true", help="exact source-bound BO/selector resume only"
    )
    cli.add_argument(
        "--execute",
        action="store_true",
        help="execute this stage; omitted prints the concrete command only",
    )
    args = cli.parse_args(argv)
    config = configuration(args.config)
    if args.resume and args.stage not in ("bo", "selector"):
        cli.error("resume applies only to BO or selector; partial model fits are not resumable")
    paths = layout(
        config,
        variables=args.variables,
        models=args.models,
        budget=args.bo_trials,
        search_seed=args.search_seed,
    )
    argv = commands(config, paths, args.stage, resume=args.resume)
    if not args.execute:
        print(
            json.dumps(
                dict(
                    status="planned_not_executed",
                    stage=args.stage,
                    variant=paths["variant"],
                    manifest=str(paths["manifest"]),
                    commands=argv,
                    shell=[shlex.join(c) for c in argv],
                ),
                indent=2,
            )
        )
        return 0
    print(
        json.dumps(
            execute(args.config, config, paths, args.stage, argv, resume=args.resume), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
