"""Executable experiment inventory; unimplemented methods remain explicit."""

import json
from pathlib import Path
import subprocess
import sys
import time

from loris.discovery.manifest import file_record
from loris._engine.rlearner import atomic_json


def wrapper_text():
    """One relocatable wrapper for generated-suite and repository layouts."""
    return """#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Explicit IDs are required; no implicit run-all. Overrides are paths, not shell code.
if [[ "$#" -lt 1 || "$#" -gt 2 || ( "$#" -eq 2 && "$2" != "--dry-run" ) ]]; then
  echo "Usage: $0 EXPERIMENT_ID [--dry-run]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
matrix_path="${LORIS_EXPERIMENT_MATRIX:-}"
if [[ -z "$matrix_path" ]]; then
  matrix_root="$script_dir"
  while [[ ! -f "$matrix_root/EXPERIMENT_MATRIX.json" ]]; do
    if [[ "$matrix_root" == "/" ]]; then
      echo "No ancestor EXPERIMENT_MATRIX.json; set LORIS_EXPERIMENT_MATRIX explicitly." >&2
      exit 2
    fi
    matrix_root="$(dirname -- "$matrix_root")"
  done
  matrix_path="$matrix_root/EXPERIMENT_MATRIX.json"
fi
# Prefer a colocated source checkout; otherwise use the explicitly installed package.
package_root="$script_dir"
while [[ ! -f "$package_root/loris/experiments/__main__.py" && "$package_root" != "/" ]]; do
  package_root="$(dirname -- "$package_root")"
done
if [[ -f "$package_root/loris/experiments/__main__.py" ]]; then
  export PYTHONPATH="$package_root${PYTHONPATH:+:$PYTHONPATH}"
fi
exec "${LORIS_PYTHON:-python}" -m loris.experiments run --matrix "$matrix_path" --id "$1" "${@:2}"
"""


def prepare(assets_path, output):
    path = Path(assets_path).resolve(strict=True)
    assets = json.loads(path.read_text())
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    experiments = []

    def add(ds, name, question, commands, kind, status="ready", blockers=None, **extra):
        if kind in (
            "online_labeling",
            "feedback_weight_control",
            "feedback_ablation",
            "external_budget_curve",
        ):
            # Historical inventories do not certify the new independent-q or
            # controlled-supervision protocol. Only workflow creates those runs.
            commands = []
            status = "blocked_assets"
            blockers = list(blockers or []) + [
                "Legacy online launch disabled: bind current independent calibration and explicit supervision controls through workflow"
            ]
            extra["legacy_online_launch_disabled"] = True
        row = dict(
            id=ds + "/" + name,
            dataset=ds,
            question=question,
            kind=kind,
            status=status,
            blockers=blockers or [],
            commands=commands,
            seed=42,
            repeats=1,
            output=str(out / "runs" / ds / name),
            **extra,
        )
        experiments.append(row)

    for ds, a in assets["datasets"].items():
        manifest = str((path.parent / a["discovery_manifest"]).resolve())
        rules = str((path.parent / a["rules"]).resolve())
        common = ["--manifest", manifest]
        dest = lambda n: str(out / "runs" / ds / n)
        add(
            ds,
            "frozen_discovery",
            "完整已保存RLearner在开发集得到什么结果？",
            [],
            "frozen_result",
            status="complete_reused",
            receipt=a["completion"],
            new_BO=0,
            new_training=0,
        )
        add(
            ds,
            "model_baselines",
            "所有模型及相同开发阈值机会下的组合表现",
            [
                [sys.executable, "-m", "loris.experiments", "baselines"]
                + common
                + ["--output", dest("model_baselines")]
            ],
            "matched_raw_predictions",
            new_BO=0,
            new_training=0,
            calibration_budget="19 thresholds/label/model; dev in-sample",
        )
        for alias in a.get("model_rdl_aliases", []):
            for calibrated in (False, True):
                name = "model_rdl_" + alias + ("_calibrated" if calibrated else "_at05")
                command = (
                    [sys.executable, "-m", "loris.experiments", "evaluate"]
                    + common
                    + ["--model", alias, "--output", dest(name)]
                )
                if calibrated:
                    command.append("--calibrate")
                add(
                    ds,
                    name,
                    "同概率/阈值/根下的完整单模型RDL可行性与P",
                    [command],
                    "complete_model_program",
                    new_BO=0,
                    new_training=0,
                    Chase="full; Gamma conflict retained; no automatic source suppression",
                )
        for variant in ("pure_ml", "no_text", "no_label", "no_ground"):
            add(
                ds,
                variant,
                (
                    "固定程序的能力依赖"
                    if variant != "no_ground"
                    else "移除ML规则文本/属性guard的反事实"
                ),
                [
                    [sys.executable, "-m", "loris.experiments", "evaluate"]
                    + common
                    + ["--rules", rules, "--variant", variant, "--output", dest(variant)]
                ],
                (
                    "fixed_program_dependency"
                    if variant != "no_ground"
                    else "paper_NoGround_operationalization"
                ),
                new_BO=0,
                new_training=0,
                Chase="full fixed point; conflict is failed result",
            )
        for selection in ("indiv", "random"):
            name = "rediscover_" + selection
            add(
                ds,
                name,
                "同K替换学习选模后的重新发现收益",
                [
                    [sys.executable, "-m", "loris.discovery", "learn-rules"]
                    + common
                    + ["--model-selection", selection, "--output", dest(name)]
                ],
                "fresh_equal_budget_BO",
                new_BO=5400,
                new_training=0,
                resource="Full 5400 trials; do not launch during script preparation",
            )
        for domain in ("no_text", "local"):
            name = "rediscover_" + domain
            add(
                ds,
                name,
                "从空Σ禁用一种候选能力后重新学习",
                [
                    [sys.executable, "-m", "loris.discovery", "learn-rules"]
                    + common
                    + ["--candidate-domain", domain, "--output", dest(name)]
                ],
                "fresh_equal_budget_BO",
                new_BO=5400,
                new_training=0,
                resource="Full 5400 trials; includes complete raw proposal archive",
            )
        labeling = a.get("labeling_manifest")
        labeling_data = json.loads((path.parent / labeling).read_text()) if labeling else None
        reliability_protocol = a.get("fit_reliability_protocol")
        if labeling and reliability_protocol:
            # Legacy asset maps remain readable, but selection-conditioned fit
            # priors cannot be launched as the independent public protocol.
            add(
                ds,
                "fit_source_reliability",
                "Historical masked-fit prior; public calibration requires independent rows",
                [],
                "fit_source_reliability",
                status="blocked_assets",
                blockers=["Use workflow calibrate with loris_independent_rule_calibration_v2"],
                complete_Chase_calls=2,
                new_BO=0,
                new_training=0,
                evaluation="historical development diagnostic only",
            )
            for mode in ("confidence_only", "reliability_only", "provenance"):
                name = "feedback_" + mode
                ready = mode == "confidence_only"
                commands = (
                    [
                        [
                            sys.executable,
                            "-m",
                            "loris.labeling",
                            "run",
                            "--manifest",
                            str((path.parent / labeling).resolve()),
                            "--strategy",
                            "paper",
                            "--weighting",
                            mode,
                            "--output",
                            dest(name),
                        ]
                    ]
                    if ready
                    else []
                )
                add(
                    ds,
                    name,
                    "Legacy matrix weighting control",
                    commands,
                    "feedback_weight_control",
                    status="ready" if ready else "blocked_assets",
                    blockers=(
                        []
                        if ready
                        else [
                            "Bind independent calibration through workflow; old pooled fit artifacts are rejected"
                        ]
                    ),
                    new_BO=0,
                    weighting=mode,
                    evaluation="development only",
                )
        for strategy in ("frozen", "supervised", "self_train", "paper"):
            for query in ("rdg", "random", "uncertainty") if strategy == "paper" else ("rdg",):
                name = "labeling_" + strategy + ("_" + query if strategy == "paper" else "")
                commands = []
                if labeling:
                    commands = [
                        [
                            sys.executable,
                            "-m",
                            "loris.labeling",
                            "run",
                            "--manifest",
                            str((path.parent / labeling).resolve()),
                            "--strategy",
                            strategy,
                            "--query-strategy",
                            query,
                            "--output",
                            dest(name),
                        ]
                    ]
                    if strategy == "self_train":
                        commands[0] += ["--candidate-check", "nocheck"]
                is_query_comparison = strategy == "paper" and query != "rdg"
                no_query = (
                    labeling_data is None
                    or labeling_data["loop"]["label_budget"] == 0
                    or labeling_data["oracle"]["kind"] == "disabled"
                )
                ready = bool(labeling) and not (is_query_comparison and no_query)
                add(
                    ds,
                    name,
                    "固定Σ的实际反馈与外部标注成本",
                    commands if ready else [],
                    "online_labeling",
                    status="ready" if ready else "blocked_assets",
                    blockers=(
                        []
                        if ready
                        else [
                            (
                                "Bind nonzero budget and a real answer provider for query-policy comparisons"
                                if is_query_comparison and no_query
                                else "Bind matching Aux checkpoint, full online context, and query-answer provider"
                            )
                        ]
                    ),
                    new_BO=0,
                    weighting=labeling_data["weighting"] if labeling_data else None,
                    label_budget=labeling_data["loop"]["label_budget"] if labeling_data else None,
                    evaluation=(
                        "development control, zero external queries"
                        if no_query
                        else "interactive oracle evaluation for explicit online manifest"
                    ),
                )
        for name, flags in (
            ("feedback_uniform", ["--weighting", "uniform"]),
            ("feedback_nocheck", ["--candidate-check", "nocheck"]),
        ):
            commands = []
            if labeling:
                commands = [
                    [
                        sys.executable,
                        "-m",
                        "loris.labeling",
                        "run",
                        "--manifest",
                        str((path.parent / labeling).resolve()),
                        "--strategy",
                        "paper",
                        "--output",
                        dest(name),
                    ]
                    + flags
                ]
            add(
                ds,
                name,
                "历史反馈配方；未核验相同监督，不能称单因素对照",
                commands,
                "feedback_ablation",
                status="ready" if labeling else "blocked_assets",
                blockers=[] if labeling else ["Bind actual matching labeling manifest"],
                new_BO=0,
                new_training="declared actual continuation budget",
                note="Uniform-bound manifests make uniform a duplicate control, not main weighted LORIS; NoCheck still checks final Chase",
            )
        external = a.get("external_labeling_manifest")
        if external:
            ext_path = (path.parent / external).resolve()
            ext = json.loads(ext_path.read_text())
            eligible = int(a["external_eligible_documents"])
            import math

            for fraction in (0.001, 0.005, 0.01, 0.05, 1.0):
                budget = math.ceil(eligible * fraction)
                for query in ("rdg", "random", "uncertainty"):
                    name = "external_%s_%g" % (query, fraction)
                    cmd = [
                        sys.executable,
                        "-m",
                        "loris.labeling",
                        "run",
                        "--manifest",
                        str(ext_path),
                        "--strategy",
                        "paper",
                        "--query-strategy",
                        query,
                        "--label-budget",
                        str(budget),
                        "--query-batch-size",
                        str(min(8, budget)),
                        "--output",
                        dest(name),
                    ]
                    add(
                        ds,
                        name,
                        "相同查询预算下质量/未回答文档收益及成本",
                        [cmd],
                        "external_budget_curve",
                        status="ready" if ext["oracle"]["kind"] != "disabled" else "blocked_assets",
                        blockers=(
                            []
                            if ext["oracle"]["kind"] != "disabled"
                            else ["No oracle provider bound"]
                        ),
                        new_BO=0,
                        label_budget=budget,
                        eligible_documents=eligible,
                        budget_fraction=fraction,
                        query_batch_size=min(8, budget),
                        weighting=ext["weighting"],
                        resource="Potentially many Aux updates; select one explicit budget point, not run-all",
                        evaluation="development score after train-query simulation; not interactive test until full target context bound",
                    )
        for name, why in (
            (
                "NoJoint",
                "Needs same-domain independent MI context/model-F1 selection; not IndivMS alias",
            ),
            ("ChiSquared", "Needs matched primitive supplier and frozen count/budget"),
            ("WeShap", "Needs trusted original implementation and matched dataset adapter"),
            ("LocalBoost", "Needs trusted original implementation and matched dataset adapter"),
            (
                "Dyn_tau_eta",
                "Counterfactual needs bounded validation search of rule labels/context; not enabled in main loop",
            ),
            (
                "incremental_vs_full",
                "Use workflow main/noinc on the same frozen assets and verify identical final states",
            ),
            (
                "RDG_exact_influence",
                "Needs explicit small diagnostic domain and exact hypothetical-Chase cost budget",
            ),
            (
                "scaling_D_Sigma_Lambda",
                "Predeclare nested axes/sample IDs before execution; no truncated Chase",
            ),
            (
                "BO_budget_sensitivity",
                "Predeclare independent fresh budgets; prefixes are not equivalent searches",
            ),
            (
                "initialization_sensitivity",
                "Paper m is overloaded: distinguish initial_trials from max_predicates",
            ),
        ):
            add(
                ds,
                name,
                why,
                [],
                "paper_requested",
                status="not_implemented",
                blockers=[why],
                new_BO=None,
                new_training=None,
            )
        for name in (
            "Snuba",
            "SelfPretraining",
            "RulePrompt",
            "DeBERTaSVM",
            "DeBERTaXGBoost",
            "RoBERTaSVM",
            "RoBERTaXGBoost",
            "BESRA",
            "CoMAL",
        ):
            add(
                ds,
                "external_" + name,
                "论文外部baseline与相同监督权限的比较",
                [],
                "original_external_baseline",
                status="blocked_external_implementation",
                blockers=[
                    "No authenticated matching executable implementation/checkpoint/provider bound"
                ],
                new_BO=None,
                new_training=None,
            )
    for ds, reason in assets.get("pending_datasets", {}).items():
        add(
            ds,
            "prepare_assets",
            reason,
            [],
            "dataset_asset_dependency",
            status="blocked_assets",
            blockers=[reason],
            new_BO=0,
            new_training=0,
        )
    matrix = dict(
        schema="loris_paper_experiment_matrix_v1",
        assets=file_record(path),
        paper=assets["paper"],
        policy=assets["policy"],
        experiments=experiments,
        historical_scores_not_recomputed=True,
        run_all_implicitly=False,
    )
    atomic_json(out / "EXPERIMENT_MATRIX.json", matrix)
    # Generated scripts invoke exact IDs; no shell interpolation of commands.
    scripts = out / "scripts"
    scripts.mkdir()
    for ds in assets["datasets"]:
        target = scripts / (ds + "_ready.sh")
        target.write_text(wrapper_text())
        target.chmod(0o755)
    return matrix


def run(matrix_path, experiment_id, dry_run=False):
    path = Path(matrix_path).resolve(strict=True)
    matrix = json.loads(path.read_text())
    found = [x for x in matrix["experiments"] if x["id"] == experiment_id]
    if len(found) != 1:
        raise ValueError("unique experiment ID required")
    exp = found[0]
    if exp["status"] == "complete_reused":
        return dict(status="complete_reused", receipt=exp["receipt"], new_computation=0)
    if exp["status"] != "ready" or not exp["commands"]:
        raise ValueError("experiment is not runnable: " + json.dumps(exp["blockers"]))
    if dry_run:
        return dict(status="dry_run", commands=exp["commands"], output=exp["output"])
    out = Path(exp["output"])
    if out.exists():
        raise FileExistsError("output exists; no implicit restart/resume/overwrite")
    receipts = path.parent / "execution_receipts"
    receipts.mkdir(exist_ok=True)
    receipt = receipts / (experiment_id.replace("/", "__") + ".json")
    if receipt.exists():
        raise FileExistsError("execution receipt already exists; choose a new matrix namespace")
    start = time.monotonic()
    state = dict(status="running", experiment=exp, matrix=file_record(path), commands=[])
    atomic_json(receipt, state)
    try:
        for i, command in enumerate(exp["commands"]):
            log = receipts / (experiment_id.replace("/", "__") + "_%02d.log" % i)
            with log.open("xb") as stream:
                process = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            state["commands"].append(
                dict(argv=command, returncode=process.returncode, log=str(log))
            )
            atomic_json(receipt, state)
            if process.returncode:
                raise RuntimeError("experiment failed; see " + str(log))
        state.update(status="complete", seconds=time.monotonic() - start)
    except BaseException as error:
        state.update(
            status="failed",
            error=type(error).__name__,
            reason=str(error),
            seconds=time.monotonic() - start,
        )
        raise
    finally:
        atomic_json(receipt, state)
    return state
