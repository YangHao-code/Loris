"""Portable, explicit experiment stages with authenticated external assets.

No stage downloads data, supplies fabricated labels, or silently resumes a failed
experiment. Paths in the asset map may use ${ASSET_ROOT} and ${REPO_ROOT}.
"""

import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time

import numpy as np
from loris.discovery.manifest import bound, file_record
from .registry import check_complete_program, check_execution_program
from .states import (
    completion_state_evidence,
    evaluation_has_valid_state,
    final_state_kind,
    final_arrays_evidence,
)

SCHEMA = "loris_release_assets_v1"
VARIANTS = (
    "main",
    "uniform",
    "reliability_only",
    "provenance",
    "confidence_only",
    "direct_rule_precision",
    "direct_rule_precision_discount",
    "complete_provenance",
    "complete_provenance_discount",
    "frozen",
    "frozen_zero_budget",
    "self_train",
    "supervised",
    "nocheck",
    "noinc",
    "trusted_replay",
    "exclude_receiver",
    "affected_channels_to_aux",
    "halt_conflicts",
    "unknown_full_rebuild",
    "random",
    "uncertainty",
)
VARIANTS += ("nollm", "frozen_prior")
# Control for the 2026-09-30 method changes: quarantine instead of validated
# Gamma wins, no undecided-cell fallback, no commit gate, original kappa
# calibration. Speed settings are those of the main manifest.
VARIANTS += ("original_semantics",)
ORIGINAL_SEMANTICS_VARIANTS = ("original_semantics", "calibration_original")
# Main with LLM proposals checked against the closure that includes the answers
# accepted earlier in the same query batch (loop.llm_check_base="refreshed").
VARIANTS += ("exact_llm_check",)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def expand(value, asset_root, repo_root):
    if isinstance(value, str):
        value = value.replace("${ASSET_ROOT}", str(asset_root)).replace(
            "${REPO_ROOT}", str(repo_root)
        )
        if "${" in value:
            raise ValueError("unresolved template variable: " + value)
        return value
    if isinstance(value, list):
        return [expand(v, asset_root, repo_root) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, asset_root, repo_root) for k, v in value.items()}
    return value


def resolve_manifest(path, asset_root, repo_root):
    """Resolve a supplied manifest in its own directory without editing it."""
    path = Path(path).resolve(strict=True)
    value = expand(read(path), asset_root, repo_root)

    def visit(item):
        if isinstance(item, list):
            return [visit(v) for v in item]
        if not isinstance(item, dict):
            return item
        result = {k: visit(v) for k, v in item.items()}
        if isinstance(result.get("path"), str):
            target = Path(result["path"]).expanduser()
            if not target.is_absolute():
                target = path.parent / target
            result["path"] = str(target.resolve())
            if "sha256" in result:
                bound(result, path.parent)
        for key in ("backend_source", "cache_dir", "gpu_lock", "lexical_cache_dir", "directory"):
            text = result.get(key)
            if isinstance(text, str) and text:
                target = Path(text).expanduser()
                result[key] = str(
                    (target if target.is_absolute() else path.parent / target).resolve()
                )
        return result

    return visit(value)


def bind_assets(asset_map, output, asset_root=None):
    source = Path(asset_map).resolve(strict=True)
    root = (
        Path(asset_root or os.environ.get("LORIS_ASSET_ROOT") or source.parent)
        .expanduser()
        .resolve()
    )
    repo = Path(__file__).resolve().parents[2]
    value = expand(read(source), root, repo)
    required = {"schema", "dataset", "scope", "rules_mode", "artifacts", "notes"}
    if set(value) != required or value["schema"] != SCHEMA:
        raise ValueError("expected explicit loris_release_assets_v1 asset map")
    if value["scope"] not in ("paper_table3", "historical_extension", "extra_dataset"):
        raise ValueError("declare paper_table3, historical_extension or extra_dataset scope")
    if value["rules_mode"] not in ("reuse", "discover"):
        raise ValueError(
            "rules_mode must be reuse or discover; incomplete caches never trigger automatic BO"
        )
    allowed = {
        "dataset_audit",
        "dataset_source",
        "discovery_manifest",
        "labeling_manifest",
        "reliability_protocol",
        "evaluation",
        "cached_program",
        "cached_completion",
    }
    if not isinstance(value["artifacts"], dict) or set(value["artifacts"]) - allowed:
        raise ValueError("unknown asset role")
    refs = {}
    for name, record in value["artifacts"].items():
        if (
            not isinstance(record, dict)
            or not set(record) <= {"path", "sha256"}
            or not record.get("path")
        ):
            raise ValueError("asset references require path and optional expected SHA256")
        target = Path(record["path"]).expanduser()
        target = (target if target.is_absolute() else source.parent / target).resolve(strict=True)
        if record.get("sha256"):
            bound(dict(path=str(target), sha256=record["sha256"]), source.parent)
        refs[name] = file_record(target)
    if "dataset_audit" not in refs:
        raise ValueError("dataset_audit manifest required before any experiment")
    if value["rules_mode"] == "reuse" and not {"cached_program", "cached_completion"} <= set(refs):
        raise ValueError("reuse needs both the complete ordered program and completion receipt")
    if value["rules_mode"] == "discover" and "discovery_manifest" not in refs:
        raise ValueError("discover needs a complete, authenticated upstream manifest")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    value.update(
        artifacts=refs, source_map=file_record(source), asset_root=str(root), repo_root=str(repo)
    )
    write(out / "ASSETS.json", value)
    return dict(status="bound_not_executed", workspace=str(out), assets=sorted(refs))


def asset(workspace, name, required=True):
    workspace = Path(workspace).resolve(strict=True)
    config = read(workspace / "ASSETS.json")
    record = config["artifacts"].get(name)
    if record is None:
        if required:
            raise ValueError(
                "missing " + name + "; supply genuine matching assets and bind a new workspace"
            )
        return None
    return bound(record, workspace)


def manifest(workspace, name):
    workspace = Path(workspace)
    config = read(workspace / "ASSETS.json")
    return resolve_manifest(asset(workspace, name), config["asset_root"], config["repo_root"])


def audit_dataset(path, output=None, asset_root=None, repo_root=None):
    path = Path(path).resolve(strict=True)
    data = resolve_manifest(path, asset_root or path.parent, repo_root or path.parent)
    if data.get("schema") != "loris_release_dataset_audit_v1":
        raise ValueError("dataset audit schema must be loris_release_dataset_audit_v1")
    labels = data["labels"]
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(x, str) or not x.strip() for x in labels)
        or len(labels) != len(set(labels))
    ):
        raise ValueError("complete unique label axis required")
    if set(data["splits"]) != {"train", "dev", "test"} or set(data["expected_counts"]) != set(
        data["splits"]
    ):
        raise ValueError("explicit full train/dev/test counts required")
    if data["duplicate_policy"] not in ("report_official", "require_disjoint"):
        raise ValueError("declare duplicate policy")
    text_policy = data.get("text_policy", dict(empty_text="reject", missing_text_reason=""))
    if (
        set(text_policy) != {"empty_text", "missing_text_reason"}
        or text_policy["empty_text"] not in ("reject", "report_missing")
        or not isinstance(text_policy["missing_text_reason"], str)
        or (
            text_policy["empty_text"] == "report_missing"
            and not text_policy["missing_text_reason"].strip()
        )
    ):
        raise ValueError("missing-text exceptions require explicit source/protocol justification")
    details, identities, texts, checked_files = {}, {}, {}, []
    for role, split in data["splits"].items():
        documents = bound(split["documents"], path.parent)
        checked_files.append(file_record(documents))
        if "truth" not in split:
            raise ValueError(role + ": complete truth asset required for dataset integrity audit")
        ids, hashes, truth_ids = [], [], []
        empty_text = 0
        feature_ids = 0
        if type(data["expected_counts"][role]) is not int or data["expected_counts"][role] < 1:
            raise ValueError(role + ": full split must have a positive integer expected count")
        with documents.open() as stream:
            for line in stream:
                row = json.loads(line)
                row_id = row[split.get("id_key", "row_id")]
                if type(row_id) not in (str, int) or not str(row_id).strip():
                    raise ValueError(role + ": explicit nonempty document ID required")
                ids.append(str(row_id))
                truth_ids.append(
                    str(row[split.get("document_truth_id_key", split.get("id_key", "row_id"))])
                )
                text = row[split.get("text_key", "text")]
                if not isinstance(text, str):
                    raise ValueError("document text must be an original ordered string")
                empty_text += int(not text.strip())
                feature_ids += int(bool(re.fullmatch(r"\s*w\d+(?:\s+w\d+)+\s*", text)))
                hashes.append(hashlib.sha256(text.encode()).hexdigest())
        if len(ids) != data["expected_counts"][role] or len(ids) != len(set(ids)):
            raise ValueError(role + ": incomplete or duplicate document ID axis")
        if empty_text and text_policy["empty_text"] == "reject":
            raise ValueError(
                role
                + ": "
                + str(empty_text)
                + " empty/blank document texts; explicit missing-text protocol required"
            )
        if feature_ids:
            raise ValueError(
                role
                + ": "
                + str(feature_ids)
                + " feature-ID documents cannot serve as original ordered natural text"
            )
        record = dict(
            rows=len(ids),
            labels=len(labels),
            duplicate_exact_text_rows=len(hashes) - len(set(hashes)),
            empty_text_rows=empty_text,
            feature_id_text_rows=feature_ids,
        )
        if "truth" in split:
            truth = bound(split["truth"], path.parent)
            checked_files.append(file_record(truth))
            with np.load(truth, allow_pickle=False) as values:
                y = values[split.get("target_key", "targets")]
                if y.shape != (len(ids), len(labels)) or not np.isin(y, [0, 1]).all():
                    raise ValueError(role + ": truth axis/value mismatch")
                truth_id_key = split.get("truth_id_key", "row_ids")
                if (
                    truth_id_key not in values
                    or [str(x) for x in values[truth_id_key].tolist()] != truth_ids
                ):
                    raise ValueError(role + ": truth document IDs differ")
                if "labels" in values and values["labels"].tolist() != labels:
                    raise ValueError(role + ": truth label ordering differs")
                if "observed" in values and (
                    values["observed"].shape != y.shape
                    or not np.isin(values["observed"], [0, 1]).all()
                    or not values["observed"].all()
                ):
                    raise ValueError(role + ": integrity truth is not a complete label table")
                record["zero_positive_label_rows"] = int((~y.astype(bool).any(axis=1)).sum())
            record["truth_verified"] = True
        details[role], identities[role], texts[role] = record, set(ids), set(hashes)
    overlaps = {}
    for a, b in itertools.combinations(details, 2):
        if identities[a] & identities[b]:
            raise ValueError("row IDs overlap: " + a + "/" + b)
        overlaps[a + "/" + b] = len(texts[a] & texts[b])
    if data["duplicate_policy"] == "require_disjoint" and any(overlaps.values()):
        raise ValueError("duplicate text crosses declared independent splits")
    result = dict(
        status="verified",
        dataset=data["dataset"],
        splits=details,
        exact_text_overlaps=overlaps,
        scope=data["scope"],
        original_source_certification=data["source_provenance"],
        test_read_for_integrity_only=True,
        test_metrics_computed=False,
        text_policy=text_policy,
        full_original_text_present=not any(v["empty_text_rows"] for v in details.values()),
        content_duplicates_are_not_removed=True,
        source_manifest=file_record(path),
        files=checked_files,
    )
    if output:
        write(output, result)
    return result


def execute(workspace, stage, argv, dry_run=False):
    if dry_run:
        return dict(status="dry_run", command=argv)
    workspace = Path(workspace)
    receipt = workspace / "receipts" / (stage + ".json")
    if receipt.exists():
        raise FileExistsError(
            "stage already attempted; inspect receipt or bind a new workspace: " + str(receipt)
        )
    start = time.monotonic()
    state = dict(status="running", argv=argv, stage=stage)
    write(receipt, state)
    try:
        process = subprocess.run(argv, check=False)
        state["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError("experiment stage failed: " + stage)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=type(error).__name__, reason=str(error))
        raise
    finally:
        state["seconds"] = time.monotonic() - start
        write(receipt, state)
    return state


def require_audit(workspace):
    path = Path(workspace) / "DATA_AUDIT.json"
    if not path.exists() or read(path).get("status") != "verified":
        raise ValueError("run workflow audit first; complete data integrity is required")
    # Recheck the audit input identity, including nested data files when the audit is run.
    bound(read(path)["source_manifest"], path.parent)
    for record in read(path)["files"]:
        bound(record, path.parent)


def initial_observations(context):
    """Initial supervision for the paper experiment, separate from online Gamma."""
    gamma = context.get("gamma")
    shape = (len(context["row_ids"]), len(context["labels"]))
    if gamma is None:
        raise ValueError("paper experiment context must declare its initial Gamma")
    pos, neg = np.asarray(gamma.pos), np.asarray(gamma.neg)
    if (
        pos.shape != shape
        or neg.shape != shape
        or pos.dtype != np.bool_
        or neg.dtype != np.bool_
        or np.any(pos & neg)
    ):
        raise ValueError("initial Gamma signed label axes differ")
    return pos | neg


def audit_application(workspace, result):
    """Bind an explicitly complete application axis to the audited split texts."""
    workspace = Path(workspace)
    config = read(workspace / "ASSETS.json")
    if "labeling_manifest" not in config["artifacts"]:
        result["application_context"] = dict(status="not_supplied_not_certified")
        return result
    data = manifest(workspace, "dataset_audit")
    splits = data.get("application_splits")
    if (
        not isinstance(splits, list)
        or not splits
        or len(splits) != len(set(splits))
        or any(role not in data["splits"] for role in splits)
    ):
        raise ValueError("declare application_splits explicitly before labeling a complete dataset")
    app = manifest(workspace, "labeling_manifest")
    context_path = bound(app["context"], workspace)
    with context_path.open("rb") as stream:
        context = pickle.load(stream)  # authenticated trusted-origin local context
    expected, source_splits = {}, {}
    for role in splits:
        split = data["splits"][role]
        with bound(split["documents"], workspace).open() as stream:
            for line in stream:
                doc = json.loads(line)
                row_id = str(doc[split.get("id_key", "row_id")])
                source_splits[row_id] = role
                expected[row_id] = hashlib.sha256(
                    doc[split.get("text_key", "text")].encode()
                ).hexdigest()
    holdout = None
    if data.get("application_holdout") is not None:
        from .application_holdout import validate_holdout
        excluded, holdout = validate_holdout(data["application_holdout"], workspace,
            context, source_splits, data["labels"])
        expected = {row: value for row, value in expected.items() if row not in excluded}
        result["files"].extend(holdout["files"])
    ids = context["row_ids"]
    if (
        not isinstance(ids, list)
        or len(ids) != len(set(ids))
        or len(ids) != len(context["docs"])
        or set(ids) != set(expected)
    ):
        raise ValueError("application context must contain exactly all declared split document IDs")
    if context["labels"] != data["labels"]:
        raise ValueError("application label axis differs from complete audited dataset")
    observed = initial_observations(context)
    root_rows = np.flatnonzero(observed.any(axis=1))
    if any(source_splits[ids[int(row)]] != "train" for row in root_rows):
        raise ValueError("paper experiment initial Gamma must use actual train IDs only")
    for row_id, doc in zip(ids, context["docs"]):
        if (
            not isinstance(doc.cnt, str)
            or hashlib.sha256(doc.cnt.encode()).hexdigest() != expected[row_id]
        ):
            raise ValueError(
                "application original text differs from audited document ID: " + str(row_id)
            )
    actual = file_record(context_path)
    result["files"].extend([actual, file_record(asset(workspace, "labeling_manifest"))])
    result["application_context"] = dict(
        status="verified_complete_declared_split_axis",
        context=actual,
        application_splits=splits,
        rows=len(ids),
        labels=len(context["labels"]),
        ordered_row_ids_sha256=hashlib.sha256(
            json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        context_order_may_differ_from_source_split_order=True,
        all_texts_match_by_id=True,
        initial_Gamma_policy="paper_train_only_by_source_document_ID",
        initial_Gamma_documents=len(root_rows),
        initial_Gamma_signed_cells=int(observed.sum()),
        test_documents_present="test" in splits,
        calibration_holdout=holdout,
        official_full_split_audit_preserved=True,
        test_scope=(
            "transductive_inputs; direct test queries require interactive_test evaluation"
            if "test" in splits
            else "test_absent"
        ),
    )
    return result


def program(workspace):
    workspace = Path(workspace)
    config = read(workspace / "ASSETS.json")
    if config["rules_mode"] == "reuse":
        p, c = asset(workspace, "cached_program"), asset(workspace, "cached_completion")
    else:
        p, c = workspace / "discovery/learning/SELECTED.json", workspace / "discovery/COMPLETE.json"
    check_execution_program(p, c)
    labels = set(manifest(workspace, "dataset_audit")["labels"])
    for rule in read(p)["rules"]:
        for atom in [rule["head"]] + rule["body"]:
            if "label" in atom and atom["label"] not in labels:
                raise ValueError("cached rule label is outside the audited dataset axis")
    return p


def _calibration_directory(workspace, data, variant="main"):
    return Path(workspace) / (
        "calibration-halt"
        if data["loop"]["conflict_policy"] == "halt"
        else (
            "calibration-full-rebuild"
            if data["loop"]["unknown_semantics"] == "full_rebuild"
            else (
                "calibration-original" if variant in ORIGINAL_SEMANTICS_VARIANTS
                else "calibration"
            )
        )
    )


def _original_semantics(data):
    """Undo the 2026-09-30 method changes of a labeling manifest (control)."""
    data["loop"]["gamma_conflict_policy"] = "quarantine"
    data["loop"]["commit_gate"] = "none"
    data["loop"].pop("commit_gate_tolerance", None)
    data.pop("fallback", None)
    data["calibration"] = dict(grid=list(data["calibration"]["grid"]), minimum_support_each_sign=1)


CALIBRATION_VARIANTS = ("calibration", "calibration_halt", "calibration_full_rebuild",
                        "calibration_original")


def _adapt_output_controls(data):
    """Keep the 2026-09-30 conflict options well defined for every variant.

    validated_gamma_wins and the refreshed LLM check base exist only for the
    retained unknown protocol; halt and full_rebuild variants fall back to
    quarantine and the batch-start check base. The fallback and the commit
    gate are unchanged in every variant (see _needs_independent_outputs).
    """
    loop = data["loop"]
    if (loop.get("conflict_policy") != "unknown_arbitrate"
            or loop.get("unknown_semantics") != "retain_consequences"):
        if loop.get("gamma_conflict_policy") == "validated_gamma_wins":
            loop["gamma_conflict_policy"] = "quarantine"
        if loop.get("llm_check_base") == "refreshed":
            loop["llm_check_base"] = "batch_start"


def _needs_independent_outputs(data):
    """Whether the run recalibrates the fallback or the commit gate per model."""
    fallback = data.get("fallback") or {}
    return bool(
        (fallback.get("enabled") and fallback.get("calibration") == "independent_calibration")
        or data["loop"].get("commit_gate", "none") == "independent_calibration"
    )


def label_manifest(workspace, variant="main", supervision_protocol="as_declared"):
    if (
        read(Path(workspace) / "DATA_AUDIT.json").get("application_context", {}).get("status")
        != "verified_complete_declared_split_axis"
    ):
        raise ValueError(
            "a complete declared application context audit is required before calibration/labeling"
        )
    data = manifest(workspace, "labeling_manifest")
    if variant == "main" and supervision_protocol == "as_declared" and (
        data["loop"].get("label_budget") == 0 or data["oracle"]["kind"] == "disabled"
    ):
        raise ValueError(
            "Main requires an external answer provider and a positive or unlimited document budget; "
            "use the explicit fixed_gamma supervision protocol for a zero-query control"
        )
    data["program"] = file_record(program(workspace))
    data["strategy"] = "paper"
    data["weighting"] = "direct_rule_precision_discount"
    data["feedback"].update(
        trusted_replay=data["feedback"].get("trusted_replay", False),
        source_filter="none",
        receiver_policy="same_model",
        disagreement_discount=data["feedback"].get("disagreement_discount", 0.5),
        candidate_check="checked",
        rule_reliability_update="current_model",
        check_failure_policy="halt",
    )
    data["loop"]["execution"] = "incremental"
    data["loop"]["saturation_rounds"] = 1
    data["loop"]["conflict_policy"] = "unknown_arbitrate"
    data["loop"]["unknown_semantics"] = "retain_consequences"
    data.setdefault(
        "arbitration",
        dict(alias=data.get("aux_extension", {}).get("source_alias", data["dynamic"]["alias"]),
             default_threshold=0.5, thresholds={}),
    )
    if variant in (
        "uniform",
        "direct_rule_precision",
        "direct_rule_precision_discount",
        "complete_provenance",
        "complete_provenance_discount",
        "reliability_only",
        "provenance",
        "confidence_only",
    ):
        data["weighting"] = variant
    elif variant in ("frozen", "frozen_zero_budget", "self_train", "supervised"):
        data["strategy"], data["weighting"] = (
            "frozen" if variant == "frozen_zero_budget" else variant
        ), "uniform"
        data["feedback"]["trusted_replay"] = variant == "supervised"
        if variant == "supervised":
            data["feedback"]["trusted_replay_fraction"] = 1.0
            data["feedback"]["rule_feedback_scope"] = "all"
        if variant == "frozen_zero_budget":
            data["loop"]["label_budget"] = 0
            data["loop"]["source_review_budget"] = 0
            data["oracle"] = dict(kind="disabled", directory=None, allow_known_source_review=False)
        if variant == "self_train":
            data["feedback"]["candidate_check"] = "nocheck"
    elif variant == "nocheck":
        data["feedback"]["candidate_check"] = "nocheck"
    elif variant == "noinc":
        data["loop"]["execution"] = "full_recompute"
    elif variant == "halt_conflicts":
        data["loop"]["conflict_policy"] = "halt"
    elif variant == "unknown_full_rebuild":
        data["loop"]["unknown_semantics"] = "full_rebuild"
    elif variant == "trusted_replay":
        data["feedback"]["trusted_replay"] = True
        data["feedback"]["trusted_replay_fraction"] = 1.0
    elif variant == "exclude_receiver":
        data["feedback"]["source_filter"] = "exclude_receiver"
    elif variant == "affected_channels_to_aux":
        data["feedback"]["receiver_policy"] = "affected_channels_to_aux"
    elif variant == "frozen_prior":
        data["feedback"]["rule_reliability_update"] = "frozen_prior"
    elif variant == "nollm":
        if (
            data["oracle"]["kind"] != "directory_llm_then_human"
            or data["loop"].get("label_budget") == 0
        ):
            raise ValueError(
                "NoLLM needs a nonzero-budget LLM/human protocol with the same human answer directory"
            )
        data["oracle"]["kind"] = "directory_human"
    elif variant in ("random", "uncertainty"):
        if data["loop"].get("label_budget") == 0 or data["oracle"]["kind"] == "disabled":
            raise ValueError(
                "query-policy comparison requires a nonzero budget and matching answer provider"
            )
        data["query"]["strategy"] = variant
    elif variant == "exact_llm_check":
        data["loop"]["llm_check_base"] = "refreshed"
    elif variant not in ("main", "calibration", "calibration_halt", "calibration_full_rebuild",
                         "original_semantics", "calibration_original"):
        raise ValueError("unknown labeling variant")
    if variant in CALIBRATION_VARIANTS:
        data["weighting"] = "uniform"
    if variant == "calibration_halt":
        data["loop"]["conflict_policy"] = "halt"
    if variant == "calibration_full_rebuild":
        data["loop"]["unknown_semantics"] = "full_rebuild"
    if variant in ORIGINAL_SEMANTICS_VARIANTS:
        _original_semantics(data)
    _adapt_output_controls(data)
    if supervision_protocol not in ("as_declared", "fixed_gamma"):
        raise ValueError("explicit as_declared or fixed_gamma supervision protocol required")
    if supervision_protocol == "fixed_gamma":
        if variant in ("nollm", "random", "uncertainty"):
            raise ValueError("annotation-policy experiments require actual external supervision")
        data["loop"]["label_budget"] = 0
        data["loop"]["query_cadence"] = "after_saturation"
        data["feedback"]["rule_feedback_scope"] = "all"
        data["loop"]["source_review_budget"] = 0
        data["oracle"] = dict(kind="disabled", directory=None, allow_known_source_review=False)
    from loris.labeling.feedback import RELIABILITY_WEIGHTINGS, canonical_weighting

    # The independent calibration artifact of this variant's semantics: rule
    # reliability for q weightings, and the M0 fallback thresholds / gate
    # baseline plus per-model recalibration for every variant that declares
    # them (frozen, uniform, self-training and supervised controls included).
    # The calibration run itself creates it.
    data["source_reliability"] = (
        file_record(_calibration_directory(workspace, data, variant) / "RELIABILITY.json")
        if variant not in CALIBRATION_VARIANTS and (
            canonical_weighting(data["weighting"]) in RELIABILITY_WEIGHTINGS
            or _needs_independent_outputs(data))
        else None
    )
    return data


def evaluate_final(workspace, variant, output, *, require_round_metrics=False):
    from .evaluation import metrics

    workspace, output = Path(workspace), Path(output)
    run = workspace / "labeling" / variant
    complete = read(run / "COMPLETE.json")
    protocol = manifest(workspace, "evaluation")
    if protocol.get("schema") != "loris_final_evaluation_v1" or protocol["role"] not in (
        "dev",
        "test",
        "interactive_test",
    ):
        raise ValueError("declare dev/test/interactive_test evaluation scope")
    app_path = workspace / "effective" / ("labeling_" + variant + ".json")
    app = read(app_path)
    state_evidence = completion_state_evidence(complete, app, run)
    input_receipt = read(run / "INPUT_RECEIPT.json")
    if input_receipt["manifest"]["sha256"] != file_record(app_path)["sha256"]:
        raise ValueError("effective labeling manifest changed since inference")
    with bound(app["context"], workspace).open("rb") as stream:
        context = pickle.load(stream)  # SHA-bound, trusted-origin local asset only.
    index = {x: i for i, x in enumerate(context["row_ids"])}
    with np.load(bound(protocol["truth"], workspace), allow_pickle=False) as truth:
        ids = truth["row_ids"].tolist()
        if len(ids) != len(set(ids)) or truth["labels"].tolist() != context["labels"]:
            raise ValueError("evaluation ID or label axis mismatch")
        if not set(ids) <= set(index):
            raise ValueError("evaluation documents absent from final inference context")
        if not np.isin(truth["targets"], [0, 1]).all():
            raise ValueError("evaluation truth must contain binary labels")
        y = np.array(truth["targets"], dtype=bool)
        if y.shape != (len(ids), len(context["labels"])):
            raise ValueError("evaluation truth dimensions differ")
    rows = np.asarray([index[x] for x in ids], dtype=int)
    initial_known = initial_observations(context)
    if protocol["role"] in ("test", "interactive_test") and initial_known[rows].any():
        raise ValueError(
            "test scoring rows already contain initial Gamma labels; paper test labels must start masked"
        )
    initial_training = set(context.get("fit_indices", ()))
    online_training = {
        i for i, role in enumerate(context["row_roles"]) if role in app["domains"]["training_roles"]
    }
    acquired = set(complete.get("acquired_documents", ()))
    if protocol["role"] in ("test", "interactive_test") and initial_training & set(rows):
        raise ValueError("test overlaps initial supervised model-fit rows")
    independent_sources = []
    if "reliability_protocol" in read(workspace / "ASSETS.json")["artifacts"]:
        independence = manifest(workspace, "reliability_protocol")
        independent_sources.append(file_record(asset(workspace, "reliability_protocol")))
        if protocol["role"] in ("test", "interactive_test"):
            for field in ("model_fit_row_ids", "rule_fit_row_ids", "calibration_row_ids"):
                if set(ids) & set(independence[field]):
                    raise ValueError("test overlaps independent-protocol " + field)
    if protocol["role"] == "test" and online_training & set(rows):
        raise ValueError("declared target online adaptation requires interactive_test scope")
    if protocol["role"] == "test" and acquired & set(rows):
        raise ValueError("queried test rows require interactive_test scope")
    with np.load(run / "FINAL_P.npz", allow_pickle=False) as facts:
        final_evidence = final_arrays_evidence(facts, context, complete)
        pos, neg = facts["positive"], facts["negative"]
        selected = pos[rows]
        scores = metrics(y, selected)
        fallback_accounting = None
        if "fallback_positive" in facts.files:
            # Final metrics include the undecided-cell Aux fallback; the rule
            # output alone (logical positives plus arbitration) is reported too.
            rule_output = facts["logical_positive"] | facts["arbitration_positive"]
            fallback_accounting = dict(
                filled_cells=int(facts["fallback_positive"][rows].sum()),
                rule_output_metrics=metrics(y, rule_output[rows]),
                final_prediction_metrics_include_fallback=True,
            )
        unanswered = np.asarray([int(r) not in acquired for r in rows])
        scores_unanswered = (
            metrics(y[unanswered], selected[unanswered]) if unanswered.any() else None
        )
        conflict_accounting = None
        if (
            state_evidence["current_Gamma_has_quarantined_fixpoint"]
            or state_evidence["current_history_has_quarantined_fixpoint"]
        ):
            fields = (
                "logical_positive",
                "logical_negative",
                "conflict_unknown",
                "remaining_unknown",
                "arbitration_positive",
                "arbitration_negative",
            )
            if any(
                name not in facts.files
                or facts[name].shape != pos.shape
                or facts[name].dtype != np.bool_
                for name in fields
            ):
                raise ValueError(
                    "quarantined evaluation requires all final logical/unknown/arbitration masks"
                )
            conflict_accounting = dict(
                scope="signed label cells on declared evaluation rows; final metrics include temporary arbitration",
                counts={name: int(facts[name][rows].sum()) for name in fields},
                final_logical_metrics=metrics(y, facts["logical_positive"][rows]),
                final_prediction_metrics_include_arbitration=True,
            )
    from .round_evaluation import evaluate_rounds

    round_evaluation = evaluate_rounds(run, complete, context, ids, y)
    if require_round_metrics and round_evaluation["status"] != "complete":
        raise ValueError(
            "RQ2 requires complete saved evaluation rows in every round; "
            + round_evaluation["status"]
        )
    providers = ("human", "simulated_human", "llm", "simulated_llm")
    annotation = dict(
        status="unknown_missing_document_source_evidence",
        evaluation_documents=len(rows),
        accepted_documents={name: None for name in providers},
        human_equivalent_unique_documents=None,
        human_source_mode="unknown",
        human_equivalent_automatic_percent=None,
        automatic_percent_status="unknown_missing_document_source_evidence",
        counts_scope="unique accepted annotation documents intersected with actual evaluation rows",
        provider_calls_are_not_human_document_counts=True,
    )
    sources = complete.get("annotation_documents")
    if sources is not None:
        if not isinstance(sources, dict) or set(sources) != set(providers):
            raise ValueError("annotation document source evidence requires all four provider axes")
        selected_sources = {}
        for provider in providers:
            supplied = sources[provider]
            if (
                not isinstance(supplied, list)
                or any(type(row) is not int or not 0 <= row < len(index) for row in supplied)
                or len(set(supplied)) != len(supplied)
            ):
                raise ValueError(
                    "annotation document source evidence contains invalid or repeated row indices"
                )
            selected_sources[provider] = set(supplied) & set(rows)
        human = selected_sources["human"] | selected_sources["simulated_human"]
        mode = (
            "mixed"
            if selected_sources["human"] and selected_sources["simulated_human"]
            else (
                "actual"
                if selected_sources["human"]
                else "simulated" if selected_sources["simulated_human"] else "none_observed"
            )
        )
        resolved = bool(len(rows) and selected.any(axis=1).all())
        eligible = (
            complete["status"] == "all_documents_resolved"
            and complete.get("task_complete") is True
            and resolved
        )
        annotation.update(
            status="verified_document_source_evidence",
            accepted_documents={name: len(values) for name, values in selected_sources.items()},
            human_equivalent_unique_documents=len(human),
            human_source_mode=mode,
            evaluation_all_documents_resolved=resolved,
            human_equivalent_automatic_percent=(
                100.0 * (1.0 - len(human) / len(rows)) if eligible else None
            ),
            automatic_percent_status=(
                "all_resolved" if eligible else "not_a_complete_resolved_evaluation"
            ),
            simulated_human_counts_are_human_equivalents=True,
        )
    trace_files = sorted(run.glob("update_*/ACTUAL_TRAINING_TRACE.json"))
    sampled = set()
    trace_records = []
    for trace_path in trace_files:
        trace = read(trace_path)
        trace_rows = trace["sampled_rows"]
        if any(type(row) is not int or not 0 <= row < len(index) for row in trace_rows):
            raise ValueError("training trace contains row indices outside the application context")
        sampled.update(trace_rows)
        trace_records.append(file_record(trace_path))
    trace_complete = (
        type(complete.get("training_attempts")) is int
        and len(trace_files) == complete["training_attempts"]
    )
    adaptation = dict(
        declared_online_training_roles=app["domains"]["training_roles"],
        declared_online_evaluation_roles=sorted(
            {context["row_roles"][int(row)] for row in rows if int(row) in online_training}
        ),
        declared_online_evaluation_documents=len(online_training & set(rows)),
        declared_permission_is_not_actual_training=True,
        actual_optimizer_updates=complete.get("actual_optimizer_updates"),
        committed_model_updates=complete.get("committed_model_updates"),
        training_trace_complete=trace_complete,
        actual_sampled_evaluation_documents=len(sampled & set(rows)) if trace_complete else None,
        sampled_training_scope="sampler observations across all attempted continuations, including uncommitted/failed candidates; not a claim of committed gradient use",
        training_trace_evidence=trace_records,
        supervision_evidence=[
            dict(arrays=file_record(p), receipt=file_record(p.with_name("RECEIPT.json")))
            for p in sorted(run.glob("supervision_*/ARRAYS.npz"))
        ],
        initial_fit_protection="context.fit_indices plus available independent-protocol model/rule/calibration row IDs",
        independent_protocol_evidence=independent_sources,
    )
    output.mkdir(parents=True, exist_ok=False)
    selected_program = read(bound(app["program"], workspace)) if "program" in app else {}
    control_path = workspace / "effective" / ("labeling_" + variant + ".experiment.json")
    control = read(control_path) if control_path.is_file() else {}
    if control and bound(control["effective_manifest"], workspace) != app_path.resolve():
        raise ValueError("experiment control receipt binds another effective labeling manifest")
    result = dict(
        status="evaluated",
        variant=variant,
        evaluation_role=protocol["role"],
        run_status=complete["status"],
        task_complete=complete.get("task_complete", False),
        resource_cap_is_not_convergence=True,
        **state_evidence,
        conflict_accounting=conflict_accounting,
        fallback_accounting=fallback_accounting,
        final_array_validation=final_evidence,
        round_evaluation=round_evaluation,
        declared_label_budget=app.get("loop", {}).get("label_budget"),
        strategy=app.get("strategy"),
        weighting=app.get("weighting"),
        supervision_protocol=control.get("supervision_protocol", "unknown_legacy"),
        same_budget_is_not_same_supervision=True,
        experiment_control=file_record(control_path) if control else None,
        discovery_proposal_strategy=selected_program.get(
            "proposal_strategy", "historical_unspecified"
        ),
        discovery_admission_order=selected_program.get("admission_order", "historical_unspecified"),
        annotation_evaluation=annotation,
        online_adaptation=adaptation,
        metrics=scores,
        not_directly_answered_metrics=scores_unanswered,
        labels=context["labels"],
        complete=file_record(run / "COMPLETE.json"),
        facts=file_record(run / "FINAL_P.npz"),
        evaluation_manifest=file_record(asset(workspace, "evaluation")),
        new_training=0,
        new_inference=0,
    )
    write(output / "RESULTS.json", result)
    return result


def collect(workspace, output):
    workspace, output = Path(workspace), Path(output)
    rows = []
    rounds = []
    predicates = []
    for p in sorted((workspace / "evaluation").glob("*/RESULTS.json")):
        item = read(p)
        m = item["metrics"]
        completion = read(bound(item["complete"], p.parent))
        if not evaluation_has_valid_state(item) or final_state_kind(item) != final_state_kind(
            completion
        ):
            raise ValueError("evaluation and completion must certify the same explicit final state")
        rows.append(
            dict(
                variant=item["variant"],
                evaluation_role=item["evaluation_role"],
                run_status=item["run_status"],
                task_complete=item.get("task_complete", False),
                discovery_proposal_strategy=item.get("discovery_proposal_strategy"),
                declared_label_budget=item.get("declared_label_budget"),
                weighting=item.get("weighting"),
                supervision_protocol=item.get("supervision_protocol", "unknown_legacy"),
                final_state_kind=item.get("final_state_kind"),
                current_Gamma_has_valid_closure=item.get("current_Gamma_has_valid_closure", False),
                current_Gamma_has_quarantined_fixpoint=item.get(
                    "current_Gamma_has_quarantined_fixpoint", False
                ),
                current_history_has_quarantined_fixpoint=item.get(
                    "current_history_has_quarantined_fixpoint", False
                ),
                conflict_policy=item.get("conflict_policy"),
                unknown_semantics=item.get("unknown_semantics"),
                active_quarantine_pairs=item.get("active_quarantine_pairs"),
                effective_execution=item.get("effective_execution"),
                full_recompute_reason=item.get("full_recompute_reason"),
                macro_f1=m["macro_f1"],
                micro_f1=m["micro_f1"],
                TP=m["TP"],
                FP=m["FP"],
                FN=m["FN"],
                coverage=m["coverage"],
                multi_label_completeness=m["multi_label_completeness"],
            )
        )
        for key in (
            "logical_positive",
            "logical_negative",
            "conflict_unknown",
            "remaining_unknown",
            "arbitration_positive",
            "arbitration_negative",
        ):
            rows[-1]["evaluation_" + key + "_cells"] = (
                (item.get("conflict_accounting") or {}).get("counts", {}).get(key)
            )
        for key in (
            "elapsed_seconds",
            "actual_Chase_calls",
            "training_attempts",
            "actual_optimizer_updates",
            "committed_model_updates",
            "actual_label_requests",
            "actual_source_reviews",
        ):
            rows[-1][key] = completion.get(key)
        for key in ("full", "chaseInc", "chaseDelta", "calibration"):
            rows[-1]["calls_" + key] = completion.get("inference_counts", {}).get(key)
        for key in (
            "chase_seconds",
            "refinement_seconds",
            "influence_seconds",
            "external_seconds",
            "calibration_seconds",
            "feedback_source_seconds",
        ):
            rows[-1][key] = completion.get("timings", {}).get(key)
        rows[-1]["rule_reliability_update"] = completion.get("rule_reliability_update")
        for key in (
            "human_answers",
            "simulated_human_answers",
            "llm_proposals",
            "simulated_llm_proposals",
            "llm_accepted",
        ):
            rows[-1]["provider_calls_" + key] = completion.get("annotation_sources", {}).get(key)
        annotation = item.get("annotation_evaluation", {})
        rows[-1]["annotation_document_evidence_status"] = annotation.get("status", "unknown")
        rows[-1]["evaluation_documents"] = annotation.get("evaluation_documents")
        for key in ("human", "simulated_human", "llm", "simulated_llm"):
            rows[-1]["evaluation_accepted_" + key + "_documents"] = annotation.get(
                "accepted_documents", {}
            ).get(key)
        for key in (
            "human_equivalent_unique_documents",
            "human_source_mode",
            "human_equivalent_automatic_percent",
            "automatic_percent_status",
        ):
            rows[-1][key] = annotation.get(key)
        adaptation = item.get("online_adaptation", {})
        for key in (
            "declared_online_evaluation_documents",
            "actual_sampled_evaluation_documents",
            "training_trace_complete",
        ):
            rows[-1][key] = adaptation.get(key)
        rows[-1]["declared_online_evaluation_roles"] = ";".join(
            adaptation.get("declared_online_evaluation_roles", [])
        )
        calibration_stage = (
            "calibrate_calibration_halt"
            if item.get("conflict_policy") == "halt"
            else (
                "calibrate_calibration_full_rebuild"
                if item.get("unknown_semantics") == "full_rebuild"
                else "calibrate_calibration"
            )
        )
        initial_calibration = workspace / "receipts" / (calibration_stage + ".json")
        initial_receipt = read(initial_calibration) if initial_calibration.is_file() else {}
        rows[-1]["initial_shared_calibration_stage_seconds"] = (
            initial_receipt.get("seconds") if initial_receipt.get("status") == "complete" else None
        )
        round_report = item.get("round_evaluation")
        round_items = (
            round_report["rounds"] if round_report else completion.get("round_history", [])
        )
        for item_round in round_items:
            change = item_round.get("decision_change") or {}
            row = dict(
                variant=item["variant"],
                round=item_round["round"],
                version=item_round["version"],
                decision_change_fraction=change.get("fraction"),
                decision_change_count=change.get("changed"),
                decision_occurrences=change.get("total"),
                evaluation_source=(
                    "independent_offline_truth" if round_report else "legacy_context_truth"
                ),
                scoring_status=item_round.get("scoring_status", "legacy_unspecified"),
                scored_documents=item_round.get("scored_documents"),
                expected_documents=item_round.get("expected_documents"),
            )
            # Older producer naming meant final output, including arbitration.
            item_round = dict(item_round)
            item_round.setdefault("output_metrics", item_round.get("labeling_metrics"))
            for source, prefix in (
                ("output_metrics", "output"),
                ("logical_metrics", "logical"),
                ("aux_model_metrics_at_0_5", "aux_at_0_5"),
            ):
                for metric in ("macro_f1", "micro_f1", "coverage", "multi_label_completeness"):
                    row[prefix + "_" + metric] = (item_round.get(source) or {}).get(metric)
            rounds.append(row)
            for occurrence, score in item_round.get("dynamic_predicate_F1", {}).items():
                predicates.append(
                    dict(
                        variant=item["variant"],
                        round=item_round["round"],
                        occurrence=occurrence,
                        label_index=item_round["occurrence_labels"][occurrence],
                        scoring_status=item_round.get("scoring_status", "legacy_unspecified"),
                        scored_documents=item_round.get("scored_documents"),
                        expected_documents=item_round.get("expected_documents"),
                        macro_f1=(score or {}).get("macro_f1"),
                        micro_f1=(score or {}).get("micro_f1"),
                    )
                )
    if not rows:
        raise ValueError("no completed evaluation receipts; run evaluate first")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "RESULTS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, values in (("ROUNDS.csv", rounds), ("DYNAMIC_PREDICATES.csv", predicates)):
        if values:
            with (output / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    write(
        output / "SUMMARY.json",
        dict(
            rows=rows,
            unexecuted_results_not_filled=True,
            no_test_based_configuration_selection=True,
            global_optimum_claimed=False,
            complete_main_result_claimed=bool([r for r in rows if r["variant"] == "main"])
            and all(r["task_complete"] for r in rows if r["variant"] == "main"),
            round_metrics_scope="post_run_declared_scoring_roles; separate from final evaluation protocol",
            matched_budget_does_not_certify_matched_supervision=True,
            fixed_gamma_comparison=(
                file_record(workspace / "FIXED_GAMMA_COMPARISON.json")
                if (workspace / "FIXED_GAMMA_COMPARISON.json").is_file()
                else None
            ),
        ),
    )
    return dict(status="collected", rows=len(rows), output=str(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("bind")
    b.add_argument("--assets", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--asset-root")
    d = sub.add_parser("audit-data")
    d.add_argument("--manifest", required=True)
    d.add_argument("--output", required=True)
    for name in ("prepare-data", "audit", "rules", "calibrate", "label", "evaluate", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--workspace", required=True)
        if name in ("prepare-data", "rules", "calibrate", "label"):
            p.add_argument("--dry-run", action="store_true")
        if name == "rules":
            p.add_argument(
                "--proposal-strategy",
                choices=("accuracy_guided", "random_joint"),
                default="accuracy_guided",
            )
        if name == "calibrate":
            p.add_argument(
                "--unknown-semantics",
                choices=("retain_consequences", "full_rebuild"),
                default="retain_consequences",
                help="Calibrate with the actual selected unknown protocol; full_rebuild uses calibration-full-rebuild.",
            )
            p.add_argument(
                "--conflict-policy",
                choices=("unknown_arbitrate", "halt"),
                default="unknown_arbitrate",
                help="Actually recalibrate under the selected inference semantics; halt uses a separate calibration-halt output.",
            )
            p.add_argument(
                "--semantics",
                choices=("current", "original"),
                default="current",
                help="original calibrates the original_semantics control into calibration-original.",
            )
        if name in ("label", "evaluate"):
            p.add_argument("--variant", choices=VARIANTS, default="main")
        if name == "evaluate":
            p.add_argument(
                "--require-round-metrics",
                action="store_true",
                help="RQ2: reject missing or incomplete saved prediction rows for any round",
            )
        if name == "label":
            p.add_argument(
                "--supervision-protocol",
                choices=("as_declared", "fixed_gamma"),
                default="as_declared",
            )
    args = parser.parse_args(argv)
    if args.stage == "bind":
        return bind_assets(args.assets, args.output, args.asset_root)
    if args.stage == "audit-data":
        return audit_dataset(args.manifest, args.output)
    work = Path(args.workspace).resolve(strict=True)
    if args.stage == "prepare-data":
        return execute(
            work,
            "prepare-data",
            [
                sys.executable,
                "-m",
                "loris.discovery",
                "prepare-data",
                "--manifest",
                str(asset(work, "dataset_source")),
                "--output",
                str(work / "prepared"),
            ],
            args.dry_run,
        )
    if args.stage == "audit":
        conf = read(work / "ASSETS.json")
        result = audit_dataset(
            asset(work, "dataset_audit"), asset_root=conf["asset_root"], repo_root=conf["repo_root"]
        )
        result = audit_application(work, result)
        write(work / "DATA_AUDIT.json", result)
        return result
    require_audit(work)
    if args.stage == "rules":
        conf = read(work / "ASSETS.json")
        if conf["rules_mode"] == "reuse":
            if args.proposal_strategy == "random_joint":
                raise ValueError(
                    "random-joint discovery comparison needs its own new discovery workspace"
                )
            result = check_execution_program(program(work), asset(work, "cached_completion"))
            result.update(scope=conf["scope"], new_BO=0, new_training=0)
            if not args.dry_run:
                write(work / "RULE_REUSE.json", result)
            return result
        data = manifest(work, "discovery_manifest")
        data["run"]["proposal_strategy"] = args.proposal_strategy
        data["run"]["admission_order"] = (
            "random_consistent"
            if args.proposal_strategy == "random_joint"
            else "standalone_score_desc"
        )
        if args.proposal_strategy == "random_joint":
            data["run"]["initializer_slots"] = 0
        path = work / "effective/discovery.json"
        if not args.dry_run:
            write(path, data)
        return execute(
            work,
            "discovery",
            [
                sys.executable,
                "-m",
                "loris.discovery",
                "learn-rules",
                "--manifest",
                str(path),
                "--proposal-strategy",
                args.proposal_strategy,
                "--admission-order",
                data["run"]["admission_order"],
                "--initializer-slots",
                str(data["run"]["initializer_slots"]),
                "--output",
                str(work / "discovery"),
            ],
            args.dry_run,
        )
    if args.stage in ("calibrate", "label"):
        if args.stage == "calibrate" and args.semantics == "original" and (
                args.conflict_policy != "unknown_arbitrate"
                or args.unknown_semantics != "retain_consequences"):
            raise ValueError("the original-semantics control uses the main conflict protocol")
        variant = (
            (
                "calibration_halt"
                if args.conflict_policy == "halt"
                else (
                    "calibration_full_rebuild"
                    if args.unknown_semantics == "full_rebuild"
                    else (
                        "calibration_original" if args.semantics == "original"
                        else "calibration"
                    )
                )
            )
            if args.stage == "calibrate"
            else args.variant
        )
        supervision_protocol = args.supervision_protocol if args.stage == "label" else "as_declared"
        data = label_manifest(work, variant, supervision_protocol)
        path = work / "effective" / ("labeling_" + variant + ".json")
        if not args.dry_run:
            write(path, data)
            if args.stage == "label":
                write(
                    path.with_suffix(".experiment.json"),
                    dict(
                        schema="loris_experiment_control_v1",
                        variant=variant,
                        supervision_protocol=(
                            "explicit_zero_budget"
                            if variant == "frozen_zero_budget"
                            and supervision_protocol == "as_declared"
                            else supervision_protocol
                        ),
                        effective_manifest=file_record(path),
                        initial_context=data["context"],
                        program=data["program"],
                        initial_checkpoint=data["dynamic"]["checkpoint"],
                        weighting=data["weighting"],
                        equal_external_permission_does_not_mean_equal_supervision=True,
                        fixed_gamma_requires_postrun_comparison=supervision_protocol
                        == "fixed_gamma",
                    ),
                )
        if args.stage == "calibrate":
            from loris.labeling.reliability import chase_semantics

            protocol = manifest(work, "reliability_protocol")
            # This is a new input plan for a real estimation run, not a relabelled
            # artifact. The independent context, answers and fit/split IDs stay fixed.
            protocol["model_binding"]["chase_semantics"] = chase_semantics(data)
            suffix = (
                "_halt"
                if args.conflict_policy == "halt"
                else ("_full_rebuild" if args.unknown_semantics == "full_rebuild"
                      else ("_original" if args.semantics == "original" else ""))
            )
            pp = work / "effective" / ("independent_calibration" + suffix + ".json")
            if not args.dry_run:
                write(pp, protocol)
            argv = [
                sys.executable,
                "-m",
                "loris.labeling",
                "calibrate-reliability",
                "--manifest",
                str(path),
                "--protocol",
                str(pp),
                "--output",
                str(_calibration_directory(work, data, variant)),
            ]
        else:
            argv = [
                sys.executable,
                "-m",
                "loris.labeling",
                "run",
                "--manifest",
                str(path),
                "--output",
                str(work / "labeling" / variant),
            ]
        return execute(work, args.stage + "_" + variant, argv, args.dry_run)
    if args.stage == "evaluate":
        return evaluate_final(
            work,
            args.variant,
            work / "evaluation" / args.variant,
            require_round_metrics=args.require_round_metrics,
        )
    return collect(work, work / "report")
