"""Explicit paper-aligned labeling and independent rule calibration manifests."""

import argparse
import copy
import json
from pathlib import Path
import pickle

import numpy as np

from . import SCHEMA
from .query_policy import CADENCES, QUERY_POOLS, parse_budget
from .feedback import RECEIVER_POLICIES, RELIABILITY_WEIGHTINGS, WEIGHTINGS, canonical_weighting
from loris.discovery.manifest import bound, file_record


def validate(data):
    required = {
        "schema",
        "context",
        "program",
        "dynamic",
        "domains",
        "loop",
        "feedback",
        "calibration",
        "oracle",
        "strategy",
        "weighting",
        "source_reliability",
        "notes",
        "query",
    }
    if (
        not isinstance(data, dict)
        or not required <= set(data)
        or set(data) - required - {"arbitration", "fallback", "aux_extension"}
        or data["schema"] != SCHEMA
    ):
        raise ValueError("explicit labeling manifest fields/schema required")
    if data["strategy"] not in ("frozen", "supervised", "self_train", "paper"):
        raise ValueError("unsupported strategy/weighting")
    weighting = canonical_weighting(data["weighting"])
    if data["strategy"] == "self_train" and weighting in RELIABILITY_WEIGHTINGS:
        raise ValueError(
            "SelfTrain predictions cannot use reliability of the independent logical source"
        )
    dynamic = data["dynamic"]
    dynamic_required = {
        "alias",
        "checkpoint",
        "backend_source",
        "configuration",
        "device",
        "gpu_lock",
        "ensemble_dependencies",
        "probability_lineage",
    }
    if not dynamic_required <= set(dynamic) or set(dynamic) - dynamic_required - {
            "initialization", "amp_skip_tolerance", "prediction_amp", "prediction_length_sort",
            "amp_init_scale", "reuse_application_predictions"}:
        raise ValueError("explicit dynamic model, checkpoint, config and ensemble policy required")
    from .trainer import AMP_MODES

    if not isinstance(dynamic["configuration"], dict) or dynamic["configuration"].get(
            "amp", "bf16") not in AMP_MODES:
        raise ValueError("dynamic configuration amp must be fp16, bf16, none or auto")
    scale = dynamic.get("amp_init_scale")
    if scale is not None and (type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0):
        raise ValueError("amp_init_scale must be a positive finite number")
    tolerance = dynamic.get("amp_skip_tolerance", 0.0)
    if type(tolerance) not in (int, float) or not 0 <= tolerance < 0.5:
        raise ValueError("amp_skip_tolerance must be a fraction in [0, 0.5)")
    if dynamic.get("prediction_amp") not in (None, "none", "fp16", "bf16"):
        raise ValueError("prediction_amp must be none, fp16 or bf16")
    if type(dynamic.get("prediction_length_sort", False)) is not bool:
        raise ValueError("prediction_length_sort must be a bool")
    if type(dynamic.get("reuse_application_predictions", False)) is not bool:
        raise ValueError("reuse_application_predictions must be a bool")
    initialization = dynamic.get("initialization", "previous")
    if initialization not in ("previous", "m0"):
        raise ValueError("Aux initialization must be previous or m0")
    if (
        not isinstance(dynamic["alias"], str)
        or not dynamic["alias"]
        or dynamic["device"] not in ("cpu", "cuda")
    ):
        raise ValueError("invalid dynamic alias/device")
    lineage = dynamic["probability_lineage"]
    if (
        not isinstance(lineage, (str, dict))
        or not lineage
        or (isinstance(lineage, str) and not lineage.strip())
    ):
        raise ValueError("probability/checkpoint lineage disclosure required")
    for ensemble in dynamic["ensemble_dependencies"]:
        if (
            set(ensemble) != {"alias", "dependencies", "operation", "vote_threshold"}
            or ensemble["operation"] != "binary_vote_mean"
            or not ensemble["dependencies"]
            or len(set(ensemble["dependencies"])) != len(ensemble["dependencies"])
            or not 0 <= ensemble["vote_threshold"] <= 1
        ):
            raise ValueError("invalid explicit ensemble dependency")
    from .aux_extension import configuration as aux_extension_configuration

    aux_extension_configuration(data)
    if set(data["domains"]) != {"training_roles", "query_roles", "scoring_roles"}:
        raise ValueError("separate train/query/scoring role lists required")
    for roles in data["domains"].values():
        if (
            not isinstance(roles, list)
            or len(set(roles)) != len(roles)
            or any(not isinstance(x, str) for x in roles)
        ):
            raise ValueError("role lists must be explicit unique strings")
    loop = data["loop"]
    positive = {
        "max_inner_rounds",
        "saturation_rounds",
        "max_chase_calls",
        "max_rdg_estimates",
        "max_update_seconds",
    }
    nonnegative = {"source_review_budget"}
    if not (positive | nonnegative) <= set(loop) or set(loop) - positive - nonnegative - {
        "execution",
        "conflict_policy",
        "unknown_semantics",
        "label_budget", "query_batch_size", "query_batch_fraction", "query_batch_min",
        "query_cadence", "intermediate_exports", "history_export_interval",
        "query_answer_mode", "query_pool",
        "gamma_conflict_policy", "commit_gate", "commit_gate_tolerance", "llm_check_base",
        "conflict_update_policy",
        "commit_gate_metric_policy", "commit_gate_comp_tolerance", "commit_gate_fn_tolerance",
        "commit_gate_aux_grid", "joint_gate_partition",
    }:
        raise ValueError("all compute and request budgets must be explicit")
    update_policy = loop.get("conflict_update_policy", "retain_history")
    if update_policy not in ("retain_history", "fresh_from_gamma"):
        raise ValueError("unsupported conflict_update_policy")
    if update_policy == "fresh_from_gamma" and (
        loop.get("conflict_policy") != "unknown_arbitrate"
        or loop.get("unknown_semantics") != "retain_consequences"
    ):
        raise ValueError("fresh_from_gamma requires retained unknown conflict semantics")
    if loop.get("llm_check_base", "batch_start") not in ("batch_start", "refreshed"):
        raise ValueError("llm_check_base must be batch_start or refreshed")
    if loop.get("llm_check_base", "batch_start") == "refreshed" and (
        loop.get("conflict_policy", "halt") != "unknown_arbitrate"
        or loop.get("unknown_semantics", "full_rebuild") != "retain_consequences"
    ):
        raise ValueError("refreshed llm_check_base requires unknown_arbitrate with retain_consequences")
    if loop.get("gamma_conflict_policy", "quarantine") not in ("quarantine", "validated_gamma_wins"):
        raise ValueError("gamma_conflict_policy must be quarantine or validated_gamma_wins")
    if loop.get("gamma_conflict_policy", "quarantine") == "validated_gamma_wins" and (
        loop.get("conflict_policy", "halt") != "unknown_arbitrate"
        or loop.get("unknown_semantics", "full_rebuild") != "retain_consequences"
    ):
        raise ValueError("validated_gamma_wins requires unknown_arbitrate with retain_consequences")
    if loop.get("commit_gate", "none") not in ("none", "independent_calibration"):
        raise ValueError("commit_gate must be none or independent_calibration")
    tolerance = loop.get("commit_gate_tolerance", 0.0)
    if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("commit_gate_tolerance must be a finite nonnegative number")
    gate_policy = loop.get("commit_gate_metric_policy", "mean_f1")
    if gate_policy not in ("mean_f1", "aux_loris_joint"):
        raise ValueError("commit_gate_metric_policy must be mean_f1 or aux_loris_joint")
    for key in ("commit_gate_comp_tolerance",):
        value = loop.get(key, 0.0)
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(key + " must be finite in [0,1]")
    if type(loop.get("commit_gate_fn_tolerance", 0)) is not int or loop.get("commit_gate_fn_tolerance", 0) < 0:
        raise ValueError("commit_gate_fn_tolerance must be a nonnegative integer")
    if gate_policy == "aux_loris_joint":
        if update_policy != "fresh_from_gamma":
            raise ValueError("joint gate requires conflict_update_policy=fresh_from_gamma to match calibration")
        if (loop.get("commit_gate") != "independent_calibration" or tolerance != 0
                or "aux_extension" not in data):
            raise ValueError("joint gate requires independent_calibration, zero mean tolerance, and isolated Aux")
        partition = loop.get("joint_gate_partition")
        if not isinstance(partition, dict) or set(partition) != {"fit_row_ids", "score_row_ids"}:
            raise ValueError("joint gate requires explicit fit and score row IDs")
        for ids in partition.values():
            if (not isinstance(ids, list) or not ids
                    or any(not isinstance(value, str) or not value for value in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("joint gate partitions require nonempty unique string row IDs")
        if set(partition["fit_row_ids"]) & set(partition["score_row_ids"]):
            raise ValueError("joint gate fit and score row IDs must be disjoint")
        from .fallback import DEFAULT_GRID
        grid = loop.get("commit_gate_aux_grid", DEFAULT_GRID)
        if (not isinstance(grid, (list, tuple)) or not grid
                or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in grid)
                or len(set(grid)) != len(grid)):
            raise ValueError("joint gate auxiliary grid must be finite unique cutoffs in [0,1]")
    if "fallback" in data:
        from .fallback import configuration

        configuration(data)
    if loop.get("execution", "incremental") not in ("incremental", "full_recompute"):
        raise ValueError("execution must be incremental or full_recompute")
    if loop.get("conflict_policy", "halt") not in ("halt", "review_instances", "unknown_arbitrate"):
        raise ValueError("unsupported conflict_policy")
    if loop.get("unknown_semantics", "full_rebuild") not in ("retain_consequences", "full_rebuild"):
        raise ValueError("unknown_semantics must be retain_consequences or full_rebuild")
    if loop.get("conflict_policy", "halt") == "unknown_arbitrate" or "arbitration" in data:
        from .conflict_protocol import arbitration_rules

        arbitration_rules(data)
    for key in positive | nonnegative:
        value = loop[key]
        if type(value) is not int or value < (1 if key in positive else 0):
            raise ValueError("invalid budget: " + key)
    if (
        loop["source_review_budget"] > 0
        and loop.get("conflict_policy", "halt") != "review_instances"
    ):
        raise ValueError(
            "positive source_review_budget requires explicit conflict_policy=review_instances"
        )
    budget = loop.get("label_budget")
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("label_budget must be a nonnegative document count or null (unlimited)")
    size = loop.get("query_batch_size")
    if size is not None and (type(size) is not int or size < 1):
        raise ValueError("query_batch_size must be positive or null for fractional batches")
    fraction = loop.get("query_batch_fraction", 0.01)
    if type(fraction) not in (int, float) or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("query_batch_fraction must be in (0,1]")
    for key, default in (("query_batch_min", 32), ("history_export_interval", 10)):
        if type(loop.get(key, default)) is not int or loop.get(key, default) < 1:
            raise ValueError(key + " must be positive")
    if loop.get("query_cadence", "after_saturation") not in CADENCES:
        raise ValueError("unsupported query_cadence")
    if loop.get("query_pool", "uncovered") not in QUERY_POOLS:
        raise ValueError("unsupported query_pool")
    if loop.get("query_answer_mode", "complete") not in ("complete", "allow_partial"):
        raise ValueError("unsupported query_answer_mode")
    if loop.get("intermediate_exports", "periodic") not in ("all", "periodic", "final"):
        raise ValueError("unsupported intermediate_exports")
    if loop["saturation_rounds"] != 1:
        raise ValueError("saturation_rounds must remain 1")
    feedback_required = {
        "trusted_replay",
        "normalization",
        "channel_policy",
        "candidate_check",
        "self_train_confidence",
    }
    if not feedback_required <= set(data["feedback"]) or set(
        data["feedback"]
    ) - feedback_required - {
        "source_filter",
        "disagreement_discount",
        "rule_reliability_update",
        "check_failure_policy",
        "receiver_policy",
        "new_verified_supervision",
        "trusted_replay_fraction",
        "rule_feedback_scope",
        "weak_mass_ratio",
        "weak_rows_per_label",
        "reliability_floor",
        "max_training_rows",
        "verified_replay_policy",
    }:
        raise ValueError("explicit replay/normalization/threshold-channel policy required")
    fb = data["feedback"]
    if fb.get("verified_replay_policy", "pending_batch") not in ("pending_batch", "cumulative_acquired"):
        raise ValueError("verified_replay_policy must be pending_batch or cumulative_acquired")
    if (fb.get("verified_replay_policy") == "cumulative_acquired"
            and fb.get("new_verified_supervision", "full_batch") != "full_batch"):
        raise ValueError("cumulative acquired answers require full_batch supervision")
    if fb.get("max_training_rows") is not None and (
            type(fb["max_training_rows"]) is not int or fb["max_training_rows"] < 1):
        raise ValueError("max_training_rows must be a positive integer or null")
    for key, low, high in (("weak_mass_ratio", 0.0, None), ("reliability_floor", 0.0, 1.0)):
        value = fb.get(key)
        if value is not None and (type(value) not in (int, float) or not np.isfinite(value)
                                  or value < low or (high is not None and value > high)):
            raise ValueError(key + " must be a finite number in range or null")
    if fb.get("weak_rows_per_label") is not None and (
            type(fb["weak_rows_per_label"]) is not int or fb["weak_rows_per_label"] < 1):
        raise ValueError("weak_rows_per_label must be a positive integer or null")
    if fb.get("reliability_floor") is not None and canonical_weighting(data["weighting"]) not in RELIABILITY_WEIGHTINGS:
        raise ValueError("reliability_floor requires a rule reliability weighting")
    if (
        type(data["feedback"]["trusted_replay"]) is not bool
        or data["feedback"]["normalization"] != "common_mass_per_accumulation_window"
        or data["feedback"]["channel_policy"] != "any_actual_occurrence_disagrees_deduplicate_pair"
    ):
        raise ValueError("unsupported feedback policy")
    replay_fraction = data["feedback"].get(
        "trusted_replay_fraction",
        1.0 if data["feedback"]["trusted_replay"] else 0.0,
    )
    if type(replay_fraction) not in (int, float) or not np.isfinite(replay_fraction) or not 0 <= replay_fraction <= 1:
        raise ValueError("trusted_replay_fraction must be finite in [0,1]")
    scope = data["feedback"].get("rule_feedback_scope", "all")
    if scope not in ("all", "new_or_changed"):
        raise ValueError("rule_feedback_scope must be all or new_or_changed")
    if (initialization == "previous" and data["strategy"] == "paper"
            and (not data["feedback"]["trusted_replay"] or replay_fraction == 0)):
        # Continuing the previous model on logical corrections alone trains only
        # on its own disagreements (observed collapse, AAPD v3 2026-09-29).
        raise ValueError("previous-initialization paper refinement requires trusted replay")
    if initialization == "m0" and (
        scope != "all"
        or not data["feedback"]["trusted_replay"]
        or replay_fraction != 1.0
        or loop.get("query_cadence") != "each_closure"
        or loop["max_inner_rounds"] != 1
        or data["feedback"].get("new_verified_supervision") != "full_batch"
    ):
        raise ValueError("m0 requires all current feedback, full accumulated answers, and one update per closure")
    if scope == "new_or_changed" and (
        loop.get("query_cadence") != "each_closure"
        or loop["max_inner_rounds"] != 1
        or data["feedback"].get("new_verified_supervision") != "full_batch"
    ):
        raise ValueError("incremental feedback requires each_closure, one update, and full fresh answers")
    if (
        data["feedback"]["candidate_check"] not in ("checked", "nocheck")
        or not 0.5 < data["feedback"]["self_train_confidence"] <= 1
    ):
        raise ValueError(
            "explicit candidate check and high-confidence SelfTrain threshold required"
        )
    if data["feedback"].get("source_filter", "none") not in ("none", "exclude_receiver"):
        raise ValueError("source_filter must be none or the exclude_receiver ablation")
    if data["feedback"].get("receiver_policy", "same_model") not in RECEIVER_POLICIES:
        raise ValueError(
            "receiver_policy must be same_model or explicit affected_channels_to_aux extension"
        )
    if data["feedback"].get("new_verified_supervision", "full_batch") not in ("corrections_only", "full_batch"):
        raise ValueError("new_verified_supervision must be corrections_only or full_batch")
    if data["feedback"].get("check_failure_policy", "halt") not in ("halt", "reject_batch"):
        raise ValueError("check_failure_policy must be halt or explicit reject_batch extension")
    if data["feedback"].get("rule_reliability_update", "current_model") not in (
        "current_model",
        "frozen_prior",
    ):
        raise ValueError(
            "rule_reliability_update must be current_model or explicit frozen_prior ablation"
        )
    discount = data["feedback"].get("disagreement_discount", 0.5)
    if type(discount) not in (int, float) or not np.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("disagreement_discount must be finite in [0,1]")
    if (
        not {"strategy", "seed"} <= set(data["query"])
        or set(data["query"]) - {"strategy", "seed", "rdg_candidate_limit", "review"}
        or data["query"]["strategy"] not in ("rdg", "random", "uncertainty", "review")
        or type(data["query"]["seed"]) is not int
        or not 0 <= data["query"]["seed"] < 2**32
    ):
        raise ValueError("explicit query strategy and uint32 seed required")
    from .query_selection import configuration as review_configuration
    review_options = review_configuration(data)
    if review_options.get("eligibility_mode", "legacy") == "evidence" and (
            data["query"]["strategy"] != "review" or loop.get("query_pool") != "needs_review"):
        raise ValueError("evidence query eligibility requires strategy=review and pool=needs_review")
    if loop.get("query_pool") == "needs_review" and data["query"]["strategy"] != "review":
        raise ValueError("needs_review requires the review query strategy")
    rdg_limit = data["query"].get("rdg_candidate_limit")
    if rdg_limit is not None and (type(rdg_limit) is not int or rdg_limit <= 0):
        raise ValueError("rdg_candidate_limit must be a positive integer or null")
    if (data["query"]["strategy"] == "rdg" and rdg_limit is not None
            and loop.get("query_batch_size") is not None
            and rdg_limit < loop["query_batch_size"]):
        raise ValueError("rdg_candidate_limit cannot be smaller than query_batch_size")
    if data["strategy"] == "supervised" and not data["feedback"]["trusted_replay"]:
        raise ValueError("supervised control requires actual trusted replay")
    c = data["calibration"]
    if (
        not {"grid", "minimum_support_each_sign"} <= set(c)
        or set(c) - {"grid", "minimum_support_each_sign", "cells", "max_step"}
        or c.get("cells", "all") not in ("all", "trusted_only")
        or (c.get("max_step") is not None and (type(c["max_step"]) not in (int, float)
                                              or not np.isfinite(c["max_step"]) or c["max_step"] <= 0))
        or not isinstance(c["grid"], list)
        or not c["grid"]
        or len(set(c["grid"])) != len(c["grid"])
        or any(
            type(t) not in (int, float) or not np.isfinite(t) or not 0 <= t <= 1 for t in c["grid"]
        )
        or type(c["minimum_support_each_sign"]) is not int
        or c["minimum_support_each_sign"] < 1
    ):
        raise ValueError("explicit finite calibration grid and signed support required")
    oracle = data["oracle"]
    expected = {"kind", "directory", "allow_known_source_review"} | (
        {"manifest"} if oracle.get("kind") == "npz_simulated_human" else set()
    )
    if set(oracle) != expected or type(oracle["allow_known_source_review"]) is not bool:
        raise ValueError("explicit oracle/provider authority required")
    if oracle["kind"] not in (
        "disabled",
        "directory_simulated_human",
        "npz_simulated_human",
        "directory_llm_then_human",
        "directory_human",
    ):
        raise ValueError(
            "unsupported oracle kind; external LLM proposals require directory_llm_then_human"
        )
    if oracle["kind"] in (
        "directory_simulated_human",
        "directory_llm_then_human",
        "directory_human",
    ) and (not isinstance(oracle["directory"], str) or not oracle["directory"]):
        raise ValueError("answer provider directory required")
    if weighting in RELIABILITY_WEIGHTINGS and data["source_reliability"] is None:
        raise ValueError(
            "reliability weighting requires an independent per-rule calibration artifact; uniform/confidence_only are named controls"
        )
    return data


def check_probability_lineage(dynamic, context):
    """Check supplied source metadata, without claiming to prove model execution.

    A human-readable source declaration remains supported. Optional structured
    fields must agree with the already loaded model/table and their axes; users
    are not required to compute or supply additional hashes.
    """
    lineage = dynamic["probability_lineage"]
    checked = []
    if isinstance(lineage, dict):
        expected = dict(
            alias=dynamic["alias"],
            checkpoint_sha256=dynamic["checkpoint"]["sha256"],
            row_ids=list(context["row_ids"]),
            labels=list(context["labels"]),
        )
        if "probabilities_sha256" in lineage:
            from loris._engine.typed_session import array_identity

            expected["probabilities_sha256"] = array_identity(
                context["model_probabilities"][dynamic["alias"]]
            )["sha256"]
        for key, value in expected.items():
            if key in lineage:
                if lineage[key] != value:
                    raise ValueError("probability lineage differs from actual " + key)
                checked.append(key)
    return dict(
        status="declared_fields_match" if checked else "source_declaration_only",
        checked_fields=checked,
        model_execution_proven=False,
    )


def configure_aux_initialization(data, initialization):
    if initialization not in ("previous", "m0"):
        raise ValueError("Aux initialization must be previous or m0")
    data["dynamic"]["initialization"] = initialization
    if initialization == "m0":
        data["loop"].update(query_cadence="each_closure", max_inner_rounds=1)
        data["feedback"].update(
            rule_feedback_scope="all", trusted_replay=True,
            trusted_replay_fraction=1.0, new_verified_supervision="full_batch",
        )


def load(
    path,
    *,
    strategy=None,
    weighting=None,
    candidate_check=None,
    conflict_policy=None,
    unknown_semantics=None,
    check_failure_policy=None,
    receiver_policy=None,
    new_verified_supervision=None,
    arbitration_threshold=None,
    query_strategy=None,
    rdg_candidate_limit=None,
    self_train_confidence=None,
    label_budget=None,
    source_review_budget=None,
    query_batch_size=None,
    query_batch_fraction=None,
    query_batch_min=None,
    query_cadence=None,
    query_pool=None,
    query_answer_mode=None,
    intermediate_exports=None,
    history_export_interval=None,
    source_reliability=None,
    ignore_reliability=False,
    aux_initialization=None,
):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text())
    data = copy.deepcopy(data)
    if strategy is not None:
        data["strategy"] = strategy
    if weighting is not None:
        data["weighting"] = weighting
    if ignore_reliability:
        data["source_reliability"] = None
    if source_reliability is not None:
        data["source_reliability"] = file_record(Path(source_reliability).expanduser())
    if candidate_check is not None:
        data["feedback"]["candidate_check"] = candidate_check
    if conflict_policy is not None:
        data["loop"]["conflict_policy"] = conflict_policy
    if unknown_semantics is not None:
        data["loop"]["unknown_semantics"] = unknown_semantics
    if check_failure_policy is not None:
        data["feedback"]["check_failure_policy"] = check_failure_policy
    if receiver_policy is not None:
        data["feedback"]["receiver_policy"] = receiver_policy
    if new_verified_supervision is not None:
        data["feedback"]["new_verified_supervision"] = new_verified_supervision
    if arbitration_threshold is not None:
        data["arbitration"] = dict(
            alias=data["dynamic"]["alias"], default_threshold=arbitration_threshold, thresholds={}
        )
    if query_strategy is not None:
        data["query"]["strategy"] = query_strategy
    if rdg_candidate_limit is not None:
        data["query"]["rdg_candidate_limit"] = rdg_candidate_limit
    if self_train_confidence is not None:
        data["feedback"]["self_train_confidence"] = self_train_confidence
    for key, value in (
        ("label_budget", label_budget),
        ("source_review_budget", source_review_budget),
        ("query_batch_size", query_batch_size),
        ("query_batch_fraction", query_batch_fraction),
        ("query_batch_min", query_batch_min),
        ("query_cadence", query_cadence),
        ("query_pool", query_pool),
        ("query_answer_mode", query_answer_mode),
        ("intermediate_exports", intermediate_exports),
        ("history_export_interval", history_export_interval),
    ):
        if value is not None:
            data["loop"][key] = None if key == "label_budget" and value == "unlimited" else value
    if query_batch_fraction is not None and query_batch_size is None:
        data["loop"]["query_batch_size"] = None
    data["loop"].setdefault("label_budget", None)
    if aux_initialization is not None:
        configure_aux_initialization(data, aux_initialization)
    validate(data)
    requested_weighting = data["weighting"]
    data["weighting"] = canonical_weighting(requested_weighting)
    data["loop"].setdefault("conflict_policy", "halt")
    # Preserve the semantics of historical manifests. New templates explicitly
    # select retain_consequences; old results are never silently reinterpreted.
    data["loop"].setdefault("unknown_semantics", "full_rebuild")
    data["feedback"].setdefault("check_failure_policy", "halt")
    data["feedback"].setdefault("receiver_policy", "same_model")
    refs = {name: bound(data[name], path.parent) for name in ("context", "program")}
    checkpoint = bound(data["dynamic"]["checkpoint"], path.parent)
    data["dynamic"]["checkpoint"] = file_record(checkpoint)
    backend = (path.parent / data["dynamic"]["backend_source"]).resolve(strict=True)
    if not (backend / "loris_ssl/chase_backend.py").is_file():
        raise ValueError("real optional Aux backend source missing")
    data["dynamic"]["backend_source"] = str(backend)
    if data["dynamic"]["gpu_lock"]:
        data["dynamic"]["gpu_lock"] = str((path.parent / data["dynamic"]["gpu_lock"]).resolve())
    if data["oracle"]["directory"]:
        data["oracle"]["directory"] = str((path.parent / data["oracle"]["directory"]).resolve())
    if data["oracle"]["kind"] == "npz_simulated_human":
        provider = bound(data["oracle"]["manifest"], path.parent)
        data["oracle"]["manifest"] = file_record(provider)
        oracle = json.loads(provider.read_text())
        if oracle.get("context") != data["context"]:
            raise ValueError("oracle is bound to another complete context")
    with refs["context"].open("rb") as f:
        context = pickle.load(f)  # authenticated trusted-local asset only
    from .query_selection import configuration as review_configuration
    if set(review_configuration(data)["aux_thresholds"]) - set(context["labels"]):
        raise ValueError("query Aux threshold label is outside the complete label axis")
    program = json.loads(refs["program"].read_text())
    records = program["rules"] if isinstance(program, dict) else program
    from loris._engine.rdl_reference import Rule

    rules = tuple(Rule.from_dict(r) for r in records)
    from .aux_extension import attach_context, extend_program, partition_audit

    # This is an in-memory extension of bound source artifacts. Never edit the
    # learned program or replace any of its model aliases with the new Aux.
    context = attach_context(context, data)
    rules = extend_program(rules, data, context["labels"])
    aux_partition = partition_audit(rules, data, context["labels"])
    from loris.rule_forms import validate_program_rule_forms

    rule_form_audit = validate_program_rule_forms(rules)
    n, k = len(context["docs"]), len(context["labels"])
    if data["loop"]["conflict_policy"] == "unknown_arbitrate":
        from .conflict_protocol import arbitration_rules

        arbitration_rules(data, context["labels"])
    if (
        len(context["row_ids"]) != n
        or len(set(context["row_ids"])) != n
        or len(set(context["labels"])) != k
        or len(context["row_roles"]) != n
        or context["gamma"].shape != (n, k)
        or np.any(context["gamma"].pos & context["gamma"].neg)
    ):
        raise ValueError("complete unique row/label/signed-root axes required")
    if data["dynamic"]["alias"] not in context["model_probabilities"]:
        raise ValueError("Aux probability channel missing from actual context")
    for alias, values in context["model_probabilities"].items():
        p = np.asarray(values)
        if p.shape != (n, k) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("invalid full model probability axis: " + alias)
    lineage_check = check_probability_lineage(data["dynamic"], context)
    # Certify the purported zero-update interpretation before any output,
    # inference or optimizer work. A declared ensemble cannot quietly replace
    # its old table at startup, nor depend on a later refresh in a bad order.
    specifications = data["dynamic"]["ensemble_dependencies"]
    names = [s["alias"] for s in specifications]
    if len(set(names)) != len(names) or data["dynamic"]["alias"] in names:
        raise ValueError("duplicate ensemble aliases or overriding Aux")
    tables = dict(context["model_probabilities"])
    done = set()
    for spec in specifications:
        alias = spec["alias"]
        deps = spec["dependencies"]
        if alias not in tables or alias in deps or any(a not in tables for a in deps):
            raise ValueError("ensemble has missing or self dependency")
        if any(a in names and a not in done for a in deps):
            raise ValueError("ensemble dependency order is not topological")
        rebuilt = np.mean(
            [np.asarray(tables[a], dtype=np.float64) >= spec["vote_threshold"] for a in deps],
            axis=0,
        ).astype(np.float32)
        if not np.array_equal(rebuilt, tables[alias]):
            raise ValueError("ensemble does not reproduce original zero-update table")
        tables[alias] = rebuilt
        done.add(alias)
    reliability = None
    if data["source_reliability"] is not None:
        rp = bound(data["source_reliability"], path.parent)
        reliability = json.loads(rp.read_text())
        if reliability.get("schema") != "loris_independent_rule_reliability_v2":
            raise ValueError(
                "public labeling requires independent per-rule calibration; historical fit pooled priors and legacy counts are not accepted"
            )
        from .reliability import validate_artifact

        validate_artifact(reliability, context, rules, data)
    from .runner import rule_channels

    from .aux_extension import adaptive_aliases
    affected = {data["dynamic"]["alias"]} | set(adaptive_aliases(data))
    for spec in specifications:
        if affected.intersection(spec["dependencies"]):
            affected.add(spec["alias"])
        else:
            raise ValueError("declared refreshed ensemble does not depend on dynamic Aux")
    channels = rule_channels(rules, affected, tuple(context["labels"]))
    if (
        data["strategy"] != "frozen"
        and not channels
        and data["loop"]["conflict_policy"] != "unknown_arbitrate"
    ):
        raise ValueError("selected program never reads the dynamic Aux or its declared ensembles")
    return (
        data,
        context,
        rules,
        reliability,
        dict(
            manifest=file_record(path),
            refs={k: file_record(v) for k, v in refs.items()},
            documents=n,
            labels=k,
            rules=len(rules),
            rule_form_audit=rule_form_audit,
            dynamic_occurrences=len(channels),
            original_thresholds=[[list(a), list(b)] for a, b in channels.items()],
            declared_ensemble_zero_update_exact=True,
            probability_lineage_check=lineage_check,
            historical_context_contains_scoring_truth="truth" in context,
            no_training_or_Chase_in_check=True,
            uniform_is_unvalidated_weight_control=data["weighting"] == "uniform",
            requested_weighting=requested_weighting,
            effective_weighting=data["weighting"],
            aux_initialization=data["dynamic"].get("initialization", "previous"),
            aux_extension=aux_partition,
            effective_feedback=data["feedback"],
            weighting_alias=(
                requested_weighting if requested_weighting != data["weighting"] else None
            ),
            conflict_policy=data["loop"]["conflict_policy"],
            unknown_semantics=data["loop"]["unknown_semantics"],
            check_failure_policy=data["feedback"]["check_failure_policy"],
            receiver_policy=data["feedback"]["receiver_policy"],
            receiver_occurrences=len(
                rule_channels(rules, {data["dynamic"]["alias"]} | set(adaptive_aliases(data)),
                              tuple(context["labels"]))
            ),
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("residual-aux", help="Gate independent saved Aux checkpoints using complete calibration closures")
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("prepare-aux-extension", help="Append an isolated adaptive Aux rule per label to a frozen learned program")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alias", default="loris_aux_online_v2")
    p.add_argument("--default-threshold", type=float, default=0.5)
    p.add_argument("--thresholds", help="JSON map of label to threshold, selected on dev/calibration only")
    p.add_argument("--protocol", help="Independent calibration protocol to bind to the new alias")
    p.add_argument("--label-budget", type=int, default=128)
    p.add_argument("--query-batch-size", type=int, default=16)
    p = sub.add_parser(
        "calibrate-reliability",
        help="Calibrate each rule using independent held-out labeled documents",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--protocol", required=True)
    p = sub.add_parser(
        "prepare-oracle", help="Prepare isolated NPZ oracle metadata without decoding truth"
    )
    p.add_argument("--context", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--truth-key", required=True)
    p.add_argument("--label-axis", required=True)
    p.add_argument("--label-axis-key", required=True)
    p.add_argument("--roles", nargs="+", required=True)
    p.add_argument("--source-rows", type=int, required=True)
    p.add_argument("--source-offset", type=int, required=True)
    p.add_argument("--output", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--strategy", choices=("frozen", "supervised", "self_train", "paper"))
        p.add_argument(
            "--weighting",
            choices=WEIGHTINGS,
            help="Main: direct_rule_precision_discount (direct source q times model confidence factor); pure direct_rule_precision and complete_provenance[_discount] are ablations. reliability_only/provenance are legacy aliases.",
        )
        p.add_argument(
            "--source-reliability",
            help="Independent per-rule calibration JSON, bound by absolute path and SHA",
        )
        p.add_argument("--candidate-check", choices=("checked", "nocheck"))
        p.add_argument(
            "--aux-initialization", choices=("previous", "m0"),
            help="previous continues committed weights; m0 reloads the original checkpoint each round and reselects current logical corrections against M0, with all accumulated answers and one update per closure.",
        )
        p.add_argument(
            "--receiver-policy",
            choices=RECEIVER_POLICIES,
            help="Main same_model: only the receiver's actual predicates select its logical corrections. affected_channels_to_aux explicitly transfers dependent ensemble corrections to Aux.",
        )
        p.add_argument(
            "--conflict-policy", choices=("halt", "review_instances", "unknown_arbitrate")
        )
        p.add_argument(
            "--unknown-semantics",
            choices=("retain_consequences", "full_rebuild"),
            help="retain_consequences isolates a conflict cell but retains its existing descendants; full_rebuild is the earlier unknown protocol. New templates select retain_consequences.",
        )
        p.add_argument(
            "--arbitration-threshold",
            type=float,
            help="Explicit common threshold for the additional conflict-only dynamic arbitration rules; per-label overrides can be declared in the manifest.",
        )
        p.add_argument(
            "--check-failure-policy",
            choices=("halt", "reject_batch"),
            help="Author-confirmed default halt: stop with diagnostics when the complete check fails; reject_batch is an explicit engineering extension, not error filtering.",
        )
        p.add_argument("--query-strategy", choices=("rdg", "random", "uncertainty", "review"))
        p.add_argument("--rdg-candidate-limit", type=int, help="Explicit uniform candidate-domain approximation for RDG; default uncapped")
        p.add_argument("--self-train-confidence", type=float)
        p.add_argument("--label-budget", type=parse_budget, help="Maximum requested documents; unlimited disables the cap. LLM-to-human fallback uses one document budget.")
        p.add_argument("--source-review-budget", type=int)
        p.add_argument("--query-batch-size", type=int, help="Fixed document batch size; overrides fractional size")
        p.add_argument("--query-batch-fraction", type=float, help="Fraction of the initial eligible query pool, not the shrinking remainder")
        p.add_argument("--query-batch-min", type=int)
        p.add_argument("--query-cadence", choices=CADENCES)
        p.add_argument("--query-pool", choices=QUERY_POOLS,
                       help="all_unanswered also queries model-covered documents lacking complete trusted answers")
        p.add_argument("--query-answer-mode", choices=("complete", "allow_partial"))
        p.add_argument("--new-verified-supervision", choices=("full_batch", "corrections_only"),
                       help="Full new annotations at unit weight, or only receiver disagreements; existing logical corrections remain weighted")
        p.add_argument("--intermediate-exports", choices=("all", "periodic", "final"))
        p.add_argument("--history-export-interval", type=int)
        if name == "run":
            p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "residual-aux":
        from .residual_aux import main_protocol

        print(json.dumps(main_protocol(args.protocol, args.output), indent=2))
        return 0
    if args.command == "prepare-aux-extension":
        from .aux_extension import prepare

        thresholds = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
        result = prepare(args.manifest, args.output, alias=args.alias,
            default_threshold=args.default_threshold, thresholds=thresholds,
            protocol_path=args.protocol, label_budget=args.label_budget,
            query_batch_size=args.query_batch_size)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "calibrate-reliability":
        from .reliability import main_calibrate

        return main_calibrate(args)
    if args.command == "prepare-oracle":
        from .oracle import prepare

        result = prepare(
            context_path=args.context,
            truth_path=args.truth,
            truth_key=args.truth_key,
            axis_path=args.label_axis,
            axis_key=args.label_axis_key,
            roles=args.roles,
            source_rows=args.source_rows,
            source_offset=args.source_offset,
            output=args.output,
        )
        print(json.dumps(result, indent=2))
        return 0
    data, context, rules, reliability, receipt = load(
        args.manifest,
        strategy=args.strategy,
        weighting=args.weighting,
        candidate_check=args.candidate_check,
        aux_initialization=args.aux_initialization,
        conflict_policy=args.conflict_policy,
        unknown_semantics=args.unknown_semantics,
        check_failure_policy=args.check_failure_policy,
        receiver_policy=args.receiver_policy,
        new_verified_supervision=args.new_verified_supervision,
        arbitration_threshold=args.arbitration_threshold,
        query_strategy=args.query_strategy,
        rdg_candidate_limit=args.rdg_candidate_limit,
        self_train_confidence=args.self_train_confidence,
        label_budget=args.label_budget,
        source_review_budget=args.source_review_budget,
        query_batch_size=args.query_batch_size,
        query_batch_fraction=args.query_batch_fraction,
        query_batch_min=args.query_batch_min,
        query_cadence=args.query_cadence,
        query_pool=args.query_pool,
        query_answer_mode=args.query_answer_mode,
        intermediate_exports=args.intermediate_exports,
        history_export_interval=args.history_export_interval,
        source_reliability=args.source_reliability,
    )
    if args.command == "check":
        print(json.dumps(receipt, indent=2))
        return 0
    from .runner import LabelingRun, write
    from .trainer import train_actual

    run = LabelingRun(
        context=context,
        rules=rules,
        manifest=data,
        output=args.output,
        trainer=train_actual,
        reliability=reliability,
    )
    write(Path(args.output) / "INPUT_RECEIPT.json", receipt)
    result = run.run()
    print(json.dumps(result, indent=2))
    return 0
