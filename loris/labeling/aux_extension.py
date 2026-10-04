"""An isolated adaptive Aux channel appended to an immutable learned program.

The learned rules retain their original aliases, thresholds and initial tables.
Only one new alias can receive trained probabilities. The ordinary typed Chase
and its vectorized unary evaluator execute the new rules.
"""

import copy
import json
from pathlib import Path

import numpy as np

from loris._engine.rdl_reference import Doc, Label, ML, Rule
from loris._engine.typed_session import digest


def configuration(manifest, labels=None):
    value = manifest.get("aux_extension")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "source_alias", "alias", "default_threshold", "thresholds"
    }:
        raise ValueError("aux_extension requires source_alias/alias/default_threshold/thresholds")
    source, alias = value["source_alias"], value["alias"]
    if (not isinstance(source, str) or not source or not isinstance(alias, str)
            or not alias or source == alias or alias != manifest["dynamic"]["alias"]):
        raise ValueError("adaptive Aux must have a new alias distinct from the frozen source")
    if manifest["dynamic"]["ensemble_dependencies"]:
        raise ValueError("isolated Aux cannot refresh any original ensemble")
    for name in ("arbitration", "fallback"):
        if name in manifest and manifest[name]["alias"] != source:
            raise ValueError("isolated Aux must keep the original " + name + " model frozen")
    if (manifest.get("fallback", {}).get("enabled")
            and manifest["fallback"]["calibration"] != "fixed"):
        raise ValueError("isolated Aux must freeze the original fallback thresholds")
    overrides = value["thresholds"]
    if not isinstance(overrides, dict) or any(not isinstance(k, str) or not k for k in overrides):
        raise ValueError("Aux thresholds must be keyed by label name")
    for threshold in [value["default_threshold"], *overrides.values()]:
        if (type(threshold) not in (int, float) or not np.isfinite(threshold)
                or not 0 <= threshold <= 1):
            raise ValueError("Aux thresholds must be finite numbers in [0,1]")
    if labels is not None and set(overrides) - set(labels):
        raise ValueError("Aux threshold label is outside the complete label axis")
    return value


def attach_context(context, manifest, *, allow_existing=False):
    """Share the initial array; never duplicate the large application table."""
    config = configuration(manifest, context["labels"])
    if config is None:
        return context
    source, alias = config["source_alias"], config["alias"]
    tables = context["model_probabilities"]
    if source not in tables:
        raise ValueError("frozen source Aux predictions are absent")
    if alias in tables and not allow_existing:
        raise ValueError("new adaptive Aux alias already exists in the source context")
    result = dict(context, model_probabilities=dict(tables))
    result["model_probabilities"].setdefault(alias, tables[source])
    if "static_model_probabilities" in context:
        static = dict(context["static_model_probabilities"])
        if source not in static:
            raise ValueError("frozen source Aux static table is absent")
        static.setdefault(alias, static[source])
        result["static_model_probabilities"] = static
    return result


def appended_rules(manifest, labels):
    config = configuration(manifest, labels)
    if config is None:
        return ()
    return tuple(
        Rule((Doc("x"), ML("x", config["alias"], label,
                           config["thresholds"].get(label, config["default_threshold"]))),
             Label("x", label))
        for label in labels
    )


def extend_program(rules, manifest, labels):
    config = configuration(manifest, labels)
    if config is None:
        return tuple(rules)
    if any(isinstance(atom, ML) and atom.alias == config["alias"]
           for rule in rules for atom in rule.body):
        raise ValueError("new adaptive Aux alias collides with a learned rule")
    return tuple(rules) + appended_rules(manifest, labels)


def partition_audit(rules, manifest, labels):
    """Prove that all adaptable occurrences are in the appended suffix."""
    config = configuration(manifest, labels)
    if config is None:
        return None
    extra = appended_rules(manifest, labels)
    if not extra or tuple(rules[-len(extra):]) != extra:
        raise ValueError("isolated Aux program is missing its declared per-label suffix")
    core = tuple(rules[:-len(extra)])
    if any(isinstance(atom, ML) and atom.alias == config["alias"]
           for rule in core for atom in rule.body):
        raise ValueError("adaptive Aux occurrence leaked into the frozen learned program")
    return dict(
        frozen_rules=len(core), added_rules=len(extra),
        frozen_program_id=digest([rule.to_dict() for rule in core]),
        source_alias=config["source_alias"], adaptive_alias=config["alias"],
        old_rule_bytes_rewritten=False, old_model_tables_refreshed=False,
        thresholds_by_label={label: float(config["thresholds"].get(
            label, config["default_threshold"])) for label in labels},
    )


def prepare(manifest_path, output, *, alias="loris_aux_online_v2", default_threshold=0.5,
            thresholds=None, protocol_path=None, label_budget=128, query_batch_size=16):
    """Write a new manifest/protocol; never rewrite source artifacts or run jobs."""
    from .cli import load, validate
    from .reliability import validate_protocol
    from loris.discovery.manifest import bound, file_record

    # This is preparation for a NEW independent calibration. Old q/threshold
    # artifacts must not be reused after the program and alias have changed.
    data, context, rules, _, receipt = load(
        manifest_path, weighting="uniform", ignore_reliability=True)
    if data.get("aux_extension") is not None:
        raise ValueError("manifest already has an adaptive Aux extension")
    source = data["dynamic"]["alias"]
    if thresholds is not None and not isinstance(thresholds, dict):
        raise ValueError("Aux thresholds must be a JSON object keyed by label")
    data["context"] = receipt["refs"]["context"]
    data["program"] = receipt["refs"]["program"]
    # Freeze the original output rules as well. Their actual calibrated
    # thresholds come from the authenticated V1 artifact, not the placeholder
    # default in its input manifest. The V2 gate is newly calibrated below.
    fallback = data.get("fallback")
    if fallback and fallback["enabled"] and fallback["calibration"] == "independent_calibration":
        original_ref = json.loads(Path(manifest_path).read_text())["source_reliability"]
        if original_ref is None:
            raise ValueError("calibrate the source fallback before freezing its thresholds for V2")
        rp = bound(original_ref, Path(manifest_path).resolve().parent)
        artifact = json.loads(rp.read_text())
        from .reliability import validate_artifact, load_fallback_record

        validate_artifact(artifact, context, rules, data)
        calibrated = load_fallback_record(file_record(rp), artifact, context, rules, data,
                                          context["labels"])
        fallback.update(calibration="fixed", thresholds=dict(calibrated["thresholds"]))
    data["dynamic"]["alias"] = alias
    data["dynamic"]["ensemble_dependencies"] = []
    lineage = data["dynamic"]["probability_lineage"]
    if isinstance(lineage, dict):
        lineage = copy.deepcopy(lineage)
        if "alias" in lineage:
            lineage["alias"] = alias
        lineage["frozen_source_alias"] = source
        data["dynamic"]["probability_lineage"] = lineage
    data["aux_extension"] = dict(source_alias=source, alias=alias,
        default_threshold=default_threshold, thresholds=dict(thresholds or {}))
    data["source_reliability"] = None
    # Covered multi-label documents can still be incomplete or incorrect.
    # A finite query budget controls this broader pool without using test gold.
    data["loop"].update(query_pool="all_unanswered", label_budget=label_budget,
                        query_batch_size=query_batch_size, query_cadence="each_closure",
                        max_inner_rounds=1)
    data["feedback"].update(new_verified_supervision="full_batch")
    data["notes"] = str(data["notes"]) + (
        " | V2 isolated Aux: learned rules/tables frozen; per-label adaptive suffix; "
        "finite all-unanswered query pool. Requires new independent calibration. "
        "No assertion of monotonic test improvement.")
    if type(label_budget) is not int or label_budget < 1:
        raise ValueError("V2 preparation requires a positive finite document budget")
    validate(data)
    extended_context = attach_context(context, data)
    extended = extend_program(rules, data, context["labels"])
    audit = partition_audit(extended, data, context["labels"])
    protocol = None
    if protocol_path is not None:
        pp = Path(protocol_path).resolve(strict=True)
        protocol = json.loads(pp.read_text())
        for key in ("context", "answers"):
            protocol[key] = file_record(bound(protocol[key], pp.parent))
        protocol["model_binding"] = dict(protocol["model_binding"],
            checkpoint_sha256=data["dynamic"]["checkpoint"]["sha256"],
            probability_lineage=data["dynamic"]["probability_lineage"])
        validate_protocol(protocol, extended_context, extended, data)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        p = output / name
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return file_record(p)
    result = dict(status="prepared_requires_new_calibration", version="V2",
        source_manifest=receipt["manifest"], frozen_program=data["program"],
        partition=audit, labeling_manifest=save("labeling.json", data),
        reliability_protocol=(save("reliability_protocol.json", protocol) if protocol else None),
        preferred_weighting=json.loads(Path(manifest_path).read_text())["weighting"],
        preparation_weighting="uniform", training_started=False, test_answers_used=False)
    save("PREPARED.json", result)
    return result
