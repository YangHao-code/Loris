"""An isolated adaptive Aux channel appended to an immutable learned program.

The learned rules retain their original aliases, thresholds and initial tables.
Only one new alias can receive trained probabilities. The ordinary typed Chase
and its vectorized unary evaluator execute the new rules.
"""

import copy
import json
from pathlib import Path

import numpy as np

from loris._engine.rdl_reference import Doc, Label, ML, Rule, Text
from loris._engine.typed_session import digest


def configuration(manifest, labels=None):
    value = manifest.get("aux_extension")
    if value is None:
        return None
    required = {"source_alias", "alias", "default_threshold", "thresholds"}
    if (not isinstance(value, dict) or not required <= set(value)
            or set(value) - required - {"initial_alias", "correction"}):
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
    initial_alias = value.get("initial_alias", source)
    if not isinstance(initial_alias, str) or not initial_alias or initial_alias == alias:
        raise ValueError("initial Aux alias must name an existing frozen model table")
    if "correction" in value:
        from .aux_correction import validate_specification
        validate_specification(value["correction"], value, labels)
    return value


def attach_context(context, manifest, *, allow_existing=False):
    """Share the initial array; never duplicate the large application table."""
    config = configuration(manifest, context["labels"])
    if config is None:
        return context
    source, alias = config["source_alias"], config["alias"]
    initial = config.get("initial_alias", source)
    tables = context["model_probabilities"]
    if source not in tables:
        raise ValueError("frozen source Aux predictions are absent")
    if initial not in tables:
        raise ValueError("initial Aux predictions are absent")
    if alias in tables and not allow_existing:
        raise ValueError("new adaptive Aux alias already exists in the source context")
    result = dict(context, model_probabilities=dict(tables))
    result["model_probabilities"].setdefault(alias, tables[initial])
    for name, table in probability_views(manifest, result["model_probabilities"][alias]).items():
        if name != alias and name in tables and not allow_existing:
            raise ValueError("new Aux complement alias already exists in the source context")
        result["model_probabilities"][name] = table
    if "static_model_probabilities" in context:
        static = dict(context["static_model_probabilities"])
        if initial not in static:
            raise ValueError("initial Aux static table is absent")
        static.setdefault(alias, static[initial])
        for name, table in probability_views(manifest, static[alias]).items():
            if name != alias and name in context["static_model_probabilities"] and not allow_existing:
                raise ValueError("new Aux complement static alias already exists in the source context")
            static[name] = table
        result["static_model_probabilities"] = static
    return result


def probability_views(manifest, probabilities):
    """The low-probability predicate is an exact float64 view of the same model.

    Recompute this whenever the adaptive table changes. It is not a separately
    trained model, and the original source/arbitration tables are never touched.
    """
    config = configuration(manifest)
    if config is None:
        return {}
    values = np.asarray(probabilities)
    if (values.ndim != 2 or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()):
        raise ValueError("adaptive Aux probabilities must be a finite matrix in [0,1]")
    result = {config["alias"]: probabilities}
    correction = config.get("correction")
    if correction is not None:
        negative = np.subtract(np.float64(1.), values, dtype=np.float64)
        negative.setflags(write=False)
        result[correction["complement_alias"]] = negative
    return result


def adaptive_aliases(manifest):
    config = configuration(manifest)
    if config is None:
        return ()
    return (config["alias"],) + ((config["correction"]["complement_alias"],)
                                if "correction" in config else ())


def channel_polarities(rules, manifest):
    """Only suffix channels participate in online threshold/model adaptation."""
    config = configuration(manifest)
    if config is None:
        return {}
    negative_alias = config.get("correction", {}).get("complement_alias")
    aliases = set(adaptive_aliases(manifest))
    return {(i, a): (-1 if atom.alias == negative_alias else 1)
            for i, rule in enumerate(rules) for a, atom in enumerate(rule.body)
            if isinstance(atom, ML) and atom.dynamic and atom.alias in aliases}


def clamp_channels(manifest, channels, rules):
    """Enforce one conservative negative threshold per label after calibration.

    Complement thresholds are ``eta=1-low``. Raising eta narrows the negative
    region. We never broaden that region to resolve conflicting calibration.
    Positive cutoffs have a minimum of min_gap so a disjoint negative region
    always exists. Old-model channel entries are copied byte-for-byte.
    """
    config = configuration(manifest)
    if config is None or "correction" not in config:
        return dict(channels)
    correction = config["correction"]
    polarities = channel_polarities(rules, manifest)
    result = dict(channels)
    positive_by_label, negative_by_label = {}, {}
    for (ri, ai), polarity in polarities.items():
        key = (ri, ai)
        if key not in result:
            raise ValueError("adaptive Aux calibration is missing a suffix channel")
        pair = result[key]
        if len(pair) != 2 or type(pair[0]) not in (int, np.int32, np.int64):
            raise ValueError("adaptive threshold channels require label index and eta")
        eta = float(pair[1])
        if not np.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("adaptive threshold channel eta must be in [0,1]")
        label = rules[ri].body[ai].label
        (positive_by_label if polarity > 0 else negative_by_label).setdefault(label, []).append(key)
    for label, keys in positive_by_label.items():
        # Usually one positive rule; if a later format includes duplicates,
        # calibrate them to the same most conservative threshold.
        hi = max(correction["min_gap"], max(float(result[k][1]) for k in keys))
        for key in keys:
            result[key] = (result[key][0], hi)
    for label, keys in negative_by_label.items():
        pos = positive_by_label.get(label)
        hi = (float(result[pos[0]][1]) if pos else
              float(config["thresholds"].get(label, config["default_threshold"])))
        eta = max(max(float(result[k][1]) for k in keys),
                  float(np.float64(1.) - np.float64(hi) + np.float64(correction["min_gap"])))
        eta = min(1., eta)  # only exact-roundoff excess can remain after hi >= gap.
        for key in keys:
            result[key] = (result[key][0], eta)
    return result


def static_correction_body(rule):
    """Only label-independent unary contexts may create a corrective negative.

    A label premise would feed its own consequences back into the correction.
    Relationships require corpus-level support accounting and are deliberately
    not guessed by this sparse-context extension.
    """
    return (isinstance(rule.head, Label) and rule.head.polarity == "pos"
            and len(rule.variables) == 1
            and all(isinstance(a, (Doc, Text, ML)) for a in rule.body))


def dormant_positive_channels(labels, initial_contexts, fit_probabilities, fit_positive,
                              fit_negative, *, grid=(.8, .9, .95, .99),
                              precision_floor=.95, minimum_support=10):
    """Opt-in dormant positive suffix, with no initial new logical facts.

    Each context is (initial Aux probability, successful logical-positive
    matrix). Fit answers are separate A-only supervision, never application
    truth. Every admitted firing must already be logical-positive in ALL
    supplied contexts (application and calibration). A full-Chase initial
    equality check and independent B joint gate remain caller obligations.
    """
    labels = tuple(labels)
    p, y, n = np.asarray(fit_probabilities, np.float64), np.asarray(fit_positive), np.asarray(fit_negative)
    if (not labels or len(set(labels)) != len(labels) or p.ndim != 2 or p.shape[1] != len(labels)
            or y.shape != p.shape or n.shape != p.shape or y.dtype != np.bool_ or n.dtype != np.bool_
            or np.any(y & n) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or type(minimum_support) is not int or minimum_support < 1
            or type(precision_floor) not in (int, float) or not np.isfinite(precision_floor)
            or not 0 <= precision_floor <= 1):
        raise ValueError("dormant positive fitting requires explicit signed A answers and valid axes")
    grid = tuple(sorted(set(grid)))
    if not grid or any(type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1 for value in grid):
        raise ValueError("dormant positive grid must contain finite thresholds in [0,1]")
    contexts = []
    for probabilities, logical_positive in initial_contexts:
        cp, lp = np.asarray(probabilities), np.asarray(logical_positive)
        if (cp.ndim != 2 or cp.shape[1] != len(labels) or lp.shape != cp.shape or lp.dtype != np.bool_
                or not np.isfinite(cp).all() or np.any((cp < 0) | (cp > 1))):
            raise ValueError("dormant positive contexts must contain matching initial logical facts")
        contexts.append((cp, lp))
    if len(contexts) < 2:
        raise ValueError("dormant positive channels require application and calibration context checks")
    active, thresholds, diagnostics = [], {}, []
    observed = y | n
    for col, label in enumerate(labels):
        chosen = None
        for threshold in grid:
            # Float64 comparison matches the typed Chase boundary.
            if any(np.any((cp[:, col].astype(np.float64, copy=False) >= threshold) & ~lp[:, col])
                   for cp, lp in contexts):
                continue
            fires = (p[:, col] >= threshold) & observed[:, col]
            support = int(fires.sum())
            precision = float(y[fires, col].mean()) if support else None
            if support >= minimum_support and precision >= precision_floor:
                chosen = dict(label=label, threshold=float(threshold), A_support=support,
                              A_precision=precision, initial_new_logical_cells=0)
                break  # Lowest safe predeclared cutoff leaves future improvement room.
        diagnostics.append(chosen or dict(label=label, status="no_equivalent_supported_positive_channel"))
        if chosen is not None:
            active.append(label)
            thresholds[label] = chosen["threshold"]
    return dict(positive_active=active, thresholds=thresholds), dict(
        fitting_rows=len(p), contexts_checked=len(contexts), labels=diagnostics,
        threshold_selection="lowest_predeclared_cutoff_with_precision_support_and_initial_logic_implication",
        full_chase_initial_equality_and_B_joint_gate_required=True,
        application_truth_used=False, old_program_rewritten=False,
        precision_floor=float(precision_floor), minimum_support=minimum_support,
    )


def appended_rules(manifest, labels, original_rules=()):
    config = configuration(manifest, labels)
    if config is None:
        return ()
    correction = config.get("correction")
    active = set(correction["positive_active"]) if correction is not None else set(labels)
    positive = tuple(
        Rule((Doc("x"), ML("x", config["alias"], label,
                           config["thresholds"].get(label, config["default_threshold"]))),
             Label("x", label))
        for label in labels if label in active
    )
    if correction is None:
        return positive
    original_rules = tuple(original_rules)
    negative = []
    for index in correction["negative_rule_indices"]:
        if index >= len(original_rules):
            raise ValueError("negative correction index is outside the frozen program")
        old = original_rules[index]
        if not static_correction_body(old):
            raise ValueError("negative correction requires a static unary Doc/Text/ML context")
        low = correction["negative_thresholds"].get(old.head.label,
                                                     correction["negative_default_threshold"])
        negative.append(Rule(old.body + (ML(old.head.var, correction["complement_alias"],
                                             old.head.label, float(np.float64(1.) - np.float64(low)),
                                             dynamic=True),),
                             Label(old.head.var, old.head.label, "neg")))
    return positive + tuple(negative)


def extend_program(rules, manifest, labels):
    config = configuration(manifest, labels)
    if config is None:
        return tuple(rules)
    aliases = set(adaptive_aliases(manifest))
    if any(isinstance(atom, ML) and atom.alias in aliases
           for rule in rules for atom in rule.body):
        raise ValueError("new adaptive Aux alias collides with a learned rule")
    return tuple(rules) + appended_rules(manifest, labels, rules)


def partition_audit(rules, manifest, labels):
    """Prove that all adaptable occurrences are in the appended suffix."""
    config = configuration(manifest, labels)
    if config is None:
        return None
    correction = config.get("correction")
    suffix_length = (len(correction["positive_active"]) + len(correction["negative_rule_indices"])
                     if correction is not None else len(labels))
    core = tuple(rules[:-suffix_length]) if suffix_length else tuple(rules)
    extra = appended_rules(manifest, labels, core)
    if len(extra) != suffix_length or (suffix_length and tuple(rules[-suffix_length:]) != extra):
        raise ValueError("isolated Aux program is missing its declared per-label suffix")
    aliases = set(adaptive_aliases(manifest))
    if any(isinstance(atom, ML) and atom.alias in aliases
           for rule in core for atom in rule.body):
        raise ValueError("adaptive Aux occurrence leaked into the frozen learned program")
    result = dict(
        frozen_rules=len(core), added_rules=len(extra),
        frozen_program_id=digest([rule.to_dict() for rule in core]),
        source_alias=config["source_alias"], adaptive_alias=config["alias"],
        old_rule_bytes_rewritten=False, old_model_tables_refreshed=False,
        thresholds_by_label={label: float(config["thresholds"].get(
            label, config["default_threshold"])) for label in labels},
    )
    if correction is not None:
        result.update(positive_rules=len(correction["positive_active"]),
                      negative_rules=len(correction["negative_rule_indices"]),
                      complement_alias=correction["complement_alias"],
                      initial_alias=config.get("initial_alias", config["source_alias"]),
                      negative_rule_indices=list(correction["negative_rule_indices"]),
                      negative_thresholds_by_label={label: float(correction["negative_thresholds"].get(
                          label, correction["negative_default_threshold"])) for label in labels},
                      correction_binding=digest(correction),
                      derived_aliases={correction["complement_alias"]: dict(
                          source=config["alias"], transformation="float64_one_minus_probability")},
                      channel_polarities={"%d:%d" % key: polarity
                                          for key, polarity in channel_polarities(rules, manifest).items()})
    return result


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
