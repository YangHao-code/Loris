"""Explicit, source-bound inputs for discovery; no test or online training.

Contexts are trusted local pickle artifacts. A checksum verifies identity, not
the safety of arbitrary downloaded Python objects. Only inspect/load contexts
created by a trusted data preparation pipeline.
"""

import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from . import SCHEMA


REQUIRED_RUN_FIELDS = {
    "seed",
    "budget_per_cluster",
    "workers",
    "initializer_slots",
    "label_rotation",
}
RUN_DEFAULTS = {
    "candidate_domain": "rich",
    "admission_order": "standalone_score_desc",
    "proposal_strategy": "accuracy_guided",
    "max_variables": 2,
    "context_probes": 32,
    "acquisition_candidates": 64,
    "expected_body_size": 3.0,
    "initialization_workers": 1,
    "initializer_candidate_archive": "winners_and_counts",
    "admission_policy": "strict",
    "admission_workers": 8,
    "admission_batch_size": 8,
    "gamma_conflict_policy": "soft_fit_precision",
    "fit_precision_floor": 0.9,
}
RUN_FIELDS = REQUIRED_RUN_FIELDS | set(RUN_DEFAULTS)


def file_record(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bound(record, base):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("file reference requires exactly path and sha256")
    if not isinstance(record["path"], str) or not record["path"]:
        raise ValueError("nonempty file path required")
    path = (base / record["path"]).resolve(strict=True)
    if not path.is_file() or file_record(path)["sha256"] != record["sha256"]:
        raise ValueError("file identity differs: " + str(path))
    return path


def load_manifest(path, overrides=None):
    path = Path(path).resolve(strict=True)
    data = read_json(path)
    required = {
        "schema",
        "dataset",
        "context",
        "model_selection",
        "pattern_supply",
        "relation_attributes",
        "run",
        "oracle_policy",
        "supervision_exposure",
    }
    allowed = required | {"lexical_cache_dir", "notes", "selection_strategy", "relation_screening"}
    if not isinstance(data, dict) or set(data) - allowed or required - set(data):
        raise ValueError("unknown or missing manifest fields")
    if data["schema"] != SCHEMA:
        raise ValueError("manifest is not a current rich discovery protocol")
    dataset = data["dataset"]
    if (
        not isinstance(dataset, dict)
        or set(dataset) != {"name", "split_protocol", "scope", "roots_contract"}
        or not isinstance(dataset["name"], str)
        or not dataset["name"]
    ):
        raise ValueError("dataset requires name, bound split protocol, scope and roots contract")
    if (
        dataset["scope"] != "train_dev_only"
        or dataset["roots_contract"] != "complete_signed_known_train"
    ):
        raise ValueError("discovery permits train/dev and complete known-train roots only")
    if data["oracle_policy"] not in {"accuracy_then_vote_bce", "bce_first"}:
        raise ValueError("explicit supported oracle policy required")
    run = data["run"]
    if not isinstance(run, dict) or set(run) - RUN_FIELDS or REQUIRED_RUN_FIELDS - set(run):
        raise ValueError("run must explicitly declare every supported execution parameter")
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    if set(overrides) - RUN_FIELDS - {"selection_strategy"}:
        raise ValueError("unsupported execution override")
    data = copy.deepcopy(data)
    data["selection_strategy"] = overrides.get(
        "selection_strategy", data.get("selection_strategy", "artifact")
    )
    if data["selection_strategy"] not in {"artifact", "learned", "quality", "indiv", "random", "all"}:
        raise ValueError("unsupported model-selection strategy")
    data["run"] = dict(RUN_DEFAULTS, **data["run"])
    data["run"].update({k: v for k, v in overrides.items() if k in RUN_FIELDS})
    run = data["run"]
    for key in (
        "seed",
        "budget_per_cluster",
        "workers",
        "initialization_workers",
        "initializer_slots",
        "max_variables",
        "context_probes",
        "acquisition_candidates",
        "admission_workers",
        "admission_batch_size",
    ):
        if type(run[key]) is not int:
            raise ValueError(key + " must be an integer, not a bool or coerced string")
    if not 0 <= run["seed"] < 2**32 or run["budget_per_cluster"] < 1 or run["workers"] < 1 or run["initialization_workers"] < 1:
        raise ValueError("invalid seed, BO budget or worker count")
    from .search.domain import variable_names
    if run["admission_policy"] not in ("strict", "batch_prefilter") or min(run["admission_workers"], run["admission_batch_size"]) < 1:
        raise ValueError("invalid admission policy/workers/batch size")
    if run["gamma_conflict_policy"] not in ("strict", "soft_fit_precision"):
        raise ValueError("gamma_conflict_policy must be strict or soft_fit_precision")
    if type(run["fit_precision_floor"]) not in (int, float) or not 0 < run["fit_precision_floor"] <= 1:
        raise ValueError("fit_precision_floor must lie in (0, 1]")

    variable_names(run["max_variables"])
    if run["initializer_candidate_archive"] not in {"all", "evaluated_with_analytic_skips", "winners_and_counts"}:
        raise ValueError("unsupported initializer candidate archive")
    if run["context_probes"] < 1 or run["acquisition_candidates"] < 1:
        raise ValueError("positive context probe and acquisition budgets required")
    if (
        type(run["expected_body_size"]) not in (int, float)
        or not np.isfinite(run["expected_body_size"])
        or run["expected_body_size"] <= 0
    ):
        raise ValueError("positive finite expected body size required")
    if not 0 <= run["initializer_slots"] <= min(30, run["budget_per_cluster"]):
        raise ValueError("initializer slots must fit the original 30 startup slots and BO budget")
    if run["label_rotation"] not in {"axis", "seeded_cluster_rotation"}:
        raise ValueError("unsupported initialization label order")
    if run["candidate_domain"] not in {"rich", "no_text", "local"}:
        raise ValueError("unsupported candidate domain")
    if run["admission_order"] not in {
        "proposal_order",
        "standalone_score_desc",
        "random_consistent",
    }:
        raise ValueError("unsupported admission order")
    if run["proposal_strategy"] not in {"accuracy_guided", "random_joint"}:
        raise ValueError("unsupported proposal strategy")
    if run["proposal_strategy"] == "random_joint":
        if run["initializer_slots"] != 0 or run["admission_order"] != "random_consistent":
            raise ValueError(
                "random_joint requires initializer_slots=0 and admission_order=random_consistent"
            )
    elif run["admission_order"] == "random_consistent":
        raise ValueError("random_consistent requires random_joint proposals")
    attrs = data["relation_attributes"]
    if (
        not isinstance(attrs, list)
        or any(not isinstance(a, str) or not a for a in attrs)
        or len(attrs) != len(set(attrs))
    ):
        raise ValueError("relation_attributes must be an explicit unique ordered string list")
    base = path.parent
    refs = {
        name: bound(data[name], base)
        for name in ("context", "model_selection", "pattern_supply", "supervision_exposure")
    }
    refs["split_protocol"] = bound(dataset["split_protocol"], base)
    cache = data.get("lexical_cache_dir")
    if cache is not None and (not isinstance(cache, str) or not cache):
        raise ValueError("lexical_cache_dir must be a path or null")
    from loris._engine.rlearner import digest

    binding = {
        "manifest": file_record(path),
        "effective_manifest": data,
        "effective_manifest_id": digest(data),
        "overrides": overrides,
        "files": {k: file_record(v) for k, v in refs.items()},
    }
    return data, refs, binding, ((base / cache).resolve() if cache else None)


def validate_context(context):
    """Validate complete axes and the discovery supervision boundary."""
    required = {
        "docs",
        "labels",
        "row_ids",
        "row_roles",
        "gamma",
        "fit_indices",
        "receiver_indices",
        "truth",
        "source_gold",
        "model_probabilities",
        "model_decisions",
        "context_id",
        "attribute_values",
        "attribute_presence",
    }
    if not isinstance(context, dict) or not required <= set(context):
        raise ValueError("complete discovery context required")
    n, k = len(context["docs"]), len(context["labels"])
    if not n or not k or len(set(context["labels"])) != k:
        raise ValueError("nonempty unique label axis required")
    rows = context["row_ids"]
    roles = np.asarray(context["row_roles"])
    if len(rows) != n or len(set(rows)) != n or roles.shape != (n,):
        raise ValueError("complete unique row/role axes required")
    if not set(roles.tolist()) <= {"source_fit", "unlabeled_train", "val_ctrl", "excluded_dev"}:
        raise ValueError("test, online and undeclared row roles cannot enter rule discovery")
    excluded = np.flatnonzero(roles == 'excluded_dev')
    if len(excluded) and not np.array_equal(excluded, context.get('excluded_dev_indices')):
        raise ValueError('Excluded dev roles require an exact declared exclusion axis')
    if len(context.get("test_indices", ())):
        raise ValueError("test rows cannot enter discovery")
    indices = {}
    for name, role in (("fit_indices", "source_fit"), ("receiver_indices", "val_ctrl")):
        values = np.asarray(context[name])
        if (
            values.ndim != 1
            or not len(values)
            or values.dtype.kind not in "iu"
            or len(set(values.tolist())) != len(values)
            or (values < 0).any()
            or (values >= n).any()
            or not np.all(roles[values] == role)
            or set(values.tolist()) != set(np.flatnonzero(roles == role).tolist())
        ):
            raise ValueError("indices must exactly match declared " + role + " rows")
        indices[name] = values
    fit, control = indices["fit_indices"], indices["receiver_indices"]
    gp, gn = np.asarray(context["gamma"].pos), np.asarray(context["gamma"].neg)
    if (
        gp.shape != (n, k)
        or gn.shape != gp.shape
        or gp.dtype != bool
        or gn.dtype != bool
        or (gp & gn).any()
    ):
        raise ValueError("complete binary, consistent signed root axes required")
    known = np.zeros(n, bool)
    known[fit] = True
    if not (gp | gn)[fit].all() or (gp | gn)[~known].any():
        raise ValueError("only complete known train answers may initialize Gamma")
    gold = context["source_gold"]
    if not isinstance(gold, dict) or set(gold) != {"fit", "val_ctrl"}:
        raise ValueError("no hidden train/test gold may enter a discovery context")
    for value, shape, name in (
        (gold["fit"], (len(fit), k), "fit"),
        (gold["val_ctrl"], (len(control), k), "control"),
        (context["truth"], (len(control), k), "scoring"),
    ):
        a = np.asarray(value)
        if a.shape != shape or not np.isin(a, (0, 1)).all():
            raise ValueError("invalid " + name + " truth axis")
    if not np.array_equal(gp[fit], gold["fit"]) or not np.array_equal(
        context["truth"], gold["val_ctrl"]
    ):
        raise ValueError("roots, fit targets and scoring truth are inconsistent")
    probabilities, decisions = context["model_probabilities"], context["model_decisions"]
    if not probabilities or set(probabilities) != set(decisions):
        raise ValueError("model probability and binary decision alias axes differ")
    for alias in probabilities:
        p, d = np.asarray(probabilities[alias]), np.asarray(decisions[alias])
        if (
            p.shape != (n, k)
            or p.dtype.kind != "f"
            or not np.isfinite(p).all()
            or (p < 0).any()
            or (p > 1).any()
            or d.shape != p.shape
            or not np.isin(d, (0, 1)).all()
        ):
            raise ValueError("model arrays differ from document/label axes: " + alias)
    av, ap = context["attribute_values"], context["attribute_presence"]
    # cnt/ttl/mtd values live in Document, not the metadata value dictionary.
    if not set(av) <= set(ap) or not set(ap) <= set(av) | {"cnt", "ttl", "mtd"}:
        raise ValueError("attribute values/presence axes differ")
    for attr in ap:
        if (
            (attr in av and len(av[attr]) != n)
            or np.asarray(ap[attr]).shape != (n,)
            or np.asarray(ap[attr]).dtype != bool
        ):
            raise ValueError("explicit attribute presence required: " + attr)
    summary = {
        "documents": n,
        "labels": k,
        "known_train": len(fit),
        "control": len(control),
        "unlabeled_train": int((roles == "unlabeled_train").sum()),
        "models": len(probabilities),
    }
    if len(excluded):
        summary['excluded_dev'] = len(excluded)
    return summary


RELATION_SCREENING_DEFAULT = {
    "policy": "discriminative_blocked_equality_v1",
    "min_groups": 100,
    "min_group_fraction": 0.05,
    "max_mean_block_partners": 128,
    "min_nonsingleton_fraction": 0.01,
}


def _attribute_values(context, attr):
    if attr in ("cnt", "ttl", "mtd"):
        values = [d.get_attr(attr) for d in context["docs"]]
        presence = np.asarray(context["attribute_presence"].get(attr, [v != "" for v in values]), bool)
        return values, presence
    return list(context["attribute_values"][attr]), np.asarray(context["attribute_presence"][attr], bool)


def relation_screening(context, attributes, policy=None):
    """Keep attribute-equality joins that can discriminate between documents.

    Equality is executed by blocking (grouping documents by the attribute
    value), so a join over attribute A costs sum over groups |g|^2. An
    attribute is kept when it has at least min(``min_groups``, max(2,
    ceil(``min_group_fraction`` x present))) distinct present
    values, the mean number of documents sharing a block is at most
    ``max_mean_block_partners`` (a near-constant attribute such as a 4-way
    cluster id makes every document a partner of a quarter of the corpus and
    carries no document-specific evidence), and at least
    ``min_nonsingleton_fraction`` of present documents share their value with
    another document (otherwise the join only matches a document with
    itself). policy={"policy": "none"} keeps every declared attribute.
    """
    if policy is None:
        policy = dict(RELATION_SCREENING_DEFAULT)
    if not isinstance(policy, dict) or policy.get("policy") not in (
            "none", RELATION_SCREENING_DEFAULT["policy"]):
        raise ValueError("relation_screening must be none or " + RELATION_SCREENING_DEFAULT["policy"])
    if policy["policy"] == "none":
        return {"policy": "none", "kept_attributes": list(attributes), "statistics": {}}
    policy = dict(RELATION_SCREENING_DEFAULT, **policy)
    for key in ("min_groups", "min_group_fraction", "max_mean_block_partners", "min_nonsingleton_fraction"):
        if type(policy[key]) not in (int, float) or policy[key] < 0:
            raise ValueError("relation_screening thresholds must be nonnegative numbers")
    stats, kept, partitions = {}, [], {}
    for attr in attributes:
        values, presence = _attribute_values(context, attr)
        counter, first = {}, {}
        codes = np.full(len(values), -1, np.int64)
        for row, (value, present) in enumerate(zip(values, presence)):
            if present:
                counter[value] = counter.get(value, 0) + 1
                codes[row] = first.setdefault(value, len(first))
        present = int(sum(counter.values()))
        sizes = np.fromiter(counter.values(), np.int64, len(counter))
        pairs = int((sizes * sizes).sum())
        nonsingleton = int(sizes[sizes > 1].sum())
        mean_partners = pairs / present if present else 0.0
        fraction = nonsingleton / present if present else 0.0
        # A corpus smaller than min_groups / min_group_fraction documents
        # cannot have min_groups groups; require 5% of its documents instead.
        required_groups = min(policy["min_groups"],
                              max(2, int(np.ceil(policy["min_group_fraction"] * present))))
        keep = (len(counter) >= required_groups
                and mean_partners <= policy["max_mean_block_partners"]
                and fraction >= policy["min_nonsingleton_fraction"])
        # Two attributes inducing the same row partition define the same join.
        partition = codes.tobytes()
        duplicate_of = partitions.get(partition) if keep else None
        stats[attr] = {"present": present, "groups": len(counter),
                       "largest_group": int(sizes.max()) if len(sizes) else 0,
                       "block_pairs": pairs, "mean_block_partners": mean_partners,
                       "nonsingleton_fraction": fraction, "required_groups": required_groups,
                       "kept": bool(keep and duplicate_of is None),
                       "identical_partition_of": duplicate_of}
        if keep and duplicate_of is None:
            kept.append(attr)
            partitions[partition] = attr
    return {"policy": policy["policy"], "thresholds": {k: policy[k] for k in RELATION_SCREENING_DEFAULT if k != "policy"},
            "kept_attributes": kept, "declared_attributes": list(attributes), "statistics": stats,
            "label_comparisons_kept": True}


def load_inputs(path, overrides=None):
    """Authenticate artifacts, then return the exact portable runner inputs."""
    data, refs, binding, cache = load_manifest(path, overrides)
    with refs["context"].open("rb") as stream:
        context = pickle.load(stream)
    summary = validate_context(context)
    from .model_selection import verify_selection_output, _control, SCHEMA as CONTROL_SCHEMA
    from . import model_selection as selection_execution
    from loris._engine.rlearner import array_identity, digest
    from loris._engine.rlearner_config import PatternSpec, ParameterDomain, paper_relations
    from loris.predicates import predicate_from_dict

    selected = verify_selection_output(refs["model_selection"])
    protocol, arrays = selected["protocol"], selected["input_arrays"]
    fit = np.asarray(context["fit_indices"])
    if (
        protocol["rows"] != [context["row_ids"][int(i)] for i in fit]
        or protocol["labels"] != context["labels"]
        or protocol["protocol"]["oracle_policy"] != data["oracle_policy"]
        or not np.array_equal(arrays["truth"], context["source_gold"]["fit"])
    ):
        raise ValueError("selector fit, label or oracle identity differs")
    interpretation = selected["model_interpretation"]
    if interpretation["row_ids"] != context["row_ids"]:
        raise ValueError("selector model interpretation has different documents")
    for alias in protocol["aliases"]:
        if (
            alias not in context["model_probabilities"]
            or not np.array_equal(
                interpretation["probabilities"][alias], context["model_probabilities"][alias]
            )
            or not np.array_equal(
                interpretation["decisions"][alias], context["model_decisions"][alias]
            )
        ):
            raise ValueError("selector and discovery use different models: " + alias)
    si = selected["result"]["selected_aliases_by_cluster"]
    strategy = data["selection_strategy"]
    from .quality_selection import SCHEMA as QUALITY_SCHEMA

    selected_schema = selected["result"].get("schema")
    if strategy == "learned" and selected_schema in (CONTROL_SCHEMA, QUALITY_SCHEMA):
        raise ValueError("learned strategy requires an actually trained learned-selector artifact")
    if strategy == "quality" and selected_schema != QUALITY_SCHEMA:
        raise ValueError("quality strategy requires a select-quality artifact")
    selection_control = selected.get("control_protocol")
    effective_k = protocol["protocol"]["k"]
    if strategy in {"indiv", "random", "all"}:
        requested_k = selected["result"].get("requested_k", effective_k)
        selection_control = _control(
            arrays,
            protocol["aliases"],
            context["labels"],
            requested_k,
            data["run"]["seed"],
            strategy,
        )
        si = selection_control["selected_aliases_by_cluster"]
        effective_k = selection_control["effective_k"]
    supply = read_json(refs["pattern_supply"])
    if supply.get("status") != "complete":
        raise ValueError("complete pattern supply required")
    pattern_path = bound(supply["patterns"], refs["pattern_supply"].parent)
    records = read_json(pattern_path)
    patterns = {}
    for cluster, items in records.items():
        patterns[cluster] = [
            PatternSpec(
                x["key"],
                predicate_from_dict(x["predicate"]),
                ParameterDomain(**x["frequency_domain"]) if x["frequency_domain"] else None,
                tuple(x["frequency_ops"]),
            )
            for x in items
        ]
    identity_path = refs["pattern_supply"].parent / "FIT_IDENTITY.json"
    identity = read_json(identity_path)
    if (
        identity["fit_row_ids"] != protocol["rows"]
        or identity["labels"] != context["labels"]
        or identity["positive"] != array_identity(context["gamma"].pos[fit])
        or identity["negative"] != array_identity(context["gamma"].neg[fit])
        or identity["clusters"] != array_identity(arrays["assignments"])
        or identity["texts"]
        != {a: digest([context["docs"][int(i)].get_attr(a) for i in fit]) for a in ("cnt", "ttl")}
        or identity.get("semantic_metadata_fields")
        != context.get("receipt", {})
        .get("binding", {})
        .get("text_interpretation", {})
        .get("semantic_metadata_fields")
    ):
        raise ValueError("pattern supply uses different fit inputs")
    if set(si) != set(patterns) or not si or data["run"]["workers"] > len(si) or data["run"]["initialization_workers"] > len(si):
        raise ValueError("Si/Pi cluster axes or worker count differ")
    if any(len(v) != effective_k or len(v) != len(set(v)) for v in si.values()):
        raise ValueError("selected models must preserve the selector K")
    available = {"cnt", "ttl", "mtd"} | set(context["attribute_values"])
    if not set(data["relation_attributes"]) <= available:
        raise ValueError("relation domain requests absent attributes")
    # Missing values still obey the unchanged runtime presence semantics.
    from .search.domain import variable_names

    screening = relation_screening(context, data["relation_attributes"],
                                   data.get("relation_screening"))
    relations = paper_relations(
        variable_names(data["run"]["max_variables"]), screening["kept_attributes"]
    )
    binding.update(
        pattern_specs=file_record(pattern_path),
        pattern_fit=file_record(identity_path),
        Si=si,
        Pi_counts={c: len(patterns[c]) for c in si},
        summary=summary,
        selection_strategy=strategy,
        selection_control=selection_control,
        effective_k=effective_k,
        selection_execution=file_record(selection_execution.__file__),
        manifest_execution=file_record(__file__),
        relation_screening=screening,
    )
    return dict(
        context=context,
        si=si,
        patterns=patterns,
        relations=relations,
        input_binding=binding,
        lexical_cache_dir=cache,
        **data["run"],
    )
