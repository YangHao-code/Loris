"""Comparable model predictions and whole-program dependency ablations.

This module never changes a frozen checkpoint or a discovery artifact. All
threshold selection here is explicitly in-sample development selection. Final
holdout/interactive evaluation belongs to a separately bound labeling manifest.
"""

import csv
import json
from pathlib import Path
import time

import numpy as np

from loris.discovery.manifest import load_inputs, file_record
from loris._engine.rdl_reference import Doc, Text, ML, Label, LabelCompare, AttrEqual, Rule
from loris._engine.rlearner import atomic_json, binary_metrics, array_identity


VARIANTS = ("full", "pure_ml", "no_text", "no_label", "no_ground")


def metrics(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    result = binary_metrics(y, p)
    tp, fp, fn = (int(sum(result[k])) for k in ("tp", "fp", "fn"))
    result.update(
        TP=tp,
        FP=fp,
        FN=fn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        coverage=float(p.any(axis=1).mean()) if len(p) else None,
    )
    multi = y.sum(axis=1) > 1
    result["multi_label_completeness"] = (
        float(((y & p)[multi].sum(axis=1) / y[multi].sum(axis=1)).mean()) if multi.any() else None
    )
    return result


def cell_changes(y, before, after):
    y, before, after = (np.asarray(x, bool) for x in (y, before, after))
    if y.shape != before.shape or y.shape != after.shape:
        raise ValueError("cell difference axes differ")
    return dict(
        recovered_FN=int((y & ~before & after).sum()),
        added_FP=int((~y & ~before & after).sum()),
        lost_TP=int((y & before & ~after).sum()),
        removed_FP=int((~y & before & ~after).sum()),
    )


def calibrate_thresholds(y, probabilities, grid):
    """Per-label F1; deterministic ties prefer the threshold nearest 0.5."""
    # The RDL runtime normalizes probability tables to float64. Match that
    # comparison exactly, including float32 values close to a grid threshold.
    y, p = np.asarray(y, bool), np.asarray(probabilities, dtype=np.float64)
    grid = np.asarray(grid, float)
    if y.shape != p.shape or grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
        raise ValueError("invalid calibration axes/grid")
    if (grid < 0).any() or (grid > 1).any() or len(set(grid.tolist())) != len(grid):
        raise ValueError("grid must contain distinct probabilities")
    order = sorted(grid.tolist(), key=lambda t: (abs(t - 0.5), t))
    best = np.full(p.shape[1], -1.0)
    thresholds = np.full(p.shape[1], 0.5)
    for t in order:
        pred = p >= t
        tp = (pred & y).sum(axis=0)
        denom = 2 * tp + (pred & ~y).sum(axis=0) + (~pred & y).sum(axis=0)
        f1 = np.divide(2.0 * tp, denom, out=np.zeros(len(tp)), where=denom != 0)
        better = f1 > best
        thresholds[better], best[better] = t, f1[better]
    return thresholds


def model_program(alias, labels, thresholds):
    if len(thresholds) != len(labels):
        raise ValueError("label/threshold axis differs")
    # Both fixed and dynamic predicates compare probabilities with kappa;
    # dynamic controls eligibility for between-closure model updates.
    return [
        Rule((Doc("x"), ML("x", alias, label, float(t), True)), Label("x", label))
        for label, t in zip(labels, thresholds)
    ]


def transform_program(rules, variant):
    if variant not in VARIANTS:
        raise ValueError("unsupported fixed-program variant")
    selected, removed = [], []
    for i, r in enumerate(rules):
        non_doc = [a for a in r.body if not isinstance(a, Doc)]
        keep = True
        if variant == "pure_ml":
            keep = (
                bool(non_doc)
                and all(isinstance(a, ML) for a in non_doc)
                and isinstance(r.head, Label)
            )
        elif variant == "no_text":
            keep = not any(isinstance(a, Text) for a in r.body)
        elif variant == "no_label":
            keep = not any(isinstance(a, (Label, LabelCompare)) for a in r.body) and not isinstance(
                r.head, LabelCompare
            )
        elif variant == "no_ground" and any(isinstance(a, ML) for a in r.body):
            # Paper NoGround is a separate, deliberately widened counterfactual;
            # it must never be called a whole-rule dependency deletion.
            r = Rule(tuple(a for a in r.body if not isinstance(a, (Text, AttrEqual))), r.head)
        if keep:
            selected.append(r)
        else:
            removed.append(i)
    return selected, removed


def syntax_statistics(rules):
    from loris.predicates import FreqPredicate

    rows = []
    for i, r in enumerate(rules):
        ml = [a for a in r.body if isinstance(a, ML)]
        text = [a for a in r.body if isinstance(a, Text)]
        rows.append(
            dict(
                index=i,
                text_conditions=len(text),
                ml_conditions=len(ml),
                label_premises=sum(isinstance(a, (Label, LabelCompare)) for a in r.body),
                relation_head=isinstance(r.head, LabelCompare),
                ml_ge_zero=sum(a.dynamic and a.eta == 0 for a in ml),
                freq_ge_zero=sum(
                    isinstance(a.predicate, FreqPredicate)
                    and getattr(a.predicate, "op", None) == ">="
                    and getattr(a.predicate, "eta", None) == 0
                    for a in text
                ),
            )
        )
    return dict(
        rules=rows,
        categories_overlap=True,
        distinct_variables_do_not_certify_distinct_documents=True,
        syntactic_nonrestrictive_conditions_are_not_extra_evidence=True,
    )


def _new_output(output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    return out


def routed_vote(context, si, fit_assignments):
    """Authenticate metadata-to-selector cluster mapping on the complete fit axis.

    Metadata names are not parsed heuristically. A mapping is established only
    from the already authenticated selector assignments and matching fit rows.
    """
    values = context["attribute_values"].get("mtd.semantic_cluster_id")
    if values is None:
        raise ValueError("no declared full-corpus routing assignments")
    fit = np.asarray(context["fit_indices"])
    if len(fit_assignments) != len(fit):
        raise ValueError("selector assignment fit axis differs")
    mapping = {}
    for row, cluster in zip(fit, fit_assignments):
        key, cluster = values[int(row)], str(cluster)
        if key in mapping and mapping[key] != cluster:
            raise ValueError("metadata group and selector assignment disagree")
        if cluster not in si:
            raise ValueError("selector cluster absent from Si")
        mapping[key] = cluster
    idx = np.asarray(context["receiver_indices"])
    if any(values[int(row)] not in mapping for row in idx):
        raise ValueError("scoring rows contain unauthenticated routing groups")
    result = np.zeros((len(idx), len(context["labels"])), bool)
    for n, row in enumerate(idx):
        aliases = si[mapping[values[int(row)]]]
        result[n] = np.stack([context["model_decisions"][a][row] for a in aliases]).sum(
            axis=0
        ) * 2 > len(aliases)
    return result, mapping


def _write_tables(out, records, labels):
    fields = [
        "name",
        "prediction_object",
        "status",
        "macro_f1",
        "micro_f1",
        "precision",
        "recall",
        "TP",
        "FP",
        "FN",
        "coverage",
        "multi_label_completeness",
        "rules",
        "seconds",
    ]
    with (out / "MAIN_RESULTS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(dict(record, **(record.get("metrics") or {})))
    with (out / "PER_LABEL.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "label", "F1", "TP", "FP", "FN"])
        for row in records:
            if row.get("metrics") is not None:
                m = row["metrics"]
                for i, label in enumerate(labels):
                    writer.writerow(
                        [
                            row["name"],
                            label,
                            m["per_label_f1"][i],
                            m["tp"][i],
                            m["fp"][i],
                            m["fn"][i],
                        ]
                    )


def baselines(manifest, output, grid):
    inputs = load_inputs(manifest)
    c = inputs["context"]
    out = _new_output(output)
    start = time.monotonic()
    idx = np.asarray(c["receiver_indices"])
    y = c["truth"]
    records, predictions, thresholds = [], {}, {}
    for alias, p in c["model_probabilities"].items():
        p = np.asarray(p, dtype=np.float64)
        t = calibrate_thresholds(y, p[idx], grid)
        thresholds[alias] = t.tolist()
        for suffix, threshold in (("at_0.5", 0.5), ("dev_calibrated", t)):
            name = alias + ":" + suffix
            pred = p[idx] >= threshold
            key = "prediction_%03d" % len(predictions)
            predictions[key] = pred
            records.append(
                dict(
                    name=name,
                    prediction_object="raw_model",
                    status="complete",
                    metrics=metrics(y, pred),
                    prediction_key=key,
                )
            )
    # Existing selection protocol: strict binary majority, negative ties.
    combinations = dict(all_models=list(c["model_decisions"]))
    combinations.update(
        {"Si_" + cluster: list(aliases) for cluster, aliases in inputs["si"].items()}
    )
    for name, aliases in combinations.items():
        pred = np.stack([c["model_decisions"][a][idx] for a in aliases]).sum(axis=0) * 2 > len(
            aliases
        )
        key = "prediction_%03d" % len(predictions)
        predictions[key] = pred
        records.append(
            dict(
                name=name,
                prediction_object="binary_vote",
                status="complete",
                aliases=aliases,
                positive_ties=False,
                metrics=metrics(y, pred),
                prediction_key=key,
            )
        )
        if name.startswith("Si_"):
            records[-1][
                "scope"
            ] = "one_cluster_combination_on_all_dev_diagnostic_not_routed_selector"
    from loris._learning.selection import verify_model_selection_output

    selection_path = Path(inputs["input_binding"]["files"]["model_selection"]["path"])
    selected = verify_model_selection_output(selection_path.parent)
    try:
        pred, mapping = routed_vote(c, inputs["si"], selected["input_arrays"]["assignments"])
        key = "prediction_%03d" % len(predictions)
        predictions[key] = pred
        records.append(
            dict(
                name="Si_actual_routed",
                prediction_object="binary_vote",
                status="complete",
                mapping=mapping,
                positive_ties=False,
                metrics=metrics(y, pred),
                prediction_key=key,
            )
        )
    except ValueError as error:
        records.append(
            dict(
                name="Si_actual_routed",
                prediction_object="binary_vote",
                status="unavailable",
                metrics=None,
                reason=str(error),
            )
        )
    np.savez_compressed(out / "PREDICTIONS.npz", **predictions)
    atomic_json(
        out / "THRESHOLDS.json",
        dict(
            grid=list(grid),
            thresholds=thresholds,
            comparison=">=",
            selection_rows=[c["row_ids"][int(i)] for i in idx],
            selection="per_label_macro_F1",
            tie_break="closest_to_0.5_then_lower",
        ),
    )
    receipt = dict(
        schema="loris_matched_model_baselines_v1",
        status="complete",
        records=records,
        implementation=file_record(__file__),
        input_binding=inputs["input_binding"],
        scoring_rows=[c["row_ids"][int(i)] for i in idx],
        labels=c["labels"],
        truth_identity=array_identity(np.asarray(y)),
        evaluation="development; same rows select thresholds and score; not independent holdout",
        model_threshold_comparison=">=",
        calibration_candidates_per_label=len(grid),
        model_training=0,
        BO_trials=0,
        seconds=time.monotonic() - start,
        baseline_predictions_are_not_Gamma_or_logical_P=True,
    )
    _write_tables(out, records, c["labels"])
    atomic_json(out / "RESULTS.json", receipt)
    atomic_json(
        out / "COMPLETE.json", dict(status="complete", results=file_record(out / "RESULTS.json"))
    )
    return receipt


def evaluate_program(
    manifest, rule_path, output, variant="full", alias=None, thresholds=None, calibrate_grid=None
):
    inputs = load_inputs(manifest)
    c = inputs["context"]
    idx = np.asarray(c["receiver_indices"])
    if alias is None:
        raw = json.loads(Path(rule_path).read_text())
        rules = [Rule.from_dict(r) for r in (raw["rules"] if isinstance(raw, dict) else raw)]
        source = file_record(rule_path)
        rules, removed = transform_program(rules, variant)
    else:
        if alias not in c["model_probabilities"]:
            raise ValueError("model alias absent from authenticated context")
        if calibrate_grid is not None:
            thresholds = calibrate_thresholds(
                c["truth"], c["model_probabilities"][alias][idx], calibrate_grid
            )
        if thresholds is None:
            thresholds = [0.5] * len(c["labels"])
        rules = model_program(alias, c["labels"], thresholds)
        source, removed = (
            dict(
                model_alias=alias,
                comparison=">=",
                thresholds=list(thresholds),
                calibration_grid=calibrate_grid,
            ),
            [],
        )
    out = _new_output(output)
    atomic_json(out / "RULES.json", dict(rules=[r.to_dict() for r in rules]))
    atomic_json(out / "SYNTAX.json", syntax_statistics(rules))
    from loris._storage.discovery_factory import make_discovery_runner
    from loris.discovery.runner import _write_provenance

    runner = make_discovery_runner(
        c,
        out / "static_cache",
        subset_direction="standard",
        lexical_cache_dir=inputs.get("lexical_cache_dir"),
    )
    started = time.monotonic()
    try:
        result = runner(tuple(rules), gamma=c["gamma"])
        good = result.status == "fixpoint"
        # A failed partial state is saved for diagnosis, never scored normally.
        np.savez_compressed(
            out / ("FACTS.npz" if good else "FAILED_PARTIAL_FACTS.npz"),
            pos=result.positive_facts,
            neg=result.negative_facts,
        )
        proofs = _write_provenance(result, out)
        m = metrics(c["truth"], result.positive_facts[idx]) if good else None
        record = dict(
            name=alias or variant,
            prediction_object="successful_Chase_P",
            status=result.status,
            metrics=m,
            rules=len(rules),
            removed_rule_indices=removed,
            seconds=time.monotonic() - started,
            conflicts=[[int(i), str(k)] for i, k in result.conflicts],
            comparison_conflicts=list(getattr(result, "comparison_conflicts", ())),
        )
        receipt = dict(
            schema="loris_fixed_program_evaluation_v1",
            record=record,
            implementation=file_record(__file__),
            intervention=(
                "paper_NoGround_guard_removal_counterfactual"
                if variant == "no_ground"
                else (
                    "single_model_complete_program"
                    if alias is not None
                    else "fixed_program_whole_rule_deletion"
                )
            ),
            input_binding=inputs["input_binding"],
            source=source,
            proofs=proofs,
            scoring_rows=[c["row_ids"][int(i)] for i in idx],
            labels=c["labels"],
            truth_identity=array_identity(np.asarray(c["truth"])),
            evaluation="development",
            training_updates=0,
            external_queries=0,
            BO_trials=0,
            failed_closures_not_scored=True,
            provenance_scope="stored witnesses; not all alternative proofs",
        )
        _write_tables(out, [record], c["labels"])
        atomic_json(out / "RESULTS.json", receipt)
        atomic_json(
            out / "COMPLETE.json",
            dict(
                status="complete_evaluation",
                chase_status=result.status,
                results=file_record(out / "RESULTS.json"),
            ),
        )
        return receipt
    except BaseException as e:
        atomic_json(
            out / "FAILURE.json", dict(status="failed", error=type(e).__name__, reason=str(e))
        )
        raise
    finally:
        runner.close()
