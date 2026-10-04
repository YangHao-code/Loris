"""Fit-only quality regression and candidate selection before rule learning.

Experimental alternatives to the original joint-vote selector. Neither utility
estimates future rule value. Inputs supply already frozen calibration and text
clusters; this command never tunes experts, thresholds, clusters or BO rules.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from .manifest import bound, file_record, read_json
from . import quality_scoring as scoring

SCHEMA = "loris_fit_quality_selection_v1"
INPUT_SCHEMA = "loris_quality_inputs_v1"
UTILITIES = ("probability_brier", "threshold_brier")
SCORERS = ("static_global", "static_cluster", "ridge", "residual_mlp")
AGGREGATIONS = ("frequency", "predicted_max")


def load_quality_inputs(manifest, quality_inputs):
    """Read complete fit truth only; preserve the frozen downstream interpretation."""
    from .model_selection import _inputs

    manifest = Path(manifest).resolve(strict=True)
    side_path = Path(quality_inputs).resolve(strict=True)
    data, protocol, arrays, interpretation = _inputs(manifest)
    side = read_json(side_path)
    required = {"schema", "selector_manifest", "arrays", "row_ids", "aliases", "labels",
                "calibration_provenance", "group_provenance"}
    if not isinstance(side, dict) or set(side) != required or side["schema"] != INPUT_SCHEMA:
        raise ValueError("quality inputs require the documented fit-only sidecar schema")
    if bound(side["selector_manifest"], side_path.parent) != manifest:
        raise ValueError("quality inputs belong to a different selector manifest")
    for key in ("row_ids", "aliases", "labels"):
        if side[key] != data[key]:
            raise ValueError("quality " + key + " axis differs from selector inputs")
    for key in ("calibration_provenance", "group_provenance"):
        if not isinstance(side[key], dict) or not side[key]:
            raise ValueError(key + " must describe the existing frozen preparation")
    with np.load(bound(side["arrays"], side_path.parent), allow_pickle=False) as z:
        if set(z.files) != {"groups", "calibrated_probabilities", "thresholds"}:
            raise ValueError("quality arrays must contain groups, calibrated_probabilities, thresholds only")
        extra = {key: z[key].copy() for key in z.files}
    n, m, labels = len(data["row_ids"]), len(data["aliases"]), len(data["labels"])
    groups = extra["groups"]
    if (groups.shape != (n,) or groups.dtype.kind not in "iuUS"
            or (groups.dtype.kind in "US" and np.any(groups.astype(str) == ""))):
        raise ValueError("one explicit nonempty connected-group ID per fit row required")
    calibrated, thresholds = extra["calibrated_probabilities"], extra["thresholds"]
    for values, shape, name in [(calibrated, (m, n, labels), "calibrated probabilities"),
                                (thresholds, (m, labels), "model/label thresholds")]:
        if (values.shape != shape or values.dtype.kind != "f" or not np.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()):
            raise ValueError("invalid " + name + " axis or range")
    row_lookup = {row: i for i, row in enumerate(interpretation["row_ids"])}
    fit_rows = [row_lookup[row] for row in data["row_ids"]]
    raw = np.stack([interpretation["probabilities"][alias][fit_rows] for alias in data["aliases"]])
    # Raw decision boundaries are authoritative. Normalized soft scores are NOT
    # substituted for decisions, including at zero/one or float32 boundaries.
    if not np.array_equal(raw.astype(np.float64) >= thresholds.astype(np.float64)[:, None, :],
                          arrays["decisions"]):
        raise ValueError("supplied thresholds do not reproduce the frozen fit decisions")
    extra["raw_probabilities"] = raw
    return data, protocol, arrays, interpretation, side, extra


def _utility(extra, truth, utility):
    if utility not in UTILITIES:
        raise ValueError("unsupported quality utility")
    values = extra["calibrated_probabilities"] if utility == "probability_brier" else scoring.threshold_scores(
        extra["raw_probabilities"], extra["thresholds"])
    return scoring.balanced_squared_utility(values, truth)


def _save_state(path, state):
    # No pickle or executable checkpoint is needed by the text-only forward.
    arrays = {key: value for key, value in state.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in state.items() if key not in arrays}
    np.savez_compressed(path, __metadata__=np.asarray(json.dumps(metadata, allow_nan=False)), **arrays)


def load_state(path):
    with np.load(path, allow_pickle=False) as z:
        state = json.loads(str(z["__metadata__"].item()))
        state.update({key: z[key].copy() for key in z.files if key != "__metadata__"})
        return state


def _sources():
    from . import model_selection, selector_backend
    from loris._learning import selection
    return [file_record(module.__file__) for module in (scoring, model_selection, selector_backend, selection)]


def select_quality(manifest, quality_inputs, output, *, utility, scorer="static_cluster",
                   aggregation="frequency", k=None, folds=3, seed=42, steps=80,
                   ridge_alpha=100., feature_scale=5., hidden_dim=32, correction_l2=.1,
                   weight_decay=.01, learning_rate=.001, threads=4, device="cpu",
                   max_combinations=100000):
    from loris._engine.rlearner import immutable_json

    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("preserve the existing selector; choose a new output directory")
    if scorer not in SCORERS or aggregation not in AGGREGATIONS:
        raise ValueError("unsupported scorer or aggregation")
    data, protocol, arrays, interpretation, side, extra = load_quality_inputs(manifest, quality_inputs)
    effective_k = protocol.k if k is None else k
    target = _utility(extra, arrays["truth"], utility)
    # Validate aggregation and its enumeration budget before any fitting.
    scoring.aggregate_candidates(np.zeros_like(target[:1]), arrays["assignments"][:1], data["aliases"],
                                 effective_k, method=aggregation, max_combinations=max_combinations)
    options = dict(method=scorer, folds=folds, seed=seed, steps=steps, ridge_alpha=ridge_alpha,
                   feature_scale=feature_scale, hidden_dim=hidden_dim, correction_l2=correction_l2,
                   weight_decay=weight_decay, learning_rate=learning_rate, threads=threads, device=device)
    definition = dict(schema=SCHEMA, utility=utility, scorer=scorer, aggregation=aggregation,
        k=effective_k, max_combinations=max_combinations, fit_options=options,
        input_manifest=file_record(manifest), quality_inputs=file_record(quality_inputs),
        source=file_record(__file__), supporting_sources=_sources(),
        target="r = U - mean_models(U); no per-document standard deviation scaling",
        probability_utility="1 - squared probability error; equal nonempty positive/negative label groups",
        threshold_utility="same squared-error utility on threshold-normalized soft scores; not calibrated probabilities",
        Si_rows="connected-group OOF predictions on fit rows only",
        checkpoint="fixed steps in every OOF fit and main fit; no validation epoch selection",
        forward="text features and text-derived cluster IDs only; no labels or expert predictions",
        calibration="supplied frozen mapping/thresholds; not fitted by this command",
        tie_break="canonical alias order; predicted_max primary within 1e-12, then mean member score, then aliases",
        paper_status="author-authorized research options, not a paper-defined unique target or proven rule utility",
        cost="full-pool fit predictions were required upstream; reading cached predictions is not inference savings")
    out.mkdir(parents=True, exist_ok=False)
    immutable_json(out / "PROTOCOL.json", definition)
    started = time.monotonic()
    try:
        fitted = scoring.fit_predict(arrays["features"], target, arrays["assignments"], data["aliases"],
                                     groups=extra["groups"], **options)
        si = scoring.aggregate_candidates(fitted["oof_scores"], arrays["assignments"], data["aliases"],
                                          effective_k, method=aggregation, max_combinations=max_combinations)
        _save_state(out / "MODEL.npz", fitted["state"])
        fold_models = []
        for fold, state in enumerate(fitted["fold_states"]):
            path = out / ("FOLD_%02d.npz" % fold)
            _save_state(path, state)
            fold_models.append(file_record(path))
        np.savez_compressed(out / "SCORES.npz", utility=target, oof_scores=fitted["oof_scores"],
                            fit_scores=fitted["fit_scores"], fold_ids=fitted["fold_ids"])
        union = sorted({alias for aliases in si.values() for alias in aliases})
        result = dict(schema=SCHEMA, strategy="quality", utility=utility, scorer=scorer,
                      aggregation=aggregation, requested_k=effective_k, effective_k=effective_k,
                      selected_aliases_by_cluster=si, model_order=data["aliases"], labels=data["labels"],
                      fit_rows=len(target), cluster_count=len(si), expert_union=union,
                      expert_union_size=len(union), full_pool_size=len(data["aliases"]),
                      new_expert_predictions=0, new_BO=0, selection_truth_permission="complete known fit only")
        immutable_json(out / "SELECTED.json", result)
        immutable_json(out / "FIT.json", dict(provenance=fitted["provenance"],
                       seconds=time.monotonic() - started, validation_labels_loaded=False,
                       relative_fit_mse=float(np.mean((fitted["fit_scores"] - scoring.relative(target))**2)),
                       relative_oof_mse=float(np.mean((fitted["oof_scores"] - scoring.relative(target))**2))))
        complete = dict(schema=SCHEMA, status="complete", protocol=file_record(out / "PROTOCOL.json"),
                        selected=file_record(out / "SELECTED.json"), scores=file_record(out / "SCORES.npz"),
                        model=file_record(out / "MODEL.npz"), fit=file_record(out / "FIT.json"),
                        fold_models=fold_models)
        immutable_json(out / "COMPLETE.json", complete)
        return dict(complete, selected_aliases_by_cluster=si, expert_union=union)
    except Exception as error:
        immutable_json(out / "FAILED.json", dict(status="failed", error=type(error).__name__, message=str(error)))
        raise


def verify_quality_output(selected_path):
    """Validate the common discovery interface without retraining a selector."""
    path = Path(selected_path).resolve(strict=True)
    if path.name not in ("SELECTED.json", "COMPLETE.json"):
        raise ValueError("quality reference must be SELECTED.json or COMPLETE.json")
    complete = read_json(path.parent / "COMPLETE.json")
    if complete.get("schema") != SCHEMA or complete.get("status") != "complete":
        raise ValueError("quality selector is incomplete")
    artifacts = {name: bound(complete[name], path.parent) for name in ("protocol", "selected", "scores", "model", "fit")}
    for name, file_path in artifacts.items():
        if file_path != path.parent / {"protocol":"PROTOCOL.json", "selected":"SELECTED.json", "scores":"SCORES.npz",
                                       "model":"MODEL.npz", "fit":"FIT.json"}[name]:
            raise ValueError("quality completion must bind its own " + name + " artifact")
    definition, result = read_json(artifacts["protocol"]), read_json(artifacts["selected"])
    if definition.get("schema") != SCHEMA or definition["source"] != file_record(__file__) or definition["supporting_sources"] != _sources():
        raise ValueError("quality selector implementation differs from the saved protocol")
    manifest = bound(definition["input_manifest"], path.parent)
    sidecar = bound(definition["quality_inputs"], path.parent)
    data, protocol, arrays, interpretation, side, extra = load_quality_inputs(manifest, sidecar)
    with np.load(artifacts["scores"], allow_pickle=False) as z:
        target, oof, folds, fit = (z[key] for key in ("utility", "oof_scores", "fold_ids", "fit_scores"))
    if not np.array_equal(target, _utility(extra, arrays["truth"], definition["utility"])):
        raise ValueError("saved quality target differs from the frozen input definition")
    expected_folds = scoring.group_fold_ids(extra["groups"], definition["fit_options"]["folds"], definition["fit_options"]["seed"])
    if folds.dtype.kind not in "iu" or not np.array_equal(folds, expected_folds):
        raise ValueError("invalid complete OOF fold assignment")
    if len(complete["fold_models"]) != definition["fit_options"]["folds"] or oof.shape != target.shape:
        raise ValueError("missing OOF models or wrong OOF axes")
    options = {key:value for key, value in definition["fit_options"].items() if key != "folds"}
    for fold, record in enumerate(complete["fold_models"]):
        fold_path = bound(record, path.parent)
        if fold_path != path.parent / ("FOLD_%02d.npz" % fold):
            raise ValueError("OOF model belongs to a different output or fold")
        fold_state = load_state(fold_path)
        held = folds == fold
        expected_options = dict(options, seed=(options["seed"] + 1009 * fold) % 2**32)
        if (fold_state["configuration"] != expected_options or fold_state["training_rows"] != int((~held).sum())
                or list(fold_state["aliases"]) != data["aliases"]):
            raise ValueError("OOF model differs from the declared fold fit")
        held_scores = scoring.predict(fold_state, arrays["features"][held], arrays["assignments"][held])
        if not np.allclose(held_scores, oof[held], atol=1e-7, rtol=1e-6):
            raise ValueError("OOF scores do not replay from the corresponding held-group model")
    state = load_state(artifacts["model"])
    if (list(state["aliases"]) != data["aliases"] or state["method"] != definition["scorer"]
            or state["configuration"] != options or state["training_rows"] != len(target)):
        raise ValueError("quality model axes or scorer differ")
    predicted = scoring.predict(state, arrays["features"], arrays["assignments"])
    if fit.shape != target.shape or not np.allclose(predicted, fit, atol=1e-7, rtol=1e-6):
        raise ValueError("saved main scores differ from text-only forward")
    expected = scoring.aggregate_candidates(oof, arrays["assignments"], data["aliases"], definition["k"],
                                            method=definition["aggregation"], max_combinations=definition["max_combinations"])
    if (result.get("schema") != SCHEMA or result["selected_aliases_by_cluster"] != expected
            or result["effective_k"] != definition["k"]
            or any(result[key] != definition[key] for key in ("utility", "scorer", "aggregation"))):
        raise ValueError("saved Si differs from fit-only OOF aggregation")
    # Only k/oracle_policy are consumed by the older discovery binding. Do not
    # return its joint-vote loss description as if it described this new target.
    return dict(result=result, input_arrays=arrays, model_interpretation=interpretation,
                protocol=dict(rows=data["row_ids"], labels=data["labels"], aliases=data["aliases"],
                              protocol=dict(k=definition["k"], oracle_policy=protocol.oracle_policy,
                                            objective=definition["utility"], schema=SCHEMA)),
                control_protocol=definition)


def predict_quality(model, inputs, output):
    """Export all-expert scores. Expert predictions and truth are forbidden here."""
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("score output already exists")
    with np.load(inputs, allow_pickle=False) as z:
        if set(z.files) != {"features", "clusters", "row_ids"}:
            raise ValueError("text-only forward accepts features, clusters, row_ids only")
        features, clusters, rows = z["features"], z["clusters"], z["row_ids"]
    if rows.ndim != 1 or len(rows) != len(features) or len(np.unique(rows)) != len(rows):
        raise ValueError("unique text row IDs required")
    state = load_state(model)
    values = scoring.predict(state, features, clusters)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        np.savez_compressed(stream, row_ids=rows, clusters=clusters, aliases=np.asarray(state["aliases"]), scores=values)
    return dict(status="complete", rows=len(rows), models=len(state["aliases"]),
                output=str(out), expert_predictions_used=False, labels_used=False,
                prediction=scoring.prediction_provenance(state, clusters))


def prepare_quality_inputs(manifest, source_arrays, output, calibration_description, group_description):
    """Bind a precomputed calibration pack to exact fit/model/label axes."""
    from .model_selection import _inputs
    from loris._engine.rlearner import immutable_json

    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("quality input output already exists")
    data, _, _, _ = _inputs(manifest)
    with np.load(source_arrays, allow_pickle=False) as z:
        expected = {"row_ids", "aliases", "labels", "groups", "calibrated_probabilities", "thresholds"}
        if set(z.files) != expected:
            raise ValueError("source pack requires exact row_ids, aliases, labels and the three quality arrays")
        for key in ("row_ids", "aliases", "labels"):
            if z[key].tolist() != data[key]:
                raise ValueError("source pack " + key + " differs from selector inputs")
        arrays = {key:z[key].copy() for key in ("groups", "calibrated_probabilities", "thresholds")}
    if not calibration_description.strip() or not group_description.strip():
        raise ValueError("describe calibration data/history and connected-group construction")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "ARRAYS.npz", **arrays)
    side = dict(schema=INPUT_SCHEMA, selector_manifest=file_record(manifest),
                arrays=file_record(out / "ARRAYS.npz"), **{key:data[key] for key in ("row_ids", "aliases", "labels")},
                calibration_provenance=dict(description=calibration_description, source=file_record(source_arrays)),
                group_provenance=dict(description=group_description, source=file_record(source_arrays)))
    immutable_json(out / "quality_inputs.json", side)
    load_quality_inputs(manifest, out / "quality_inputs.json")
    return dict(status="complete", quality_inputs=str(out / "quality_inputs.json"), fitted=False)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="bind a precomputed, axis-labelled fit calibration pack; no fitting")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--source-arrays", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--calibration-description", required=True, help="frozen mapping and thresholds, data role and historical exposure")
    prepare.add_argument("--group-description", required=True, help="document duplicate/connected-group definition")
    fit = sub.add_parser("fit", help="freeze fit-only Si, then pass COMPLETE.json to rule discovery")
    fit.add_argument("--manifest", required=True, help="existing fit selector input manifest")
    fit.add_argument("--quality-inputs", required=True, help="frozen calibration/groups sidecar")
    fit.add_argument("--output", required=True, help="new output directory")
    fit.add_argument("--utility", choices=UTILITIES, required=True)
    fit.add_argument("--scorer", choices=SCORERS, default="static_cluster")
    fit.add_argument("--aggregation", choices=AGGREGATIONS, default="frequency")
    fit.add_argument("--k", type=int, help="default: K in the selector input manifest")
    for name, default in [("folds",3), ("seed",42), ("steps",80), ("hidden-dim",32), ("threads",4), ("max-combinations",100000)]:
        fit.add_argument("--" + name, type=int, default=default)
    for name, default in [("ridge-alpha",100.), ("feature-scale",5.), ("correction-l2",.1), ("weight-decay",.01), ("learning-rate",.001)]:
        fit.add_argument("--" + name, type=float, default=default)
    fit.add_argument("--device", default="cpu", help="MLP training device, e.g. cpu or cuda:0")
    score = sub.add_parser("score", help="text-only all-expert scoring from a saved MODEL.npz")
    score.add_argument("--model", required=True)
    score.add_argument("--inputs", required=True, help="NPZ: features, clusters, row_ids; no truth/predictions")
    score.add_argument("--output", required=True)
    args = vars(cli.parse_args(argv))
    command = args.pop("command")
    try:
        value = {"fit":select_quality, "score":predict_quality, "prepare":prepare_quality_inputs}[command](**args)
        print(json.dumps(value, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="failed", error=type(error).__name__, message=str(error))), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
