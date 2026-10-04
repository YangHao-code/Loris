"""Executable model-selection controls using the same authenticated fit inputs.

These controls do not imitate a learned selector or train a replacement model.
`learned` delegates the existing real selector; other strategies use its input
manifest without requiring a previously trained selector checkpoint.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .manifest import bound, file_record, read_json

SCHEMA = "loris_fit_model_selection_control_v1"
STRATEGIES = ("learned", "indiv", "random", "all")


def _inputs(manifest):
    from .selector_backend import validate_selector_manifest
    from loris._learning.selection import (
        ModelSelectionProtocol,
        _validate,
        _normalize_model_interpretation,
    )

    path = Path(manifest).resolve(strict=True)
    data = read_json(path)
    validate_selector_manifest(data)
    protocol = ModelSelectionProtocol(**data["protocol"])
    array_path = bound(data["arrays"], path.parent)
    with np.load(array_path, allow_pickle=False) as z:
        x, d, y, a, rows, aliases, labels = _validate(
            z["features"],
            z["decisions"],
            z["truth"],
            z["cluster_assignments"],
            data["row_ids"],
            data["aliases"],
            data["labels"],
            data["input_binding"],
            protocol,
        )
        description = data.get("model_interpretation")
        if not isinstance(description, dict):
            raise ValueError(
                "complete frozen model interpretation required for downstream discovery"
            )
        interpretation = {"row_ids": description["row_ids"]}
        for kind in ("probabilities", "decisions"):
            if set(description.get(kind, {})) != set(aliases):
                raise ValueError("every model needs full probability and decision axes")
            interpretation[kind] = {
                alias: np.array(z[key], copy=True) for alias, key in description[kind].items()
            }
    _normalize_model_interpretation(interpretation, aliases, labels, rows, d)
    return data, protocol, dict(features=x, decisions=d, truth=y, assignments=a), interpretation


def _control(arrays, aliases, labels, k, seed, strategy):
    from loris._engine.rlearner import binary_metrics

    if strategy not in ("indiv", "random", "all"):
        raise ValueError("control requires indiv, random or all")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("control selection seed must be an unsigned32 integer")
    if type(k) is not int or not 1 <= k <= len(aliases):
        raise ValueError("K must be a valid fixed pool size")
    assignments, votes, truth = arrays["assignments"], arrays["decisions"], arrays["truth"]
    groups = np.unique(assignments)
    if seed + 1009 * (len(groups) - 1) >= 2**32:
        raise ValueError("per-cluster random seeds exceed unsigned32")
    result, details = {}, []
    predictions = np.zeros_like(truth)
    effective_k = len(aliases) if strategy == "all" else k
    for position, cluster in enumerate(groups):
        rows = assignments == cluster
        scores = [binary_metrics(truth[rows], votes[i, rows]) for i in range(len(aliases))]
        if strategy == "indiv":
            # Rank only the same assigned fit rows. No dev or global fallback.
            chosen = sorted(range(len(aliases)), key=lambda i: (-scores[i]["macro_f1"], i))[:k]
        elif strategy == "random":
            chosen = (
                np.random.RandomState(seed + 1009 * position)
                .choice(len(aliases), k, replace=False)
                .tolist()
            )
        else:
            chosen = list(range(len(aliases)))
        chosen = sorted(chosen)  # Retain the declared model axis inside each Si.
        prediction = 2 * votes[chosen][:, rows].sum(axis=0) > effective_k
        predictions[rows] = prediction
        result[str(int(cluster))] = [aliases[i] for i in chosen]
        details.append(
            dict(
                cluster=str(int(cluster)),
                fit_rows=int(rows.sum()),
                individual_metrics={alias: scores[i] for i, alias in enumerate(aliases)},
                selected_indices=chosen,
                vote_metrics=binary_metrics(truth[rows], prediction),
            )
        )
    return dict(
        selected_aliases_by_cluster=result,
        strategy=strategy,
        requested_k=k,
        effective_k=effective_k,
        same_k_comparison=effective_k == k,
        all_is_capacity_control=strategy == "all",
        seed=seed,
        model_order=list(aliases),
        cluster_order=list(result),
        labels=list(labels),
        clusters=details,
        fit_vote_metrics=binary_metrics(truth, predictions),
        completed_optimizer_steps=0,
        scoring_rows="assigned fit rows only; zero-support labels remain in the fixed macro axis",
        tie_break="individual F1 ties by declared model order; even vote ties are negative",
        fitted_pool_updated=False,
        new_BO=0,
    )


def select_models(manifest, output, *, strategy, seed=None, resume=False, cublas_workspace=None):
    if strategy not in STRATEGIES:
        raise ValueError("unsupported model selection strategy")
    if strategy == "learned":
        if seed is not None:
            raise ValueError("learned seed is frozen in its manifest, not an unbound override")
        from .selector import train_selector

        return train_selector(manifest, output, resume=resume, cublas_workspace=cublas_workspace)
    if resume or cublas_workspace is not None:
        raise ValueError(
            "fit controls have no training/resume/CUDA-workspace option; choose a new output"
        )
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(
            "selection control output exists; preserve it and choose a new namespace"
        )
    manifest = Path(manifest).resolve(strict=True)
    data, protocol, arrays, _ = _inputs(manifest)
    effective_seed = protocol.seed if seed is None else seed
    result = _control(arrays, data["aliases"], data["labels"], protocol.k, effective_seed, strategy)
    from loris._engine.rlearner import immutable_json
    from loris._engine import rlearner
    from loris._learning import selection as original_selection
    from . import selector_backend

    definition = dict(
        schema=SCHEMA,
        strategy=strategy,
        seed=effective_seed,
        input_manifest=file_record(manifest),
        source=file_record(__file__),
        supporting_sources=[
            file_record(rlearner.__file__),
            file_record(original_selection.__file__),
            file_record(selector_backend.__file__),
        ],
        k=protocol.k,
        selection_truth_permission="complete known fit only",
        checkpoint_required=False,
        training_updates=0,
        distinction="author-defined selection control, not the latest paper NoJoint or an external method",
    )
    out.mkdir(parents=True, exist_ok=False)
    immutable_json(out / "PROTOCOL.json", definition)
    immutable_json(out / "SELECTED.json", dict(result, schema=SCHEMA))
    complete = dict(
        schema=SCHEMA,
        status="complete",
        protocol=file_record(out / "PROTOCOL.json"),
        selected=file_record(out / "SELECTED.json"),
        input_manifest=file_record(manifest),
        source=file_record(__file__),
        training_updates=0,
        new_BO=0,
    )
    immutable_json(out / "COMPLETE.json", complete)
    return dict(complete, selected_aliases_by_cluster=result["selected_aliases_by_cluster"])


def verify_selection_output(selected_path):
    """Return the common selector interface, with no training or output writes."""
    path = Path(selected_path).resolve(strict=True)
    raw = read_json(path)
    if path.name not in ("SELECTED.json", "COMPLETE.json"):
        raise ValueError("model selection reference must identify SELECTED.json or COMPLETE.json")
    from .quality_selection import SCHEMA as QUALITY_SCHEMA, verify_quality_output

    if raw.get("schema") == QUALITY_SCHEMA:
        return verify_quality_output(path)
    if raw.get("schema") != SCHEMA:
        from loris._learning.selection import verify_model_selection_output

        return verify_model_selection_output(path.parent)
    complete = raw if path.name == "COMPLETE.json" else read_json(path.parent / "COMPLETE.json")
    if path.name == "COMPLETE.json":
        selected = bound(complete["selected"], path.parent)
        if selected.parent != path.parent or selected.name != "SELECTED.json":
            raise ValueError("control completion must identify its own selected artifact")
        path, raw = selected, read_json(selected)
    if complete.get("status") != "complete" or bound(complete["selected"], path.parent) != path:
        raise ValueError("selection control is incomplete or differs from its selected artifact")
    definition = read_json(bound(complete["protocol"], path.parent))
    source = bound(complete["source"], path.parent)
    if source != Path(__file__).resolve() or definition["source"] != complete["source"]:
        raise ValueError(
            "control implementation differs; do not silently reinterpret a historical selector"
        )
    from loris._engine import rlearner
    from loris._learning import selection as original_selection
    from . import selector_backend

    if definition["supporting_sources"] != [
        file_record(rlearner.__file__),
        file_record(original_selection.__file__),
        file_record(selector_backend.__file__),
    ]:
        raise ValueError("control scoring or input-validation implementation differs")
    manifest = bound(definition["input_manifest"], path.parent)
    if definition["input_manifest"] != complete["input_manifest"]:
        raise ValueError("control input identity differs")
    data, protocol, arrays, interpretation = _inputs(manifest)
    expected = dict(
        _control(
            arrays,
            data["aliases"],
            data["labels"],
            protocol.k,
            definition["seed"],
            definition["strategy"],
        ),
        schema=SCHEMA,
    )
    if raw != expected or definition["k"] != protocol.k:
        raise ValueError("stored control selection is not the declared fit-only strategy")
    return dict(
        result=raw,
        input_arrays=arrays,
        model_interpretation=interpretation,
        protocol=dict(
            rows=data["row_ids"],
            labels=data["labels"],
            aliases=data["aliases"],
            protocol=dict(protocol.manifest(), k=raw["effective_k"]),
        ),
        control_protocol=definition,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="existing upstream selector-input manifest; fit only"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument(
        "--seed", type=int, help="indiv/random/all only; default is the frozen selector seed"
    )
    parser.add_argument(
        "--resume", action="store_true", help="learned only, exact existing selector identity"
    )
    parser.add_argument(
        "--cublas-workspace", choices=(":4096:8", ":16:8"), help="learned CUDA only"
    )
    args = parser.parse_args(argv)
    try:
        print(json.dumps(select_models(**vars(args)), indent=2, allow_nan=False))
        return 0
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(dict(status="failed", error_type=type(error).__name__, reason=str(error))),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
