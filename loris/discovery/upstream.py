"""Manifest-bound rich-discovery features, context and selector preparation.

Known training supervision comes exclusively from supervision.py. MiniLM may
encode the complete train/dev text axis, but KMeans fits known-train rows only.
Model-pool scores are authenticated, not generated or relabeled by this module.
"""

import argparse
import contextlib
import fcntl
import json
import sys
from importlib.metadata import version
import os
import pickle
import time
from pathlib import Path

import numpy as np

from .manifest import bound, file_record, validate_context
from .supervision import load_supervision

FEATURE_SCHEMA = "loris_fit_semantic_features_v1"
DEFAULT_BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SELECTOR_PROTOCOL = {
    "k": 4,
    "epochs": 100,
    "hidden_dim": 128,
    "num_samples": 500,
    "sigma": 0.05,
    "learning_rate": 0.001,
    "lambda_task": 0.1,
    "lambda_entropy": 0.01,
    "imitation_positive_weight": 2.0,
    "vote_smoothing": 0.1,
    "dropout": 0.1,
    "seed": 42,
    "device": "cuda:0",
    "oracle_policy": "accuracy_then_vote_bce",
}


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def document_views(
    records, semantic_cluster_ids=None, *, title_field="title", semantic_metadata_fields=None
):
    """One text interpretation shared by pattern fit and full-context inference.

    Empty fields stay absent. Source row IDs are not invented relation fields.
    Derived cluster IDs are a separately documented attribute; they are not
    written back into original metadata or mistaken for entity relations.
    """
    from loris.document import Document
    from loris._engine.rdl_reference import prepare_attributes
    from loris._engine.rlearner import canonical

    records = list(records)
    if not records:
        raise ValueError("nonempty document records required")
    fields = sorted(set().union(*(set(row["metadata"]) for row in records)))
    if semantic_metadata_fields is not None:
        if (
            not isinstance(semantic_metadata_fields, list)
            or any(not isinstance(field, str) or not field for field in semantic_metadata_fields)
            or len(set(semantic_metadata_fields)) != len(semantic_metadata_fields)
            or not set(semantic_metadata_fields) <= set(fields)
        ):
            raise ValueError(
                "semantic_metadata_fields must be unique existing source fields, or []"
            )
        fields = sorted(semantic_metadata_fields)
    if "semantic_cluster_id" in fields and semantic_cluster_ids is not None:
        raise ValueError("derived semantic_cluster_id would overwrite a source field")
    values = {"mtd." + key: [] for key in fields}
    if semantic_cluster_ids is not None:
        ids = np.asarray(semantic_cluster_ids)
        if ids.shape != (len(records),) or ids.dtype.kind not in "iu" or (ids < 0).any():
            raise ValueError("complete nonnegative semantic cluster axis required")
        values["mtd.semantic_cluster_id"] = []
    docs = []
    for index, record in enumerate(records):
        original_metadata = record["metadata"]
        if not isinstance(record["text"], str) or not isinstance(original_metadata, dict):
            raise ValueError("real text and metadata records required")
        metadata = {field: original_metadata.get(field, "") for field in fields}
        if any(not isinstance(v, str) for v in metadata.values()):
            raise ValueError("declared semantic metadata values must remain strings")
        available = {key: value for key, value in metadata.items() if value.strip()}
        title = metadata.get(title_field, "")
        docs.append(
            Document(
                cnt=record["text"],
                ttl=title if title.strip() else "",
                mtd=canonical(available) if available else "",
            )
        )
        for field in fields:
            value = metadata.get(field, "")
            values["mtd." + field].append(value if value.strip() else "")
        if semantic_cluster_ids is not None:
            values["mtd.semantic_cluster_id"].append("fit_kmeans:" + str(int(ids[index])))
    _, presence = prepare_attributes(docs, values)
    return {"docs": docs, "attribute_values": values, "attribute_presence": presence}


def records_to_documents(records, *, title_field="title", semantic_metadata_fields=None):
    return document_views(
        records, title_field=title_field, semantic_metadata_fields=semantic_metadata_fields
    )["docs"]


def fit_clusters(features, fit_indices, *, n_clusters=4, seed=42, n_init=10, max_iter=300):
    """Fit centroids using only the frozen known-train representation rows."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    x, fit = np.asarray(features), np.asarray(fit_indices)
    if (
        x.ndim != 2
        or not len(x)
        or not np.isfinite(x).all()
        or fit.ndim != 1
        or fit.dtype.kind not in "iu"
        or not len(fit)
        or len(set(fit.tolist())) != len(fit)
        or (fit < 0).any()
        or (fit >= len(x)).any()
    ):
        raise ValueError("valid feature and complete fit axes required")
    if type(n_clusters) is not int or not 1 <= n_clusters <= len(fit):
        raise ValueError("cluster count must fit the known sample")
    with threadpool_limits(limits=1):
        model = KMeans(
            n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=seed
        ).fit(x[fit])
        assigned = model.predict(x).astype(np.int64)
    if set(assigned[fit].tolist()) != set(range(n_clusters)):
        raise ValueError("requested clusters must have actual fit support")
    return model, assigned


def _encode_documents(texts, backbone, *, device, max_length, batch_size):
    # The public model must already exist locally; this code does not download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(backbone), device=device)
    model.max_seq_length = max_length
    features = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return features


@contextlib.contextmanager
def _feature_gpu_lock(path, device):
    start = time.monotonic()
    if path is None or not str(device).startswith("cuda"):
        yield 0.0
        return
    lock_path = Path(path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        print(json.dumps({"event": "waiting_feature_gpu_lock", "path": str(lock_path)}), flush=True)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield time.monotonic() - start
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def prepare_features(
    supervision_manifest,
    output_dir,
    *,
    backbone=DEFAULT_BACKBONE,
    device="cuda",
    max_length=256,
    batch_size=128,
    n_clusters=4,
    seed=42,
    n_init=10,
    max_iter=300,
    gpu_lock=None,
):
    """Encode train/dev text then fit known-train KMeans; no target scoring."""
    from loris._engine.rlearner import array_identity

    start = time.monotonic()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("new feature namespace required: " + str(output))
    data = load_supervision(supervision_manifest, include_dev_targets=False)
    if (
        type(max_length) is not int
        or max_length <= 0
        or type(batch_size) is not int
        or batch_size <= 0
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise ValueError("valid explicit embedding parameters required")
    if (
        type(n_clusters) is not int
        or not 1 <= n_clusters <= len(data["fit_indices"])
        or type(n_init) is not int
        or n_init < 1
        or type(max_iter) is not int
        or max_iter < 1
    ):
        raise ValueError(
            "cluster count/initializations/iterations must be positive and fit the known sample"
        )
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": FEATURE_SCHEMA,
        "supervision": data["manifest_record"],
        "backbone": str(backbone),
        "device": device,
        "max_length": max_length,
        "batch_size": batch_size,
        "normalize_embeddings": True,
        "gpu_lock": str(Path(gpu_lock).resolve()) if gpu_lock else None,
        "n_clusters": n_clusters,
        "seed": seed,
        "n_init": n_init,
        "max_iter": max_iter,
        "feature_domain": "all train+dev text",
        "centroid_fit_domain": "known train only",
        "dev_or_test_used_to_fit_centroids": False,
        "dev_targets_read": False,
        "test_documents_or_truth_read": False,
        "source": file_record(__file__),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scikit-learn": version("scikit-learn"),
            "sentence-transformers": version("sentence-transformers"),
            "torch": version("torch"),
        },
    }
    # A local checkpoint snapshot path can be bound file-by-file. Public cached
    # names remain explicit; caller must supply a resolved path for frozen bytes.
    try:
        model_path = Path(backbone)
        if not model_path.is_dir():
            from huggingface_hub import snapshot_download, constants

            cache = (
                os.environ.get("HF_HUB_CACHE")
                or os.environ.get("HUGGINGFACE_HUB_CACHE")
                or (
                    str(Path(os.environ["HF_HOME"]) / "hub")
                    if os.environ.get("HF_HOME")
                    else constants.HF_HUB_CACHE
                )
            )
            model_path = Path(
                snapshot_download(repo_id=str(backbone), cache_dir=cache, local_files_only=True)
            )
        if not (model_path / "modules.json").is_file():
            raise ValueError("local SentenceTransformer snapshot modules.json required")
        protocol["resolved_backbone"] = str(model_path.resolve())
        protocol["backbone_files"] = [
            file_record(p) for p in sorted(model_path.rglob("*")) if p.is_file()
        ]
        _write(output / "PROTOCOL.json", protocol)
        cost = {}
        with _feature_gpu_lock(gpu_lock, device) as wait_seconds:
            cost["gpu_lock_wait_seconds"] = wait_seconds
            on_cuda = str(device).startswith("cuda")
            if on_cuda:
                import torch

                torch.cuda.reset_peak_memory_stats(device)
            encoding_start = time.monotonic()
            features = _encode_documents(
                [r["text"] for r in data["docs"]],
                model_path,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
            )
            if on_cuda:
                torch.cuda.synchronize(device)
                cost["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                cost["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
                cost["gpu_name"] = torch.cuda.get_device_name(device)
                torch.cuda.empty_cache()
            else:
                cost["peak_gpu_allocated_bytes"] = None
                cost["peak_gpu_reserved_bytes"] = None
            cost["encoding_seconds"] = time.monotonic() - encoding_start
        if (
            features.shape != (len(data["docs"]), 384)
            or features.dtype != np.float32
            or not np.isfinite(features).all()
        ):
            raise ValueError(
                "MiniLM feature shape/dtype differs from the declared384-dimensional family"
            )
        cluster_start = time.monotonic()
        model, assignment = fit_clusters(
            features,
            data["fit_indices"],
            n_clusters=n_clusters,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
        )
        cost["clustering_seconds"] = time.monotonic() - cluster_start
        np.save(output / "features.npy", features)
        fit = data["fit_indices"]
        np.savez_compressed(
            output / "CLUSTER_ARRAYS.npz",
            cluster_assignments=assignment[fit],
            discovery_cluster_ids=assignment,
            fit_indices=fit,
            centroids=model.cluster_centers_,
        )
        clustering = {
            "schema": FEATURE_SCHEMA,
            "n_clusters": n_clusters,
            "n_init": n_init,
            "max_iter": max_iter,
            "seed": seed,
            "fit_indices": fit.tolist(),
            "fit_row_ids": data["fit_row_ids"],
            "feature_identity": array_identity(features[fit]),
            "assignments": array_identity(assignment[fit]),
            "centroids": array_identity(model.cluster_centers_),
            "sizes": np.bincount(assignment[fit], minlength=n_clusters).tolist(),
            "n_iter": int(model.n_iter_),
            "inertia": float(model.inertia_),
            "all_rows": len(features),
            "full_feature_file": file_record(output / "features.npy"),
            "dev_or_test_used_to_fit_centroids": False,
            "supervision": data["manifest_record"],
        }
        _write(output / "CLUSTERING.json", clustering)
        result = {
            "schema": FEATURE_SCHEMA,
            "status": "complete",
            "protocol": file_record(output / "PROTOCOL.json"),
            "supervision": data["manifest_record"],
            "scores": file_record(output / "features.npy"),
            "clusters": file_record(output / "CLUSTERING.json"),
            "cluster_arrays": file_record(output / "CLUSTER_ARRAYS.npz"),
            "shape": list(features.shape),
            "row_ids": data["row_ids"],
            "seconds": time.monotonic() - start,
            "model_training_calls": 0,
            "BO_calls": 0,
            "test_read": False,
            "dev_targets_read": False,
            "cost": cost,
        }
        _write(output / "FEATURES_COMPLETE.json", result)
        return result
    except Exception as error:
        _write(
            output / "FAILED.json",
            {
                "status": "failed",
                "reason": str(error),
                "error_type": type(error).__name__,
                "seconds": time.monotonic() - start,
            },
        )
        raise


def _feature_data(receipt_path, data, *, load_features=False):
    from loris._engine.rlearner import array_identity

    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = _read(receipt_path)
    if (
        receipt.get("schema") != FEATURE_SCHEMA
        or receipt.get("status") != "complete"
        or receipt["supervision"] != data["manifest_record"]
        or receipt["row_ids"] != data["row_ids"]
    ):
        raise ValueError("feature receipt differs from the complete supervision axis")
    paths = {
        k: bound(receipt[k], receipt_path.parent)
        for k in ("scores", "clusters", "cluster_arrays", "protocol")
    }
    clustering = _read(paths["clusters"])
    if (
        clustering["fit_indices"] != data["fit_indices"].tolist()
        or clustering["fit_row_ids"] != data["fit_row_ids"]
        or clustering["dev_or_test_used_to_fit_centroids"] is not False
    ):
        raise ValueError("clustering fit scope differs")
    with np.load(paths["cluster_arrays"], allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    if (
        not np.array_equal(arrays["fit_indices"], data["fit_indices"])
        or arrays["discovery_cluster_ids"].shape != (len(data["docs"]),)
        or not np.array_equal(
            arrays["cluster_assignments"], arrays["discovery_cluster_ids"][data["fit_indices"]]
        )
        or clustering["assignments"] != array_identity(arrays["cluster_assignments"])
    ):
        raise ValueError("cluster arrays/receipt differ")
    features = np.load(paths["scores"], allow_pickle=False) if load_features else None
    if features is not None and (
        list(features.shape) != receipt["shape"]
        or clustering["feature_identity"] != array_identity(features[data["fit_indices"]])
    ):
        raise ValueError("feature matrix/fit identity differs")
    return receipt, clustering, arrays, paths, features


def prepare_context(
    supervision_manifest,
    pool_handoff,
    feature_receipt,
    output_dir,
    *,
    title_field="title",
    semantic_metadata_fields=None,
):
    """Bind a real model pool to full signed known roots and dev-only scoring."""
    from loris.chase import GammaSeed
    from loris._engine.rlearner import array_identity, digest

    start = time.monotonic()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("new context namespace required")
    data = load_supervision(supervision_manifest)
    receipt, clustering, cluster_arrays, feature_paths, _ = _feature_data(feature_receipt, data)
    hand_path = Path(pool_handoff).resolve(strict=True)
    hand = _read(hand_path)
    if (
        hand.get("status") != "complete"
        or hand.get("supervision_manifest") != data["manifest_record"]
        or hand.get("row_ids") != data["row_ids"]
        or hand.get("labels") != data["labels"]
        or hand.get("row_roles") != data["row_roles"]
        or not hand.get("channels")
    ):
        raise ValueError("model handoff must authenticate the same complete supervision axis")
    n, k = len(data["docs"]), len(data["labels"])
    probs = {}
    inputs = {r["path"]: r["sha256"] for r in data["files_read"].values()}
    for p in (
        Path(supervision_manifest),
        hand_path,
        Path(feature_receipt),
        *feature_paths.values(),
        Path(__file__),
    ):
        record = file_record(p)
        inputs[record["path"]] = record["sha256"]
    order = hand.get("model_order")
    if (
        not isinstance(order, list)
        or len(set(order)) != len(order)
        or set(order) != set(hand["channels"])
    ):
        raise ValueError("handoff requires explicit fixed model_order")
    for alias in order:
        info = hand["channels"][alias]
        if not isinstance(alias, str) or not alias or alias in probs:
            raise ValueError("real unique model aliases required")
        path = bound(info["probabilities"], hand_path.parent)
        with np.load(path, allow_pickle=False) as values:
            scores = values["scores"]
            if (
                scores.shape != (n, k)
                or scores.dtype != np.float32
                or not np.isfinite(scores).all()
                or (scores < 0).any()
                or (scores > 1).any()
                or values["labels"].tolist() != data["labels"]
                or values["row_ids"].tolist() != data["row_ids"]
                or not np.array_equal(values["global_rows"], np.arange(n, dtype=np.int64))
            ):
                raise ValueError("model probability identity/axes/range differs")
            probs[alias] = scores.copy()
        inputs[str(path)] = info["probabilities"]["sha256"]
    if not isinstance(title_field, str) or not title_field:
        raise ValueError("explicit nonempty title metadata field required")
    views = document_views(
        data["docs"],
        cluster_arrays["discovery_cluster_ids"],
        title_field=title_field,
        semantic_metadata_fields=semantic_metadata_fields,
    )
    binding = {
        "schema": "loris_complete_discovery_context_v1",
        "supervision": data["manifest_record"],
        "models": {key: array_identity(value) for key, value in probs.items()},
        "model_order": order,
        "row_ids": data["row_ids"],
        "fit": data["fit_indices"].tolist(),
        "labels": data["labels"],
        "inputs": inputs,
        "attribute_values": views["attribute_values"],
        "texts": digest([r["text"] for r in data["docs"]]),
        "semantic_attribute": "fit-only KMeans assignment; unchanged AttrEqual, not entity identity",
        "text_interpretation": {
            "cnt": "prepared text",
            "title_field": title_field,
            "mtd": "canonical nonempty projected source metadata; None projection retains all original fields",
            "semantic_metadata_fields": semantic_metadata_fields,
        },
    }
    context = dict(
        views,
        schema=binding["schema"],
        context_id=digest(binding),
        labels=data["labels"],
        row_ids=data["row_ids"],
        row_roles=data["row_roles"],
        raw_indices=np.arange(n, dtype=np.int64),
        identities=[
            {"role": row["role"], "raw_id": row["row_id"], "global_row": i}
            for i, row in enumerate(data["docs"])
        ],
        fit_indices=data["fit_indices"],
        receiver_indices=data["receiver_indices"],
        test_indices=np.asarray([], np.int64),
        gamma=GammaSeed(data["gamma_pos"], data["gamma_neg"]),
        source_gold={"fit": data["fit_targets"], "val_ctrl": data["dev_targets"]},
        truth=data["dev_targets"],
        model_probabilities=probs,
        scores=probs,
        model_decisions={key: value >= 0.5 for key, value in probs.items()},
        raw_metadata=[row["metadata"] for row in data["docs"]],
        theta_id=digest(binding["models"]),
        input_files=inputs,
        receipt={"binding": binding, "test_read": False, "unmasked_train_truth_read": False},
    )
    validate_context(context)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "discovery_context.pkl").open("xb") as stream:
        pickle.dump(context, stream, protocol=4)
    result = {
        "schema": binding["schema"],
        "status": "complete",
        "context": file_record(output / "discovery_context.pkl"),
        "supervision": data["manifest_record"],
        "pool_handoff": file_record(hand_path),
        "features": file_record(feature_receipt),
        "rows": n,
        "labels": k,
        "known": len(data["fit_indices"]),
        "model_aliases": list(probs),
        "model_order": list(probs),
        "title_field": title_field,
        "semantic_metadata_fields": semantic_metadata_fields,
        "seconds": time.monotonic() - start,
        "test_read": False,
        "unmasked_train_truth_read": False,
        "training_calls": 0,
        "BO_calls": 0,
    }
    _write(output / "COMPLETE.json", result)
    return result


def prepare_selector(
    supervision_manifest, context_receipt, feature_receipt, output_dir, *, protocol=None
):
    """Prepare the existing learned selector; this function does not train it."""
    start = time.monotonic()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("new selector-input namespace required")
    data = load_supervision(supervision_manifest, include_dev_targets=False)
    feature_done, clustering, clusters, _, features = _feature_data(
        feature_receipt, data, load_features=True
    )
    receipt_path = Path(context_receipt).resolve(strict=True)
    receipt = _read(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("supervision") != data["manifest_record"]:
        raise ValueError("context receipt differs from supervision")
    context_path = bound(receipt["context"], receipt_path.parent)
    # Trusted local pipeline pickle. It contains dev scoring labels by design;
    # only source_gold.fit is serialized into selector training inputs.
    with context_path.open("rb") as stream:
        context = pickle.load(stream)
    validate_context(context)
    if context["row_ids"] != data["row_ids"] or context["labels"] != data["labels"]:
        raise ValueError("context axes differ")
    if not np.array_equal(context["source_gold"]["fit"], data["fit_targets"]):
        raise ValueError("context known targets differ")
    aliases = list(context["model_probabilities"])
    if receipt.get("model_order") != aliases:
        raise ValueError("context receipt model order differs")
    config = dict(DEFAULT_SELECTOR_PROTOCOL if protocol is None else protocol)
    if (
        set(config) != set(DEFAULT_SELECTOR_PROTOCOL)
        or type(config["k"]) is not int
        or not 1 <= config["k"] <= len(aliases)
    ):
        raise ValueError("complete declared selector protocol and valid K required")
    fit = data["fit_indices"]
    arrays = {
        "features": features[fit],
        "truth": data["fit_targets"],
        "cluster_assignments": clusters["cluster_assignments"],
    }
    interpretation = {"row_ids": data["row_ids"], "probabilities": {}, "decisions": {}}
    votes = []
    for i, alias in enumerate(aliases):
        pk, dk = "p%02d" % i, "d%02d" % i
        arrays[pk], arrays[dk] = (
            context["model_probabilities"][alias],
            context["model_decisions"][alias],
        )
        interpretation["probabilities"][alias], interpretation["decisions"][alias] = pk, dk
        votes.append(arrays[dk][fit])
    arrays["decisions"] = np.stack(votes)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "SELECTOR_INPUTS.npz", **arrays)
    manifest = {
        "arrays": file_record(output / "SELECTOR_INPUTS.npz"),
        "row_ids": data["fit_row_ids"],
        "aliases": aliases,
        "labels": data["labels"],
        "protocol": config,
        "model_interpretation": interpretation,
        "input_binding": {
            "role": "fit",
            "expected_fit_rows": len(fit),
            "feature_provenance": clustering,
            "decision_provenance": {
                "policy": "probability_ge_0_5",
                "context": file_record(context_path),
                "alias_order": aliases,
            },
            "supervised_fit_truth": {
                "supervision": data["manifest_record"],
                "field": "known_train_targets",
            },
            "historical_train_dev_use_disclosed": True,
            "official_test_truth_used": False,
            "context_contains_dev_scoring_truth": True,
            "dev_truth_used_as_selector_targets": False,
        },
    }
    _write(output / "select_models.manifest.json", manifest)
    result = {
        "status": "complete",
        "manifest": file_record(output / "select_models.manifest.json"),
        "supervision": data["manifest_record"],
        "source": file_record(__file__),
        "seconds": time.monotonic() - start,
        "training_calls": 0,
        "BO_calls": 0,
    }
    _write(output / "COMPLETE.json", result)
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("features", "context", "selector"))
    cli.add_argument("--supervision", required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--features", help="FEATURES_COMPLETE.json, required for context/selector")
    cli.add_argument("--pool", help="HANDOFF.json, required for context")
    cli.add_argument(
        "--title-field", default="title", help="actual metadata title field; missing remains empty"
    )
    cli.add_argument(
        "--semantic-metadata-fields",
        type=json.loads,
        help="explicit JSON list of semantic source fields; [] excludes all source metadata; omitted retains historical all-field view",
    )
    cli.add_argument("--context", help="context COMPLETE.json, required for selector")
    cli.add_argument("--backbone", default=DEFAULT_BACKBONE)
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--max-length", type=int, default=256)
    cli.add_argument("--batch-size", type=int, default=128)
    cli.add_argument("--n-clusters", type=int, default=4)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--n-init", type=int, default=10)
    cli.add_argument("--max-iter", type=int, default=300)
    cli.add_argument(
        "--gpu-lock", help="optional explicit shared lock path for CUDA feature extraction"
    )
    cli.add_argument(
        "--protocol", help="complete selector protocol JSON; omitted uses declared defaults"
    )
    args = cli.parse_args(argv)
    if args.stage == "features":
        result = prepare_features(
            args.supervision,
            args.output,
            backbone=args.backbone,
            device=args.device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            n_clusters=args.n_clusters,
            seed=args.seed,
            n_init=args.n_init,
            max_iter=args.max_iter,
            gpu_lock=args.gpu_lock,
        )
    elif args.stage == "context":
        if not args.pool or not args.features:
            cli.error("context requires --pool and --features")
        result = prepare_context(
            args.supervision,
            args.pool,
            args.features,
            args.output,
            title_field=args.title_field,
            semantic_metadata_fields=args.semantic_metadata_fields,
        )
    else:
        if not args.context or not args.features:
            cli.error("selector requires --context and --features")
        result = prepare_selector(
            args.supervision,
            args.context,
            args.features,
            args.output,
            protocol=_read(args.protocol) if args.protocol else None,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
