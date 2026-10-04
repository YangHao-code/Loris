"""Text-only quality scoring and fit-domain candidate aggregation.

This module does not load datasets, calibrate experts, select epochs, or run
rules. Utilities are supervision, never prediction inputs. ``fit_predict``
constructs group-held-out predictions at a fixed training budget, then fits a
separate deployment model on all supplied training rows. Callers must bind the
feature, model, document, label and calibration axes outside this module.

States contain only arrays, scalars, lists and dictionaries. ``predict`` uses
NumPy only, including for states trained with the optional torch dependency.
"""

import itertools
import math

import numpy as np


SCHEMA = "loris_text_quality_scoring_v1"
METHODS = ("static_global", "static_cluster", "ridge", "residual_mlp")


def _finite(value, name, ndim):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != ndim or not np.isfinite(a).all():
        raise ValueError("%s must be a finite %d-dimensional array" % (name, ndim))
    return a


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % name)
    if int(value) < minimum:
        raise ValueError("%s must be at least %d" % (name, minimum))
    return int(value)


def _positive(value, name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("%s must be a finite number" % name)
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError("%s must be finite and %s" % (name, "nonnegative" if allow_zero else "positive"))
    return value


def _clusters(clusters, rows):
    a = np.asarray(clusters)
    if a.shape != (rows,) or a.dtype.kind not in "iu" or np.any(a < 0):
        raise ValueError("clusters must be a row-aligned vector of nonnegative integers")
    if a.dtype.kind == "u" and np.any(a > np.iinfo(np.int64).max):
        raise ValueError("cluster identifiers must fit int64")
    return a.astype(np.int64, copy=False)


def _aliases(aliases, count):
    a = np.asarray(aliases)
    if a.shape != (count,) or not all(isinstance(v, (str, np.str_)) and v for v in a):
        raise ValueError("aliases must be nonempty strings aligned to the model axis")
    names = a.tolist()
    if len(set(names)) != count:
        raise ValueError("aliases must be unique")
    return np.asarray(names, dtype=str)


def balanced_squared_utility(p, truth):
    """Return U[N,M] = 1 minus squared error averaged over nonempty sign groups.

    Positive and negative *true-label* groups receive equal weight when both
    exist. This outcome-dependent balance is not a claim of a strictly proper
    probability scoring rule. No endpoint clipping is needed for squared error.
    """
    p = _finite(p, "p", 3)
    y = np.asarray(truth)
    if min(p.shape) < 1 or y.shape != p.shape[1:] or not np.isin(y, [0, 1]).all():
        raise ValueError("truth must be binary [N,L] aligned to nonempty p[M,N,L]")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p must lie in [0,1]")
    y = y.astype(bool)
    error = (p - y[None]) ** 2
    positive, negative = y.sum(1), (~y).sum(1)
    plus = np.divide((error * y[None]).sum(2), positive[None],
                     out=np.zeros(p.shape[:2]), where=positive[None] > 0)
    minus = np.divide((error * (~y)[None]).sum(2), negative[None],
                      out=np.zeros(p.shape[:2]), where=negative[None] > 0)
    groups = (positive > 0).astype(int) + (negative > 0).astype(int)
    return (1 - (plus + minus) / groups[None]).T


def threshold_scores(raw, thresholds):
    """Reproduce the existing threshold-normalized float32 score convention.

    Both logit arguments are clipped to [1e-7,1-1e-7]; exact raw endpoints are
    preserved. These scores are not calibrated probabilities. Hard decisions
    must still compare float64 raw scores to their original thresholds: clipping
    and float32 rounding can affect the 0.5 boundary, particularly at endpoints.
    """
    p = _finite(raw, "raw", 3)
    t = _finite(thresholds, "thresholds", 2)
    if min(p.shape) < 1 or t.shape != (p.shape[0], p.shape[2]):
        raise ValueError("thresholds must have shape [M,L] for raw[M,N,L]")
    if np.any((p < 0) | (p > 1)) or np.any((t < 0) | (t > 1)):
        raise ValueError("raw scores and thresholds must lie in [0,1]")
    pc, tc = np.clip(p, 1e-7, 1 - 1e-7), np.clip(t, 1e-7, 1 - 1e-7)
    z = np.log(pc) - np.log1p(-pc) - (np.log(tc) - np.log1p(-tc))[:, None, :]
    result = (1 / (1 + np.exp(-np.clip(z, -70, 70)))).astype(np.float32)
    result[p == 0] = 0
    result[p == 1] = 1
    return result


def relative(utility):
    """Center each document across models; never divide by row standard deviation."""
    u = _finite(utility, "utility", 2)
    if u.shape[1] < 1:
        raise ValueError("utility must have a nonempty model axis")
    return u - u.mean(1, keepdims=True)


def _order(values, aliases):
    canonical = np.argsort(aliases, kind="stable")
    return canonical[np.argsort(-values[..., canonical], axis=-1, kind="stable")]


def aggregate_candidates(scores, clusters, aliases, K, method="frequency", max_combinations=100000):
    """Return {str(cluster_id): [aliases]} from fit-domain predicted scores.

    ``frequency`` counts document Top-K votes, resolving exact ties by alias.
    ``predicted_max`` enumerates sets maximizing mean document max score; ties
    within 1e-12 use mean selected-member score, then lexicographic alias tuple.
    Input contains no truth. Callers must supply OOF scores for OOF aggregation.
    """
    s = _finite(scores, "scores", 2)
    if min(s.shape) < 1:
        raise ValueError("scores must have nonempty document and model axes")
    c, names = _clusters(clusters, len(s)), _aliases(aliases, s.shape[1])
    K = _integer(K, "K", 1)
    limit = _integer(max_combinations, "max_combinations", 1)
    if K > len(names) or method not in ("frequency", "predicted_max"):
        raise ValueError("invalid K or aggregation method")
    combinations = None
    if method == "predicted_max":
        count = math.comb(len(names), K)
        if count > limit:
            raise ValueError("predicted_max needs %d combinations, exceeding limit %d" % (count, limit))
        combinations = np.asarray(list(itertools.combinations(np.argsort(names), K)), dtype=np.int64)
    result = {}
    for cluster in np.unique(c):
        rows = s[c == cluster]
        if method == "frequency":
            votes = np.bincount(_order(rows, names)[:, :K].ravel(), minlength=len(names))
            ids = _order(votes, names)[:K]
        else:
            # Bound intermediate gathers independently of the number of documents.
            objectives = np.empty(len(combinations), dtype=np.float64)
            for start in range(0, len(combinations), 256):
                sets = combinations[start:start + 256]
                totals = np.zeros(len(sets), dtype=np.float64)
                row_chunk = max(1, 1000000 // (len(sets) * K))
                for offset in range(0, len(rows), row_chunk):
                    totals += rows[offset:offset + row_chunk, sets].max(2).sum(0)
                objectives[start:start + len(sets)] = totals / len(rows)
            tied = np.flatnonzero(objectives >= objectives.max() - 1e-12)
            secondary = rows.mean(0)[combinations[tied]].mean(1)
            chosen = tied[np.flatnonzero(secondary >= secondary.max() - 1e-12)[0]]
            ids = combinations[chosen]
        result[str(int(cluster))] = names[ids].tolist()
    return result


def _configuration(method, seed, steps, ridge_alpha, feature_scale, hidden_dim,
                   correction_l2, weight_decay, learning_rate, threads, device):
    if method not in METHODS:
        raise ValueError("unknown quality scoring method: %s" % method)
    seed = _integer(seed, "seed")
    if seed >= 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    return dict(method=method, seed=seed, steps=_integer(steps, "steps"),
                ridge_alpha=_positive(ridge_alpha, "ridge_alpha"),
                feature_scale=_positive(feature_scale, "feature_scale"),
                hidden_dim=_integer(hidden_dim, "hidden_dim", 1),
                correction_l2=_positive(correction_l2, "correction_l2", True),
                weight_decay=_positive(weight_decay, "weight_decay", True),
                learning_rate=_positive(learning_rate, "learning_rate"),
                threads=_integer(threads, "threads", 1), device=str(device))


def _fit_arrays(features, utility, clusters, aliases):
    x, u = _finite(features, "features", 2), _finite(utility, "utility", 2)
    if min(x.shape) < 1 or u.shape[0] != len(x) or u.shape[1] < 1:
        raise ValueError("nonempty features[N,D] and utility[N,M] must align")
    if np.any((u < 0) | (u > 1)):
        raise ValueError("training utility must lie in [0,1]")
    return x, u, _clusters(clusters, len(x)), _aliases(aliases, u.shape[1])


def _prior(state, clusters):
    ids = np.asarray(state["cluster_ids"], dtype=np.int64)
    table = np.asarray(state["cluster_prior"], dtype=np.float64)
    global_prior = np.asarray(state["global_prior"], dtype=np.float64)
    baseline = np.tile(global_prior, (len(clusters), 1))
    if state["method"] != "static_global":
        for i, cluster in enumerate(ids):
            baseline[clusters == cluster] = table[i]
    return baseline


def prediction_provenance(state, clusters):
    """Describe unseen-cluster fallback without reading any labels."""
    a = np.asarray(clusters)
    a = _clusters(a, len(a))
    unknown = ~np.isin(a, state["cluster_ids"])
    return dict(rows=len(a), unknown_cluster_ids=np.unique(a[unknown]).tolist(),
                unknown_cluster_rows=int(unknown.sum()),
                fallback_rows=int(unknown.sum()) if state["method"] != "static_global" else 0,
                unknown_cluster_policy="fit_global_prior; retain text correction if present")


def predict(state, features, clusters):
    """Predict relative qualities using text and cluster IDs only, without torch.

    Unknown clusters use this model's fit-only global prior. Call
    ``prediction_provenance`` to record their count during standalone deployment.
    States may contain NumPy arrays or their JSON-list equivalents.
    """
    if state.get("schema") != SCHEMA or state.get("method") not in METHODS:
        raise ValueError("unsupported quality-scoring state")
    x = _finite(features, "features", 2)
    if x.shape[1] != state["feature_dim"]:
        raise ValueError("prediction feature dimension differs from fitted state")
    c = _clusters(clusters, len(x))
    base = _prior(state, c)
    method = state["method"]
    if method in ("static_global", "static_cluster"):
        return base
    if method == "ridge":
        delta = (x * state["feature_scale"]) @ np.asarray(state["coef"], dtype=np.float64).T
    else:
        scaled = (x * state["feature_scale"]).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("features overflow float32 MLP inputs")
        hidden = np.maximum(scaled @ np.asarray(state["w1"], dtype=np.float32).T
                            + np.asarray(state["b1"], dtype=np.float32), 0)
        delta = hidden @ np.asarray(state["w2"], dtype=np.float32).T + np.asarray(state["b2"], dtype=np.float32)
    # Match MLP float32 centering before adding its float64 prior.
    delta = delta - delta.mean(1, keepdims=True)
    scores = base + delta
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite quality predictions")
    return scores


def _train_mlp(state, x, residual, config):
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("residual_mlp requires the optional torch dependency") from exc
    device = torch.device(config["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("residual_mlp supports cpu or cuda devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    scaled = (x * config["feature_scale"]).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("features overflow float32 MLP inputs")
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    losses = []
    try:
        torch.set_num_threads(config["threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        # Initialization is on CPU; preserve the caller's CPU RNG. No stochastic
        # GPU operations are used and no global CUDA seed needs to be changed.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config["seed"])
            model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], config["hidden_dim"]),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(config["hidden_dim"], residual.shape[1]))
            torch.nn.init.zeros_(model[-1].weight)
            torch.nn.init.zeros_(model[-1].bias)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        tensor_x = torch.as_tensor(scaled, device=device)
        tensor_y = torch.as_tensor(residual.astype(np.float32), device=device)
        for step in range(config["steps"]):
            optimizer.zero_grad()
            correction = model(tensor_x)
            correction = correction - correction.mean(1, keepdim=True)
            mse = (correction - tensor_y).square().mean()
            penalty = config["correction_l2"] * correction.square().mean()
            loss = mse + penalty
            if not torch.isfinite(loss):
                raise ValueError("nonfinite residual MLP training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(dict(step=step + 1, mse=float(mse.detach().cpu()),
                               correction_penalty=float(penalty.detach().cpu())))
        state.update(w1=model[0].weight.detach().cpu().numpy().copy(),
                     b1=model[0].bias.detach().cpu().numpy().copy(),
                     w2=model[2].weight.detach().cpu().numpy().copy(),
                     b2=model[2].bias.detach().cpu().numpy().copy())
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return losses


def fit(features, utility, clusters, aliases, *, method="residual_mlp", seed=42,
        steps=80, ridge_alpha=100., feature_scale=5., hidden_dim=32,
        correction_l2=.1, weight_decay=.01, learning_rate=.001, threads=4, device="cpu"):
    """Fit one model at a fixed budget; no validation or implicit epoch selection."""
    config = _configuration(method, seed, steps, ridge_alpha, feature_scale, hidden_dim,
                            correction_l2, weight_decay, learning_rate, threads, device)
    x, u, c, names = _fit_arrays(features, utility, clusters, aliases)
    r = relative(u)
    ids = np.unique(c)
    state = dict(schema=SCHEMA, method=method, aliases=names.tolist(), feature_dim=x.shape[1],
                 feature_scale=config["feature_scale"], cluster_ids=ids,
                 cluster_prior=np.stack([r[c == cluster].mean(0) for cluster in ids]),
                 global_prior=r.mean(0), configuration=config,
                 unknown_cluster_policy="fit_global_prior; retain text correction if present")
    residual = r - _prior(state, c)
    if method == "ridge":
        from sklearn.linear_model import Ridge
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=config["threads"]):
            estimator = Ridge(alpha=config["ridge_alpha"], fit_intercept=False, solver="cholesky")
            estimator.fit(x * config["feature_scale"], residual)
        state["coef"] = estimator.coef_.copy()
    elif method == "residual_mlp":
        state["training_history"] = _train_mlp(state, x, residual, config)
    state["training_rows"] = len(x)
    state["training_relative_mse"] = float(np.mean((predict(state, x, c) - r) ** 2))
    return state


def group_fold_ids(groups, folds=3, seed=42):
    """Return the declared deterministic complete-group OOF assignment.

    Sort unique group identifiers, permute with ``RandomState(seed)``, then
    assign whole groups cyclically to folds. This public helper allows artifact
    verification to reconstruct membership without fitting any model.
    """
    folds = _integer(folds, "folds", 2)
    seed = _integer(seed, "seed")
    if seed >= 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    g = np.asarray(groups)
    if g.ndim != 1 or not len(g) or g.dtype.kind not in "iuUS":
        raise ValueError("groups must be explicit integer or string identifiers")
    if g.dtype.kind in "US" and np.any(g.astype(str) == ""):
        raise ValueError("group identifiers must not be empty")
    unique = np.unique(g)
    if len(unique) < folds:
        raise ValueError("group OOF requires at least folds distinct groups")
    shuffled = np.random.RandomState(seed).permutation(unique)
    mapping = {group: i % folds for i, group in enumerate(shuffled)}
    return np.asarray([mapping[group] for group in g], dtype=np.int64)


def fit_predict(features, utility, clusters, aliases, *, method="residual_mlp", groups=None,
                query_features=None, query_clusters=None, folds=3, seed=42, steps=80,
                ridge_alpha=100., feature_scale=5., hidden_dim=32, correction_l2=.1,
                weight_decay=.01, learning_rate=.001, threads=4, device="cpu"):
    """Return group-OOF, full-fit and optional text-only query predictions.

    Group IDs are required: callers must supply their document connected groups,
    not infer independence from row order. All folds and the final fit use the
    same fixed steps; priors are estimated separately on each fit subset. A
    cluster absent from a fold's training subset falls back to that fold's global
    prior and is counted in provenance. ``fold_states`` retains each fold model,
    in fold order, so a verifier can replay held-row predictions without fitting.
    """
    config = _configuration(method, seed, steps, ridge_alpha, feature_scale, hidden_dim,
                            correction_l2, weight_decay, learning_rate, threads, device)
    x, u, c, names = _fit_arrays(features, utility, clusters, aliases)
    folds = _integer(folds, "folds", 2)
    g = np.asarray(groups)
    if g.shape != (len(x),) or g.dtype.kind not in "iuUS":
        raise ValueError("groups must be explicit row-aligned integer or string identifiers")
    fold_ids = group_fold_ids(g, folds=folds, seed=config["seed"])
    if (query_features is None) != (query_clusters is None):
        raise ValueError("query_features and query_clusters must be supplied together")
    qx = qc = None
    if query_features is not None:
        qx = _finite(query_features, "query_features", 2)
        if qx.shape[1] != x.shape[1]:
            raise ValueError("query feature dimension differs from fitting features")
        qc = _clusters(query_clusters, len(qx))
    oof = np.empty_like(u)
    receipts = []
    fold_states = []
    for fold in range(folds):
        held = fold_ids == fold
        train = ~held
        fold_options = dict(config, seed=(config["seed"] + 1009 * fold) % 2**32)
        state = fit(x[train], u[train], c[train], names, **fold_options)
        fold_states.append(state)
        oof[held] = predict(state, x[held], c[held])
        receipts.append(dict(fold=fold, fit_rows=int(train.sum()), held_rows=int(held.sum()),
                             fit_groups=len(np.unique(g[train])), held_groups=len(np.unique(g[held])),
                             group_overlap=0, seed=fold_options["seed"],
                             steps=config["steps"] if method == "residual_mlp" else 0,
                             prediction=prediction_provenance(state, c[held])))
    state = fit(x, u, c, names, **config)
    return dict(state=state, fold_states=fold_states, oof_scores=oof, fit_scores=predict(state, x, c),
                query_scores=None if qx is None else predict(state, qx, qc), fold_ids=fold_ids,
                provenance=dict(method=method, folds=receipts, configuration=config,
                                prior="corresponding fit subset only; row-centered utility without row-std scaling",
                                epoch_selection=False, fixed_budget_for_folds_and_main=True,
                                query=None if qc is None else prediction_provenance(state, qc)))
