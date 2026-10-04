"""Synthetic checks of the portable text-quality scoring contract."""

import builtins
import itertools
import json

import numpy as np
import pytest

from loris.discovery import quality_scoring as qs


def fixture():
    rng = np.random.RandomState(17)
    x = rng.normal(size=(18, 5))
    utility = rng.uniform(.1, .95, size=(18, 4))
    clusters = np.repeat([0, 1, 2], 6)
    groups = np.repeat(np.arange(9), 2)
    return x, utility, clusters, np.array(["d", "b", "a", "c"]), groups


def test_utility_uses_only_nonempty_true_sign_groups_and_preserves_endpoints():
    y = np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]])
    p = np.array([[[.8, .2, .4], [1, 1, 1], [0, 0, 0]],
                  [[.2, .9, .7], [0, 0, 0], [1, 1, 1]]])
    expected = np.array([[1 - (.04 + (.04 + .16) / 2) / 2,
                          1 - (.64 + (.81 + .49) / 2) / 2], [1, 0], [1, 0]])
    np.testing.assert_allclose(qs.balanced_squared_utility(p, y), expected)
    np.testing.assert_allclose(qs.balanced_squared_utility(np.array([[[.9, .1]],
        [[.6, .4]], [[.99, .99]]]), [[1, 0]]), [[.99, .84, .5099]])


@pytest.mark.parametrize("p,y", [(np.zeros((2, 2, 3)), [[1, 0, 0]]),
    (np.ones((1, 1, 1)) * 1.1, [[1]]), (np.zeros((1, 1, 1)), [[2]]),
    (np.zeros((1, 1, 0)), np.zeros((1, 0))), (np.full((1, 1, 1), np.nan), [[1]])])
def test_invalid_utility_arrays_are_rejected(p, y):
    with pytest.raises(ValueError):
        qs.balanced_squared_utility(p, y)


def test_threshold_scores_match_existing_clipped_logit_and_float32_convention():
    raw = np.array([[[0, .3, 1], [1e-10, .6, .9]],
                    [[.8, .7, .2], [1, 0, .5]]])
    thresholds = np.array([[0, .3, 1], [.7, .1, .4]])
    cp = np.clip(raw, 1e-7, 1 - 1e-7)
    ct = np.clip(thresholds, 1e-7, 1 - 1e-7)
    z = np.log(cp / (1 - cp)) - np.log(ct / (1 - ct))[:, None, :]
    expected = (1 / (1 + np.exp(-z))).astype(np.float32)
    expected[raw == 0], expected[raw == 1] = 0, 1
    actual = qs.threshold_scores(raw, thresholds)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert actual[0, 0, 0] == 0  # Preserve raw endpoint; not the raw>=0 decision.
    assert actual[0, 0, 1] == .5
    with pytest.raises(ValueError):
        qs.threshold_scores(raw, thresholds[:, :2])


def test_relative_centers_without_standardizing_the_gap():
    u = np.array([[.9, .8, .7], [.99, .98, .97]])
    r = qs.relative(u)
    np.testing.assert_allclose(r.sum(1), 0, atol=1e-15)
    np.testing.assert_allclose(r[0], 10 * r[1], atol=1e-15)


@pytest.mark.parametrize("method", ["frequency", "predicted_max"])
def test_candidate_ties_and_model_axis_permutation_are_canonical(method):
    aliases = np.array(["d", "b", "a", "c"])
    scores = np.array([[1, 1, 1, 1], [1, 1, 1, 1.]])
    clusters = np.array([0, 0])
    assert qs.aggregate_candidates(scores, clusters, aliases, 2, method) == {"0": ["a", "b"]}
    permutation = np.array([2, 3, 0, 1])
    assert qs.aggregate_candidates(scores[:, permutation], clusters, aliases[permutation], 2, method) == {"0": ["a", "b"]}


def test_predicted_max_uses_coverage_then_member_mean_and_is_row_shift_invariant():
    aliases = ["a", "b", "c", "d"]
    scores = np.array([[1, .8, .2, 0], [1, .7, .9, 0.]])
    # Every pair containing a has maximum coverage. b wins the secondary mean.
    assert qs.aggregate_candidates(scores, [0, 0], aliases, 2, "predicted_max") == {"0": ["a", "b"]}
    shifted = scores + np.array([[10], [-7]])
    assert qs.aggregate_candidates(shifted, [0, 0], aliases, 2, "predicted_max") == {"0": ["a", "b"]}
    coverage = np.array([[1, .9, 0], [0, .9, 1.]])
    assert qs.aggregate_candidates(coverage, [0, 0], ["a", "b", "c"], 2, "predicted_max") == {"0": ["a", "c"]}


def test_predicted_max_agrees_with_independent_small_enumeration_and_limit():
    rng = np.random.RandomState(6)
    scores = rng.normal(size=(27, 8))
    aliases = np.array(list("hgfedcba"))
    choices = list(itertools.combinations(sorted(aliases), 3))
    by_name = {a: i for i, a in enumerate(aliases)}
    values = [np.mean([max(row[by_name[a]] for a in choice) for row in scores]) for choice in choices]
    expected = list(choices[int(np.argmax(values))])
    assert qs.aggregate_candidates(scores, np.zeros(27, int), aliases, 3, "predicted_max") == {"0": expected}
    with pytest.raises(ValueError, match="exceeding limit"):
        qs.aggregate_candidates(scores, np.zeros(27, int), aliases, 3, "predicted_max", 2)


@pytest.mark.parametrize("method", ["static_global", "static_cluster", "ridge"])
def test_group_oof_replays_independent_fold_fits_and_never_uses_held_targets(method):
    x, u, c, aliases, groups = fixture()
    result = qs.fit_predict(x, u, c, aliases, method=method, groups=groups,
                            query_features=x[:2], query_clusters=[99, 99])
    assert result["provenance"]["query"]["unknown_cluster_rows"] == 2
    np.testing.assert_array_equal(result["fold_ids"], qs.group_fold_ids(groups, folds=3, seed=42))
    assert len(result["fold_states"]) == 3
    for fold in range(3):
        held = result["fold_ids"] == fold
        assert not set(groups[held]) & set(groups[~held])
        state = qs.fit(x[~held], u[~held], c[~held], aliases, method=method)
        np.testing.assert_allclose(result["oof_scores"][held], qs.predict(state, x[held], c[held]), atol=1e-12)
        np.testing.assert_array_equal(result["oof_scores"][held],
                                      qs.predict(result["fold_states"][fold], x[held], c[held]))
    held = result["fold_ids"] == 1
    perturbed = u.copy()
    perturbed[held] = 1 - perturbed[held]
    rerun = qs.fit_predict(x, perturbed, c, aliases, method=method, groups=groups)
    np.testing.assert_array_equal(result["oof_scores"][held], rerun["oof_scores"][held])


def test_missing_fold_clusters_use_only_that_folds_global_prior():
    x, u, _, aliases, groups = fixture()
    # Each cluster is also exactly one connected group, absent when held out.
    result = qs.fit_predict(x, u, groups, aliases, method="static_cluster", groups=groups)
    assert sum(f["prediction"]["fallback_rows"] for f in result["provenance"]["folds"]) == len(x)
    for fold in range(3):
        held = result["fold_ids"] == fold
        expected = qs.relative(u[~held]).mean(0)
        np.testing.assert_allclose(result["oof_scores"][held], np.tile(expected, (held.sum(), 1)))


def test_group_and_feature_contracts_fail_before_fitting():
    x, u, c, aliases, groups = fixture()
    with pytest.raises(ValueError, match="explicit"):
        qs.fit_predict(x, u, c, aliases, method="static_cluster")
    with pytest.raises(ValueError, match="distinct"):
        qs.fit_predict(x, u, c, aliases, method="static_cluster", groups=np.zeros(len(x), int))
    with pytest.raises(ValueError, match="nonnegative integers"):
        qs.fit(x, u, c.astype(float), aliases, method="static_cluster")
    with pytest.raises(ValueError, match="together"):
        qs.fit_predict(x, u, c, aliases, method="static_cluster", groups=groups, query_features=x)


def test_group_fold_helper_matches_declared_formula_and_rejects_empty_group_ids():
    groups = np.array(["z", "a", "a", "c", "b", "z", "e", "d"])
    shuffled = np.random.RandomState(19).permutation(np.unique(groups))
    expected = np.array([next(i % 3 for i, group in enumerate(shuffled) if group == value)
                         for value in groups])
    np.testing.assert_array_equal(qs.group_fold_ids(groups, 3, 19), expected)
    for empty in [np.array(["", "a", "b"]), np.array([b"", b"a", b"b"])]:
        with pytest.raises(ValueError, match="empty"):
            qs.group_fold_ids(empty)


@pytest.mark.parametrize("steps", [0, 3])
def test_mlp_numpy_deployment_and_serialization_need_no_torch_import(steps, monkeypatch):
    torch = pytest.importorskip("torch")
    x, u, c, aliases, groups = fixture()
    result = qs.fit_predict(x, u, c, aliases, method="residual_mlp", groups=groups,
                            steps=steps, hidden_dim=7, threads=1)
    state = result["state"]
    assert all(r["steps"] == steps for r in result["provenance"]["folds"])
    assert state["configuration"]["steps"] == steps
    assert state["w1"].shape == (7, 5)
    if steps == 0:
        np.testing.assert_array_equal(state["w2"], 0)
        for cluster in np.unique(c):
            np.testing.assert_allclose(result["fit_scores"][c == cluster],
                np.tile(qs.relative(u[c == cluster]).mean(0), ((c == cluster).sum(), 1)))
    else:
        assert np.any(state["w2"] != 0)
    # A separate torch forward checks the exported array orientation and the
    # centering order; deployment itself must remain independent of torch.
    xx = torch.tensor((x * state["feature_scale"]).astype(np.float32))
    hidden = torch.relu(torch.nn.functional.linear(xx, torch.tensor(state["w1"]),
                                                   torch.tensor(state["b1"])))
    delta = torch.nn.functional.linear(hidden, torch.tensor(state["w2"]),
                                       torch.tensor(state["b2"]))
    delta -= delta.mean(1, keepdim=True)
    base = np.stack([state["cluster_prior"][np.flatnonzero(state["cluster_ids"] == ci)[0]]
                     for ci in c])
    np.testing.assert_allclose(result["fit_scores"], base + delta.numpy(), atol=2e-7)
    serialized = json.loads(json.dumps(state, default=lambda a: a.tolist()))
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("deployment must not import torch")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    np.testing.assert_array_equal(qs.predict(serialized, x, c), result["fit_scores"])
    assert qs.predict(serialized, np.empty((0, 5)), np.array([], int)).shape == (0, 4)


def test_mlp_held_group_targets_do_not_affect_their_oof_predictions():
    pytest.importorskip("torch")
    x, u, c, aliases, groups = fixture()
    kwargs = dict(method="residual_mlp", groups=groups, steps=2, hidden_dim=5, threads=1)
    initial = qs.fit_predict(x, u, c, aliases, **kwargs)
    held = initial["fold_ids"] == 0
    changed = u.copy()
    changed[held] = 1 - changed[held]
    modified = qs.fit_predict(x, changed, c, aliases, **kwargs)
    np.testing.assert_array_equal(initial["oof_scores"][held], modified["oof_scores"][held])
    for fold, state in enumerate(initial["fold_states"]):
        held = initial["fold_ids"] == fold
        np.testing.assert_array_equal(initial["oof_scores"][held], qs.predict(state, x[held], c[held]))
