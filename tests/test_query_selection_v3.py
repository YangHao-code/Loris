import numpy as np
import pytest

from loris.labeling.query_selection import (
    configuration, rank_review_candidates, review_candidates, train_label_cardinality,
)


def plan(pos, aux, *, mean=3.0, ratio=.7, gamma_pos=None,
         gamma_neg=None, unknown=None, query_rows=None):
    pos, aux = np.asarray(pos, bool), np.asarray(aux, float)
    z = np.zeros_like(pos)
    return review_candidates(
        positive=pos, negative=z, unknown=z if unknown is None else unknown,
        gamma_positive=z if gamma_pos is None else gamma_pos,
        gamma_negative=z if gamma_neg is None else gamma_neg,
        query_rows=np.ones(len(pos), bool) if query_rows is None else query_rows,
        aux_probabilities=aux, aux_thresholds=np.full(pos.shape[1], .5),
        train_mean=mean, ratio=ratio)


def test_cardinality_uses_only_complete_designated_training_rows():
    pos = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], bool)
    neg = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]], bool)
    result = train_label_cardinality(pos, neg, np.array([1, 1, 0], bool))
    assert result["mean"] == 2
    assert result["rows"] == 1
    assert result["incomplete_training_rows"] == 1
    with pytest.raises(ValueError, match="complete training"):
        train_label_cardinality(pos, neg, np.zeros(3, bool))


def test_strict_count_threshold_is_not_rounded_before_comparison():
    pos = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    p = plan(pos, pos, mean=3, ratio=2 / 3)
    assert p["parameters"]["strict_count_threshold"] == 2
    assert p["parameters"]["maximum_sparse_count"] == 1
    assert p["eligible"].tolist() == [True, False, False]
    p = plan(pos, pos, mean=3, ratio=.7)
    assert p["eligible"].tolist() == [True, True, False]


def test_over_labeled_document_still_selected_on_aux_disagreement():
    p = plan([[1, 1, 1], [1, 1, 1]], [[.01, .1, .9], [.9, .9, .9]])
    assert p["eligible"].tolist() == [True, False]
    assert p["reasons"]["disagreement"].tolist() == [True, False]


def test_complete_answers_and_non_query_roles_never_selected():
    pos = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    gp = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], bool)
    gn = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], bool)
    p = plan(pos, np.full((3, 3), .5), gamma_pos=gp, gamma_neg=gn,
             query_rows=np.array([1, 0, 1], bool))
    rows, _ = rank_review_candidates(p, limit=128, seed=1)
    assert rows == [2]


def test_unknown_is_reviewed_even_with_sufficient_positive_labels():
    p = plan([[1, 1, 1]], [[.9, .9, .9]],
             unknown=np.array([[0, 1, 0]], bool))
    assert p["eligible"].tolist() == [True]
    assert p["reasons"]["conflict"].tolist() == [True]


def test_random_audit_covers_agreement_outside_review_domain_without_duplicates():
    pos = np.ones((16, 3), bool)
    p = plan(pos, np.full((16, 3), .9))
    assert not p["eligible"].any()
    rows, audit = rank_review_candidates(p, limit=8, seed=17, audit_fraction=.5)
    assert len(rows) == len(set(rows)) == 8
    assert audit["random_audit_rows"] == rows
    assert rows == rank_review_candidates(p, limit=8, seed=17, audit_fraction=.5)[0]


def test_configuration_is_optional_and_rejects_typos_and_bad_parameters():
    assert configuration({}) == dict(cardinality_ratio=.7, audit_fraction=.125,
                                    aux_default_threshold=.5, aux_thresholds={})
    assert configuration(dict(query=dict(review=dict(cardinality_ratio=1))))["cardinality_ratio"] == 1
    with pytest.raises(ValueError, match="unsupported"):
        configuration(dict(query=dict(review=dict(cardinality_ratioo=.5))))
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        configuration(dict(query=dict(review=dict(audit_fraction=2))))
    with pytest.raises(ValueError, match="number"):
        configuration(dict(query=dict(review=dict(audit_fraction=True))))


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_independent_aux_threshold_configuration_is_validated(threshold):
    with pytest.raises(ValueError, match="finite"):
        configuration(dict(query=dict(review=dict(aux_thresholds={"A": threshold}))))


def test_independent_aux_thresholds_preserve_label_axis_names():
    value = dict(query=dict(review=dict(aux_default_threshold=.4, aux_thresholds={"A": .8})))
    assert configuration(value)["aux_thresholds"] == {"A": .8}


def test_zero_budget_and_zero_mean_are_defined():
    p = plan([[0, 0]], [[0, 0]], mean=0)
    assert p["eligible"].tolist() == [True]
    assert p["parameters"]["maximum_sparse_count"] == -1
    assert rank_review_candidates(p, limit=0, seed=1)[0] == []


def test_zero_audit_fraction_disables_sampling_outside_risk_gates():
    p = plan([[1, 1, 1]], [[.9, .9, .9]])
    assert rank_review_candidates(p, limit=1, seed=1, audit_fraction=0)[0] == []


def test_invalid_axes_and_probabilities_fail_before_ranking():
    with pytest.raises(ValueError, match="Aux probabilities"):
        plan([[0, 0]], [[-1, 0]])
    with pytest.raises(ValueError, match="positive"):
        plan([[0, 0]], [[0, 0]], ratio=0)
    with pytest.raises(ValueError, match="integer"):
        rank_review_candidates(plan([[0, 0]], [[0, 0]]), limit=1.5, seed=1)
