"""Explicit temporary arbitration does not turn ordinary missing labels negative."""

import copy
import json

import numpy as np
import pytest

from loris.labeling.arbitration import (
    SCHEMA,
    AmbiguousArbitrationError,
    ArbitrationRule,
    arbitrate,
    paired_rules,
)


def test_roundtrip_declares_both_decisions_and_signs_without_an_implicit_threshold():
    rules = paired_rules("Aux", "A", 0.37)
    records = json.loads(json.dumps([rule.to_dict() for rule in rules]))
    assert tuple(ArbitrationRule.from_dict(record) for record in records) == rules
    assert records[0]["schema"] == SCHEMA
    assert records[0]["body"]["model_decision"]["value"] is True
    assert records[1]["body"]["model_decision"]["value"] is False
    assert records[0]["head"]["polarity"] == "pos"
    assert records[1]["head"]["polarity"] == "neg"
    with pytest.raises(TypeError):
        paired_rules("Aux", "A")


@pytest.mark.parametrize("field", ["guard", "threshold", "sign", "schema", "extra"])
def test_serialized_rules_cannot_hide_the_guard_threshold_or_negative_decision(field):
    record = paired_rules("Aux", "A", 0.5)[1].to_dict()
    if field == "guard":
        record["body"]["conflict_unknown"] = False
    elif field == "threshold":
        del record["body"]["model_decision"]["kappa"]
    elif field == "sign":
        record["head"]["polarity"] = "pos"
    elif field == "schema":
        record["schema"] = "ordinary_ML"
    else:
        record["default_threshold"] = 0.5
    with pytest.raises(ValueError):
        ArbitrationRule.from_dict(record)


def test_only_conflict_pairs_fire_and_other_labels_same_document_stay_untouched():
    unknown = np.array([[True, False, True], [False, True, False]])
    table = np.array([[0.8, 0.01, 0.2], [0.1, 0.4, 0.8]])
    before_mask, before_table = unknown.copy(), table.copy()
    rules = paired_rules("Aux", "A", 0.6) + paired_rules("Aux", "B", 0.6)
    result = arbitrate(rules, {"Aux": table}, unknown, ["A", "B", "C"], model_state="pre")
    assert np.array_equal(result.positive, [[True, False, False], [False, False, False]])
    assert np.array_equal(result.negative, [[False, False, False], [False, True, False]])
    assert np.array_equal(result.remaining_unknown, [[False, False, True], [False, False, False]])
    assert np.array_equal(unknown, before_mask) and np.array_equal(table, before_table)
    assert result.audit["training_calls"] == result.audit["model_calls"] == 0
    assert result.audit["writes_Gamma"] is False
    assert result.audit["remaining_unknown_pairs"] == 1
    assert {(x["row"], x["label"], x["polarity"]) for x in result.evidence} == {
        (0, "A", "pos"),
        (1, "B", "neg"),
    }


def test_float64_threshold_equality_and_strict_complement_are_exact():
    values = np.array([[np.nextafter(0.5, 0.0)], [0.5], [np.nextafter(0.5, 1.0)]])
    result = arbitrate(
        paired_rules("Aux", "A", 0.5),
        {"Aux": values},
        np.ones((3, 1), bool),
        ["A"],
        model_state="post",
        model_version="candidate-7",
    )
    assert result.positive[:, 0].tolist() == [False, True, True]
    assert result.negative[:, 0].tolist() == [True, False, False]
    assert not result.remaining_unknown.any()
    assert [source["probability"] for source in result.evidence] == values[:, 0].tolist()
    assert all(source["model_version"] == "candidate-7" for source in result.evidence)
    assert result.audit["frozen_probability_tables"]["Aux"]["dtype"] == "float64"


@pytest.mark.parametrize("threshold, expected", [(0.0, [True, True]), (1.0, [False, True])])
def test_endpoint_thresholds_use_the_declared_complement(threshold, expected):
    result = arbitrate(
        paired_rules("Aux", "A", threshold),
        {"Aux": np.array([[0.0], [1.0]])},
        np.ones((2, 1), bool),
        ["A"],
        model_state="pre",
    )
    assert result.positive[:, 0].tolist() == expected
    assert np.array_equal(result.negative, ~result.positive)


def test_same_sign_multiple_models_and_thresholds_keep_all_sources_order_independently():
    rules = (
        paired_rules("Aux", "A", 0.3)
        + paired_rules("Aux", "A", 0.7)
        + paired_rules("Other", "A", 0.6)
    )
    tables = {"Aux": np.array([[0.8]]), "Other": np.array([[0.9]])}
    first = arbitrate(rules, tables, np.ones((1, 1), bool), ["A"], model_state="post")
    reverse = arbitrate(rules[::-1], tables, np.ones((1, 1), bool), ["A"], model_state="post")
    assert first.positive[0, 0] and not first.negative[0, 0]
    assert len(first.evidence) == 3
    assert first.evidence == reverse.evidence and first.audit == reverse.audit
    assert len({source["rule_id"] for source in first.evidence}) == 3


@pytest.mark.parametrize("other_model", [False, True])
def test_opposite_model_or_threshold_decisions_reject_all_results_with_evidence(other_model):
    alias = "Other" if other_model else "Aux"
    rules = paired_rules("Aux", "A", 0.3) + paired_rules(alias, "A", 0.7)
    tables = {"Aux": np.array([[0.5]]), "Other": np.array([[0.5]])}
    diagnostics = []
    for order in (rules, rules[::-1]):
        with pytest.raises(AmbiguousArbitrationError, match="ambiguous") as error:
            arbitrate(order, tables, np.ones((1, 1), bool), ["A"], model_state="pre")
        diagnostics.append(error.value.conflicts)
    assert diagnostics[0] == diagnostics[1]
    assert {source["polarity"] for source in diagnostics[0][0]["sources"]} == {"pos", "neg"}


def test_no_rules_retains_input_unknown_and_empty_guard_does_not_create_conflicts():
    mask = np.array([[True, False]])
    result = arbitrate([], {}, mask, ["A", "B"], model_state="pre")
    assert np.array_equal(result.remaining_unknown, mask)
    assert not (result.positive | result.negative).any() and not result.evidence
    contradictory = paired_rules("Aux", "A", 0.2) + paired_rules("Aux", "A", 0.8)
    empty = arbitrate(
        contradictory,
        {"Aux": np.array([[0.5, 0.5]])},
        np.zeros((1, 2), bool),
        ["A", "B"],
        model_state="pre",
    )
    assert not empty.evidence


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1, True, "0.5"])
def test_invalid_or_implicit_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="kappa"):
        paired_rules("Aux", "A", value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_bad_probabilities_rejected_even_outside_active_conflict_mask(value):
    with pytest.raises(ValueError, match="probabilities"):
        arbitrate(
            paired_rules("Aux", "A", 0.5),
            {"Aux": np.array([[0.8], [value]])},
            np.array([[True], [False]]),
            ["A"],
            model_state="pre",
        )


@pytest.mark.parametrize("fault", ["missing_table", "shape", "mask", "label", "pair", "duplicate"])
def test_missing_or_wrong_bindings_never_invent_a_probability_or_threshold(fault):
    rules, tables = paired_rules("Aux", "A", 0.5), {"Aux": np.array([[0.2]])}
    mask, labels = np.ones((1, 1), bool), ["A"]
    if fault == "missing_table":
        tables = {}
    elif fault == "shape":
        tables["Aux"] = np.array([[0.2, 0.8]])
    elif fault == "mask":
        mask = np.ones((1, 1), int)
    elif fault == "label":
        labels = ["B"]
    elif fault == "pair":
        rules = rules[:1]
    else:
        rules = rules + rules
    with pytest.raises(ValueError):
        arbitrate(rules, tables, mask, labels, model_state="pre")


def test_pre_and_post_are_separate_frozen_snapshots_and_outputs_cannot_be_mutated():
    rules = paired_rules("Aux", "A", 0.6)
    serialized = copy.deepcopy([rule.to_dict() for rule in rules])
    tables, mask = {"Aux": np.array([[0.2]])}, np.ones((1, 1), bool)
    before = arbitrate(serialized, tables, mask, ["A"], model_state="pre", model_version="v0")
    tables["Aux"][0, 0] = 0.8
    after = arbitrate(serialized, tables, mask, ["A"], model_state="post", model_version="v1")
    assert before.negative[0, 0] and after.positive[0, 0]
    assert before.evidence[0]["probability"] == 0.2
    assert before.evidence[0]["table_sha256"] != after.evidence[0]["table_sha256"]
    assert serialized == [rule.to_dict() for rule in rules]
    for array in (before.positive, before.negative, before.remaining_unknown):
        with pytest.raises(ValueError):
            array[0, 0] = False
    json.dumps(before.audit, allow_nan=False)
    json.dumps(before.evidence, allow_nan=False)
