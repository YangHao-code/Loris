"""Sparse correction isolation, numerical boundaries and held-out fit scope."""

import copy
import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, ML, Rule
from loris._storage.discovery_factory import make_discovery_runner
from loris.labeling.aux_extension import (configuration, attach_context, extend_program,
    partition_audit, probability_views, channel_polarities, clamp_channels)
from loris.labeling.aux_correction import fit_correction
from loris.labeling.unknown_incremental import retaining_chase
from tests.test_labeling import fixture


def corrected(*, positive=()):
    context, rules, manifest = fixture()
    manifest["dynamic"]["alias"] = "adaptive"
    manifest["aux_extension"] = dict(source_alias="aux", alias="adaptive", initial_alias="fixed",
        default_threshold=.9, thresholds={}, correction=dict(complement_alias="adaptive_low",
        positive_active=list(positive), negative_rule_indices=[1],
        negative_default_threshold=.1, negative_thresholds={}, min_gap=.05))
    return context, rules, manifest


def test_separate_initial_model_does_not_overwrite_frozen_source_and_prefix():
    context, old, manifest = corrected()
    before = [r.to_dict() for r in old]
    ctx = attach_context(context, manifest)
    assert ctx["model_probabilities"]["adaptive"] is context["model_probabilities"]["fixed"]
    assert ctx["model_probabilities"]["aux"] is context["model_probabilities"]["aux"]
    assert "adaptive" not in context["model_probabilities"]
    rules = extend_program(old, manifest, ctx["labels"])
    assert [r.to_dict() for r in rules[:len(old)]] == before
    assert rules[-1].head.polarity == "neg" and rules[-1].body[:-1] == old[1].body
    audit = partition_audit(rules, manifest, ctx["labels"])
    assert audit["positive_rules"] == 0 and audit["negative_rules"] == 1
    assert audit["added_rules"] == 1 and audit["frozen_rules"] == len(old)
    assert channel_polarities(rules, manifest) == {(2, 2): -1}
    assert audit["derived_aliases"]["adaptive_low"]["source"] == "adaptive"


def test_empty_screened_suffix_and_tampered_negative_fail_closed():
    context, old, manifest = corrected()
    manifest["aux_extension"]["correction"]["negative_rule_indices"] = []
    assert extend_program(old, manifest, context["labels"]) == old
    assert partition_audit(old, manifest, context["labels"])["added_rules"] == 0
    _, old, manifest = corrected()
    rules = extend_program(old, manifest, context["labels"])
    changed = rules[:-1] + (Rule(rules[-1].body, Label("x", "B", "neg")),)
    with pytest.raises(ValueError, match="suffix"):
        partition_audit(changed, manifest, context["labels"])


def test_float32_low_boundary_uses_same_checkpoint_float64_complement():
    _, _, manifest = corrected()
    lo = float(np.float32(.1))
    values = np.array([[np.nextafter(np.float32(.1), np.float32(0)), .5],
                       [np.float32(.1), np.nextafter(np.float32(.1), np.float32(1))]], np.float32)
    tables = probability_views(manifest, values)
    np.testing.assert_array_equal(tables["adaptive_low"] >= 1. - lo, values.astype(np.float64) <= lo)
    assert tables["adaptive"] is values
    assert tables["adaptive_low"].dtype == np.float64 and not tables["adaptive_low"].flags.writeable
    changed = np.ones_like(values)
    updated = probability_views(manifest, changed)
    assert not updated["adaptive_low"].any()
    assert np.array_equal(values, tables["adaptive"])


@pytest.mark.parametrize("change", [
    {"negative_rule_indices": [1, 1]}, {"negative_rule_indices": [True]},
    {"negative_rule_indices": [1, 0]}, {"positive_active": ["A", "A"]},
    {"positive_active": ["unseen"]}, {"complement_alias": "aux"},
    {"negative_thresholds": {"A": .88}}, {"min_gap": 0},
])
def test_invalid_correction_schema_rejected(change):
    context, _, manifest = corrected()
    manifest["aux_extension"]["correction"].update(change)
    with pytest.raises(ValueError):
        configuration(manifest, context["labels"])


def test_label_dependent_context_and_out_of_range_index_rejected():
    context, _, manifest = corrected()
    old = (Rule((Doc("x"), Label("x", "B")), Label("x", "A")),)
    manifest["aux_extension"]["correction"]["negative_rule_indices"] = [0]
    with pytest.raises(ValueError, match="static unary"):
        extend_program(old, manifest, context["labels"])
    manifest["aux_extension"]["correction"]["negative_rule_indices"] = [1]
    with pytest.raises(ValueError, match="outside"):
        extend_program(old, manifest, context["labels"])


def test_corrective_negative_creates_unknown_but_cannot_override_trusted_gamma(tmp_path):
    context, old, manifest = corrected()
    context["model_probabilities"]["fixed"] = np.full((4, 2), .02, np.float32)
    ctx = attach_context(context, manifest)
    rules = extend_program(old, manifest, ctx["labels"])
    runner = make_discovery_runner(ctx, tmp_path / "cache", subset_direction="standard")
    try:
        closure = retaining_chase(runner, rules, ctx["gamma"], gamma_wins=True)
    finally:
        runner.close()
    assert closure.status == "fixpoint"
    assert closure.positive_facts[0, 0] and not closure.conflict_unknown[0, 0]
    assert closure.conflict_unknown[1, 0] and not closure.positive_facts[1, 0]
    assert not closure.negative_facts[2, 0]  # context "red" is required, not a global veto.
    assert np.array_equal(ctx["gamma"].pos, context["gamma"].pos)


def test_round_calibration_clamps_joint_gap_and_shares_negative_threshold():
    context, old, manifest = corrected(positive=["A"])
    manifest["aux_extension"]["correction"]["negative_rule_indices"] = [0, 1]
    rules = extend_program(old, manifest, context["labels"])
    channels = {(0, 1): (0, .8), (2, 1): (0, .01), (3, 2): (0, .1), (4, 2): (0, .7)}
    actual = clamp_channels(manifest, channels, rules)
    assert actual[(0, 1)] == channels[(0, 1)]  # old dynamic=True predicate remains frozen.
    assert actual[(2, 1)][1] == .05
    assert actual[(3, 2)][1] == actual[(4, 2)][1] == 1.
    assert channels[(2, 1)][1] == .01


def fit_arrays():
    labels = ["A", "B"]
    rules = (Rule((Doc("x"),), Label("x", "A")),)
    y = np.array([[1, 1], [1, 1], [0, 0], [0, 0], [1, 0], [0, 1]], bool)
    p = np.array([[.99, .99], [.99, .99], [.01, .01], [.01, .01], [.9, .01], [.01, .9]])
    fires = np.ones((1, 6), bool)
    mask = np.array([1, 1, 1, 1, 0, 0], bool)
    return dict(labels=labels, rules=rules, truth=y, probabilities=p,
                rule_fires=fires, fit_mask=mask, complement_alias="fitted_low", min_support=2)


def test_dev_fit_support_precision_gates_and_no_confirmation_gold_leak():
    args = fit_arrays()
    update, receipt = fit_correction(**args)
    assert update["correction"]["positive_active"] == ["B"]
    assert update["correction"]["negative_rule_indices"] == [0]
    assert receipt["fit_rows"] == 4 and receipt["confirmation_rows"] == 2
    assert receipt["actual_chase_confirmation_required"] and not receipt["truth_entered_chase"]
    args["truth"] = args["truth"].copy()
    args["truth"][~args["fit_mask"]] = ~args["truth"][~args["fit_mask"]]
    update_again, receipt_again = fit_correction(**args)
    assert update_again == update and receipt_again == receipt


def test_insufficient_support_does_not_invent_positive_or_negative_rules():
    args = fit_arrays()
    args["min_support"] = 10
    update, _ = fit_correction(**args)
    assert update["correction"]["positive_active"] == []
    assert update["correction"]["negative_rule_indices"] == []


def test_nonboolean_fit_mask_cannot_silently_index_integer_rows():
    args = fit_arrays()
    args["fit_mask"] = np.array([1, 1, 1, 1, 0, 0])
    with pytest.raises(ValueError, match="boolean fit_mask"):
        fit_correction(**args)
