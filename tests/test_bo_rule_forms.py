"""Real BO construction/scoring/admission of the third, relation-headed form.

Controlled ordinary startup configurations make relation directions observable
without relying on the random sampler to happen to propose a specific rule.
They still use the production compact codec, Optuna transaction, frozen Chase,
standalone scores, marginal admission and authenticated completed-run resume.
"""

import json

import numpy as np
import optuna
import pytest

from loris._engine.compact_storage import ParameterCodec
from loris._engine.rdl_reference import AttrEqual, Label, LabelCompare, Rule
from loris._engine.rlearner_config import InvalidConfiguration, PatternSpec
from loris.discovery.runner import run_discovery
from loris.discovery.search.admission import admit_ordered_cluster
from loris.discovery.search.configuration import AuthorFreeCombinationSpace, build_config_space
from loris.predicates import MatchPredicate
from loris.rule_forms import AuthorRuleFormError, REJECTION, parse_author_rule
from tests.test_discovery_runner import inputs


def relation_parameters(space, operation, pair="x|y"):
    parameters = space.empty_parameters(operation, pair=pair)
    parameters.update({"p1.x.include": 1, "p2.y.include": 1, "r0.include": 1})
    return parameters


def relation_inputs():
    kwargs = inputs()
    kwargs["patterns"]["0"].extend([
        PatternSpec("source0", MatchPredicate("ttl", "^0$")),
        PatternSpec("target4", MatchPredicate("ttl", "^4$")),
    ])
    kwargs.update(initializer_slots=0, budget_per_cluster=1)
    return kwargs


@pytest.mark.parametrize("operation", ["subset", "eq", "strict_subset"])
@pytest.mark.parametrize("pair", ["x|y", "y|x"])
def test_real_BO_scores_and_admits_legal_relation_head_without_Label_body(
    tmp_path, monkeypatch, operation, pair,
):
    def controlled_start(self, count, *, seed, **kwargs):
        assert count == 1
        yield relation_parameters(self, operation, pair)

    monkeypatch.setattr(AuthorFreeCombinationSpace, "iter_initial_configurations", controlled_start)
    kwargs = relation_inputs()
    output = tmp_path / "bo"
    complete = run_discovery(output=output, **kwargs)
    folder = output / "learning/cluster_00"
    trial = json.loads((folder / "trial_00000.json").read_text())
    rule = parse_author_rule(trial["rule"])
    assert isinstance(rule.head, LabelCompare) and rule.head.op == operation
    assert not any(isinstance(atom, Label) for atom in rule.body)
    assert (rule.head.left_var, rule.head.right_var) == tuple(pair.split("|"))
    assert trial["configuration"]["admissible"]
    assert trial["standalone"]["evaluation"]["status"] == "fixpoint"
    admission = json.loads((folder / "admission/candidate_00000.json").read_text())
    transmits = pair == "x|y" or operation == "eq"
    assert admission["decision"]["accepted"] is transmits
    assert complete["selected_rules"] == int(transmits)
    assert complete["rule_form_rejected_candidates"] == 0
    assert complete["rule_form_audit"]["invalid_count"] == 0
    assert trial["standalone"]["score"] == (0.5 if transmits else 0.0)
    with np.load(output / "FACTS.npz") as facts:
        assert bool(facts["pos"][4, 0]) is transmits
    # A completed legal relation program is loadable unchanged. No keyword
    # gate rejects LabelCompare or silently removes the selected rule.
    saved = {p: p.read_bytes() for p in (output / "COMPLETE.json", folder / "trial_00000.json",
                                        folder / "optuna.sqlite")}
    assert run_discovery(output=output, resume=True, **kwargs) == complete
    assert all(path.read_bytes() == body for path, body in saved.items())


@pytest.mark.parametrize("operation", ["subset", "eq", "strict_subset"])
def test_actual_optuna_suggest_and_compact_codec_preserve_relation_rule(operation):
    kwargs = relation_inputs()
    space = build_config_space(cluster_id="0", labels=kwargs["context"]["labels"],
                               patterns=kwargs["patterns"]["0"], models=(),
                               relations=kwargs["relations"])
    parameters = relation_parameters(space, operation)
    proposed = space.suggest(optuna.trial.FixedTrial(parameters))
    codec = ParameterCodec.from_space(space)
    assert codec.decode(codec.encode(proposed)) == proposed == parameters
    receipt = space.configuration_receipt(proposed)
    assert receipt["admissible"] and receipt["rule"] == space.instantiate(parameters).to_dict()
    # The same cross body cannot be transferred to a single-label head unless
    # an actual Label premise is present; model/text atoms are not substitutes.
    single = space.empty_parameters("add", label="A", variable="y")
    single.update({key: value for key, value in parameters.items() if not key.startswith("head.")})
    with pytest.raises(InvalidConfiguration, match=REJECTION):
        space.instantiate(single)
    single["l.x.k1.neg.include"] = 1
    assert isinstance(space.instantiate(single).head, Label)


def test_completed_trial_admission_rechecks_loaded_cross_rule_without_silent_filter(tmp_path):
    # The admission entry is a second guard for cached/deserialized rules, not
    # only a condition checked by the parameter factory for newly drawn rules.
    from loris._engine.rdl_reference import Doc

    invalid = Rule((Doc("x"), Doc("y"), AttrEqual("x", "mtd", "y", "mtd")), Label("y", "A"))
    raw = invalid.to_dict()
    record = dict(number=0, rule=raw, standalone=dict(score=1.0))
    with pytest.raises(AuthorRuleFormError, match="invalid rule indices"):
        admit_ordered_cluster(order="standalone_score_desc", directory=tmp_path,
                              records=[record], selected=[], evaluator=None,
                              rule_parser=parse_author_rule)
    audit = json.loads((tmp_path / "RULE_FORM_AUDIT.json").read_text())
    assert audit["invalid_count"] == 1 and audit["rules_dropped"] == 0
    assert record["rule"] == raw
    assert not list(tmp_path.glob("candidate_*.json"))
