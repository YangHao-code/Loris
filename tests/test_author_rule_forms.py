"""Public author rule-form gate and explicitly accounted search rejection."""

import copy
import json

import pytest

from loris._engine.rdl_reference import (
    Doc,
    AttrEqual,
    Label,
    LabelCompare,
    ML,
    Rule,
    ReferenceChase,
    RefGamma,
)
from loris._engine.rlearner_config import InvalidConfiguration
from loris.discovery.manifest import file_record
from loris.discovery.runner import run_discovery
from loris.discovery.search.configuration import (
    AuthorFreeCombinationSpace,
    build_config_space,
    SCHEMA,
)
from loris.discovery.search.random_joint import RandomJointSpace
from loris.labeling import cli
from loris.document import Document
from loris.rule_forms import (
    AuthorRuleFormError,
    POLICY,
    REJECTION,
    audit_program_rule_forms,
    validate_author_rule_form,
    validate_program_rule_forms,
)
from tests.test_labeling_reliability_weights import runtime_manifest
from tests.test_discovery_runner import inputs


def cross_rule(head=None, premise=(), relation=None):
    return Rule(
        (Doc("x"), Doc("y"), relation or AttrEqual("x", "cnt", "y", "cnt")) + tuple(premise),
        head or Label("y", "A"),
    )


@pytest.mark.parametrize(
    "relation",
    [
        AttrEqual("x", "cnt", "y", "ttl"),
        LabelCompare("x", "eq", "y"),
        LabelCompare("x", "subset", "y"),
        LabelCompare("x", "strict_subset", "y"),
    ],
)
@pytest.mark.parametrize("sign", ["pos", "neg"])
def test_cross_single_label_head_needs_explicit_Label_body(relation, sign):
    invalid = cross_rule(Label("y", "A", sign), relation=relation)
    with pytest.raises(AuthorRuleFormError, match="cross_single_label_head"):
        validate_author_rule_form(invalid)
    # ML is a model predicate, not an observed Label premise.
    with pytest.raises(AuthorRuleFormError):
        validate_author_rule_form(
            cross_rule(Label("y", "A", sign), [ML("x", "m", "B", 0.5)], relation)
        )
    for source in ("x", "y"):
        # Author's minimum restriction: any explicit Label, not necessarily
        # the target label, matching sign, or the other variable.
        legal = cross_rule(Label("y", "A", sign), [Label(source, "B", "neg")], relation)
        assert validate_author_rule_form(legal) is legal


@pytest.mark.parametrize("op", ["eq", "subset", "strict_subset"])
def test_cross_relation_head_without_Label_is_valid_collection_propagation(op):
    rule = cross_rule(LabelCompare("x", op, "y"))
    assert validate_program_rule_forms([rule])["invalid_count"] == 0


@pytest.mark.parametrize(
    "relation", [AttrEqual("x", "cnt", "x", "ttl"), LabelCompare("x", "eq", "x")]
)
def test_same_variable_comparison_is_not_cross_and_generic_history_remains_readable(relation):
    rule = Rule((Doc("x"), relation), Label("x", "A"))
    assert audit_program_rule_forms([rule])["cross_rules"] == 0
    assert validate_author_rule_form(rule) is rule
    historical = cross_rule()
    parsed = Rule.from_dict(historical.to_dict())
    result = ReferenceChase(["A"]).run(
        [Document(cnt="same"), Document(cnt="same")], (parsed,), RefGamma.empty(2, 1)
    )
    assert result.status == "fixpoint" and result.positive_facts.all()
    with pytest.raises(AuthorRuleFormError):
        validate_author_rule_form(parsed)


def test_whole_program_audit_preserves_original_indices_and_never_deletes():
    rules = [Rule((Doc("x"),), Label("x", "A"))] * 167
    rules[165] = cross_rule()
    records = [r.to_dict() for r in rules]
    snapshot = copy.deepcopy(records)
    with pytest.raises(AuthorRuleFormError, match=r"indices=\[165\]") as caught:
        validate_program_rule_forms(records)
    audit = caught.value.audit
    assert audit["rules_checked"] == 167 and audit["invalid_count"] == 1
    assert audit["rules_dropped"] == 0 and audit["violations"][0]["rule_index"] == 165
    assert records == snapshot


def test_public_labeling_cli_rejects_invalid_loaded_program_before_running(tmp_path):
    manifest, data = runtime_manifest(tmp_path)
    program = data["program"]["path"]
    with open(program, "w") as f:
        json.dump([cross_rule().to_dict()], f)
    data["program"] = file_record(program)
    manifest.write_text(json.dumps(data))
    with pytest.raises(AuthorRuleFormError, match="invalid rule indices"):
        cli.load(manifest)


def test_public_LabelingRun_constructor_rejects_before_output_or_trainer(tmp_path):
    from loris.labeling.runner import LabelingRun
    from tests.test_labeling import fixture

    context, _, manifest = fixture()
    calls = []

    def trainer(**kwargs):
        calls.append(kwargs)
        raise AssertionError("invalid public program must not invoke training")

    output = tmp_path / "run"
    with pytest.raises(AuthorRuleFormError, match="invalid rule indices"):
        LabelingRun(
            context=context,
            rules=(cross_rule(),),
            manifest=manifest,
            output=output,
            trainer=trainer,
        )
    assert not output.exists() and not calls


def test_both_public_search_spaces_reject_same_rule_and_keep_actual_trial_reason(tmp_path):
    space = build_config_space(
        cluster_id="0",
        labels=["A"],
        patterns=(),
        models=(),
        relations=(AttrEqual("x", "cnt", "y", "cnt"),),
    )
    params = space.empty_parameters("add", label="A", variable="y")
    params["r0.include"] = 1
    for wrapper in (space, RandomJointSpace(space, tmp_path / "random")):
        receipt = wrapper.configuration_receipt(params)
        assert not receipt["admissible"] and receipt["rule"] is None
        assert receipt["invalid_reason"].startswith(REJECTION)
        with pytest.raises(InvalidConfiguration):
            wrapper.instantiate(params)
    manifest = space.manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["author_combination_policy"]["public_rule_form"] == POLICY
    # New startup rejection is retained as a trial instead of silently adding
    # a Label premise or retrying this proposal.
    probability = {
        f.include_key: (1.0 - 1e-12 if f.kind == "relation" else 1e-12) for f in space.features
    }
    proposed = next(
        space.iter_initial_configurations(1, seed=42, feature_probabilities=probability)
    )
    assert not space.configuration_receipt(proposed)["admissible"]


def test_real_BO_invalid_author_form_consumes_trial_and_has_no_scoring_Chase(tmp_path, monkeypatch):
    original = AuthorFreeCombinationSpace.iter_initial_configurations

    def first_invalid(self, count, *, seed, **kwargs):
        params = self.empty_parameters("add", label="A", variable="y")
        params["r0.include"] = 1
        yield params
        yield from original(self, count - 1, seed=seed, **kwargs)

    monkeypatch.setattr(AuthorFreeCombinationSpace, "iter_initial_configurations", first_invalid)
    kw = inputs()
    kw.update(initializer_slots=0, budget_per_cluster=3)
    done = run_discovery(output=tmp_path / "bo", **kw)
    trial = json.loads((tmp_path / "bo/learning/cluster_00/trial_00000.json").read_text())
    assert trial["rule"] is None and trial["no_body_reason"].startswith(REJECTION)
    assert trial["standalone"]["evaluation"] is None and trial["standalone"]["score"] == -1
    assert done["complete_BO_trials"] == 3 and done["rule_form_rejected_candidates"] >= 1
    assert done["rule_form_audit"]["invalid_count"] == 0


def test_actual_random_joint_trials_use_same_gate_and_report_rejections(tmp_path):
    kw = inputs()
    kw.update(
        initializer_slots=0,
        budget_per_cluster=35,
        proposal_strategy="random_joint",
        admission_order="random_consistent",
    )
    done = run_discovery(output=tmp_path / "random", **kw)
    records = [
        json.loads(p.read_text())
        for p in sorted((tmp_path / "random/learning/cluster_00").glob("trial_*.json"))
    ]
    rejected = [r for r in records if str(r.get("no_body_reason", "")).startswith(REJECTION)]
    assert len(records) == 35 and rejected
    assert done["rule_form_rejected_candidates"] == len(rejected)
    assert all(r["rule"] is None and r["standalone"]["evaluation"] is None for r in rejected)
    assert (
        validate_program_rule_forms(r["rule"] for r in records if r["rule"] is not None)[
            "invalid_count"
        ]
        == 0
    )
