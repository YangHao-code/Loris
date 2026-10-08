"""Actual Chase/query-loop tests; annotations are explicit simulated fixtures."""
import json

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, Rule, Text
from loris.chase import GammaSeed
from loris.labeling.runner import LabelingRun
from loris.predicates import MatchPredicate
from tests.test_labeling import fixture


def configured(tmp_path, name="run", *, pool="needs_review"):
    ctx, rules, m = fixture()
    # Every query document already has A. The red document disagrees with Aux,
    # while the other documents agree. A fully answered source row supplies
    # the training-only cardinality reference.
    ctx["model_probabilities"]["aux"][2:, 0] = .9
    m["strategy"] = "frozen"
    m["query"] = dict(strategy="review", seed=42,
                      review=dict(cardinality_ratio=.7, audit_fraction=0))
    m["loop"].update(query_pool=pool, label_budget=1, query_batch_size=1,
                      query_cadence="each_closure", query_answer_mode="complete")
    answers = tmp_path / (name + "_answers")
    answers.mkdir()
    (answers / "00000001.json").write_text(json.dumps(dict(
        row_id="r1", labels=["A", "B"], targets=[1, 1],
        provenance="explicit simulated human test fixture")))
    m["oracle"].update(kind="directory_simulated_human", directory=str(answers))
    return ctx, rules, m


def forbidden_training(**kwargs):
    raise AssertionError("frozen query tests must not invoke a trainer")


def close(run):
    for runner in run.sessions:
        runner.close()


def test_review_queries_document_that_already_has_one_label(tmp_path):
    ctx, rules, m = configured(tmp_path)
    run = LabelingRun(context=ctx, rules=rules, manifest=m,
                      output=tmp_path / "run", trainer=forbidden_training)
    report = run.run()
    assert report["actual_label_requests"] == 1
    assert run.gamma.pos[1].tolist() == [True, True]
    assert report["annotation_sources"]["human_answers"] == 0
    assert report["annotation_sources"]["simulated_human_answers"] == 1
    event = next(e for e in run.events if e["kind"] == "query_batch_selected")
    assert event["rows"] == [1]
    # Query answers cannot contaminate the reference cardinality.
    assert run.review_reference["mean"] == 1
    assert run.review_reference["rows"] == 1


def test_misspelled_review_aux_threshold_label_is_rejected_before_chase(tmp_path):
    ctx, rules, manifest = configured(tmp_path)
    manifest["query"]["review"]["aux_thresholds"] = {"unseen_label": .7}
    with pytest.raises(ValueError, match="label axis"):
        LabelingRun(context=ctx, rules=rules, manifest=manifest,
                    output=tmp_path / "invalid", trainer=forbidden_training)


def test_legacy_uncovered_pool_does_not_query_already_positive_document(tmp_path):
    ctx, rules, m = configured(tmp_path, pool="uncovered")
    run = LabelingRun(context=ctx, rules=rules, manifest=m,
                      output=tmp_path / "run", trainer=forbidden_training)
    report = run.run()
    assert report["actual_label_requests"] == 0
    assert not (run.gamma.pos[1] | run.gamma.neg[1]).any()


def test_complete_trusted_answer_excluded_even_when_aux_disagrees(tmp_path):
    ctx, rules, m = configured(tmp_path)
    gp, gn = ctx["gamma"].pos.copy(), ctx["gamma"].neg.copy()
    gp[1, 0] = True
    gn[1, 1] = True
    ctx["gamma"] = GammaSeed(gp, gn)
    # The unanswered blue document also has A but Aux proposes an extra B.
    ctx["model_probabilities"]["aux"][2, 1] = .9
    run = LabelingRun(context=ctx, rules=rules, manifest=m,
                      output=tmp_path / "run", trainer=forbidden_training)
    try:
        result, session, _ = run.chase(run.version, phase="query_test")
        assert run.eligible_query_rows(result).tolist() == [False, False, True, False]
        assert run.rdg(result, session, limit=1) == 2
        assert run.review_reference["rows"] == 1
    finally:
        close(run)


def test_review_ignores_arbitrated_conflict_that_has_no_remaining_unknown(tmp_path):
    ctx, rules, m = configured(tmp_path)
    ctx["model_probabilities"]["aux"][1, 0] = .9
    m["loop"].update(conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences",
                      gamma_conflict_policy="validated_gamma_wins")
    m["arbitration"] = dict(alias="aux", default_threshold=.5, thresholds={})
    opposite = Rule((Doc("x"), Text("x", MatchPredicate("cnt", "red"))), Label("x", "A", "neg"))
    run = LabelingRun(context=ctx, rules=rules + (opposite,), manifest=m,
                      output=tmp_path / "run", trainer=forbidden_training)
    try:
        result, session, _ = run.chase(run.version, phase="query_test")
        assert result.conflict_unknown[1, 0]
        assert not result.remaining_unknown[1, 0]
        assert not run.review_plan(result)["reasons"]["conflict"][1]
        assert not run.eligible_query_rows(result)[1]
    finally:
        close(run)
