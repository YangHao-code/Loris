"""Regression tests for the second 2026-10-01 review (LLM checks, check base, fallback binding)."""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, RefGamma, Rule, Text
from loris._engine.typed_runtime import IndexedChaseRunner
from loris.document import Document
from loris.labeling.runner import LabelingRun
from loris.labeling.unknown_incremental import retaining_chase
from loris.predicates import MatchPredicate
from tests.test_conflict_protocol_loop import close, forbidden_training, setup


def at(var, row):
    return Text(var, MatchPredicate("cnt", "^node%d$" % row))


def make(tmp_path, n, rules, policy="validated_gamma_wins", execution="incremental", gamma=None,
         check_base="batch_start"):
    ctx, manifest = setup()
    k = len(ctx["labels"])
    ctx.update(docs=[Document(cnt="node%d" % i, ttl="", mtd="") for i in range(n)],
               row_ids=["r%d" % i for i in range(n)], row_roles=["train"] * n,
               gamma=gamma if gamma is not None else RefGamma.empty(n, k),
               model_probabilities={"aux": np.full((n, k), 0.2, np.float32)},
               truth=np.empty((0, k), bool))
    manifest["loop"].update(unknown_semantics="retain_consequences", gamma_conflict_policy=policy,
                            execution=execution, label_budget=None, query_answer_mode="allow_partial",
                            llm_check_base=check_base)
    run = LabelingRun(context=ctx, rules=rules, manifest=manifest, output=tmp_path / "run",
                      trainer=forbidden_training)
    run.manifest["oracle"].update(kind="directory_llm_then_human", directory=str(tmp_path / "answers"))
    return run


def propose(run, tmp_path, row, targets, result, session):
    (tmp_path / "answers").mkdir(exist_ok=True)
    (tmp_path / "answers" / ("%08d.llm.json" % row)).write_text(json.dumps(dict(
        row_id="r%d" % row, labels=list(run.labels), targets=targets, provider="llm",
        provenance="review2 regression")))
    reasons = []
    original = run.event

    def event(kind, **kw):
        if kind == "LLM_proposal_needs_human":
            reasons.append(kw.get("reason"))
        return original(kind, **kw)

    run.event = event
    try:
        accepted = bool(run.oracle_answer(row, "review2", result=result, session=session, version=run.version))
    finally:
        run.event = original
    return accepted, reasons


def independent_case(tmp_path, execution, extra_rule=None):
    # Trusted Gamma: doc0 is not B. r1 derives B@0 on its own (blocked before any
    # proposal); r2 derives B@0 from A@0. A proposal A@0 adds a second path only.
    neg = np.zeros((1, 4), bool)
    neg[0, 1] = True
    rules = [Rule((Doc("x"), at("x", 0)), Label("x", "B")),
             Rule((Doc("x"), at("x", 0), Label("x", "A")), Label("x", "B"))]
    if extra_rule is not None:
        rules.append(extra_rule)
    return make(tmp_path, 1, tuple(rules), execution=execution, gamma=RefGamma(np.zeros((1, 4), bool), neg))


@pytest.mark.parametrize("execution", ["incremental", "full_recompute"])
def test_llm_check_decision_does_not_depend_on_execution(tmp_path, execution):
    run = independent_case(tmp_path, execution)
    try:
        result, session, _ = run.chase(run.version, phase="initial_zero_update")
        accepted, reasons = propose(run, tmp_path, 0, [1, None, None, None], result, session)
        assert accepted and not reasons
    finally:
        close(run)


@pytest.mark.parametrize("execution", ["incremental", "full_recompute"])
def test_a_new_contradiction_with_trusted_gamma_is_still_rejected_in_both_modes(tmp_path, execution):
    # A@0 -> D@0, and trusted Gamma says not D: only the proposal derives it.
    neg = np.zeros((1, 4), bool)
    neg[0, 3] = True
    rules = (Rule((Doc("x"), at("x", 0), Label("x", "A")), Label("x", "D")),)
    run = make(tmp_path, 1, rules, execution=execution, gamma=RefGamma(np.zeros((1, 4), bool), neg))
    try:
        result, session, _ = run.chase(run.version, phase="initial_zero_update")
        accepted, reasons = propose(run, tmp_path, 0, [1, None, None, None], result, session)
        assert not accepted and reasons == ["proposal_consequence_contradicts_validated_labels"]
    finally:
        close(run)


@pytest.mark.parametrize("execution", ["incremental", "full_recompute"])
def test_contradiction_with_the_proposal_itself_is_a_conflict(tmp_path, execution):
    rules = (Rule((Doc("x"), at("x", 0), Label("x", "A")), Label("x", "B", "neg")),)
    run = make(tmp_path, 1, rules, execution=execution)
    try:
        result, session, _ = run.chase(run.version, phase="initial_zero_update")
        accepted, reasons = propose(run, tmp_path, 0, [1, 1, None, None], result, session)
        assert not accepted and reasons == ["proposal_adds_conflict_or_invalid_relation"]
    finally:
        close(run)


def test_a_blocked_temporary_root_is_not_attributed_to_the_candidate(tmp_path):
    ctx = dict(docs=[Document(cnt="node%d" % i, ttl="", mtd="") for i in range(2)], labels=["A", "B"],
               gamma=RefGamma.empty(2, 2), model_probabilities={}, model_decisions={})
    neg = np.zeros((2, 2), bool)
    neg[0, 0] = True
    gamma = RefGamma(np.zeros((2, 2), bool), neg)
    seed = np.zeros((2, 2), bool)
    seed[0, 0] = True                         # temporary positive root contradicting Gamma
    rules = (Rule((Doc("x"), at("x", 1)), Label("x", "B")),)
    runner = IndexedChaseRunner(dict(ctx, gamma=gamma), tmp_path / "a")
    try:
        before = retaining_chase(runner, rules, gamma, gamma_wins=True, initial_positive=seed)
        assert before.gamma_blocked == 1
        neg2 = neg.copy()
        neg2[1, 0] = True                     # candidate: doc1 is not A
        checked_gamma = RefGamma(np.zeros((2, 2), bool), neg2)
        candidate = (np.zeros((2, 2), bool), neg2 & ~neg)
        later = IndexedChaseRunner(dict(ctx, gamma=checked_gamma), tmp_path / "b")
        try:
            for execution in ("incremental", "full_recompute"):
                checked = retaining_chase(later, rules, checked_gamma, before=before, update_kind="check",
                                          gamma_wins=True, check_candidates=candidate, execution=execution)
                assert checked.candidate_contradictions == frozenset()
        finally:
            later.close()
    finally:
        runner.close()


def earlier_answer_case(tmp_path, policy, check_base):
    cross = lambda sign: Rule((Doc("x"), Doc("y"), at("x", 0), at("y", 2), Label("x", "A")), Label("y", "B", sign))
    run = make(tmp_path, 3, (cross("pos"), cross("neg")), policy=policy, check_base=check_base)
    result, session, _ = run.chase(run.version, phase="initial_zero_update")
    pos, neg = run.gamma.pos.copy(), run.gamma.neg.copy()
    pos[0, 0] = True
    neg[0, 1:] = True                         # accepted earlier in the batch: doc0 has only A
    run.gamma = RefGamma(pos, neg)
    return run, result, session


@pytest.mark.parametrize("policy", ["quarantine", "validated_gamma_wins"])
def test_batch_start_check_base_keeps_the_documented_approximation(tmp_path, policy):
    run, result, session = earlier_answer_case(tmp_path, policy, "batch_start")
    try:
        accepted, reasons = propose(run, tmp_path, 1, [0, 0, 0, 1], result, session)
        assert not accepted and reasons == ["proposal_adds_conflict_or_invalid_relation"]
    finally:
        close(run)


@pytest.mark.parametrize("policy", ["quarantine", "validated_gamma_wins"])
def test_refreshed_check_base_includes_answers_accepted_earlier_in_the_batch(tmp_path, policy):
    run, result, session = earlier_answer_case(tmp_path, policy, "refreshed")
    try:
        refreshes = []
        original = run.event

        def event(kind, **kw):
            if kind.startswith("LLM_check_base"):
                refreshes.append(kind)
            return original(kind, **kw)

        run.event = event
        accepted, reasons = propose(run, tmp_path, 1, [0, 0, 0, 1], result, session)
        assert accepted and not reasons
        assert refreshes == ["LLM_check_base_refreshed"]
        # Gamma changed (the accepted proposal): a later check refreshes once more.
        propose(run, tmp_path, 2, [0, 0, 1, 0], result, session)
        assert refreshes.count("LLM_check_base_refreshed") == 2
        run._drop_check_base()
        assert run._check_base is None
    finally:
        close(run)


def test_refreshed_check_base_validation_and_workflow_variant(tmp_path):
    from loris.labeling import cli
    from loris.experiments import workflow

    _, manifest = setup()
    manifest["feedback"].update(trusted_replay=True, trusted_replay_fraction=0.05)
    manifest["loop"].update(unknown_semantics="retain_consequences", llm_check_base="refreshed")
    cli.validate(manifest)
    for bad in ("other", None):
        broken = copy.deepcopy(manifest)
        broken["loop"]["llm_check_base"] = bad
        with pytest.raises(ValueError):
            cli.validate(broken)
    broken = copy.deepcopy(manifest)
    broken["loop"]["unknown_semantics"] = "full_rebuild"
    with pytest.raises(ValueError):
        cli.validate(broken)
    assert "exact_llm_check" in workflow.VARIANTS
    data = dict(loop=dict(conflict_policy="halt", unknown_semantics="retain_consequences",
                          llm_check_base="refreshed"), strategy="frozen", weighting="uniform",
                feedback={}, dynamic=dict(alias="aux"))
    workflow._adapt_output_controls(data)
    assert data["loop"]["llm_check_base"] == "batch_start"


def calibrated(tmp_path, grid=(0.1, 1.0)):
    from loris.labeling.reliability import main_calibrate
    from tests.test_fit_reliability import calibration_fixture

    args, _, _ = calibration_fixture(tmp_path)
    data = json.loads(Path(args.manifest).read_text())
    data["loop"].update(conflict_policy="unknown_arbitrate", unknown_semantics="retain_consequences",
                        gamma_conflict_policy="validated_gamma_wins", commit_gate="independent_calibration")
    data["arbitration"] = dict(alias=data["dynamic"]["alias"], default_threshold=0.5, thresholds={})
    data["fallback"] = dict(enabled=True, alias=data["dynamic"]["alias"], calibration="independent_calibration",
                            default_threshold=1.0, thresholds={}, grid=list(grid))
    data["calibration"].update(cells="trusted_only", max_step=0.1, minimum_support_each_sign=1)
    Path(args.manifest).write_text(json.dumps(data))
    assert main_calibrate(args) == 0
    return args


def start(tmp_path, args, change=None):
    from loris.labeling import cli
    from tests.test_current_model_reliability import scripted_trainer

    manifest = json.loads(Path(args.manifest).read_text())
    if change is not None:
        change(manifest)
    path = tmp_path / ("manifest_%d.json" % len(list(tmp_path.glob("manifest_*.json"))))
    path.write_text(json.dumps(manifest))
    effective, ctx, rules, artifact, _ = cli.load(path, weighting="direct_rule_precision_discount",
                                                  source_reliability=tmp_path / "calibrated/RELIABILITY.json")
    trainer, _ = scripted_trainer(ctx)
    out = Path(tempfile.mkdtemp(dir=tmp_path)) / "run"
    return LabelingRun(context=ctx, rules=rules, manifest=effective, output=out, trainer=trainer,
                       reliability=artifact)


def test_fallback_record_must_match_the_run(tmp_path):
    args = calibrated(tmp_path)
    run = start(tmp_path, args)
    record = json.loads((tmp_path / "calibrated/FALLBACK_CALIBRATION.json").read_text())
    assert run.fallback_thresholds.tolist() == [record["thresholds"][l] for l in run.labels]
    for change in (lambda m: m["fallback"].update(grid=[0.25, 0.75]),
                   lambda m: m["arbitration"].update(default_threshold=0.9),
                   lambda m: m["fallback"].update(calibration="fixed", thresholds={"A": 0.3})):
        with pytest.raises(ValueError, match="FALLBACK_CALIBRATION.json does not match"):
            start(tmp_path, args, change)


def test_tampered_or_unbound_fallback_record_is_rejected(tmp_path):
    args = calibrated(tmp_path)
    path = tmp_path / "calibrated/FALLBACK_CALIBRATION.json"
    original = path.read_text()
    record = json.loads(original)
    path.write_text(json.dumps(dict(record, thresholds={k: 0.123 for k in record["thresholds"]})))
    with pytest.raises(ValueError, match="file identity differs"):
        start(tmp_path, args)
    path.write_text(original)
    start(tmp_path, args)  # restored bytes are accepted again
    results = tmp_path / "calibrated/RESULTS.json"
    receipt = json.loads(results.read_text())
    assert receipt["fallback_calibration"]["path"] == str(path.resolve())
