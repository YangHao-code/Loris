"""Chase calls that differ only in Gamma reuse one runner with identical results."""
import numpy as np
import pytest

from loris._engine.rdl_reference import Label, RefGamma
from loris.labeling.conflict_protocol import state_identity
from loris.labeling.runner import LabelingRun
from tests.test_conflict_protocol_loop import close, forbidden_training, rule, setup


def make(tmp_path, name):
    ctx, manifest = setup()
    manifest["loop"]["unknown_semantics"] = "retain_consequences"
    return LabelingRun(context=ctx, rules=(rule("A"), rule("B", Label("x", "A"))), manifest=manifest,
                       output=tmp_path / name, trainer=forbidden_training)


def checks(run):
    result, session, _ = run.chase(run.version, phase="initial_zero_update")
    outputs = []
    for column in (2, 3):
        pos = run.gamma.pos.copy()
        pos[1, column] = True
        seed = RefGamma(pos, run.gamma.neg, run.gamma.subset, run.gamma.strict_subset)
        checked, temporary, _ = run.chase(run.version, seed, phase="LLM_proposal_validation",
                                          previous=(result, session), update_kind="check",
                                          initial_unknown=result.conflict_unknown)
        outputs.append((checked, temporary))
    return result, session, outputs


@pytest.mark.parametrize("reuse", [True, False])
def test_check_calls_reuse_the_live_runner_and_close_with_the_last_session(tmp_path, reuse, monkeypatch):
    run = make(tmp_path, "run")
    if not reuse:
        monkeypatch.setattr(LabelingRun, "_reusable_runner", lambda self, *a: None)
    try:
        result, session, outputs = checks(run)
        runners = {id(session.runner)} | {id(t.runner) for _, t in outputs}
        assert len(runners) == (1 if reuse else 3)
        for _, temporary in outputs:
            run.release(temporary)
        assert session.runner in run.sessions
        session.runner.verify_code()  # still open and usable
        run.release(session)
        assert not run.sessions
    finally:
        close(run)


def test_reused_runner_results_equal_fresh_runner_results(tmp_path, monkeypatch):
    reused = make(tmp_path, "a")
    base = make(tmp_path, "b")
    try:
        first = checks(reused)
        monkeypatch.setattr(LabelingRun, "_reusable_runner", lambda self, *a: None)
        second = checks(base)
        pairs = [(first[0], second[0])] + [(a[0], b[0]) for a, b in zip(first[2], second[2])]
        for a, b in pairs:
            np.testing.assert_array_equal(a.positive_facts, b.positive_facts)
            np.testing.assert_array_equal(a.negative_facts, b.negative_facts)
            np.testing.assert_array_equal(a.conflict_unknown, b.conflict_unknown)
            assert a.reference_audit["context_id"] == b.reference_audit["context_id"]
            assert state_identity(a)["sha256"] == state_identity(b)["sha256"]
    finally:
        close(reused)
        close(base)


def test_changed_model_table_builds_a_new_runner(tmp_path):
    run = make(tmp_path, "run")
    try:
        _, session, _ = run.chase(run.version, phase="initial_zero_update")
        run.version.probabilities = np.full_like(run.version.probabilities, 0.9)
        _, other, _ = run.chase(run.version, phase="initial_zero_update")
        assert other.runner is not session.runner
    finally:
        close(run)
