"""Decision checks and compact roots retain complete logical update semantics."""
import copy
import gzip
import json

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, ML, RefGamma, Rule
from loris._engine.typed_runtime import IndexedChaseRunner
from loris.labeling import unknown_incremental as engine
from loris.labeling.exports import export_facts, export_provenance
from tests.test_unknown_incremental import at, context, edge, local, same


@pytest.mark.parametrize("gamma_wins", [False, True])
@pytest.mark.parametrize("execution", ["incremental", "full_recompute"])
def test_compact_history_and_decision_checks_match_explicit_across_updates(tmp_path, gamma_wins, execution):
    ctx = context(5)
    p, n = ctx["gamma"].pos.copy(), ctx["gamma"].neg.copy()
    p[0, 0] = True
    n[3, 3] = True
    gamma = ctx["gamma"] = RefGamma(p, n, ((0, 1),))
    rules = (local(Label("x", "B"), Label("x", "A")),
             local(Label("x", "A", "neg"), Label("x", "B"), at("x", 1)),
             local(Label("x", "C"), Label("x", "B")),
             local(Label("x", "D"), Label("x", "C"), ML("x", "aux", "D", .5)),
             edge(2, 3, Label("x", "A")))
    runner = IndexedChaseRunner(ctx, tmp_path / "a")
    later = None
    try:
        states = [engine.retaining_chase(runner, rules, gamma, gamma_wins=gamma_wins,
                                         root_storage=storage, execution=execution)
                  for storage in ("explicit", "compact")]
        same(*states)
        for result in states:
            assert result.status == "fixpoint"
        p2 = p.copy(); p2[2, 0] = True
        checked_gamma = RefGamma(p2, n, gamma.subset)
        candidates = (p2 & ~p, np.zeros_like(n)) if gamma_wins else None
        checked = []
        for result in states:
            identity = engine.history_digest(result)
            kwargs = dict(before=result, update_kind="check", gamma_wins=gamma_wins,
                          execution=execution, check_candidates=candidates)
            full = engine.retaining_chase(runner, rules, checked_gamma, **kwargs)
            fast = engine.retaining_chase(runner, rules, checked_gamma, decision_only=True, **kwargs)
            same(full, fast)
            assert full.candidate_contradictions == fast.candidate_contradictions
            assert not hasattr(fast, "history_nodes")
            assert engine.history_digest(result) == identity
            checked.append(full)
        same(*checked)
        assert checked[0].candidate_contradictions == checked[1].candidate_contradictions
        # A committed annotation resolves old unknown and may withdraw a
        # quarantined version and its descendants under validated Gamma wins.
        p3, n3 = p.copy(), n.copy(); n3[1, 0] = True
        committed_gamma = RefGamma(p3, n3, gamma.subset)
        committed = [engine.retaining_chase(runner, rules, committed_gamma, before=old,
                      update_kind="chaseInc", gamma_wins=gamma_wins, execution=execution,
                      initial_unknown=np.zeros_like(n)) for old in states]
        same(*committed)
        changed = copy.deepcopy(ctx)
        changed["model_probabilities"]["aux"][:] = .1
        later = IndexedChaseRunner(changed, tmp_path / "b")
        delta = [engine.retaining_chase(later, rules, committed_gamma, before=old,
                 old_runner=runner, old_rules=rules, update_kind="chaseDelta",
                 gamma_wins=gamma_wins, execution=execution) for old in committed]
        same(*delta)
        views = [engine.recorded_feedback_view(later, rules, result, committed_gamma)
                 for result in delta]
        same(*views)
        for result in delta:
            for identifier, node in result.history_nodes.items():
                for parent in node.get("history_dependencies", ()):
                    assert parent in result.history_nodes
    finally:
        runner.close()
        if later is not None:
            later.close()


def test_compact_export_binds_initial_root_arrays_and_keeps_explicit_history(tmp_path):
    ctx = context(4)
    p = ctx["gamma"].pos.copy(); p[0, 0] = True
    ctx["gamma"] = RefGamma(p, ctx["gamma"].neg)
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = engine.retaining_chase(runner, (local(Label("x", "B"), Label("x", "A")),),
                                        ctx["gamma"], root_storage="compact")
        export_facts(tmp_path, result)
        export_provenance(tmp_path, result)
        with gzip.open(tmp_path / "HISTORY.json.gz", "rt") as stream:
            record = json.load(stream)
        assert record["schema"] == engine.COMPACT_HISTORY_SCHEMA
        assert record["nodes"]["roots"]["count"] == 5
        assert len(record["nodes"]["explicit"]) == 1
        assert next(iter(record["nodes"]["explicit"].values()))["history_dependencies"] == ["g:pos:0:0"]
        with np.load(tmp_path / "FACTS.npz") as arrays:
            np.testing.assert_array_equal(arrays["initial_root_gamma_positive"], p)
        assert engine.history_digest(result) == engine.digest(engine.export_history(result))
    finally:
        runner.close()


def test_committed_history_fingerprint_reused_and_replacement_invalidates(tmp_path, monkeypatch):
    ctx = context(1)
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        result = engine.retaining_chase(runner, (), ctx["gamma"])
        identity = result.history_identity
        original = engine.export_history
        def forbidden(*args, **kwargs):
            pytest.fail("immutable history must reuse its committed fingerprint")
        monkeypatch.setattr(engine, "export_history", forbidden)
        assert engine.history_digest(result) == identity
        monkeypatch.setattr(engine, "export_history", original)
        result.history_retired = frozenset({"h0"})
        assert engine.history_digest(result) != identity
        with pytest.raises(TypeError):
            result.history_current["bad"] = "h0"
    finally:
        runner.close()


def test_large_gamma_check_does_not_enumerate_shared_history_or_all_roots(tmp_path, monkeypatch):
    n, k = 2501, 100
    ctx = context(n, labels=tuple("L%d" % j for j in range(k)))
    neg = np.ones((n, k), bool); neg[-1] = False
    gamma = ctx["gamma"] = RefGamma(np.zeros_like(neg), neg)
    runner = IndexedChaseRunner(ctx, tmp_path / "cache")
    try:
        before = engine.retaining_chase(runner, (), gamma, gamma_wins=True)
        assert before.history_schema == engine.COMPACT_HISTORY_SCHEMA
        assert not before.history_nodes.mapping.delta
        def forbidden(*args, **kwargs):
            pytest.fail("full root/history enumeration on a sparse check")
        monkeypatch.setattr(engine, "_roots", forbidden)
        monkeypatch.setattr(engine._RootMap, "__iter__", forbidden)
        pos = gamma.pos.copy(); pos[-1, 0] = True
        after = engine.retaining_chase(runner, (), RefGamma(pos, neg), before=before,
                  update_kind="check", gamma_wins=True, check_candidates=(pos, np.zeros_like(pos)),
                  decision_only=True)
        assert after.status == "fixpoint" and after.positive_facts[-1, 0]
        assert after.reference_audit["maintenance"]["root_candidates_processed"] == 1
        committed = engine.retaining_chase(runner, (), RefGamma(pos, neg), before=before,
                  update_kind="chaseInc", gamma_wins=True)
        assert len(committed.history_nodes.mapping.delta) == 1
    finally:
        runner.close()
