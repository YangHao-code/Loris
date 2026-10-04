"""Adaptive relation storage preserves exact rows, snapshots and identities."""

import copy
import random
import tracemalloc

import numpy as np
import pytest

from loris._engine.relation_rows import RelationRows, RELATION_SCHEMA
from loris._engine.typed_runtime import RelationFacts
from loris._constraints.finite_comparison import ExactFiniteComparisonOracle
from loris._storage.selection_relations import ImmutableRelationCache


@pytest.mark.parametrize("reflexive", [False, True])
def test_random_row_mutations_transpose_and_immutable_snapshots(reflexive):
    rng = random.Random(91)
    n = 173
    rows = RelationRows(n, reflexive=reflexive)
    expected = [(1 << i) if reflexive else 0 for i in range(n)]
    for _ in range(100):
        row = rng.randrange(n)
        value = sum(1 << col for col in rng.sample(range(n), rng.randrange(9)))
        rows[row] = value
        expected[row] = value
    assert list(rows) == expected
    assert list(rows.transpose()) == [sum(1 << source for source, value in enumerate(expected)
                                         if value & (1 << target)) for target in range(n)]
    assert rows.transpose().transpose() == rows
    expected_edges = {(i, j) for i, value in enumerate(expected) for j in range(n)
                      if value & (1 << j)}
    assert set(rows.edges()) == expected_edges and rows.count() == len(expected_edges)
    frozen = rows.copy(frozen=True)
    copied = copy.deepcopy(frozen)
    digest = frozen.sha256()
    rows[0] ^= 1
    assert copied == frozen and frozen.sha256() == digest
    assert list(frozen) == expected
    with pytest.raises(TypeError):
        frozen[0] = 0
    with pytest.raises(AttributeError):
        frozen.reflexive = not reflexive
    with pytest.raises(TypeError):
        frozen._data[0] = 0


def test_relation_hash_is_canonical_across_defaults_streaming_inputs_and_dense_rows():
    n = 2048
    identity = RelationRows(n, reflexive=True)
    streamed = RelationRows.from_rows((1 << i for i in range(n)))
    assert identity.sha256() == streamed.sha256()
    assert RelationFacts(identity).identity() == RelationFacts(streamed).identity()
    identity[0] |= 1 << (n - 1)
    identity[17] = (1 << n) - 1
    streamed[0], streamed[17] = identity[0], identity[17]
    assert identity.sha256() == streamed.sha256()
    facts = RelationFacts(identity)
    assert facts.identity()["schema"] == RELATION_SCHEMA
    assert facts.rows.storage_counts()["sparse_rows"] == 1
    assert facts.rows.storage_counts()["dense_rows"] == 1
    assert (0, n - 1) in facts and (17, n - 1) in facts
    assert facts.to_dict()["rows"]["default_diagonal"] is True
    assert RelationFacts(RelationRows.from_dict(facts.to_dict()["rows"])).identity() == facts.identity()
    identity[0] = 0
    assert (0, n - 1) in facts
    assert facts.sha256 != RelationFacts(identity).sha256


def test_reflexive_storage_hash_and_snapshots_do_not_allocate_quadratic_bitsets():
    tracemalloc.start()
    try:
        rows = RelationRows(100_000, reflexive=True)
        facts = RelationFacts(rows)
        transpose = facts.rows.transpose()
        frozen = transpose.copy(frozen=True)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert facts.rows.storage_counts()["stored_rows"] == 0
    assert transpose.storage_counts()["stored_rows"] == 0
    assert len(facts) == 100_000 and frozen[-1] == 1 << 99_999
    # The old two identity-row lists alone exceed 1 GB at this N. The bound
    # checks actual traced allocations including hash and immutable snapshots.
    assert current < 1_000_000 and peak < 1_000_000


def test_exact_relation_cache_reuses_only_equal_frozen_storage():
    rows = RelationRows(1200, reflexive=True)
    cache = ImmutableRelationCache()
    before = cache.materialize(rows)
    assert cache.materialize(rows) is before
    rows[3] |= 1 << 1199
    after = cache.materialize(rows)
    assert after is not before and (3, 1199) not in before and (3, 1199) in after
    rows[3] = 1 << 3
    assert cache.materialize(rows) is before


def test_finite_oracle_certificates_snapshot_adaptive_rows_without_aliasing():
    pos = np.array([[1, 0], [1, 1]], bool)
    neg = np.zeros_like(pos)
    subset = RelationRows(2, reflexive=True)
    subset[0] |= 2
    strict = RelationRows(2)
    strict[0] = 2
    oracle = ExactFiniteComparisonOracle()
    assert oracle.feasible_rows(pos, neg, subset, strict)
    branch = copy.deepcopy(oracle)
    strict[1] = 1  # Newly close a proper cycle; old certificate cannot persist.
    assert not branch.feasible_rows(pos, neg, subset, strict)
    assert oracle._witness.checked_proper[1] == 0


@pytest.mark.parametrize("rows", [[-1], [2], [0, 4]])
def test_invalid_external_rows_cannot_enter_immutable_facts(rows):
    with pytest.raises(ValueError):
        RelationFacts(rows)


def test_real_conflict_certificate_rebuild_uses_empty_adaptive_rows(tmp_path, monkeypatch):
    from loris.document import Document
    from loris.predicates import MatchPredicate
    from loris._engine.rdl_reference import Doc, Label, RefGamma, Rule, Text
    from loris._engine.typed_session import TypedChaseSession, complete_state_identity, seed_identity
    from loris._learning.recovery import conflict_certificate, root_keys
    from loris._learning import signed_patterns

    n = 1200
    gamma = RefGamma.empty(n, 1)
    pos = gamma.pos.copy()
    pos[0, 0] = True
    gamma = RefGamma(pos, gamma.neg)
    context = dict(docs=[Document(cnt="row", ttl=str(i)) for i in range(n)], labels=["A"], gamma=gamma,
                   model_probabilities={}, model_decisions={})
    rules = [Rule((Doc("x"), Text("x", MatchPredicate("ttl", "^0$"))), Label("x", "A", "neg"))]
    session = TypedChaseSession(context, rules, tmp_path / "certificate", model_snapshot_id="fixed")
    try:
        sources = {key: {"origin": "validated_Gamma"} for key in root_keys(gamma)}
        result = session.chase(gamma, root_sources=sources)
        assert result.status == "conflict"
        certificate = conflict_certificate(result, session=session, seed=gamma,
                                           root_sources=sources, stage="storage_test")
        certificate["complete_proof_dag"] = result.proof_dag.to_dict()
        certificate["conflict_id"] = signed_patterns.digest(
            {key: value for key, value in certificate.items() if key != "conflict_id"})
        created = []

        def recording_empty_rows(axis):
            rows = RelationRows(axis)
            assert not rows.reflexive and rows.count() == 0
            created.append(rows)
            return rows

        monkeypatch.setattr(signed_patterns, "RelationRows", recording_empty_rows)
        signed_patterns._verify_conflict_certificate(
            certificate, complete_state_identity(result), session.receipt()["calls"][0],
            [rule.to_dict() for rule in rules], ["A"], seed_identity(gamma))
        assert len(created) == 2
        assert created[0].count() == n and created[1].count() == 0
        assert created[0].storage_counts()["sparse_rows"] > 0
    finally:
        session.close()
