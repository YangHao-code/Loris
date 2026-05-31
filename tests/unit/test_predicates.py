"""Characterization tests for the 9 LORIS predicate types.

Runs against legacy ``pattern_extraction.predicates`` or new ``loris.predicates``
(see tests/conftest.py + LORIS_TEST_TARGET). These pin the observable behavior
of every predicate's ``__call__`` truth table and the to_dict/from_dict
round-trip, so the Phase-1 migration can be proven behavior-preserving.

Only deterministic paths are exercised (no ``sim=True`` embedding paths and no
fitted ML models), so the suite is fast and reproducible without GPU/network.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# MatchPredicate — match(x.A, r)
# ---------------------------------------------------------------------------

def test_match_predicate_basic(predicates_mod, document_cls):
    P = predicates_mod.MatchPredicate
    doc = document_cls(cnt="The bank reported profits.", ttl="Finance News")
    assert P("cnt", "bank")(doc) is True
    assert P("cnt", "missing")(doc) is False
    # targets the chosen attribute only
    assert P("ttl", "Finance")(doc) is True
    assert P("ttl", "bank")(doc) is False


def test_match_predicate_empty_text(predicates_mod, document_cls):
    P = predicates_mod.MatchPredicate
    doc = document_cls(cnt="")
    assert P("cnt", "anything")(doc) is False


# ---------------------------------------------------------------------------
# CooccurPredicate — cooccur(x.A, r1, r2), unordered
# ---------------------------------------------------------------------------

def test_cooccur_predicate(predicates_mod, document_cls):
    P = predicates_mod.CooccurPredicate
    doc = document_cls(cnt="The bank and the market reacted.")
    assert P("cnt", "bank", "market")(doc) is True
    # unordered: both directions hold
    assert P("cnt", "market", "bank")(doc) is True
    # one term missing -> False
    assert P("cnt", "bank", "absent")(doc) is False


# ---------------------------------------------------------------------------
# BeforePredicate — before(x.A, r1, r2): start(r1) < start(r2)
# ---------------------------------------------------------------------------

def test_before_predicate_order(predicates_mod, document_cls):
    P = predicates_mod.BeforePredicate
    doc = document_cls(cnt="The bank and the market reacted.")
    assert P("cnt", "bank", "market")(doc) is True
    assert P("cnt", "market", "bank")(doc) is False
    # missing second term -> False
    assert P("cnt", "bank", "absent")(doc) is False


# ---------------------------------------------------------------------------
# FreqPredicate — freq(x.A, r) >= eta
# ---------------------------------------------------------------------------

def test_freq_predicate_threshold(predicates_mod, document_cls):
    P = predicates_mod.FreqPredicate
    doc = document_cls(cnt="spam spam spam eggs")
    # 3 occurrences of "spam"
    assert P("cnt", "spam", ">=", 3)(doc) is True
    assert P("cnt", "spam", ">=", 4)(doc) is False
    assert P("cnt", "eggs", ">=", 1)(doc) is True
    assert P("cnt", "spam", "==", 3)(doc) is True


def test_freq_predicate_eta_is_float(predicates_mod, document_cls):
    """eta is normalised to float for serialization stability (memory note)."""
    p = predicates_mod.FreqPredicate("cnt", "x", ">=", 2)
    assert isinstance(p.eta, float)


# ---------------------------------------------------------------------------
# LabelPredicate — read-only label tests (contains / eq / subset / strict)
# ---------------------------------------------------------------------------

def test_label_predicate_contains(predicates_mod, document_cls):
    LP = predicates_mod.LabelPredicate
    doc = document_cls(cnt="x", lbl={"finance", "markets"})
    assert LP(label="finance", op="contains")(doc) is True
    assert LP(label="sports", op="contains")(doc) is False


def test_label_predicate_is_read_only(predicates_mod, document_cls):
    """A label predicate must NOT mutate doc.lbl (paper: mutation is a rule
    consequence, never a predicate side effect). This pins the post-`minus`
    invariant; against legacy it documents that contains/eq are pure."""
    LP = predicates_mod.LabelPredicate
    doc = document_cls(cnt="x", lbl={"a", "b"})
    before = set(doc.lbl)
    LP(label="a", op="contains")(doc)
    assert doc.lbl == before


# ---------------------------------------------------------------------------
# Serialization round-trip for the deterministic predicate types
# ---------------------------------------------------------------------------

def _roundtrip(mod, pred):
    d = mod.predicate_to_dict(pred)
    back = mod.predicate_from_dict(d)
    return d, back, mod.predicate_to_dict(back)


@pytest.mark.parametrize("factory", [
    lambda m: m.MatchPredicate("cnt", "bank"),
    lambda m: m.CooccurPredicate("cnt", "bank", "market"),
    lambda m: m.BeforePredicate("ttl", "a", "b"),
    lambda m: m.FreqPredicate("cnt", "spam", ">=", 2),
    lambda m: m.LabelPredicate(label="finance", op="contains"),
])
def test_predicate_serialization_roundtrip(predicates_mod, factory):
    pred = factory(predicates_mod)
    d, back, d2 = _roundtrip(predicates_mod, pred)
    # to_dict is stable across a from_dict round-trip (canonical form)
    assert d == d2, (d, d2)
    # reconstructed predicate is of the same type
    assert type(back) is type(pred)
