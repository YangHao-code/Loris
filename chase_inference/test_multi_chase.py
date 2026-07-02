"""
chase_inference/test_multi_chase.py
------------------------------------
Unit tests for the Multi-Label Chase algorithm.

Run:  cd /root/autodl-tmp/Loris && python -m pytest chase_inference/test_multi_chase.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pattern_extraction.document import Document
from pattern_extraction.predicates import (
    LabelPredicate,
    MatchPredicate,
    _Pattern,
)
from rule_discovery.loris_rule_discovery import RDL, RDLSet
from chase_inference.multi_chase import (
    BulkLBL,
    ChaseResult,
    MultiChase,
    _SEED_RULE,
    _TRANSITIVITY_RULE,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _pat(s: str) -> _Pattern:
    """Shorthand: create a case-insensitive pattern."""
    import re
    return _Pattern(raw=s, flags=re.IGNORECASE)


def _doc(text: str, lbl: set | None = None) -> Document:
    return Document(cnt=text, lbl=lbl or set())


LABELS = ["finance", "economy", "sports", "tech"]


# ── Test 1: Fixpoint with text-only rules ────────────────────────────


class TestTextOnlyFixpoint:
    """Text-only rules converge in 1 round."""

    def test_basic(self):
        rules = [
            RDL(body=(MatchPredicate(attr="cnt", r=_pat("bank")),), consequence="finance"),
            RDL(body=(MatchPredicate(attr="cnt", r=_pat("goal")),), consequence="sports"),
        ]
        docs = [
            _doc("The bank reported profits"),
            _doc("He scored a goal"),
            _doc("Nothing relevant here"),
        ]
        chase = MultiChase(rules, LABELS, enable_transitivity=False)
        result = chase.run(docs)

        assert result.status == "fixpoint"
        # doc 0 → finance
        assert result.predictions[0, LABELS.index("finance")] == 1.0
        assert result.predictions[0, LABELS.index("sports")] == 0.0
        # doc 1 → sports
        assert result.predictions[1, LABELS.index("sports")] == 1.0
        # doc 2 → nothing
        assert result.predictions[2].sum() == 0.0


# ── Test 2: Label-dependent chain ────────────────────────────────────


class TestLabelDependentChain:
    """Rule A: match→+finance, Rule B: label(finance)→+economy → 2 rounds."""

    def test_chain(self):
        rules = [
            RDL(
                body=(MatchPredicate(attr="cnt", r=_pat("bank")),),
                consequence="finance",
            ),
            RDL(
                body=(LabelPredicate(label="finance", op="contains"),),
                consequence="economy",
            ),
        ]
        docs = [
            _doc("The bank reported profits"),
            _doc("Weather is sunny today"),
        ]
        chase = MultiChase(rules, LABELS, enable_transitivity=False)
        result = chase.run(docs)

        assert result.status == "fixpoint"
        # doc 0 gets both finance and economy
        assert result.predictions[0, LABELS.index("finance")] == 1.0
        assert result.predictions[0, LABELS.index("economy")] == 1.0
        # doc 1 gets neither
        assert result.predictions[1, LABELS.index("finance")] == 0.0
        assert result.predictions[1, LABELS.index("economy")] == 0.0


# ── Test 3: Conflict detection (halt mode) ───────────────────────────


class TestConflictHalt:
    """Rule A: →+finance, Rule B: →-finance on same doc → conflict."""

    def test_conflict(self):
        rules = [
            RDL(
                body=(MatchPredicate(attr="cnt", r=_pat("bank")),),
                consequence="finance",
                consequence_op="add",
            ),
            RDL(
                body=(MatchPredicate(attr="cnt", r=_pat("bank")),),
                consequence="finance",
                consequence_op="remove",
            ),
        ]
        docs = [_doc("The bank reported profits")]
        chase = MultiChase(rules, LABELS, conflict_mode="halt",
                           enable_transitivity=False)
        result = chase.run(docs)

        assert result.status == "conflict"
        assert len(result.conflicts) > 0
        assert (0, "finance") in result.conflicts


# ── Test 4: Conflict resolution (negative_wins) ─────────────────────


class TestConflictNegativeWins:
    """Same conflict but negative_wins → label removed from pos."""

    def test_negative_wins(self):
        rules = [
            RDL(
                body=(MatchPredicate(attr="cnt", r=_pat("bank")),),
                consequence="finance",
                consequence_op="add",
            ),
            RDL(
                body=(MatchPredicate(attr="cnt", r=_pat("bank")),),
                consequence="finance",
                consequence_op="remove",
            ),
        ]
        docs = [_doc("The bank reported profits")]
        chase = MultiChase(rules, LABELS, conflict_mode="negative_wins",
                           enable_transitivity=False)
        result = chase.run(docs)

        assert result.status == "fixpoint"
        # finance should be removed (negative wins)
        assert result.predictions[0, LABELS.index("finance")] == 0.0


# ── Test 5: Transitivity propagation ─────────────────────────────────


class TestTransitivity:
    """doc1.lbl ⊆ doc0.lbl → new label on doc1 propagates to doc0."""

    def test_propagation(self):
        labels = ["A", "B", "C"]
        # doc0 starts with {A, B}, doc1 starts with {A}
        # → doc1.lbl ⊆ doc0.lbl
        # Rule: match "trigger" → +C (will fire on doc1)
        # Transitivity should propagate C from doc1 to doc0
        rules = [
            RDL(
                body=(MatchPredicate(attr="cnt", r=_pat("trigger")),),
                consequence="C",
            ),
        ]
        docs = [
            _doc("No trigger here", lbl={"A", "B"}),
            _doc("Has trigger word", lbl={"A"}),
        ]
        chase = MultiChase(rules, labels, enable_transitivity=True)
        result = chase.run(docs)

        assert result.status == "fixpoint"
        # doc1 gets C from rule
        assert result.predictions[1, labels.index("C")] == 1.0
        # doc0 should also get C via transitivity (doc1 ⊆ doc0)
        assert result.predictions[0, labels.index("C")] == 1.0


# ── Test 6: No cycles ────────────────────────────────────────────────


class TestNoCycles:
    """label(A)→+B ∧ label(B)→+A with both seeded should not loop."""

    def test_no_infinite_loop(self):
        labels = ["A", "B"]
        rules = [
            RDL(body=(LabelPredicate(label="A", op="contains"),), consequence="B"),
            RDL(body=(LabelPredicate(label="B", op="contains"),), consequence="A"),
        ]
        # Seed doc0 with label A → should get B → then A already exists → done
        docs = [_doc("anything", lbl={"A"})]
        chase = MultiChase(rules, labels, max_rounds=50,
                           enable_transitivity=False)
        result = chase.run(docs)

        assert result.status == "fixpoint"
        assert result.predictions[0, labels.index("A")] == 1.0
        assert result.predictions[0, labels.index("B")] == 1.0
        # Should converge quickly (not hit max_rounds)
        assert result.n_rounds <= 5


# ── Test 7: Empty rules ─────────────────────────────────────────────


class TestEmptyRules:
    """No rules → base_predictions returned unchanged."""

    def test_empty(self):
        base = np.array([[1, 0, 1, 0]], dtype=np.float32)
        docs = [_doc("anything")]
        chase = MultiChase([], LABELS, enable_transitivity=False)
        result = chase.run(docs, base_predictions=base)

        assert result.status == "fixpoint"
        np.testing.assert_array_equal(result.predictions, base)


# ── Test 8: Base predictions seeding ─────────────────────────────────


class TestBasePredictionsSeeding:
    """Labels from base_predictions appear in lbl_pos."""

    def test_seeding(self):
        base = np.array([
            [1, 0, 0, 0],  # doc0: finance
            [0, 1, 0, 0],  # doc1: economy
        ], dtype=np.float32)
        # Rule: label(finance) → +tech
        rules = [
            RDL(body=(LabelPredicate(label="finance", op="contains"),),
                consequence="tech"),
        ]
        docs = [_doc("doc0"), _doc("doc1")]
        chase = MultiChase(rules, LABELS, enable_transitivity=False)
        result = chase.run(docs, base_predictions=base)

        assert result.status == "fixpoint"
        # doc0 gets tech (because finance was seeded from base)
        assert result.predictions[0, LABELS.index("tech")] == 1.0
        # doc1 doesn't get tech (no finance label)
        assert result.predictions[1, LABELS.index("tech")] == 0.0


# ── Test 9: Integration with RDLSet.chase_predict ────────────────────


class TestRDLSetIntegration:
    """RDLSet.chase_predict() works end-to-end."""

    def test_chase_predict(self):
        rules = [
            RDL(body=(MatchPredicate(attr="cnt", r=_pat("bank")),),
                consequence="finance"),
        ]
        rdl_set = RDLSet(rules, LABELS)
        docs = [_doc("The bank is open"), _doc("Nothing here")]
        result = rdl_set.chase_predict(docs, enable_transitivity=False)

        assert result.status == "fixpoint"
        assert result.predictions[0, LABELS.index("finance")] == 1.0
        assert result.predictions[1, LABELS.index("finance")] == 0.0


# ── Test 10: Max rounds termination ──────────────────────────────────


class TestMaxRounds:
    """Verify max_rounds cap works."""

    def test_max_rounds(self):
        # This rule chain won't cycle (evaluated set prevents it)
        # but let's set max_rounds=1 to verify early termination
        rules = [
            RDL(body=(MatchPredicate(attr="cnt", r=_pat("x")),),
                consequence="finance"),
            RDL(body=(LabelPredicate(label="finance", op="contains"),),
                consequence="economy"),
        ]
        docs = [_doc("x marks the spot")]
        chase = MultiChase(rules, LABELS, max_rounds=1,
                           enable_transitivity=False)
        result = chase.run(docs)

        # Should terminate at max_rounds before economy can be derived
        assert result.status == "max_rounds"
        assert result.predictions[0, LABELS.index("finance")] == 1.0


# ── Test 11: Provenance tracking ─────────────────────────────────────


class TestProvenance:
    """Track which rules derived which labels."""

    def test_provenance(self):
        rules = [
            RDL(body=(MatchPredicate(attr="cnt", r=_pat("bank")),),
                consequence="finance"),
        ]
        docs = [_doc("The bank")]
        chase = MultiChase(rules, LABELS, track_provenance=True,
                           enable_transitivity=False)
        result = chase.run(docs)

        assert (0, "finance") in result.provenance
        assert 0 in result.provenance[(0, "finance")]


# ── Test 12: BulkLBL operations ──────────────────────────────────────


class TestBulkLBL:
    """Verify BulkLBL data structure."""

    def test_create(self):
        lbl = BulkLBL.create(3, 4)
        assert lbl.pos.shape == (3, 4)
        assert lbl.neg.shape == (3, 4)
        assert not np.any(lbl.pos)
        assert not np.any(lbl.neg)
        assert len(lbl.sub) == 3
        assert len(lbl.sup) == 3

    def test_conflict_detection(self):
        lbl = BulkLBL.create(2, 3)
        lbl.pos[0, 1] = True
        lbl.neg[0, 1] = True
        assert np.any(lbl.has_conflict(0))
        assert not np.any(lbl.has_conflict(1))
        assert lbl.any_conflict(np.array([0, 1]))

    def test_no_conflict(self):
        lbl = BulkLBL.create(2, 3)
        lbl.pos[0, 0] = True
        lbl.neg[0, 1] = True
        assert not np.any(lbl.has_conflict(0))


# ── Test 13: Empty document list ─────────────────────────────────────


class TestEmptyDocs:
    """Chase on empty doc list returns empty result."""

    def test_empty(self):
        chase = MultiChase([], LABELS)
        result = chase.run([])
        assert result.status == "fixpoint"
        assert result.predictions.shape == (0, 4)
        assert result.n_rounds == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
