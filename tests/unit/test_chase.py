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

import scipy.sparse as sp

from loris.document import Document
from loris.predicates import (
    GroupPredicate,
    LabelPredicate,
    MatchPredicate,
)
from loris.predicates._core import _Pattern
from loris.rules.group_propagation import discover_equal_rules
from loris.rules.rdl import RDL, RDLSet
from loris.chase.multi_chase import (
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
    """B-5: coincidental prediction-bitmap containment is NOT a subset relation.

    Previously the chase inferred ``y.lbl ⊆ x.lbl`` from ``y_pos & ~x_pos`` and
    propagated labels up those bogus edges (the B1 bug). That source is removed;
    ``sub``/``sup`` are now populated only by the comparison consequence
    (legitimate co-membership source added in C-9). So bitmap containment must
    NOT cause propagation.
    """

    def test_bitmap_containment_does_not_propagate(self):
        labels = ["A", "B", "C"]
        # doc0 starts with {A, B}, doc1 starts with {A} → doc1.pos ⊆ doc0.pos
        # (a coincidental bitmap containment, NOT a real subset relation).
        # Rule: match "trigger" → +C (fires on doc1 only).
        rules = [
            RDL(
                body=(MatchPredicate(attr="cnt", r=_pat("trigger")),),
                consequence="C",
            ),
        ]
        docs = [
            # doc0 deliberately does NOT contain "trigger" — it can only acquire
            # C through (the now-removed) bitmap-containment transitivity.
            _doc("Nothing relevant in this document", lbl={"A", "B"}),
            _doc("Has trigger word", lbl={"A"}),
        ]
        # Even with transitivity explicitly ON, there is no legitimate sub/sup
        # source, so C must stay confined to doc1.
        chase = MultiChase(rules, labels, enable_transitivity=True)
        result = chase.run(docs)

        assert result.status == "fixpoint"
        # doc1 gets C from the rule
        assert result.predictions[1, labels.index("C")] == 1.0
        # doc0 must NOT inherit C from coincidental bitmap containment (B1 fixed)
        assert result.predictions[0, labels.index("C")] == 0.0

    def test_transitivity_default_off(self):
        # B-5: enable_transitivity now defaults to False.
        import inspect
        assert inspect.signature(MultiChase.__init__).parameters[
            "enable_transitivity"].default is False

    def test_conflict_mode_default_negative_wins(self):
        # C-8: chase converges (not halt) on conflict, preserving remove rules.
        import inspect
        assert inspect.signature(MultiChase.__init__).parameters[
            "conflict_mode"].default == "negative_wins"


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


# ── Comparison consequence: x.lbl = y.lbl (consequence_op="equal") ───

def _membership_csr(rows_values, n_values):
    """Build a (n_docs × n_values) 0/1 csr from per-doc value sets."""
    rows, cols = [], []
    for i, vals in enumerate(rows_values):
        for v in vals:
            rows.append(i)
            cols.append(v)
    data = np.ones(len(rows), dtype=np.int8)
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(len(rows_values), n_values), dtype=np.int8
    )


def _equal_rule(attr="A", n_values=2):
    """An x.lbl=y.lbl comparison-consequence rule over attribute `attr`."""
    return RDL(
        body=(GroupPredicate(attr_name=attr, group_count=n_values),),
        consequence="",
        consequence_op="equal",
    )


FIN = LABELS.index("finance")
SPORT = LABELS.index("sports")
ECON = LABELS.index("economy")


class TestEqualConsequence:
    """x.A=y.A → x.lbl=y.lbl: batched label-set copy across co-members."""

    def test_copies_labels_across_comembers(self):
        # docs 0,1 share value 0; doc 2 has value 1 alone. Seed finance on d0.
        docs = [_doc("a", {"finance"}), _doc("b"), _doc("c")]
        M = _membership_csr([{0}, {0}, {1}], 2)
        chase = MultiChase([_equal_rule()], LABELS, enable_transitivity=False,
                           virtual_attrs={"A": M})
        result = chase.run(docs)
        assert result.status == "fixpoint"
        assert result.predictions[0, FIN] == 1.0   # keeps its own
        assert result.predictions[1, FIN] == 1.0   # copied from co-member d0
        assert result.predictions[2, FIN] == 0.0   # shares no value

    def test_symmetry(self):
        # d0=finance, d1=sports, share value 0 → both end {finance, sports}.
        docs = [_doc("a", {"finance"}), _doc("b", {"sports"})]
        M = _membership_csr([{0}, {0}], 1)
        chase = MultiChase([_equal_rule(n_values=1)], LABELS,
                           enable_transitivity=False, virtual_attrs={"A": M})
        result = chase.run(docs)
        for d in (0, 1):
            assert result.predictions[d, FIN] == 1.0
            assert result.predictions[d, SPORT] == 1.0

    def test_transitive_closure_shared_value_chain(self):
        # d0:{v0}, d1:{v0,v1}, d2:{v1}; seed finance on d0. Multi-hop: d0→d1→d2.
        docs = [_doc("a", {"finance"}), _doc("b"), _doc("c")]
        M = _membership_csr([{0}, {0, 1}, {1}], 2)
        chase = MultiChase([_equal_rule()], LABELS, enable_transitivity=False,
                           virtual_attrs={"A": M})
        result = chase.run(docs)
        assert result.status == "fixpoint"
        assert result.predictions[2, FIN] == 1.0   # reached across the chain
        assert result.n_rounds >= 2                # genuinely multi-hop

    def test_no_op_when_no_equal_rules(self):
        # An add/text rule + non-empty virtual_attrs must be byte-identical to
        # running with virtual_attrs=None (backs the golden-neutrality claim).
        rules = [RDL(body=(MatchPredicate(attr="cnt", r=_pat("bank")),),
                     consequence="finance")]
        docs = [_doc("the bank"), _doc("nothing")]
        M = _membership_csr([{0}, {0}], 1)
        r_with = MultiChase(rules, LABELS, enable_transitivity=False,
                            virtual_attrs={"A": M}).run(docs)
        r_without = MultiChase(rules, LABELS, enable_transitivity=False,
                               virtual_attrs=None).run(docs)
        assert np.array_equal(r_with.predictions, r_without.predictions)

    def test_self_inclusion_harmless(self):
        # Lone doc with its own value: ends with exactly its seeded label.
        docs = [_doc("a", {"finance"})]
        M = _membership_csr([{0}], 1)
        chase = MultiChase([_equal_rule(n_values=1)], LABELS,
                           enable_transitivity=False, virtual_attrs={"A": M})
        result = chase.run(docs)
        assert result.predictions[0, FIN] == 1.0
        assert result.predictions[0].sum() == 1.0   # nothing spurious added

    def test_fixpoint_terminates_dense(self):
        # All docs share one value (dense clique); two seeds → all get both.
        docs = [_doc("a", {"finance"}), _doc("b", {"sports"}), _doc("c")]
        M = _membership_csr([{0}, {0}, {0}], 1)
        chase = MultiChase([_equal_rule(n_values=1)], LABELS,
                           enable_transitivity=False, virtual_attrs={"A": M})
        result = chase.run(docs)
        assert result.status == "fixpoint"
        for d in (0, 1, 2):
            assert result.predictions[d, FIN] == 1.0
            assert result.predictions[d, SPORT] == 1.0

    def test_conflict_interaction_resolution(self):
        # d0 seeded finance; a remove-rule fires -finance on d1 (text "bank");
        # d0,d1 share a value so equal copies finance into d1.pos → conflict.
        docs = [_doc("a", {"finance"}), _doc("bank")]
        M = _membership_csr([{0}, {0}], 1)
        rules = [
            RDL(body=(MatchPredicate(attr="cnt", r=_pat("bank")),),
                consequence="finance", consequence_op="remove"),
            _equal_rule(n_values=1),
        ]
        # negative_wins: d1 finance excluded (pos & ~neg)
        neg = MultiChase(rules, LABELS, conflict_mode="negative_wins",
                         enable_transitivity=False, virtual_attrs={"A": M}).run(docs)
        assert neg.predictions[1, FIN] == 0.0
        # positive_wins: d1 finance kept (pos)
        pos = MultiChase(rules, LABELS, conflict_mode="positive_wins",
                         enable_transitivity=False, virtual_attrs={"A": M}).run(docs)
        assert pos.predictions[1, FIN] == 1.0

    def test_serialization_and_classification(self):
        rule = _equal_rule()
        d = rule.to_dict()
        assert d["consequence"] == "" and d["consequence_op"] == "equal"
        rt = RDL.from_dict(d)
        assert rt.consequence == "" and rt.consequence_op == "equal"
        # _classify_rules routes equal-rule to _equal_rule_idxs, a plain group
        # add-rule to _group_rule_idxs, a text rule to _text_rule_idxs.
        group_add = RDL(
            body=(GroupPredicate(attr_name="A", group_count=2),
                  LabelPredicate(label="finance", op="contains")),
            consequence="finance", consequence_op="add",
        )
        text = RDL(body=(MatchPredicate(attr="cnt", r=_pat("bank")),),
                   consequence="finance")
        chase = MultiChase([rule, group_add, text], LABELS,
                           enable_transitivity=False)
        assert chase._equal_rule_idxs == [0]
        assert chase._group_rule_idxs == [1]
        assert chase._text_rule_idxs == [2]


class TestEqualRuleDiscovery:
    """B-6: discover_equal_rules admits attributes accuracy-guided (paper §5.2)."""

    def test_admits_beneficial_attribute(self):
        # docs 0,1 share value 0 (truth finance); 2,3 share value 1 (truth sports).
        # base: only d0 has finance, d2 has sports → equal-rule copies to 1 and 3.
        M = sp.csr_matrix(np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int8))
        val = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32)
        existing = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        out = discover_equal_rules({"A": M}, val, existing,
                                   min_fires=1, min_corr_prec=0.5)
        assert len(out) == 1
        _trial, rule = out[0]
        assert rule.consequence_op == "equal" and rule.consequence == ""
        assert rule.body[0].attr_name == "A"
        assert rule.val_stats["corr_prec"] == 1.0
        assert rule.score > 0

    def test_rejects_harmful_attribute(self):
        # Attribute B groups docs with CONFLICTING truth → copying labels hurts
        # precision → must be rejected by the corr_prec / f1_gain gate.
        M = sp.csr_matrix(np.array([[1], [1]], dtype=np.int8))  # d0,d1 co-members
        val = np.array([[1, 0], [0, 1]], dtype=np.float32)       # disjoint truth
        existing = np.array([[1, 0], [0, 1]], dtype=np.float32)  # already correct
        out = discover_equal_rules({"B": M}, val, existing,
                                   min_fires=1, min_corr_prec=0.6)
        # Copying would add finance→d1 and sports→d0, both wrong → 0 improved.
        assert out == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
