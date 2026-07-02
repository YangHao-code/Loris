"""Unit tests for Lever D — multi-literal conjunctive joins x.A=y.A ∧ x.B=y.B.

Two guarantees: (1) discovery mines a 2-GroupPredicate rule where neither single
attribute is precise but their conjunction is, only when ``multiattr=True``
(``multiattr=False`` is golden-neutral); (2) the chase enforces BOTH group
literals conjunctively (the historical code kept only the first via ``next()``).
"""
import numpy as np
import scipy.sparse as sp

from loris.document import Document
from loris.predicates import GroupPredicate, LabelPredicate
from loris.rules.group_propagation import discover_group_rules
from loris.rules.rdl import RDL
from loris.chase.multi_chase import MultiChase


def _csr(value_per_doc, n_values):
    n = len(value_per_doc)
    rows = np.arange(n)
    cols = np.asarray(value_per_doc)
    return sp.csr_matrix((np.ones(n, np.int8), (rows, cols)), shape=(n, n_values))


def _conjunction_only_attrs():
    """40 docs, label L0 true for docs 0–9. Attr cluster (a0={0..19}) and attr
    phrase (b0={0..9,20..29}) each predict L0 at prec 0.50; only their
    conjunction (= docs 0..9) is precise (1.0)."""
    A = _csr([0] * 20 + [1] * 20, 2)                       # family: cluster
    B = _csr([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10, 2)  # family: phrase
    attrs = {"cluster_2": A, "phrase": B}
    n, L = 40, 2
    val_labels = np.zeros((n, L), np.float32)
    val_labels[:10, 0] = 1
    label_state = val_labels.copy()
    existing = np.zeros((n, L), np.float32)
    docs = [Document(cnt="same filler text") for _ in range(n)]
    names = ["L0", "L1"]
    return attrs, label_state, names, val_labels, existing, docs


def _two_group_rules(out, label="L0"):
    return [r for _t, r in out
            if r.consequence == label
            and sum(isinstance(p, GroupPredicate) for p in r.body) >= 2]


def test_multiattr_emits_conjunction_join():
    attrs, ls, names, vy, ex, docs = _conjunction_only_attrs()
    out = discover_group_rules(attrs, ls, names, vy, ex, docs,
                               min_fires=3, min_corr_prec=0.60, min_f1_gain=0.0,
                               multiattr=True)
    multi = _two_group_rules(out)
    assert multi, "multiattr=True should mine a cluster∧phrase join for L0"
    fams = {p.attr_name for p in multi[0].body if isinstance(p, GroupPredicate)}
    assert fams == {"cluster_2", "phrase"}
    assert multi[0].val_stats.get("corr_prec", 0) >= 0.60


def test_no_multiattr_when_flag_off():
    attrs, ls, names, vy, ex, docs = _conjunction_only_attrs()
    out = discover_group_rules(attrs, ls, names, vy, ex, docs,
                               min_fires=3, min_corr_prec=0.60, min_f1_gain=0.0,
                               multiattr=False)
    assert not _two_group_rules(out), "multiattr=False must not emit 2-group rules"


def test_flag_off_is_golden_neutral():
    """multiattr=False must reproduce the discovery output exactly (the default)."""
    attrs, ls, names, vy, ex, docs = _conjunction_only_attrs()
    a = discover_group_rules(attrs, ls, names, vy, ex, docs,
                             min_fires=3, min_corr_prec=0.60, min_f1_gain=0.0)
    b = discover_group_rules(attrs, ls, names, vy, ex, docs,
                             min_fires=3, min_corr_prec=0.60, min_f1_gain=0.0,
                             multiattr=False)
    assert [r.body for _, r in a] == [r.body for _, r in b]
    assert [r.consequence for _, r in a] == [r.consequence for _, r in b]


def test_chase_enforces_both_group_literals():
    """A 2-GroupPredicate rule fires only on a doc sharing BOTH attributes with a
    seed — not a doc sharing only one (which the old next()-truncation would let
    through)."""
    LABELS = ["finance", "economy"]
    FIN = 0
    # d0 seed; d1 shares A & B; d2 shares A only; d3 shares B only; d4 neither.
    docs = [Document(cnt="x", lbl={"finance"})] + [Document(cnt="x") for _ in range(4)]
    A = _csr([0, 0, 0, 1, 1], 2)   # value a0 = {0,1,2}
    B = _csr([0, 0, 1, 0, 1], 2)   # value b0 = {0,1,3}
    rule = RDL(
        body=(GroupPredicate(attr_name="A", group_count=2),
              GroupPredicate(attr_name="B", group_count=2),
              LabelPredicate(label="finance", op="contains")),
        consequence="finance", consequence_op="add")
    chase = MultiChase([rule], LABELS, enable_transitivity=False,
                       virtual_attrs={"A": A, "B": B})
    result = chase.run(docs)
    assert result.predictions[0, FIN] == 1.0   # seed keeps its label
    assert result.predictions[1, FIN] == 1.0   # shares BOTH → gets finance
    assert result.predictions[2, FIN] == 0.0   # shares only A → must NOT fire
    assert result.predictions[3, FIN] == 0.0   # shares only B → must NOT fire
    assert result.predictions[4, FIN] == 0.0   # shares neither
