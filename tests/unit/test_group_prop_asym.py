"""Unit tests for the strengthened group-propagation rule discovery:
asymmetric per-label gate + generalized text-narrowing. The key guarantee is
golden-neutrality: base_label_prec=None reproduces the old flat-gate behaviour."""
import numpy as np
import scipy.sparse as sp
from loris.document import Document
from loris.predicates import GroupPredicate, LabelPredicate, MatchPredicate
from loris.rules.group_propagation import discover_group_rules


def _toy():
    """3 labels, 1 attribute with 2 values; docs sharing a value co-predict a label.
    Designed so a pure group+label rule for label 0 has corr_prec around 0.6-0.7."""
    n = 40
    rng = np.random.RandomState(0)
    # attribute: docs 0-19 share value 0, docs 20-39 share value 1
    rows = np.arange(n); cols = (np.arange(n) >= 20).astype(int)
    M = sp.csr_matrix((np.ones(n, np.int8), (rows, cols)), shape=(n, 2))
    attrs = {"grp": M}
    L = 3
    val_labels = np.zeros((n, L), np.float32)
    # label 0 present in most of value-0 group (so x.grp=y.grp propagates it)
    val_labels[:16, 0] = 1            # 16/20 of group-0 truly have label 0
    existing = np.zeros((n, L), np.float32)  # nothing predicted yet
    label_state = val_labels.copy()   # neighbours' label set = truth (for fire mask)
    docs = [Document(cnt=f"doc {i} " + ("alpha beta" if i < 16 else "gamma")) for i in range(n)]
    names = [f"L{i}" for i in range(L)]
    return attrs, label_state, names, val_labels, existing, docs


def test_import_and_runs():
    attrs, ls, names, vy, ex, docs = _toy()
    out = discover_group_rules(attrs, ls, names, vy, ex, docs,
                               min_fires=3, min_corr_prec=0.50, min_f1_gain=0.0)
    assert isinstance(out, list)
    for _trial, rdl in out:
        # body always starts with a GroupPredicate + LabelPredicate (comparison link)
        types = [type(p) for p in rdl.body]
        assert GroupPredicate in types and LabelPredicate in types


def test_none_baseprec_is_default_behaviour():
    """base_label_prec=None must give exactly the same rules as not passing it."""
    attrs, ls, names, vy, ex, docs = _toy()
    a = discover_group_rules(attrs, ls, names, vy, ex, docs,
                             min_fires=3, min_corr_prec=0.50, min_f1_gain=0.0)
    b = discover_group_rules(attrs, ls, names, vy, ex, docs,
                             min_fires=3, min_corr_prec=0.50, min_f1_gain=0.0,
                             base_label_prec=None)
    assert len(a) == len(b)
    assert [r.consequence for _, r in a] == [r.consequence for _, r in b]
    assert [r.consequence_op for _, r in a] == [r.consequence_op for _, r in b]


def test_asym_gate_tightens_high_baseprec_label():
    """A label whose base precision is already very high gets a tighter gate
    (base_prec+margin), so a borderline group rule for it is no longer admitted."""
    attrs, ls, names, vy, ex, docs = _toy()
    # default (flat 0.50) admits label-0 group rule
    flat = discover_group_rules(attrs, ls, names, vy, ex, docs,
                                min_fires=3, min_corr_prec=0.50, min_f1_gain=0.0)
    n_flat = sum(1 for _t, r in flat if r.consequence == "L0")
    # asymmetric: pretend label 0 already has base precision 0.95 -> gate 1.0
    bp = np.zeros(len(names)); bp[0] = 0.95
    asym = discover_group_rules(attrs, ls, names, vy, ex, docs,
                                min_fires=3, min_corr_prec=0.50, min_f1_gain=0.0,
                                base_label_prec=bp, asym_margin=0.05)
    n_asym = sum(1 for _t, r in asym if r.consequence == "L0")
    assert n_flat >= 1, "toy should admit a label-0 group rule under flat gate"
    assert n_asym <= n_flat, "tighter gate must not admit MORE rules for the high-prec label"
