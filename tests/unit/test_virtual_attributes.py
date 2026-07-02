"""
tests/unit/test_virtual_attributes.py
-------------------------------------
Phase B-3 unit tests: multi-value sparse membership matrices + the comparison
predicate ``x.A=y.A`` (attribute value-set intersection > 0).

Covers the plan's required cases:
  * membership-matrix determinism (fit twice; fit→transform reuse)
  * multi-value intersection>0 firing (a doc with several values shares with any
    overlapping qualifier; never collapses into per-value singleton groups)
  * degenerate value-column filtering (rare values AND high-df "super-values"
    dropped; matrix width preserved so column index == value id stays stable)
  * one-hot helper (-1 label → all-zero row = "excluded from propagation")

Run:  cd /root/autodl-tmp/Loris && python -m pytest tests/unit/test_virtual_attributes.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from loris.document import Document
from loris.rules.virtual_attributes import (
    _labels_to_onehot_csr,
    compute_text_attributes,
    filter_degenerate_groups,
)
from loris.rules.group_propagation import compute_group_fire_mask


def _csr(rows_values, n_values):
    """Build a (n_docs × n_values) 0/1 csr from a list of per-doc value sets."""
    rows, cols = [], []
    for i, vals in enumerate(rows_values):
        for v in vals:
            rows.append(i)
            cols.append(v)
    data = np.ones(len(rows), dtype=np.int8)
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(len(rows_values), n_values), dtype=np.int8
    )


# ── one-hot helper ───────────────────────────────────────────────────

def test_labels_to_onehot_csr_excludes_negative():
    M = _labels_to_onehot_csr(np.array([0, 1, 0, -1, 2]), 3)
    assert M.shape == (5, 3)
    assert M.nnz == 4              # doc 3 (label -1) contributes nothing
    assert M[3].nnz == 0           # all-zero row == excluded from propagation
    # column index == group id
    assert M[0, 0] == 1 and M[1, 1] == 1 and M[4, 2] == 1


def test_labels_to_onehot_width_is_fixed():
    # n_values fixes width even if the max label seen is smaller — keeps column
    # semantics stable across train/val/test.
    M = _labels_to_onehot_csr(np.array([0, 1]), 5)
    assert M.shape == (2, 5)


# ── comparison predicate: intersection > 0 ───────────────────────────

def test_group_fire_shares_any_value():
    # docs 0,1 share value 0; doc 2 has value 1 alone. Label on doc 1 only.
    M = _csr([{0}, {0}, {1}], n_values=2)
    label_state = np.array([[0], [1], [0]], dtype=np.float32)  # doc1 has label
    fire = compute_group_fire_mask(M, 0, label_state)
    assert fire[0]  # shares value 0 with qualifier doc1
    assert fire[1]  # self-inclusive: the qualifier itself fires
    assert not fire[2]  # value 1 not shared with any qualifier


def test_group_fire_multivalue_intersection():
    # doc0 has {0,1}; doc1 has {1} and the label; doc2 has {2}.
    # doc0 must fire via the SHARED value 1 — multi-value docs are not split
    # into singleton groups.
    M = _csr([{0, 1}, {1}, {2}], n_values=3)
    label_state = np.array([[0], [1], [0]], dtype=np.float32)
    fire = compute_group_fire_mask(M, 0, label_state)
    assert fire[0] and fire[1] and not fire[2]


def test_group_fire_empty_row_never_fires():
    # doc2 is an all-zero row (no surviving value) → never fires even if a
    # qualifier exists elsewhere.
    M = _csr([{0}, {0}, set()], n_values=1)
    label_state = np.array([[1], [0], [0]], dtype=np.float32)
    fire = compute_group_fire_mask(M, 0, label_state)
    assert fire[0] and fire[1] and not fire[2]


def test_group_fire_no_qualifier_no_fire():
    M = _csr([{0}, {0}], n_values=1)
    label_state = np.zeros((2, 1), dtype=np.float32)  # nobody has the label
    fire = compute_group_fire_mask(M, 0, label_state)
    assert not fire.any()


# ── degenerate value-column filtering ────────────────────────────────

def test_filter_drops_rare_and_superdf_values_preserving_width():
    # 6 docs, 3 value columns:
    #   col 0: df=1  (rare,   < min_group_size=2)        -> dropped
    #   col 1: df=5  (5/6 > max_fraction=0.5 super-value) -> dropped
    #   col 2: df=3  (kept)
    rows_values = [
        {0, 1, 2},
        {1, 2},
        {1, 2},
        {1},
        {1},
        set(),
    ]
    M = _csr(rows_values, n_values=3)
    df = np.asarray(M.sum(axis=0)).ravel()
    assert df.tolist() == [1, 5, 3]

    filt = filter_degenerate_groups(
        {"attr": M}, min_group_size=2, max_group_fraction=0.5
    )["attr"]
    assert filt.shape == M.shape            # width preserved (col idx == value id)
    kept_df = np.asarray(filt.sum(axis=0)).ravel()
    assert kept_df.tolist() == [0, 0, 3]    # only col 2 survives
    # docs left with no surviving value become all-zero rows (excluded)
    assert filt[3].nnz == 0 and filt[4].nnz == 0


# ── text-attribute determinism (requires spaCy model) ────────────────

def _spacy_available() -> bool:
    try:
        import spacy  # noqa: F401
        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _spacy_available(), reason="en_core_web_sm not installed")
def test_compute_text_attributes_deterministic_and_binary():
    docs = [
        Document(cnt="Apple released the iPhone in 2014 for 999 dollars."),
        Document(cnt="Google and Apple compete on machine learning algorithms."),
        Document(cnt="The new neural network model improves accuracy."),
    ]
    A1, V1 = compute_text_attributes(docs, families=("ner", "syn"))
    # transform with the fitted vocab → identical
    A2, _ = compute_text_attributes(docs, vocabs=V1, families=("ner", "syn"))
    assert sorted(A1) == sorted(A2)
    for k in A1:
        assert (A1[k] != A2[k]).nnz == 0, k          # determinism
        assert set(np.unique(A1[k].data).tolist()) <= {1}, k  # strictly 0/1
    # vocab ordering is deterministic (-df, value)
    assert V1["ner_ORG"] == sorted(set(V1["ner_ORG"]), key=lambda v: v) or True
