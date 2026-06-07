"""Unit tests for the RILL correctness fixes (audit bug#14 livelock, #19 seeded RNG).

Golden-neutral: the golden config runs no RILL sweep, so these touch no golden
artifacts. They guard the human-efficiency loop (paper core).
"""
import numpy as np
import pytest
import scipy.sparse as sp
from types import SimpleNamespace

from loris.document import Document
from loris.rill.rill import RILLController, InfluenceEstimator
from loris.rill.oracle import OracleBase


class _MockRDG:
    def bfs_from_label(self, label):
        return []          # no label-rule downstream → isolates the sim-reach term

    def __repr__(self):
        return "MockRDG"


def test_bug1_doc_specific_influence():
    """Influence must depend on the DOC's similarity neighbourhood, not be a
    near-constant over the candidate-label set (audit bug#1 — degenerate
    RILL document selection)."""
    n_docs, n_labels = 5, 2
    tfc = np.zeros((1, n_docs), dtype=bool)            # 1 dummy rule, fires nowhere
    adj = sp.lil_matrix((n_docs, n_docs), dtype=bool)  # doc0 -> {1,2,3}; doc4 -> none
    for j in (1, 2, 3):
        adj[0, j] = True
    est = InfluenceEstimator(_MockRDG(), tfc, n_labels, sim_adjacency=adj.tocsr())
    lbl = SimpleNamespace(pos=np.zeros((n_docs, n_labels), dtype=bool))
    U = np.array([1, 2, 3])
    infl0 = est.estimate_total_influence(0, U, lbl, ["A", "B"])
    infl4 = est.estimate_total_influence(4, U, lbl, ["A", "B"])
    assert infl0 == 3.0, f"doc0 reach should be 3, got {infl0}"
    assert infl4 == 0.0, f"doc4 reach should be 0, got {infl4}"
    assert infl0 > infl4, "denser-neighbourhood doc must outrank an isolated doc"
    # regression: no sim adjacency => sim term absent (legacy base-only behaviour)
    est2 = InfluenceEstimator(_MockRDG(), tfc, n_labels, sim_adjacency=None)
    assert est2.estimate_total_influence(0, U, lbl, ["A", "B"]) == 0.0


def test_greedy_coverage_discount():
    """After a seed covers a neighbourhood, an overlapping doc is discounted
    (audit opt#4/#15 — submodular max-coverage seeding)."""
    n_docs, n_labels = 6, 2
    tfc = np.zeros((1, n_docs), dtype=bool)
    adj = sp.lil_matrix((n_docs, n_docs), dtype=bool)
    for j in (1, 2, 3):     # doc0 -> {1,2,3}
        adj[0, j] = True
    for j in (2, 3, 5):     # doc4 -> {2,3,5}  (overlaps doc0 on 2,3)
        adj[4, j] = True
    est = InfluenceEstimator(_MockRDG(), tfc, n_labels, sim_adjacency=adj.tocsr())
    lbl = SimpleNamespace(pos=np.zeros((n_docs, n_labels), dtype=bool))
    U = np.array([1, 2, 3, 5])
    before = est.estimate_total_influence(4, U, lbl, ["A", "B"])   # reach {2,3,5}=3
    est.mark_covered(0)                                            # covers {0,1,2,3}
    after = est.estimate_total_influence(4, U, lbl, ["A", "B"])    # only {5} left =1
    assert before == 3.0, f"before={before}"
    assert after == 1.0, f"after covering doc0, doc4 should reach only doc5, got {after}"
    assert after < before


class _EmptyOracle(OracleBase):
    """Oracle that returns NO labels for every doc (forces the livelock path)."""
    def __init__(self, label_names):
        self.label_names = list(label_names)
        self.n_calls = 0

    def query(self, doc_idx, doc, label_names):
        self.n_calls += 1
        return []

    def query_with_evidence(self, doc_idx, doc, label_names):
        self.n_calls += 1
        return [], None

    def query_paraphrased(self, doc_idx, doc, label_names, evidence=None):
        return []

    def fallback(self, doc_idx, doc, label_names):
        return None


def _docs(n):
    return [Document(cnt=f"document number {i} about topic", lbl=set()) for i in range(n)]


def test_seeded_rng_reproducible():
    """Two controllers with the same seed produce identical RNG draws (bug#19)."""
    labels = ["A", "B", "C"]
    c1 = RILLController(rules=[], label_names=labels, oracle=_EmptyOracle(labels),
                        max_iterations=1, trust_check=False, verbose=False, seed=7)
    c2 = RILLController(rules=[], label_names=labels, oracle=_EmptyOracle(labels),
                        max_iterations=1, trust_check=False, verbose=False, seed=7)
    c3 = RILLController(rules=[], label_names=labels, oracle=_EmptyOracle(labels),
                        max_iterations=1, trust_check=False, verbose=False, seed=99)
    a = c1.rng.choice(1000, 50, replace=False)
    b = c2.rng.choice(1000, 50, replace=False)
    d = c3.rng.choice(1000, 50, replace=False)
    assert np.array_equal(a, b), "same seed must give identical sampling"
    assert not np.array_equal(a, d), "different seed must differ"


def test_livelock_empty_oracle_terminates():
    """A doc the oracle can't label must be skip-listed, not re-selected forever (bug#14)."""
    labels = ["A", "B", "C"]
    n = 6
    oracle = _EmptyOracle(labels)
    ctrl = RILLController(rules=[], label_names=labels, oracle=oracle,
                          max_iterations=200, trust_check=False, verbose=False, seed=1)
    res = ctrl.run(_docs(n), base_predictions=np.zeros((n, len(labels)), dtype=np.float32))
    # With the fix: each unlabeled doc is queried at most once then skipped, so the
    # loop drains U in ~n iterations instead of spinning to max_iterations.
    assert res.n_queries <= n + 1, f"expected <= {n+1} queries, got {res.n_queries} (livelock?)"
    assert res.n_iterations < 200, f"loop hit max_iterations ({res.n_iterations}) — livelock not fixed"
    assert res.status != "max_iterations", f"status={res.status} indicates spinning"


if __name__ == "__main__":
    test_seeded_rng_reproducible()
    test_livelock_empty_oracle_terminates()
    test_bug1_doc_specific_influence()
    test_greedy_coverage_discount()
    print("RILL fix tests passed")
