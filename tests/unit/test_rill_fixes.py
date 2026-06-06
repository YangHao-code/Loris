"""Unit tests for the RILL correctness fixes (audit bug#14 livelock, #19 seeded RNG).

Golden-neutral: the golden config runs no RILL sweep, so these touch no golden
artifacts. They guard the human-efficiency loop (paper core).
"""
import numpy as np
import pytest

from loris.document import Document
from loris.rill.rill import RILLController
from loris.rill.oracle import OracleBase


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
    print("RILL fix tests passed")
