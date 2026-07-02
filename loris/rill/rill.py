"""
chase_inference/rill.py
------------------------
RILL — Recursive Inference for Logical Labeling (paper §6.2).

When the Chase engine reaches fixpoint with unlabeled documents remaining,
RILL selects the most influential document, queries an oracle for its label,
injects the label, and resumes Chase.  Repeats until all documents are
labeled or a budget is exhausted.

Three main classes:
- ``InfluenceEstimator`` — ranks unlabeled documents by downstream impact.
- ``TrustChecker`` — three-layer defense against oracle hallucination.
- ``RILLController`` — the outer active-learning loop.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from loris.document import Document
from loris.predicates import LabelPredicate, TextualPredicate

from loris.chase.multi_chase import BulkLBL, ChaseResult, MultiChase
from loris.rill.oracle import GroundTruthOracle, OracleBase
from loris.chase.rdg import RuleDependencyGraph

logger = logging.getLogger(__name__)


# =========================================================================
# InfluenceEstimator
# =========================================================================


class InfluenceEstimator:
    """Estimate downstream influence of labeling a document.

    Uses the Rule Dependency Graph and precomputed text-fire cache to
    estimate how many unlabeled documents would be affected by assigning
    a specific label to a document.  All operations are vectorised NumPy
    — no regex or model inference at runtime.

    Parameters
    ----------
    rdg : RuleDependencyGraph
        Static rule dependency graph.
    text_fire_cache : np.ndarray
        ``(n_rules, n_docs)`` bool matrix from MultiChase.
    n_labels : int
        Number of label columns.
    """

    def __init__(
        self,
        rdg: RuleDependencyGraph,
        text_fire_cache: np.ndarray,
        n_labels: int,
        sim_adjacency=None,
    ) -> None:
        self.rdg = rdg
        self.text_fire_cache = text_fire_cache
        self.n_labels = n_labels
        # bug#1: doc-specific cross-document reach. sim_adjacency[x] gives the
        # similarity neighbours of x; labeling x propagates its labels to them,
        # so a doc in a denser *unlabeled* neighbourhood is more valuable to
        # query. Without this, influence depends only on the candidate-label set
        # (≈constant across docs) → degenerate (≈random) document selection.
        self._sim_adj = sim_adjacency
        self._n_docs = int(text_fire_cache.shape[1]) if text_fire_cache is not None else 0
        self._u_fp = None
        self._u_mask_arr = None
        self._covered = None   # opt#4/#15: greedy max-coverage discount mask

        # Pre-compute downstream rule sets for each label (cached)
        self._label_downstream: Dict[str, List[int]] = {}

    def mark_covered(self, doc_idx: int) -> None:
        """Greedy max-coverage (opt#4/#15): after a doc is queried, mark it and
        its similarity neighbours as covered so subsequent influence scores
        discount already-reachable docs (submodular 1−1/e selection) instead of
        re-querying the same dense neighbourhood."""
        if self._sim_adj is None or not self._n_docs:
            return
        if self._covered is None:
            self._covered = np.zeros(self._n_docs, dtype=bool)
        self._covered[doc_idx] = True
        row = self._sim_adj[doc_idx]
        neigh = (row.indices if hasattr(row, "indices")
                 else np.nonzero(np.asarray(row).ravel())[0])
        if len(neigh):
            self._covered[neigh] = True

    def _u_mask(self, U_indices: np.ndarray) -> np.ndarray:
        """Boolean (n_docs,) membership mask for U, rebuilt only when U changes."""
        fp = (len(U_indices),
              int(U_indices[0]) if len(U_indices) else -1,
              int(U_indices[-1]) if len(U_indices) else -1)
        if self._u_fp != fp:
            m = np.zeros(self._n_docs, dtype=bool)
            if len(U_indices):
                m[U_indices] = True
            self._u_mask_arr, self._u_fp = m, fp
        return self._u_mask_arr

    def _get_downstream_rules(self, label: str) -> List[int]:
        """Return list of rule indices reachable from *label* (cached)."""
        if label not in self._label_downstream:
            downstream = self.rdg.bfs_from_label(label)
            self._label_downstream[label] = sorted(downstream)
        return self._label_downstream[label]

    def estimate_influence(
        self,
        label: str,
        U_indices: np.ndarray,
    ) -> int:
        """Estimate how many docs in U would be affected by assigning *label*.

        Implements: ``|spset(φ, D) ∩ U|`` from the paper.

        Parameters
        ----------
        label : str
            The label being considered.
        U_indices : np.ndarray
            Indices of currently-unlabeled documents.

        Returns
        -------
        int
            Number of U docs reachable through downstream rules.
        """
        downstream = self._get_downstream_rules(label)
        if not downstream or len(U_indices) == 0:
            return 0

        # text_fire_cache[downstream_rules, :][:, U_indices].any(axis=0)
        sub_cache = self.text_fire_cache[downstream][:, U_indices]
        reachable = sub_cache.any(axis=0)
        return int(reachable.sum())

    def estimate_total_influence(
        self,
        doc_idx: int,
        U_indices: np.ndarray,
        lbl: BulkLBL,
        label_names: List[str],
    ) -> float:
        """Average influence over all candidate labels for *doc_idx*.

        For each label τ that *doc_idx* does not already possess, compute
        influence and average.

        Parameters
        ----------
        doc_idx : int
            Document being evaluated.
        U_indices : np.ndarray
            Current unlabeled document set.
        lbl : BulkLBL
            Current label state.
        label_names : List[str]
            Ordered label vocabulary.

        Returns
        -------
        float
            Average influence score.
        """
        candidates = [
            (τ_idx, τ_name)
            for τ_idx, τ_name in enumerate(label_names)
            if not lbl.pos[doc_idx, τ_idx]
        ]
        if not candidates:
            return 0.0

        total = 0
        for _, τ_name in candidates:
            total += self.estimate_influence(τ_name, U_indices)
        base = total / len(candidates)
        # bug#1: add doc-SPECIFIC cross-document reach so influence is not
        # ≈constant across docs. Labeling doc_idx propagates its labels to its
        # similarity neighbours, so count how many are still unlabeled (in U).
        if self._sim_adj is not None and self._n_docs and len(U_indices):
            row = self._sim_adj[doc_idx]
            neigh = (row.indices if hasattr(row, "indices")
                     else np.nonzero(np.asarray(row).ravel())[0])
            if len(neigh):
                reachable = self._u_mask(U_indices)[neigh]
                # opt#4/#15: discount neighbours already covered by earlier seeds
                if self._covered is not None:
                    reachable = reachable & ~self._covered[neigh]
                base += float(reachable.sum())
        return base

    def rank_unlabeled(
        self,
        U_indices: np.ndarray,
        lbl: BulkLBL,
        label_names: List[str],
    ) -> np.ndarray:
        """Return U_indices sorted by descending influence score."""
        scores = np.array([
            self.estimate_total_influence(int(x), U_indices, lbl, label_names)
            for x in U_indices
        ])
        order = np.argsort(-scores)
        return U_indices[order]


# =========================================================================
# TrustChecker — three-layer defense against oracle hallucination
# =========================================================================


class TrustChecker:
    """Three-layer trust verification for oracle labels (paper §6.2).

    Layer 1 (Evidence): Verify evidence document E against rules.
    Layer 2 (Stability): Paraphrased re-query must agree with original.
    Layer 3 (Sandbox): Mini-Chase must not produce conflicts.

    ``GroundTruthOracle`` skips Layers 1–2 (no hallucination risk).

    Parameters
    ----------
    rdg : RuleDependencyGraph
        Rule dependency graph.
    text_fire_cache : np.ndarray
        ``(n_rules, n_docs)`` bool.
    label_names : List[str]
        Ordered label vocabulary.
    rules : list
        RDL rule objects.
    max_bfs_depth : int
        BFS depth for sandbox rule selection (default 2).
    """

    def __init__(
        self,
        rdg: RuleDependencyGraph,
        text_fire_cache: np.ndarray,
        label_names: List[str],
        rules: list,
        max_bfs_depth: int = 2,
    ) -> None:
        self.rdg = rdg
        self.text_fire_cache = text_fire_cache
        self.label_names = list(label_names)
        self._label2idx = {n: i for i, n in enumerate(self.label_names)}
        self.rules = list(rules)
        self.max_bfs_depth = max_bfs_depth

    def check(
        self,
        doc_idx: int,
        labels: List[str],
        lbl: BulkLBL,
        oracle: OracleBase,
        doc: Document,
        label_names: List[str],
    ) -> bool:
        """Run trust verification.  Returns ``True`` = reject, ``False`` = accept.

        Parameters
        ----------
        doc_idx : int
            Document being labeled.
        labels : List[str]
            All proposed labels from oracle (one human interaction).
        lbl : BulkLBL
            Current label state.
        oracle : OracleBase
            The oracle that provided the labels.
        doc : Document
            The document being labeled.
        label_names : List[str]
            Label vocabulary.

        Returns
        -------
        bool
            ``True`` if the labels should be **rejected** (untrusted).
        """
        if not labels:
            return True  # no labels → reject

        # Validate all labels
        label_indices = []
        for label in labels:
            idx = self._label2idx.get(label)
            if idx is None:
                logger.warning("TrustChecker: unknown label %r — rejecting", label)
                return True
            label_indices.append(idx)

        # GroundTruthOracle: skip evidence + stability checks
        if not isinstance(oracle, GroundTruthOracle):
            # Step 1: Evidence document consistency (check each label)
            evidence = oracle.get_last_evidence()
            if evidence is not None:
                for label, lidx in zip(labels, label_indices):
                    if not self._check_evidence_consistency(evidence, label, lidx):
                        logger.debug(
                            "TrustChecker: doc %d label %r failed evidence check",
                            doc_idx, label,
                        )
                        return True

            # Step 2: Stability (mandatory for LLM oracles)
            if not self._check_stability(oracle, doc_idx, doc, label_names, labels, evidence):
                logger.debug(
                    "TrustChecker: doc %d labels %r failed stability check",
                    doc_idx, labels,
                )
                return True

        # Step 3: Sandbox mini-Chase (all oracle types) — inject all labels
        return self._check_sandbox_consistency_multi(
            doc_idx, labels, label_indices, lbl,
        )

    # ----- Step 1: Evidence Document Consistency -----

    def _check_evidence_consistency(
        self,
        evidence_text: str,
        label: str,
        label_idx: int,
    ) -> bool:
        """Check if evidence doc E is consistent with label τ.

        Returns ``True`` = consistent, ``False`` = inconsistent (reject).
        """
        evidence_doc = Document(cnt=evidence_text, lbl=set())

        # Collect rules related to this label
        related_rules = set(self.rdg.rules_producing_label(label))
        related_rules.update(self.rdg.rules_depending_on_label(label))

        for r_idx in related_rules:
            rule = self.rules[r_idx]
            # Check text predicates only (label preds depend on E's label state)
            text_preds = [
                p for p in rule.body if isinstance(p, TextualPredicate)
            ]
            if not text_preds:
                continue

            all_text_fire = all(p(evidence_doc) for p in text_preds)
            # If a rule that *produces* this label fires on E → consistent
            if rule.consequence == label and all_text_fire:
                return True

        # No supporting rule fires on E → inconsistent
        return False

    # ----- Step 2: Stability -----

    def _check_stability(
        self,
        oracle: OracleBase,
        doc_idx: int,
        doc: Document,
        label_names: List[str],
        labels: List[str],
        evidence: Optional[str],
    ) -> bool:
        """Paraphrased re-query must agree with original (all labels must match).

        Returns ``True`` = stable, ``False`` = unstable (reject).
        """
        labels_2 = oracle.query_paraphrased(doc_idx, doc, label_names, evidence)
        return set(labels_2) == set(labels)

    # ----- Step 3: Sandbox mini-Chase -----

    def _check_sandbox_consistency_multi(
        self,
        doc_idx: int,
        labels: List[str],
        label_indices: List[int],
        lbl: BulkLBL,
    ) -> bool:
        """Run mini-Chase in a sandbox with multiple labels.

        Returns ``True`` = conflict (reject).
        Uses ``conflict_mode="halt"`` hardcoded.
        """
        # Get local rule subset (2-hop BFS from all labels)
        R_local: Set[int] = set()
        for label in labels:
            R_local |= self.rdg.bfs_from_label(label, max_depth=self.max_bfs_depth)
        if not R_local:
            return False

        R_local_list = sorted(R_local)

        # Collect affected documents
        affected_doc_set: Set[int] = set()
        for r_idx in R_local_list:
            affected_doc_set.update(np.where(self.text_fire_cache[r_idx])[0])
        affected_doc_set.add(doc_idx)

        if not affected_doc_set:
            return False

        D_local = sorted(affected_doc_set)
        D_local_arr = np.array(D_local)

        # Create sandbox copies
        sandbox_pos = lbl.pos[D_local_arr].copy()
        sandbox_neg = lbl.neg[D_local_arr].copy()

        local_idx_map = {d: i for i, d in enumerate(D_local)}
        local_x = local_idx_map.get(doc_idx)
        if local_x is None:
            return False

        # Inject all proposed labels
        for lidx in label_indices:
            sandbox_pos[local_x, lidx] = True
            # Immediate conflict check
            if sandbox_neg[local_x, lidx]:
                return True

        # Mini-chase: iterate label-dependent rules on local docs
        n_local = len(D_local)
        changed = True
        max_iters = 10
        iteration = 0

        while changed and iteration < max_iters:
            changed = False
            iteration += 1

            for r_idx in R_local_list:
                rule = self.rules[r_idx]
                consequence_label_idx = self._label2idx.get(rule.consequence)
                if consequence_label_idx is None:
                    continue

                for local_i in range(n_local):
                    global_d = D_local[local_i]

                    if not self.text_fire_cache[r_idx, global_d]:
                        continue

                    label_preds = [
                        p for p in rule.body if isinstance(p, LabelPredicate)
                    ]
                    label_ok = True
                    for pred in label_preds:
                        if pred.op == "contains":
                            pidx = self._label2idx.get(pred.label)
                            if pidx is None or not sandbox_pos[local_i, pidx]:
                                label_ok = False
                                break
                    if not label_ok:
                        continue

                    op = rule.consequence_op
                    if op == "add":
                        if not sandbox_pos[local_i, consequence_label_idx]:
                            sandbox_pos[local_i, consequence_label_idx] = True
                            changed = True
                    elif op == "remove":
                        if not sandbox_neg[local_i, consequence_label_idx]:
                            sandbox_neg[local_i, consequence_label_idx] = True
                            changed = True

                    # Conflict check — halt immediately (hardcoded)
                    if (sandbox_pos[local_i] & sandbox_neg[local_i]).any():
                        return True

        return False


# =========================================================================
# RILLResult — outcome of the active-learning loop
# =========================================================================


@dataclass
class RILLResult:
    """Result of a RILL active-learning run.

    Attributes
    ----------
    status : str
        ``"complete"`` — all documents labeled.
        ``"max_iterations"`` — budget exhausted.
        ``"no_unlabeled"`` — no unlabeled docs at start.
    predictions : np.ndarray
        ``(n_docs, n_labels)`` float32 multi-hot.
    query_log : List[Dict]
        Per-iteration records: iteration, doc_idx, label, influence, U_size.
    n_iterations : int
        Total RILL iterations executed.
    n_queries : int
        Total oracle queries made.
    chase_result : Optional[ChaseResult]
        The last ChaseResult from the underlying engine.
    """

    status: str
    predictions: np.ndarray
    query_log: List[Dict[str, Any]] = field(default_factory=list)
    n_iterations: int = 0
    n_queries: int = 0
    chase_result: Optional[ChaseResult] = None


# =========================================================================
# RILLController — the outer active-learning loop
# =========================================================================


class RILLController:
    """RILL active-learning controller (paper §6.2).

    When Chase reaches fixpoint with unlabeled documents remaining:
    1. Rank unlabeled docs by influence (via RDG + text_fire_cache).
    2. Query oracle for the most influential document's label.
    3. Trust-check the label (3-layer defense).
    4. Inject label and resume incremental Chase.
    5. Repeat until all docs labeled or budget exhausted.

    Parameters
    ----------
    rules : list
        RDL rule objects.
    label_names : List[str]
        Ordered label vocabulary.
    oracle : OracleBase
        Label oracle (GroundTruthOracle or LLMOracle).
    max_iterations : int
        Maximum RILL iterations (oracle queries).
    trust_check : bool
        Whether to run trust verification.
    trust_bfs_depth : int
        BFS depth for TrustChecker sandbox.
    conflict_mode : str
        Chase conflict mode (default ``"negative_wins"`` for RILL).
    verbose : bool
        Whether to log progress.
    """

    def __init__(
        self,
        rules: list,
        label_names: List[str],
        oracle: OracleBase,
        max_iterations: int = 1000,
        trust_check: bool = True,
        trust_bfs_depth: int = 2,
        conflict_mode: str = "negative_wins",
        verbose: bool = True,
        sim_graphs: Optional[Dict] = None,
        sim_decay: float = 1.0,
        sim_conf_threshold: float = 0.0,
        virtual_attrs: Optional[Dict[str, "np.ndarray"]] = None,
        seed: int = 42,
        active_relabel: bool = False,
    ) -> None:
        self.rules = list(rules)
        self.label_names = list(label_names)
        self.n_labels = len(label_names)
        self._label2idx = {n: i for i, n in enumerate(self.label_names)}
        self.oracle = oracle
        self.max_iterations = max_iterations
        self.trust_check = trust_check
        self.trust_bfs_depth = trust_bfs_depth
        self.conflict_mode = conflict_mode
        self.verbose = verbose
        self.sim_graphs = sim_graphs
        self.sim_decay = sim_decay
        self.sim_conf_threshold = sim_conf_threshold
        self.virtual_attrs = virtual_attrs
        # active_relabel: query the most-influential docs regardless of whether
        # they already carry base+rules labels (the human-cost budget sweep needs
        # to reach any budget B; otherwise U collapses to the few all-zero docs).
        self.active_relabel = active_relabel
        # bug#19/opt#9: seeded RNG so large-U sampling (and thus query order) is
        # reproducible across runs instead of using the global unseeded np.random.
        self.rng = np.random.RandomState(seed)

    def run(
        self,
        docs: List[Document],
        base_predictions: Optional[np.ndarray] = None,
    ) -> RILLResult:
        """Execute the RILL active-learning loop.

        Parameters
        ----------
        docs : List[Document]
            Input documents.
        base_predictions : np.ndarray, optional
            ``(n_docs, n_labels)`` base model predictions.

        Returns
        -------
        RILLResult
        """
        n_docs = len(docs)
        if n_docs == 0:
            return RILLResult(
                status="no_unlabeled",
                predictions=np.zeros((0, self.n_labels), dtype=np.float32),
            )

        # Step 1–2: Initial Chase
        chase = MultiChase(
            rules=self.rules,
            label_names=self.label_names,
            conflict_mode=self.conflict_mode,
            enable_transitivity=True,
            track_provenance=False,
            sim_graphs=self.sim_graphs,
            sim_decay=self.sim_decay,
            sim_conf_threshold=self.sim_conf_threshold,
            virtual_attrs=self.virtual_attrs,
        )
        initial_result = chase.run_persistent(docs, base_predictions)
        lbl = chase.lbl
        text_fire_cache = chase.text_fire_cache

        if self.verbose:
            n_labeled = int((lbl.pos.sum(axis=1) > 0).sum())
            logger.info(
                "RILL: initial Chase done — %d/%d docs labeled, status=%s",
                n_labeled, n_docs, initial_result.status,
            )

        # Step 5: Build RDG
        rdg = RuleDependencyGraph(self.rules, self.label_names)
        if self.verbose:
            logger.info("RILL: RDG built — %s", rdg)

        # Step 6: Estimator & TrustChecker
        # bug#1: combine all sim-graph thresholds into one adjacency so the
        # estimator can score each doc's cross-document propagation reach.
        _sim_adj = None
        if self.sim_graphs:
            _mats = [g for g in self.sim_graphs.values() if g is not None]
            if _mats:
                _acc = _mats[0].astype(bool)
                for _g in _mats[1:]:
                    _acc = _acc + _g.astype(bool)
                _sim_adj = _acc.tocsr()
        estimator = InfluenceEstimator(rdg, text_fire_cache, self.n_labels,
                                       sim_adjacency=_sim_adj)
        trust_checker = TrustChecker(
            rdg, text_fire_cache, self.label_names, self.rules,
            max_bfs_depth=self.trust_bfs_depth,
        )

        # Step 8: Identify unlabeled documents
        # U = docs with no positive labels AND whose neg hasn't exhausted all labels.
        # active_relabel: candidates = ALL docs not yet queried (so the human-cost
        # sweep can query up to any budget B, even on base+rules-labeled docs).
        queried: set = set()
        if self.active_relabel:
            U = np.arange(n_docs)
        else:
            U = np.where(
                (lbl.pos.sum(axis=1) == 0) & (lbl.neg.sum(axis=1) < self.n_labels)
            )[0]

        # Skip set: docs where trust check failed and fallback returned None
        skip_set: set = set()

        query_log: List[Dict[str, Any]] = []
        iteration = 0
        n_queries = 0

        if self.verbose:
            logger.info("RILL: starting loop — %d unlabeled docs", len(U))

        # Step 9: Main loop
        while len(U) > 0 and iteration < self.max_iterations:
            # Step 10–12: Rank by influence, pick best
            if len(U) <= 50:
                # Small U: compute exact scores
                scores = np.array([
                    estimator.estimate_total_influence(
                        int(x), U, lbl, self.label_names,
                    )
                    for x in U
                ])
            else:
                # Large U: sample a subset for scoring efficiency
                sample_size = min(200, len(U))
                sample_idx = self.rng.choice(len(U), sample_size, replace=False)
                sample_U = U[sample_idx]
                scores = np.array([
                    estimator.estimate_total_influence(
                        int(x), U, lbl, self.label_names,
                    )
                    for x in sample_U
                ])
                best_in_sample = sample_idx[np.argmax(scores)]
                # Map back
                U_reordered = np.concatenate([
                    [U[best_in_sample]],
                    np.delete(U, best_in_sample),
                ])
                scores = np.array([scores[np.argmax(scores)]])
                U_for_pick = U_reordered
                x_star = int(U_for_pick[0])
                influence = float(scores[0])

            if len(U) <= 50:
                best_idx = int(np.argmax(scores))
                x_star = int(U[best_idx])
                influence = float(scores[best_idx])

            # Step 14–15: Query oracle — returns ALL labels for the doc
            #   (one human interaction = one query)
            labels_star, evidence = self.oracle.query_with_evidence(
                x_star, docs[x_star], self.label_names,
            )
            n_queries += 1

            # Step 17–22: Trust check
            rejected = False
            if self.trust_check and labels_star:
                if trust_checker.check(
                    x_star, labels_star, lbl, self.oracle,
                    docs[x_star], self.label_names,
                ):
                    # Trust check failed — try fallback
                    fallback_labels = self.oracle.fallback(
                        x_star, docs[x_star], self.label_names,
                    )
                    if fallback_labels is None:
                        skip_set.add(x_star)
                        rejected = True
                        if self.verbose:
                            logger.debug(
                                "RILL iter %d: doc %d labels %r rejected, "
                                "no fallback — skipping",
                                iteration, x_star, labels_star,
                            )
                    else:
                        labels_star = fallback_labels

            if not rejected:
                # Step 24–25: Inject ALL labels and resume Chase
                label_indices = [
                    self._label2idx[τ]
                    for τ in labels_star
                    if τ in self._label2idx
                ] if labels_star else []
                if label_indices:
                    chase.inject_labels_and_resume(x_star, label_indices)
                    estimator.mark_covered(x_star)   # opt#4/#15 greedy coverage

                    if self.verbose and iteration % 50 == 0:
                        n_labeled = int((lbl.pos.sum(axis=1) > 0).sum())
                        logger.info(
                            "RILL iter %d: queried doc %d → %r "
                            "(influence=%.1f), %d/%d labeled, %d unlabeled",
                            iteration, x_star, labels_star, influence,
                            n_labeled, n_docs, len(U),
                        )
                else:
                    # bug#14: oracle returned no injectable label (empty, or all
                    # out-of-vocab) — skip x_star, else it is never given a pos
                    # label, stays in U, and is re-selected every iteration
                    # (livelock, especially under constant influence #1).
                    skip_set.add(x_star)

            # Step 27–29: Update U
            queried.add(x_star)
            if self.active_relabel:
                # candidates = all docs not yet queried (relabel-any mode)
                U = np.array([i for i in range(n_docs)
                              if i not in queried and i not in skip_set])
            else:
                U = np.where(
                    (lbl.pos.sum(axis=1) == 0) & (lbl.neg.sum(axis=1) < self.n_labels)
                )[0]
                # Exclude skipped docs
                if skip_set:
                    U = np.array([x for x in U if x not in skip_set])

            # Log
            query_log.append({
                "iteration": iteration,
                "doc_idx": x_star,
                "labels": labels_star,
                "n_labels_injected": len(labels_star) if not rejected else 0,
                "influence": influence,
                "U_size": len(U),
                "rejected": rejected,
            })

            iteration += 1

        # Final status
        if len(U) == 0:
            status = "complete"
        else:
            status = "max_iterations"

        # B6 fix: finalise via the chase's conflict-aware builder (pos & ~neg),
        # not pos alone — otherwise every REMOVE rule's suppression is silently
        # dropped from RILL's output, inconsistent with the rest of the pipeline.
        predictions = chase._build_predictions(chase.lbl)

        if self.verbose:
            n_labeled = int((lbl.pos.sum(axis=1) > 0).sum())
            logger.info(
                "RILL finished: status=%s, %d iterations, %d queries, "
                "%d/%d docs labeled, %d skipped",
                status, iteration, n_queries, n_labeled, n_docs, len(skip_set),
            )

        return RILLResult(
            status=status,
            predictions=predictions,
            query_log=query_log,
            n_iterations=iteration,
            n_queries=n_queries,
            chase_result=initial_result,
        )
