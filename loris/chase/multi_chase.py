"""
chase_inference/multi_chase.py
-------------------------------
Multi-Label Chase algorithm with NumPy bitmap acceleration.

Implements the paper's chase semantics:
  - Label-equivalence classes: [x.lbl]_= (pos), [x.lbl]_≠ (neg), [x]_⊆ (sub)
  - Incremental queue-based evaluation (only re-evaluate affected docs)
  - Conflict detection (lbl_pos ∩ lbl_neg ≠ ∅)
  - Transitivity propagation via reverse index
  - Fixpoint termination

Design notes
------------
* lbl_pos / lbl_neg are stored as (n_docs, n_labels) bool matrices in BulkLBL
  for vectorised set operations.
* Text/ML predicates are evaluated once (immutable) and cached; only label-
  dependent predicates are re-evaluated each round.
* Queue items are (doc_idx, label_idx, rule_idx) tuples.
* Borrowing incremental patterns from LBoost's prob_gar_chase.h:
  new-item tracking, delta-document filtering, evaluated-set deduplication.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from loris.document import Document
import scipy.sparse as sp

from loris.predicates import (
    GroupPredicate,
    LabelPredicate,
    MLPredicate,
    MLThresholdPredicate,
    Predicate,
    SimPredicate,
    TextualPredicate,
)

logger = logging.getLogger(__name__)

# Queue item sentinel rule indices
_SEED_RULE = -1          # seeded from base_predictions / doc.lbl
_TRANSITIVITY_RULE = -2  # propagated via subset transitivity
_EQUAL_RULE = -3         # copied via comparison consequence x.lbl=y.lbl


# =========================================================================
# BulkLBL — vectorised label state for all documents
# =========================================================================

@dataclass
class BulkLBL:
    """Vectorised label-equivalence-class state for all documents.

    Attributes
    ----------
    pos : np.ndarray
        (n_docs, n_labels) bool — positive labels ``[x.lbl]_=``.
    neg : np.ndarray
        (n_docs, n_labels) bool — excluded labels ``[x.lbl]_≠``.
    sub : List[Set[int]]
        Per-doc forward subset refs: ``sub[x]`` = {y : y.lbl ⊆ x.lbl}.
    sup : List[Set[int]]
        Per-doc reverse subset refs: ``sup[y]`` = {x : y ∈ sub[x]}.
    """

    pos: np.ndarray
    neg: np.ndarray
    sub: List[Set[int]]
    sup: List[Set[int]]

    @staticmethod
    def create(n_docs: int, n_labels: int) -> "BulkLBL":
        return BulkLBL(
            pos=np.zeros((n_docs, n_labels), dtype=bool),
            neg=np.zeros((n_docs, n_labels), dtype=bool),
            sub=[set() for _ in range(n_docs)],
            sup=[set() for _ in range(n_docs)],
        )

    def has_conflict(self, doc_idx: int) -> np.ndarray:
        """Return bool vector of conflicting labels for *doc_idx*."""
        return self.pos[doc_idx] & self.neg[doc_idx]

    def any_conflict(self, doc_indices: np.ndarray) -> bool:
        """Fast check: any conflict among *doc_indices*?"""
        return bool(np.any(self.pos[doc_indices] & self.neg[doc_indices]))


# =========================================================================
# ChaseResult — outcome container
# =========================================================================

@dataclass
class ChaseResult:
    """Result of a multi-label chase run.

    Attributes
    ----------
    status : str
        ``"fixpoint"`` (normal convergence), ``"conflict"`` (halted on
        contradiction), or ``"max_rounds"`` / ``"timeout"``.
    predictions : np.ndarray
        (n_docs, n_labels) float32 multi-hot prediction matrix.
    conflicts : List[Tuple[int, str]]
        (doc_idx, label_name) pairs that were in conflict.
    n_rounds : int
        Number of chase rounds executed.
    provenance : Dict[Tuple[int, str], List[int]]
        Optional mapping ``(doc_idx, label_name) → [rule_indices]``
        that derived that label.  Empty if ``track_provenance=False``.
    """

    status: str
    predictions: np.ndarray
    conflicts: List[Tuple[int, str]]
    n_rounds: int
    provenance: Dict[Tuple[int, str], List[int]] = field(default_factory=dict)


# =========================================================================
# MultiChase — the chase engine
# =========================================================================

# Type alias for queue items: (doc_idx, label_idx, rule_idx)
_QItem = Tuple[int, int, int]


class MultiChase:
    """Multi-label chase with incremental queue and NumPy bitmap state.

    Parameters
    ----------
    rules : List
        List of RDL rule objects (from ``rule_discovery``).
    label_names : List[str]
        Ordered label vocabulary (determines column indices).
    max_rounds : int
        Hard cap on chase iterations.
    time_limit_sec : float or None
        Wall-clock time limit (None = unlimited).
    conflict_mode : str
        ``"halt"`` — return ⊥ on first conflict (Church-Rosser ⊥ output).
        ``"negative_wins"`` (DEFAULT, C-8) — continue past conflicts; final
            labels = ``pos & ~neg`` (a removed label suppresses a conflicting
            add). The paper requires the chase to CONVERGE rather than halt on
            conflict, and this dual-set resolution preserves corrective remove
            rules and matches the fast-path test evaluator's semantics.
        ``"positive_wins"`` — continue, but final labels = ``pos`` only, i.e.
            negations are dropped entirely (a removed label is re-asserted if
            anything added it). NOT the default: it nullifies every remove rule.
    enable_transitivity : bool
        Whether to maintain and propagate ``[x]_⊆`` subset relations. Defaults
        to ``False`` (B-5): the only former source of ``sub``/``sup`` was the
        prediction-bitmap inference (the B1 bug), now removed. A legitimate
        co-membership source + negative transitivity is added in C-9, which will
        re-enable this. The ``=`` comparison consequence (B-4) propagates labels
        via SpMV closure independently of this flag.
    track_provenance : bool
        Record which rules derived which labels.
    """

    def __init__(
        self,
        rules: List,
        label_names: List[str],
        max_rounds: int = 100,
        time_limit_sec: Optional[float] = None,
        conflict_mode: str = "negative_wins",
        enable_transitivity: bool = False,
        track_provenance: bool = False,
        sim_graphs: Optional[Dict[float, sp.csr_matrix]] = None,
        sim_decay: float = 1.0,
        sim_conf_threshold: float = 0.0,
        virtual_attrs: Optional[Dict[str, sp.csr_matrix]] = None,
        disable_incremental: bool = False,
    ) -> None:
        if conflict_mode not in ("halt", "negative_wins", "positive_wins"):
            raise ValueError(f"Invalid conflict_mode: {conflict_mode!r}")

        self.rules = list(rules)
        self.label_names = list(label_names)
        self.n_labels = len(label_names)
        self._label2idx: Dict[str, int] = {
            name: i for i, name in enumerate(label_names)
        }
        self.max_rounds = max_rounds
        self.time_limit_sec = time_limit_sec
        self.conflict_mode = conflict_mode
        self.enable_transitivity = enable_transitivity
        self.track_provenance = track_provenance
        self.sim_graphs = sim_graphs or {}
        self.sim_decay = sim_decay
        self.sim_conf_threshold = sim_conf_threshold
        self.virtual_attrs = virtual_attrs or {}
        # noInc ablation: bypass the incremental affected-docs/evaluated optimisation
        # and re-scan ALL rules over ALL docs each round (same fixpoint, slower).
        self.disable_incremental = disable_incremental

        # Pre-classify rules
        self._text_rule_idxs: List[int] = []
        self._label_rule_idxs: List[int] = []
        self._sim_rule_idxs: List[int] = []
        self._group_rule_idxs: List[int] = []
        self._equal_rule_idxs: List[int] = []
        self._classify_rules()

    @classmethod
    def from_rdlset(cls, rdl_set, **kwargs) -> "MultiChase":
        """Construct from an existing ``RDLSet``."""
        return cls(
            rules=rdl_set.rules,
            label_names=rdl_set.label_names,
            **kwargs,
        )

    # -----------------------------------------------------------------
    # Rule classification
    # -----------------------------------------------------------------

    def _classify_rules(self) -> None:
        """Split rules into equal-, group-, sim-, label-, and text-dependent.

        Equal rules (consequence_op="equal", comparison consequence x.lbl=y.lbl)
        are checked FIRST: although their body holds a GroupPredicate, they are
        NOT add-rules — they copy y's whole label set via _eval_equal_rules, and
        must not fall into _group_rule_idxs (whose consequence is a single label;
        an equal-rule's consequence is "" → would be silently dropped there).
        Sim rules (SimPredicate) are evaluated via SpMV each round. Group rules
        (GroupPredicate) are evaluated via group membership; they may also contain
        LabelPredicate (checked on neighbour y, not x). Pure label rules use the
        existing per-doc check.
        """
        for idx, rule in enumerate(self.rules):
            has_sim = any(isinstance(p, SimPredicate) for p in rule.body)
            has_group = any(isinstance(p, GroupPredicate) for p in rule.body)
            has_label = any(isinstance(p, LabelPredicate) for p in rule.body)
            if rule.consequence_op == "equal":
                self._equal_rule_idxs.append(idx)
            elif has_group:
                self._group_rule_idxs.append(idx)
            elif has_sim:
                self._sim_rule_idxs.append(idx)
            elif has_label:
                self._label_rule_idxs.append(idx)
            else:
                self._text_rule_idxs.append(idx)

    # -----------------------------------------------------------------
    # Text / ML predicate fire cache
    # -----------------------------------------------------------------

    def _precompute_text_fires(
        self, docs: List[Document],
    ) -> np.ndarray:
        """Compute (n_rules, n_docs) bool cache for non-label predicates.

        For text-only rules, the entire body is cached (all predicates are
        immutable).  For label-dependent rules, only the non-label predicates
        are cached (label predicates are re-evaluated per round).

        Optimization: precompute per-unique-predicate masks first, then AND
        them per rule.  This avoids redundant per-doc evaluation when many
        rules share the same text/ML predicates.
        """
        n_rules = len(self.rules)
        n_docs = len(docs)
        _t0 = time.monotonic()

        # Step 1: collect all unique text predicates across all rules
        _pred_key_to_mask: Dict[str, np.ndarray] = {}
        _all_unique_preds: Dict[str, "Predicate"] = {}  # key → pred object
        for rule in self.rules:
            for p in rule.body:
                if isinstance(p, (LabelPredicate, SimPredicate, GroupPredicate)):
                    continue
                key = str(p)
                if key not in _all_unique_preds:
                    _all_unique_preds[key] = p

        # Step 2: evaluate each unique predicate once on all docs.
        # Env-gated fast-path (CHASE_BATCH_ML): MLThresholdPredicate.__call__ does one model
        # forward *per doc*; when M is an LLM (Mistral-in-pool) that is intractable on a big
        # test chase. Batch those predicates per model — one batched predict_proba over all
        # docs, then threshold. Output is identical (proba[:,label] >= t, same as __call__);
        # off by default so golden/other runs are unchanged.
        _batched_keys: set = set()
        if os.environ.get("CHASE_BATCH_ML"):
            try:
                from loris.predicates._core import MLThresholdPredicate, _get_ml_model
                by_model: Dict[str, list] = {}
                for key, pred in _all_unique_preds.items():
                    if isinstance(pred, MLThresholdPredicate):
                        by_model.setdefault(pred.model_name, []).append((key, pred))
                if by_model:
                    texts = [d.cnt for d in docs]
                    for mn, plist in by_model.items():
                        model = _get_ml_model(mn)
                        clf = getattr(model, "_clf", model)
                        proba = np.asarray(clf.predict_proba(texts), dtype=np.float64)
                        if proba.ndim == 1:
                            proba = proba.reshape(n_docs, -1)
                        for key, pred in plist:
                            li = model.label_index(pred.label)
                            _pred_key_to_mask[key] = (proba[:, li] >= pred.threshold)
                            _batched_keys.add(key)
                    logger.info(
                        "  [CHASE_BATCH_ML] batched %d ML predicate masks over %d model(s)",
                        len(_batched_keys), len(by_model),
                    )
            except Exception as _e:  # pragma: no cover — fall back to per-doc
                logger.warning(
                    "  [CHASE_BATCH_ML] fast-path failed (%s); using per-doc eval", _e
                )
                _batched_keys = set()

        _todo = [(k, p) for k, p in _all_unique_preds.items() if k not in _batched_keys]
        logger.info(
            "  Computing %d unique predicate masks on %d docs ...",
            len(_todo), n_docs,
        )
        for i, (key, pred) in enumerate(_todo):
            mask = np.array([bool(pred(doc)) for doc in docs], dtype=bool)
            _pred_key_to_mask[key] = mask
            if (i + 1) % 50 == 0 or i == len(_todo) - 1:
                _elapsed = time.monotonic() - _t0
                logger.info(
                    "  predicate masks: %d/%d (%.1fs elapsed)",
                    i + 1, len(_todo), _elapsed,
                )

        # Step 3: AND per-predicate masks to build per-rule cache
        cache = np.ones((n_rules, n_docs), dtype=bool)
        for rule_idx, rule in enumerate(self.rules):
            text_preds = [
                p for p in rule.body
                if not isinstance(p, (LabelPredicate, SimPredicate, GroupPredicate))
            ]
            if not text_preds:
                continue
            for p in text_preds:
                cache[rule_idx] &= _pred_key_to_mask[str(p)]

        logger.info(
            "Text fire cache done: %d rules, %d unique predicates, %d docs (%.1fs)",
            n_rules, len(_all_unique_preds), n_docs, time.monotonic() - _t0,
        )
        return cache

    # -----------------------------------------------------------------
    # Label predicate evaluation against BulkLBL state
    # -----------------------------------------------------------------

    def _check_label_predicates(
        self, rule, lbl: BulkLBL, doc_idx: int,
    ) -> bool:
        """Evaluate label predicates in *rule.body* against BulkLBL state.

        Uses a virtual label set (snapshot of pos) so that chained ``minus``
        predicates within the same body don't corrupt the state.
        """
        label_preds = [
            p for p in rule.body if isinstance(p, LabelPredicate)
        ]
        if not label_preds:
            return True

        # Virtual label set for "minus" side-effect simulation
        # Convert bitmap to string set only when needed
        virtual_pos: Optional[Set[str]] = None

        for pred in label_preds:
            if pred.op == "contains":
                lidx = self._label2idx.get(pred.label)
                if lidx is None or not lbl.pos[doc_idx, lidx]:
                    return False

            elif pred.op == "minus":
                # Need the virtual set for chained minus
                if virtual_pos is None:
                    virtual_pos = set(
                        self.label_names[j]
                        for j in range(self.n_labels)
                        if lbl.pos[doc_idx, j]
                    )
                if pred.label not in virtual_pos:
                    return False
                virtual_pos.discard(pred.label)

            elif pred.op in ("eq", "subset", "strict_subset"):
                # Need the full label set
                if virtual_pos is None:
                    virtual_pos = set(
                        self.label_names[j]
                        for j in range(self.n_labels)
                        if lbl.pos[doc_idx, j]
                    )
                target = pred._label_set()
                if pred.op == "eq":
                    if virtual_pos != target:
                        return False
                elif pred.op == "subset":
                    if not (virtual_pos <= target):
                        return False
                elif pred.op == "strict_subset":
                    if not (len(virtual_pos) > 0 and virtual_pos < target):
                        return False

        return True

    # -----------------------------------------------------------------
    # SimPredicate evaluation (SpMV batch)
    # -----------------------------------------------------------------

    def _eval_sim_rules(
        self,
        lbl: BulkLBL,
        text_fire_cache: np.ndarray,
        evaluated: Set[Tuple[int, int, int]],
    ) -> List[_QItem]:
        """Batch-evaluate all sim rules via SpMV.

        For each sim rule:
          1. Build y_mask from LabelPredicate (check **y**'s labels)
          2. SpMV: has_qualifying_neighbor = (adj @ y_mask) > 0
          3. AND with text_fire_cache[rule_idx] (text preds on **x**)
          4. Collect (doc_idx, label_idx, rule_idx) fires

        Returns list of queue items to enqueue.
        """
        fires: List[_QItem] = []
        for rule_idx in self._sim_rule_idxs:
            rule = self.rules[rule_idx]
            consequence_lidx = self._label2idx.get(rule.consequence)
            if consequence_lidx is None:
                continue

            # Lever D symmetry: enforce ALL SimPredicates conjunctively (was
            # next() → first only). Single-SimPredicate bodies are unchanged.
            sim_preds = [p for p in rule.body if isinstance(p, SimPredicate)]
            adjs = [self.sim_graphs.get(sp.threshold) for sp in sim_preds]
            if any(a is None for a in adjs):
                continue

            # Step 1: y_mask — which docs y satisfy LabelPredicate?
            label_preds = [p for p in rule.body if isinstance(p, LabelPredicate)]
            n_docs = lbl.pos.shape[0]
            y_mask = np.ones(n_docs, dtype=np.float32)
            for lp in label_preds:
                lidx = self._label2idx.get(lp.label)
                if lidx is None:
                    y_mask[:] = 0
                    break
                if lp.op == "contains":
                    y_mask *= lbl.pos[:, lidx].astype(np.float32)

            # Step 2: SpMV — which x have qualifying neighbours under EVERY graph?
            has_neighbor = None
            for adj in adjs:
                _hn = np.asarray((adj @ y_mask) > 0).ravel()
                has_neighbor = _hn if has_neighbor is None else (has_neighbor & _hn)

            # Step 3: AND with text predicates on x
            combined = text_fire_cache[rule_idx] & has_neighbor

            # Step 4: exclude docs that already have the consequence label
            combined &= ~lbl.pos[:, consequence_lidx]

            for doc_idx in np.where(combined)[0]:
                key = (int(doc_idx), consequence_lidx, rule_idx)
                if key not in evaluated:
                    fires.append(key)

        return fires

    # -----------------------------------------------------------------
    # GroupPredicate evaluation (attribute equality batch)
    # -----------------------------------------------------------------

    def _eval_group_rules(
        self,
        lbl: BulkLBL,
        text_fire_cache: np.ndarray,
        evaluated: Set[Tuple[int, int, int]],
    ) -> List[_QItem]:
        """Batch-evaluate all group rules via the comparison predicate x.A=y.A.

        For each group rule:
          1. Get the membership matrix (n_docs × n_values csr) from
             virtual_attrs[attr_name]
          2. Build y_mask from LabelPredicate (check y's labels)
          3. SpMV: a doc x fires iff it shares >=1 attribute value with some
             y_mask qualifier — membership @ (membershipᵀ @ y_mask) > 0
          4. AND with text_fire_cache[rule_idx] (text preds on x)
          5. Collect (doc_idx, label_idx, rule_idx) fires
        """
        fires: List[_QItem] = []
        for rule_idx in self._group_rule_idxs:
            rule = self.rules[rule_idx]
            consequence_lidx = self._label2idx.get(rule.consequence)
            if consequence_lidx is None:
                continue

            # Lever D: a rule may carry MORE THAN ONE GroupPredicate
            # (x.A=y.A ∧ x.B=y.B → +τ). Enforce ALL of them conjunctively — the
            # historical code took only the first via next(), silently ignoring a
            # 2nd literal. For a single-GroupPredicate body this loop is
            # bit-identical to the old behaviour (golden-neutral).
            group_preds = [p for p in rule.body if isinstance(p, GroupPredicate)]
            if not group_preds:
                continue

            # Step 1: y_mask — which docs y satisfy LabelPredicate?
            label_preds = [p for p in rule.body if isinstance(p, LabelPredicate)]
            n_docs = lbl.pos.shape[0]
            y_mask = np.ones(n_docs, dtype=bool)
            for lp in label_preds:
                lidx = self._label2idx.get(lp.label)
                if lidx is None:
                    y_mask[:] = False
                    break
                if lp.op == "contains":
                    y_mask &= lbl.pos[:, lidx].astype(bool)

            # Step 2: comparison-predicate fire via SpMV (x.A=y.A ∧ label∈y.lbl),
            # AND'd across every group literal. `membership` is a
            # (n_docs × n_values) 0/1 csr; doc x fires iff it shares >=1 attribute
            # value with some y_mask qualifier. Never materializes the
            # n_docs×n_docs co-membership matrix:
            #   col  = membershipᵀ @ y_mask    (qualifiers per value)
            #   fire = (membership @ col) > 0  (shares a value with a qualifier)
            # Mirrors group_propagation.compute_group_fire_mask; self-inclusion is
            # intentional (Step 4 excludes docs already holding the consequence).
            group_fire = None
            _missing = False
            for gp in group_preds:
                membership = self.virtual_attrs.get(gp.attr_name)
                if membership is None:
                    _missing = True
                    break
                _col = membership.T.dot(y_mask.astype(np.float32))
                _gf = np.asarray(membership.dot(_col)).ravel() > 0
                group_fire = _gf if group_fire is None else (group_fire & _gf)
            if _missing or group_fire is None:
                continue

            # Step 3: AND with text predicates on x
            combined = text_fire_cache[rule_idx] & group_fire

            # Step 4: exclude docs that already have the consequence label
            combined &= ~lbl.pos[:, consequence_lidx]

            for doc_idx in np.where(combined)[0]:
                key = (int(doc_idx), consequence_lidx, rule_idx)
                if key not in evaluated:
                    fires.append(key)

        return fires

    # -----------------------------------------------------------------
    # Comparison consequence: x.lbl = y.lbl  (consequence_op="equal")
    # -----------------------------------------------------------------

    def _eval_equal_rules(
        self,
        lbl: BulkLBL,
        queue: Deque[_QItem],
    ) -> Set[int]:
        """Apply all comparison-consequence (``x.lbl = y.lbl``) rules in batch.

        Paper §6.1 (chase step, condition 2, ⊗ = "="): for every pair of docs
        (x, y) sharing >=1 value of the rule's attribute A (``x.A = y.A``), copy
        y's positive labels into x (and, symmetrically, x's into y). Iterated to
        fixpoint over the chase rounds, every doc in a co-membership connected
        component ends with the union of that component's positive labels.

        Implemented as one batched SpMV per equal-rule — generalizing
        :func:`group_propagation.compute_group_fire_mask` from a single label
        column to the full label matrix, never materializing the n_docs×n_docs
        co-membership matrix::

            new_pos = (membership @ (membershipᵀ @ pos)) > 0   # per (doc, label)

        ``new_pos[i, l]`` is True iff some co-member of doc i (including i itself
        — self-inclusion is intentional and idempotent) currently has label l.
        We OR this into ``lbl.pos`` (monotone — bits are only set, never cleared,
        which guarantees termination and Church-Rosser) and enqueue every newly
        set (doc, label) so the surrounding round loop re-runs to fixpoint and
        the existing conflict / transitivity machinery sees the new labels.

        Returns the set of docs whose ``pos`` row gained at least one label.
        """
        affected: Set[int] = set()
        for rule_idx in self._equal_rule_idxs:
            rule = self.rules[rule_idx]
            group_pred = next(
                (p for p in rule.body if isinstance(p, GroupPredicate)), None
            )
            if group_pred is None:
                continue
            membership = self.virtual_attrs.get(group_pred.attr_name)
            if membership is None:
                continue

            pos = lbl.pos.astype(np.float32)
            # x.A=y.A label-set copy: a doc gets every label held by any doc it
            # shares an attribute value with.
            new_pos = np.asarray(membership.dot(membership.T.dot(pos))) > 0
            newly = new_pos & ~lbl.pos
            if not newly.any():
                continue

            lbl.pos |= new_pos  # monotone OR
            doc_idxs, label_idxs = np.where(newly)
            for d, l in zip(doc_idxs.tolist(), label_idxs.tolist()):
                queue.append((d, l, _EQUAL_RULE))
                affected.add(d)

        return affected

    # -----------------------------------------------------------------
    # Consequence application
    # -----------------------------------------------------------------

    def _apply_consequence(
        self,
        rule_idx: int,
        doc_idx: int,
        lbl: BulkLBL,
        queue: Deque[_QItem],
    ) -> List[int]:
        """Apply rule consequence; return list of newly-set label indices."""
        rule = self.rules[rule_idx]
        label_idx = self._label2idx.get(rule.consequence)
        if label_idx is None:
            return []

        newly_added: List[int] = []
        op = rule.consequence_op

        if op == "add":
            if not lbl.pos[doc_idx, label_idx]:
                lbl.pos[doc_idx, label_idx] = True
                newly_added.append(label_idx)
                queue.append((doc_idx, label_idx, rule_idx))

        elif op == "remove":
            if not lbl.neg[doc_idx, label_idx]:
                lbl.neg[doc_idx, label_idx] = True
                queue.append((doc_idx, label_idx, rule_idx))

        elif op == "replace":
            # Non-monotonic: evict all other positive labels to neg
            current_pos = np.where(lbl.pos[doc_idx])[0]
            for existing_lidx in current_pos:
                if existing_lidx != label_idx:
                    lbl.neg[doc_idx, existing_lidx] = True
            if not lbl.pos[doc_idx, label_idx]:
                lbl.pos[doc_idx, label_idx] = True
                newly_added.append(label_idx)
            queue.append((doc_idx, label_idx, rule_idx))

        return newly_added

    # -----------------------------------------------------------------
    # Transitivity
    # -----------------------------------------------------------------

    def _update_subset_relations(
        self,
        lbl: BulkLBL,
        affected_docs: Set[int],
        all_doc_indices: np.ndarray,
    ) -> None:
        """No-op (B-5): subset relations are NOT derived from the prediction bitmap.

        The previous implementation inferred ``y.lbl ⊆ x.lbl`` whenever doc y's
        *predicted* labels happened to be a subset of doc x's (``y_pos & ~x_pos``),
        then propagated labels up those edges (`_propagate_transitivity`). That is
        the B1 bug: coincidental bitmap containment is NOT a semantic subset
        relation — it manufactures spurious ⊆ edges (e.g. any doc with a smaller
        predicted label set becomes a "child" of every larger one) and leaks
        labels in both directions.

        Per the paper (§6.1), ``sub``/``sup`` may be populated ONLY by the
        comparison consequence ``x.lbl ⊗ y.lbl`` (the legitimate source). B-4
        realizes the ``=`` consequence directly via SpMV label-set closure
        (`_eval_equal_rules`), so no explicit ``sub``/``sup`` is needed for it;
        deriving ``sub``/``sup`` from the real co-membership relation
        (``virtual_attrs``) — to drive negative transitivity — is deferred to C-9.
        Until then this stays empty and `_propagate_transitivity` is inert.
        """
        return

    def _propagate_transitivity(
        self,
        doc_idx: int,
        new_label_idxs: List[int],
        lbl: BulkLBL,
        queue: Deque[_QItem],
    ) -> None:
        """Propagate new labels from doc_idx to all parent documents."""
        # sup[doc_idx] = set of x where doc_idx ∈ sub[x]
        # meaning doc_idx.lbl ⊆ x.lbl → x should inherit doc_idx's labels
        parents = list(lbl.sup[doc_idx])  # snapshot for safe iteration
        for x_idx in parents:
            for lidx in new_label_idxs:
                if not lbl.pos[x_idx, lidx]:
                    lbl.pos[x_idx, lidx] = True
                    queue.append((x_idx, lidx, _TRANSITIVITY_RULE))

    # -----------------------------------------------------------------
    # Conflict detection & resolution
    # -----------------------------------------------------------------

    def _handle_conflicts(
        self,
        lbl: BulkLBL,
        affected_docs: Set[int],
    ) -> List[Tuple[int, str]]:
        """Detect conflicts (pos & neg both true). Record only — do not modify sets.

        Monotonicity: pos and neg never shrink, preserving Church-Rosser.
        Conflict resolution is deferred to _build_predictions.
        """
        conflicts: List[Tuple[int, str]] = []
        for doc_idx in affected_docs:
            conflict_mask = lbl.pos[doc_idx] & lbl.neg[doc_idx]
            if not np.any(conflict_mask):
                continue

            for lidx in np.where(conflict_mask)[0]:
                label_name = self.label_names[lidx]
                conflicts.append((doc_idx, label_name))

        return conflicts

    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------

    def _build_predictions(self, lbl: BulkLBL) -> np.ndarray:
        """Convert BulkLBL to float32 prediction matrix using dual-set semantics."""
        if self.conflict_mode == "positive_wins":
            return lbl.pos.astype(np.float32)
        return (lbl.pos & ~lbl.neg).astype(np.float32)

    # -----------------------------------------------------------------
    # Main chase loop
    # -----------------------------------------------------------------

    def run(
        self,
        docs: List[Document],
        base_predictions: Optional[np.ndarray] = None,
    ) -> ChaseResult:
        """Execute the multi-label chase.

        Parameters
        ----------
        docs : List[Document]
            Input documents.
        base_predictions : np.ndarray, optional
            (n_docs, n_labels) base model predictions to seed from.

        Returns
        -------
        ChaseResult
        """
        n_docs = len(docs)
        if n_docs == 0:
            return ChaseResult(
                status="fixpoint",
                predictions=np.zeros((0, self.n_labels), dtype=np.float32),
                conflicts=[],
                n_rounds=0,
            )

        start_time = time.monotonic()

        # -- Initialise BulkLBL --
        lbl = BulkLBL.create(n_docs, self.n_labels)

        # Seed from base_predictions
        if base_predictions is not None:
            bp = np.asarray(base_predictions)
            lbl.pos[:] = bp > 0

        # Seed from doc.lbl
        for i, doc in enumerate(docs):
            if doc.lbl:
                for label_name in doc.lbl:
                    lidx = self._label2idx.get(label_name)
                    if lidx is not None:
                        lbl.pos[i, lidx] = True

        # -- Precompute text fire cache --
        logger.info(
            "Precomputing text fire cache: %d rules × %d docs",
            len(self.rules), n_docs,
        )
        text_fire_cache = self._precompute_text_fires(docs)

        # -- Provenance tracking --
        provenance: Dict[Tuple[int, str], List[int]] = {}

        def _record_provenance(doc_idx: int, label_idx: int, rule_idx: int):
            if self.track_provenance and rule_idx >= 0:
                key = (doc_idx, self.label_names[label_idx])
                provenance.setdefault(key, []).append(rule_idx)

        # -- Queue & evaluated set --
        queue: Deque[_QItem] = deque()
        evaluated: Set[Tuple[int, int, int]] = set()  # (doc_idx, label_idx, rule_idx)
        all_conflicts: List[Tuple[int, str]] = []

        # ==============================================================
        # FIRST ROUND: full scan of all rules on all documents
        # ==============================================================
        logger.info("Chase round 0: full scan")

        # 1) Text-only rules
        for rule_idx in self._text_rule_idxs:
            rule = self.rules[rule_idx]
            label_idx = self._label2idx.get(rule.consequence)
            if label_idx is None:
                continue

            firing_docs = np.where(text_fire_cache[rule_idx])[0]
            for doc_idx in firing_docs:
                key = (int(doc_idx), label_idx, rule_idx)
                if key in evaluated:
                    continue
                new_lidxs = self._apply_consequence(
                    rule_idx, int(doc_idx), lbl, queue,
                )
                for lidx in new_lidxs:
                    _record_provenance(int(doc_idx), lidx, rule_idx)
                evaluated.add(key)

        # 2) Label-dependent rules
        for rule_idx in self._label_rule_idxs:
            rule = self.rules[rule_idx]
            label_idx = self._label2idx.get(rule.consequence)
            if label_idx is None:
                continue

            firing_text_docs = np.where(text_fire_cache[rule_idx])[0]
            for doc_idx in firing_text_docs:
                key = (int(doc_idx), label_idx, rule_idx)
                if key in evaluated:
                    continue
                if self._check_label_predicates(rule, lbl, int(doc_idx)):
                    new_lidxs = self._apply_consequence(
                        rule_idx, int(doc_idx), lbl, queue,
                    )
                    for lidx in new_lidxs:
                        _record_provenance(int(doc_idx), lidx, rule_idx)
                    evaluated.add(key)

        # 3) Sim-dependent rules (SpMV batch)
        sim_hop = 0
        if self._sim_rule_idxs and self.sim_graphs:
            if self.sim_decay ** sim_hop >= self.sim_conf_threshold:
                sim_fires = self._eval_sim_rules(lbl, text_fire_cache, evaluated)
                for doc_idx, label_idx, rule_idx in sim_fires:
                    new_lidxs = self._apply_consequence(rule_idx, doc_idx, lbl, queue)
                    for lidx in new_lidxs:
                        _record_provenance(doc_idx, lidx, rule_idx)
                    evaluated.add((doc_idx, label_idx, rule_idx))
                if sim_fires:
                    sim_hop += 1

        # 4) Group-dependent rules (attribute equality batch)
        if self._group_rule_idxs and self.virtual_attrs:
            group_fires = self._eval_group_rules(lbl, text_fire_cache, evaluated)
            for doc_idx, label_idx, rule_idx in group_fires:
                new_lidxs = self._apply_consequence(rule_idx, doc_idx, lbl, queue)
                for lidx in new_lidxs:
                    _record_provenance(doc_idx, lidx, rule_idx)
                evaluated.add((doc_idx, label_idx, rule_idx))

        # 5) Comparison-consequence rules (x.lbl=y.lbl) — batched label-set copy
        if self._equal_rule_idxs and self.virtual_attrs:
            self._eval_equal_rules(lbl, queue)

        # First-round conflict check
        all_doc_set = set(range(n_docs))
        conflicts = self._handle_conflicts(lbl, all_doc_set)
        if conflicts:
            all_conflicts.extend(conflicts)
            if self.conflict_mode == "halt":
                return ChaseResult(
                    status="conflict",
                    predictions=self._build_predictions(lbl),
                    conflicts=all_conflicts,
                    n_rounds=1,
                    provenance=provenance,
                )

        # First-round transitivity
        if self.enable_transitivity:
            all_doc_indices = np.arange(n_docs)
            self._update_subset_relations(lbl, all_doc_set, all_doc_indices)
            # Propagate initial labels through subset relations
            for y_idx in range(n_docs):
                pos_lidxs = list(np.where(lbl.pos[y_idx])[0])
                if pos_lidxs:
                    self._propagate_transitivity(y_idx, pos_lidxs, lbl, queue)

        # ==============================================================
        # NON-INCREMENTAL (naive) CHASE — noInc ablation
        # ==============================================================
        # Repeat FULL scans of every rule over EVERY doc until the monotone label
        # state stops growing. Same fixpoint as the incremental loop, but O(rules ×
        # docs) work per round (no affected-docs narrowing, no `evaluated` skip) —
        # this isolates the incremental optimisation's runtime win (Exp-3 / noInc).
        if self.disable_incremental:
            round_num = 1
            prev_state = int(lbl.pos.sum() + lbl.neg.sum())
            while round_num < self.max_rounds:
                if self.time_limit_sec is not None:
                    if (time.monotonic() - start_time) >= self.time_limit_sec:
                        logger.info("Chase (noInc) terminated: time limit")
                        return ChaseResult(
                            status="timeout",
                            predictions=self._build_predictions(lbl),
                            conflicts=all_conflicts, n_rounds=round_num,
                            provenance=provenance,
                        )
                scan_q: Deque[_QItem] = deque()  # collected but not used to drive rounds
                # (1) text-only rules over ALL firing docs
                for rule_idx in self._text_rule_idxs:
                    if self._label2idx.get(self.rules[rule_idx].consequence) is None:
                        continue
                    for doc_idx in np.where(text_fire_cache[rule_idx])[0]:
                        for lidx in self._apply_consequence(rule_idx, int(doc_idx), lbl, scan_q):
                            _record_provenance(int(doc_idx), lidx, rule_idx)
                # (2) label-dependent rules over ALL firing docs
                for rule_idx in self._label_rule_idxs:
                    rule = self.rules[rule_idx]
                    if self._label2idx.get(rule.consequence) is None:
                        continue
                    for doc_idx in np.where(text_fire_cache[rule_idx])[0]:
                        if self._check_label_predicates(rule, lbl, int(doc_idx)):
                            for lidx in self._apply_consequence(rule_idx, int(doc_idx), lbl, scan_q):
                                _record_provenance(int(doc_idx), lidx, rule_idx)
                # (3) sim rules (fresh evaluated set ⇒ full re-eval)
                if self._sim_rule_idxs and self.sim_graphs:
                    for doc_idx, label_idx, rule_idx in self._eval_sim_rules(lbl, text_fire_cache, set()):
                        for lidx in self._apply_consequence(rule_idx, doc_idx, lbl, scan_q):
                            _record_provenance(doc_idx, lidx, rule_idx)
                # (4) group rules
                if self._group_rule_idxs and self.virtual_attrs:
                    for doc_idx, label_idx, rule_idx in self._eval_group_rules(lbl, text_fire_cache, set()):
                        for lidx in self._apply_consequence(rule_idx, doc_idx, lbl, scan_q):
                            _record_provenance(doc_idx, lidx, rule_idx)
                # (5) comparison-consequence rules
                if self._equal_rule_idxs and self.virtual_attrs:
                    self._eval_equal_rules(lbl, scan_q)
                # conflicts over all docs
                conflicts = self._handle_conflicts(lbl, set(range(n_docs)))
                if conflicts:
                    all_conflicts.extend(conflicts)
                    if self.conflict_mode == "halt":
                        return ChaseResult(
                            status="conflict",
                            predictions=self._build_predictions(lbl),
                            conflicts=all_conflicts, n_rounds=round_num,
                            provenance=provenance,
                        )
                round_num += 1
                state = int(lbl.pos.sum() + lbl.neg.sum())
                if state == prev_state:
                    break  # fixpoint: a full pass changed nothing
                prev_state = state
            logger.info("Chase (noInc) reached fixpoint in %d rounds", round_num)
            # Write labels back to doc.lbl (mirror the incremental path).
            for i, doc in enumerate(docs):
                doc.lbl = set(self.label_names[j] for j in range(self.n_labels)
                              if lbl.pos[i, j])
            return ChaseResult(
                status="fixpoint",
                predictions=self._build_predictions(lbl),
                conflicts=all_conflicts, n_rounds=round_num,
                provenance=provenance,
            )

        # ==============================================================
        # INCREMENTAL ROUNDS
        # ==============================================================
        round_num = 1
        all_doc_indices = np.arange(n_docs)

        while queue:
            # Time check
            if self.time_limit_sec is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.time_limit_sec:
                    logger.info("Chase terminated: time limit (%.1fs)", elapsed)
                    return ChaseResult(
                        status="timeout",
                        predictions=self._build_predictions(lbl),
                        conflicts=all_conflicts,
                        n_rounds=round_num,
                        provenance=provenance,
                    )

            if round_num >= self.max_rounds:
                logger.info("Chase terminated: max rounds (%d)", self.max_rounds)
                return ChaseResult(
                    status="max_rounds",
                    predictions=self._build_predictions(lbl),
                    conflicts=all_conflicts,
                    n_rounds=round_num,
                    provenance=provenance,
                )

            # Drain queue → current batch
            current_batch: List[_QItem] = []
            while queue:
                current_batch.append(queue.popleft())

            affected_docs: Set[int] = {item[0] for item in current_batch}

            logger.debug(
                "Chase round %d: %d queue items, %d affected docs",
                round_num, len(current_batch), len(affected_docs),
            )

            # Re-evaluate label-dependent rules on affected documents
            for rule_idx in self._label_rule_idxs:
                rule = self.rules[rule_idx]
                label_idx = self._label2idx.get(rule.consequence)
                if label_idx is None:
                    continue

                for doc_idx in affected_docs:
                    key = (doc_idx, label_idx, rule_idx)
                    if key in evaluated:
                        continue

                    # Text predicates must also pass
                    if not text_fire_cache[rule_idx, doc_idx]:
                        continue

                    if self._check_label_predicates(rule, lbl, doc_idx):
                        new_lidxs = self._apply_consequence(
                            rule_idx, doc_idx, lbl, queue,
                        )
                        for lidx in new_lidxs:
                            _record_provenance(doc_idx, lidx, rule_idx)
                        if new_lidxs and self.enable_transitivity:
                            self._propagate_transitivity(
                                doc_idx, new_lidxs, lbl, queue,
                            )
                        evaluated.add(key)

            # Re-evaluate sim-dependent rules (global SpMV, label state changed)
            if self._sim_rule_idxs and self.sim_graphs:
                if self.sim_decay ** sim_hop >= self.sim_conf_threshold:
                    sim_fires = self._eval_sim_rules(lbl, text_fire_cache, evaluated)
                    for doc_idx, label_idx, rule_idx in sim_fires:
                        new_lidxs = self._apply_consequence(rule_idx, doc_idx, lbl, queue)
                        for lidx in new_lidxs:
                            _record_provenance(doc_idx, lidx, rule_idx)
                        if new_lidxs and self.enable_transitivity:
                            self._propagate_transitivity(doc_idx, new_lidxs, lbl, queue)
                        evaluated.add((doc_idx, label_idx, rule_idx))
                    if sim_fires:
                        sim_hop += 1

            # Re-evaluate group-dependent rules (label state changed)
            if self._group_rule_idxs and self.virtual_attrs:
                group_fires = self._eval_group_rules(lbl, text_fire_cache, evaluated)
                for doc_idx, label_idx, rule_idx in group_fires:
                    new_lidxs = self._apply_consequence(rule_idx, doc_idx, lbl, queue)
                    for lidx in new_lidxs:
                        _record_provenance(doc_idx, lidx, rule_idx)
                    if new_lidxs and self.enable_transitivity:
                        self._propagate_transitivity(doc_idx, new_lidxs, lbl, queue)
                    evaluated.add((doc_idx, label_idx, rule_idx))

            # Re-evaluate comparison-consequence rules (x.lbl=y.lbl)
            if self._equal_rule_idxs and self.virtual_attrs:
                self._eval_equal_rules(lbl, queue)

            # Conflict detection
            conflicts = self._handle_conflicts(lbl, affected_docs)
            if conflicts:
                all_conflicts.extend(conflicts)
                if self.conflict_mode == "halt":
                    return ChaseResult(
                        status="conflict",
                        predictions=self._build_predictions(lbl),
                        conflicts=all_conflicts,
                        n_rounds=round_num + 1,
                        provenance=provenance,
                    )

            # Update subset relations for changed docs
            if self.enable_transitivity and affected_docs:
                self._update_subset_relations(lbl, affected_docs, all_doc_indices)

            round_num += 1

        # ==============================================================
        # FIXPOINT
        # ==============================================================
        logger.info("Chase reached fixpoint in %d rounds", round_num)

        # Write labels back to doc.lbl
        for i, doc in enumerate(docs):
            doc.lbl = set(
                self.label_names[j]
                for j in range(self.n_labels)
                if lbl.pos[i, j]
            )

        return ChaseResult(
            status="fixpoint",
            predictions=self._build_predictions(lbl),
            conflicts=all_conflicts,
            n_rounds=round_num,
            provenance=provenance,
        )

    # -----------------------------------------------------------------
    # RILL extensions: persistent state for incremental Chase
    # -----------------------------------------------------------------

    @property
    def lbl(self) -> BulkLBL:
        """Return the persistent BulkLBL state (only after ``run_persistent``)."""
        if not hasattr(self, "_lbl"):
            raise RuntimeError("Call run_persistent() before accessing .lbl")
        return self._lbl

    @property
    def text_fire_cache(self) -> np.ndarray:
        """Return the ``(n_rules, n_docs)`` text-fire cache."""
        if not hasattr(self, "_text_fire_cache"):
            raise RuntimeError(
                "Call run_persistent() before accessing .text_fire_cache"
            )
        return self._text_fire_cache

    def run_persistent(
        self,
        docs: List[Document],
        base_predictions: Optional[np.ndarray] = None,
    ) -> ChaseResult:
        """Run chase and persist internal state for subsequent ``inject_and_resume``.

        Identical to ``run()`` but stores ``lbl``, ``text_fire_cache``,
        ``evaluated``, ``docs``, and ``all_doc_indices`` on ``self`` so
        that RILL can inject oracle labels and resume incrementally.
        """
        n_docs = len(docs)
        if n_docs == 0:
            return ChaseResult(
                status="fixpoint",
                predictions=np.zeros((0, self.n_labels), dtype=np.float32),
                conflicts=[],
                n_rounds=0,
            )

        start_time = time.monotonic()

        # -- Initialise BulkLBL --
        lbl = BulkLBL.create(n_docs, self.n_labels)

        if base_predictions is not None:
            bp = np.asarray(base_predictions)
            lbl.pos[:] = bp > 0

        for i, doc in enumerate(docs):
            if doc.lbl:
                for label_name in doc.lbl:
                    lidx = self._label2idx.get(label_name)
                    if lidx is not None:
                        lbl.pos[i, lidx] = True

        # -- Precompute text fire cache --
        logger.info(
            "Precomputing text fire cache: %d rules × %d docs",
            len(self.rules), n_docs,
        )
        text_fire_cache = self._precompute_text_fires(docs)

        # -- Provenance tracking --
        provenance: Dict[Tuple[int, str], List[int]] = {}

        def _record_provenance(doc_idx: int, label_idx: int, rule_idx: int):
            if self.track_provenance and rule_idx >= 0:
                key = (doc_idx, self.label_names[label_idx])
                provenance.setdefault(key, []).append(rule_idx)

        # -- Queue & evaluated set --
        queue: Deque[_QItem] = deque()
        evaluated: Set[Tuple[int, int, int]] = set()
        all_conflicts: List[Tuple[int, str]] = []

        # === FIRST ROUND: full scan ===
        logger.info("Chase round 0: full scan (persistent mode)")

        for rule_idx in self._text_rule_idxs:
            rule = self.rules[rule_idx]
            label_idx = self._label2idx.get(rule.consequence)
            if label_idx is None:
                continue
            firing_docs = np.where(text_fire_cache[rule_idx])[0]
            for doc_idx in firing_docs:
                key = (int(doc_idx), label_idx, rule_idx)
                if key in evaluated:
                    continue
                new_lidxs = self._apply_consequence(
                    rule_idx, int(doc_idx), lbl, queue,
                )
                for lidx in new_lidxs:
                    _record_provenance(int(doc_idx), lidx, rule_idx)
                evaluated.add(key)

        for rule_idx in self._label_rule_idxs:
            rule = self.rules[rule_idx]
            label_idx = self._label2idx.get(rule.consequence)
            if label_idx is None:
                continue
            firing_text_docs = np.where(text_fire_cache[rule_idx])[0]
            for doc_idx in firing_text_docs:
                key = (int(doc_idx), label_idx, rule_idx)
                if key in evaluated:
                    continue
                if self._check_label_predicates(rule, lbl, int(doc_idx)):
                    new_lidxs = self._apply_consequence(
                        rule_idx, int(doc_idx), lbl, queue,
                    )
                    for lidx in new_lidxs:
                        _record_provenance(int(doc_idx), lidx, rule_idx)
                    evaluated.add(key)

        # Sim-dependent rules (SpMV batch) — persistent first round
        sim_hop = 0
        if self._sim_rule_idxs and self.sim_graphs:
            if self.sim_decay ** sim_hop >= self.sim_conf_threshold:
                sim_fires = self._eval_sim_rules(lbl, text_fire_cache, evaluated)
                for doc_idx, label_idx, rule_idx in sim_fires:
                    new_lidxs = self._apply_consequence(rule_idx, doc_idx, lbl, queue)
                    for lidx in new_lidxs:
                        _record_provenance(doc_idx, lidx, rule_idx)
                    evaluated.add((doc_idx, label_idx, rule_idx))
                if sim_fires:
                    sim_hop += 1

        # Group-dependent rules — persistent first round
        if self._group_rule_idxs and self.virtual_attrs:
            group_fires = self._eval_group_rules(lbl, text_fire_cache, evaluated)
            for doc_idx, label_idx, rule_idx in group_fires:
                new_lidxs = self._apply_consequence(rule_idx, doc_idx, lbl, queue)
                for lidx in new_lidxs:
                    _record_provenance(doc_idx, lidx, rule_idx)
                evaluated.add((doc_idx, label_idx, rule_idx))

        # Comparison-consequence rules (x.lbl=y.lbl) — persistent first round
        if self._equal_rule_idxs and self.virtual_attrs:
            self._eval_equal_rules(lbl, queue)

        all_doc_set = set(range(n_docs))
        conflicts = self._handle_conflicts(lbl, all_doc_set)
        if conflicts:
            all_conflicts.extend(conflicts)
            if self.conflict_mode == "halt":
                # Still persist state so caller can inspect
                self._persist_state(
                    lbl, text_fire_cache, evaluated, docs, np.arange(n_docs),
                    provenance,
                )
                return ChaseResult(
                    status="conflict",
                    predictions=self._build_predictions(lbl),
                    conflicts=all_conflicts,
                    n_rounds=1,
                    provenance=provenance,
                )

        if self.enable_transitivity:
            all_doc_indices = np.arange(n_docs)
            self._update_subset_relations(lbl, all_doc_set, all_doc_indices)
            for y_idx in range(n_docs):
                pos_lidxs = list(np.where(lbl.pos[y_idx])[0])
                if pos_lidxs:
                    self._propagate_transitivity(y_idx, pos_lidxs, lbl, queue)

        # === INCREMENTAL ROUNDS ===
        round_num = 1
        all_doc_indices = np.arange(n_docs)

        while queue:
            if self.time_limit_sec is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.time_limit_sec:
                    break
            if round_num >= self.max_rounds:
                break

            current_batch: List[_QItem] = []
            while queue:
                current_batch.append(queue.popleft())

            affected_docs: Set[int] = {item[0] for item in current_batch}

            for rule_idx in self._label_rule_idxs:
                rule = self.rules[rule_idx]
                label_idx = self._label2idx.get(rule.consequence)
                if label_idx is None:
                    continue
                for doc_idx in affected_docs:
                    key = (doc_idx, label_idx, rule_idx)
                    if key in evaluated:
                        continue
                    if not text_fire_cache[rule_idx, doc_idx]:
                        continue
                    if self._check_label_predicates(rule, lbl, doc_idx):
                        new_lidxs = self._apply_consequence(
                            rule_idx, doc_idx, lbl, queue,
                        )
                        for lidx in new_lidxs:
                            _record_provenance(doc_idx, lidx, rule_idx)
                        if new_lidxs and self.enable_transitivity:
                            self._propagate_transitivity(
                                doc_idx, new_lidxs, lbl, queue,
                            )
                        evaluated.add(key)

            # Sim-dependent rules — persistent incremental
            if self._sim_rule_idxs and self.sim_graphs:
                if self.sim_decay ** sim_hop >= self.sim_conf_threshold:
                    sim_fires = self._eval_sim_rules(lbl, text_fire_cache, evaluated)
                    for doc_idx, label_idx, rule_idx in sim_fires:
                        new_lidxs = self._apply_consequence(rule_idx, doc_idx, lbl, queue)
                        for lidx in new_lidxs:
                            _record_provenance(doc_idx, lidx, rule_idx)
                        if new_lidxs and self.enable_transitivity:
                            self._propagate_transitivity(doc_idx, new_lidxs, lbl, queue)
                        evaluated.add((doc_idx, label_idx, rule_idx))
                    if sim_fires:
                        sim_hop += 1

            # Group-dependent rules — persistent incremental
            if self._group_rule_idxs and self.virtual_attrs:
                group_fires = self._eval_group_rules(lbl, text_fire_cache, evaluated)
                for doc_idx, label_idx, rule_idx in group_fires:
                    new_lidxs = self._apply_consequence(rule_idx, doc_idx, lbl, queue)
                    for lidx in new_lidxs:
                        _record_provenance(doc_idx, lidx, rule_idx)
                    if new_lidxs and self.enable_transitivity:
                        self._propagate_transitivity(doc_idx, new_lidxs, lbl, queue)
                    evaluated.add((doc_idx, label_idx, rule_idx))

            # Comparison-consequence rules (x.lbl=y.lbl) — persistent incremental
            if self._equal_rule_idxs and self.virtual_attrs:
                self._eval_equal_rules(lbl, queue)

            conflicts = self._handle_conflicts(lbl, affected_docs)
            if conflicts:
                all_conflicts.extend(conflicts)
                if self.conflict_mode == "halt":
                    break

            if self.enable_transitivity and affected_docs:
                self._update_subset_relations(lbl, affected_docs, all_doc_indices)

            round_num += 1

        # === PERSIST STATE ===
        self._persist_state(
            lbl, text_fire_cache, evaluated, docs, all_doc_indices, provenance,
        )

        status = "fixpoint" if not queue else "max_rounds"
        if all_conflicts and self.conflict_mode == "halt":
            status = "conflict"

        logger.info(
            "Chase (persistent) reached %s in %d rounds", status, round_num,
        )

        return ChaseResult(
            status=status,
            predictions=self._build_predictions(lbl),
            conflicts=all_conflicts,
            n_rounds=round_num,
            provenance=provenance,
        )

    def _persist_state(
        self,
        lbl: BulkLBL,
        text_fire_cache: np.ndarray,
        evaluated: Set[Tuple[int, int, int]],
        docs: List[Document],
        all_doc_indices: np.ndarray,
        provenance: Dict[Tuple[int, str], List[int]],
    ) -> None:
        """Store internal state on ``self`` for ``inject_and_resume``."""
        self._lbl = lbl
        self._text_fire_cache = text_fire_cache
        self._evaluated = evaluated
        self._docs = docs
        self._all_doc_indices = all_doc_indices
        self._provenance = provenance

    def inject_and_resume(
        self,
        doc_idx: int,
        label_idx: int,
    ) -> ChaseResult:
        """Inject a seed label and resume incremental Chase.

        Called by RILL after the oracle provides a label for ``doc_idx``.

        Parameters
        ----------
        doc_idx : int
            Document index to label.
        label_idx : int
            Label column index to assign (positive).

        Returns
        -------
        ChaseResult
            Updated result after incremental propagation.
        """
        if not hasattr(self, "_lbl"):
            raise RuntimeError("Call run_persistent() before inject_and_resume()")

        lbl = self._lbl
        text_fire_cache = self._text_fire_cache
        evaluated = self._evaluated

        # Inject the seed label
        lbl.pos[doc_idx, label_idx] = True

        # Create a new queue seeded with this injection
        queue: Deque[_QItem] = deque()
        queue.append((doc_idx, label_idx, _SEED_RULE))

        all_conflicts: List[Tuple[int, str]] = []
        round_num = 0
        sim_hop = 0

        while queue:
            if round_num >= self.max_rounds:
                break

            current_batch: List[_QItem] = []
            while queue:
                current_batch.append(queue.popleft())

            affected_docs: Set[int] = {item[0] for item in current_batch}

            # Re-evaluate label-dependent rules on affected documents
            for rule_idx in self._label_rule_idxs:
                rule = self.rules[rule_idx]
                rlabel_idx = self._label2idx.get(rule.consequence)
                if rlabel_idx is None:
                    continue
                for d_idx in affected_docs:
                    key = (d_idx, rlabel_idx, rule_idx)
                    if key in evaluated:
                        continue
                    if not text_fire_cache[rule_idx, d_idx]:
                        continue
                    if self._check_label_predicates(rule, lbl, d_idx):
                        new_lidxs = self._apply_consequence(
                            rule_idx, d_idx, lbl, queue,
                        )
                        if new_lidxs and self.enable_transitivity:
                            self._propagate_transitivity(
                                d_idx, new_lidxs, lbl, queue,
                            )
                        if self.track_provenance:
                            for lidx in new_lidxs:
                                pkey = (d_idx, self.label_names[lidx])
                                self._provenance.setdefault(pkey, []).append(rule_idx)
                        evaluated.add(key)

            # Also re-evaluate text-only rules on affected docs
            # (they may now fire "add" for a label the doc didn't have)
            for rule_idx in self._text_rule_idxs:
                rule = self.rules[rule_idx]
                rlabel_idx = self._label2idx.get(rule.consequence)
                if rlabel_idx is None:
                    continue
                for d_idx in affected_docs:
                    key = (d_idx, rlabel_idx, rule_idx)
                    if key in evaluated:
                        continue
                    if not text_fire_cache[rule_idx, d_idx]:
                        continue
                    new_lidxs = self._apply_consequence(
                        rule_idx, d_idx, lbl, queue,
                    )
                    if new_lidxs and self.enable_transitivity:
                        self._propagate_transitivity(
                            d_idx, new_lidxs, lbl, queue,
                        )
                    evaluated.add(key)

            # Sim-dependent rules — inject_and_resume
            if self._sim_rule_idxs and self.sim_graphs:
                if self.sim_decay ** sim_hop >= self.sim_conf_threshold:
                    sim_fires = self._eval_sim_rules(lbl, text_fire_cache, evaluated)
                    for d_idx, l_idx, r_idx in sim_fires:
                        new_lidxs = self._apply_consequence(r_idx, d_idx, lbl, queue)
                        if new_lidxs and self.enable_transitivity:
                            self._propagate_transitivity(d_idx, new_lidxs, lbl, queue)
                        evaluated.add((d_idx, l_idx, r_idx))
                    if sim_fires:
                        sim_hop += 1

            # Group-dependent rules — inject_and_resume
            if self._group_rule_idxs and self.virtual_attrs:
                group_fires = self._eval_group_rules(lbl, text_fire_cache, evaluated)
                for d_idx, l_idx, r_idx in group_fires:
                    new_lidxs = self._apply_consequence(r_idx, d_idx, lbl, queue)
                    if new_lidxs and self.enable_transitivity:
                        self._propagate_transitivity(d_idx, new_lidxs, lbl, queue)
                    evaluated.add((d_idx, l_idx, r_idx))

            # Comparison-consequence rules (x.lbl=y.lbl) — inject_and_resume
            if self._equal_rule_idxs and self.virtual_attrs:
                self._eval_equal_rules(lbl, queue)

            conflicts = self._handle_conflicts(lbl, affected_docs)
            if conflicts:
                all_conflicts.extend(conflicts)
                if self.conflict_mode == "halt":
                    break

            if self.enable_transitivity and affected_docs:
                self._update_subset_relations(
                    lbl, affected_docs, self._all_doc_indices,
                )

            round_num += 1

        status = "fixpoint" if not queue else "conflict"

        return ChaseResult(
            status=status,
            predictions=self._build_predictions(lbl),
            conflicts=all_conflicts,
            n_rounds=round_num,
        )

    def inject_labels_and_resume(
        self,
        doc_idx: int,
        label_indices: List[int],
    ) -> ChaseResult:
        """Inject **multiple** seed labels for one document and resume Chase.

        Called by RILL after the oracle provides all labels for ``doc_idx``
        (one human interaction → all applicable labels).

        Parameters
        ----------
        doc_idx : int
            Document index to label.
        label_indices : List[int]
            All label column indices to assign (positive).

        Returns
        -------
        ChaseResult
        """
        if not hasattr(self, "_lbl"):
            raise RuntimeError(
                "Call run_persistent() before inject_labels_and_resume()"
            )

        lbl = self._lbl
        text_fire_cache = self._text_fire_cache
        evaluated = self._evaluated

        # Inject all seed labels
        queue: Deque[_QItem] = deque()
        for lidx in label_indices:
            lbl.pos[doc_idx, lidx] = True
            queue.append((doc_idx, lidx, _SEED_RULE))

        all_conflicts: List[Tuple[int, str]] = []
        round_num = 0
        sim_hop = 0

        while queue:
            if round_num >= self.max_rounds:
                break

            current_batch: List[_QItem] = []
            while queue:
                current_batch.append(queue.popleft())

            affected_docs: Set[int] = {item[0] for item in current_batch}

            for rule_idx in self._label_rule_idxs:
                rule = self.rules[rule_idx]
                rlabel_idx = self._label2idx.get(rule.consequence)
                if rlabel_idx is None:
                    continue
                for d_idx in affected_docs:
                    key = (d_idx, rlabel_idx, rule_idx)
                    if key in evaluated:
                        continue
                    if not text_fire_cache[rule_idx, d_idx]:
                        continue
                    if self._check_label_predicates(rule, lbl, d_idx):
                        new_lidxs = self._apply_consequence(
                            rule_idx, d_idx, lbl, queue,
                        )
                        if new_lidxs and self.enable_transitivity:
                            self._propagate_transitivity(
                                d_idx, new_lidxs, lbl, queue,
                            )
                        if self.track_provenance:
                            for ll in new_lidxs:
                                pkey = (d_idx, self.label_names[ll])
                                self._provenance.setdefault(pkey, []).append(
                                    rule_idx
                                )
                        evaluated.add(key)

            for rule_idx in self._text_rule_idxs:
                rule = self.rules[rule_idx]
                rlabel_idx = self._label2idx.get(rule.consequence)
                if rlabel_idx is None:
                    continue
                for d_idx in affected_docs:
                    key = (d_idx, rlabel_idx, rule_idx)
                    if key in evaluated:
                        continue
                    if not text_fire_cache[rule_idx, d_idx]:
                        continue
                    new_lidxs = self._apply_consequence(
                        rule_idx, d_idx, lbl, queue,
                    )
                    if new_lidxs and self.enable_transitivity:
                        self._propagate_transitivity(
                            d_idx, new_lidxs, lbl, queue,
                        )
                    evaluated.add(key)

            # Sim-dependent rules — inject_labels_and_resume
            if self._sim_rule_idxs and self.sim_graphs:
                if self.sim_decay ** sim_hop >= self.sim_conf_threshold:
                    sim_fires = self._eval_sim_rules(lbl, text_fire_cache, evaluated)
                    for d_idx, l_idx, r_idx in sim_fires:
                        new_lidxs = self._apply_consequence(r_idx, d_idx, lbl, queue)
                        if new_lidxs and self.enable_transitivity:
                            self._propagate_transitivity(d_idx, new_lidxs, lbl, queue)
                        evaluated.add((d_idx, l_idx, r_idx))
                    if sim_fires:
                        sim_hop += 1

            # Group-dependent rules — inject_labels_and_resume
            if self._group_rule_idxs and self.virtual_attrs:
                group_fires = self._eval_group_rules(lbl, text_fire_cache, evaluated)
                for d_idx, l_idx, r_idx in group_fires:
                    new_lidxs = self._apply_consequence(r_idx, d_idx, lbl, queue)
                    if new_lidxs and self.enable_transitivity:
                        self._propagate_transitivity(d_idx, new_lidxs, lbl, queue)
                    evaluated.add((d_idx, l_idx, r_idx))

            # Comparison-consequence rules (x.lbl=y.lbl) — inject_labels_and_resume
            if self._equal_rule_idxs and self.virtual_attrs:
                self._eval_equal_rules(lbl, queue)

            conflicts = self._handle_conflicts(lbl, affected_docs)
            if conflicts:
                all_conflicts.extend(conflicts)
                if self.conflict_mode == "halt":
                    break

            if self.enable_transitivity and affected_docs:
                self._update_subset_relations(
                    lbl, affected_docs, self._all_doc_indices,
                )

            round_num += 1

        status = "fixpoint" if not queue else "conflict"

        return ChaseResult(
            status=status,
            predictions=self._build_predictions(lbl),
            conflicts=all_conflicts,
            n_rounds=round_num,
        )
