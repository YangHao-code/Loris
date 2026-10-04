"""Complete finite facts for rule selection, without materializing proofs.

The score evaluator needs complete signed facts and comparison consistency.
Proof construction is independent of those computations in the typed runner.
This explicitly different execution protocol keeps rule order, per-head state
boundaries and successful fixed points. Exact static attribute blocks check
conflicts at block boundaries. Its inherited general matcher
prioritizes actual complete-body strict diagonal witnesses; failed partial
states and first-conflict order are not claimed identical to the prior runtime.
It is not a feedback/human Chase result and never claims an empty complete DAG.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass
import copy
import inspect
from pathlib import Path

import numpy as np

from loris._engine.array_cache import sha
from loris._engine.hashing import cached_file_sha as file_sha
from loris._engine.rdl_reference import AttrEqual, Label, LabelCompare, SEMANTICS
from loris._engine.typed_runtime import IndexedChaseRunner, RelationFacts, bits, indices, _gamma_blocks
from loris._engine.relation_rows import RelationRows
from loris._engine.attribute_index import row_bits, shared_group_positions
from loris._engine.typed_runtime import unbits
from loris._constraints.finite_comparison import ExactFiniteComparisonOracle
from .state_execution import ExactBookkeepingState, state_execution_binding
from .state_execution import source_stamp as state_execution_source_stamp, _verify_original
from . import selection_prefix, selection_relations


SCHEMA = 'paper36_complete_fact_strict_diagonal_priority_selection_v12_reflexive_attribute_blocks'
_ORIGINAL_RUNNER_METHODS = {
    name: getattr(IndexedChaseRunner, name)
    for name, value in vars(IndexedChaseRunner).items()
    if inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod, property))
}


@dataclass
class SelectionChaseResult:
    """Scoring-only facts: deliberately no proof_dag or provenance attribute.

    A distinct type and absent proof fields make the original full-state and
    feedback consumers reject this object rather than hash a None/empty DAG.
    """
    status: str
    predictions: np.ndarray
    conflicts: list
    n_rounds: int
    positive_facts: np.ndarray
    negative_facts: np.ndarray
    proof_materialized: bool = False
    selection_only: bool = True
    feedback_permitted: bool = False


class SelectionFactsState(ExactBookkeepingState):
    """The same monotone factual state, with no per-fact provenance objects."""
    def __init__(self, runner, gamma):
        self.owner = runner
        self.n, self.k = gamma.shape
        self.pos, self.neg = gamma.pos.copy(), gamma.neg.copy()
        self.label_bits = {'pos': [bits(self.pos[:, j]) for j in range(self.k)],
                           'neg': [bits(self.neg[:, j]) for j in range(self.k)]}
        self._positive_row_bits = [bits(row) for row in self.pos]
        self._positive_closed = False
        self._signed_overlap = bool(np.any(self.pos & self.neg))
        self._strict_edge_count = 0
        self.out = RelationRows(self.n, reflexive=True)
        self.inc = RelationRows(self.n, reflexive=True)
        self.proper = RelationRows(self.n)
        self.proper_inc = RelationRows(self.n)
        self.proof = self.base_proof = None
        self.provenance = {}
        self.conflict = None
        self.version = 0
        self.last_valid_version = -1
        self.feasibility_checks = 0
        self.relation_version = 0
        self.stats = Counter()
        self._comparison_oracle = ExactFiniteComparisonOracle()
        self.gamma_bits = {sign: tuple(rows) for sign, rows in self.label_bits.items()}
        self.gamma_protection = False
        self.blocked_bits = {'pos': {}, 'neg': {}}
        self.gamma_blocked = 0
        # The old seed proof records do not establish any factual edge. Apply
        # every comparison seed in exactly the same sorted Gamma order.
        for a, b in gamma.subset:
            self.add_edge(a, b, False, initialize=True)
        for a, b in gamma.strict_subset:
            self.add_edge(a, b, True, initialize=True)
        self.propagate_all_seed_labels()
        self.gamma_protection = bool(getattr(runner, 'gamma_protection', False))

    def get_proof(self, key):
        raise RuntimeError('selection-only state does not materialize proofs')

    def remember(self, *args, **kwargs):
        raise RuntimeError('selection-only state must not construct proof records')

    def set_label(self, sign, i, j, origin=None, dependencies=(), rule_id=None,
                  valuation=None, static=(), propagate=True):
        matrix = self.pos if sign == 'pos' else self.neg
        if matrix[i, j]:
            return False
        if self.gamma_protection and _gamma_blocks(self, sign, i, j):
            return False
        was_closed = self._positive_closed
        matrix[i, j] = True
        if sign == 'pos':
            self._positive_closed = False
            self._positive_row_bits[i] |= 1 << int(j)
            self._signed_overlap |= bool(self.neg[i, j])
        else:
            self._signed_overlap |= bool(self.pos[i, j])
        self.label_bits[sign][j] |= 1 << i
        self.version += 1
        if sign == 'pos' and propagate:
            for target in indices(self.out[i] & ~self.label_bits['pos'][j]):
                self.set_label('pos', target, j, propagate=False)
            # A complete row propagation preserves an already closed state;
            # it cannot certify other externally deferred label insertions.
            self._positive_closed = was_closed
        return True

    def propagate_all_seed_labels(self):
        for source in self.out.nonidentity_sources():
            for target in indices(self.out[source] & ~(1 << source)):
                difference = self._positive_row_bits[source] & ~self._positive_row_bits[target]
                for j in indices(difference):
                    self.set_label('pos', target, j, propagate=False)
        self._positive_closed = True

    def _install(self, kind, source, target, dependencies=(), origin=None):
        out, inc = (self.proper, self.proper_inc) if kind == 'strict_subset' else (self.out, self.inc)
        bit = 1 << target
        if out[source] & bit:
            return False
        out[source] |= bit
        inc[target] |= 1 << source
        self.version += 1
        self.relation_version += 1
        if kind == 'strict_subset':
            self._strict_edge_count += 1
        self.stats['new_' + kind] += 1
        return True

    def add_edge(self, a, b, strict, initialize=False):
        if (not strict and not self._strict_edge_count and not initialize
                and self._positive_closed and getattr(self.owner, '_relation_batching', False)):
            return self._add_edge_rectangle(a, b)
        was_closed = self._positive_closed
        self._positive_closed = False
        changed = self._add_edge_scalar(a, b, strict, initialize)
        # Comparison initialization intentionally defers positive propagation.
        # Otherwise the original scalar algorithm preserves a closed input.
        self._positive_closed = was_closed if not changed or not initialize else False
        return changed

    def _add_edge_rectangle(self, a, b):
        """Exact non-strict insertion R <- R union predecessors(a) x successors(b).

        All relation rows and their positive labels are closed on entry. No
        rule matching, validity check, or provenance consumer runs inside this
        call. Its return state and counters match scalar edge insertion; no
        first-proof or internal per-edge order is claimed for this fact path.
        """
        if self.out[a] & (1 << b):
            return False
        # Transitive closure already contains every edge from inc[b] to
        # out[b], and every edge from inc[a] to out[a]. Therefore all new
        # edges lie in these two exact differences. In particular, joining
        # a new node into an existing component need not scan that entire
        # component again when its members' rows are already complete.
        predecessors = self.inc[a] & ~self.inc[b]
        successors = self.out[b] & ~self.out[a]
        successor_count = bin(successors).count('1')
        source_labels = self._positive_row_bits[a]
        protection = self.gamma_protection
        added_relations = 0
        for source in indices(predecessors):
            if protection:
                # A validated negative at a (or between) may have blocked a
                # predecessor's label there; every new pair still copies it.
                source_labels |= self._positive_row_bits[source]
            delta = successors & ~self.out[source]
            if delta:
                self.out[source] |= successors
                # Component joins commonly add this entire same successor
                # set to many rows. Count that set once; partial rows still
                # use their own exact difference cardinality.
                added_relations += successor_count if delta == successors else bin(delta).count('1')
        for target in indices(successors):
            self.inc[target] |= predecessors
        self.version += added_relations
        self.relation_version += added_relations
        self.stats['new_subset'] += added_relations
        # Because every predecessor already reaches a, its positive labels
        # are contained in pos[a] (without Gamma protection; with it, the
        # union of the predecessors' labels was collected above, which equals
        # pos[a] whenever nothing was blocked). Every old successor of a
        # already has those labels. Only the new successor difference needs
        # copies.
        for label in indices(source_labels):
            targets = successors & ~self.label_bits['pos'][label]
            if targets and self.gamma_protection:
                protected = targets & self.gamma_bits['neg'][label]
                if protected:
                    # Validated negatives win; count each protected cell once.
                    new = protected & ~self.blocked_bits['pos'].get(label, 0)
                    self.blocked_bits['pos'][label] = self.blocked_bits['pos'].get(label, 0) | protected
                    self.gamma_blocked += bin(new).count('1')
                    targets &= ~protected
            if not targets:
                continue
            self.label_bits['pos'][label] |= targets
            self._signed_overlap |= bool(targets & self.label_bits['neg'][label])
            self.version += successor_count if targets == successors else bin(targets).count('1')
            label_bit = 1 << label
            for target in indices(targets):
                self.pos[target, label] = True
                self._positive_row_bits[target] |= label_bit
        return True

    def _add_edge_scalar(self, a, b, strict, initialize=False):
        target_relation = self.proper if strict else self.out
        if target_relation[a] & (1 << b):
            return False
        closes_cycle = bool(self.out[b] & (1 << a))
        predecessors = tuple(indices(self.inc[a]))
        successors = self.out[b]
        proper_suffix = self.proper[b]
        proper_prefix = self.proper_inc[a]
        new_subset = []
        for source in predecessors:
            for target in indices(successors & ~self.out[source]):
                if self._install('subset', source, target):
                    new_subset.append((source, target))
            proper_targets = successors if strict or (proper_prefix & (1 << source)) else proper_suffix
            for target in indices(proper_targets & ~self.proper[source]):
                self._install('strict_subset', source, target)
        changed = closes_cycle and bool(strict or self._strict_edge_count)
        while changed:
            changed = False
            for source in range(self.n):
                if not self.proper[source]:
                    continue
                for middle in tuple(indices(self.proper[source])):
                    for target in indices(self.out[middle] & ~self.proper[source]):
                        changed |= self._install('strict_subset', source, target)
                for predecessor in indices(self.inc[source] & ~(1 << source)):
                    for target in indices(self.proper[source] & ~self.proper[predecessor]):
                        changed |= self._install('strict_subset', predecessor, target)
        if not initialize:
            for source, target in new_subset:
                difference = self._positive_row_bits[source] & ~self._positive_row_bits[target]
                for j in indices(difference):
                    self.set_label('pos', target, j)
        return True

    def add_edge_block(self, sources, targets, *, equality=False):
        """Exact closure of a complete non-strict Cartesian head block.

        For closed R, adding A x B establishes precisely
        R union (union inc[A]) x (union out[B]). For equality, every member
        of A union B becomes mutually reachable. All predecessor labels are
        unioned: Gamma may have blocked a label at an intermediate node.
        Strict relations and uncertified label closure use the scalar path.
        """
        if self._strict_edge_count or not self._positive_closed:
            raise ValueError('attribute blocks require closed non-strict state')
        if not sources or not targets:
            return False
        if equality:
            sources = targets = sources | targets
        predecessors = self.inc.union_rows(indices(sources))
        successors = self.out.union_rows(indices(targets))
        successor_rows = np.asarray(list(indices(successors)), dtype=np.intp)
        successor_count = len(successor_rows)
        byte_offsets = successor_rows >> 3
        bit_masks = np.left_shift(np.uint8(1), (successor_rows & 7).astype(np.uint8))
        byte_width = (self.n + 7) // 8

        def member_mask(value):
            # Read only this block's columns from the packed N-axis bitmap.
            # A full unpackbits/flatnonzero scan costs O(N) for every label;
            # a published-date block usually has only a few dozen rows.
            packed = np.frombuffer(value.to_bytes(byte_width, 'little'), np.uint8)
            return (packed[byte_offsets] & bit_masks) != 0
        source_labels, added = 0, 0
        forward = {}
        predecessor_rows = tuple(indices(predecessors))
        for source in predecessor_rows:
            source_labels |= self._positive_row_bits[source]
            old = self.out[source]
            overlap = old & successors
            if overlap == successors:
                continue
            # Initial/reflexive rows have a singleton overlap. Avoid an
            # N-wide binary conversion for that common exact count.
            overlap_count = (0 if not overlap else 1 if not overlap & (overlap - 1)
                             else bin(overlap).count('1'))
            added += successor_count - overlap_count
            forward.setdefault(old | successors, []).append(source)
        if not added:
            return False
        for value, rows in forward.items():
            self.out.set_many(rows, value)
        backward = {}
        for target in successor_rows.tolist():
            old = self.inc[target]
            value = old | predecessors
            if value != old:
                backward.setdefault(value, []).append(target)
        for value, rows in backward.items():
            self.inc.set_many(rows, value)
        self.version += added
        self.relation_version += added
        self.stats['new_subset'] += added
        self.stats['attribute_relation_blocks'] += 1
        for label in indices(source_labels):
            targets = successors & ~self.label_bits['pos'][label]
            if targets and self.gamma_protection:
                protected = targets & self.gamma_bits['neg'][label]
                if protected:
                    previous = self.blocked_bits['pos'].get(label, 0)
                    self.blocked_bits['pos'][label] = previous | protected
                    self.gamma_blocked += int(np.count_nonzero(member_mask(protected & ~previous)))
                    targets &= ~protected
            if not targets:
                continue
            target_rows = successor_rows[member_mask(targets)]
            self.pos[target_rows, label] = True
            self.label_bits['pos'][label] |= targets
            self._signed_overlap |= bool(targets & self.label_bits['neg'][label])
            self.version += len(target_rows)
            label_bit = 1 << label
            for target in target_rows.tolist():
                self._positive_row_bits[target] |= label_bit
        return True


def _descriptors(cls):
    return {name: value for name, value in vars(cls).items()
            if inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod, property))}


def _verify_methods():
    for name, method in _ORIGINAL_RUNNER_METHODS.items():
        if getattr(IndexedChaseRunner, name) is not method:
            raise RuntimeError('original selection matching method changed: ' + name)
    if _descriptors(SelectionFactsState) != _STATE_DESCRIPTORS:
        raise RuntimeError('bound selection state method changed')
    if _descriptors(SelectionChaseRunner) != _RUNNER_DESCRIPTORS:
        raise RuntimeError('bound selection runner method changed')
    for cls, expected in _PREFIX_DESCRIPTORS.items():
        if _descriptors(cls) != expected:
            raise RuntimeError('bound first-pass snapshot method changed')
    if selection_prefix.fork_state is not _PREFIX_FORK:
        raise RuntimeError('bound first-pass snapshot fork changed')
    if (selection_relations.ImmutableRelationCache is not _RELATION_CACHE_CLASS
            or _descriptors(_RELATION_CACHE_CLASS) != _RELATION_CACHE_DESCRIPTORS
            or selection_relations.RelationFacts is not RelationFacts):
        raise RuntimeError('bound immutable relation materialization changed')
    # Also checks the supported original State source and the inherited exact
    # finite-comparison/label-bookkeeping implementation identities (the rest
    # of that binding is a pure function of source_stamp, checked by callers).
    _verify_original()


_SOURCES = {}


def _source_paths():
    if not _SOURCES:
        _SOURCES['paths'] = (str(Path(__file__).resolve()),
                             str(Path(selection_prefix.__file__).resolve()),
                             str(Path(selection_relations.__file__).resolve()))
    return _SOURCES['paths']


def binding_stamp():
    """Content identity of every source the execution binding is built from.

    The binding is a pure function of the policy flags, these file hashes
    and module constants, so an unchanged stamp implies an unchanged binding.
    """
    return tuple(file_sha(path) for path in _source_paths()) + state_execution_source_stamp()


def selection_execution_binding(*, prefix_reuse=True, prefix_cache_entries=2, relation_batching=True, unary_label_batching=True, skip_reflexive_attribute_blocks=True):
    _verify_methods()
    if (type(prefix_reuse) is not bool or type(relation_batching) is not bool
            or type(unary_label_batching) is not bool
            or type(skip_reflexive_attribute_blocks) is not bool
            or type(prefix_cache_entries) is not int or prefix_cache_entries < 2):
        raise ValueError('explicit Boolean reuse policy and at least two snapshot slots required')
    source, prefix_source, relation_source = _source_paths()
    return {
        'schema': SCHEMA,
        'implementation': {'path': source, 'sha256': file_sha(source)},
        'state_bookkeeping': state_execution_binding(),
        'unary_label_blocks': {
            'enabled': unary_label_batching,
            'precondition': 'one_variable_no_binary_body_no_nonidentity_or_strict_relations_no_new_signed_conflict',
            'labels': 'one_exact_column_bitset_union_and_numpy_scatter',
            'Gamma_protection': 'opposite_immutable_roots_win_and_unique_blocked_cells_counted',
            'first_conflict': 'potential_new_signed_overlap_uses_original_valuation_fallback',
            'matching_counters_rounds_and_final_facts_preserved': True,
        },
        'non_strict_relation_batching': {
            'enabled': relation_batching,
            'precondition': 'non_strict_head_and_no_proper_relation_and_positive_closure_certified',
            'relation_update': 'R_union_predecessors_a_cartesian_successors_b',
            'effective_domains': 'inc_a_minus_inc_b_cartesian_out_b_minus_out_a',
            'label_update': 'copy_preinsertion_positive_labels_a_to_out_b_minus_out_a',
            'label_update_under_gamma_protection':
                'copy_union_of_preinsertion_positive_labels_of_new_predecessors_to_out_b_minus_out_a',
            'initialization_and_strict_relations_use_scalar_fallback': True,
            'complete_add_edge_return_state_and_counters_preserved': True,
            'rule_valuation_round_and_conflict_check_boundaries_preserved': False,
            'internal_fact_insertion_order_or_first_proof_claimed': False,
        },
        'attribute_relation_blocks': {
            'enabled': relation_batching,
            'skip_reflexive_singletons': skip_reflexive_attribute_blocks,
            'reflexive_singleton_semantics': 'same_filtered_single_row_on_both_sides_keeps_ordered_match_and_conflict_check_without_readding_reflexivity',
            'precondition': ('two_distinct_head_variables_single_cross_attribute_equality_'
                'no_strict_relations_positive_closure_certified'),
            'negative_label_domains': 'unchanged_during_non_strict_relation_head_execution',
            'positive_label_domains': 'snapshot_each_ordered_rule_execution_refreshed_next_fixpoint_pass',
            'model_domains': 'frozen_probability_tables_during_each_runner_call',
            'closure': 'R_union_union_inc_sources_cartesian_union_out_targets',
            'equality': 'merge_union_of_both_domains_as_one_component',
            'labels': 'union_all_predecessor_positive_labels_with_exact_Gamma_protection',
            'attribute_groups': 'context_bound_compact_stable_rows_lazy_bitmaps_or_legacy_eager_exact_bitmaps',
            'missing_attribute_values': 'excluded_before_matching',
            'strict_or_unsupported_relation_shape': 'original_valuation_fallback',
            'successful_fixedpoint_facts_and_factual_counters_preserved': True,
            'failed_partial_facts_or_first_conflict_order_claimed': False,
        },
        'first_pass_prefix': {
            'enabled': prefix_reuse, 'maximum_entries': prefix_cache_entries,
            'schema': selection_prefix.SCHEMA,
            'implementation': {'path': prefix_source, 'sha256': file_sha(prefix_source)},
            'cut': 'first_pass_after_ordered_selected_prefix_before_appended_candidate',
            'promotion': 'reuse_completed_candidate_first_pass_as_next_ordered_prefix',
            'fixedpoint_warmstart': True,
            'fixedpoint_warmstart_rule': ('when the complete fixpoint of the ordered prefix is cached, '
                'fork it, apply the appended rule, then repeat full rounds until no change; '
                'facts only grow, so the least fixpoint and conflict status are unchanged'),
            'fixedpoint_cache_entries': 3, 'persistent_dynamic_cache': False,
            'cache_key': 'complete_context_Gamma_ordered_prefix_and_execution_policy',
            'actual_and_logical_matching_statistics_separate': True,
        },
        'relation_materialization': {
            'schema': selection_relations.SCHEMA, 'maximum_entries': 2,
            'implementation': {'path': relation_source, 'sha256': file_sha(relation_source)},
            'hit_requires': 'same_n_and_complete_ordered_integer_rows_equal',
            'hit_returns': 'same_original_immutable_RelationFacts_object',
            'miss': 'exact_adaptive_RelationFacts_constructor_v2',
            'no_mutable_Chase_state_or_proofs_cached': True,
            'persistent_cache': False,
        },
        'scope': 'rule_selection_only',
        'proof_materialized': False,
        'feedback_permitted': False,
        'full_signed_facts_and_comparison_closure': True,
        'original_rule_order_preserved': True,
        'valuation_order': 'exact_static_attribute_blocks_else_strict_diagonal_witness_first_else_original',
        'successful_fixedpoint_facts_preserved': True,
        'failed_partial_facts_first_conflict_or_proofs_match_prior_order': False,
        'finite_comparison_semantics_preserved': True,
        'static_evidence_and_proof_dependencies_are_not_factual_premises': True,
        'no_rules_rows_or_fixedpoint_rounds_omitted': True,
        'original_sources_classes_or_globals_modified': False,
        'requires_new_score_cache_binding': True,
    }


class SelectionChaseRunner(IndexedChaseRunner):
    def __init__(self, *args, prefix_reuse=True, prefix_cache_entries=2, relation_batching=True, unary_label_batching=True, skip_reflexive_attribute_blocks=True, **kwargs):
        # This scoring loop has its own execution schedule. Unlike the copied
        # proof runner, it does not yet apply per-instance scope permissions.
        # Reject them explicitly rather than score an unapproved application.
        if kwargs.get('application_scope') is not None:
            raise ValueError('selection-only acceleration does not support application_scope; use the complete typed runner')
        self._prefix_reuse = prefix_reuse
        self._prefix_cache_entries = prefix_cache_entries
        self._relation_batching = relation_batching
        self._unary_label_batching = unary_label_batching
        self._skip_reflexive_attribute_blocks = skip_reflexive_attribute_blocks
        self._binding_stamp = binding_stamp()
        self._selection_execution_binding = selection_execution_binding(
            prefix_reuse=prefix_reuse, prefix_cache_entries=prefix_cache_entries,
            relation_batching=relation_batching, unary_label_batching=unary_label_batching,
            skip_reflexive_attribute_blocks=skip_reflexive_attribute_blocks)
        self._prefix_cache = selection_prefix.FirstPassCache(prefix_cache_entries)
        self._fixpoints = OrderedDict()
        self._fixpoint_stats = Counter()
        self._relation_facts_cache = selection_relations.ImmutableRelationCache(2)
        self._attribute_blocks_cache = {}
        self._logical_skipped_matching = Counter()
        super().__init__(*args, **kwargs)

    @property
    def selection_execution_binding(self):
        return copy.deepcopy(self._selection_execution_binding)

    @property
    def binding(self):
        result = super().binding
        result['selection_execution'] = self.selection_execution_binding
        return result

    def verify_code(self):
        super().verify_code()
        _verify_methods()
        if (type(self._skip_reflexive_attribute_blocks) is not bool
                or self._skip_reflexive_attribute_blocks != self._selection_execution_binding['attribute_relation_blocks']['skip_reflexive_singletons']):
            raise RuntimeError('bound reflexive attribute block policy changed')
        for name in set(_ORIGINAL_RUNNER_METHODS) | set(_RUNNER_DESCRIPTORS):
            if name in self.__dict__:
                raise RuntimeError('selection runner method override: ' + name)
        if binding_stamp() != self._binding_stamp:
            if selection_execution_binding(prefix_reuse=self._prefix_reuse,
                    prefix_cache_entries=self._prefix_cache_entries,
                    relation_batching=self._relation_batching,
                    unary_label_batching=self._unary_label_batching,
                    skip_reflexive_attribute_blocks=self._skip_reflexive_attribute_blocks) != self._selection_execution_binding:
                raise RuntimeError('bound selection execution source or policy changed')
            self._binding_stamp = binding_stamp()
        if (type(self._prefix_cache) is not selection_prefix.FirstPassCache
                or self._prefix_cache.entries != self._prefix_cache_entries):
            raise RuntimeError('bound first-pass cache policy changed')
        if (type(self._relation_facts_cache) is not _RELATION_CACHE_CLASS
                or self._relation_facts_cache.entries != 2
                or any(name in vars(self._relation_facts_cache) for name in _RELATION_CACHE_DESCRIPTORS)):
            raise RuntimeError('bound immutable relation cache policy changed')

    def program_key(self, rules, gamma=None):
        # The stored binding is only read here; no defensive copy is needed.
        return sha({'schema': SCHEMA,
                    'logical_program_key': super().program_key(rules, gamma),
                    'selection_execution': self._selection_execution_binding})

    def close(self):
        self._prefix_cache.clear()
        self._fixpoints.clear()
        self._relation_facts_cache.clear()
        self._attribute_blocks_cache.clear()
        super().close()

    def audit(self):
        result = super().audit()
        logical = Counter(self.stats)
        logical.update(self._logical_skipped_matching)
        result.update(protocol=SCHEMA, selection_only=True, proof_materialized=False,
                      feedback_permitted=False,
                      selection_execution_binding=self.selection_execution_binding,
                      logical_stats=dict(logical), first_pass_prefix=self._prefix_cache.audit(),
                      fixpoint_warmstart=dict(self._fixpoint_stats),
                      relation_materialization=self._relation_facts_cache.audit())
        return result

    def _prefix_key(self, rules, gamma):
        return sha({'schema': selection_prefix.SCHEMA, 'context': self.context_id,
            'Gamma': self._seed_key(gamma), 'ordered_rules': [key for _, key in rules],
            'execution_policy': self.selection_execution_binding})

    def _matching_counts(self):
        return Counter({key: self.stats.get(key, 0)
                        for key in ('indexed_prefixes', 'matched_valuations')})

    def _gamma_protection_record(self, rules, prepared, state, status):
        """Blocked Gamma contradictions; for one Label-head rule also its agreements.

        Agreements are validated Gamma cells of the head label and polarity whose
        row has at least one complete body valuation in the final state, i.e.
        the rule's correct firings on known (fit) rows. None when not computed.
        """
        if not self.gamma_protection:
            return None
        agreements = None
        if status == 'fixpoint' and len(rules) == 1 and isinstance(rules[0][0].head, Label):
            rule = rules[0][0]
            head = rule.head
            candidates = state.gamma_bits[head.polarity][self.label_index[head.label]]
            agreements = bin(self._head_witness_rows(rule, prepared[0], state, candidates)).count('1')
        return {'blocked': int(state.gamma_blocked), 'agreements': agreements}

    def _head_witness_rows(self, rule, prepared, state, candidates):
        """Rows in candidates for the head variable with a complete body valuation."""
        head_var = rule.head.var
        order = [head_var] + [v for v in rule.variables if v != head_var]
        assignment = {}

        def domain_of(var):
            domain = prepared['domains'][var]
            for atom in prepared['unary'][var]:
                domain &= state.label_bits[atom.polarity][self.label_index[atom.label]]
            for atom in prepared['binary']:
                if var in (atom.left_var, atom.right_var):
                    domain &= self._relation_domain(atom, var, assignment, state)
            return domain

        def exists(depth):
            if depth == len(order):
                return True
            var = order[depth]
            for row in indices(domain_of(var)):
                assignment[var] = row
                found = exists(depth + 1)
                del assignment[var]
                if found:
                    return True
            return False

        witnessed = 0
        for row in indices(domain_of(head_var) & candidates):
            assignment[head_var] = row
            if exists(1):
                witnessed |= 1 << row
            del assignment[head_var]
        return witnessed

    def _remember_fixpoint(self, key, frozen_state):
        # The empty program (seeded Gamma) is the prefix of every standalone
        # score; it is never evicted. Three further complete fixpoints are kept.
        self._fixpoints[key] = frozen_state
        self._fixpoints.move_to_end(key)
        self._fixpoint_stats['fixpoints_cached'] += 1
        empty = self._prefix_key((), self.gamma)
        while len(self._fixpoints) > 4:
            victim = next(k for k in self._fixpoints if k != empty)
            del self._fixpoints[victim]

    @staticmethod
    def _state_reads(prepared):
        """State-dependent premises of a rule: signed label columns and relations."""
        labels = {(atom.polarity, atom.label) for atoms in prepared['unary'].values() for atom in atoms}
        relations = any(not isinstance(atom, AttrEqual) for atom in prepared['binary'])
        return labels, relations

    def _semi_naive(self, rules, prepared, state, first):
        """Run rule indices `first` once, then only rules reading changed facts.

        At a fixpoint of the other rules, a rule whose Label/LabelCompare
        premises are unchanged has no new satisfied valuation (static atoms and
        attribute equality never change), so skipping it cannot omit a fact.
        Label copies along subset edges are part of the state update itself.
        """
        reads = [self._state_reads(p) for p in prepared]
        pending, rounds, status = list(first), 0, 'fixpoint'
        while pending:
            rounds += 1
            before = {sign: list(state.label_bits[sign]) for sign in ('pos', 'neg')}
            relations_before = state.relation_version
            for rule_id in pending:
                _, status = self._execute_rule_range(rules, prepared, state, rule_id, rule_id + 1)
                if status == 'conflict':
                    return rounds, status
            changed = {(sign, self.labels[j]) for sign in ('pos', 'neg')
                       for j, value in enumerate(state.label_bits[sign]) if value != before[sign][j]}
            relations = state.relation_version != relations_before
            self._fixpoint_stats['semi_naive_rounds'] += 1
            pending = [i for i, (labels, uses_relations) in enumerate(reads)
                       if (labels & changed) or (uses_relations and relations)]
        return rounds, status

    def _execute_rule_range(self, rules, prepared, state, start, stop):
        changed = False
        for rule_id in range(start, stop):
            rule, _ = rules[rule_id]
            blocked = self._execute_unary_label_block(rule, prepared[rule_id], state)
            if blocked is None:
                blocked = self._execute_attribute_blocks(rule, prepared[rule_id], state)
            if blocked is not None:
                block_changed, status = blocked
                changed |= block_changed
                if status == 'conflict':
                    return changed, status
                continue
            for valuation in self._valuations(rule, prepared[rule_id], state):
                # _evidence only builds proof dependency/static records;
                # all factual/static matching has already been performed.
                head = rule.head
                if isinstance(head, Label):
                    changed |= state.set_label(head.polarity, valuation[head.var],
                                               self.label_index[head.label])
                else:
                    a, b = self._edge(head, valuation)
                    edges = [(a, b), (b, a)] if head.op == 'eq' else [(a, b)]
                    for x, y in edges:
                        changed |= state.add_edge(x, y, head.op == 'strict_subset')
                if state.invalid():
                    return changed, 'conflict'
        return changed, 'fixpoint'

    def _execute_unary_label_block(self, rule, prepared, state):
        """Exact unary insertions when every row has only its reflexive edge.

        Unary premises on other labels are unchanged by this rule. A premise
        equal to the head cannot gain a new matching row: its same-sign cells
        already have the head, and opposite-sign cells either stay protected or
        trigger the unchanged scalar conflict path. Thus the initial domain is
        exactly the original ordered matcher domain. No rule/validity consumer
        observes intermediate insertions on this conflict-free fragment.
        """
        head = rule.head
        if (not self._unary_label_batching or not isinstance(head, Label)
                or len(rule.variables) != 1 or prepared['binary']
                or state.relation_version or state._strict_edge_count
                or not state._positive_closed or state._signed_overlap or state.conflict):
            return None
        var = rule.variables[0]
        domain = prepared['domains'][var]
        for atom in prepared['unary'][var]:
            domain &= state.label_bits[atom.polarity][self.label_index[atom.label]]
        sign, label = head.polarity, self.label_index[head.label]
        opposite = 'neg' if sign == 'pos' else 'pos'
        domain &= ~state.label_bits[sign][label]
        if not domain:
            return False, 'fixpoint'
        protected = domain & state.gamma_bits[opposite][label] if state.gamma_protection else 0
        targets = domain & ~protected
        if targets & state.label_bits[opposite][label]:
            # The old matcher must choose the first conflict and retain its
            # exact partial state, including only earlier blocked valuations.
            return None
        count = bin(domain).count('1')
        self.stats['indexed_prefixes'] += count
        self.stats['matched_valuations'] += count
        self.stats['unary_label_blocks'] += 1
        if protected:
            previous = state.blocked_bits[sign].get(label, 0)
            state.blocked_bits[sign][label] = previous | protected
            state.gamma_blocked += bin(protected & ~previous).count('1')
        if targets:
            rows = np.fromiter(indices(targets), dtype=np.intp)
            (state.pos if sign == 'pos' else state.neg)[rows, label] = True
            state.label_bits[sign][label] |= targets
            state.version += len(rows)
            if sign == 'pos':
                label_bit = 1 << label
                for row in rows.tolist():
                    state._positive_row_bits[row] |= label_bit
            # With only reflexive relations, no further positive copies or
            # comparison witness checks are needed; closure remains certified.
            assert not state.invalid()
        return bool(targets), 'fixpoint'

    def _execute_attribute_blocks(self, rule, prepared, state):
        """Apply attribute joins as exact Cartesian closure blocks.

        Unary label domains are rebuilt from the current state on every call.
        Positive premises may grow during a block; ordinary fixpoint scheduling
        revisits this rule when those labels change. Negative premises remain
        stable, and ML probabilities are frozen for the runner context.
        """
        head = rule.head
        binary = prepared['binary']
        if (not self._relation_batching or state._strict_edge_count or not state._positive_closed
                or not isinstance(head, LabelCompare) or head.op == 'strict_subset'
                or head.left_var == head.right_var or len(rule.variables) != 2
                or len(binary) != 1 or not isinstance(binary[0], AttrEqual)
                or binary[0].left_var == binary[0].right_var):
            return None
        atom = binary[0]
        cache_key = (atom.left_attr, atom.right_attr)
        groups = self._attribute_blocks_cache.get(cache_key)
        if groups is None:
            _, mappings = self._attribute_index(*cache_key)
            left, right = mappings
            if self.compact_attribute_index:
                groups = (left, right, *shared_group_positions(left, right))
            else:
                groups = tuple((rows, right[code]) for code, rows in left.items() if code in right)
            self._attribute_blocks_cache[cache_key] = groups
        domains = dict(prepared['domains'])
        for var, atoms in prepared['unary'].items():
            for premise in atoms:
                domains[var] &= state.label_bits[premise.polarity][self.label_index[premise.label]]
        if self.compact_attribute_index:
            left_groups, right_groups, left_positions, right_positions = groups
            left_mask = unbits(domains[atom.left_var], self.n)
            right_mask = unbits(domains[atom.right_var], self.n)
            active = (left_groups.matching_groups(left_mask)[left_positions]
                      & right_groups.matching_groups(right_mask)[right_positions])

            def matched_blocks():
                # Keep the old left-dict order and exact per-rule snapshot.
                # Filter before constructing any full-axis integer bitmap.
                for lpos, rpos in zip(left_positions[active], right_positions[active]):
                    left_rows = left_groups.group_rows(lpos)
                    right_rows = right_groups.group_rows(rpos)
                    left_rows = left_rows[left_mask[left_rows]]
                    right_rows = right_rows[right_mask[right_rows]]
                    if (self._skip_reflexive_attribute_blocks
                            and len(left_rows) == len(right_rows) == 1
                            and left_rows[0] == right_rows[0]):
                        # Both subset and equality add only an existing
                        # reflexive edge. Preserve the original block's place
                        # in match counting and conflict checks, without two
                        # full-axis integers or redundant closure work.
                        yield None, None
                    else:
                        yield row_bits(left_rows), row_bits(right_rows)
            blocks = matched_blocks()
        else:
            blocks = groups
        changed = False
        for left, right in blocks:
            if left is None:
                self.stats['attribute_blocks_matched'] += 1
                if state.invalid():
                    return changed, 'conflict'
                continue
            left &= domains[atom.left_var]
            right &= domains[atom.right_var]
            if not left or not right:
                continue
            a, b = (left, right) if head.left_var == atom.left_var else (right, left)
            if self.subset_direction == 'paper_p7_reverse' and head.op != 'eq':
                a, b = b, a
            self.stats['attribute_blocks_matched'] += 1
            changed |= state.add_edge_block(a, b, equality=head.op == 'eq')
            if state.invalid():
                return changed, 'conflict'
        return changed, 'fixpoint'

    def __call__(self, rules, gamma=None):
        self.verify_code()
        rules = tuple(self._rule(value) for value in rules)
        prepared = [self._prepare(rule, key) for rule, key in rules]
        gamma = self.gamma if gamma is None else self._gamma(gamma)
        matching_before = self._matching_counts()
        inherited_matching = Counter()
        reused = False
        warm = None
        if self._prefix_reuse and rules:
            prefix_fixpoint_key = self._prefix_key(rules[:-1], gamma)
            warm = self._fixpoints.get(prefix_fixpoint_key)
            if warm is not None:
                self._fixpoints.move_to_end(prefix_fixpoint_key)
            elif len(rules) == 1:
                # The empty program's fixpoint is the seeded Gamma state itself.
                seed = SelectionFactsState(self, gamma)
                if not seed.invalid():
                    warm = selection_prefix.fork_state(seed, readonly=True)
                    self._remember_fixpoint(self._prefix_key((), gamma), warm)
        if warm is not None:
            # Exact semi-naive warm start from the complete fixpoint of the prefix.
            self._fixpoint_stats['warm_starts'] += 1
            state = selection_prefix.fork_state(warm)
            reused = True
            rounds, status = self._semi_naive(rules, prepared, state, [len(rules) - 1])
            repeat = False
        elif self._prefix_reuse:
            prefix = rules[:-1] if rules else rules
            prefix_key = self._prefix_key(prefix, gamma)
            program_key = self._prefix_key(rules, gamma)
            snapshot = self._prefix_cache.get(prefix_key)
            if snapshot is None:
                state = SelectionFactsState(self, gamma)
                rounds, changed = 0, False
                status = 'conflict' if state.invalid() else 'fixpoint'
                if status != 'conflict':
                    rounds = 1
                    changed, status = self._execute_rule_range(rules, prepared, state, 0, len(prefix))
                self._prefix_cache.put(prefix_key, state, status=status, rounds=rounds,
                    changed=changed, matching_counts=self._matching_counts() - matching_before,
                    full_program=False)
            else:
                state = snapshot.fork(self)
                rounds, changed, status = snapshot.rounds, snapshot.changed, snapshot.status
                inherited_matching.update(dict(snapshot.matching_counts))
                self._logical_skipped_matching.update(inherited_matching)
                reused = True
            if status != 'conflict' and rules:
                candidate_changed, status = self._execute_rule_range(
                    rules, prepared, state, len(prefix), len(rules))
                changed |= candidate_changed
            # Seal before later rounds mutate this branch. A subsequent call
            # whose actual selected prefix is this program promotes this cut.
            first_pass_matching = inherited_matching + (self._matching_counts() - matching_before)
            self._prefix_cache.put(program_key, state, status=status, rounds=rounds,
                changed=changed, matching_counts=first_pass_matching,
                full_program=True, protect=prefix_key)
            repeat = status != 'conflict' and changed
        else:
            state = SelectionFactsState(self, gamma)
            rounds = 0
            status = 'conflict' if state.invalid() else 'fixpoint'
            repeat = status != 'conflict'
        while repeat:
            rounds += 1
            changed, status = self._execute_rule_range(rules, prepared, state, 0, len(rules))
            repeat = status != 'conflict' and changed
        if self._prefix_reuse and status == 'fixpoint':
            key = self._prefix_key(rules, gamma)
            if key not in self._fixpoints:
                self._remember_fixpoint(key, selection_prefix.fork_state(state, readonly=True))
            self._fixpoints.move_to_end(key)
        conflicts = [(int(i), self.labels[int(j)]) for i, j in np.argwhere(state.pos & state.neg)]
        result = SelectionChaseResult(status=status, predictions=(state.pos & ~state.neg).astype(np.float32),
            conflicts=conflicts, n_rounds=rounds,
            positive_facts=state.pos.copy(), negative_facts=state.neg.copy())
        result.gamma_protection = self._gamma_protection_record(rules, prepared, state, status)
        result.subset_facts = self._relation_facts_cache.materialize(state.out)
        result.strict_subset_facts = self._relation_facts_cache.materialize(state.proper)
        result.comparison_conflicts = ([state.conflict] if state.conflict and
            state.conflict['kind'] != 'positive_negative_label_conflict' else [])
        self.stats['Chase_calls'] += 1
        result.reference_audit = {
            'semantics': SEMANTICS, 'executor': SCHEMA, 'context_id': self.context_id,
            'full_rows': self.n, 'label_names': list(self.labels),
            'subset_direction': self.subset_direction,
            'subset_direction_pending_author_confirmation': False,
            'subset_direction_author_confirmed': self.subset_direction == 'standard',
            'subset_direction_legacy_compatibility': self.subset_direction == 'paper_p7_reverse',
            'complete_without_cutoff': True,
            'program_sha256': sha([rule.to_dict() for rule, _ in rules]),
            'comparison_premises_from_partial_bitmaps': False,
            'document_lbl_used_as_Gamma': False, 'model_outputs_unioned_into_Gamma': False,
            'negative_contraposition_added': False,
            'finite_comparison_solver_calls': state.feasibility_checks,
            'finite_comparison_certificate_statistics': dict(state._comparison_oracle.stats),
            'relation_fact_representation': 'complete_immutable_bitset_rows',
            'proof_representation': 'not_materialized_selection_only',
            'proof_materialized': False,
            'rounds_and_first_witness_not_claimed_identical_to_reference': True,
            'incremental_relation_statistics': dict(state.stats),
        }
        result.selection_execution_audit = {
            'schema': SCHEMA, 'scope': 'rule_selection_only', 'proof_materialized': False,
            'feedback_permitted': False, 'complete_without_cutoff': True,
            'execution_binding': self.selection_execution_binding,
            'first_pass_prefix_reused': reused,
            'inherited_logical_matching_counts': dict(inherited_matching),
            'actual_matching_counts': dict(self._matching_counts() - matching_before),
            'relation_materialization': self._relation_facts_cache.audit(),
            'full_proof_replay_required_before_provenance_or_feedback_use': True,
        }
        return result

    run = __call__


_STATE_DESCRIPTORS = _descriptors(SelectionFactsState)
_RUNNER_DESCRIPTORS = _descriptors(SelectionChaseRunner)
_PREFIX_DESCRIPTORS = {cls: _descriptors(cls)
    for cls in (selection_prefix.FirstPassSnapshot, selection_prefix.FirstPassCache)}
_PREFIX_FORK = selection_prefix.fork_state
_RELATION_CACHE_CLASS = selection_relations.ImmutableRelationCache
_RELATION_CACHE_DESCRIPTORS = _descriptors(_RELATION_CACHE_CLASS)


def make_selection_runner(context, cache_dir, **runner_options):
    """Create an explicitly bound complete-fact runner for rule scoring only."""
    return SelectionChaseRunner(context, cache_dir, **runner_options)
