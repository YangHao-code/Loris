"""Indexed full finite Chase for explicitly scoped PDF36 RDL.

Static unary masks and attribute equality indexes replace Cartesian scans.
Dynamic facts never enter persistent caches. The rule order is fixed. For a
strict head over at most its two variables, a real complete-body diagonal
valuation is tried first; it goes through the ordinary head/proof/conflict path.
Other valuations retain their original order. This may change failed partial
states and first-conflict witnesses, never a successful fixed point.
All finite valuations remain supported, including disconnected/many-variable
bodies: these use the same exact indexed nested join without a cutoff.

The finite oracle remains independent in rdl_reference.py. Its strict inclusion
feasibility checker is shared: solver completions are never observed facts.
"""
from collections import Counter
from collections.abc import Mapping
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import sys
from types import MappingProxyType

import numpy as np
from loris._constraints.finite_comparison import ExactFiniteComparisonOracle
import scipy

from loris.chase import ChaseResult, GammaSeed
from loris.document import Document, VALID_ATTRS
from loris._engine.array_cache import PersistentArrayCache, canonical, file_sha, sha, _readonly
from loris._engine.hashing import array_hash, immutable_buffer
from loris._engine.relation_rows import RelationRows, RELATION_SCHEMA
from loris._engine.attribute_index import CompactAttributeIndex, SCHEMA as ATTRIBUTE_INDEX_SCHEMA
from loris._engine.rdl_reference import (Doc, Label, Text, ML, AttrEqual, LabelCompare,
    Rule, RefGamma, ReferenceChase, comparison_feasible, prepare_attributes, SEMANTICS, SCHEMA)

PROTOCOL = 'paper36_explicit_variable_indexed_v6_fixed_label_attribute_iteration'


def bits(mask):
    """Numpy row mask to little-endian arbitrary length exact integer bitset."""
    return int.from_bytes(np.packbits(np.asarray(mask, bool), bitorder='little').tobytes(), 'little')


def unbits(value, n):
    """Exact inverse of bits(): row bitset -> Boolean mask of length n.

    Identical to np.array([bool(value & (1 << i)) for i in range(n)]) for
    every nonnegative value below 1 << n (little-endian bit order).
    """
    if value < 0 or value >> n:
        raise ValueError('row bitset outside the complete document axis')
    raw = np.frombuffer(value.to_bytes((n + 7) // 8, 'little'), np.uint8)
    return np.unpackbits(raw, bitorder='little')[:n].astype(bool)


def _gamma_blocks(state, sign, i, j):
    """Validated Gamma wins: True if the opposite Gamma fact protects (i, j).

    Only active once a state enables gamma_protection after its seed closure.
    Each protected cell is counted once in state.gamma_blocked.
    """
    opposite = 'neg' if sign == 'pos' else 'pos'
    bit = 1 << int(i)
    if not (state.gamma_bits[opposite][j] & bit):
        return False
    seen = state.blocked_bits[sign]
    if not (seen.get(j, 0) & bit):
        seen[j] = seen.get(j, 0) | bit
        state.gamma_blocked += 1
    return True


def indices(value):
    # Same ascending row sequence as the plain lowest-bit loop. Every big-int
    # step of that loop is O(bit_length), so after a few rows a wide positive
    # bitset is decoded once with np.unpackbits (2026-09-30 optimization).
    for _ in range(8):
        if not value:
            return
        low = value & -value
        yield low.bit_length() - 1
        value ^= low
    if value > 0 and value.bit_length() > 256:
        raw = np.frombuffer(value.to_bytes((value.bit_length() + 7) // 8, 'little'), np.uint8)
        yield from np.flatnonzero(np.unpackbits(raw, bitorder='little')).tolist()
        return
    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low


def _immutable(value):
    if value is None or type(value) in (str, int, float, bool): return value
    if isinstance(value, Mapping): return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)): return tuple(_immutable(item) for item in value)
    return value


def _plain(value):
    if value is None or type(value) in (str, int, float, bool): return value
    if isinstance(value, Mapping): return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(item) for item in value]
    return value


_SHARED_SEED_PROOFS = {}


def _mapping_only(value):
    # json default equivalent to encoding _plain(value): Mappings become dicts,
    # tuples/lists are native, anything else is not JSON (as after _plain).
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError('Object of type %s is not JSON serializable' % type(value).__name__)


def canonical_node_bytes(node):
    """UTF-8 of canonical(_plain(node)); raises TypeError where canonical does."""
    return json.dumps(node, default=_mapping_only, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


class ProofDAG(Mapping):
    """Complete immutable proof map with deterministic cached layered identity."""
    def __init__(self, derived, base=None):
        self._derived = _immutable(derived)
        self._base = base
        if base is not None and any(key in base for key in self._derived):
            raise ValueError('derived proof may not shadow a validated proof')
        self._identity = None
        self._node_bytes = None

    @classmethod
    def adopt(cls, derived, immutable=(), encoded_from=None):
        """ProofDAG(derived) where values listed by identity in ``immutable``
        are already outputs of _immutable and are reused as they are.

        ``immutable`` maps keys to such values (for example an earlier
        ProofDAG); a value is reused only if it is that very object, so the
        content equals ProofDAG(derived) exactly (2026-09-30 optimization).
        An adopted DAG caches the canonical bytes of each node, by node object,
        and inherits cached bytes of reused nodes from ``immutable`` and from
        ``encoded_from`` (another adopted ProofDAG).
        """
        result = cls.__new__(cls)
        known = immutable._derived if isinstance(immutable, ProofDAG) and immutable._base is None \
            else immutable
        get = known.get if known else (lambda key: None)
        result._derived = MappingProxyType({
            key: value if get(key) is value and value is not None else _immutable(value)
            for key, value in derived.items()})
        result._base = None
        result._identity = None
        # id(node) -> bytes. The DAG holds every node, so an id is never reused
        # while its entry exists; inherited entries are for the same objects.
        cache = {}
        for source in (immutable, encoded_from):
            saved = getattr(source, '_node_bytes', None)
            if saved:
                cache.update(saved)
        live = {id(value) for value in result._derived.values()}
        result._node_bytes = {key: value for key, value in cache.items() if key in live}
        return result

    def canonical_local_bytes(self):
        """UTF-8 of canonical(_plain(derived part)), assembled from cached node
        bytes; None when this DAG keeps no node cache. Raises TypeError exactly
        where canonical(_plain(...)) would. The DAG is immutable, so the bytes
        are computed once."""
        if getattr(self, '_node_bytes', None) is None:
            # Plain ProofDAGs and subclasses keep the original encoding path.
            return None
        saved = getattr(self, '_local_bytes', None)
        if saved is not None:
            return saved
        keys = sorted(self._derived)
        if any(type(key) is not str for key in keys):
            return None
        joined = ''.join(keys)
        # Printable ASCII keys without quote/backslash are their own JSON body.
        plain_keys = (joined.isascii() and joined.isprintable() and '"' not in joined
                      and '\\' not in joined)
        cache = self._node_bytes
        derived = self._derived
        parts = []
        for key in keys:
            node = derived[key]
            encoded = cache.get(id(node))
            if encoded is None:
                encoded = cache[id(node)] = canonical_node_bytes(node)
            head = (b'"' + key.encode() + b'":') if plain_keys else (
                json.dumps(key, ensure_ascii=False).encode() + b':')
            parts.append(head + encoded)
        self._local_bytes = b'{' + b','.join(parts) + b'}'
        return self._local_bytes

    def __len__(self): return len(self._derived) + (len(self._base) if self._base is not None else 0)
    def __iter__(self):
        yield from self._derived
        if self._base is not None: yield from self._base
    def __getitem__(self, key):
        if key in self._derived: return self._derived[key]
        if self._base is None: raise KeyError(key)
        return self._base[key]

    def identity(self):
        if self._identity is None:
            local = self.canonical_local_bytes()
            self._identity = {'schema': 'exact_immutable_layered_proof_dag_v1', 'count': len(self),
                'local_sha256': (sha(_plain(self._derived)) if local is None
                                 else hashlib.sha256(local).hexdigest()),
                'base_identity': self._base.identity() if self._base is not None else None}
            self._identity['sha256'] = sha(self._identity)
        return copy.deepcopy(self._identity)

    def to_dict(self): return {key: _plain(self[key]) for key in self}

    def as_dict(self):
        """dict(self.items()) in the same key order, without per-key lookups."""
        if self._base is None:
            return dict(self._derived)
        return dict(self.items())


class RelationFacts:
    """Immutable exact relation with implicit default and adaptive stored rows."""
    __slots__ = ('_rows', '_n', '_size', '_digest')
    def __init__(self, rows):
        object.__setattr__(self, '_rows', RelationRows.from_rows(rows, frozen=True))
        object.__setattr__(self, '_n', len(self._rows))
        object.__setattr__(self, '_size', self._rows.count())
        object.__setattr__(self, '_digest', self._rows.sha256())

    def __setattr__(self, key, value): raise AttributeError('complete relation is immutable')
    @property
    def rows(self): return self._rows
    @property
    def n(self): return self._n
    def __len__(self): return self._size
    def __iter__(self):
        yield from self.rows.edges()
    def __contains__(self, edge):
        if not isinstance(edge, tuple) or len(edge) != 2: return False
        i, j = edge
        return isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer)) and 0 <= i < self.n and 0 <= j < self.n and bool(self.rows[int(i)] & (1 << int(j)))
    @property
    def sha256(self): return self._digest
    def identity(self):
        return {'schema': RELATION_SCHEMA, 'n': self.n, 'count': len(self), 'sha256': self.sha256}
    def to_dict(self):
        return {**self.identity(), 'rows': self.rows.to_dict()}


class _State:
    def __init__(self, runner, gamma):
        self.owner = runner
        self.n, self.k = gamma.shape
        self.pos, self.neg = gamma.pos.copy(), gamma.neg.copy()
        self.label_bits = {'pos': [bits(self.pos[:, j]) for j in range(self.k)],
                           'neg': [bits(self.neg[:, j]) for j in range(self.k)]}
        # Validated Gamma cells; protection starts after the seed closure.
        self.gamma_bits = {sign: tuple(rows) for sign, rows in self.label_bits.items()}
        self.gamma_protection = False
        self.blocked_bits = {'pos': {}, 'neg': {}}
        self.gamma_blocked = 0
        self.out = RelationRows(self.n, reflexive=True)
        self.inc = RelationRows(self.n, reflexive=True)
        self.proper = RelationRows(self.n)
        self.proper_inc = RelationRows(self.n)
        self.proof = {}
        self.base_proof = runner._seed_proof(gamma)
        self.provenance = {}
        self.conflict = None
        self.version = 0
        self.last_valid_version = -1
        self.feasibility_checks = 0
        self.relation_version = 0
        self.stats = Counter()
        # A private complete integer witness can certify consistency without
        # rebuilding the finite MILP. Its values never enter factual labels.
        self._comparison_oracle = ExactFiniteComparisonOracle()
        # Reference seeds all comparison edges before its first closure. Retain
        # their validated origins even if an earlier edge entails a later one.
        for kind, edges in [('subset', gamma.subset), ('strict_subset', gamma.strict_subset)]:
            for a, b in edges: self.remember(kind, a, b, 'validated_Gamma')
        for a, b in gamma.subset: self.add_edge(a, b, False, initialize=True)
        for a, b in gamma.strict_subset: self.add_edge(a, b, True, initialize=True)
        self.propagate_all_seed_labels()
        self.gamma_protection = bool(getattr(runner, 'gamma_protection', False))

    def key(self, kind, i, j): return '%s:%d:%d' % (kind, i, j)

    def get_proof(self, key):
        return self.proof[key] if key in self.proof else self.base_proof[key]

    def remember(self, kind, i, j, origin, dependencies=(), rule_id=None, valuation=None, static=()):
        key = self.key(kind, i, j)
        if key not in self.proof and key not in self.base_proof:
            ids = {rule_id} if rule_id is not None else set()
            for dependency in dependencies: ids.update(self.get_proof(dependency).get('support_rule_ids', ()))
            self.proof[key] = {'kind': kind, 'row': int(i), 'column': int(j), 'origin': origin,
                'dependencies': list(dependencies), 'rule_id': rule_id, 'valuation': dict(valuation or {}),
                'static_evidence': list(static), 'support_rule_ids': sorted(ids)}
            if kind in ('pos', 'neg') and ids:
                target = (int(i), self.owner.labels[j])
                self.provenance[target] = sorted(set(self.provenance.get(target, ())) | ids)
        return key

    def set_label(self, sign, i, j, origin, dependencies=(), rule_id=None, valuation=None, static=(), propagate=True):
        matrix = self.pos if sign == 'pos' else self.neg
        if matrix[i, j]: return False
        if self.gamma_protection and _gamma_blocks(self, sign, i, j): return False
        matrix[i, j] = True
        self.label_bits[sign][j] |= 1 << i
        self.version += 1
        self.remember(sign, i, j, origin, dependencies, rule_id, valuation, static)
        if sign == 'pos' and propagate:
            for target in indices(self.out[i] & ~self.label_bits['pos'][j]):
                self.set_label('pos', target, j, 'inclusion_label_copy',
                    [self.key('subset', i, target), self.key('pos', i, j)], propagate=False)
        return True

    def propagate_all_seed_labels(self):
        # Reachability is already complete; direct paths cover every possible
        # positive copy. Negative membership is never contraposed.
        for source in self.out.nonidentity_sources():
            for target in indices(self.out[source] & ~(1 << source)):
                for j in np.flatnonzero(self.pos[source] & ~self.pos[target]):
                    self.set_label('pos', target, int(j), 'inclusion_label_copy',
                        [self.key('subset', source, target), self.key('pos', source, int(j))], propagate=False)

    def _install(self, kind, source, target, dependencies, origin):
        out, inc = (self.proper, self.proper_inc) if kind == 'strict_subset' else (self.out, self.inc)
        bit = 1 << target
        if out[source] & bit: return False
        out[source] |= bit
        inc[target] |= 1 << source
        self.version += 1; self.relation_version += 1
        self.remember(kind, source, target, origin, dependencies)
        self.stats['new_' + kind] += 1
        return True

    def add_edge(self, a, b, strict, initialize=False):
        """Exact incremental transitive closure over (subset, proper) paths.

        Before an insertion both relations are transitively closed. Every new
        path is predecessor(a) -> inserted edge -> successor(b). A path is
        proper iff its prefix, inserted edge, or suffix contains a proper edge.
        Snapshot operands prevent recursive proof dependencies.
        """
        target_relation = self.proper if strict else self.out
        if target_relation[a] & (1 << b): return False
        closes_cycle = bool(self.out[b] & (1 << a))
        predecessors = tuple(indices(self.inc[a]))
        successors = self.out[b]
        proper_suffix = self.proper[b]
        proper_prefix = self.proper_inc[a]
        edge_kind = 'strict_subset' if strict else 'subset'
        edge_key = self.key(edge_kind, a, b)
        if edge_key not in self.proof and edge_key not in self.base_proof:
            raise RuntimeError('new comparison edge has no authentic derivation')
        new_subset = []
        for source in predecessors:
            prefix = self.key('subset', source, a)
            for target in indices(successors & ~self.out[source]):
                deps = [prefix, edge_key, self.key('subset', b, target)]
                if self._install('subset', source, target, deps, 'inclusion_transitivity'):
                    new_subset.append((source, target))
            proper_targets = successors if strict or (proper_prefix & (1 << source)) else proper_suffix
            for target in indices(proper_targets & ~self.proper[source]):
                if strict:
                    deps = [prefix, edge_key, self.key('subset', b, target)]
                elif proper_prefix & (1 << source):
                    deps = [self.key('strict_subset', source, a), edge_key, self.key('subset', b, target)]
                else:
                    deps = [prefix, edge_key, self.key('strict_subset', b, target)]
                self._install('strict_subset', source, target, deps, 'proper_inclusion_transitivity')
        # A newly introduced cycle with an existing proper segment can require
        # proper paths crossing the inserted edge twice. Saturate only changed
        # strict rows, using the now-complete subset relation, without cutoffs.
        # Without a pre-existing b -> a path, every new path crosses the
        # inserted edge at most once; the loops above are already complete.
        # The omitted scan would add no facts, including no first proofs.
        changed = closes_cycle and bool(strict or any(self.proper))
        while changed:
            changed = False
            for source in range(self.n):
                if not self.proper[source]: continue
                for middle in tuple(indices(self.proper[source])):
                    for target in indices(self.out[middle] & ~self.proper[source]):
                        changed |= self._install('strict_subset', source, target,
                            [self.key('strict_subset', source, middle), self.key('subset', middle, target)],
                            'proper_inclusion_transitivity')
                for predecessor in indices(self.inc[source] & ~(1 << source)):
                    for target in indices(self.proper[source] & ~self.proper[predecessor]):
                        changed |= self._install('strict_subset', predecessor, target,
                            [self.key('subset', predecessor, source), self.key('strict_subset', source, target)],
                            'proper_inclusion_transitivity')
        if not initialize:
            for source, target in new_subset:
                for j in np.flatnonzero(self.pos[source] & ~self.pos[target]):
                    self.set_label('pos', target, int(j), 'inclusion_label_copy',
                        [self.key('subset', source, target), self.key('pos', source, int(j))])
        return True

    def invalid(self):
        if self.last_valid_version == self.version: return self.conflict is not None
        if np.any(self.pos & self.neg):
            self.conflict = {'kind': 'positive_negative_label_conflict'}
        else:
            if any(self.proper) and self._comparison_oracle.try_certify_rows(
                    self.pos, self.neg, self.out, self.proper):
                # A checked completion satisfies every label bound and every
                # actual comparison, so all following contradiction tests are
                # false. UNKNOWN retains the original ordered diagnostics and
                # complete, unlimited MILP fallback below.
                self.last_valid_version = self.version
                return self.conflict is not None
            for source, row in enumerate(self.proper):
                for target in indices(row):
                    if self.out[target] & (1 << source):
                        self.conflict = {'kind': 'strict_inclusion_cycle', 'source': source, 'target': target}
                        break
                    if not np.any(~self.pos[source] & ~self.neg[target]):
                        self.conflict = {'kind': 'strict_inclusion_has_no_possible_witness', 'source': source, 'target': target}
                        break
                if self.conflict: break
            if self.conflict is None and any(self.proper):
                self.feasibility_checks += 1
                if not comparison_feasible(self.pos, self.neg, set(RelationFacts(self.out)), set(RelationFacts(self.proper))):
                    self.conflict = {'kind': 'finite_label_comparisons_unsatisfiable'}
        self.last_valid_version = self.version
        return self.conflict is not None


class IndexedChaseRunner:
    """Own one immutable full data/model context; execute changing typed Sigma.

    ``runner(rules, gamma=None)`` returns full arrays plus compact complete
    RelationFacts and an immutable Mapping proof DAG (derived facts + immutable Gamma).
    No history RDL, score, ground truth, or final Sigma is loaded by this class.
    """
    def __init__(self, context, cache_dir, *, subset_direction='standard', verify_source_files=True,
                 memory_bytes=256 << 20, disk_bytes=1 << 30, lexical_cache_dir=None,
                 application_scope=None, gamma_protection=False, compact_attribute_index=True,
                 attribute_label_iteration=True):
        from .instance_scope import FrozenInstanceScope, scope_corpus_identity
        if type(gamma_protection) is not bool:
            raise TypeError('gamma_protection must be an explicit bool')
        self.gamma_protection = gamma_protection
        if type(compact_attribute_index) is not bool:
            raise TypeError('compact_attribute_index must be an explicit bool')
        self.compact_attribute_index = compact_attribute_index
        if type(attribute_label_iteration) is not bool:
            raise TypeError('attribute_label_iteration must be an explicit bool')
        self.attribute_label_iteration = attribute_label_iteration
        self.scope_corpus_id = scope_corpus_identity(context)
        if application_scope is not None:
            if not isinstance(application_scope, FrozenInstanceScope):
                raise TypeError('explicit FrozenInstanceScope required')
            application_scope.validate(context)
        self._application_scope = application_scope
        self._docs = tuple(Document(cnt=d.cnt, ttl=d.ttl, mtd=copy.deepcopy(d.mtd)) for d in context['docs'])
        self.labels = tuple(context['labels']); self.n = len(self._docs); self.k = len(self.labels)
        for doc in self._docs:
            if any(not isinstance(doc.get_attr(attr), str) for attr in VALID_ATTRS):
                raise ValueError('typed Document view requires explicit string cnt/ttl/mtd; encode metadata upstream')
        validated = ReferenceChase(self.labels,
            model_probabilities=context.get('model_probabilities', context.get('scores', {})),
            model_decisions=context.get('model_decisions', {}), subset_direction=subset_direction,
            attribute_presence=context.get('attribute_presence', {}), attribute_values=context.get('attribute_values', {}),
            static_model_probabilities=context.get('static_model_probabilities'))
        self.probabilities = MappingProxyType({key: _readonly(value) for key, value in validated.model_probabilities.items()})
        self.static_probabilities = MappingProxyType({key: _readonly(value) for key, value in validated.static_model_probabilities.items()})
        self.decisions = MappingProxyType({key: _readonly(value) for key, value in validated.model_decisions.items()})
        if any(a.shape != (self.n, self.k) for tables in [self.probabilities, self.static_probabilities, self.decisions] for a in tables.values()):
            raise ValueError('all model tables must match complete document and label axes')
        values, presence = prepare_attributes(self._docs, context.get('attribute_values', {}), context.get('attribute_presence', {}))
        self.attribute_values = MappingProxyType({key: tuple(column) for key, column in values.items()})
        self.presence = MappingProxyType({key: _readonly(mask) for key, mask in presence.items()})
        self.subset_direction = subset_direction
        self.label_index = {label: i for i, label in enumerate(self.labels)}
        gamma = context['gamma']
        self.gamma = self._gamma(gamma)
        self.stats = Counter(); self._rule_cache = {}; self._static_bits = {}; self._seed_cache = {}
        self._all = (1 << self.n) - 1
        files = {str(Path(path).resolve(strict=True)): digest for path, digest in context.get('input_files', {}).items()}
        if verify_source_files:
            for path, digest in files.items():
                if file_sha(path) != digest: raise ValueError('input SHA mismatch: ' + path)
        code_paths = [Path(__file__), Path(inspect.getfile(ReferenceChase)), Path(inspect.getfile(PersistentArrayCache)),
                      Path(inspect.getfile(Document)), Path(inspect.getfile(GammaSeed)), Path(inspect.getfile(array_hash))]
        from loris.predicates import predicate_to_dict
        from loris._engine.primitive_masks import TextPrimitiveCache
        code_paths.append(Path(inspect.getfile(predicate_to_dict)))
        code_paths.append(Path(inspect.getfile(TextPrimitiveCache)))
        code_paths.append(Path(inspect.getfile(ExactFiniteComparisonOracle)))
        code_paths.append(Path(inspect.getfile(RelationRows)))
        code_paths.append(Path(inspect.getfile(CompactAttributeIndex)))
        code_paths.append(Path(inspect.getfile(FrozenInstanceScope)))
        self.lexical = TextPrimitiveCache(self._docs,
            lexical_cache_dir if lexical_cache_dir is not None else Path(cache_dir) / 'lexical_primitives',
            attribute_presence=self.presence, memory_bytes=memory_bytes, disk_bytes=disk_bytes)
        self._code = {str(path.resolve()): file_sha(path) for path in code_paths}
        self._code_stat = {p: self._stat(p) for p in self._code}
        self._binding = {'protocol': PROTOCOL, 'reference_semantics': SEMANTICS, 'rule_schema': SCHEMA,
            'subset_direction': subset_direction, 'labels': list(self.labels), 'full_rows': self.n,
            'documents': sha([[d.cnt, d.ttl, d.mtd] for d in self._docs]),
            'probabilities': {k: array_hash(v) for k, v in self.probabilities.items()},
            'static_probabilities': {k: array_hash(v) for k, v in self.static_probabilities.items()},
            'decisions': {k: array_hash(v) for k, v in self.decisions.items()},
            'attribute_presence': {k: array_hash(v) for k, v in self.presence.items()},
            'attribute_values': {k: sha(v) for k, v in self.attribute_values.items()},
            'lexical_primitive_context_id': self.lexical.context_id,
            'source_context_id': context.get('context_id'), 'input_files': files, 'code_files': self._code,
            'scope_corpus_identity': self.scope_corpus_id,
            'versions': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__},
            'no_dynamic_facts_in_persistent_cache': True, 'no_cutoffs': True,
            'attribute_label_iteration': {
                'enabled': attribute_label_iteration,
                'fragment': 'target_first_same_signed_label_same_attribute_with_static_masks_and_other_label_columns_no_nonidentity_relations',
                'grounding_order_proofs_conflicts_and_matching_counts_preserved': True},
            'attribute_index': {'schema': ATTRIBUTE_INDEX_SCHEMA,
                'compact': compact_attribute_index,
                'persistent_group_bitmaps': not compact_attribute_index,
                'group_and_row_order_preserved': True},
            'strict_diagonal_priority': {
                'scope': 'all_body_variables_are_head_variables_at_most_two_without_application_scope',
                'premises': 'actual_static_masks_signed_label_facts_and_full_relation_facts',
                'execution': 'ordinary_evidence_head_application_and_conflict_check',
                'successful_fixedpoint_unchanged': True,
                'failed_partial_facts_first_conflict_and_proofs_match_prior_order': False}}
        if self._application_scope is not None:
            self._binding['application_scope'] = self._application_scope.receipt()
        if self.gamma_protection:
            self._binding['gamma_conflict_semantics'] = {
                'policy': 'validated_Gamma_wins',
                'derived_fact_contradicting_validated_Gamma': 'blocked_not_inserted_and_counted',
                'derived_vs_derived_contradiction': 'chase_conflict_unchanged',
                'seed_closure_contradiction': 'chase_conflict_unchanged'}
        self.context_id = sha(self._binding)
        path = Path(cache_dir).resolve() / self.context_id; path.mkdir(parents=True, exist_ok=True)
        (path / 'binding.json').write_text(canonical(self._binding))
        self._db = sqlite3.connect(str(path / 'typed_exact.sqlite'))
        self._db.execute('PRAGMA journal_mode=WAL'); self._db.execute('PRAGMA synchronous=FULL')
        self._db.execute('CREATE TABLE IF NOT EXISTS arrays(namespace TEXT,key TEXT,payload BLOB,sha256 TEXT,bytes INTEGER,created INTEGER,PRIMARY KEY(namespace,key))')
        self.masks = PersistentArrayCache(self._db, 'unary_masks', memory_bytes=memory_bytes, disk_bytes=disk_bytes)
        self.conjunctions = PersistentArrayCache(self._db, 'unary_conjunctions', memory_bytes=memory_bytes, disk_bytes=disk_bytes)
        self.attribute_codes = PersistentArrayCache(self._db, 'attribute_equal_codes', memory_bytes=memory_bytes, disk_bytes=disk_bytes)
        self._attribute_indexes = {}; self._prepared_rules = {}; self._attribute_code_arrays = {}

    @staticmethod
    def _stat(path):
        value = Path(path).stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def verify_code(self):
        if self.attribute_label_iteration != self._binding['attribute_label_iteration']['enabled']:
            raise RuntimeError('bound attribute label iteration policy changed')
        if self.compact_attribute_index != self._binding['attribute_index']['compact']:
            raise RuntimeError('bound attribute index policy changed')
        scope = self._application_scope
        if ((scope.receipt() if scope is not None else None) != self._binding.get('application_scope')):
            raise RuntimeError('bound rule application scope changed')
        for path, expected in self._code.items():
            now = self._stat(path)
            if now != self._code_stat[path]:
                if file_sha(path) != expected: raise RuntimeError('bound semantic source changed: ' + path)
                self._code_stat[path] = now

    @property
    def application_scope(self):
        return self._application_scope

    def application_allowed(self, rule_index, rule, valuation):
        if self._application_scope is None:
            return True
        allowed = self._application_scope.allows(rule_index, rule, valuation)
        if not allowed:
            self.stats['approved_rule_instances_skipped'] += 1
        return allowed

    @property
    def docs(self):
        return tuple(Document(cnt=d.cnt, ttl=d.ttl, mtd=d.mtd) for d in self._docs)

    @property
    def binding(self): return copy.deepcopy(self._binding)

    def _gamma(self, value):
        if not isinstance(value, (GammaSeed, RefGamma)) or value.shape != (self.n, self.k):
            raise ValueError('explicit full-axis GammaSeed/RefGamma required')
        if (type(value) is RefGamma and immutable_buffer(value.pos) and immutable_buffer(value.neg)
                and value.pos.dtype == bool and value.neg.dtype == bool):
            # Already the exact normalized form this method builds: validated
            # RefGamma (sorted unique edges) over permanently read-only bytes.
            return value
        result = RefGamma(value.pos, value.neg, getattr(value, 'subset', ()), getattr(value, 'strict_subset', ()))
        object.__setattr__(result, 'pos', _readonly(result.pos)); object.__setattr__(result, 'neg', _readonly(result.neg))
        return result

    def _seed_key(self, gamma):
        return sha({'pos': array_hash(gamma.pos), 'neg': array_hash(gamma.neg),
                    'subset': gamma.subset, 'strict_subset': gamma.strict_subset})

    def _seed_proof(self, gamma):
        key = self._seed_key(gamma)
        previous = self._seed_cache.get(key)
        if previous is not None: return previous
        # The immutable seed proof depends only on Gamma and n, so runners of
        # one process share it (2026-09-30).
        shared = _SHARED_SEED_PROOFS.get((key, self.n))
        if shared is not None:
            if len(self._seed_cache) >= 2: self._seed_cache.pop(next(iter(self._seed_cache)))
            self._seed_cache[key] = shared
            return shared
        result = {}
        for sign, matrix in [('pos', gamma.pos), ('neg', gamma.neg)]:
            for i, j in np.argwhere(matrix):
                result['%s:%d:%d' % (sign, i, j)] = {'kind': sign, 'row': int(i), 'column': int(j),
                    'origin': 'validated_Gamma', 'dependencies': [], 'rule_id': None, 'valuation': {}, 'static_evidence': [], 'support_rule_ids': []}
        for i in range(self.n):
            result['subset:%d:%d' % (i, i)] = {'kind': 'subset', 'row': i, 'column': i,
                'origin': 'identity', 'dependencies': [], 'rule_id': None, 'valuation': {}, 'static_evidence': [], 'support_rule_ids': []}
        result = ProofDAG(result)
        # Alternative Gamma (feedback weak roots) cannot grow this cache forever.
        if len(self._seed_cache) >= 2: self._seed_cache.pop(next(iter(self._seed_cache)))
        self._seed_cache[key] = result
        if len(_SHARED_SEED_PROOFS) >= 4: _SHARED_SEED_PROOFS.pop(next(iter(_SHARED_SEED_PROOFS)))
        _SHARED_SEED_PROOFS[(key, self.n)] = result
        return result

    def _rule(self, value):
        if not isinstance(value, Rule): raise TypeError('typed Rule required; legacy RDL is not an implicit adapter')
        previous = self._rule_cache.get(id(value))
        if previous is not None and previous[0] is value: return previous[1:]
        record = value.to_dict(); rule = Rule.from_dict(record); identifier = sha(record)
        for atom in rule.body + (rule.head,):
            if isinstance(atom, (Label, ML)) and atom.label not in self.label_index: raise ValueError('unknown frozen-axis label: ' + atom.label)
        self._rule_cache[id(value)] = (value, rule, identifier)
        return rule, identifier

    def program_key(self, rules, gamma=None):
        gamma = self.gamma if gamma is None else self._gamma(gamma)
        return sha({'context': self.context_id, 'program': [self._rule(rule)[1] for rule in rules], 'gamma': self._seed_key(gamma)})

    def _static(self, atom):
        record = atom.to_dict(); record.pop('var'); key = canonical(record)
        if key in self._static_bits: return self._static_bits[key]
        mask = self.masks.get(key)
        if mask is None:
            if isinstance(atom, Text):
                mask = self.lexical.mask(atom.predicate)
            else:
                table = self.probabilities if atom.dynamic else self.static_probabilities
                if atom.alias not in table: raise ValueError('missing explicit model probability table: ' + atom.alias)
                column = table[atom.alias][:, self.label_index[atom.label]]
                mask = column >= atom.eta
            mask = self.masks.put(key, mask); self.stats['static_computations'] += 1
        result = bits(mask); self._static_bits[key] = result
        return result

    def _attribute_codes(self, left, right):
        """The code matrix of _attribute_index(left, right) without its row groups."""
        key = (left, right)
        if left not in self.attribute_values or right not in self.attribute_values:
            raise ValueError('every named attribute must be explicitly declared on this context')
        if key in self._attribute_indexes: return self._attribute_indexes[key][0]
        saved = self._attribute_code_arrays.get(key)
        if saved is not None: return saved
        codes = self.attribute_codes.get(key)
        if codes is None:
            codes = self._encode_attributes(key)
        self._attribute_code_arrays[key] = codes
        return codes

    def _encode_attributes(self, key):
        lookup = {}; columns = []
        for attr in key:
            values = np.full(self.n, -1, np.int64)
            for i, item in enumerate(self.attribute_values[attr]):
                if self.presence[attr][i]: values[i] = lookup.setdefault(item, len(lookup))
            columns.append(values)
        return self.attribute_codes.put(key, np.stack(columns))

    def _attribute_index(self, left, right):
        key = (left, right)
        if left not in self.attribute_values or right not in self.attribute_values:
            raise ValueError('every named attribute must be explicitly declared on this context')
        if key in self._attribute_indexes: return self._attribute_indexes[key]
        codes = self._attribute_code_arrays.get(key)
        if codes is None:
            codes = self.attribute_codes.get(key)
        if codes is None:
            codes = self._encode_attributes(key)
        if self.compact_attribute_index:
            left_groups = CompactAttributeIndex(codes[0])
            # Identical attributes have exactly the same presence and row axis.
            groups = [left_groups, left_groups if left == right else CompactAttributeIndex(codes[1])]
        else:
            groups = []
            for column in codes:
                mapping = {}
                for i, value in enumerate(column.tolist()):
                    if value >= 0: mapping[value] = mapping.get(value, 0) | (1 << i)
                groups.append(mapping)
        result = (codes, groups); self._attribute_indexes[key] = result
        return result

    def _prepare(self, rule, key):
        if key in self._prepared_rules: return self._prepared_rules[key]
        unary = {v: [] for v in rule.variables}; binary = []
        static = {v: [] for v in rule.variables}
        for atom in rule.body:
            if isinstance(atom, (Text, ML)): static[atom.var].append(atom)
            elif isinstance(atom, Label): unary[atom.var].append(atom)
            elif isinstance(atom, (AttrEqual, LabelCompare)):
                binary.append(atom)
                # Codes are validated/cached now; row groups are built on first use.
                if isinstance(atom, AttrEqual): self._attribute_codes(atom.left_attr, atom.right_attr)
        domains = {}
        for var, atoms in static.items():
            records = [a.to_dict() for a in atoms]
            for record in records: record.pop('var')
            conjunction_key = canonical(records)
            mask = self.conjunctions.get(conjunction_key)
            if mask is None:
                domain = self._all
                for atom in atoms: domain &= self._static(atom)
                mask = unbits(domain, self.n)
                mask = self.conjunctions.put(conjunction_key, mask)
            domains[var] = bits(mask)
        prepared = {'unary': unary, 'binary': binary, 'domains': domains}
        self._prepared_rules[key] = prepared
        return prepared

    def precompute(self, rules):
        self.verify_code()
        for value in rules:
            rule, key = self._rule(value); self._prepare(rule, key)
        return {'new_Chase_calls': 0, 'audit': self.audit()}

    def _relation_domain(self, atom, var, assignment, state):
        left, right = atom.left_var, atom.right_var
        if left == right:
            if isinstance(atom, AttrEqual):
                codes = self._attribute_codes(atom.left_attr, atom.right_attr)
                return bits((codes[0] >= 0) & (codes[0] == codes[1]))
            return self._all if atom.op != 'strict_subset' else sum(1 << i for i in range(self.n) if state.proper[i] & (1 << i))
        other = right if var == left else left
        if other not in assignment: return self._all
        row = assignment[other]
        if isinstance(atom, AttrEqual):
            codes, groups = self._attribute_index(atom.left_attr, atom.right_attr)
            side = 0 if var == left else 1
            return groups[side].get(int(codes[1 - side, row]), 0)
        source_left = self.subset_direction == 'standard' or atom.op == 'eq'
        variable_source = (var == left) == source_left
        if atom.op == 'eq': return state.out[row] & state.inc[row]
        out, inc = (state.proper, state.proper_inc) if atom.op == 'strict_subset' else (state.out, state.inc)
        return inc[row] if variable_source else out[row]

    def _strict_diagonal_valuations(self, rule, prepared, state):
        """Yield only authentic full-body self-inclusion witnesses.

        This is an execution-order optimization, not a candidate filter. Extra
        body variables and explicit application scopes use the unchanged join.
        No label bitmap is interpreted as a set-comparison premise: relation
        atoms are checked against the same actual relation state as the join.
        The caller still constructs evidence, applies the head, and checks the
        ordinary strict-cycle failure. A successful state cannot contain an
        applied strict diagonal; no old failed partial-state identity is claimed.
        """
        head = rule.head
        if (not isinstance(head, LabelCompare) or head.op != 'strict_subset'
                or set(rule.variables) != {head.left_var, head.right_var}
                or len(rule.variables) > 2 or self._application_scope is not None):
            return
        domain = self._all
        for var in rule.variables:
            domain &= prepared['domains'][var]
            for atom in prepared['unary'][var]:
                domain &= state.label_bits[atom.polarity][self.label_index[atom.label]]
        for row in indices(domain):
            valuation = {var: row for var in rule.variables}
            bit = 1 << row
            if any(not (self._relation_domain(atom, atom.left_var, valuation, state) & bit)
                   for atom in prepared['binary']):
                continue
            self.stats['strict_diagonal_body_witnesses'] += 1
            self.stats['matched_valuations'] += 1
            yield valuation

    def _attribute_label_valuations(self, rule, prepared, state):
        """Original target-first join order without a full-width mask per row.

        This fragment copies a signed label between equal values of one
        attribute, intersecting Text/ML conditions and other label columns.
        With identity-only relations, this rule only changes its head label;
        other label columns are therefore fixed throughout this enumeration.
        Without nonidentity relations, inserting the head can
        only add that same label inside an already active attribute group.
        Therefore the group activity snapshot cannot become stale. Individual
        row labels remain live: a newly inserted earlier target may be the
        original matcher's first source witness for a later target. Protected
        targets still yield every currently matching source, preserving all
        blocked groundings and the ordinary head/conflict/proof path.
        """
        head = rule.head
        if (not self.attribute_label_iteration or not self.compact_attribute_index
                or not isinstance(head, Label) or len(rule.variables) != 2
                or rule.variables[0] != head.var or len(rule.body) < 4
                or self._application_scope is not None
                or getattr(state, 'relation_version', None) != 0
                or not getattr(state, '_positive_closed', True)):
            return None
        labels = [a for a in rule.body if isinstance(a, Label)]
        attrs = [a for a in rule.body if isinstance(a, AttrEqual)]
        docs = [a for a in rule.body if isinstance(a, Doc)]
        copying = [a for a in labels if a.var != head.var
                   and a.label == head.label and a.polarity == head.polarity]
        if (len(copying) != 1 or len(attrs) != 1 or len(docs) != 2
                or any(not isinstance(a, (Doc, Label, AttrEqual, Text, ML))
                       for a in rule.body)):
            return None
        premise, atom = copying[0], attrs[0]
        fixed_labels = [a for a in labels if a is not premise]
        if any(a.label == head.label for a in fixed_labels):
            return None
        if (premise.var == head.var or premise.label != head.label
                or premise.polarity != head.polarity
                or atom.left_var == atom.right_var
                or atom.left_attr != atom.right_attr
                or {a.var for a in docs} != set(rule.variables)
                or {atom.left_var, atom.right_var} != set(rule.variables)):
            return None
        codes, groups = self._attribute_index(atom.left_attr, atom.right_attr)
        group = groups[0]
        matrix = state.pos if head.polarity == 'pos' else state.neg
        label = self.label_index[head.label]
        live_column = matrix[:, label]
        fixed_domains = dict(prepared['domains'])
        for condition in fixed_labels:
            fixed_domains[condition.var] &= state.label_bits[condition.polarity][self.label_index[condition.label]]
        source_mask = unbits(fixed_domains[premise.var], self.n)
        active = group.matching_groups(live_column & source_mask)
        # Code order need not be contiguous; vectorized positions avoid one
        # binary search and one N-bit bitmap allocation for each target row.
        positions = np.searchsorted(group.codes, codes[0])
        target_mask = unbits(fixed_domains[head.var], self.n)
        target_rows = np.flatnonzero(target_mask & ~live_column)

        def ordered():
            for raw_target in target_rows:
                target = int(raw_target)
                if live_column[target]:
                    continue
                # The general join counts this outer prefix even if its
                # attribute is absent or no source in its group has the label.
                self.stats['indexed_prefixes'] += 1
                if codes[0, target] < 0:
                    continue
                position = int(positions[target])
                if not active[position]:
                    continue
                for raw_source in group.group_rows(position):
                    if live_column[target]:
                        break
                    source = int(raw_source)
                    if not source_mask[source] or not live_column[source]:
                        continue
                    self.stats['indexed_prefixes'] += 1
                    self.stats['matched_valuations'] += 1
                    yield {head.var: target, premise.var: source}
        return ordered()

    def _valuations(self, rule, prepared, state):
        attribute_values = self._attribute_label_valuations(rule, prepared, state)
        if attribute_values is not None:
            yield from attribute_values
            return
        # For a unary Label head, inserting that head cannot make another
        # body's label column true. A same-sign head premise already implies
        # the head (and is excluded), while an opposite-sign premise is never
        # modified by this insertion. Hence the body domain is fixed. Positive
        # propagation may establish the head on a later candidate; check that
        # one cell before yielding, preserving the original ascending order.
        # The original proof states propagate synchronously before returning;
        # the selection state additionally exposes a closure certificate. An
        # uncertified selection state retains the general matcher below.
        # This avoids rebuilding an N-bit domain for each of its matching rows.
        head = rule.head
        if (isinstance(head, Label) and len(rule.variables) == 1 and not prepared['binary']
                and getattr(state, '_positive_closed', True)):
            var = rule.variables[0]
            label = self.label_index[head.label]
            domain = prepared['domains'][var]
            for atom in prepared['unary'][var]:
                domain &= state.label_bits[atom.polarity][self.label_index[atom.label]]
            domain &= ~state.label_bits[head.polarity][label]
            matrix = state.pos if head.polarity == 'pos' else state.neg
            for row in indices(domain):
                if matrix[row, label]:
                    continue
                self.stats['indexed_prefixes'] += 1
                self.stats['matched_valuations'] += 1
                yield {var: row}
            return
        yield from self._strict_diagonal_valuations(rule, prepared, state)
        assignment = {}; variables = rule.variables
        def visit(depth):
            if depth == len(variables):
                self.stats['matched_valuations'] += 1
                yield dict(assignment)
                return
            var = variables[depth]; previous = -1
            while True:
                domain = prepared['domains'][var] & ~((1 << (previous + 1)) - 1)
                for atom in prepared['unary'][var]: domain &= state.label_bits[atom.polarity][self.label_index[atom.label]]
                for atom in prepared['binary']:
                    if var in (atom.left_var, atom.right_var): domain &= self._relation_domain(atom, var, assignment, state)
                # Once a head is known, no remaining valuation for that head
                # can add a fact or alter the first-witness proof.
                head = rule.head
                if isinstance(head, Label):
                    known = state.label_bits[head.polarity][self.label_index[head.label]]
                    if head.var == var: domain &= ~known
                    elif head.var in assignment and known & (1 << assignment[head.var]): return
                elif var in (head.left_var, head.right_var):
                    if head.left_var == head.right_var or (head.right_var if var == head.left_var else head.left_var) in assignment:
                        domain &= ~self._relation_domain(head, var, assignment, state)
                if not domain: break
                low = domain & -domain; previous = low.bit_length() - 1
                assignment[var] = previous
                self.stats['indexed_prefixes'] += 1
                yield from visit(depth + 1)
                del assignment[var]
        yield from visit(0)

    def _edge(self, atom, valuation):
        a, b = valuation[atom.left_var], valuation[atom.right_var]
        return (b, a) if self.subset_direction == 'paper_p7_reverse' and atom.op != 'eq' else (a, b)

    def _evidence(self, rule, valuation):
        dependencies, static = [], []
        for atom in rule.body:
            if isinstance(atom, Label): dependencies.append('%s:%d:%d' % (atom.polarity, valuation[atom.var], self.label_index[atom.label]))
            elif isinstance(atom, (Text, ML)): static.append({'atom': atom.to_dict(), 'row': valuation[atom.var]})
            elif isinstance(atom, AttrEqual): static.append({'atom': atom.to_dict(), 'rows': [valuation[atom.left_var], valuation[atom.right_var]]})
            elif isinstance(atom, LabelCompare):
                a, b = self._edge(atom, valuation)
                dependencies.append('%s:%d:%d' % ('strict_subset' if atom.op == 'strict_subset' else 'subset', a, b))
                if atom.op == 'eq': dependencies.append('subset:%d:%d' % (b, a))
        return dependencies, static

    def __call__(self, rules, gamma=None):
        self.verify_code()
        rules = tuple(self._rule(value) for value in rules)
        if self._application_scope is not None:
            self._application_scope.validate_rules([r for r, _ in rules])
        prepared = [self._prepare(rule, key) for rule, key in rules]
        gamma = self.gamma if gamma is None else self._gamma(gamma)
        state = _State(self, gamma); rounds = 0
        status = 'conflict' if state.invalid() else 'fixpoint'
        if status != 'conflict':
            while True:
                rounds += 1; changed = False
                for rule_id, ((rule, _), preparation) in enumerate(zip(rules, prepared)):
                    for valuation in self._valuations(rule, preparation, state):
                        if not self.application_allowed(rule_id, rule, valuation):
                            continue
                        dependencies, static = self._evidence(rule, valuation)
                        head = rule.head
                        if isinstance(head, Label):
                            changed |= state.set_label(head.polarity, valuation[head.var], self.label_index[head.label],
                                'rule_application', dependencies, rule_id, valuation, static)
                        else:
                            a, b = self._edge(head, valuation)
                            kind = 'strict_subset' if head.op == 'strict_subset' else 'subset'
                            # Equality establishes both directed subset facts.
                            # Install their authentic rule proofs before closure.
                            edges = [(a, b), (b, a)] if head.op == 'eq' else [(a, b)]
                            for x, y in edges: state.remember(kind, x, y, 'rule_application', dependencies, rule_id, valuation, static)
                            for x, y in edges: changed |= state.add_edge(x, y, head.op == 'strict_subset')
                        if state.invalid(): status = 'conflict'; break
                    if status == 'conflict': break
                if status == 'conflict' or not changed: break
        conflicts = [(int(i), self.labels[int(j)]) for i, j in np.argwhere(state.pos & state.neg)]
        result = ChaseResult(status=status, predictions=(state.pos & ~state.neg).astype(np.float32), conflicts=conflicts,
            n_rounds=rounds, provenance=state.provenance, positive_facts=state.pos.copy(), negative_facts=state.neg.copy())
        result.subset_facts = RelationFacts(state.out); result.strict_subset_facts = RelationFacts(state.proper)
        result.proof_dag = ProofDAG(state.proof, state.base_proof)
        result.comparison_conflicts = [state.conflict] if state.conflict and state.conflict['kind'] != 'positive_negative_label_conflict' else []
        self.stats['Chase_calls'] += 1
        result.reference_audit = {'semantics': SEMANTICS, 'executor': PROTOCOL, 'context_id': self.context_id,
            'full_rows': self.n, 'label_names': list(self.labels), 'subset_direction': self.subset_direction,
            'subset_direction_pending_author_confirmation': False,
            'subset_direction_author_confirmed': self.subset_direction == 'standard',
            'subset_direction_legacy_compatibility': self.subset_direction == 'paper_p7_reverse',
            'complete_without_cutoff': True,
            'program_sha256': sha([rule.to_dict() for rule, _ in rules]), 'comparison_premises_from_partial_bitmaps': False,
            'document_lbl_used_as_Gamma': False, 'model_outputs_unioned_into_Gamma': False,
            'negative_contraposition_added': False, 'finite_comparison_solver_calls': state.feasibility_checks,
            'finite_comparison_certificate_statistics': dict(state._comparison_oracle.stats),
            'relation_fact_representation': 'complete_immutable_bitset_rows',
            'proof_representation': 'exact_immutable_layered_proof_dag_v1',
            'rounds_and_first_witness_not_claimed_identical_to_reference': True,
            'incremental_relation_statistics': dict(state.stats)}
        if self._application_scope is not None:
            result.reference_audit['application_scope'] = self._application_scope.receipt()
        return result

    run = __call__

    def audit(self):
        return {'protocol': PROTOCOL, 'context_id': self.context_id, 'binding': self.binding, 'stats': dict(self.stats),
                'lexical_primitives': self.lexical.audit(),
                'caches': {name: cache.audit() for name, cache in [('masks', self.masks), ('conjunctions', self.conjunctions), ('attribute_codes', self.attribute_codes)]}}

    def close(self):
        self._db.close()
        self.lexical.close()
