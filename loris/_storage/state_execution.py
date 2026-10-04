"""Exact state bookkeeping acceleration, preserving the original schedule.

Only complete Boolean-label reductions and empty strict-relation scans are
replaced by maintained facts. Positive row bitsets enumerate exactly the same
ascending label differences as numpy.flatnonzero. The full general closure,
finite-comparison oracle, first witnesses, and conflict order are retained.
This is an isolated, source-bound runner adapter; no original global is edited.
"""
import copy
import functools
import hashlib
import inspect
import os
from pathlib import Path
from types import FunctionType

import numpy as np

from loris._engine.hashing import cached_file_sha as file_sha
from loris._engine.typed_runtime import (
    IndexedChaseRunner, _State, RelationFacts, bits, indices, comparison_feasible, _gamma_blocks)


PROTOCOL = 'paper36_exact_incremental_state_bookkeeping_v1'
_EXPECTED_ORIGINAL = {
    '__init__': 'c934c00be833da5b2237ecff0db5b8e623763de3e17f84b15bbecb2f0fec6a5b',
    'set_label': '5726c95bec17516d7160abb5edcc1085344c7772e84aa89dbf87a482be6abc3e',
    'propagate_all_seed_labels': '96e02b1aa2b89b3d2ef2096e38423df3565aed6591956bc8b9683e7dff88fdc9',
    '_install': '064a1e677264ce7ce6340824b0dbd1ba67b81e60385e2ad72c196b378ac2d459',
    'add_edge': 'c33eb362449f8613941c19957b69578d911920d6af328579929ecec2ba859f8d',
    'invalid': '4d731ad3d04f1a3df9b7338da8e048b269a7691ee17edffa133e2f85f260baad',
}
_ORIGINAL_METHODS = {name: getattr(_State, name) for name in _EXPECTED_ORIGINAL}
_ORIGINAL_CALL = IndexedChaseRunner.__call__
# 2026-09-15: a6c2185 added explicit immutable application-scope validation,
# per-valuation permission checks, and the scope receipt. With scope=None the
# remaining schedule AST is identical to 97c5d7e. This adapter copies the actual
# new function, including its scope checks, and still verifies the exact hash.
_EXPECTED_CALL = 'b43eef5f79a9039a20d21fedb01714f3328526cde7333b1a26bf53b24813dd38'


_METHOD_SHA_CACHE = {}


def _method_sha(method):
    """SHA of the exact method source.

    inspect.getsource already trusts linecache (size/mtime) for staleness.
    This cache is stricter: it re-extracts whenever the function's code object
    or the source file's device, inode, size, mtime_ns or ctime_ns differ.
    """
    path = inspect.getsourcefile(method)
    stat = os.stat(path)
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    code = getattr(method, '__code__', None)
    saved = _METHOD_SHA_CACHE.get(method)
    if saved is not None and saved[0] is code and saved[1] == identity:
        return saved[2]
    value = hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
    _METHOD_SHA_CACHE[method] = (code, identity, value)
    return value


def _verify_original():
    for name, expected in _EXPECTED_ORIGINAL.items():
        method = getattr(_State, name)
        if method is not _ORIGINAL_METHODS[name] or _method_sha(method) != expected:
            raise RuntimeError('unsupported original state method changed: ' + name)
    if (IndexedChaseRunner.__call__ is not _ORIGINAL_CALL
            or _method_sha(_ORIGINAL_CALL) != _EXPECTED_CALL
            or _ORIGINAL_CALL.__globals__.get('_State') is not _State):
        raise RuntimeError('unsupported original runner schedule changed')
    for name, method in _IMPLEMENTATION_METHODS.items():
        if getattr(ExactBookkeepingState, name) is not method:
            raise RuntimeError('bound accelerated state method changed: ' + name)


class ExactBookkeepingState(_State):
    def __init__(self, runner, gamma):
        # Install before the original constructor calls virtual insertion and
        # seed-propagation methods. The original owns all arrays and proofs.
        self._positive_row_bits = [bits(row) for row in gamma.pos]
        self._signed_overlap = bool(np.any(gamma.pos & gamma.neg))
        self._strict_edge_count = 0
        super().__init__(runner, gamma)

    def set_label(self, sign, i, j, origin, dependencies=(), rule_id=None,
                  valuation=None, static=(), propagate=True):
        matrix = self.pos if sign == 'pos' else self.neg
        if matrix[i, j]:
            return False
        if self.gamma_protection and _gamma_blocks(self, sign, i, j):
            return False
        matrix[i, j] = True
        if sign == 'pos':
            self._positive_row_bits[i] |= 1 << int(j)
            self._signed_overlap |= bool(self.neg[i, j])
        else:
            self._signed_overlap |= bool(self.pos[i, j])
        self.label_bits[sign][j] |= 1 << i
        self.version += 1
        self.remember(sign, i, j, origin, dependencies, rule_id, valuation, static)
        if sign == 'pos' and propagate:
            for target in indices(self.out[i] & ~self.label_bits['pos'][j]):
                self.set_label('pos', target, j, 'inclusion_label_copy',
                    [self.key('subset', i, target), self.key('pos', i, j)], propagate=False)
        return True

    def propagate_all_seed_labels(self):
        for source in self.out.nonidentity_sources():
            for target in indices(self.out[source] & ~(1 << source)):
                difference = self._positive_row_bits[source] & ~self._positive_row_bits[target]
                for j in indices(difference):
                    self.set_label('pos', target, j, 'inclusion_label_copy',
                        [self.key('subset', source, target), self.key('pos', source, j)], propagate=False)

    def _install(self, kind, source, target, dependencies, origin):
        changed = super()._install(kind, source, target, dependencies, origin)
        if changed and kind == 'strict_subset':
            self._strict_edge_count += 1
        return changed

    def add_edge(self, a, b, strict, initialize=False):
        # Keep the exact original edge and proof order; only the last label
        # difference and the presence test for strict edges are replaced.
        target_relation = self.proper if strict else self.out
        if target_relation[a] & (1 << b):
            return False
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
        changed = closes_cycle and bool(strict or self._strict_edge_count)
        while changed:
            changed = False
            for source in range(self.n):
                if not self.proper[source]:
                    continue
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
                difference = self._positive_row_bits[source] & ~self._positive_row_bits[target]
                # Snapshot, as with the original np.flatnonzero: earlier label
                # insertions may recursively establish later labels.
                for j in indices(difference):
                    self.set_label('pos', target, j, 'inclusion_label_copy',
                        [self.key('subset', source, target), self.key('pos', source, j)])
        return True

    def invalid(self):
        if self.last_valid_version == self.version:
            return self.conflict is not None
        if self._signed_overlap:
            self.conflict = {'kind': 'positive_negative_label_conflict'}
        elif self._strict_edge_count:
            if self._comparison_oracle.try_certify_rows(self.pos, self.neg, self.out, self.proper):
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
                if self.conflict:
                    break
            if self.conflict is None:
                self.feasibility_checks += 1
                if not comparison_feasible(self.pos, self.neg, set(RelationFacts(self.out)), set(RelationFacts(self.proper))):
                    self.conflict = {'kind': 'finite_label_comparisons_unsatisfiable'}
        self.last_valid_version = self.version
        return self.conflict is not None


_IMPLEMENTATION_METHODS = {name: getattr(ExactBookkeepingState, name)
                           for name in _EXPECTED_ORIGINAL}


_SOURCE_PATHS = {}


def _paths():
    if not _SOURCE_PATHS:
        _SOURCE_PATHS['source'] = str(Path(__file__).resolve())
        _SOURCE_PATHS['runtime'] = str(Path(inspect.getfile(_State)).resolve())
    return _SOURCE_PATHS['source'], _SOURCE_PATHS['runtime']


def source_stamp():
    """Content identity of the two bound sources (stat-cached SHA)."""
    source, runtime = _paths()
    return file_sha(source), file_sha(runtime)


def state_execution_binding():
    _verify_original()
    source, runtime = _paths()
    return {
        'protocol': PROTOCOL,
        'implementation': {'path': source, 'sha256': file_sha(source)},
        'original_runtime': {'path': runtime, 'sha256': file_sha(runtime)},
        'original_state_methods_sha256': copy.deepcopy(_EXPECTED_ORIGINAL),
        'original_schedule_sha256': _EXPECTED_CALL,
        'preserves_complete_facts_first_witnesses_rounds_and_conflicts': True,
        'signed_conflict': 'monotone_exact_cell_overlap_flag',
        'strict_presence': 'exact_successful_strict_insertion_count',
        'positive_row_difference': 'unbounded_integer_bits_in_original_ascending_label_order',
        'candidate_order_or_fixedpoint_changed': False,
        'original_sources_or_globals_modified': False,
    }


def make_exact_state_runner_class(base=IndexedChaseRunner):
    """Layer over the original runner or its exact ProofDAG-storage adapter."""
    _verify_original()
    if not isinstance(base, type) or not issubclass(base, IndexedChaseRunner):
        raise TypeError('base must be an IndexedChaseRunner subclass')
    function = base.__call__
    if (not inspect.isfunction(function) or function.__code__ is not _ORIGINAL_CALL.__code__
            or function.__globals__.get('_State') is not _State):
        raise ValueError('only the original schedule with its original _State may be adapted')
    namespace = dict(function.__globals__)
    namespace['_State'] = ExactBookkeepingState
    call = FunctionType(function.__code__, namespace, function.__name__,
                        function.__defaults__, function.__closure__)
    functools.update_wrapper(call, function)
    call.__kwdefaults__ = copy.copy(function.__kwdefaults__)

    class ExactStateExecutionRunner(base):
        def __init__(self, *args, **kwargs):
            self._state_execution_binding = state_execution_binding()
            super().__init__(*args, **kwargs)

        @property
        def state_execution_binding(self):
            return copy.deepcopy(self._state_execution_binding)

        def verify_code(self):
            super().verify_code()
            _verify_original()
            source = self._state_execution_binding['implementation']
            if file_sha(source['path']) != source['sha256']:
                raise RuntimeError('bound state execution source changed: ' + source['path'])

    ExactStateExecutionRunner.__call__ = call
    ExactStateExecutionRunner.run = call
    return ExactStateExecutionRunner
