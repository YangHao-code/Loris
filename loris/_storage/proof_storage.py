"""Exact layered proof storage with one canonical encoding per typed record.

The proof algorithm, insertion order, first witnesses, schema, and canonical
bytes are unchanged. Standard typed records are frozen as owned JSON bytes;
hashing reuses those bytes without recreating millions of immutable mappings.
Other values retain recursive freezing and bounded JSON staging.

The optional runner factory clones only the two original methods that construct
ProofDAG.  It is an explicitly bound, isolated execution adapter, not a global
monkey patch and not authorization to change an existing experiment protocol.
"""
from collections.abc import Mapping
import copy
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import FunctionType, MappingProxyType

from loris._engine.array_cache import file_sha, sha
from loris._engine.typed_runtime import IndexedChaseRunner, ProofDAG


PROTOCOL = 'paper36_exact_layered_proof_storage_acceleration_v2'
_JSON_OPTIONS = dict(sort_keys=True, ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False)
_ENCODER = json.JSONEncoder(**_JSON_OPTIONS)


def fast_immutable(value):
    """Match typed_runtime._immutable, including container subclasses."""
    kind = type(value)
    if kind is str or kind is int or value is None or kind is bool or kind is float:
        return value
    if kind is dict or kind is MappingProxyType:
        return MappingProxyType({key: fast_immutable(item) for key, item in value.items()})
    if kind is list or kind is tuple:
        return tuple(fast_immutable(item) for item in value)
    # Retain the original semantics for unusual user supplied Mapping/list
    # subclasses.  The common primitive path avoids repeated ABC checks.
    if isinstance(value, Mapping):
        return MappingProxyType({key: fast_immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(fast_immutable(item) for item in value)
    return value


def fast_plain(value):
    """Match typed_runtime._plain without ABC checks for every primitive."""
    kind = type(value)
    if kind is str or kind is int or value is None or kind is bool or kind is float:
        return value
    if kind is dict or kind is MappingProxyType:
        return {key: fast_plain(item) for key, item in value.items()}
    if kind is list or kind is tuple:
        return [fast_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {key: fast_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [fast_plain(item) for item in value]
    return value


def iter_canonical_chunks(value):
    """Yield exactly canonical(fast_plain(value)), with ordinary string keys.

    Non-string keys and exotic mappings use the standard encoder over a plain
    copy, preserving its key conversion, sort behavior, and rejection cases.
    The fast path stages at most one record plus the sorted top-level key list;
    it does not claim constant space for a single exceptionally large record.
    """
    kind = type(value)
    if (kind is dict or kind is MappingProxyType) and all(type(key) is str for key in value):
        yield '{'
        first = True
        for key in sorted(value):
            if not first:
                yield ','
            first = False
            yield _ENCODER.encode(key)
            yield ':'
            yield _ENCODER.encode(fast_plain(value[key]))
        yield '}'
    else:
        yield from _ENCODER.iterencode(fast_plain(value))


def canonical_sha256(value):
    result = hashlib.sha256()
    for chunk in iter_canonical_chunks(value):
        result.update(chunk.encode('utf-8'))
    return result.hexdigest()


_PROOF_FIELDS = frozenset(('kind', 'row', 'column', 'origin', 'dependencies',
                         'rule_id', 'valuation', 'static_evidence', 'support_rule_ids'))


def _plain_json(value):
    """Only exact JSON types; unusual containers retain the original path."""
    kind = type(value)
    if kind in (str, int, bool) or value is None:
        return True
    if kind is float:
        return math.isfinite(value)
    if kind in (list, tuple):
        return all(_plain_json(item) for item in value)
    if kind is dict:
        return all(type(key) is str and _plain_json(item) for key, item in value.items())
    return False


def _standard_record(value):
    """Recognize exactly the JSON-safe records written by typed _State.

    Empty valuations and static evidence dominate transitive closure. Checking
    their known schema avoids walking and rebuilding each record twice.
    """
    if type(value) is not dict or value.keys() != _PROOF_FIELDS:
        return False
    if (type(value['kind']) is not str or type(value['origin']) is not str
            or type(value['row']) is not int or type(value['column']) is not int
            or (value['rule_id'] is not None and type(value['rule_id']) is not int)):
        return False
    for name, scalar in (('dependencies', str), ('support_rule_ids', int)):
        values = value[name]
        if type(values) not in (list, tuple) or any(type(item) is not scalar for item in values):
            return False
    valuation = value['valuation']
    return (type(valuation) is dict
            and all(type(key) is str and type(row) is int for key, row in valuation.items())
            and type(value['static_evidence']) in (list, tuple)
            and _plain_json(value['static_evidence']))


class _EncodedRecord(bytes):
    """An immutable canonical record, distinct from unsupported raw bytes."""
    __slots__ = ()


class _FrozenProofLayer(Mapping):
    """Owned immutable record bytes, with lazy immutable views for lookups.

    During Chase only the small validated base is read. Derived proof records
    are normally consumed by identity(), which never needs those views.
    """
    def __init__(self, records):
        values = {}
        for key, value in records.items():
            if _standard_record(value):
                try:
                    values[key] = _EncodedRecord(_ENCODER.encode(value).encode('utf-8'))
                    continue
                except (ValueError, UnicodeError, OverflowError, RecursionError):
                    # Preserve deferred JSON errors, including lone surrogates.
                    pass
            values[key] = fast_immutable(value)
        self._values = MappingProxyType(values)
        self._decoded = {}

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __contains__(self, key):
        return key in self._values

    def __getitem__(self, key):
        value = self._values[key]
        if type(value) is _EncodedRecord:
            if key not in self._decoded:
                self._decoded[key] = fast_immutable(json.loads(value))
            return self._decoded[key]
        return value

    def plain(self, key):
        value = self._values[key]
        return json.loads(value) if type(value) is _EncodedRecord else fast_plain(value)

    def canonical_sha256(self):
        result = hashlib.sha256(b'{')
        first = True
        for key in sorted(self._values):
            if not first:
                result.update(b',')
            first = False
            result.update(_ENCODER.encode(key).encode('utf-8'))
            result.update(b':')
            value = self._values[key]
            result.update(value if type(value) is _EncodedRecord else
                          _ENCODER.encode(fast_plain(value)).encode('utf-8'))
        result.update(b'}')
        return result.hexdigest()


class ExactProofDAG(ProofDAG):
    """Original ProofDAG interface, immutable values and identical identity."""
    def __init__(self, derived, base=None):
        self._derived = (_FrozenProofLayer(derived)
                         if type(derived) is dict and all(type(key) is str for key in derived)
                         else fast_immutable(derived))
        self._base = base
        if base is not None and any(key in base for key in self._derived):
            raise ValueError('derived proof may not shadow a validated proof')
        self._identity = None

    def __contains__(self, key):
        return key in self._derived or (self._base is not None and key in self._base)

    def identity(self):
        if self._identity is None:
            self._identity = {
                'schema': 'exact_immutable_layered_proof_dag_v1',
                'count': len(self),
                'local_sha256': (self._derived.canonical_sha256()
                                 if type(self._derived) is _FrozenProofLayer
                                 else canonical_sha256(self._derived)),
                'base_identity': self._base.identity() if self._base is not None else None,
            }
            self._identity['sha256'] = sha(self._identity)
        return copy.deepcopy(self._identity)

    def to_dict(self):
        if type(self._derived) is not _FrozenProofLayer:
            return {key: fast_plain(self[key]) for key in self}
        result = {key: self._derived.plain(key) for key in self._derived}
        if self._base is not None:
            result.update({key: fast_plain(self._base[key]) for key in self._base})
        return result


def storage_binding():
    """Explicit source identity for receipts and a future protocol migration."""
    source = str(Path(__file__).resolve())
    original = str(Path(inspect.getfile(ProofDAG)).resolve())
    return {
        'protocol': PROTOCOL,
        'proof_schema': 'exact_immutable_layered_proof_dag_v1',
        'implementation': {'path': source, 'sha256': file_sha(source)},
        'original_runtime': {'path': original, 'sha256': file_sha(original)},
        'preserves_first_witness_and_canonical_bytes': True,
        'changes_chase_algorithm': False,
        'changes_original_sources_or_globals': False,
        'standard_proof_records': 'owned_canonical_UTF8_bytes_with_lazy_immutable_views',
        'generic_values': 'original_recursive_freeze_and_canonical_encoding',
    }


def _clone_with_proof_class(function):
    if not inspect.isfunction(function) or 'ProofDAG' not in function.__code__.co_names:
        raise TypeError('expected an original Python method that directly constructs ProofDAG')
    if function.__globals__.get('ProofDAG') is not ProofDAG:
        raise ValueError('unexpected original ProofDAG binding; do not silently stack adapters')
    namespace = dict(function.__globals__)
    namespace['ProofDAG'] = ExactProofDAG
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    functools.update_wrapper(result, function)
    result.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return result


def make_exact_proof_runner_class(base=IndexedChaseRunner):
    """Build an isolated runner preserving logical context and cache identity.

    The separate ``proof_storage_binding`` authenticates changed execution code.
    A production caller must bind it in its new execution protocol; the unchanged
    ``context_id`` alone is deliberately not an execution-source receipt.
    """
    if not isinstance(base, type) or not issubclass(base, IndexedChaseRunner):
        raise TypeError('base must be an IndexedChaseRunner subclass')
    seed = _clone_with_proof_class(base._seed_proof)
    call = _clone_with_proof_class(base.__call__)

    class ExactProofStorageRunner(base):
        def __init__(self, *args, **kwargs):
            self._proof_storage_binding = storage_binding()
            super().__init__(*args, **kwargs)

        @property
        def proof_storage_binding(self):
            return copy.deepcopy(self._proof_storage_binding)

        def verify_code(self):
            super().verify_code()
            source = self._proof_storage_binding['implementation']
            if file_sha(source['path']) != source['sha256']:
                raise RuntimeError('bound proof storage source changed: ' + source['path'])

    ExactProofStorageRunner._seed_proof = seed
    ExactProofStorageRunner.__call__ = call
    ExactProofStorageRunner.run = call
    return ExactProofStorageRunner
