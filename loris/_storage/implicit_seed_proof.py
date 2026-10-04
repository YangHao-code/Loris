"""Exact addressable Gamma/identity proofs without one object per root cell.

Logical keys, records and iteration order match the original seed ProofDAG.
The explicitly versioned identity hashes immutable bitmaps, not expanded JSON;
this is a storage migration and does not claim an identical old physical hash.
"""
import copy
import gzip
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np
from loris._engine.array_cache import _readonly, sha
from loris._engine.hashing import array_hash
from loris._engine.typed_runtime import ProofDAG
from loris._learning.artifacts import file_record

SCHEMA = 'paper36_exact_implicit_seed_proof_v1'


class ImplicitSeedProof(ProofDAG):
    def __init__(self, positive, negative):
        pos, neg = np.asarray(positive), np.asarray(negative)
        if (pos.dtype != np.bool_ or neg.dtype != np.bool_ or pos.ndim != 2
                or pos.shape != neg.shape or np.any(pos & neg)):
            raise ValueError('Consistent complete-axis Boolean Gamma bitmaps required')
        self.positive, self.negative = _readonly(pos), _readonly(neg)
        self.n, self.k = pos.shape
        self._count = int(np.count_nonzero(pos)) + int(np.count_nonzero(neg)) + self.n
        self._base = None
        # Explicit diagnostic expansion through inherited to_dict/as_dict is
        # still possible; production export recognizes this compact layer.
        self._derived = self
        self._identity = dict(schema=SCHEMA, count=self._count, shape=[self.n, self.k],
            positive_sha256=array_hash(self.positive), negative_sha256=array_hash(self.negative),
            identity_subset=True, key_order='positive_row_major_then_negative_row_major_then_identity',
            record_schema='original_typed_root_nine_fields_v1')
        self._identity['sha256'] = sha(self._identity)

    def __len__(self):
        return self._count

    def _parts(self, key):
        if type(key) is not str:
            raise KeyError(key)
        parts = key.split(':')
        try:
            kind, row, column = parts[0], int(parts[1]), int(parts[2])
        except (ValueError, IndexError):
            raise KeyError(key) from None
        if len(parts) != 3 or key != '%s:%d:%d' % (kind, row, column) or not 0 <= row < self.n:
            raise KeyError(key)
        if kind == 'subset':
            valid = column == row
        elif kind in ('pos', 'neg'):
            valid = 0 <= column < self.k and bool((self.positive if kind == 'pos' else self.negative)[row, column])
        else:
            valid = False
        if not valid:
            raise KeyError(key)
        return kind, row, column

    def __contains__(self, key):
        try:
            self._parts(key)
            return True
        except KeyError:
            return False

    def __getitem__(self, key):
        kind, row, column = self._parts(key)
        return MappingProxyType(dict(kind=kind, row=row, column=column,
            origin='identity' if kind == 'subset' else 'validated_Gamma',
            dependencies=(), rule_id=None, valuation=MappingProxyType({}),
            static_evidence=(), support_rule_ids=()))

    def __iter__(self):
        for kind, matrix in (('pos', self.positive), ('neg', self.negative)):
            for row in np.flatnonzero(matrix.any(axis=1)):
                for column in np.flatnonzero(matrix[row]):
                    yield '%s:%d:%d' % (kind, row, column)
        for row in range(self.n):
            yield 'subset:%d:%d' % (row, row)

    def identity(self):
        return copy.deepcopy(self._identity)

    def save(self, path):
        path = Path(path)
        if path.exists():
            previous = load(dict(bitmap=file_record(path), identity=self.identity()), path.parent)
            if previous.identity() != self.identity():
                raise ValueError('Existing compact root bitmaps differ')
        else:
            temporary = path.with_name(path.name + '.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, positive=self.positive, negative=self.negative)
            temporary.replace(path)
        return dict(schema=SCHEMA, identity=self.identity(), bitmap=file_record(path),
            expanded_root_count=len(self), complete_root_records_reconstructible=True)


def seed_proof(runner, gamma):
    """Factory-owned alternative to the original eager _seed_proof only."""
    key = SCHEMA + ':' + runner._seed_key(gamma)
    previous = runner._seed_cache.get(key)
    if previous is not None:
        return previous
    result = ImplicitSeedProof(gamma.pos, gamma.neg)
    if len(runner._seed_cache) >= 2:
        runner._seed_cache.pop(next(iter(runner._seed_cache)))
    runner._seed_cache[key] = result
    return result


def load(record, directory):
    reference = record['bitmap']
    path = Path(reference['path'])
    if not path.is_absolute():
        path = Path(directory) / path
    if file_record(path)['sha256'] != reference['sha256']:
        raise ValueError('Compact Gamma proof bitmap SHA differs')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'positive', 'negative'}:
            raise ValueError('Compact Gamma proof bitmap schema differs')
        result = ImplicitSeedProof(data['positive'], data['negative'])
    if result.identity() != record['identity']:
        raise ValueError('Compact Gamma proof bitmap identity differs')
    if record.get('expanded_root_count', len(result)) != len(result):
        raise ValueError('Compact Gamma proof root count differs')
    return result


def provenance_root_records(path, *, validate=True):
    """Collect and authenticate every root sidecar referenced by a proof stream."""
    path = Path(path)
    records = []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            value = json.loads(line)
            if value.get('type') == 'implicit_seed_roots':
                record = value['record']
                if record.get('schema') != SCHEMA:
                    raise ValueError('Unknown compact Gamma proof schema')
                if validate:
                    load(record, path.parent)
                records.append(record)
    return records


def validate_proof_roots(report, directory=None):
    """Validate all compact root layers declared by an authenticated proof report.

    Historical expanded-root reports contain no such layers and remain valid.
    A compact Merkle identity may never silently lose its external bitmap.
    """
    if isinstance(report, (str, Path)):
        path = Path(report)
        directory = path.parent
        report = json.loads(path.read_text())
    records = report.get('implicit_seed_roots', [])
    expected = []
    identity = report.get('proof_identity')
    while identity is not None:
        if identity.get('schema') == SCHEMA:
            expected.append(identity)
        identity = identity.get('base_identity')
    if [record.get('identity') for record in records] != expected:
        raise ValueError('Compact proof root references differ from the layered proof identity')
    for record in records:
        if record.get('schema') != SCHEMA or record.get('complete_root_records_reconstructible') is not True:
            raise ValueError('Incomplete compact proof root declaration')
        load(record, directory or '.')
    return records
