"""Exact array cache utilities, independent of historical chase schedulers."""
from collections import Counter, OrderedDict
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp

def _size(value):
    if sp.issparse(value):
        return value.data.nbytes + value.indices.nbytes + value.indptr.nbytes
    return value.nbytes


class _LRU:
    def __init__(self, limit):
        if limit < 0:
            raise ValueError("cache byte limits must be nonnegative")
        self.limit = int(limit)
        self.bytes = 0
        self.entries = OrderedDict()
        self.evictions = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
        return value

    def put(self, key, value):
        size = _size(value)
        if size > self.limit:
            return value
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes -= _size(old)
        while self.bytes + size > self.limit and self.entries:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= _size(removed)
            self.evictions += 1
        self.entries[key] = value
        self.bytes += size
        return value

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _readonly(value):
    array = np.ascontiguousarray(value)
    # A non-writeable owner can become writeable again.  Reuse only arrays
    # backed by immutable bytes (possibly through NumPy/memoryview views).
    base = array
    while isinstance(base, (np.ndarray, memoryview)):
        if isinstance(base, np.ndarray):
            if base.base is None:
                break
            base = base.base
        else:
            if not base.readonly:
                break
            base = base.obj
    if not array.flags.writeable and isinstance(base, bytes):
        return array
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _frozen(value):
    if sp.issparse(value):
        value = value.tocsr(copy=True)
        value.data, value.indices, value.indptr = map(_readonly, (value.data, value.indices, value.indptr))
        return value
    return _readonly(value)


def _key_bytes(value):
    """Typed, length-delimited cache identity including full source-mask bytes."""
    if isinstance(value, bytes):
        return b'b' + str(len(value)).encode() + b':' + value
    if isinstance(value, tuple):
        parts = [_key_bytes(v) for v in value]
        return b't' + str(len(parts)).encode() + b':' + b''.join(parts)
    encoded = canonical(value).encode()
    return b'j' + str(len(encoded)).encode() + b':' + encoded


def _pack(value):
    stream = io.BytesIO()
    if sp.issparse(value):
        np.savez_compressed(stream, kind=np.array('csr'), shape=np.asarray(value.shape, np.int64),
                            data=value.data, indices=value.indices, indptr=value.indptr)
    else:
        np.savez_compressed(stream, kind=np.array('array'), value=np.asarray(value))
    return stream.getvalue()


def _unpack(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        kind = str(data['kind'])
        if kind == 'csr':
            result = sp.csr_matrix((data['data'], data['indices'], data['indptr']),
                                   shape=tuple(int(x) for x in data['shape']))
        elif kind == 'array':
            result = data['value']
        else:
            raise RuntimeError('unknown persistent cache value type')
    return _frozen(result)


class PersistentArrayCache:
    """Bounded RAM LRU plus transactional, SHA-checked SQLite array blobs.

    Limits evict storage only. A miss always recomputes the exact operation;
    corruption fails loudly. SQLite commits are atomic across interruptions.
    No pickle or executable object is stored.
    """
    def __init__(self, db, namespace, *, memory_bytes, disk_bytes):
        if disk_bytes < 0:
            raise ValueError('disk cache byte limit must be nonnegative')
        self.db, self.namespace = db, namespace
        self.memory = _LRU(memory_bytes)
        self.disk_limit = int(disk_bytes)
        self.stats = Counter()

    @property
    def bytes(self): return self.memory.bytes

    @property
    def entries(self): return self.memory.entries

    @property
    def evictions(self): return self.memory.evictions

    def get(self, key):
        identifier = hashlib.sha256(_key_bytes(key)).hexdigest()
        value = self.memory.get(identifier)
        if value is not None:
            self.stats['memory_hits'] += 1
            return value
        row = self.db.execute('SELECT payload,sha256 FROM arrays WHERE namespace=? AND key=?',
                              (self.namespace, identifier)).fetchone()
        if row is None:
            self.stats['misses'] += 1
            return None
        payload, expected = row
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError('persistent cache SHA mismatch: ' + self.namespace + '/' + identifier)
        value = _unpack(payload)
        self.stats['disk_hits'] += 1
        self.memory.put(identifier, value)
        return value

    def put(self, key, value):
        identifier = hashlib.sha256(_key_bytes(key)).hexdigest()
        value = _frozen(value)
        self.memory.put(identifier, value)
        if not self.disk_limit or _size(value) > self.disk_limit:
            self.stats['disk_storage_skips'] += 1
            return value
        payload = _pack(value)
        if len(payload) > self.disk_limit:
            self.stats['disk_storage_skips'] += 1
            return value
        digest = hashlib.sha256(payload).hexdigest()
        with self.db:
            previous = self.db.execute('SELECT sha256 FROM arrays WHERE namespace=? AND key=?',
                                      (self.namespace, identifier)).fetchone()
            if previous and previous[0] != digest:
                raise RuntimeError('same exact cache key produced a different value')
            self.db.execute('INSERT OR IGNORE INTO arrays VALUES(?,?,?,?,?,?)',
                (self.namespace, identifier, payload, digest, len(payload), time.time_ns()))
            total = self.db.execute('SELECT COALESCE(SUM(bytes),0) FROM arrays WHERE namespace=?',
                                    (self.namespace,)).fetchone()[0]
            while total > self.disk_limit:
                oldest = self.db.execute('SELECT key,bytes FROM arrays WHERE namespace=? ORDER BY created,key LIMIT 1',
                                         (self.namespace,)).fetchone()
                self.db.execute('DELETE FROM arrays WHERE namespace=? AND key=?', (self.namespace, oldest[0]))
                total -= oldest[1]
                self.stats['disk_evictions'] += 1
        self.stats['writes'] += 1
        return value

    def audit(self):
        row = self.db.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM arrays WHERE namespace=?',
                              (self.namespace,)).fetchone()
        return {'stats': dict(self.stats), 'memory_bytes': self.bytes,
            'memory_limit': self.memory.limit, 'memory_evictions': self.evictions,
            'disk_entries': row[0], 'disk_bytes': row[1], 'disk_limit': self.disk_limit}
