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


def _install_namespace_accounting(db):
    """Atomically migrate old databases; row triggers maintain exact shared totals.

    The marker is written in the same transaction as backfill and triggers.
    A savepoint also works when a caller already owns an outer transaction.
    """
    # SQLite otherwise omits DELETE triggers for INSERT OR REPLACE. Normal
    # cache writes use INSERT OR IGNORE; enable correct replacement accounting
    # for every connection that instantiates this cache too.
    db.execute('PRAGMA recursive_triggers=ON')
    # In WAL mode, a deferred savepoint followed by a schema read may acquire
    # a read snapshot that cannot later upgrade after another writer commits.
    # Reserve the writer before any migration reads when we own the transaction;
    # retain savepoint/rollback semantics inside an existing caller transaction.
    owns_transaction = not db.in_transaction
    if owns_transaction:
        db.execute('BEGIN IMMEDIATE')
    savepoint_active = False
    try:
        db.execute('SAVEPOINT array_cache_accounting_install')
        savepoint_active = True
        db.execute('CREATE TABLE IF NOT EXISTS array_cache_schema(name TEXT PRIMARY KEY,version INTEGER NOT NULL)')
        # This write obtains SQLite's writer lock before migration state is read.
        db.execute("INSERT OR IGNORE INTO array_cache_schema VALUES('namespace_bytes',0)")
        version = db.execute("SELECT version FROM array_cache_schema WHERE name='namespace_bytes'").fetchone()[0]
        if version not in (0, 1):
            raise RuntimeError('unsupported persistent array cache accounting version')
        if version == 0:
            db.execute('CREATE TABLE array_cache_namespace_bytes(namespace TEXT PRIMARY KEY,bytes INTEGER NOT NULL)')
            db.execute('INSERT INTO array_cache_namespace_bytes SELECT namespace,COALESCE(SUM(bytes),0) FROM arrays GROUP BY namespace')
            db.execute("""CREATE TRIGGER array_cache_bytes_insert AFTER INSERT ON arrays BEGIN
                INSERT OR IGNORE INTO array_cache_namespace_bytes VALUES(NEW.namespace,0);
                UPDATE array_cache_namespace_bytes SET bytes=bytes+NEW.bytes WHERE namespace=NEW.namespace;
            END""")
            db.execute("""CREATE TRIGGER array_cache_bytes_delete AFTER DELETE ON arrays BEGIN
                UPDATE array_cache_namespace_bytes SET bytes=bytes-OLD.bytes WHERE namespace=OLD.namespace;
            END""")
            db.execute("""CREATE TRIGGER array_cache_bytes_update AFTER UPDATE OF namespace,bytes ON arrays BEGIN
                UPDATE array_cache_namespace_bytes SET bytes=bytes-OLD.bytes WHERE namespace=OLD.namespace;
                INSERT OR IGNORE INTO array_cache_namespace_bytes VALUES(NEW.namespace,0);
                UPDATE array_cache_namespace_bytes SET bytes=bytes+NEW.bytes WHERE namespace=NEW.namespace;
            END""")
            db.execute('CREATE INDEX array_cache_oldest ON arrays(namespace,created,key,bytes)')
            db.execute("UPDATE array_cache_schema SET version=1 WHERE name='namespace_bytes'")
        db.execute('RELEASE SAVEPOINT array_cache_accounting_install')
        savepoint_active = False
        if owns_transaction:
            db.commit()
    except BaseException:
        if savepoint_active:
            db.execute('ROLLBACK TO SAVEPOINT array_cache_accounting_install')
            db.execute('RELEASE SAVEPOINT array_cache_accounting_install')
        if owns_transaction:
            db.rollback()
        raise


class PersistentArrayCache:
    """Bounded RAM LRU plus transactional, SHA-checked SQLite array blobs.

    Limits evict storage only. A miss always recomputes the exact operation;
    corruption fails loudly. SQLite commits are atomic across interruptions.
    No pickle or executable object is stored.
    """
    def __init__(self, db, namespace, *, memory_bytes, disk_bytes):
        if disk_bytes < 0:
            raise ValueError('disk cache byte limit must be nonnegative')
        _install_namespace_accounting(db)
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
            total = self.db.execute('SELECT bytes FROM array_cache_namespace_bytes WHERE namespace=?',
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
