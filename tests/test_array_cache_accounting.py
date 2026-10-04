import importlib.util
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SCHEMA = 'CREATE TABLE arrays(namespace TEXT,key TEXT,payload BLOB,sha256 TEXT,bytes INTEGER,created INTEGER,PRIMARY KEY(namespace,key))'


def load(name):
    from loris._engine import array_cache
    path = (HERE / 'fixtures/array_cache_before_v2.py' if name == 'original'
            else Path(array_cache.__file__))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def database(path=':memory:'):
    db = sqlite3.connect(str(path))
    db.execute('CREATE TABLE IF NOT EXISTS arrays(namespace TEXT,key TEXT,payload BLOB,sha256 TEXT,bytes INTEGER,created INTEGER,PRIMARY KEY(namespace,key))')
    return db


def cache(module, db, namespace='masks', limit=1 << 20):
    return module.PersistentArrayCache(db, namespace, memory_bytes=0, disk_bytes=limit)


def assert_accounting(db):
    actual = dict(db.execute('SELECT namespace,SUM(bytes) FROM arrays GROUP BY namespace'))
    recorded = dict(db.execute('SELECT namespace,bytes FROM array_cache_namespace_bytes'))
    assert {k: v for k, v in recorded.items() if v} == actual


@pytest.mark.parametrize('limit', [0, 1, 900, 1800, 1 << 20])
def test_exact_rows_outputs_stats_and_evictions(limit):
    modules = [load('original'), load('optimized')]
    connections = [database(), database()]
    for module in modules:
        # Fixed timestamps also test the existing key tie breaker.
        module.time = SimpleNamespace(time_ns=lambda: 42)
    caches = [[cache(m, d, ns, limit) for ns in ('a', 'b')]
              for m, d in zip(modules, connections)]
    for i in range(24):
        for m, cs in zip(modules, caches):
            value = np.arange(32, dtype=np.int32) + i
            result = cs[i % 2].put(('key', i), value)
            np.testing.assert_array_equal(result, value)
            assert not result.flags.writeable
        for key in [0, max(0, i - 2), i]:
            lhs, rhs = [cs[key % 2].get(('key', key)) for cs in caches]
            if lhs is None:
                assert rhs is None
            else:
                np.testing.assert_array_equal(lhs, rhs)
        rows = [d.execute('SELECT * FROM arrays ORDER BY namespace,key').fetchall() for d in connections]
        assert rows[0] == rows[1]
        assert [c.audit() for c in caches[0]] == [c.audit() for c in caches[1]]
        assert_accounting(connections[1])


def test_existing_database_backfill_only_once(tmp_path):
    original, optimized = load('original'), load('optimized')
    path = tmp_path / 'old.sqlite'
    db = database(path)
    for ns in ('a', 'b'):
        c = cache(original, db, ns)
        for i in range(20):
            c.put(i, np.arange(i + 1))
    rows = db.execute('SELECT * FROM arrays ORDER BY namespace,key').fetchall()
    db.close()
    db = database(path)
    trace = []
    db.set_trace_callback(trace.append)
    a = cache(optimized, db, 'a')
    b = cache(optimized, db, 'b')
    assert len([s for s in trace if s.startswith('INSERT INTO array_cache_namespace_bytes SELECT')]) == 1
    assert db.execute('SELECT * FROM arrays ORDER BY namespace,key').fetchall() == rows
    assert_accounting(db)
    np.testing.assert_array_equal(a.get(19), np.arange(20))
    db.close()
    db = database(path)
    trace = []
    db.set_trace_callback(trace.append)
    b = cache(optimized, db, 'b')
    assert not [s for s in trace if 'SUM(bytes)' in s]
    np.testing.assert_array_equal(b.get(19), np.arange(20))
    assert_accounting(db)


def test_multiple_instances_connections_and_limits(tmp_path):
    optimized = load('optimized')
    path = tmp_path / 'shared.sqlite'
    db1, db2 = database(path), database(path)
    a1, a2 = cache(optimized, db1, 'a', 900), cache(optimized, db2, 'a', 900)
    b1 = cache(optimized, db1, 'b', 1800)
    for i in range(20):
        (a1 if i % 2 else a2).put(i, np.arange(16) + i)
        b1.put(i, np.arange(16) + i)
        assert_accounting(db1)
        assert_accounting(db2)
        assert dict(db1.execute('SELECT namespace,bytes FROM array_cache_namespace_bytes'))['a'] <= 900
    db1.close()
    db2.close()


def test_duplicate_conflict_replacement_update_and_delete():
    optimized = load('optimized')
    db = database()
    c = cache(optimized, db)
    c.put('x', np.array([1, 2]))
    before = db.execute('SELECT * FROM arrays').fetchall()
    c.put('x', np.array([1, 2]))
    assert db.execute('SELECT * FROM arrays').fetchall() == before
    with pytest.raises(RuntimeError, match='different value'):
        c.put('x', np.array([9, 9]))
    assert db.execute('SELECT * FROM arrays').fetchall() == before
    assert_accounting(db)
    row = list(before[0])
    row[4] += 10
    with db:
        db.execute('INSERT OR REPLACE INTO arrays VALUES(?,?,?,?,?,?)', row)
    assert_accounting(db)
    with db:
        db.execute("UPDATE arrays SET namespace='other',bytes=bytes+100")
    assert_accounting(db)
    with db:
        db.execute('DELETE FROM arrays')
    assert_accounting(db)


def test_transaction_rollback_restores_rows_and_totals():
    optimized = load('optimized')
    db = database()
    c = cache(optimized, db)
    c.put('x', np.array([1, 2]))
    before = db.execute('SELECT * FROM arrays').fetchall()
    totals = db.execute('SELECT * FROM array_cache_namespace_bytes').fetchall()
    db.execute('BEGIN')
    db.execute('DELETE FROM arrays')
    assert_accounting(db)
    db.rollback()
    assert db.execute('SELECT * FROM arrays').fetchall() == before
    assert db.execute('SELECT * FROM array_cache_namespace_bytes').fetchall() == totals
    db.execute('CREATE TRIGGER abort_insert AFTER INSERT ON arrays BEGIN SELECT RAISE(ABORT,"injected"); END')
    with pytest.raises(sqlite3.IntegrityError, match='injected'):
        c.put('y', np.arange(5))
    assert db.execute('SELECT * FROM arrays').fetchall() == before
    assert db.execute('SELECT * FROM array_cache_namespace_bytes').fetchall() == totals


def test_install_rollback_and_reopen():
    optimized = load('optimized')
    db = database()
    db.execute('BEGIN')
    cache(optimized, db)
    assert db.in_transaction
    db.rollback()
    assert not db.execute("SELECT name FROM sqlite_master WHERE name='array_cache_schema'").fetchall()
    c = cache(optimized, db)
    c.put('x', np.arange(5))
    assert_accounting(db)


def test_blob_corruption_still_fails():
    optimized = load('optimized')
    db = database()
    c = cache(optimized, db)
    c.put('x', np.arange(5))
    with db:
        db.execute("UPDATE arrays SET payload=X'00'")
    with pytest.raises(RuntimeError, match='SHA mismatch'):
        c.get('x')


def test_efficient_lookup_and_oldest_plans():
    optimized = load('optimized')
    db = database()
    cache(optimized, db)
    total_plan = db.execute('EXPLAIN QUERY PLAN SELECT bytes FROM array_cache_namespace_bytes WHERE namespace=?', ('a',)).fetchall()
    oldest_plan = db.execute('EXPLAIN QUERY PLAN SELECT key,bytes FROM arrays WHERE namespace=? ORDER BY created,key LIMIT 1', ('a',)).fetchall()
    assert 'SEARCH' in str(total_plan) and 'SCAN' not in str(total_plan)
    assert 'COVERING INDEX array_cache_oldest' in str(oldest_plan)
    assert 'TEMP B-TREE' not in str(oldest_plan)


@pytest.mark.parametrize('journal', ['DELETE', 'WAL'])
def test_concurrent_connections_migrate_and_write(tmp_path, journal):
    path = tmp_path / 'concurrent.sqlite'
    db = database(path)
    db.execute('PRAGMA journal_mode=' + journal)
    db.close()
    barrier = threading.Barrier(2)

    def worker(identity):
        module = load('optimized')
        db = database(path)
        barrier.wait()
        for i in range(40):
            # Reopen cache views while the other connection writes. Production
            # prewarming does this on the shared WAL lexical database too.
            c = cache(module, db, 'shared', 1800)
            c.put((identity, i), np.arange(16) + i)
        db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, [0, 1]))
    db = database(path)
    assert_accounting(db)
    assert db.execute('SELECT bytes FROM array_cache_namespace_bytes').fetchone()[0] <= 1800


def test_abrupt_process_exit_rolls_back(tmp_path):
    path = tmp_path / 'crash.sqlite'
    module = load('optimized')
    db = database(path)
    c = cache(module, db)
    c.put('x', np.arange(5))
    before = db.execute('SELECT * FROM arrays').fetchall()
    totals = db.execute('SELECT * FROM array_cache_namespace_bytes').fetchall()
    db.close()
    script = "import os,sqlite3,sys; d=sqlite3.connect(sys.argv[1]); d.execute('BEGIN IMMEDIATE'); d.execute('DELETE FROM arrays'); os._exit(23)"
    completed = subprocess.run([sys.executable, '-c', script, str(path)], check=False)
    assert completed.returncode == 23
    db = database(path)
    cache(module, db)
    assert db.execute('SELECT * FROM arrays').fetchall() == before
    assert db.execute('SELECT * FROM array_cache_namespace_bytes').fetchall() == totals
    assert_accounting(db)
