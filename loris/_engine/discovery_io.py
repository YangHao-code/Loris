"""Operational persistence policy; never a change to BO proposals or scores.

LORIS_DISCOVERY_IO=durable (default) retains per-artifact fsync. The opt-in
restartable profile uses SQLite WAL/NORMAL and atomic file replacement without
fsync. It supports process interruption while the OS/filesystem stays alive,
not machine/power loss. Resume still checks all receipts, RNG and SHA chains;
an inconsistent state is rejected, never repaired by resampling. After host
loss restore a stopped, consistent backup instead. Prefer durable on local SSD
when that already meets the throughput budget. Do not rsync active SQLite/WAL
and sampler files independently to create such a backup.
"""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import time


def profile():
    value = os.environ.get('LORIS_DISCOVERY_IO', 'durable')
    if value not in ('durable', 'restartable'):
        raise ValueError('LORIS_DISCOVERY_IO must be durable or restartable')
    return value


def sync_file(stream):
    stream.flush()
    if profile() == 'durable':
        os.fsync(stream.fileno())


def configure_sqlite(db):
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=' + ('NORMAL' if profile() == 'restartable' else 'FULL'))
    db.execute('PRAGMA busy_timeout=60000')
    return db


def optuna_storage(path):
    """Apply connection-local PRAGMAs before SQLAlchemy uses each connection."""
    import optuna
    path = str(Path(path).resolve())
    if profile() == 'durable':
        # Preserve the original Optuna storage configuration by default.
        return 'sqlite:///' + path

    def connect():
        db = sqlite3.connect(path, timeout=60, check_same_thread=False)
        try:
            return configure_sqlite(db)
        except BaseException:
            db.close()
            raise

    return optuna.storages.RDBStorage('sqlite:///' + path, engine_kwargs={'creator': connect})


def policy_record():
    mode = profile()
    return dict(schema='loris_discovery_io_v1', profile=mode,
                sqlite=('WAL/NORMAL' if mode == 'restartable' else 'original/FULL'),
                artifact_fsync=mode == 'durable', atomic_replace=True,
                score_and_sampler_protocol_unchanged=True,
                host_loss_recovery=('restore a stopped consistent backup; reject inconsistent live state'
                                    if mode == 'restartable' else 'validate all original receipts'))


class ProgressThrottle:
    """Only diagnostic progress is coalesced; trial/RNG commits are never skipped."""
    def __init__(self):
        default = '5' if profile() == 'restartable' else '0'
        self.seconds = float(os.environ.get('LORIS_DISCOVERY_PROGRESS_SECONDS', default))
        if not 0 <= self.seconds <= 3600:
            raise ValueError('discovery progress interval must be in [0, 3600] seconds')
        self.last = None

    def due(self, record):
        now = time.monotonic()
        final = record.get('complete') == record.get('budget') and 'budget' in record
        if self.last is None or final or now - self.last >= self.seconds:
            self.last = now
            return True
        return False
