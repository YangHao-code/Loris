"""Audited storage-only adapter for resuming an unchanged historical release.

Example (use the original PYTHONPATH and original entry script/options):
  python discovery_fast_resume.py --audit RUN/IO_ADAPTER_ATTEMPTS.jsonl \
      --script original_driver.py -- original arguments

No installed source, proposal, evaluator, identity check, or checkpoint field
is changed. The audit records the original source files and adapter hash. Only
file synchronization and SQLite durability change. All original resume checks
still execute. This is an explicit restartable I/O run: process interruption is
supported, host/power-loss recovery requires a stopped consistent backup. It is
not a way to load checkpoints with a changed selection engine or sampler.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import runpy
import sqlite3
import sys
import time


def install(audit_path):
    from loris._engine import rlearner, compact_storage
    if getattr(rlearner, '_restartable_io_adapter', False):
        raise RuntimeError('storage adapter is already installed')
    original_connect = sqlite3.connect

    class RestartableConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if ''.join(sql.upper().split()).rstrip(';') == 'PRAGMASYNCHRONOUS=FULL':
                sql = 'PRAGMA synchronous=NORMAL'
            return super().execute(sql, parameters)

    def connect(*args, **kwargs):
        if 'factory' in kwargs or len(args) > 5:
            raise ValueError('custom SQLite connection factory outside audited adapter scope')
        kwargs['factory'] = RestartableConnection
        db = original_connect(*args, **kwargs)
        database = str(args[0] if args else kwargs['database'])
        try:
            if 'mode=ro' not in database and 'immutable=1' not in database:
                db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=NORMAL')
            db.execute('PRAGMA busy_timeout=60000')
            return db
        except BaseException:
            db.close()
            raise

    def atomic_bytes(path, raw):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
        with temporary.open('wb') as stream:
            stream.write(raw)
            stream.flush()
        os.replace(temporary, path)

    def atomic_json(path, value):
        # Match the original writer's locale encoding, JSON representation and
        # trailing newline exactly. Atomic replacement/ordering is unchanged.
        import locale
        atomic_bytes(path, (rlearner.canonical(value) + '\n').encode(locale.getpreferredencoding(False)))

    def atomic_pickle(path, value):
        raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_bytes(path, raw)
        return hashlib.sha256(raw).hexdigest()

    def immutable_bytes(path, raw):
        path = Path(path)
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError('immutable compact artifact differs: ' + str(path))
        else:
            atomic_bytes(path, raw)

    source_root = Path(rlearner.__file__).resolve().parents[1]
    source_files = [Path(rlearner.__file__), Path(compact_storage.__file__),
                    source_root / '_engine/primitive_masks.py',
                    source_root / '_storage/selection_evaluator.py',
                    source_root / '_storage/selection_execution.py',
                    source_root / 'discovery/runner.py',
                    source_root / 'discovery/search/sampler.py',
                    source_root / 'discovery/search/storage.py']
    record = dict(schema='loris_audited_restartable_io_adapter_v1',
                  adapter_path=str(Path(__file__).resolve()),
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_files={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source_files},
                  started_unix=time.time(), pid=os.getpid(),
                  changes=['atomic writers retain flush/replace, omit fsync',
                           'new SQLite connections WAL/NORMAL; cache FULL redirected to NORMAL'],
                  semantic_code_and_all_resume_checks_unchanged=True,
                  host_loss_recovery='restore a stopped consistent backup; never resample failed validation')
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    replacements = [(rlearner.atomic_json, atomic_json), (rlearner.atomic_pickle, atomic_pickle),
                    (compact_storage._immutable_bytes, immutable_bytes)]
    # Replace aliases already imported by library modules, if any. All immutable
    # validation routines and source-record functions remain the originals.
    for name, module in list(sys.modules.items()):
        if module is not None and (name == 'loris' or name.startswith('loris.')):
            for attr, value in list(vars(module).items()):
                for old, new in replacements:
                    if value is old:
                        setattr(module, attr, new)
    sqlite3.connect = sqlite3.dbapi2.connect = connect
    rlearner._restartable_io_adapter = True
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument('--script')
    entry.add_argument('--module')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    install(args.audit)
    remaining = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.argv = [args.script or args.module] + remaining
    if args.script:
        runpy.run_path(args.script, run_name='__main__')
    else:
        runpy.run_module(args.module, run_name='__main__', alter_sys=True)
