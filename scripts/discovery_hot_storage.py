"""Relocate stopped BO hot directories without changing any source/input binding.

Run with the ORIGINAL release still installed at its original absolute path:
  python scripts/discovery_hot_storage.py --output RUN --hot-root /ssd/loris

All discovery writers must be stopped. The output lock is required and held
throughout copying/verification/switching. Original directories are retained as
immutable rollback copies. Only BO cluster/score directories move; initialization,
lexical/prior paths and all receipt bytes stay unchanged. A symlink does not make
a different release compatible: normal exact resume checks remain mandatory.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def _manifest(directory):
    value = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('nested symlinks are not supported: ' + str(path))
        if path.is_file():
            value[str(path.relative_to(directory))] = [_sha(path), path.stat().st_size]
    return value


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def relocate(output, hot_root):
    output, hot_root = Path(output).resolve(), Path(hot_root).resolve()
    if not (output / 'INPUTS.json').is_file():
        raise ValueError('an existing discovery INPUTS.json is required')
    if (output / 'COMPLETE.json').exists():
        raise ValueError('completed archives must remain at their recorded resolved paths')
    if hot_root == output or output in hot_root.parents:
        raise ValueError('hot storage must be outside the run directory')
    hot_root.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('discovery is running; stop all writers before relocation')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        token = hashlib.sha256(str(output).encode()).hexdigest()[:12]
        destination = hot_root / (output.name + '-' + token + '-' + stamp)
        destination.mkdir(exist_ok=False)
        candidates = sorted((output / 'learning').glob('cluster_[0-9][0-9]'))
        if (output / 'worker_scores').is_dir():
            candidates.append(output / 'worker_scores')
        if not candidates:
            raise ValueError('no BO hot directories found')
        if any(path.is_symlink() for path in candidates):
            raise ValueError('already relocated directories require explicit manual rollback first')
        report_path = destination / 'RELOCATION.json'
        report = dict(schema='loris_stopped_hot_storage_relocation_v1', output=str(output),
                      source_inputs_sha256=_sha(output / 'INPUTS.json'), status='copying',
                      source_code_changed=False, original_paths_preserved_by_symlinks=True,
                      exact_resume_validation_required=True, directories=[])
        _write(report_path, report)
        for source in candidates:
            target = destination / str(source.relative_to(output)).replace('/', '__')
            before = _manifest(source)
            shutil.copytree(source, target)
            if _manifest(target) != before or _manifest(source) != before:
                raise RuntimeError('source changed during offline copy or copy checksum differs')
            for path in target.rglob('*'):
                if path.is_file():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
            for path in sorted(target.rglob('*'), reverse=True):
                if path.is_dir():
                    _sync_directory(path)
            _sync_directory(target)
            for path in target.rglob('*.sqlite'):
                db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
                try:
                    if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                        raise ValueError('copied SQLite failed quick_check: ' + str(path))
                finally:
                    db.close()
            # A read-only WAL integrity probe may recreate shared-memory files;
            # they contain no durable state. Bind the source copy before probe.
            backup = source.with_name(source.name + '.pre_hot_' + stamp)
            entry = dict(original=str(source), target=str(target), rollback=str(backup),
                         files=len(before), bytes=sum(v[1] for v in before.values()),
                         content_manifest_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                         status='verified')
            report['directories'].append(entry)
            _write(report_path, report)
            os.replace(source, backup)
            try:
                source.symlink_to(target, target_is_directory=True)
                _sync_directory(source.parent)
            except BaseException:
                if source.is_symlink():
                    source.unlink()
                os.replace(backup, source)
                raise
            entry['status'] = 'relocated'
            _write(report_path, report)
        report['status'] = 'relocated_exact_resume_required'
        _write(report_path, report)
        return report_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hot-root', required=True)
    args = parser.parse_args()
    print(relocate(args.output, args.hot_root))
