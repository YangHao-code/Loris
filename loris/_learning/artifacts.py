"""Small, durable, source-bound artifacts shared by the aligned workflow."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
from collections.abc import Mapping

import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value.resolve())
    if is_dataclass(value):
        return {key: plain(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [plain(item) for item in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}


def bound_path(record, base=None):
    if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
        raise ValueError('each input requires path and SHA256')
    path = Path(record['path'])
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    actual = file_record(path)
    if actual['sha256'] != record['sha256']:
        raise ValueError('input checksum changed: ' + str(path))
    return Path(actual['path'])


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                      allow_nan=False, indent=2) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def append_event(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(plain(value), sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())

