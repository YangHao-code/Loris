"""Private immutable probability snapshots with exact, bounded-lifetime identities.

Only an ndarray whose entire base chain ends in immutable bytes is memoized.
Mutable arrays, readonly views over mutable owners and mmap inputs retain full
content revalidation. Shape/dtype/strides are checked on every cache access.
"""
import hashlib
import weakref
import numpy as np
from loris._engine.hashing import _immutable_identity

_IDENTITIES = {}


def freeze_probability(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise TypeError('Probability snapshots cannot contain Python objects')
    # Replaces the existing private np.array(copy=True): one owned data copy,
    # not a readonly flag on a potentially writable external alias.
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)


def _digests(value):
    value = np.asarray(value)
    identity = _immutable_identity(value)
    immutable = identity is not None
    owner, signature = identity if immutable else (None, None)
    saved = _IDENTITIES.get(id(value)) if immutable else None
    if saved is not None and saved[0]() is value and saved[1] == signature and saved[3] is owner:
        return saved[2]
    c = np.ascontiguousarray(value)
    raw = memoryview(c).cast('B') if c.size else b''
    content = hashlib.sha256(raw).hexdigest()
    h = hashlib.sha256()
    h.update(str(c.dtype).encode())
    h.update(repr(c.shape).encode())
    h.update(raw)
    result = (tuple(c.shape), str(c.dtype), content, h.hexdigest())
    if immutable:
        key = id(value)
        def discard(ref):
            current = _IDENTITIES.get(key)
            if current is not None and current[0] is ref:
                _IDENTITIES.pop(key, None)
        ref = weakref.ref(value, discard)
        _IDENTITIES[key] = (ref, signature, result, owner)
    return result


def probability_identity(value):
    shape, dtype, content, _ = _digests(value)
    # A fresh dict/list prevents callers from changing a memoized receipt.
    return {'shape': list(shape), 'dtype': dtype, 'sha256': content}


def probability_hash(value):
    return _digests(value)[3]
