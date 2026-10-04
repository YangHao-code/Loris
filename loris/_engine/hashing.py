"""Stable numerical array identities."""
import hashlib
import os
import weakref

import numpy as np


def array_hash(value):
    cached = _immutable_lookup(_ARRAY_HASHES, value)
    if cached is not None:
        return cached
    value_c = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value_c.dtype).encode())
    h.update(repr(value_c.shape).encode())
    h.update(value_c.tobytes())
    result = h.hexdigest()
    _immutable_store(_ARRAY_HASHES, value, result)
    return result


# --- exact identity caches (2026-09-30 optimization) -------------------------
# An array whose buffer is ultimately an immutable ``bytes`` object and whose
# WRITEABLE flag is off cannot be changed through any NumPy handle (NumPy
# refuses to re-enable writing for it). Its content identity can therefore be
# reused only while its immutable buffer binding and metadata remain identical.
# Every other array is hashed on every call, exactly as before.
_ARRAY_HASHES = {}


def immutable_buffer(value):
    if not isinstance(value, np.ndarray) or value.flags.writeable:
        return False
    base = value
    while isinstance(base, (np.ndarray, memoryview)):
        if isinstance(base, np.ndarray):
            if base.flags.writeable:
                return False
            if base.base is None:
                return False
            base = base.base
        else:
            if not base.readonly:
                return False
            base = base.obj
    return isinstance(base, bytes)


def _immutable_identity(value):
    if not immutable_buffer(value):
        return None
    owner = value
    while isinstance(owner, (np.ndarray, memoryview)):
        owner = owner.base if isinstance(owner, np.ndarray) else owner.obj
    signature = (tuple(value.shape), str(value.dtype), tuple(value.strides),
                 value.__array_interface__['data'][0])
    return owner, signature


def _immutable_lookup(cache, value):
    identity = _immutable_identity(value)
    if identity is None:
        return None
    owner, signature = identity
    saved = cache.get(id(value))
    if (saved is not None and saved[0]() is value
            and saved[2] is owner and saved[3] == signature):
        return saved[1]
    return None


def _immutable_store(cache, value, result):
    identity = _immutable_identity(value)
    if identity is None:
        return
    owner, signature = identity
    key = id(value)
    def discard(ref):
        saved = cache.get(key)
        if saved is not None and saved[0] is ref:
            cache.pop(key, None)
    # Keep the immutable buffer alive until this array/cache entry disappears,
    # preventing object-id or data-pointer recycling after ndarray.data rebind.
    cache[key] = (weakref.ref(value, discard), result, owner, signature)


# Source files bound by verification checks are re-hashed only when their
# device, inode, size, mtime or ctime change (the same exact policy as
# IndexedChaseRunner.verify_code). Only used for source-code verification.
_SOURCE_SHA = {}


def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _cached_source(path):
    """(realpath, sha256) of a source file, recomputed when its stat identity changes.

    os.stat follows symbolic links, so a retargeted link changes the identity.
    """
    key = os.fspath(path)
    identity = _identity(os.stat(key))
    saved = _SOURCE_SHA.get(key)
    if saved is not None and saved[0] == identity:
        return saved[1], saved[2]
    real = os.path.realpath(key)
    digest = hashlib.sha256()
    with open(real, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    value = digest.hexdigest()
    if _identity(os.stat(key)) == identity and _identity(os.stat(real)) == identity:
        _SOURCE_SHA[key] = (identity, real, value)
    return real, value


def cached_file_sha(path):
    return _cached_source(path)[1]


def cached_file_record(path):
    real, value = _cached_source(path)
    if not os.path.isfile(real):
        raise FileNotFoundError(real)
    return {'path': real, 'sha256': value}
