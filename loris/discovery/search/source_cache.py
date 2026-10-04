"""Cache repeated source extraction without removing method/source checks.

Each method still passes the original expected SHA and identity checks. A
cached extraction is used only while its code object and source inode, size,
mtime and ctime are unchanged. BO boundaries also perform the original full
source check. The cache is local to one new worker, never injected live.
"""
from contextlib import contextmanager
import inspect
from pathlib import Path

from loris._storage import state_execution


class MethodHashes:
    def __init__(self, original):
        self.original = original
        self.values = {}
        self.hits = self.misses = 0

    def __call__(self, method):
        path = Path(inspect.getsourcefile(method))
        stat = path.stat()
        identity = (method.__code__, stat.st_dev, stat.st_ino, stat.st_size,
                    stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self.values.get(method)
        if cached is not None and cached[0] == identity:
            self.hits += 1
            return cached[1]
        value = self.original(method)
        self.values[method] = (identity, value)
        self.misses += 1
        return value

    def verify(self):
        for method, (identity, value) in self.values.items():
            if method.__code__ is not identity[0] or self.original(method) != value:
                raise RuntimeError('BO source/code changed during cached verification')


@contextmanager
def cached_source_extraction():
    original = state_execution._method_sha
    cache = MethodHashes(original)
    state_execution._verify_original()
    state_execution._method_sha = cache
    try:
        yield cache
        cache.verify()
    finally:
        state_execution._method_sha = original
    state_execution._verify_original()
