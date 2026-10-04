"""Bounded shared exact token statistics for canonical ASCII word regexes.

Non-ASCII rows and unsupported patterns retain Python re semantics. This is an
execution choice, never a pattern filter. Arrays returned are one regex's
count/first vectors; no pattern-by-document dense matrix is constructed.
"""
from array import array
from collections import Counter
import os
import re
import sys

import numpy as np


TOKEN = re.compile(r'[A-Za-z0-9_]+')
PREFIX, SUFFIX = r'\b(?:', r')\b'
FLAGS = (0, int(re.UNICODE), int(re.IGNORECASE), int(re.UNICODE | re.IGNORECASE))
_MISSING = object()


def configuration(policy=None, memory_bytes=None):
    policy = policy if policy is not None else os.environ.get('LORIS_SHARED_LITERAL_POLICY', 'auto')
    if policy not in ('off', 'auto', 'always'):
        raise ValueError('shared literal policy must be off, auto or always')
    memory_bytes = (int(os.environ.get('LORIS_SHARED_LITERAL_BYTES', 64 << 20))
                    if memory_bytes is None else memory_bytes)
    if type(memory_bytes) is not int or memory_bytes < 0:
        raise ValueError('shared literal memory bytes must be a nonnegative integer')
    return {'policy': policy, 'memory_bytes': memory_bytes,
            'minimum_misses': 32, 'minimum_scan_passes': 4,
            'domain': 'canonical_ascii_single_token_alternatives_ascii_rows',
            'accounting': 'Python dict/key/packed-array sizes plus conservative transient reserve checked before growth; output vector/source texts/RSS allocator overhead excluded'}


def canonical_forms(raw, flags, metadata=_MISSING):
    """Recognize the exact canonical grammar; never trust metadata alone.

    PatternSpec carries raw/flags but not anchor metadata, so its strict grammar
    can be reconstructed. Explicit noncanonical metadata still forces fallback.
    """
    if not isinstance(raw, str) or type(flags) is not int or flags not in FLAGS:
        return None
    if metadata is _MISSING:
        if not raw.startswith(PREFIX) or not raw.endswith(SUFFIX):
            return None
        forms = raw[len(PREFIX):-len(SUFFIX)].split('|')
    else:
        if not isinstance(metadata, (list, tuple)) or len(metadata) != 1:
            return None
        forms = metadata[0]
    if (not isinstance(forms, (list, tuple)) or not forms or
            any(not isinstance(v, str) or TOKEN.fullmatch(v) is None for v in forms)):
        return None
    expected = PREFIX + '|'.join(sorted(set(forms), key=lambda v: (-len(v), v))) + SUFFIX
    if raw != expected:
        return None
    return tuple(sorted({v.lower() if flags & re.IGNORECASE else v for v in forms}))


class SharedLiteralStats:
    """Adaptive, byte-limited postings with exact per-pattern fallback.

    Auto admission records original candidate scan work: at least 32 eligible
    cache misses and four ASCII-field character passes. Sparse tiny batches stay
    on the existing superset index. Thresholds choose execution only, not output.
    """
    def __init__(self, values, *, config=None, stats=None):
        self.values = values
        self.config = dict(config or configuration())
        self.stats = stats if stats is not None else Counter()
        self.indices = {}
        self.blocked = set()
        self.work = {}
        self.field_sizes = {}
        self.bytes = 0
        self.stats['shared_literal_budget_bytes'] = self.config['memory_bytes']
        self.stats['shared_literal_current_bytes'] = 0

    def _field_size(self, attr):
        if attr not in self.field_sizes:
            self.field_sizes[attr] = sum(len(t) for t in self.values[attr] if t.isascii())
        return self.field_sizes[attr]

    def _build(self, attr, ignorecase):
        index = {}
        allocated = sys.getsizeof(index)
        available = self.config['memory_bytes'] - self.bytes
        if allocated > available:
            self.stats['shared_literal_builds_over_budget'] += 1
            return None
        nonascii = False
        for row, text in enumerate(self.values[attr]):
            if not text.isascii():
                nonascii = True
                continue
            self.stats['shared_literal_token_rows_scanned'] += 1
            for match in TOKEN.finditer(text):
                # Do not first create an arbitrarily long token/lowercase copy
                # and only then discover that it cannot fit the scratch budget.
                token_reserve = 2 * (sys.getsizeof('') + match.end() - match.start())
                if allocated + token_reserve > available:
                    self.stats['shared_literal_builds_over_budget'] += 1
                    return None
                word = match.group()
                if ignorecase:
                    word = word.lower()
                posting = index.get(word)
                if posting is None:
                    # Check key size before allocating a new posting. Any
                    # residual dict/array allocator growth is bounded by one
                    # insertion and then checked immediately below.
                    # A dict resize can temporarily retain both old and new
                    # tables. Reserve two old-table sizes before insertion;
                    # actual persistent sizes are checked again afterwards.
                    reserve = token_reserve + sys.getsizeof(array('q', (0, 0, 0))) + 2 * sys.getsizeof(index)
                    if allocated + reserve > available:
                        self.stats['shared_literal_builds_over_budget'] += 1
                        return None
                    before = sys.getsizeof(index)
                    posting = array('q', (row, 1, match.start()))
                    index[word] = posting
                    allocated += sys.getsizeof(word) + sys.getsizeof(posting) + sys.getsizeof(index) - before
                elif posting[-3] == row:
                    posting[-2] += 1
                else:
                    before = sys.getsizeof(posting)
                    if allocated + token_reserve + 2 * before + 128 > available:
                        self.stats['shared_literal_builds_over_budget'] += 1
                        return None
                    posting.extend((row, 1, match.start()))
                    allocated += sys.getsizeof(posting) - before
                self.stats['shared_literal_token_occurrences'] += 1
                self.stats['shared_literal_peak_accounted_bytes'] = max(
                    self.stats['shared_literal_peak_accounted_bytes'], self.bytes + allocated)
                if allocated > available:
                    self.stats['shared_literal_builds_over_budget'] += 1
                    return None
        self.bytes += allocated
        self.stats['shared_literal_current_bytes'] = self.bytes
        self.stats['shared_literal_builds'] += 1
        return index, nonascii

    def maybe_counts(self, attr, raw, flags, *, metadata=_MISSING, candidate_rows=None):
        """Return exact (2,N) array or None to use the original evaluator."""
        if self.config['policy'] == 'off':
            self.stats['shared_literal_disabled'] += 1
            return None
        forms = canonical_forms(raw, flags, metadata)
        if forms is None:
            self.stats['shared_literal_unsupported'] += 1
            return None
        ignorecase = bool(flags & re.IGNORECASE)
        key = attr, ignorecase
        if key in self.blocked:
            self.stats['shared_literal_budget_fallbacks'] += 1
            return None
        if key not in self.indices:
            work = self.work.setdefault(key, [0, 0])
            work[0] += 1
            texts = self.values[attr]
            candidates = range(len(texts)) if candidate_rows is None else candidate_rows
            work[1] += sum(len(texts[i]) for i in candidates if texts[i].isascii())
            size = self._field_size(attr)
            if not size or (self.config['policy'] == 'auto' and
                    (work[0] < self.config['minimum_misses'] or
                     work[1] < self.config['minimum_scan_passes'] * size)):
                self.stats['shared_literal_auto_deferred'] += 1
                return None
            built = self._build(attr, ignorecase)
            if built is None:
                self.blocked.add(key)
                return None
            self.indices[key] = built
        index, nonascii = self.indices[key]
        result = np.zeros((2, len(self.values[attr])), dtype=np.int64)
        result[1] = -1
        for word in forms:
            posting = index.get(word)
            if posting is None:
                continue
            # Packed triples are row-unique per word and preserve char offsets.
            values = np.frombuffer(posting, dtype=np.int64).reshape(-1, 3)
            rows, counts, first = values.T
            result[0, rows] += counts
            old = result[1, rows]
            result[1, rows] = np.where(old < 0, first, np.minimum(old, first))
        if nonascii:
            compiled = re.compile(raw, flags)
            for row, text in enumerate(self.values[attr]):
                if text.isascii():
                    continue
                count, first = 0, -1
                for match in compiled.finditer(text):
                    if count == 0:
                        first = match.start()
                    count += 1
                result[:, row] = count, first
                self.stats['shared_literal_nonascii_regex_rows'] += 1
        self.stats['shared_literal_hits'] += 1
        return result

    def audit(self):
        return {'configuration': self.config, 'accounted_bytes': self.bytes,
                'built_field_case_indices': len(self.indices),
                'blocked_field_case_indices': len(self.blocked),
                'observed_original_scan_work': [dict(attr=k[0], ignorecase=k[1], misses=v[0],
                    ascii_candidate_characters=v[1]) for k, v in self.work.items()]}
