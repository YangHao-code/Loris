"""Exact reusable lexical statistics, independent of rules, labels and models.

A regex's count and first match position are scanned once per document field.
Match/freq/cooccur/before and every subsequent frequency threshold are derived
from these statistics. Bindings include actual text, observed-field masks and
implementation hashes, but no Gamma, Sigma, labels, BO score or subset mode.
"""
from collections import Counter
import inspect
from pathlib import Path
import sqlite3
import time

import numpy as np

from loris.document import Document, VALID_ATTRS
from loris.predicates import MatchPredicate, FreqPredicate, CooccurPredicate, BeforePredicate, predicate_to_dict
from loris._engine.array_cache import PersistentArrayCache, canonical, sha, file_sha, _readonly
from loris._engine.shared_literal_stats import SharedLiteralStats, configuration as shared_configuration
from loris._engine.discovery_io import configure_sqlite


SCHEMA = 'paper36_exact_regex_count_first_position_cache_v1'


class TextPrimitiveCache:
    def __init__(self, docs, cache_dir, attribute_presence=None, memory_bytes=256 << 20, disk_bytes=4 << 30,
                 *, shared_literal_policy=None, shared_literal_bytes=None):
        docs = tuple(docs)
        self.values = {attr: tuple(doc.get_attr(attr) for doc in docs) for attr in sorted(VALID_ATTRS)}
        if any(not isinstance(text, str) for values in self.values.values() for text in values):
            raise ValueError('lexical primitive fields must be explicit strings')
        self.n = len(docs)
        self.presence = {}
        supplied = attribute_presence or {}
        for attr, values in self.values.items():
            observed = np.asarray(supplied.get(attr, [v != '' for v in values]))
            if observed.shape != (self.n,) or not np.isin(observed, [0, 1]).all():
                raise ValueError('exact text-field presence must match complete row axis')
            self.presence[attr] = _readonly(observed.astype(bool))
        shared_config = shared_configuration(shared_literal_policy, shared_literal_bytes)
        self.binding = {'schema': SCHEMA, 'rows': self.n,
            'attribute_texts': {attr: sha(values) for attr, values in self.values.items()},
            'attribute_presence': {attr: sha(mask.tolist()) for attr, mask in self.presence.items()},
            'implementation': {str(Path(path).resolve()): file_sha(path) for path in
                [__file__, inspect.getfile(predicate_to_dict), inspect.getfile(PersistentArrayCache),
                 inspect.getfile(SharedLiteralStats)]},
            'shared_literal': shared_config,
            'semantic_text_matching': False, 'label_model_rule_inputs': False}
        self.context_id = sha(self.binding)
        directory = Path(cache_dir) / self.context_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'binding.json').write_text(canonical(self.binding))
        self.db = sqlite3.connect(str(directory / 'lexical.sqlite'), timeout=60)
        configure_sqlite(self.db)
        self.db.execute('CREATE TABLE IF NOT EXISTS arrays(namespace TEXT,key TEXT,payload BLOB,sha256 TEXT,bytes INTEGER,created INTEGER,PRIMARY KEY(namespace,key))')
        self.regex = PersistentArrayCache(self.db, 'regex_count_first', memory_bytes=memory_bytes, disk_bytes=disk_bytes)
        self.masks = PersistentArrayCache(self.db, 'predicate_masks', memory_bytes=memory_bytes, disk_bytes=disk_bytes)
        self.stats = Counter()
        self.shared_literals = SharedLiteralStats(self.values, config=shared_config, stats=self.stats)
        self.compute_seconds = 0.

    def regex_stats(self, attr, pattern):
        if attr not in self.values:
            raise ValueError('lexical primitives only support declared cnt/ttl/mtd text fields')
        key = (attr, pattern.raw, int(pattern.flags))
        result = self.regex.get(key)
        if result is not None:
            return result
        started, memo = time.perf_counter(), {}
        values = self.shared_literals.maybe_counts(attr, pattern.raw, int(pattern.flags))
        if values is None:
            self.stats['regex_scans'] += 1
            values = np.empty((2, self.n), np.int64)
            compiled = pattern._compiled()
            for i, text in enumerate(self.values[attr]):
                item = memo.get(text)
                if item is None:
                    count, first = 0, -1
                    for match in compiled.finditer(text):
                        if count == 0:
                            first = match.start()
                        count += 1
                    item = (count, first)
                    memo[text] = item
                values[:, i] = item
        result = self.regex.put(key, values)
        self.stats['regex_statistics_computed'] += 1
        self.stats['distinct_text_scans'] += len(memo)
        self.compute_seconds += time.perf_counter() - started
        return result

    def mask(self, predicate):
        if type(predicate) not in (MatchPredicate, FreqPredicate, CooccurPredicate, BeforePredicate) or predicate.sim:
            raise TypeError('cache only supports exact paper lexical primitives')
        key = predicate_to_dict(predicate)
        result = self.masks.get(key)
        if result is not None:
            return result
        if isinstance(predicate, (MatchPredicate, FreqPredicate)):
            count = self.regex_stats(predicate.attr, predicate.r)[0]
            if isinstance(predicate, MatchPredicate):
                result = count > 0
                if predicate.negate:
                    result = ~result
            else:
                operations = {'==': np.equal, '!=': np.not_equal, '<': np.less,
                    '<=': np.less_equal, '>': np.greater, '>=': np.greater_equal}
                result = operations[predicate.op](count, predicate.eta)
        else:
            first = self.regex_stats(predicate.attr, predicate.r1)
            second = self.regex_stats(predicate.attr, predicate.r2)
            result = (first[0] > 0) & (second[0] > 0)
            if isinstance(predicate, BeforePredicate):
                result &= first[1] < second[1]
        result = self.masks.put(key, result & self.presence[predicate.attr])
        self.stats['derived_masks'] += 1
        return result

    def audit(self):
        return {'context_id': self.context_id, 'rows': self.n, 'computations': dict(self.stats),
                'actual_regex_seconds': self.compute_seconds, 'regex_cache': dict(self.regex.stats),
                'mask_cache': dict(self.masks.stats), 'shared_literal': self.shared_literals.audit()}

    def close(self):
        self.db.close()
