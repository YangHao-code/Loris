"""Bounded exact reuse of already materialized immutable relation objects.

This caches the representation of completed facts, never a Chase state or a
logical inference. Misses use the current exact adaptive RelationFacts constructor.
"""
from collections import Counter

from loris._engine.typed_runtime import RelationFacts
from loris._engine.relation_rows import RelationRows


SCHEMA = 'paper36_exact_immutable_relation_materialization_cache_v1'


class ImmutableRelationCache:
    def __init__(self, entries=2):
        if type(entries) is not int or not 1 <= entries <= 2:
            raise ValueError('relation materialization cache requires one or two entries')
        self.entries = entries
        self._values = []
        self.stats = Counter()

    def materialize(self, rows):
        # Snapshot sparse/default storage without eagerly constructing all
        # diagonal integers. A caller may mutate its state after this returns.
        actual = RelationRows.from_rows(rows, frozen=True)
        self.stats['lookups'] += 1
        for index, old in enumerate(self._values):
            if len(actual) == old.n and actual == old.rows:
                self._values.append(self._values.pop(index))
                self.stats['exact_hits'] += 1
                return old
        # Complete exact identity uses the versioned adaptive-row encoding.
        value = RelationFacts(actual)
        self.stats['original_constructions'] += 1
        self._values.append(value)
        if len(self._values) > self.entries:
            self._values.pop(0)
            self.stats['evictions'] += 1
        return value

    def clear(self):
        self._values.clear()

    def audit(self):
        return {'schema': SCHEMA, 'maximum_entries': self.entries, 'entries': len(self._values),
                'statistics': dict(self.stats), 'complete_ordered_rows_compared': True,
                'mutable_state_or_proofs_cached': False, 'persistent_cache': False}
