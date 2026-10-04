"""Private complete-fact snapshots at the first-pass candidate boundary.

Snapshots are never Gamma, never final fixed points, and never persisted in the
original static cache. Immutable integer rows can be shared; every mutable
label array, row list and finite-comparison witness is private to a branch.
"""
from collections import Counter, OrderedDict
import copy
from dataclasses import dataclass
from types import MappingProxyType
from loris._engine.relation_rows import RelationRows


SCHEMA = 'paper36_fact_first_pass_prefix_snapshot_v1'
STATE_FIELDS = frozenset((
    'owner', 'n', 'k', 'pos', 'neg', 'label_bits', '_positive_row_bits', '_positive_closed',
    '_signed_overlap', '_strict_edge_count', 'out', 'inc', 'proper', 'proper_inc',
    'proof', 'base_proof', 'provenance', 'conflict', 'version', 'last_valid_version',
    'feasibility_checks', 'relation_version', 'stats', '_comparison_oracle',
    'gamma_bits', 'gamma_protection', 'blocked_bits', 'gamma_blocked'))
ROW_LISTS = ('_positive_row_bits', 'out', 'inc', 'proper', 'proper_inc')


def fork_state(state, *, readonly=False):
    if set(vars(state)) != STATE_FIELDS:
        raise ValueError('unreviewed complete-fact snapshot fields')
    if state.proof is not None or state.base_proof is not None or state.provenance:
        raise ValueError('first-pass fact snapshots cannot contain full proofs or provenance')
    branch = object.__new__(type(state))
    branch.__dict__.update(vars(state))
    for name in ('pos', 'neg'):
        value = getattr(state, name).copy()
        if readonly:
            value.setflags(write=False)
        setattr(branch, name, value)
    for name in ROW_LISTS:
        value = getattr(state, name)
        setattr(branch, name, value.copy(frozen=readonly) if isinstance(value, RelationRows)
                else tuple(value) if readonly else list(value))
    labels = {sign: tuple(rows) if readonly else list(rows)
              for sign, rows in state.label_bits.items()}
    branch.label_bits = MappingProxyType(labels) if readonly else labels
    branch.provenance = {}
    branch.blocked_bits = {sign: dict(rows) for sign, rows in state.blocked_bits.items()}
    branch.conflict = copy.deepcopy(state.conflict)
    branch.stats = state.stats.copy()
    branch._comparison_oracle = copy.deepcopy(state._comparison_oracle)
    return branch


@dataclass(frozen=True)
class FirstPassSnapshot:
    state: object
    status: str
    rounds: int
    changed: bool
    matching_counts: tuple
    full_program: bool

    def fork(self, owner):
        if self.state.owner is not owner:
            raise ValueError('prefix snapshot belongs to a different frozen runner')
        return fork_state(self.state)


class FirstPassCache:
    def __init__(self, entries=2):
        if type(entries) is not int or entries < 2:
            raise ValueError('first-pass cache requires at least two bounded entries')
        self.entries = entries
        self._values = OrderedDict()
        self.stats = Counter()

    def get(self, key):
        value = self._values.get(key)
        if value is None:
            self.stats['misses'] += 1
            return None
        self._values.move_to_end(key)
        self.stats['hits'] += 1
        if value.full_program:
            self.stats['completed_first_pass_reused_as_longer_prefix'] += 1
        return value

    def put(self, key, state, *, status, rounds, changed, matching_counts,
            full_program, protect=None):
        if status not in ('fixpoint', 'conflict') or rounds not in (0, 1):
            raise ValueError('snapshot must stop during the first pass or initial conflict')
        # If full program == prefix (empty rules), retain the same authenticated
        # cut. Never turn a final fixed-point state into a first-pass snapshot.
        if key in self._values:
            self._values.move_to_end(key)
            return
        self._values[key] = FirstPassSnapshot(
            fork_state(state, readonly=True), status, int(rounds), bool(changed),
            tuple(sorted(matching_counts.items())), bool(full_program))
        self.stats['snapshots_built'] += 1
        while len(self._values) > self.entries:
            victim = next(candidate for candidate in self._values if candidate != protect)
            del self._values[victim]
            self.stats['evictions'] += 1

    def clear(self):
        self._values.clear()

    def audit(self):
        return {'schema': SCHEMA, 'maximum_entries': self.entries,
                'entries': len(self._values), 'persistent_dynamic_cache': False,
                'statistics': dict(self.stats)}
