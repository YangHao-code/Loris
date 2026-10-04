"""Certificate-only acceleration of PDF36 finite label comparison feasibility.

An accepted Boolean completion is a mathematical witness, never a new observed
label. Failure to find a cheap witness does NOT mean infeasibility: the original
unrestricted MILP remains the complete fallback. No constraints, rows, labels,
solver limits, scores or Chase scheduling are changed by this helper.

The optional initial completion can describe a whole frozen static program. It
is still checked against the actual state before every successful decision.
Unchanged witness/constraint pairs need not be rechecked: immutable snapshots of
previously checked relations certify them, and every newly present pair is
checked. Membership bounds are fully checked on every call.
"""
from collections import Counter

import numpy as np

from loris._engine.rdl_reference import comparison_feasible as _reference_feasible
from loris._engine.relation_rows import RelationRows


def _indices(value):
    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low


def _relation_rows(edges, n):
    rows = RelationRows(n)
    for source, target in edges:
        if not 0 <= source < n or not 0 <= target < n:
            raise ValueError('comparison edge outside complete document axis')
        rows[int(source)] |= 1 << int(target)
    return rows.copy(frozen=True)


def _edges(rows):
    return {(source, target) for source, row in enumerate(rows)
            for target in _indices(row)}


class _Witness:
    """Private integer completion plus reusable exact relation certificates."""

    def __init__(self, completion):
        self.completion = np.array(completion, dtype=bool, copy=True)
        self.completion.setflags(write=False)
        self.n, self.k = self.completion.shape
        self.all_rows = (1 << self.n) - 1
        self.column_bits = tuple(int.from_bytes(column.tobytes(), 'little')
            for column in np.packbits(self.completion.T, axis=1, bitorder='little'))
        self.row_bits = tuple(int.from_bytes(row.tobytes(), 'little')
            for row in np.packbits(self.completion, axis=1, bitorder='little'))
        equal = {}
        for source, labels in enumerate(self.row_bits):
            equal[labels] = equal.get(labels, 0) | (1 << source)
        self.allowed = {}
        self.strict_allowed = {}
        for labels, equal_rows in equal.items():
            targets = self.all_rows
            for label in _indices(labels):
                targets &= self.column_bits[label]
            self.allowed[labels] = targets
            self.strict_allowed[labels] = targets & ~equal_rows
        self.checked_inclusion = RelationRows(self.n).copy(frozen=True)
        self.checked_proper = RelationRows(self.n).copy(frozen=True)

    def accepts(self, pos, neg, inclusion_rows, proper_rows, stats):
        if self.completion.shape != pos.shape:
            return False
        # Bounds are not monotonicity-assumed: label mutations are checked too.
        if np.any(pos & ~self.completion) or np.any(neg & self.completion):
            return False
        for source, (included, strict, old_included, old_strict) in enumerate(zip(
                inclusion_rows, proper_rows, self.checked_inclusion, self.checked_proper)):
            new_included = included & ~old_included
            new_strict = strict & ~old_strict
            if not new_included and not new_strict:
                continue
            labels = self.row_bits[source]
            stats['changed_relation_rows_checked'] += 1
            if new_included & ~self.allowed[labels]:
                return False
            # Strict pairs must satisfy inclusion even when callers did not
            # explicitly duplicate them into the non-strict relation.
            if new_strict & ~self.strict_allowed[labels]:
                return False
        # Commit only a complete successful validation, never a partial scan.
        self.checked_inclusion = inclusion_rows
        self.checked_proper = proper_rows
        return True


class ExactFiniteComparisonOracle:
    """Exactly preserve reference feasibility; cheaply certify common states.

    ``initial_completion`` must be a complete binary [N,K] matrix. It is copied
    into private storage and has no connection to Chase factual arrays. The
    callable fallback must implement the original full feasibility decision.
    """

    def __init__(self, initial_completion=None, fallback=None):
        self._witness = None
        self._initial = None
        if initial_completion is not None:
            initial = np.asarray(initial_completion)
            if initial.ndim != 2 or not np.isin(initial, [0, 1]).all():
                raise ValueError('initial completion must be a complete binary matrix')
            self._initial = np.array(initial, dtype=bool, copy=True)
        self._fallback = fallback or _reference_feasible
        self.stats = Counter()

    @staticmethod
    def _bounds(pos, neg):
        pos, neg = np.asarray(pos), np.asarray(neg)
        if (pos.ndim != 2 or neg.shape != pos.shape or
                (pos.dtype != np.bool_ and not np.isin(pos, [0, 1]).all()) or
                (neg.dtype != np.bool_ and not np.isin(neg, [0, 1]).all())):
            raise ValueError('complete matching binary positive/negative axes required')
        return np.asarray(pos, bool), np.asarray(neg, bool)

    def feasible(self, pos, neg, inclusion, proper):
        pos, neg = self._bounds(pos, neg)
        return self.feasible_rows(pos, neg, _relation_rows(inclusion, len(pos)),
                                  _relation_rows(proper, len(pos)), _bounds_checked=True)

    def _normalize(self, pos, neg, inclusion_rows, proper_rows, _bounds_checked=False):
        if not _bounds_checked:
            pos, neg = self._bounds(pos, neg)
        n = len(pos)
        inclusion_rows = RelationRows.from_rows(inclusion_rows, frozen=True)
        proper_rows = RelationRows.from_rows(proper_rows, frozen=True)
        if len(inclusion_rows) != n or len(proper_rows) != n:
            raise ValueError('complete comparison bitsets must use the exact document axis')
        return pos, neg, inclusion_rows, proper_rows

    def feasible_rows(self, pos, neg, inclusion_rows, proper_rows, _bounds_checked=False):
        """Check complete integer-bitset rows without materializing N² tuples."""
        pos, neg, inclusion_rows, proper_rows = self._normalize(
            pos, neg, inclusion_rows, proper_rows, _bounds_checked)
        self.stats['calls'] += 1
        # Match the original function's no-proper short circuit; callers check
        # label conflicts independently, just as in the original _State.
        if not any(proper_rows):
            self.stats['no_proper'] += 1
            return True
        if self._try_certify_rows(pos, neg, inclusion_rows, proper_rows):
            return True
        self.stats['complete_milp_fallback'] += 1
        return bool(self._fallback(pos, neg, _edges(inclusion_rows), _edges(proper_rows)))

    def try_certify_rows(self, pos, neg, inclusion_rows, proper_rows):
        """Return True only for a valid complete witness; False means UNKNOWN.

        This method never calls MILP and never claims infeasibility. A runtime
        can place it before its original ordered conflict diagnostics, retaining
        that entire old path on False, including its precise conflict evidence.
        """
        normalized = self._normalize(pos, neg, inclusion_rows, proper_rows)
        self.stats['certificate_only_calls'] += 1
        return self._try_certify_rows(*normalized)

    def _try_certify_rows(self, pos, neg, inclusion_rows, proper_rows):
        if self._witness is not None and self._witness.accepts(
                pos, neg, inclusion_rows, proper_rows, self.stats):
            self.stats['cached_witness'] += 1
            return True

        def try_completion(completion, kind):
            witness = _Witness(completion)
            if not witness.accepts(pos, neg, inclusion_rows, proper_rows, self.stats):
                return False
            self._witness = witness
            self.stats['certificate_' + kind] += 1
            return True

        if self._initial is not None:
            initial, self._initial = self._initial, None
            if initial.shape != pos.shape:
                raise ValueError('initial completion differs from actual document/label axes')
            if try_completion(initial, 'initial'):
                return True
        if try_completion(pos, 'positive_lower_bound'):
            return True
        # Mark every non-reflexive comparison target with all labels still
        # permitted by its explicit negatives. This certifies disjoint
        # source→target stars/bicliques, but is merely a candidate in general.
        targets = 0
        for source, (included, strict) in enumerate(zip(inclusion_rows, proper_rows)):
            targets |= (included | strict) & ~(1 << source)
        completion = pos.copy()
        target_indices = list(_indices(targets))
        if target_indices:
            completion[target_indices] = ~neg[target_indices]
        if try_completion(completion, 'incoming_upper_bound'):
            return True
        self.stats['certificate_unknown'] += 1
        return False


def comparison_feasible(pos, neg, inclusion, proper):
    """Stateless drop-in Boolean wrapper; stateful users can reuse the class."""
    return ExactFiniteComparisonOracle().feasible(pos, neg, inclusion, proper)
