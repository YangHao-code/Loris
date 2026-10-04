"""Exact attribute groups stored in O(rows + groups) space.

The old mapping stored one full-axis integer bitmap per distinct value. Sparse
high-row groups then used quadratic space. This immutable mapping retains the
same insertion order and ascending member rows, and constructs a bitmap only
for a requested value; no bitmap is retained in the index.
"""
from collections.abc import Mapping

import numpy as np


SCHEMA = 'stable_compact_attribute_rows_lazy_exact_bitmap_v1'


def row_bits(rows):
    """Exact bitmap for ascending distinct nonnegative integer rows."""
    if len(rows) <= 8:
        value = 0
        for row in rows:
            value |= 1 << int(row)
        return value
    raw = np.zeros((int(rows[-1]) >> 3) + 1, dtype=np.uint8)
    np.bitwise_or.at(raw, rows >> 3, (1 << (rows & 7)).astype(np.uint8))
    return int.from_bytes(raw.tobytes(), 'little')


class CompactAttributeIndex(Mapping):
    """Read-only code -> row bitmap compatibility view over compact row groups."""
    __slots__ = ('codes', 'rows', 'offsets', '_order')

    def __init__(self, column):
        column = np.asarray(column)
        dtype = np.int32 if len(column) <= np.iinfo(np.int32).max else np.int64
        present = np.flatnonzero(column >= 0).astype(dtype, copy=False)
        order = np.argsort(column[present], kind='stable')
        self.rows = present[order]
        self.codes, starts = np.unique(column[self.rows], return_index=True)
        self.offsets = np.append(starts, len(present)).astype(dtype, copy=False)
        # A right attribute may encounter codes in a different order from the
        # left attribute. Match dict insertion order, not numeric code order.
        first_rows = self.rows[self.offsets[:-1]]
        self._order = np.argsort(first_rows, kind='stable').astype(dtype, copy=False)
        for array in (self.codes, self.rows, self.offsets, self._order):
            array.flags.writeable = False

    def __len__(self):
        return len(self.codes)

    def __iter__(self):
        for position in self._order:
            yield int(self.codes[position])

    def __contains__(self, code):
        if not isinstance(code, (int, np.integer)):
            return False
        position = int(np.searchsorted(self.codes, code))
        return position < len(self.codes) and int(self.codes[position]) == int(code)

    def __getitem__(self, code):
        if not isinstance(code, (int, np.integer)):
            raise KeyError(code)
        position = int(np.searchsorted(self.codes, code))
        if position >= len(self.codes) or int(self.codes[position]) != int(code):
            raise KeyError(code)
        return row_bits(self.group_rows(position))

    def group_rows(self, position):
        return self.rows[self.offsets[position]:self.offsets[position + 1]]

    def matching_groups(self, mask):
        """Which groups contain a row in an immutable per-rule domain snapshot."""
        if not len(self.codes):
            return np.zeros(0, dtype=bool)
        return np.logical_or.reduceat(mask[self.rows], self.offsets[:-1])

    @property
    def storage_bytes(self):
        return sum(x.nbytes for x in (self.codes, self.rows, self.offsets, self._order))


def shared_group_positions(left, right):
    """Shared groups in the exact insertion order of the left mapping."""
    _, lpos, rpos = np.intersect1d(left.codes, right.codes, assume_unique=True,
                                  return_indices=True)
    order = np.argsort(left.rows[left.offsets[lpos]], kind='stable')
    lpos, rpos = lpos[order], rpos[order]
    lpos.flags.writeable = rpos.flags.writeable = False
    return lpos, rpos
