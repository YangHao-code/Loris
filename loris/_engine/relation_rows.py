"""Exact fixed-axis relation rows with implicit empty/reflexive defaults.

The Chase still reads and updates integer bitsets. Only their resident storage
changes: sparse rows keep sorted column indices, dense rows retain integers.
Snapshots never materialize every diagonal bitset. Truly dense relations can
still require quadratic storage; this is not a bound on a dense closure.
"""

from collections.abc import Sequence
import hashlib
import operator
from types import MappingProxyType


RELATION_SCHEMA = "exact_complete_relation_adaptive_rows_v2"


def _indices(value):
    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low


def _count(value):
    return value.bit_count() if hasattr(value, "bit_count") else bin(value).count("1")


def _sparse_wins(count, width):
    # Conservative CPython storage estimate: tuple slots plus column integers
    # versus the packed integer digits. This choice affects storage only.
    return 40 + 36 * count < 24 + 4 * ((width + 29) // 30)


def _encode(value):
    count = _count(value)
    return tuple(_indices(value)) if _sparse_wins(count, value.bit_length()) else value


def _encode_columns(columns):
    columns = tuple(columns)
    if not columns:
        return 0
    if _sparse_wins(len(columns), columns[-1] + 1):
        return columns
    result = 0
    for column in columns:
        result |= 1 << column
    return result


class RelationRows(Sequence):
    """Mutable bitset-compatible rows; ``copy(frozen=True)`` is immutable."""

    __slots__ = ("n", "reflexive", "_data", "_frozen")

    def __init__(self, n, *, reflexive=False):
        n = operator.index(n)
        if n < 0 or type(reflexive) is not bool:
            raise ValueError("nonnegative axis and explicit reflexive Boolean required")
        self.n = n
        self.reflexive = reflexive
        self._data = {}
        self._frozen = False

    def __setattr__(self, key, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("relation row snapshot is immutable")
        object.__setattr__(self, key, value)

    @classmethod
    def from_rows(cls, rows, *, frozen=False):
        if isinstance(rows, cls):
            return rows.copy(frozen=frozen)
        # Stream unknown-length iterables into adaptive entries. Never first
        # collect the large integer row values in a list or tuple.
        data = {}
        n = 0
        for n, value in enumerate(rows, 1):
            value = operator.index(value)
            if value < 0:
                raise ValueError("relation rows must be nonnegative")
            if value:
                data[n - 1] = _encode(value)
        result = cls(n)
        for value in data.values():
            if (value[-1] + 1 if isinstance(value, tuple) else value.bit_length()) > n:
                raise ValueError("relation rows must lie on their exact N axis")
        result._data = data
        return result.copy(frozen=True) if frozen else result

    def copy(self, *, frozen=False):
        result = type(self)(self.n, reflexive=self.reflexive)
        data = dict(self._data)
        result._data = MappingProxyType(data) if frozen else data
        result._frozen = frozen
        return result

    def __copy__(self):
        return self.copy(frozen=self._frozen)

    def __deepcopy__(self, memo):
        # Encoded rows are immutable integers/tuples. Only the container can
        # change, so an independent dictionary is a complete deep snapshot.
        result = self.copy(frozen=self._frozen)
        memo[id(self)] = result
        return result

    def __len__(self):
        return self.n

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += self.n
        if not 0 <= index < self.n:
            raise IndexError("relation row outside its axis")
        return index

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(self.n)))
        index = self._index(index)
        value = self._data.get(index)
        if value is None:
            return (1 << index) if self.reflexive else 0
        if isinstance(value, tuple):
            result = 0
            for column in value:
                result |= 1 << column
            return result
        return value

    def __setitem__(self, index, value):
        if self._frozen:
            raise TypeError("relation row snapshot is immutable")
        index = self._index(index)
        value = operator.index(value)
        if value < 0 or value.bit_length() > self.n:
            raise ValueError("relation rows must lie on their exact N axis")
        if value == ((1 << index) if self.reflexive else 0):
            self._data.pop(index, None)
        else:
            self._data[index] = _encode(value)

    def set_many(self, rows, value):
        """Assign identical exact rows, sharing their immutable encoding.

        Equality groups often establish the same row for hundreds of nodes.
        Encoding that row once avoids repeated wide-integer sparse scans and
        also lets snapshots share a single immutable tuple for the group.
        """
        if self._frozen:
            raise TypeError("relation row snapshot is immutable")
        value = operator.index(value)
        if value < 0 or value.bit_length() > self.n:
            raise ValueError("relation rows must lie on their exact N axis")
        rows = tuple(self._index(row) for row in rows)
        encoded = _encode(value)
        for row in rows:
            if value == ((1 << row) if self.reflexive else 0):
                self._data.pop(row, None)
            else:
                self._data[row] = encoded

    def union_rows(self, rows):
        """Union rows while decoding a shared immutable encoding only once."""
        result, seen = 0, set()
        for row in rows:
            row = self._index(row)
            value = self._data.get(row)
            if value is None:
                if self.reflexive:
                    result |= 1 << row
                continue
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(value, tuple):
                for column in value:
                    result |= 1 << column
            else:
                result |= value
        return result

    def __eq__(self, other):
        if isinstance(other, RelationRows):
            if self.n != other.n:
                return False
            if self.reflexive == other.reflexive:
                return self._data == other._data
        if not isinstance(other, Sequence) or len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self, other))

    def columns(self, index):
        """Stream a row's exact sorted columns without allocating a bitset."""
        index = self._index(index)
        value = self._data.get(index)
        if value is None:
            if self.reflexive:
                yield index
        elif isinstance(value, tuple):
            yield from value
        else:
            yield from _indices(value)

    def edges(self):
        for source in range(self.n):
            for target in self.columns(source):
                yield source, target

    def nonidentity_sources(self):
        """Ascending rows with a non-reflexive edge, omitting default rows."""
        for row in sorted(self._data):
            value = self._data[row]
            if isinstance(value, tuple):
                nontrivial = any(column != row for column in value)
            else:
                nontrivial = bool(value & ~(1 << row))
            if nontrivial:
                yield row

    def count(self):
        result = self.n if self.reflexive else 0
        for value in self._data.values():
            result += (len(value) if isinstance(value, tuple) else _count(value)) - int(self.reflexive)
        return result

    def transpose(self):
        result = type(self)(self.n, reflexive=self.reflexive)
        changed = {}
        for source in self._data:
            columns = tuple(self.columns(source))
            if self.reflexive and source not in columns:
                changed.setdefault(source, {source}).discard(source)
            for target in columns:
                if self.reflexive and source == target:
                    continue
                changed.setdefault(target, {target} if self.reflexive else set()).add(source)
        result._data = {row: _encode_columns(sorted(columns)) for row, columns in changed.items()}
        return result

    def sha256(self):
        """Canonical exact rows; no N-wide zero padding for sparse rows.

        I/Z identify a singleton diagonal/empty row; other rows have a stable
        adaptive encoding independent of the container's default mode. Byte
        lengths and row/column counts prevent concatenation ambiguities.
        """
        digest = hashlib.sha256((RELATION_SCHEMA + ":" + str(self.n) + ":").encode())
        pending = bytearray()
        for row in range(self.n):
            value = self._data.get(row)
            if value is None:
                pending.extend(b"I" if self.reflexive else b"Z")
            elif value == 0:
                pending.extend(b"Z")
            elif (isinstance(value, tuple) and value == (row,)) or (
                isinstance(value, int) and value == (1 << row)
            ):
                pending.extend(b"I")
            else:
                digest.update(pending)
                pending.clear()
                if isinstance(value, tuple):
                    digest.update(b"S" + len(value).to_bytes(8, "little"))
                    for column in value:
                        digest.update(column.to_bytes(8, "little"))
                else:
                    size = (value.bit_length() + 7) // 8
                    digest.update(b"D" + size.to_bytes(8, "little"))
                    digest.update(value.to_bytes(size, "little"))
            if len(pending) >= 8192:
                digest.update(pending)
                pending.clear()
        digest.update(pending)
        return digest.hexdigest()

    def to_dict(self):
        return dict(n=self.n, default_diagonal=self.reflexive,
                    overrides=list(self.iter_overrides()))

    def iter_overrides(self):
        for row, value in sorted(self._data.items()):
            yield ([row, "columns", list(value)] if isinstance(value, tuple)
                   else [row, "bits_hex", hex(value)])

    @classmethod
    def from_dict(cls, value, *, frozen=False):
        if not isinstance(value, dict) or set(value) != {"n", "default_diagonal", "overrides"}:
            raise ValueError("complete adaptive row encoding required")
        result = cls(value["n"], reflexive=value["default_diagonal"])
        seen = set()
        for row, encoding, data in value["overrides"]:
            row = operator.index(row)
            if not 0 <= row < result.n or row in seen:
                raise ValueError("unique in-axis override rows required")
            seen.add(row)
            if encoding == "bits_hex":
                result[row] = int(data, 16)
            elif encoding == "columns":
                columns = tuple(operator.index(column) for column in data)
                if (columns != tuple(sorted(set(columns))) or
                        any(column < 0 or column >= result.n for column in columns)):
                    raise ValueError("sorted unique in-axis columns required")
                if columns != ((row,) if result.reflexive else ()):
                    result._data[row] = _encode_columns(columns)
            else:
                raise ValueError("unknown adaptive row encoding")
        return result.copy(frozen=True) if frozen else result

    def storage_counts(self):
        return dict(rows=self.n, default_diagonal=self.reflexive,
                    stored_rows=len(self._data),
                    sparse_rows=sum(isinstance(v, tuple) for v in self._data.values()),
                    dense_rows=sum(isinstance(v, int) for v in self._data.values()))
