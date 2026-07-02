"""Document data structure for the LORIS framework.

Each document is a text variable ``x`` with four attributes as defined in
the paper:

    x.lbl  — (possibly soft) label set from label universe Λ
    x.mtd  — auxiliary metadata (e.g. source, timestamp)
    x.ttl  — short title or heading string
    x.cnt  — primary textual content

A *dataset* is a finite collection of :class:`Document` objects.

Migrated verbatim from ``pattern_extraction/document.py`` (Phase 1). The
legacy module re-exports these names as a shim during migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set

#: Valid document attribute names used by textual predicates.
VALID_ATTRS: FrozenSet[str] = frozenset({"mtd", "ttl", "cnt"})


@dataclass
class Document:
    """Text variable ``x`` with attributes ``(x.lbl, x.mtd, x.ttl, x.cnt)``.

    This is the central data unit for all predicate evaluation and rule
    application in the LORIS pipeline. Labels (``x.lbl``) are intentionally
    mutable — rule inference can add or remove soft labels at runtime.

    Parameters
    ----------
    cnt : str
        Primary textual content of the document (required).
    lbl : Set[str]
        Possibly soft label set from the label universe Λ. Hard labels are
        the special case where ``|lbl| == 1``. Defaults to an empty set.
    mtd : str
        Auxiliary metadata string, e.g. ``"source=Reuters date=2024-01-15"``.
    ttl : str
        Short title or heading, e.g. a news headline or log category.

    Examples
    --------
    >>> doc = Document(cnt="The bank reported record profits.", ttl="Finance")
    >>> doc.lbl.add("finance")
    >>> doc.get_attr("ttl")
    'Finance'
    """

    cnt: str
    lbl: Set[str] = field(default_factory=set)
    mtd: str = ""
    ttl: str = ""

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def get_attr(self, attr: str) -> str:
        """Return the text value of document attribute *attr*.

        Parameters
        ----------
        attr : str
            One of ``"mtd"``, ``"ttl"``, or ``"cnt"``.

        Raises
        ------
        ValueError
            If *attr* is not a valid document attribute name.
        """
        if attr not in VALID_ATTRS:
            raise ValueError(
                f"Invalid attribute {attr!r}. Must be one of {sorted(VALID_ATTRS)}."
            )
        return getattr(self, attr)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict) -> "Document":
        """Build a :class:`Document` from a plain dict.

        Expected keys: ``"cnt"`` (required), ``"lbl"``, ``"mtd"``, ``"ttl"``
        (all optional). ``"lbl"`` may be a list or set of strings.
        """
        return cls(
            cnt=str(d.get("cnt", "")),
            lbl=set(d.get("lbl", [])),
            mtd=str(d.get("mtd", "")),
            ttl=str(d.get("ttl", "")),
        )

    def to_dict(self) -> Dict:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "cnt": self.cnt,
            "lbl": sorted(self.lbl),
            "mtd": self.mtd,
            "ttl": self.ttl,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        preview = self.cnt[:60] + ("…" if len(self.cnt) > 60 else "")
        return f"Document(ttl={self.ttl!r}, lbl={self.lbl!r}, cnt={preview!r})"
