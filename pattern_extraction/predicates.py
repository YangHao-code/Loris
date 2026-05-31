"""
pattern_extraction/predicates.py
----------------------------------
Predicate hierarchy for the LORIS framework.

Implements the four textual predicates from the paper's RDL (Rule-based
Document Labeling) specification:

    match(x.A, r)           — MatchPredicate
    freq(x.A, r) ⊕ η        — FreqPredicate
    cooccur(x.A, r1, r2)    — CooccurPredicate
    before(x.A, r1, r2)     — BeforePredicate

where A ∈ {mtd, ttl, cnt} is a document attribute and r, r1, r2 are textual
patterns (plain-string keywords or regular expressions).

Extended predicates:

    MLPredicate(model_name, label)   — ML model-based classification
    LabelPredicate(label, op)        — label-set operations on x.lbl

Design notes
------------
* Every predicate is a **frozen dataclass** and is therefore hashable and
  usable as a dict key or set member — enabling deduplication in the
  pattern abstraction pipeline and in rule stores.
* Patterns (``r``, ``r1``, ``r2``) are represented by :class:`_Pattern`,
  which wraps a raw string and integer ``re`` flags.  The compiled
  ``re.Pattern`` is cached on the instance via a module-level WeakValueDictionary
  so that deserialization (which creates new ``_Pattern`` objects) still
  benefits from compiled-pattern caching without violating frozen-dataclass
  constraints.
* All predicates accept ``Union[str, re.Pattern]`` in their constructors and
  normalise to ``_Pattern`` via :func:`_as_pattern`.
* The ``__call__(doc)`` interface makes every predicate a callable boolean
  function over :class:`~pattern_extraction.document.Document` objects,
  suitable for direct use in rule bodies.
* Textual predicates support optional ``sim=True`` + ``threshold`` for
  semantic similarity soft-matching via sentence embeddings.
"""

from __future__ import annotations

import operator as _op
import re
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from pattern_extraction.document import Document, VALID_ATTRS


# ---------------------------------------------------------------------------
# Operator dispatch table for FreqPredicate
# ---------------------------------------------------------------------------

#: Maps comparison operator strings to stdlib ``operator`` callables.
_COMPARE_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "==": _op.eq,
    "!=": _op.ne,
    "<":  _op.lt,
    "<=": _op.le,
    ">":  _op.gt,
    ">=": _op.ge,
}

#: Valid operator strings (exposed for validation and serialization).
VALID_OPS: frozenset = frozenset(_COMPARE_OPS.keys())

# ---------------------------------------------------------------------------
# Flag name → re constant mapping (for config-file driven regex loading)
# ---------------------------------------------------------------------------

_FLAG_MAP: Dict[str, int] = {
    "IGNORECASE": re.IGNORECASE,
    "I":          re.IGNORECASE,
    "MULTILINE":  re.MULTILINE,
    "M":          re.MULTILINE,
    "DOTALL":     re.DOTALL,
    "S":          re.DOTALL,
}


def flags_from_names(names: List[str]) -> int:
    """
    Convert a list of flag name strings to a combined ``re`` flags int.

    Parameters
    ----------
    names : List[str]
        Flag names, e.g. ``["IGNORECASE", "MULTILINE"]``.

    Returns
    -------
    int
        Combined ``re`` flags value (OR of all named flags), or ``0``.

    Raises
    ------
    ValueError
        If any name is not a recognised ``re`` flag.
    """
    result = 0
    for name in names:
        key = name.upper()
        if key not in _FLAG_MAP:
            raise ValueError(
                f"Unknown re flag name {name!r}. "
                f"Valid names: {sorted(_FLAG_MAP.keys())}"
            )
        result |= _FLAG_MAP[key]
    return result


# ---------------------------------------------------------------------------
# _Pattern — internal pattern wrapper
# ---------------------------------------------------------------------------

# Module-level cache: (raw, flags) → compiled re.Pattern
# WeakValueDictionary lets the GC reclaim unused compiled patterns.
_PATTERN_CACHE: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


@dataclass(frozen=True)
class _Pattern:
    """
    Immutable wrapper around a textual pattern string with optional ``re`` flags.

    Provides ``.search()`` and ``.findall()`` that operate on compiled
    ``re.Pattern`` objects, with a module-level cache to avoid redundant
    compilation.

    Parameters
    ----------
    raw : str
        The raw pattern string (keyword or regular expression).
    flags : int
        Combined ``re`` flags integer.  Default ``0`` (no flags).

    Notes
    -----
    To perform **exact keyword matching** (no regex interpretation), pass
    ``re.escape(word)`` as *raw*, or wrap the term with word boundaries:
    ``rf"\\b{re.escape(word)}\\b"``.
    """

    raw: str
    flags: int = 0
    sim_text: str = ""  # plain-language anchor for embedding (used when sim=True)

    # ------------------------------------------------------------------
    # Compiled pattern property (not stored in dataclass — derived on demand)
    # ------------------------------------------------------------------

    def _compiled(self) -> re.Pattern[str]:
        """Return the compiled ``re.Pattern``, using the module cache."""
        key = (self.raw, self.flags)
        pat = _PATTERN_CACHE.get(key)
        if pat is None:
            pat = re.compile(self.raw, self.flags)
            # WeakValueDictionary requires a strong reference holder; store via
            # a local strong ref — the caller (search/findall) keeps it alive
            # for the duration of the call.  The dict may evict it later.
            try:
                _PATTERN_CACHE[key] = pat
            except TypeError:
                pass  # unhashable key (shouldn't happen, but be safe)
        return pat

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------

    def search(self, text: str) -> Optional[re.Match[str]]:
        """Return the first match of this pattern in *text*, or ``None``."""
        return self._compiled().search(text)

    def findall(self, text: str) -> List[str]:
        """Return all non-overlapping matches of this pattern in *text*."""
        return self._compiled().findall(text)

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:
        flag_str = f", flags={self.flags}" if self.flags else ""
        return f"_Pattern({self.raw!r}{flag_str})"


def _as_pattern(r: Union[str, re.Pattern[str], "_Pattern"]) -> _Pattern:
    """
    Normalise *r* to a :class:`_Pattern`.

    Accepted inputs:

    * :class:`_Pattern`        — returned as-is.
    * ``re.Pattern``           — ``pattern`` and ``flags`` are extracted.
    * ``str``                  — used as ``raw`` with ``flags=0``.
    """
    if isinstance(r, _Pattern):
        return r
    if isinstance(r, re.Pattern):
        return _Pattern(raw=r.pattern, flags=r.flags)
    return _Pattern(raw=str(r))


# ---------------------------------------------------------------------------
# Semantic similarity helpers (lazy-loaded)
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = None
_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}


def _get_embedding_model():
    """Lazily load the SentenceTransformer model for semantic similarity."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def _get_embedding(text: str) -> np.ndarray:
    """Compute (and cache) the embedding vector for *text*."""
    if text not in _EMBEDDING_CACHE:
        model = _get_embedding_model()
        _EMBEDDING_CACHE[text] = model.encode(text, convert_to_numpy=True)
    return _EMBEDDING_CACHE[text]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences by ``.!?`` followed by whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in parts if s]


# ---------------------------------------------------------------------------
# Predicate — top-level abstract base
# ---------------------------------------------------------------------------

class Predicate(ABC):
    """
    Top-level abstract base class for all predicates in the LORIS framework.

    Every predicate is callable: ``pred(doc) -> bool``.
    """

    @abstractmethod
    def __call__(self, doc: Document) -> bool:
        """Evaluate the predicate on *doc*. Return ``True`` if satisfied."""
        ...


# ---------------------------------------------------------------------------
# TextualPredicate — abstract base for attribute-targeting predicates
# ---------------------------------------------------------------------------

class TextualPredicate(Predicate):
    """
    Abstract base class for all textual predicates over document attributes.

    Every concrete predicate targets a specific document attribute
    ``A ∈ {mtd, ttl, cnt}`` and evaluates to a boolean when called with a
    :class:`~pattern_extraction.document.Document` instance.

    Subclasses are frozen dataclasses, making them hashable and usable as
    rule identifiers in downstream rule-discovery components.
    """

    #: Validated in ``__post_init__`` of every concrete subclass.
    attr: str

    def __post_init__(self) -> None:
        # Validate attr and normalise pattern fields in subclasses.
        if self.attr not in VALID_ATTRS:
            raise ValueError(
                f"attr must be one of {sorted(VALID_ATTRS)}, got {self.attr!r}."
            )

    @abstractmethod
    def __call__(self, doc: Document) -> bool:
        """Evaluate the predicate on *doc*. Return ``True`` if satisfied."""
        ...

    def attr_value(self, doc: Document) -> str:
        """Retrieve the targeted attribute string from *doc*."""
        return doc.get_attr(self.attr)

    # ------------------------------------------------------------------
    # String representation (shared format for all subclasses)
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# MatchPredicate — match(x.A, r)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchPredicate(TextualPredicate):
    """
    ``match(x.A, r)`` — tests whether pattern *r* appears in attribute *A*.

    Parameters
    ----------
    attr : str
        Target attribute: ``"mtd"``, ``"ttl"``, or ``"cnt"``.
    r : str, re.Pattern, or _Pattern
        Textual pattern (keyword or regular expression).
    sim : bool
        If True, use semantic similarity instead of regex matching.
    threshold : float
        Cosine similarity threshold (only used when sim=True).

    Examples
    --------
    >>> p = MatchPredicate("cnt", r"\\bbank\\b")
    >>> doc = Document(cnt="The bank reported profits.")
    >>> p(doc)
    True
    """

    attr: str
    r: _Pattern
    sim: bool = False
    threshold: float = 0.85

    def __post_init__(self) -> None:
        # Normalise r to _Pattern before frozen check fires
        object.__setattr__(self, "r", _as_pattern(self.r))
        super().__post_init__()

    def __call__(self, doc: Document) -> bool:
        text = self.attr_value(doc)
        if self.sim:
            emb_pat = _get_embedding(self.r.sim_text or self.r.raw)
            for sent in _split_sentences(text):
                if _cosine_similarity(emb_pat, _get_embedding(sent)) >= self.threshold:
                    return True
            return False
        return self.r.search(text) is not None

    def __repr__(self) -> str:
        sim_str = f", sim={self.threshold}" if self.sim else ""
        return f"match(x.{self.attr}, {self.r.raw!r}{sim_str})"


# ---------------------------------------------------------------------------
# FreqPredicate — freq(x.A, r) ⊕ η
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreqPredicate(TextualPredicate):
    """
    ``freq(x.A, r) ⊕ η`` — tests if the occurrence count of *r* in *A*
    satisfies the comparison ``count ⊕ η``.

    Parameters
    ----------
    attr : str
        Target attribute.
    r : str, re.Pattern, or _Pattern
        Pattern to count occurrences of.
    op : str
        Comparison operator: one of ``"=="``, ``"!="`` , ``"<"``,
        ``"<="`` , ``">"``, ``">="`` .
    eta : float
        Numeric threshold (right-hand side of the comparison).
    sim : bool
        If True, count sentences semantically similar to *r* instead of regex matches.
    threshold : float
        Cosine similarity threshold (only used when sim=True).

    Examples
    --------
    >>> p = FreqPredicate("cnt", r"\\bthe\\b", op=">=", eta=3)
    >>> doc = Document(cnt="The the the bank reported.")
    >>> p(doc)
    True
    """

    attr: str
    r: _Pattern
    op: str
    eta: float
    sim: bool = False
    threshold: float = 0.85

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", _as_pattern(self.r))
        # Normalise eta to float so that repr is stable after serialization
        # round-trips (JSON always decodes numbers as float/int uniformly).
        object.__setattr__(self, "eta", float(self.eta))
        if self.op not in VALID_OPS:
            raise ValueError(
                f"op must be one of {sorted(VALID_OPS)}, got {self.op!r}."
            )
        super().__post_init__()

    def __call__(self, doc: Document) -> bool:
        text = self.attr_value(doc)
        if self.sim:
            sentences = _split_sentences(text)
            emb_pat = _get_embedding(self.r.sim_text or self.r.raw)
            count = sum(
                1 for s in sentences
                if _cosine_similarity(emb_pat, _get_embedding(s)) >= self.threshold
            )
        else:
            count = len(self.r.findall(text))
        return _COMPARE_OPS[self.op](count, self.eta)

    def __repr__(self) -> str:
        sim_str = f", sim={self.threshold}" if self.sim else ""
        return f"freq(x.{self.attr}, {self.r.raw!r}{sim_str}) {self.op} {self.eta}"


# ---------------------------------------------------------------------------
# CooccurPredicate — cooccur(x.A, r1, r2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CooccurPredicate(TextualPredicate):
    """
    ``cooccur(x.A, r1, r2)`` — tests whether both *r1* and *r2* appear in
    attribute *A* (unordered).

    Parameters
    ----------
    attr : str
        Target attribute.
    r1 : str, re.Pattern, or _Pattern
        First pattern.
    r2 : str, re.Pattern, or _Pattern
        Second pattern.
    sim : bool
        If True, use semantic similarity for matching.
    threshold : float
        Cosine similarity threshold (only used when sim=True).

    Notes
    -----
    Co-occurrence is **unordered**: ``CooccurPredicate("cnt", a, b)`` and
    ``CooccurPredicate("cnt", b, a)`` are semantically equivalent but are
    **not** collapsed to the same hash (unlike the old ``CooccurPattern``).
    If canonical ordering is desired, normalise ``r1``/``r2`` before
    construction (e.g. sort their ``raw`` strings).

    Examples
    --------
    >>> p = CooccurPredicate("cnt", "bank", "market")
    >>> doc = Document(cnt="The bank and the market reacted.")
    >>> p(doc)
    True
    """

    attr: str
    r1: _Pattern
    r2: _Pattern
    sim: bool = False
    threshold: float = 0.85

    def __post_init__(self) -> None:
        object.__setattr__(self, "r1", _as_pattern(self.r1))
        object.__setattr__(self, "r2", _as_pattern(self.r2))
        super().__post_init__()

    def __call__(self, doc: Document) -> bool:
        text = self.attr_value(doc)
        if self.sim:
            sentences = _split_sentences(text)
            emb_r1 = _get_embedding(self.r1.sim_text or self.r1.raw)
            emb_r2 = _get_embedding(self.r2.sim_text or self.r2.raw)
            found_r1 = any(
                _cosine_similarity(emb_r1, _get_embedding(s)) >= self.threshold
                for s in sentences
            )
            found_r2 = any(
                _cosine_similarity(emb_r2, _get_embedding(s)) >= self.threshold
                for s in sentences
            )
            return found_r1 and found_r2
        return (
            self.r1.search(text) is not None
            and self.r2.search(text) is not None
        )

    def __repr__(self) -> str:
        sim_str = f", sim={self.threshold}" if self.sim else ""
        return f"cooccur(x.{self.attr}, {self.r1.raw!r}, {self.r2.raw!r}{sim_str})"


# ---------------------------------------------------------------------------
# BeforePredicate — before(x.A, r1, r2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeforePredicate(TextualPredicate):
    """
    ``before(x.A, r1, r2)`` — tests whether the first match of *r1* starts
    at a strictly lower character offset than the first match of *r2* in *A*.

    Parameters
    ----------
    attr : str
        Target attribute.
    r1 : str, re.Pattern, or _Pattern
        The pattern that must appear first.
    r2 : str, re.Pattern, or _Pattern
        The pattern that must appear after *r1*.
    sim : bool
        If True, use semantic similarity for position comparison.
    threshold : float
        Cosine similarity threshold (only used when sim=True).

    Notes
    -----
    Position is measured by the **start** character offset of the first
    (leftmost) match of each pattern, using ``re.search`` semantics.  This
    is consistent with the paper's definition and handles multi-character
    patterns correctly without whitespace tokenisation.

    Examples
    --------
    >>> p = BeforePredicate("cnt", "bank", "market")
    >>> doc = Document(cnt="The bank and the market reacted.")
    >>> p(doc)
    True
    >>> p2 = BeforePredicate("cnt", "market", "bank")
    >>> p2(doc)
    False
    """

    attr: str
    r1: _Pattern
    r2: _Pattern
    sim: bool = False
    threshold: float = 0.85

    def __post_init__(self) -> None:
        object.__setattr__(self, "r1", _as_pattern(self.r1))
        object.__setattr__(self, "r2", _as_pattern(self.r2))
        super().__post_init__()

    def __call__(self, doc: Document) -> bool:
        text = self.attr_value(doc)
        if self.sim:
            sentences = _split_sentences(text)
            emb_r1 = _get_embedding(self.r1.sim_text or self.r1.raw)
            emb_r2 = _get_embedding(self.r2.sim_text or self.r2.raw)
            pos_r1 = None
            pos_r2 = None
            for i, s in enumerate(sentences):
                emb_s = _get_embedding(s)
                if pos_r1 is None and _cosine_similarity(emb_r1, emb_s) >= self.threshold:
                    pos_r1 = i
                if pos_r2 is None and _cosine_similarity(emb_r2, emb_s) >= self.threshold:
                    pos_r2 = i
            return pos_r1 is not None and pos_r2 is not None and pos_r1 < pos_r2
        m1 = self.r1.search(text)
        m2 = self.r2.search(text)
        return m1 is not None and m2 is not None and m1.start() < m2.start()

    def __repr__(self) -> str:
        sim_str = f", sim={self.threshold}" if self.sim else ""
        return f"before(x.{self.attr}, {self.r1.raw!r}, {self.r2.raw!r}{sim_str})"


# ---------------------------------------------------------------------------
# MLPredicate — ML model-based classification
# ---------------------------------------------------------------------------

_ML_MODEL_CACHE: Dict[str, Any] = {}


def register_ml_model(name: str, model: Any) -> None:
    """
    Register a custom ML model for use with :class:`MLPredicate`.

    The model must have a ``predict(text) -> str`` method that returns
    a predicted label string.

    Parameters
    ----------
    name : str
        Model identifier (used in ``MLPredicate.model_name``).
    model : Any
        Model object with a ``predict(text) -> str`` method.
    """
    _ML_MODEL_CACHE[name] = model


def _get_ml_model(model_name: str) -> Any:
    """Load or retrieve a cached ML model."""
    if model_name in _ML_MODEL_CACHE:
        return _ML_MODEL_CACHE[model_name]
    try:
        from transformers import pipeline
        pipe = pipeline("text-classification", model=model_name)
        _ML_MODEL_CACHE[model_name] = pipe
        return pipe
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load ML model {model_name!r}. "
            f"Register it with register_ml_model() or install transformers. "
            f"Original error: {exc}"
        ) from exc


@dataclass(frozen=True)
class MLPredicate(Predicate):
    """
    ML model-based predicate: evaluates whether a document is classified
    as *label* by the specified ML model.

    Parameters
    ----------
    model_name : str
        Identifier for the ML model (registered via :func:`register_ml_model`
        or a Hugging Face model name).
    label : str
        Target class label that the model should predict for a positive match.
    """

    model_name: str
    label: str

    def __call__(self, doc: Document) -> bool:
        model = _get_ml_model(self.model_name)
        # Support two protocols: (1) object with predict(text)->str,
        # (2) transformers pipeline returning [{"label": ..., "score": ...}]
        if hasattr(model, "predict"):
            prediction = model.predict(doc.cnt)
            return prediction == self.label
        # transformers pipeline protocol
        result = model(doc.cnt)
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("label", "") == self.label
        return False

    def __repr__(self) -> str:
        return f"ml({self.model_name!r}, label={self.label!r})"

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class MLThresholdPredicate(Predicate):
    """
    Threshold-based ML predicate for multi-label classification.

    Unlike :class:`MLPredicate` which uses argmax (single-label), this
    predicate fires when the model's predicted probability for *label*
    exceeds *threshold*.  This enables coverage on rare labels that would
    never be the argmax prediction.
    """

    model_name: str
    label: str
    threshold: float = 0.5

    def __call__(self, doc: Document) -> bool:
        model = _get_ml_model(self.model_name)
        if hasattr(model, "predict_proba_single"):
            proba = model.predict_proba_single(doc.cnt)
            label_idx = model.label_index(self.label)
            return float(proba[label_idx]) >= self.threshold
        # fallback to argmax
        if hasattr(model, "predict"):
            return model.predict(doc.cnt) == self.label
        return False

    def __repr__(self) -> str:
        return (f"ml_thresh({self.model_name!r}, label={self.label!r}, "
                f"thresh={self.threshold:.2f})")

    def __str__(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# LabelPredicate — label-set operations on x.lbl
# ---------------------------------------------------------------------------

_VALID_LABEL_OPS = frozenset({"contains", "eq", "subset", "strict_subset", "minus"})


@dataclass(frozen=True)
class LabelPredicate(Predicate):
    """
    Label-set predicate: evaluates conditions on a document's label set.

    Parameters
    ----------
    label : str
        Target label (or comma-separated label set for set operations).
    op : str
        Operation type:
        - ``"contains"`` : ``label in doc.lbl``
        - ``"eq"``       : ``doc.lbl == label_set``
        - ``"subset"``   : ``doc.lbl <= label_set``
        - ``"strict_subset"`` : ``doc.lbl < label_set``
        - ``"minus"``    : remove label from ``doc.lbl``, return whether removed
    """

    label: str
    op: str = "contains"

    def __post_init__(self) -> None:
        if self.op not in _VALID_LABEL_OPS:
            raise ValueError(
                f"op must be one of {sorted(_VALID_LABEL_OPS)}, got {self.op!r}."
            )

    def _label_set(self) -> set:
        """Parse comma-separated label string into a set."""
        return {s.strip() for s in self.label.split(",") if s.strip()}

    def __call__(self, doc: Document) -> bool:
        if self.op == "contains":
            return self.label in doc.lbl
        target = self._label_set()
        if self.op == "eq":
            return doc.lbl == target
        if self.op == "subset":
            return doc.lbl <= target
        if self.op == "strict_subset":
            return len(doc.lbl) > 0 and doc.lbl < target
        if self.op == "minus":
            # Side-effect: remove label from doc.lbl
            removed = self.label in doc.lbl
            doc.lbl.discard(self.label)
            return removed
        return False

    def __repr__(self) -> str:
        return f"label({self.label!r}, op={self.op!r})"

    def __str__(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# SimPredicate — document similarity predicate for Chase propagation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimPredicate(Predicate):
    """
    Document similarity predicate: ``sim(x, y) > threshold``.

    A normal body predicate.  When a rule body contains a SimPredicate,
    the rule becomes *pairwise*: for every document *x*, iterate over
    neighbours *y* with ``cosine(x, y) > threshold``.

    - :class:`LabelPredicate` instances in the same body check **y**'s
      labels (the neighbour), not *x*'s.
    - Textual predicates still check **x**'s text.
    - The rule consequence applies to **x**.

    Example rules::

        sim(x, y, 0.85) ∧ τ ∈ y.lbl → add τ to x.lbl
        sim(x, y, 0.90) ∧ A ∈ y.lbl ∧ match(x.cnt, "kw") → add B to x.lbl

    ``__call__`` is a stub — actual evaluation uses SpMV on a precomputed
    sparse adjacency matrix:
    ``(sim_graph[θ] @ label_state[:, τ_idx]) > 0``.
    """

    threshold: float = 0.85

    def __call__(self, doc: Document) -> bool:
        return True  # stub — evaluated via sim_graph

    def __repr__(self) -> str:
        return f"sim(x, y, >{self.threshold:.2f})"

    def __str__(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# GroupPredicate — attribute-equality predicate for group-based propagation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupPredicate(Predicate):
    """
    Attribute-equality predicate: ``x.attr == y.attr`` (cross-document grouping).

    Paper formalism: ``doc(x) ∧ doc(y) ∧ x.attr == y.attr ∧ label(A∈y.lbl) → add A to x``

    When a rule body contains a GroupPredicate, the rule is *pairwise*:
    for document *x*, find all documents *y* sharing the same group
    (same value of ``attr_name``).  If any *y* has the required label
    (checked via LabelPredicate in the same body), the rule fires on *x*.

    Virtual attributes (cluster IDs, prediction patterns from intermediate
    models) are stored externally as numpy arrays indexed by document position.

    ``__call__`` is a stub — actual evaluation uses precomputed group
    membership arrays.
    """

    attr_name: str
    group_count: int = 0

    def __call__(self, doc: Document) -> bool:
        return True  # stub — evaluated via group membership arrays

    def __repr__(self) -> str:
        return f"x.{self.attr_name} == y.{self.attr_name}"

    def __str__(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def predicate_to_dict(p: Predicate) -> Dict[str, Any]:
    """
    Serialise a :class:`Predicate` to a JSON-safe plain dict.

    Parameters
    ----------
    p : Predicate

    Returns
    -------
    dict
    """
    base: Dict[str, Any] = {"type": type(p).__name__}

    def _pat_dict(pat: _Pattern) -> Dict[str, Any]:
        d: Dict[str, Any] = {"raw": pat.raw, "flags": pat.flags}
        if pat.sim_text:
            d["sim_text"] = pat.sim_text
        return d

    if isinstance(p, MatchPredicate):
        base["attr"] = p.attr
        base["r"] = _pat_dict(p.r)
        base["sim"] = p.sim
        base["threshold"] = p.threshold

    elif isinstance(p, FreqPredicate):
        base["attr"] = p.attr
        base["r"] = _pat_dict(p.r)
        base["op"] = p.op
        base["eta"] = p.eta
        base["sim"] = p.sim
        base["threshold"] = p.threshold

    elif isinstance(p, (CooccurPredicate, BeforePredicate)):
        base["attr"] = p.attr
        base["r1"] = _pat_dict(p.r1)
        base["r2"] = _pat_dict(p.r2)
        base["sim"] = p.sim
        base["threshold"] = p.threshold

    elif isinstance(p, MLThresholdPredicate):
        base["model_name"] = p.model_name
        base["label"] = p.label
        base["threshold"] = p.threshold

    elif isinstance(p, MLPredicate):
        base["model_name"] = p.model_name
        base["label"] = p.label

    elif isinstance(p, LabelPredicate):
        base["label"] = p.label
        base["op"] = p.op

    elif isinstance(p, SimPredicate):
        base["threshold"] = p.threshold

    elif isinstance(p, GroupPredicate):
        base["attr_name"] = p.attr_name
        base["group_count"] = p.group_count

    else:
        raise TypeError(f"Unknown predicate type: {type(p).__name__!r}")

    return base


def predicate_from_dict(d: Dict[str, Any]) -> Predicate:
    """
    Reconstruct a :class:`Predicate` from a plain dict produced by
    :func:`predicate_to_dict`.

    Parameters
    ----------
    d : dict

    Returns
    -------
    Predicate

    Raises
    ------
    ValueError
        If ``d["type"]`` is not a recognised predicate class name.
    """
    ptype = d["type"]

    if ptype == "MatchPredicate":
        return MatchPredicate(
            attr=d["attr"],
            r=_Pattern(**d["r"]),
            sim=d.get("sim", False),
            threshold=d.get("threshold", 0.85),
        )
    if ptype == "FreqPredicate":
        return FreqPredicate(
            attr=d["attr"],
            r=_Pattern(**d["r"]),
            op=d["op"],
            eta=float(d["eta"]),
            sim=d.get("sim", False),
            threshold=d.get("threshold", 0.85),
        )
    if ptype == "CooccurPredicate":
        return CooccurPredicate(
            attr=d["attr"],
            r1=_Pattern(**d["r1"]),
            r2=_Pattern(**d["r2"]),
            sim=d.get("sim", False),
            threshold=d.get("threshold", 0.85),
        )
    if ptype == "BeforePredicate":
        return BeforePredicate(
            attr=d["attr"],
            r1=_Pattern(**d["r1"]),
            r2=_Pattern(**d["r2"]),
            sim=d.get("sim", False),
            threshold=d.get("threshold", 0.85),
        )
    if ptype == "MLThresholdPredicate":
        return MLThresholdPredicate(
            model_name=d["model_name"],
            label=d["label"],
            threshold=float(d.get("threshold", 0.5)),
        )
    if ptype == "MLPredicate":
        return MLPredicate(
            model_name=d["model_name"],
            label=d["label"],
        )
    if ptype == "LabelPredicate":
        return LabelPredicate(
            label=d["label"],
            op=d.get("op", "contains"),
        )
    if ptype == "SimPredicate":
        return SimPredicate(
            threshold=float(d.get("threshold", 0.85)),
        )
    if ptype == "GroupPredicate":
        return GroupPredicate(
            attr_name=d["attr_name"],
            group_count=int(d.get("group_count", 0)),
        )
    raise ValueError(
        f"Unknown predicate type {ptype!r}. "
        "Expected one of: MatchPredicate, FreqPredicate, "
        "CooccurPredicate, BeforePredicate, MLPredicate, MLThresholdPredicate, "
        "LabelPredicate, SimPredicate, GroupPredicate."
    )
