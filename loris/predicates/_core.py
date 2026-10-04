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

from loris.document import Document, VALID_ATTRS


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
    """Lazily load the SentenceTransformer for per-predicate semantic similarity.

    Pinned to CPU on purpose: this model is invoked inside a predicate's
    ``__call__`` (per-document), which the rule discoverer fans out across
    ``Parallel(n_jobs=-1)`` worker *processes* (one per core). On a CUDA host
    each worker would otherwise load its own copy onto the GPU, creating N CUDA
    contexts + N model copies and exhausting GPU memory (CUDA OOM). The per-doc
    embedding is tiny, so CPU is both correct and faster here. Heavy batched
    embeddings (clustering, sim-graphs) and the ML model pool use their own
    GPU-capable paths and are unaffected.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from loris.models.embeddings import get_sentence_encoder
        _EMBEDDING_MODEL = get_sentence_encoder(device="cpu")
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
    negate: bool = False    # B5/Phase-2: True ⇒ fires when r is ABSENT (¬match)

    def __post_init__(self) -> None:
        # Normalise r to _Pattern before frozen check fires
        object.__setattr__(self, "r", _as_pattern(self.r))
        super().__post_init__()

    def _matches(self, doc: Document) -> bool:
        text = self.attr_value(doc)
        if self.sim:
            emb_pat = _get_embedding(self.r.sim_text or self.r.raw)
            for sent in _split_sentences(text):
                if _cosine_similarity(emb_pat, _get_embedding(sent)) >= self.threshold:
                    return True
            return False
        return self.r.search(text) is not None

    def __call__(self, doc: Document) -> bool:
        m = self._matches(doc)
        return (not m) if self.negate else m

    def __repr__(self) -> str:
        sim_str = f", sim={self.threshold}" if self.sim else ""
        neg = "¬" if self.negate else ""
        return f"{neg}match(x.{self.attr}, {self.r.raw!r}{sim_str})"


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
_ML_LABEL_AXES: Dict[str, Tuple[str, ...]] = {}

# Per-(model_name, document-text) prediction memo.
# MLThresholdPredicate.__call__ runs one model forward per doc; the staged Stage-3
# co-occurrence step (pipeline/steps.py:1984/1998) re-predicts the *full* rule set over
# thousands of docs, so an LLM-in-pool (e.g. Mistral) turns that into 1e4–1e5 redundant
# forwards and the run appears to hang. This memo caches each model's output per unique
# document text so repeated evals are O(1) — value-identical to recomputing.
# CRITICAL (staleness): the memo is a snapshot. It is invalidated for a model name
# whenever that name is (re)registered via register_ml_model — so a retrained model
# (iterative path) can never serve stale predictions. The standard (re)train path goes
# through register_ml_model; do NOT mutate a registered model in place without
# re-registering, or its memo will be stale.
_ML_PRED_MEMO: Dict[Tuple[str, ...], Any] = {}
_MEMO_MISS = object()  # sentinel: distinguishes "not cached" from a cached falsy value


def clear_ml_pred_memo(name: Optional[str] = None) -> None:
    """Drop memoized per-doc predictions — for *name* only, or all models if None."""
    if name is None:
        _ML_PRED_MEMO.clear()
    else:
        for _k in [k for k in _ML_PRED_MEMO if k[0] == name]:
            del _ML_PRED_MEMO[_k]


def register_ml_model(name: str, model: Any, *, labels=None) -> None:
    """Register a model and invalidate predictions from its previous version.

    Probability predicates accept ``predict_proba([text]) -> (1, K)`` or
    ``predict_proba_single(text) -> (K,)``. Supply ordered ``labels`` for their
    output axis, or implement ``label_index(label)`` on the model. Public
    document classifiers use the batch protocol, even for one document.

    The separate argmax ``MLPredicate`` retains its single-label
    ``predict(text) -> str`` protocol. It is not a threshold fallback.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("registered model name must be a non-empty string")
    axis = None
    if labels is not None:
        if isinstance(labels, (str, bytes, set, frozenset, dict)):
            raise ValueError("labels must be an ordered sequence of label names")
        axis = tuple(labels)
        if (not axis or any(not isinstance(label, str) or not label for label in axis)
                or len(set(axis)) != len(axis)):
            raise ValueError("labels must be non-empty, unique label names")
        if hasattr(model, "num_labels") and model.num_labels != len(axis):
            raise ValueError("registered label axis differs from model.num_labels")
    _ML_MODEL_CACHE[name] = model
    _ML_LABEL_AXES.pop(name, None)
    if axis is not None:
        _ML_LABEL_AXES[name] = axis
    # Invalidate any memoized predictions for this name: a (re)registered model may be
    # freshly retrained, so previously cached per-doc outputs are now stale.
    clear_ml_pred_memo(name)


def _ml_probability(model_name: str, text: str, label: str) -> float:
    """Read one validated probability on an explicit label axis; never argmax."""
    model = _get_ml_model(model_name)
    axis = _ML_LABEL_AXES.get(model_name)
    if axis is not None:
        if label not in axis:
            raise ValueError("label is not on the registered model axis: " + label)
        index = axis.index(label)
    elif callable(getattr(model, "label_index", None)):
        index = model.label_index(label)
        if isinstance(index, (bool, np.bool_)):
            raise ValueError("label_index must return a nonnegative integer")
        try:
            index = _op.index(index)
        except TypeError:
            raise ValueError("label_index must return a nonnegative integer") from None
    else:
        raise ValueError("probability predicates require registered labels or label_index(label)")

    key = (model_name, text, "proba_exact")
    proba = _ML_PRED_MEMO.get(key)
    if proba is None:
        single = getattr(model, "predict_proba_single", None)
        batch = getattr(model, "predict_proba", None)
        if callable(single):
            values = np.asarray(single(text), dtype=np.float64)
        elif callable(batch):
            matrix = np.asarray(batch([text]), dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[0] != 1:
                raise ValueError("predict_proba([text]) must return shape (1, K)")
            values = matrix[0]
        else:
            raise RuntimeError("probability predicates require predict_proba or predict_proba_single; "
                               "argmax cannot implement a threshold")
        expected = len(axis) if axis is not None else getattr(model, "num_labels", None)
        if (values.ndim != 1 or not values.size
                or expected is not None and values.size != expected
                or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
            raise ValueError("model probabilities must be finite [0, 1] values on the declared label axis")
        if not 0 <= index < values.size:
            raise ValueError("label_index is outside the model probability axis")
        proba = np.array(values, copy=True)
        proba.setflags(write=False)
        _ML_PRED_MEMO[key] = proba
    if not 0 <= index < proba.size:
        raise ValueError("label_index is outside the model probability axis")
    return float(proba[index])


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


def _get_ml_proba_batch(model_name: str, texts: List[str]) -> np.ndarray:
    """Return approximate batched probabilities with a typed per-text memo.

    A registered neural classifier may produce slightly different scores for
    ``predict_proba(texts)`` and repeated ``predict_proba([text])`` calls.  The
    Chase therefore calls this helper only for the explicit
    ``CHASE_BATCH_ML=approximate`` opt-in; exact rule semantics use the scalar
    predicate path.

    Only cache misses are sent to the registered model's batch
    ``predict_proba`` implementation.  Rows are stored under separate
    ``(model_name, text, "proba_batch_approx")`` keys.  They are deliberately
    never reused by exact per-document predicates, so an approximate run cannot
    contaminate a later exact Chase in the same process.  Re-registering
    ``model_name`` invalidates both caches through :func:`register_ml_model`.

    The batch protocol is deliberately strict: the model must return a
    non-empty two-dimensional matrix with one row per distinct missing text,
    and all cached/fresh rows for a model must have the same width.  Validation
    happens before fresh rows are committed, preventing a malformed batch from
    partially poisoning the memo.
    """
    model = _get_ml_model(model_name)
    text_list = list(texts)
    if not text_list:
        return np.empty((0, 0), dtype=np.float64)

    keys = [
        (model_name, text, "proba_batch_approx") for text in text_list
    ]
    rows_by_key: Dict[Tuple[str, ...], np.ndarray] = {}
    missing_keys: List[Tuple[str, ...]] = []
    missing_texts: List[str] = []
    pending_keys = set()
    expected_width: Optional[int] = None

    # Deduplicate repeated document text without changing output row order.
    for key, text in zip(keys, text_list):
        if key in rows_by_key:
            continue
        cached = _ML_PRED_MEMO.get(key, _MEMO_MISS)
        if cached is _MEMO_MISS:
            if key in pending_keys:
                continue
            pending_keys.add(key)
            missing_keys.append(key)
            missing_texts.append(text)
            continue
        row = np.asarray(cached, dtype=np.float64)
        if row.ndim != 1 or row.size == 0:
            raise ValueError(
                f"Cached ML probability for model {model_name!r} must be a "
                f"non-empty 1-D row, got shape {row.shape}"
            )
        if expected_width is None:
            expected_width = int(row.size)
        elif row.size != expected_width:
            raise ValueError(
                f"Cached ML probabilities for model {model_name!r} have "
                f"inconsistent widths {expected_width} and {row.size}"
            )
        rows_by_key[key] = row

    if missing_texts:
        predictor = getattr(model, "_clf", model)
        predict_proba = getattr(predictor, "predict_proba", None)
        if predict_proba is None:
            raise RuntimeError(
                f"Registered ML model {model_name!r} has no batched "
                "predict_proba() implementation"
            )
        fresh = np.asarray(predict_proba(missing_texts), dtype=np.float64)
        if fresh.ndim != 2:
            raise ValueError(
                f"Batched predict_proba for model {model_name!r} must return "
                f"a 2-D matrix, got shape {fresh.shape}"
            )
        if fresh.shape[0] != len(missing_texts):
            raise ValueError(
                f"Batched predict_proba for model {model_name!r} returned "
                f"{fresh.shape[0]} rows for {len(missing_texts)} texts"
            )
        if fresh.shape[1] == 0:
            raise ValueError(
                f"Batched predict_proba for model {model_name!r} returned "
                "zero probability columns"
            )
        if expected_width is not None and fresh.shape[1] != expected_width:
            raise ValueError(
                f"Batched predict_proba for model {model_name!r} returned "
                f"width {fresh.shape[1]}, but cached rows have width "
                f"{expected_width}"
            )

        # Validate the complete matrix before mutating the shared memo.
        for idx, key in enumerate(missing_keys):
            row = np.array(fresh[idx], dtype=np.float64, copy=True)
            rows_by_key[key] = row
        for key in missing_keys:
            _ML_PRED_MEMO[key] = rows_by_key[key]

    return np.stack([rows_by_key[key] for key in keys], axis=0)


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
    is at least *threshold*. This enables coverage on rare labels that would
    never be the argmax prediction.
    """

    model_name: str
    label: str
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("ML threshold must be finite and within [0, 1]")

    def __call__(self, doc: Document) -> bool:
        return _ml_probability(self.model_name, doc.cnt, self.label) >= float(self.threshold)

    def __repr__(self) -> str:
        return (f"ml_thresh({self.model_name!r}, label={self.label!r}, "
                f"thresh={self.threshold:.2f})")

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class MLScoreBandPredicate(Predicate):
    """Target-side model-confidence interval predicate.

    The predicate is true exactly when the registered multi-label model assigns
    ``label`` a probability in the half-open interval
    ``[lower, upper)``.  It is intentionally distinct from a pair of threshold
    predicates: the rule language has no generic predicate negation, and the
    half-open upper bound is required to express a residual rule without
    re-admitting the ordinary ``MLThresholdPredicate -> ADD`` prediction.

    In a relational rule, immutable ML predicates are evaluated on receiver
    ``x`` by :class:`~loris.chase.multi_chase.MultiChase`; a neighbouring
    ``LabelPredicate`` is evaluated only on source ``y``.  Thus this predicate
    is never a label seed and never supplies source evidence.
    """

    model_name: str
    label: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("ML score-band bounds must be finite")
        if not 0.0 <= float(self.lower) < float(self.upper) <= 1.0:
            raise ValueError(
                "ML score band must satisfy 0 <= lower < upper <= 1, got "
                f"[{self.lower!r}, {self.upper!r})"
            )

    def __call__(self, doc: Document) -> bool:
        score = _ml_probability(self.model_name, doc.cnt, self.label)
        return float(self.lower) <= score < float(self.upper)

    def __repr__(self) -> str:
        return (
            f"ml_band({self.model_name!r}, label={self.label!r}, "
            f"low={self.lower:.4f}, high={self.upper:.4f})"
        )

    def __str__(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# LabelPredicate — label-set operations on x.lbl
# ---------------------------------------------------------------------------

_VALID_LABEL_OPS = frozenset({"contains"})


@dataclass(frozen=True)
class LabelPredicate(Predicate):
    """Read positive membership without treating partial labels as a closed set.

    ``Document.lbl`` contains known positives; an empty value means no positive
    label is known. Literal equality/subset checks on it do not establish RDL
    comparison facts. Use scoped ``LabelCompare`` with the Chase relation state
    for such premises or conclusions. Membership is a pure read; this predicate
    never inserts, rejects or removes labels.
    """

    label: str
    op: str = "contains"

    def __post_init__(self) -> None:
        if self.op in {"eq", "subset", "strict_subset"}:
            raise ValueError("label-set comparisons require explicit relation facts; use RDL LabelCompare, "
                             "not a comparison of partial Document.lbl sets")
        if self.op not in _VALID_LABEL_OPS:
            raise ValueError(
                f"op must be one of {sorted(_VALID_LABEL_OPS)}, got {self.op!r}."
            )

    def __call__(self, doc: Document) -> bool:
        if self.op == "contains":
            return self.label in doc.lbl
        # Old pickles can bypass __post_init__; they must not revive the
        # incomplete-set comparison semantics at execution time.
        raise ValueError("only positive membership is supported here; use RDL LabelCompare for relations")

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

    Direct single-document evaluation is invalid.  Actual evaluation uses SpMV
    on a precomputed sparse adjacency matrix:
    ``(sim_graph[θ] @ label_state[:, τ_idx]) > 0``.
    """

    threshold: float = 0.85
    min_sources: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or not -1.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"threshold must be finite and in [-1, 1], got {self.threshold!r}."
            )
        if isinstance(self.min_sources, bool) or not isinstance(self.min_sources, int):
            raise TypeError("min_sources must be an integer")
        if self.min_sources < 1:
            raise ValueError("min_sources must be at least 1")

    def __call__(self, doc: Document) -> bool:
        raise RuntimeError(
            "SimPredicate requires cross-document graph context and cannot be "
            "evaluated on a single Document"
        )

    def __repr__(self) -> str:
        support = f", support>={self.min_sources}" if self.min_sources != 1 else ""
        return f"sim(x, y, >{self.threshold:.2f}{support})"

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
    for document *x*, find all documents *y* that **share at least one value**
    of attribute ``attr_name`` with *x*.  If any such *y* has the required label
    (checked via LabelPredicate in the same body), the rule fires on *x*.

    Attribute A is **multi-valued**: each document holds a *set* of values for a
    given attribute type (e.g. several NER ORG entities, several regex hits), so
    ``x.A=y.A`` means *value-set intersection is non-empty*, not scalar equality.
    Membership is stored externally as a ``scipy.sparse`` (n_docs × n_values)
    0/1 csr per attribute type (see :mod:`loris.rules.virtual_attributes`);
    cluster IDs degenerate to a one-hot row. ``group_count`` records the number
    of value columns (formerly the number of discrete group ids).

    Direct single-document evaluation is invalid.  Actual evaluation uses the
    precomputed membership matrix via SpMV
    (``membership @ (membershipᵀ @ y_mask) > 0``).
    """

    attr_name: str
    group_count: int = 0

    def __call__(self, doc: Document) -> bool:
        raise RuntimeError(
            "GroupPredicate requires cross-document membership context and "
            "cannot be evaluated on a single Document"
        )

    def __repr__(self) -> str:
        return f"x.{self.attr_name} == y.{self.attr_name}"

    def __str__(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# TopKPredicate — rank-based document-neighbour relation
# ---------------------------------------------------------------------------

_VALID_TOPK_MODES = frozenset({"directed", "mutual", "symmetric_union"})
_VALID_TOPK_SOURCE_POLICIES = frozenset({
    "validated", "recursive", "anchored_recursive",
})


@dataclass(frozen=True)
class TopKPredicate(Predicate):
    """Rank-based cross-document relation ``y in TopK(x)``.

    The corresponding sparse graph uses the receiver/source convention
    ``A[x, y] = 1``: row ``x`` lists the sources whose labels may support target
    document ``x``.  ``mode`` controls how a directed kNN relation is converted:

    ``directed``
        Keep ``y in TopK(x)`` exactly (the standard target-centric kNN query).
    ``mutual``
        Keep an edge only when each document is in the other's top-k set.
    ``symmetric_union``
        Keep an edge when either directed relation exists.

    ``scope_attr`` optionally names a precomputed partition (for example
    ``cluster_50``); graph construction then searches only within a source's
    partition. ``min_sources`` is a rule-evaluation quorum and deliberately is
    not part of :attr:`graph_key`, so several rules can share the same graph.
    ``source_policy='validated'`` restricts label evidence to explicit
    :math:`Gamma`/human facts; ``'recursive'`` also permits Chase-derived facts
    and must be treated as a separately reported ablation.
    ``'anchored_recursive'`` permits current Chase-derived sources for the
    total ``min_sources`` quorum but additionally requires
    ``min_validated_sources`` same-label :math:`Gamma` sources.  Its Chase
    implementation is intentionally limited to a single graph-only,
    same-label residual ADD rule so this two-quorum interpretation is never
    silently applied to an ambiguous join.

    Unlike the legacy Sim/Group predicate stubs, direct per-document evaluation
    fails loudly.  This predicate has meaning only with a graph and a second
    document variable; treating it as a local boolean silently changes the rule.
    """

    attr: str = "cnt"
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    k: int = 10
    mode: str = "directed"
    scope_attr: str = ""
    min_similarity: float = 0.0
    min_sources: int = 1
    source_policy: str = "validated"
    min_validated_sources: int = 1

    def __post_init__(self) -> None:
        if self.attr not in VALID_ATTRS:
            raise ValueError(
                f"attr must be one of {sorted(VALID_ATTRS)}, got {self.attr!r}."
            )
        if not isinstance(self.encoder, str) or not self.encoder.strip():
            raise ValueError("encoder must be a non-empty string")
        if isinstance(self.k, bool) or not isinstance(self.k, int):
            raise TypeError("k must be an integer")
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.mode not in _VALID_TOPK_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_TOPK_MODES)}, "
                f"got {self.mode!r}."
            )
        if not isinstance(self.scope_attr, str):
            raise TypeError("scope_attr must be a string")
        if not np.isfinite(self.min_similarity) or not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError(
                "min_similarity must be finite and in [-1, 1], "
                f"got {self.min_similarity!r}."
            )
        if isinstance(self.min_sources, bool) or not isinstance(self.min_sources, int):
            raise TypeError("min_sources must be an integer")
        if self.min_sources < 1:
            raise ValueError("min_sources must be at least 1")
        if self.source_policy not in _VALID_TOPK_SOURCE_POLICIES:
            raise ValueError(
                "source_policy must be one of "
                f"{sorted(_VALID_TOPK_SOURCE_POLICIES)}, "
                f"got {self.source_policy!r}."
            )
        if isinstance(self.min_validated_sources, bool) or not isinstance(
            self.min_validated_sources, int,
        ):
            raise TypeError("min_validated_sources must be an integer")
        if self.min_validated_sources < 1:
            raise ValueError("min_validated_sources must be at least 1")

    @property
    def graph_key(self) -> Tuple[str, str, int, str, str, float]:
        """Stable key for the immutable neighbour graph used by this predicate."""
        return (
            self.attr,
            self.encoder,
            self.k,
            self.mode,
            self.scope_attr,
            float(self.min_similarity),
        )

    def __call__(self, doc: Document) -> bool:
        raise RuntimeError(
            "TopKPredicate requires cross-document graph context and cannot be "
            "evaluated on a single Document"
        )

    def __repr__(self) -> str:
        scope = f", scope={self.scope_attr!r}" if self.scope_attr else ""
        floor = (
            f", cos>={self.min_similarity:.3f}"
            if self.min_similarity > -1.0 else ""
        )
        support = f", support>={self.min_sources}" if self.min_sources != 1 else ""
        policy = f", sources={self.source_policy}"
        anchor_support = (
            f", gamma_support>={self.min_validated_sources}"
            if self.source_policy == "anchored_recursive" else ""
        )
        return (
            f"topk_{self.mode}(y in N_{self.k}(x), attr={self.attr!r}"
            f"{scope}{floor}{support}{policy}{anchor_support})"
        )

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
        base["negate"] = p.negate

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

    elif isinstance(p, MLScoreBandPredicate):
        base["model_name"] = p.model_name
        base["label"] = p.label
        base["lower"] = p.lower
        base["upper"] = p.upper

    elif isinstance(p, MLPredicate):
        base["model_name"] = p.model_name
        base["label"] = p.label

    elif isinstance(p, LabelPredicate):
        base["label"] = p.label
        base["op"] = p.op

    elif isinstance(p, SimPredicate):
        base["threshold"] = p.threshold
        base["min_sources"] = p.min_sources

    elif isinstance(p, TopKPredicate):
        base["attr"] = p.attr
        base["encoder"] = p.encoder
        base["k"] = p.k
        base["mode"] = p.mode
        base["scope_attr"] = p.scope_attr
        base["min_similarity"] = p.min_similarity
        base["min_sources"] = p.min_sources
        base["source_policy"] = p.source_policy
        base["min_validated_sources"] = p.min_validated_sources

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
            negate=d.get("negate", False),
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
    if ptype == "MLScoreBandPredicate":
        return MLScoreBandPredicate(
            model_name=d["model_name"],
            label=d["label"],
            lower=float(d["lower"]),
            upper=float(d["upper"]),
        )
    if ptype == "MLPredicate":
        return MLPredicate(
            model_name=d["model_name"],
            label=d["label"],
        )
    if ptype == "LabelPredicate":
        op = d.get("op", "contains")
        if op == "minus":
            # Legacy side-effecting op removed (label mutation is a rule
            # consequence, not a predicate). Historical rules that stored a
            # minus body predicate are no longer constructible; surface this
            # explicitly so callers migrate to consequence_op="remove".
            raise ValueError(
                "LabelPredicate op='minus' is no longer supported. Label "
                "removal is a rule consequence (consequence_op='remove'), not "
                "a body predicate. Re-discover or migrate the affected rule."
            )
        return LabelPredicate(
            label=d["label"],
            op=op,
        )
    if ptype == "SimPredicate":
        return SimPredicate(
            threshold=float(d.get("threshold", 0.85)),
            min_sources=int(d.get("min_sources", 1)),
        )
    if ptype == "TopKPredicate":
        return TopKPredicate(
            attr=d.get("attr", "cnt"),
            encoder=d.get(
                "encoder", "sentence-transformers/all-MiniLM-L6-v2",
            ),
            k=int(d.get("k", 10)),
            mode=d.get("mode", "directed"),
            scope_attr=d.get("scope_attr", ""),
            min_similarity=float(d.get("min_similarity", 0.0)),
            min_sources=int(d.get("min_sources", 1)),
            source_policy=d.get("source_policy", "validated"),
            min_validated_sources=int(d.get("min_validated_sources", 1)),
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
        "MLScoreBandPredicate, "
        "LabelPredicate, SimPredicate, TopKPredicate, GroupPredicate."
    )
