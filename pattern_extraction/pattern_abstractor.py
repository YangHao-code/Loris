"""
pattern_extraction/pattern_abstractor.py
-----------------------------------------
Three-step pattern abstraction pipeline from the LORIS paper.

Steps
-----
(1) Clustering       — pretrained embeddings + k-means (k given or auto).
(2) Candidate retrieval — TF-IDF terms, spaCy noun/verb chunks, regex
                          entity spans; combined into CooccurPredicate and
                          BeforePredicate pairs (each targeting attr="cnt").
(3) Lightweight screening — filter by coverage ∈ [min, max] and by entropy
                            discriminability < threshold.

Mathematical note on entropy (Step 3):
  H = -Σ_{i=1}^{|Λ|} P_i log(P_i)
  where P_i = |{docs from class i satisfying the pattern}|
              / |{all docs satisfying the pattern}|.
  The smaller H, the better the pattern discriminates between classes.

Regex configuration
-------------------
Entity-regex patterns are loaded from a YAML or JSON file, not hardcoded.
Pass ``regex_config_path`` to use a custom file; ``None`` loads the bundled
``regex_patterns.yaml`` (sibling of this module).  The file can be copied
and edited for domain-specific entity types without touching Python code.
See :func:`load_regex_config` for the expected schema.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

from pattern_extraction.document import Document
from pattern_extraction.predicates import (
    BeforePredicate,
    CooccurPredicate,
    FreqPredicate,
    MatchPredicate,
    TextualPredicate,
    _Pattern,
    flags_from_names,
)

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded with clear install messages
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required for PatternAbstractor.\n"
        "Install with:  pip install sentence-transformers"
    ) from _e

try:
    import spacy  # type: ignore[import]
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore[assignment]
    _SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Path to the bundled default regex config shipped alongside this module.
_DEFAULT_REGEX_CONFIG: Path = Path(__file__).parent / "regex_patterns.yaml"


# ---------------------------------------------------------------------------
# Regex config loader
# ---------------------------------------------------------------------------

def load_regex_config(
    path: Optional[str] = None,
) -> List[Tuple[str, re.Pattern[str]]]:
    """
    Load entity-regex patterns from a YAML or JSON configuration file.

    Parameters
    ----------
    path : str or None
        Path to a ``.yaml``/``.yml`` or ``.json`` config file.
        When ``None`` (default) the bundled ``regex_patterns.yaml`` is used.

    Returns
    -------
    List[Tuple[str, re.Pattern[str]]]
        A list of ``(name, compiled_pattern)`` tuples, in the order they
        appear in the config file.

    File schema (YAML example)::

        patterns:
          - name: id
            pattern: '\\b[A-Z]{2,}\\d+\\b'
            flags: []
          - name: error
            pattern: '\\bERR[-_]\\w+\\b'
            flags: [IGNORECASE]

    The ``flags`` list accepts: ``IGNORECASE`` (or ``I``),
    ``MULTILINE`` (``M``), ``DOTALL`` (``S``).

    Raises
    ------
    ImportError
        If a YAML path is given but PyYAML is not installed.
    FileNotFoundError
        If the specified file does not exist.
    KeyError / ValueError
        If a required field is missing or a flag name is unrecognised.
    """
    target = Path(path) if path is not None else _DEFAULT_REGEX_CONFIG

    if not target.exists():
        raise FileNotFoundError(
            f"Regex config file not found: {target}\n"
            "Pass regex_config_path=None to use the bundled default."
        )

    suffix = target.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import]
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to load YAML regex config files.\n"
                "Install with:  pip install pyyaml\n"
                "Alternatively, convert your config to JSON."
            ) from err
        with target.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

    elif suffix == ".json":
        with target.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

    else:
        raise ValueError(
            f"Unsupported config file extension {suffix!r}. "
            "Expected .yaml, .yml, or .json."
        )

    result: List[Tuple[str, re.Pattern[str]]] = []
    for entry in raw.get("patterns", []):
        name: str = entry["name"]
        pattern_str: str = entry["pattern"]
        flag_names: List[str] = entry.get("flags", [])
        combined_flags = flags_from_names(flag_names)
        compiled = re.compile(pattern_str, combined_flags)
        result.append((name, compiled))

    logger.debug(
        "Loaded %d entity-regex patterns from %s.", len(result), target
    )
    return result


# ---------------------------------------------------------------------------
# PatternAbstractor — main pipeline class
# ---------------------------------------------------------------------------

class PatternAbstractor:
    """
    Three-step pattern abstraction pipeline (LORIS framework).

    Step 1 — **Clustering**: embeds documents with a pretrained
    sentence-transformer model and groups them with k-means.  ``k`` can be
    supplied explicitly or selected automatically via the silhouette score.

    Step 2 — **Candidate retrieval**: for each cluster, extracts anchor terms
    from three complementary sources (TF-IDF, spaCy chunks, regex spans) and
    forms :class:`~pattern_extraction.predicates.CooccurPredicate` /
    :class:`~pattern_extraction.predicates.BeforePredicate` pairs targeting
    ``attr="cnt"`` (document content).  Single-anchor TF-IDF and regex terms
    are also emitted as
    :class:`~pattern_extraction.predicates.MatchPredicate` objects.

    Step 3 — **Lightweight screening**: retains only predicates whose
    *coverage* (fraction of documents matched) lies in
    ``[min_coverage, max_coverage]`` and whose label-distribution *entropy*
    is below ``max_entropy_threshold``.

    Parameters
    ----------
    n_clusters : int or None, optional
        Number of k-means clusters.  ``None`` (default) selects the best
        ``k`` automatically via the silhouette score.
    max_auto_k : int, optional
        Upper bound for automatic ``k`` search. Default 15.
    embedding_model : str, optional
        sentence-transformers model identifier.
        Default ``"sentence-transformers/all-MiniLM-L6-v2"``.
    tfidf_top_k : int, optional
        Number of top TF-IDF anchors to extract per cluster. Default 20.
    spacy_model : str, optional
        spaCy model name for noun chunks and verb lemmas.
        Default ``"en_core_web_sm"``.
    min_coverage : float, optional
        Minimum fraction of total documents a predicate must match. Default 0.02.
    max_coverage : float, optional
        Maximum fraction of total documents a predicate must match. Default 0.90.
    max_entropy_threshold : float, optional
        Predicates with label-distribution entropy ≥ this value are discarded.
        Entropy is in nats (natural log). Default 1.0.
    max_pairs_per_cluster : int, optional
        Cap on anchor pairs per cluster to prevent O(k²) blowup. Default 500.
    random_state : int, optional
        Random seed for k-means reproducibility. Default 42.
    regex_config_path : str or None, optional
        Path to a YAML/JSON entity-regex config file.  ``None`` loads the
        bundled ``regex_patterns.yaml``.  Users can copy the bundled file,
        edit it for their domain, and pass the custom path here.
    auto_regex : bool, optional
        If ``True``, enable automatic discovery of high-frequency structural
        token patterns from the corpus.  Discovered patterns are appended to
        the entity-regex list before Step 2 anchor extraction.  Default ``False``.
    auto_regex_min_freq : int, optional
        Minimum corpus-wide frequency for a token shape to be converted into
        an auto-discovered regex.  Only used when ``auto_regex=True``.
        Default 3.

    Attributes
    ----------
    patterns_ : List[TextualPredicate]
        Predicates that survived Step 3 screening (populated after ``fit``).
    cluster_labels_ : np.ndarray, shape (n_docs,)
        k-means cluster assignment for each training document.
    n_clusters_ : int
        Actual number of clusters used.
    label_encoder_ : LabelEncoder or None
        Fitted encoder when string labels are provided; ``None`` otherwise.
    n_classes_ : int
        Number of unique classes observed during ``fit``.
    is_fitted : bool
        ``True`` after a successful ``fit`` call.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        max_auto_k: int = 15,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        tfidf_top_k: int = 20,
        spacy_model: str = "en_core_web_sm",
        min_coverage: float = 0.02,
        max_coverage: float = 0.90,
        max_entropy_threshold: float = 1.0,
        max_pairs_per_cluster: int = 500,
        random_state: int = 42,
        regex_config_path: Optional[str] = None,
        auto_regex: bool = False,
        auto_regex_min_freq: int = 3,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_auto_k = max_auto_k
        self.embedding_model = embedding_model
        self.tfidf_top_k = tfidf_top_k
        self.spacy_model = spacy_model
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.max_entropy_threshold = max_entropy_threshold
        self.max_pairs_per_cluster = max_pairs_per_cluster
        self.random_state = random_state
        self.regex_config_path = regex_config_path
        self.auto_regex = auto_regex
        self.auto_regex_min_freq = auto_regex_min_freq

        # Load entity-regex config immediately so config errors surface early.
        self._entity_regexes: List[Tuple[str, re.Pattern[str]]] = (
            load_regex_config(regex_config_path)
        )

        self.is_fitted: bool = False

        # Lazy-loaded heavy objects
        self._encoder: Optional[SentenceTransformer] = None
        self._nlp: Optional[object] = None  # spaCy Language object
        self._auto_regex_extractor: Optional["AutoRegexExtractor"] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: Union[List[str], List[Document]],
        labels: Union[List[str], np.ndarray],
    ) -> "PatternAbstractor":
        """
        Run the full three-step LORIS pattern abstraction pipeline.

        Parameters
        ----------
        texts : List[str] or List[Document]
            Training corpus.  Plain strings are automatically wrapped into
            :class:`~pattern_extraction.document.Document` objects
            (``cnt=text``).  Pass ``List[Document]`` to leverage ``mtd``
            and ``ttl`` attributes in downstream predicate evaluation.
        labels : List[str] or np.ndarray
            Document labels:

            * **Single-label** — 1-D sequence of string or integer class
              identifiers, e.g. ``["finance", "sports", "finance"]``.
            * **Multi-hot** — 2-D ``np.ndarray`` of shape
              ``(n_docs, n_classes)`` with binary values.

        Returns
        -------
        PatternAbstractor
            ``self``, enabling method chaining.

        Raises
        ------
        ValueError
            If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("`texts` must be a non-empty list.")

        # Normalise to List[Document]
        docs: List[Document] = [
            Document(cnt=t) if isinstance(t, str) else t
            for t in texts
        ]
        # Raw content strings used for embedding and TF-IDF
        raw_texts: List[str] = [d.cnt for d in docs]

        n_docs = len(docs)
        logger.info("PatternAbstractor.fit() called on %d documents.", n_docs)

        # -- Label normalisation -------------------------------------------
        class_indices_per_doc = self._normalize_labels(labels, n_docs)

        # -- Step 1: Clustering -------------------------------------------
        logger.info("Step 1 | Embedding and clustering %d documents.", n_docs)
        _, cluster_labels = self._step1_cluster(raw_texts)
        self.cluster_labels_ = cluster_labels
        logger.info("Step 1 | Selected k=%d clusters.", self.n_clusters_)

        # -- Auto-regex discovery (optional, before Step 2) ----------------
        if self.auto_regex:
            from pattern_extraction.auto_regex import AutoRegexExtractor

            self._auto_regex_extractor = AutoRegexExtractor(
                min_freq=self.auto_regex_min_freq,
            )
            self._auto_regex_extractor.fit(
                raw_texts, existing_regexes=self._entity_regexes
            )
            auto_patterns = self._auto_regex_extractor.get_patterns()
            self._entity_regexes.extend(auto_patterns)
            logger.info(
                "Auto-regex: discovered %d structural patterns.",
                len(auto_patterns),
            )

        # -- Step 2: Candidate pattern generation -------------------------
        logger.info("Step 2 | Anchor extraction and pattern generation.")
        candidates: List[TextualPredicate] = []

        for k in range(self.n_clusters_):
            indices = [i for i, c in enumerate(cluster_labels) if c == k]
            c_texts = [raw_texts[i] for i in indices]

            if not c_texts:
                logger.warning("Cluster %d is empty; skipping.", k)
                continue

            anchors = self._collect_anchors(c_texts)
            cluster_pats = self._step2_generate_patterns(anchors, c_texts)
            candidates.extend(cluster_pats)

            logger.info(
                "  Cluster %d | %d docs | %d anchors | %d candidate predicates",
                k, len(c_texts), len(anchors), len(cluster_pats),
            )

        # Global dedup: frozen dataclasses are hashable
        candidates = list(dict.fromkeys(candidates))
        logger.info(
            "Step 2 | Total unique candidate predicates: %d", len(candidates)
        )

        # -- Step 3: Lightweight screening --------------------------------
        logger.info(
            "Step 3 | Screening by coverage [%.3f, %.3f] and entropy < %.3f.",
            self.min_coverage, self.max_coverage, self.max_entropy_threshold,
        )
        self.patterns_: List[TextualPredicate] = self._step3_screen_patterns(
            candidates, raw_texts, class_indices_per_doc
        )
        logger.info(
            "Step 3 | Predicates surviving screening: %d / %d",
            len(self.patterns_), len(candidates),
        )
        if not self.patterns_:
            logger.warning(
                "No predicates survived screening.  Consider relaxing "
                "`min_coverage`, `max_coverage`, or `max_entropy_threshold`."
            )

        self.is_fitted = True
        return self

    def to_store(self) -> "PatternStore":  # type: ignore[name-defined]
        """
        Wrap fitted predicates in a :class:`~pattern_extraction.pattern_store.PatternStore`.

        The store provides ``apply(docs)`` for applying predicates to any
        new corpus and ``save``/``load`` for persistence.

        Returns
        -------
        PatternStore

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        self._check_fitted()
        # Import here to avoid circular import (pattern_store imports from here)
        from pattern_extraction.pattern_store import PatternStore

        label_names: Optional[List[str]] = (
            self.label_encoder_.classes_.tolist()
            if self.label_encoder_ is not None
            else None
        )
        return PatternStore(
            patterns=self.patterns_,
            n_classes=self.n_classes_,
            label_names=label_names,
        )

    def get_pattern_stats(self) -> List[Dict]:
        """
        Return a list of dicts describing each surviving predicate.

        Each dict has keys: ``"predicate"`` (repr string), ``"type"``,
        ``"attr"``.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        self._check_fitted()
        return [
            {"predicate": repr(p), "type": type(p).__name__, "attr": p.attr}
            for p in self.patterns_
        ]

    # ------------------------------------------------------------------
    # Step 1: Clustering
    # ------------------------------------------------------------------

    def _step1_cluster(
        self, texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed *texts* and assign each to a k-means cluster.

        If ``self.n_clusters`` is ``None``, the best ``k`` in
        ``[2, min(self.max_auto_k, floor(sqrt(n)))]`` is selected by
        maximising the silhouette score on a random subsample of ≤ 2,000
        documents.

        Returns
        -------
        embeddings : np.ndarray, shape (n, d)
        cluster_labels : np.ndarray, shape (n,)
        """
        if self._encoder is None:
            logger.info(
                "Loading sentence-transformer model '%s'.", self.embedding_model
            )
            self._encoder = SentenceTransformer(self.embedding_model)

        embeddings: np.ndarray = self._encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        n = len(texts)

        if self.n_clusters is not None:
            k = min(self.n_clusters, n)
            if k != self.n_clusters:
                logger.warning(
                    "n_clusters=%d exceeds n_docs=%d; clamped to %d.",
                    self.n_clusters, n, k,
                )
        else:
            k = self._auto_select_k(embeddings)
            logger.info("Auto k-selection chose k=%d.", k)

        self.n_clusters_ = k
        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        cluster_labels: np.ndarray = km.fit_predict(embeddings)
        return embeddings, cluster_labels

    def _auto_select_k(self, embeddings: np.ndarray) -> int:
        """
        Select the number of clusters that maximises the silhouette score.

        Searches ``k ∈ [2, min(max_auto_k, floor(sqrt(n)))]``.
        Silhouette scores are computed on a random subsample of at most
        2,000 points to keep runtime tractable for large corpora.

        Returns
        -------
        int  Best ``k`` found.
        """
        n = len(embeddings)
        k_max = min(self.max_auto_k, max(2, int(math.floor(math.sqrt(n)))))

        if k_max < 2:
            logger.warning(
                "Dataset too small for auto k-selection (n=%d); defaulting to k=2.", n
            )
            return 2

        _MAX_SAMPLE = 2000
        rng = np.random.default_rng(self.random_state)
        if n > _MAX_SAMPLE:
            sample_idx = rng.choice(n, size=_MAX_SAMPLE, replace=False)
            emb_sample = embeddings[sample_idx]
        else:
            emb_sample = embeddings

        best_k, best_score = 2, -1.0
        for k in range(2, k_max + 1):
            km = KMeans(
                n_clusters=k, n_init=5, random_state=self.random_state
            )
            labels_sample = km.fit_predict(emb_sample)

            if len(np.unique(labels_sample)) < 2:
                continue

            score = float(silhouette_score(emb_sample, labels_sample))
            logger.debug("  k=%d  silhouette=%.4f", k, score)

            if score > best_score:
                best_score, best_k = score, k

        return best_k

    # ------------------------------------------------------------------
    # Step 2: Anchor extraction
    # ------------------------------------------------------------------

    def _collect_anchors(self, cluster_texts: List[str]) -> List[str]:
        """
        Merge anchors from TF-IDF, spaCy chunks, and regex spans.

        Returns deduplicated lowercase anchor terms preserving insertion
        order (TF-IDF → chunk → regex).
        """
        a_tfidf = self._anchors_tfidf(cluster_texts)
        a_chunk = self._anchors_spacy(cluster_texts)
        a_regex = self._anchors_regex(cluster_texts)

        return list(
            dict.fromkeys(
                t.lower().strip()
                for t in (a_tfidf + a_chunk + a_regex)
                if t.strip()
            )
        )

    def _anchors_tfidf(self, cluster_texts: List[str]) -> List[str]:
        """
        Extract top-``tfidf_top_k`` anchor terms by mean TF-IDF score.

        Bigrams are included (``ngram_range=(1, 2)``).
        """
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words="english",
        )
        try:
            X = vectorizer.fit_transform(cluster_texts)
        except ValueError:
            return []

        mean_scores: np.ndarray = np.asarray(X.mean(axis=0)).flatten()
        top_n = min(self.tfidf_top_k, len(mean_scores))
        top_indices = mean_scores.argsort()[::-1][:top_n]
        feature_names = vectorizer.get_feature_names_out()
        return [feature_names[i] for i in top_indices]

    def _anchors_spacy(self, cluster_texts: List[str]) -> List[str]:
        """
        Extract noun chunks and verb lemmas using spaCy.

        If spaCy or the requested model is unavailable, returns ``[]`` and
        logs a one-time warning.
        """
        if not _SPACY_AVAILABLE:
            logger.warning(
                "spaCy is not installed; skipping chunk anchors.  "
                "Install with:  pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )
            return []

        if self._nlp is None:
            try:
                self._nlp = spacy.load(
                    self.spacy_model,
                    disable=["ner"],
                )
                logger.info("Loaded spaCy model '%s'.", self.spacy_model)
            except OSError:
                logger.warning(
                    "spaCy model '%s' not found; skipping chunk anchors.  "
                    "Install with:  python -m spacy download %s",
                    self.spacy_model, self.spacy_model,
                )
                self._nlp = False  # type: ignore[assignment]
                return []

        if self._nlp is False:
            return []

        anchors: set = set()
        for doc in self._nlp.pipe(cluster_texts, batch_size=64):  # type: ignore[union-attr]
            try:
                for chunk in doc.noun_chunks:
                    text = chunk.text.lower().strip()
                    if text:
                        anchors.add(text)
            except ValueError:
                pass

            for token in doc:
                if token.pos_ in ("VERB", "AUX") and not token.is_stop:
                    lemma = token.lemma_.lower().strip()
                    if lemma:
                        anchors.add(lemma)

        return list(anchors)

    def _anchors_regex(self, cluster_texts: List[str]) -> List[str]:
        """
        Extract entity-bearing spans using entity-regex patterns loaded from
        the config file (``self._entity_regexes``).

        Returns unique matched span strings (lowercased).  The raw spans
        (rather than the regex pattern strings) become anchors, because they
        represent concrete surface forms that co-occur with other anchors in
        the corpus.
        """
        anchors: set = set()
        for text in cluster_texts:
            for _, pattern in self._entity_regexes:
                for match in pattern.finditer(text):
                    span = match.group(0).lower().strip()
                    if span:
                        anchors.add(span)
        return list(anchors)

    # ------------------------------------------------------------------
    # Step 2: Pattern generation from anchors
    # ------------------------------------------------------------------

    def _step2_generate_patterns(
        self,
        anchors: List[str],
        cluster_texts: List[str],
    ) -> List[TextualPredicate]:
        """
        Form textual predicates from the merged anchor list for one cluster.

        Emitted predicates (all with ``attr="cnt"``):

        * :class:`MatchPredicate` — one per anchor term (single-anchor coverage).
        * :class:`CooccurPredicate` — one per unordered pair ``(a, b)``.
        * :class:`BeforePredicate` ``(a, b)`` / ``(b, a)`` — emitted when the
          ordering is attested in at least one cluster document.

        The anchor list is capped so that the number of pairs stays within
        ``max_pairs_per_cluster``.

        Parameters
        ----------
        anchors : List[str]
            Deduplicated lowercase anchor terms.
        cluster_texts : List[str]
            Documents in this cluster (used to detect ordering for before-predicates).

        Returns
        -------
        List[TextualPredicate]
        """
        if not anchors:
            return []

        # -- Single-anchor MatchPredicates + FreqPredicates ---------------
        # MatchPredicate covers presence; FreqPredicate covers repetition.
        # Word-boundary regex for single tokens; literal match for phrases.
        match_preds: List[TextualPredicate] = []
        for anchor in anchors:
            if " " in anchor:
                raw = re.escape(anchor)            # phrase: exact literal
            else:
                raw = rf"\b{re.escape(anchor)}\b"  # word: boundary-anchored
            a_pat = _Pattern(raw=raw)
            match_preds.append(MatchPredicate(attr="cnt", r=a_pat))

            # Emit FreqPredicate only when the anchor genuinely recurs in some
            # cluster documents (median non-zero count >= 2).  This avoids
            # redundancy with MatchPredicate for rare or single-occurrence terms.
            eta = self._freq_eta(a_pat, cluster_texts)
            if eta is not None:
                match_preds.append(
                    FreqPredicate(attr="cnt", r=a_pat, op=">=", eta=eta)
                )

        if len(anchors) < 2:
            return match_preds

        # -- Pair-level predicates -----------------------------------------
        # Cap anchors to keep pairs ≤ max_pairs_per_cluster
        max_anchors = int(math.ceil(math.sqrt(2 * self.max_pairs_per_cluster))) + 1
        sampled = anchors[:max_anchors]

        pair_preds: List[TextualPredicate] = []
        seen: set = set()

        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                a, b = sampled[i], sampled[j]
                if a == b:
                    continue

                # Build _Pattern wrappers for the pair
                a_pat = _Pattern(raw=rf"\b{re.escape(a)}\b" if " " not in a else re.escape(a))
                b_pat = _Pattern(raw=rf"\b{re.escape(b)}\b" if " " not in b else re.escape(b))

                # -- CooccurPredicate ---------------------------------------
                key_co = ("cooccur", min(a, b), max(a, b))
                if key_co not in seen:
                    pair_preds.append(
                        CooccurPredicate(attr="cnt", r1=a_pat, r2=b_pat)
                    )
                    seen.add(key_co)

                # -- Ordering statistics over cluster docs ------------------
                before_ab = 0
                before_ba = 0

                for text in cluster_texts:
                    m_a = a_pat.search(text)
                    m_b = b_pat.search(text)
                    if m_a is not None and m_b is not None:
                        if m_a.start() < m_b.start():
                            before_ab += 1
                        elif m_b.start() < m_a.start():
                            before_ba += 1

                # -- BeforePredicate(a, b) ----------------------------------
                if before_ab > 0:
                    key_ab = ("before", a, b)
                    if key_ab not in seen:
                        pair_preds.append(
                            BeforePredicate(attr="cnt", r1=a_pat, r2=b_pat)
                        )
                        seen.add(key_ab)

                # -- BeforePredicate(b, a) ----------------------------------
                if before_ba > 0:
                    key_ba = ("before", b, a)
                    if key_ba not in seen:
                        pair_preds.append(
                            BeforePredicate(attr="cnt", r1=b_pat, r2=a_pat)
                        )
                        seen.add(key_ba)

                if len(pair_preds) >= self.max_pairs_per_cluster * 3:
                    return match_preds + pair_preds

        return match_preds + pair_preds

    def _freq_eta(
        self,
        anchor_pat: _Pattern,
        cluster_texts: List[str],
    ) -> Optional[float]:
        """
        Compute a representative frequency threshold for :class:`FreqPredicate`.

        Strategy
        --------
        1. Count occurrences of *anchor_pat* in every cluster document.
        2. Collect non-zero counts (≥ 2) — documents where the anchor recurs.
        3. Return the median of those counts (as float, for ``FreqPredicate.eta``
           normalisation stability).
        4. Return ``None`` when the median would be < 2, making a
           ``FreqPredicate`` redundant with the co-emitted ``MatchPredicate``.

        Using the **median** guards against outlier documents that repeat an
        anchor an atypically large number of times.

        Returns
        -------
        float or None
        """
        counts = [len(anchor_pat.findall(t)) for t in cluster_texts]
        nonzero = [c for c in counts if c >= 2]
        if not nonzero:
            return None
        eta = float(np.median(nonzero))
        return eta if eta >= 2.0 else None

    # ------------------------------------------------------------------
    # Step 3: Lightweight screening
    # ------------------------------------------------------------------

    def _step3_screen_patterns(
        self,
        candidates: List[TextualPredicate],
        texts: List[str],
        class_indices_per_doc: List[List[int]],
    ) -> List[TextualPredicate]:
        """
        Filter candidate predicates by coverage and entropy discriminability.

        Coverage criterion:
            ``min_coverage ≤ |matching_docs| / |total_docs| ≤ max_coverage``

        Entropy criterion (LORIS paper, §3):
            ``H = -Σ P_i log(P_i) < max_entropy_threshold``
            where ``P_i`` is the share of class-``i`` documents among all
            documents that match the predicate.  Lower entropy ⟹ better
            discriminability.

        Parameters
        ----------
        candidates : List[TextualPredicate]
        texts : List[str]
            Raw content strings (``doc.cnt``).
        class_indices_per_doc : List[List[int]]
            Active class indices per document.

        Returns
        -------
        List[TextualPredicate]
        """
        n_total = len(texts)
        surviving: List[TextualPredicate] = []

        for pred in candidates:
            # Evaluate predicate on each document (wrap as Document for __call__)
            matching: List[int] = [
                i for i, text in enumerate(texts)
                if pred(Document(cnt=text))
            ]

            # -- Coverage check -------------------------------------------
            coverage = len(matching) / n_total
            if not (self.min_coverage <= coverage <= self.max_coverage):
                continue

            # -- Entropy check --------------------------------------------
            entropy = self._compute_entropy(matching, class_indices_per_doc)
            if entropy >= self.max_entropy_threshold:
                continue

            surviving.append(pred)

        return surviving

    def _compute_entropy(
        self,
        matching_doc_indices: List[int],
        class_indices_per_doc: List[List[int]],
    ) -> float:
        """
        Compute the Shannon entropy of the class distribution over matched docs.

        For multi-label documents each active class contributes independently.
        Entropy is measured in **nats** (natural logarithm), consistent with
        the formula in the paper:  H = -Σ P_i ln(P_i).

        Returns
        -------
        float
            H ≥ 0.  Returns ``0.0`` for perfectly discriminative predicates.
        """
        class_counts: Counter = Counter()
        for idx in matching_doc_indices:
            for cls_idx in class_indices_per_doc[idx]:
                class_counts[cls_idx] += 1

        total = sum(class_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in class_counts.values():
            p_i = count / total
            if p_i > 0.0:
                entropy -= p_i * math.log(p_i)

        return entropy

    # ------------------------------------------------------------------
    # Label normalisation
    # ------------------------------------------------------------------

    def _normalize_labels(
        self,
        labels: Union[List[str], np.ndarray],
        n_docs: int,
    ) -> List[List[int]]:
        """
        Convert *labels* to a unified ``List[List[int]]`` format.

        Three accepted input formats:

        1. **Multi-hot 2-D array** — shape ``(n_docs, n_classes)``.
        2. **1-D string array / list** — encoded with :class:`LabelEncoder`.
        3. **1-D integer array / list** — used directly.

        Populates ``self.label_encoder_`` and ``self.n_classes_``.
        """
        labels_arr = np.asarray(labels)

        if labels_arr.ndim == 2:
            label_matrix = labels_arr.astype(np.int32)
            self.label_encoder_: Optional[LabelEncoder] = None
            self.n_classes_: int = label_matrix.shape[1]
            return [list(np.where(row)[0]) for row in label_matrix]

        if labels_arr.dtype.kind in ("U", "S", "O"):
            enc = LabelEncoder()
            int_labels = enc.fit_transform(labels_arr).astype(np.int32)
            self.label_encoder_ = enc
        else:
            int_labels = labels_arr.astype(np.int32)
            self.label_encoder_ = None

        self.n_classes_ = int(np.max(int_labels)) + 1
        return [[int(c)] for c in int_labels]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise ``RuntimeError`` if ``fit()`` has not been called yet."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet.  "
                "Call fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        k_info = (
            f"n_clusters_={self.n_clusters_}"
            if self.is_fitted
            else f"n_clusters={self.n_clusters!r}"
        )
        return (
            f"{self.__class__.__name__}("
            f"{k_info}, "
            f"embedding_model={self.embedding_model!r}, "
            f"status={status})"
        )
