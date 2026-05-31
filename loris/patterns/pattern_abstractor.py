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
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

from loris.document import Document
from loris.predicates import (
    BeforePredicate,
    CooccurPredicate,
    FreqPredicate,
    MatchPredicate,
    TextualPredicate,
    flags_from_names,
)
from loris.predicates._core import _Pattern

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
        pattern_mode: str = "full",  # "full" | "fast" | "sim"
        extra_stop_words: Optional[Set[str]] = None,
        anchor_min_df: int = 3,
        sim_threshold: Optional[float] = None,
        encoder: object = None,
        glove_path: Optional[str] = None,
        synonym_min_sim: float = 0.65,
        synonym_top_k: int = 5,
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
        self.pattern_mode = pattern_mode
        self.extra_stop_words: Set[str] = (
            {w.lower() for w in extra_stop_words} if extra_stop_words else set()
        )
        self.anchor_min_df = max(anchor_min_df, 1)
        self.sim_threshold = sim_threshold  # None → use default (0.45 sim, 0.85 exact)
        # In fast mode, always enable auto_regex
        if pattern_mode == "fast":
            self.auto_regex = True

        # Load entity-regex config immediately so config errors surface early.
        self._entity_regexes: List[Tuple[str, re.Pattern[str]]] = (
            load_regex_config(regex_config_path)
        )

        self.is_fitted: bool = False

        # GloVe synonym expansion
        self._glove_path = glove_path
        self._synonym_min_sim = synonym_min_sim
        self._synonym_top_k = synonym_top_k
        self._glove_model = None  # lazy-loaded KeyedVectors

        # Lazy-loaded heavy objects (encoder can be injected via __init__)
        self._encoder: Optional[SentenceTransformer] = encoder
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
        if self.n_clusters == 1:
            # 单集群模式：跳过 SentenceTransformer 编码和 KMeans
            self.n_clusters_ = 1
            self.cluster_labels_ = np.zeros(n_docs, dtype=np.int32)
            logger.info("Step 1 | n_clusters=1, skipping embedding and clustering "
                        "(%d documents).", n_docs)
        else:
            logger.info("Step 1 | Embedding and clustering %d documents.", n_docs)
            _, cluster_labels = self._step1_cluster(raw_texts)
            self.cluster_labels_ = cluster_labels
            logger.info("Step 1 | Selected k=%d clusters.", self.n_clusters_)
        cluster_labels = self.cluster_labels_

        # -- Auto-regex discovery (optional, before Step 2) ----------------
        if self.auto_regex:
            from loris.patterns.auto_regex import AutoRegexExtractor

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

        # Decide which fields to mine: always cnt; ttl only if any doc has one.
        anchor_fields: List[str] = ["cnt"]
        any_ttl = any(bool(getattr(d, "ttl", "")) for d in docs)
        any_mtd = any(bool(getattr(d, "mtd", "")) for d in docs)
        if any_ttl:
            anchor_fields.append("ttl")
        if any_mtd:
            anchor_fields.append("mtd")
        logger.info("Step 2 | Mining anchors over fields: %s", anchor_fields)

        for k in range(self.n_clusters_):
            indices = [i for i, c in enumerate(cluster_labels) if c == k]
            if not indices:
                logger.warning("Cluster %d is empty; skipping.", k)
                continue

            for field in anchor_fields:
                c_field_texts = [(getattr(docs[i], field) or "") for i in indices]
                if field != "cnt" and not any(t.strip() for t in c_field_texts):
                    continue  # all empty for this field in this cluster

                anchors = self._collect_anchors(c_field_texts)
                cluster_pats, _cluster_fcm = self._step2_generate_patterns(
                    anchors, c_field_texts, field=field,
                )
                candidates.extend(cluster_pats)

                logger.info(
                    "  Cluster %d (%s) | %d docs | %d anchors | %d candidate predicates",
                    k, field, len(c_field_texts), len(anchors), len(cluster_pats),
                )

        # Global dedup: frozen dataclasses are hashable
        candidates = list(dict.fromkeys(candidates))
        logger.info(
            "Step 2 | Total unique candidate predicates: %d", len(candidates)
        )

        # -- Step 3: Lightweight screening --------------------------------
        # Adapt entropy threshold to number of classes:
        # For many classes, max possible entropy (ln(n)) is high, so a fixed
        # threshold easily rejects everything.  Use at least 85% of ln(n).
        max_possible_entropy = math.log(max(self.n_classes_, 2))
        adaptive_threshold = max(
            self.max_entropy_threshold,
            0.85 * max_possible_entropy,
        )
        logger.info(
            "Step 3 | Screening by coverage [%.3f, %.3f] and entropy < %.3f "
            "(n_classes=%d, ln(n)=%.3f, adaptive_threshold=%.3f).",
            self.min_coverage, self.max_coverage, adaptive_threshold,
            self.n_classes_, max_possible_entropy, adaptive_threshold,
        )
        # Temporarily apply adaptive threshold
        orig_threshold = self.max_entropy_threshold
        self.max_entropy_threshold = adaptive_threshold
        self.patterns_: List[TextualPredicate] = self._step3_screen_patterns(
            candidates, raw_texts, class_indices_per_doc, docs=docs,
        )
        self.max_entropy_threshold = orig_threshold
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
        from loris.patterns.pattern_store import PatternStore

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

        In "fast" mode: TF-IDF + regex only (skip spaCy).

        Returns deduplicated lowercase anchor terms preserving insertion
        order (TF-IDF → chunk/entity → regex).
        """
        a_tfidf_raw = self._anchors_tfidf(cluster_texts)
        # Lemmatize TF-IDF anchors (noun plurals, verb tenses) if spaCy available
        if self._nlp is not None:
            a_tfidf = []
            for tok_text in a_tfidf_raw:
                lemma = " ".join(
                    t.lemma_.lower() for t in self._nlp(tok_text) if not t.is_space
                ).strip()
                a_tfidf.append(lemma if lemma else tok_text)
        else:
            a_tfidf = a_tfidf_raw

        if self.pattern_mode == "fast":
            # In fast mode, skip regex surface forms — entity regex templates
            # are added directly as MatchPredicates in _step2_generate_patterns().
            # Only use TF-IDF anchors to avoid anchor explosion.
            a_extra = []
        else:
            a_extra = self._anchors_spacy(cluster_texts) + self._anchors_regex(cluster_texts)

        merged = list(
            dict.fromkeys(
                t.lower().strip()
                for t in (a_tfidf + a_extra)
                if t.strip()
            )
        )
        # Filter out domain-specific stopwords from all anchor sources.
        if self.extra_stop_words:
            merged = [
                a for a in merged
                if not all(tok in self.extra_stop_words for tok in a.split())
            ]
        # Filter out English stopwords (covers spaCy anchors that bypass TF-IDF's stop_words="english")
        merged = [
            a for a in merged
            if not all(tok in self._EXPANSION_STOPWORDS for tok in a.split())
        ]
        return merged

    def _anchors_tfidf(self, cluster_texts: List[str]) -> List[str]:
        """
        Extract top-``tfidf_top_k`` anchor terms by mean TF-IDF score.

        Bigrams are included (``ngram_range=(1, 2)``).
        """
        # Merge sklearn English stopwords with any domain-specific extras.
        if self.extra_stop_words:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            merged_stop = list(ENGLISH_STOP_WORDS | self.extra_stop_words)
        else:
            merged_stop = "english"

        vectorizer = TfidfVectorizer(
            max_features=max(self.tfidf_top_k * 3, 500),
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words=merged_stop,
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

        from collections import Counter
        anchor_df: Counter = Counter()

        for doc in self._nlp.pipe(cluster_texts, batch_size=256):  # type: ignore[union-attr]
            doc_anchors: set = set()
            try:
                for chunk in doc.noun_chunks:
                    lemma_text = " ".join(
                        tok.lemma_.lower() for tok in chunk
                        if not tok.is_space and not tok.is_stop and not tok.is_punct
                    ).strip()
                    if lemma_text:
                        doc_anchors.add(lemma_text)
            except ValueError:
                pass

            for token in doc:
                if token.pos_ in ("VERB", "AUX") and not token.is_stop:
                    lemma = token.lemma_.lower().strip()
                    if lemma:
                        doc_anchors.add(lemma)

            for a in doc_anchors:
                anchor_df[a] += 1

        n_total = len(anchor_df)
        # Dynamic DF: an anchor with DF < min_coverage * n_docs cannot
        # possibly produce a predicate passing Step 3 coverage screening,
        # so we can safely prune it here (zero information loss).
        n_cluster = len(cluster_texts)
        effective_min_df = max(
            self.anchor_min_df,
            int(n_cluster * self.min_coverage),
        )
        if effective_min_df > 1:
            filtered = [a for a, cnt in anchor_df.items()
                        if cnt >= effective_min_df]
        else:
            filtered = list(anchor_df.keys())
        logger.info(
            "spaCy anchors: %d unique → %d after DF≥%d filter "
            "(dynamic from min_coverage=%.3f × %d docs)",
            n_total, len(filtered), effective_min_df,
            self.min_coverage, n_cluster,
        )
        return filtered

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
    # Morphological variant expansion
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Stopwords excluded from synonym/derivation expansion
    # ------------------------------------------------------------------
    _EXPANSION_STOPWORDS: frozenset = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could",
        "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
        "she", "her", "it", "its", "they", "them", "their",
        "this", "that", "these", "those", "who", "what", "which", "where",
        "when", "how", "why", "all", "each", "every", "both", "few", "more",
        "most", "some", "any", "no", "not", "only", "very", "just", "also",
        "than", "too", "so", "if", "or", "and", "but", "nor", "for", "yet",
        "in", "on", "at", "to", "by", "of", "up", "out", "off", "from",
        "with", "about", "into", "over", "after", "before", "between",
        "new", "one", "two", "first", "last", "long", "great", "little",
        "own", "other", "old", "right", "big", "high", "small", "large",
        "next", "early", "young", "same", "able", "get", "got", "go",
        "come", "came", "make", "made", "take", "took", "give", "gave",
        # Academic domain high-frequency non-discriminative words
        "paper", "study", "method", "approach", "results", "show", "propose",
        "proposed", "based", "using", "used", "model", "data", "system",
    })

    # ------------------------------------------------------------------
    # Morphological helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _noun_plurals(lemma: str) -> set:
        """Generate plural forms for a noun lemma."""
        forms: set = set()
        forms.add(lemma + "s")
        if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
            forms.add(lemma[:-1] + "ies")
        elif lemma.endswith(("s", "x", "z", "sh", "ch")):
            forms.add(lemma + "es")
        return forms

    @staticmethod
    def _verb_inflections(lemma: str) -> set:
        """Generate verb inflections from a lemma."""
        forms: set = set()
        if lemma.endswith("e"):
            forms.add(lemma + "d")           # love → loved
            forms.add(lemma[:-1] + "ing")    # love → loving
            forms.add(lemma + "s")           # love → loves
        elif (len(lemma) in (3, 4)
              and lemma[-1] not in "aeiouwy"
              and lemma[-2] in "aeiou"
              and (len(lemma) < 3 or lemma[-3] not in "aeiou")):
            # CVC doubling: run → running, stop → stopped
            forms.add(lemma + lemma[-1] + "ing")
            forms.add(lemma + lemma[-1] + "ed")
            forms.add(lemma + "s")
        else:
            forms.add(lemma + "ed")
            forms.add(lemma + "ing")
            forms.add(lemma + "s")
            if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
                forms.add(lemma[:-1] + "ied")
                forms.add(lemma[:-1] + "ies")
        return forms

    @staticmethod
    def _adj_derivations(lemma: str) -> set:
        """Generate adverb, noun, and comparative/superlative forms for adjectives."""
        forms: set = set()
        # -ly adverb
        if lemma.endswith("y") and len(lemma) > 2:
            forms.add(lemma[:-1] + "ily")    # happy → happily
        elif lemma.endswith("le"):
            forms.add(lemma[:-1] + "y")      # simple → simply
        elif lemma.endswith("ic"):
            forms.add(lemma + "ally")        # historic → historically
        else:
            forms.add(lemma + "ly")          # dark → darkly
        # comparative / superlative (short adjectives)
        if len(lemma) <= 6 and not lemma.endswith(("ous", "ful", "ive", "ing", "ed", "al")):
            if lemma.endswith("e"):
                forms.add(lemma + "r")       # large → larger
                forms.add(lemma + "st")      # large → largest
            elif (lemma.endswith("y") and len(lemma) > 2
                  and lemma[-2] not in "aeiou"):
                forms.add(lemma[:-1] + "ier")   # happy → happier
                forms.add(lemma[:-1] + "iest")  # happy → happiest
            else:
                forms.add(lemma + "er")      # dark → darker
                forms.add(lemma + "est")     # dark → darkest
        # -ness noun derivation
        if lemma.endswith("y") and len(lemma) > 2:
            forms.add(lemma[:-1] + "iness")  # happy → happiness
        else:
            forms.add(lemma + "ness")        # dark → darkness
        # -ity noun derivation (for -al, -ive, -ous adjectives)
        if lemma.endswith("al"):
            forms.add(lemma[:-2] + "ality")  # spiritual → spirituality
        elif lemma.endswith("ive"):
            forms.add(lemma[:-3] + "ivity")  # creative → creativity
        elif lemma.endswith("ous"):
            forms.add(lemma[:-3] + "osity")  # generous → generosity
        return forms

    # ------------------------------------------------------------------
    # Irregular forms (static lookup)
    # ------------------------------------------------------------------
    _IRREGULAR_FORMS: Dict[str, List[str]] = {
        "child": ["children"], "woman": ["women"], "man": ["men"],
        "person": ["people", "persons"], "die": ["death", "dying", "dead"],
        "mouse": ["mice"], "foot": ["feet"], "tooth": ["teeth"],
        "goose": ["geese"], "ox": ["oxen"], "life": ["lives"],
        "knife": ["knives"], "wife": ["wives"], "half": ["halves"],
        "self": ["selves"], "leaf": ["leaves"], "thief": ["thieves"],
        "wolf": ["wolves"], "shelf": ["shelves"], "calf": ["calves"],
    }

    @staticmethod
    def _verb_derivations(lemma: str) -> set:
        """Generate agent nouns and nominalizations from a verb lemma.

        E.g. teach→teacher, create→creation/creator, develop→development
        """
        forms: set = set()
        if len(lemma) < 4:
            return forms
        # -er agent noun
        if lemma.endswith("e"):
            forms.add(lemma + "r")            # create → creater  (also creator below)
        elif (len(lemma) in (3, 4)
              and lemma[-1] not in "aeiouwy"
              and lemma[-2] in "aeiou"
              and (len(lemma) < 3 or lemma[-3] not in "aeiou")):
            forms.add(lemma + lemma[-1] + "er")  # run → runner
        else:
            forms.add(lemma + "er")           # teach → teacher
        # -or agent noun (for -ate verbs)
        if lemma.endswith("ate"):
            forms.add(lemma[:-3] + "ator")    # create → creator
        # -tion / -ation nominalization
        if lemma.endswith("e"):
            forms.add(lemma[:-1] + "ation")   # create → creation
            forms.add(lemma[:-1] + "ion")     # describe → description (approx)
        elif lemma.endswith("ify"):
            forms.add(lemma[:-1] + "ication") # classify → classification
        else:
            forms.add(lemma + "ation")        # inform → information (approx)
            forms.add(lemma + "ion")          # detect → detection (approx)
        # -ment nominalization
        forms.add(lemma + "ment")             # develop → development
        if lemma.endswith("e"):
            forms.add(lemma[:-1] + "ment")    # manage → management
        return forms

    @staticmethod
    def _noun_to_adj(lemma: str) -> set:
        """Generate adjective derivations from a noun lemma.

        E.g. history→historical, hope→hopeful/hopeless, mystery→mysterious
        """
        forms: set = set()
        if len(lemma) < 4:
            return forms
        # -ful / -less
        forms.add(lemma + "ful")              # hope → hopeful
        forms.add(lemma + "less")             # hope → hopeless
        if lemma.endswith("y") and len(lemma) > 2:
            forms.add(lemma[:-1] + "iful")    # beauty → beautiful
            forms.add(lemma[:-1] + "iless")   # (rare but consistent)
        # -al / -ial / -ical
        forms.add(lemma + "al")               # music → musical
        if lemma.endswith("y") and len(lemma) > 2:
            forms.add(lemma[:-1] + "ical")    # history → historical
        elif lemma.endswith("e"):
            forms.add(lemma[:-1] + "al")      # nature → natural
        # -ous / -ious / -eous
        forms.add(lemma + "ous")              # danger → dangerous
        if lemma.endswith("y") and len(lemma) > 2:
            forms.add(lemma[:-1] + "ious")    # mystery → mysterious
        elif lemma.endswith("e"):
            forms.add(lemma[:-1] + "ous")     # adventure → adventurous
        # -ive
        if lemma.endswith("tion"):
            forms.add(lemma[:-4] + "tive")    # fiction → fictive (approx)
        elif lemma.endswith("ion"):
            forms.add(lemma[:-3] + "ive")     # passion → passive (approx)
        return forms

    # Module-level cache: avoid re-loading GloVe for every PatternAbstractor instance
    _glove_cache: Dict[str, object] = {}

    def _load_glove(self):
        """Lazy-load GloVe word vectors for synonym expansion (cached across instances)."""
        if self._glove_model is not None:
            return
        if not self._glove_path:
            return
        # Check module-level cache first
        if self._glove_path in PatternAbstractor._glove_cache:
            self._glove_model = PatternAbstractor._glove_cache[self._glove_path]
            logger.info("GloVe reused from cache: %d words", len(self._glove_model))
            return
        try:
            from gensim.models import KeyedVectors
            logger.info("Loading GloVe vectors from %s ...", self._glove_path)
            self._glove_model = KeyedVectors.load_word2vec_format(
                self._glove_path, no_header=True)
            PatternAbstractor._glove_cache[self._glove_path] = self._glove_model
            logger.info("GloVe loaded: %d words", len(self._glove_model))
        except Exception as e:
            logger.warning("Failed to load GloVe: %s — synonym expansion disabled", e)
            self._glove_path = None  # don't retry

    def _glove_synonyms(self, word: str) -> List[str]:
        """Get GloVe nearest neighbors for a word (single-word, filtered)."""
        if self._glove_model is None:
            return []
        try:
            sims = self._glove_model.most_similar(word.lower(), topn=self._synonym_top_k * 2)
        except KeyError:
            return []
        result = []
        for w, score in sims:
            if score < self._synonym_min_sim:
                break
            # Filter: alphabetic, not too short, not stopword, not the original
            if (w == word or not w.isalpha() or len(w) < 3
                    or w in self._EXPANSION_STOPWORDS):
                continue
            result.append(w)
            if len(result) >= self._synonym_top_k:
                break
        return result

    @staticmethod
    def _wordnet_synonyms(word: str, max_syns: int = 4) -> List[str]:
        """Get synonyms from WordNet (first synset only, single-word lemmas)."""
        try:
            from nltk.corpus import wordnet as wn
            synsets = wn.synsets(word)
            if not synsets:
                return []
            syns: List[str] = []
            for lem in synsets[0].lemmas():
                name = lem.name().replace("_", " ")
                low = name.lower()
                if (low != word.lower()
                        and " " not in name
                        and len(name) > 2
                        and name.isalpha()
                        and low not in PatternAbstractor._EXPANSION_STOPWORDS):
                    syns.append(low)
            return syns[:max_syns]
        except Exception:
            return []

    def _morph_variants(self, lemma: str, pos: Optional[str] = None) -> set:
        """Generate morphological variants for a lemma given its POS."""
        variants = {lemma}
        if pos == "VERB":
            variants.update(self._verb_inflections(lemma))
            variants.update(self._verb_derivations(lemma))
        elif pos == "ADJ":
            variants.update(self._adj_derivations(lemma))
            variants.update(self._noun_plurals(lemma))
        elif pos == "NOUN" or pos is None:
            variants.update(self._noun_plurals(lemma))
            variants.update(self._noun_to_adj(lemma))
            variants.update(self._verb_inflections(lemma))
        elif pos == "ADV":
            if lemma.endswith("ly") and len(lemma) > 3:
                adj_base = lemma[:-2]
                variants.add(adj_base)
                variants.update(self._adj_derivations(adj_base))
        # Irregular forms (POS-independent)
        variants.update(self._IRREGULAR_FORMS.get(lemma, []))
        return variants

    def _lemma_to_morph_regex(self, token: str) -> str:
        """Expand a single-token alphabetic word to a regex alternation
        matching morphological variants + GloVe/WordNet synonyms.

        E.g. 'murder' (NOUN) → r'\\b(?:slaying|...|murdering|murdered|murders|murder)\\b'
        """
        if " " in token or not token.isalpha():
            return rf"\b{re.escape(token)}\b"

        # Lemmatize + POS tag via spaCy
        lemma = token
        pos = None
        if self._nlp is not None and self._nlp is not False:
            doc = self._nlp(token)
            if doc:
                lemma = doc[0].lemma_.lower()
                pos = doc[0].pos_

        # Core morphological variants of the anchor itself
        variants = {lemma, token}
        # If anchor is a stopword, skip derivation and synonym expansion
        if lemma in self._EXPANSION_STOPWORDS:
            alts = sorted(variants, key=lambda x: (-len(x), x))
            escaped = [re.escape(v) for v in alts]
            return rf"\b(?:{'|'.join(escaped)})\b"

        variants.update(self._morph_variants(lemma, pos))

        # Filter morphological variants to only corpus-attested forms
        if hasattr(self, '_corpus_vocab') and self._corpus_vocab:
            variants = {v for v in variants if v in self._corpus_vocab} | {lemma, token}

        # Synonym expansion (GloVe + WordNet): add synonym + plural only
        # (skip verb inflections for synonyms to avoid noise like "alwaysing")
        self._load_glove()
        synonyms = set(self._glove_synonyms(lemma))
        synonyms.update(self._wordnet_synonyms(lemma))
        for syn in synonyms:
            variants.add(syn)
            variants.update(self._noun_plurals(syn))

        # Cap total variants to avoid regex explosion
        if len(variants) > 30:
            # Keep original + lemma + morph variants, trim synonym variants
            core = {lemma, token}
            core.update(self._morph_variants(lemma, pos))
            extras = sorted(variants - core)[:30 - len(core)]
            variants = core | set(extras)

        # Sort longest-first for deterministic alternation
        alts = sorted(variants, key=lambda x: (-len(x), x))
        escaped = [re.escape(v) for v in alts]
        return rf"\b(?:{'|'.join(escaped)})\b"

    def _phrase_to_morph_regex(self, phrase: str) -> str:
        """Expand a multi-word phrase so each word matches morphological variants.

        E.g. 'young adult' → r'\\b(?:young)\\b\\s+\\b(?:adults|adult)\\b'
             'home cook'   → r'\\b(?:homes|home)\\b\\s+\\b(?:cooking|cooked|cooks|cook)\\b'
        """
        words = phrase.strip().split()
        if not words:
            return re.escape(phrase)
        parts = []
        for w in words:
            if w.isalpha():
                # Get morph regex for each word (returns \b(..)\b)
                part = self._lemma_to_morph_regex(w)
            else:
                part = rf"\b{re.escape(w)}\b"
            parts.append(part)
        return r"\s+".join(parts)

    # ------------------------------------------------------------------
    # Step 2: Pattern generation from anchors
    # ------------------------------------------------------------------

    def _step2_generate_patterns(
        self,
        anchors: List[str],
        cluster_texts: List[str],
        field: str = "cnt",
    ) -> Tuple[List["TextualPredicate"], Dict[int, List[int]]]:
        """
        Form textual predicates from the merged anchor list for one cluster.

        Emitted predicates (all with ``attr=field``):

        * :class:`MatchPredicate` — one per anchor term (single-anchor coverage).
        * :class:`CooccurPredicate` — one per unordered pair ``(a, b)``.
        * :class:`BeforePredicate` ``(a, b)`` / ``(b, a)`` — emitted when the
          ordering is attested in at least one cluster document.

        ``field`` selects the document attribute (``cnt``/``ttl``/``mtd``)
        that emitted predicates fire against; ``cluster_texts`` should be
        the corresponding text strings drawn from that attribute.

        The anchor list is capped so that the number of pairs stays within
        ``max_pairs_per_cluster``.

        Returns
        -------
        (predicates, fire_counts_map)
            fire_counts_map: ``{id(predicate): counts_per_doc}`` for
            MatchPredicates only (used by Step 3 to skip redundant regex scans).
        """
        if not anchors:
            return [], {}

        # Pre-compute corpus vocabulary for morphological expansion filtering
        self._corpus_vocab: set = set()
        for text in cluster_texts:
            self._corpus_vocab.update(re.findall(r'\b[a-z]+\b', text.lower()))

        # Semantic similarity mode: predicates compare embeddings instead of regex
        use_sim = (self.pattern_mode == "sim")
        # Use a lenient threshold for screening; Optuna will tighten it later
        default_th = 0.45 if use_sim else 0.85
        sim_threshold = self.sim_threshold if self.sim_threshold is not None else default_th

        # -- Single-anchor MatchPredicates + FreqPredicates ---------------
        match_preds: List[TextualPredicate] = []
        fire_counts_map: Dict[int, List[int]] = {}

        for anchor in anchors:
            if " " in anchor:
                raw = self._phrase_to_morph_regex(anchor)
            elif anchor.isalpha():
                raw = self._lemma_to_morph_regex(anchor)
            else:
                escaped = re.escape(anchor)
                raw = rf"\b{escaped}\b" if anchor.isalnum() else escaped
            a_pat = _Pattern(raw=raw, flags=re.IGNORECASE, sim_text=anchor)
            mp = MatchPredicate(attr=field, r=a_pat, sim=use_sim, threshold=sim_threshold)
            match_preds.append(mp)

            # One-pass regex scan: reused for freq_eta + Step 3 screening
            counts = [len(a_pat.findall(t)) for t in cluster_texts]
            fire_counts_map[id(mp)] = counts

            # Inline _freq_eta: emit FreqPredicate when median non-zero >= 2
            nonzero = [c for c in counts if c >= 2]
            if nonzero:
                eta = float(np.median(nonzero))
                if eta >= 2.0:
                    match_preds.append(
                        FreqPredicate(attr=field, r=a_pat, op=">=", eta=eta,
                                      sim=use_sim, threshold=sim_threshold)
                    )

        if len(anchors) < 2 or self.pattern_mode == "fast":
            # In fast mode, also add entity regex templates as MatchPredicates
            if self.pattern_mode == "fast":
                for _name, compiled_re in self._entity_regexes:
                    match_preds.append(
                        MatchPredicate(attr=field, r=_Pattern(raw=compiled_re.pattern), sim=use_sim)
                    )
            return match_preds, fire_counts_map

        # -- Pair-level predicates -----------------------------------------
        # Cap anchors to keep pairs ≤ max_pairs_per_cluster
        max_anchors = int(math.ceil(math.sqrt(2 * self.max_pairs_per_cluster))) + 1
        sampled = anchors[:max_anchors]

        def _anchor_raw(tok: str) -> str:
            if " " in tok:
                return self._phrase_to_morph_regex(tok)
            if tok.isalpha():
                return self._lemma_to_morph_regex(tok)
            esc = re.escape(tok)
            return rf"\b{esc}\b" if tok.isalnum() else esc

        pair_preds: List[TextualPredicate] = []
        seen: set = set()

        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                a, b = sampled[i], sampled[j]
                if a == b:
                    continue

                a_pat = _Pattern(raw=_anchor_raw(a), flags=re.IGNORECASE, sim_text=a)
                b_pat = _Pattern(raw=_anchor_raw(b), flags=re.IGNORECASE, sim_text=b)

                # -- CooccurPredicate ---------------------------------------
                key_co = ("cooccur", min(a, b), max(a, b))
                if key_co not in seen:
                    pair_preds.append(
                        CooccurPredicate(attr=field, r1=a_pat, r2=b_pat, sim=use_sim, threshold=sim_threshold)
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
                            BeforePredicate(attr=field, r1=a_pat, r2=b_pat, sim=use_sim, threshold=sim_threshold)
                        )
                        seen.add(key_ab)

                # -- BeforePredicate(b, a) ----------------------------------
                if before_ba > 0:
                    key_ba = ("before", b, a)
                    if key_ba not in seen:
                        pair_preds.append(
                            BeforePredicate(attr=field, r1=b_pat, r2=a_pat, sim=use_sim, threshold=sim_threshold)
                        )
                        seen.add(key_ba)

                if len(pair_preds) >= self.max_pairs_per_cluster * 3:
                    return match_preds + pair_preds, fire_counts_map

        return match_preds + pair_preds, fire_counts_map

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

    # def _step3_screen_patterns(
    #     self,
    #     candidates: List[TextualPredicate],
    #     texts: List[str],
    #     class_indices_per_doc: List[List[int]],
    # ) -> List[TextualPredicate]:
    #     """
    #     Filter candidate predicates by coverage and entropy discriminability.

    #     Coverage criterion:
    #         ``min_coverage ≤ |matching_docs| / |total_docs| ≤ max_coverage``

    #     Entropy criterion (LORIS paper, §3):
    #         ``H = -Σ P_i log(P_i) < max_entropy_threshold``
    #         where ``P_i`` is the share of class-``i`` documents among all
    #         documents that match the predicate.  Lower entropy ⟹ better
    #         discriminability.

    #     Parameters
    #     ----------
    #     candidates : List[TextualPredicate]
    #     texts : List[str]
    #         Raw content strings (``doc.cnt``).
    #     class_indices_per_doc : List[List[int]]
    #         Active class indices per document.

    #     Returns
    #     -------
    #     List[TextualPredicate]
    #     """
    #     n_total = len(texts)
    #     surviving: List[TextualPredicate] = []

    #     for pred in candidates:
    #         # Evaluate predicate on each document (wrap as Document for __call__)
    #         matching: List[int] = [
    #             i for i, text in enumerate(texts)
    #             if pred(Document(cnt=text))
    #         ]

    #         # -- Coverage check -------------------------------------------
    #         coverage = len(matching) / n_total
    #         if not (self.min_coverage <= coverage <= self.max_coverage):
    #             continue

    #         # -- Entropy check --------------------------------------------
    #         entropy = self._compute_entropy(matching, class_indices_per_doc)
    #         if entropy >= self.max_entropy_threshold:
    #             continue

    #         surviving.append(pred)

    #     return surviving

    def _step3_screen_patterns(
        self,
        candidates: List[TextualPredicate],
        texts: List[str],
        class_indices_per_doc: List[List[int]],
        docs: Optional[List[Document]] = None,
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
        docs : Optional[List[Document]]
            Full Document objects (with ``ttl``/``mtd`` populated). When
            provided, predicates with ``attr != "cnt"`` evaluate against
            the proper field. Falls back to ``Document(cnt=text)`` when None.

        Returns
        -------
        List[TextualPredicate]
        """
        n_total = len(texts)
        surviving: List[TextualPredicate] = []

        if docs is None:
            docs = [Document(cnt=text) for text in texts]

        from tqdm import tqdm

        # Diagnostic counters
        n_failed_coverage = 0
        n_failed_entropy = 0
        entropy_values = []
        # Per-field min_coverage: title fields are short, allow lower coverage
        ttl_min_coverage = max(self.min_coverage * 0.4, 0.001)
        mtd_min_coverage = max(self.min_coverage * 0.4, 0.001)

        for pred in tqdm(candidates, desc="Step 3 Screening", leave=False):
            matching: List[int] = [
                i for i, doc in enumerate(docs)
                if pred(doc)
            ]

            # -- Coverage check (per-field threshold) ---------------------
            coverage = len(matching) / n_total
            field = getattr(pred, "attr", "cnt")
            if field == "ttl":
                lo = ttl_min_coverage
            elif field == "mtd":
                lo = mtd_min_coverage
            else:
                lo = self.min_coverage
            if not (lo <= coverage <= self.max_coverage):
                n_failed_coverage += 1
                continue

            # -- Entropy check --------------------------------------------
            entropy = self._compute_entropy(matching, class_indices_per_doc)
            entropy_values.append(entropy)
            if entropy >= self.max_entropy_threshold:
                n_failed_entropy += 1
                continue

            surviving.append(pred)

        # Diagnostic summary
        n_passed_coverage = len(entropy_values)
        logger.info(
            "Step 3 | Screening breakdown: %d/%d failed coverage, "
            "%d passed coverage, %d failed entropy, %d survived.",
            n_failed_coverage, len(candidates),
            n_passed_coverage, n_failed_entropy, len(surviving),
        )
        if entropy_values:
            entropy_values.sort()
            logger.info(
                "Step 3 | Entropy stats of coverage-passing candidates: "
                "min=%.3f  median=%.3f  max=%.3f  threshold=%.3f",
                entropy_values[0],
                entropy_values[len(entropy_values) // 2],
                entropy_values[-1],
                self.max_entropy_threshold,
            )

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
