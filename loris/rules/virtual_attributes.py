"""
Virtual attribute computation for group-based comparison predicates (x.A=y.A).

Generates **multi-value sparse membership matrices** from:
  1. K-Means clustering on sentence embeddings (fit on train, predict on target)
     — single-valued, represented as a one-hot csr for uniformity.
  2. Real document attributes via a single spaCy pass over doc.cnt:
       - NER entity types (ner_ORG, ner_PERSON, ner_GPE, ner_DATE, ...)
       - syntactic features (syn_root, syn_nsubj, syn_posgram, syn_nounchunk)
       - regex entities (regex_id, regex_money, ... from regex_patterns.yaml)

Each attribute is a scipy.sparse.csr_matrix (n_docs × n_values), dtype int8,
entry[i, v] = 1 iff doc i possesses value v of that attribute type. The
comparison predicate x.A=y.A holds iff the two docs' value-sets for attr A
intersect (>=1 shared value). A doc with no values for an attribute is an
all-zero row and shares nothing (the multi-value replacement for the old -1
"excluded from propagation" sentinel).

NOTE (B-3 paper alignment): the old single-value group-id arrays + Cantor
prediction-pattern signatures were fabricated attributes; replaced here by real
NER/syntactic/regex attributes. Cluster_k is kept (as one-hot csr). The legacy
single-value helpers (compute_prediction_pattern_attributes, _cantor_pair) have
been REMOVED (dead + non-paper; the latter also held a latent NameError).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def compute_cluster_attributes(
    train_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    k_list: List[int] = [50, 100, 200, 500],
    kmeans_models: Optional[Dict[int, object]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[int, object]]:
    """K-Means fit on train, predict on target.

    Parameters
    ----------
    train_embeddings : (n_train, dim) float32 — used for fitting only
    target_embeddings : (n_target, dim) float32 — predictions computed here
    k_list : cluster counts to try
    kmeans_models : if provided, skip fitting and reuse these models

    Returns
    -------
    (attrs_dict, models) where attrs_dict maps "cluster_{k}" -> (n_target,) int32
    """
    from sklearn.cluster import MiniBatchKMeans

    n_train = len(train_embeddings)
    if kmeans_models is None:
        kmeans_models = {}
        for k in k_list:
            # KMeans requires n_clusters <= n_samples. On small corpora a
            # configured k can exceed the training-set size; clamp it so the
            # fit never raises (B0). On full-size data (k < n_train) this is a
            # no-op, so cluster assignments are unchanged.
            k_eff = min(k, n_train)
            if k_eff < k:
                logger.warning(
                    "  KMeans k=%d > n_train=%d; clamping to %d", k, n_train, k_eff)
            km = MiniBatchKMeans(n_clusters=k_eff, random_state=42, batch_size=1024)
            km.fit(train_embeddings)
            kmeans_models[k] = km
            logger.info("  KMeans k=%d fit on %d train docs", k_eff, n_train)

    attrs: Dict[str, np.ndarray] = {}
    for k, km in kmeans_models.items():
        labels = km.predict(target_embeddings).astype(np.int32)
        attrs[f"cluster_{k}"] = labels
        logger.info("  cluster_%d: %d unique groups on %d target docs",
                    k, len(np.unique(labels)), len(target_embeddings))

    return attrs, kmeans_models


def _labels_to_onehot_csr(labels: np.ndarray, n_values: int) -> sp.csr_matrix:
    """Convert a (n_docs,) int group-id array to a one-hot (n_docs × n_values) csr.

    A label of -1 (degenerate / excluded) becomes an all-zero row, which shares
    no value with any other doc — the multi-value equivalent of the old
    "excluded from propagation" sentinel. n_values fixes the column count so the
    column index == group id is stable across train/val/test.
    """
    labels = np.asarray(labels)
    valid = labels >= 0
    rows = np.nonzero(valid)[0]
    cols = labels[valid].astype(np.int64)
    data = np.ones(len(rows), dtype=np.int8)
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(len(labels), n_values), dtype=np.int8
    )


def _cluster_attrs_to_csr(
    cluster_attrs: Dict[str, np.ndarray],
    kmeans_models: Dict[int, object],
) -> Dict[str, sp.csr_matrix]:
    """Wrap single-value cluster_k group-id arrays as one-hot csr matrices.

    n_values per cluster_k = the model's actual cluster count (n_clusters), which
    is fixed across BO/select/test because the same fitted KMeans is reused.
    """
    out: Dict[str, sp.csr_matrix] = {}
    for k, km in kmeans_models.items():
        name = f"cluster_{k}"
        if name not in cluster_attrs:
            continue
        n_clusters = int(getattr(km, "n_clusters", 0)) or (
            int(cluster_attrs[name].max()) + 1 if cluster_attrs[name].size else 0
        )
        out[name] = _labels_to_onehot_csr(cluster_attrs[name], n_clusters)
    return out

# NOTE (paper-vs-code audit, 2026-06-04): the fabricated "prediction-pattern"
# join attributes (compute_prediction_pattern_attributes + the Cantor-pairing
# helper) were REMOVED here — they have no basis in the paper (the comparison
# predicate joins on real attributes x.A, A∈{mtd,ttl,cnt}, not Cantor-paired
# top-2 prediction signatures) and were dead in the active pipeline. Their
# orphaned _cantor_pair body (signature already deleted) also held a latent
# NameError. Removing both fixes the bug and drops the non-paper dead code.


# ── Real text attributes via a single spaCy pass ────────────────────────────

_SPACY_NLP_CACHE: Dict[str, object] = {}
_NOUNCHUNK_KEEP_POS = {"NOUN", "PROPN"}
_NER_FAMILY, _SYN_FAMILY, _REGEX_FAMILY = "ner", "syn", "regex"


def _get_spacy_nlp(spacy_model: str):
    """Load (and cache) a spaCy model with NER ENABLED. Raise loudly on failure.

    Unlike pattern_abstractor (which disables NER and swallows OSError), text
    attributes need NER and treat the model as a hard dependency — a missing
    model must fail loud, never silently degrade to cluster-only attributes.
    """
    nlp = _SPACY_NLP_CACHE.get(spacy_model)
    if nlp is None:
        import spacy  # noqa: PLC0415 — optional heavy dep, imported on use
        nlp = spacy.load(spacy_model)  # NER enabled; raises OSError if absent
        _SPACY_NLP_CACHE[spacy_model] = nlp
    return nlp


def _clean_noun_chunk(chunk) -> str:
    """Strip a noun chunk to its content nouns (NOUN/PROPN, non-stop, non-punct).

    "a new machine learning algorithm" -> "machine learning algorithm".
    Prevents adjective+det noise from filling the truncated vocab window.
    """
    toks = [
        t.lemma_.lower()
        for t in chunk
        if t.pos_ in _NOUNCHUNK_KEEP_POS
        and not t.is_stop
        and not t.is_punct
        and t.lemma_.strip()
    ]
    return " ".join(toks)


def _extract_doc_values(spacy_doc, raw_text: str, regex_patterns, families) -> Dict[str, set]:
    """Collect {attr_key: set(values)} for one document. Deterministic."""
    vals: Dict[str, set] = {}

    def _add(key: str, value: str) -> None:
        value = value.strip()
        if value:
            vals.setdefault(key, set()).add(value)

    if _NER_FAMILY in families:
        for ent in spacy_doc.ents:
            _add(f"ner_{ent.label_}", ent.text.lower())

    if _SYN_FAMILY in families:
        # main-sentence root verb(s) + subjects (nsubj)
        for tok in spacy_doc:
            if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
                _add("syn_root", tok.lemma_.lower())
            if tok.dep_ == "nsubj":
                _add("syn_nsubj", tok.lemma_.lower())
        for chunk in spacy_doc.noun_chunks:
            cleaned = _clean_noun_chunk(chunk)
            if cleaned:
                _add("syn_nounchunk", cleaned)
        # single-value POS trigram signature of the leading content tokens
        lead = [t.pos_ for t in spacy_doc if not t.is_space and not t.is_punct][:3]
        if len(lead) == 3:
            _add("syn_posgram", "_".join(lead))

    if _REGEX_FAMILY in families and regex_patterns:
        for name, pat in regex_patterns:
            for m in pat.findall(raw_text):
                _add(f"regex_{name}", m if isinstance(m, str) else str(m))

    return vals


def compute_text_attributes(
    docs: Sequence,
    vocabs: Optional[Dict[str, List[str]]] = None,
    regex_config_path: Optional[str] = None,
    spacy_model: str = "en_core_web_sm",
    families: Tuple[str, ...] = ("ner", "syn", "regex"),
    max_vocab_per_family: int = 2000,
) -> Tuple[Dict[str, sp.csr_matrix], Dict[str, List[str]]]:
    """Single spaCy pass over doc.cnt → multi-value membership matrices.

    fit mode  (vocabs is None): build per-attr value vocabularies from `docs`,
                                ordered by (-document_frequency, value) and
                                truncated to max_vocab_per_family.
    transform mode (vocabs given): reuse them; out-of-vocab values are dropped
                                (mirrors KMeans predict-on-target).

    Returns (attrs_csr, vocabs) where attrs_csr maps attr_key -> csr (n_docs ×
    n_values) int8, and vocabs maps attr_key -> ordered value list.

    CSR is built once per attribute from COO triplets (never incrementally
    stacked in the loop — that is an O(N^2) fragmentation trap).
    """
    n_docs = len(docs)
    fit_mode = vocabs is None

    regex_patterns = None
    if _REGEX_FAMILY in families:
        from loris.patterns import load_regex_config  # noqa: PLC0415
        regex_patterns = load_regex_config(regex_config_path)

    nlp = _get_spacy_nlp(spacy_model)
    texts = [getattr(d, "cnt", "") or "" for d in docs]

    # Pass 1: collect per-doc value sets (single spaCy pipe pass).
    per_doc_vals: List[Dict[str, set]] = []
    df_counter: Dict[str, Dict[str, int]] = {}
    for i, sdoc in enumerate(nlp.pipe(texts, batch_size=256)):
        vals = _extract_doc_values(sdoc, texts[i], regex_patterns, families)
        per_doc_vals.append(vals)
        if fit_mode:
            for key, vset in vals.items():
                bucket = df_counter.setdefault(key, {})
                for v in vset:
                    bucket[v] = bucket.get(v, 0) + 1

    # Build / reuse vocabularies (deterministic ordering).
    if fit_mode:
        vocabs = {}
        for key, bucket in df_counter.items():
            ordered = sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))
            vocabs[key] = [v for v, _ in ordered[:max_vocab_per_family]]

    vocab_index: Dict[str, Dict[str, int]] = {
        key: {v: j for j, v in enumerate(vlist)} for key, vlist in vocabs.items()
    }

    # Pass 2: assemble COO triplets per attribute, build csr once each.
    rows: Dict[str, List[int]] = {key: [] for key in vocabs}
    cols: Dict[str, List[int]] = {key: [] for key in vocabs}
    for i, vals in enumerate(per_doc_vals):
        for key, vset in vals.items():
            idx_map = vocab_index.get(key)
            if not idx_map:
                continue
            for v in vset:
                col = idx_map.get(v)
                if col is not None:
                    rows[key].append(i)
                    cols[key].append(col)

    attrs: Dict[str, sp.csr_matrix] = {}
    for key, vlist in vocabs.items():
        n_vals = len(vlist)
        if n_vals == 0:
            continue
        r = np.asarray(rows[key], dtype=np.int64)
        c = np.asarray(cols[key], dtype=np.int64)
        data = np.ones(len(r), dtype=np.int8)
        M = sp.csr_matrix((data, (r, c)), shape=(n_docs, n_vals), dtype=np.int8)
        # COO sums duplicates (same doc, same value twice) → force 0/1.
        if M.nnz:
            M.data[:] = 1
        attrs[key] = M

    logger.info("compute_text_attributes (%s): %d docs, %d attr types",
                "fit" if fit_mode else "transform", n_docs, len(attrs))
    return attrs, vocabs


def compute_phrase_attributes(
    train_docs: Sequence,
    train_labels: np.ndarray,
    target_docs: Sequence,
    label_names: List[str],
    phrase_vocab: Optional[List[str]] = None,
    ngram_range: Tuple[int, int] = (1, 2),
    min_support: int = 5,
    min_lift: float = 2.0,
    min_co_count: int = 4,
    max_phrases: int = 400,
) -> Tuple[Optional[sp.csr_matrix], List[str]]:
    """Discriminative-phrase membership attribute for honest x.A=y.A propagation.

    OPT-5 (AAPD-propagation-findings): cluster_k / NER / frequency-ranked values
    carry NO label signal, so under honest (predicted) neighbour labels group
    discovery finds 0 rules. This builds a membership matrix whose VALUES are
    n-grams that are *label-discriminative on TRAIN* (lift = P(label|phrase)/P(label)
    above ``min_lift`` with support), so two docs sharing such a phrase form a
    label-coherent propagation group and ``x.phrase = y.phrase`` can fire honestly.

    Mined on TRAIN only (no leakage); transformed onto ``target_docs``. The mining
    is deterministic, so re-calling on the same train set at BO / select / test
    yields the same vocabulary (column semantics stay consistent).

    Returns ``(csr (n_target × n_phrases) int8 | None, phrase_vocab)``. None when
    no phrase clears the gate (caller simply omits the attribute).
    """
    from sklearn.feature_extraction.text import CountVectorizer  # noqa: PLC0415

    train_texts = [getattr(d, "cnt", "") or "" for d in train_docs]
    Y = np.asarray(train_labels)
    if Y.ndim == 1:                      # single-label → one-hot
        Y = np.eye(len(label_names), dtype=np.int8)[Y]
    n_train = len(train_texts)

    if phrase_vocab is None:
        # Fit a binary uni/bi-gram vectoriser on train and rank terms by max
        # per-label lift (deterministic; ties broken by -df then term).
        vec = CountVectorizer(ngram_range=ngram_range, binary=True,
                              min_df=min_support, max_features=50000)
        try:
            Xtr = vec.fit_transform(train_texts).astype(np.float32)
        except ValueError:               # empty vocabulary on tiny corpora
            return None, []
        terms = vec.get_feature_names_out()
        df = np.asarray(Xtr.sum(axis=0)).ravel()              # (V,)
        co = np.asarray(Xtr.T.dot(Y.astype(np.float32)))      # (V, L) term∧label
        label_freq = Y.sum(axis=0).astype(np.float64)         # (L,)
        p_label = np.maximum(label_freq / max(n_train, 1), 1e-9)
        p_l_given_t = co / np.maximum(df[:, None], 1.0)       # (V, L)
        lift = p_l_given_t / p_label[None, :]                 # (V, L)
        best_l = lift.argmax(axis=1)
        best_lift = lift[np.arange(len(terms)), best_l]
        best_co = co[np.arange(len(terms)), best_l]
        keep = (best_lift >= min_lift) & (df >= min_support) & (best_co >= min_co_count)
        cand = np.nonzero(keep)[0]
        # rank by lift, then coverage; deterministic
        order = sorted(cand, key=lambda t: (-best_lift[t], -df[t], terms[t]))
        phrase_vocab = [str(terms[t]) for t in order[:max_phrases]]
        if not phrase_vocab:
            return None, []

    # Transform target docs onto the fixed phrase vocabulary (binary presence).
    target_texts = [getattr(d, "cnt", "") or "" for d in target_docs]
    tvec = CountVectorizer(ngram_range=ngram_range, binary=True,
                           vocabulary=phrase_vocab)
    M = tvec.transform(target_texts).astype(np.int8).tocsr()
    if M.nnz:
        M.data[:] = 1
    return M, list(phrase_vocab)


def llm_doc_hash(text: str) -> str:
    """Stable content hash used to key the offline LLM attribute/judgment cache.
    The precompute script and the pipeline MUST agree on this so a doc resolves
    to the same cached values at BO / select / test (fit-once-and-reuse)."""
    import hashlib  # noqa: PLC0415
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def compute_llm_attributes(
    docs: Sequence,
    cache_path: Optional[str],
    vocab: Optional[List[str]] = None,
    attr_name: str = "llm",
) -> Tuple[Optional[sp.csr_matrix], List[str]]:
    """Closed-ontology LLM membership attribute for ``x.A=y.A`` (Lever C).

    Reads a precomputed cache ``{doc_hash: [value, ...]}`` produced OFFLINE by a
    local LLM (see ``precompute_llm_attributes.py``) over a **closed** ontology,
    and builds a (n_docs × n_values) 0/1 csr — the same shape as every other
    virtual attribute, so it auto-enrols into group/equal/multi-literal discovery.

    The LLM is non-deterministic, so its output is captured ONCE in the cache and
    reused verbatim, keyed by ``llm_doc_hash(doc.cnt)``. ``vocab=None`` fits the
    value vocabulary (deterministic, ``(-df, value)``); pass the returned vocab at
    select/test to keep column semantics identical (the contract group rules need).

    Returns ``(csr | None, vocab)``. ``None`` (caller omits the attribute) when the
    cache is absent/empty — so a missing cache is golden-neutral, never a crash.
    """
    import json, os  # noqa: PLC0415
    if not cache_path or not os.path.exists(cache_path):
        return None, []
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except Exception as e:                                   # never break discovery
        logger.warning("compute_llm_attributes: cannot read cache %s: %s", cache_path, e)
        return None, []

    texts = [getattr(d, "cnt", "") or "" for d in docs]
    per_doc = [
        [str(v).strip().lower() for v in (cache.get(llm_doc_hash(t)) or []) if str(v).strip()]
        for t in texts
    ]

    if vocab is None:                                        # fit value vocabulary
        df: Dict[str, int] = {}
        for vals in per_doc:
            for v in set(vals):
                df[v] = df.get(v, 0) + 1
        vocab = [v for v, _ in sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))]
        if not vocab:
            return None, []

    idx = {v: j for j, v in enumerate(vocab)}
    rows: List[int] = []
    cols: List[int] = []
    for i, vals in enumerate(per_doc):
        for v in set(vals):
            j = idx.get(v)
            if j is not None:
                rows.append(i)
                cols.append(j)
    n_docs, n_vals = len(docs), len(vocab)
    if n_vals == 0:
        return None, []
    M = sp.csr_matrix(
        (np.ones(len(rows), np.int8),
         (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
        shape=(n_docs, n_vals), dtype=np.int8,
    )
    if M.nnz:
        M.data[:] = 1
    logger.info("compute_llm_attributes (%s): %d docs, %d closed-ontology values",
                os.path.basename(cache_path), n_docs, n_vals)
    return M, list(vocab)


def compute_all_virtual_attributes(
    train_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    train_docs: Sequence,
    target_docs: Sequence,
    label_names: List[str],
    k_list: List[int] = [50, 100, 200, 500],
    kmeans_models: Optional[Dict[int, object]] = None,
    text_vocabs: Optional[Dict[str, List[str]]] = None,
    regex_config_path: Optional[str] = None,
    spacy_model: str = "en_core_web_sm",
    text_families: Tuple[str, ...] = ("ner", "syn", "regex"),
    enable_text_attrs: bool = True,
    enable_cluster_attrs: bool = True,
    text_max_vocab: int = 2000,
    enable_llm_attrs: bool = False,
    llm_cache_path: Optional[str] = None,
) -> Tuple[Dict[str, sp.csr_matrix], Dict[int, object], Dict[str, List[str]]]:
    """Compute all virtual attributes as multi-value csr membership matrices.

    cluster_k (one-hot csr) + real text attributes (NER/syntactic/regex). The
    fabricated Cantor prediction-pattern attributes are no longer produced.

    Text-attribute value vocabularies are fit on `train_docs` (first call,
    text_vocabs=None) and reused on `target_docs` so attribute column semantics
    are identical across BO / select / test — mirroring KMeans fit/predict.

    Returns
    -------
    (all_attrs, kmeans_models, text_vocabs)
        all_attrs : {attr_name: csr (n_target × n_values) int8}
        text_vocabs : fit on first call, pass back in for stable transforms.
    """
    all_attrs: Dict[str, sp.csr_matrix] = {}

    if enable_cluster_attrs:
        cluster_attrs, kmeans_models = compute_cluster_attributes(
            train_embeddings, target_embeddings, k_list, kmeans_models
        )
        all_attrs.update(_cluster_attrs_to_csr(cluster_attrs, kmeans_models))

    if enable_text_attrs:
        # Fit vocab on train (once) if not supplied, then transform target.
        if text_vocabs is None:
            _, text_vocabs = compute_text_attributes(
                train_docs, vocabs=None, regex_config_path=regex_config_path,
                spacy_model=spacy_model, families=text_families,
                max_vocab_per_family=text_max_vocab,
            )
        text_attrs, text_vocabs = compute_text_attributes(
            target_docs, vocabs=text_vocabs, regex_config_path=regex_config_path,
            spacy_model=spacy_model, families=text_families,
            max_vocab_per_family=text_max_vocab,
        )
        all_attrs.update(text_attrs)
        if not text_attrs:
            raise RuntimeError(
                "enable_text_attrs=True but compute_text_attributes produced "
                "zero attributes — spaCy pass likely broken (fail loud rather "
                "than silently degrade to cluster-only)."
            )

    # Lever C: closed-ontology LLM membership attribute (offline cache). The
    # group rule references it by NAME and the fire-mask is computed intra-matrix
    # per call, so no cross-call column threading is needed — a fresh fit from the
    # SAME cache is consistent. Default OFF / no cache ⇒ no attribute (golden-neutral).
    if enable_llm_attrs and llm_cache_path:
        _llm_M, _ = compute_llm_attributes(target_docs, llm_cache_path)
        if _llm_M is not None and _llm_M.shape[1] > 0:
            all_attrs["llm"] = _llm_M

    # Fail-loud structural guard (catches silent shape/type corruption).
    n_target = len(target_docs)
    for name, M in all_attrs.items():
        assert sp.issparse(M) and M.shape[0] == n_target, (
            f"virtual attr {name!r} is not a (n_docs={n_target} × n_values) csr: "
            f"{type(M)} shape={getattr(M, 'shape', None)}"
        )
        logger.info("  attr %s: %d values, nnz=%d", name, M.shape[1], M.nnz)

    logger.info("Total virtual attributes: %d", len(all_attrs))
    return all_attrs, kmeans_models, text_vocabs


def filter_degenerate_groups(
    virtual_attrs: Dict[str, sp.csr_matrix],
    min_group_size: int = 3,
    max_group_fraction: float = 0.33,
) -> Dict[str, sp.csr_matrix]:
    """Drop degenerate attribute VALUES (columns) from each membership matrix.

    A value column is degenerate if its document-frequency is:
      - < min_group_size (too rare for meaningful propagation), or
      - > max_group_fraction * n_docs (a "super-value" like DATE=2014 that would
        merge hundreds of unrelated docs → catastrophic false positives).

    Operates per value-column (not per group). Matrix WIDTH is preserved (kept
    columns are zeroed in place) so column index == value id stays stable and
    consistent with the value vocabulary. Docs left with no surviving value
    become all-zero rows and are naturally excluded from propagation — the
    multi-value equivalent of the old -1 sentinel.

    Parameters
    ----------
    virtual_attrs : {attr_name: csr (n_docs × n_values) int8}
    min_group_size : minimum document-frequency per value
    max_group_fraction : maximum fraction of docs per value

    Returns
    -------
    Filtered copy of virtual_attrs (degenerate value columns zeroed out).
    """
    filtered: Dict[str, sp.csr_matrix] = {}
    for attr_name, M in virtual_attrs.items():
        M = M.tocsr()
        n_docs = M.shape[0]
        max_size = int(max_group_fraction * n_docs)
        df = np.asarray(M.sum(axis=0)).ravel()
        keep = (df >= min_group_size) & (df <= max_size)
        n_drop = int((~keep).sum())

        if n_drop:
            # Zero dropped columns while preserving width: scale columns by keep.
            diag = sp.diags(keep.astype(np.int8), format="csr")
            M = (M @ diag).tocsr()
            M.eliminate_zeros()

        n_remaining = int(keep.sum())
        if n_drop:
            logger.info("  %s: dropped %d degenerate values, %d remaining",
                        attr_name, n_drop, n_remaining)
        filtered[attr_name] = M

    return filtered
