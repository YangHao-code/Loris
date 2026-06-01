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
single-value helpers (compute_prediction_pattern_attributes, _cantor_pair) are
retained for backward-compatible imports but no longer wired into the pipeline.
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
    """Cantor pairing function for two non-negative integers."""
    return (a + b) * (a + b + 1) // 2 + b


def compute_prediction_pattern_attributes(
    ml_proba_cache: Dict[str, np.ndarray],
    label_names: List[str],
    top_k_list: List[int] = [1, 2],
) -> Dict[str, np.ndarray]:
    """Intermediate model predictions → group IDs.

    For each model:
      - top1: group_id = argmax label index (0..n_labels-1)
      - top2: group_id = cantor_pair(sorted top-2 indices)

    Parameters
    ----------
    ml_proba_cache : {model_name: (n_docs, n_labels) float proba array}
    label_names : label name list (for logging)
    top_k_list : which top-k patterns to compute

    Returns
    -------
    {attr_name: (n_docs,) int32}
    """
    attrs: Dict[str, np.ndarray] = {}
    n_labels = len(label_names)

    for model_name, proba in ml_proba_cache.items():
        if proba.ndim != 2 or proba.shape[1] != n_labels:
            logger.warning("  Skipping %s: shape %s doesn't match %d labels",
                          model_name, proba.shape, n_labels)
            continue

        short_name = model_name.replace("loris_", "")

        if 1 in top_k_list:
            top1 = np.argmax(proba, axis=1).astype(np.int32)
            attr_name = f"{short_name}_top1"
            attrs[attr_name] = top1
            n_groups = len(np.unique(top1))
            logger.info("  %s: %d unique groups", attr_name, n_groups)

        if 2 in top_k_list:
            top2_idx = np.argsort(proba, axis=1)[:, -2:]
            n_docs = len(proba)
            top2_ids = np.empty(n_docs, dtype=np.int32)
            for i in range(n_docs):
                a, b = sorted([int(top2_idx[i, 0]), int(top2_idx[i, 1])])
                top2_ids[i] = _cantor_pair(a, b)
            attr_name = f"{short_name}_top2"
            attrs[attr_name] = top2_ids
            n_groups = len(np.unique(top2_ids))
            logger.info("  %s: %d unique groups", attr_name, n_groups)

    return attrs


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
