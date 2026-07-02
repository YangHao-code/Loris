"""Unit tests for Lever C — LLM closed-ontology membership attribute.

Guarantees: (1) an absent/empty cache returns ``None`` (golden-neutral — the
attribute is simply omitted); (2) a cache groups docs sharing ontology values;
(3) the doc hash is stable and matches the precompute key.
"""
import json
import os
import tempfile

import numpy as np

from loris.document import Document
from loris.rules.virtual_attributes import compute_llm_attributes, llm_doc_hash


def test_absent_cache_is_none():
    docs = [Document(cnt="a"), Document(cnt="b")]
    M, vocab = compute_llm_attributes(docs, "/does/not/exist.json")
    assert M is None and vocab == []
    M2, vocab2 = compute_llm_attributes(docs, None)
    assert M2 is None and vocab2 == []


def test_cache_groups_shared_values():
    docs = [Document(cnt="beamforming paper"),
            Document(cnt="graph coloring paper"),
            Document(cnt="mimo wireless paper")]
    cache = {
        llm_doc_hash(docs[0].cnt): ["wireless-comms"],
        llm_doc_hash(docs[1].cnt): ["graph-theory"],
        llm_doc_hash(docs[2].cnt): ["Wireless-Comms"],   # case-normalised on read
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    json.dump(cache, open(path, "w"))
    try:
        M, vocab = compute_llm_attributes(docs, path)
        assert M is not None and M.shape == (3, 2)
        col = vocab.index("wireless-comms")
        assert M[0, col] == 1 and M[2, col] == 1 and M[1, col] == 0
        # the two wireless docs share a value → the group fire mask links them
        co = (M @ M.T).toarray()
        assert co[0, 2] >= 1 and co[0, 1] == 0
    finally:
        os.remove(path)


def test_hash_is_stable_and_content_keyed():
    assert llm_doc_hash("hello") == llm_doc_hash("hello")
    assert llm_doc_hash("hello") != llm_doc_hash("world")
    assert isinstance(llm_doc_hash("x"), str) and len(llm_doc_hash("x")) == 16


def test_empty_cache_values_is_none():
    docs = [Document(cnt="a"), Document(cnt="b")]
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    json.dump({llm_doc_hash("a"): [], llm_doc_hash("b"): []}, open(path, "w"))
    try:
        M, vocab = compute_llm_attributes(docs, path)
        assert M is None and vocab == []   # no values anywhere → omit attribute
    finally:
        os.remove(path)
