"""LORIS data subpackage.

Dataset preparation, configuration, and loading:
  * ``prepare``  — domain stopwords, ``prepare_<dataset>`` downloaders,
                   :class:`DatasetConfig`, and :data:`DATASET_REGISTRY`
  * ``config``   — :class:`HParams` pipeline hyperparameters
  * ``loader``   — ``configure_logging`` and ``load_data``

Migrated from ``run_loris_multi_pipeline.py`` SECTION 1 (Phase 5). This is the
single authoritative home for the data layer; the legacy pipeline modules
re-export these names as shims so both old pipelines keep importing them.
"""

from __future__ import annotations

from loris.data.config import HParams
from loris.data.prepare import (
    DatasetConfig,
    DATASET_REGISTRY,
    AAPD_STOP_WORDS,
    EURLEX_STOP_WORDS,
    RCV1_STOP_WORDS,
    BGC_STOP_WORDS,
    REUTERS21578_STOP_WORDS,
    PUBMED_STOP_WORDS,
    HUPD_STOP_WORDS,
    GOODREADS_STOP_WORDS,
    prepare_aapd,
    prepare_eurlex,
    prepare_rcv1,
    prepare_bgc,
    prepare_reuters21578,
    prepare_pubmed,
    prepare_hupd,
    prepare_goodreads,
)
from loris.data.loader import configure_logging, load_data

__all__ = [
    "HParams",
    "DatasetConfig",
    "DATASET_REGISTRY",
    "AAPD_STOP_WORDS",
    "EURLEX_STOP_WORDS",
    "RCV1_STOP_WORDS",
    "BGC_STOP_WORDS",
    "REUTERS21578_STOP_WORDS",
    "PUBMED_STOP_WORDS",
    "HUPD_STOP_WORDS",
    "GOODREADS_STOP_WORDS",
    "prepare_aapd",
    "prepare_eurlex",
    "prepare_rcv1",
    "prepare_bgc",
    "prepare_reuters21578",
    "prepare_pubmed",
    "prepare_hupd",
    "prepare_goodreads",
    "configure_logging",
    "load_data",
]
