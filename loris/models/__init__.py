"""LORIS document classifiers.

Pluggable model pool used by the pipeline's model-selection step. Concrete
classifiers (TF-IDF, neural, pretrained-encoder, LoRA-SLM) implement
:class:`~loris.models.base.BaseDocumentClassifier`. Heavier classifiers that
pull optional deps (torch encoders, PEFT) are imported lazily by the pipeline,
so they are intentionally NOT eagerly re-exported here.

Migrated from the top-level ``models/`` package (Phase 6). The legacy package
re-exports these as shims.
"""

from __future__ import annotations

from loris.models.base import BaseDocumentClassifier
from loris.models.tfidf_classifier import TFIDFClassifier
from loris.models.neural_classifier import NeuralClassifier

__all__ = [
    "BaseDocumentClassifier",
    "TFIDFClassifier",
    "NeuralClassifier",
]
