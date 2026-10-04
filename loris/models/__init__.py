"""LORIS document classifiers.

TF-IDF and the common interface need only the base dependencies. Neural models
are loaded on demand so importing a lightweight classifier does not require
Torch. Explicitly importing ``NeuralClassifier`` still loads its dependencies.
"""

from __future__ import annotations

from loris.models.base import BaseDocumentClassifier
from loris.models.tfidf_classifier import TFIDFClassifier

__all__ = [
    "BaseDocumentClassifier",
    "TFIDFClassifier",
    "NeuralClassifier",
]


def __getattr__(name):
    if name == "NeuralClassifier":
        from loris.models.neural_classifier import NeuralClassifier

        globals()[name] = NeuralClassifier
        return NeuralClassifier
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
