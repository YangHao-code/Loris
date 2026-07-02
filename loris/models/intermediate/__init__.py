"""Intermediate feature models for the Loris rule discovery pipeline.

These models produce stable structured signals (not direct label predictions)
that rules can compose via MLThresholdPredicate. They all implement
BaseDocumentClassifier so they integrate with the existing _PredictWrapper
and register_ml_model infrastructure.
"""

from loris.models.intermediate.embedding_region import EmbeddingRegionModel
from loris.models.intermediate.feature_density import FeatureDensityModel
from loris.models.intermediate.ensemble_vote import EnsembleVoteModel
from loris.models.intermediate.structural import StructuralFeatureModel

__all__ = [
    "EmbeddingRegionModel",
    "FeatureDensityModel",
    "EnsembleVoteModel",
    "StructuralFeatureModel",
]
