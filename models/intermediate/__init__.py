"""Intermediate feature models for the Loris rule discovery pipeline.

These models produce stable structured signals (not direct label predictions)
that rules can compose via MLThresholdPredicate. They all implement
BaseDocumentClassifier so they integrate with the existing _PredictWrapper
and register_ml_model infrastructure.
"""

from models.intermediate.embedding_region import EmbeddingRegionModel
from models.intermediate.feature_density import FeatureDensityModel
from models.intermediate.ensemble_vote import EnsembleVoteModel
from models.intermediate.structural import StructuralFeatureModel

__all__ = [
    "EmbeddingRegionModel",
    "FeatureDensityModel",
    "EnsembleVoteModel",
    "StructuralFeatureModel",
]
