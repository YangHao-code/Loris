"""EmbeddingRegionModel — cosine similarity to per-label centroids.

Computes document embedding similarity to each label's centroid (mean of
positive-sample embeddings). Produces stable coverage across splits because
it operates in continuous semantic space rather than exact word matching.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from loris.models.base import BaseDocumentClassifier


class EmbeddingRegionModel(BaseDocumentClassifier):
    """Label-centroid cosine similarity as probability output.

    Parameters
    ----------
    num_labels : int
        Number of output labels.
    precomputed_embeddings : np.ndarray, optional
        If provided, skip encoding during fit() and use these directly.
        Shape (n_train, embed_dim).
    model_name : str
        Sentence-transformer model for encoding (used if no precomputed).
    """

    def __init__(
        self,
        num_labels: int,
        precomputed_embeddings: Optional[np.ndarray] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels, **kwargs)
        self._precomputed = precomputed_embeddings
        self._model_name = model_name
        self._centroids: Optional[np.ndarray] = None
        self._st_model = None

    def _get_encoder(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self._model_name)
        return self._st_model

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = self._get_encoder()
        emb = model.encode(texts, batch_size=256, show_progress_bar=False,
                           normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "EmbeddingRegionModel":
        y_train = np.asarray(y_train)

        if self._precomputed is not None and self._precomputed.shape[0] == len(X_train):
            embeddings = self._precomputed
        else:
            embeddings = self._encode(X_train)

        # L2-normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        # Compute per-label centroids from positive samples
        self._centroids = np.zeros(
            (self.num_labels, embeddings.shape[1]), dtype=np.float32
        )
        for i in range(self.num_labels):
            pos_mask = y_train[:, i] > 0
            if pos_mask.sum() > 0:
                centroid = embeddings[pos_mask].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-8:
                    centroid /= norm
                self._centroids[i] = centroid

        self.is_fitted = True
        return self

    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        self._check_fitted()
        embeddings = self._encode(X_test)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        # Cosine similarity → [0, 1] via (sim + 1) / 2
        sims = embeddings @ self._centroids.T
        proba = (sims + 1.0) / 2.0
        return proba.astype(np.float32)

    def predict(self, X_test: List[str]) -> np.ndarray:
        proba = self.predict_proba(X_test)
        return (proba >= 0.5).astype(np.int8)

    def predict_proba_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Fast path: compute proba from pre-computed embeddings directly."""
        self._check_fitted()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms
        sims = embeddings @ self._centroids.T
        return ((sims + 1.0) / 2.0).astype(np.float32)
