"""
models/neural_classifier.py
---------------------------
PyTorch-based multi-label text classifiers: TextCNN and BiLSTM.

Both architectures share the same training loop, vocabulary pipeline,
and output head (BCEWithLogitsLoss). Select the architecture via the
``variant`` parameter: ``"textcnn"`` (default) or ``"bilstm"``.
Select the output head via ``head_type``: ``"linear"`` (default) or
``"cosine"``.

Design notes
~~~~~~~~~~~~
* Vocabulary is built **only from training data** to prevent leakage.
* The best checkpoint (highest val Micro-F1) is restored after training.
* Prediction uses ``torch.no_grad()`` and processes data in batches to
  avoid GPU OOM on large test sets.
* Both models are GPU/CPU adaptive via ``get_device()``.
* ``_CosineHead`` computes cosine similarity between text features and
  learnable per-label prototype vectors. The resulting scores are in
  [-1, 1] and serve directly as logits for BCEWithLogitsLoss.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from loris.models.base import BaseDocumentClassifier
from loris.models.utils import (
    build_vocab,
    get_device,
    multilabel_dataloader,
    texts_to_ids,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output head modules
# ---------------------------------------------------------------------------

class _CosineHead(nn.Module):
    """
    Learnable label-prototype cosine-similarity head.

    Computes the cosine similarity between the input feature vector and
    ``num_labels`` learnable prototype vectors. The resulting scores lie in
    ``[-1, 1]`` and are used directly as logits for ``BCEWithLogitsLoss``.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input feature vector.
    num_labels : int
        Number of output labels (= number of prototype vectors).
    """

    def __init__(self, in_features: int, num_labels: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_labels, in_features))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Cosine similarity scores, shape ``(batch, num_labels)``.
        """
        x_norm = F.normalize(x, p=2, dim=-1)             # (batch, F)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)  # (L, F)
        return x_norm @ p_norm.T                           # (batch, L)


def _make_head(head_type: str, in_features: int, num_labels: int) -> nn.Module:
    """Factory: return a Linear or CosineHead based on ``head_type``."""
    if head_type == "cosine":
        return _CosineHead(in_features, num_labels)
    return nn.Linear(in_features, num_labels)


# ---------------------------------------------------------------------------
# Internal model modules
# ---------------------------------------------------------------------------

class _TextCNN(nn.Module):
    """
    Text CNN for sentence classification.

    Architecture
    ~~~~~~~~~~~~
    Embedding → parallel Conv1d branches (one per filter size)
    → ReLU + AdaptiveMaxPool1d(1) → Concat → Dropout → Head

    Parameters
    ----------
    vocab_size : int
    embed_dim : int
    num_labels : int
    num_filters : int
        Number of feature maps per filter size.
    filter_sizes : Sequence[int]
        List of convolutional kernel sizes.
    dropout : float
    head_type : str
        ``"linear"`` (default) or ``"cosine"``.
    pad_idx : int
        Embedding padding index (zero-initialized, no gradient).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_labels: int,
        num_filters: int = 128,
        filter_sizes: Sequence[int] = (3, 4, 5),
        dropout: float = 0.3,
        head_type: str = "linear",
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        feature_dim = num_filters * len(filter_sizes)
        self.head = _make_head(head_type, feature_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len)``, dtype ``long``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_labels)``.
        """
        # (batch, seq_len, embed_dim) → (batch, embed_dim, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = []
        for conv in self.convs:
            # (batch, num_filters, seq_len - kernel + 1)
            h = torch.relu(conv(emb))
            # (batch, num_filters, 1) → (batch, num_filters)
            h = F.adaptive_max_pool1d(h, 1).squeeze(2)
            pooled.append(h)
        # (batch, num_filters * len(filter_sizes))
        cat = torch.cat(pooled, dim=1)
        return self.head(self.dropout(cat))


class _BiLSTM(nn.Module):
    """
    Bidirectional LSTM for sentence classification.

    Architecture
    ~~~~~~~~~~~~
    Embedding → BiLSTM (2 layers) → last-step hidden state
    (forward ‖ backward) → Dropout → Head

    Parameters
    ----------
    vocab_size : int
    embed_dim : int
    hidden_dim : int
        Total hidden size; each direction uses ``hidden_dim // 2`` units.
    num_labels : int
    num_layers : int
    dropout : float
    head_type : str
        ``"linear"`` (default) or ``"cosine"``.
    pad_idx : int
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        head_type: str = "linear",
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = _make_head(head_type, hidden_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len)``, dtype ``long``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_labels)``.
        """
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(emb)
        # h_n: (num_layers * 2, batch, hidden_dim // 2)
        # Take last layer's forward and backward hidden states
        forward_h = h_n[-2]   # (batch, hidden_dim // 2)
        backward_h = h_n[-1]  # (batch, hidden_dim // 2)
        h = torch.cat([forward_h, backward_h], dim=1)  # (batch, hidden_dim)
        return self.head(self.dropout(h))


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

class NeuralClassifier(BaseDocumentClassifier):
    """
    PyTorch-based multi-label classifier (TextCNN or BiLSTM).

    Parameters
    ----------
    num_labels : int
        Number of output labels.
    variant : str
        Architecture to use: ``"textcnn"`` (default) or ``"bilstm"``.
    head_type : str
        Output head type: ``"linear"`` (default MLP head) or ``"cosine"``
        (learnable label-prototype cosine-similarity head).
    max_vocab_size : int
        Maximum vocabulary size (default 30 000).
    min_freq : int
        Minimum token frequency for vocabulary inclusion (default 2).
    max_len : int
        Sequence length after truncation / padding (default 256).
    embed_dim : int
        Embedding dimension (default 128).
    hidden_dim : int
        Hidden size for BiLSTM, or total feature size for TextCNN (default 256).
    num_filters : int
        Number of feature maps per filter size in TextCNN (default 128).
    filter_sizes : Sequence[int]
        Kernel sizes for TextCNN conv branches (default (3, 4, 5)).
    num_layers : int
        Number of LSTM layers in BiLSTM (default 2).
    num_epochs : int
        Maximum number of training epochs (default 10).
    batch_size : int
        Training batch size (default 32).
    lr : float
        Learning rate for Adam optimizer (default 1e-3).
    dropout : float
        Dropout probability (default 0.3).
    threshold : float
        Sigmoid threshold for binarising predictions (default 0.5).
    pretrained_vectors : np.ndarray, optional
        Pre-trained embedding matrix of shape ``(vocab_size, embed_dim)``.
        When provided, the embedding layer is initialised from these weights.

    Attributes
    ----------
    vocab : Dict[str, int]
        Token-to-index vocabulary built during ``fit()``.
    model : nn.Module
        Underlying PyTorch model (``_TextCNN`` or ``_BiLSTM``).
    device : torch.device
        Inference / training device selected automatically.
    """

    def __init__(
        self,
        num_labels: int,
        variant: str = "textcnn",
        head_type: str = "linear",
        max_vocab_size: int = 30_000,
        min_freq: int = 2,
        max_len: int = 256,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_filters: int = 128,
        filter_sizes: Sequence[int] = (3, 4, 5),
        num_layers: int = 2,
        num_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        dropout: float = 0.3,
        threshold: float = 0.5,
        pretrained_vectors: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels, **kwargs)
        variant = variant.lower()
        head_type = head_type.lower()
        if variant not in ("textcnn", "bilstm"):
            raise ValueError(f"variant must be 'textcnn' or 'bilstm', got '{variant}'")
        if head_type not in ("linear", "cosine"):
            raise ValueError(f"head_type must be 'linear' or 'cosine', got '{head_type}'")

        self.variant = variant
        self.head_type = head_type
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.filter_sizes = list(filter_sizes)
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.threshold = threshold
        self.pretrained_vectors = pretrained_vectors

        self.vocab: dict = {}
        self.vocab_size: int = 0
        self.model: Optional[nn.Module] = None
        self.device: torch.device = get_device()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "NeuralClassifier":
        """
        Build vocabulary, instantiate the model, and run the training loop.

        The best model (by validation Micro-F1) is restored after training.

        Parameters
        ----------
        X_train, y_train, X_val, y_val
            See ``BaseDocumentClassifier.fit``.

        Returns
        -------
        NeuralClassifier
            ``self``.
        """
        y_train = np.asarray(y_train, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        # Detect single-label vs multi-label
        self._single_label = (
            y_train.ndim == 2
            and (y_train.sum(axis=1) == 1).all()
        )

        # 1. Build vocabulary from training data only
        self.vocab = build_vocab(
            X_train,
            max_vocab_size=self.max_vocab_size,
            min_freq=self.min_freq,
        )
        self.vocab_size = len(self.vocab)
        logger.info("Vocabulary size: %d", self.vocab_size)

        # 2. Tokenise and pad
        ids_train = texts_to_ids(X_train, self.vocab, self.max_len)
        ids_val = texts_to_ids(X_val, self.vocab, self.max_len)

        # 3. Build DataLoaders
        train_loader = multilabel_dataloader(
            ids_train, y_train, batch_size=self.batch_size, shuffle=True
        )
        val_loader = multilabel_dataloader(
            ids_val, y_val, batch_size=self.batch_size, shuffle=False
        )

        # 4. Instantiate model
        self.model = self._build_model()
        self.model.to(self.device)

        # Optionally initialise embedding from pre-trained vectors
        if self.pretrained_vectors is not None:
            self._load_pretrained_vectors()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self._single_label:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        best_val_f1 = -1.0
        best_state: Optional[dict] = None

        # 5. Training loop
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for batch_ids, batch_labels in train_loader:
                batch_ids = batch_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_ids)
                if self._single_label:
                    loss = criterion(logits, batch_labels.argmax(dim=1))
                else:
                    loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # 6. Validation Micro-F1
            val_f1 = self._evaluate_loader(val_loader, y_val)
            avg_loss = epoch_loss / len(train_loader)
            logger.info(
                "Epoch %d/%d | loss=%.4f | val_micro_f1=%.4f",
                epoch, self.num_epochs, avg_loss, val_f1,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(self.model.state_dict())

        # 7. Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        logger.info("Training complete. Best val Micro-F1: %.4f", best_val_f1)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_test: List[str]) -> np.ndarray:
        """Return multi-hot binary predictions, shape ``(n, num_labels)``."""
        proba = self.predict_proba(X_test)
        if self._single_label:
            preds = np.zeros_like(proba, dtype=np.int8)
            preds[np.arange(len(proba)), proba.argmax(axis=1)] = 1
            return preds
        return (proba >= self.threshold).astype(np.int8)

    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        """
        Return per-label probability estimates, shape ``(n, num_labels)``.

        Processes data in mini-batches with ``torch.no_grad()`` to avoid OOM.
        """
        self._check_fitted()
        ids = texts_to_ids(X_test, self.vocab, self.max_len)
        ids_tensor = torch.from_numpy(ids)

        self.model.eval()
        all_logits: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(ids_tensor), self.batch_size):
                batch = ids_tensor[start : start + self.batch_size].to(self.device)
                logits = self.model(batch)
                all_logits.append(logits.cpu().numpy())

        logits_all = np.concatenate(all_logits, axis=0)
        if self._single_label:
            # softmax probabilities
            exp_l = np.exp(logits_all - logits_all.max(axis=1, keepdims=True))
            return (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)
        else:
            return (1.0 / (1.0 + np.exp(-logits_all))).astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Instantiate the selected architecture and head."""
        if self.variant == "textcnn":
            return _TextCNN(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                num_labels=self.num_labels,
                num_filters=self.num_filters,
                filter_sizes=self.filter_sizes,
                dropout=self.dropout,
                head_type=self.head_type,
            )
        # bilstm
        return _BiLSTM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_labels=self.num_labels,
            num_layers=self.num_layers,
            dropout=self.dropout,
            head_type=self.head_type,
        )

    def _load_pretrained_vectors(self) -> None:
        """Initialise the embedding layer from ``self.pretrained_vectors``."""
        vectors = torch.from_numpy(self.pretrained_vectors.astype(np.float32))
        vocab_size = min(vectors.size(0), self.vocab_size)
        with torch.no_grad():
            self.model.embedding.weight[:vocab_size].copy_(vectors[:vocab_size])
        logger.info("Loaded pre-trained vectors for %d tokens.", vocab_size)

    def _evaluate_loader(
        self, loader: Any, y_true: np.ndarray
    ) -> float:
        """Compute Micro-F1 on a DataLoader without training."""
        self.model.eval()
        all_logits: List[np.ndarray] = []
        with torch.no_grad():
            for batch_ids, _ in loader:
                batch_ids = batch_ids.to(self.device)
                logits = self.model(batch_ids)
                all_logits.append(logits.cpu().numpy())
        logits_all = np.concatenate(all_logits, axis=0)

        if self._single_label:
            preds = np.zeros_like(y_true, dtype=np.int8)
            preds[np.arange(len(logits_all)), logits_all.argmax(axis=1)] = 1
        else:
            proba = 1.0 / (1.0 + np.exp(-logits_all))  # sigmoid
            preds = (proba >= self.threshold).astype(np.int8)

        return float(f1_score(y_true.astype(np.int8), preds, average="micro", zero_division=0))
