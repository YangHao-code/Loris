"""
models/pretrained_encoder_classifier.py
----------------------------------------
Fine-tuned pre-trained encoder classifier for multi-label text classification.

Supported backbones: any HuggingFace model compatible with
``AutoModelForSequenceClassification`` (encoder) or ``AutoModel`` (feature
extractor), e.g. RoBERTa, DeBERTa-v3.

Two ``classifier_head`` modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``"mlp"`` (default): End-to-end fine-tune with a linear classification head
  on top of the ``[CLS]`` token (``AutoModelForSequenceClassification``).
* ``"xgboost"``: Freeze the encoder, extract ``[CLS]`` embeddings, then fit
  a ``MultiOutputClassifier(XGBClassifier(...))`` on the frozen features.
  Requires the ``xgboost`` package (``pip install xgboost``).

Key design choices (MLP path)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``problem_type="multi_label_classification"`` → HuggingFace uses
  ``BCEWithLogitsLoss``; we compute it ourselves for full control.
* Label tensors must be ``float32`` (required by BCEWithLogitsLoss).
* Mixed-precision training (AMP) enabled by default on CUDA.
* Optional gradient checkpointing reduces GPU memory.
* Early stopping on validation Micro-F1 with configurable patience.
* Larger prediction batch (no gradients needed).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from models.base import BaseDocumentClassifier
from models.utils import get_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

class _EncoderDataset(Dataset):
    """Wraps tokenised inputs and float labels for a DataLoader."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

class PretrainedEncoderClassifier(BaseDocumentClassifier):
    """
    Pre-trained encoder for multi-label classification (MLP or XGBoost head).

    Parameters
    ----------
    num_labels : int
        Number of output labels.
    model_name : str
        HuggingFace model identifier (default ``"roberta-base"``).
        Other examples: ``"microsoft/deberta-v3-base"``.
    classifier_head : str
        Output head type: ``"mlp"`` (default, end-to-end fine-tune) or
        ``"xgboost"`` (frozen encoder features + gradient-boosted trees).
    max_length : int
        Tokeniser max sequence length (default 512).
    batch_size : int
        Training batch size for MLP path (default 8).
    pred_batch_size : int
        Prediction / feature-extraction batch size (default 32).
    num_epochs : int
        Maximum training epochs for MLP path (default 5).
    lr : float
        Peak learning rate for AdamW in MLP path (default 2e-5).
    threshold : float
        Sigmoid threshold for binarising predictions (default 0.5).
    warmup_ratio : float
        Fraction of training steps used for linear warmup (default 0.1).
    weight_decay : float
        AdamW weight decay in MLP path (default 0.01).
    use_amp : bool
        Enable automatic mixed precision on CUDA (default ``True``).
    patience : int
        Early-stopping patience for MLP path (default 3).
    gradient_checkpointing : bool
        Enable gradient checkpointing to trade compute for memory (default
        ``False``). Incompatible with ``torch.compile()``.
    xgb_n_estimators : int
        Number of trees for XGBoost (default 300).
    xgb_max_depth : int
        Maximum tree depth for XGBoost (default 6).
    **kwargs
        Forwarded to ``BaseDocumentClassifier.__init__``.

    Attributes
    ----------
    tokenizer : AutoTokenizer
        Loaded tokeniser.
    model : AutoModelForSequenceClassification or None
        Fine-tuned MLP classification model (MLP path only).
    encoder : AutoModel or None
        Frozen feature-extraction encoder (XGBoost path only).
    xgb_clf : MultiOutputClassifier or None
        Fitted XGBoost multi-label classifier (XGBoost path only).
    device : torch.device
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = "roberta-base",
        classifier_head: str = "mlp",
        max_length: int = 512,
        batch_size: int = 8,
        pred_batch_size: int = 32,
        num_epochs: int = 5,
        lr: float = 2e-5,
        threshold: float = 0.5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        use_amp: bool = True,
        patience: int = 3,
        gradient_checkpointing: bool = False,
        xgb_n_estimators: int = 300,
        xgb_max_depth: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels, **kwargs)
        classifier_head = classifier_head.lower()
        if classifier_head not in ("mlp", "xgboost"):
            raise ValueError(
                f"classifier_head must be 'mlp' or 'xgboost', got '{classifier_head}'"
            )
        self.model_name = model_name
        self.classifier_head = classifier_head
        self.max_length = max_length
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.threshold = threshold
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.use_amp = use_amp and torch.cuda.is_available()
        self.patience = patience
        self.gradient_checkpointing = gradient_checkpointing
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth

        self.device: torch.device = get_device()
        self.tokenizer: Optional[AutoTokenizer] = None
        # MLP path
        self.model: Optional[AutoModelForSequenceClassification] = None
        # XGBoost path
        self.encoder = None
        self.xgb_clf = None

    # ------------------------------------------------------------------
    # Training — dispatcher
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "PretrainedEncoderClassifier":
        """
        Train the classifier.

        Dispatches to ``_fit_mlp`` or ``_fit_xgboost`` based on
        ``self.classifier_head``.

        Parameters
        ----------
        X_train, y_train, X_val, y_val
            See ``BaseDocumentClassifier.fit``.

        Returns
        -------
        PretrainedEncoderClassifier
            ``self``.
        """
        y_train = np.asarray(y_train, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        logger.info("Loading tokeniser: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.classifier_head == "xgboost":
            return self._fit_xgboost(X_train, y_train, X_val, y_val)
        return self._fit_mlp(X_train, y_train, X_val, y_val)

    # ------------------------------------------------------------------
    # MLP fine-tune path
    # ------------------------------------------------------------------

    def _fit_mlp(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "PretrainedEncoderClassifier":
        """End-to-end fine-tune with BCEWithLogitsLoss."""
        logger.info("MLP path — loading %s", self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification",
        )
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_params = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_params, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        best_val_f1 = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_loss += loss.item()

            val_f1 = self._val_micro_f1_mlp(val_loader, y_val)
            logger.info(
                "Epoch %d/%d | loss=%.4f | val_micro_f1=%.4f",
                epoch, self.num_epochs, epoch_loss / len(train_loader), val_f1,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                import copy
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        logger.info("MLP training complete. Best val Micro-F1: %.4f", best_val_f1)
        return self

    # ------------------------------------------------------------------
    # XGBoost path
    # ------------------------------------------------------------------

    def _fit_xgboost(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "PretrainedEncoderClassifier":
        """Extract frozen [CLS] embeddings then fit a multi-output XGBoost."""
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for classifier_head='xgboost'. "
                "Install it with: pip install xgboost"
            ) from exc
        from sklearn.multioutput import MultiOutputClassifier
        from transformers import AutoModel

        logger.info("XGBoost path — loading frozen encoder: %s", self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.to(self.device)

        logger.info("Extracting training embeddings …")
        train_embs = self._extract_cls_embeddings(X_train)  # (n_train, H)

        logger.info("Fitting XGBoost …")
        self.xgb_clf = MultiOutputClassifier(
            XGBClassifier(
                n_estimators=self.xgb_n_estimators,
                max_depth=self.xgb_max_depth,
                eval_metric="logloss",
                n_jobs=1,
                verbosity=0,
            ),
            n_jobs=-1,
        )
        self.xgb_clf.fit(train_embs, y_train.astype(int))

        self.is_fitted = True
        logger.info("XGBoost fitting complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_test: List[str]) -> np.ndarray:
        """Return multi-hot binary predictions, shape ``(n, num_labels)``."""
        proba = self.predict_proba(X_test)
        return (proba >= self.threshold).astype(np.int8)

    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        """
        Return per-label probability estimates, shape ``(n, num_labels)``.

        MLP path: batched forward pass with ``torch.no_grad()``.
        XGBoost path: extract frozen CLS embeddings, call XGB's predict_proba.
        """
        self._check_fitted()

        if self.classifier_head == "xgboost":
            embs = self._extract_cls_embeddings(X_test)
            # predict_proba returns List[L] of (n, 2); take positive-class column
            proba_list = self.xgb_clf.predict_proba(embs)
            return np.stack([p[:, 1] for p in proba_list], axis=1).astype(np.float32)

        # MLP path
        loader = self._make_loader(X_test, labels=None, shuffle=False,
                                   batch_size=self.pred_batch_size)
        self.model.eval()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits
                probs = torch.sigmoid(logits).cpu().float().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenise(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _make_loader(
        self,
        texts: List[str],
        labels: Optional[np.ndarray],
        shuffle: bool = False,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        enc = self._tokenise(texts)
        label_tensor = (
            torch.from_numpy(labels.astype(np.float32)) if labels is not None else None
        )
        dataset = _EncoderDataset(enc["input_ids"], enc["attention_mask"], label_tensor)
        bs = batch_size if batch_size is not None else self.batch_size
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _extract_cls_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract [CLS] / first-token hidden states from the frozen encoder.

        Parameters
        ----------
        texts : List[str]

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), hidden_size)``, dtype ``float32``.
        """
        enc = self._tokenise(texts)
        all_embs: List[np.ndarray] = []
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        with torch.no_grad():
            for i in range(0, len(ids), self.pred_batch_size):
                out = self.encoder(
                    input_ids=ids[i : i + self.pred_batch_size].to(self.device),
                    attention_mask=mask[i : i + self.pred_batch_size].to(self.device),
                )
                # Use [CLS] token (index 0 of last hidden state)
                cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
                all_embs.append(cls)
        return np.concatenate(all_embs, axis=0)

    def _val_micro_f1_mlp(self, loader: DataLoader, y_true: np.ndarray) -> float:
        """Compute validation Micro-F1 for the MLP path."""
        from sklearn.metrics import f1_score

        self.model.eval()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits
                probs = torch.sigmoid(logits).cpu().float().numpy()
                all_probs.append(probs)

        proba = np.concatenate(all_probs, axis=0)
        preds = (proba >= self.threshold).astype(np.int8)
        return float(f1_score(y_true.astype(np.int8), preds, average="micro", zero_division=0))
