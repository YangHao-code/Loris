"""GROUP B baselines — frozen-encoder embeddings + per-label classical head.

Pipeline (paper §7, "encoder + shallow head" family):

1. Load ``AutoTokenizer`` + ``AutoModel(model_name)`` (a *frozen* feature
   extractor — no fine-tuning).
2. Mean-pool (attention-masked) the last hidden states into one fixed vector
   per document for train / val / test.
3. Fit a per-label head on the frozen features:
     * ``head="svm"``    → ``OneVsRestClassifier(LinearSVC)`` (``decision_function``
       gives per-label scores, val-tuned global threshold).
     * ``head="logreg"`` → ``OneVsRestClassifier(LogisticRegression)``
       (``predict_proba`` per-label scores).
     * ``head="xgboost"``→ one ``XGBClassifier`` per label (lazy import;
       clear error if xgboost missing).
4. Score on test with the LORIS metric.

BASELINES keys:
    deberta_svm      microsoft/deberta-v3-base + SVM head
    deberta_xgboost  microsoft/deberta-v3-base + XGBoost head
    roberta_svm      roberta-base              + SVM head
    roberta_xgboost  roberta-base              + XGBoost head

``deberta-v3-base`` weights and ``xgboost``/``sentencepiece`` may be absent on
the smoke box; the ``roberta_svm`` key is smoked on CPU. The ``deberta_*`` and
``*_xgboost`` keys run once the server downloads ``microsoft/deberta-v3-base``
and ``pip install xgboost sentencepiece``.

On datasets without raw text (``has_raw_text==False``, e.g. rcv1) every key
raises ``NotImplementedError`` — encoder baselines need real text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from loris.baselines.common import (
    Split,
    Timer,
    has_raw_text,
    score,
    score_from_scores,
)


# ---------------------------------------------------------------------------
# Frozen-encoder + classical-head classifier
# ---------------------------------------------------------------------------

class EncoderHeadClassifier:
    """Frozen transformer encoder → mean-pooled embeddings → per-label head.

    Parameters
    ----------
    num_labels : int
    model_name : str
        HuggingFace identifier, e.g. ``"roberta-base"`` or
        ``"microsoft/deberta-v3-base"``.
    head : str
        ``"svm"`` | ``"logreg"`` | ``"xgboost"``.
    pooling : str
        ``"mean"`` (attention-masked mean of last hidden state, default) or
        ``"cls"`` (first token).
    max_length : int
        Tokeniser max sequence length (default 256 — embeddings only, so we can
        afford to keep it modest for speed).
    batch_size : int
        Embedding-extraction batch size (default 32).
    seed : int
        Used to seed the head where applicable.
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = "roberta-base",
        head: str = "svm",
        pooling: str = "mean",
        max_length: int = 256,
        batch_size: int = 32,
        seed: int = 0,
        xgb_n_estimators: int = 300,
        xgb_max_depth: int = 6,
    ) -> None:
        head = head.lower()
        if head not in ("svm", "logreg", "xgboost"):
            raise ValueError(
                f"head must be 'svm', 'logreg' or 'xgboost', got '{head}'"
            )
        pooling = pooling.lower()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got '{pooling}'")
        self.num_labels = num_labels
        self.model_name = model_name
        self.head = head
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth

        self.tokenizer = None
        self.encoder = None
        self.device = None
        self.clf = None
        # which label columns are degenerate (single class in train) — for
        # heads that can't fit a 1-class problem (xgboost). 0 ⇒ never predict.
        self._dead_labels: Optional[np.ndarray] = None

    # -- encoder loading / embedding extraction ----------------------------

    def _load_encoder(self) -> None:
        if self.encoder is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        from loris.models.utils import get_device

        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.to(self.device)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return frozen pooled embeddings, shape ``(len(texts), hidden)``."""
        import torch

        self._load_encoder()
        all_embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state  # (b, T, H)
                if self.pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:  # masked mean
                    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (b, T, 1)
                    summed = (hidden * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1.0)
                    pooled = summed / counts
                all_embs.append(pooled.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)

    # -- head fitting ------------------------------------------------------

    def fit(self, Xtr: List[str], ytr: np.ndarray, Xval=None, yval=None) -> "EncoderHeadClassifier":
        ytr = np.asarray(ytr, dtype=np.int8)
        emb_tr = self.embed(list(Xtr))

        if self.head == "svm":
            from sklearn.multiclass import OneVsRestClassifier
            from sklearn.svm import LinearSVC

            self.clf = OneVsRestClassifier(
                LinearSVC(random_state=self.seed), n_jobs=-1
            )
            self.clf.fit(emb_tr, ytr)
        elif self.head == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.multiclass import OneVsRestClassifier

            self.clf = OneVsRestClassifier(
                LogisticRegression(
                    random_state=self.seed, max_iter=1000, n_jobs=None
                ),
                n_jobs=-1,
            )
            self.clf.fit(emb_tr, ytr)
        else:  # xgboost: one classifier per label
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "xgboost is required for head='xgboost'. "
                    "Install it with: pip install xgboost"
                ) from exc
            self.clf = []
            self._dead_labels = np.zeros(self.num_labels, dtype=bool)
            for j in range(self.num_labels):
                col = ytr[:, j]
                if col.sum() == 0 or col.sum() == len(col):
                    # single-class column: XGB can't fit; remember constant.
                    self._dead_labels[j] = True
                    self.clf.append(int(col.sum() > 0))  # constant prediction
                    continue
                model = XGBClassifier(
                    n_estimators=self.xgb_n_estimators,
                    max_depth=self.xgb_max_depth,
                    eval_metric="logloss",
                    random_state=self.seed,
                    n_jobs=-1,
                    verbosity=0,
                )
                model.fit(emb_tr, col.astype(int))
                self.clf.append(model)
        return self

    # -- scoring -----------------------------------------------------------

    def decision_scores(self, X: List[str]) -> np.ndarray:
        """Per-label real-valued scores, shape ``(n, num_labels)``.

        SVM → ``decision_function`` (raw margin). logreg/xgboost → P(label=1).
        """
        emb = self.embed(list(X))
        if self.head == "svm":
            scores = self.clf.decision_function(emb)
            if scores.ndim == 1:  # single-label OvR edge case
                scores = scores.reshape(-1, 1)
            return scores.astype(np.float32)
        if self.head == "logreg":
            # OneVsRestClassifier.predict_proba → (n, n_labels) for multilabel
            return np.asarray(self.clf.predict_proba(emb), dtype=np.float32)
        # xgboost: per-label probability
        n = emb.shape[0]
        out = np.zeros((n, self.num_labels), dtype=np.float32)
        for j, model in enumerate(self.clf):
            if isinstance(model, int):  # constant column
                out[:, j] = float(model)
                continue
            proba = model.predict_proba(emb)  # (n, 2)
            out[:, j] = proba[:, 1]
        return out


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def _run_encoder_head(
    split: Split,
    model_name: str,
    head: str,
    seed: int = 0,
    pooling: str = "mean",
    max_length: int = 256,
    batch_size: int = 32,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fit a frozen-encoder + classical head baseline and score on test."""
    if not has_raw_text(split.dataset):
        raise NotImplementedError(
            f"encoder_head baselines require raw text; dataset "
            f"'{split.dataset}' has none (e.g. rcv1 is hashed TF-IDF only)."
        )

    clf = EncoderHeadClassifier(
        num_labels=split.n_labels,
        model_name=model_name,
        head=head,
        pooling=pooling,
        max_length=max_length,
        batch_size=batch_size,
        seed=seed,
    )

    with Timer() as t:
        clf.fit(split.train_X, split.train_y)
        test_scores = clf.decision_scores(split.test_X)
        val_scores = clf.decision_scores(split.val_X)

    metrics = score_from_scores(
        test_scores, split.test_y, val_scores, split.val_y
    )

    n_train = int(np.asarray(split.train_y).shape[0])
    return {
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "subset_accuracy": metrics["subset_accuracy"],
        "n_annotations": n_train,
        "extra": {
            "model_name": model_name,
            "head": head,
            "pooling": pooling,
            "threshold": metrics.get("threshold"),
            "wall_sec": round(t.sec, 2),
        },
    }


# ── Contract: BASELINES dict ─────────────────────────────────────────────────

def deberta_svm(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """DeBERTa-v3-base frozen embeddings + OvR LinearSVC head.

    kwargs: pooling ('mean'|'cls'), max_length, batch_size.
    """
    return _run_encoder_head(
        split, "microsoft/deberta-v3-base", "svm", seed=seed, **kwargs
    )


def deberta_xgboost(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """DeBERTa-v3-base frozen embeddings + per-label XGBoost head.

    kwargs: pooling, max_length, batch_size. Needs xgboost + sentencepiece.
    """
    return _run_encoder_head(
        split, "microsoft/deberta-v3-base", "xgboost", seed=seed, **kwargs
    )


def roberta_svm(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """roberta-base frozen embeddings + OvR LinearSVC head.

    kwargs: pooling ('mean'|'cls'), max_length, batch_size.
    """
    return _run_encoder_head(
        split, "roberta-base", "svm", seed=seed, **kwargs
    )


def roberta_xgboost(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """roberta-base frozen embeddings + per-label XGBoost head.

    kwargs: pooling, max_length, batch_size. Needs xgboost.
    """
    return _run_encoder_head(
        split, "roberta-base", "xgboost", seed=seed, **kwargs
    )


BASELINES = {
    "deberta_svm": deberta_svm,
    "deberta_xgboost": deberta_xgboost,
    "roberta_svm": roberta_svm,
    "roberta_xgboost": roberta_xgboost,
}
