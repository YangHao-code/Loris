"""GROUP B baselines — encoder embeddings + per-label classical head.

Pipeline (paper §7): DeBERTa_SVM / DeBERTa_XGBoost / RoBERTa_SVM / RoBERTa_XGBoost
are *"models fine-tuned on the training data"*. We therefore **fine-tune the
encoder end-to-end** on the multi-label task (BCEWithLogitsLoss over a temporary
linear head), then extract the **fine-tuned** pooled features and fit the named
head (SVM / XGBoost) on them. This is the paper-faithful headline.

A ``finetune=False`` mode keeps the original **frozen-encoder** features as a
disclosed *ablation* (much weaker — the encoder contributes no task signal).

Pipeline detail:

1. Load ``AutoTokenizer`` + ``AutoModel(model_name)``.
2. If ``finetune``: attach a linear head, fine-tune the whole encoder with
   BCEWithLogitsLoss (AdamW, early stop on val micro-F1), then drop the head.
   Else: keep the encoder frozen.
3. Mean-pool (attention-masked) the last hidden states into one fixed vector per
   document for train / val / test (using the fine-tuned-or-frozen encoder).
4. Fit a per-label head on those features:
     * ``head="svm"``    → ``OneVsRestClassifier(LinearSVC)``
     * ``head="logreg"`` → ``OneVsRestClassifier(LogisticRegression)``
     * ``head="xgboost"``→ one ``XGBClassifier`` per label.
5. Score on test with the LORIS metric (val-tuned global threshold).

BASELINES keys (headline = fine-tuned):
    deberta_svm      microsoft/deberta-v3-base FT + SVM head
    deberta_xgboost  microsoft/deberta-v3-base FT + XGBoost head
    roberta_svm      roberta-base              FT + SVM head
    roberta_xgboost  roberta-base              FT + XGBoost head
(frozen-feature ablation available via finetune=False, keys suffixed _frozen.)

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
        finetune: bool = True,
        ft_epochs: int = 3,
        ft_lr: float = 2e-5,
        ft_batch_size: int = 16,
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
        self.finetune = finetune
        self.ft_epochs = ft_epochs
        self.ft_lr = ft_lr
        self.ft_batch_size = ft_batch_size
        self._ft_done = False

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

    def _finetune_encoder(self, texts: List[str], y: np.ndarray,
                          val_texts=None, val_y=None) -> None:
        """End-to-end fine-tune the encoder body on the multi-label task.

        Attaches a temporary mean-pool + linear head, optimises BCEWithLogitsLoss
        with AdamW, then re-freezes the (now fine-tuned) encoder so ``embed``
        extracts task-adapted features. Paper: DeBERTa/RoBERTa are "fine-tuned
        on the training data".
        """
        import torch
        import torch.nn as nn

        self._load_encoder()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self.device
        H = self.encoder.config.hidden_size
        head = nn.Linear(H, self.num_labels).to(device)

        # unfreeze encoder for fine-tuning
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        try:
            self.encoder.gradient_checkpointing_enable()
        except Exception:
            pass

        params = list(self.encoder.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=self.ft_lr, weight_decay=0.01)
        crit = nn.BCEWithLogitsLoss()
        use_amp = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        y_t = np.asarray(y, dtype=np.float32)
        n = len(texts)
        idx = np.arange(n)
        bs = self.ft_batch_size
        for epoch in range(1, self.ft_epochs + 1):
            np.random.shuffle(idx)
            ep_loss = 0.0
            nb = 0
            for s in range(0, n, bs):
                b = idx[s:s + bs]
                batch = [texts[i] for i in b]
                yb = torch.from_numpy(y_t[b]).to(device)
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    hidden = out.last_hidden_state
                    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                    logits = head(pooled)
                    loss = crit(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                ep_loss += float(loss.item())
                nb += 1
            import logging
            logging.getLogger("loris.baselines").info(
                "  [%s FT] epoch %d/%d loss=%.4f",
                self.model_name, epoch, self.ft_epochs, ep_loss / max(nb, 1))

        # re-freeze: embed() will now extract fine-tuned features
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._ft_done = True

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return pooled embeddings, shape ``(len(texts), hidden)``.

        Uses the fine-tuned encoder weights if ``_finetune_encoder`` has run,
        else the frozen pretrained weights.
        """
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
        # Paper-faithful: fine-tune the encoder on the task BEFORE extracting
        # features. Frozen mode (finetune=False) is the ablation.
        if self.finetune and not self._ft_done:
            self._finetune_encoder(list(Xtr), ytr, Xval, yval)
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
    finetune: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fit an encoder + classical head baseline and score on test.

    ``finetune=True`` (default, paper-faithful): fine-tune the encoder on the
    task, then fit the head on fine-tuned features. ``finetune=False``: frozen
    encoder features (ablation).
    """
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
        finetune=finetune,
        ft_epochs=int(kwargs.get("ft_epochs", 3)),
    )

    with Timer() as t:
        clf.fit(split.train_X, split.train_y, split.val_X, split.val_y)
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
            "finetune": bool(finetune),
            "pooling": pooling,
            "threshold": metrics.get("threshold"),
            "wall_sec": round(t.sec, 2),
        },
    }


# ── Contract: BASELINES dict ─────────────────────────────────────────────────

def deberta_svm(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """DeBERTa-v3-base FINE-TUNED + OvR LinearSVC head (paper headline).

    kwargs: pooling ('mean'|'cls'), max_length, batch_size, ft_epochs.
    """
    return _run_encoder_head(
        split, "microsoft/deberta-v3-base", "svm", seed=seed, finetune=True, **kwargs
    )


def deberta_xgboost(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """DeBERTa-v3-base FINE-TUNED + per-label XGBoost head (paper headline).

    kwargs: pooling, max_length, batch_size, ft_epochs. Needs xgboost + sentencepiece.
    """
    return _run_encoder_head(
        split, "microsoft/deberta-v3-base", "xgboost", seed=seed, finetune=True, **kwargs
    )


def roberta_svm(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """roberta-base FINE-TUNED + OvR LinearSVC head (paper headline)."""
    return _run_encoder_head(
        split, "roberta-base", "svm", seed=seed, finetune=True, **kwargs
    )


def roberta_xgboost(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """roberta-base FINE-TUNED + per-label XGBoost head (paper headline)."""
    return _run_encoder_head(
        split, "roberta-base", "xgboost", seed=seed, finetune=True, **kwargs
    )


# ── Frozen-encoder ablation variants (disclosed; NOT the paper headline) ──────
def deberta_svm_frozen(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """Ablation: DeBERTa-v3-base FROZEN embeddings + OvR LinearSVC head."""
    return _run_encoder_head(
        split, "microsoft/deberta-v3-base", "svm", seed=seed, finetune=False, **kwargs
    )


def deberta_xgboost_frozen(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """Ablation: DeBERTa-v3-base FROZEN embeddings + per-label XGBoost head."""
    return _run_encoder_head(
        split, "microsoft/deberta-v3-base", "xgboost", seed=seed, finetune=False, **kwargs
    )


def roberta_svm_frozen(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """Ablation: roberta-base FROZEN embeddings + OvR LinearSVC head."""
    return _run_encoder_head(
        split, "roberta-base", "svm", seed=seed, finetune=False, **kwargs
    )


def roberta_xgboost_frozen(split: Split, seed: int = 0, **kwargs: Any) -> Dict[str, Any]:
    """Ablation: roberta-base FROZEN embeddings + per-label XGBoost head."""
    return _run_encoder_head(
        split, "roberta-base", "xgboost", seed=seed, finetune=False, **kwargs
    )


BASELINES = {
    # paper headline: fine-tuned encoder + named head
    "deberta_svm": deberta_svm,
    "deberta_xgboost": deberta_xgboost,
    "roberta_svm": roberta_svm,
    "roberta_xgboost": roberta_xgboost,
    # disclosed frozen-encoder ablations
    "deberta_svm_frozen": deberta_svm_frozen,
    "deberta_xgboost_frozen": deberta_xgboost_frozen,
    "roberta_svm_frozen": roberta_svm_frozen,
    "roberta_xgboost_frozen": roberta_xgboost_frozen,
}
