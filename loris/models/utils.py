"""
models/utils.py
---------------
Shared utilities for the MLTC classifier framework.
Used by NeuralClassifier and LoRASLMClassifier.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_vocab(
    texts: List[str],
    max_vocab_size: int = 30_000,
    min_freq: int = 2,
    special_tokens: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Build a token-to-index vocabulary from whitespace-tokenized texts.

    Parameters
    ----------
    texts : List[str]
        Raw text samples to build the vocabulary from (training split only).
    max_vocab_size : int
        Maximum number of tokens in the vocabulary (including special tokens).
    min_freq : int
        Minimum token frequency required for inclusion.
    special_tokens : List[str], optional
        Tokens prepended at indices 0, 1, … before regular tokens.
        Defaults to ["<PAD>", "<UNK>"].

    Returns
    -------
    Dict[str, int]
        Mapping from token string to integer index.
    """
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>"]

    counter: Counter = Counter()
    for text in texts:
        counter.update(text.lower().split())

    vocab: Dict[str, int] = {}
    for tok in special_tokens:
        vocab[tok] = len(vocab)

    for token, freq in counter.most_common(max_vocab_size - len(special_tokens)):
        if freq < min_freq:
            break
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def texts_to_ids(
    texts: List[str],
    vocab: Dict[str, int],
    max_len: int = 256,
    pad_idx: int = 0,
    unk_idx: int = 1,
) -> np.ndarray:
    """
    Convert a list of raw strings to a zero-padded integer array.

    Tokens are lowercased and split on whitespace. Sequences longer than
    ``max_len`` are truncated from the right (first ``max_len`` tokens kept).

    Parameters
    ----------
    texts : List[str]
        Input text samples.
    vocab : Dict[str, int]
        Vocabulary mapping from token to index.
    max_len : int
        Fixed sequence length after padding/truncation.
    pad_idx : int
        Index used for padding (must match the embedding's ``padding_idx``).
    unk_idx : int
        Index used for out-of-vocabulary tokens.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(len(texts), max_len)``, dtype ``int64``.
    """
    result = np.full((len(texts), max_len), fill_value=pad_idx, dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = text.lower().split()[:max_len]
        for j, tok in enumerate(tokens):
            result[i, j] = vocab.get(tok, unk_idx)
    return result


def multilabel_dataloader(
    token_ids: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Wrap token_id and label arrays into a PyTorch DataLoader.

    Parameters
    ----------
    token_ids : np.ndarray
        Shape ``(n_samples, max_len)``, dtype ``int64``.
    labels : np.ndarray
        Shape ``(n_samples, num_labels)``, dtype ``float32``.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle samples each epoch.
    num_workers : int
        DataLoader worker processes (0 = main process only).

    Returns
    -------
    DataLoader
        Yields ``(token_id_batch, label_batch)`` tuples.
    """
    ids_tensor = torch.from_numpy(token_ids).long()
    labels_tensor = torch.from_numpy(labels.astype(np.float32))
    dataset = TensorDataset(ids_tensor, labels_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def sigmoid_threshold(
    logits: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply sigmoid activation then binarize at ``threshold``.

    Parameters
    ----------
    logits : np.ndarray
        Raw model output, shape ``(n_samples, num_labels)``.
    threshold : float
        Decision boundary in probability space (after sigmoid).

    Returns
    -------
    np.ndarray
        Multi-hot binary array of dtype ``int8``, same shape as ``logits``.
    """
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return (probs >= threshold).astype(np.int8)
