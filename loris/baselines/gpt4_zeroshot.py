"""GPT4 zero-shot multi-label baseline (paper §7, Group B).

For each TEST document we ask ``gpt-4.1`` to return the subset of the fixed
candidate ``label_names`` that apply (zero-shot, no training).  To control
cost we subsample the test set to ``max_test`` documents (default 1000) with a
seeded ``RandomState`` and score **only on that subsample** (the matching
``test_y`` rows).

Offline / CI: ``LORIS_LLM_MOCK=1`` (or no ``OPENAI_API_KEY``) makes the client
return empty predictions, so macro-F1 ~ 0.  That is expected — the smoke test
verifies the baseline RUNS, not its accuracy.  Mock results must NOT be
reported as real numbers.

Inapplicable on datasets without raw text (e.g. rcv1 ships as hashed TF-IDF):
those raise ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from loris.baselines.common import Split, Timer, has_raw_text, score


def _gpt4_zeroshot(split: Split, seed: int = 0, **kwargs) -> Dict:
    """Zero-shot GPT-4 multi-label classification on a test subsample.

    kwargs:
        max_test (int, default 1000): max number of test docs to query/score.
    """
    if not has_raw_text(split.dataset):
        raise NotImplementedError(
            f"GPT4 zero-shot needs raw document text; dataset "
            f"'{split.dataset}' ships without it (e.g. rcv1 hashed TF-IDF)."
        )

    from loris.baselines.llm_client import LLMClient

    max_test = int(kwargs.get("max_test", 1000))
    cli = LLMClient()
    if cli.mock:
        print(
            "[gpt4_zeroshot] NOTE: running in MOCK mode (LORIS_LLM_MOCK=1 or no "
            "OPENAI_API_KEY) — predictions are empty, metrics ~0. "
            "Do NOT report these as real numbers."
        )

    n_test = len(split.test_X)
    k = min(max_test, n_test)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_test, size=k, replace=False)
    idx.sort()

    label_names = split.label_names
    label_to_col = {l: j for j, l in enumerate(label_names)}
    n_labels = len(label_names)

    preds = np.zeros((k, n_labels), dtype=np.int8)
    with Timer() as t:
        for i, doc_i in enumerate(idx):
            applied = cli.classify_multilabel(split.test_X[doc_i], label_names)
            for lab in applied:
                col = label_to_col.get(lab)
                if col is not None:
                    preds[i, col] = 1

    y_true = np.asarray(split.test_y, dtype=np.int8)[idx]
    metrics = score(y_true, preds)

    extra = cli.cost_summary()
    extra["n_scored"] = int(k)
    extra["wall_sec"] = float(t.sec)

    return {
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "subset_accuracy": metrics["subset_accuracy"],
        "n_annotations": 0,
        "extra": extra,
    }


BASELINES = {"gpt4": _gpt4_zeroshot}
