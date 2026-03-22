"""
demo.py
-------
End-to-end demonstration of all four MLTC classifiers on synthetic data.

Run examples
~~~~~~~~~~~~
    # Fast, CPU-friendly (no model downloads needed)
    python demo.py --classifier tfidf
    python demo.py --classifier neural --variant textcnn
    python demo.py --classifier neural --variant bilstm

    # Requires GPU + pre-downloaded model weights
    python demo.py --classifier encoder --model_name roberta-base
    python demo.py --classifier lora --model_name meta-llama/Meta-Llama-3-8B

    # Run all CPU-compatible models at once
    python demo.py --classifier all

Output (per model)
~~~~~~~~~~~~~~~~~~
    [TFIDFClassifier] Micro-F1:        0.4123
    [TFIDFClassifier] Macro-F1:        0.3891
    [TFIDFClassifier] Subset Accuracy: 0.1600
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("demo")


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

# Keyword pools: label i is positively associated with keywords in LABEL_WORDS[i]
LABEL_WORDS = [
    ["sports", "game", "team", "player", "match", "win", "goal"],
    ["science", "research", "study", "data", "experiment", "result"],
    ["politics", "election", "government", "policy", "vote", "law"],
    ["technology", "software", "hardware", "computer", "internet", "app"],
    ["health", "medical", "doctor", "hospital", "patient", "disease"],
    ["finance", "market", "stock", "investment", "economy", "bank"],
]


def make_dummy_data(
    n_train: int = 300,
    n_val: int = 80,
    n_test: int = 80,
    num_labels: int = 6,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic multi-label classification data.

    Each sample is a short sentence of random common words, with a few
    domain-specific keywords injected to create a learnable signal.
    Labels are correlated with injected keywords.

    Parameters
    ----------
    n_train, n_val, n_test : int
        Number of samples per split.
    num_labels : int
        Number of labels. Must be ≤ len(LABEL_WORDS) (6).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_val, y_val, X_test, y_test)``
    """
    rng = np.random.default_rng(seed)
    num_labels = min(num_labels, len(LABEL_WORDS))

    # General-purpose filler vocabulary
    filler = [
        "the", "a", "is", "in", "on", "at", "for", "with", "this",
        "that", "are", "was", "has", "have", "be", "it", "its",
        "new", "old", "good", "bad", "big", "small", "many", "few",
    ]

    def make_sample(label_mask: np.ndarray) -> str:
        words = list(rng.choice(filler, size=rng.integers(6, 12), replace=True))
        for label_idx in np.where(label_mask)[0]:
            kws = LABEL_WORDS[label_idx]
            words.insert(
                rng.integers(0, len(words) + 1),
                kws[rng.integers(0, len(kws))],
            )
        rng.shuffle(words)
        return " ".join(words)

    def make_split(n: int) -> tuple:
        # Each sample gets 1–3 random labels with ~60 % probability
        labels = (rng.random((n, num_labels)) < 0.35).astype(np.float32)
        # Ensure at least one label per sample
        for i in range(n):
            if labels[i].sum() == 0:
                labels[i, rng.integers(0, num_labels)] = 1.0
        texts = [make_sample(labels[i]) for i in range(n)]
        return texts, labels

    X_train, y_train = make_split(n_train)
    X_val, y_val = make_split(n_val)
    X_test, y_test = make_split(n_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Demo runners
# ---------------------------------------------------------------------------

def run_tfidf(
    num_labels: int,
    clf_type: str = "svm",
    **kwargs,
) -> None:
    """Demonstrate TFIDFClassifier (SVM or Logistic Regression)."""
    from models import TFIDFClassifier

    X_train, y_train, X_val, y_val, X_test, y_test = make_dummy_data(
        num_labels=num_labels
    )
    clf = TFIDFClassifier(num_labels=num_labels, classifier_type=clf_type)
    clf.fit(X_train, y_train, X_val, y_val)

    sample = ["the sports team won the game with a great goal"]
    proba = clf.predict_proba(sample)
    pred = clf.predict(sample)
    logger.info("Sample predict_proba:\n%s", proba)
    logger.info("Sample predict (multi-hot):\n%s", pred)

    metrics = clf.evaluate(X_test, y_test)
    _print_metrics(f"TFIDFClassifier[{clf_type}]", metrics)


def run_neural(
    num_labels: int,
    variant: str = "textcnn",
    head_type: str = "linear",
    **kwargs,
) -> None:
    """Demonstrate NeuralClassifier (TextCNN or BiLSTM, linear or cosine head)."""
    from models import NeuralClassifier

    X_train, y_train, X_val, y_val, X_test, y_test = make_dummy_data(
        num_labels=num_labels
    )
    clf = NeuralClassifier(
        num_labels=num_labels,
        variant=variant,
        head_type=head_type,
        num_epochs=5,
        batch_size=32,
    )
    clf.fit(X_train, y_train, X_val, y_val)

    sample = ["science research data experiment new study result"]
    proba = clf.predict_proba(sample)
    pred = clf.predict(sample)
    logger.info("Sample predict_proba:\n%s", proba)
    logger.info("Sample predict (multi-hot):\n%s", pred)

    metrics = clf.evaluate(X_test, y_test)
    _print_metrics(f"NeuralClassifier[{variant}+{head_type}]", metrics)


def run_encoder(
    num_labels: int,
    model_name: str = "roberta-base",
    classifier_head: str = "mlp",
    **kwargs,
) -> None:
    """Demonstrate PretrainedEncoderClassifier (MLP or XGBoost head)."""
    from models import PretrainedEncoderClassifier

    X_train, y_train, X_val, y_val, X_test, y_test = make_dummy_data(
        num_labels=num_labels
    )
    clf = PretrainedEncoderClassifier(
        num_labels=num_labels,
        model_name=model_name,
        classifier_head=classifier_head,
        num_epochs=3,
        batch_size=8,
    )
    clf.fit(X_train, y_train, X_val, y_val)

    sample = ["the election government policy vote law new"]
    proba = clf.predict_proba(sample)
    pred = clf.predict(sample)
    logger.info("Sample predict_proba:\n%s", proba)
    logger.info("Sample predict (multi-hot):\n%s", pred)

    metrics = clf.evaluate(X_test, y_test)
    _print_metrics(f"PretrainedEncoderClassifier[{model_name}+{classifier_head}]", metrics)


def run_lora(
    num_labels: int,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    peft_method: str = "lora",
    **kwargs,
) -> None:
    """Demonstrate LoRASLMClassifier (LoRA or IA³)."""
    from models import LoRASLMClassifier

    X_train, y_train, X_val, y_val, X_test, y_test = make_dummy_data(
        num_labels=num_labels
    )
    clf = LoRASLMClassifier(
        num_labels=num_labels,
        model_name=model_name,
        peft_method=peft_method,
        num_epochs=2,
        batch_size=2,
        accumulation_steps=8,
    )
    clf.fit(X_train, y_train, X_val, y_val)

    sample = ["hospital patient medical doctor disease health"]
    proba = clf.predict_proba(sample)
    pred = clf.predict(sample)
    logger.info("Sample predict_proba:\n%s", proba)
    logger.info("Sample predict (multi-hot):\n%s", pred)

    metrics = clf.evaluate(X_test, y_test)
    _print_metrics(f"LoRASLMClassifier[{model_name}+{peft_method}]", metrics)


def _print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{'=' * 55}")
    print(f"  Model : {name}")
    print(f"{'=' * 55}")
    print(f"  Micro-F1        : {metrics['micro_f1']:.4f}")
    print(f"  Macro-F1        : {metrics['macro_f1']:.4f}")
    print(f"  Subset Accuracy : {metrics['subset_accuracy']:.4f}")
    print(f"{'=' * 55}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLTC Model Pool — end-to-end demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--classifier",
        choices=["tfidf", "neural", "encoder", "lora", "all"],
        default="tfidf",
        help="Which classifier to demonstrate.",
    )
    # TFIDFClassifier options
    parser.add_argument(
        "--clf_type",
        choices=["svm", "logistic_regression"],
        default="svm",
        help="TFIDFClassifier backend: 'svm' (LinearSVC) or 'logistic_regression'.",
    )
    # NeuralClassifier options
    parser.add_argument(
        "--variant",
        choices=["textcnn", "bilstm"],
        default="textcnn",
        help="Architecture variant for NeuralClassifier.",
    )
    parser.add_argument(
        "--head_type",
        choices=["linear", "cosine"],
        default="linear",
        help="Output head for NeuralClassifier: 'linear' (MLP) or 'cosine' (prototype).",
    )
    # PretrainedEncoderClassifier options
    parser.add_argument(
        "--classifier_head",
        choices=["mlp", "xgboost"],
        default="mlp",
        help="Output head for PretrainedEncoderClassifier.",
    )
    # Shared (encoder / lora)
    parser.add_argument(
        "--model_name",
        default=None,
        help=(
            "HuggingFace model identifier for encoder/lora classifiers. "
            "Defaults: encoder→roberta-base, lora→meta-llama/Meta-Llama-3-8B."
        ),
    )
    # LoRASLMClassifier options
    parser.add_argument(
        "--peft_method",
        choices=["lora", "ia3"],
        default="lora",
        help="PEFT adapter type for LoRASLMClassifier.",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=6,
        help="Number of output labels (max 6 for dummy data).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_labels = min(args.num_labels, 6)

    if args.classifier == "all":
        # Run all CPU-compatible combinations
        run_tfidf(num_labels, clf_type="svm")
        run_tfidf(num_labels, clf_type="logistic_regression")
        run_neural(num_labels, variant="textcnn", head_type="linear")
        run_neural(num_labels, variant="textcnn", head_type="cosine")
        run_neural(num_labels, variant="bilstm", head_type="linear")
        run_neural(num_labels, variant="bilstm", head_type="cosine")
        return

    if args.classifier == "tfidf":
        run_tfidf(num_labels, clf_type=args.clf_type)

    elif args.classifier == "neural":
        run_neural(num_labels, variant=args.variant, head_type=args.head_type)

    elif args.classifier == "encoder":
        model_name = args.model_name or "roberta-base"
        run_encoder(num_labels, model_name=model_name, classifier_head=args.classifier_head)

    elif args.classifier == "lora":
        model_name = args.model_name or "meta-llama/Meta-Llama-3-8B"
        run_lora(num_labels, model_name=model_name, peft_method=args.peft_method)


if __name__ == "__main__":
    main()
