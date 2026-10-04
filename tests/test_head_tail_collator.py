"""Actual offline tokenizer/collator budgets include required special tokens."""

import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def collator_class():
    pytest.importorskip("torch")
    source = os.environ.get("LORIS_SSL_SOURCE")
    if source:
        from loris.labeling.trainer import load_backend
        load_backend(source)
    module = pytest.importorskip("loris_ssl.data")
    return module.TextBatchCollator


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertTokenizerFast

    vocabulary = {token: i for i, token in enumerate(
        ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijkl")
    )}
    backend = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer()
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])],
    )
    return BertTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                             cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


@pytest.mark.parametrize("max_length", [0, 1])
def test_negative_content_budget_is_rejected(collator_class, tokenizer, max_length):
    with pytest.raises(ValueError, match="required special tokens"):
        collator_class(tokenizer, max_length)([{"text": "a b c"}])


@pytest.mark.parametrize("text", ["", "a", "a b c d e"])
def test_zero_content_budget_retains_only_required_specials(collator_class, tokenizer, text):
    result = collator_class(tokenizer, 2)([{"text": text}])
    assert result["input_ids"].shape == result["attention_mask"].shape == (1, 2)
    assert tokenizer.convert_ids_to_tokens(result["input_ids"][0]) == ["[CLS]", "[SEP]"]
    assert result["attention_mask"].all()


@pytest.mark.parametrize("max_length, expected", [
    (3, ["a"]),
    (4, ["a", "b"]),
    (5, ["a", "b", "l"]),
    (6, ["a", "b", "c", "l"]),
    (10, ["a", "b", "c", "d", "e", "f", "k", "l"]),
])
def test_zero_tail_and_normal_three_to_one_truncation(collator_class, tokenizer, max_length, expected):
    result = collator_class(tokenizer, max_length)([{"text": "a b c d e f g h i j k l"}])
    assert result["input_ids"].shape == (1, max_length)
    assert tokenizer.convert_ids_to_tokens(result["input_ids"][0]) == ["[CLS]"] + expected + ["[SEP]"]


def test_short_and_empty_texts_pad_without_duplication_or_label_changes(collator_class, tokenizer):
    result = collator_class(tokenizer, 384)([
        {"text": "a b", "labels": np.array([1., 0.], np.float32)},
        {"text": "", "labels": np.array([0., 1.], np.float32)},
    ])
    assert result["input_ids"].shape == (2, 4)
    assert result["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert tokenizer.convert_ids_to_tokens(result["input_ids"][0]) == ["[CLS]", "a", "b", "[SEP]"]
    assert tokenizer.convert_ids_to_tokens(result["input_ids"][1]) == ["[CLS]", "[SEP]", "[PAD]", "[PAD]"]
    assert result["labels"].tolist() == [[1., 0.], [0., 1.]]


@pytest.mark.parametrize("max_length, head, tail", [(384, 286, 96), (512, 382, 128)])
def test_usual_limits_preserve_actual_long_text_head_and_tail(
    collator_class, tokenizer, max_length, head, tail
):
    text = " ".join(["a"] * 600 + ["b"] * 600)
    result = collator_class(tokenizer, max_length)([{"text": text}])
    assert result["input_ids"].shape == (1, max_length)
    assert result["attention_mask"].sum().item() == max_length
    assert tokenizer.convert_ids_to_tokens(result["input_ids"][0]) == (
        ["[CLS]"] + ["a"] * head + ["b"] * tail + ["[SEP]"]
    )
