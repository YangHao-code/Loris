"""The public probability API preserves thresholds, axes and model versions."""
import numpy as np
import pytest

from loris.document import Document
from loris.models.tfidf_classifier import TFIDFClassifier
from loris.predicates import (
    MLPredicate, MLThresholdPredicate, MLScoreBandPredicate,
    register_ml_model, predicate_to_dict, predicate_from_dict,
)
from loris.predicates import _core


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(_core, "_ML_MODEL_CACHE", {})
    monkeypatch.setattr(_core, "_ML_LABEL_AXES", {})
    monkeypatch.setattr(_core, "_ML_PRED_MEMO", {})


class BatchModel:
    def __init__(self, scores=(.6, .4)):
        self.scores = scores
        self.calls = []

    def predict(self, text):
        return "A"

    def predict_proba(self, texts):
        self.calls.append(texts)
        return np.tile(self.scores, (len(texts), 1))


def test_high_threshold_cannot_fall_back_to_argmax_and_band_shares_exact_score():
    model = BatchModel()
    register_ml_model("m", model, labels=["A", "B"])
    doc = Document(cnt="text")
    assert MLPredicate("m", "A")(doc)
    assert not MLThresholdPredicate("m", "A", .99)(doc)
    assert MLThresholdPredicate("m", "A", .6)(doc)
    assert MLScoreBandPredicate("m", "A", .6, .7)(doc)
    assert not MLScoreBandPredicate("m", "A", .5, .6)(doc)
    assert model.calls == [["text"]]
    predicate = MLThresholdPredicate("m", "B", .5)
    assert not predicate_from_dict(predicate_to_dict(predicate))(doc)


def test_single_document_protocol_and_reregistration_replace_scores_and_axis():
    class Single:
        def label_index(self, label):
            return {"A": 0, "B": 1}[label]

        def predict_proba_single(self, text):
            assert text == "text"
            return np.asarray([.8, .2], np.float32)

    doc = Document(cnt="text")
    register_ml_model("m", Single())
    assert MLThresholdPredicate("m", "A", .7)(doc)
    register_ml_model("m", BatchModel((.9, .1)), labels=["B", "A"])
    assert not MLThresholdPredicate("m", "A", .7)(doc)
    register_ml_model("m", Single())
    assert MLThresholdPredicate("m", "A", .7)(doc)


def test_real_public_tfidf_uses_list_input_and_declared_label_order():
    texts = ["red apple fruit", "red cherry fruit", "red berry fruit",
             "blue sky cloud", "blue ocean water", "blue sea water"]
    targets = np.asarray([[1, 0]] * 3 + [[0, 1]] * 3)
    model = TFIDFClassifier(2, classifier_type="svm", min_df=1, n_jobs=1)
    model.fit(texts, targets, [], np.zeros((0, 2)))
    register_ml_model("real", model, labels=["fruit", "blue"])
    p = model.predict_proba([texts[0]])[0]
    for index, label in enumerate(["fruit", "blue"]):
        for threshold in (0., .5, .99, 1.):
            assert MLThresholdPredicate("real", label, threshold)(Document(cnt=texts[0])) == (
                float(p[index]) >= threshold
            )


@pytest.mark.parametrize("values", [[], [.5], [.5, np.nan], [.5, np.inf], [-.1, .5], [.5, 1.1]])
def test_invalid_probability_rows_are_rejected_without_populating_cache(values):
    register_ml_model("bad", BatchModel(values), labels=["A", "B"])
    with pytest.raises(ValueError, match="probabilities"):
        MLThresholdPredicate("bad", "A")(Document(cnt="text"))
    assert not _core._ML_PRED_MEMO


def test_missing_axis_missing_probabilities_and_malformed_batch_are_explicit():
    register_ml_model("no_axis", BatchModel())
    with pytest.raises(ValueError, match="registered labels"):
        MLThresholdPredicate("no_axis", "A")(Document(cnt="text"))

    class Argmax:
        def predict(self, text):
            return "A"

    register_ml_model("argmax", Argmax(), labels=["A"])
    with pytest.raises(RuntimeError, match="argmax cannot"):
        MLThresholdPredicate("argmax", "A")(Document(cnt="text"))

    class WrongBatch:
        def predict_proba(self, texts):
            return np.asarray([.6, .4])

    register_ml_model("shape", WrongBatch(), labels=["A", "B"])
    with pytest.raises(ValueError, match="shape"):
        MLThresholdPredicate("shape", "A")(Document(cnt="text"))


@pytest.mark.parametrize("index", [-1, 2, .5, True])
def test_invalid_native_label_index_is_not_used_for_numpy_indexing(index):
    class Native(BatchModel):
        def label_index(self, label):
            return index

    register_ml_model("native", Native())
    with pytest.raises(ValueError, match="label_index"):
        MLThresholdPredicate("native", "A")(Document(cnt="text"))


@pytest.mark.parametrize("axis", [[], ["A", "A"], ["A", ""], "AB",
                                  {"A", "B"}, frozenset({"A", "B"}), {"A": 0, "B": 1}])
def test_invalid_registered_axes_are_rejected(axis):
    with pytest.raises(ValueError, match="labels"):
        register_ml_model("m", BatchModel(), labels=axis)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -.1, 1.1])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        MLThresholdPredicate("m", "A", threshold)
