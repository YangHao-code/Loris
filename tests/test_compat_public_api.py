"""Small real-model regressions for the release's public compatibility boundaries."""
import numpy as np
import pytest

from loris.document import Document
from loris.models import embeddings
from loris.models.neural_classifier import NeuralClassifier, _BiLSTM
from loris.models.tfidf_classifier import TFIDFClassifier


TEXTS = ["red apple fruit", "red cherry fruit", "red berry fruit",
         "blue sky cloud", "blue ocean water", "blue sea water"]


def fit(method, targets, **kwargs):
    targets = np.asarray(targets)
    return TFIDFClassifier(
        targets.shape[1], classifier_type=method, min_df=1, n_jobs=1, **kwargs
    ).fit(TEXTS, targets, [], np.zeros((0, targets.shape[1])))


@pytest.mark.parametrize("method", ["svm", "logistic_regression"])
@pytest.mark.parametrize("target", [0, 1, "mixed"])
def test_one_label_axis_and_constant_probabilities(method, target):
    targets = np.asarray([[i >= 3 if target == "mixed" else target] for i in range(6)])
    model = fit(method, targets)
    p, y = model.predict_proba(TEXTS), model.predict(TEXTS)
    assert p.shape == y.shape == (6, 1)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    if target != "mixed":
        assert (p == target).all() and (y == target).all()


@pytest.mark.parametrize("method", ["svm", "logistic_regression"])
def test_multilabel_constant_columns_keep_their_declared_positions(method):
    model = fit(method, [[0, 1, i >= 3] for i in range(6)])
    p, y = model.predict_proba(TEXTS), model.predict(TEXTS)
    assert p.shape == y.shape == (6, 3)
    assert (p[:, 0] == 0).all() and (p[:, 1] == 1).all()
    assert (y[:, 0] == 0).all() and (y[:, 1] == 1).all()
    assert ((0 < p[:, 2]) & (p[:, 2] < 1)).all()


def test_single_label_lr_restores_missing_class_columns():
    model = fit("logistic_regression", [[1, 0, 0]] * 3 + [[0, 0, 1]] * 3,
                task_type="single_label")
    p = model.predict_proba(TEXTS)
    assert list(model.classifier.classes_) == [0, 2]
    assert p.shape == (6, 3) and (p[:, 1] == 0).all()
    expected = model.classifier.predict_proba(model.vectorizer.transform(TEXTS))
    np.testing.assert_allclose(p[:, [0, 2]], expected)
    assert model.predict(TEXTS).shape == (6, 3)


def test_single_observed_lr_class_has_exact_probability():
    model = fit("logistic_regression", [[0, 0, 1]] * 6)
    expected = np.tile([0, 0, 1], (6, 1))
    np.testing.assert_array_equal(model.predict_proba(TEXTS), expected)
    np.testing.assert_array_equal(model.predict(TEXTS), expected)


def test_calibrated_svm_also_preserves_single_label_axis():
    model = fit("svm", [[i >= 3] for i in range(6)], use_calibration=True)
    p = model.predict_proba(TEXTS)
    assert p.shape == model.predict(TEXTS).shape == (6, 1)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()


@pytest.mark.parametrize("method", ["svm", "logistic_regression"])
@pytest.mark.parametrize("constant", [0, 1])
def test_main_fixed_axis_one_constant_column_uses_original_binary_targets(method, constant):
    from loris.discovery.model_pool import FixedAxisTFIDF
    model = FixedAxisTFIDF(num_labels=1, classifier_type=method, max_features=50,
        ngram_range=[1, 1], min_df=1, C=1., use_calibration=False, max_iter=2000, n_jobs=1)
    model.fit(TEXTS, np.full((6, 1), constant, np.int32))
    np.testing.assert_array_equal(model.predict_proba(TEXTS), np.full((6, 1), constant))
    del model.constant_labels  # Historical fitted-object layout.
    np.testing.assert_array_equal(model.predict_proba(TEXTS), np.full((6, 1), constant))


def test_lr_constructor_without_removed_keyword(monkeypatch):
    import sklearn.linear_model
    import loris.models.tfidf_classifier as public
    from loris.discovery.model_pool import FixedAxisTFIDF

    original = sklearn.linear_model.LogisticRegression
    calls = []

    def newer_signature(*, C, max_iter, solver):
        calls.append((C, max_iter, solver))
        return original(C=C, max_iter=max_iter, solver=solver)

    # This checks compatibility with the removed keyword, not a full sklearn
    # version matrix: the actual optimization still uses installed sklearn.
    monkeypatch.setattr(sklearn.linear_model, "LogisticRegression", newer_signature)
    monkeypatch.setattr(public, "LogisticRegression", newer_signature)
    y = np.asarray([[1, 0, 0]] * 3 + [[0, 0, 1]] * 3)
    assert fit("logistic_regression", y, task_type="single_label").predict_proba(TEXTS).shape == (6, 3)
    main = FixedAxisTFIDF(num_labels=3, classifier_type="logistic_regression",
        max_features=50, ngram_range=[1, 1], min_df=1, C=1., use_calibration=False,
        max_iter=2000, n_jobs=1).fit(TEXTS, y)
    assert main.predict_proba(TEXTS).shape == (6, 3)
    assert len(calls) == 2


@pytest.mark.parametrize("targets", [np.zeros((6, 2)), np.full((6, 3), .5)])
def test_public_tfidf_rejects_wrong_or_nonbinary_target_axis(targets):
    model = TFIDFClassifier(3, min_df=1, n_jobs=1)
    with pytest.raises(ValueError, match="declared label axis"):
        model.fit(TEXTS, targets, [], np.zeros((0, 3)))
    assert model.vectorizer is None and not model.is_fitted


def test_none_metadata_cannot_connect_documents_via_equality():
    from loris._engine.rdl_reference import (
        ReferenceChase, RefGamma, Rule, Doc, AttrEqual, Label, prepare_attributes,
    )
    docs = [Document.from_dict(dict(cnt=t, mtd=None, ttl=None)) for t in ("left", "right")]
    assert [d.mtd for d in docs] == ["", ""]
    assert not prepare_attributes(docs)[1]["mtd"].any()
    gamma = RefGamma(np.array([[1], [0]], bool), np.zeros((2, 1), bool))
    rule = Rule((Doc("x"), Doc("y"), AttrEqual("x", "mtd", "y", "mtd"),
                 Label("x", "A")), Label("y", "A"))
    closure = ReferenceChase(["A"]).run(docs, [rule], gamma)
    assert closure.positive_facts.tolist() == [[True], [False]]
    assert Document.from_dict({"cnt": None}).cnt == ""
    assert Document.from_dict({"cnt": "text", "mtd": "None"}).mtd == "None"


@pytest.mark.parametrize("path_suffix", [".npy", ""])
def test_existing_unbound_embedding_cache_is_rejected_before_encoder(tmp_path, monkeypatch, path_suffix):
    path = tmp_path / "old.npy"
    np.save(path, np.array([[2., 0.], [0., 3.]], np.float32))
    original = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before loading/downloading an encoder")

    monkeypatch.setattr(embeddings, "get_sentence_encoder", forbidden)
    with pytest.raises(ValueError, match="reuse is disabled"):
        embeddings.encode_texts(["different", "texts"], normalize=True,
                                cache_path=tmp_path / ("old" + path_suffix))
    assert path.read_bytes() == original


def test_fresh_embedding_export_preserves_explicit_encoding_settings(tmp_path, monkeypatch):
    calls = []

    class Encoder:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.eye(2)

    monkeypatch.setattr(embeddings, "get_sentence_encoder", lambda *args: Encoder())
    result = embeddings.encode_texts(["a", "b"], normalize=True,
                                    cache_path=tmp_path / "new")
    assert calls[0][1]["normalize_embeddings"] is True
    np.testing.assert_array_equal(np.load(tmp_path / "new.npy"), result)
    assert result.dtype == np.float32


@pytest.mark.parametrize("dimension", [0, -2, 7, 1, 8., True])
def test_bilstm_rejects_invalid_total_hidden_size_at_both_entry_points(dimension):
    with pytest.raises(ValueError, match="positive even"):
        NeuralClassifier(2, variant="bilstm", hidden_dim=dimension)
    with pytest.raises(ValueError, match="positive even"):
        _BiLSTM(8, 4, dimension, 2)


def test_valid_bilstm_shape_and_textcnn_independent_parameter_remain_supported():
    import torch
    model = _BiLSTM(8, 4, 6, 2, dropout=0.)
    assert model(torch.tensor([[2, 3, 0], [4, 0, 0]])).shape == (2, 2)
    # hidden_dim is unused by TextCNN and should not gain a BiLSTM restriction.
    assert NeuralClassifier(2, variant="textcnn", hidden_dim=7).hidden_dim == 7
