"""Public fixed-axis scoring and explicit tasks; no pretrained downloads."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from loris.models.base import BaseDocumentClassifier, fixed_axis_metrics
from loris.models.tfidf_classifier import TFIDFClassifier


TEXTS = ["red apple fruit", "red cherry fruit", "red berry fruit",
         "blue sky cloud", "blue ocean water", "blue sea water"]
ONEHOT = np.asarray([[1, 0]] * 3 + [[0, 1]] * 3, np.float32)


class FixedPredictions(BaseDocumentClassifier):
    def fit(self, *args):
        return self

    def predict(self, texts):
        return self.predictions

    def predict_proba(self, texts):
        return self.predictions.astype(np.float32)


@pytest.mark.parametrize("truth,prediction,micro,macro,subset", [
    ([[1], [0]], [[1], [1]], 2 / 3, 2 / 3, .5),
    ([[0], [0]], [[0], [0]], 0., 0., 1.),
    ([[1], [1]], [[1], [1]], 1., 1., 1.),
    ([[1], [0]], [[0], [0]], 0., 0., .5),
    ([[1, 0], [0, 1]], [[1, 1], [0, 1]], .8, 5 / 6, .5),
])
def test_all_three_public_f1_entry_points_keep_the_label_axis(truth, prediction, micro, macro, subset):
    y, p = np.asarray(truth), np.asarray(prediction)
    model = FixedPredictions(y.shape[1])
    model.predictions, model.is_fitted = p, True
    scores = model.evaluate(["text"] * len(y), y)
    assert scores == pytest.approx(dict(micro_f1=micro, macro_f1=macro, subset_accuracy=subset))

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from loris.models.neural_classifier import NeuralClassifier
    from loris.models.pretrained_encoder_classifier import PretrainedEncoderClassifier

    class Logits(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            out = torch.tensor(p * 2 - 1, dtype=torch.float32)[input_ids[:, 0]]
            return out if attention_mask is None else SimpleNamespace(logits=out)

    ids = torch.arange(len(y)).reshape(-1, 1)
    neural = NeuralClassifier(y.shape[1])
    neural.device, neural.model = torch.device("cpu"), Logits()
    assert neural._evaluate_loader([(ids, torch.tensor(y))], y) == pytest.approx(micro)
    encoder = PretrainedEncoderClassifier(y.shape[1], use_amp=False)
    encoder.device, encoder.model = torch.device("cpu"), Logits()
    loader = [dict(input_ids=ids, attention_mask=torch.ones_like(ids))]
    assert encoder._val_micro_f1_mlp(loader, y) == pytest.approx(micro)


@pytest.mark.parametrize("truth,prediction", [
    ([[1]], [[1, 0]]), ([[.5]], [[1]]), ([[1]], [[2]]), ([], []),
])
def test_metrics_reject_invalid_axes_and_nonbinary_values(truth, prediction):
    with pytest.raises(ValueError, match="binary document-by-label"):
        fixed_axis_metrics(truth, prediction)


@pytest.mark.parametrize("method", ["svm", "logistic_regression"])
@pytest.mark.parametrize("task_type", ["multilabel", "single_label"])
def test_tfidf_task_is_explicit_and_survives_serialization(method, task_type):
    from sklearn.multiclass import OneVsRestClassifier

    model = TFIDFClassifier(2, classifier_type=method, min_df=1, n_jobs=1, task_type=task_type)
    model.fit(TEXTS, ONEHOT, [], np.zeros((0, 2)))
    assert model._single_label is (task_type == "single_label")
    assert isinstance(model.classifier, OneVsRestClassifier) is not (
        method == "logistic_regression" and task_type == "single_label"
    )
    restored = pickle.loads(pickle.dumps(model))
    assert restored.task_type == restored.config["task_type"] == task_type
    np.testing.assert_array_equal(restored.predict(TEXTS), model.predict(TEXTS))
    np.testing.assert_allclose(restored.predict_proba(TEXTS), model.predict_proba(TEXTS))
    if task_type == "single_label":
        assert (model.predict(TEXTS).sum(axis=1) == 1).all()


def test_onehot_rows_do_not_force_single_label_decoding():
    # Real LR sees indistinguishable documents with each class observed equally.
    # The default must retain independent binary estimators, even for one-hot y.
    texts = ["same shared text"] * 6
    multi = TFIDFClassifier(2, classifier_type="logistic_regression", min_df=1, n_jobs=1)
    single = TFIDFClassifier(2, classifier_type="logistic_regression", min_df=1, n_jobs=1,
                             task_type="single_label")
    for model in (multi, single):
        model.fit(texts, ONEHOT, [], np.zeros((0, 2)))
    assert not multi._single_label and single._single_label
    np.testing.assert_allclose(multi.predict_proba(texts), .5)
    # sklearn's LR predict uses its strict score cutoff; this checks the model
    # structure rather than incorrectly claiming that LR predict uses >= .5.
    assert len(multi.classifier.estimators_) == 2
    assert (multi.predict(texts).sum(axis=1) == 0).all()
    assert (single.predict(texts).sum(axis=1) == 1).all()


@pytest.mark.parametrize("kind", ["tfidf", "neural", "encoder"])
def test_invalid_single_label_targets_rejected_before_allocating_a_model(kind):
    if kind == "tfidf":
        cls, kwargs = TFIDFClassifier, dict(min_df=1, n_jobs=1)
    else:
        pytest.importorskip("torch")
        if kind == "neural":
            from loris.models.neural_classifier import NeuralClassifier as cls
        else:
            pytest.importorskip("transformers")
            from loris.models.pretrained_encoder_classifier import PretrainedEncoderClassifier as cls
        kwargs = {}
    with pytest.raises(ValueError, match="automatic inference is unsupported"):
        cls(2, task_type="auto", **kwargs)
    model = cls(2, task_type="single_label", **kwargs)
    for invalid in (np.ones((6, 2)), np.zeros((6, 2))):
        with pytest.raises(ValueError, match="exactly one positive"):
            model.fit(TEXTS, invalid, TEXTS, ONEHOT)
        assert not model.is_fitted


@pytest.mark.parametrize("method", ["svm", "logistic_regression"])
def test_historical_fitted_objects_keep_prediction_mode_but_cannot_implicitly_refit(method):
    model = TFIDFClassifier(2, classifier_type=method, min_df=1, n_jobs=1,
                            task_type="single_label")
    model.fit(TEXTS, ONEHOT, [], np.zeros((0, 2)))
    del model.task_type
    del model.config["task_type"]
    expected = model.predict(TEXTS)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(TEXTS), expected)
    with pytest.raises(ValueError, match="historical model requires an explicit task_type"):
        restored.fit(TEXTS, ONEHOT, [], np.zeros((0, 2)))


@pytest.mark.parametrize("task_type", ["multilabel", "single_label"])
def test_real_public_neural_training_respects_task_and_roundtrip(task_type):
    torch = pytest.importorskip("torch")
    from loris.models.neural_classifier import NeuralClassifier
    from loris.models.utils import texts_to_ids

    torch.set_num_threads(1)
    torch.manual_seed(41)
    model = NeuralClassifier(2, embed_dim=4, hidden_dim=8, num_filters=2, filter_sizes=(1,),
                             max_len=5, min_freq=1, num_epochs=1, batch_size=2,
                             dropout=0., task_type=task_type)
    model.device = torch.device("cpu")
    model.fit(TEXTS, ONEHOT, TEXTS, ONEHOT)
    assert model._single_label is (task_type == "single_label")
    model.model.eval()
    with torch.no_grad():
        logits = model.model(torch.from_numpy(texts_to_ids(TEXTS, model.vocab, model.max_len)))
        expected = torch.softmax(logits, dim=1) if model._single_label else torch.sigmoid(logits)
    np.testing.assert_allclose(model.predict_proba(TEXTS), expected.numpy(), rtol=1e-6)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.task_type == task_type
    np.testing.assert_array_equal(restored.predict(TEXTS), model.predict(TEXTS))


@pytest.mark.parametrize("constant", [0, 1])
def test_public_xgboost_one_constant_label_needs_no_encoder(monkeypatch, constant):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from loris.models.pretrained_encoder_classifier import PretrainedEncoderClassifier
    from transformers import AutoTokenizer, AutoModel

    def forbidden(*args, **kwargs):
        raise AssertionError("constant columns do not need pretrained weights or tokens")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(AutoModel, "from_pretrained", forbidden)
    model = PretrainedEncoderClassifier(1, classifier_head="xgboost", model_name="must_not_load")
    model.fit(TEXTS, np.full((6, 1), constant), [], np.zeros((0, 1)))
    assert model.xgb_clf is None and model.encoder is None
    assert model._xgb_columns == () and model._xgb_constants == {0: float(constant)}
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(TEXTS), np.full((6, 1), constant))
    np.testing.assert_array_equal(restored.predict(TEXTS), np.full((6, 1), constant))


@pytest.mark.parametrize("task_type", ["multilabel", "single_label"])
def test_real_encoder_xgboost_constant_columns_and_roundtrip(tiny_encoder_checkpoint, monkeypatch, task_type):
    import torch
    pytest.importorskip("xgboost")
    import sklearn.multioutput
    from loris.models.pretrained_encoder_classifier import PretrainedEncoderClassifier

    original = sklearn.multioutput.MultiOutputClassifier
    # Only reduce parallelism in this test. Both transformers and XGB estimators
    # are real; no predictions or training calls are mocked.
    monkeypatch.setattr(sklearn.multioutput, "MultiOutputClassifier",
                        lambda estimator, n_jobs: original(estimator, n_jobs=1))
    targets = (np.column_stack([np.zeros(6), ONEHOT[:, 0], np.ones(6), ONEHOT[:, 1]])
               if task_type == "multilabel" else np.column_stack([np.zeros(6), ONEHOT]))
    model = PretrainedEncoderClassifier(targets.shape[1], model_name=str(tiny_encoder_checkpoint),
        classifier_head="xgboost", max_length=8, pred_batch_size=2,
        xgb_n_estimators=2, xgb_max_depth=1, task_type=task_type)
    model.device = torch.device("cpu")
    model.fit(TEXTS, targets, [], np.zeros((0, targets.shape[1])))
    monkeypatch.undo()  # Restore the class binding before the real pickle roundtrip.
    p = model.predict_proba(TEXTS)
    assert p.shape == targets.shape and np.isfinite(p).all()
    assert (p[:, 0] == 0).all()
    assert model._xgb_columns == ((1, 3) if task_type == "multilabel" else (1, 2))
    assert len(model.xgb_clf.estimators_) == 2
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    varying_expected = model.xgb_clf.predict_proba(model._extract_cls_embeddings(TEXTS))
    for column, probabilities in zip(model._xgb_columns, varying_expected):
        np.testing.assert_allclose(p[:, column], probabilities[:, 1])
    if task_type == "multilabel":
        assert (p[:, 2] == 1).all()
    else:
        assert (model.predict(TEXTS).sum(axis=1) == 1).all()
    restored = pickle.loads(pickle.dumps(model))
    assert restored.task_type == task_type
    np.testing.assert_array_equal(restored.predict_proba(TEXTS), p)
    np.testing.assert_array_equal(restored.predict(TEXTS), model.predict(TEXTS))


def test_historical_xgboost_zero_class_probability_is_restored(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    xgb = pytest.importorskip("xgboost")
    from sklearn.multioutput import MultiOutputClassifier
    from loris.models.pretrained_encoder_classifier import PretrainedEncoderClassifier

    # Reproduce the old stored object: every label has an estimator, including
    # an all-zero estimator whose XGBoost 1.7 probability output is nonzero.
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    targets = np.column_stack([np.zeros(6), ONEHOT[:, 0]])
    model = PretrainedEncoderClassifier(2, classifier_head="xgboost")
    model.xgb_clf = MultiOutputClassifier(xgb.XGBClassifier(n_estimators=2, max_depth=1,
        n_jobs=1, eval_metric="logloss", verbosity=0), n_jobs=1).fit(features, targets)
    model.device, model.is_fitted = torch.device("cpu"), True
    del model.task_type
    del model.config["task_type"]
    monkeypatch.setattr(model, "_extract_cls_embeddings", lambda texts: features)
    p = model.predict_proba(TEXTS)
    assert p.shape == (6, 2) and (p[:, 0] == 0).all()
    np.testing.assert_allclose(p[:, 1], model.xgb_clf.predict_proba(features)[1][:, 1])


@pytest.fixture(scope="module")
def tiny_encoder_checkpoint(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertModel, BertTokenizerFast

    torch.set_num_threads(1)
    torch.manual_seed(43)
    path = tmp_path_factory.mktemp("public_api_tiny_bert")
    (path / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nred\nblue\nfruit\nsky\n")
    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    BertModel(BertConfig(vocab_size=9, hidden_size=8, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=16,
                        max_position_embeddings=32)).save_pretrained(path)
    return path


@pytest.mark.parametrize("task_type", ["multilabel", "single_label"])
def test_real_public_encoder_mlp_training_respects_task_and_roundtrip(tiny_encoder_checkpoint, task_type):
    import torch
    from loris.models.pretrained_encoder_classifier import PretrainedEncoderClassifier

    torch.manual_seed(47)
    model = PretrainedEncoderClassifier(2, model_name=str(tiny_encoder_checkpoint),
        max_length=8, batch_size=2, pred_batch_size=2, num_epochs=1, use_amp=False,
        task_type=task_type)
    model.device = torch.device("cpu")
    model.fit(TEXTS, ONEHOT, TEXTS, ONEHOT)
    assert model.model.config.problem_type == (
        "single_label_classification" if task_type == "single_label" else "multi_label_classification"
    )
    restored = pickle.loads(pickle.dumps(model))
    assert restored.task_type == task_type
    np.testing.assert_array_equal(restored.predict(TEXTS), model.predict(TEXTS))
    with torch.no_grad():
        encoded = model._tokenise(TEXTS)
        logits = model.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits
        expected = torch.softmax(logits, dim=1) if model._single_label else torch.sigmoid(logits)
    np.testing.assert_allclose(model.predict_proba(TEXTS), expected.numpy(), rtol=1e-6)
