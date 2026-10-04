"""Dataset-independent real seven-model pool with a frozen supervision contract.

Architecture/budget defaults follow Reuters train_pool_v2. Native neural
checkpoint selection is the declared author supplement: minimum development
BCE, while fixed-.5 F1 remains a reported diagnostic. No unknown/test targets
are available through the supervision loader and no task checkpoint is reused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import contextlib
import fcntl
import gc
import importlib
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import resource
import sys
import time
from types import SimpleNamespace
import warnings

import numpy as np

from .manifest import file_record, read_json, bound
from loris._engine.rlearner import atomic_json, immutable_json, digest

NATIVE = (
    "tfidf_svm_unigram",
    "tfidf_svm_bigram",
    "tfidf_lr_bigram",
    "textcnn",
    "bilstm",
    "encoder_mlp",
)
MODELS = NATIVE + ("aux_dynamic",)
SCHEMA = "loris_real7_supervised_pool_dev_bce_v1"


@dataclass(frozen=True)
class PoolSettings:
    seed: int = 42
    device: str = "cuda"
    cpu_threads: int = 4
    linear_max_iter: int = 2000
    neural_epochs: int = 60
    neural_min_epochs: int = 10
    neural_patience: int = 6
    encoder_epochs: int = 24
    encoder_min_epochs: int = 8
    encoder_patience: int = 4
    aux_epochs: int = 16
    aux_min_epochs: int = 10
    aux_patience: int = 5
    max_phase_seconds: int = 14400

    def validate(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("uint32 seed required")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must explicitly be cpu or cuda")
        for name, value in asdict(self).items():
            if name not in ("seed", "device") and (type(value) is not int or value < 1):
                raise ValueError("positive integer " + name + " required")
        for family in ("neural", "encoder", "aux"):
            if getattr(self, family + "_min_epochs") > getattr(self, family + "_epochs"):
                raise ValueError(family + " minimum epochs exceeds cap")


def _backend(source):
    """Load the explicitly supplied real optional package; never substitute it."""
    root = Path(source).resolve(strict=True)
    if not (root / "loris_ssl/chase_backend.py").is_file():
        raise ValueError("backend_source must contain loris_ssl/chase_backend.py")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("loris_ssl.chase_backend")
    if Path(module.__file__).resolve() != root / "loris_ssl/chase_backend.py":
        raise ValueError("loaded loris_ssl comes from a different source than declared")
    return module


def model_configs(labels, settings, pretrained_model, backend_source):
    settings.validate()
    L = len(labels)
    if not L or len(set(labels)) != L:
        raise ValueError("unique nonempty labels required")
    result = {}
    for name in NATIVE:
        if name.startswith("tfidf"):
            result[name] = dict(
                num_labels=L,
                classifier_type="logistic_regression" if "_lr_" in name else "svm",
                max_features=50000,
                ngram_range=[1, 1] if "unigram" in name else [1, 2],
                min_df=2,
                C=1.0,
                use_calibration=False,
                max_iter=settings.linear_max_iter,
                n_jobs=settings.cpu_threads,
            )
        elif name == "encoder_mlp":
            result[name] = dict(
                num_labels=L,
                model_name=str(pretrained_model),
                max_length=512,
                batch_size=16,
                pred_batch_size=64,
                num_epochs=settings.encoder_epochs,
                lr=2e-5,
                warmup_ratio=0.1,
                weight_decay=0.01,
                use_amp=settings.device == "cuda",
                patience=settings.encoder_patience,
                gradient_checkpointing=True,
            )
        else:
            result[name] = dict(
                num_labels=L,
                variant=name,
                max_vocab_size=15000,
                min_freq=2,
                max_len=256,
                embed_dim=64,
                hidden_dim=128,
                num_filters=64 if name == "textcnn" else 128,
                num_layers=2,
                num_epochs=settings.neural_epochs,
                batch_size=32,
                lr=0.001,
                dropout=0.3,
                threshold=0.5,
            )
    cfg = _backend(backend_source).ChaseBackendConfig(
        model_name=str(pretrained_model),
        cache_dir=str(Path(pretrained_model).parent),
        offline=True,
        head="label_attention_linear",
        max_length=384,
        truncation="head_tail",
        projection_dim=128,
        loss="masked_weighted_bce",
        epochs=settings.aux_epochs,
        min_epochs=settings.aux_min_epochs,
        patience=settings.aux_patience,
        batch_size=8,
        prediction_batch_size=64,
        validation_batch_size=64,
        grad_accum_steps=2,
        learning_rate=2e-5,
        head_learning_rate=5e-5,
        seed=settings.seed,
        amp="bf16" if settings.device == "cuda" else "none",
    )
    result["aux_dynamic"] = cfg.to_dict()
    return result


def _source_records(backend_source):
    from loris.models import (
        neural_classifier,
        pretrained_encoder_classifier,
        tfidf_classifier,
        utils,
    )
    from . import supervision

    paths = [Path(__file__)] + [
        Path(m.__file__)
        for m in (
            neural_classifier,
            pretrained_encoder_classifier,
            tfidf_classifier,
            utils,
            supervision,
        )
    ]
    paths += sorted(Path(neural_classifier.__file__).parent.glob("*.py"))
    paths += sorted((Path(backend_source) / "loris_ssl").rglob("*.py"))
    return [file_record(p) for p in sorted(set(paths))]


def _environment():
    import torch, sklearn, transformers, scipy, platform

    return dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        transformers=transformers.__version__,
        scipy=scipy.__version__,
    )


def freeze_pool(
    supervision_manifest, output, *, pretrained_model, backend_source, settings=None, gpu_lock=None
):
    from .supervision import load_supervision

    settings = PoolSettings() if settings is None else settings
    if not isinstance(settings, PoolSettings):
        raise TypeError("PoolSettings instance required")
    settings.validate()
    D = load_supervision(supervision_manifest)
    model = Path(pretrained_model).resolve(strict=True)
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError("explicit local pretrained snapshot required")
    if not any((model / n).is_file() for n in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError("pretrained snapshot lacks real weights")
    backend_source = Path(backend_source).resolve(strict=True)
    config = model_configs(D["labels"], settings, model, backend_source)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        schema=SCHEMA,
        supervision=file_record(supervision_manifest),
        settings=asdict(settings),
        models=config,
        labels=D["labels"],
        row_ids=D["row_ids"],
        row_roles=D["row_roles"],
        model_order=["loris_clf_" + name for name in MODELS],
        actual_model_classes=dict(
            encoder_mlp="AutoModelForSequenceClassification, full pretrained encoder fine-tuning",
            aux_dynamic="loris_ssl.ChaseTransformerBackend, label_attention_linear",
            textcnn="loris.models.neural_classifier._TextCNN",
            bilstm="loris.models.neural_classifier._BiLSTM",
        ),
        environment=_environment(),
        fit_indices=np.asarray(D["fit_indices"]).tolist(),
        receiver_indices=np.asarray(D["receiver_indices"]).tolist(),
        sources=_source_records(backend_source),
        backend_source=str(backend_source),
        pretrained_snapshot=[file_record(p) for p in sorted(model.iterdir()) if p.is_file()],
        gpu_lock=str(Path(gpu_lock).resolve()) if gpu_lock is not None else None,
        checkpoint_selection=dict(
            native_neural="minimum dev BCE from logits, improvement > 1e-6, earliest tie",
            aux="existing real backend dev weighted BCE; student/EMA selection and tie policy unchanged",
        ),
        author_supplement="native neural checkpoint selection switches from dev Macro-F1@0.5 to BCE before seeing results",
        original_architecture="Reuters real native6 plus loris_ssl Aux7, no Strong12 or old task checkpoint substitution",
        initial_training="complete fit answers only; unknown train and dev texts inference only",
        dev_role="validation/checkpoint/calibration diagnostic, never gradient targets",
        test_read=False,
        unknown_train_truth_read=False,
        online_feedback_updates=0,
        linear_multilabel_contract="fixed-axis OvR even if the small fit subset happens to be one-hot; exact 0/1 constant estimators",
        calibration_grid=[i / 20 for i in range(21)],
        calibration_ties="closest .5 then larger eta",
        checkpoint_resume="completed model verification only; partial model training is preserved and rejected",
        timeout="hard per-phase elapsed budget includes lock wait, each model in terminable process",
        timing_scope="training_seconds includes setup/validation/checkpoint selection; inference_seconds only full declared text-axis prediction; no pure SGD time claim",
        exposure=D["manifest"].get("exposure", None),
    )
    immutable_json(out / "PROTOCOL.json", protocol)
    np.savez_compressed(
        out / "TRAINING_AXIS.npz",
        global_rows=np.asarray(D["fit_indices"]),
        row_ids=np.asarray(D["row_ids"])[D["fit_indices"]],
        targets=np.asarray(D["fit_targets"], np.float32),
        labels=np.asarray(D["labels"]),
    )
    immutable_json(
        out / "EXPOSURE.json",
        dict(
            supervision=D["manifest_record"],
            known_train_answers=len(D["fit_indices"]),
            known_signed_cells=len(D["fit_indices"]) * len(D["labels"]),
            dev_selection_answers=len(D["receiver_indices"]),
            unknown_train_text_only_rows=len(D["train_indices"]) - len(D["fit_indices"]),
            zero_known_positive_labels=int((np.asarray(D["fit_targets"]).sum(0) == 0).sum()),
            unknown_truth_read=False,
            test_read=False,
            old_task_checkpoint_reused=False,
            external_pretraining="bound local pretrained snapshot",
            historical_supervision="see bound supervision manifest/exposure",
        ),
    )
    return dict(
        status="frozen_not_trained",
        protocol=file_record(out / "PROTOCOL.json"),
        models=list(MODELS),
        shape=[len(D["docs"]), len(D["labels"])],
        fit_answers=len(D["fit_indices"]),
        new_training=0,
    )


def verify_pool(output):
    from .supervision import load_supervision

    out = Path(output).resolve(strict=True)
    P = read_json(out / "PROTOCOL.json")
    if P["schema"] != SCHEMA:
        raise ValueError("unsupported pool protocol")
    if P["environment"] != _environment():
        raise ValueError("pool environment differs from frozen runtime")
    if P["model_order"] != ["loris_clf_" + name for name in MODELS]:
        raise ValueError("declared model axis differs")
    for record in P["sources"] + P["pretrained_snapshot"] + [P["supervision"]]:
        bound(record, out)
    D = load_supervision(P["supervision"]["path"])
    S = PoolSettings(**P["settings"])
    S.validate()
    if (
        D["labels"] != P["labels"]
        or D["row_ids"] != P["row_ids"]
        or D["row_roles"] != P["row_roles"]
        or np.asarray(D["fit_indices"]).tolist() != P["fit_indices"]
        or np.asarray(D["receiver_indices"]).tolist() != P["receiver_indices"]
    ):
        raise ValueError("supervision axes changed after freeze")
    return P, D, S


def metrics(y, p):
    y = np.asarray(y, bool)
    p = np.asarray(p)
    if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probability axis/values differ")
    b = p >= 0.5
    tp = (y & b).sum(0)
    fp = (~y & b).sum(0)
    fn = (y & ~b).sum(0)
    den = 2 * tp + fp + fn
    f = np.divide(2.0 * tp, den, out=np.zeros(len(tp)), where=den > 0)
    return dict(
        macro_f1=float(f.mean()),
        micro_f1=float(2 * tp.sum() / max(1, den.sum())),
        TP=int(tp.sum()),
        FP=int(fp.sum()),
        FN=int(fn.sum()),
        coverage=float(b.any(1).mean()),
        rows=len(y),
        labels=y.shape[1],
        bce=float(
            -(
                y * np.log(np.clip(p, 1e-7, 1 - 1e-7))
                + (~y) * np.log(np.clip(1 - p, 1e-7, 1 - 1e-7))
            ).mean()
        ),
        per_label=dict(TP=tp.tolist(), FP=fp.tolist(), FN=fn.tolist(), f1=f.tolist()),
    )


class FixedAxisTFIDF:
    """Actual TF-IDF/OvR models, including classes absent from known fit."""

    def __init__(self, **config):
        if (
            config["classifier_type"] not in ("svm", "logistic_regression")
            or config["use_calibration"] is not False
        ):
            raise ValueError(
                "this protocol supports raw fixed-axis SVM/LR without probability calibration"
            )
        self.config = config

    def fit(self, texts, targets):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.multiclass import OneVsRestClassifier

        c = self.config
        y = np.asarray(targets, np.int32)
        if y.shape != (len(texts), c["num_labels"]) or not np.isin(y, (0, 1)).all():
            raise ValueError("fully observed fixed label axis required")
        self.constant_labels = {
            i: int(column[0]) for i, column in enumerate(y.T)
            if len(column) and np.all(column == column[0])
        }
        self.vectorizer = TfidfVectorizer(
            max_features=c["max_features"],
            ngram_range=tuple(c["ngram_range"]),
            min_df=c["min_df"],
            sublinear_tf=True,
        )
        x = self.vectorizer.fit_transform(texts)
        inner = (
            LogisticRegression(C=c["C"], max_iter=c["max_iter"], solver="lbfgs")
            if c["classifier_type"] == "logistic_regression"
            else LinearSVC(C=c["C"], max_iter=c["max_iter"])
        )
        self.classifier = OneVsRestClassifier(inner, n_jobs=c["n_jobs"]).fit(x, y)
        if len(self.classifier.estimators_) != c["num_labels"]:
            raise ValueError("OvR lost the declared label axis")
        return self

    def predict_proba(self, texts):
        from scipy.special import expit

        x = self.vectorizer.transform(texts)
        values = []
        constants = getattr(self, "constant_labels", {})
        if self.config["num_labels"] == 1 and len(self.classifier.classes_) == 1:
            # Historical fitted objects lack constant_labels, but sklearn still
            # retains the sole original class in classes_.
            constants = {0: int(self.classifier.classes_[0])}
        for i, estimator in enumerate(self.classifier.estimators_):
            if i in constants:
                values.append(np.full(len(texts), float(constants[i])))
            elif hasattr(estimator, "y_"):
                value = np.asarray(estimator.y_).reshape(-1)
                if len(value) != 1 or value[0] not in (0, 1):
                    raise ValueError("invalid constant binary estimator")
                values.append(np.full(len(texts), float(value[0])))
            elif self.config["classifier_type"] == "logistic_regression":
                values.append(estimator.predict_proba(x)[:, 1])
            else:
                values.append(expit(estimator.decision_function(x)))
        return np.stack(values, axis=1).astype(np.float32)


def _seed(seed, threads):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _neural_fit(name, config, D, directory, S):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from loris.models.neural_classifier import NeuralClassifier
    from loris.models.pretrained_encoder_classifier import PretrainedEncoderClassifier
    from loris.models.utils import build_vocab, texts_to_ids, multilabel_dataloader

    texts = [D["docs"][int(i)]["text"] for i in D["fit_indices"]]
    y = np.asarray(D["fit_targets"], np.float32)
    devtexts = [D["docs"][int(i)]["text"] for i in D["receiver_indices"]]
    yv = np.asarray(D["dev_targets"], np.float32)
    encoder = name == "encoder_mlp"
    device = torch.device(S.device)
    amp = encoder and device.type == "cuda"
    m = PretrainedEncoderClassifier(**config) if encoder else NeuralClassifier(**config)
    m._single_label = False
    m.device = device
    if encoder:
        m.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], local_files_only=True)
        m.model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=len(D["labels"]),
            problem_type="multi_label_classification",
            local_files_only=True,
        ).to(device)
        if config.get("gradient_checkpointing", True):
            m.model.gradient_checkpointing_enable()
        tr = m._make_loader(texts, y, shuffle=True)
        va = m._make_loader(devtexts, yv, shuffle=False,
                            batch_size=config.get("pred_batch_size", config["batch_size"]))
        no_decay = ("bias", "LayerNorm.weight")
        groups = [
            dict(
                params=[
                    p for n, p in m.model.named_parameters() if not any(t in n for t in no_decay)
                ],
                weight_decay=config["weight_decay"],
            ),
            dict(
                params=[p for n, p in m.model.named_parameters() if any(t in n for t in no_decay)],
                weight_decay=0.0,
            ),
        ]
        optimizer = torch.optim.AdamW(groups, lr=config["lr"])
        steps = len(tr) * config["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(steps * config["warmup_ratio"]), steps
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        minimum = S.encoder_min_epochs
        patience = S.encoder_patience
    else:
        m.vocab = build_vocab(texts, max_vocab_size=m.max_vocab_size, min_freq=m.min_freq)
        m.vocab_size = len(m.vocab)
        m.model = m._build_model().to(device)
        tr = multilabel_dataloader(
            texts_to_ids(texts, m.vocab, m.max_len), y, batch_size=m.batch_size, shuffle=True
        )
        va = multilabel_dataloader(
            texts_to_ids(devtexts, m.vocab, m.max_len), yv, batch_size=128, shuffle=False
        )
        optimizer = torch.optim.Adam(m.model.parameters(), lr=config["lr"])
        scheduler = scaler = None
        minimum = S.neural_min_epochs
        patience = S.neural_patience

    def unpack(batch):
        return (
            (
                {k: v.to(device) for k, v in batch.items() if k != "labels"},
                batch["labels"].to(device),
            )
            if encoder
            else (batch[0].to(device), batch[1].to(device))
        )

    def forward(x):
        return m.model(**x).logits if encoder else m.model(x)

    best = math.inf
    best_epoch = 0
    bad = 0
    updates = skipped = 0
    history = []
    criterion = torch.nn.BCEWithLogitsLoss()
    with (directory / "EPOCHS.jsonl").open("x") as log:
        for epoch in range(1, config["num_epochs"] + 1):
            started = time.monotonic()
            m.model.train()
            loss_sum = 0.0
            cells = 0
            for batch in tr:
                x, target = unpack(batch)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    logits = forward(x)
                    loss = criterion(logits, target)
                if encoder:
                    oldscale = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(m.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    success = scaler.get_scale() >= oldscale
                    if success:
                        scheduler.step()
                        updates += 1
                    else:
                        skipped += 1
                else:
                    loss.backward()
                    optimizer.step()
                    updates += 1
                loss_sum += float(loss.item()) * target.numel()
                cells += target.numel()
            m.model.eval()
            probs = []
            validation_sum = 0.0
            validation_cells = 0
            with torch.no_grad():
                for batch in va:
                    x, target = unpack(batch)
                    with torch.cuda.amp.autocast(enabled=amp):
                        logits = forward(x)
                    validation_sum += float(
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            logits.float(), target, reduction="sum"
                        ).item()
                    )
                    validation_cells += target.numel()
                    probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            p = np.concatenate(probs)
            value = validation_sum / validation_cells
            improved = value < best - 1e-6
            if improved:
                best = value
                best_epoch = epoch
                bad = 0
                torch.save(
                    {k: v.detach().cpu() for k, v in m.model.state_dict().items()},
                    directory / "best_state.pt",
                )
                np.save(directory / "best_dev_scores.npy", p)
            else:
                bad += 1
            row = dict(
                epoch=epoch,
                train_stream_bce=loss_sum / cells,
                dev=metrics(yv, p),
                dev_bce_logits=value,
                new_best=improved,
                successful_updates=updates,
                amp_skips=skipped,
                seconds=time.monotonic() - started,
            )
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            history.append(row)
            if epoch >= minimum and bad >= patience:
                break
    m.model.load_state_dict(
        torch.load(directory / "best_state.pt", map_location="cpu"), strict=True
    )
    m.is_fitted = True
    immutable_json(
        directory / "TRAINING.json",
        dict(
            config=config,
            best_epoch=best_epoch,
            best_dev_bce=best,
            best_dev_macro_f1=history[best_epoch - 1]["dev"]["macro_f1"],
            epochs_run=len(history),
            successful_updates=updates,
            amp_skips=skipped,
            stop_reason=(
                "early_stopping" if len(history) >= minimum and bad >= patience else "epoch_cap"
            ),
            selection="minimum_dev_BCE_logits",
            minimum_epochs=minimum,
            patience=patience,
            fit_rows=len(texts),
            fit_cells=int(y.size),
            unknown_targets_used=False,
        ),
    )
    return m


def _aux_fit(config, D, directory, P, S):
    import torch

    module = _backend(P["backend_source"])

    class LoggedBackend(module.ChaseTransformerBackend):
        @torch.no_grad()
        def _validation_loss(self, model, loader):
            model.eval()
            num = den = 0.0
            ps = []
            ys = []
            for batch in loader:
                with self._autocast():
                    logits = self._forward_logits(model, batch)
                    _, bn, bd = self._loss_parts(logits, batch)
                num += float(bn.cpu())
                den += float(bd.cpu())
                ps.append(torch.sigmoid(logits.float()).cpu().numpy())
                ys.append(batch["labels"].cpu().numpy())
            if den <= 0:
                raise ValueError("empty validation weight")
            with (directory / "VALIDATION.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            validation_loss=num / den,
                            dev=metrics(np.concatenate(ys), np.concatenate(ps)),
                        )
                    )
                    + "\n"
                )
            return num / den

    m = LoggedBackend(module.ChaseBackendConfig(**config), len(D["labels"]), device=S.device)
    view = lambda a: SimpleNamespace(
        targets=np.asarray(a, np.float32),
        observed=np.ones(np.asarray(a).shape, bool),
        weights=np.ones(np.asarray(a).shape, np.float32),
    )
    fit = [D["docs"][int(i)]["text"] for i in D["fit_indices"]]
    dev = [D["docs"][int(i)]["text"] for i in D["receiver_indices"]]
    audit = m.fit(fit, view(D["fit_targets"]), dev, view(D["dev_targets"]))
    immutable_json(directory / "TRAINING.json", audit)
    m.save_checkpoint(directory / "backend_checkpoint")
    torch.save(
        dict(
            state_dict={k: v.detach().cpu().clone() for k, v in m._model.state_dict().items()},
            label_names=D["labels"],
            model_name=config["model_name"],
            head=config["head"],
            dropout=config["dropout"],
            projection_dim=config["projection_dim"],
            training_config=config,
            source=SCHEMA,
            fit_row_ids=[D["row_ids"][int(i)] for i in D["fit_indices"]],
        ),
        directory / "model_only_checkpoint.pt",
    )
    with (directory / "EPOCHS.jsonl").open("x") as f:
        for row in audit["history"]:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    return m


def _verify_complete(directory, P):
    done = read_json(directory / "COMPLETE.json")
    if done["protocol"] != file_record(directory.parent / "PROTOCOL.json"):
        raise ValueError("model belongs to different pool protocol")
    for name in ("scores", "model", "training"):
        bound(done[name], directory)
    with np.load(done["scores"]["path"], allow_pickle=False) as z:
        if (
            z["labels"].tolist() != P["labels"]
            or z["row_ids"].tolist() != P["row_ids"]
            or z["scores"].shape != (len(P["row_ids"]), len(P["labels"]))
        ):
            raise ValueError("model output axes differ")
        if not np.isfinite(z["scores"]).all() or (z["scores"] < 0).any() or (z["scores"] > 1).any():
            raise ValueError("invalid saved probability values")
    return done


def _train_one(output, name):
    import torch
    from sklearn.exceptions import ConvergenceWarning

    out = Path(output)
    P, D, S = verify_pool(out)
    _seed(S.seed, S.cpu_threads)
    if name not in MODELS:
        raise ValueError("unknown model")
    d = out / name
    d.mkdir(exist_ok=False)
    started = time.monotonic()
    config = P["models"][name]
    if not name.startswith("tfidf") and S.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    immutable_json(
        d / "STARTED.json",
        dict(name=name, pid=os.getpid(), protocol=file_record(out / "PROTOCOL.json")),
    )
    texts = [r["text"] for r in D["docs"]]
    fit = np.asarray(D["fit_indices"])
    dev = np.asarray(D["receiver_indices"])
    if name.startswith("tfidf"):
        from joblib import parallel_backend

        m = FixedAxisTFIDF(**config)
        with warnings.catch_warnings(record=True) as caught, parallel_backend(
            "threading", n_jobs=S.cpu_threads
        ):
            warnings.simplefilter("always")
            m.fit([texts[int(i)] for i in fit], D["fit_targets"])
        immutable_json(
            d / "TRAINING.json",
            dict(
                config=config,
                fit_rows=len(fit),
                fit_cells=int(np.asarray(D["fit_targets"]).size),
                warnings=[str(w.message) for w in caught],
                converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                iterations=[
                    np.asarray(e.n_iter_).tolist() if hasattr(e, "n_iter_") else None
                    for e in m.classifier.estimators_
                ],
                constant_class_indices=[
                    i for i, e in enumerate(m.classifier.estimators_) if hasattr(e, "y_")
                ],
                unknown_targets_used=False,
            ),
        )
    elif name == "aux_dynamic":
        m = _aux_fit(config, D, d, P, S)
    else:
        m = _neural_fit(name, config, D, d, S)
    training_seconds = time.monotonic() - started
    if name == "aux_dynamic":
        checkpoint = d / "model_only_checkpoint.pt"
    else:
        if hasattr(m, "model"):
            m.model.cpu()
            m.device = torch.device("cpu")
        with (d / "model.pkl").open("xb") as f:
            pickle.dump(m, f, pickle.HIGHEST_PROTOCOL)
        checkpoint = d / "model.pkl"
        if hasattr(m, "model"):
            m.model.to(S.device)
            m.device = torch.device(S.device)
    checkpoint_seconds = time.monotonic() - started - training_seconds
    inference_started = time.monotonic()
    unique = list(dict.fromkeys(texts))
    positions = {text: i for i, text in enumerate(unique)}
    raw = m.predict_proba(unique)
    scores = raw[np.asarray([positions[t] for t in texts])].astype(np.float32)
    if (
        scores.shape != (len(texts), len(D["labels"]))
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("new model output axis or probabilities differ")
    inference_seconds = time.monotonic() - inference_started
    np.savez_compressed(
        d / "probabilities.npz",
        scores=scores,
        labels=np.asarray(D["labels"]),
        row_ids=np.asarray(D["row_ids"]),
        global_rows=np.arange(len(texts)),
    )
    etas = []
    truth = np.asarray(D["dev_targets"], bool)
    for j in range(len(D["labels"])):
        ranked = []
        for index, eta in enumerate(P["calibration_grid"]):
            pred = scores[dev, j] >= eta
            t = truth[:, j]
            tp = int((pred & t).sum())
            fp = int((pred & ~t).sum())
            fn = int((~pred & t).sum())
            ranked.append((2 * tp / max(1, 2 * tp + fp + fn), -abs(eta - 0.5), eta))
        etas.append(max(ranked)[2])
    done = dict(
        status="complete",
        alias="loris_clf_" + name,
        name=name,
        seed=S.seed,
        protocol=file_record(out / "PROTOCOL.json"),
        scores=file_record(d / "probabilities.npz"),
        model=file_record(checkpoint),
        training=file_record(d / "TRAINING.json"),
        shape=list(scores.shape),
        labels=D["labels"],
        fit_metrics=metrics(D["fit_targets"], scores[fit]),
        dev_metrics=metrics(D["dev_targets"], scores[dev]),
        seconds=time.monotonic() - started,
        unique_texts=len(unique),
        training_seconds=training_seconds,
        checkpoint_seconds=checkpoint_seconds,
        inference_seconds=inference_seconds,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_cuda_bytes=(
            int(torch.cuda.max_memory_allocated())
            if not name.startswith("tfidf") and S.device == "cuda"
            else 0
        ),
        same_dev_calibrated_diagnostic=dict(
            eta=etas,
            metrics=metrics(truth, (scores[dev] >= np.asarray(etas)).astype(float)),
            not_applied_to_raw_scores=True,
        ),
        training_axis=file_record(out / "TRAINING_AXIS.npz"),
        test_read=False,
        unknown_truth_read=False,
    )
    immutable_json(d / "COMPLETE.json", done)


def handoff_pool(output):
    out = Path(output).resolve()
    P, D, S = verify_pool(out)
    completed = {name: _verify_complete(out / name, P) for name in MODELS}
    result = dict(
        schema=SCHEMA,
        status="complete",
        channels={
            r["alias"]: dict(
                probabilities=r["scores"],
                checkpoint=r["model"],
                dev_macro_f1=r["dev_metrics"]["macro_f1"],
            )
            for r in completed.values()
        },
        shape=[len(D["docs"]), len(D["labels"])],
        labels=D["labels"],
        row_ids=D["row_ids"],
        row_roles=D["row_roles"],
        protocol=file_record(out / "PROTOCOL.json"),
        model_order=P["model_order"],
        supervision_manifest=D["manifest_record"],
        exposure=file_record(out / "EXPOSURE.json"),
        test_read=False,
        unknown_truth_read=False,
        online_feedback_updates=0,
        sum_model_seconds=sum(r["seconds"] for r in completed.values()),
        sum_training_seconds=sum(r["training_seconds"] for r in completed.values()),
        sum_inference_seconds=sum(r["inference_seconds"] for r in completed.values()),
    )
    immutable_json(out / "HANDOFF.json", result)
    return result


def run_pool(output, *, phase):
    """Run real models under a hard elapsed phase budget; no partial-fit resume."""
    import torch

    phase_started = time.monotonic()
    out = Path(output).resolve()
    P, D, S = verify_pool(out)
    if phase == "handoff":
        return handoff_pool(out)
    names = (
        MODELS[:3]
        if phase == "cpu"
        else MODELS[3:] if phase == "gpu" else MODELS if phase == "all" else None
    )
    if names is None:
        raise ValueError("phase must be cpu/gpu/all/handoff")
    if phase != "cpu" and S.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("declared CUDA is unavailable")
    deadline = phase_started + S.max_phase_seconds
    completed = []
    reused = []
    for name in names:
        if (out / name / "COMPLETE.json").exists():
            _verify_complete(out / name, P)
            reused.append(name)
            continue
        if (out / name).exists():
            raise FileExistsError(
                "partial model training is preserved; new namespace required: " + name
            )
        lock = None
        child = None
        try:
            if not name.startswith("tfidf") and S.device == "cuda" and P["gpu_lock"]:
                path = Path(P["gpu_lock"])
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = path.open("a+")
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("phase budget exhausted waiting for GPU")
                        time.sleep(0.2)
            if time.monotonic() >= deadline:
                raise TimeoutError("phase budget exhausted before next model")
            child = multiprocessing.get_context("spawn").Process(
                target=_train_one, args=(str(out), name)
            )
            child.start()
            child.join(max(0.0, deadline - time.monotonic()))
            if child.is_alive():
                child.terminate()
                child.join()
                raise TimeoutError("hard phase budget exhausted during " + name)
            if child.exitcode != 0:
                raise RuntimeError("real model process failed: " + name)
            _verify_complete(out / name, P)
            completed.append(name)
        except BaseException as error:
            if child is not None and child.is_alive():
                child.terminate()
                child.join()
            atomic_json(
                out / "LAST_FAILURE.json",
                dict(
                    phase=phase,
                    model=name,
                    error_type=type(error).__name__,
                    reason=str(error),
                    completed=completed,
                    reused=reused,
                    partial_artifacts_preserved=True,
                ),
            )
            raise
        finally:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
    receipt = dict(
        status="phase_complete",
        phase=phase,
        trained=completed,
        reused=reused,
        protocol=file_record(out / "PROTOCOL.json"),
        wall_seconds=time.monotonic() - phase_started,
        actual_training_seconds=sum(
            read_json(out / n / "COMPLETE.json")["training_seconds"] for n in completed
        ),
        actual_inference_seconds=sum(
            read_json(out / n / "COMPLETE.json")["inference_seconds"] for n in completed
        ),
    )
    with (out / "PHASE_ATTEMPTS.jsonl").open("a") as stream:
        stream.write(json.dumps(receipt) + "\n")
    return handoff_pool(out) if phase == "all" else receipt


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser(
        "freeze", help="bind inputs, exact seven model configs and budget; no training"
    )
    freeze.add_argument("--supervision-manifest", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument(
        "--pretrained-model", required=True, help="local real pretrained snapshot directory"
    )
    freeze.add_argument(
        "--backend-source", required=True, help="source root containing actual loris_ssl package"
    )
    freeze.add_argument(
        "--gpu-lock", required=True, help="shared exclusive GPU lock used one model at a time"
    )
    freeze.add_argument(
        "--settings", required=True, help="JSON with every PoolSettings field explicitly declared"
    )
    run = sub.add_parser(
        "run",
        help="train real models or authenticate completed models; partial fit resume unsupported",
    )
    run.add_argument("--output", required=True)
    run.add_argument("--phase", choices=("cpu", "gpu", "all", "handoff"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            values = read_json(args.settings)
            if set(values) != set(asdict(PoolSettings())):
                raise ValueError("settings JSON must explicitly declare all PoolSettings fields")
            result = freeze_pool(
                args.supervision_manifest,
                args.output,
                pretrained_model=args.pretrained_model,
                backend_source=args.backend_source,
                settings=PoolSettings(**values),
                gpu_lock=args.gpu_lock,
            )
        else:
            result = run_pool(args.output, phase=args.phase)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (
        ValueError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TimeoutError,
        ImportError,
    ) as error:
        print(
            json.dumps(dict(status="failed", error_type=type(error).__name__, reason=str(error))),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
