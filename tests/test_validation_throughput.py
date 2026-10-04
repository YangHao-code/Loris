"""Independent validation batching keeps training windows and full holdout coverage."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_labeling_actual_aux import assets, Tokenizer, Encoder
from loris.labeling.trainer import load_backend


def test_legacy_checkpoint_config_fingerprint_is_preserved(tmp_path):
    source, _, raw = assets(tmp_path)
    backend = load_backend(source)
    cfg = backend.ChaseBackendConfig(**raw)
    legacy = cfg.to_dict()
    legacy.pop('validation_batch_size')
    for key in list(legacy):
        if key.startswith('initial_checkpoint_'):
            legacy.pop(key)
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(',', ':'),
                                          allow_nan=False).encode()).hexdigest()
    assert cfg.fingerprint == expected
    assert replace(cfg, validation_batch_size=8).fingerprint != expected
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='validation_batch_size'):
            replace(cfg, validation_batch_size=invalid)


def test_full_validation_uses_independent_batch_without_changing_training(tmp_path):
    source, _, raw = assets(tmp_path)
    backend = load_backend(source)
    cfg = backend.ChaseBackendConfig(**raw)
    texts = ['first red', 'second blue', 'third red', 'fourth blue']
    holdout = ['holdout item %d' % i for i in range(7)]

    def view(n):
        y = (np.arange(n * 2).reshape(n, 2) % 3 == 0).astype(np.float32)
        return SimpleNamespace(targets=y, observed=np.ones_like(y, bool), weights=np.ones_like(y))

    audits = []
    for size in (None, 5):
        seen = []

        class ObservedBackend(backend.ChaseTransformerBackend):
            def _validation_loss(self, model, loader):
                seen.append((loader.batch_size, len(loader.dataset)))
                return super()._validation_loss(model, loader)

        model = ObservedBackend(replace(cfg, validation_batch_size=size), 2, device='cpu',
                                tokenizer_factory=lambda _: Tokenizer(),
                                model_factory=lambda _, k: Encoder(k))
        audit = model.fit(texts, view(len(texts)), holdout, view(len(holdout)))
        assert seen == [(size or cfg.batch_size, 7)] * 2  # student and EMA; no holdout sampling
        assert audit['optimizer_updates'] == 1
        record = audit['history'][0]
        assert record['training_seconds'] >= 0
        assert record['validation_student_seconds'] >= 0
        assert record['validation_ema_seconds'] >= 0
        assert record['epoch_seconds'] >= record['training_seconds']
        audits.append(audit)
    assert audits[0]['best_validation_loss'] == pytest.approx(audits[1]['best_validation_loss'], abs=1e-6)
    assert audits[0]['supervision_sha256'] == audits[1]['supervision_sha256']
