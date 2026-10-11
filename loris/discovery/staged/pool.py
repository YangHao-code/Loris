"""Calibrate expert scores, then train the existing neural Top-K selector."""
from pathlib import Path
import json

import numpy as np

from .metrics import f1, quality

GRID = np.linspace(.025, .975, 39)
POOL_ALIAS = 'loris_calibrated_pool'


def _probabilities(value, shape=None):
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 2 or (shape is not None and p.shape != shape):
        raise ValueError('Aligned document x label probabilities required')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Finite probabilities in [0, 1] required')
    return p


def calibrate_expert(probabilities, truth):
    """Fit on development A; shrink rare-label logit cuts toward one global cut."""
    y = np.asarray(truth)
    if y.ndim != 2 or not y.size or y.dtype != np.bool_:
        raise ValueError('Nonempty complete boolean calibration targets required')
    p = _probabilities(probabilities, y.shape)
    objective = lambda q: .5 * (q['macro_f1'] + q['micro_f1'])
    global_cut = min(GRID.tolist() + [.5],
                     key=lambda c: (-objective(quality(p >= c, y)), abs(c - .5), c))
    cuts = []
    for j, positives in enumerate(y.sum(0)):
        if positives == 0:
            cuts.append(float(global_cut))
            continue
        candidates = np.unique(np.r_[GRID, .5, global_cut])
        def key(cut):
            v = p[:, j] >= cut
            tp = int((v & y[:, j]).sum())
            fp = int((v & ~y[:, j]).sum())
            return (-float(f1(tp, fp, int(positives) - tp)), abs(cut - global_cut), cut)
        chosen = min(candidates, key=key)
        weight = float(positives / (positives + 20.))
        logit = lambda x: np.log(x / (1. - x))
        cuts.append(float(1. / (1. + np.exp(-(
            weight * logit(chosen) + (1. - weight) * logit(global_cut))))))
    return dict(global_threshold=float(global_cut), per_label_thresholds=cuts,
                shrink_pseudocount=20., per_label_positive_support=y.sum(0).tolist())


def calibrated_probabilities(probabilities, cuts):
    p = _probabilities(probabilities)
    c = np.asarray(cuts, dtype=np.float64)
    if c.shape != (p.shape[1],) or not np.isfinite(c).all() or ((c <= 0) | (c >= 1)).any():
        raise ValueError('One finite interior threshold per native label required')
    p = np.clip(p, 1e-6, 1 - 1e-6)
    c = np.clip(c, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) - np.log(c / (1 - c)))))


def pool_probabilities(values, config):
    """Use precisely the fitted members and calibrated-margin combination."""
    members = config['members']
    if not members or len(set(members)) != len(members):
        raise ValueError('Nonempty unique learned model members required')
    shape = (len(values[members[0]]), len(config['labels']))
    total = np.zeros(shape, dtype=np.float64)
    for alias in members:
        p = np.clip(_probabilities(values[alias], shape), 1e-6, 1 - 1e-6)
        c = np.asarray(config['expert_cuts'][alias], dtype=np.float64)
        if c.shape != (shape[1],) or not np.isfinite(c).all() or ((c <= 0) | (c >= 1)).any():
            raise ValueError('Invalid fitted expert thresholds')
        c = np.clip(c, 1e-6, 1 - 1e-6)
        total += np.log(p / (1 - p)) - np.log(c / (1 - c))
    return 1 / (1 + np.exp(-total / len(members)))


def extend_context(context, config):
    if list(context['labels']) != config['labels']:
        raise ValueError('Pool native label order differs')
    alias = config.get('alias', POOL_ALIAS)
    if alias in context['model_probabilities']:
        raise ValueError('Pool alias must not overwrite an existing frozen expert')
    values = dict(context['model_probabilities'])
    values[alias] = pool_probabilities(values, config)
    decisions = dict(context.get('model_decisions', {}))
    decisions[alias] = values[alias] >= .5
    # The native runner hashes the actual context arrays. Do not reuse a caller
    # supplied context identifier for a newly introduced predicate.
    result = dict(context, model_probabilities=values, model_decisions=decisions)
    result.pop('context_id', None)
    return result


def fit_pool(*, features, probabilities, truth, row_ids, labels, input_binding,
             output, device='cpu', k=4, protocol=None):
    """Full-batch MLP/MC-TopK on A only; final members come from trained frequency."""
    from loris._learning.selection import (
        ModelSelectionProtocol, train_model_selection, verify_model_selection_output)
    from loris.discovery.selector import configure_workspace
    aliases = sorted(probabilities)
    if len(aliases) < k:
        raise ValueError('K exceeds the supplied expert pool')
    y = np.asarray(truth)
    details = {a: calibrate_expert(probabilities[a], y) for a in aliases}
    cuts = {a: d['per_label_thresholds'] for a, d in details.items()}
    normalized = {a: calibrated_probabilities(probabilities[a], cuts[a]) for a in aliases}
    decisions = np.stack([np.asarray(probabilities[a]) >= np.asarray(cuts[a]) for a in aliases])
    if protocol is None:
        protocol = ModelSelectionProtocol(
            k=k, epochs=100, hidden_dim=128, num_samples=64, sigma=.1,
            learning_rate=.001, lambda_task=.1, lambda_entropy=.01,
            imitation_positive_weight=2., vote_smoothing=.1, dropout=.1,
            seed=42, device=device, oracle_policy='global_margin_macro_f1',
            task_objective='pooled_margin_macro_f1')
    if (protocol.k != k or protocol.oracle_policy != 'global_margin_macro_f1'
            or protocol.task_objective != 'pooled_margin_macro_f1'):
        raise ValueError('Calibrated pool requires matching margin oracle, task and K')
    configure_workspace(protocol.device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train_model_selection(
        features=features, decisions=decisions, truth=y,
        cluster_assignments=np.zeros(len(y), np.int64), row_ids=row_ids,
        aliases=aliases, labels=labels, input_binding=input_binding, protocol=protocol,
        output_dir=output / 'selector', model_interpretation=dict(
            row_ids=row_ids, probabilities=normalized,
            decisions={a: decisions[i] for i, a in enumerate(aliases)}))
    actual = verify_model_selection_output(output / 'selector')['result']
    members = next(iter(actual['selected_aliases_by_cluster'].values()))
    config = dict(schema='loris_calibrated_neural_pool_v1', alias=POOL_ALIAS,
                  labels=list(labels), members=members, expert_cuts=cuts,
                  selection='trained_hard_topk_frequency', calibration=details,
                  protocol=protocol.manifest())
    (output / 'POOL.json').write_text(json.dumps(config, indent=2, allow_nan=False))
    return config
