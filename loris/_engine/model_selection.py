"""Durable per-cluster model selection from complete, frozen fit-row inputs.

PDF36 §5 specifies hard Top-K, an MC backward approximation, hybrid training,
and aggregation by selection frequency over D_i. Its numeric defaults and the
multi-label oracle/task surrogate are unfinished. This implementation declares
those choices explicitly: negative per-document binary cross-entropy oracle,
weighted imitation BCE, BCE on selected mean probabilities, and softmax-score
entropy. This is an explicit multi-label protocol, not a claim that the PDF
uniquely prescribes it or that BCE solves its majority-vote optimization.

Every cluster uses all its assigned fit rows for every configured epoch and
for final frequency aggregation. Only the selection network is trained. No
base model, old rule, candidate bank, validation truth, or test data is read.
Features are caller-supplied and bound; their extraction is not reproduced by
this module. No row subsampling, early stopping, pool fallback, or auto-device
fallback is supported. Checkpoints retain model, optimizer and random state.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import inspect
import json
import math
import os
import pickle
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch

from loris.selection.dynamic_router import SelectionNetwork, HybridLoss
from loris._engine import rlearner as _durable
from loris._engine.rlearner import atomic_json, immutable_json, array_identity, canonical, digest


SCHEMA = 'paper36_per_cluster_frozen_pool_multilabel_selection_v1'


@dataclass(frozen=True)
class SelectionModelSpec:
    alias: str
    dynamic: bool

    def __post_init__(self):
        if not isinstance(self.alias, str) or not self.alias or type(self.dynamic) is not bool:
            raise ValueError('model alias and explicit dynamic Boolean required')


@dataclass(frozen=True)
class SelectionProtocol:
    """All training hyperparameters are explicit experiment choices."""
    k: int
    epochs: int
    hidden_dim: int
    num_samples: int
    sigma: float
    learning_rate: float
    lambda_task: float
    lambda_entropy: float
    imitation_positive_weight: float
    dropout: float
    seed: int
    device: str
    backend: str

    def __post_init__(self):
        for name in ('k', 'epochs', 'hidden_dim', 'num_samples'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be an explicit positive integer')
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 63:
            raise ValueError('seed must be an explicit nonnegative integer below 2**63')
        for name in ('sigma', 'learning_rate', 'imitation_positive_weight'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be positive and finite')
        for name in ('lambda_task', 'lambda_entropy', 'dropout'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be nonnegative and finite')
        if self.dropout >= 1:
            raise ValueError('dropout must be below one')
        if self.backend not in ('custom', 'perturbed'):
            raise ValueError('only declared hard-Top-K MC-gradient backends are supported')
        if self.device != 'cpu' and not (
                isinstance(self.device, str) and self.device.startswith('cuda:')
                and self.device[5:].isdigit()):
            raise ValueError('device must explicitly be cpu or cuda:<index>; no fallback')

    def manifest(self):
        return dict(asdict(self), schema=SCHEMA,
            paper_numeric_defaults_complete=False,
            multi_label_extension='negative per-row BCE oracle; BCE of selected mean probabilities',
            loss='weighted_imitation_BCE + lambda_task * multilabel_BCE - lambda_entropy * softmax_entropy',
            oracle_probability_clip=1e-7, task_probability_clip=1e-6,
            optimizer={'name': 'Adam', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.},
            training='one full-cluster batch per epoch; all configured epochs',
            final_frequency_rows='same complete fit cluster D_i, not validation',
            oracle_and_frequency_ties='stable supplied model order',
            per_document_topk_ties='torch.topk on bound backend/device/version',
            backend_implementation=('DifferentiableTopK custom hard-forward/MC-backward' if self.backend == 'custom'
                else 'perturbations.perturbed with explicit device and hard_fwd=True; not the legacy auto-device special wrapper'),
            cluster_order='ascending explicit integer cluster identifier',
            cluster_seed='(seed + 1009 * position) modulo 2**63',
            base_model_fit_calls=0, base_model_inference_calls=0,
            majority_vote_global_optimality_claimed=False)


def _sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def _immutable_npz(path, values):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(values) or any(
                    saved[key].dtype != value.dtype or not np.array_equal(saved[key], value)
                    for key, value in values.items()):
                raise ValueError('immutable array artifact differs: ' + str(path))
    else:
        temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))


def _atomic_checkpoint(path, value):
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('wb') as stream:
        torch.save(_checkpoint_tensors(value), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _checkpoint_tensors(value):
    """Keep selector snapshots compatible with Torch's weights-only loader."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        result = copy.copy(value)  # Preserve state_dict's OrderedDict metadata.
        for key, child in value.items():
            result[key] = _checkpoint_tensors(child)
        return result
    if isinstance(value, (tuple, list)):
        return type(value)(_checkpoint_tensors(child) for child in value)
    return value


def _load_selector_checkpoint(path, *, map_location):
    """Read only internal selector artifacts after the caller checks their binding/hash.

    New Torch uses its restricted loader explicitly. Torch 1.11 has no such
    option and retains its original trusted-local-artifact loading boundary.
    There is no unrestricted retry if a newer Torch rejects a legacy pickle.
    """
    options = {'map_location': map_location}
    restricted = 'weights_only' in inspect.signature(torch.load).parameters
    if restricted:
        options['weights_only'] = True
    try:
        state = torch.load(path, **options)
    except pickle.UnpicklingError as error:
        if not restricted:
            raise ValueError('Invalid internal selector checkpoint; no alternate loader is attempted.') from error
        raise ValueError(
            'Selector checkpoint is unsupported by the restricted loader; '
            'regenerate legacy NumPy-pickle checkpoints with this release. '
            'Unrestricted loading is not attempted.'
        ) from error
    # Restore only the numerical snapshot fields; model/optimizer/RNG tensors
    # retain their original device and type for exact training resumption.
    for container, fields in ((state, ('scores', 'masks')),
                              (state.get('selection', {}), ('scores', 'masks', 'frequency'))):
        for key in fields:
            if isinstance(container.get(key), torch.Tensor):
                container[key] = container[key].detach().cpu().numpy()
    return state


def _state_identity(state):
    return digest({name: array_identity(value.detach().cpu().numpy())
                   for name, value in sorted(state.items())})


def _input_arrays(*, features, truth, probabilities, model_specs, model_decisions,
                  cluster_assignments, row_ids, labels, input_binding, protocol):
    if not isinstance(protocol, SelectionProtocol):
        raise ValueError('explicit SelectionProtocol required')
    if not isinstance(input_binding, Mapping) or input_binding.get('role') != 'fit':
        raise ValueError('input binding must explicitly declare role=fit')
    if not isinstance(input_binding.get('feature_provenance'), Mapping) or not input_binding['feature_provenance']:
        raise ValueError('feature extraction provenance must be supplied')
    features = np.asarray(features)
    truth = np.asarray(truth)
    if features.ndim != 2 or not features.shape[0] or not features.shape[1] or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError('complete finite float32 feature matrix required')
    n = features.shape[0]
    if type(input_binding.get('expected_fit_rows')) is not int or input_binding['expected_fit_rows'] != n:
        raise ValueError('complete expected fit row count must be bound explicitly')
    labels = tuple(labels)
    if not labels or any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError('nonempty unique ordered label names required')
    if truth.shape != (n, len(labels)) or not np.isin(truth, [0, 1]).all():
        raise ValueError('complete binary fit truth label axis required')
    assignments = np.asarray(cluster_assignments)
    if assignments.shape != (n,) or not np.issubdtype(assignments.dtype, np.integer) or np.any(assignments < 0):
        raise ValueError('every fit row requires one explicit nonnegative integer cluster assignment')
    rows = list(row_ids)
    if len(rows) != n or any(isinstance(row, (bool, np.bool_)) or not isinstance(row, (str, int, np.integer)) for row in rows):
        raise ValueError('full row identity axis requires unique integer/string IDs')
    rows = [int(row) if isinstance(row, (int, np.integer)) else row for row in rows]
    if any(isinstance(row, str) and not row for row in rows) or len(set(map(canonical, rows))) != n:
        raise ValueError('full row identity axis requires unique integer/string IDs')
    model_specs = tuple(model_specs)
    if not model_specs or any(not isinstance(model, SelectionModelSpec) for model in model_specs):
        raise ValueError('explicit complete SelectionModelSpec roster required')
    aliases = [model.alias for model in model_specs]
    if len(set(aliases)) != len(aliases) or not 1 <= protocol.k <= len(aliases):
        raise ValueError('unique model aliases and K within the complete pool required')
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(aliases):
        raise ValueError('probability tables must exactly cover the declared model pool; no fallback')
    decisions = {} if model_decisions is None else model_decisions
    if not isinstance(decisions, Mapping) or set(decisions) - set(aliases):
        raise ValueError('decision tables refer to an undeclared model')
    matrices, frozen_decisions = [], {}
    for model in model_specs:
        p = np.asarray(probabilities[model.alias])
        if p.shape != truth.shape or p.dtype != np.float32 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('complete finite float32 probability table required: ' + model.alias)
        matrices.append(np.array(p, copy=True))
        if not model.dynamic and model.alias not in decisions:
            raise ValueError('static model requires explicit binary decision table: ' + model.alias)
        if model.alias in decisions:
            value = np.asarray(decisions[model.alias])
            if value.shape != truth.shape or not np.isin(value, [0, 1]).all():
                raise ValueError('complete binary model decision table required: ' + model.alias)
            frozen_decisions[model.alias] = np.array(value, bool, copy=True)
    return {'features': np.array(features, copy=True), 'truth': np.array(truth, bool, copy=True),
            'probabilities': np.stack(matrices), 'decisions': frozen_decisions,
            'assignments': np.array(assignments, np.int64, copy=True), 'row_ids': rows,
            'labels': labels, 'models': model_specs, 'aliases': aliases}


def build_multilabel_oracle(probabilities, truth, k):
    """Rank each model's full-label BCE, with stable input-order ties."""
    p, y = np.asarray(probabilities), np.asarray(truth)
    if p.ndim != 3 or p.shape[1:] != y.shape or y.ndim != 2 or not y.size:
        raise ValueError('oracle needs model x row x label probabilities and row x label truth')
    if type(k) is not int or not 1 <= k <= p.shape[0] or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all():
        raise ValueError('invalid complete oracle inputs')
    # Use float64 for a fixed, numerically explicit oracle computation; the
    # selection network itself consumes float32 feature/probability tensors.
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    ce = -(y[None] * np.log(p) + (1 - y[None]) * np.log1p(-p)).mean(axis=2).T
    chosen = np.argsort(ce, axis=1, kind='stable')[:, :k]
    mask = np.zeros(ce.shape, np.float32)
    np.put_along_axis(mask, chosen, 1., axis=1)
    return mask, ce


@contextmanager
def _random_scope(device, seed):
    devices = [] if device.type == 'cpu' else [device.index]
    deterministic = torch.are_deterministic_algorithms_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _rng_state(device):
    return {'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def _restore_rng(state, device):
    torch.set_rng_state(state['cpu'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def _snapshot(net, features, aliases, k):
    net.eval()
    with torch.no_grad():
        mask, scores = net(features, return_scores=True)
    masks, values = mask.detach().cpu().numpy(), scores.detach().cpu().numpy()
    if not np.isfinite(values).all() or not np.isin(masks, [0, 1]).all() or not np.all(masks.sum(axis=1) == k):
        raise RuntimeError('selection network did not return finite exact hard Top-K outputs')
    frequency = masks.sum(axis=0, dtype=np.int64)
    selected = np.argsort(-frequency, kind='stable')[:k].tolist()
    return {'scores': values, 'masks': masks, 'frequency': frequency,
            'selected_indices': selected, 'selected_aliases': [aliases[i] for i in selected]}


def _checkpoint_receipt(checkpoint, filename, checkpoint_sha256):
    snapshot = checkpoint['selection']
    return {'schema': SCHEMA, 'binding_id': checkpoint['binding_id'],
        'cluster_id': checkpoint['cluster_id'], 'epoch': checkpoint['epoch'],
        'optimizer_steps_completed': checkpoint['epoch'],
        'losses_before_update': checkpoint['losses_before_update'],
        'model_state_sha256': _state_identity(checkpoint['model']),
        'post_update_scores': array_identity(snapshot['scores']),
        'post_update_masks': array_identity(snapshot['masks']),
        'frequency': snapshot['frequency'].tolist(),
        'selected_indices': snapshot['selected_indices'], 'selected_aliases': snapshot['selected_aliases'],
        'checkpoint': filename, 'checkpoint_sha256': checkpoint_sha256}


def _cluster(directory, data, indices, cluster_id, protocol, binding_id, seed, device, on_progress, sources):
    directory.mkdir(parents=True, exist_ok=True)
    p, y = data['probabilities'][:, indices], data['truth'][indices]
    oracle, oracle_ce = build_multilabel_oracle(p, y, protocol.k)
    _immutable_npz(directory / 'oracle.npz', {'fit_row_indices': indices,
        'oracle_mask': oracle, 'per_model_binary_cross_entropy': oracle_ce})
    immutable_json(directory / 'INPUTS.json', {'binding_id': binding_id, 'cluster_id': cluster_id,
        'rows': len(indices), 'fit_row_indices': indices.tolist(), 'row_ids': [data['row_ids'][i] for i in indices],
        'seed': seed, 'oracle': {'file': 'oracle.npz', 'sha256': _sha_file(directory / 'oracle.npz'),
            'mask': array_identity(oracle), 'per_model_bce': array_identity(oracle_ce)}})
    with _random_scope(device, seed):
        net = SelectionNetwork(input_dim=data['features'].shape[1], hidden_dim=protocol.hidden_dim,
            n_models=len(data['models']), k=protocol.k, num_samples=protocol.num_samples,
            sigma=protocol.sigma, dropout=protocol.dropout, device=device, backend='custom').to(device)
        if protocol.backend == 'perturbed':
            # The legacy special wrapper omits its device argument when
            # sampling noise. The general implementation preserves explicit
            # CPU/CUDA placement and the same declared hard-forward/MC method.
            from loris.selection import perturbations
            def hard_topk(scores):
                indices = torch.topk(scores, protocol.k, dim=-1).indices
                return torch.zeros_like(scores).scatter_(-1, indices, 1.)
            net._topk_layer = perturbations.perturbed(hard_topk, num_samples=protocol.num_samples,
                sigma=protocol.sigma, noise='normal', batched=True, device=device, hard_fwd=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=protocol.learning_rate,
            betas=(.9, .999), eps=1e-8, weight_decay=0.)
        loss_fn = HybridLoss(lambda_task=protocol.lambda_task, lambda_ent=protocol.lambda_entropy,
            pos_weight=protocol.imitation_positive_weight).to(device)
        feature_t = torch.from_numpy(data['features'][indices]).to(device)
        oracle_t = torch.from_numpy(oracle).to(device)
        p_t, y_t = torch.from_numpy(p).to(device), torch.from_numpy(y.astype(np.float32)).to(device)
        receipts = sorted(directory.glob('epoch_*.json'))
        checkpoint = None
        for epoch, path in enumerate(receipts):
            record = json.loads(path.read_text())
            if path.name != 'epoch_%05d.json' % epoch or record.get('epoch') != epoch or epoch > protocol.epochs:
                raise ValueError('noncontiguous or excessive committed selection epochs')
            filename = 'checkpoint_%05d.pt' % epoch
            target = directory / filename
            if record.get('checkpoint') != filename or _sha_file(target) != record.get('checkpoint_sha256'):
                raise ValueError('selection checkpoint checksum differs')
            # Only local SHA-verified checkpoints created by this API are read.
            checkpoint = _load_selector_checkpoint(target, map_location=device)
            if checkpoint['binding_id'] != binding_id or checkpoint['cluster_id'] != cluster_id or checkpoint['epoch'] != epoch:
                raise ValueError('selection checkpoint context/cluster/epoch differs')
            if _checkpoint_receipt(checkpoint, filename, record['checkpoint_sha256']) != record:
                raise ValueError('selection epoch receipt differs from checkpoint')
        if checkpoint is not None:
            net.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            _restore_rng(checkpoint['rng_state'], device)
        start = len(receipts)
        for epoch in range(start, protocol.epochs + 1):
            for path, expected in sources.items():
                if _sha_file(path) != expected:
                    raise RuntimeError('bound selection source changed during run: ' + path)
            losses = None
            if epoch:
                net.train()
                mask, scores = net(feature_t, return_scores=True)
                imitation = loss_fn.imitation_loss(scores, oracle_t)
                task = loss_fn.task_loss_multilabel(mask, p_t, y_t)
                entropy = loss_fn.entropy_loss(scores)
                total = imitation + protocol.lambda_task * task - protocol.lambda_entropy * entropy
                if not torch.isfinite(total):
                    raise RuntimeError('nonfinite full-cluster model selection loss')
                optimizer.zero_grad()
                total.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in net.parameters()):
                    raise RuntimeError('nonfinite selection gradient; no fallback update')
                optimizer.step()
                losses = {'imitation': float(imitation.detach().cpu()), 'task': float(task.detach().cpu()),
                          'entropy': float(entropy.detach().cpu()), 'total': float(total.detach().cpu())}
            selected = _snapshot(net, feature_t, data['aliases'], protocol.k)
            checkpoint = {'schema': SCHEMA, 'binding_id': binding_id, 'cluster_id': cluster_id,
                'epoch': epoch, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                'rng_state': _rng_state(device), 'losses_before_update': losses, 'selection': selected}
            for path, expected in sources.items():
                if _sha_file(path) != expected:
                    raise RuntimeError('bound selection source changed during epoch: ' + path)
            target = directory / ('checkpoint_%05d.pt' % epoch)
            _atomic_checkpoint(target, checkpoint)
            record = _checkpoint_receipt(checkpoint, target.name, _sha_file(target))
            # This atomic JSON is the commit point. An orphan checkpoint from
            # a pre-commit crash is recomputed from the last committed state.
            immutable_json(directory / ('epoch_%05d.json' % epoch), record)
            if on_progress is not None:
                on_progress(copy.deepcopy(dict(record, phase='model_selection_epoch', rows=len(indices))))
        snapshot = checkpoint['selection']
        _immutable_npz(directory / 'final_selection.npz', {'fit_row_indices': indices,
            'scores': snapshot['scores'], 'hard_topk_masks': snapshot['masks'], 'frequency': snapshot['frequency']})
        result = {'schema': SCHEMA, 'binding_id': binding_id, 'cluster_id': cluster_id,
            'rows': len(indices), 'completed_epochs': protocol.epochs, 'all_fit_cluster_rows_used': True,
            'selected_aliases': snapshot['selected_aliases'], 'selected_indices': snapshot['selected_indices'],
            'selected_models': [asdict(data['models'][i]) for i in snapshot['selected_indices']],
            'frequency': snapshot['frequency'].tolist(),
            'final_selection_sha256': _sha_file(directory / 'final_selection.npz'),
            'final_model_state_sha256': _state_identity(checkpoint['model']),
            'final_epoch_receipt_sha256': _sha_file(directory / ('epoch_%05d.json' % protocol.epochs))}
        immutable_json(directory / 'SELECTION.json', result)
        immutable_json(directory / 'COMPLETE.json', {'selection_sha256': digest(result),
            'completed_epochs': protocol.epochs, 'binding_id': binding_id})
        return result


def run_cluster_model_selection(directory, *, features, truth, probabilities, model_specs,
        cluster_assignments, row_ids, labels, input_binding, protocol, model_decisions=None, on_progress=None):
    """Return complete per-cluster S_i records; resume only identical inputs.

    The row axis is the entire fit split, never a validation subset. Metadata
    ``input_binding`` requires role='fit', expected_fit_rows, and nonempty
    feature_provenance. Probabilities have shape (fit rows, labels) per alias;
    features are explicit float32; static model specs additionally require
    frozen binary decision tables. Decisions certify downstream static ML
    semantics; all selection oracle/task losses use the probability tables.
    """
    data = _input_arrays(features=features, truth=truth, probabilities=probabilities,
        model_specs=model_specs, model_decisions=model_decisions, cluster_assignments=cluster_assignments,
        row_ids=row_ids, labels=labels, input_binding=input_binding, protocol=protocol)
    device = torch.device(protocol.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or device.index >= torch.cuda.device_count()):
        raise RuntimeError('explicit requested CUDA device unavailable; no CPU fallback')
    source_paths = {Path(__file__).resolve(), Path(inspect.getfile(SelectionNetwork)).resolve(),
                    Path(_durable.__file__).resolve()}
    if protocol.backend == 'perturbed':
        from loris.selection import perturbations
        source_paths.add(Path(perturbations.__file__).resolve())
    sources = {str(path): _sha_file(path) for path in source_paths}
    binding = {'schema': SCHEMA, 'protocol': protocol.manifest(),
        'input_binding': json.loads(canonical(input_binding)), 'labels': list(data['labels']),
        'row_ids': data['row_ids'], 'models': [asdict(model) for model in data['models']],
        'features': array_identity(data['features']), 'truth': array_identity(data['truth']),
        'probabilities': array_identity(data['probabilities']),
        'decisions': {key: array_identity(value) for key, value in data['decisions'].items()},
        'cluster_assignments': array_identity(data['assignments']), 'source_files': sources,
        'versions': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                     'torch_threads': torch.get_num_threads(), 'cuda': torch.version.cuda,
                     'cudnn': torch.backends.cudnn.version()},
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another writer owns this selection run') from exc
        immutable_json(directory / 'PROTOCOL.json', binding)
        binding_id = digest(binding)
        results = []
        for position, cluster_id in enumerate(np.unique(data['assignments'])):
            for path, expected in sources.items():
                if _sha_file(path) != expected:
                    raise RuntimeError('bound selection source changed during run: ' + path)
            indices = np.flatnonzero(data['assignments'] == cluster_id)
            results.append(_cluster(directory / ('cluster_%02d' % position), data, indices, int(cluster_id),
                protocol, binding_id, (protocol.seed + 1009 * position) % (2 ** 63), device, on_progress, sources))
        result = {'schema': SCHEMA, 'binding_id': binding_id, 'complete_fit_rows': len(data['row_ids']),
            'clusters': results, 'completed_optimizer_steps': protocol.epochs * len(results),
            'base_model_fit_calls': 0, 'old_rule_or_bank_inputs': 0,
            'static_decisions_used_for_selection_loss': False}
        immutable_json(directory / 'SELECTION.json', result)
        immutable_json(directory / 'COMPLETE.json', {'selection_sha256': digest(result), 'binding_id': binding_id})
        return result
