"""PDF36 §5 model selection with explicit vote or calibrated-F1 supervision.

Opt-in global F1 targets retain the MLP, MC Top-K and final frequency selection.
The margin task evaluates the same calibrated probability mixture as discovery;
its first 25 epochs use imitation only. Historical vote defaults remain available.

Hard Top-K forward, Gaussian-smoothed Monte Carlo backward, weighted oracle
BCE + vote-count two-class CE - score entropy, then all-row frequency Top-K.
For multi-label data each label is a two-class vote; ties choose negative.
CE uses (positive_votes+alpha)/(K+2*alpha), preserving the majority boundary.
Numerically safe vote ratios retain the original calculation. At floating-point
boundaries the same BCE uses float64 log counts; no epsilon or probability
clipping is introduced. Unrepresentable mask gradients are rejected explicitly.
The oracle exhausts EVERY K-subset. The main policy maximizes each document's
number of correct binary votes, then minimizes vote BCE, then model-order ties.
The bce_first comparison reverses those first two priorities. Both use only
binary votes, not base-model confidence. These are local oracle completions,
not claims to globally maximize Macro-F1.
All subset global Macro-F1 scores and the exact best global subset are retained
as diagnostics. The final S remains the paper's frequency aggregation.

For z~N(0,I), the correct Jacobian is E[K(s+sigma*z) z^T]/sigma
(Berthet et al., NeurIPS 2020, Proposition 3.1). PDF36 Eq3 omits this scale.
This module declares the correction rather than silently claiming literal Eq3.
"""
from dataclasses import asdict, dataclass
import copy
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from loris._engine.rlearner import array_identity, canonical, digest, immutable_json
from loris._engine import rlearner as _rlearner_source
from loris._engine import model_selection as _storage_source
from loris._engine.model_selection import (_random_scope, _rng_state, _restore_rng,
    _immutable_npz, _atomic_checkpoint, _sha_file, _load_selector_checkpoint)
from .patterns import macro_f1
from . import patterns as _patterns_source

SCHEMA = 'paper36_aligned_joint_vote_selection_v1'
ORACLE_POLICIES = {
    'accuracy_then_vote_bce': 'all combinations; local binary vote correctness, then vote BCE, then lexicographic model indices',
    'bce_first': 'all combinations; vote BCE, then local binary vote correctness, then lexicographic model indices',
    'global_macro_f1': 'all combinations; whole fitted group macro F1, micro F1 tie break, then stable model order; supervised target for the unchanged neural selector',
    'global_balanced_f1': 'all combinations; mean whole fitted group macro and micro F1, then stable model order; supervised target for the unchanged neural selector',
    'global_margin_macro_f1': 'all combinations; calibrated probability margin ensemble macro F1, then micro; teacher only, final output remains neural frequency selection',
}


@dataclass(frozen=True)
class ModelSelectionProtocol:
    k: int
    epochs: int
    hidden_dim: int
    num_samples: int
    sigma: float
    learning_rate: float
    lambda_task: float
    lambda_entropy: float
    imitation_positive_weight: float
    vote_smoothing: float
    dropout: float
    seed: int
    device: str
    oracle_policy: str = 'accuracy_then_vote_bce'
    task_objective: str = 'vote_ce'

    def __post_init__(self):
        for name in ('k', 'epochs', 'hidden_dim', 'num_samples'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        for name in ('sigma', 'learning_rate', 'imitation_positive_weight', 'vote_smoothing'):
            x = getattr(self, name)
            if type(x) not in (float, int) or not math.isfinite(x) or x <= 0:
                raise ValueError(name + ' must be positive and finite')
        for name in ('lambda_task', 'lambda_entropy', 'dropout'):
            x = getattr(self, name)
            if type(x) not in (float, int) or not math.isfinite(x) or x < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if self.dropout >= 1 or type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('invalid dropout or seed')
        if self.device != 'cpu' and not (isinstance(self.device, str) and self.device.startswith('cuda:') and self.device[5:].isdigit()):
            raise ValueError('explicit cpu or cuda:index required; no fallback')
        if not isinstance(self.oracle_policy, str) or self.oracle_policy not in ORACLE_POLICIES:
            raise ValueError('unknown oracle_policy')
        if self.task_objective not in ('vote_ce', 'pooled_macro_f1', 'pooled_margin_macro_f1'):
            raise ValueError('unknown selector task objective')

    def manifest(self):
        return dict(asdict(self), schema=SCHEMA,
            oracle=ORACLE_POLICIES[self.oracle_policy],
            task=('mean two-class cross entropy on smoothed binary vote counts' if self.task_objective == 'vote_ce'
                  else '0.8 soft macro + 0.2 soft micro F1 of hard frequency pool, using calibrated probability margins' if self.task_objective=='pooled_margin_macro_f1'
                  else '0.8 soft macro + 0.2 soft micro F1 of hard frequency-aggregated selected pool; same MC TopK gradient'),
            mc_control_variate='unperturbed hard mask' if self.task_objective=='pooled_margin_macro_f1' else None,
            vote_bce_numerics='original_safe_vote_ratio_else_float64_equivalent_log_counts; reject_unrepresentable_mask_gradients',
            vote_ties='negative', topk_ties='stable supplied model order',
            gradient='E[TopK(s+sigma*z) z^T] / sigma, z standard Gaussian',
            gradient_is_literal_pdf36_eq3=False,
            numeric_defaults_prescribed_by_pdf=False, global_macro_optimality_claimed=False,
            training=('all assigned fit rows every epoch, one full batch; margin task begins after 25 imitation-only epochs' if self.task_objective=='pooled_margin_macro_f1' else 'all assigned fit rows every configured epoch, one full batch'),
            final_selection='all-fit-row hard TopK selection frequency, stable ties',
            query_budget_independent=True)


def hard_topk(scores, k):
    if scores.ndim < 1 or type(k) is not int or not 1 <= k <= scores.shape[-1] or not torch.isfinite(scores).all():
        raise ValueError('finite scores and valid K required')
    indices = torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :k]
    return torch.zeros_like(scores).scatter_(-1, indices, 1.)


class PerturbedTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, num_samples, sigma):
        if scores.ndim != 2 or not scores.is_floating_point() or type(num_samples) is not int or num_samples < 1:
            raise ValueError('row x model floating scores and positive MC count required')
        if type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('positive finite Gaussian scale required')
        ctx.save_for_backward(scores)
        ctx.k, ctx.num_samples, ctx.sigma = k, num_samples, sigma
        return hard_topk(scores, k)

    @staticmethod
    def backward(ctx, grad_output):
        (scores,) = ctx.saved_tensors
        if ctx.k == scores.shape[-1]:
            # The optimizer is exactly constant: remove finite-sample noise.
            return torch.zeros_like(scores), None, None, None
        noise = torch.randn(ctx.num_samples, *scores.shape, dtype=scores.dtype, device=scores.device)
        selected = hard_topk(scores[None] + ctx.sigma * noise, ctx.k)
        inner = (selected * grad_output[None]).sum(dim=-1)
        gradient = (noise * inner[..., None]).mean(dim=0) / ctx.sigma
        return gradient, None, None, None


class CenteredPerturbedTopK(PerturbedTopK):
    """Same hard forward and expected MC Jacobian, with a constant control variate.

    E[hard_topk(scores) * z] is zero. Removing this term suppresses gradients
    due solely to sample noise when the selected set did not change.
    """
    @staticmethod
    def backward(ctx, grad_output):
        (scores,)=ctx.saved_tensors
        if ctx.k==scores.shape[-1]:return torch.zeros_like(scores),None,None,None
        noise=torch.randn(ctx.num_samples,*scores.shape,dtype=scores.dtype,device=scores.device)
        selected=hard_topk(scores[None]+ctx.sigma*noise,ctx.k)-hard_topk(scores,ctx.k)[None]
        inner=(selected*grad_output[None]).sum(-1)
        return (noise*inner[...,None]).mean(0)/ctx.sigma,None,None,None


def vote_probabilities(mask, decisions, k, smoothing):
    """Differentiable vote fractions; decisions is MODEL x ROW x LABEL."""
    if decisions.ndim != 3 or mask.shape != (decisions.shape[1], decisions.shape[0]):
        raise ValueError('vote decision/mask axes differ')
    if type(k) is not int or not 1 <= k <= decisions.shape[0] or type(smoothing) not in (int, float) or not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('valid K and positive smoothing required')
    if not torch.isfinite(decisions).all() or not ((decisions == 0) | (decisions == 1)).all():
        raise ValueError('explicit binary model decisions required; probabilities are not votes')
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all() or not (mask.sum(dim=1) == k).all():
        raise ValueError('mask must be exact hard K selection')
    return (torch.einsum('rm,mrl->rl', mask, decisions) + smoothing) / (k + 2*smoothing)


def vote_cross_entropy(mask, decisions, truth, k, smoothing):
    p = vote_probabilities(mask, decisions, k, smoothing)
    if truth.shape != p.shape or not ((truth == 0) | (truth == 1)).all():
        raise ValueError('complete binary truth required')
    # NLL of a two-class categorical distribution per label, identical to
    # binary CE but derived from hard decisions, never base probabilities.
    # Preserve normal forward values and gradients bit for bit. Endpoint
    # rounding, scalar overflow and unrepresentable reciprocal derivatives
    # require a different numerical expression of this same loss.
    if (torch.isfinite(p).all() and ((p > 0) & (p < 1)).all()
            and torch.isfinite(p.reciprocal()).all()
            and torch.isfinite((1-p).reciprocal()).all()):
        return -(truth * torch.log(p) + (1-truth) * torch.log1p(-p)).mean()
    votes = torch.einsum('rm,mrl->rl', mask.to(torch.float64), decisions.to(torch.float64))
    correct = torch.where(truth.bool(), votes, k-votes)
    # The true derivative is d BCE/d votes = +/-1/(correct_votes+alpha),
    # followed by the all-cell mean and the binary-vote linear map. Detect an
    # actually unrepresentable input gradient instead of returning NaN/inf or
    # silently clipping it. This does not impose an extra smoothing epsilon.
    with torch.no_grad():
        derivative = torch.where(truth.bool(), -1., 1.).to(torch.float64)
        derivative = (derivative / truth.numel()) / (correct + smoothing)
        mask_gradient = torch.einsum('rl,mrl->rm', derivative, decisions.to(torch.float64))
        if (not torch.isfinite(mask_gradient).all()
                or (mask_gradient.abs() > torch.finfo(mask.dtype).max).any()):
            raise ValueError('smoothed vote BCE mask gradient is not representable in ' + str(mask.dtype))
    if smoothing >= k:
        # Cancel log(alpha) analytically; neither 2*alpha nor a huge sum is
        # ever formed, including for the largest finite float64 alpha.
        losses = math.log(2 + k/smoothing) - torch.log1p(correct/smoothing)
    else:
        losses = torch.empty_like(correct)
        nonzero = correct > 0
        c = correct[nonzero]
        # Keep the small positive loss of an entirely correct vote, even when
        # adding alpha to its integer count would round alpha away.
        losses[nonzero] = (torch.log(k/c) + math.log1p(2*(smoothing/k))
            - torch.log1p(smoothing/c))
        losses[~nonzero] = (math.log(k) + math.log1p(2*(smoothing/k))
            - torch.log(correct[~nonzero] + smoothing))
    return losses.mean().to(p.dtype)


def _oracle_vote_cross_entropy(votes, truth, k, smoothing):
    """Same vote BCE; preserve the original finite ratio path exactly."""
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        p = (votes + smoothing)/(k + 2*smoothing)
    if np.isfinite(p).all() and ((p > 0) & (p < 1)).all():
        return -(truth*np.log(p)+(1-truth)*np.log1p(-p)).mean(axis=1)
    correct = np.where(truth, votes, k-votes).astype(np.float64)
    if smoothing >= k:
        losses = math.log(2 + k/smoothing) - np.log1p(correct/smoothing)
    else:
        losses = np.empty_like(correct)
        nonzero = correct > 0
        c = correct[nonzero]
        losses[nonzero] = (np.log(k/c) + math.log1p(2*(smoothing/k))
            - np.log1p(smoothing/c))
        losses[~nonzero] = math.log(k) + math.log1p(2*(smoothing/k)) - math.log(smoothing)
    return losses.mean(axis=1)


def majority_predictions(decisions, mask):
    d, m = np.asarray(decisions), np.asarray(mask)
    if d.ndim != 3 or m.shape != (d.shape[1], d.shape[0]) or not np.isin(d, [0, 1]).all() or not np.isin(m, [0, 1]).all():
        raise ValueError('complete binary votes and row x model masks required')
    k = m.sum(axis=1)
    if np.any(k < 1):
        raise ValueError('at least one model per row required')
    votes = np.einsum('rm,mrl->rl', m.astype(np.int64), d.astype(np.int64))
    return 2*votes > k[:, None]


def build_joint_vote_oracle(decisions, truth, k, smoothing, *, oracle_policy='accuracy_then_vote_bce', margins=None):
    """Exact all-subset local oracle plus global subset diagnostics; no cap."""
    d, y = np.asarray(decisions), np.asarray(truth)
    if d.ndim != 3 or y.ndim != 2 or not y.size or d.shape[1:] != y.shape or not np.isin(d, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('complete binary model x row x label decisions required')
    if type(k) is not int or not 1 <= k <= len(d) or type(smoothing) not in (int, float) or not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('valid K and positive vote smoothing required')
    if not isinstance(oracle_policy, str) or oracle_policy not in ORACLE_POLICIES:
        raise ValueError('unknown oracle_policy')
    best_correct = np.full(len(y), -1, dtype=np.int64)
    best_ce = np.full(len(y), np.inf)
    oracle = np.zeros((len(y), len(d)), np.float32)
    records = []
    if oracle_policy=='global_margin_macro_f1':
        margins=np.asarray(margins)
        if margins.shape!=d.shape or not np.isfinite(margins).all():raise ValueError('complete calibrated margin supervision required')
    elif margins is not None:raise ValueError('margin supervision requires its explicit oracle policy')
    for subset in itertools.combinations(range(len(d)), k):
        votes = d[list(subset)].astype(np.int64).sum(axis=0)
        pred = margins[list(subset)].sum(0)>=0 if margins is not None else 2*votes > k
        correct = (pred == y).sum(axis=1)
        ce = _oracle_vote_cross_entropy(votes, y, k, smoothing)
        if oracle_policy == 'accuracy_then_vote_bce':
            better = (correct > best_correct) | ((correct == best_correct) & (ce < best_ce))
        elif oracle_policy == 'bce_first':
            better = (ce < best_ce) | ((ce == best_ce) & (correct > best_correct))
        else:
            # Local correctness is diagnostic only for the population-F1 targets.
            better = (correct > best_correct) | ((correct == best_correct) & (ce < best_ce))
        oracle[better] = 0
        for i in subset:
            oracle[better, i] = 1
        best_correct[better], best_ce[better] = correct[better], ce[better]
        records.append({'indices': list(subset), 'macro_f1': macro_f1(y, pred),
            'binary_accuracy': float((pred == y).mean()), 'vote_cross_entropy': float(ce.mean())})
        if oracle_policy.startswith('global_'):
            target = y.astype(bool, copy=False)
            tp = int((pred & target).sum()); fp = int((pred & ~target).sum()); fn = int((~pred & target).sum())
            records[-1]['micro_f1'] = 2*tp/max(2*tp+fp+fn, 1)
    # Stable combination order breaks equal global Macro-F1 scores.
    global_best = max(records, key=lambda r: r['macro_f1'])
    if oracle_policy.startswith('global_'):
        key = ((lambda r: (r['macro_f1'], r['micro_f1'])) if oracle_policy in ('global_macro_f1','global_margin_macro_f1')
               else (lambda r: ((r['macro_f1']+r['micro_f1'])*.5,)))
        teacher = max(records, key=key)
        oracle[:] = 0.; oracle[:, teacher['indices']] = 1.
    return {'mask': oracle, 'correct_labels': best_correct, 'vote_cross_entropy': best_ce,
        'oracle_policy': oracle_policy, 'oracle': ORACLE_POLICIES[oracle_policy],
        'all_subset_scores': records, 'subsets_evaluated': len(records),
        'best_global_macro_subset': global_best, 'local_oracle_global_optimality_claimed': False}


def pooled_macro_task(mask, decisions, truth, k, smoothing, num_samples, sigma):
    """Train the same hard-frequency pool that the original selector exports.

    No learned subset is substituted with the teacher or a hand-picked pool.
    Gradients pass through the original MC TopK at both document and pool levels.
    """
    frequency = mask.mean(0, keepdim=True)
    pool = PerturbedTopK.apply(frequency, k, num_samples, sigma)
    probabilities = (torch.einsum('m,mrl->rl', pool[0], decisions) + smoothing)/(k+2*smoothing)
    tp=(probabilities*truth).sum(0)
    den=probabilities.sum(0)+truth.sum(0)
    macro=(2*tp/den.clamp_min(1e-12)).mean()
    micro=2*tp.sum()/den.sum().clamp_min(1e-12)
    return 1.-(.8*macro+.2*micro)


def interpretation_margins(interpretation, aliases, rows):
    if interpretation is None:raise ValueError('bound calibrated model interpretation required')
    index={r:i for i,r in enumerate(interpretation['row_ids'])};take=[index[r] for r in rows]
    p=np.stack([interpretation['probabilities'][a][take] for a in aliases]).astype(np.float64)
    if not np.isfinite(p).all() or (p<=0).any() or (p>=1).any():raise ValueError('strictly interior calibrated probabilities required')
    return np.log(p/(1-p))


def margin_predictions(margins, mask):
    return np.einsum('rm,mrl->rl',mask.astype(np.float64),margins)>=0


def pooled_margin_task(mask,margins,truth,k,num_samples,sigma):
    pool=CenteredPerturbedTopK.apply(mask.mean(0,keepdim=True),k,num_samples,sigma)
    p=torch.sigmoid(torch.einsum('m,mrl->rl',pool[0],margins)/k)
    tp=(p*truth).sum(0);den=p.sum(0)+truth.sum(0)
    return 1-(.8*(2*tp/den.clamp_min(1e-12)).mean()+.2*2*tp.sum()/den.sum().clamp_min(1e-12))


def _validate(features, decisions, truth, assignments, row_ids, aliases, labels, binding, protocol):
    if not isinstance(protocol, ModelSelectionProtocol):
        raise ValueError('explicit ModelSelectionProtocol required')
    x, y, a = np.asarray(features), np.asarray(truth), np.asarray(assignments)
    if x.ndim != 2 or not x.size or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('full finite float32 feature matrix required')
    if y.ndim != 2 or not y.size or y.shape[0] != len(x) or not np.isin(y, [0, 1]).all():
        raise ValueError('complete binary fit truth required')
    aliases, labels, row_ids = list(aliases), list(labels), list(row_ids)
    for values in (aliases, labels):
        if not values or any(type(s) is not str or not s for s in values) or len(set(values)) != len(values):
            raise ValueError('unique nonempty string model/label axes required')
    if len(labels) != y.shape[1] or protocol.k > len(aliases):
        raise ValueError('K or label axis differs')
    if len(row_ids) != len(x) or any(type(r) not in (str, int) or r == '' for r in row_ids) or len({canonical(r) for r in row_ids}) != len(row_ids):
        raise ValueError('complete unique row IDs required')
    if a.shape != (len(x),) or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) or np.any(a > np.iinfo(np.int64).max):
        raise ValueError('complete nonnegative integer cluster assignments required')
    if isinstance(decisions, Mapping):
        if set(decisions) != set(aliases):
            raise ValueError('decision mapping must cover exactly the full pool')
        d = np.stack([decisions[k] for k in aliases])
    else:
        d = np.asarray(decisions)
    if d.dtype != bool or d.shape != (len(aliases), len(x), len(labels)):
        raise ValueError('explicit boolean decisions for full model x row x label axes required')
    if not isinstance(binding, Mapping) or binding.get('role') != 'fit' or type(binding.get('expected_fit_rows')) is not int or binding.get('expected_fit_rows') != len(x):
        raise ValueError('fit-only binding and complete expected row count required')
    if not isinstance(binding.get('feature_provenance'), Mapping) or not binding['feature_provenance'] or not isinstance(binding.get('decision_provenance'), Mapping) or not binding['decision_provenance']:
        raise ValueError('feature and binary-decision provenance required')
    return x.copy(), d.copy(), y.astype(bool, copy=True), a.astype(np.int64, copy=True), row_ids, aliases, labels


def _normalize_model_interpretation(value, aliases, labels, fit_rows, fit_decisions):
    if value is None:
        return None, {}
    if not isinstance(value, Mapping) or set(value) != {'row_ids', 'probabilities', 'decisions'}:
        raise ValueError('model_interpretation requires row_ids, probabilities and decisions')
    rows = list(value['row_ids'])
    if not rows or any(type(r) not in (str,int) or r == '' for r in rows) or len({canonical(r) for r in rows}) != len(rows):
        raise ValueError('complete unique model interpretation row axis required')
    positions = {canonical(r):i for i,r in enumerate(rows)}
    if any(canonical(r) not in positions for r in fit_rows):
        raise ValueError('model interpretation omits selector fit rows')
    fit_indices = [positions[canonical(r)] for r in fit_rows]
    metadata = {'rows':rows, 'probabilities':{}, 'decisions':{}}
    arrays = {}
    for kind, prefix in [('probabilities','p'),('decisions','d')]:
        tables = value[kind]
        if not isinstance(tables, Mapping) or set(tables)-set(aliases):
            raise ValueError('model interpretation contains undeclared aliases')
        for alias in aliases:
            if alias not in tables:
                continue
            table = np.asarray(tables[alias])
            if table.shape != (len(rows),len(labels)):
                raise ValueError('model interpretation axes differ')
            if kind == 'probabilities':
                if table.dtype.kind != 'f' or not np.isfinite(table).all() or np.any((table<0)|(table>1)):
                    raise ValueError('complete finite probability interpretation required')
            elif table.dtype != bool or not np.array_equal(table[fit_indices],fit_decisions[aliases.index(alias)]):
                raise ValueError('fixed model decisions differ from selector fit votes')
            table = np.array(table,copy=True)
            key = prefix+'%04d'%aliases.index(alias)
            arrays[key] = table
            metadata[kind][alias] = {'array_key':key, 'identity':array_identity(table)}
    if set(metadata['probabilities']) | set(metadata['decisions']) != set(aliases):
        raise ValueError('complete model pool interpretation required')
    return metadata, arrays


def train_model_selection(*, features, decisions, truth, cluster_assignments, row_ids,
        aliases, labels, input_binding, protocol, output_dir, on_progress=None, model_interpretation=None):
    """Train all clusters, resume identical committed epochs; no hidden budget.

    Epochs/MC samples/subsets are unrelated to the LLM/human query budget.
    Only the selector is trained. Caller supplies full fitted-model hard votes
    and embedding features with their own provenance; no hidden model inference.
    """
    x, d, y, a, rows, aliases, labels = _validate(features, decisions, truth,
        cluster_assignments, row_ids, aliases, labels, input_binding, protocol)
    device = torch.device(protocol.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or device.index >= torch.cuda.device_count()):
        raise RuntimeError('requested CUDA device unavailable; no fallback')
    sources = {str(Path(p).resolve()): _sha_file(p) for p in
        (__file__, _rlearner_source.__file__, _storage_source.__file__, _patterns_source.__file__)}
    binding = {'schema': SCHEMA, 'protocol': protocol.manifest(), 'input_binding': dict(input_binding),
        'features': array_identity(x), 'decisions': array_identity(d), 'truth': array_identity(y),
        'assignments': array_identity(a), 'rows': rows, 'aliases': aliases, 'labels': labels,
        'source_files': sources, 'versions': {'python': sys.version, 'numpy': np.__version__,
            'torch': torch.__version__, 'torch_threads': torch.get_num_threads(), 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version()},
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    interpretation_metadata, interpretation_arrays = _normalize_model_interpretation(
        model_interpretation, aliases, labels, rows, d)
    margin_values=(interpretation_margins(model_interpretation,aliases,rows)
                   if protocol.oracle_policy=='global_margin_macro_f1' else None)
    if (protocol.task_objective=='pooled_margin_macro_f1') != (margin_values is not None):
        raise ValueError('margin oracle and margin task must use the same calibrated interpretation')
    binding['artifact_contract'] = 'complete_input_arrays_training_chain_and_model_interpretation_v2'
    binding['model_interpretation'] = interpretation_metadata
    bid = digest(binding)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    with (directory/'.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('selection directory already has a writer') from exc
        immutable_json(directory/'PROTOCOL.json', binding)
        _immutable_npz(directory/'INPUT_ARRAYS.npz', {'features':x, 'decisions':d, 'truth':y, 'assignments':a})
        if interpretation_metadata is not None:
            _immutable_npz(directory/'MODEL_INTERPRETATION.npz', interpretation_arrays)
        for pos, cluster in enumerate(np.unique(a)):
            idx = np.flatnonzero(a == cluster)
            out = directory/('cluster_%02d' % pos)
            out.mkdir(exist_ok=True)
            oracle = build_joint_vote_oracle(d[:, idx], y[idx], protocol.k, protocol.vote_smoothing,
                oracle_policy=protocol.oracle_policy,margins=margin_values[:,idx] if margin_values is not None else None)
            immutable_json(out/'ORACLE.json', {k:v for k,v in oracle.items() if not isinstance(v, np.ndarray)})
            _immutable_npz(out/'oracle.npz', {k:v for k,v in oracle.items() if isinstance(v, np.ndarray)})
            immutable_json(out/'INPUTS.json', {'binding_id': bid, 'cluster': int(cluster),
                'oracle_policy': protocol.oracle_policy,
                'rows': idx.tolist(), 'row_ids': [rows[i] for i in idx],
                'oracle_sha256': _sha_file(out/'oracle.npz'), 'oracle_scores_sha256': _sha_file(out/'ORACLE.json')})
            seed = (protocol.seed + 1009*pos) % 2**63
            with _random_scope(device, seed):
                net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], protocol.hidden_dim), torch.nn.ReLU(),
                    torch.nn.Dropout(protocol.dropout), torch.nn.Linear(protocol.hidden_dim, len(aliases))).to(device)
                opt = torch.optim.Adam(net.parameters(), lr=protocol.learning_rate)
                xt = torch.from_numpy(x[idx]).to(device)
                dt = torch.from_numpy(d[:,idx].astype(np.float32)).to(device)
                yt = torch.from_numpy(y[idx].astype(np.float32)).to(device)
                ot = torch.from_numpy(oracle['mask']).to(device)
                mt=torch.from_numpy(margin_values[:,idx].astype(np.float32)).to(device) if margin_values is not None else None
                receipts = sorted(out.glob('epoch_*.json'))
                previous = None
                for epoch, path in enumerate(receipts):
                    rec = json.loads(path.read_text())
                    if path.name != 'epoch_%05d.json' % epoch or rec.get('epoch') != epoch or epoch > protocol.epochs or rec.get('binding_id') != bid:
                        raise ValueError('noncontiguous/excessive/wrong-bound committed epochs')
                    cp = out/('checkpoint_%05d.pt' % epoch)
                    if rec.get('checkpoint') != cp.name or rec.get('sha256') != _sha_file(cp):
                        raise ValueError('committed checkpoint checksum differs')
                    previous = _load_selector_checkpoint(cp, map_location=device)
                    if previous.get('binding_id') != bid or previous.get('epoch') != epoch or previous.get('cluster') != int(cluster):
                        raise ValueError('checkpoint binding differs')
                    expected_receipt = {'binding_id':bid, 'epoch':epoch, 'checkpoint':cp.name,
                        'sha256':_sha_file(cp), 'rows_used':len(idx),
                        'losses_before_update':previous['losses_before_update'], 'summary':previous['summary']}
                    if rec != expected_receipt:
                        raise ValueError('committed epoch receipt differs from checkpoint')
                if previous is not None:
                    net.load_state_dict(previous['model'], strict=True)
                    opt.load_state_dict(previous['optimizer'])
                    _restore_rng(previous['rng'], device)
                for epoch in range(len(receipts), protocol.epochs+1):
                    if any(_sha_file(p) != h for p,h in sources.items()):
                        raise RuntimeError('bound selection source changed')
                    losses = None
                    if epoch:
                        net.train()
                        scores = net(xt)
                        topk=CenteredPerturbedTopK if mt is not None else PerturbedTopK
                        mask = topk.apply(scores, protocol.k, protocol.num_samples, protocol.sigma)
                        imitate = F.binary_cross_entropy_with_logits(scores, ot,
                            pos_weight=torch.tensor(protocol.imitation_positive_weight, device=device))
                        task = (pooled_margin_task(mask,mt,yt,protocol.k,protocol.num_samples,protocol.sigma) if mt is not None else
                                vote_cross_entropy(mask, dt, yt, protocol.k, protocol.vote_smoothing)
                                if protocol.task_objective == 'vote_ce' else pooled_macro_task(
                                    mask, dt, yt, protocol.k, protocol.vote_smoothing,
                                    protocol.num_samples, protocol.sigma))
                        logp = F.log_softmax(scores, dim=-1)
                        entropy = -(logp.exp()*logp).sum(dim=-1).mean()
                        task_weight=0. if mt is not None and epoch<=25 else protocol.lambda_task
                        loss = imitate + task_weight*task - protocol.lambda_entropy*entropy
                        if not torch.isfinite(loss):
                            raise RuntimeError('nonfinite full-cluster loss')
                        opt.zero_grad()
                        loss.backward()
                        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                            raise RuntimeError('nonfinite selector gradient')
                        opt.step()
                        losses = {k:float(v.detach().cpu()) for k,v in
                            [('imitation',imitate),('task',task),('entropy',entropy),('total',loss)]}
                    net.eval()
                    with torch.no_grad():
                        scores = net(xt)
                        mask = hard_topk(scores, protocol.k)
                    score_np, mask_np = scores.cpu().numpy(), mask.cpu().numpy()
                    freq = mask_np.sum(axis=0, dtype=np.int64)
                    chosen = np.argsort(-freq, kind='stable')[:protocol.k].tolist()
                    static_mask = np.zeros_like(mask_np)
                    static_mask[:, chosen] = 1
                    summary = {'frequency': freq.tolist(), 'selected_indices': chosen,
                        'selected_aliases': [aliases[i] for i in chosen],
                        'selected_subset_macro_f1': macro_f1(y[idx], margin_predictions(margin_values[:,idx],static_mask) if mt is not None else majority_predictions(d[:,idx], static_mask)),
                        'per_document_selection_macro_f1': macro_f1(y[idx], margin_predictions(margin_values[:,idx],mask_np) if mt is not None else majority_predictions(d[:,idx], mask_np))}
                    previous = {'binding_id':bid, 'epoch':epoch, 'cluster':int(cluster), 'model':net.state_dict(),
                        'optimizer':opt.state_dict(), 'rng':_rng_state(device), 'scores':score_np,
                        'masks':mask_np, 'losses_before_update':losses, 'summary':summary}
                    if any(_sha_file(p) != h for p,h in sources.items()):
                        raise RuntimeError('bound source changed during epoch')
                    cp = out/('checkpoint_%05d.pt' % epoch)
                    _atomic_checkpoint(cp, previous)
                    rec = {'binding_id':bid, 'epoch':epoch, 'checkpoint':cp.name, 'sha256':_sha_file(cp),
                           'rows_used':len(idx), 'losses_before_update':losses, 'summary':summary}
                    immutable_json(out/('epoch_%05d.json' % epoch), rec)
                    if on_progress is not None:
                        on_progress(copy.deepcopy(dict(rec, cluster_id=int(cluster))))
                _immutable_npz(out/'final_selection.npz', {'row_indices':idx, 'scores':previous['scores'],
                    'masks':previous['masks'], 'frequency':np.asarray(previous['summary']['frequency'],np.int64)})
                result = dict(previous['summary'], schema=SCHEMA, binding_id=bid, cluster_id=int(cluster),
                    oracle_policy=protocol.oracle_policy,
                    rows=len(idx), completed_epochs=protocol.epochs, optimizer_steps=protocol.epochs,
                    oracle_subsets_evaluated=oracle['subsets_evaluated'],
                    oracle_best_global_macro_subset=oracle['best_global_macro_subset'],
                    final_arrays_sha256=_sha_file(out/'final_selection.npz'),
                    final_checkpoint_sha256=_sha_file(out/('checkpoint_%05d.pt' % protocol.epochs)))
                immutable_json(out/'SELECTION.json', result)
                immutable_json(out/'COMPLETE.json', {'result_sha256':digest(result), 'binding_id':bid})
                results.append(result)
        training_files = {str(path.relative_to(directory)):_sha_file(path)
            for pos in range(len(results)) for path in sorted((directory/('cluster_%02d'%pos)).iterdir())
            if path.is_file() and (path.suffix in ('.json','.pt','.npz'))}
        total = {'schema':SCHEMA, 'binding_id':bid, 'clusters':results,
            'oracle_policy': protocol.oracle_policy,
            'complete_fit_rows':len(rows), 'completed_optimizer_steps':len(results)*protocol.epochs,
            'base_model_fit_calls':0, 'base_model_inference_calls':0, 'full_pool_used':True,
            'test_truth_reads':0, 'query_budget_used':0, 'global_macro_optimality_claimed':False,
            'selected_aliases_by_cluster': {str(r['cluster_id']):r['selected_aliases'] for r in results},
            'training_artifact_sha256':training_files,
            'input_arrays_sha256':_sha_file(directory/'INPUT_ARRAYS.npz'),
            'model_interpretation_sha256':_sha_file(directory/'MODEL_INTERPRETATION.npz') if interpretation_metadata is not None else None}
        immutable_json(directory/'SELECTION.json', total)
        immutable_json(directory/'SELECTED.json', total)
        immutable_json(directory/'COMPLETE.json', {'result_sha256':digest(total), 'binding_id':bid})
        return total


def verify_model_selection_output(directory):
    """Authenticate full producer input/training/selection chain, no retraining."""
    from .selection_validation import verify_model_selection_output as verify
    return verify(directory)
