"""Figure 6's two separate evaluations, with durable fresh configuration search.

Line 7 sees exactly one candidate and the fixed input Gamma.  Line 12 sees the
current selected program plus that candidate.  No previous rule bank is an
input to this module.  Unspecified author choices are explicit protocol fields.
"""
from __future__ import annotations

import copy
import base64
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import time
import weakref

import numpy as np

from loris._engine.discovery_io import sync_file, configure_sqlite, optuna_storage, policy_record


SCHEMA = 'paper36_author_confirmed_two_evaluations_fresh_rules_v1'
LINE7_MODES = ('positive_macro_f1', 'head_polarity_f1')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False,
                      default=lambda obj: dict(obj) if isinstance(obj, Mapping) else _unsupported_json(obj))


def _unsupported_json(value):
    raise TypeError('non-JSON artifact type: ' + type(value).__name__)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


_ARRAY_IDENTITIES = {}


def array_identity(value):
    from loris._engine.hashing import _immutable_lookup, _immutable_store
    cached = _immutable_lookup(_ARRAY_IDENTITIES, value)
    if cached is not None:
        return copy.deepcopy(cached)
    a = np.ascontiguousarray(value)
    result = {'shape': list(a.shape), 'dtype': str(a.dtype),
              'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
    _immutable_store(_ARRAY_IDENTITIES, value, copy.deepcopy(result))
    return result


def _frozen_array(value, dtype=None):
    """Private bytes-backed copy that no NumPy handle can make writeable."""
    a = np.ascontiguousarray(np.array(value, dtype=dtype, copy=True))
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _frozen_gamma(gamma):
    """Immutable private copy of Gamma with identical identity, or the input.

    The evaluator scores only this private copy, so external mutation of the
    caller's object cannot change a score; identities are computed once.
    """
    try:
        from loris._engine.rdl_reference import RefGamma
        frozen = RefGamma(np.asarray(gamma.pos), np.asarray(gamma.neg),
                          tuple(getattr(gamma, 'subset', ())), tuple(getattr(gamma, 'strict_subset', ())))
        object.__setattr__(frozen, 'pos', _frozen_array(frozen.pos))
        object.__setattr__(frozen, 'neg', _frozen_array(frozen.neg))
    except (TypeError, ValueError, AttributeError):
        return gamma
    return frozen if gamma_identity(frozen) == gamma_identity(gamma) else gamma


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as stream:
        stream.write(canonical(value) + '\n')
        sync_file(stream)
    os.replace(str(tmp), str(path))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('immutable artifact differs: ' + str(path))
    else:
        atomic_json(path, value)


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('wb') as stream:
        stream.write(data)
        sync_file(stream)
    os.replace(str(tmp), str(path))
    return hashlib.sha256(data).hexdigest()


def rule_record(rule):
    raw = rule.to_dict() if hasattr(rule, 'to_dict') else dict(rule)
    # All meaning, including variable scope and negative polarity, is retained.
    return json.loads(canonical(raw))


_RULE_CANONICAL = {}


def _rule_canonical(rule):
    """canonical(rule_record(rule)), cached for frozen typed Rule objects."""
    from loris._engine.rdl_reference import Rule
    if type(rule) is not Rule:
        return canonical(rule_record(rule))
    saved = _RULE_CANONICAL.get(id(rule))
    if saved is not None and saved[0]() is rule:
        return saved[1]
    text = canonical(rule_record(rule))
    key = id(rule)
    _RULE_CANONICAL[key] = (weakref.ref(rule, lambda _r, key=key: _RULE_CANONICAL.pop(key, None)), text)
    return text


def binary_metrics(truth, predictions):
    y, p = np.asarray(truth, bool), np.asarray(predictions, bool)
    if y.ndim != 2 or y.shape != p.shape or y.shape[1] == 0:
        raise ValueError('binary metric axes differ')
    tp = np.sum(y & p, axis=0, dtype=np.int64)
    fp = np.sum(~y & p, axis=0, dtype=np.int64)
    fn = np.sum(y & ~p, axis=0, dtype=np.int64)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2. * tp, denom, out=np.zeros(y.shape[1]), where=denom != 0)
    total = int(denom.sum())
    micro = float(2 * tp.sum() / total) if total else 0.
    return {'macro_f1': float(f1.mean()), 'micro_f1': micro,
            'per_label_f1': f1.tolist(), 'tp': tp.tolist(), 'fp': fp.tolist(),
            'fn': fn.tolist(), 'rows': y.shape[0], 'labels': y.shape[1]}


def _result_field(result, name):
    return result[name] if isinstance(result, dict) else getattr(result, name)


def gamma_identity(gamma):
    return {'positive': array_identity(gamma.pos), 'negative': array_identity(gamma.neg),
            'subset': [list(p) for p in getattr(gamma, 'subset', ())],
            'strict_subset': [list(p) for p in getattr(gamma, 'strict_subset', ())]}


def _optional_field(result, name, default=()):
    return result.get(name, default) if isinstance(result, dict) else getattr(result, name, default)


def _sampler_record(sampler, completed_trial):
    raw = pickle.dumps(sampler, protocol=pickle.HIGHEST_PROTOCOL)
    return {'completed_trial': completed_trial, 'sha256': hashlib.sha256(raw).hexdigest(),
            'pickle_base64': base64.b64encode(raw).decode('ascii')}


def _restore_sampler(record):
    # This is a trusted local checkpoint, never an externally supplied pickle.
    raw = base64.b64decode(record['pickle_base64'], validate=True)
    if hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('sampler checkpoint checksum differs')
    return pickle.loads(raw)


class TwoFilterEvaluator:
    """Fixed-data standalone efficacy and marginal whole-program evaluation.

    ``head_info`` returns (label, negative_bool) for a label consequence, or
    None for a set consequence.  It is supplied by the typed rule grammar,
    rather than guessing polarity from a display string.
    """
    def __init__(self, runner, *, labels, receiver_indices, truth, gamma,
                 context_id, line7_mode, head_info, cache_path=None, runtime_binding=None,
                 fit_precision_floor=None):
        if line7_mode not in LINE7_MODES:
            raise ValueError('line7_mode must explicitly name an author-defined score')
        if fit_precision_floor is not None and (
                type(fit_precision_floor) not in (int, float) or not 0 < fit_precision_floor <= 1):
            raise ValueError('fit_precision_floor must be None or in (0, 1]')
        if fit_precision_floor is not None and not getattr(runner, 'gamma_protection', False):
            raise ValueError('a fit precision floor requires a Gamma-protected runner')
        self.fit_precision_floor = fit_precision_floor
        self.runner, self.gamma, self.head_info = runner, gamma, head_info
        self.labels = tuple(labels)
        self.indices = _frozen_array(receiver_indices, dtype=np.int64)
        self.truth = np.asarray(truth)
        self.line7_mode = line7_mode
        pos, neg = np.asarray(gamma.pos), np.asarray(gamma.neg)
        if pos.dtype != bool or neg.dtype != bool or pos.shape != neg.shape:
            raise ValueError('Gamma requires equal Boolean positive/negative axes')
        if pos.ndim != 2 or pos.shape[1] != len(self.labels):
            raise ValueError('Gamma label axes differ')
        if len(set(self.labels)) != len(self.labels) or not self.labels:
            raise ValueError('labels must be nonempty and unique')
        if (self.indices.ndim != 1 or not len(self.indices)
                or len(np.unique(self.indices)) != len(self.indices)
                or np.any(self.indices < 0) or np.any(self.indices >= pos.shape[0])):
            raise ValueError('receiver indices invalid')
        if self.truth.shape != (len(self.indices), len(self.labels)) or not np.isin(self.truth, [0, 1]).all():
            raise ValueError('truth axes or binary values differ')
        if pos[self.indices].any() or neg[self.indices].any():
            raise ValueError('validation truth/facts must not seed Gamma')
        self.truth = _frozen_array(self.truth, dtype=bool)
        self.gamma_signature = gamma_identity(gamma)
        self.gamma = _frozen_gamma(gamma)
        self.binding = {'schema': SCHEMA, 'context_id': context_id, 'labels': list(self.labels),
                        'receivers': array_identity(self.indices), 'truth': array_identity(self.truth),
                        'gamma': self.gamma_signature, 'line7_mode': line7_mode,
                        'line12_mode': 'positive_macro_f1', 'line7_selected_prefix': False,
                        'runtime': runtime_binding if runtime_binding is not None else getattr(runner, 'binding', None),
                        'evaluator_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if fit_precision_floor is not None:
            self.binding['fit_precision_gate'] = {
                'floor': float(fit_precision_floor),
                'precision': 'agreements / (agreements + blocked) on validated Gamma cells of one Label-head rule',
                'no_Gamma_evidence': 'no gate (validation F1 decides)',
                'non_Label_head_rules': 'any blocked Gamma contradiction rejects',
                'line7': 'score -1 below floor', 'line12': 'rejected below floor'}
        self.binding_id = digest(self.binding)
        self.memory, self.actual_calls, self.cache_hits = {}, 0, 0
        self.actual_compute_seconds = 0.
        self.db = None
        if cache_path is not None:
            path = Path(cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(path), timeout=60)
            configure_sqlite(self.db)
            self.db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS programs(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            old = self.db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
            encoded = canonical(self.binding)
            if old and old[0] != encoded:
                self.db.close()
                raise ValueError('cannot reuse a cache with a different data/score/engine binding')
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('binding',?)", (encoded,))
            self.db.commit()

    def program_key(self, rules):
        # Byte-identical to digest({'binding_id': ..., 'ordered_rules': [rule_record(r)...]}):
        # canonical JSON of a dict with sorted keys is the concatenation of the
        # canonical JSON of its values; per-rule text is cached per immutable rule.
        parts = [_rule_canonical(r) for r in rules]
        raw = ('{"binding_id":' + canonical(self.binding_id) + ',"ordered_rules":['
               + ','.join(parts) + ']}')
        return hashlib.sha256(raw.encode()).hexdigest()

    def evaluate(self, rules):
        rules = tuple(rules)
        current_gamma = gamma_identity(self.gamma)
        if current_gamma != self.gamma_signature:
            raise ValueError('fixed discovery Gamma changed')
        key = self.program_key(rules)
        value = self.memory.get(key)
        if value is None and self.db is not None:
            row = self.db.execute('SELECT value FROM programs WHERE key=?', (key,)).fetchone()
            value = json.loads(row[0]) if row else None
        if value is not None:
            self.cache_hits += 1
            return copy.deepcopy(value)
        started = time.perf_counter()
        result = self.runner(rules, gamma=self.gamma)
        elapsed = time.perf_counter() - started
        if gamma_identity(self.gamma) != self.gamma_signature:
            raise ValueError('Chase mutated the fixed discovery Gamma')
        status = _result_field(result, 'status')
        if status not in ('fixpoint', 'conflict'):
            raise ValueError('incomplete Chase cannot be a rule-discovery score')
        pos = np.asarray(_result_field(result, 'positive_facts'))
        neg = np.asarray(_result_field(result, 'negative_facts'))
        pred = np.asarray(_result_field(result, 'predictions'))
        if (pos.shape != self.gamma.pos.shape or neg.shape != pos.shape or pred.shape != pos.shape
                or pos.dtype != bool or neg.dtype != bool):
            raise ValueError('complete corpus result and exact signed axes required')
        if np.any(self.gamma.pos & ~pos) or np.any(self.gamma.neg & ~neg):
            raise ValueError('Chase discarded validated signed Gamma facts')
        conflicts = [[int(i), str(label)] for i, label in _result_field(result, 'conflicts')]
        comparison_conflicts = _optional_field(result, 'comparison_conflicts')
        if status == 'fixpoint' and (conflicts or comparison_conflicts or np.any(pos & neg)):
            raise ValueError('fixpoint contains signed conflicts')
        if status == 'conflict' and not conflicts:
            # A strict-comparison contradiction may not be a label-cell clash.
            # The typed engine can expose structured relation conflicts below.
            if not comparison_conflicts:
                raise ValueError('conflict status lacks a recorded conflict')
        metrics = binary_metrics(self.truth, pos[self.indices]) if status == 'fixpoint' else None
        negative_metrics = binary_metrics(~self.truth, neg[self.indices]) if status == 'fixpoint' else None
        value = {'program_key': key, 'status': status,
                 'n_rounds': int(_result_field(result, 'n_rounds')), 'conflicts': conflicts,
                 'positive_metrics': metrics, 'negative_metrics': negative_metrics,
                 'full_state': {name: array_identity(a) for name, a in
                                [('positive_facts', pos), ('negative_facts', neg), ('predictions', pred)]}}
        for field in ('subset_facts', 'strict_subset_facts'):
            facts = _optional_field(result, field)
            required = set(getattr(self.gamma, 'subset' if field == 'subset_facts' else 'strict_subset', ()))
            if any(pair not in facts for pair in required):
                raise ValueError('Chase discarded validated comparison Gamma facts')
            value['full_state'][field] = facts.identity() if hasattr(facts, 'identity') else {
                'count': len(facts), 'sha256': digest([list(map(int, pair)) for pair in facts])}
        value['comparison_conflicts'] = json.loads(canonical(list(comparison_conflicts)))
        protection = _optional_field(result, 'gamma_protection', None)
        value['gamma_protection'] = (None if protection is None else
                                     {'blocked': int(protection['blocked']),
                                      'agreements': (None if protection['agreements'] is None
                                                     else int(protection['agreements']))})
        proof = _optional_field(result, 'proof_dag', None)
        if proof is not None:
            value['full_state']['proof_dag'] = proof.identity() if hasattr(proof, 'identity') else {
                'count': len(proof), 'sha256': digest(proof)}
        self.actual_calls += 1
        self.actual_compute_seconds += elapsed
        self.memory[key] = copy.deepcopy(value)
        if self.db is not None:
            self.db.execute('INSERT INTO programs VALUES(?,?)', (key, canonical(value)))
            self.db.commit()
        return copy.deepcopy(value)

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def fit_precision_gate(self, rule, result=None):
        """None if the rule passes the fit precision floor, else a reason string."""
        if self.fit_precision_floor is None:
            return None
        result = self.evaluate((rule,)) if result is None else result
        if result['status'] != 'fixpoint':
            return None
        record = result.get('gamma_protection') or {'blocked': 0, 'agreements': None}
        blocked, agreements = record['blocked'], record['agreements']
        if agreements is None:
            return 'blocked_Gamma_contradiction' if blocked else None
        if blocked + agreements == 0:
            return None
        precision = agreements / (agreements + blocked)
        return None if precision >= self.fit_precision_floor else 'fit_precision_below_floor'

    def standalone(self, rule):
        # A selected-prefix argument is intentionally absent.
        result = self.evaluate((rule,))
        score = -1.
        metric_name = self.line7_mode
        gate = self.fit_precision_gate(rule, result)
        if gate is not None:
            return {'line': 7, 'score': -1., 'metric': gate, 'selected_prefix_used': False,
                    'standalone_rule_count': 1, 'evaluation': result}
        if result['status'] == 'fixpoint':
            head = self.head_info(rule)
            if self.line7_mode == 'head_polarity_f1' and head is not None:
                label, is_negative = head
                index = self.labels.index(label)
                kind = 'negative_metrics' if is_negative else 'positive_metrics'
                score = result[kind]['per_label_f1'][index]
                metric_name = 'negative_head_f1' if is_negative else 'positive_head_f1'
            else:
                score = result['positive_metrics']['macro_f1']
                metric_name = 'positive_macro_f1'
        return {'line': 7, 'score': float(score), 'metric': metric_name,
                'selected_prefix_used': False, 'standalone_rule_count': 1, 'evaluation': result}

    def admit(self, selected, rule, epsilon=1e-12):
        baseline = self.evaluate(tuple(selected))
        if baseline['status'] != 'fixpoint':
            raise ValueError('selected program must be consistent')
        gate = self.fit_precision_gate(rule)
        if gate is not None:
            return {'line': 12, 'accepted': False, 'criterion': 'positive_macro_f1',
                    'delta_macro_f1': None, 'baseline': baseline,
                    'evaluation': self.evaluate((rule,)), 'reason': gate}
        proposed = self.evaluate(tuple(selected) + (rule,))
        delta = None
        if proposed['status'] == 'fixpoint':
            delta = proposed['positive_metrics']['macro_f1'] - baseline['positive_metrics']['macro_f1']
        accepted = delta is not None and delta > epsilon
        return {'line': 12, 'accepted': bool(accepted), 'criterion': 'positive_macro_f1',
                'delta_macro_f1': delta, 'baseline': baseline, 'evaluation': proposed,
                'reason': 'strict_macro_improvement' if accepted else
                          ('conflict' if proposed['status'] == 'conflict' else 'no_strict_macro_improvement')}


_FAMILY_AXES = {}


def _family_counts(space, params, compact_value=None):
    """Per-family ON counts over the complete axis (every family key present).

    With a sealed compact value the ON indices are exactly the params' 1 bits,
    so counting them equals the dense per-feature sum.
    """
    features = tuple(getattr(space, 'features', ()))
    if compact_value is None or not features:
        counts = {}
        for feature in features:
            counts[feature.kind] = counts.get(feature.kind, 0) + int(params[feature.include_key])
        return counts
    saved = _FAMILY_AXES.get(id(space))
    if saved is None or saved[0] is not features:
        kinds = tuple(feature.kind for feature in features)
        saved = (features, kinds, tuple(dict.fromkeys(kinds)))
        _FAMILY_AXES.clear()
        _FAMILY_AXES[id(space)] = saved
    counts = dict.fromkeys(saved[2], 0)
    for index in compact_value['on_indices']:
        counts[saved[1][index]] += 1
    return counts


def run_cluster_search(*, directory, space, evaluator, seed, budget,
                       rule_parser, sampler_factory=None, sampler_binding=None, on_progress=None,
                       storage_mode='dense_v1'):
    """All fresh BO evaluations precede any admission in this cluster.

    Space API: .manifest(), .suggest(trial) -> params, .configuration_receipt,
    and optional .initial_configurations(seed=..., count=...). Every configured
    body (including an invalid body's reason) is sealed before evaluation.
    A pending record holds the sampler immediately before tell; a committed
    checkpoint holds it after tell. Both are atomic, allowing recovery even if
    the process stops between Optuna's transaction and our checkpoint commit.
    """
    import optuna
    from loris._engine.compact_storage import (STORAGE_MODE, FIXED_ATTR, ParameterCodec,
                                              CompactTrial, SamplerCheckpointStore)
    if storage_mode not in ('dense_v1', STORAGE_MODE):
        raise ValueError('unknown explicit BO persistence protocol')
    compact = storage_mode == STORAGE_MODE
    if budget <= 0:
        raise ValueError('positive complete BO budget required')
    if sampler_factory is not None and sampler_binding is None:
        raise ValueError('custom BO sampler must have an explicit persisted protocol')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Operational history is separate from the proposal/score binding. Changing
    # durability never changes or bypasses the exact resume checks below.
    with (directory / 'IO_ATTEMPTS.jsonl').open('a') as stream:
        stream.write(canonical(dict(policy_record(), pid=os.getpid(), started_unix=time.time())) + '\n')
        sync_file(stream)
    codec = ParameterCodec.from_space(space) if compact else None
    state_store = SamplerCheckpointStore(directory / 'sampler_states', codec) if compact else None
    save_sampler = state_store.save if compact else _sampler_record
    restore_sampler = state_store.restore if compact else _restore_sampler
    protocol = {'schema': SCHEMA, 'space': space.manifest(), 'evaluator': evaluator.binding,
                'seed': int(seed), 'budget': int(budget), 'whole_program_prefix_used_in_BO': False,
                'historical_candidates_loaded': False,
                'sampler': sampler_binding or {'class': 'optuna.samplers.TPESampler',
                    'optuna_version': optuna.__version__, 'n_startup_trials': 30}}
    if compact:
        protocol['persistence'] = {'mode': STORAGE_MODE, 'zero_default': 0,
            'include_axis_sha256': codec.axis_sha256, 'all_include_indicators_retained': True,
            'proposal_vectors_and_RNG_unchanged': True, 'conditional_parameters': 'original Optuna Trial',
            'sampler_state': 'SHA-bound static snapshot, dynamic RNG snapshots, complete observation prefix',
            'source_sha256': hashlib.sha256(Path(__file__).with_name('compact_storage.py').read_bytes()).hexdigest()}
    protocol_id = digest(protocol)
    immutable_json(directory / 'protocol.json', protocol)
    # Persist the complete manifest, then release its detached Python copy.
    # Every subsequent record is bound to the same canonical protocol_id.
    del protocol
    sampler_path, pending_path = directory / 'sampler.json', directory / 'pending.json'
    if sampler_path.exists():
        checkpoint = json.loads(sampler_path.read_text())
        sampler = restore_sampler(checkpoint)
    else:
        sampler = (sampler_factory(seed) if sampler_factory else
                   optuna.samplers.TPESampler(seed=seed, n_startup_trials=30))
        checkpoint = save_sampler(sampler, -1)
        atomic_json(sampler_path, checkpoint)
    study = optuna.create_study(study_name=directory.name, direction='maximize', sampler=sampler,
                               storage=optuna_storage(directory / 'optuna.sqlite'),
                               load_if_exists=True)
    def validate_complete_records():
        for finished in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            record = json.loads((directory / ('trial_%05d.json' % finished.number)).read_text())
            actual = codec.frozen_parameters(finished) if compact else finished.params
            if (record['number'] != finished.number or record['protocol_id'] != protocol_id
                    or record['params'] != actual or record['standalone']['score'] != finished.value
                    or record['rule'] != finished.user_attrs.get('rule_body')):
                raise ValueError('completed Optuna trial differs from its durable body/score receipt')
            params = codec.decode(record['params']) if compact else record['params']
            configuration = space.configuration_receipt(params)
            if compact:
                configuration['params'] = {'same_as_trial_params': True}
            if record['configuration'] != configuration:
                raise ValueError('saved configuration no longer instantiates to the same typed rule')
    validate_complete_records()
    if hasattr(space, 'initial_configurations'):
        initial_kwargs = {}
        if hasattr(sampler, 'feature_probabilities'):
            initial_kwargs['feature_probabilities'] = sampler.feature_probabilities
        initializer = (getattr(space, 'iter_initial_configurations', space.initial_configurations)
                       if compact else space.initial_configurations)
        configs = initializer(seed=seed, count=min(30, budget), **initial_kwargs)
        init = [codec.encode(p) for p in configs] if compact else list(configs)
        immutable_json(directory / 'initial_configurations.json', init)
        existing = study.get_trials(deepcopy=False)
        # Also verify a fully enqueued prefix and completed starts. A complete
        # enqueue transaction is not evidence that its stored fixed parameters
        # still belong to this seed/manifest after an interruption.
        for i, trial in enumerate(existing[:len(init)]):
            fixed = trial.user_attrs.get(FIXED_ATTR) if compact else trial.system_attrs.get('fixed_params')
            if (fixed != init[i] or (compact and
                trial.system_attrs.get('fixed_params') != init[i]['active_parameters'])):
                raise ValueError('initial configuration order differs')
        # Complete an interrupted initialization without adding duplicate starts.
        if len(existing) < len(init):
            if any(t.state != optuna.trial.TrialState.WAITING for t in existing):
                raise ValueError('BO started before initialization completed')
            for params in init[len(existing):]:
                if compact:
                    study.enqueue_trial(params['active_parameters'], user_attrs={FIXED_ATTR: params})
                else:
                    study.enqueue_trial(params)
    while True:
        trials = study.get_trials(deepcopy=False)
        complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        if any(t.state in (optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED) for t in trials):
            raise ValueError('failed/pruned trials do not fulfill the full budget')
        if len(complete) > budget:
            raise ValueError('budget exceeded')
        if pending_path.exists():
            pending = json.loads(pending_path.read_text())
            if pending['protocol_id'] != protocol_id:
                raise ValueError('interrupted body belongs to a different protocol')
            previous = trials[pending['number']]
            actual = codec.frozen_parameters(previous) if compact else previous.params
            if actual != pending['params']:
                raise ValueError('interrupted Optuna parameters differ from sealed configuration')
            if previous.state == optuna.trial.TrialState.COMPLETE:
                receipt = directory / ('trial_%05d.json' % pending['number'])
                if not receipt.exists():
                    raise ValueError('completed BO trial is missing its durable body and score')
                record = json.loads(receipt.read_text())
                if any(record.get(k) != v for k, v in pending.items()):
                    raise ValueError('completed trial receipt differs from sealed body')
                if previous.value != record['standalone']['score']:
                    raise ValueError('completed Optuna score differs from receipt')
                if checkpoint['completed_trial'] == pending['number'] - 1:
                    # tell committed but our checkpoint did not. Replay just
                    # its sampler callback, never its expensive Chase score.
                    study.sampler = restore_sampler(pending['sampler_before_tell'])
                    study.sampler.after_trial(study, previous, previous.state, previous.values)
                    checkpoint = save_sampler(study.sampler, pending['number'])
                    atomic_json(sampler_path, checkpoint)
                elif checkpoint['completed_trial'] != pending['number']:
                    raise ValueError('sampler commit and pending trial sequence differ')
                pending_path.unlink()
                continue
            if previous.state != optuna.trial.TrialState.RUNNING:
                raise ValueError('interrupted trial state differs')
            if checkpoint['completed_trial'] != pending['number'] - 1:
                raise ValueError('running trial does not follow sampler checkpoint')
            study.sampler = restore_sampler(pending['sampler_before_tell'])
            trial = optuna.trial.Trial(study, previous._trial_id)
        else:
            if len(complete) == budget:
                if checkpoint['completed_trial'] != budget - 1:
                    raise ValueError('completed budget lacks its final sampler checkpoint')
                break
            if any(t.state == optuna.trial.TrialState.RUNNING for t in trials):
                raise ValueError('unsealed interrupted candidate; recovery must not silently resample')
            if checkpoint['completed_trial'] != len(complete) - 1:
                raise ValueError('sampler checkpoint does not follow completed trials')
            trial = study.ask()
            suggestion_trial = CompactTrial(trial, codec) if compact else trial
            params = space.suggest(suggestion_trial)
            if params != suggestion_trial.params:
                raise ValueError('complete space parameters must match actual Optuna suggestions')
            stored_params = suggestion_trial.seal() if compact else dict(params)
            configuration = space.configuration_receipt(params)
            if compact:
                # Canonical configuration_id remains the SHA of the original
                # dense receipt, while parameters are stored only once.
                configuration['params'] = {'same_as_trial_params': True}
            actual_by_family = _family_counts(space, params, stored_params if compact else None)
            pending = {'number': trial.number, 'params': stored_params, 'rule': configuration['rule'],
                       'configuration': configuration, 'no_body_reason': configuration['invalid_reason'],
                       'actual_body_indicators': sum(actual_by_family.values()),
                       'actual_body_indicators_by_family': actual_by_family,
                       'sampler_before_tell': save_sampler(study.sampler, trial.number - 1),
                       'protocol_id': protocol_id}
            if hasattr(study.sampler, 'last_diagnostic'):
                pending['sampler_diagnostic'] = copy.deepcopy(study.sampler.last_diagnostic)
            atomic_json(pending_path, pending)
            # Compact receipts retain all indicators through the immutable
            # axis and ON indices. These temporary dense proposal dictionaries
            # are no longer needed while the full Chase is being evaluated.
            del params, suggestion_trial, configuration
        raw = pending['rule']
        path = directory / ('trial_%05d.json' % trial.number)
        if path.exists():
            record = json.loads(path.read_text())
            if any(record.get(k) != v for k, v in pending.items()):
                raise ValueError('interrupted evaluation receipt differs from sealed body')
            evaluation = record['standalone']
        else:
            evaluation = evaluator.standalone(rule_parser(raw)) if raw is not None else {
                'line': 7, 'score': -1., 'selected_prefix_used': False,
                'evaluation': None, 'reason': pending['no_body_reason']}
            record = dict(pending, standalone=evaluation)
            immutable_json(path, record)
        if trial.user_attrs.get('rule_body') is None:
            trial.set_user_attr('rule_body', raw)
        study.tell(trial, evaluation['score'])
        checkpoint = save_sampler(study.sampler, trial.number)
        atomic_json(sampler_path, checkpoint)
        pending_path.unlink()
        if on_progress:
            on_progress({'phase': 'line7_standalone_BO', 'complete': len(complete) + 1,
                         'budget': budget, 'last_standalone_score': evaluation['score'],
                         'actual_chase_calls': evaluator.actual_calls, 'cache_hits': evaluator.cache_hits})
    records = []
    validate_complete_records()
    for number in range(budget):
        path = directory / ('trial_%05d.json' % number)
        record = json.loads(path.read_text())
        if record['number'] != number or record['protocol_id'] != protocol_id:
            raise ValueError('complete original trial order and protocol required')
        records.append(record)
    immutable_json(directory / 'BO_COMPLETE.json', {
        'protocol_id': protocol_id, 'complete_trials': budget,
        'all_trial_records_sha256': digest(records), 'newly_generated': True})
    return records


def admit_cluster(*, directory, records, selected, evaluator, rule_parser, on_progress=None):
    directory = Path(directory)
    # Called only with a completed cluster BO receipt; it must not be possible
    # to start stage 12 while stage 7 is still generating configurations.
    complete_path = directory.parent / 'BO_COMPLETE.json'
    if not complete_path.exists():
        raise ValueError('finish all cluster BO trials before sequential admission')
    complete = json.loads(complete_path.read_text())
    protocol = json.loads((directory.parent / 'protocol.json').read_text())
    if (complete['protocol_id'] != digest(protocol) or protocol['evaluator'] != evaluator.binding
            or protocol['whole_program_prefix_used_in_BO'] or protocol['historical_candidates_loaded']):
        raise ValueError('admission requires the same fresh BO data/score/runtime protocol')
    del protocol
    if (complete['complete_trials'] != len(records)
            or complete['all_trial_records_sha256'] != digest(records)):
        raise ValueError('admission candidates differ from the completed fresh BO')
    selected = list(selected)
    immutable_json(directory / 'before.json', {'rules': [rule_record(r) for r in selected],
                                              'evaluation': evaluator.evaluate(selected)})
    for i, record in enumerate(records):
        raw = record['rule']
        if raw is None:
            decision = {'line': 12, 'accepted': False, 'reason': record['no_body_reason']}
        else:
            rule = rule_parser(raw)
            decision = evaluator.admit(selected, rule)
        receipt = {'trial_number': i, 'candidate': raw, 'standalone': record['standalone'],
                   'decision': decision, 'selected_before': len(selected)}
        if decision['accepted']:
            selected.append(rule)
        receipt['selected_after'] = len(selected)
        receipt['selected_program_key'] = evaluator.program_key(selected)
        immutable_json(directory / ('candidate_%05d.json' % i), receipt)
        if on_progress:
            on_progress({'phase': 'line12_whole_program_admission', 'processed': i + 1,
                         'budget': len(records), 'selected': len(selected),
                         'actual_chase_calls': evaluator.actual_calls, 'cache_hits': evaluator.cache_hits})
    immutable_json(directory / 'after.json', {'rules': [rule_record(r) for r in selected],
                                             'evaluation': evaluator.evaluate(selected),
                                             'all_candidates_processed': len(records)})
    return selected


def learn_fresh_program(*, directory, spaces, evaluator, seed, budget_per_cluster,
                        rule_parser, on_progress=None, storage_mode='dense_v1'):
    """Figure 6's cluster order, with a single unified BO study per cluster.

    Selected patterns/model inputs are supplied explicitly by the caller; this
    function does not claim to reproduce their upstream extraction/selection.
    All candidate bodies and both evaluations remain available after completion.
    """
    from loris._engine.sparse_bo import JointBinaryTPESampler
    spaces, directory = tuple(spaces), Path(directory)
    if not spaces or len({s.cluster_id for s in spaces}) != len(spaces):
        raise ValueError('nonempty uniquely identified cluster spaces required')
    protocol = {'schema': SCHEMA, 'seed': int(seed), 'budget_per_cluster': int(budget_per_cluster),
        'total_BO_budget': len(spaces) * int(budget_per_cluster), 'initial_sigma': [],
        'historical_rules_or_banks_loaded': False, 'evaluator': evaluator.binding,
        'spaces': [s.manifest() for s in spaces], 'studies_per_cluster': 1,
        'algorithm': 'cluster BO complete, then sequential whole-program admission',
        'BO_prior': 'equal expected length mass across all represented predicate families',
        'upstream_scope': 'explicit supplied P_i and S_i; upstream extraction/model selection is caller-owned',
        'seed_policy': 'one experiment seed, deterministic cluster offset 1009 * cluster_position'}
    if storage_mode != 'dense_v1':
        protocol['persistence_mode'] = storage_mode
    protocol_id = digest(protocol)
    immutable_json(directory / 'protocol.json', protocol)
    del protocol
    selected, cluster_results = [], []
    def progress(record):
        payload = dict(record, total_clusters=len(spaces), protocol_id=protocol_id,
                       actual_compute_seconds_this_process=evaluator.actual_compute_seconds)
        atomic_json(directory / 'progress.json', payload)
        if on_progress:
            on_progress(payload)
    for position, space in enumerate(spaces):
        cluster_seed = int(seed) + 1009 * position
        families = {f.include_key: f.kind for f in space.features}
        def factory(_seed):
            return JointBinaryTPESampler(tuple(families), seed=_seed, feature_families=families)
        sampler_binding = factory(cluster_seed).protocol
        folder = directory / ('cluster_%02d' % position)
        def update(record):
            progress(dict(record, cluster_position=position, cluster_id=space.cluster_id))
        records = run_cluster_search(directory=folder, space=space, evaluator=evaluator,
            seed=cluster_seed, budget=budget_per_cluster, rule_parser=rule_parser,
            sampler_factory=factory, sampler_binding=sampler_binding, on_progress=update,
            storage_mode=storage_mode)
        selected = admit_cluster(directory=folder / 'admission', records=records, selected=selected,
                                 evaluator=evaluator, rule_parser=rule_parser, on_progress=update)
        cluster_results.append({'cluster_id': space.cluster_id, 'complete_BO_trials': len(records),
            'admissible_configurations': sum(r['rule'] is not None for r in records),
            'standalone_conflicts': sum(r['standalone'].get('evaluation') is not None and
                r['standalone']['evaluation']['status'] == 'conflict' for r in records),
            'standalone_zero_scores': sum(r['standalone']['score'] == 0 for r in records),
            'selected_after_cluster': len(selected)})
    final = {'schema': SCHEMA, 'protocol_id': protocol_id, 'rules': [rule_record(r) for r in selected],
             'evaluation': evaluator.evaluate(selected), 'clusters': cluster_results,
             'complete_BO_trials': len(spaces) * budget_per_cluster,
             'all_candidates_sequentially_evaluated': True, 'historical_candidates_added': 0}
    immutable_json(directory / 'SELECTED.json', final)
    immutable_json(directory / 'COMPLETE.json', {'protocol_id': protocol_id,
        'selected_artifact_sha256': digest(final), 'selected_rules': len(selected),
        'complete_BO_trials': len(spaces) * budget_per_cluster})
    progress({'phase': 'complete', 'selected': len(selected), 'complete_BO_trials': len(spaces) * budget_per_cluster})
    return selected
