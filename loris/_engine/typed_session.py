"""Frozen scoped-RDL sessions with complete states and explicit root trust.

This is a new adapter around IndexedChaseRunner; the legacy runtime is not
modified. Whole-result reuse is exact, bounded, and returns a new result object
with immutable arrays. No inference, rule selection, or training is skipped.
"""
from collections import OrderedDict
from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from loris.chase import ChaseResult, GammaSeed
from loris._engine.rdl_reference import ML, RefGamma, Rule
from loris._engine.typed_runtime import IndexedChaseRunner, ProofDAG, RelationFacts


SCHEMA = 'paper36_scoped_frozen_chase_session_v1'
PHASES = ('labeling', 'feedback', 'external_evidence_check', 'influence_counterfactual')


def plain(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_default(value):
    # Let the JSON encoder traverse native containers/scalars in C. Walking
    # every scalar through Mapping/NumPy checks first dominated large history
    # fingerprints. These are exactly the conversions performed by plain().
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Object of type %s is not JSON serializable' % type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, default=_json_default, sort_keys=True,
        ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_identity(value):
    value = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def seed_identity(seed):
    return {'positive': array_identity(seed.pos), 'negative': array_identity(seed.neg),
            'subset': list(getattr(seed, 'subset', ())),
            'strict_subset': list(getattr(seed, 'strict_subset', ()))}


def complete_state_identity(result):
    """Bind all logical facts and their complete first-witness proof DAG."""
    for field in ('subset_facts', 'strict_subset_facts', 'comparison_conflicts', 'proof_dag'):
        if not hasattr(result, field):
            raise ValueError('complete typed Chase state required; missing ' + field)
    def relation(value):
        return value.identity() if isinstance(value, RelationFacts) else {
            'schema': 'complete_relation_pairs', 'pairs': sorted(tuple(x) for x in value)}
    proof = result.proof_dag
    body = {'status': result.status, 'rounds': int(result.n_rounds),
        'positive': array_identity(result.positive_facts), 'negative': array_identity(result.negative_facts),
        'predictions': array_identity(result.predictions),
        'label_conflicts': plain(result.conflicts), 'comparison_conflicts': plain(result.comparison_conflicts),
        'subset': relation(result.subset_facts), 'strict_subset': relation(result.strict_subset_facts),
        'proof_dag': proof.identity() if isinstance(proof, ProofDAG) else digest(proof),
        'legacy_provenance': sorted([[int(i), str(label), list(ids)]
                                     for (i, label), ids in result.provenance.items()])}
    return {**body, 'sha256': digest(body)}


def _readonly(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _clone(result):
    clone = ChaseResult(status=result.status, predictions=_readonly(result.predictions),
        conflicts=copy.deepcopy(result.conflicts), n_rounds=result.n_rounds,
        provenance=copy.deepcopy(result.provenance), positive_facts=_readonly(result.positive_facts),
        negative_facts=_readonly(result.negative_facts))
    clone.subset_facts = result.subset_facts
    clone.strict_subset_facts = result.strict_subset_facts
    clone.proof_dag = result.proof_dag
    clone.comparison_conflicts = copy.deepcopy(result.comparison_conflicts)
    clone.reference_audit = copy.deepcopy(result.reference_audit)
    if hasattr(result, 'typed_session_audit'):
        clone.typed_session_audit = copy.deepcopy(result.typed_session_audit)
    return clone


def _root_keys(seed):
    keys = set()
    for sign, array in (('pos', seed.pos), ('neg', seed.neg)):
        keys.update('%s:%d:%d' % (sign, i, j) for i, j in np.argwhere(array))
    keys.update('subset:%d:%d' % edge for edge in getattr(seed, 'subset', ()))
    keys.update('strict_subset:%d:%d' % edge for edge in getattr(seed, 'strict_subset', ()))
    return keys


class TypedChaseSession:
    """One complete selected Sigma and one immutable model interpretation.

    Without selection_directory, the session is explicitly a diagnostic fixture.
    A production caller supplies the completed fresh learner's directory; its
    full protocol and selected program are authenticated before any execution.
    ``root_sources`` is a mapping from ``pos:i:j`` / ``neg:i:j`` (or relation
    fact keys) to source receipts. Feedback requires an exact receipt for every
    weak root; it never silently treats the runner's default Gamma as a seed.
    """
    def __init__(self, context, rules, cache_dir, *, model_snapshot_id,
                 subset_direction='standard', selection_directory=None,
                 cache_entries=4, verify_source_files=True, **runner_options):
        if not isinstance(model_snapshot_id, str) or not model_snapshot_id:
            raise ValueError('explicit immutable model_snapshot_id required')
        if type(cache_entries) is not int or cache_entries < 0:
            raise ValueError('cache_entries must be a nonnegative integer')
        self.rules = tuple(Rule.from_dict(r.to_dict() if isinstance(r, Rule) else r) for r in rules)
        self.program_id = digest([r.to_dict() for r in self.rules])
        self.model_snapshot_id = self.snapshot_id = model_snapshot_id
        self._snapshot_id = model_snapshot_id
        self.selection = self._selection(selection_directory)
        self.runner = IndexedChaseRunner(context, cache_dir, subset_direction=subset_direction,
            verify_source_files=verify_source_files, **runner_options)
        if self.runner.application_scope is not None:
            self.runner.application_scope.validate_rules(self.rules)
        self.labels = self.runner.labels
        if self.selection.get('selection_complete_claim') and tuple(self.selection['labels']) != self.labels:
            self.runner.close()
            raise ValueError('selected program and inference label axes differ')
        self.context_id = self.runner.context_id
        self.shape = (self.runner.n, self.runner.k)
        self.cache_entries = cache_entries
        self._cache = OrderedDict()
        self.calls = []
        self.actual_chase_calls = self.cache_hits = 0
        self._source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._binding = {'schema': SCHEMA, 'context_id': self.context_id, 'Sigma': self.program_id,
            'theta': self.snapshot_id, 'selection': self.selection,
            'complete_state_and_proof_bound': True, 'result_reuse_policy': 'exact_frozen_seed_phase_and_root_receipts',
            'feedback_default_Gamma_forbidden': True, 'source_sha256': self._source_sha}
        if self.runner.application_scope is not None:
            self._binding['application_scope'] = self.runner.application_scope.receipt()

    def _selection(self, directory):
        if directory is None:
            return {'mode': 'diagnostic_fixture', 'selection_complete_claim': False}
        directory = Path(directory).resolve(strict=True)
        selected = json.loads((directory / 'SELECTED.json').read_text())
        complete = json.loads((directory / 'COMPLETE.json').read_text())
        protocol = json.loads((directory / 'protocol.json').read_text())
        if (complete.get('selected_artifact_sha256') != digest(selected)
                or complete.get('protocol_id') != digest(protocol)
                or selected.get('protocol_id') != digest(protocol)
                or selected.get('rules') != [r.to_dict() for r in self.rules]
                or selected.get('historical_candidates_added') != 0
                or selected.get('all_candidates_sequentially_evaluated') is not True
                or selected.get('complete_BO_trials') != protocol.get('total_BO_budget')
                or complete.get('complete_BO_trials') != protocol.get('total_BO_budget')
                or complete.get('selected_rules') != len(self.rules)):
            raise ValueError('selected Sigma does not match completed fresh rule-learning receipt')
        if not protocol.get('spaces') or protocol.get('historical_rules_or_banks_loaded') is not False:
            raise ValueError('fresh learning protocol required')
        budget = protocol.get('budget_per_cluster')
        if (type(budget) is not int or budget < 1
                or len(protocol['spaces']) * budget != protocol.get('total_BO_budget')):
            raise ValueError('declared full BO budget is inconsistent')
        previous = []
        for index, space in enumerate(protocol['spaces']):
            folder = directory / ('cluster_%02d' % index)
            bo = json.loads((folder / 'BO_COMPLETE.json').read_text())
            cluster_protocol = json.loads((folder / 'protocol.json').read_text())
            before = json.loads((folder / 'admission' / 'before.json').read_text())
            admission = json.loads((folder / 'admission' / 'after.json').read_text())
            if (bo.get('complete_trials') != budget or admission.get('all_candidates_processed') != budget
                    or bo.get('protocol_id') != digest(cluster_protocol)
                    or cluster_protocol.get('evaluator') != protocol.get('evaluator')
                    or before.get('rules') != previous):
                raise ValueError('every cluster must finish its full BO and admission budget')
            records = [json.loads((folder / ('trial_%05d.json' % number)).read_text()) for number in range(budget)]
            if bo.get('all_trial_records_sha256') != digest(records):
                raise ValueError('completed BO candidates have changed')
            for number, candidate in enumerate(records):
                decision = json.loads((folder / 'admission' / ('candidate_%05d.json' % number)).read_text())
                if (decision.get('trial_number') != number or decision.get('candidate') != candidate.get('rule')
                        or decision.get('selected_before') != len(previous)
                        or type(decision.get('decision', {}).get('accepted')) is not bool):
                    raise ValueError('admission trace differs from the complete BO candidate sequence')
                if decision['decision']['accepted']:
                    if candidate.get('rule') is None:
                        raise ValueError('empty candidate cannot be accepted')
                    previous = previous + [candidate['rule']]
                expected_key = digest({'binding_id': digest(protocol['evaluator']), 'ordered_rules': previous})
                if (decision.get('selected_after') != len(previous)
                        or decision.get('selected_program_key') != expected_key):
                    raise ValueError('admission trace changed its ordered selected program')
            if admission.get('rules') != previous:
                raise ValueError('cluster final Sigma differs from its sequential decisions')
        if previous != selected['rules']:
            raise ValueError('final Sigma differs from the last completed admission trace')
        return {'mode': 'completed_fresh_selection', 'directory': str(directory),
            'selection_complete_claim': True, 'selected_artifact_sha256': digest(selected),
            'protocol_id': digest(protocol), 'complete_BO_trials': complete['complete_BO_trials'],
            'labels': list(protocol['evaluator']['labels']),
            'learning_context_id': protocol['evaluator']['context_id']}

    @property
    def docs(self):
        return self.runner.docs

    @property
    def binding(self):
        return copy.deepcopy(self._binding)

    def active_labels(self, alias):
        active = np.zeros(len(self.labels), bool)
        index = {label: i for i, label in enumerate(self.labels)}
        for rule in self.rules:
            for atom in rule.body:
                if isinstance(atom, ML) and atom.dynamic and atom.alias == alias:
                    active[index[atom.label]] = True
        return _readonly(active)

    def _sources(self, seed, phase, supplied):
        keys = _root_keys(seed)
        if supplied is None:
            if phase == 'feedback' and keys:
                raise ValueError('every feedback weak root requires an explicit correction source')
            supplied = {key: {'origin': 'validated_Gamma' if phase == 'labeling'
                             else 'counterfactual_assumption'} for key in keys}
        if not isinstance(supplied, Mapping) or set(supplied) != keys:
            raise ValueError('root_sources must cover exactly the explicitly supplied seed facts')
        result = plain(supplied)
        for key, source in result.items():
            if not isinstance(source, dict) or not isinstance(source.get('origin'), str):
                raise ValueError('every root source requires an explicit origin')
            if phase == 'feedback':
                if (source['origin'] not in ('feedback_weak_root', 'accepted_weak_root')
                        or not isinstance(source.get('correction_id'), str) or not source['correction_id']):
                    raise ValueError('feedback roots must name their weak correction, never validated truth')
            elif phase == 'labeling' and source['origin'] not in (
                    'validated_Gamma', 'ground_truth', 'human', 'llm_assumption'):
                raise ValueError('labeling roots must explicitly retain validated or revocable LLM sources')
        digest(result)
        return result

    def chase(self, seed, *, phase='labeling', root_sources=None):
        if phase not in PHASES:
            raise ValueError('explicit supported logical phase required')
        if not isinstance(seed, (GammaSeed, RefGamma)) or seed.shape != self.shape:
            raise ValueError('explicit full-axis GammaSeed or RefGamma required')
        if phase == 'feedback' and (getattr(seed, 'subset', ()) or getattr(seed, 'strict_subset', ())):
            raise ValueError('section 6.2 weak roots are instantiated signed label corrections')
        if (self.snapshot_id != self._snapshot_id or self.model_snapshot_id != self._snapshot_id
                or digest([r.to_dict() for r in self.rules]) != self.program_id
                or self.runner.context_id != self.context_id or self.context_id != self._binding['context_id']
                or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self._source_sha):
            raise RuntimeError('frozen session rules, model identity or adapter source changed')
        self.runner.verify_code()
        scope = self.runner.application_scope
        if ((scope.receipt() if scope is not None else None)
                != self._binding.get('application_scope')):
            raise RuntimeError('frozen application scope changed; create a new session')
        sources = self._sources(seed, phase, root_sources)
        seed_before = digest(seed_identity(seed))
        key = digest({'binding': self._binding, 'seed': seed_before, 'phase': phase, 'root_sources': sources})
        hit = key in self._cache
        if hit:
            self.cache_hits += 1
            stored = self._cache.pop(key)
            self._cache[key] = stored
        else:
            result = self.runner(self.rules, gamma=seed)
            self.actual_chase_calls += 1
            if result.status not in ('fixpoint', 'conflict'):
                raise RuntimeError('only complete fixpoint or explicit conflict can enter a session')
            proof = result.proof_dag.to_dict()
            for fact, source in sources.items():
                if fact not in proof:
                    raise RuntimeError('Chase dropped an explicit seed proof: ' + fact)
                if proof[fact]['origin'] != 'validated_Gamma' and not (
                        fact.startswith('subset:') and fact.split(':')[1] == fact.split(':')[2]
                        and proof[fact]['origin'] in ('identity', 'reflexivity')):
                    raise RuntimeError('Chase seed proof lost its root identity: ' + fact)
                proof[fact]['origin'] = source['origin']
                proof[fact]['root_source'] = source
                proof[fact]['logical_phase'] = phase
            result.proof_dag = ProofDAG(proof)
            result.typed_session_audit = {'schema': SCHEMA, 'phase': phase, 'theta': self.snapshot_id,
                'Sigma': self.program_id, 'context_id': self.context_id, 'seed_sha256': seed_before,
                'n_rules': len(self.rules), 'labels': list(self.labels),
                'model_probability_tables': {alias: array_identity(table)
                                            for alias, table in self.runner.probabilities.items()},
                'dynamic_threshold_specs': {alias: sorted({(self.labels.index(atom.label), atom.eta)
                    for rule in self.rules for atom in rule.body
                    if isinstance(atom, ML) and atom.dynamic and atom.alias == alias})
                    for alias in self.runner.probabilities},
                'root_sources_sha256': digest(sources), 'explicit_weak_roots': len(sources) if phase == 'feedback' else 0,
                'validated_Gamma_unioned': False, 'selection': self.selection}
            if scope is not None:
                result.typed_session_audit['application_scope'] = scope.receipt()
            result.typed_session_audit['complete_state_sha256'] = complete_state_identity(result)['sha256']
            stored = _clone(result)
            if self.cache_entries:
                self._cache[key] = stored
                while len(self._cache) > self.cache_entries:
                    self._cache.popitem(last=False)
        if digest(seed_identity(seed)) != seed_before:
            raise RuntimeError('Chase mutated its explicit immutable seed')
        output = _clone(stored)
        output.typed_session_audit['exact_result_cache_hit'] = hit
        self.calls.append({**copy.deepcopy(output.typed_session_audit), 'status': output.status,
                           'rounds': output.n_rounds})
        return output

    def feedback_chase(self, seed, *, root_sources):
        return self.chase(seed, phase='feedback', root_sources=root_sources)

    def receipt(self):
        return {'binding': self.binding, 'actual_chase_calls': self.actual_chase_calls,
            'exact_result_cache_hits': self.cache_hits, 'calls': copy.deepcopy(self.calls),
            'runner': self.runner.audit()}

    def close(self):
        self._cache.clear()
        self.runner.close()
