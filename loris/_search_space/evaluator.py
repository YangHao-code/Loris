"""Connect every actually used rule variable before any cache access.

The new BO study produces its own proposals and observations. Legacy SQLite
sources are immutable function-result caches, never proposal/history inputs.
V3 and frozen V4 function results retain their entire original payloads in a
new provenance envelope. Only authenticated provenance fields are normalized
for comparison; original facts, predictions, proof identities, rounds, metrics
and unknown fields must match exactly. No BO history is restored here.
"""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

import numpy as np

from loris._engine.rdl_reference import Rule
from loris._engine.rlearner import (TwoFilterEvaluator, canonical, digest,
                                    gamma_identity, rule_record)
from .configuration import guard_policy_binding, validate_rule_guard


SCHEMA = 'paper36_connected_exact_score_cache_v1'
V4_SCHEMA = 'paper36_guarded_exact_score_cache_v1'


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def frozen_v4_binding(base_binding):
    """Historical scores belong to the original immutable interpreter.

    This release changes predicate semantics and source identities. Ordered
    complete programs remain reusable through the public workflow, but old
    score caches must not be authenticated as scores from this implementation.
    """
    raise ValueError('historical score-cache import is unavailable in this release; '
                     'reuse a verified complete SELECTED program or evaluate afresh')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _stat(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _no_sidecars(path):
    # immutable=1 may ignore a WAL. Never silently omit a committed transaction.
    if any(Path(str(path) + suffix).exists() for suffix in ('-wal', '-shm', '-journal')):
        raise ValueError('legacy score source must be an authenticated standalone SQLite snapshot')


def _validate_payload(value, expected_key, binding):
    required = {'program_key', 'status', 'n_rounds', 'conflicts', 'comparison_conflicts',
                'positive_metrics', 'negative_metrics', 'full_state'}
    if not isinstance(value, dict) or not required <= set(value) or value['program_key'] != expected_key:
        raise ValueError('legacy score payload lacks its exact ordered program identity or complete receipt')
    if value['status'] not in ('fixpoint', 'conflict') or type(value['n_rounds']) is not int or value['n_rounds'] < 0:
        raise ValueError('legacy score payload is not a completed original Chase result')
    state = value['full_state']
    if not isinstance(state, dict) or not {'positive_facts', 'negative_facts', 'predictions',
                                          'subset_facts', 'strict_subset_facts', 'proof_dag'} <= set(state):
        raise ValueError('legacy score payload omits original full-state identities')
    def valid_sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    for name in ('positive_facts', 'negative_facts', 'predictions'):
        record = state[name]
        if (not isinstance(record, dict) or record.get('shape') != binding['gamma']['positive']['shape']
                or not isinstance(record.get('dtype'), str) or not record['dtype']
                or not valid_sha(record.get('sha256'))):
            raise ValueError('legacy full-state array identity differs from the complete bound axes')
        try:
            dtype = np.dtype(record.get('dtype'))
        except (TypeError, ValueError):
            raise ValueError('legacy array identity has an invalid original dtype')
        if (name != 'predictions' and dtype != np.dtype(bool)) or dtype.kind not in 'biuf':
            raise ValueError('legacy array identity has an unsupported original dtype')
    for name in ('subset_facts', 'strict_subset_facts', 'proof_dag'):
        if (not isinstance(state[name], dict) or type(state[name].get('count')) is not int
                or state[name]['count'] < 0 or not valid_sha(state[name].get('sha256'))):
            raise ValueError('legacy comparison or proof identity is incomplete')
    if not isinstance(value['conflicts'], list) or not isinstance(value['comparison_conflicts'], list):
        raise ValueError('legacy conflict receipts are incomplete')
    if value['status'] == 'fixpoint':
        if value['conflicts'] or value['comparison_conflicts']:
            raise ValueError('legacy fixpoint receipt contains a conflict')
        for name in ('positive_metrics', 'negative_metrics'):
            metrics = value[name]
            if (not isinstance(metrics, dict) or not {'macro_f1', 'micro_f1', 'per_label_f1', 'tp', 'fp', 'fn', 'rows', 'labels'} <= set(metrics)
                    or metrics['rows'] != binding['receivers']['shape'][0]
                    or metrics['labels'] != len(binding['labels'])):
                raise ValueError('legacy completed score metrics omit the complete receiver/label axes')
            for field in ('per_label_f1', 'tp', 'fp', 'fn'):
                if not isinstance(metrics[field], list) or len(metrics[field]) != len(binding['labels']):
                    raise ValueError('legacy metric vectors omit the complete label axes')
            for field in ('tp', 'fp', 'fn'):
                if any(type(v) is not int or not 0 <= v <= metrics['rows'] for v in metrics[field]):
                    raise ValueError('legacy metric counts are invalid')
            tp, fp, fn = (np.asarray(metrics[field], np.int64) for field in ('tp', 'fp', 'fn'))
            if np.any(tp + fp > metrics['rows']) or np.any(tp + fn > metrics['rows']):
                raise ValueError('legacy metric counts exceed the full receiver axis')
            denominator = 2 * tp + fp + fn
            f1 = np.divide(2. * tp, denominator, out=np.zeros(len(tp)), where=denominator != 0)
            total = int(denominator.sum())
            if (metrics['per_label_f1'] != f1.tolist() or metrics['macro_f1'] != float(f1.mean())
                    or metrics['micro_f1'] != (float(2 * tp.sum() / total) if total else 0.)):
                raise ValueError('legacy metric receipts disagree with their original exact counts')
    else:
        if not (value['conflicts'] or value['comparison_conflicts']):
            raise ValueError('legacy conflict status lacks conflict evidence')
        if value['positive_metrics'] is not None or value['negative_metrics'] is not None:
            raise ValueError('legacy conflict must retain the original absent score metrics')
    canonical(value)  # Reject NaN and non-JSON values; preserve every unknown provenance field.


class ReadOnlyLegacyScoreSource:
    """A closed SHA-bound source with an exact base or reviewed V4 binding."""
    def __init__(self, record, expected_binding, expected_v4_binding=None):
        if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
            raise ValueError('legacy source requires an explicit immutable path and complete file SHA')
        supplied_path = Path(record['path'])
        path = supplied_path.resolve(strict=True)
        if not path.is_file() or supplied_path.is_symlink():
            raise ValueError('legacy score source must be a regular immutable snapshot')
        self.record = {'path': str(path), 'sha256': record['sha256']}
        self.base_binding = copy.deepcopy(expected_binding)
        self.binding = None
        self.kind = None
        self._signature = None
        self.db = None
        self.check()
        uri = 'file:' + quote(str(path), safe='/') + '?mode=ro&immutable=1'
        self.db = sqlite3.connect(uri, uri=True)
        try:
            row = self.db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
            actual = json.loads(row[0]) if row is not None else None
            if actual == self.base_binding:
                self.kind = 'original_v3'
            elif expected_v4_binding is not None and actual == expected_v4_binding:
                self.kind = 'frozen_guarded_v4'
            else:
                raise ValueError('legacy source differs in data, Gamma, truth, model, engine or scoring binding')
            self.binding = actual
            if self.db.execute('PRAGMA quick_check').fetchone() != ('ok',):
                raise ValueError('legacy source SQLite integrity check failed')
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        path = Path(self.record['path'])
        _no_sidecars(path)
        before = _stat(path)
        if before != self._signature:
            if file_sha(path) != self.record['sha256'] or _stat(path) != before:
                raise ValueError('authenticated legacy score source changed')
            self._signature = before

    def lookup(self, key):
        self.check()
        row = self.db.execute('SELECT value FROM programs WHERE key=?', (key,)).fetchone()
        value = json.loads(row[0]) if row is not None else None
        if value is not None:
            _validate_payload(value, key, self.binding)
        self.check()
        return value

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None


class GuardedTwoFilterEvaluator(TwoFilterEvaluator):
    """Original two scores, with an explicit hard grammar guard before lookup."""
    def __init__(self, runner, *, labels, receiver_indices, truth, gamma,
                 context_id, line7_mode, head_info, cache_path=None,
                 runtime_binding=None, legacy_score_sources=(), fit_precision_floor=None):
        records = list(legacy_score_sources)
        if any(not isinstance(r, dict) or set(r) != {'path', 'sha256'} for r in records):
            raise ValueError('legacy source requires an explicit immutable path and complete file SHA')
        if len({str(Path(r['path']).resolve()) for r in records}) != len(records):
            raise ValueError('legacy score source paths must be distinct')
        if cache_path is not None:
            writable = Path(cache_path)
            if any(writable.resolve() == Path(r['path']).resolve() or
                   (writable.exists() and Path(r['path']).exists() and writable.samefile(r['path']))
                   for r in records):
                raise ValueError('new writable score cache must not alias a legacy immutable source')
        original_runtime = copy.deepcopy(runtime_binding if runtime_binding is not None else getattr(runner, 'binding', None))
        guarded_runtime = {'schema': SCHEMA,
            'original_runtime_binding': original_runtime,
            'guard': guard_policy_binding(),
            'guarded_evaluator_source_sha256': file_sha(__file__),
            'score_function': 'unchanged_original_TwoFilterEvaluator_for_guard_admissible_programs',
            'legacy_cache_policy': 'exact_current_proposal_ordered_program_lookup_only_no_BO_history_import'}
        self.legacy_sources, self.legacy_cache_hits, self.legacy_reuse_receipts = [], 0, []
        self._origin_sources, self._provenance_files = {}, {}
        super().__init__(runner, labels=labels, receiver_indices=receiver_indices,
            truth=truth, gamma=gamma, context_id=context_id, line7_mode=line7_mode,
            head_info=head_info, cache_path=cache_path, runtime_binding=guarded_runtime,
            fit_precision_floor=fit_precision_floor)
        self.original_evaluator_binding = copy.deepcopy(self.binding)
        self.original_evaluator_binding['runtime'] = original_runtime
        try:
            # New discovery computes scores under this release's interpreter.
            # Loading complete learned rules is a separate public workflow.
            self.v4_evaluator_binding = (
                frozen_v4_binding(self.original_evaluator_binding) if records else None)
            for record in sorted(records, key=lambda r: str(Path(r['path']).resolve())):
                self.legacy_sources.append(ReadOnlyLegacyScoreSource(record,
                    self.original_evaluator_binding, self.v4_evaluator_binding))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _rules(rules):
        rules = tuple(Rule.from_dict(rule) if isinstance(rule, dict) else rule for rule in rules)
        for rule in rules:
            validate_rule_guard(rule)
        return rules

    def _has_current(self, key):
        return key in self.memory or (self.db is not None and
            self.db.execute('SELECT 1 FROM programs WHERE key=?', (key,)).fetchone() is not None)

    def _authenticate_provenance_file(self, record):
        if (not isinstance(record, dict) or set(record) != {'path', 'sha256'}
                or not _valid_sha(record['sha256'])):
            raise ValueError('legacy provenance requires its complete source file SHA')
        supplied = Path(record['path'])
        path = supplied.resolve(strict=True)
        if not path.is_file() or supplied.is_symlink():
            raise ValueError('legacy provenance file must be a regular immutable file')
        signature = _stat(path)
        key = (str(path), record['sha256'])
        if self._provenance_files.get(key) != signature:
            if file_sha(path) != record['sha256'] or _stat(path) != signature:
                raise ValueError('legacy provenance file changed')
            self._provenance_files[key] = signature

    def _base_origin_source(self, record):
        if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
            raise ValueError('V4 origin source requires an exact immutable cache record')
        key = (str(Path(record['path']).resolve()), record['sha256'])
        if key not in self._origin_sources:
            self._origin_sources[key] = ReadOnlyLegacyScoreSource(record, self.original_evaluator_binding)
        return self._origin_sources[key]

    def _canonical_base_payload(self, source, value, base_key):
        """Authenticate wrappers before removing them from exact-result comparison."""
        if 'connected_cache_origin' in value:
            raise ValueError('recursive connected cache import needs a separately reviewed protocol')
        restored = copy.deepcopy(value)
        if 'guarded_cache_origin' in restored:
            if source.kind != 'frozen_guarded_v4':
                raise ValueError('V4 provenance cannot appear under an unguarded source binding')
            origin = restored.pop('guarded_cache_origin')
            restored['program_key'] = base_key
            records = origin.get('source_caches') if isinstance(origin, dict) else None
            if not isinstance(records, list) or not records:
                raise ValueError('V4 reuse provenance requires all original source cache references')
            expected = {'schema': V4_SCHEMA,
                'reuse_kind': 'exact_result_for_independently_proposed_current_program',
                'old_evaluator_binding_id': digest(self.original_evaluator_binding),
                'new_evaluator_binding_id': digest(source.binding),
                'old_program_key': base_key, 'original_payload_sha256': digest(restored),
                'source_caches': records, 'old_facts_metrics_rounds_and_full_proof_identity_preserved': True,
                'new_Chase_execution_performed': False, 'old_candidates_or_BO_observations_imported': 0}
            if (origin != expected or origin['new_Chase_execution_performed'] is not False
                    or origin['old_facts_metrics_rounds_and_full_proof_identity_preserved'] is not True
                    or type(origin['old_candidates_or_BO_observations_imported']) is not int):
                raise ValueError('V4 provenance no longer authenticates its exact original complete payload')
            paths = [str(Path(r['path']).resolve()) for r in records]
            if paths != sorted(set(paths)):
                raise ValueError('V4 origin cache list is not complete deterministic distinct provenance')
            for record in records:
                prior = self._base_origin_source(record).lookup(base_key)
                if prior is None or canonical(prior) != canonical(restored):
                    raise ValueError('V4 origin cache does not contain its claimed exact original payload')
        else:
            restored['program_key'] = base_key
        _validate_payload(restored, base_key, self.original_evaluator_binding)
        result = copy.deepcopy(restored)
        if 'migration_origin' in result:
            origin = result.pop('migration_origin')
            required = {'schema', 'new_binding_id', 'new_execution_performed',
                'original_n_rounds_and_proof_identity_retained', 'source_binding_id',
                'source_program_key', 'source_evaluation_sha256', 'source_full_state_sha256',
                'equivalence_receipt', 'origin_primary', 'origin_snapshot',
                'original_artifact_manifest_sha256', 'original_queue_manifest_sha256'}
            if (not isinstance(origin, dict) or set(origin) != required
                    or origin['schema'] != 'paper36_imported_evaluation_origin_v1'
                    or origin['new_binding_id'] != digest(self.original_evaluator_binding)
                    or origin['new_execution_performed'] is not False
                    or origin['original_n_rounds_and_proof_identity_retained'] is not True
                    or origin['source_full_state_sha256'] != digest(result['full_state'])):
                raise ValueError('old migration provenance does not preserve the complete result')
            for field in ('source_binding_id', 'source_program_key', 'source_evaluation_sha256',
                          'original_artifact_manifest_sha256', 'original_queue_manifest_sha256'):
                if not _valid_sha(origin[field]):
                    raise ValueError('old migration provenance identity is incomplete')
            if any(not isinstance(origin[field], str) or not origin[field] for field in ('origin_primary', 'origin_snapshot')):
                raise ValueError('old migration provenance location is incomplete')
            previous = dict(result, program_key=origin['source_program_key'])
            if digest(previous) != origin['source_evaluation_sha256']:
                raise ValueError('old migration provenance changed an original result field')
            self._authenticate_provenance_file(origin['equivalence_receipt'])
        return result, restored

    def _lookup_legacy(self, rules, new_key):
        ordered = [rule_record(rule) for rule in rules]
        base_key = digest({'binding_id': digest(self.original_evaluator_binding), 'ordered_rules': ordered})
        hits = []
        for source in self.legacy_sources:
            old_key = digest({'binding_id': digest(source.binding), 'ordered_rules': ordered})
            value = source.lookup(old_key)
            if value is not None:
                raw, restored = self._canonical_base_payload(source, value, base_key)
                hits.append((source, value, raw, restored))
        if not hits:
            return None
        original = hits[0][2]
        if any(canonical(raw) != canonical(original) for _, _, raw, _ in hits[1:]):
            raise ValueError('legacy sources disagree on complete facts/predictions/proofs/rounds/score or unknown fields')
        result = copy.deepcopy(original)
        result['program_key'] = new_key
        result['connected_cache_origin'] = {'schema': SCHEMA,
            'reuse_kind': 'exact_result_for_independently_proposed_current_program',
            'base_evaluator_binding_id': digest(self.original_evaluator_binding),
            'new_evaluator_binding_id': self.binding_id,
            'base_program_key': base_key, 'canonical_base_payload_sha256': digest(original),
            'source_payloads': [{'source_cache': copy.deepcopy(source.record),
                'source_kind': source.kind, 'source_evaluator_binding': copy.deepcopy(source.binding),
                'source_program_key': value['program_key'], 'source_payload_sha256': digest(value),
                'original_complete_payload': copy.deepcopy(value),
                'restored_base_payload_sha256': digest(restored)} for source, value, _, restored in hits],
            'only_authenticated_provenance_wrappers_normalized': True,
            'old_facts_metrics_rounds_and_full_proof_identity_preserved': True,
            'new_Chase_execution_performed': False, 'old_candidates_or_BO_observations_imported': 0}
        self.legacy_reuse_receipts.append(copy.deepcopy(result['connected_cache_origin']))
        self.legacy_cache_hits += 1
        return result

    def evaluate(self, rules):
        rules = self._rules(rules)  # Before either the local or legacy cache can be queried.
        if gamma_identity(self.gamma) != self.gamma_signature:
            raise ValueError('fixed discovery Gamma changed')
        key = self.program_key(rules)
        if self.legacy_sources and not self._has_current(key):
            value = self._lookup_legacy(rules, key)
            if value is not None:
                if self.db is not None:
                    self.db.execute('INSERT INTO programs VALUES(?,?)', (key, canonical(value)))
                    self.db.commit()
                self.memory[key] = copy.deepcopy(value)
        return super().evaluate(rules)

    def standalone(self, rule):
        rule, = self._rules((rule,))
        return super().standalone(rule)

    def admit(self, selected, rule, epsilon=1e-12):
        # Validate the proposal too before evaluating even the selected baseline.
        selected = self._rules(selected)
        rule, = self._rules((rule,))
        return super().admit(selected, rule, epsilon)

    def close(self):
        for source in getattr(self, 'legacy_sources', ()):
            source.close()
        for source in getattr(self, '_origin_sources', {}).values():
            source.close()
        super().close()


ConnectedTwoFilterEvaluator = GuardedTwoFilterEvaluator
