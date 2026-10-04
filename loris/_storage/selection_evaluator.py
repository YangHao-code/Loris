"""Full-fact rule scoring in an explicit namespace without proof expansion.

The original two F1 filters and connected-rule guard are retained. These
results cannot certify feedback provenance; selected programs require a full
provenance replay before entering a proof-consuming session.
"""
import copy
import inspect
import json
from pathlib import Path
import sqlite3
from types import FunctionType
from urllib.parse import quote

import numpy as np

from loris._engine.discovery_io import configure_sqlite

from loris._engine.rlearner import TwoFilterEvaluator, array_identity, canonical, digest, gamma_identity
from loris._search_space.evaluator import GuardedTwoFilterEvaluator
from loris._search_space.configuration import validate_rule_guard
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._engine.relation_rows import RELATION_SCHEMA
from loris._learning import noop_admission
from loris._learning.artifacts import file_record, write_json
from loris._engine.hashing import cached_file_record
from .selection_execution import SelectionChaseRunner, selection_execution_binding


SCHEMA = 'paper36_selection_facts_score_v1'
FACT_FIELDS = ('positive_facts', 'negative_facts', 'predictions', 'subset_facts', 'strict_subset_facts')
# n_rounds is an execution statistic (naive rounds of one schedule), not a
# semantic quantity: a fixpoint warm start reaches the same least fixpoint in
# fewer rounds. It stays in every payload but not in semantic comparisons.
SEMANTIC_FIELDS = ('status', 'conflicts', 'comparison_conflicts',
                   'positive_metrics', 'negative_metrics')
_PAYLOAD_FIELDS = set(SEMANTIC_FIELDS) | {'n_rounds', 'program_key', 'full_state', 'scoring_schema',
                                         'proof_materialized', 'selection_only', 'gamma_protection'}


def semantic_result_projection(value):
    """Compare full-fact semantics across namespaces, never import a cache.

    Deliberately excludes the program key and proof identity. Callers must
    authenticate each ordered program and its input binding independently.
    """
    if not isinstance(value, dict) or not set(SEMANTIC_FIELDS) <= set(value):
        raise ValueError('complete scoring semantics required')
    state = value.get('full_state')
    if not isinstance(state, dict) or not set(FACT_FIELDS) <= set(state):
        raise ValueError('all five complete fact identities required')
    return {**{name: copy.deepcopy(value[name]) for name in SEMANTIC_FIELDS},
            'full_state': {name: copy.deepcopy(state[name]) for name in FACT_FIELDS}}


def execution_binding(runner=None):
    if runner is not None:
        _runner(runner)
    policy = selection_execution_binding() if runner is None else runner.selection_execution_binding
    records = {}
    def collect(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256'}:
                if file_record(value['path']) != value:
                    raise ValueError('selection execution source changed')
                records[value['path']] = value
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
    collect(policy)
    for obj in (SelectionChaseRunner, IndexedChaseRunner, TwoFilterEvaluator,
                GuardedTwoFilterEvaluator, validate_rule_guard, noop_admission.StaticBodyNoOpGate):
        record = file_record(inspect.getfile(obj)); records[record['path']] = record
    record = file_record(__file__); records[record['path']] = record
    return {'schema': SCHEMA, 'sources': [records[path] for path in sorted(records)],
            'runner_execution': policy, 'proof_materialized': False, 'selection_only': True,
            'full_fact_closure_conflicts_and_both_F1_filters_preserved': True,
            'legacy_full_proof_score_cache_imported': False,
            'static_noop': 'complete_static_body_superset_empty_only',
            'downstream_requirement': 'replay selected program with provenance before feedback'}


def _runner(runner):
    if type(runner) is not SelectionChaseRunner:
        raise TypeError('exact SelectionChaseRunner required')
    runner.verify_code()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _validate_payload(value, key, binding):
    if (not isinstance(value, dict) or set(value) != _PAYLOAD_FIELDS
            or value['program_key'] != key or value['scoring_schema'] != SCHEMA
            or value['proof_materialized'] is not False or value['selection_only'] is not True):
        raise ValueError('selection cache must contain an explicit complete facts-only payload')
    semantic_result_projection(value)
    protection = value['gamma_protection']
    if protection is not None and (
            not isinstance(protection, dict) or set(protection) != {'blocked', 'agreements'}
            or type(protection['blocked']) is not int or protection['blocked'] < 0
            or (protection['agreements'] is not None and
                (type(protection['agreements']) is not int or protection['agreements'] < 0))):
        raise ValueError('selection Gamma-protection record is malformed')
    state = value['full_state']; shape = binding['gamma']['positive']['shape']
    if set(state) != set(FACT_FIELDS):
        raise ValueError('selection payload cannot contain a fabricated or full proof identity')
    for name in FACT_FIELDS[:3]:
        item = state[name]
        if (not isinstance(item, dict) or set(item) != {'shape', 'dtype', 'sha256'}
                or item['shape'] != shape or not _sha(item['sha256'])
                or item['dtype'] != ('float32' if name == 'predictions' else 'bool')):
            raise ValueError('selection full-corpus array identity differs')
    for name in FACT_FIELDS[3:]:
        item = state[name]
        if (not isinstance(item, dict) or item.get('schema') != RELATION_SCHEMA
                or item.get('n') != shape[0] or type(item.get('count')) is not int
                or not 0 <= item['count'] <= shape[0] ** 2 or not _sha(item.get('sha256'))):
            raise ValueError('selection complete relation identity differs')
    if (value['status'] not in ('fixpoint', 'conflict') or type(value['n_rounds']) is not int
            or value['n_rounds'] < 0 or not isinstance(value['conflicts'], list)
            or not isinstance(value['comparison_conflicts'], list)):
        raise ValueError('selection requires complete termination and conflict records')
    if value['status'] == 'conflict':
        if (not value['conflicts'] and not value['comparison_conflicts']) or any(
                value[name] is not None for name in ('positive_metrics', 'negative_metrics')):
            raise ValueError('invalid conflicting selection score')
        return
    if value['conflicts'] or value['comparison_conflicts']:
        raise ValueError('selection fixpoint contains conflicts')
    rows, labels = binding['receivers']['shape'][0], len(binding['labels'])
    for name in ('positive_metrics', 'negative_metrics'):
        metrics = value[name]
        fields = {'macro_f1', 'micro_f1', 'per_label_f1', 'tp', 'fp', 'fn', 'rows', 'labels'}
        if (not isinstance(metrics, dict) or set(metrics) != fields
                or metrics['rows'] != rows or metrics['labels'] != labels):
            raise ValueError('selection metric axes differ')
        for field in ('tp', 'fp', 'fn'):
            if (not isinstance(metrics[field], list) or len(metrics[field]) != labels
                    or any(type(x) is not int or not 0 <= x <= rows for x in metrics[field])):
                raise ValueError('invalid selection confusion counts')
        tp, fp, fn = (np.asarray(metrics[field], np.int64) for field in ('tp', 'fp', 'fn'))
        denominator = 2 * tp + fp + fn
        f1 = np.divide(2. * tp, denominator, out=np.zeros(labels), where=denominator != 0)
        total = int(denominator.sum())
        if (metrics['per_label_f1'] != f1.tolist() or metrics['macro_f1'] != float(f1.mean())
                or metrics['micro_f1'] != (float(2 * tp.sum() / total) if total else 0.)
                or np.any(tp + fp > rows) or np.any(tp + fn > rows)):
            raise ValueError('selection metrics differ from complete confusion counts')


class SelectionEvaluator(GuardedTwoFilterEvaluator):
    """Original guarded standalone/admit functions, new authenticated cache."""
    def __init__(self, runner, *, legacy_score_sources=(), cache_path=None, runtime_binding=None, **kwargs):
        _runner(runner)
        if tuple(legacy_score_sources):
            raise ValueError('legacy full-proof score caches are forbidden for selection')
        if runtime_binding is not None and runtime_binding != runner.binding:
            raise ValueError('selection runtime binding differs from actual runner')
        super().__init__(runner, cache_path=None, runtime_binding=runner.binding,
                         legacy_score_sources=(), **kwargs)
        self.execution_binding = execution_binding(runner)
        self.binding.update(schema=SCHEMA, selection_scoring=self.execution_binding)
        self.binding_id = digest(self.binding)
        self._fixed_binding = copy.deepcopy(self.binding)
        self._fixed_runner_binding = copy.deepcopy(runner.binding)
        self._fixed_head_info = self.head_info
        if cache_path is not None:
            path = Path(cache_path); path.parent.mkdir(parents=True, exist_ok=True)
            encoded = canonical(self.binding)
            if path.exists():
                probe = sqlite3.connect('file:' + quote(str(path.resolve()), safe='/') + '?mode=ro', uri=True)
                try:
                    old = probe.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
                    if old is None or old[0] != encoded:
                        raise ValueError('selection cache binding differs; full-proof caches cannot be reused')
                finally:
                    probe.close()
            self.db = sqlite3.connect(str(path), timeout=60)
            try:
                configure_sqlite(self.db)
                self.db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
                self.db.execute('CREATE TABLE IF NOT EXISTS programs(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
                old = self.db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
                if old is not None and old[0] != encoded:
                    raise ValueError('selection cache binding differs; full-proof caches cannot be reused')
                self.db.execute("INSERT OR IGNORE INTO metadata VALUES('binding',?)", (encoded,))
                self.db.commit()
            except BaseException:
                self.close(); raise

    def _verify(self):
        _runner(self.runner)
        if (self.binding != self._fixed_binding or self.binding_id != digest(self.binding)
                or self.execution_binding != self.binding['selection_scoring']
                or self.runner.binding != self._fixed_runner_binding
                or list(self.labels) != self.binding['labels']
                or self.line7_mode != self.binding['line7_mode']
                or self.head_info is not self._fixed_head_info
                or gamma_identity(self.gamma) != self.binding['gamma']
                or array_identity(self.truth) != self.binding['truth']
                or array_identity(self.indices) != self.binding['receivers']):
            raise ValueError('fixed selection data or evaluator binding changed')
        for record in self.execution_binding['sources']:
            if cached_file_record(record['path']) != record:
                raise ValueError('selection execution source changed')

    def evaluate(self, rules):
        self._verify(); rules = self._rules(rules); key = self.program_key(rules)
        value = self.memory.get(key)
        if value is None and self.db is not None:
            row = self.db.execute('SELECT value FROM programs WHERE key=?', (key,)).fetchone()
            value = json.loads(row[0]) if row else None
        if value is not None:
            _validate_payload(value, key, self.binding)
            self.cache_hits += 1
            return copy.deepcopy(value)
        # The original scorer builds the entire factual payload and updates
        # actual-call counters. Persist only the finalized selection envelope,
        # so a crash can never leave an unmarked payload in this namespace.
        database, self.db = self.db, None
        try:
            value = TwoFilterEvaluator.evaluate(self, rules)
        finally:
            self.db = database
            self.memory.pop(key, None)
        value.update(scoring_schema=SCHEMA, proof_materialized=False, selection_only=True)
        _validate_payload(value, key, self.binding)
        self._verify()
        if self.db is not None:
            self.db.execute('INSERT INTO programs VALUES(?,?)', (key, canonical(value)))
            self.db.commit()
        self.memory[key] = copy.deepcopy(value)
        return copy.deepcopy(value)


_initializer = noop_admission.StaticBodyNoOpGate.__init__
_GATE_METHODS = {name: getattr(noop_admission.StaticBodyNoOpGate, name)
                 for name in ('__init__', 'certify', 'verify')}
if _initializer.__globals__.get('IndexedChaseRunner') is not IndexedChaseRunner:
    raise ValueError('unexpected original static gate constructor')
_namespace = dict(_initializer.__globals__, IndexedChaseRunner=SelectionChaseRunner)
_gate_init = FunctionType(_initializer.__code__, _namespace, _initializer.__name__,
                          _initializer.__defaults__, _initializer.__closure__)


class SelectionStaticNoOpGate(noop_admission.StaticBodyNoOpGate):
    def __init__(self, runner):
        _runner(runner); _gate_init(self, runner)

    def verify(self):
        if any(getattr(noop_admission.StaticBodyNoOpGate, name) is not method
               for name, method in _GATE_METHODS.items()):
            raise ValueError('original complete static-body gate changed')
        _runner(self.runner); super().verify()


class SelectionAdmissionReuse:
    """Exact empty-static-body reuse with an honest facts-only certificate."""
    def __init__(self, evaluator, runner, receipt_dir):
        if type(evaluator) is not SelectionEvaluator or evaluator.runner is not runner:
            raise TypeError('matching SelectionEvaluator and runner required')
        self.evaluator, self.gate = evaluator, SelectionStaticNoOpGate(runner)
        self.directory = Path(receipt_dir).resolve()
        self.execution_binding = dict(execution_binding(runner), durable_receipt_directory=str(self.directory))
        self.reused_admissions = 0
        self.last_execution_receipt = None

    def __getattr__(self, name):
        return getattr(self.evaluator, name)

    def admit(self, selected, rule, epsilon=1e-12):
        e = self.evaluator; e._verify()
        selected = e._rules(selected); rule, = e._rules((rule,))
        self.last_execution_receipt = None
        self.gate.verify()
        certificate = self.gate.certify(rule)
        key = e.program_key(selected + (rule,))
        cached = key in e.memory or (e.db is not None and
                 e.db.execute('SELECT 1 FROM programs WHERE key=?', (key,)).fetchone() is not None)
        if certificate is None or cached:
            return e.admit(selected, rule, epsilon)
        baseline = e.evaluate(selected)
        if baseline['status'] != 'fixpoint':
            raise ValueError('selected program must be consistent')
        _validate_payload(baseline, e.program_key(selected), e.binding)
        proposed = copy.deepcopy(baseline); proposed['program_key'] = key
        _validate_payload(proposed, key, e.binding)
        delta = proposed['positive_metrics']['macro_f1'] - baseline['positive_metrics']['macro_f1']
        accepted = delta > epsilon
        result = dict(line=12, accepted=bool(accepted), criterion='positive_macro_f1',
                      delta_macro_f1=delta, baseline=baseline, evaluation=proposed,
                      reason='strict_macro_improvement' if accepted else 'no_strict_macro_improvement')
        receipt = dict(schema=SCHEMA, execution_binding=self.execution_binding,
                       certificate=certificate, ordered_selected_rules=[r.to_dict() for r in selected],
                       appended_rule=rule.to_dict(), original_baseline_payload=baseline,
                       proposed_program_key=key, proposed_payload_sha256=digest(proposed),
                       complete_facts_rounds_conflicts_and_metrics_preserved=True,
                       proof_materialized=False, selection_only=True,
                       new_proposal_Chase_performed=False, score_cache_written=False)
        self.gate.verify(); e._verify()
        path = self.directory / (key + '.json')
        if path.exists():
            if json.loads(path.read_text()) != receipt:
                raise ValueError('saved selection no-op certificate differs')
        else:
            write_json(path, receipt, exclusive=True)
        self.last_execution_receipt = copy.deepcopy(receipt)
        self.reused_admissions += 1
        return result


def make_selection_evaluator(runner, **kwargs):
    return SelectionEvaluator(runner, **kwargs)


def wrap_selection_evaluator(evaluator, runner, receipt_dir):
    return SelectionAdmissionReuse(evaluator, runner, receipt_dir)
