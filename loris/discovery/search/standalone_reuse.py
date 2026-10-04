"""Reuse authenticated, completed BO standalone evaluations during admission.

This does not reuse a proposed-prefix decision. The original admission method
still evaluates that prefix and its candidate. Only the fixed single-rule
input to the fit precision gate is cached from the same completed BO binding.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from loris._engine.rlearner import TwoFilterEvaluator, canonical, digest, rule_record
from loris._storage.selection_evaluator import SelectionEvaluator, _validate_payload
from loris.rule_forms import parse_author_rule
from . import parallel_admission as parallel

SCHEMA = 'loris_same_binding_completed_BO_standalone_reuse_v1'


def _raw_key(raw):
    return canonical(raw)


def authenticated_scores(directory, records, evaluator):
    """Authenticate the complete BO before extracting any function result."""
    evaluator = getattr(evaluator, 'evaluator', evaluator)
    if type(evaluator) is not SelectionEvaluator:
        raise TypeError('same-binding complete selection evaluator required')
    evaluator._verify()
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    complete = json.loads((directory / 'BO_COMPLETE.json').read_text())
    if (complete['protocol_id'] != digest(protocol)
            or protocol['evaluator'] != evaluator.binding
            or protocol['whole_program_prefix_used_in_BO']
            or protocol['historical_candidates_loaded']
            or complete['complete_trials'] != len(records)
            or complete['all_trial_records_sha256'] != digest(records)
            or [r['number'] for r in records] != list(range(len(records)))):
        raise ValueError('standalone reuse requires unchanged completed same-binding BO')
    values = {}
    for record in records:
        raw = record['rule']
        if raw is None:
            continue
        rule = parse_author_rule(raw)
        value = record['standalone']['evaluation']
        key = evaluator.program_key((rule,))
        _validate_payload(value, key, evaluator.binding)
        # Run the original score/gate method on its authenticated function
        # result, without invoking an engine or mutating the parent cache.
        view = SimpleNamespace(evaluate=lambda rules: value,
                               fit_precision_gate=evaluator.fit_precision_gate,
                               head_info=evaluator.head_info, labels=evaluator.labels,
                               line7_mode=evaluator.line7_mode)
        if TwoFilterEvaluator.standalone(view, rule) != record['standalone']:
            raise ValueError('standalone score or fit gate differs from its complete evaluation')
        raw_key = _raw_key(raw)
        if raw_key in values and values[raw_key] != value:
            raise ValueError('same BO rule has differing complete standalone results')
        values[raw_key] = copy.deepcopy(value)
    evaluator._verify()
    return values, dict(schema=SCHEMA, evaluator_binding_id=evaluator.binding_id,
                        protocol_id=complete['protocol_id'],
                        all_trial_records_sha256=complete['all_trial_records_sha256'],
                        complete_trials=len(records), distinct_rules=len(values),
                        proposed_prefix_decisions_reused=False)


def _score_with_standalone(task):
    prefix, raw, order, binding_id, standalone = task
    _, evaluator, _ = parallel._worker
    evaluator._verify()
    rule = parse_author_rule(raw)
    key = evaluator.program_key((rule,))
    if binding_id != evaluator.binding_id:
        raise ValueError('worker standalone evaluator binding differs')
    _validate_payload(standalone, key, evaluator.binding)
    old = evaluator.memory.get(key)
    if old is not None:
        # Existing cache entries may record a different number of executed
        # rounds after a warm start. Every factual/metric/counter field must
        # still agree; retain the already recorded local execution statistic.
        if {k:v for k,v in old.items() if k != 'n_rounds'} != {k:v for k,v in standalone.items() if k != 'n_rounds'}:
            raise ValueError('worker single-rule result differs from completed BO')
    # With an empty prefix the proposed program is the single-rule cache key.
    # Inserting it could change the no-op adapter's cached/uncached branch and
    # its complete receipt, so preserve that first-prefix path verbatim.
    if prefix and old is None:
        evaluator.memory[key] = copy.deepcopy(standalone)
    decision, cost = parallel._score((prefix, raw, order))
    cost['authenticated_BO_standalone_reused'] = bool(prefix)
    return decision, cost


class BOStandaloneScorer(parallel.ParallelScorer):
    """ParallelScorer with an explicit, authenticated cluster registration."""
    def register_cluster(self, directory, records, evaluator):
        self.standalone, self.standalone_receipt = authenticated_scores(directory, records, evaluator)
        return copy.deepcopy(self.standalone_receipt)

    def __call__(self, selected, rules, order):
        if not hasattr(self, 'standalone'):
            raise ValueError('completed BO must be registered before standalone reuse')
        prefix = [rule_record(r) for r in selected]
        raw = [rule_record(r) for r in rules]
        tasks = [(prefix, r, order, self.standalone_receipt['evaluator_binding_id'],
                  self.standalone[_raw_key(r)]) for r in raw]
        results = list(self.pool.map(_score_with_standalone, tasks, chunksize=1))
        self.calls += sum(r[1]['calls'] for r in results)
        self.compute_seconds += sum(r[1]['compute_seconds'] for r in results)
        return [r[0] for r in results], [r[1] for r in results]
