"""Exact line-12 reuse when the appended rule cannot ever match its body.

This optional adapter is not installed in any running study. Only a complete
empty static-body join is certified. Dynamic signed/label-comparison atoms are
ignored to form a superset of possible bodies, never interpreted as false.
No known-head pruning is allowed. Unknown cases execute the original admit.
"""
import copy
import hashlib
from pathlib import Path

from loris._engine.rdl_reference import Rule, Doc, Text, ML, Label, AttrEqual, LabelCompare
from loris._engine.typed_runtime import IndexedChaseRunner, indices
from loris._engine.rlearner import TwoFilterEvaluator, digest, gamma_identity, array_identity
from loris._engine.compressed_equal import ExactEqualityDispatch, UnsupportedDenseEquality
from loris._search_space.configuration import validate_rule_guard, guard_policy_binding
from loris._search_space.evaluator import GuardedTwoFilterEvaluator, _validate_payload

SCHEMA = 'paper36_complete_static_body_noop_admission_v1'
BASE_FIELDS = frozenset(('program_key','status','n_rounds','conflicts','comparison_conflicts',
                        'positive_metrics','negative_metrics','full_state'))


class StaticBodyNoOpGate:
    def __init__(self, runner):
        if type(runner) is not IndexedChaseRunner:
            raise TypeError('original IndexedChaseRunner required')
        self.runner = runner
        self.context_binding = copy.deepcopy(runner.binding)
        self.context_id = runner.context_id
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.policy = guard_policy_binding()
        self._cache = {}
        self.last_audit = None

    def verify(self):
        self.runner.verify_code()
        if (self.runner.binding != self.context_binding or self.runner.context_id != self.context_id
                or self.policy != guard_policy_binding()
                or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self.source_sha256):
            raise ValueError('frozen no-op gate context/source/guard changed')

    def certify(self, rule):
        self.verify()
        if not isinstance(rule, Rule):
            raise TypeError('typed Rule required')
        validate_rule_guard(rule)
        key = digest(rule.to_dict())
        if key in self._cache:
            self.last_audit = copy.deepcopy(self._cache[key])
            return copy.deepcopy(self.last_audit) if self.last_audit['certified_noop'] else None
        # _prepare uses every declared Text/ML domain and attribute-presence
        # aware equality index on the complete corpus, with no rule valuation.
        runner = self.runner
        parsed, identity = runner._rule(rule)
        prepared = runner._prepare(parsed, identity)
        if any(not isinstance(a, (Doc,Text,ML,Label,AttrEqual,LabelCompare)) for a in rule.body):
            raise ValueError('unsupported body atom cannot be an empty-body certificate')
        static_binary = tuple(a for a in prepared['binary'] if isinstance(a, AttrEqual))
        assignment = {}
        visited = 0
        def witness(depth):
            nonlocal visited
            if depth == len(rule.variables):
                return dict(assignment)
            var = rule.variables[depth]
            domain = prepared['domains'][var]
            for atom in static_binary:
                if var in (atom.left_var,atom.right_var):
                    domain &= runner._relation_domain(atom,var,assignment,None)
            for row in indices(domain):
                visited += 1
                assignment[var] = row
                found = witness(depth+1)
                if found is not None:
                    return found
            assignment.pop(var,None)
            return None
        found = witness(0)
        audit = {'schema':SCHEMA,'certified_noop':found is None,
            'context_id':self.context_id,'context_binding_sha256':digest(self.context_binding),
            'rule_sha256':key,'source_sha256':self.source_sha256,'guard':copy.deepcopy(self.policy),
            'full_rows':runner.n,'labels':list(runner.labels),'all_variable_axes':list(rule.variables),
            'visited_static_prefixes':visited,'static_witness':found,
            'ignored_monotone_atom_indices':[i for i,a in enumerate(rule.body) if isinstance(a,(Label,LabelCompare))],
            'known_head_pruning_used':False,'negative_unknown_as_false_used':False,
            'empty_result_covers_full_declared_domain':found is None,
            'theorem':'empty_static_body_superset_implies_appended_rule_never_fires_in_any_round'}
        self._cache[key] = copy.deepcopy(audit)
        self.last_audit = copy.deepcopy(audit)
        return audit if found is None else None


class NoOpAdmissionAdapter:
    """Optional exact score adapter; owns no database and writes no score cache.

    Existing function results and provenance-bearing baselines use original
    admit. Plain completed baseline payloads may be reused after authentication.
    The independent execution receipt retains the complete source payload.
    """
    def __init__(self, evaluator, runner):
        if type(evaluator) not in (TwoFilterEvaluator, GuardedTwoFilterEvaluator):
            raise TypeError('original or connected TwoFilterEvaluator required')
        actual = evaluator.runner
        if actual is not runner and not (type(actual) is ExactEqualityDispatch and actual.runner is runner):
            raise ValueError('evaluator must execute this exact runner context')
        self.evaluator = evaluator
        self._runner_object = actual
        self.gate = StaticBodyNoOpGate(runner)
        self.binding = copy.deepcopy(evaluator.binding)
        self.binding_id = evaluator.binding_id
        self._runner_binding = copy.deepcopy(getattr(actual,'binding',None))
        runtime = self.binding['runtime']
        if type(evaluator) is GuardedTwoFilterEvaluator:
            if runtime.get('guard') != guard_policy_binding():
                raise ValueError('connected evaluator guard differs')
            runtime = runtime['original_runtime_binding']
        if runtime != self._runner_binding:
            raise ValueError('evaluator runtime binding differs from actual callable')
        self.receipts = []
        self.last_execution_receipt = None
        # Score keys remain the original exact function keys. The enclosing
        # NEW experiment protocol must separately bind this execution policy.
        self.execution_binding = {'schema':SCHEMA,'source_sha256':self.gate.source_sha256,
            'guard':copy.deepcopy(self.gate.policy),'original_evaluator_binding_id':self.binding_id,
            'scope':'complete_empty_static_body_join_only',
            'all_trial_records_still_enter_line12':True,'score_cache_written':False}
        self._verify()

    def __getattr__(self, name):
        # Expose original actual_calls/cache_hits/timing and sampler-facing
        # properties without manufacturing execution statistics.
        return getattr(self.evaluator,name)

    def evaluate(self, rules):
        self._verify()
        rules = tuple(rules)
        for rule in rules:
            validate_rule_guard(rule)
        return self.evaluator.evaluate(rules)

    def standalone(self, rule):
        self._verify()
        validate_rule_guard(rule)
        return self.evaluator.standalone(rule)

    def program_key(self, rules):
        return self.evaluator.program_key(rules)

    def close(self):
        self.evaluator.close()

    def _verify(self):
        self.gate.verify()
        e = self.evaluator
        if (e.runner is not self._runner_object or e.binding != self.binding or e.binding_id != self.binding_id or digest(e.binding) != self.binding_id
                or gamma_identity(e.gamma) != self.binding['gamma']
                or array_identity(e.indices) != self.binding['receivers']
                or array_identity(e.truth) != self.binding['truth']
                or list(e.labels) != self.binding['labels']
                or getattr(e.runner,'binding',None) != self._runner_binding):
            raise ValueError('fixed evaluator Gamma/data/model/score binding changed')

    def _cached(self,key):
        e = self.evaluator
        return key in e.memory or (e.db is not None and
            e.db.execute('SELECT 1 FROM programs WHERE key=?',(key,)).fetchone() is not None)

    def _original_indexed_proof_scope(self, selected, rule, baseline):
        if baseline['full_state'].get('proof_dag',{}).get('schema') != 'exact_immutable_layered_proof_dag_v1':
            return False
        actual = self.evaluator.runner
        if type(actual) is ExactEqualityDispatch:
            # A compressed recipe binds its entire program, and a change of
            # executor can change the first proof even for an inert rule.
            gamma = actual.runner._gamma(self.evaluator.gamma)
            for program in (selected,selected+(rule,)):
                try:
                    actual.compressed._certify(program,gamma)
                except UnsupportedDenseEquality:
                    continue
                return False
        return True

    def admit(self, selected, rule, epsilon=1e-12):
        self._verify()
        selected = tuple(selected)
        for value in selected+(rule,):
            validate_rule_guard(value)
        e = self.evaluator
        key = e.program_key(selected+(rule,))
        certificate = self.gate.certify(rule)
        self.last_execution_receipt = None
        if certificate is None or self._cached(key):
            return e.admit(selected,rule,epsilon)
        baseline = e.evaluate(selected)
        self._verify()
        if baseline['status'] != 'fixpoint':
            raise ValueError('selected program must be consistent')
        # Do not reattach an old program's migration/cache provenance to a new
        # program. Such payloads conservatively take the original execution.
        if set(baseline) != BASE_FIELDS or not self._original_indexed_proof_scope(selected,rule,baseline):
            return e.admit(selected,rule,epsilon)
        _validate_payload(baseline,e.program_key(selected),self.binding)
        proposed = copy.deepcopy(baseline)
        proposed['program_key'] = key
        delta = proposed['positive_metrics']['macro_f1']-baseline['positive_metrics']['macro_f1']
        accepted = delta > epsilon
        result = {'line':12,'accepted':bool(accepted),'criterion':'positive_macro_f1',
            'delta_macro_f1':delta,'baseline':baseline,'evaluation':proposed,
            'reason':'strict_macro_improvement' if accepted else 'no_strict_macro_improvement'}
        receipt = {'schema':SCHEMA,'certificate':certificate,'evaluator_binding_id':self.binding_id,
            'ordered_selected_rules':[r.to_dict() for r in selected],'appended_rule':rule.to_dict(),
            'original_baseline_payload':copy.deepcopy(baseline),'baseline_payload_sha256':digest(baseline),
            'proposed_payload_sha256':digest(proposed),'proposed_program_key':key,
            'all_facts_proof_identity_rounds_metrics_preserved':True,
            'original_rule_indices_unchanged':True,'appended_rule_never_fires':True,
            'both_programs_use_original_indexed_proof_schema':True,
            'new_proposal_Chase_performed':False,'score_cache_written':False,
            'complete_proof_restoration':'replay original ordered baseline with original bound Gamma/context; appended rule has no matching body'}
        self.last_execution_receipt = copy.deepcopy(receipt)
        self.receipts.append(copy.deepcopy(receipt))
        return result
