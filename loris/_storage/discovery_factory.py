"""Exact state bookkeeping, proof storage and static no-op for NEW discovery.

The original runner's context and evaluator program keys are unchanged. Bind
``execution_binding()`` in the enclosing new protocol before construction. This
module never changes original classes, functions, globals, or running studies.
Only its factory-owned runner class can use the compatibility gate.
"""
import copy
import functools
import inspect
import json
from pathlib import Path
from types import FunctionType

from loris._engine.typed_runtime import IndexedChaseRunner, ProofDAG, _State
from loris._engine.rlearner import TwoFilterEvaluator
from loris._engine.compressed_equal import ExactEqualityDispatch
from loris._search_space.evaluator import GuardedTwoFilterEvaluator
from loris._learning import noop_admission as original_noop
from loris._learning.artifacts import file_record, write_json
from . import proof_storage, state_execution, implicit_seed_proof

SCHEMA = 'paper36_strict_diagonal_compact_seed_proof_discovery_v4'

# Factory ownership is exact, not an isinstance test on arbitrary subclasses.
_RUNNER_CLASS = state_execution.make_exact_state_runner_class(
    proof_storage.make_exact_proof_runner_class(IndexedChaseRunner))
_RUNNER_CLASS._seed_proof = implicit_seed_proof.seed_proof
_ORIGINAL_METHODS = {name: getattr(IndexedChaseRunner, name) for name in (
    '__init__', '_rule', '_prepare', '_static', '_relation_domain', '_valuations',
    '_evidence', '_gamma', 'program_key', 'close')}
_CLONED_METHODS = {name: getattr(_RUNNER_CLASS, name)
                   for name in ('__init__', '__call__', 'run', '_seed_proof', 'verify_code')}
_ORIGINAL_PROOF_METHODS = {name: getattr(IndexedChaseRunner, name)
                          for name in ('__call__', '_seed_proof')}
def _descriptors(cls):
    return {name: value for name, value in vars(cls).items()
            if inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod, property))}


_ORIGINAL_DESCRIPTORS = _descriptors(IndexedChaseRunner)
_OWNED_DESCRIPTORS = _descriptors(_RUNNER_CLASS)
_NOOP_DESCRIPTORS = {cls: _descriptors(cls) for cls in
                     (original_noop.StaticBodyNoOpGate, original_noop.NoOpAdmissionAdapter)}
_SOURCE_MODULES = (proof_storage, state_execution, implicit_seed_proof, original_noop,
                   inspect.getmodule(IndexedChaseRunner),
                   inspect.getmodule(TwoFilterEvaluator),
                   inspect.getmodule(ExactEqualityDispatch),
                   inspect.getmodule(GuardedTwoFilterEvaluator),
                   inspect.getmodule(original_noop.validate_rule_guard))
_SOURCE_PATHS = tuple(dict.fromkeys([str(Path(__file__).resolve())] +
                     [str(Path(module.__file__).resolve()) for module in _SOURCE_MODULES]))
_EXECUTION_POLICY = {
    'schema': SCHEMA,
    'sources': [file_record(path) for path in _SOURCE_PATHS],
    'proof_storage': proof_storage.storage_binding(),
    'root_storage': dict(schema=implicit_seed_proof.SCHEMA, physical_identity_changed=True,
                         complete_logical_keys_records_order_and_counts_preserved=True,
                         canonical_expanded_root_JSON_hash_unchanged=False),
    'state_execution': state_execution.state_execution_binding(),
    'runner': 'factory_owned_original_schedule_with_exact_state_bookkeeping_and_ProofDAG_storage',
    'no_op': 'original_complete_static_body_superset_empty_certificate_and_admission',
    'storage_wrapper_preserves_current_runtime005_base_binding_and_program_keys': True,
    'prior_runtime004_binding_or_score_cache_reuse_permitted': False,
    'successful_fixedpoint_facts_and_scores_preserved': True,
    'valuation_order': 'actual_complete_body_strict_diagonal_witness_first_else_original',
    'failed_partial_facts_first_conflict_and_proofs_match_prior_order': False,
    'all_candidates_keep_original_order_and_both_filter_evaluations': True,
    'compressed_equality_enabled_by_this_factory': False,
    'score_cache_written_by_noop_adapter': False,
    'original_sources_classes_or_globals_modified': False,
    'requires_new_enclosing_execution_protocol': True,
}


def _verify_execution_sources():
    for record in _EXECUTION_POLICY['sources']:
        if file_record(record['path']) != record:
            raise RuntimeError('bound exact discovery execution source changed: ' + record['path'])
    if proof_storage.storage_binding() != _EXECUTION_POLICY['proof_storage']:
        raise RuntimeError('proof storage execution binding changed')
    if state_execution.state_execution_binding() != _EXECUTION_POLICY['state_execution']:
        raise RuntimeError('state execution binding changed')
    for cls, expected in _NOOP_DESCRIPTORS.items():
        if _descriptors(cls) != expected:
            raise ValueError('original no-op method identity changed')


def execution_binding():
    """Source-bound NEW execution policy, separate from the original score key."""
    _verify_execution_sources()
    return copy.deepcopy(_EXECUTION_POLICY)


def _function(value):
    return value.__func__ if inspect.ismethod(value) else value


def _verify_owned_runner(runner):
    if type(runner) is not _RUNNER_CLASS:
        raise TypeError('runner must be created by make_discovery_runner; arbitrary subclasses are not certified')
    _verify_execution_sources()
    if _descriptors(IndexedChaseRunner) != _ORIGINAL_DESCRIPTORS or _descriptors(_RUNNER_CLASS) != _OWNED_DESCRIPTORS:
        raise ValueError('original or exact runner method/descriptor identity changed')
    for name in _ORIGINAL_DESCRIPTORS:
        if name in runner.__dict__:
            raise ValueError('runner instance method/descriptor override: ' + name)
    if runner.proof_storage_binding != _EXECUTION_POLICY['proof_storage']:
        raise ValueError('runner proof-storage identity differs from the bound factory')
    if runner.state_execution_binding != _EXECUTION_POLICY['state_execution']:
        raise ValueError('runner state-execution identity differs from the bound factory')
    for name, original in _ORIGINAL_METHODS.items():
        if (getattr(IndexedChaseRunner, name) is not original
                or _function(getattr(runner, name)) is not getattr(_RUNNER_CLASS, name)):
            raise ValueError('original runner method identity changed: ' + name)
        # __init__ is intentionally supplied by the storage subclass. All
        # matching, domains, validation and scoring methods are inherited.
        if name != '__init__' and getattr(_RUNNER_CLASS, name) is not original:
            raise ValueError('uncertified runner override: ' + name)
    for name, method in _CLONED_METHODS.items():
        if getattr(_RUNNER_CLASS, name) is not method or _function(getattr(runner, name)) is not method:
            raise ValueError('exact proof runner method changed: ' + name)
    for name, original in _ORIGINAL_PROOF_METHODS.items():
        clone = _CLONED_METHODS[name]
        if name == '_seed_proof':
            if (getattr(IndexedChaseRunner, name) is not original
                    or clone is not implicit_seed_proof.seed_proof):
                raise ValueError('isolated compact seed proof binding changed')
            continue
        if (getattr(IndexedChaseRunner, name) is not original
                or clone.__code__ is not original.__code__
                or original.__globals__.get('ProofDAG') is not ProofDAG
                or clone.__globals__.get('ProofDAG') is not proof_storage.ExactProofDAG):
            raise ValueError('isolated proof constructor binding changed: ' + name)
    if _CLONED_METHODS['run'] is not _CLONED_METHODS['__call__']:
        raise ValueError('run and __call__ no longer share the exact constructor')
    if (IndexedChaseRunner.__call__.__globals__.get('_State') is not _State
            or _CLONED_METHODS['__call__'].__globals__.get('_State') is not
            state_execution.ExactBookkeepingState):
        raise ValueError('isolated state constructor binding changed')


def _clone_initializer(function, name, expected, replacement):
    """Preserve the original constructor code without a global monkey patch."""
    if function.__globals__.get(name) is not expected or name not in function.__code__.co_names:
        raise ValueError('unexpected original no-op constructor binding: ' + name)
    namespace = dict(function.__globals__)
    namespace[name] = replacement
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    functools.update_wrapper(clone, function)
    clone.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return clone


_GATE_INIT = _clone_initializer(original_noop.StaticBodyNoOpGate.__init__,
                               'IndexedChaseRunner', IndexedChaseRunner, _RUNNER_CLASS)


class ExactProofStaticNoOpGate(original_noop.StaticBodyNoOpGate):
    """The unchanged full-row gate, with narrowly certified runner enrollment."""
    def __init__(self, runner):
        _verify_owned_runner(runner)
        _GATE_INIT(self, runner)

    def verify(self):
        _verify_owned_runner(self.runner)
        super().verify()


_ADAPTER_INIT = _clone_initializer(original_noop.NoOpAdmissionAdapter.__init__,
                                  'StaticBodyNoOpGate', original_noop.StaticBodyNoOpGate,
                                  ExactProofStaticNoOpGate)


class ExactProofAdmissionReuse(original_noop.NoOpAdmissionAdapter):
    """Original admission result, durable full reuse receipts, exact storage.

    This remains synchronous. It certifies a candidate before its cache miss
    invokes the runner; it does not prefill a running worker's SQLite database.
    ``reused_admissions`` counts successful certified calls, not distinct rules.
    """
    def __init__(self, evaluator, runner, receipt_dir):
        _verify_owned_runner(runner)
        self.directory = Path(receipt_dir).resolve()
        self.reused_admissions = 0
        self._owned_runner = runner
        _ADAPTER_INIT(self, evaluator, runner)
        original_binding = copy.deepcopy(self.execution_binding)
        self.execution_binding = {
            'schema': SCHEMA,
            'execution_policy': execution_binding(),
            'original_noop_execution_binding': original_binding,
            'original_evaluator_binding_id': self.binding_id,
            'durable_receipt_directory': str(self.directory),
            'reuse_certificate_saved_before_admission_return': True,
            'new_source_binding_does_not_relabel_old_execution_as_new': True,
        }

    def _verify(self):
        _verify_owned_runner(self._owned_runner)
        super()._verify()

    def admit(self, selected, rule, epsilon=1e-12):
        result = super().admit(selected, rule, epsilon)
        if self.last_execution_receipt is not None:
            receipt = copy.deepcopy(self.last_execution_receipt)
            receipt['exact_storage_execution_binding'] = copy.deepcopy(self.execution_binding)
            path = self.directory / (receipt['proposed_program_key'] + '.json')
            if path.exists():
                if json.loads(path.read_text()) != receipt:
                    raise ValueError('existing exact proof/no-op execution receipt differs')
            else:
                write_json(path, receipt, exclusive=True)
            self.last_execution_receipt = copy.deepcopy(receipt)
            self.receipts.clear()  # Full evidence now lives in the durable file.
            self.reused_admissions += 1
        return result


def make_discovery_runner(context, cache_dir, **runner_options):
    """Create the original schedule with exact bookkeeping and proof storage."""
    _verify_execution_sources()
    runner = _RUNNER_CLASS(context, cache_dir, **runner_options)
    try:
        _verify_owned_runner(runner)
    except BaseException:
        runner.close()
        raise
    return runner


def wrap_discovery_evaluator(evaluator, runner, receipt_dir):
    """Wrap an original or connected evaluator; returns its original scores.

    A caller may already have an ExactEqualityDispatch. Its compressed branch
    remains outside no-op reuse: the original adapter checks BOTH programs for
    fallback-only scope, preserving its program-dependent proof identity rule.
    This function does not install such a dispatcher itself.
    """
    return ExactProofAdmissionReuse(evaluator, runner, receipt_dir)
