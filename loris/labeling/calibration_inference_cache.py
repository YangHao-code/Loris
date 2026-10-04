"""Exact, explicitly authenticated reuse of arbitration-free calibration inference.

This stores only logical output matrices and already observed per-rule scores.
It is not a checkpoint, retained-history cache, or a substitute for current-model
prediction, fallback threshold fitting, gate scoring, or a new full binding.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import numpy as np

from loris.discovery.manifest import bound, file_record
from loris._engine.rdl_reference import ML
from loris._engine.typed_session import array_identity, digest, seed_identity

SCHEMA = 'loris_exact_independent_calibration_inference_cache_v1'
COLLECTION = 'observed_receivers_and_new_heads'


def dependency_binding(runner, context, rules, gamma, pos, neg, selected, semantics):
    """Project actual ML dependencies, not alias names guessed from a checkpoint.

    The runner has already verified input file references. Their provenance
    paths and the opaque source context id are excluded here because their
    *contents* are independently bound below and in the current full artifact.
    Every other execution binding field is retained, including static input
    identities, complete source paths/hashes, scope, versions, and policies.
    """
    execution = runner.binding
    excluded = {key: execution.pop(key, None) for key in ('input_files', 'source_context_id')}
    del excluded  # Full current provenance stays in INPUTS/RELIABILITY, not this semantic key.
    used = {'probabilities': set(), 'static_probabilities': set()}
    for rule in rules:
        for atom in rule.body:
            if isinstance(atom, ML):
                used['probabilities' if atom.dynamic else 'static_probabilities'].add(atom.alias)
    for table, aliases in used.items():
        execution[table] = {a: execution[table][a] for a in sorted(aliases)}
    # Typed ML execution consumes probabilities, never legacy binary decisions.
    execution['decisions'] = {}
    package = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(package)): file_record(p)['sha256']
            for p in sorted(package.rglob('*.py'))}
    return dict(schema=SCHEMA, execution=execution, semantic_sources=code,
                rules=[r.to_dict() for r in rules], chase_semantics=semantics,
                masked_Gamma=seed_identity(gamma), row_ids=digest(context['row_ids']),
                labels=list(context['labels']), selected=array_identity(np.asarray(selected, bool)),
                observed_positive=array_identity(pos), observed_negative=array_identity(neg),
                inference='fresh_retained_without_arbitration_or_temporary_seeds',
                conclusion_collection=COLLECTION,
                output_use='logical_matrices_and_rule_scores_only_not_history_or_proofs')


def load_parent(reference, expected):
    """Only a SHA-bound complete parent RESULTS receipt may nominate a cache."""
    if reference is None:
        return None
    receipt_path = bound(reference, Path('.'))
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'complete':
        raise ValueError('calibration cache parent receipt is not complete')
    record = receipt.get('rule_inference_cache')
    if record is None:
        return None  # Historical artifacts remain valid but cannot provide this cache.
    path = bound(record, receipt_path.parent)
    value = json.loads(path.read_text())
    if value.get('schema') != SCHEMA:
        raise ValueError('unknown independent calibration inference cache schema')
    if value.get('dependency_binding') != expected:
        return None  # A changed premise, source, axis, Gamma, rule or answer needs a fresh chase.
    if value.get('rule_scores') != receipt.get('rule_scores'):
        raise ValueError('cached rule scores differ from the authenticated parent receipt')
    arrays_path = bound(value['arrays'], path.parent)
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != {'positive', 'negative', 'unknown'}:
            raise ValueError('calibration cache array names differ')
        arrays = {key: archive[key].copy() for key in archive.files}
    shape = (expected['execution']['full_rows'], len(expected['labels']))
    for key, array in arrays.items():
        if array.dtype != np.bool_ or array.shape != shape or array_identity(array) != value['array_identities'][key]:
            raise ValueError('calibration cache matrix identity/axes differ: ' + key)
    if np.any(arrays['positive'] & arrays['negative']) or np.any((arrays['positive'] | arrays['negative']) & arrays['unknown']):
        raise ValueError('cached retained facts are not conflict-quarantined')
    scores = copy.deepcopy(value['rule_scores'])
    if len(scores) != len(expected['rules']):
        raise ValueError('cached rule score axis differs')
    for index, score in enumerate(scores):
        if (score['rule_id'] != index or score['rule_sha256'] != digest(expected['rules'][index])
                or score['q'] != (score['correct'] + 1.0) / (score['support'] + 2.0)
                or not 0 <= score['correct'] <= score['support']):
            raise ValueError('cached per-rule q record is inconsistent')
    closure = SimpleNamespace(status='fixpoint', positive_facts=arrays['positive'],
                              negative_facts=arrays['negative'], conflict_unknown=arrays['unknown'])
    return closure, scores, dict(parent_receipt=file_record(receipt_path), cache=file_record(path),
                                arrays=file_record(arrays_path), original_execution=value['original_execution'])


def store(output, binding, closure, scores, reused_from=None):
    output = Path(output)
    if reused_from is None:
        path = output / 'RULE_INFERENCE.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, positive=closure.positive_facts, negative=closure.negative_facts,
                                unknown=closure.conflict_unknown)
        arrays = file_record(path)
        execution = dict(n_rounds=closure.n_rounds, maintenance=closure.reference_audit['maintenance'])
    else:
        arrays = reused_from['arrays']
        # Verify again immediately before publishing the new reference.
        bound(arrays, Path('.'))
        execution = reused_from['original_execution']
    value = dict(schema=SCHEMA, dependency_binding=binding, arrays=arrays,
                 array_identities={key: array_identity(array) for key, array in (
                     ('positive', closure.positive_facts), ('negative', closure.negative_facts),
                     ('unknown', closure.conflict_unknown))}, rule_scores=scores,
                 original_execution=execution, reused_from=reused_from,
                 fresh_inference_executed=reused_from is None,
                 current_model_inputs_bound_separately=True,
                 fallback_and_gate_evaluated_for_current_manifest=True)
    path = output / 'RULE_INFERENCE.json'
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False);stream.write('\n')
    return file_record(path)
