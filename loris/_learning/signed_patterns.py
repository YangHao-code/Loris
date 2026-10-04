"""Signed pattern groups selected by complete Chase and fixed-control Macro-F1.

This is the user-specified completion of the paper's pattern-selection step,
not an assertion that the paper specifies this algorithm.  B is a prediction
table, never a fact seed.  Successful Chase yields P and N; the scored table is
(B & ~N) | P.  Any logical conflict invalidates the whole proposed rule group.
After every accepted group all remaining groups are evaluated again against
the entire accumulated program.  The separate legacy AddOnly path is unchanged.
"""
from dataclasses import asdict, dataclass
import base64
import binascii
import copy
import inspect
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from loris.chase import GammaSeed
from loris.document import Document
from loris._engine.rdl_reference import Label, RefGamma, ReferenceChase, Rule, Text, prepare_attributes
from loris._engine.rlearner import array_identity, canonical, digest
from loris._engine.rlearner_config import PatternSpec
from loris._engine.typed_session import TypedChaseSession, complete_state_identity, seed_identity
from loris._engine.typed_runtime import ProofDAG, RelationFacts
from loris._engine.relation_rows import RelationRows
from loris.predicates import predicate_to_dict
from .artifacts import file_record, write_json
from .patterns import _binary, _pattern_spec, _rows, macro_f1
from .recovery import conflict_certificate, root_keys


SCHEMA = 'paper36_signed_chase_patterns_v1'
CORRECTION = '(B & ~N) | P after successful complete Chase'


def signed_context_identity(context):
    """Pure identity of actual full-corpus semantics; creates no cache or files.

    Document.lbl is deliberately excluded: labels enter logical state only via
    explicit Gamma.  Probability and decision identities use the interpreter's
    normalized float64/bool tables, and missing-attribute presence is explicit.
    File checksums are declared here and checked by evaluator/artifact gates.
    """
    docs, labels = tuple(context['docs']), tuple(context['labels'])
    reference = ReferenceChase(labels,
        model_probabilities=context.get('model_probabilities', context.get('scores', {})),
        model_decisions=context.get('model_decisions', {}), subset_direction='standard',
        attribute_values=context.get('attribute_values', {}), attribute_presence=context.get('attribute_presence', {}))
    values, presence = prepare_attributes(docs, reference.attribute_values, reference.attribute_presence)
    n, k = len(docs), len(labels)
    if any(a.shape != (n, k) for tables in (reference.model_probabilities, reference.model_decisions) for a in tables.values()):
        raise ValueError('model interpretation must match the complete document/label axes')
    gamma = context['gamma']
    if not isinstance(gamma, (GammaSeed, RefGamma)) or gamma.shape != (n, k):
        raise ValueError('explicit full-axis Gamma required')
    value = {'full_rows': n, 'labels': list(labels), 'subset_direction': 'standard',
        'documents': digest([[d.cnt, d.ttl, d.mtd] for d in docs]),
        'attribute_values': {key: digest(column) for key, column in values.items()},
        'attribute_presence': {key: array_identity(mask) for key, mask in presence.items()},
        'probabilities': {key: array_identity(value) for key, value in reference.model_probabilities.items()},
        'decisions': {key: array_identity(value) for key, value in reference.model_decisions.items()},
        'gamma': seed_identity(gamma), 'source_context_id': context.get('context_id'),
        'input_files': {str(Path(path).resolve(strict=True)): sha for path, sha in context.get('input_files', {}).items()}}
    return json.loads(canonical(value))


def _indices(values, n, name):
    items = list(values)
    if (not items or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                         or x < 0 or x >= n for x in items)
            or len(set(map(int, items))) != len(items)):
        raise ValueError(name + ' must be nonempty unique integer indices in the full document axis')
    return np.asarray(items, dtype=np.int64)


def _immutable_binary(value, shape=None):
    value = _binary(value, shape)
    return np.frombuffer(value.tobytes(), dtype=bool).reshape(value.shape)


def _signed_fact_record(positive, negative):
    """Compact canonical full-axis facts, independently rescorable from disk."""
    return {'schema': 'paper36_signed_facts_packbits_v1', 'shape': list(positive.shape), 'bitorder': 'little',
        **{sign: base64.b64encode(np.packbits(value.reshape(-1), bitorder='little').tobytes()).decode('ascii')
           for sign, value in (('positive', positive), ('negative', negative))}}


def _read_signed_facts(record, shape):
    if isinstance(record, dict) and record.get("schema") == "paper36_exact_unary_signed_fact_delta_v1":
        from .analytic_unary_patterns import read_delta
        return read_delta(record, shape)
    if (not isinstance(record, dict) or set(record) != {'schema', 'shape', 'bitorder', 'positive', 'negative'}
            or record.get('schema') != 'paper36_signed_facts_packbits_v1'
            or not isinstance(record.get('shape'), list) or any(type(size) is not int for size in record['shape'])
            or record.get('shape') != list(shape) or record.get('bitorder') != 'little'):
        raise ValueError('successful Chase requires canonical complete signed-fact bitmaps')
    count, values = math.prod(shape), []
    for sign in ('positive', 'negative'):
        encoded = record[sign]
        try:
            packed = base64.b64decode(encoded, validate=True)
        except (TypeError, ValueError, binascii.Error):
            raise ValueError('signed-fact bitmap is not canonical base64') from None
        if (not isinstance(encoded, str) or base64.b64encode(packed).decode('ascii') != encoded
                or len(packed) != (count + 7) // 8
                or count % 8 and packed[-1] >> (count % 8)):
            raise ValueError('signed-fact bitmap has wrong length or nonzero padding')
        values.append(np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=count,
                                    bitorder='little').astype(bool).reshape(shape))
    return tuple(values)


def correction_diagnostics(truth, before, after):
    """Four disjoint changes, counted only on the fixed scoring rows/labels."""
    y = _binary(truth)
    old, new = _binary(before, y.shape), _binary(after, y.shape)
    additions, removals = ~old & new, old & ~new
    masks = {'fn_to_tp': additions & y, 'tn_to_fp': additions & ~y,
             'fp_to_tn': removals & ~y, 'tp_to_fn': removals & y}
    return {**{key: int(mask.sum()) for key, mask in masks.items()},
            'per_label': {key: np.count_nonzero(mask, axis=0).tolist() for key, mask in masks.items()},
            'scored_rows': len(y), 'scored_labels': y.shape[1]}


@dataclass(frozen=True)
class SignedPatternCandidate:
    key: str
    pattern_specs: tuple
    rules: tuple
    fit_provenance: Mapping

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError('signed candidate requires a nonempty key')
        specs = tuple(_pattern_spec(s.to_dict(), s.key) if isinstance(s, PatternSpec)
                      else _pattern_spec(s, s.get('key')) for s in self.pattern_specs)
        if not specs or len({s.key for s in specs}) != len(specs):
            raise ValueError('each signed candidate requires unique explicit PatternSpecs')
        rules = tuple(Rule.from_dict(r.to_dict() if isinstance(r, Rule) else r) for r in self.rules)
        # Empty groups are retained as honest raw no-effect candidates.  Auxiliary
        # comparison heads may accompany at least one explicit signed label head.
        if rules and not any(isinstance(r.head, Label) for r in rules):
            raise ValueError('a nonempty signed group needs an explicit target label and polarity')
        represented = {digest(predicate_to_dict(s.predicate)) for s in specs}
        used = {digest(predicate_to_dict(a.predicate)) for r in rules for a in r.body if isinstance(a, Text)}
        if not used.issubset(represented) or (rules and not represented.issubset(used)):
            raise ValueError('candidate text conditions and exported PatternSpecs must match exactly')
        if not isinstance(self.fit_provenance, Mapping):
            raise ValueError('explicit fit provenance required')
        provenance = json.loads(canonical(dict(self.fit_provenance)))
        object.__setattr__(self, 'pattern_specs', specs)
        object.__setattr__(self, 'rules', rules)
        object.__setattr__(self, 'fit_provenance', provenance)

    def to_dict(self):
        return {'key': self.key, 'pattern_specs': [s.to_dict() for s in self.pattern_specs],
                'rules': [r.to_dict() for r in self.rules],
                'fit_provenance': copy.deepcopy(self.fit_provenance)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'key', 'pattern_specs', 'rules', 'fit_provenance'}:
            raise ValueError('exact signed-candidate schema required')
        return cls(value['key'], tuple(value['pattern_specs']), tuple(value['rules']), value['fit_provenance'])


@dataclass(frozen=True)
class SignedPatternSelectionProtocol:
    expected_candidates: int
    min_gain: float = 0.

    def __post_init__(self):
        if type(self.expected_candidates) is not int or self.expected_candidates < 0:
            raise ValueError('full expected candidate count must be explicit and nonnegative')
        if type(self.min_gain) not in (int, float) or not math.isfinite(self.min_gain) or self.min_gain < 0:
            raise ValueError('min_gain must be finite and nonnegative')


@dataclass(frozen=True)
class SignedChaseEvaluation:
    feasible: bool
    macro_f1: object
    positive_facts: np.ndarray
    negative_facts: np.ndarray
    corrected_predictions: object
    diagnostics: object
    status: str
    chase_receipt: Mapping
    state_identity: Mapping
    conflict_certificate: object = None

    def record(self):
        return {'feasible': self.feasible, 'score': self.macro_f1,
                'diagnostics': copy.deepcopy(self.diagnostics), 'status': self.status,
                'state_identity': copy.deepcopy(self.state_identity),
                'chase_receipt': copy.deepcopy(self.chase_receipt),
                'conflict_certificate': copy.deepcopy(self.conflict_certificate),
                'signed_facts': _signed_fact_record(self.positive_facts, self.negative_facts) if self.feasible else None}


class SignedChaseEvaluator:
    """Shared real-Chase evaluator for fit head fitting and control selection.

    The caller of fit evaluation must first remove scored fit roots (including
    incident relation roots) from a COPY of Gamma.  This low-level evaluator
    receives that explicit seed; it never adds baseline predictions or truth.
    The selection entry point additionally enforces control/Gamma separation.
    """
    def __init__(self, context, *, baseline_predictions, score_indices, truth,
                 cache_dir, model_snapshot_id=None, background_rules=()):
        n, k = len(context['docs']), len(context['labels'])
        if not n or not k or len(set(context['labels'])) != k:
            raise ValueError('nonempty complete document and unique label axes required')
        self.indices = _indices(score_indices, n, 'score_indices')
        self.indices.flags.writeable = False
        self.truth = _immutable_binary(truth, (len(self.indices), k))
        self.baseline = _immutable_binary(baseline_predictions, (n, k))
        gamma = context['gamma']
        if not isinstance(gamma, (GammaSeed, RefGamma)) or gamma.shape != (n, k):
            raise ValueError('explicit full-axis Gamma required')
        self.seed = RefGamma(gamma.pos.copy(), gamma.neg.copy(),
                             tuple(getattr(gamma, 'subset', ())), tuple(getattr(gamma, 'strict_subset', ())))
        self._context = {'docs': [Document(cnt=d.cnt, ttl=d.ttl, mtd=copy.deepcopy(d.mtd)) for d in context['docs']],
            'labels': list(context['labels']), 'gamma': self.seed,
            'model_probabilities': {key: np.array(value, copy=True) for key, value in
                                    context.get('model_probabilities', context.get('scores', {})).items()},
            'model_decisions': {key: np.array(value, copy=True) for key, value in context.get('model_decisions', {}).items()},
            'attribute_values': copy.deepcopy(context.get('attribute_values', {})),
            'attribute_presence': {key: np.array(value, copy=True) for key, value in context.get('attribute_presence', {}).items()},
            'input_files': dict(context.get('input_files', {})), 'context_id': context.get('context_id')}
        self.background_rules = tuple(Rule.from_dict(r.to_dict() if isinstance(r, Rule) else r) for r in background_rules)
        snapshot_binding = {'probabilities': {key: array_identity(value) for key, value in self._context['model_probabilities'].items()},
            'decisions': {key: array_identity(value) for key, value in self._context['model_decisions'].items()},
            'input_files': self._context['input_files']}
        self.model_snapshot_id = model_snapshot_id or digest(snapshot_binding)
        self.cache_dir = Path(cache_dir)
        session = self._session(())
        try:
            self._binding = {'context_id': session.context_id, 'identity': signed_context_identity(self._context),
                'runtime': session.runner.binding,
                'gamma': seed_identity(self.seed), 'model_snapshot': snapshot_binding,
                'model_snapshot_id': self.model_snapshot_id, 'labels': list(context['labels']),
                'score_indices': self.indices.tolist(), 'truth': array_identity(self.truth),
                'baseline': array_identity(self.baseline),
                'background_rules': [r.to_dict() for r in self.background_rules],
                'correction': CORRECTION, 'score': 'fixed-axis Macro-F1; zero denominator contributes zero',
                'candidate_evaluation_only': True, 'selection_complete_claim': False}
        finally:
            session.close()

    def _session(self, rules):
        return TypedChaseSession(self._context, self.background_rules + tuple(rules), self.cache_dir,
            model_snapshot_id=self.model_snapshot_id, subset_direction='standard', verify_source_files=True)

    @property
    def binding(self):
        return copy.deepcopy(self._binding)

    def evaluate(self, rules, *, previous_predictions=None):
        old = self.baseline if previous_predictions is None else _immutable_binary(previous_predictions, self.baseline.shape)
        session = self._session(rules)
        try:
            if session.context_id != self._binding['context_id']:
                raise RuntimeError('frozen signed-evaluation context changed')
            sources = {key: {'origin': 'validated_Gamma'} for key in root_keys(self.seed)}
            result = session.chase(self.seed, phase='labeling', root_sources=sources)
            state, receipt = complete_state_identity(result), session.receipt()
            positive = _immutable_binary(result.positive_facts, self.baseline.shape)
            negative = _immutable_binary(result.negative_facts, self.baseline.shape)
            feasible = result.status == 'fixpoint' and not np.any(positive & negative) and not result.comparison_conflicts
            if not feasible:
                # No score, no repaired subset, no partial program admission.
                certificate = conflict_certificate(result, session=session, seed=self.seed,
                    root_sources=sources, stage='signed_pattern_candidate_evaluation')
                # The selected conflict branches are sufficient for reading K;
                # retain the complete DAG as well so a file-only verifier can
                # reconstruct the actual proof identity in this Chase receipt.
                certificate['complete_proof_dag'] = result.proof_dag.to_dict()
                certificate['conflict_id'] = digest({key: value for key, value in certificate.items()
                                                     if key != 'conflict_id'})
                return SignedChaseEvaluation(False, None, positive, negative, None, None,
                                             result.status, receipt, state, certificate)
            corrected = _immutable_binary((self.baseline & ~negative) | positive)
            score = macro_f1(self.truth, corrected[self.indices])
            changes = correction_diagnostics(self.truth, old[self.indices], corrected[self.indices])
            return SignedChaseEvaluation(True, score, positive, negative, corrected, changes,
                                         result.status, receipt, state)
        finally:
            session.close()

    def close(self):
        """Sessions close after every complete evaluation; present for uniform use."""


def make_signed_evaluator(execution_backend="reference", **kwargs):
    if execution_backend == "reference":
        return SignedChaseEvaluator(**kwargs)
    if execution_backend == "analytic_unary_v1":
        from .analytic_unary_patterns import AnalyticUnarySignedChaseEvaluator
        return AnalyticUnarySignedChaseEvaluator(**kwargs)
    raise ValueError("unknown signed pattern execution backend")


def _source_files():
    # The runtime supplies its complete interpretation dependency list as well.
    from . import artifacts, patterns, recovery
    from loris._engine import rlearner_config, rlearner, typed_session
    paths = [Path(__file__), *[Path(inspect.getfile(m)) for m in
                              (artifacts, patterns, recovery, rlearner_config, rlearner, typed_session)]]
    paths.extend(Path(__file__).with_name(name) for name in
                 ('signed_candidate_generation.py', 'candidate_generation.py', 'semantic_patterns.py', 'analytic_unary_patterns.py'))
    return {str(p.resolve()): file_record(p)['sha256'] for p in paths}


def _verify_files(files):
    for path, expected in files.items():
        if file_record(path)['sha256'] != expected:
            raise ValueError('bound signed-pattern file changed: ' + str(path))


def _external_files(value, base):
    """Collect explicit source receipts, including nested raw extraction inputs."""
    result = {}
    if isinstance(value, Mapping):
        if set(value) == {'path', 'sha256'}:
            path = Path(value['path'])
            actual = file_record(path if path.is_absolute() else base / path)
            if actual['sha256'] != value['sha256']:
                raise ValueError('signed-pattern upstream receipt changed: ' + str(path))
            result[actual['path']] = actual['sha256']
        else:
            for item in value.values():
                result.update(_external_files(item, base))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.update(_external_files(item, base))
    return result


def _validate_candidate(candidate, *, fit_row_ids, control_row_ids, labels):
    p = candidate.fit_provenance
    rows = p.get('label_fit_row_ids')
    if p.get('role') != 'fit' or not isinstance(rows, list) or not rows:
        raise ValueError('signed targets require explicit nonempty fit-only provenance')
    _rows(rows, len(rows))
    known, control = set(map(canonical, fit_row_ids)), set(map(canonical, control_row_ids))
    if not set(map(canonical, rows)).issubset(known) or set(map(canonical, rows)) & control:
        raise ValueError('signed target fitting must use only declared source_fit rows')
    _array_record(p.get('fit_truth'), (len(rows), len(labels)), 'fit_truth')
    if type(p.get('fit_group_feasible', True)) is not bool:
        raise ValueError('fit_group_feasible must be boolean')
    masks = p.get('pattern_masks')
    if not isinstance(masks, dict) or set(masks) != {s.key for s in candidate.pattern_specs}:
        raise ValueError('every exported PatternSpec requires fit/control mask identities')
    for record in masks.values():
        if not isinstance(record, dict) or set(record) != {'fit', 'control'}:
            raise ValueError('exact fit/control pattern mask records required')
        _array_record(record['fit'], (len(rows),), 'fit mask')
        _array_record(record['control'], (len(control_row_ids),), 'control mask')
    for rule in candidate.rules:
        for atom in rule.body + (rule.head,):
            if isinstance(atom, Label) and atom.label not in labels:
                raise ValueError('candidate label is outside fixed label axis')


def _array_record(value, shape, name):
    if (not isinstance(value, dict) or value.get('shape') != list(shape)
            or value.get('dtype') != 'bool' or not isinstance(value.get('sha256'), str)
            or len(value['sha256']) != 64 or any(c not in '0123456789abcdef' for c in value['sha256'])):
        raise ValueError(name + ' requires a complete boolean array identity on the declared axis')


def select_signed_patterns(candidates, *, context, baseline_predictions, input_binding,
                           protocol, output_dir, model_snapshot_id=None, background_rules=(), execution_backend="reference"):
    """Select best strict positive control marginal, rescoring all remaining groups.

    ``context`` carries full row_ids/row_roles/fit_indices/receiver_indices and
    control-only truth in addition to the complete typed runtime inputs.
    All input candidates are retained, including zero-effect and infeasible
    groups.  An existing output directory is never overwritten.
    """
    if not isinstance(protocol, SignedPatternSelectionProtocol):
        raise ValueError('explicit SignedPatternSelectionProtocol required')
    n, k = len(context['docs']), len(context['labels'])
    rows = _rows(context['row_ids'], n)
    fit = _indices(context['fit_indices'], n, 'fit_indices')
    control = _indices(context['receiver_indices'], n, 'receiver_indices')
    if np.intersect1d(fit, control).size:
        raise ValueError('fit and control axes must be disjoint')
    roles = list(context['row_roles'])
    if (len(roles) != n or set(np.flatnonzero(np.asarray(roles) == 'source_fit')) != set(fit)
            or set(np.flatnonzero(np.asarray(roles) == 'val_ctrl')) != set(control)):
        raise ValueError('complete source_fit and val_ctrl role axes must match declared indices')
    labels = list(context['labels'])
    if any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != k:
        raise ValueError('complete unique nonempty label strings required')
    if (not isinstance(input_binding, Mapping) or input_binding.get('role') != 'control'
            or type(input_binding.get('expected_rows')) is not int or input_binding['expected_rows'] != len(control)):
        raise ValueError('explicit complete control role and row count required')
    gamma = context['gamma']
    if not isinstance(gamma, (GammaSeed, RefGamma)) or gamma.shape != (n, k):
        raise ValueError('explicit full-axis Gamma required')
    allowed = np.zeros(n, bool); allowed[fit] = True
    if (np.any(gamma.pos[~allowed]) or np.any(gamma.neg[~allowed])
            or any(not allowed[a] or not allowed[b] for field in ('subset', 'strict_subset')
                   for a, b in getattr(gamma, field, ()))):
        raise ValueError('selection Gamma may contain only declared source_fit roots; never B or control truth')
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    evaluator = make_signed_evaluator(execution_backend, context=context, baseline_predictions=baseline_predictions,
        score_indices=control, truth=context['truth'], cache_dir=directory / 'chase_cache',
        model_snapshot_id=model_snapshot_id, background_rules=background_rules)
    source_files = _source_files()
    source_files.update(evaluator.binding['runtime']['code_files'])
    binding = {'schema': SCHEMA, 'protocol': asdict(protocol), 'input_binding': dict(input_binding),
        'rows': [rows[i] for i in control], 'control_row_ids': [rows[i] for i in control],
        'full_row_ids': rows, 'row_roles': roles, 'fit_indices': fit.tolist(), 'control_indices': control.tolist(),
        'labels': labels, 'truth': array_identity(evaluator.truth), 'baseline': array_identity(evaluator.baseline),
        'context': evaluator.binding, 'source_files': source_files,
        'method': 'SignedChase', 'order': 'best strict marginal; original ordinal breaks exact ties',
        'rescore': 'every remaining group after every accepted group', 'conflicts': 'whole proposal infeasible; score null',
        'correction': CORRECTION, 'control_rows_dropped': 0, 'baseline_or_truth_added_to_gamma': False}
    write_json(directory / 'PROTOCOL.json', binding, exclusive=True)
    raw = []
    seen = set()
    fit_binding_context = {**context, **evaluator._context, 'row_ids': rows, 'row_roles': roles,
                           'fit_indices': fit, 'receiver_indices': control}
    for supplied in candidates:
        if not isinstance(supplied, SignedPatternCandidate):
            raise ValueError('all raw items must be SignedPatternCandidate')
        candidate = SignedPatternCandidate.from_dict(supplied.to_dict())
        if candidate.key in seen:
            raise ValueError('candidate keys must be unique')
        seen.add(candidate.key)
        _validate_candidate(candidate, fit_row_ids=[rows[i] for i in fit],
                            control_row_ids=binding['rows'], labels=labels)
        from .signed_candidate_generation import validate_signed_candidate_fit_binding
        validate_signed_candidate_fit_binding(candidate, context=fit_binding_context,
            baseline_predictions=evaluator.baseline, background_rules=evaluator.background_rules)
        raw.append(candidate)
    write_json(directory / 'CANDIDATES.json', [c.to_dict() for c in raw], exclusive=True)
    if len(raw) != protocol.expected_candidates:
        raise ValueError('complete expected candidate count differs from supplied raw stream')
    base = Path(input_binding['manifest']['path']).parent if 'manifest' in input_binding else Path.cwd()
    external_files = _external_files([binding['input_binding'], [c.fit_provenance for c in raw]], base)
    initial = evaluator.evaluate(())
    write_json(directory / 'BASELINE.json', initial.record(), exclusive=True)
    if not initial.feasible:
        raise ValueError('fixed background/Gamma is infeasible; no pattern selection can be completed')
    accepted, remaining, rounds = [], list(range(len(raw))), []
    current = initial
    while remaining:
        _verify_files(source_files)
        entries, best_ordinal, best_eval, best_gain = [], None, None, protocol.min_gain
        selected_rules = tuple(rule for ordinal in accepted for rule in raw[ordinal].rules)
        for ordinal in remaining:
            candidate = raw[ordinal]
            evaluated = evaluator.evaluate(selected_rules + candidate.rules,
                                           previous_predictions=current.corrected_predictions)
            fit_feasible = candidate.fit_provenance.get('fit_group_feasible', True)
            eligible = fit_feasible and evaluated.feasible
            gain = evaluated.macro_f1 - current.macro_f1 if evaluated.feasible else None
            entry = {'ordinal': ordinal, 'key': candidate.key, **evaluated.record(),
                     'gain': gain, 'eligible': eligible, 'fit_group_feasible': fit_feasible}
            entries.append(entry)
            if eligible and gain > best_gain:
                best_ordinal, best_eval, best_gain = ordinal, evaluated, gain
        rounds.append({'round': len(rounds), 'current_score': current.macro_f1,
                       'evaluations': entries, 'accepted_ordinal': best_ordinal})
        write_json(directory / 'ROUNDS.json', rounds)
        if best_ordinal is None:
            break
        accepted.append(best_ordinal)
        remaining.remove(best_ordinal)
        current = best_eval
    if not rounds:
        write_json(directory / 'ROUNDS.json', [], exclusive=True)
    final_reference_check = None
    if hasattr(evaluator, "verify_final"):
        final_reference_check = evaluator.verify_final(tuple(rule for ordinal in accepted for rule in raw[ordinal].rules), current)
    specs, used = [], {}
    for ordinal in accepted:
        for spec in raw[ordinal].pattern_specs:
            encoded = spec.to_dict()
            if spec.key in used and used[spec.key] != encoded:
                raise ValueError('selected groups reuse a PatternSpec key with different semantics')
            if spec.key not in used:
                used[spec.key] = encoded; specs.append(encoded)
    result = {'schema': SCHEMA, 'protocol_id': digest(binding), 'method': 'SignedChase',
        'raw_candidates': len(raw), 'selected_ordinals': accepted,
        'selected_keys': [raw[i].key for i in accepted], 'selected_pattern_specs': specs,
        'selected_rules': [r.to_dict() for i in accepted for r in raw[i].rules],
        'background_rules': evaluator.binding['background_rules'], 'initial_score': initial.macro_f1,
        'final_score': current.macro_f1, 'baseline_raw_score': macro_f1(evaluator.truth, evaluator.baseline[control]),
        'diagnostics': correction_diagnostics(evaluator.truth, evaluator.baseline[control], current.corrected_predictions[control]),
        'final_state_identity': current.state_identity, 'final_chase_receipt': current.chase_receipt,
        'full_positive_facts': array_identity(current.positive_facts), 'full_negative_facts': array_identity(current.negative_facts),
        'full_corrected_predictions': array_identity(current.corrected_predictions), 'round_count': len(rounds),
        'control_rows_dropped': 0, 'selection_complete_claim': True,
        'claim_scope': 'pattern selection completed; candidate rule groups are not the final RLearner Sigma',
        'execution_backend': execution_backend, 'final_reference_check': final_reference_check}
    write_json(directory / 'SELECTED.json', result, exclusive=True)
    with (directory / 'FINAL.npz').open('xb') as stream:
        np.savez_compressed(stream, baseline=evaluator.baseline, positive_facts=current.positive_facts,
            negative_facts=current.negative_facts, corrected_predictions=current.corrected_predictions,
            control_indices=control, truth=evaluator.truth)
    _verify_files(source_files)
    _verify_files(external_files)
    names = ('PROTOCOL.json', 'CANDIDATES.json', 'BASELINE.json', 'ROUNDS.json', 'SELECTED.json', 'FINAL.npz')
    write_json(directory / 'COMPLETE.json', {'schema': SCHEMA, 'protocol_id': digest(binding),
        'selected_artifact_sha256': digest(result),
        'external_files': external_files,
        'files': {name: file_record(directory / name)['sha256'] for name in names}}, exclusive=True)
    return result


def verify_signed_pattern_output(directory):
    """Authenticate files, current semantic source, axes and complete decisions.

    This validates the stored selection trace; callers binding another corpus
    must additionally replay primitive masks and Chase against their own inputs.
    No incomplete selection is exported as a completed pattern vocabulary.
    """
    directory = Path(directory).resolve(strict=True)
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    names = {'PROTOCOL.json', 'CANDIDATES.json', 'BASELINE.json', 'ROUNDS.json', 'SELECTED.json', 'FINAL.npz'}
    if complete.get('schema') != SCHEMA or set(complete.get('files', {})) != names:
        raise ValueError('complete signed-pattern artifact inventory required')
    files = {str(directory / name): checksum for name, checksum in complete['files'].items()}
    _verify_files(files)
    protocol = json.loads((directory / 'PROTOCOL.json').read_text())
    result = json.loads((directory / 'SELECTED.json').read_text())
    initial = json.loads((directory / 'BASELINE.json').read_text())
    rounds = json.loads((directory / 'ROUNDS.json').read_text())
    if (protocol.get('schema') != SCHEMA or result.get('schema') != SCHEMA
            or complete.get('protocol_id') != digest(protocol) or result.get('protocol_id') != digest(protocol)
            or complete.get('selected_artifact_sha256') != digest(result)):
        raise ValueError('signed-pattern protocol/result binding mismatch')
    if not isinstance(protocol.get('source_files'), dict) or not _source_files().items() <= protocol['source_files'].items():
        raise ValueError('signed-pattern producer source is not current and complete')
    _verify_files(protocol['source_files'])
    _verify_files(protocol['context']['runtime']['input_files'])
    if not protocol['context']['runtime']['code_files'].items() <= protocol['source_files'].items():
        raise ValueError('runtime semantic dependencies are not source-bound')
    raw = tuple(SignedPatternCandidate.from_dict(c) for c in json.loads((directory / 'CANDIDATES.json').read_text()))
    ib = protocol['input_binding']
    base = Path(ib['manifest']['path']).parent if 'manifest' in ib else Path.cwd()
    external_files = _external_files([ib, [c.fit_provenance for c in raw]], base)
    if complete.get('external_files') != external_files:
        raise ValueError('complete upstream source inventory mismatch')
    p = SignedPatternSelectionProtocol(**protocol['protocol'])
    if len(raw) != p.expected_candidates or len({c.key for c in raw}) != len(raw) or result.get('raw_candidates') != len(raw):
        raise ValueError('raw candidate inventory is incomplete or duplicated')
    full_rows = _rows(protocol['full_row_ids'], len(protocol['full_row_ids']))
    fit = _indices(protocol['fit_indices'], len(full_rows), 'fit_indices')
    control = _indices(protocol['control_indices'], len(full_rows), 'control_indices')
    rows = [full_rows[i] for i in control]
    if rows != protocol['rows'] or rows != protocol['control_row_ids'] or np.intersect1d(fit, control).size:
        raise ValueError('signed-pattern row-axis binding mismatch')
    shape = (len(full_rows), len(protocol['labels']))
    with np.load(directory / 'FINAL.npz', allow_pickle=False) as stored:
        if set(stored.files) != {'baseline', 'positive_facts', 'negative_facts', 'corrected_predictions', 'control_indices', 'truth'}:
            raise ValueError('complete final signed arrays required')
        final_arrays = {key: stored[key] for key in stored.files}
    baseline = _binary(final_arrays['baseline'], shape)
    truth = _binary(final_arrays['truth'], (len(control), shape[1]))
    if (not np.array_equal(final_arrays['control_indices'], control)
            or array_identity(baseline) != protocol['baseline'] or array_identity(truth) != protocol['truth']):
        raise ValueError('signed evaluation requires the authenticated complete baseline and fixed control truth')
    for candidate in raw:
        _validate_candidate(candidate, fit_row_ids=[full_rows[i] for i in fit], control_row_ids=rows, labels=protocol['labels'])
    accepted, remaining, current = [], list(range(len(raw))), initial.get('score')
    last_accepted_evaluation = initial
    if initial.get('feasible') is not True or initial.get('status') != 'fixpoint' or not _score(current):
        raise ValueError('fixed-background evaluation must be complete and feasible')
    _verify_program_receipt(initial, protocol, (), raw)
    current_predictions = _verify_success_evaluation(initial, baseline, truth, control, baseline)
    for round_id, item in enumerate(rounds):
        if item.get('round') != round_id or item.get('current_score') != current:
            raise ValueError('round index/current score mismatch')
        entries = item.get('evaluations', [])
        if [entry.get('ordinal') for entry in entries] != remaining:
            raise ValueError('every remaining candidate must be rescored in original order')
        best, gain, best_predictions = None, p.min_gain, None
        for entry in entries:
            ordinal = entry['ordinal']
            fit_feasible = raw[ordinal].fit_provenance.get('fit_group_feasible', True)
            if (entry.get('key') != raw[ordinal].key or entry.get('fit_group_feasible') != fit_feasible
                    or entry.get('eligible') != (fit_feasible and entry.get('feasible') is True)):
                raise ValueError('evaluation/candidate fit eligibility mismatch')
            _verify_program_receipt(entry, protocol, tuple(accepted) + (ordinal,), raw)
            if entry.get('feasible') is True:
                evaluated_predictions = _verify_success_evaluation(
                    entry, baseline, truth, control, current_predictions)
                if (entry.get('status') != 'fixpoint' or not _score(entry.get('score'))
                        or entry.get('gain') != entry['score'] - current
                        or entry.get('diagnostics', {}).get('scored_rows') != len(control)
                        or entry.get('diagnostics', {}).get('scored_labels') != len(protocol['labels'])):
                    raise ValueError('feasible candidate score/gain/fixed-axis diagnostics mismatch')
                if entry['eligible'] and entry['gain'] > gain:
                    best, gain = ordinal, entry['gain']
                    best_predictions = evaluated_predictions
            elif (entry.get('status') != 'conflict' or entry.get('score') is not None
                  or entry.get('gain') is not None or entry.get('diagnostics') is not None):
                raise ValueError('a conflict must reject the entire proposal without a score')
        if item.get('accepted_ordinal') != best:
            raise ValueError('selection is not the best strict marginal with stable ordinal ties')
        if best is None:
            if round_id != len(rounds) - 1:
                raise ValueError('selection must stop when no strict positive eligible gain remains')
        else:
            last_accepted_evaluation = next(e for e in entries if e['ordinal'] == best)
            current = last_accepted_evaluation['score']
            current_predictions = best_predictions
            accepted.append(best); remaining.remove(best)
    if remaining and (not rounds or rounds[-1]['accepted_ordinal'] is not None):
        raise ValueError('incomplete selection: remaining candidates lack a terminal rescore')
    selected = tuple(raw[i] for i in accepted)
    specs = {}
    for candidate in selected:
        for spec in candidate.pattern_specs:
            value = spec.to_dict()
            if spec.key in specs and specs[spec.key] != value:
                raise ValueError('selected PatternSpec key collision')
            specs.setdefault(spec.key, value)
    if (result.get('selected_ordinals') != accepted or result.get('selected_keys') != [c.key for c in selected]
            or result.get('selected_pattern_specs') != list(specs.values())
            or result.get('selected_rules') != [r.to_dict() for c in selected for r in c.rules]
            or result.get('background_rules') != protocol['context']['background_rules']
            or result.get('initial_score') != initial['score'] or result.get('final_score') != current
            or result.get('final_state_identity') != last_accepted_evaluation['state_identity']
            or result.get('final_chase_receipt') != last_accepted_evaluation['chase_receipt']
            or result.get('round_count') != len(rounds) or result.get('control_rows_dropped') != 0
            or result.get('selection_complete_claim') is not True):
        raise ValueError('selected result does not match complete raw selection trace')
    _verify_program_receipt({'chase_receipt': result['final_chase_receipt'],
                             'state_identity': result['final_state_identity']}, protocol, accepted, raw)
    if protocol['context'].get('execution_backend') == 'analytic_unary_v1':
        check = result.get('final_reference_check', {})
        if check.get('facts_and_score_equal') is not True or check.get('final_real_Chase_calls') != 1:
            raise ValueError('analytic selected group requires final complete real-Chase comparison')
        actual_state = check.get('real_state_identity', {})
        _verify_program_receipt({'chase_receipt': check.get('real_chase_receipt', {}),
                                 'state_identity': actual_state}, protocol, accepted, raw)
        if (actual_state.get('status') != 'fixpoint'
                or actual_state.get('positive') != result['full_positive_facts']
                or actual_state.get('negative') != result['full_negative_facts']):
            raise ValueError('real-Chase final facts differ from analytic selected group')
    positive = _binary(final_arrays['positive_facts'], shape)
    negative = _binary(final_arrays['negative_facts'], shape)
    corrected = _binary(final_arrays['corrected_predictions'], shape)
    if (np.any(positive & negative) or not np.array_equal(corrected, (baseline & ~negative) | positive)
            or not np.array_equal(corrected, current_predictions)
            or array_identity(positive) != result['full_positive_facts']
            or array_identity(negative) != result['full_negative_facts']
            or array_identity(positive) != result['final_state_identity']['positive']
            or array_identity(negative) != result['final_state_identity']['negative']
            or array_identity(corrected) != result['full_corrected_predictions']
            or macro_f1(truth, corrected[control]) != current
            or macro_f1(truth, baseline[control]) != result['baseline_raw_score']
            or correction_diagnostics(truth, baseline[control], corrected[control]) != result['diagnostics']):
        raise ValueError('final signed facts, predictions or fixed-control score mismatch')
    files.update(protocol['source_files'])
    files.update(protocol['context']['runtime']['input_files'])
    files.update(external_files)
    files[str(directory / 'COMPLETE.json')] = file_record(directory / 'COMPLETE.json')['sha256']
    _verify_files(files)
    return {'protocol': protocol, 'result': result, 'candidates': raw,
            'selected_candidates': selected, 'files': files}


def _score(value):
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1


def _verify_success_evaluation(evaluation, baseline, truth, control, previous):
    positive, negative = _read_signed_facts(evaluation.get('signed_facts'), baseline.shape)
    state = evaluation['state_identity']
    if (np.any(positive & negative) or state['label_conflicts'] or state['comparison_conflicts']
            or array_identity(positive) != state['positive'] or array_identity(negative) != state['negative']
            or array_identity((positive & ~negative).astype(np.float32)) != state['predictions']):
        raise ValueError('successful signed-fact bitmaps differ from the actual complete Chase state')
    corrected = (baseline & ~negative) | positive
    if (evaluation.get('score') != macro_f1(truth, corrected[control])
            or evaluation.get('diagnostics') != correction_diagnostics(truth, previous[control], corrected[control])):
        raise ValueError('successful evaluation score/diagnostics differ from its complete signed facts')
    return corrected


def _verify_program_receipt(evaluation, protocol, ordinals, raw):
    receipt = evaluation.get('chase_receipt', {})
    program = protocol['context']['background_rules'] + [r.to_dict() for i in ordinals for r in raw[i].rules]
    if receipt.get('schema') == 'paper36_exact_unary_signed_evaluation_v1':
        from .analytic_unary_patterns import verify_analytic_receipt
        verify_analytic_receipt(evaluation, protocol, program)
        return
    expected = digest(program)
    calls = receipt.get('calls', [])
    state = evaluation.get('state_identity', {})
    if state.get('sha256') != digest({key: value for key, value in state.items() if key != 'sha256'}):
        raise ValueError('complete Chase state identity is inconsistent')
    if (receipt.get('binding', {}).get('Sigma') != expected or len(calls) != 1
            or calls[0].get('Sigma') != expected or calls[0].get('context_id') != protocol['context']['context_id']
            or calls[0].get('theta') != protocol['context']['model_snapshot_id']
            or calls[0].get('seed_sha256') != digest(protocol['context']['gamma'])
            or calls[0].get('complete_state_sha256') != evaluation.get('state_identity', {}).get('sha256')
            or calls[0].get('validated_Gamma_unioned') is not False
            or calls[0].get('status') != state.get('status')
            or evaluation.get('status', state.get('status')) != state.get('status')):
        raise ValueError('complete real-Chase receipt does not bind the exact proposed rule group and frozen seed')
    if state.get('status') == 'conflict':
        _verify_conflict_certificate(evaluation.get('conflict_certificate'), state, calls[0], program,
                                     protocol['labels'], protocol['context']['gamma'])
    elif state.get('status') != 'fixpoint' or evaluation.get('conflict_certificate') is not None:
        raise ValueError('successful complete Chase must have no Conflict certificate')


def _verify_conflict_certificate(certificate, state, call, program, labels, gamma):
    """Authenticate readable witnesses against the same complete Chase call.

    This checks complete recorded facts and proof structure, not the primitive
    text/ML predicates of another corpus; those still require the documented
    context replay gate when the artifact is consumed with new inputs.
    """
    if (not isinstance(certificate, dict)
            or certificate.get('schema') != 'paper36_outer_conflict_certificate_v1'
            or certificate.get('conflict_id') != digest({key: value for key, value in certificate.items()
                                                        if key != 'conflict_id'})
            or certificate.get('outcome') != 'Conflict' or certificate.get('failed_state') != state
            or certificate.get('Sigma') != call['Sigma'] or certificate.get('rules') != program
            or certificate.get('theta') != call['theta'] or certificate.get('context_id') != call['context_id']
            or certificate.get('labels') != labels or certificate.get('candidate_roots') != gamma
            or certificate.get('last_rule_presumed_wrong') is not False
            or certificate.get('confidence_based_repair') is not False
            or certificate.get('failed_derived_facts_committed') is not False):
        raise ValueError('Conflict certificate must bind the same rules, roots and failed Chase state')
    complete = certificate.get('complete_proof_dag')
    branches = certificate.get('proof_nodes')
    sources = certificate.get('root_sources')
    if (not isinstance(complete, dict) or not isinstance(branches, dict) or not isinstance(sources, dict)
            or ProofDAG(complete).identity() != state.get('proof_dag')
            or digest(sources) != call.get('root_sources_sha256')
            or any(complete.get(key) != node for key, node in branches.items())):
        raise ValueError('Conflict requires complete authentic proof records and both retained branches')
    n, k = state['positive']['shape']
    signed = {sign: np.zeros((n, k), bool) for sign in ('pos', 'neg')}
    # Reconstruct only facts present in the certificate. In particular, do
    # not pre-add identity edges; use empty adaptive rows until proved.
    relations = {kind: RelationRows(n) for kind in ('subset', 'strict_subset')}
    root_signed = {sign: np.zeros((n, k), bool) for sign in ('pos', 'neg')}
    root_relations = {kind: set() for kind in relations}
    for key, node in complete.items():
        try:
            kind, row, column = key.split(':')
            row, column = int(row), int(column)
        except (AttributeError, ValueError):
            raise ValueError('Conflict proof fact keys must identify signed labels or relations') from None
        if (kind not in (*signed, *relations) or not 0 <= row < n
                or not 0 <= column < (k if kind in signed else n) or not isinstance(node, dict)
                or (node.get('kind'), node.get('row'), node.get('column')) != (kind, row, column)
                or not isinstance(node.get('dependencies'), list)
                or any(dependency not in complete for dependency in node['dependencies'])):
            raise ValueError('Conflict proof fact axes or premise references are invalid')
        if kind in signed:
            signed[kind][row, column] = True
        else:
            relations[kind][row] |= 1 << column
        if node.get('origin') == 'rule_application':
            rule_id = node.get('rule_id')
            if type(rule_id) is not int or not 0 <= rule_id < len(program):
                raise ValueError('Conflict derivation must name a rule in this exact proposal')
        if key in sources:
            if sources[key] != {'origin': 'validated_Gamma'} or node.get('root_source') != sources[key]:
                raise ValueError('Conflict root testimony differs from the actual explicit Gamma source')
            if kind in signed:
                root_signed[kind][row, column] = True
            else:
                root_relations[kind].add((row, column))
    if not set(sources).issubset(complete):
        raise ValueError('Conflict proof dropped an explicit root source')
    if (array_identity(signed['pos']) != state['positive'] or array_identity(signed['neg']) != state['negative']
            or any(RelationFacts(rows).identity() != state[kind] for kind, rows in relations.items())
            or array_identity(root_signed['pos']) != gamma['positive']
            or array_identity(root_signed['neg']) != gamma['negative']
            or any(root_relations[kind] != set(map(tuple, gamma[kind])) for kind in relations)):
        raise ValueError('Conflict actual proof facts do not reconstruct the recorded state and Gamma')
    collisions = [{'document_index': int(row), 'label_index': int(label), 'label': labels[int(label)],
                   'positive_source': 'pos:%d:%d' % (row, label), 'negative_source': 'neg:%d:%d' % (row, label)}
                  for row, label in np.argwhere(signed['pos'] & signed['neg'])]
    if (certificate.get('collisions') != collisions
            or state['label_conflicts'] != [[item['document_index'], item['label']] for item in collisions]
            or certificate.get('comparison_conflicts') != state['comparison_conflicts']
            or not collisions and not state['comparison_conflicts']):
        raise ValueError('Conflict must retain its actual signed collision or relation infeasibility')
    needed = [item[sign] for item in collisions for sign in ('positive_source', 'negative_source')]
    if state['comparison_conflicts']:
        needed.extend(complete)
    visited = set()
    while needed:
        key = needed.pop()
        if key in visited:
            continue
        if key not in branches:
            raise ValueError('Conflict is missing a positive/negative source or a supporting premise')
        visited.add(key)
        needed.extend(branches[key]['dependencies'])


def export_selected_signed_pattern_specs(directory):
    checked = verify_signed_pattern_output(directory)
    return tuple(_pattern_spec(raw, raw['key']) for raw in checked['result']['selected_pattern_specs'])
