"""Fit signed rule groups from every raw textual instance, without control gold.

Each raw pattern remains one candidate. For each label, both signed heads are
tried with the fixed background and a complete Chase; the best strictly useful
fit direction is frozen. Their joint rule group is also chased, since separate
feasible heads need not remain feasible together. No raw instance is pruned.
"""
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from loris._engine.rdl_reference import Doc, Label, RefGamma, ReferenceChase, Rule, Text, prepare_attributes
from loris._engine.rlearner import array_identity, canonical, digest
from loris._engine.hashing import array_hash
from .artifacts import bound_path, file_record
from .candidate_generation import exact_pattern_mask, pattern_instances
from .patterns import _binary, _pattern_spec, _rows
from .semantic_patterns import (
    SCHEMA as RAW_SCHEMA, _axes, _domain, _integer_axis, _read, _readonly, _unpack)
from .signed_patterns import SignedChaseEvaluator, SignedPatternCandidate, signed_context_identity, _array_record


SCHEMA = 'paper36_signed_pattern_fit_generation_v1'
GAIN_TOLERANCE = 0.


def _candidate_axes(context, fit_indices, control_indices):
    n = len(context['docs'])
    rows = _rows(context['row_ids'], n)
    roles = list(context['row_roles'])
    fit = _integer_axis(fit_indices, 'signed pattern fit axis')
    control = _integer_axis(control_indices, 'signed pattern control axis')
    if (len(roles) != n or not len(fit) or not len(control)
            or np.any(fit >= n) or np.any(control >= n)
            or len(set(fit.tolist())) != len(fit)
            or len(set(control.tolist())) != len(control)
            or set(fit.tolist()) & set(control.tolist())
            or any(roles[i] not in ('fit', 'source_fit') for i in fit)
            or any(roles[i] not in ('control', 'val_ctrl') for i in control)):
        raise ValueError('disjoint complete fit/control scoring axes with matching roles required')
    return fit, control, rows


def fit_context_without_scored_roots(context, fit_indices):
    """Copy Gamma while hiding scored fit labels and incident relation roots.

    Background rules and other documents' legitimate roots remain available.
    This does not mutate the labeling/discovery context or add model B to Gamma.
    """
    fit = _integer_axis(fit_indices, 'signed pattern fit axis')
    n, k = len(context['docs']), len(context['labels'])
    if not len(fit) or np.any(fit >= n) or len(set(fit.tolist())) != len(fit):
        raise ValueError('unique in-range nonempty fit scoring rows required')
    source = context['gamma']
    if not hasattr(source, 'pos') or not hasattr(source, 'neg'):
        raise ValueError('explicit positive and negative Gamma arrays required')
    original = RefGamma(_binary(source.pos, (n, k)), _binary(source.neg, (n, k)),
        tuple(getattr(source, 'subset', ())), tuple(getattr(source, 'strict_subset', ())))
    if 'row_roles' in context:
        roles = list(context['row_roles'])
        if len(roles) != n:
            raise ValueError('complete row roles required for fit Gamma isolation')
        allowed = np.array([role in ('fit', 'source_fit') for role in roles], bool)
        if (np.any(original.pos[~allowed]) or np.any(original.neg[~allowed])
                or any(not allowed[a] or not allowed[b]
                       for edge_set in (original.subset, original.strict_subset) for a, b in edge_set)):
            raise ValueError('fit candidate Gamma may contain only source_fit roots; no control or test roots')
    pos, neg = original.pos.copy(), original.neg.copy()
    removed_pos, removed_neg = int(pos[fit].sum()), int(neg[fit].sum())
    pos[fit] = False
    neg[fit] = False
    scored = set(fit.tolist())
    subset = tuple(edge for edge in original.subset if not scored.intersection(edge))
    strict = tuple(edge for edge in original.strict_subset if not scored.intersection(edge))
    clean = RefGamma(pos, neg, subset, strict)
    copied = dict(context)
    # Scorers receive fit truth separately; the context's usual `truth` is the
    # control truth and must never become a fitting input.
    copied.pop('truth', None)
    copied['gamma'] = clean
    policy = {'policy': 'remove_scored_fit_pos_neg_and_incident_relation_roots',
        'scored_fit_indices': fit.tolist(),
        'removed_positive_roots': removed_pos, 'removed_negative_roots': removed_neg,
        'removed_subset_roots': len(original.subset) - len(subset),
        'removed_strict_subset_roots': len(original.strict_subset) - len(strict),
        'remaining_gamma': {'positive': array_identity(clean.pos), 'negative': array_identity(clean.neg),
            'subset': [list(edge) for edge in subset], 'strict_subset': [list(edge) for edge in strict]},
        'other_legitimate_roots_retained': True, 'baseline_injected_as_gamma': False,
        'root_roles': 'source_fit or fit only when row_roles are declared',
        'background_rules_recomputed': True}
    return copied, policy


def validate_signed_candidate_fit_binding(candidate, *, context, baseline_predictions, background_rules=()):
    """Check a fitted group's actual input identities without Chase or training.

    Both automatic and externally supplied groups require the complete binding
    of their fit evaluator.  The CLI separately authenticates the actual fit
    truth array and cluster row list; this function cross-checks that declared
    truth identity against the fit evaluator and the current corpus/B/Gamma.
    It does not refit external ML/cross-document rules or infer truth from hashes.
    """
    if not isinstance(candidate, SignedPatternCandidate):
        raise ValueError('fit binding requires a SignedPatternCandidate')
    p = candidate.fit_provenance
    required = {'role', 'label_fit_row_ids', 'fit_truth', 'fit_baseline', 'full_baseline',
                'fit_seed_policy', 'fit_evaluator_binding', 'pattern_masks'}
    if not isinstance(p, Mapping) or not required.issubset(p) or p['role'] != 'fit':
        raise ValueError('signed candidate requires a complete fit binding, including B and sanitized context')
    n, k = len(context['docs']), len(context['labels'])
    rows = _rows(context['row_ids'], n)
    fit_rows = _rows(p['label_fit_row_ids'], len(p['label_fit_row_ids']))
    positions = {canonical(row): i for i, row in enumerate(rows)}
    if not fit_rows or any(canonical(row) not in positions for row in fit_rows):
        raise ValueError('fit binding row identity differs from the actual complete corpus')
    fit = np.asarray([positions[canonical(row)] for row in fit_rows], dtype=np.int64)
    control = _integer_axis(context['receiver_indices'], 'fit binding control axis')
    fit, control, _ = _candidate_axes(context, fit, control)
    all_fit = _integer_axis(context['fit_indices'], 'fit binding full source_fit axis')
    if not set(fit.tolist()).issubset(set(all_fit.tolist())):
        raise ValueError('fit binding uses rows outside the declared source_fit universe')
    baseline = _binary(baseline_predictions, (n, k))
    _array_record(p['fit_truth'], (len(fit), k), 'fit binding truth')
    clean, policy = fit_context_without_scored_roots(context, fit)
    expected_identity = signed_context_identity(clean)
    background = [r.to_dict() if isinstance(r, Rule) else Rule.from_dict(r).to_dict() for r in background_rules]
    binding = p['fit_evaluator_binding']
    expected_snapshot = {
        'probabilities': {alias: array_identity(np.asarray(values)) for alias, values in
                          context.get('model_probabilities', context.get('scores', {})).items()},
        'decisions': {alias: array_identity(np.asarray(values)) for alias, values in context.get('model_decisions', {}).items()},
        'input_files': dict(context.get('input_files', {}))}
    if (p['full_baseline'] != array_identity(baseline) or p['fit_baseline'] != array_identity(baseline[fit])
            or p['fit_seed_policy'] != policy or not isinstance(binding, Mapping)
            or binding.get('baseline') != array_identity(baseline)
            or binding.get('truth') != p['fit_truth'] or binding.get('score_indices') != fit.tolist()
            or binding.get('labels') != list(context['labels']) or binding.get('identity') != expected_identity
            or binding.get('gamma') != policy['remaining_gamma']
            or binding.get('background_rules') != background or binding.get('model_snapshot') != expected_snapshot
            or not isinstance(binding.get('model_snapshot_id'), str) or not binding['model_snapshot_id']):
        raise ValueError('signed candidate fit binding differs from current B, fit truth/rows, Gamma, models or background')
    # The human-readable semantic identity and the actual runtime receipt must
    # describe the same tables, not merely carry matching context-ID strings.
    runtime = binding.get('runtime')
    reference = ReferenceChase(context['labels'],
        model_probabilities=context.get('model_probabilities', context.get('scores', {})),
        model_decisions=context.get('model_decisions', {}), subset_direction='standard',
        attribute_values=context.get('attribute_values', {}), attribute_presence=context.get('attribute_presence', {}))
    values, presence = prepare_attributes(context['docs'], reference.attribute_values, reference.attribute_presence)
    expected_runtime = {key: expected_identity[key] for key in
                        ('full_rows', 'labels', 'subset_direction', 'documents', 'attribute_values', 'source_context_id', 'input_files')}
    expected_runtime.update(
        probabilities={alias: array_hash(table) for alias, table in reference.model_probabilities.items()},
        decisions={alias: array_hash(table) for alias, table in reference.model_decisions.items()},
        attribute_presence={name: array_hash(mask) for name, mask in presence.items()})
    if (not isinstance(runtime, Mapping) or binding.get('context_id') != digest(runtime)
            or any(runtime.get(key) != value for key, value in expected_runtime.items())):
        raise ValueError('fit runtime receipt differs from the actual complete context interpretation')
    masks = p['pattern_masks']
    if not isinstance(masks, Mapping) or set(masks) != {s.key for s in candidate.pattern_specs}:
        raise ValueError('fit binding requires actual fit/control masks for every PatternSpec')
    for spec in candidate.pattern_specs:
        mask = exact_pattern_mask(spec.predicate, context['docs'], context.get('attribute_presence'))
        expected = {'fit': array_identity(mask[fit]), 'control': array_identity(mask[control])}
        if masks[spec.key] != expected:
            raise ValueError('fit binding pattern masks differ from the actual document/attribute interpretation')
    return {'fit_indices': fit.tolist(), 'fit_truth': dict(p['fit_truth']),
            'full_baseline': array_identity(baseline), 'fit_context_identity': expected_identity}


def _evaluation_record(evaluation, baseline_score):
    record = evaluation.record()
    record.update(macro_f1=evaluation.macro_f1,
        gain=None if not evaluation.feasible else float(evaluation.macro_f1 - baseline_score))
    return record


def _fit_signed_groups(instances, *, context, fit_indices, control_indices,
                       fit_truth, baseline_predictions, background_rules,
                       cache_dir, model_snapshot_id, execution_backend="reference"):
    fit, control, rows = _candidate_axes(context, fit_indices, control_indices)
    labels = tuple(context['labels'])
    truth = _binary(fit_truth, (len(fit), len(labels)))
    baseline = _binary(baseline_predictions, (len(rows), len(labels)))
    fit_context, seed_policy = fit_context_without_scored_roots(context, fit)
    generator_source = file_record(__file__)
    from .signed_patterns import make_signed_evaluator
    evaluator = make_signed_evaluator(execution_backend, context=fit_context, baseline_predictions=baseline,
        score_indices=fit, truth=truth, cache_dir=cache_dir,
        model_snapshot_id=model_snapshot_id, background_rules=tuple(background_rules))
    try:
        initial = evaluator.evaluate(())
        if not initial.feasible:
            raise ValueError('fixed background and sanitized fit Gamma have a Chase conflict')
        seen = set()
        for spec, fit_mask, control_mask, raw_provenance in instances:
            if spec.key in seen:
                raise ValueError('raw signed pattern keys must remain unique')
            seen.add(spec.key)
            fit_mask = _binary(fit_mask, (len(fit),))
            control_mask = _binary(control_mask, (len(control),))
            chosen, head_fit = [], []
            for label_index, label in enumerate(labels):
                best_rule, best_score, attempts = None, initial.macro_f1, []
                for polarity in ('pos', 'neg'):
                    rule = Rule((Doc('x'), Text('x', spec.predicate)), Label('x', label, polarity))
                    outcome = evaluator.evaluate((rule,), previous_predictions=initial.corrected_predictions)
                    attempts.append(dict(_evaluation_record(outcome, initial.macro_f1), polarity=polarity))
                    if outcome.feasible and outcome.macro_f1 > best_score + GAIN_TOLERANCE:
                        best_rule, best_score = rule, outcome.macro_f1
                if best_rule is not None:
                    chosen.append(best_rule)
                head_fit.append({'label_index': label_index, 'label': label,
                    'selected_polarity': None if best_rule is None else best_rule.head.polarity,
                    'attempts': attempts})
            combined = evaluator.evaluate(tuple(chosen), previous_predictions=initial.corrected_predictions)
            provenance = {'schema': SCHEMA, 'role': 'fit',
                'generator_source': generator_source,
                'label_fit_row_ids': [rows[i] for i in fit], 'fit_truth': array_identity(truth),
                'fit_baseline': array_identity(baseline[fit]),
                'full_baseline': array_identity(baseline), 'fit_seed_policy': seed_policy,
                'fit_evaluator_binding': evaluator.binding,
                'raw_candidate_key': spec.key, 'raw_extraction': dict(raw_provenance),
                'pattern_masks': {spec.key: {'fit': array_identity(fit_mask), 'control': array_identity(control_mask)}},
                'label_assignment': 'per label best strict positive full-fit Macro-F1 gain; positive before negative on ties',
                'fit_gain_tolerance': GAIN_TOLERANCE, 'head_fit': head_fit,
                'fit_background': _evaluation_record(initial, initial.macro_f1),
                'fit_group': _evaluation_record(combined, initial.macro_f1),
                'fit_group_feasible': bool(combined.feasible),
                'fit_group_has_heads': bool(chosen),
                'control_truth_used_for_fitting': False, 'all_raw_instances_retained': True,
                'frequency_instance_exact': True, 'legacy_coverage_entropy_prefilter': False,
                'missing_attribute_semantics': 'false, matching Chase primitive masks'}
            yield SignedPatternCandidate(spec.key, (spec,), tuple(chosen), provenance)
        bound_path(generator_source)
    finally:
        evaluator.close()
        if hasattr(instances, 'close'):
            instances.close()


def build_signed_pattern_candidates(templates, *, context, fit_indices, control_indices,
                                    fit_truth, baseline_predictions, cache_dir,
                                    input_provenance, background_rules=(), model_snapshot_id=None, execution_backend="reference"):
    """Expand the complete finite template stream and fit each signed rule group.

    ``baseline_predictions`` is the fixed full-corpus binary B, not Gamma.
    Fit and control truth are never concatenated; only fit truth is accepted.
    """
    if not isinstance(input_provenance, Mapping):
        raise ValueError('declare raw candidate domain provenance')
    unfiltered = input_provenance.get('raw_unfiltered_vocabulary') is True
    bounded_fit = (input_provenance.get('candidate_domain_policy') == 'fit_unigram_document_frequency_v1'
        and input_provenance.get('raw_unfiltered_vocabulary') is False
        and input_provenance.get('min_fit_document_support') == 5
        and input_provenance.get('max_fit_document_fraction') == .95
        and input_provenance.get('max_candidates_per_cluster') == 64
        and input_provenance.get('ordering') == 'descending fit document frequency then exact token string'
        and isinstance(input_provenance.get('full_fit_vocabulary'), Mapping)
        and isinstance(input_provenance.get('screening_receipt'), Mapping))
    bounded_semantic = (input_provenance.get('candidate_domain_policy') == 'fit_semantic_four_family_content_budget_v2'
        and input_provenance.get('raw_unfiltered_vocabulary') is False
        and input_provenance.get('min_fit_document_support') == 5
        and input_provenance.get('max_fit_document_fraction') == .95
        and input_provenance.get('max_candidates_per_cluster') == 64
        and input_provenance.get('per_family_budget') == 16
        and input_provenance.get('families') == ['Match','Cooccur','Before','Freq']
        and input_provenance.get('frequency_values') == [2]
        and input_provenance.get('frequency_ops') == ['>=']
        and input_provenance.get('parse_document_support_prefilter') == 5
        and input_provenance.get('anchor_validity') == 'unicode_surface_word_len_ge_2_alpha_not_spacy_stopwords'
        and input_provenance.get('ordering') == 'descending actual predicate fit document support then fixed PatternSpec key'
        and all(isinstance(input_provenance.get(key), Mapping) for key in
                ('full_semantic_domain','screening_complete','screening_receipt','validity_protocol','stop_words')))
    if not (unfiltered or bounded_fit or bounded_semantic):
        raise ValueError('declare unfiltered vocabulary or explicit author-approved fit-only candidate domain')
    if bounded_fit:
        bound_path(input_provenance['full_fit_vocabulary'])
        bound_path(input_provenance['screening_receipt'])
    if bounded_semantic:
        for key in ('full_semantic_domain','screening_complete','screening_receipt','validity_protocol','stop_words'):
            bound_path(input_provenance[key])
    fit, control, _ = _candidate_axes(context, fit_indices, control_indices)

    def instances():
        for spec in pattern_instances(templates):
            mask = exact_pattern_mask(spec.predicate, context['docs'], context.get('attribute_presence'))
            yield spec, mask[fit], mask[control], input_provenance

    yield from _fit_signed_groups(instances(), context=context, fit_indices=fit,
        control_indices=control, fit_truth=fit_truth, baseline_predictions=baseline_predictions,
        background_rules=background_rules, cache_dir=cache_dir, model_snapshot_id=model_snapshot_id, execution_backend=execution_backend)


def iter_signed_semantic_candidates(raw_dir, cluster_id, *, context, fit_truth,
                                    baseline_predictions, cache_dir, background_rules=(),
                                    model_snapshot_id=None):
    """Read every authenticated extract/materialize mask; freshly fit its heads.

    Raw singleton frequency keys are preserved rather than expanded a second
    time. Saved AddOnly labels and historical selected Pi are never loaded.
    """
    if type(cluster_id) is not int or cluster_id < 0:
        raise ValueError('raw semantic cluster ID must be a nonnegative integer')
    raw_dir = Path(raw_dir).resolve(strict=True)
    complete_record = file_record(raw_dir / 'COMPLETE.json')
    raw = _read(bound_path(complete_record))
    if raw.get('schema') != RAW_SCHEMA + '_masks' or raw.get('all_raw_masks_saved') is not True:
        raise ValueError('complete raw semantic mask archive required')
    domain_record = bound_path(raw['domain'])
    domain, policy = _domain(domain_record.parent)
    if (raw['clusters'] != domain['clusters'] or raw['raw_candidates'] != domain['expected_candidates']
            or raw['expected_candidates'] != domain['expected_candidates']):
        raise ValueError('raw semantic mask/domain complete counts differ')
    _, _, _, _, actual_axes = _axes(context, policy['axes']['cluster_assignments'])
    if actual_axes != policy['axes']:
        raise ValueError('raw semantic context text/row/presence binding differs')
    if str(cluster_id) not in raw['clusters']:
        raise ValueError('raw semantic cluster is not declared')
    detail = raw['clusters'][str(cluster_id)]
    fit, control, rows = _candidate_axes(context, detail['fit_indices'], detail['control_indices'])
    if ([rows[i] for i in fit] != detail['fit_row_ids']
            or [rows[i] for i in control] != detail['control_row_ids']):
        raise ValueError('raw semantic fit/control row identities differ')
    database = bound_path(raw['database'])
    if any(Path(str(database) + suffix).exists() for suffix in ('-wal', '-shm', '-journal')):
        raise ValueError('raw semantic database must be a closed immutable snapshot')
    provenance = {'schema': RAW_SCHEMA, 'raw_unfiltered_vocabulary': True,
        'raw_complete': complete_record, 'domain_sha256': raw['domain']['sha256'],
        'raw_database_sha256': raw['database']['sha256'], 'cluster': cluster_id,
        'expected_raw_candidates': detail['expected_candidates'],
        'old_fitted_labels_or_selected_patterns_reused': False}

    def instances():
        db = _readonly(database)
        count = 0
        try:
            for ordinal, key, encoded, fm, fs, cm, cs in db.execute(
                    'SELECT ordinal,key,spec,fit_mask,fit_sha,control_mask,control_sha '
                    'FROM raw WHERE cluster=? ORDER BY ordinal', (cluster_id,)):
                if ordinal != count or count >= detail['expected_candidates']:
                    raise ValueError('raw semantic candidate ordinal/count differs')
                spec = _pattern_spec(json.loads(encoded), key)
                fit_mask = _unpack(fm, fs, len(fit))
                control_mask = _unpack(cm, cs, len(control))
                yield spec, fit_mask, control_mask, dict(provenance, raw_ordinal=ordinal)
                count += 1
            if count != detail['expected_candidates']:
                raise ValueError('complete raw semantic candidate count differs')
            bound_path(complete_record)
            bound_path(raw['database'])
            bound_path(raw['domain'])
            _domain(domain_record.parent)
        finally:
            db.close()

    yield from _fit_signed_groups(instances(), context=context, fit_indices=fit,
        control_indices=control, fit_truth=fit_truth, baseline_predictions=baseline_predictions,
        background_rules=background_rules, cache_dir=cache_dir, model_snapshot_id=model_snapshot_id)
