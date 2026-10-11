"""Portable discovery entry point: frozen experts, original BO, native checks."""
import collections
import json
from pathlib import Path
import time

import numpy as np

from loris._engine.rdl_reference import Doc, Label, ML, Rule
from loris._storage.discovery_factory import make_discovery_runner
from loris.rule_forms import validate_program_rule_forms
from . import frozen, helpers, inventory, program as rb, prune, search
from .metrics import quality
from .pool import extend_context


def choose(entries, *, priority='diversity', macro_tolerance=0., micro_tolerance=.002):
    """Compare fitted-dev programs only; no target labels or scores accepted."""
    if priority not in ('diversity', 'macro'):
        raise ValueError('priority must be diversity or macro')
    if any(not np.isfinite(v) or v < 0 or v > 1
           for v in (macro_tolerance, micro_tolerance)):
        raise ValueError('Quality tolerances must be finite absolute F1 differences')
    base = entries[0]['quality_B']
    eligible = [e for e in entries
                if e['quality_B']['macro_f1'] >= base['macro_f1'] - macro_tolerance - 1e-12
                and e['quality_B']['micro_f1'] >= base['micro_f1'] - micro_tolerance - 1e-12]
    def key(e):
        diversity = sum(v['necessary_TP'] > 0 for v in e['effective_families_B'].values())
        macro, micro = e['quality_B']['macro_f1'], e['quality_B']['micro_f1']
        return ((diversity, macro, micro, -len(e['rules'])) if priority == 'diversity'
                else (macro, micro, diversity, -len(e['rules'])))
    return max(eligible, key=key)['candidate_id']


def _rows(context, indices, targets, name):
    rows, y = np.asarray(indices), np.asarray(targets)
    n, k = len(context['docs']), len(context['labels'])
    if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
            or len(set(rows.tolist())) != len(rows)
            or (rows < 0).any() or (rows >= n).any()
            or y.dtype != np.bool_ or y.shape != (len(rows), k)):
        raise ValueError('Invalid complete ' + name + ' axes')
    if (context['gamma'].pos | context['gamma'].neg)[rows].any():
        raise ValueError(name + ' answers must not already occur in Gamma')
    roles = context.get('row_roles')
    if roles is None or len(roles) != n:
        raise ValueError('Explicit train/dev/test row_roles required')
    if any(str(roles[int(i)]) not in ('dev', 'calibration') for i in rows):
        raise ValueError(name + ' must contain only dev/calibration rows')
    return rows.astype(np.int64), y


def discover(*, context, pool, old_rules, A, truth_A, B, truth_B, output,
             pattern_supply=None, lexical_cache_dir=None, mode='diverse',
             priority='diversity', macro_tolerance=0., micro_tolerance=.002,
             seed=28101, allow_replacement=True):
    """Fit on A, optionally prune on B, lock programs, verify native closures.

    ``frozen`` reproduces the frozen-base rich search. ``diverse`` adds optional
    replacement and B pruning. B is fitted development data in the latter mode.
    No test truth is an input. Apply SELECTED.json with the ordinary Chase.
    """
    if mode not in ('frozen', 'diverse') or type(allow_replacement) is not bool:
        raise ValueError('Explicit supported mode and replacement policy required')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    A, yA = _rows(context, A, truth_A, 'A')
    B, yB = _rows(context, B, truth_B, 'B')
    if np.intersect1d(A, B).size or len(A) < 3:
        raise ValueError('Disjoint A/B and at least three A rows required')
    # Discovery contexts may not contain target text/relations: otherwise a
    # transductive path could influence development selection before locking.
    if any(str(role) not in ('train', 'dev', 'calibration') for role in context['row_roles']):
        raise ValueError('Use a train/dev-only context for rule discovery')
    train = np.asarray([str(role) == 'train' for role in context['row_roles']])
    observed = context['gamma'].pos | context['gamma'].neg
    if observed[~train].any() or not observed[train].all():
        raise ValueError('Gamma must contain exactly the complete supplied training labels')
    if len(set(context['labels'])) != len(context['labels']):
        raise ValueError('Unique native labels required')
    context = extend_context(context, pool)
    alias, members = pool.get('alias', 'loris_calibrated_pool'), list(pool['members'])
    base = [Rule((Doc('x'), ML('x', alias, label, .5)), Label('x', label))
            for label in context['labels']]
    old_rules = [Rule.from_dict(r) if isinstance(r, dict) else r for r in old_rules]
    rules = list(base)
    base_ids = list(range(len(base)))
    programs = {'model_base': base_ids}
    traces = {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    settings = dict(mode=mode, priority=priority, seed=seed,
                    macro_tolerance=macro_tolerance, micro_tolerance=micro_tolerance,
                    allow_replacement=allow_replacement,
                    B_used_for_pruning=mode == 'diverse', independent_B_validation=False,
                    objective=(.9 if mode == 'diverse' else .8),
                    per_label_rule_cap=None, minimum_positive_support=None,
                    test_used_for_selection=False, new_Aux_training=False)
    (output / 'SETTINGS.json').write_text(json.dumps(settings, indent=2))
    (output / 'POOL.json').write_text(json.dumps(pool, indent=2, allow_nan=False))
    cuts = dict(pool['expert_cuts'], **{alias: [.5] * len(base)})
    if mode == 'frozen':
        rules += [r for r in old_rules if rb.family(r) != 'legacy_ml']
        oldrich = list(range(len(base), len(rules)))
        templates = []
        for guard, roles, family in helpers.guard_templates(old_rules):
            templates.append(dict(index=len(rules), guard=guard, roles=roles, family=family))
            rules.append(guard)
    runner = make_discovery_runner(context, output / 'cache',
                                   lexical_cache_dir=lexical_cache_dir, gamma_protection=True)
    try:
        program = rb.Program(runner, rules)
        if mode == 'frozen':
            metric = rb.Metric(A, yA, program.n)
            folds = [rb.Metric(A[np.arange(len(A)) % 3 == f],
                               yA[np.arange(len(A)) % 3 == f], program.n) for f in range(3)]
            programs['old_rich_joint'], traces['old_rich_joint'] = frozen.admit(
                program, base_ids, oldrich, metric, folds)
            for name, stages in [('text_then_propagation', [{'text'}, {'label', 'relation'}]),
                                 ('propagation_then_text', [{'label', 'relation'}, {'text'}])]:
                ids, log = list(base_ids), []
                for index, families in enumerate(stages):
                    ids, record = frozen.fit(context, program, rules, templates, ids, families,
                                             members, alias, cuts, A, yA,
                                             seed + index + (100 if name.startswith('propagation') else 0))
                    log.append(record)
                    if index == 0:
                        first = 'text' if name.startswith('text') else 'propagation'
                        programs[first], traces[first] = list(ids), list(log)
                programs[name], traces[name] = list(ids), log
        else:
            factory = inventory.Factory(context, runner, program, rules)
            patterns = inventory.pattern_inventory(pattern_supply or {})
            patterns, matrix, _ = inventory.text_features(context, A, patterns, output / 'text_cache')
            ids = context.get('row_ids', list(range(program.n)))
            split = helpers.folds(yA, [ids[int(i)] for i in A], 3, seed)
            g, p = split != 0, split == 0
            bundles, audit = search.proposals(
                context, factory, base_ids, A[g], yA[g], members, cuts,
                patterns, matrix[g], old_rules, seed=seed, ensemble=alias,
                allow_replacement=allow_replacement)
            (output / 'PROPOSALS.json').write_text(json.dumps(dict(
                audit=audit, bundles=bundles, rules=[r.to_dict() for r in rules]), allow_nan=False))
            options = [('add_only', True)] + ([('replace', False)] if allow_replacement else [])
            for name, only_add in options:
                grown, growth = search.accept(program, rules, base_ids, bundles,
                    A[g], yA[g], A[p], yA[p], add_only=only_add)
                retained, pruning = prune.prune(program, rules, grown, base_ids, B, yB)
                programs[name] = retained
                traces[name] = dict(growth=growth, B_pruning=pruning)
                # Save each finished candidate before beginning another slow one.
                (output / (name + '.json')).write_text(json.dumps(dict(
                    rules=[rules[i].to_dict() for i in retained], trace=traces[name]), allow_nan=False))
        entries = []
        rows = np.r_[A, B]
        for name, ids in programs.items():
            selected = tuple(rules[i] for i in ids)
            validate_program_rule_forms(selected)
            state = program.closure(program.seed, ids)
            prediction = program.prediction(state, rows)
            actual = runner(selected, gamma=context['gamma'])
            if actual.status != 'fixpoint' or not np.array_equal(
                    actual.positive_facts, program.prediction(state, np.arange(program.n))):
                raise RuntimeError('Bitset/native complete closure mismatch')
            if not np.array_equal(actual.negative_facts, context['gamma'].neg):
                raise RuntimeError('Positive rule discovery changed negative roots')
            effective = search.effective_families(program, rules, ids, B)
            before = prediction[len(A):]
            for family, item in effective.items():
                sub = [i for i in ids if inventory.family(rules[i]) != family]
                without = program.prediction(program.closure(program.seed, sub), B)
                item['necessary_TP'] = int((before & ~without & yB).sum())
                item['necessary_FP'] = int((before & ~without & ~yB).sum())
            entries.append(dict(candidate_id=name, rules=[r.to_dict() for r in selected],
                families=dict(collections.Counter(inventory.family(r) for r in selected)),
                quality_A=quality(prediction[:len(A)], yA), quality_B=quality(before, yB),
                effective_families_B=effective, trace=traces.get(name, [])))
        chosen = choose(entries, priority=priority, macro_tolerance=macro_tolerance,
                        micro_tolerance=micro_tolerance)
        selected = next(e for e in entries if e['candidate_id'] == chosen)
        result = dict(schema='loris_staged_discovery_v1', settings=settings,
                      selected_candidate_id=chosen, entries=entries,
                      complete_native_closures_verified=True, seconds=time.monotonic() - started)
        (output / 'SELECTED.json').write_text(json.dumps(dict(
            rules=selected['rules'], pool_alias=alias, candidate_id=chosen), indent=2, allow_nan=False))
        (output / 'COMPLETE.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        runner.close()
