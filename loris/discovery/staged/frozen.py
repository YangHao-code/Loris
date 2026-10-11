"""Rich-rule BO on an immutable model foundation; evaluate complete-closure gain."""
import collections
import copy
import json

import numpy as np
import optuna

from . import program as rb
from . import helpers as prior


def admissible(before, after):
    return (all(after[k] >= before[k] - 1e-12 for k in ('macro_f1', 'micro_f1'))
            and prior.objective(after) > prior.objective(before) + 1e-12)


def changes(old_counts, new_counts):
    return dict(added_TP=int((new_counts[0] - old_counts[0]).sum()),
                added_FP=int((new_counts[1] - old_counts[1]).sum()),
                removed_FN=int((old_counts[2] - new_counts[2]).sum()))


def admit(program, selected, candidates, metric, folds):
    selected = list(selected)
    protected = set(selected)
    current = program.closure(program.seed, selected)
    counts = metric.counts(current)
    remaining = sorted(set(candidates) - set(selected))
    log = []
    while remaining:
        before = metric.quality(counts)
        watch = prior.watchers(program, selected)
        best = None
        for i in remaining:
            after = program.incremental(current, selected, i, watch)
            if after == current:
                continue
            new = metric.delta(current, after, counts)
            q = metric.quality(new)
            if not admissible(before, q):
                continue
            stable = prior.folded_gain(program, current, after, folds)
            if stable <= 1e-12:
                continue
            value = prior.objective(q) - prior.objective(before) + .25 * stable
            if best is None or value > best[0] + 1e-15:
                best = value, i, after, new, q
        if best is None:
            break
        value, i, after, new, q = best
        log.append(dict(rule_index=i, quality_A=q, robust_score=value,
                        **changes(counts, new)))
        current, counts = after, new
        selected.append(i)
        remaining.remove(i)
        assert protected.issubset(selected)
    return selected, log


def fit(ctx, program, rules, templates, selected, families, members, alias, cuts, A, y, seed):
    templates = [x for x in templates if x['family'] in families]
    if not templates:
        return list(selected), dict(trials=0, templates=0, accepted=0, trajectory=[])
    aliases = sorted(set(members) | {alias})
    keys = ['r%d.m%d' % (r, m)
            for r in range(max(len(x['roles']) for x in templates)) for m in range(len(aliases))]
    label_index = {l: j for j, l in enumerate(ctx['labels'])}
    metric = rb.Metric(A, y, program.n)
    folds = [rb.Metric(A[np.arange(len(A)) % 3 == f], y[np.arange(len(A)) % 3 == f], program.n)
             for f in range(3)]
    current = program.closure(program.seed, selected)
    counts = metric.counts(current)
    before = metric.quality(counts)
    watch = prior.watchers(program, selected)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=prior.sampler(keys, seed))
    # A guard without an ML condition is essential: adding the already-covered
    # head's ensemble predicate at its base threshold would only be redundant.
    for t, info in enumerate(templates):
        naked = dict.fromkeys(keys, 0)
        naked['template'] = t
        study.enqueue_trial(naked)
        guarded = dict(naked)
        key = 'r0.m%d' % aliases.index(alias)
        guarded[key] = 1
        guarded['cut.' + key] = float(cuts[alias][label_index[info['roles'][0][1]]])
        study.enqueue_trial(guarded)
    limit = 2 * len(templates) + 4 * len(ctx['labels'])
    candidates, records, seen = [], [], set()

    def evaluate(trial):
        t = trial.suggest_categorical('template', list(range(len(templates))))
        info = templates[t]
        active = {k: trial.suggest_categorical(k, [0, 1]) for k in keys}
        atoms = []
        for r, (var, label) in enumerate(info['roles']):
            for m, a in enumerate(aliases):
                key = 'r%d.m%d' % (r, m)
                if active[key]:
                    atoms.append((var, a, label, trial.suggest_float('cut.' + key, .001, .999)))
        rule = prior.make_rule(info['guard'].head.label, atoms, info['guard'])
        signature = json.dumps(rule.to_dict(), sort_keys=True)
        if signature in seen:
            return -1.
        seen.add(signature)
        item = copy.copy(program.items[info['index']])
        item['domains'] = dict(item['domains'])
        item['cache'] = collections.OrderedDict()
        for var, a, label, cut in atoms:
            item['domains'][var] &= rb.bits(np.asarray(ctx['model_probabilities'][a][:, label_index[label]], np.float64) >= cut)
        idx = len(rules)
        assert idx == len(program.items)
        rules.append(rule)
        program.items.append(item)
        after = program.incremental(current, selected, idx, watch)
        new = metric.delta(current, after, counts)
        q = metric.quality(new)
        if admissible(before, q):
            stable = prior.folded_gain(program, current, after, folds)
            value = prior.objective(q) - prior.objective(before) + .25 * stable - 1e-6 * max(0, len(atoms) - 1)
        else:
            value = -1.
        candidates.append(idx)
        records.append(dict(trial=trial.number, template=t, rule_index=idx,
                            quality_A=q, value=float(value), **changes(counts, new)))
        return float(value)

    study.optimize(evaluate, n_trials=limit, show_progress_bar=False)
    selected, log = admit(program, selected, candidates, metric, folds)
    return selected, dict(trials=limit, templates=len(templates), accepted=len(log),
                          initial=before, trajectory=log, records=records)
