"""Vectorized exact fit-initializer scan (archive mode ``winners_and_counts``).

Reproduces, candidate by candidate, the per-label winners and reason counts of
the reference loop in initialization.py under its analytic-skip semantics:

* candidate order and numbering  L x (M x (P+1) + M(M-1)/2);
* thresholds  eta = at_or_above(domain, nextafter(max negative score in the
  candidate's scope)) or the domain minimum when the scope has no negative;
* reasons  boundary_outside_domain / direct_root_conflict / zero_support /
  eligible, and the ranking (fit F1, TP, fewer body predicates, number).

Only the per-candidate JSONL archive is not written. The reference loop is
used whenever a model is static or its threshold domain is not a closed float
interval, so this module never guesses a domain rule.
"""

import numpy as np
import scipy.sparse as sp

from loris.predicates import FreqPredicate


def supported(space):
    return all(
        m.dynamic and m.threshold_domain is not None and not m.threshold_domain.values
        and not m.threshold_domain.integer
        for m in space.models
    )


def _eta(domain, raw):
    """_at_or_above for a closed float interval, element-wise (exact)."""
    answer = np.maximum(float(domain.low), raw)
    legal = (answer >= float(domain.low)) & (answer <= float(domain.high))
    return answer, legal


def _first_eta(domain):
    eta = 0.5 if domain.contains(0.5) else (domain.low + domain.high) / 2
    return eta


def scan(space, prior, masks, has_positive):
    """Return (counts, winners) exactly as the reference analytic-skip loop."""
    labels, models = space.labels, space.models
    L, M, P = len(labels), len(models), len(space.patterns)
    block = M * (P + 1) + M * (M - 1) // 2
    pos, neg = np.asarray(prior.pos, bool), np.asarray(prior.neg, bool)
    n = pos.shape[0]
    positive_counts = pos.sum(axis=0)
    counts = {"total": 0, "zero_support": 0, "boundary_outside_domain": 0,
              "direct_root_conflict": 0, "eligible": 0,
              "analytically_omitted_zero_text_positive": 0}
    if P:
        rows = [np.flatnonzero(m) for m in masks]
        indptr = np.zeros(P + 1, np.int64)
        indptr[1:] = np.cumsum([len(r) for r in rows])
        indices = np.concatenate(rows) if indptr[-1] else np.zeros(0, np.int64)
    else:
        indptr, indices = np.zeros(1, np.int64), np.zeros(0, np.int64)
    has_positive = np.asarray(has_positive, bool).reshape(P, L)
    winners = {}
    for li in range(L):
        pos_l, neg_l = pos[:, li], neg[:, li]
        pcount = int(positive_counts[li])
        base = li * block
        q = np.flatnonzero(has_positive[:, li])
        skipped = P - len(q)
        counts["total"] += block
        counts["analytically_omitted_zero_text_positive"] += M * skipped
        best = None  # (f1, tp, -body, -number, payload)

        def consider(number, body, tp, fp, legal, payload):
            nonlocal best
            if not legal:
                counts["boundary_outside_domain"] += 1
                return
            if fp:
                counts["direct_root_conflict"] += 1
                return
            if not tp:
                counts["zero_support"] += 1
                return
            counts["eligible"] += 1
            f1 = 2 * tp / (2 * tp + fp + (pcount - tp))
            rank = (f1, tp, -body, -number)
            if best is None or rank > best[0]:
                best = (rank, payload)

        # Pattern rows restricted to Q (every Q pattern has a positive row).
        if len(q):
            seg_start = indptr[q]
            seg_len = indptr[q + 1] - seg_start
            gather = np.repeat(seg_start - np.concatenate(([0], np.cumsum(seg_len)[:-1])), seg_len) \
                + np.arange(int(seg_len.sum()))
            rows_q = indices[gather]
            starts = np.concatenate(([0], np.cumsum(seg_len)[:-1]))
            negsel = neg_l[rows_q]
            possel = pos_l[rows_q]
        for mi, model in enumerate(models):
            domain = model.threshold_domain
            low = float(domain.low)
            s = np.asarray(prior.probabilities[model.alias][:, li], np.float64)
            # Pure ML candidate: scope is every fit row.
            if neg_l.any():
                raw = float(np.nextafter(np.max(s[neg_l]), np.inf))
            else:
                raw = domain.low  # the reference records the declared object
            eta, legal = _eta(domain, np.float64(raw))
            eta, legal = float(eta), bool(legal)
            if legal:
                firing = s >= eta
                tp, fp = int(np.sum(firing & pos_l)), int(np.sum(firing & neg_l))
            else:
                tp = fp = 0
            consider(base + mi * (P + 1), 1, tp, fp, legal,
                     ("pure", mi, None, eta, raw))
            if not len(q):
                continue
            data = s[rows_q]
            masked = np.where(negsel, data, -np.inf)
            rowmax = np.maximum.reduceat(masked, starts)
            has_neg = np.logical_or.reduceat(negsel, starts)
            raw_q = np.where(has_neg, np.nextafter(rowmax, np.inf), low)
            eta_q, legal_q = _eta(domain, raw_q)
            eta_rep = np.repeat(eta_q, seg_len)
            fire = data >= eta_rep
            tp_q = np.add.reduceat((fire & possel).astype(np.int64), starts)
            fp_q = np.add.reduceat((fire & negsel).astype(np.int64), starts)
            tp_q = np.where(legal_q, tp_q, 0)
            fp_q = np.where(legal_q, fp_q, 0)
            # Counts for the whole block, then the winner among eligible rows.
            illegal = ~legal_q
            conflict = legal_q & (fp_q > 0)
            zero = legal_q & (fp_q == 0) & (tp_q == 0)
            eligible = legal_q & (fp_q == 0) & (tp_q > 0)
            counts["boundary_outside_domain"] += int(illegal.sum())
            counts["direct_root_conflict"] += int(conflict.sum())
            counts["zero_support"] += int(zero.sum())
            counts["eligible"] += int(eligible.sum())
            e = np.flatnonzero(eligible)
            if len(e):
                tp_e = tp_q[e]
                f1_e = (2 * tp_e) / (2 * tp_e + fp_q[e] + (pcount - tp_e)).astype(np.float64)
                numbers = base + mi * (P + 1) + 1 + q[e]
                # Max of (f1, tp, -2, -number): primary key last in lexsort.
                order = np.lexsort((-numbers, tp_e, f1_e))
                j = e[order[-1]]
                number = int(base + mi * (P + 1) + 1 + q[j])
                tp_j = int(tp_q[j])
                f1_j = 2 * tp_j / (2 * tp_j + int(fp_q[j]) + (pcount - tp_j))
                rank = (f1_j, tp_j, -2, -number)
                if best is None or rank > best[0]:
                    raw_j = float(raw_q[j]) if has_neg[j] else domain.low
                    best = (rank, ("text", mi, int(q[j]), float(eta_q[j]), raw_j))
        for pair_index, (a, b) in enumerate((a, b) for a in range(M) for b in range(a + 1, M)):
            number = base + M * (P + 1) + pair_index
            ma, mb = models[a], models[b]
            first = _first_eta(ma.threshold_domain)
            sa = np.asarray(prior.probabilities[ma.alias][:, li], np.float64)
            sb = np.asarray(prior.probabilities[mb.alias][:, li], np.float64)
            scope = sa >= first
            opposite = scope & neg_l
            raw = (float(np.nextafter(np.max(sb[opposite]), np.inf)) if np.any(opposite)
                   else mb.threshold_domain.low)
            eta_b, legal = _eta(mb.threshold_domain, np.float64(raw))
            eta_b, legal = float(eta_b), bool(legal)
            if legal:
                scope = scope & (sb >= eta_b)
                tp, fp = int(np.sum(scope & pos_l)), int(np.sum(scope & neg_l))
            else:
                tp = fp = 0
            consider(number, 2, tp, fp, legal, ("pair", a, b, first, eta_b, raw))
        if best is not None:
            winners[li] = best
    return counts, winners


def winner_record(space, prior, li, rank, payload):
    """Rebuild the reference winner record and dense parameters exactly."""
    label = space.labels[li]
    variable = space.variables[0]
    f1, tp, neg_body, neg_number = rank
    params = space.empty_parameters("add", label=label, variable=variable)
    kind = payload[0]
    if kind == "pair":
        _, a, b, first, eta_b, raw = payload
        model_indices, pattern_index = [a, b], None
        thresholds = [
            {"alias": space.models[a].alias, "eta": first, "raw_boundary": None},
            {"alias": space.models[b].alias, "eta": eta_b, "raw_boundary": raw},
        ]
        etas = {a: first, b: eta_b}
    else:
        _, mi, pattern_index, eta, raw = payload
        model_indices = [mi]
        thresholds = [{"alias": space.models[mi].alias, "eta": eta, "raw_boundary": raw}]
        etas = {mi: eta}
    mfeatures = {(f.index, f.label): f for f in space.features if f.kind == "model" and f.var == variable}
    pfeatures = {f.index: f for f in space.features if f.kind == "pattern" and f.var == variable}
    if pattern_index is not None:
        f = pfeatures[pattern_index]
        pred = space.patterns[pattern_index].predicate
        params[f.include_key] = 1
        if isinstance(pred, FreqPredicate):
            params[f.key + ".op"] = pred.op
            params[f.key + ".eta"] = pred.eta
    for mi in model_indices:
        f = mfeatures[(mi, label)]
        params[f.include_key] = 1
        params[f.key + ".eta"] = etas[mi]
    fp = 0
    fn = int(prior.pos[:, li].sum()) - tp
    record = {
        "candidate_number": -neg_number,
        "label_index": li,
        "label": label,
        "pattern_index": pattern_index,
        "model_indices": list(model_indices),
        "thresholds": thresholds,
        "fit_tp": tp,
        "fit_fp": fp,
        "fit_fn": fn,
        "fit_f1": f1,
        "reason": "eligible",
    }
    return record, params
