"""Fit-conditioned rich mixtures plus an unmodified full-axis TPE branch.

This changes proposal probabilities, not RDL/Chase/admission semantics. Every
attempt is one budgeted BO trial. No fit threshold grid is scored or imported.
All failures fall back to the same original binary draw and are visible.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

from loris.predicates import FreqPredicate
from loris._engine.rdl_reference import AttrEqual, LabelCompare
from loris._engine.sparse_bo import JointBinaryTPESampler
from loris._engine.primitive_masks import TextPrimitiveCache
from loris._learning.structured_bo import FitWitnessPrior, build_fit_witness_prior

SCHEMA = "paper36_rich_fit_mixture_tpe_20260916_v1"
# Fixed before scores. This is a proposal schedule, not a grammar restriction.
RECIPES = (
    "pure_ml",
    "multi_ml",
    "text_ml",
    "multi_text_ml",
    "label_ml",
    "label_text",
    "propagation",
    "propagation_rich",
    "relation_head",
    "broad",
    "broad",
    "broad",
)
_LOCAL_PATTERN_MASKS = WeakKeyDictionary()


def pattern_mask(prior, pattern_index, active=None):
    """Exact fit-only text mask, computed once per instantiated predicate."""
    from loris.predicates import predicate_to_dict
    from loris._engine.rlearner_config import _canonical

    pred = prior.patterns[pattern_index].predicate
    if active is not None and isinstance(pred, FreqPredicate):
        pred = replace(pred, op=active["op"], eta=active["eta"])
    cache = _LOCAL_PATTERN_MASKS.setdefault(prior, {})
    key = _canonical(predicate_to_dict(pred))
    if key not in cache:
        if prior.cache_dir is not None:
            # Share the exact original count/first-position cache with the
            # scalar fit witness lookup. A new Freq threshold derives a mask
            # from cached counts instead of rescanning the text. prior.close()
            # owns this connection; no live SQLite object enters checkpoints.
            from loris._learning.structured_bo import _MASKS
            from loris._engine.primitive_masks import TextPrimitiveCache

            primitive = _MASKS.get(prior)
            if primitive is None:
                primitive = TextPrimitiveCache(
                    prior.docs,
                    prior.cache_dir,
                    attribute_presence={k: prior.presence[k] for k in ("cnt", "ttl", "mtd")},
                )
                _MASKS[prior] = primitive
            cache[key] = primitive.mask(pred)
        else:
            presence = prior.presence[pred.attr]
            cache[key] = np.fromiter(
                (bool(presence[r]) and bool(pred(doc)) for r, doc in enumerate(prior.docs)),
                bool,
                count=len(prior.docs),
            )
        cache[key].setflags(write=False)
    return cache[key]


def source_record(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def _distribution(domain):
    if domain.values:
        return CategoricalDistribution(domain.values)
    if domain.integer:
        return IntDistribution(int(domain.low), int(domain.high))
    return FloatDistribution(float(domain.low), float(domain.high))


def _at_or_above(domain, value):
    """Smallest legal threshold >= value; no numerical domain clipping."""
    if domain.values:
        legal = [v for v in domain.values if v >= value]
        return min(legal) if legal else None
    answer = max(float(domain.low), float(value))
    if domain.integer:
        answer = int(np.ceil(answer))
    return answer if domain.contains(answer) else None


class RichCombinationTPESampler(JointBinaryTPESampler):
    def __init__(self, include_keys, *, seed, prior, context_probes=32, **kwargs):
        if type(prior) is not FitWitnessPrior:
            raise TypeError("exact detached FitWitnessPrior required")
        if tuple(include_keys) != tuple(f.include_key for f in prior.features):
            raise ValueError("complete feature axis required")
        if type(context_probes) is not int or context_probes < 1:
            raise ValueError("positive context probe budget required")
        prior.validate()
        super().__init__(include_keys, seed=seed, **kwargs)
        self.prior = prior
        self.context_probes = context_probes
        self.model_features = {
            (f.var, f.index, f.label): i for i, f in enumerate(prior.features) if f.kind == "model"
        }
        self.protocol = {
            "schema": SCHEMA,
            "base_TPE": copy.deepcopy(self.protocol),
            "fit_witness_id": prior.binding_id,
            "fit_witness_binding": copy.deepcopy(prior.binding),
            "schedule": list(RECIPES),
            "schedule_index": "trial.number modulo 12",
            "label_schedule": "(trial.number // 12 + cluster integer offset) modulo K; decimal-string integer cluster IDs parsed exactly; other cluster IDs use zero; fit-supported fallback",
            "label_head_sign": "independent seeded Bernoulli .1 remove/.9 add per structured label-head trial; not coupled to label cycle",
            "relation_heads": "eq/subset/strict_subset cyclic; actual distinct observed AttrEqual witness",
            "broad": "3/12 slots retain original full-axis joint binary TPE without conditioning",
            "structured_extra_atoms": "retain original ON atoms true at the shared fit valuation; no body cap",
            "ML_context": "optional; never remove ML for lack of Text or AttrEqual",
            "thresholds": "original domain, including values below .5 and endpoints; fit negative boundary / negative .9 quantile / TPE",
            "threshold_mixture": "4/6 nextafter(max opposite known fit probability,+inf), 1/6 opposite .9 quantile, 1/6 original TPE",
            "threshold_boundary_scope": "same-variable current conjunctive fit body mask, no dev truth, no independent candidate score",
            "threshold_unavailable": "retain original TPE threshold, never declare unsupported rule reliable",
            "context_probes": context_probes,
            "pattern_choice": "fit witness match, at most declared probes; no standalone Text head requirement",
            "negative_facts": "explicit gamma_neg only; unknown is never negative",
            "relations": "actual observed nonmissing values only; no truth-created relation",
            "fallback": "same pre-existing full-axis TPE draw with conditional delegate restored",
            "initialization": "caller-declared fit initializer replaces existing first 30 startup slots; fallback is optional-context sparse; no Stage0 or historical bank",
            "fit_reanchoring": "single-variable candidate may change its fit witness to an explicitly known head row satisfying the same complete proposed body; no parameter changes, extra scored candidates or relation construction",
            "all_legal_candidates_enter_line12": True,
            "hard_quality_or_zero_error_gate": False,
            "models_patterns_heads_parameter_axes_unchanged": True,
            "control_truth_for_proposals": False,
            "historical_rules_imported": False,
            "source": source_record(__file__),
            "prior_source": source_record(inspect.getfile(FitWitnessPrior)),
            "text_primitive_source": source_record(inspect.getfile(TextPrimitiveCache)),
            "joint_source": source_record(inspect.getfile(JointBinaryTPESampler)),
            "equivalent_to_old_BO_sequence": False,
        }

    def _draw(self, study, trial, key, distribution):
        return self.delegate.sample_independent(study, trial, key, distribution)

    def _active(self, feature, study, trial):
        if feature.kind == "pattern" and isinstance(
            self.prior.patterns[feature.index].predicate, FreqPredicate
        ):
            spec = self.prior.patterns[feature.index]
            return {
                feature.key
                + ".op": self._draw(
                    study, trial, feature.key + ".op", CategoricalDistribution(spec.frequency_ops)
                ),
                feature.key
                + ".eta": self._draw(
                    study, trial, feature.key + ".eta", _distribution(spec.frequency_domain)
                ),
            }
        if feature.kind == "model" and self.prior.models[feature.index].dynamic:
            domain = self.prior.models[feature.index].threshold_domain
            return {
                feature.key
                + ".eta": self._draw(study, trial, feature.key + ".eta", _distribution(domain))
            }
        return {}

    def _local_mask(self, chosen, active, variable):
        """Exact conjunction of same-variable features on the detached fit rows.

        Cross-document attributes and source facts cannot be treated as target
        facts here. They remain checked by actual shared witness/full Chase.
        """
        prior = self.prior
        mask = np.ones(len(prior.rows), bool)
        for i in chosen:
            f = prior.features[i]
            if f.var != variable:
                continue
            if f.kind == "label":
                mask &= (prior.pos if f.polarity == "pos" else prior.neg)[
                    :, prior.label_index[f.label]
                ]
            elif f.kind == "model":
                m = prior.models[f.index]
                li = prior.label_index[f.label]
                mask &= (
                    prior.probabilities[m.alias][:, li].astype(np.float64) >= active[f.key + ".eta"]
                    if m.dynamic
                    else prior.decisions[m.alias][:, li]
                )
            elif f.kind == "pattern":
                args = (
                    {"op": active[f.key + ".op"], "eta": active[f.key + ".eta"]}
                    if isinstance(prior.patterns[f.index].predicate, FreqPredicate)
                    else None
                )
                mask &= pattern_mask(prior, f.index, args)
        return mask

    def _threshold(self, index, chosen, active, planned, assignment, study, trial):
        prior = self.prior
        f = prior.features[index]
        m = prior.models[f.index]
        values = self._active(f, study, trial)
        if not m.dynamic:
            return values, {"policy": "static original decision"}
        label = planned.get("head.label")
        target = planned.get("head.var")
        detail = {"policy": "original TPE", "alias": m.alias, "model_label": f.label}
        # Only a head-target atom admits the following fit-known counterexample
        # calculation. Cross-label/source atoms retain their original domain.
        if f.var == target and label is not None:
            li = prior.label_index[label]
            opposite = prior.neg[:, li] if planned["head.kind"] == "add" else prior.pos[:, li]
            scope = self._local_mask(chosen, active, target)
            negatives = opposite & scope
            scores = prior.probabilities[m.alias][:, prior.label_index[f.label]].astype(np.float64)
            mode = int(self.rng.randint(6))
            eta = None
            if np.any(negatives) and mode < 5:
                boundary = (
                    float(np.max(scores[negatives]))
                    if mode < 4
                    else float(np.quantile(scores[negatives], 0.9))
                )
                eta = _at_or_above(m.threshold_domain, float(np.nextafter(boundary, np.inf)))
                detail.update(
                    policy="fit_opposite_max_boundary" if mode < 4 else "fit_opposite_q90_boundary",
                    opposite_known_cells=int(negatives.sum()),
                    boundary=boundary,
                    scope_rows=int(scope.sum()),
                    legal_boundary=eta,
                )
            elif not np.any(negatives) and mode < 5:
                eta = (
                    _at_or_above(m.threshold_domain, float(m.threshold_domain.low or 0))
                    if not m.threshold_domain.values
                    else min(m.threshold_domain.values)
                )
                detail.update(
                    policy="no_explicit_opposite_in_fit_scope_domain_min",
                    scope_rows=int(scope.sum()),
                )
            if eta is not None:
                values[f.key + ".eta"] = eta
        # This is one proposal, not retry search. If a boundary excludes its
        # witness the whole structured recipe falls back, preserving the trial.
        detail["eta"] = values[f.key + ".eta"]
        return values, detail

    def sample_relative(self, study, trial, search_space):
        broad = super().sample_relative(study, trial, search_space)
        base = copy.deepcopy(self.last_diagnostic)
        recipe = RECIPES[trial.number % len(RECIPES)]
        self.last_diagnostic = {
            "schema": SCHEMA,
            "trial": trial.number,
            "base": base,
            "recipe": recipe,
            "fit_witness_id": self.prior.binding_id,
            "conditional_parameters": {},
        }
        if recipe == "broad" or len(self.prior.variables) > 2:
            self.last_diagnostic["branch"] = "original_full_axis_TPE"
            return broad
        delegate_before = copy.deepcopy(self.delegate)

        def fallback(reason):
            self.delegate = delegate_before
            self.last_diagnostic.update(
                branch="original_full_axis_TPE",
                conditioning_fallback=reason,
                conditional_parameters={},
                fallback_delegate_restored=True,
            )
            return broad

        prior = self.prior
        cycle = trial.number // len(RECIPES)
        offset = (
            prior.cluster_id
            if type(prior.cluster_id) is int
            else (
                int(prior.cluster_id)
                if isinstance(prior.cluster_id, str) and prior.cluster_id.isdecimal()
                else 0
            )
        )
        label = prior.labels[(cycle + offset) % len(prior.labels)]
        kind = "remove" if self.rng.random_sample() < 0.1 else "add"
        if recipe == "relation_head":
            if len(prior.variables) < 2:
                return fallback("relation recipe needs two variables")
            kinds = tuple(k for k in prior.head_kinds if k not in ("add", "remove"))
            kind = kinds[cycle % len(kinds)]
        target = prior.variables[cycle % len(prior.variables)]
        planned = {"head.kind": kind}
        source = target
        chosen, active, threshold_receipts = set(), {}, []
        reanchor_receipts = []
        cross = recipe in ("propagation", "propagation_rich", "relation_head")
        if cross:
            if not prior.attr_joins:
                return fallback("no actual nonmissing distinct-row attribute join")
            join = prior.attr_joins[int(self.rng.randint(len(prior.attr_joins)))]
            ri, left_rows, index = join
            atom = prior.relations[ri]
            left = int(left_rows[int(self.rng.randint(len(left_rows)))])
            rights = tuple(r for r in index[prior.values[atom.left_attr][left]] if r != left)
            right = int(rights[int(self.rng.randint(len(rights)))])
            assignment = {atom.left_var: left, atom.right_var: right}
            source = next(v for v in prior.variables if v != target)
            chosen.add(prior.feature_index["r%d.include" % ri])
            if kind not in ("add", "remove"):
                planned["head.pair"] = prior.pairs[cycle % len(prior.pairs)]
            else:
                facts = prior.neg if kind == "remove" else prior.pos
                supported = facts[assignment[source]] & facts[assignment[target]]
                if not supported[prior.label_index[label]]:
                    choices = np.flatnonzero(supported)
                    if not len(choices):
                        return fallback("no shared explicit fit head label at observed relation")
                    label = prior.labels[int(choices[int(self.rng.randint(len(choices)))])]
        else:
            facts = prior.neg if kind == "remove" else prior.pos
            rows = np.flatnonzero(facts[:, prior.label_index[label]])
            if not len(rows):
                return fallback("scheduled head label has no explicit fit support")
            assignment = {target: int(rows[int(self.rng.randint(len(rows)))])}
        if kind in ("add", "remove"):
            planned.update({"head.var": target, "head.label": label})

        def enable(index, values=None):
            f = prior.features[index]
            values = self._active(f, study, trial) if values is None else values
            if not prior.holds(index, assignment, values):
                return False
            chosen.add(index)
            active.update(values)
            return True

        # Condition the full TPE draw on this shared valuation, never delete ML
        # merely because it lacks a lexical or attribute companion.
        for i, f in enumerate(prior.features):
            if not broad[f.include_key]:
                continue
            needed = (
                (f.var,)
                if f.var is not None
                else (prior.relations[f.index].left_var, prior.relations[f.index].right_var)
            )
            if any(v not in assignment for v in needed):
                continue
            if f.kind == "label" and f.var == target and f.label == planned.get("head.label"):
                continue
            if f.kind == "relation" and isinstance(prior.relations[f.index], LabelCompare):
                a = prior.relations[f.index]
                if a.op == kind and planned.get("head.pair") in (
                    a.left_var + "|" + a.right_var,
                    a.right_var + "|" + a.left_var if kind == "eq" else "",
                ):
                    continue
            # Pure/multi-ML slots are genuinely ML-only. Other families remain
            # present in the full-axis broad branch and the other recipes.
            if recipe in ("pure_ml", "multi_ml") and f.kind != "model":
                continue
            enable(i)

        def add_patterns(variable, count):
            candidates = list(prior.pattern_features[variable])
            if not candidates:
                return 0
            order = self.rng.permutation(len(candidates))[: self.context_probes]
            found = 0
            for j in order:
                i = candidates[int(j)]
                if i in chosen:
                    continue
                if enable(i):
                    found += 1
                    if found == count:
                        break
            return found

        if recipe in ("text_ml", "multi_text_ml", "label_text", "propagation_rich"):
            if add_patterns(target, 2 if recipe == "multi_text_ml" else 1) < 1:
                return fallback("no matched current pattern within fixed fit probe budget")
        if recipe in ("label_ml", "label_text"):
            candidates = [
                i
                for (v, lab, pol), i in prior.label_features.items()
                if v == target and lab != label and prior.holds(i, assignment, {})
            ]
            if not candidates:
                return fallback("no distinct explicit source label for same-document label recipe")
            enable(candidates[int(self.rng.randint(len(candidates)))], {})
        if recipe in ("propagation", "propagation_rich"):
            index = prior.label_features[(source, label, "neg" if kind == "remove" else "pos")]
            if not enable(index, {}):
                raise RuntimeError("declared shared source fact vanished")
        if recipe in (
            "pure_ml",
            "multi_ml",
            "text_ml",
            "multi_text_ml",
            "label_ml",
            "propagation_rich",
        ):
            if not prior.models:
                return fallback("no model in current Si")
            count = 2 if recipe == "multi_ml" and len(prior.models) > 1 else 1
            order = self.rng.permutation(len(prior.models))[:count]
            for position, mi in enumerate(order):
                # First uses same-label model; second may use any model label.
                ml_label = (
                    label
                    if position == 0 or self.rng.random_sample() < 0.5
                    else prior.labels[int(self.rng.randint(len(prior.labels)))]
                )
                i = self.model_features[(target, int(mi), ml_label)]
                if i in chosen:
                    continue
                values, receipt = self._threshold(
                    i, chosen, active, planned, assignment, study, trial
                )
                threshold_receipts.append(receipt)
                if not enable(i, values):
                    if cross:
                        return fallback(
                            "fit-calibrated model atom does not hold at selected relation witness"
                        )
                    proposed_active = dict(active, **values)
                    support = self._local_mask(chosen | {i}, proposed_active, target)
                    support &= (prior.neg if kind == "remove" else prior.pos)[
                        :, prior.label_index[label]
                    ]
                    rows = np.flatnonzero(support)
                    if not len(rows):
                        return fallback(
                            "fit-calibrated complete body has zero explicit head support"
                        )
                    previous = assignment[target]
                    assignment[target] = int(rows[int(self.rng.randint(len(rows)))])
                    reanchor_receipts.append(
                        {
                            "old_row_id": prior.rows[previous],
                            "new_row_id": prior.rows[assignment[target]],
                            "eligible_rows": len(rows),
                        }
                    )
                    if not enable(i, values):
                        raise RuntimeError("fit body reanchoring lost its exact support")
        if not chosen:
            return fallback("no substantive body retained")
        planned.update(active)
        if not all(prior.holds(i, assignment, planned) for i in chosen):
            raise RuntimeError("rich proposal lost its shared fit witness")
        result = {f.include_key: int(i in chosen) for i, f in enumerate(prior.features)}
        self.last_diagnostic.update(
            branch="rich_fit_conditioned_TPE",
            conditional_parameters=planned,
            witness={v: prior.rows[r] for v, r in assignment.items()},
            source_variable=source,
            target_variable=target,
            threshold_receipts=threshold_receipts,
            joint_fit_body_witness_verified=True,
            fit_reanchoring=reanchor_receipts,
            selected_body_indicators=len(chosen),
            known_head_used_as_target_body=False,
            no_context_ML_deleted=True,
        )
        return result

    def sample_independent(self, study, trial, param_name, param_distribution):
        diagnostic = self.last_diagnostic or {}
        planned = diagnostic.get("conditional_parameters", {})
        if diagnostic.get("trial") == trial.number and param_name in planned:
            value = planned[param_name]
            if not param_distribution._contains(param_distribution.to_internal_repr(value)):
                raise ValueError("fit-conditioned parameter escaped original domain")
            return value
        return super().sample_independent(study, trial, param_name, param_distribution)
