"""Thirty fit-derived ordinary BO starts with complete preselection receipts.

Only current declared Pi/Si and explicit complete fit roots are read. No Chase
or dev score is called. Every retained start consumes a normal BO trial and is
subject to both original evaluator stages; nothing is injected into Sigma.
"""

import copy
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from loris.predicates import FreqPredicate
from loris._engine.compact_storage import ParameterCodec
from loris._engine.rlearner import immutable_json, digest
from .sampler import source_record, pattern_mask, _at_or_above

SCHEMA = "paper36_rich_fit_initialization_20260916_v1"


class _TrackedParameters(dict):
    def __init__(self, initial, codec):
        super().__init__(initial)
        self.codec = codec
        self.changed_bits = {}
        self.deleted_bits = set()
        self.active = {k: v for k, v in initial.items() if k not in codec.key_set}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key in self.codec.key_set:
            self.deleted_bits.discard(key)
            self.changed_bits[key] = value
        else:
            self.active[key] = value

    def __delitem__(self, key):
        super().__delitem__(key)
        if key in self.codec.key_set:
            self.changed_bits.pop(key, None)
            self.deleted_bits.add(key)
        else:
            self.active.pop(key, None)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *default):
        if len(default) > 1:
            raise TypeError('pop expected at most 2 arguments')
        if key in self:
            value = self[key]
            del self[key]
            return value
        if default:
            return default[0]
        raise KeyError(key)

    def popitem(self):
        if not self:
            raise KeyError('popitem(): dictionary is empty')
        key = next(reversed(self))
        return key, self.pop(key)

    def clear(self):
        for key in list(self):
            del self[key]


class _ReusableParameterBuffer:
    def __init__(self, space, codec, variable):
        self.space, self.codec, self.variable = space, codec, variable
        self.params = None
        self.index = {key: i for i, key in enumerate(codec.include_keys)}
        self.frame = None

    def reset(self, label):
        if self.params is None:
            # Exactly the same real-space constructor as the original first
            # candidate, and one complete original codec validation of zeros.
            initial = self.space.empty_parameters('add', label=label, variable=self.variable)
            self.frame = self.codec.encode(initial)
            if self.frame['on_indices']:
                raise ValueError('the initializer requires its actual all-zero template')
            self.params = _TrackedParameters(initial, self.codec)
        else:
            p = self.params
            for key in set(p.changed_bits) | p.deleted_bits:
                dict.__setitem__(p, key, 0)
            for key in p.active:
                dict.__delitem__(p, key)
            p.changed_bits.clear()
            p.deleted_bits.clear()
            p.active.clear()
            p.update({'head.kind': 'add', 'head.label': label, 'head.var': self.variable})
        return self.params

    def encode(self):
        p = self.params
        if p.deleted_bits:
            raise ValueError('every include bit must be an explicit integer zero or one')
        on = []
        for key, value in p.changed_bits.items():
            if type(value) is not int or value not in (0, 1):
                raise ValueError('every include bit must be an explicit integer zero or one')
            if value:
                on.append(self.index[key])
        value = dict(self.frame, on_indices=sorted(on), active_parameters=dict(p.active))
        # The unchanged full axis is already validated. This still checks the
        # exact compact schema, active fields and finite JSON representation.
        return self.codec.validate(value)



class FitInitializedSpace:
    def __init__(self, base, directory):
        self._base = base
        self._directory = Path(directory).resolve()
        self._complete = json.loads((self._directory / "COMPLETE.json").read_text())
        self._complete_file = source_record(self._directory / "COMPLETE.json")
        self._seeds = json.loads((self._directory / "SEEDS.json").read_text())["seeds"]
        self._complete_id = digest(self._complete)
        self._seed_id = digest(self._seeds)
        self._validate()

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _validate(self):
        if (
            digest(self._complete) != self._complete_id
            or digest(self._seeds) != self._seed_id
            or source_record(self._directory / "COMPLETE.json") != self._complete_file
            or self._complete["source"] != source_record(__file__)
            or self._complete["space_manifest_id"] != digest(self._base.manifest())
        ):
            raise ValueError("fit initialization binding changed")
        for record in self._complete["files"]:
            if source_record(record["path"]) != record:
                raise ValueError("fit initialization artifact changed")

    def manifest(self):
        self._validate()
        return dict(
            self._base.manifest(),
            initialization_override={
                "schema": SCHEMA,
                "receipt": source_record(self._directory / "COMPLETE.json"),
                "seeds_counted_in_BO_budget": True,
                "direct_Sigma_injection": False,
            },
        )

    def iter_initial_configurations(self, count, *, seed, **kwargs):
        self._validate()
        original = self._base.iter_initial_configurations(count, seed=seed, **kwargs)
        for i, fallback in enumerate(original):
            yield copy.deepcopy(self._seeds[i]["parameters"]) if i < len(self._seeds) else fallback

    def initial_configurations(self, count, *, seed, **kwargs):
        return list(self.iter_initial_configurations(count, seed=seed, **kwargs))


def build_fit_initialized_space(
    space,
    prior,
    *,
    output_dir,
    initializer_slots=30,
    label_rotation="axis",
    rotation_seed=42,
    cluster_position=0,
    initializer_candidate_archive="all",
):
    if type(initializer_slots) is not int or not 0 <= initializer_slots <= 30:
        raise ValueError("fit-derived startup slots must be in [0,30]")
    if label_rotation not in ("axis", "seeded_cluster_rotation"):
        raise ValueError("explicit axis or seeded_cluster_rotation initialization required")
    if type(rotation_seed) is not int or not 0 <= rotation_seed < 2**32:
        raise ValueError("valid explicit RandomState seed required")
    if type(cluster_position) is not int or cluster_position < 0:
        raise ValueError("nonnegative frozen cluster position required")
    if initializer_candidate_archive not in ("all", "evaluated_with_analytic_skips"):
        raise ValueError("explicit supported initializer candidate archive required")
    analytic_skips = initializer_candidate_archive == "evaluated_with_analytic_skips"
    candidate_filename = ("EVALUATED_FIT_CANDIDATES.jsonl" if analytic_skips
                          else "ALL_FIT_CANDIDATES.jsonl")
    options = dict(
        initializer_slots=initializer_slots,
        label_rotation=label_rotation,
        rotation_seed=rotation_seed,
        cluster_position=cluster_position,
        initializer_candidate_archive=initializer_candidate_archive,
    )
    prior.validate(space)
    if not np.all(prior.pos | prior.neg):
        raise ValueError(
            "this initialization protocol requires explicitly complete signed fit answers"
        )
    directory = Path(output_dir)
    if (directory / "COMPLETE.json").exists():
        if (
            json.loads((directory / "COMPLETE.json").read_text()).get("initialization_options")
            != options
        ):
            raise ValueError("resumed fit initialization options differ")
        return FitInitializedSpace(space, directory)
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    protocol = {
        "schema": SCHEMA,
        "source": source_record(__file__),
        "space_id": space.space_id,
        "fit_witness_id": prior.binding_id,
        "fit_witness_binding": prior.binding,
        "max_seeds": initializer_slots,
        "max_per_label": 1,
        "initialization_options": options,
        "label_rotation_role": "startup proposal order only; does not restrict BO label/head domain",
        "candidate_domain": "pureML per Si; each original Pi default instance + each Si; each model pair with first eta=.5 when legal, else domain midpoint, then conditioned second boundary",
        "threshold": "smallest domain eta strictly above highest explicit negative fit probability in current same-document scope; domain minimum if no negative; one threshold per candidate, no scored grid",
        "ranking": "per-label fit F1, TP, fewer body predicates, stable generation order",
        "selection_requirement": "positive fit support and no direct explicit-negative root collision on complete fit; initializer only, not universal correctness claim or later BO gate",
        "unknown_as_negative": False,
        "dev_or_test_truth_read": False,
        "BO_calls": 0,
        "Chase_calls": 0,
        "seed_budget": "first min(30,budget) existing startup slots",
        "remaining_starts": "original optional-context sparse initializer at the same positions",
        "ML_grammar_domain_unchanged": True,
        "Pi_domain_unchanged": True,
        "candidate_archive": initializer_candidate_archive,
        "candidate_rows": candidate_filename,
        "analytic_skip_scope": (
            "same-document Text AND model positive-label startups with no positive fit root inside Text scope; "
            "such starts cannot have positive fit support or become eligible; original grid numbering retained"
            if analytic_skips else None
        ),
    }
    if initializer_slots == 0:
        protocol.update(
            fit_grid_skipped_because_zero_slots=True,
            fit_grid_skip_reason=(
                "initializer_slots=0 requests no fit-derived startup replacements; "
                "preserve original BO fallback starts without enumerating fit candidates"
            ),
            fit_grid_candidate_count=0,
            seed_budget="zero fit-derived replacements; original startup sequence unchanged",
        )
    immutable_json(directory / "PROTOCOL.json", protocol)
    codec = ParameterCodec.from_space(space)
    immutable_json(
        directory / "PARAMETER_AXIS.json",
        {
            "space_id": space.space_id,
            "include_keys": list(codec.include_keys),
            "codec": "paper36_lossless_zero_default_parameters_v1",
        },
    )
    if initializer_slots == 0:
        # The parameter axis is still complete, but there is no seed to select.
        # In particular, do not construct masks or enumerate the Pi x Si x label
        # fit grid merely to discard every resulting winner.
        with (directory / candidate_filename).open("x"):
            pass
        if analytic_skips:
            immutable_json(directory / "ANALYTICALLY_SKIPPED.json", {
                "patterns": [], "model_indices": list(range(len(space.models))),
                "fit_grid_skipped_because_zero_slots": True,
                "reason": "No startup replacements requested; the grid was not enumerated",
            })
        label_order = list(range(len(space.labels)))
        if label_rotation == "seeded_cluster_rotation":
            label_order = (
                np.random.RandomState(rotation_seed).permutation(len(space.labels)).tolist()
            )
        skipped = {
            "fit_grid_skipped_because_zero_slots": True,
            "fit_grid_skip_reason": protocol["fit_grid_skip_reason"],
            "fit_grid_candidate_count": 0,
        }
        immutable_json(
            directory / "SEEDS.json",
            dict(
                seeds=[],
                count=0,
                label_order=label_order,
                selected_label_indices=[],
                eligible_but_outside_startup_budget=[],
                labels_without_seed=[],
                labels_not_evaluated=list(space.labels),
                **skipped,
            ),
        )
        immutable_json(
            directory / "COMPLETE.json",
            dict(
                schema=SCHEMA,
                source=source_record(__file__),
                space_manifest_id=digest(space.manifest()),
                fit_witness_id=prior.binding_id,
                counts=dict(
                    total=0,
                    zero_support=0,
                    boundary_outside_domain=0,
                    direct_root_conflict=0,
                    eligible=0,
                ),
                seed_count=0,
                initialization_options=options,
                elapsed_seconds=time.monotonic() - start,
                files=[source_record(p) for p in sorted(directory.iterdir()) if p.is_file()],
                BO_calls=0,
                Chase_calls=0,
                direct_rules_imported=0,
                **skipped,
            ),
        )
        return FitInitializedSpace(space, directory)
    variable = space.variables[0]
    mfeatures = {
        (f.index, f.label): f for f in space.features if f.kind == "model" and f.var == variable
    }
    pfeatures = {f.index: f for f in space.features if f.kind == "pattern" and f.var == variable}
    allmask = np.ones(len(prior.rows), bool)
    masks = [pattern_mask(prior, i) for i in range(len(space.patterns))]
    has_positive = None
    if analytic_skips:
        has_positive = [prior.pos[mask].any(axis=0) for mask in masks]
        immutable_json(directory / "ANALYTICALLY_SKIPPED.json", {
            "reason": "Text mask has no positive fit root for this add-head label; no ML conjunction can create positive fit support",
            "scope": "single same-document Text plus one model positive-head startup only",
            "original_reason_not_computed": True,
            "model_indices": list(range(len(space.models))),
            "candidate_number_formula": "label_index * (M*(P+1)+M*(M-1)//2) + model_index*(P+1) + 1 + pattern_index",
            "M": len(space.models), "P": len(space.patterns),
            "patterns": [{"pattern_index": pi, "label_indices": np.flatnonzero(~positive).tolist()}
                         for pi, positive in enumerate(has_positive) if not positive.all()],
        })
    counts, winners = {
        "total": 0,
        "zero_support": 0,
        "boundary_outside_domain": 0,
        "direct_root_conflict": 0,
        "eligible": 0,
    }, {}

    # A label is visited once in fixed order. Retain only its model columns,
    # avoiding repeated float64 conversion for every pattern without caching
    # the full model x label x document product again.
    probability_columns = {}
    positive_counts = prior.pos.sum(axis=0)

    if analytic_skips:
        counts["analytically_omitted_zero_text_positive"] = 0

    def probability(mi, li):
        key = (mi, li)
        if key not in probability_columns:
            model = space.models[mi]
            value = (
                prior.probabilities[model.alias][:, li].astype(np.float64)
                if model.dynamic
                else prior.decisions[model.alias][:, li]
            )
            value.setflags(write=False)
            probability_columns[key] = value
        return probability_columns[key]

    def boundary(mi, li, scope):
        model = space.models[mi]
        if not model.dynamic:
            return None, probability(mi, li), None
        scores = probability(mi, li)
        opposite = scope & prior.neg[:, li]
        raw = (
            float(np.nextafter(np.max(scores[opposite]), np.inf))
            if np.any(opposite)
            else (
                min(model.threshold_domain.values)
                if model.threshold_domain.values
                else model.threshold_domain.low
            )
        )
        eta = _at_or_above(model.threshold_domain, raw)
        return eta, np.zeros_like(scope) if eta is None else scores >= eta, raw

    parameter_buffer = _ReusableParameterBuffer(space, codec, variable)
    with (directory / candidate_filename).open("x") as stream:

        def candidate(li, model_indices, pi=None, first_eta=None):
            if analytic_skips and pi is not None and len(model_indices) == 1 and not has_positive[pi][li]:
                counts["total"] += 1
                counts["analytically_omitted_zero_text_positive"] += 1
                return
            label = space.labels[li]
            params = parameter_buffer.reset(label)
            scope = allmask.copy() if pi is None else masks[pi].copy()
            body_size = len(model_indices) + int(pi is not None)
            if pi is not None:
                f = pfeatures[pi]
                pred = space.patterns[pi].predicate
                params[f.include_key] = 1
                if isinstance(pred, FreqPredicate):
                    params[f.key + ".op"] = pred.op
                    params[f.key + ".eta"] = pred.eta
            thresholds = []
            legal = True
            for position, mi in enumerate(model_indices):
                model = space.models[mi]
                f = mfeatures[(mi, label)]
                params[f.include_key] = 1
                if first_eta is not None and position == 0 and model.dynamic:
                    eta = first_eta
                    firing = probability(mi, li) >= eta
                    raw = None
                else:
                    eta, firing, raw = boundary(mi, li, scope)
                if model.dynamic:
                    if eta is None:
                        legal = False
                        break
                    params[f.key + ".eta"] = eta
                thresholds.append({"alias": model.alias, "eta": eta, "raw_boundary": raw})
                scope &= firing
            tp = int(np.sum(scope & prior.pos[:, li])) if legal else 0
            fp = int(np.sum(scope & prior.neg[:, li])) if legal else 0
            fn = int(positive_counts[li]) - tp
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            reason = (
                "boundary_outside_domain"
                if not legal
                else "direct_root_conflict" if fp else "zero_support" if not tp else "eligible"
            )
            counts["total"] += 1
            counts[reason] += 1
            number = counts["total"] - 1
            record = {
                "candidate_number": number,
                "label_index": li,
                "label": label,
                "pattern_index": pi,
                "model_indices": list(model_indices),
                "thresholds": thresholds,
                "fit_tp": tp,
                "fit_fp": fp,
                "fit_fn": fn,
                "fit_f1": f1,
                "reason": reason,
                "parameters": parameter_buffer.encode() if legal else None,
            }
            stream.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
            if reason == "eligible":
                # The compact path validates the full axis binding plus every
                # active domain and ordinary rule-form guard. Omitted bits are
                # certified zeros; this is not a relaxed grammar or admission.
                compact_factory = getattr(space, "_instantiate_compact", None)
                rule = (compact_factory(record["parameters"])
                        if compact_factory is not None else space.instantiate(params))
                rank = (f1, tp, -body_size, -number)
                if li not in winners or rank > winners[li][0]:
                    winners[li] = (rank, dict(record, parameters=dict(params), rule=rule.to_dict()))

        for li in range(len(space.labels)):
            probability_columns.clear()
            for mi in range(len(space.models)):
                candidate(li, (mi,))
                for pi in range(len(space.patterns)):
                    candidate(li, (mi,), pi)
            for a, b in itertools.combinations(range(len(space.models)), 2):
                model = space.models[a]
                eta = None
                if model.dynamic:
                    domain = model.threshold_domain
                    eta = (
                        0.5
                        if domain.contains(0.5)
                        else (
                            domain.values[len(domain.values) // 2]
                            if domain.values
                            else (domain.low + domain.high) / 2
                        )
                    )
                    if domain.integer:
                        eta = int(eta)
                candidate(li, (a, b), first_eta=eta)
    label_order = list(range(len(space.labels)))
    if label_rotation == "seeded_cluster_rotation":
        permutation = np.random.RandomState(rotation_seed).permutation(len(space.labels)).tolist()
        # Match the established Reuters protocol at slots=30; the offset is
        # tied to declared startup slots, never to observed label effects.
        offset = (cluster_position * initializer_slots) % len(permutation)
        label_order = permutation[offset:] + permutation[:offset]
    eligible = [i for i in label_order if i in winners]
    chosen = eligible[:initializer_slots]
    seeds = [winners[i][1] for i in chosen]
    immutable_json(
        directory / "SEEDS.json",
        {
            "seeds": seeds,
            "count": len(seeds),
            "label_order": label_order,
            "selected_label_indices": chosen,
            "eligible_but_outside_startup_budget": [
                winners[i][1] for i in eligible[initializer_slots:]
            ],
            "labels_without_seed": [
                space.labels[i] for i in range(len(space.labels)) if i not in winners
            ],
        },
    )
    complete = {
        "schema": SCHEMA,
        "source": source_record(__file__),
        "space_manifest_id": digest(space.manifest()),
        "fit_witness_id": prior.binding_id,
        "counts": counts,
        "seed_count": len(seeds),
        "initialization_options": options,
        "elapsed_seconds": time.monotonic() - start,
        "files": [source_record(p) for p in sorted(directory.iterdir()) if p.is_file()],
        "BO_calls": 0,
        "Chase_calls": 0,
        "direct_rules_imported": 0,
    }
    immutable_json(directory / "COMPLETE.json", complete)
    return FitInitializedSpace(space, directory)
