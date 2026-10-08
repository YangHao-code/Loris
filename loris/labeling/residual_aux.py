"""Opt-in signed Aux rules with a gate on the complete calibration closure.

Threshold fitting proposes rules; only a fresh ordinary Chase admits a model.
The original program, probability tables, Gamma, arbitration and fallback stay
fixed. Saved probabilities bind a checkpoint but do not prove its execution.
"""
import copy
import json
from pathlib import Path
import pickle

import numpy as np

from loris._engine.rdl_reference import Doc, Label, ML, RefGamma, Rule
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._engine.typed_session import digest, seed_identity
from loris._engine.instance_scope import scope_corpus_identity
from .probability_tables import freeze_probability, probability_identity

POSITIVE_GRID = (.05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .99, 1.)
NEGATIVE_GRID = (None, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90)
QUANTILE_ANCHORS = (.005, .01, .015, .02, .025, .03, .04, .05, .10, .20, .30,
                    .40, .50, .60, .70, .80, .90, .95, .99, 1.)
EPS = 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _boolean(value, shape, name):
    value = np.asarray(value)
    _require(value.dtype == np.bool_ and value.shape == shape, name + " Boolean axes differ")
    return value


def _probability(value, shape):
    value = np.asarray(value)
    _require(value.dtype in (np.float32, np.float64) and value.shape == shape
             and np.isfinite(value).all() and not ((value < 0) | (value > 1)).any(),
             "Complete finite native Aux probability axes required")
    return freeze_probability(value)


def _counts(prediction, truth):
    return np.stack(((prediction & truth).sum(0), (prediction & ~truth).sum(0),
                     (~prediction & truth).sum(0))).astype(np.int64)


def _f1(counts):
    denominator = 2 * counts[0] + counts[1] + counts[2]
    return np.divide(2 * counts[0], denominator, out=np.zeros_like(denominator, float),
                     where=denominator > 0)


def _micro(counts):
    total = counts.sum(1)
    denominator = 2 * total[0] + total[1] + total[2]
    return float(2 * total[0] / denominator) if denominator else 0.


def quality(prediction, truth):
    truth = np.asarray(truth)
    _require(truth.ndim == 2 and min(truth.shape) > 0, "Nonempty complete calibration axes required")
    truth = _boolean(truth, truth.shape, "Calibration truth")
    prediction = _boolean(prediction, truth.shape, "Actual endpoint")
    counts = _counts(prediction, truth)
    return dict(macro_f1=float(_f1(counts).mean()), micro_f1=_micro(counts),
                TP=int(counts[0].sum()), FP=int(counts[1].sum()), FN=int(counts[2].sum()))


def dual_gate(candidate, incumbent):
    values = [q[key] for q in (candidate, incumbent) for key in ("macro_f1", "micro_f1")]
    _require(all(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1 for v in values),
             "Finite calibration macro/micro required")
    return (candidate["macro_f1"] + EPS >= incumbent["macro_f1"]
            and candidate["micro_f1"] + EPS >= incumbent["micro_f1"]
            and candidate["macro_f1"] + candidate["micro_f1"] >
            incumbent["macro_f1"] + incumbent["micro_f1"] + EPS)


def _proposal_policy(value=None, *, minimum_removals=None, maximum_true_loss=None):
    """Explicit presets; the default retains the historical coarse policy."""
    presets = {
        "coarse": dict(minimum_removals=5, maximum_true_loss=.10,
                       rare_loss_allowance=0, quantile_points=0),
        "quantile41_rare_allowance_one": dict(minimum_removals=3, maximum_true_loss=.10,
                                             rare_loss_allowance=1, quantile_points=41),
    }
    value = "coarse" if value is None else value
    if isinstance(value, str):
        _require(value in presets, "Unknown residual proposal policy")
        result = dict(presets[value], name=value)
    else:
        _require(isinstance(value, dict) and isinstance(value.get("name"), str)
                 and not set(value) - {"name", "minimum_removals", "maximum_true_loss",
                                       "rare_loss_allowance", "quantile_points"}
                 and value["name"] in presets, "Explicit residual proposal configuration required")
        result = dict(presets[value["name"]], name=value["name"])
        result.update(copy.deepcopy(value))
    for key, override in (("minimum_removals", minimum_removals),
                          ("maximum_true_loss", maximum_true_loss)):
        if override is not None:
            _require(not isinstance(value, dict) or key not in value or value[key] == override,
                     "Conflicting residual screening bounds")
            result[key] = override
    _require(type(result["minimum_removals"]) is int and result["minimum_removals"] >= 1
             and type(result["maximum_true_loss"]) in (int, float)
             and np.isfinite(result["maximum_true_loss"]) and 0 <= result["maximum_true_loss"] <= 1
             and type(result["rare_loss_allowance"]) is int
             and type(result["quantile_points"]) is int,
             "Explicit residual screening bounds required")
    preset = presets[result["name"]]
    _require(result["rare_loss_allowance"] == preset["rare_loss_allowance"]
             and result["quantile_points"] == preset["quantile_points"],
             "Named policy fixes its quantile count and rare-label allowance")
    result["maximum_true_loss"] = float(result["maximum_true_loss"])
    return result


def _quantile_grid(values, points=41):
    """Adjacent widened-probability midpoints, never rounded to float32."""
    values = np.unique(np.asarray(values, dtype=np.float64))
    boundaries = (values[:-1] + values[1:]) * .5 if len(values) > 1 else values
    if len(boundaries):
        try:
            selected = np.quantile(boundaries, np.linspace(0, 1, points), method="nearest")
        except TypeError:  # NumPy versions before the method keyword.
            selected = np.quantile(boundaries, np.linspace(0, 1, points), interpolation="nearest")
    else:
        selected = np.asarray([], dtype=np.float64)
    return np.unique(np.r_[selected, QUANTILE_ANCHORS])


def fit_thresholds(old, probabilities, truth, *, minimum_removals=None,
                   maximum_true_loss=None, proposal_policy=None):
    """Vectorized per-label proposal; this matrix is never the admission score.

    Coarse screening is backward compatible. The named quantile preset adds
    41 per-label midpoint quantiles and explicitly allows one lost true label
    when the ten-percent bound would otherwise be zero. Native cold Chase,
    including original conflict arbitration/fallback, remains the final gate.
    """
    _require(np.asarray(old).ndim == 2 and min(np.asarray(old).shape) > 0,
             "Nonempty complete residual axes required")
    truth = _boolean(truth, np.asarray(old).shape, "Calibration truth")
    old = _boolean(old, truth.shape, "Old complete endpoint")
    p = _probability(probabilities, truth.shape).astype(np.float64)
    policy = _proposal_policy(proposal_policy, minimum_removals=minimum_removals,
                              maximum_true_loss=maximum_true_loss)
    current = _counts(old, truth); choices = []
    for column in range(old.shape[1]):
        before = float(_f1(current[:, column:column + 1])[0]); options = []
        if policy["quantile_points"]:
            positives = _quantile_grid(p[~old[:, column], column], policy["quantile_points"])
            negatives = _quantile_grid(p[old[:, column], column], policy["quantile_points"])
            negatives = np.r_[-1., negatives[negatives < 1.]]
        else:
            positives = np.asarray(POSITIVE_GRID, dtype=np.float64)
            negatives = np.asarray([-1. if v is None else v for v in NEGATIVE_GRID], dtype=np.float64)
        added = (~old[:, column])[None, :] & (p[:, column][None, :] >= positives[:, None])
        removed = old[:, column][None, :] & (p[:, column][None, :] <= negatives[:, None])
        atp = (added & truth[:, column][None, :]).sum(1)
        afp = (added & ~truth[:, column][None, :]).sum(1)
        rtp = (removed & truth[:, column][None, :]).sum(1)
        rfp = (removed & ~truth[:, column][None, :]).sum(1)
        counts = np.stack(np.broadcast_arrays(current[0, column] + atp[:, None] - rtp[None, :],
            current[1, column] + afp[:, None] - rfp[None, :],
            current[2, column] - atp[:, None] + rtp[None, :]))
        scores = _f1(counts)
        true_count = int(truth[:, column].sum())
        maximum_loss = max(policy["rare_loss_allowance"],
                           int(np.floor(true_count * policy["maximum_true_loss"] + EPS)))
        if policy["name"] == "coarse":
            true_loss_safe = rtp / true_count <= policy["maximum_true_loss"] if true_count else np.zeros_like(rtp, bool)
        else:
            true_loss_safe = rtp <= maximum_loss
        safe_negative = ((negatives < 0) | (((rtp + rfp) >= policy["minimum_removals"])
                         & true_loss_safe & (true_count > 0)))
        valid = ((negatives[None, :] < positives[:, None]) & safe_negative[None, :]
                 & (scores > before + EPS))
        for a, b in np.argwhere(valid):
            options.append((float(scores[a, b]), float(positives[a]), float(negatives[b]), counts[:, a, b].copy()))
        options.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
        if options:
            choices.append((options[0][0] - before, column, options))
    choices.sort(key=lambda x: (-x[0], x[1]))
    positive, negative = {}, {}
    for _, column, options in choices:
        for _, pos, neg, counts in options:
            proposed = current.copy(); proposed[:, column] = counts
            if _micro(proposed) + EPS < _micro(current):
                continue
            current = proposed; positive[column] = pos
            if neg >= 0:
                negative[column] = neg
            break
    return positive, negative


def aliases(aux_alias):
    _require(isinstance(aux_alias, str) and aux_alias.strip(), "Independent new Aux alias required")
    return dict(positive=aux_alias, complement=aux_alias + "__complement",
                old_present=aux_alias + "__old_present", old_absent=aux_alias + "__old_absent",
                scope=aux_alias + "__scope")


def residual_rules(prefix, labels, names, positive, negative):
    """Ordinary Doc/ML rules, with no old-output Label premise or new cycle."""
    _require(not (set(positive) | set(negative)) - set(labels), "Residual cut outside native labels")
    tail = []
    for label in labels:
        scope = ML("x", names["scope"], label, .5, dynamic=False)
        if label in positive:
            tail.append(Rule((Doc("x"), ML("x", names["positive"], label, positive[label]),
                              ML("x", names["old_absent"], label, .5, dynamic=False), scope),
                             Label("x", label)))
        if label in negative:
            _require(0 <= negative[label] < positive.get(label, 1.), "Negative cut must be below positive")
            tail.append(Rule((Doc("x"), ML("x", names["complement"], label, 1. - negative[label]),
                              ML("x", names["old_present"], label, .5, dynamic=False), scope),
                             Label("x", label, "neg")))
    return tuple(prefix) + tuple(tail)


def _policy(value, contexts, labels):
    _require(isinstance(value, dict) and set(value) == {"conflict_policy", "gamma_wins", "arbitration", "fallback"}
             and value["conflict_policy"] in ("halt", "unknown_arbitrate")
             and type(value["gamma_wins"]) is bool, "Explicit frozen endpoint policy required")
    value = copy.deepcopy(value)
    for key in ("arbitration", "fallback"):
        config = value[key]
        if config is None:
            continue
        _require(set(config) == {"alias", "default_threshold", "thresholds"}
                 and all(config["alias"] in ctx["model_probabilities"] for ctx in contexts.values())
                 and isinstance(config["thresholds"], dict)
                 and not set(config["thresholds"]) - set(labels), "Frozen " + key + " binding differs")
        for threshold in [config["default_threshold"], *config["thresholds"].values()]:
            _require(type(threshold) in (int, float) and np.isfinite(threshold) and 0 <= threshold <= 1,
                     "Finite frozen endpoint thresholds required")
    _require(value["conflict_policy"] != "unknown_arbitrate" or value["arbitration"] is not None,
             "Unknown arbitration requires its original frozen model")
    return value


def _cold(context, rules, policy, cache):
    from .arbitration import arbitrate, paired_rules
    from .conflict_protocol import available_predictions, unknown_mask
    from .fallback import fill
    from .unknown_incremental import retaining_chase
    runner = IndexedChaseRunner(context, cache, gamma_protection=policy["gamma_wins"])
    try:
        runner.precompute(rules)
        if policy["conflict_policy"] == "unknown_arbitrate":
            closure = retaining_chase(runner, rules, context["gamma"], before=None,
                gamma_wins=policy["gamma_wins"], root_storage="compact")
        else:
            closure = runner(rules, context["gamma"])
        _require(closure.status == "fixpoint", "Actual complete residual Chase failed")
        unknown = unknown_mask(closure); arbitration = None
        if policy["arbitration"] is not None:
            config = policy["arbitration"]
            paired = tuple(rule for label in context["labels"] for rule in paired_rules(
                config["alias"], label, config["thresholds"].get(label, config["default_threshold"])))
            result = arbitrate(paired, context["model_probabilities"], unknown,
                              context["labels"], model_state="post", model_version="frozen")
            closure.arbitration_positive, closure.arbitration_negative = result.positive, result.negative
            arbitration = (result.positive, result.negative)
        if policy["fallback"] is not None:
            config = policy["fallback"]
            cuts = np.asarray([config["thresholds"].get(label, config["default_threshold"])
                               for label in context["labels"]], float)
            closure.fallback_positive = fill(closure.positive_facts, closure.negative_facts,
                unknown, context["gamma"].neg, context["model_probabilities"][config["alias"]], cuts, arbitration)
        return freeze_probability(available_predictions(closure).astype(bool))
    finally:
        runner.close()


class ResidualAuxWorkflow:
    """Attach ordered trained checkpoints to a frozen program using calibration only.

    ``contexts`` includes ``calibration`` and any application contexts. Calibration
    answers cover only ``calibration_rows``; no test-answer argument exists. The
    backend remains caller-owned: pass each saved checkpoint's newly predicted
    arrays to ``consider``. A rejected checkpoint leaves the accepted state intact.
    """
    def __init__(self, *, contexts, rules, alias, receiver_masks, calibration_rows,
                 calibration_truth, policy, cache_dir, minimum_removals=None,
                 maximum_true_loss=None, proposal_policy=None):
        _require("calibration" in contexts and set(contexts) == set(receiver_masks), "Explicit context scopes required")
        self.rules = tuple(Rule.from_dict(r.to_dict() if isinstance(r, Rule) else r) for r in rules)
        self.names = aliases(alias); self.contexts = {}; self.scope = {}
        self.labels = tuple(contexts["calibration"]["labels"])
        _require(self.labels and len(set(self.labels)) == len(self.labels), "Distinct native labels required")
        for name, source in contexts.items():
            shape = (len(source["docs"]), len(self.labels))
            _require(tuple(source["labels"]) == self.labels and len(source["row_ids"]) == shape[0]
                     and len(set(source["row_ids"])) == shape[0], "Context native axes differ")
            occupied = set(source["model_probabilities"]) | set(source.get("static_model_probabilities", {}))
            _require(not occupied & set(self.names.values()), "New Aux alias collides with frozen Pi")
            ctx = dict(source, docs=copy.deepcopy(source["docs"]), row_ids=list(source["row_ids"]),
                       labels=list(self.labels))
            for key in ("attribute_values", "attribute_presence", "identities", "input_files"):
                if key in source:
                    ctx[key] = copy.deepcopy(source[key])
            ctx["model_decisions"] = {a: freeze_probability(_boolean(p, shape, "Frozen decisions"))
                                      for a, p in source.get("model_decisions", {}).items()}
            for key in ("model_probabilities", "static_model_probabilities"):
                if key in ctx:
                    ctx[key] = {a: _probability(p, shape) for a, p in ctx[key].items()}
            gamma = source["gamma"]
            gp, gn = _boolean(gamma.pos, shape, "Original positive Gamma"), _boolean(gamma.neg, shape, "Original negative Gamma")
            _require(not (gp & gn).any(), "Original Gamma polarities overlap")
            ctx["gamma"] = RefGamma(freeze_probability(gp), freeze_probability(gn),
                                    tuple(getattr(gamma, "subset", ())), tuple(getattr(gamma, "strict_subset", ())))
            scope = _boolean(receiver_masks[name], (shape[0],), "Receiver scope")
            self.scope[name] = freeze_probability(scope[:, None] & ~(ctx["gamma"].pos | ctx["gamma"].neg))
            self.contexts[name] = ctx
        _require(not any(isinstance(atom, ML) and atom.alias in self.names.values()
                         for rule in self.rules for atom in rule.body), "New alias occurs in frozen prefix")
        self.policy = _policy(policy, self.contexts, self.labels)
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        shape = (len(self.contexts["calibration"]["docs"]), len(self.labels))
        self.calibration_rows = freeze_probability(_boolean(calibration_rows, (shape[0],), "Calibration role"))
        _require(self.calibration_rows.any() and not (self.contexts["calibration"]["gamma"].pos
                 | self.contexts["calibration"]["gamma"].neg)[self.calibration_rows].any(),
                 "Calibration answers cannot seed Gamma")
        self.truth = freeze_probability(_boolean(calibration_truth,
            (int(self.calibration_rows.sum()), shape[1]), "Complete calibration truth"))
        self.proposal_policy = _proposal_policy(proposal_policy, minimum_removals=minimum_removals,
                                                maximum_true_loss=maximum_true_loss)
        self.minimum_removals = self.proposal_policy["minimum_removals"]
        self.maximum_true_loss = self.proposal_policy["maximum_true_loss"]
        self.binding = self._binding()
        self.old_outputs = {name: _cold(ctx, self.rules, self.policy, self.cache / name)
                            for name, ctx in self.contexts.items()}
        self.old_output_binding = {name: probability_identity(value) for name, value in self.old_outputs.items()}
        self.outputs = dict(self.old_outputs); self.current_outputs = dict(self.old_outputs)
        self.accepted_rules = self.rules; self.current_rules = self.rules
        self.accepted_checkpoint = None; self.accepted_candidate = "frozen_old_program"
        self.best_quality = quality(self.outputs["calibration"][self.calibration_rows], self.truth)
        self.initial_quality = dict(self.best_quality)
        self.trajectory = []; self.old_closure_calls = len(self.contexts)

    def _binding(self):
        return digest(dict(program=[r.to_dict() for r in self.rules], policy=self.policy,
            proposal_policy=self.proposal_policy,
            screening_bounds=dict(minimum_removals=self.minimum_removals,
                                  maximum_true_loss=self.maximum_true_loss),
            calibration_rows=probability_identity(self.calibration_rows),
            calibration_truth=probability_identity(self.truth),
            scope={name: probability_identity(value) for name, value in self.scope.items()},
            contexts={name: dict(rows=list(ctx["row_ids"]), labels=list(ctx["labels"]),
                Gamma=seed_identity(ctx["gamma"]), corpus=scope_corpus_identity(ctx),
                context_id=ctx.get("context_id"), input_files=ctx.get("input_files", {}),
                decisions={a: probability_identity(p) for a, p in ctx["model_decisions"].items()},
                tables={key: {a: probability_identity(p)
                for a, p in ctx.get(key, {}).items()} for key in ("model_probabilities", "static_model_probabilities")})
                for name, ctx in self.contexts.items()}))

    def candidate_context(self, name, probabilities):
        ctx = self.contexts[name]; shape = self.old_outputs[name].shape
        p = _probability(probabilities, shape).astype(np.float64)
        scope, old = self.scope[name], self.old_outputs[name]
        dynamic = dict(ctx["model_probabilities"]); static = dict(ctx.get("static_model_probabilities", ctx["model_probabilities"]))
        tables = {self.names["positive"]: freeze_probability(p),
                  self.names["complement"]: freeze_probability(1. - p),
                  self.names["old_present"]: freeze_probability((old & scope).astype(float)),
                  self.names["old_absent"]: freeze_probability((~old & scope).astype(float)),
                  self.names["scope"]: freeze_probability(scope.astype(float))}
        dynamic.update(tables)
        for key in ("old_present", "old_absent", "scope"):
            static[self.names[key]] = tables[self.names[key]]
        return dict(ctx, model_probabilities=dynamic, static_model_probabilities=static)

    def consider(self, candidate_id, checkpoint, probabilities, *, training_receipt=None):
        try:
            return self._consider(candidate_id, checkpoint, probabilities, training_receipt=training_receipt)
        except Exception as error:
            if (isinstance(candidate_id, str) and candidate_id and candidate_id != "frozen_old_program"
                    and candidate_id not in {r["candidate_id"] for r in self.trajectory}):
                self.trajectory.append(dict(candidate_id=candidate_id, checkpoint=copy.deepcopy(checkpoint),
                    accepted=False, status="failed", error_type=type(error).__name__, reason=str(error),
                    threshold_proposal_policy=copy.deepcopy(self.proposal_policy),
                    retained_candidate=self.accepted_candidate, B_or_test_used_for_selection=False))
            raise

    def _consider(self, candidate_id, checkpoint, probabilities, *, training_receipt=None):
        _require(isinstance(candidate_id, str) and candidate_id and candidate_id != "frozen_old_program"
                 and candidate_id not in {r["candidate_id"] for r in self.trajectory}, "Unique ordered candidate required")
        _require(set(probabilities) == set(self.contexts), "Fresh predictions for every native context required")
        _require(isinstance(checkpoint, dict) and set(checkpoint) == {"path", "sha256"}
                 and isinstance(checkpoint["path"], str) and checkpoint["path"]
                 and isinstance(checkpoint["sha256"], str) and len(checkpoint["sha256"]) == 64
                 and all(c in "0123456789abcdef" for c in checkpoint["sha256"]),
                 "Saved checkpoint reference required")
        _require(self._binding() == self.binding, "Frozen program/Pi/Gamma/endpoint policy changed")
        _require({name: probability_identity(value) for name, value in self.old_outputs.items()}
                 == self.old_output_binding, "Frozen old endpoint constants changed")
        contexts = {name: self.candidate_context(name, probabilities[name]) for name in self.contexts}
        old = self.old_outputs["calibration"][self.calibration_rows]
        p = contexts["calibration"]["model_probabilities"][self.names["positive"]][self.calibration_rows]
        positive, negative = fit_thresholds(old, p, self.truth,
            minimum_removals=self.minimum_removals, maximum_true_loss=self.maximum_true_loss,
            proposal_policy=self.proposal_policy)
        positive = {self.labels[j]: cut for j, cut in positive.items()}
        negative = {self.labels[j]: cut for j, cut in negative.items()}
        # Widen before complementing. Casting the table or its midpoint cuts
        # back to float32 can move an inclusive negative boundary by one bin.
        # A lossy float64 complement at extreme tiny scores is rejected rather
        # than silently compiling a rule that differs from P <= its cut.
        for name, ctx in contexts.items():
            for label, cut in negative.items():
                column = self.labels.index(label)
                region = self.scope[name][:, column] & self.old_outputs[name][:, column]
                raw = ctx["model_probabilities"][self.names["positive"]][:, column]
                complement = ctx["model_probabilities"][self.names["complement"]][:, column]
                _require(np.array_equal(region & (complement >= 1. - cut), region & (raw <= cut)),
                         "Negative complement boundary cannot represent the exact probability cut")
        rules = residual_rules(self.rules, self.labels, self.names, positive, negative)
        outputs = {name: _cold(ctx, rules, self.policy, self.cache / name) for name, ctx in contexts.items()}
        actual = outputs["calibration"]
        score = quality(actual[self.calibration_rows], self.truth)
        accepted = dual_gate(score, self.best_quality)
        record = dict(candidate_id=candidate_id, status="complete", checkpoint=copy.deepcopy(checkpoint),
            training_receipt=copy.deepcopy(training_receipt), accepted=accepted,
            retained_candidate=candidate_id if accepted else self.accepted_candidate,
            incumbent_before=self.accepted_candidate, incumbent_quality=dict(self.best_quality),
            actual_complete_calibration_quality=score, positive_thresholds=positive, negative_thresholds=negative,
            complete_program_id=digest([r.to_dict() for r in rules]), frozen_binding=self.binding,
            threshold_proposal_policy=copy.deepcopy(self.proposal_policy),
            predictions={name: probability_identity(ctx["model_probabilities"][self.names["positive"]])
                         for name, ctx in contexts.items()}, B_or_test_used_for_selection=False,
            actual_complete_outputs={name: probability_identity(value) for name, value in outputs.items()},
            saved_predictions_prove_checkpoint_execution=False, pseudo_promoted_to_Gamma=False)
        _require(self._binding() == self.binding, "Closure changed frozen knowledge")
        self.current_outputs = outputs
        self.current_rules = rules
        if accepted:
            self.outputs, self.accepted_rules = outputs, rules
            self.best_quality = score; self.accepted_candidate = candidate_id
            self.accepted_checkpoint = copy.deepcopy(checkpoint)
        self.trajectory.append(record)
        return copy.deepcopy(record)


def masked_teacher_bce(logits, targets, observed, weights):
    """Optional SSL loss: detached teacher, full-cell mean, unknown gradient zero.

    Caller must certify teacher cells using calibration/training data, keep them
    outside Gamma, and replay trusted supervision separately at unit weight.
    """
    import torch
    from torch.nn import functional as F
    _require(logits.ndim == 2 and logits.numel() and logits.shape == targets.shape == observed.shape == weights.shape
             and observed.dtype == torch.bool and torch.isfinite(targets).all()
             and not ((targets != 0) & (targets != 1)).any() and torch.isfinite(weights).all()
             and not ((weights < 0) | (weights > 1)).any()
             and not ((targets > 0) & ~observed).any() and not ((weights > 0) & ~observed).any(),
             "Explicit certified teacher cell axes required")
    active = observed & (weights > 0)
    _require(torch.isfinite(logits[active]).all(), "Nonfinite active teacher logits")
    working = torch.where(active, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    values = F.binary_cross_entropy_with_logits(working, targets.detach().float(), reduction="none")
    return torch.where(active, values * weights.detach().float(), torch.zeros_like(values)).sum() / logits.numel()


def main_protocol(protocol_path, output):
    """Portable saved-checkpoint CLI; no training, annotation or test truth loader."""
    from loris.discovery.manifest import bound, file_record
    protocol_path = Path(protocol_path).resolve(strict=True); base = protocol_path.parent
    protocol = json.loads(protocol_path.read_text())
    fields = {"schema", "contexts", "program", "alias", "policy", "calibration_answers", "candidates"}
    _require(fields <= set(protocol) and not set(protocol) - fields - {"proposal_policy"}
             and protocol["schema"] == "loris_residual_aux_protocol_v1", "Explicit residual protocol required")
    destination = Path(output).resolve(); destination.mkdir(parents=True, exist_ok=False)
    contexts, masks = {}, {}
    for name, entry in protocol["contexts"].items():
        _require(set(entry) == {"context", "receiver_row_ids"}, "Explicit receiver row IDs required")
        with bound(entry["context"], base).open("rb") as stream:
            contexts[name] = pickle.load(stream)  # Only authenticated caller-owned context files.
        ids = contexts[name]["row_ids"]
        _require(len(set(entry["receiver_row_ids"])) == len(entry["receiver_row_ids"])
                 and not set(entry["receiver_row_ids"]) - set(ids), "Receiver row identity differs")
        masks[name] = np.isin(ids, entry["receiver_row_ids"])
    program = json.loads(bound(protocol["program"], base).read_text())
    rules = program["rules"] if isinstance(program, dict) else program
    cal = contexts["calibration"]
    with np.load(bound(protocol["calibration_answers"], base), allow_pickle=False) as packet:
        _require(set(packet.files) == {"row_ids", "labels", "targets", "observed"}, "Calibration packet inventory differs")
        rows = np.isin(cal["row_ids"], packet["row_ids"])
        _require(np.asarray(cal["row_ids"])[rows].tolist() == packet["row_ids"].tolist()
                 and cal["labels"] == packet["labels"].tolist()
                 and packet["observed"].dtype == np.bool_ and packet["observed"].all()
                 and packet["observed"].shape == packet["targets"].shape,
                 "Complete calibration native axes required")
        truth = packet["targets"].copy()
    workflow = ResidualAuxWorkflow(contexts=contexts, rules=rules, alias=protocol["alias"],
        receiver_masks=masks, calibration_rows=rows, calibration_truth=truth,
        policy=protocol["policy"], cache_dir=destination / "cache",
        proposal_policy=protocol.get("proposal_policy"))
    def save_outputs(prefix, outputs):
        for name, prediction in outputs.items():
            np.savez_compressed(destination / (prefix + name + "_output.npz"),
                prediction=prediction, row_ids=contexts[name]["row_ids"], labels=workflow.labels)
    def save_program(name, rules):
        (destination / name).write_text(json.dumps([r.to_dict() for r in rules], indent=2) + "\n")
    save_outputs("initial_", workflow.old_outputs)
    save_program("INITIAL_PROGRAM.json", workflow.rules)
    for index, candidate in enumerate(protocol["candidates"]):
        _require(set(candidate) == {"candidate_id", "checkpoint", "probabilities"}, "Explicit ordered saved candidate required")
        checkpoint = file_record(bound(candidate["checkpoint"], base)); probabilities = {}
        _require(set(candidate["probabilities"]) == set(contexts), "Candidate context inventory differs")
        for name, record in candidate["probabilities"].items():
            with np.load(bound(record, base), allow_pickle=False) as packet:
                _require(set(packet.files) == {"row_ids", "labels", "probabilities"}
                         and packet["row_ids"].tolist() == list(contexts[name]["row_ids"])
                         and packet["labels"].tolist() == list(contexts[name]["labels"]),
                         "Saved model probability native axes differ")
                probabilities[name] = packet["probabilities"].copy()
        try:
            record = workflow.consider(candidate["candidate_id"], checkpoint, probabilities)
        except Exception:
            (destination / "FAILED.json").write_text(json.dumps(dict(
                trajectory=workflow.trajectory, retained_candidate=workflow.accepted_candidate,
                retained_checkpoint=workflow.accepted_checkpoint, artifacts_preserved=True), indent=2) + "\n")
            raise
        save_outputs("candidate_%05d_" % index, workflow.current_outputs)
        save_outputs("retained_%05d_" % index, workflow.outputs)
        save_program("candidate_%05d_PROGRAM.json" % index, workflow.current_rules)
        save_program("retained_%05d_PROGRAM.json" % index, workflow.accepted_rules)
        (destination / ("candidate_%05d.json" % index)).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    for name, prediction in workflow.outputs.items():
        np.savez_compressed(destination / (name + "_output.npz"), prediction=prediction,
                            row_ids=contexts[name]["row_ids"], labels=workflow.labels)
    result = dict(schema="loris_residual_aux_complete_v1", accepted_candidate=workflow.accepted_candidate,
        protocol=file_record(protocol_path), initial_complete_calibration_quality=workflow.initial_quality,
        threshold_proposal_policy=dict(workflow.proposal_policy,
            positive_grid=list(POSITIVE_GRID) if not workflow.proposal_policy["quantile_points"] else None,
            negative_grid=list(NEGATIVE_GRID) if not workflow.proposal_policy["quantile_points"] else None,
            quantile_anchor_grid=list(QUANTILE_ANCHORS) if workflow.proposal_policy["quantile_points"] else None,
            gate="complete_calibration_macro_micro_nondecrease_and_strict_sum_increase", epsilon=EPS),
        accepted_checkpoint=workflow.accepted_checkpoint, actual_complete_calibration_quality=workflow.best_quality,
        trajectory=workflow.trajectory, frozen_binding=workflow.binding, old_closure_calls=workflow.old_closure_calls,
        B_or_test_used_for_selection=False, new_annotation_queries=0, training_performed_by_this_cli=False,
        saved_predictions_prove_checkpoint_execution=False)
    (destination / "ACCEPTED_PROGRAM.json").write_text(json.dumps([r.to_dict() for r in workflow.accepted_rules], indent=2) + "\n")
    (destination / "COMPLETE.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
