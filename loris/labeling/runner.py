"""Concrete bounded latest-paper loop using the real complete typed Chase."""

from dataclasses import dataclass
import copy
import gzip
import json
from pathlib import Path
import time
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from loris._engine.rdl_reference import ML, Label, Rule, RefGamma
from loris._engine.typed_runtime import IndexedChaseRunner
from loris._engine.typed_session import (
    array_identity,
    digest,
    seed_identity,
    complete_state_identity,
    plain,
)
from loris._engine.instance_scope import FrozenInstanceScope, scope_corpus_identity
from loris._learning.rdg import RuleDependencyGraph
from loris.rule_forms import validate_program_rule_forms
from .query_policy import remaining_budget, batch_size
from .probability_tables import freeze_probability, probability_identity, probability_hash
from .exports import export_facts, export_provenance, should_export
from .feedback import (
    assignments,
    incremental_batch,
    limit_training_rows,
    cumulative_verified_rows,
    batch_id,
    calibrate,
    canonical_weighting,
    RELIABILITY_WEIGHTINGS,
    DIRECT_RULE_WEIGHTINGS,
    DISCOUNT_WEIGHTINGS,
)
from .conflict_protocol import (
    POLICY,
    arbitration_rules,
    unknown_mask,
    available_predictions,
    pending_rows,
    state_identity,
)


class ResourceLimit(RuntimeError):
    pass


class FeedbackCheckInvariantError(RuntimeError):
    """Entailed corrections changed a frozen closure; this is not a quality filter."""


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(plain(value), f, indent=2, allow_nan=False)


def signed_gamma(seed):
    return RefGamma(
        np.array(seed.pos, bool, copy=True),
        np.array(seed.neg, bool, copy=True),
        tuple(getattr(seed, "subset", ())),
        tuple(getattr(seed, "strict_subset", ())),
    )


def rule_channels(rules, alias, labels):
    aliases = {alias} if isinstance(alias, str) else set(alias)
    result = {}
    for ri, rule in enumerate(rules):
        for ai, atom in enumerate(rule.body):
            if isinstance(atom, ML) and atom.alias in aliases and atom.dynamic:
                result[(ri, ai)] = (labels.index(atom.label), atom.eta)
    return result


def with_channels(rules, channels):
    records = [r.to_dict() for r in rules]
    for (ri, ai), (_, value) in channels.items():
        records[ri]["body"][ai]["eta"] = value
    return tuple(Rule.from_dict(r) for r in records)


def facts_id(result):
    return digest(
        dict(
            p=array_identity(result.positive_facts),
            n=array_identity(result.negative_facts),
            subset=result.subset_facts.identity(),
            strict=result.strict_subset_facts.identity(),
            conflict_unknown=array_identity(unknown_mask(result)),
        )
    )


def metrics(truth, prediction):
    y, p = np.asarray(truth, bool), np.asarray(prediction, bool)
    if y.shape != p.shape:
        raise ValueError("scoring axes differ")
    tp = (y & p).sum(0)
    fp = (~y & p).sum(0)
    fn = (y & ~p).sum(0)
    multi = y.sum(1) > 1

    def divide(a, b):
        return np.divide(
            a, b, out=np.zeros_like(np.asarray(a), dtype=float), where=np.asarray(b) > 0
        )

    return dict(
        macro_f1=float(np.mean(divide(2 * tp, 2 * tp + fp + fn))),
        micro_f1=float(divide(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum())),
        precision=float(divide(tp.sum(), tp.sum() + fp.sum())),
        recall=float(divide(tp.sum(), tp.sum() + fn.sum())),
        TP=int(tp.sum()),
        FP=int(fp.sum()),
        FN=int(fn.sum()),
        documents=len(y),
        covered_documents=int(p.any(1).sum()),
        coverage=float(p.any(1).mean()) if len(p) else 0.0,
        multi_label_completeness=(
            float(((y & p)[multi].sum(1) / y[multi].sum(1)).mean()) if multi.any() else None
        ),
        per_label=[
            dict(TP=int(a), FP=int(b), FN=int(c), F1=float(d))
            for a, b, c, d in zip(tp, fp, fn, divide(2 * tp, 2 * tp + fp + fn))
        ],
    )


def _table_digests(tables):
    if tables is None:
        return None
    return {alias: probability_hash(value) for alias, value in tables.items()}


@dataclass
class Version:
    probabilities: np.ndarray
    checkpoint: dict
    channels: dict
    identity: str


class LabelingRun:
    """Only the declared dynamic alias and declared ensembles can change.

    Trainer injection is a test seam; the CLI always installs train_actual.
    Oracle files are opened for one selected row only. Source review is a
    separate request, with existing trusted answers reused rather than queried.
    """

    def __init__(self, *, context, rules, manifest, output, trainer, reliability=None):
        self.context = dict(context)
        self.rules = tuple(Rule.from_dict(r.to_dict() if isinstance(r, Rule) else r) for r in rules)
        rule_form_audit = validate_program_rule_forms(self.rules)
        self.manifest = copy.deepcopy(manifest)
        self.requested_weighting = self.manifest["weighting"]
        self.manifest["weighting"] = canonical_weighting(self.requested_weighting)
        from .feedback import RECEIVER_POLICIES

        self.receiver_policy = self.manifest["feedback"].setdefault("receiver_policy", "same_model")
        if self.receiver_policy not in RECEIVER_POLICIES:
            raise ValueError("unsupported correction receiver policy")
        manifest = self.manifest
        self.aux_initialization = manifest["dynamic"].get("initialization", "previous")
        if self.aux_initialization not in ("previous", "m0"):
            raise ValueError("unsupported Aux initialization")
        if (self.aux_initialization == "m0"
                or manifest["feedback"].get("verified_replay_policy", "pending_batch") != "pending_batch"
                or manifest["loop"].get("commit_gate_metric_policy", "mean_f1") != "mean_f1"):
            from .cli import validate

            validate(manifest)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.trainer, self.reliability = trainer, reliability
        self.labels = tuple(context["labels"])
        from .aux_extension import partition_audit

        self.aux_extension = partition_audit(self.rules, manifest, self.labels)
        self.rows = tuple(context["row_ids"])
        self.roles = np.asarray(context["row_roles"])
        self.n = len(self.rows)
        self.k = len(self.labels)
        self.gamma = signed_gamma(context["gamma"])
        self.initial_gamma = seed_identity(self.gamma)
        self.answered = set(np.flatnonzero((self.gamma.pos | self.gamma.neg).all(1)).tolist())
        self.initial_answers = set(self.answered)
        self.decisions = []
        self.pending_verified_rows = set()
        self.verified_replay_policy = manifest["feedback"].get("verified_replay_policy", "pending_batch")
        self.previous_feedback_state = None
        self.query_batches = 0
        self.initial_query_pool_size = int(np.count_nonzero(
            np.isin(self.roles, manifest["domains"]["query_roles"])
            & ~(self.gamma.pos | self.gamma.neg).all(1)))
        self.events = []
        self.sessions = []
        self.actual_chases = 0
        self.inference_counts = {"full": 0, "chaseInc": 0, "chaseDelta": 0, "calibration": 0}
        self.round_history = []
        self.annotation_sources = {
            "human_answers": 0,
            "simulated_human_answers": 0,
            "llm_proposals": 0,
            "simulated_llm_proposals": 0,
            "llm_accepted": 0,
        }
        self.annotation_documents = {
            name: set() for name in ("human", "simulated_human", "llm", "simulated_llm")
        }
        self.timings = {
            "chase_seconds": 0.0,
            "chase_setup_seconds": 0.0,
            "chase_export_seconds": 0.0,
            "feedback_weights_seconds": 0.0,
            "threshold_calibration_seconds": 0.0,
            "refinement_seconds": 0.0,
            "influence_seconds": 0.0,
            "external_seconds": 0.0,
            "calibration_seconds": 0.0,
            "feedback_source_seconds": 0.0,
        }
        self.training_attempts = self.actual_updates = self.committed_updates = self.queries = (
            self.reviews
        ) = 0
        self.unmeasured_failed_updates = 0
        self.supervised_done = False
        self.query_random = np.random.RandomState(manifest["query"]["seed"])
        self.rdg_estimates = 0
        self.alias = manifest["dynamic"]["alias"]
        self.quarantine_enabled = manifest["loop"].get("conflict_policy", "halt") == POLICY
        self.retain_unknown = (
            self.quarantine_enabled
            and manifest["loop"].get("unknown_semantics", "full_rebuild") == "retain_consequences"
        )
        self.conflict_update_policy = manifest["loop"].get("conflict_update_policy", "retain_history")
        if self.conflict_update_policy not in ("retain_history", "fresh_from_gamma"):
            raise ValueError("unsupported conflict_update_policy")
        if self.conflict_update_policy == "fresh_from_gamma" and not self.retain_unknown:
            raise ValueError("fresh_from_gamma requires the retained unknown conflict protocol")
        self.arbitration_rules = (
            arbitration_rules(manifest, self.labels) if self.quarantine_enabled else ()
        )
        from . import fallback as fallback_module

        self.fallback = fallback_module.configuration(manifest)
        if self.fallback is not None and not self.fallback["enabled"]:
            self.fallback = None
        self.fallback_thresholds = (
            fallback_module.declared_thresholds(self.fallback, self.labels)
            if self.fallback is not None else None
        )
        self.gamma_wins = manifest["loop"].get("gamma_conflict_policy", "quarantine") == "validated_gamma_wins"
        # batch_start (default): LLM proposals of a query batch are checked
        # against the closure at the start of the batch. refreshed: against the
        # closure that also contains the answers accepted earlier in the batch.
        self.llm_check_base = manifest["loop"].get("llm_check_base", "batch_start")
        self._check_base = None
        self._preflight_precision(manifest)
        self.commit_gate = manifest["loop"].get("commit_gate", "none")
        self.commit_gate_metric_policy = manifest["loop"].get("commit_gate_metric_policy", "mean_f1")
        self.current_joint_gate = None
        self.best_gate_score = None
        self.started = time.monotonic()
        source_alias = (self.aux_extension["source_alias"]
                        if self.aux_extension is not None else None)
        share_initial_aux = (source_alias is not None and
            context["model_probabilities"][self.alias]
            is context["model_probabilities"][source_alias])
        self.tables = {a: freeze_probability(p) for a, p in context["model_probabilities"].items()
                       if not (share_initial_aux and a == self.alias)}
        if share_initial_aux:
            self.tables[self.alias] = self.tables[source_alias]
        self.initial_tables = {a: probability_identity(p) for a, p in self.tables.items()}
        from .aux_extension import adaptive_aliases, channel_polarities
        self.affected = {self.alias} | set(adaptive_aliases(manifest))
        declared = set()
        for ensemble in manifest["dynamic"]["ensemble_dependencies"]:
            name = ensemble["alias"]
            if name == self.alias or name in declared or name not in self.tables:
                raise ValueError("duplicate/new ensemble alias or illegal override of Aux")
            if name in ensemble["dependencies"]:
                raise ValueError("self-dependent ensemble")
            if self.alias in ensemble["dependencies"] or self.affected.intersection(
                ensemble["dependencies"]
            ):
                self.affected.add(name)
                declared.add(name)
            else:
                raise ValueError("declared refreshed ensemble does not depend on dynamic Aux")
        self.channels = rule_channels(self.rules, self.affected, self.labels)
        if self.aux_extension is not None and any(
            ri < self.aux_extension["frozen_rules"] for ri, _ in self.channels
        ):
            raise ValueError("old learned rules must not have adaptable channels in isolated Aux mode")
        self.receiver_occurrences = set(rule_channels(
            self.rules, {self.alias} | set(adaptive_aliases(manifest)), self.labels))
        self.channel_polarities = channel_polarities(self.rules, manifest)
        if manifest["strategy"] != "frozen" and not self.channels and not self.quarantine_enabled:
            raise ValueError("declared adaptable model and ensembles have no dynamic ML occurrence")
        self.training_rows = np.isin(self.roles, manifest["domains"]["training_roles"])
        self.query_rows = np.isin(self.roles, manifest["domains"]["query_roles"])
        self.review_reference = None
        self.review_evidence = False
        if manifest["query"]["strategy"] == "review":
            from .query_selection import train_label_cardinality, configuration
            options = configuration(manifest)
            self.review_evidence = options.get("eligibility_mode", "legacy") == "evidence"
            if self.review_evidence and manifest["loop"].get("query_pool", "uncovered") != "needs_review":
                raise ValueError("evidence query eligibility requires needs_review")
            if set(options["aux_thresholds"]) - set(self.labels):
                raise ValueError("query Aux threshold label is outside the complete label axis")
            # Query answers must never change this training-only reference.
            reference_rows = self.training_rows & ~self.query_rows
            self.review_reference = train_label_cardinality(
                self.gamma.pos, self.gamma.neg, reference_rows)
            write(self.output / "QUERY_TRAIN_CARDINALITY.json", self.review_reference)
        self.version = Version(
            self.tables[self.alias],
            manifest["dynamic"]["checkpoint"],
            self.channels,
            digest(
                dict(
                    checkpoint=manifest["dynamic"]["checkpoint"],
                    p=array_identity(self.tables[self.alias]),
                    channels=[[list(k), list(v)] for k, v in self.channels.items()],
                )
            ),
        )
        self.reliability_mode = manifest["feedback"].get("rule_reliability_update", "current_model")
        self.initial_version = copy.deepcopy(self.version)
        self.current_reliability_record = manifest["source_reliability"]
        self.reliability_version = self.version.identity
        self.calibration_prediction_texts = None
        if reliability is not None and self.reliability_mode == "frozen_prior":
            from .reliability import FROZEN_POLICY

            self.reliability = copy.deepcopy(reliability)
            self.reliability["transfer_policy"] = FROZEN_POLICY
        self.fallback_independent = (
            self.fallback is not None and self.fallback["calibration"] == "independent_calibration"
        )
        if self.commit_gate not in ("none", "independent_calibration"):
            raise ValueError("commit_gate must be none or independent_calibration")
        # The undecided-cell fallback and the commit gate are recalibrated on
        # the independent calibration documents for every candidate model, in
        # every strategy and weighting (2026-10-02): a comparison variant then
        # differs from the main protocol only in the mechanism it studies.
        independent_outputs = (
            self.fallback_independent or self.commit_gate == "independent_calibration")
        if reliability is not None and (
            independent_outputs
            or (
                manifest["strategy"] == "paper"
                and manifest["weighting"] in RELIABILITY_WEIGHTINGS
                and self.reliability_mode == "current_model"
            )
        ):
            from .reliability import calibration_texts

            self.calibration_prediction_texts = calibration_texts(reliability)
        if independent_outputs:
            if reliability is None or manifest["source_reliability"] is None:
                raise ValueError(
                    "independent fallback calibration / commit gate need the bound independent "
                    "calibration artifact (source_reliability); run the calibrate stage first"
                )
            from .reliability import load_fallback_record

            # The record must be the one bound to this reliability artifact
            # and match this run's configuration, labels, model and program.
            initial_record = load_fallback_record(
                manifest["source_reliability"], reliability, self.context, self.rules, manifest,
                self.labels)
            if self.fallback_independent:
                self.fallback_thresholds = np.array(
                    [float(initial_record["thresholds"][label]) for label in self.labels])
            self.best_gate_score = float(initial_record["gate"]["mean_f1"])
            if self.commit_gate_metric_policy == "aux_loris_joint":
                if "joint_gate" not in initial_record:
                    raise ValueError("strict joint gate requires new A/B-bound calibration; recalibrate")
                self.current_joint_gate = copy.deepcopy(initial_record["joint_gate"])
        initial = self.tables_for(self.version)
        if any(not np.array_equal(initial[a], self.tables[a]) for a in self.affected):
            raise ValueError(
                "declared ensemble computation does not exactly reproduce frozen M0; zero update would change baseline"
            )
        write(self.output / "EFFECTIVE_MANIFEST.json", manifest)
        write(self.output / "ROOTS_INITIAL.json", self.initial_gamma)
        write(self.output / "RULE_FORM_AUDIT.json", rule_form_audit)
        write(self.output / "MODEL_PREDICATE_MAP.json", self.model_predicate_map())
        if self.aux_extension is not None:
            write(self.output / "AUX_EXTENSION.json", self.aux_extension)
        if self.quarantine_enabled:
            write(
                self.output / "MODEL_ARBITRATION_RULES.json",
                [r.to_dict() for r in self.arbitration_rules],
            )

    @staticmethod
    def _preflight_precision(manifest):
        """Reject an unsupported GPU precision before any Chase or training work."""
        dynamic = manifest["dynamic"]
        if manifest["strategy"] == "frozen" or dynamic.get("device") != "cuda":
            return
        amps = {dynamic.get("configuration", {}).get("amp"), dynamic.get("prediction_amp")}
        if "bf16" not in amps:
            return
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but no CUDA device is visible")
        if int(torch.cuda.get_device_capability()[0]) < 8:
            raise RuntimeError(
                "bf16 AMP requires an Ampere-or-newer GPU; on V100 declare amp='fp16' "
                "with an amp_skip_tolerance, amp='auto', or amp='none'")

    def model_predicate_map(self):
        """Record concrete symbol bindings; these choices are not paper formulas."""
        return dict(
            aux_extension=self.aux_extension,
            parameter_blocks={
                self.alias: dict(
                    symbol="omega_Aux", receives_gradient=True, checkpoint=self.version.checkpoint
                ),
            },
            models=[
                dict(
                    alias=alias,
                    initial_table=self.initial_tables[alias],
                    refreshed_after_Aux_update=alias in self.affected,
                    receives_gradient=alias == self.alias,
                )
                for alias in sorted(self.tables)
            ],
            ensembles=copy.deepcopy(self.manifest["dynamic"]["ensemble_dependencies"]),
            occurrences=[
                dict(
                    rule_index=ri,
                    atom_index=ai,
                    alias=atom.alias,
                    label=atom.label,
                    dynamic=atom.dynamic,
                    threshold=atom.eta,
                    threshold_symbol="kappa",
                    label_symbol="tau",
                    threshold_refined=(ri, ai) in self.channels,
                    interpretation=("current_version_table"
                                    if atom.dynamic and atom.alias in self.affected
                                    else "initial_table"),
                    training_receiver=(
                        self.alias
                        if (ri, ai) in self.channels
                        and (
                            self.receiver_policy == "affected_channels_to_aux"
                            or (ri, ai) in self.receiver_occurrences
                        )
                        else None
                    ),
                )
                for ri, rule in enumerate(self.rules)
                for ai, atom in enumerate(rule.body)
                if isinstance(atom, ML)
            ],
            correction_selection=(
                "any_actual_receiver_occurrence_disagrees_then_deduplicate_document_label"
                if self.receiver_policy == "same_model"
                else "any_refined_occurrence_disagrees_then_transfer_to_Aux_deduplicate_document_label"
            ),
            receiver_policy=self.receiver_policy,
            receiver_occurrences=[list(o) for o in sorted(self.receiver_occurrences)],
            no_receiver_occurrences_means_no_logical_corrections=self.receiver_policy
            == "same_model",
            training_receiver=self.alias,
            discount_probability_source=self.alias,
            discount_active=self.manifest["weighting"] in DISCOUNT_WEIGHTINGS,
            weighting_requested=self.requested_weighting,
            weighting_effective=self.manifest["weighting"],
            threshold_objective="weighted_binary_accuracy_grid_per_occurrence",
            delta_statistics_unit="document_dynamic_predicate_occurrence",
            paper_status="implementation_choices_requiring_author_symbol_mapping_confirmation",
            conflict_semantics=(
                "author_conflict_quarantine_extension"
                if self.quarantine_enabled
                else "paper_strict_chase"
            ),
            arbitration_rules=[r.to_dict() for r in self.arbitration_rules],
            arbitration_changes_receiver_occurrences=False,
        )

    def event(self, kind, **fields):
        value = dict(kind=kind, **fields)
        self.events.append(value)
        with (self.output / "EVENTS.jsonl").open("a") as f:
            f.write(json.dumps(value, allow_nan=False) + "\n")
            f.flush()

    def release(self, session):
        runner = session.runner
        if runner in self.sessions:
            self.sessions.remove(runner)
        # A runner reused by several live sessions closes with the last one.
        if not any(item is runner for item in self.sessions):
            runner.close()

    def _reusable_runner(self, ctx, rules, decisions):
        """A live runner built from the same context except Gamma, or None.

        Retaining and quarantine Chase calls receive Gamma explicitly; an
        IndexedChaseRunner reads its constructor Gamma only as a default. Model
        tables are compared by content digest, rules and instance decisions by
        value, and the remaining context entries are the run's own unchanged
        objects (2026-09-30 optimization; results are identical).
        """
        pool = getattr(self, "_runner_pool", None)
        if pool is None or not any(item is pool["runner"] for item in self.sessions):
            return None
        if tuple(rules) != pool["rules"] or list(decisions or []) != pool["decisions"]:
            return None
        if set(ctx) != set(pool["context"]):
            return None
        for key, value in ctx.items():
            if key == "gamma":
                continue
            if key in ("model_probabilities", "static_model_probabilities"):
                if _table_digests(value) != pool["tables"][key]:
                    return None
            elif value is not pool["context"][key]:
                return None
        return pool["runner"], pool["scope"]

    def tables_for(self, version):
        tables = dict(self.tables)
        tables[self.alias] = version.probabilities
        if self.aux_extension is not None and self.aux_extension.get("derived_aliases"):
            from .aux_extension import probability_views
            identity = probability_identity(version.probabilities)
            saved = getattr(self, "_aux_view_cache", None)
            if saved is None or saved[0] is not version.probabilities or saved[1] != identity:
                views = probability_views(self.manifest, version.probabilities)
                views = {a: (p if a == self.alias else freeze_probability(p))
                         for a, p in views.items()}
                self._aux_view_cache = (version.probabilities, identity, views)
            tables.update(self._aux_view_cache[2])
        for spec in self.manifest["dynamic"]["ensemble_dependencies"]:
            deps = spec["dependencies"]
            if not all(a in tables for a in deps):
                raise ValueError("missing declared ensemble dependency")
            if spec["operation"] != "binary_vote_mean":
                raise ValueError("only declared binary-vote ensemble refresh is implemented")
            tables[spec["alias"]] = np.mean(
                [np.asarray(tables[a], dtype=np.float64) >= spec["vote_threshold"] for a in deps],
                axis=0,
            ).astype(np.float32)
        for alias, p in tables.items():
            if alias not in self.affected and probability_identity(p) != self.initial_tables[alias]:
                raise RuntimeError("fixed core model probability table changed")
        return tables

    def predicate_id(self, version):
        tables = self.tables_for(version)
        rules = with_channels(self.rules, version.channels)
        values = []
        for ri, rule in enumerate(rules):
            for ai, atom in enumerate(rule.body):
                if isinstance(atom, ML) and atom.dynamic and atom.alias in self.affected:
                    k = self.labels.index(atom.label)
                    values.append(
                        [
                            ri,
                            ai,
                            array_identity(
                                np.asarray(tables[atom.alias][:, k], dtype=np.float64) >= atom.eta
                            ),
                        ]
                    )
        for rule in self.arbitration_rules:
            k = self.labels.index(rule.label)
            decision = np.asarray(tables[rule.alias][:, k], float) >= rule.kappa
            values.append([rule.rule_id, array_identity(decision if rule.decision else ~decision)])
        return digest(values)

    def decision_change(self, before, after):
        old, new = self.channel_tables(before), self.channel_tables(after)
        changed = total = 0
        for occurrence, (label, threshold) in before.channels.items():
            previous = np.asarray(old[occurrence][:, label], float) >= threshold
            current = np.asarray(new[occurrence][:, label], float) >= after.channels[occurrence][1]
            changed += int((previous != current).sum())
            total += len(previous)
        old_tables, new_tables = self.tables_for(before), self.tables_for(after)
        for rule in self.arbitration_rules:
            label = self.labels.index(rule.label)
            previous = np.asarray(old_tables[rule.alias][:, label], float) >= rule.kappa
            current = np.asarray(new_tables[rule.alias][:, label], float) >= rule.kappa
            changed += int((previous != current).sum())
            total += len(previous)
        return dict(
            changed=changed,
            total=total,
            fraction=changed / total if total else 0.0,
            unit="document_dynamic_predicate_occurrence",
        )

    def record_round(self, version, result, decision_change=None):
        # Save predictions without reading evaluation gold during learning.
        ids = np.flatnonzero(np.isin(self.roles, self.manifest["domains"]["scoring_roles"]))
        folder = self.output / "rounds"
        folder.mkdir(exist_ok=True)
        path = folder / ("%05d.npz" % len(self.round_history))
        tables = self.channel_tables(version)
        decisions = {
            ("%d_%d" % occurrence): np.asarray(tables[occurrence][ids, label], float) >= threshold
            for occurrence, (label, threshold) in version.channels.items()
        }
        np.savez_compressed(
            path,
            rows=ids,
            row_ids=np.asarray(self.rows, dtype=str)[ids],
            labels=np.asarray(self.labels, dtype=str),
            logical=result.positive_facts[ids],
            output_positive=available_predictions(result)[ids],
            rule_output_positive=(result.positive_facts
                                  | getattr(result, "arbitration_positive", False))[ids],
            fallback_positive=np.asarray(getattr(
                result, "fallback_positive", np.zeros_like(result.positive_facts)))[ids],
            conflict_unknown=unknown_mask(result)[ids],
            aux_raw=version.probabilities[ids] >= 0.5,
            **decisions,
        )
        self.round_history.append(
            dict(
                round=len(self.round_history),
                version=version.identity,
                file=str(path.relative_to(self.output)),
                decision_change=decision_change,
                occurrence_labels={
                    ("%d_%d" % o): label for o, (label, _) in version.channels.items()
                },
                gamma=seed_identity(self.gamma),
            )
        )

    def _refresh_fallback(self, result, version):
        """Recompute the undecided-cell fallback of result under new thresholds."""
        if self.fallback is None or result.status != "fixpoint":
            return
        from . import fallback as fallback_module

        arbitration = (
            (result.arbitration_positive, result.arbitration_negative)
            if hasattr(result, "arbitration_positive") else None
        )
        result.fallback_positive = fallback_module.fill(
            np.asarray(result.positive_facts, bool), np.asarray(result.negative_facts, bool),
            unknown_mask(result), np.asarray(self.gamma.neg, bool),
            self.tables_for(version)[self.fallback["alias"]], self.fallback_thresholds, arbitration,
        )
        result.fallback_thresholds = self.fallback_thresholds.copy()
        result.predictions = available_predictions(result).astype(np.float32)

    def channel_tables(self, version):
        tables = self.tables_for(version)
        return {(ri, ai): tables[self.rules[ri].body[ai].alias] for ri, ai in version.channels}

    def chase(
        self,
        version,
        gamma=None,
        *,
        phase,
        rules=None,
        decisions=None,
        previous=None,
        update_kind=None,
        initial_unknown=None,
        temporary_seed=None,
        seed_arbitration=None,
        check_candidates=None,
        decision_only=False,
    ):
        if decision_only and not (self.retain_unknown and previous is not None
                                  and update_kind == "check"
                                  and phase == "LLM_proposal_validation"):
            raise ValueError("decision-only Chase is restricted to retained LLM proposal checks")
        if self.actual_chases >= self.manifest["loop"]["max_chase_calls"]:
            raise ResourceLimit("max_complete_Chase_calls reached; no partial closure scored")
        setup_started = time.monotonic()
        gamma = self.gamma if gamma is None else gamma
        fresh_update = (
            self.conflict_update_policy == "fresh_from_gamma"
            and update_kind in ("chaseInc", "chaseDelta") and not decision_only
        )
        if fresh_update:
            # A changed negative rule can invalidate a positive ancestor.
            # Previous weak descendants and arbitration seeds are not trusted
            # roots. Rebuild from current Gamma using the warm static indexes.
            previous, initial_unknown, temporary_seed, seed_arbitration = None, None, None, None
            self.event("conflict_update_from_verified_roots", requested_update=update_kind,
                       previous_weak_history_reused=False, temporary_arbitration_roots_reused=False)
        rules = with_channels(self.rules, version.channels) if rules is None else tuple(rules)
        ctx = dict(
            self.context,
            gamma=gamma,
            model_probabilities=self.tables_for(version),
            static_model_probabilities=self.tables,
        )
        decisions = self.decisions if decisions is None else decisions
        reused = self._reusable_runner(ctx, rules, decisions)
        if reused is None:
            scope = FrozenInstanceScope(ctx, rules, decisions) if decisions else None
        else:
            runner, scope = reused
        folder = self.output / ("chase_%05d" % self.actual_chases)
        folder.mkdir()
        self.actual_chases += 1
        if reused is None:
            tables = {key: _table_digests(ctx[key])
                      for key in ("model_probabilities", "static_model_probabilities") if key in ctx}
            runner = IndexedChaseRunner(ctx, self.output / "cache", application_scope=scope)
            self._runner_pool = dict(runner=runner, scope=scope, rules=tuple(rules),
                                     decisions=copy.deepcopy(list(decisions or [])),
                                     context=dict(ctx), tables=tables)
        self.sessions.append(runner)
        setup_seconds = time.monotonic() - setup_started
        self.timings["chase_setup_seconds"] += setup_seconds
        started = time.monotonic()
        execution = self.manifest["loop"].get("execution", "incremental")
        if self.quarantine_enabled:
            from .quarantine import quarantine_chase
            from .arbitration import arbitrate

            if temporary_seed is None and previous is not None:
                prior = arbitrate(
                    self.arbitration_rules,
                    ctx["model_probabilities"],
                    unknown_mask(previous[0]),
                    self.labels,
                    model_state="post",
                    model_version=version.identity,
                )
                temporary_seed = (prior.positive, prior.negative)
                seed_arbitration = dict(
                    audit=prior.audit,
                    instances=prior.evidence,
                    previous_conflict_unknown=array_identity(unknown_mask(previous[0])),
                    current_model_version=version.identity,
                )
            seed_positive, seed_negative = (
                temporary_seed if temporary_seed is not None else (None, None)
            )
            if self.retain_unknown:
                from .unknown_incremental import retaining_chase

                before, old_session = previous if previous is not None else (None, None)
                if before is not None and update_kind in ("chaseInc", "chaseDelta"):
                    # A new model/annotation round unlocks the previous conflict
                    # cells. The engine must reconsider all of their producers.
                    if initial_unknown is None:
                        initial_unknown = np.zeros_like(before.positive_facts)
                result = retaining_chase(
                    runner,
                    rules,
                    gamma,
                    before=before,
                    old_runner=old_session.runner if old_session is not None else None,
                    old_rules=old_session.rules if old_session is not None else None,
                    old_gamma=old_session.gamma if old_session is not None else None,
                    update_kind="fresh" if fresh_update else update_kind,
                    initial_positive=seed_positive,
                    initial_negative=seed_negative,
                    initial_unknown=initial_unknown,
                    execution=execution,
                    gamma_wins=self.gamma_wins,
                    check_candidates=check_candidates,
                    decision_only=decision_only,
                    collect_rule_conclusions=(
                        self.aux_extension is not None
                        and bool(self.aux_extension.get("derived_aliases"))
                        and before is None and not decision_only
                    ),
                    rule_conclusion_observed=(
                        ~(gamma.pos | gamma.neg)
                        if self.aux_extension is not None
                        and self.aux_extension.get("derived_aliases")
                        and before is None and not decision_only
                        else None
                    ),
                )
                count_kind = (
                    update_kind
                    if before is not None
                    and execution == "incremental"
                    and update_kind in ("chaseInc", "chaseDelta")
                    else "full"
                )
                self.inference_counts[count_kind] += 1
            else:
                result = quarantine_chase(
                    runner,
                    rules,
                    gamma,
                    initial_positive=seed_positive,
                    initial_negative=seed_negative,
                    initial_unknown=initial_unknown,
                )
                self.inference_counts["full"] += 1
            if decision_only:
                inference_seconds = time.monotonic() - started
                self.timings["chase_seconds"] += inference_seconds
                export_started = time.monotonic()
                write(folder / "CHECK.json", dict(
                    phase=phase, version=version.identity, status=result.status,
                    Gamma=seed_identity(gamma), base_history=before.history_identity,
                    conflict_unknown_cells=int(result.conflict_unknown.sum()),
                    new_conflict_unknown_cells=int(np.count_nonzero(
                        result.conflict_unknown & ~before.conflict_unknown)),
                    candidate_contradictions=sorted(result.candidate_contradictions or ()),
                    comparison_conflicts=result.comparison_conflicts,
                    inference_audit=result.reference_audit,
                    decision_only=True, persistent_history_changed=False,
                ))
                export_seconds = time.monotonic() - export_started
                self.timings["chase_export_seconds"] += export_seconds
                self.event("Chase", phase=phase, status=result.status, folder=str(folder),
                           version=version.identity, rounds=result.n_rounds,
                           execution=result.reference_audit["executor"],
                           maintenance=result.reference_audit["maintenance"],
                           timings=dict(setup_seconds=setup_seconds,
                                        inference_seconds=inference_seconds,
                                        export_seconds=export_seconds),
                           decision_only=True, full_provenance_exported=False)
                if self.review_evidence:
                    result.query_aux_probabilities = version.probabilities
                    result.query_aux_channels = version.channels
                return (result, SimpleNamespace(runner=runner, rules=rules, gamma=signed_gamma(gamma)),
                        folder)
            if not self.retain_unknown:
                result.temporary_seed = temporary_seed
            result.seed_arbitration = seed_arbitration
            result.query_aux_probabilities = version.probabilities
            result.query_aux_channels = version.channels
            if temporary_seed is not None:
                write(
                    folder / "SEED_ARBITRATION.json",
                    dict(
                        source=seed_arbitration,
                        positive=array_identity(seed_positive),
                        negative=array_identity(seed_negative),
                        purpose="temporary_rule_output_roots_or_exact_recheck_of_those_roots",
                        persistent_Gamma_changed=False,
                    ),
                )
            result.feedback_view = None
            if result.status == "fixpoint":
                if (
                    not self.retain_unknown
                    and temporary_seed is not None
                    and (seed_positive.any() or seed_negative.any())
                ):
                    if self.actual_chases >= self.manifest["loop"]["max_chase_calls"]:
                        raise ResourceLimit(
                            "Chase budget exhausted before arbitration-free feedback closure"
                        )
                    self.actual_chases += 1
                    self.inference_counts["full"] += 1
                    result.feedback_view = quarantine_chase(
                        runner, rules, gamma, initial_unknown=result.conflict_unknown
                    )
                    if result.feedback_view.status != "fixpoint":
                        raise ValueError("arbitration-free feedback closure failed")
                arbitration = arbitrate(
                    self.arbitration_rules,
                    ctx["model_probabilities"],
                    result.conflict_unknown,
                    self.labels,
                    model_state="post",
                    model_version=version.identity,
                )
                result.arbitration_positive = arbitration.positive
                result.arbitration_negative = arbitration.negative
                result.remaining_unknown = arbitration.remaining_unknown
                result.predictions = available_predictions(result).astype(np.float32)
                write(
                    folder / "ARBITRATION.json",
                    dict(
                        audit=arbitration.audit,
                        instances=arbitration.evidence,
                        stored_Gamma_unchanged=True,
                        predictions_are_temporary=True,
                    ),
                )
            result.reference_audit["requested_execution"] = execution
            result.reference_audit["full_recompute_reason"] = (
                "withdraw_obsolete_weak_consequences_from_verified_roots"
                if fresh_update else "nonmonotone_conflict_quarantine_and_temporary_model_seeds"
                if not self.retain_unknown
                else None
            )
        elif previous is not None and execution == "incremental":
            from .inference import chase_inc, chase_delta

            before, old_session = previous
            if update_kind == "chaseInc":
                result = chase_inc(runner, rules, before, old_session.gamma, gamma)
            elif update_kind == "chaseDelta":
                result = chase_delta(
                    old_session.runner, runner, old_session.rules, rules, before, gamma
                )
            else:
                raise ValueError("explicit incremental update kind required")
            self.inference_counts[update_kind] += 1
        else:
            result = runner(rules, gamma)
            self.inference_counts["full"] += 1
        if self.fallback is not None and result.status == "fixpoint":
            from . import fallback as fallback_module

            arbitration = (
                (result.arbitration_positive, result.arbitration_negative)
                if hasattr(result, "arbitration_positive") else None
            )
            result.fallback_positive = fallback_module.fill(
                np.asarray(result.positive_facts, bool), np.asarray(result.negative_facts, bool),
                unknown_mask(result), np.asarray(gamma.neg, bool),
                ctx["model_probabilities"][self.fallback["alias"]], self.fallback_thresholds, arbitration,
            )
            result.fallback_thresholds = self.fallback_thresholds.copy()
            result.predictions = available_predictions(result).astype(np.float32)
        inference_seconds = time.monotonic() - started
        self.timings["chase_seconds"] += inference_seconds
        export_started = time.monotonic()
        identity = state_identity(result)
        history_arrays = {}
        if hasattr(result, "history_nodes"):
            history_arrays = dict(
                execution_gamma_positive=result.persistent_gamma.pos,
                execution_gamma_negative=result.persistent_gamma.neg,
                temporary_seed_positive=result.temporary_seed[0],
                temporary_seed_negative=result.temporary_seed[1],
            )
            roots = getattr(getattr(result.history_nodes, "mapping", None), "base", None)
            if roots is not None:
                history_arrays.update(initial_root_gamma_positive=roots.roots.gamma.pos,
                                      initial_root_gamma_negative=roots.roots.gamma.neg)
        np.savez_compressed(
            folder / "FACTS.npz",
            positive=result.positive_facts,
            negative=result.negative_facts,
            conflict_unknown=unknown_mask(result),
            arbitration_positive=getattr(
                result, "arbitration_positive", np.zeros_like(result.positive_facts)
            ),
            arbitration_negative=getattr(
                result, "arbitration_negative", np.zeros_like(result.negative_facts)
            ),
            fallback_positive=getattr(
                result, "fallback_positive", np.zeros_like(result.positive_facts)
            ),
            **history_arrays,
        )
        full_export = should_export(self.manifest["loop"], self.actual_chases, phase)
        if full_export:
            export_provenance(folder, result)
        export_seconds = time.monotonic() - export_started
        self.timings["chase_export_seconds"] += export_seconds
        write(
            folder / "STATE.json",
            dict(
                phase=phase,
                version=version.identity,
                Gamma=seed_identity(gamma),
                rules=[r.to_dict() for r in rules],
                state=identity,
                persistent_Gamma=seed_identity(self.gamma),
                proof_scope="complete_first_witness_not_all_supports",
                execution_roots_are_weak=phase == "feedback_check",
                scope=scope.receipt() if scope else None,
                inference_audit=result.reference_audit,
                full_provenance_exported=full_export,
                in_memory_provenance_preserved=True,
                timings=dict(setup_seconds=setup_seconds, inference_seconds=inference_seconds,
                             export_seconds=export_seconds),
            ),
        )
        self.event(
            "Chase",
            phase=phase,
            status=result.status,
            folder=str(folder),
            version=version.identity,
            rounds=result.n_rounds,
            execution=result.reference_audit["executor"],
            maintenance=result.reference_audit.get("maintenance"),
            timings=dict(setup_seconds=setup_seconds, inference_seconds=inference_seconds,
                         export_seconds=export_seconds),
            full_provenance_exported=full_export,
        )
        if self.review_evidence:
            result.query_aux_probabilities = version.probabilities
            result.query_aux_channels = version.channels
        return (
            result,
            SimpleNamespace(runner=runner, rules=rules, gamma=signed_gamma(gamma)),
            folder,
        )

    def independent(self, version, initial_unknown=None):
        current = with_channels(self.rules, version.channels)
        selected = [
            (ri, r)
            for ri, r in enumerate(current)
            if not any(isinstance(a, ML) and a.alias in self.affected for a in r.body)
        ]
        # Exact source decisions are remapped only for identical retained rules;
        # no forbidden scoped application is silently re-enabled.
        scope = []
        selected_rules = [r for _, r in selected]
        mapping = {old: new for new, (old, _) in enumerate(selected)}
        for item in self.decisions:
            if item["rule_index"] not in mapping:
                continue
            record = copy.deepcopy(item)
            record["rule_index"] = mapping[item["rule_index"]]
            record["program_id"] = digest([r.to_dict() for r in selected_rules])
            scope.append(record)
        result, session, _ = self.chase(
            version,
            phase="independent_source_existence",
            rules=selected_rules,
            decisions=scope,
            initial_unknown=initial_unknown,
        )
        self.release(session)
        return result

    def feedback(self, result, version, *, session=None):
        actual_result = result
        result = getattr(result, "feedback_view", None) or result
        recorded_source_audit = None
        if self.retain_unknown and self.manifest["strategy"] == "paper":
            from .unknown_incremental import recorded_feedback_view

            if session is None:
                raise ValueError("recorded source checking requires the current Chase session")
            started = time.monotonic()
            result = recorded_feedback_view(session.runner, session.rules, result, self.gamma)
            elapsed = time.monotonic() - started
            self.timings["feedback_source_seconds"] += elapsed
            recorded_source_audit = dict(
                result.reference_audit["recorded_source_check"], seconds=elapsed
            )
        blocked = unknown_mask(actual_result) if self.quarantine_enabled else None
        reliability_gamma = (
            RefGamma(
                self.gamma.pos & ~blocked,
                self.gamma.neg & ~blocked,
                self.gamma.subset,
                self.gamma.strict_subset,
            )
            if blocked is not None
            else self.gamma
        )
        require_independent = (
            self.manifest["feedback"].get("source_filter", "none") == "exclude_receiver"
        )
        independent = (
            self.independent(version, initial_unknown=blocked)
            if self.manifest["strategy"] == "paper" and require_independent
            else None
        )
        weights_started = time.monotonic()
        support_weights = support_audit = None
        if (
            self.manifest["strategy"] == "paper"
            and self.manifest["weighting"] in RELIABILITY_WEIGHTINGS
        ):
            from .reliability import lineage_scores, direct_rule_weights

            if (
                self.reliability_mode == "current_model"
                and self.reliability_version != version.identity
            ):
                raise ValueError(
                    "current model has no completed independent rule calibration; stale q is forbidden"
                )
            current = with_channels(self.rules, version.channels)

            def subset_chase(selected, gamma):
                mapping = {
                    old: new
                    for new, r in enumerate(selected)
                    for old, full in enumerate(current)
                    if r is full
                }
                decisions = []
                for item in self.decisions:
                    if item["rule_index"] not in mapping:
                        continue
                    record = copy.deepcopy(item)
                    record["rule_index"] = mapping[item["rule_index"]]
                    record["program_id"] = digest([r.to_dict() for r in selected])
                    decisions.append(record)
                closure, session, _ = self.chase(
                    version,
                    gamma,
                    phase="derivation_reliability",
                    rules=selected,
                    decisions=decisions,
                    initial_unknown=blocked,
                )
                self.release(session)
                return closure

            excluded = [
                ri
                for ri, r in enumerate(current)
                if require_independent
                and any(isinstance(a, ML) and a.alias in self.affected for a in r.body)
            ]
            if self.manifest["weighting"] in DIRECT_RULE_WEIGHTINGS:
                if session is None:
                    raise ValueError("direct rule weighting needs the current frozen Chase session")
                support_weights, support_audit = direct_rule_weights(
                    session.runner,
                    current,
                    independent if require_independent else result,
                    reliability_gamma,
                    self.reliability,
                    excluded_rule_ids=excluded,
                )
            else:
                support_weights, support_audit = lineage_scores(
                    subset_chase,
                    current,
                    reliability_gamma,
                    self.reliability,
                    excluded_rule_ids=excluded,
                )
        weights_finished = time.monotonic()
        self.timings["feedback_weights_seconds"] += weights_finished - weights_started
        selection_version = (
            Version(self.initial_version.probabilities, self.initial_version.checkpoint,
                    version.channels, self.initial_version.identity)
            if self.aux_initialization == "m0" else version
        )
        batch = assignments(
            self.gamma,
            result,
            independent,
            selection_version.probabilities,
            version.channels,
            self.training_rows,
            strategy=self.manifest["strategy"],
            weighting=self.manifest["weighting"],
            reliability=self.reliability,
            reliability_scores=support_weights,
            reliability_audit=support_audit,
            require_independent=require_independent,
            disagreement_discount=self.manifest["feedback"].get("disagreement_discount", 0.5),
            probabilities_by_channel=self.channel_tables(selection_version),
            channel_polarities=self.channel_polarities,
            receiver_policy=self.receiver_policy,
            receiver_occurrences=self.receiver_occurrences,
            replay=self.manifest["feedback"]["trusted_replay"],
            replay_fraction=self.manifest["feedback"].get(
                "trusted_replay_fraction",
                1.0 if self.manifest["feedback"]["trusted_replay"] else 0.0,
            ),
            self_train_confidence=self.manifest["feedback"]["self_train_confidence"],
            weak_mass_ratio=self.manifest["feedback"].get("weak_mass_ratio"),
            weak_rows_per_label=self.manifest["feedback"].get("weak_rows_per_label"),
            reliability_floor=self.manifest["feedback"].get("reliability_floor"),
            selection_seed=int(self.manifest["query"]["seed"]) + self.training_attempts,
        )
        fresh_rows = np.array(sorted(self.pending_verified_rows), dtype=int)
        required_rows = fresh_rows
        acquired_rows = np.array([], dtype=int)
        if self.verified_replay_policy == "cumulative_acquired":
            acquired_rows, required_rows = cumulative_verified_rows(
                self.gamma, self.training_rows, self.answered - self.initial_answers,
                self.pending_verified_rows,
            )
        if (len(required_rows) and self.manifest["strategy"] != "frozen"
                and self.manifest["feedback"].get("new_verified_supervision", "full_batch") == "full_batch"):
            fresh = np.zeros_like(batch.observed)
            fresh[required_rows] = (self.gamma.pos | self.gamma.neg)[required_rows]
            fresh &= self.training_rows[:, None]
            batch.targets[fresh] = self.gamma.pos[fresh]
            batch.weights[fresh] = 1.0
            batch.true_mask[fresh] = True
            batch.weak_mask[fresh] = False
            batch.observed[fresh] = True
            batch.audit.update(true_cells=int(batch.true_mask.sum()),
                               weak_cells=int(batch.weak_mask.sum()),
                               true_mass=float(batch.weights[batch.true_mask].sum()),
                               weak_mass=float(batch.weights[batch.weak_mask].sum()))
            batch.audit["fresh_verified_batch_cells"] = int(fresh.sum())
            batch.audit["fresh_verified_batch_rows"] = fresh_rows.tolist()
        batch.audit["new_verified_supervision"] = self.manifest["feedback"].get(
            "new_verified_supervision", "full_batch")
        batch.audit["new_verified_rows_available"] = fresh_rows.tolist()
        batch.audit["new_verified_correction_cells"] = int(batch.true_mask[fresh_rows].sum())
        batch.audit["feedback_weights_seconds"] = weights_finished - weights_started
        batch.audit["reliability_artifact"] = self.current_reliability_record
        batch.audit["rule_reliability_update"] = self.reliability_mode
        batch.audit["runtime_model_version"] = version.identity
        batch.audit["aux_initialization"] = self.aux_initialization
        batch.audit["selection_probability_reference"] = (
            "m0_with_current_thresholds" if self.aux_initialization == "m0" else "current_model"
        )
        batch.audit["confidence_policy"] = (
            "initial_receiver_probability_of_current_A"
            if self.aux_initialization == "m0" else batch.audit["confidence_policy"]
        )
        batch.audit["runtime_Gamma"] = seed_identity(self.gamma)
        batch.audit["runtime_source_review_count"] = len(self.decisions)
        batch.audit["arbitration_predictions_are_logical_training_targets"] = False
        batch.audit["arbitration_only_descendants_excluded"] = self.quarantine_enabled
        batch.audit["recorded_source_check"] = recorded_source_audit
        batch.audit["quarantined_Gamma_supervision"] = (
            "stored_verified_targets_only_not_model_arbitration"
        )
        batch.audit["requested_weighting"] = self.requested_weighting
        batch.audit["weighting_alias"] = (
            self.requested_weighting
            if self.requested_weighting != self.manifest["weighting"]
            else None
        )
        if self.manifest["feedback"].get("rule_feedback_scope", "all") == "new_or_changed":
            batch = incremental_batch(
                batch, self.gamma, self.training_rows, required_rows,
                self.manifest["feedback"].get("trusted_replay_fraction", 0.05)
                if self.manifest["feedback"]["trusted_replay"] else 0.0,
                self.previous_feedback_state,
            )
        batch = limit_training_rows(
            batch, self.manifest["feedback"].get("max_training_rows"), required_rows,
            self.previous_feedback_state,
            seed=int(self.manifest["query"]["seed"]) + self.training_attempts,
        )
        if self.verified_replay_policy == "cumulative_acquired":
            active = np.flatnonzero(batch.observed.any(1))
            trained_acquired = np.intersect1d(active, acquired_rows)
            batch.audit.update(
                verified_replay_policy="cumulative_acquired",
                acquired_verified_rows_available=acquired_rows.tolist(),
                cumulative_verified_training_rows=trained_acquired.tolist(),
                required_verified_training_rows=required_rows.tolist(),
                fresh_verified_batch_rows=fresh_rows.tolist(),
                fresh_verified_batch_cells=int(batch.true_mask[fresh_rows].sum()),
                new_verified_correction_cells=int(batch.true_mask[fresh_rows].sum()),
                cumulative_verified_cells=int(batch.true_mask[trained_acquired].sum()),
                cumulative_verified_mass=float(batch.weights[trained_acquired][
                    batch.true_mask[trained_acquired]].sum(dtype=np.float64)),
                trusted_correction_policy="all_acquired_complete_answers_plus_sampled_initial_replay",
            )
        weak_pos = batch.weak_mask & (batch.targets == 1)
        weak_neg = batch.weak_mask & (batch.targets == 0)
        entailed = not (
            np.any(weak_pos & ~result.positive_facts) or np.any(weak_neg & ~result.negative_facts)
        )
        batch.audit["candidate_check_role"] = (
            "fixed_retained_history_Gamma_union_W_guard"
            if self.retain_unknown
            else (
                "fixed_quarantine_and_temporary_seed_Gamma_union_W_guard"
                if self.quarantine_enabled
                else "frozen_monotone_closure_idempotence_guard"
            )
        )
        batch.audit["candidate_check_is_independent_correctness_validation"] = False
        batch.audit["weak_candidates_entailed_by_supplied_closure"] = bool(entailed)
        batch.audit["rejected_by_consistency_check"] = 0
        batch.audit["check_failure_policy"] = self.manifest["feedback"].get(
            "check_failure_policy", "halt"
        )
        if self.manifest["strategy"] == "paper" and not entailed:
            raise FeedbackCheckInvariantError(
                "logical-A extraction returned a non-entailed correction"
            )
        if (
            batch.weak_mask.any()
            and self.manifest["feedback"]["candidate_check"] == "checked"
            and self.manifest["strategy"] == "paper"
        ):
            seed = RefGamma(
                self.gamma.pos | weak_pos,
                self.gamma.neg | weak_neg,
                self.gamma.subset,
                self.gamma.strict_subset,
            )
            checked, temporary, _ = self.chase(
                version,
                seed,
                phase="feedback_check",
                previous=(actual_result, session) if self.retain_unknown else None,
                update_kind="check" if self.retain_unknown else None,
                initial_unknown=blocked,
                temporary_seed=getattr(actual_result, "temporary_seed", None),
                seed_arbitration=getattr(actual_result, "seed_arbitration", None),
            )
            self.release(temporary)
            if checked.status != "fixpoint" or facts_id(checked) != facts_id(actual_result):
                original_count = int(batch.weak_mask.sum())
                failure_folder = self.output / ("feedback_check_failure_%05d" % self.actual_chases)
                failure_folder.mkdir()
                np.savez_compressed(
                    failure_folder / "CANDIDATES.npz", positive=weak_pos, negative=weak_neg
                )
                write(
                    failure_folder / "DIAGNOSTIC.json",
                    dict(
                        policy=batch.audit["check_failure_policy"],
                        version=version.identity,
                        Gamma=seed_identity(self.gamma),
                        candidate_batch_id=batch_id(batch),
                        before=facts_id(actual_result),
                        training_view=facts_id(result),
                        after=facts_id(checked),
                        after_status=checked.status,
                        candidate_cells=original_count,
                        W_subset_successful_closure=bool(entailed),
                        interpretation="unexpected_closure_invariant_failure_not_evidence_of_wrong_labels",
                    ),
                )
                if batch.audit["check_failure_policy"] != "reject_batch":
                    raise FeedbackCheckInvariantError(
                        "Gamma union entailed W failed or changed facts under the frozen model/program; inspect closure invariant"
                    )
                # An explicitly named engineering alternative, not the paper's
                # unspecified per-candidate rejection algorithm.
                batch.observed[batch.weak_mask] = False
                batch.weights[batch.weak_mask] = 0
                batch.targets[batch.weak_mask] = 0
                batch.weak_mask[:] = False
                batch.audit["feedback_check"] = "explicit_reject_batch_engineering_policy"
                batch.audit["rejected_by_consistency_check"] = original_count
                batch.audit["weak_mass"] = 0.0
                batch.audit["weak_cells"] = 0
                batch.audit["weak_positive"] = batch.audit["weak_negative"] = 0
            else:
                batch.audit["feedback_check"] = "complete_Gamma_union_W_success"
                batch.audit["logical_A_check_redundant_for_already_entailed_A"] = (
                    self.manifest["strategy"] == "paper" and not self.quarantine_enabled
                )
        else:
            batch.audit["feedback_check"] = (
                (
                    "self_training_own_predictions"
                    if self.manifest["strategy"] == "self_train"
                    else "NoCheck_ablation_skipped"
                )
                if batch.weak_mask.any()
                else "no_effective_weak_candidates"
            )
        folder = self.output / ("supervision_%05d" % len(list(self.output.glob("supervision_*"))))
        folder.mkdir()
        np.savez_compressed(
            folder / "ARRAYS.npz",
            targets=batch.targets,
            observed=batch.observed,
            weights=batch.weights,
            true_mask=batch.true_mask,
            weak_mask=batch.weak_mask,
        )
        write(folder / "RECEIPT.json", dict(batch.audit, checked_assignments_id=batch_id(batch)))
        self.event("supervision", **batch.audit, checked_assignments_id=batch_id(batch))
        return batch

    def _refreshing_llm_check_base(self):
        return self.llm_check_base == "refreshed" and self.retain_unknown

    def _llm_check_base_for(self, result, session, version):
        """The closure an LLM proposal of the current query batch is checked against.

        batch_start: the closure at the start of the batch. refreshed: the
        closure the batch commit would compute now, i.e. chaseInc from the
        batch-start closure with the answers accepted so far (computed once per
        Gamma). It depends only on the batch start and the current Gamma.
        """
        if not self._refreshing_llm_check_base():
            return result, session
        current = seed_identity(self.gamma)
        if seed_identity(result.persistent_gamma) == current:
            return result, session
        cached = self._check_base
        if cached is not None and cached["origin"] is result and cached["gamma"] == current:
            return cached["result"], cached["session"]
        self._drop_check_base()
        refreshed, refreshed_session, _ = self.chase(
            version, phase="LLM_check_base_refresh", previous=(result, session),
            update_kind="chaseInc")
        if refreshed.status != "fixpoint":
            self.release(refreshed_session)
            self.event("LLM_check_base_refresh_failed", status=refreshed.status, Gamma=current,
                       fallback="batch_start_closure")
            return result, session
        self._check_base = dict(origin=result, gamma=current, result=refreshed,
                                session=refreshed_session)
        self.event("LLM_check_base_refreshed", Gamma=current)
        return refreshed, refreshed_session

    def _drop_check_base(self):
        cached, self._check_base = self._check_base, None
        if cached is not None:
            self.release(cached["session"])

    def oracle_answer(self, row, reason, *, result=None, session=None, version=None):
        if ((row in self.answered and not self.review_evidence)
                or (self.gamma.pos[row] | self.gamma.neg[row]).all()):
            raise ValueError("known complete answers must not be queried twice")
        if not self.query_rows[row]:
            return False
        if remaining_budget(self.manifest["loop"].get("label_budget"), self.queries) == 0:
            return False
        oracle = self.manifest["oracle"]
        if oracle["kind"] == "disabled":
            return False
        if oracle["kind"] not in (
            "directory_simulated_human",
            "npz_simulated_human",
            "directory_llm_then_human",
            "directory_human",
        ):
            raise ValueError("unsupported external answer provider")
        self.queries += 1
        self.event(
            "label_request",
            request=self.queries,
            row=row,
            row_id=self.rows[row],
            reason=reason,
            provider=oracle["kind"],
        )
        # Learner disclosure is one answer after dispatch. The isolated NPZ
        # provider owns/decodes its bank internally; this is not an OS sandbox.
        if oracle["kind"] in ("directory_llm_then_human", "directory_human"):
            from .external import external_answer, human_answer

            def validate_llm(answer):
                if (self.manifest["loop"].get("query_answer_mode", "complete") == "complete"
                        and any(v is None for v in answer["targets"])):
                    self.event("LLM_proposal_needs_human", row=row, reason="complete_label_set_required")
                    return False
                if (
                    result is None
                    or session is None
                    or version is None
                    or result.status != "fixpoint"
                ):
                    return False
                base, base_session = self._llm_check_base_for(result, session, version)
                values = answer["targets"]
                pos = np.array([v == 1 for v in values])
                neg = np.array([v == 0 for v in values])
                if self.quarantine_enabled and ((pos | neg) & unknown_mask(base)[row]).any():
                    self.event(
                        "LLM_proposal_needs_human",
                        row=row,
                        reason="target_is_conflict_unknown_not_logically_validated",
                    )
                    return False
                if (pos & base.negative_facts[row]).any() or (
                    neg & base.positive_facts[row]
                ).any():
                    return False
                p, n = self.gamma.pos.copy(), self.gamma.neg.copy()
                p[row] |= pos
                n[row] |= neg
                seed = RefGamma(p, n, self.gamma.subset, self.gamma.strict_subset)
                candidates = None
                if self.gamma_wins:
                    # The proposal's own new cells, so the check can tell its
                    # consequences from those of answers accepted earlier.
                    candidates = (p & ~self.gamma.pos, n & ~self.gamma.neg)
                checked, temporary, _ = self.chase(
                    version,
                    seed,
                    phase="LLM_proposal_validation",
                    previous=(
                        (base, base_session)
                        if self.retain_unknown or not self.quarantine_enabled
                        else None
                    ),
                    update_kind="check" if self.retain_unknown else "chaseInc",
                    initial_unknown=unknown_mask(base) if self.quarantine_enabled else None,
                    temporary_seed=getattr(base, "temporary_seed", None),
                    seed_arbitration=getattr(base, "seed_arbitration", None),
                    check_candidates=candidates,
                    decision_only=self.retain_unknown,
                )
                valid = checked.status == "fixpoint" and not (
                    self.quarantine_enabled
                    and np.any(unknown_mask(checked) & ~unknown_mask(base))
                )
                rejection = None
                if not valid and self.quarantine_enabled:
                    rejection = dict(reason="proposal_adds_conflict_or_invalid_relation")
                # Under validated Gamma wins a contradiction with the proposal's
                # own labels is a conflict (above); a derivation that contradicts
                # trusted Gamma is blocked, and is attributed to the proposal
                # unless it already follows from the checked closure.
                caused = getattr(checked, "candidate_contradictions", None)
                if valid and self.gamma_wins and caused:
                    valid = False
                    rejection = dict(reason="proposal_consequence_contradicts_validated_labels",
                                     blocked_cells=len(caused))
                self.release(temporary)
                if rejection is not None:
                    self.event("LLM_proposal_needs_human", row=row, **rejection)
                return valid

            answer, receipt = (
                human_answer(int(row), self.rows[row], self.labels, oracle["directory"])
                if oracle["kind"] == "directory_human"
                else external_answer(
                    int(row), self.rows[row], self.labels, oracle["directory"], validate_llm
                )
            )
            for name in self.annotation_sources:
                self.annotation_sources[name] += receipt.get(name, 0)
            write(self.output / ("EXTERNAL_%05d.json" % self.queries), receipt)
            if answer is None:
                self.event("label_pending", row=row, reason=receipt["status"])
                return False
        elif oracle["kind"] == "directory_simulated_human":
            path = Path(oracle["directory"]) / ("%08d.json" % row)
            if not path.is_file():
                self.event("label_pending", row=row, reason="provider_answer_unavailable")
                return False
            answer = json.loads(path.read_text())
        else:
            from loris.discovery.manifest import file_record

            ref = oracle["manifest"]
            if file_record(ref["path"]) != ref:
                raise ValueError("oracle manifest changed")
            child = subprocess.run(
                [sys.executable, "-m", "loris.labeling.oracle", "--manifest", ref["path"]],
                input=json.dumps(dict(row=row, row_id=self.rows[row], labels=list(self.labels))),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if child.returncode:
                self.event(
                    "label_pending",
                    row=row,
                    reason="provider_process_failed",
                    error=child.stderr[-4000:],
                )
                return False
            answer = json.loads(child.stdout)
        if answer.get("row_id") != self.rows[row] or answer.get("labels") != list(self.labels):
            raise ValueError("queried answer row/label identity differs")
        values = answer.get("targets")
        partial_allowed = oracle["kind"] in ("directory_llm_then_human", "directory_human")
        permitted = (0, 1, None) if partial_allowed else (0, 1)
        if (
            not isinstance(values, list)
            or len(values) != self.k
            or any(v not in permitted for v in values)
        ):
            raise ValueError("explicit signed answer with unknown=null required")
        pos = np.array([v == 1 for v in values])
        neg = np.array([v == 0 for v in values])
        if (self.manifest["loop"].get("query_answer_mode", "complete") == "complete"
                and not (pos | neg).all()):
            self.event("label_pending", row=row, reason="complete_label_set_required")
            return False
        if (self.gamma.pos[row] & neg).any() or (self.gamma.neg[row] & pos).any():
            self.event(
                "label_pending",
                row=row,
                reason="answer_conflicts_with_existing_trusted_partial_root",
            )
            return False
        p, n = self.gamma.pos.copy(), self.gamma.neg.copy()
        p[row] |= pos
        n[row] |= neg
        self.gamma = RefGamma(p, n, self.gamma.subset, self.gamma.strict_subset)
        self.answered.add(row)
        self.pending_verified_rows.add(int(row))
        if oracle["kind"] not in ("directory_llm_then_human", "directory_human"):
            self.annotation_sources["simulated_human_answers"] += 1
        provider = answer["provider"] if partial_allowed else "simulated_human"
        self.annotation_documents[provider].add(int(row))
        write(
            self.output / ("ANSWER_%05d.json" % self.queries),
            dict(
                answer,
                provider=answer.get("provider", "development_truth_simulated_human"),
                reason=reason,
            ),
        )
        self.event(
            "answer_accepted",
            row=row,
            row_id=self.rows[row],
            provider=provider,
            Gamma=seed_identity(self.gamma),
        )
        return True

    def resolve(self, result, session, version):
        """Only known wrong direct Label applications may be simulated-approved.

        Relation/chain ambiguity stays pending for an explicit source decision;
        no label-level exception is inserted into an intact subset relation.
        """
        policy = self.manifest["loop"].get("conflict_policy", "halt")
        if result.status == "conflict" and policy in ("halt", POLICY):
            self.event(
                "fixed_program_conflict_pending",
                version=version.identity,
                policy=policy,
                effective_ground_program_changed=False,
                conflicts=plain(result.conflicts),
            )
            return result, session, False
        if policy not in ("halt", "review_instances", POLICY):
            raise ValueError("unknown conflict policy")
        while result.status == "conflict":
            handled = False
            for row, label in result.conflicts:
                k = self.labels.index(label)
                row = int(row)
                if not (self.gamma.pos[row, k] or self.gamma.neg[row, k]):
                    if self.oracle_answer(row, "actual_label_conflict"):
                        handled = True
                        break
                    continue
                if not self.manifest["oracle"]["allow_known_source_review"]:
                    continue
                if self.reviews >= self.manifest["loop"]["source_review_budget"]:
                    continue
                wrong = "neg" if self.gamma.pos[row, k] else "pos"
                key = "%s:%d:%d" % (wrong, row, k)
                node = result.proof_dag[key]
                if node.get("origin") != "rule_application" or node.get("rule_id") is None:
                    continue  # Review of relational dependencies requires a human source decision.
                ri = int(node["rule_id"])
                rule = session.rules[ri]
                if not isinstance(rule.head, Label):
                    continue
                self.reviews += 1
                conflict_id = complete_state_identity(result)["sha256"]
                approval = dict(
                    status="approved",
                    action="disable_rule_instance",
                    reason="existing complete/partial trusted answer disproves this exact direct automatic label head",
                    approval_receipt_id="simulated_source_review_%05d" % self.reviews,
                    reviewer="simulated_human_reuses_obtained_answer",
                    authority_scope="exact_instance_same_corpus_program",
                    parent_conflict_id=conflict_id,
                    corpus_id=scope_corpus_identity(self.context),
                    program_id=digest([r.to_dict() for r in session.rules]),
                    rule_index=ri,
                    rule_sha256=digest(rule.to_dict()),
                    valuation=plain(node["valuation"]),
                    provider="development_truth_review",
                    development_only=True,
                )
                self.decisions.append(approval)
                write(self.output / ("SOURCE_REVIEW_%05d.json" % self.reviews), approval)
                self.event(
                    "source_review_approved",
                    row=row,
                    rule_index=ri,
                    review=self.reviews,
                    repeated_label_query=False,
                    scope_action="whole_exact_rule_instance",
                    effective_ground_program_changed=True,
                    mechanism="explicit_rule_revision_extension",
                )
                handled = True
                break
            if not handled:
                return result, session, False
            previous = session
            result, session, _ = self.chase(version, phase="after_external_revision_from_roots")
            self.release(previous)
        return result, session, result.status == "fixpoint"

    def eligible_query_rows(self, result):
        eligible = self.query_rows & ~(self.gamma.pos | self.gamma.neg).all(1)
        pool = self.manifest["loop"].get("query_pool", "uncovered")
        if pool == "uncovered":
            eligible &= pending_rows(result, self.gamma)
        elif pool == "needs_review":
            from .query_selection import configuration
            if configuration(self.manifest)["audit_fraction"] == 0:
                eligible &= self.review_plan(result)["eligible"]
        elif pool != "all_unanswered":
            raise ValueError("unsupported query_pool")
        if self.answered and not self.review_evidence:
            eligible[np.asarray(sorted(self.answered), dtype=int)] = False
        return eligible

    def has_query_work(self, result):
        if self.review_evidence:
            return bool(self.eligible_query_rows(result).any())
        return pending_rows(result, self.gamma).any() or (
            self.manifest["loop"].get("query_pool", "uncovered") in ("all_unanswered", "needs_review")
            and self.eligible_query_rows(result).any()
        )

    def review_plan(self, result, probabilities=None):
        from .query_selection import configuration, review_candidates
        options = configuration(self.manifest)
        if self.review_reference is None:
            raise ValueError("review selection requires a training-only cardinality reference")
        rows = self.query_rows.copy()
        if self.answered and not self.review_evidence:
            rows[np.asarray(sorted(self.answered), dtype=int)] = False
        thresholds = np.array([options.get("aux_thresholds", {}).get(
            label, options.get("aux_default_threshold", .5)) for label in self.labels], float)
        for occurrence, (label, value) in getattr(result, "query_aux_channels", {}).items():
            if self.channel_polarities.get(occurrence, 1) > 0:
                thresholds[label] = value
        if self.review_evidence and probabilities is None and not hasattr(result,"query_aux_probabilities"):
            raise ValueError("evidence review requires probabilities from the current closure")
        conflicts = getattr(result, "remaining_unknown", unknown_mask(result))
        if self.review_evidence:
            conflicts = unknown_mask(result) & conflicts
        return review_candidates(
            positive=available_predictions(result), negative=result.negative_facts,
            unknown=conflicts, gamma_positive=self.gamma.pos,
            gamma_negative=self.gamma.neg, query_rows=rows,
            aux_probabilities=(getattr(result, "query_aux_probabilities", self.version.probabilities)
                               if probabilities is None else probabilities),
            aux_thresholds=thresholds,
            train_mean=self.review_reference["mean"], ratio=options["cardinality_ratio"],
            eligibility_mode=options.get("eligibility_mode", "legacy"),
            near_boundary_ratio=options.get("near_boundary_ratio", .9))

    def rdg(self, result, session, limit=1):
        eligible = np.flatnonzero(self.eligible_query_rows(result)).tolist()
        if not eligible:
            return None
        policy = self.manifest["query"]["strategy"]
        if policy == "review":
            from .query_selection import configuration, rank_review_candidates
            options = configuration(self.manifest)
            plan = self.review_plan(result, session.runner.probabilities[self.alias])
            # Respect an explicitly narrower domain while retaining random audits.
            permitted = np.zeros(self.n, bool)
            permitted[eligible] = True
            plan["unanswered"] &= permitted
            plan["eligible"] &= permitted
            chosen_rows, audit = rank_review_candidates(
                plan, limit=limit, seed=self.manifest["query"]["seed"] + self.queries,
                audit_fraction=options["audit_fraction"])
            if not chosen_rows:
                return None
            chosen = chosen_rows[0] if limit == 1 else chosen_rows
            write(self.output / ("RDG_%05d.json" % self.queries), dict(
                selected=chosen, criterion="review", audit=audit,
                parameters=plan["parameters"], query_pool=self.manifest["loop"].get("query_pool"),
                training_reference=self.review_reference, unqueried_truth_used=False))
            return chosen
        if policy in ("random", "uncertainty"):
            if policy == "random":
                chosen_rows = ([eligible[int(self.query_random.randint(len(eligible)))]]
                               if limit == 1 else self.query_random.permutation(eligible)[:limit].tolist())
                chosen = chosen_rows[0] if limit == 1 else chosen_rows
                scores = None
            else:
                p = session.runner.probabilities[self.alias]
                scores = [
                    dict(row=row, uncertainty=float(np.mean(1 - np.abs(2 * p[row] - 1))))
                    for row in eligible
                ]
                chosen_rows = [x["row"] for x in sorted(scores, key=lambda x: (-x["uncertainty"], x["row"]))[:limit]]
                chosen = chosen_rows[0] if limit == 1 else chosen_rows
            write(
                self.output / ("RDG_%05d.json" % self.queries),
                dict(
                    selected=chosen,
                    candidates=eligible,
                    scores=scores,
                    criterion=policy,
                    seed=self.manifest["query"]["seed"],
                    same_uncovered_query_domain=self.manifest["loop"].get("query_pool", "uncovered") == "uncovered",
                    query_pool=self.manifest["loop"].get("query_pool", "uncovered"),
                    unqueried_truth_used=False,
                ),
            )
            return chosen
        candidate_limit = self.manifest["query"].get("rdg_candidate_limit")
        if candidate_limit is not None:
            # Explicit candidate-domain approximation only. Every sampled row
            # still receives the original RDG estimate for all native labels.
            if candidate_limit < min(limit, len(eligible)):
                raise ValueError("rdg_candidate_limit cannot reduce the requested query batch")
            full_eligible = eligible
            if len(eligible) > candidate_limit:
                eligible = sorted(np.random.RandomState(self.manifest["query"]["seed"]).choice(
                    np.asarray(eligible, dtype=np.int64), size=candidate_limit, replace=False).tolist())
            domain_audit = dict(
                schema="loris_rdg_candidate_domain_v1",
                full_eligible_count=len(full_eligible),
                full_eligible_row_indices_sha256=digest(full_eligible),
                sampled_row_indices=eligible,
                sampled_row_ids=[self.rows[row] for row in eligible],
                sampled_count=len(eligible), configured_limit=candidate_limit,
                seed=self.manifest["query"]["seed"],
                sampler="numpy_RandomState_fresh_seed_uniform_without_replacement_sorted_rows",
                approximate_domain=len(eligible) < len(full_eligible),
                all_native_labels_evaluated=self.k,
                query_documents_already_used=self.queries,
                unqueried_truth_used=False,
                only_candidate_domain_subsampled=True,
            )
            write(self.output / ("RDG_DOMAIN_%05d.json" % self.queries), domain_audit)
            self.event("query_candidate_domain", **domain_audit)
        estimator = RuleDependencyGraph(session.rules)
        uncovered = np.flatnonzero(pending_rows(result, self.gamma)).tolist()
        scores = []
        self.event("query_ranking_started", candidates=len(eligible), labels=self.k,
                   hypotheses=len(eligible) * self.k,
                   execution="one_frozen_state_with_exact_per_hypothesis_rollback")
        estimates = estimator.estimate_batch(
            session, result,
            ((row, label) for row in eligible for label in range(self.k)),
            uncovered,
        )
        # Retain the existing grounded RDG estimate and all labels; never use
        # score truth or change selection to uncertainty without naming it.
        for row in eligible:
            values = []
            for label in range(self.k):
                if self.rdg_estimates >= self.manifest["loop"]["max_rdg_estimates"]:
                    raise ResourceLimit("RDG estimate budget exhausted before complete selection")
                values.append(next(estimates))
                self.rdg_estimates += 1
            scores.append(dict(row=row, per_label=values, mean=float(np.mean(values))))
            if len(scores) % 100 == 0 or len(scores) == len(eligible):
                self.event("query_ranking_progress", completed_documents=len(scores),
                           candidates=len(eligible), completed_hypotheses=len(scores) * self.k)
        chosen_rows = [x["row"] for x in sorted(scores, key=lambda x: (-x["mean"], x["row"]))[:limit]]
        chosen = chosen_rows[0] if limit == 1 else chosen_rows
        write(
            self.output / ("RDG_%05d.json" % self.queries),
            dict(
                candidates=scores,
                selected=chosen,
                batch_selection="top_ranked_documents_under_same_frozen_closure",
                hypothesis_execution="one_frozen_state_with_exact_per_hypothesis_rollback",
                criterion="all_label_mean_grounded_support_including_query_document",
                unqueried_truth_used=False,
                not_exact_actual_coverage_gain=True,
                **({"candidate_domain": domain_audit} if candidate_limit is not None else {}),
            ),
        )
        return chosen

    def query_batch(self, result, session, version):
        """Rank once using only the frozen closure; disclose complete answers individually.

        Eligible documents are RDG-ranked (or an explicitly named ablation/domain cap).
        Collect the batch before one chaseInc; LLM validation retains its own
        complete check and human fallback. A missing answer stops the batch,
        propagating any earlier accepted answers before returning pending.
        """
        eligible = int(np.count_nonzero(self.eligible_query_rows(result)))
        count = batch_size(self.manifest["loop"], self.initial_query_pool_size, eligible, self.queries)
        if count == 0:
            return result, session, "label_budget_exhausted" if eligible else "no_eligible_uncovered_documents"
        started = time.monotonic()
        rows = self.rdg(result, session, limit=count)
        self.timings["influence_seconds"] += time.monotonic() - started
        if rows is None:
            return result, session, "no_eligible_uncovered_documents"
        rows = [rows] if isinstance(rows, (int, np.integer)) else list(rows)
        self.query_batches += 1
        self.event("query_batch_selected", batch=self.query_batches, rows=rows,
                   requested_size=count, initial_pool_size=self.initial_query_pool_size,
                   budget_unit="documents; LLM and human fallback together count once")
        accepted = 0
        pending = False
        try:
            for row in rows:
                started = time.monotonic()
                okay = self.oracle_answer(row, "RDG_uncovered_document", result=result,
                                         session=session, version=version)
                self.timings["external_seconds"] += time.monotonic() - started
                if not okay:
                    pending = True
                    break
                accepted += 1
        finally:
            # Check bases are batch-local; the batch commit starts from result.
            self._drop_check_base()
        if accepted:
            previous = session
            result, session, _ = self.chase(version, phase="answers_only_zero_update",
                                           previous=(result, session), update_kind="chaseInc")
            self.release(previous)
            result, session, valid = self.resolve(result, session, version)
            if not valid:
                return result, session, "pending_answer_conflict"
            self.valid_gamma_id = digest(seed_identity(self.gamma))
            self.record_round(version, result)
        self.event("query_batch_completed", batch=self.query_batches, selected=len(rows),
                   accepted=accepted, pending_external_answer=pending)
        return result, session, "pending_external_answer" if pending else None

    def finish(self, status, result, version):
        current_valid = (
            result is not None
            and result.status == "fixpoint"
            and self.valid_gamma_id == digest(seed_identity(self.gamma))
        )
        np.savez_compressed(
            self.output / "TRUSTED_GAMMA_FINAL.npz",
            positive=self.gamma.pos,
            negative=self.gamma.neg,
        )
        if result is not None:
            final_started = time.monotonic()
            final_history = self.output / "final_provenance"
            final_history.mkdir()
            export_facts(final_history, result)
            export_provenance(final_history, result)
            self.timings["chase_export_seconds"] += time.monotonic() - final_started
        report = dict(
            query_policy=dict(cadence=self.manifest["loop"].get("query_cadence", "after_saturation"),
                              pool=self.manifest["loop"].get("query_pool", "uncovered"),
                              budget_unit="requested_documents_including_provider_fallback", 
                              label_budget=self.manifest["loop"].get("label_budget"),
                              batches=self.query_batches, initial_pool_size=self.initial_query_pool_size,
                              **({"rdg_candidate_limit": self.manifest["query"]["rdg_candidate_limit"]}
                                 if "rdg_candidate_limit" in self.manifest["query"] else {})),
            unresolved_documents=int(pending_rows(result, self.gamma).sum()) if result is not None else None,
            status=status,
            current_Gamma_has_valid_closure=bool(current_valid and not self.quarantine_enabled),
            current_Gamma_has_quarantined_fixpoint=bool(
                current_valid and self.quarantine_enabled and not self.retain_unknown
            ),
            current_history_has_quarantined_fixpoint=bool(current_valid and self.retain_unknown),
            active_quarantine_pairs=int(unknown_mask(result).sum()) if result is not None else 0,
            effective_execution=(
                "fresh_from_gamma_updates_with_retained_proposal_checks"
                if self.conflict_update_policy == "fresh_from_gamma" else
                (
                    "retained_incremental"
                    if self.manifest["loop"].get("execution", "incremental") == "incremental"
                    else "retained_full_scan"
                )
                if self.retain_unknown
                else (
                    "full_recompute_quarantine"
                    if self.quarantine_enabled
                    else self.manifest["loop"].get("execution", "incremental")
                )
            ),
            full_recompute_reason=(
                "withdraw_obsolete_weak_consequences_from_verified_roots"
                if self.conflict_update_policy == "fresh_from_gamma" else
                "nonmonotone_conflict_quarantine_and_temporary_model_seeds"
                if self.quarantine_enabled and not self.retain_unknown
                else None
            ),
            actual_Chase_calls=self.actual_chases,
            inference_counts=self.inference_counts,
            training_attempts=self.training_attempts,
            actual_optimizer_updates=self.actual_updates,
            committed_model_updates=self.committed_updates,
            failed_training_attempts_without_step_receipt=self.unmeasured_failed_updates,
            actual_optimizer_update_count_complete=self.unmeasured_failed_updates == 0,
            actual_label_requests=self.queries,
            actual_source_reviews=self.reviews,
            annotation_sources=self.annotation_sources,
            annotation_documents={
                name: sorted(rows) for name, rows in self.annotation_documents.items()
            },
            acquired_documents=sorted(self.answered - self.initial_answers),
            RDG_estimates=self.rdg_estimates,
            version=version.identity,
            checkpoint=version.checkpoint,
            Gamma=seed_identity(self.gamma),
            core_structure_changed=False,
            aux_extension=self.aux_extension,
            conflict_policy=self.manifest["loop"].get("conflict_policy", "halt"),
            unknown_semantics=self.manifest["loop"].get("unknown_semantics", "full_rebuild"),
            conflict_update_policy=self.conflict_update_policy,
            rule_revision_extension_used=self.reviews > 0,
            effective_ground_program_changed=bool(self.decisions),
            inference_semantics_changed_by_quarantine=self.quarantine_enabled,
            additional_arbitration_rules=[r.to_dict() for r in self.arbitration_rules],
            active_disabled_rule_instances=len(self.decisions),
            instance_scope_digest=digest(self.decisions),
            symbolic_program_digest=digest(
                [r.to_dict() for r in with_channels(self.rules, version.channels)]
            ),
            only_declared_dynamic_models_changed=True,
            elapsed_seconds=time.monotonic() - self.started,
            timings=self.timings,
            normal_metrics=None,
            task_complete=status == "all_documents_resolved",
            supervised_adaptation=(
                dict(
                    status=(
                        "completed"
                        if self.committed_updates == 1 and self.actual_updates > 0
                        else (
                            "no_eligible_Gamma"
                            if self.supervised_done and self.training_attempts == 0
                            else "not_completed"
                        )
                    ),
                    attempted=self.training_attempts > 0,
                    optimizer_updates=self.actual_updates,
                    committed_model_updates=self.committed_updates,
                    kappa_frozen=True,
                )
                if self.manifest["strategy"] == "supervised"
                else None
            ),
            round_history=self.round_history,
            rule_reliability_update=self.reliability_mode,
            final_reliability_artifact=self.current_reliability_record,
            rule_reliability_used=self.manifest["strategy"] == "paper"
            and self.manifest["weighting"] in RELIABILITY_WEIGHTINGS,
            weighting_requested=self.requested_weighting,
            weighting_effective=self.manifest["weighting"],
            candidate_check_role=(
                "fixed_retained_history_Gamma_union_W_guard"
                if self.retain_unknown
                else (
                    "fixed_quarantine_and_temporary_seed_Gamma_union_W_guard"
                    if self.quarantine_enabled
                    else "frozen_monotone_closure_idempotence_guard"
                )
            ),
            consistency_rejected_cells=sum(
                e.get("rejected_by_consistency_check", 0)
                for e in self.events
                if e["kind"] == "supervision"
            ),
        )
        if current_valid:
            np.savez_compressed(
                self.output / "FINAL_P.npz",
                row_ids=np.asarray(self.rows, dtype=str),
                labels=np.asarray(self.labels, dtype=str),
                positive=available_predictions(result),
                negative=result.negative_facts | getattr(result, "arbitration_negative", False),
                logical_positive=result.positive_facts,
                logical_negative=result.negative_facts,
                conflict_unknown=unknown_mask(result),
                remaining_unknown=getattr(result, "remaining_unknown", unknown_mask(result)),
                arbitration_positive=getattr(
                    result, "arbitration_positive", np.zeros_like(result.positive_facts)
                ),
                arbitration_negative=getattr(
                    result, "arbitration_negative", np.zeros_like(result.negative_facts)
                ),
                fallback_positive=getattr(
                    result, "fallback_positive", np.zeros_like(result.positive_facts)
                ),
            )
            report["state"] = state_identity(result)
            report["fallback"] = (
                None if self.fallback is None else dict(
                    schema="loris_undecided_cell_model_fallback_v1",
                    calibration=self.fallback["calibration"],
                    thresholds={label: float(t) for label, t in zip(self.labels, self.fallback_thresholds)},
                    filled_cells=int(np.asarray(getattr(result, "fallback_positive", False)).sum()),
                ))
            report["commit_gate"] = dict(policy=self.commit_gate, best_score=self.best_gate_score,
                                         rejected=sum(e["kind"] == "update_rejected_by_commit_gate"
                                                      for e in self.events))
            report["gamma_conflict_policy"] = "validated_gamma_wins" if self.gamma_wins else "quarantine"
            # Scoring is post-run only; none of these targets enters feedback,
            # calibration, RDG or trainer through this class.
            ids = np.asarray(self.context.get("receiver_indices", ()), dtype=int)
            if len(ids) and "truth" in self.context:
                allowed = np.isin(self.roles[ids], self.manifest["domains"]["scoring_roles"])
                y = np.asarray(self.context["truth"])
                if y.shape != (len(ids), self.k):
                    raise ValueError("declared scoring truth axis differs")
                ids = ids[allowed]
                y = y[allowed]
                report["normal_metrics"] = (
                    metrics(y, available_predictions(result)[ids]) if len(ids) else None
                )
                unanswered = np.array([int(i) not in self.answered for i in ids], bool)
                report["not_directly_answered_metrics"] = (
                    metrics(y[unanswered], available_predictions(result)[ids[unanswered]])
                    if unanswered.any()
                    else None
                )
                report["evaluation_scope"] = (
                    "declared_development_or_interactive_scoring_not_automatically_independent_test"
                )
                for item in report["round_history"]:
                    with np.load(self.output / item["file"]) as snapshot:
                        positions = {int(row): i for i, row in enumerate(snapshot["rows"])}
                        if any(int(row) not in positions for row in ids):
                            raise ValueError("round snapshot omits declared scoring rows")
                        selected = np.asarray([positions[int(row)] for row in ids], dtype=int)
                        item["labeling_metrics"] = (
                            metrics(y, snapshot["output_positive"][selected]) if len(ids) else None
                        )
                        item["output_metrics"] = item["labeling_metrics"]
                        item["logical_metrics"] = (
                            metrics(y, snapshot["logical"][selected]) if len(ids) else None
                        )
                        item["aux_model_metrics_at_0_5"] = (
                            metrics(y, snapshot["aux_raw"][selected]) if len(ids) else None
                        )
                        item["dynamic_predicate_F1"] = {}
                        item["negative_event_F1"] = {}
                        for occurrence, label in item["occurrence_labels"].items():
                            polarity = self.channel_polarities.get(
                                tuple(map(int, occurrence.split("_"))), 1)
                            event = snapshot[occurrence][selected, None]
                            item["dynamic_predicate_F1"][occurrence] = (
                                metrics(
                                    y[:, label : label + 1], event if polarity > 0 else ~event
                                )
                                if len(ids)
                                else None
                            )
                            if polarity < 0:
                                item["negative_event_F1"][occurrence] = (
                                    metrics(~y[:, label : label + 1].astype(bool), event)
                                    if len(ids) else None)
        write(self.output / "COMPLETE.json", report)
        return report

    def run(self):
        """Algorithm 2: new unseen decisions, or one annotation, then repeat."""
        version = self.version
        result = None
        session = None
        self.valid_gamma_id = None
        seen = set()
        refinements_since_answer = 0
        round_index = 0
        try:
            result, session, _ = self.chase(version, phase="initial_zero_update")
            result, session, valid = self.resolve(result, session, version)
            if not valid:
                return self.finish("pending_initial_conflict", None, version)
            self.valid_gamma_id = digest(seed_identity(self.gamma))
            self.record_round(version, result)
            # The paper's Supervised control adapts once on Gamma even when
            # the initial closure already labels every document. Other methods
            # retain Algorithm 2's unresolved-document stopping condition.
            while self.has_query_work(result) or (
                self.pending_verified_rows and self.manifest["strategy"] != "frozen"
                and self.manifest["feedback"].get("new_verified_supervision", "full_batch") == "full_batch"
            ) or (
                self.manifest["strategy"] == "supervised" and not self.supervised_done
            ):
                if (self.manifest["loop"].get("query_cadence", "after_saturation") == "each_closure"
                        and self.has_query_work(result)):
                    if remaining_budget(self.manifest["loop"].get("label_budget"), self.queries) == 0:
                        return self.finish("label_budget_exhausted", result, version)
                    result, session, query_status = self.query_batch(result, session, version)
                    if query_status is not None:
                        return self.finish(query_status, result, version)
                    refinements_since_answer = 0
                can_refine = self.manifest["strategy"] != "frozen" and not (
                    self.manifest["strategy"] == "supervised" and self.supervised_done
                )
                if can_refine:
                    if refinements_since_answer >= self.manifest["loop"]["max_inner_rounds"]:
                        return self.finish("inner_round_cap_not_saturated", result, version)
                    batch = self.feedback(result, version, session=session)
                    if batch.observed.any():
                        self.training_attempts += 1
                        refinements_since_answer += 1
                        folder = self.output / ("update_%05d" % self.training_attempts)
                        old_decisions = copy.deepcopy(self.decisions)
                        fit_counted = False
                        test_session = None
                        try:
                            training_started = time.monotonic()
                            prediction_options = (
                                {
                                    "prediction_sets": {
                                        "independent_calibration": self.calibration_prediction_texts
                                    }
                                }
                                if self.calibration_prediction_texts is not None
                                else {}
                            )
                            training_parent = (
                                self.initial_version.checkpoint
                                if self.aux_initialization == "m0" else version.checkpoint
                            )
                            self.event(
                                "model_training_started",
                                initialization=self.aux_initialization,
                                training_parent=training_parent,
                                previous_committed_checkpoint=version.checkpoint,
                                training_attempt=self.training_attempts,
                                training_rows=int(batch.observed.any(1).sum()),
                            )
                            speed_options = {
                                key: self.manifest["dynamic"][key]
                                for key in ("amp_skip_tolerance", "prediction_amp",
                                            "prediction_length_sort", "amp_init_scale",
                                            "reuse_application_predictions")
                                if key in self.manifest["dynamic"]
                            }
                            probabilities, fit = self.trainer(
                                **speed_options,
                                texts=[d.cnt for d in self.context["docs"]],
                                labels=self.labels,
                                batch=batch,
                                checkpoint=training_parent["path"],
                                checkpoint_sha256=training_parent["sha256"],
                                source=self.manifest["dynamic"]["backend_source"],
                                configuration=self.manifest["dynamic"]["configuration"],
                                device=self.manifest["dynamic"]["device"],
                                output=folder,
                                gpu_lock=self.manifest["dynamic"]["gpu_lock"],
                                max_seconds=self.manifest["loop"]["max_update_seconds"],
                                **prediction_options,
                            )
                            self.timings["refinement_seconds"] += (
                                time.monotonic() - training_started
                            )
                            self.actual_updates += int(fit["optimizer_updates"])
                            fit_counted = True
                            probabilities = freeze_probability(probabilities)
                            provisional = Version(
                                probabilities, fit["checkpoint"], version.channels, "uncommitted"
                            )
                            threshold_started = time.monotonic()
                            if self.manifest["strategy"] == "supervised":
                                channels = dict(version.channels)
                                calibration = dict(
                                    strategy="supervised",
                                    initial_kappa_frozen=True,
                                    reason="one adaptation of omega on Gamma, then freeze",
                                )
                            else:
                                channels, calibration = calibrate(
                                    probabilities,
                                    batch,
                                    version.channels,
                                    probabilities_by_channel=self.channel_tables(provisional),
                                    channel_polarities=self.channel_polarities,
                                    **self.manifest["calibration"],
                                )
                                from .aux_extension import clamp_channels
                                constrained = clamp_channels(self.manifest, channels, self.rules)
                                calibration["constraints"] = dict(
                                    changed=[list(key) for key in channels if channels[key] != constrained[key]],
                                    policy="disjoint_positive_negative_regions_shared_low_per_label")
                                channels = constrained
                            self.timings["threshold_calibration_seconds"] += time.monotonic() - threshold_started
                            write(folder / "CALIBRATION.json", calibration)
                            candidate = Version(
                                probabilities,
                                fit["checkpoint"],
                                channels,
                                digest(
                                    dict(
                                        checkpoint=fit["checkpoint"],
                                        probabilities=array_identity(probabilities),
                                        channels=[[list(k), list(v)] for k, v in channels.items()],
                                    )
                                ),
                            )
                            old_state = self.predicate_id(version)
                            new_state = self.predicate_id(candidate)
                            repeated = (
                                new_state == old_state or new_state in seen
                            ) and self.manifest["strategy"] != "supervised" and (
                                self.manifest["feedback"].get("rule_feedback_scope", "all") != "new_or_changed"
                            ) and self.aux_initialization != "m0"
                            if self.manifest["strategy"] == "supervised":
                                self.supervised_done = True
                            if repeated:
                                self.pending_verified_rows.clear()
                                self.event(
                                    "refinement_no_new_decisions",
                                    round=round_index,
                                    reason=(
                                        "unchanged" if new_state == old_state else "previously_seen"
                                    ),
                                    candidate_not_installed=True,
                                    retained_version=version.identity,
                                    seen_decision_states=len(seen),
                                )
                            else:
                                program_id = digest(
                                    [r.to_dict() for r in with_channels(self.rules, channels)]
                                )
                                self.decisions = [
                                    d for d in self.decisions if d["program_id"] == program_id
                                ]
                                if len(self.decisions) != len(old_decisions):
                                    self.event(
                                        "scope_requires_new_approval",
                                        removed_incompatible_prior_scope=len(old_decisions)
                                        - len(self.decisions),
                                        reason="exact_program_kappa_changed",
                                    )
                                tested, test_session, _ = self.chase(
                                    candidate,
                                    phase="candidate_updated_model",
                                    previous=(result, session),
                                    update_kind="chaseDelta",
                                )
                                if tested.status != "fixpoint":
                                    tested, test_session, resolved = self.resolve(
                                        tested, test_session, candidate
                                    )
                                    if not resolved:
                                        self.release(test_session)
                                        self.decisions = old_decisions
                                        self.event(
                                            "update_not_committed",
                                            reason="candidate_Chase_conflict",
                                            retained=version.identity,
                                            Gamma_retained=True,
                                        )
                                        if self.valid_gamma_id != digest(seed_identity(self.gamma)):
                                            previous = session
                                            result, session, _ = self.chase(
                                                version,
                                                phase="rollback_model_keep_new_answers",
                                                previous=(
                                                    (result, session)
                                                    if self.retain_unknown
                                                    else None
                                                ),
                                                update_kind=(
                                                    "chaseInc" if self.retain_unknown else None
                                                ),
                                            )
                                            self.release(previous)
                                            result, session, valid = self.resolve(
                                                result, session, version
                                            )
                                            if not valid:
                                                return self.finish(
                                                    "pending_current_Gamma_after_candidate_rejection",
                                                    None,
                                                    version,
                                                )
                                            self.valid_gamma_id = digest(seed_identity(self.gamma))
                                        return self.finish(
                                            "candidate_conflict_retaining_current_Gamma_state",
                                            result,
                                            version,
                                        )
                                next_reliability = self.reliability
                                next_reliability_record = self.current_reliability_record
                                if self.calibration_prediction_texts is not None:
                                    from .reliability import refresh_current

                                    prediction = fit.get("prediction_sets", {}).get(
                                        "independent_calibration"
                                    )
                                    if prediction is None:
                                        raise ValueError(
                                            "candidate lacks independent calibration predictions; current-model q cannot be reused"
                                        )
                                    if (
                                        self.actual_chases
                                        >= self.manifest["loop"]["max_chase_calls"]
                                    ):
                                        raise ResourceLimit(
                                            "Chase budget exhausted before current-model rule calibration"
                                        )
                                    candidate_manifest = copy.deepcopy(self.manifest)
                                    candidate_manifest["dynamic"][
                                        "checkpoint"
                                    ] = candidate.checkpoint
                                    candidate_context = dict(
                                        self.context,
                                        gamma=self.gamma,
                                        model_probabilities=self.tables_for(candidate),
                                        static_model_probabilities=self.tables,
                                    )

                                    def account_calibration():
                                        if (
                                            self.actual_chases
                                            >= self.manifest["loop"]["max_chase_calls"]
                                        ):
                                            raise ResourceLimit(
                                                "Chase budget exhausted before current-model rule calibration"
                                            )
                                        self.actual_chases += 1
                                        self.inference_counts["calibration"] += 1

                                    calibration_started = time.monotonic()
                                    try:
                                        next_reliability, next_reliability_record = refresh_current(
                                            self.reliability,
                                            candidate_context,
                                            with_channels(self.rules, channels),
                                            candidate_manifest,
                                            prediction,
                                            folder / "reliability",
                                            before_chase=account_calibration,
                                        )
                                    finally:
                                        self.timings["calibration_seconds"] += (
                                            time.monotonic() - calibration_started
                                        )
                                    self.event(
                                        "candidate_rule_calibration_complete",
                                        candidate=candidate.identity,
                                        artifact=next_reliability_record,
                                        model_checkpoint=candidate.checkpoint,
                                        zero_support_rules=sum(
                                            r["support"] == 0
                                            for r in next_reliability["rule_scores"]
                                        ),
                                    )
                                gate_record = None
                                if self.fallback_independent or self.commit_gate == "independent_calibration":
                                    from .reliability import load_fallback_record

                                    gate_record = load_fallback_record(
                                        next_reliability_record, next_reliability, candidate_context,
                                        with_channels(self.rules, channels), candidate_manifest, self.labels)
                                    write(folder / "GATE.json", dict(
                                        record=gate_record, best_before=self.best_gate_score,
                                        commit_gate=self.commit_gate))
                                if self.commit_gate == "independent_calibration":
                                    score = float(gate_record["gate"]["mean_f1"])
                                    tolerance = float(self.manifest["loop"].get("commit_gate_tolerance", 0.0))
                                    joint_decision = None
                                    if self.commit_gate_metric_policy == "aux_loris_joint":
                                        from .fallback import joint_gate_decision
                                        if self.current_joint_gate is None or "joint_gate" not in gate_record:
                                            raise ValueError("strict joint gate candidate lacks bound A/B metrics")
                                        if (gate_record["joint_gate"]["scoring_binding"] !=
                                                self.current_joint_gate["scoring_binding"]):
                                            raise ValueError("strict joint gate scoring rows/answers changed")
                                        joint_decision = joint_gate_decision(
                                            self.current_joint_gate, gate_record["joint_gate"],
                                            completeness_tolerance=self.manifest["loop"].get("commit_gate_comp_tolerance", 0.0),
                                            false_negative_tolerance=self.manifest["loop"].get("commit_gate_fn_tolerance", 0),
                                        )
                                        write(folder / "JOINT_GATE_DECISION.json", dict(
                                            parent=self.current_joint_gate, candidate=gate_record["joint_gate"],
                                            decision=joint_decision, checkpoint=candidate.checkpoint,
                                            candidate_version=candidate.identity))
                                    rejected = (not joint_decision["accepted"] if joint_decision is not None
                                                else self.best_gate_score is not None and score < self.best_gate_score - tolerance)
                                    if rejected:
                                        self.release(test_session)
                                        self.decisions = old_decisions
                                        self.pending_verified_rows.clear()
                                        self.event(
                                            "update_rejected_by_commit_gate",
                                            round=round_index,
                                            candidate_score=score,
                                            best_committed_score=self.best_gate_score,
                                            tolerance=tolerance,
                                            retained=version.identity,
                                            Gamma_retained=True,
                                            joint_gate=joint_decision,
                                        )
                                        if self.valid_gamma_id != digest(seed_identity(self.gamma)):
                                            previous = session
                                            result, session, _ = self.chase(
                                                version,
                                                phase="gate_rejection_keep_new_answers",
                                                previous=(result, session) if self.retain_unknown else None,
                                                update_kind="chaseInc" if self.retain_unknown else None,
                                            )
                                            self.release(previous)
                                            result, session, valid = self.resolve(result, session, version)
                                            if not valid:
                                                return self.finish(
                                                    "pending_current_Gamma_after_gate_rejection", None, version)
                                            self.valid_gamma_id = digest(seed_identity(self.gamma))
                                        continue
                                unchanged = facts_id(result) == facts_id(tested)
                                # Algorithm 2 inserts the old state only after a
                                # successful new state; repeated candidates never
                                # replace the current probabilities or thresholds.
                                seen.add(old_state)
                                change = self.decision_change(version, candidate)
                                self.release(session)
                                result, session, version = tested, test_session, candidate
                                if gate_record is not None:
                                    if self.commit_gate_metric_policy == "aux_loris_joint":
                                        self.current_joint_gate = copy.deepcopy(gate_record["joint_gate"])
                                    if self.fallback_independent:
                                        self.fallback_thresholds = np.array(
                                            [float(gate_record["thresholds"][label]) for label in self.labels])
                                        self._refresh_fallback(result, version)
                                    score = float(gate_record["gate"]["mean_f1"])
                                    self.best_gate_score = (score if self.best_gate_score is None
                                                            else max(self.best_gate_score, score))
                                if self.reliability_mode != "frozen_prior":
                                    # frozen_prior keeps its prior q; a refresh then
                                    # only supplies fallback thresholds and the gate.
                                    self.reliability = next_reliability
                                    self.current_reliability_record = next_reliability_record
                                self.reliability_version = candidate.identity
                                self.committed_updates += 1
                                self.previous_feedback_state = batch.feedback_state
                                self.pending_verified_rows.difference_update(
                                    batch.audit.get("new_verified_rows_available", ()))
                                round_index += 1
                                self.valid_gamma_id = digest(seed_identity(self.gamma))
                                self.record_round(version, result, change)
                                self.event(
                                    "model_update_committed",
                                    round=round_index,
                                    version=version.identity,
                                    unchanged_complete_facts=unchanged,
                                    all_actual_dynamic_occurrences_unchanged=change["changed"] == 0,
                                    seen_decision_states=len(seen),
                                )
                                continue
                        except (RuntimeError, ValueError, TimeoutError) as exc:
                            if not fit_counted:
                                failure = folder / "FAILED.json"
                                receipt = (
                                    json.loads(failure.read_text()) if failure.is_file() else {}
                                )
                                if type(receipt.get("actual_optimizer_updates")) is int:
                                    self.actual_updates += receipt["actual_optimizer_updates"]
                                else:
                                    self.unmeasured_failed_updates += 1
                            self.decisions = old_decisions
                            if test_session is not None and test_session is not session:
                                self.release(test_session)
                            if isinstance(exc, ResourceLimit):
                                raise
                            self.event(
                                "update_not_committed",
                                error_type=type(exc).__name__,
                                reason=str(exc),
                                Gamma_retained=True,
                            )
                            # A candidate's conflict resolution may have acquired
                            # real Gamma before its independent q calibration failed.
                            # Keep those answers and restore closure under the old
                            # model/scopes, not an obsolete pre-answer result.
                            if self.valid_gamma_id != digest(seed_identity(self.gamma)):
                                previous = session
                                result, session, _ = self.chase(
                                    version,
                                    phase="rollback_model_keep_new_answers_after_failed_refresh",
                                    previous=(result, session) if self.retain_unknown else None,
                                    update_kind="chaseInc" if self.retain_unknown else None,
                                )
                                self.release(previous)
                                result, session, valid = self.resolve(result, session, version)
                                if not valid:
                                    return self.finish(
                                        "pending_current_Gamma_after_failed_refresh", None, version
                                    )
                                self.valid_gamma_id = digest(seed_identity(self.gamma))
                            return self.finish(
                                "update_failed_retaining_zero_update_state", result, version
                            )
                    else:
                        self.pending_verified_rows.clear()
                        if self.manifest["strategy"] == "supervised":
                            self.supervised_done = True
                            self.event(
                                "supervised_not_applicable", reason="no_eligible_Gamma_supervision"
                            )
                        self.event(
                            "refinement_no_new_decisions",
                            reason="no_retained_corrections",
                            candidate_not_installed=True,
                        )
                if not self.has_query_work(result):
                    break
                if self.manifest["loop"].get("query_cadence", "after_saturation") == "each_closure":
                    continue
                if remaining_budget(self.manifest["loop"].get("label_budget"), self.queries) == 0:
                    return self.finish(
                        "conflict_unknown_saturated" if self.quarantine_enabled and unknown_mask(result).any()
                        else "label_budget_exhausted", result, version)
                result, session, query_status = self.query_batch(result, session, version)
                if query_status is not None:
                    return self.finish(query_status, result, version)
                refinements_since_answer = 0
            return self.finish("all_documents_resolved", result, version)
        except ResourceLimit as exc:
            self.event("resource_limit", reason=str(exc))
            return self.finish("pending_resource_limit", result, version)
        except BaseException as exc:
            write(
                self.output / "FAILED.json",
                dict(
                    error_type=type(exc).__name__,
                    error=str(exc),
                    events=self.events,
                    actual_Chase_calls=self.actual_chases,
                    old_artifacts_unchanged=True,
                ),
            )
            raise
        finally:
            for runner in {id(item): item for item in self.sessions}.values():
                runner.close()
