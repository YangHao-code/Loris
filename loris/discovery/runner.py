"""Portable rich RLearner, preserving the established two evaluation stages.

The caller authenticates a manifest; this runner also validates exact axes,
complete fit roots, Pi/Si, immutable execution identities, and safe resumption.
No model training, upstream pattern selection or online labeling is launched.
"""

from dataclasses import dataclass, asdict
import copy
import fcntl
import gzip
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sqlite3
import time
from collections.abc import Mapping

import numpy as np
from loris._engine.rdl_reference import Rule, Label, AttrEqual, LabelCompare
from loris._engine.rlearner import (
    admit_cluster,
    array_identity,
    atomic_json,
    binary_metrics,
    canonical,
    digest,
    gamma_identity,
    immutable_json,
)
from loris._engine.rlearner_config import ModelSpec, ParameterDomain, PatternSpec
from loris._learning.artifacts import file_record
from loris._engine.array_cache import _readonly
from loris._storage.selection_execution import make_selection_runner
from loris._storage.selection_evaluator import (
    make_selection_evaluator,
    semantic_result_projection,
    wrap_selection_evaluator,
    execution_binding as selection_binding,
    _validate_payload,
)
from loris._storage.discovery_factory import (
    make_discovery_runner,
    execution_binding as proof_binding,
)
from .search.configuration import policy_binding
from .search.domain import build_config_space, DOMAINS, variable_names
from .search.admission import admit_ordered_cluster, ORDERS
from .search.sampler import RichCombinationTPESampler, build_fit_witness_prior
from .search.execution import execution_binding, run_rich_cluster_search
from .search.initialization import build_fit_initialized_space, FitInitializedSpace
from .search.random_joint import RandomJointSampler, RandomJointSpace
from loris.rule_forms import parse_author_rule, validate_program_rule_forms, trial_rule_form_counts

SCHEMA = "loris_portable_rich_discovery_v1"


@dataclass(frozen=True)
class RunOptions:
    seed: int
    budget_per_cluster: int
    workers: int
    initializer_slots: int
    label_rotation: str
    candidate_domain: str = "rich"
    admission_order: str = "standalone_score_desc"
    proposal_strategy: str = "accuracy_guided"
    max_variables: int = 2
    context_probes: int = 32
    acquisition_candidates: int = 64
    expected_body_size: float = 3.0
    initialization_workers: int = 1
    initializer_candidate_archive: str = "winners_and_counts"
    admission_policy: str = "strict"
    admission_workers: int = 8
    admission_batch_size: int = 8
    gamma_conflict_policy: str = "soft_fit_precision"
    fit_precision_floor: float = 0.9

    @property
    def gamma_protection(self):
        return self.gamma_conflict_policy == "soft_fit_precision"

    @property
    def precision_floor(self):
        return float(self.fit_precision_floor) if self.gamma_protection else None

    def validate(self, cluster_count):
        if self.gamma_conflict_policy not in GAMMA_CONFLICT_POLICIES:
            raise ValueError("gamma_conflict_policy must be strict or soft_fit_precision")
        if (type(self.fit_precision_floor) not in (int, float)
                or not 0 < self.fit_precision_floor <= 1):
            raise ValueError("fit_precision_floor must lie in (0, 1]")
        from .search.parallel_admission import POLICIES
        if self.admission_policy not in POLICIES:
            raise ValueError("unsupported admission policy")
        for v in (self.admission_workers, self.admission_batch_size):
            if type(v) is not int or v < 1:
                raise ValueError("positive admission workers and batch size required")
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or self.seed + 1009 * (cluster_count - 1) >= 2**32
        ):
            raise ValueError("seed and per-cluster seeds must be valid unsigned32 values")
        if type(self.budget_per_cluster) is not int or self.budget_per_cluster < 1:
            raise ValueError("positive integer BO budget per cluster required")
        if type(self.workers) is not int or not 1 <= self.workers <= cluster_count:
            raise ValueError("workers must be between one and the cluster count")
        if (type(self.initialization_workers) is not int
                or not 1 <= self.initialization_workers <= cluster_count):
            raise ValueError("initialization_workers must be between one and the cluster count")
        if self.initializer_candidate_archive not in ("all", "evaluated_with_analytic_skips", "winners_and_counts"):
            raise ValueError("unsupported initializer candidate archive")
        variable_names(self.max_variables)
        for value in (self.context_probes, self.acquisition_candidates):
            if type(value) is not int or value < 1:
                raise ValueError("positive integer context probe and acquisition budgets required")
        if (
            type(self.expected_body_size) not in (int, float)
            or not np.isfinite(self.expected_body_size)
            or self.expected_body_size <= 0
        ):
            raise ValueError("positive finite expected body size required")
        if type(self.initializer_slots) is not int or not 0 <= self.initializer_slots <= min(
            30, self.budget_per_cluster
        ):
            raise ValueError(
                "fit initializer slots must fit the existing first min(30,budget) startup slots"
            )
        if self.label_rotation not in ("axis", "seeded_cluster_rotation"):
            raise ValueError("explicit supported label_rotation required")
        if self.candidate_domain not in DOMAINS or self.admission_order not in ORDERS:
            raise ValueError("explicit supported candidate_domain and admission_order required")
        if self.proposal_strategy not in ("accuracy_guided", "random_joint"):
            raise ValueError("unsupported proposal_strategy")
        if self.proposal_strategy == "random_joint":
            if self.initializer_slots != 0 or self.admission_order != "random_consistent":
                raise ValueError(
                    "random_joint requires initializer_slots=0 and admission_order=random_consistent"
                )
        elif self.admission_order == "random_consistent":
            raise ValueError("random_consistent admission requires random_joint proposals")


def validate_inputs(context, si, patterns, relations, input_binding):
    # The public Python API must enforce exactly the CLI data boundary too.
    from .manifest import validate_context

    validate_context(context)
    required = {
        "docs",
        "labels",
        "gamma",
        "fit_indices",
        "receiver_indices",
        "truth",
        "source_gold",
        "row_ids",
        "row_roles",
        "model_probabilities",
        "model_decisions",
        "attribute_values",
        "attribute_presence",
        "context_id",
    }
    if not isinstance(context, Mapping) or not required <= set(context):
        raise ValueError("complete authenticated discovery context required")
    if not isinstance(input_binding, Mapping) or not input_binding:
        raise ValueError("manifest and upstream input identity binding required")
    canonical(input_binding)
    if (
        not isinstance(si, Mapping)
        or not si
        or any(
            not isinstance(c, str) or not c or c in (".", "..") or "/" in c or "\\" in c for c in si
        )
    ):
        raise ValueError("ordered nonempty string cluster axis required")
    if not isinstance(patterns, Mapping) or set(patterns) != set(si):
        raise ValueError("Pi and Si cluster axes differ")
    labels = tuple(context["labels"])
    n = len(context["docs"])
    k = len(labels)
    if (
        not n
        or not k
        or len(set(labels)) != k
        or any(not isinstance(x, str) or not x for x in labels)
    ):
        raise ValueError("nonempty unique string label axis required")
    if (
        len(context["row_ids"]) != n
        or len(set(context["row_ids"])) != n
        or len(context["row_roles"]) != n
    ):
        raise ValueError("unique full row identity and roles required")
    axes = {}
    for name in ("fit_indices", "receiver_indices"):
        a = np.asarray(context[name])
        if (
            a.ndim != 1
            or a.dtype.kind not in "iu"
            or len(a) == 0
            or len(np.unique(a)) != len(a)
            or np.any((a < 0) | (a >= n))
        ):
            raise ValueError("valid unique nonempty " + name + " required")
        axes[name] = a
    fit, receivers = axes["fit_indices"], axes["receiver_indices"]
    if np.intersect1d(fit, receivers).size:
        raise ValueError("fit roots and scoring rows must be disjoint")
    if any(
        role not in ("source_fit", "unlabeled_train", "val_ctrl", "excluded_dev") for role in context["row_roles"]
    ):
        raise ValueError("discovery permits only declared fit/train/dev row roles")
    if any(context["row_roles"][i] != "source_fit" for i in fit) or any(
        context["row_roles"][i] != "val_ctrl" for i in receivers
    ):
        raise ValueError("fit/receiver roles differ from their declared axes")
    if (
        any(context["row_roles"][i] == "test" for i in range(n))
        or np.asarray(context.get("test_indices", ())).size
    ):
        raise ValueError("test rows cannot enter discovery context or transductive scoring")
    gamma = context["gamma"]
    if gamma.pos.shape != (n, k) or gamma.neg.shape != (n, k) or np.any(gamma.pos & gamma.neg):
        raise ValueError("consistent full-axis signed Gamma required")
    if not np.all(gamma.pos[fit] | gamma.neg[fit]):
        raise ValueError("fit initialization needs complete signed fit answers")
    outside_fit = np.ones(n, bool)
    outside_fit[fit] = False
    if gamma.pos[outside_fit].any() or gamma.neg[outside_fit].any():
        raise ValueError("only explicit fit answers may seed discovery Gamma")
    truth = np.asarray(context["truth"])
    fit_truth = np.asarray(context["source_gold"].get("fit"))
    if truth.shape != (len(receivers), k) or not np.isin(truth, (0, 1)).all():
        raise ValueError("exact scoring truth axis required")
    if fit_truth.shape != (len(fit), k) or not np.array_equal(fit_truth, gamma.pos[fit]):
        raise ValueError("fit truth must equal explicit positive fit roots")
    if set(context["source_gold"]) - {"fit", "val_ctrl"}:
        raise ValueError("undeclared truth stores cannot enter rule discovery")
    for cluster, aliases in si.items():
        if not aliases or len(set(aliases)) != len(aliases):
            raise ValueError("each Si must contain real unique selected model aliases")
        for alias in aliases:
            if alias not in context["model_probabilities"]:
                raise ValueError("selected model missing from frozen probability tables")
            a = np.asarray(context["model_probabilities"][alias])
            if a.shape != (n, k) or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
                raise ValueError("selected model probability axis or values differ")
            d = np.asarray(context["model_decisions"].get(alias))
            if d.shape != (n, k) or not np.isin(d, (0, 1)).all():
                raise ValueError("selected model decision table is missing or differs")
        if any(not isinstance(p, PatternSpec) for p in patterns[cluster]):
            raise TypeError("Pi requires actual PatternSpec objects")
        if len({p.key for p in patterns[cluster]}) != len(patterns[cluster]):
            raise ValueError("Pi pattern keys must be unique within each cluster")
    if any(not isinstance(r, (AttrEqual, LabelCompare)) for r in relations):
        raise TypeError("explicit existing RDL relation atoms required")
    return tuple(si)


GAMMA_CONFLICT_POLICIES = ("strict", "soft_fit_precision")


def evaluator_pair(context, out, worker_tag=None, lexical_cache_dir=None, persistent_scores=True,
                   gamma_protection=False, fit_precision_floor=None):
    cache = out / "static_cache" if worker_tag is None else out / "worker_static_cache" / worker_tag
    reuse = out / "admission_reuse" if worker_tag is None else out / "worker_receipts" / worker_tag
    score_cache = (
        out / "program_scores.sqlite"
        if worker_tag is None
        else out / "worker_scores" / (worker_tag + ".sqlite")
    )
    runner = make_selection_runner(
        context, cache, subset_direction="standard", lexical_cache_dir=lexical_cache_dir,
        gamma_protection=gamma_protection,
    )
    evaluator = make_selection_evaluator(
        runner,
        labels=context["labels"],
        receiver_indices=context["receiver_indices"],
        truth=context["truth"],
        gamma=context["gamma"],
        context_id=context["context_id"],
        line7_mode="positive_macro_f1",
        head_info=lambda r: (
            (r.head.label, r.head.polarity == "neg") if isinstance(r.head, Label) else None
        ),
        cache_path=score_cache if persistent_scores else None,
        legacy_score_sources=[],
        fit_precision_floor=fit_precision_floor,
    )
    return runner, evaluator, wrap_selection_evaluator(evaluator, runner, reuse)


def _shared_fit_arrays(context, fit, aliases):
    """One immutable full-fit interpretation, with per-cluster alias views.

    This does not restrict fit rows to a cluster or change any source value.
    FitWitnessPrior still owns its dictionaries and model/feature axes.
    """
    return dict(
        probabilities={a: _readonly(context['model_probabilities'][a][fit].astype(np.float64))
                       for a in aliases},
        decisions={a: _readonly(context['model_decisions'][a][fit]) for a in aliases},
        positive=_readonly(context['gamma'].pos[fit]),
        negative=_readonly(context['gamma'].neg[fit]),
        head_labels=_readonly(np.asarray(context['source_gold']['fit'], bool)),
    )


def bo_worker(context, space, prior, position, out, options, lexical_cache_dir):
    """No selected prefix is accepted by this independent single-rule worker."""
    started = time.monotonic()
    folder = out / "learning" / ("cluster_%02d" % position)
    folder.mkdir(parents=True, exist_ok=True)
    runner, evaluator, admission = evaluator_pair(
        context, out, "cluster_%02d" % position, lexical_cache_dir,
        gamma_protection=options.gamma_protection, fit_precision_floor=options.precision_floor,
    )
    seed = options.seed + 1009 * position
    families = {f.include_key: f.kind for f in space.features}

    def factory(s):
        if options.proposal_strategy == "random_joint":
            return RandomJointSampler(tuple(families), seed=s, feature_families=families)
        return RichCombinationTPESampler(
            tuple(families),
            seed=s,
            prior=prior,
            feature_families=families,
            context_probes=options.context_probes,
            acquisition_candidates=options.acquisition_candidates,
            expected_body_size=options.expected_body_size,
        )

    binding = dict(factory(seed).protocol, execution_specialization=execution_binding())
    from loris._engine.discovery_io import ProgressThrottle
    progress_throttle = ProgressThrottle()

    def progress(record):
        if not progress_throttle.due(record):
            return
        value = dict(
            record,
            cluster=space.cluster_id,
            worker_seconds=time.monotonic() - started,
            actual_compute_seconds=evaluator.actual_compute_seconds,
        )
        atomic_json(folder / "worker_status.json", value)
        with (folder / "worker_progress.jsonl").open("a") as f:
            f.write(canonical(value) + "\n")

    try:
        run_rich_cluster_search(
            directory=folder,
            space=space,
            evaluator=admission,
            seed=seed,
            budget=options.budget_per_cluster,
            rule_parser=parse_author_rule,
            sampler_factory=factory,
            sampler_binding=binding,
            on_progress=progress,
            storage_mode="compact_full_indicators_v1",
        )
        receipt = dict(
            status="complete",
            cluster=space.cluster_id,
            worker_seconds=time.monotonic() - started,
            actual_Chase_calls=evaluator.actual_calls,
            cache_hits=evaluator.cache_hits,
            actual_compute_seconds=evaluator.actual_compute_seconds,
        )
        # Attempt costs are append-only; a resumed completed study need not have
        # the same wall time as its first process.
        with (folder / "WORKER_ATTEMPTS.jsonl").open("a") as f:
            f.write(canonical(receipt) + "\n")
    except BaseException as e:
        atomic_json(folder / "WORKER_FAILURE.json", dict(type=type(e).__name__, message=str(e)))
        with (folder / "WORKER_ATTEMPTS.jsonl").open("a") as f:
            f.write(
                canonical(
                    dict(
                        status="failed",
                        cluster=space.cluster_id,
                        worker_seconds=time.monotonic() - started,
                        actual_Chase_calls=evaluator.actual_calls,
                        cache_hits=evaluator.cache_hits,
                        actual_compute_seconds=evaluator.actual_compute_seconds,
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                )
                + "\n"
            )
        raise
    finally:
        evaluator.close()
        runner.close()
        if prior is not None:
            prior.close()


def merge_worker_scores(evaluator, out, cluster_count):
    """Import only exact same-schema/binding scores, checking every full payload.

    Independent worker databases avoid SELECT/run/INSERT duplicate-key races.
    This is not legacy-score reuse and never changes a score or candidate.
    """
    sources = []
    unique = set()
    for i in range(cluster_count):
        path = out / "worker_scores" / ("cluster_%02d.sqlite" % i)
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            binding = db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
            if binding is None or json.loads(binding[0]) != evaluator.binding:
                raise ValueError("worker scoring binding differs")
            checksum = hashlib.sha256()
            count = 0
            for key, value in db.execute("SELECT key,value FROM programs ORDER BY key"):
                _validate_payload(json.loads(value), key, evaluator.binding)
                checksum.update(canonical([key, value]).encode())
                count += 1
                unique.add(key)
                old = evaluator.db.execute(
                    "SELECT value FROM programs WHERE key=?", (key,)
                ).fetchone()
                if old is not None and old[0] != value:
                    raise RuntimeError("same program produced different complete scores")
                if old is None:
                    evaluator.db.execute("INSERT INTO programs VALUES(?,?)", (key, value))
            sources.append(
                dict(path=str(path), programs=count, ordered_rows_sha256=checksum.hexdigest())
            )
        finally:
            db.close()
    evaluator.db.commit()
    immutable_json(
        out / "SCORE_CACHE_MERGE.json",
        dict(
            schema="same_binding_exact_worker_score_merge_v1",
            binding_id=digest(evaluator.binding),
            sources=sources,
            unique_source_programs=len(unique),
            every_payload_validated=True,
            duplicate_keys_require_identical_complete_payload=True,
        ),
    )


def prove(context, rules, expected, out, lexical_cache_dir=None, gamma_protection=False):
    from loris._storage.implicit_seed_proof import provenance_root_records, validate_proof_roots
    if (out / "PROOF_REPLAY.json").exists():
        previous = json.loads((out / "PROOF_REPLAY.json").read_text())
        if (
            semantic_result_projection(previous["evaluation"])
            != semantic_result_projection(expected)
            or previous["facts"] != file_record(out / "FACTS.npz")
            or previous["execution"] != proof_binding()
            or previous["proof_records"] != file_record(out / "PROVENANCE.jsonl.gz")
        ):
            raise RuntimeError("existing proof receipt belongs to different facts or execution")
        validate_proof_roots(previous, out)
        roots = provenance_root_records(out / "PROVENANCE.jsonl.gz")
        if previous.get("implicit_seed_roots", []) != roots:
            raise RuntimeError("existing compact proof root bindings differ")
        return previous
    runner = make_discovery_runner(
        context,
        out / "proof_cache",
        subset_direction="standard",
        lexical_cache_dir=lexical_cache_dir,
        gamma_protection=gamma_protection,
    )
    started = time.monotonic()
    try:
        result = runner(tuple(rules), gamma=context["gamma"])
        idx = np.asarray(context["receiver_indices"])
        pos = result.positive_facts
        neg = result.negative_facts
        full = dict(
            status=result.status,
            n_rounds=result.n_rounds,
            conflicts=[[int(i), str(label)] for i, label in result.conflicts],
            comparison_conflicts=json.loads(
                canonical(list(getattr(result, "comparison_conflicts", ())))
            ),
            positive_metrics=(
                binary_metrics(context["truth"], pos[idx]) if result.status == "fixpoint" else None
            ),
            negative_metrics=(
                binary_metrics(~np.asarray(context["truth"], bool), neg[idx])
                if result.status == "fixpoint"
                else None
            ),
            full_state={
                k: array_identity(getattr(result, k))
                for k in ("positive_facts", "negative_facts", "predictions")
            },
        )
        for name in ("subset_facts", "strict_subset_facts"):
            full["full_state"][name] = getattr(result, name).identity()
        if semantic_result_projection(full) != semantic_result_projection(expected):
            raise RuntimeError("complete provenance replay differs from scoring execution")
        if result.status != "fixpoint":
            raise RuntimeError("selected program failed complete proof replay")
        proof = result.proof_dag.identity()
        if proof["schema"] != "exact_immutable_layered_proof_dag_v1":
            raise RuntimeError("complete proof required")
        proof_records = _write_provenance(result, out)
        if (out / "FACTS.npz").exists():
            with np.load(out / "FACTS.npz", allow_pickle=False) as z:
                if (
                    set(z.files) != {"pos", "neg"}
                    or not np.array_equal(z["pos"], pos)
                    or not np.array_equal(z["neg"], neg)
                ):
                    raise RuntimeError("interrupted proof facts differ from fresh replay")
        else:
            with (out / "FACTS.npz").open("xb") as f:
                np.savez_compressed(f, pos=pos, neg=neg)
        report = dict(
            evaluation=full,
            proof_identity=proof,
            seconds=time.monotonic() - started,
            complete_facts_conflicts_rounds_scores_equal=True,
            dev_documents=len(idx),
            dev_covered_documents=int(pos[idx].any(axis=1).sum()),
            dev_coverage=float(pos[idx].any(axis=1).mean()),
            facts=file_record(out / "FACTS.npz"),
            execution=proof_binding(),
            first_witness_is_not_all_supports=True,
        )
        report["proof_records"] = proof_records
        report["implicit_seed_roots"] = provenance_root_records(out / "PROVENANCE.jsonl.gz")
        report["proof_scope"] = (
            "complete first-witness DAG with explicit derived records, reconstructible bitmap roots, "
            "and exact full relation rows; not all alternative proofs"
        )
        immutable_json(out / "PROOF_REPLAY.json", report)
        return report
    finally:
        runner.close()


def _write_provenance(result, out):
    """Stream the final DAG and exact relation rows without duplicating the DAG.

    Layer identities retain root/derived separation. Runtime first witnesses
    are preserved; their availability never claims all alternative supports.
    """
    from loris._engine.typed_runtime import _plain
    from loris._storage.implicit_seed_proof import ImplicitSeedProof

    target = out / "PROVENANCE.jsonl.gz"
    temporary = out / "PROVENANCE.partial.jsonl.gz"
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:

            def emit(value):
                stream.write((canonical(value) + "\n").encode("utf-8"))

            emit(
                dict(
                    type="header",
                    schema="loris_complete_final_provenance_v3",
                    identity=result.proof_dag.identity(),
                )
            )
            layer = result.proof_dag
            depth = 0
            while layer is not None:
                emit(dict(type="layer", depth=depth, identity=layer.identity()))
                if isinstance(layer, ImplicitSeedProof):
                    record = layer.save(out / ("PROVENANCE_ROOTS_%02d.npz" % depth))
                    emit(dict(type="implicit_seed_roots", depth=depth, record=record))
                    layer = layer._base
                    depth += 1
                    continue
                for key in sorted(layer._derived):
                    value = (
                        layer._derived.plain(key)
                        if hasattr(layer._derived, "plain")
                        else _plain(layer._derived[key])
                    )
                    emit(dict(type="proof", depth=depth, key=key, value=value))
                layer = layer._base
                depth += 1
            for name in ("subset_facts", "strict_subset_facts"):
                relation = getattr(result, name)
                emit(
                    dict(
                        type="relation",
                        name=name,
                        identity=relation.identity(),
                        row_encoding="adaptive_defaults_v2",
                        default_diagonal=relation.rows.reflexive,
                        stored_rows=relation.rows.storage_counts()["stored_rows"],
                    )
                )
                # All omitted rows have the declared exact empty/diagonal
                # default. Stream only actual overrides; a reflexive relation
                # must not produce N padded hexadecimal integers on disk.
                for row, encoding, value in relation.rows.iter_overrides():
                    emit(
                        dict(
                            type="relation_row", name=name, row=row, encoding=encoding, value=value
                        )
                    )
    if target.exists():
        if file_record(target)["sha256"] != file_record(temporary)["sha256"]:
            raise ValueError("interrupted final provenance differs from complete replay")
        temporary.unlink()
    else:
        os.replace(temporary, target)
    return file_record(target)



def _initialize_worker(space, prior, cluster, position, out, options):
    """One independent fit initializer, with no BO or admission calls."""
    try:
        build_fit_initialized_space(
            space, prior, output_dir=out / "initialization" / cluster,
            initializer_slots=options.initializer_slots,
            label_rotation=options.label_rotation,
            rotation_seed=options.seed, cluster_position=position,
            initializer_candidate_archive=options.initializer_candidate_archive,
        )
    except BaseException as error:
        atomic_json(out / "initialization" / cluster / "WORKER_FAILURE.json",
                    dict(cluster=cluster, cluster_position=position,
                         type=type(error).__name__, message=str(error)))
        raise
    finally:
        prior.close()
        # The per-prior in-memory map outlives the SQLite connection; without
        # this release serial setup retains every completed cluster's masks.
        # Durable lexical arrays remain available for the subsequent BO worker.
        from .search.sampler import _LOCAL_PATTERN_MASKS
        _LOCAL_PATTERN_MASKS.pop(prior, None)


def initialize_spaces(spaces, priors, clusters, out, options, started):
    """Initialize independently; collect in declared order, never finish order.

    Fork shares immutable fit arrays, not an open connection. Each parallel
    prior must own a distinct lexical cache directory. No initializer uses a
    shared RNG or modifies the later per-cluster BO seed convention.
    """
    if not (len(spaces) == len(priors) == len(clusters)):
        raise ValueError("initialization axes differ")
    for prior in priors:
        prior.close()
    workers = options.initialization_workers
    if workers == 1:
        for position, (space, prior, cluster) in enumerate(zip(spaces, priors, clusters)):
            _initialize_worker(space, prior, cluster, position, out, options)
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("parallel initialization requires fork; use initialization_workers=1")
        cache_dirs = [str(Path(prior.cache_dir).resolve()) for prior in priors
                      if prior.cache_dir is not None]
        if len(set(cache_dirs)) != len(cache_dirs):
            raise ValueError("parallel initializers require independent lexical cache directories")
        children, next_position = {}, 0
        try:
            while next_position < len(spaces) or any(p.is_alive() for p in children.values()):
                if any(p.exitcode not in (None, 0) for p in children.values()):
                    raise RuntimeError("independent fit initializer failed; inspect WORKER_FAILURE.json")
                alive = sum(p.is_alive() for p in children.values())
                while next_position < len(spaces) and alive < workers:
                    i = next_position
                    child = multiprocessing.get_context("fork").Process(
                        target=_initialize_worker,
                        args=(spaces[i], priors[i], clusters[i], i, out, options),
                    )
                    child.start()
                    children[i] = child
                    next_position += 1
                    alive += 1
                atomic_json(out / "STATUS.json", dict(
                    phase="parallel_independent_fit_initialization",
                    workers=[dict(cluster=clusters[i], cluster_position=i,
                                  pid=p.pid, exitcode=p.exitcode)
                             for i, p in children.items()],
                    seconds=time.monotonic() - started,
                ))
                time.sleep(0.1)
            for child in children.values():
                child.join()
            if any(p.exitcode != 0 for p in children.values()):
                raise RuntimeError("independent fit initializer failed; inspect WORKER_FAILURE.json")
        except BaseException:
            for child in children.values():
                if child.is_alive():
                    child.terminate()
            for child in children.values():
                child.join()
            raise
    return [FitInitializedSpace(space, out / "initialization" / cluster)
            for space, cluster in zip(spaces, clusters)]


def all_bo(context, spaces, priors, out, options, lexical_cache_dir, started):
    for prior in priors:
        if prior is not None:
            prior.close()
    if options.workers == 1:
        for i, (space, prior) in enumerate(zip(spaces, priors)):
            bo_worker(context, space, prior, i, out, options, lexical_cache_dir)
        return
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("parallel rich discovery currently requires fork; use workers=1")
    children = {}
    next_position = 0
    try:
        while next_position < len(spaces) or any(p.is_alive() for p in children.values()):
            if any(p.exitcode not in (None, 0) for p in children.values()):
                raise RuntimeError("independent BO worker failed; preserve unfinished transactions")
            alive = sum(p.is_alive() for p in children.values())
            while next_position < len(spaces) and alive < options.workers:
                i = next_position
                child = multiprocessing.get_context("fork").Process(
                    target=bo_worker,
                    args=(context, spaces[i], priors[i], i, out, options, lexical_cache_dir),
                )
                child.start()
                children[i] = child
                next_position += 1
                alive += 1
            atomic_json(
                out / "STATUS.json",
                dict(
                    phase="parallel_independent_line7_BO",
                    workers=[
                        dict(cluster_position=i, pid=p.pid, exitcode=p.exitcode)
                        for i, p in children.items()
                    ],
                    total_budget=len(spaces) * options.budget_per_cluster,
                    seconds=time.monotonic() - started,
                ),
            )
            time.sleep(0.2)
        for child in children.values():
            child.join()
        if any(p.exitcode != 0 for p in children.values()):
            raise RuntimeError("independent BO worker failed")
    except BaseException:
        for child in children.values():
            if child.is_alive():
                child.terminate()
        for child in children.values():
            child.join()
        raise


def _run_identity(
    context, si, patterns, relations, input_binding, options, clusters, lexical_cache_dir
):
    import optuna
    from loris.predicates import predicate_to_dict

    pattern_axis = {
        c: [
            dict(
                key=p.key,
                predicate=predicate_to_dict(p.predicate),
                frequency_domain=(
                    None if p.frequency_domain is None else dict(p.frequency_domain.__dict__)
                ),
                frequency_ops=list(p.frequency_ops),
            )
            for p in patterns[c]
        ]
        for c in clusters
    }
    # All source references are to installed library modules, never experiment
    # scripts. Old records remain valid historical artifacts in their namespace.
    sources = [file_record(__file__)]
    for path in sorted(Path(__file__).with_name("search").glob("*.py")):
        sources.append(file_record(path))
    raw = dict(
        schema=SCHEMA,
        options=asdict(options),
        cluster_order=list(clusters),
        Si={c: list(si[c]) for c in clusters},
        Pi=pattern_axis,
        relations=[r.to_dict() for r in relations],
        input_binding=input_binding,
        row_ids=list(context["row_ids"]),
        labels=list(context["labels"]),
        row_roles=list(context["row_roles"]),
        fit_indices=np.asarray(context["fit_indices"]).tolist(),
        receiver_indices=np.asarray(context["receiver_indices"]).tolist(),
        gamma=gamma_identity(context["gamma"]),
        truth=array_identity(np.asarray(context["truth"])),
        probabilities={
            a: array_identity(np.asarray(p)) for a, p in context["model_probabilities"].items()
        },
        decisions={a: array_identity(np.asarray(p)) for a, p in context["model_decisions"].items()},
        document_text_metadata_sha256=digest([[d.cnt, d.ttl, d.mtd] for d in context["docs"]]),
        context_id=context["context_id"],
        sources=sources,
        grammar=policy_binding(),
        selection_execution=selection_binding(),
        proof_execution=proof_binding(),
        sampler_execution=execution_binding(),
        environment=dict(
            python=platform.python_version(), numpy=np.__version__, optuna=optuna.__version__
        ),
        lexical_cache_dir=(
            None if lexical_cache_dir is None else str(Path(lexical_cache_dir).resolve())
        ),
        old_namespace_resume_supported=False,
        initial_sigma=[],
        online_training=False,
    )

    # Attribute values may be tuples/arrays of strings or nested metadata.
    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value

    raw["attributes_sha256"] = digest(
        plain(dict(values=context["attribute_values"], presence=context["attribute_presence"]))
    )
    return plain(raw)


def _archive_index(out, clusters, options):
    entries = []
    for i, cluster in enumerate(clusters):
        folder = out / "learning" / ("cluster_%02d" % i)
        complete = json.loads((folder / "BO_COMPLETE.json").read_text())
        record_identity = _record_identity(folder, options.budget_per_cluster)
        if record_identity["trials_sha256"] != complete["all_trial_records_sha256"]:
            raise ValueError("BO completion differs from archived proposals")
        entries.append(
            dict(
                cluster=cluster,
                cluster_position=i,
                complete=file_record(folder / "BO_COMPLETE.json"),
                trial_count=complete["complete_trials"],
                all_trial_records_sha256=complete["all_trial_records_sha256"],
                records=record_identity,
                raw_trial_template=str(folder / "trial_%05d.json"),
                admission_template=str(folder / "admission/candidate_%05d.json"),
                optuna_database=file_record(folder / "optuna.sqlite"),
                sqlite_wal_present=(folder / "optuna.sqlite-wal").exists(),
                sampler_checkpoint=file_record(folder / "sampler.json"),
                sampler_state_directory=str(folder / "sampler_states"),
                initialization_complete=file_record(
                    out / "initialization" / cluster / "COMPLETE.json"
                ),
            )
        )
    return dict(
        schema="loris_completed_rich_archive_index_v1",
        clusters=entries,
        selected=file_record(out / "learning/SELECTED.json"),
        facts=file_record(out / "FACTS.npz"),
        proof=file_record(out / "PROOF_REPLAY.json"),
        proof_records=file_record(out / "PROVENANCE.jsonl.gz"),
        implicit_seed_roots=json.loads((out / "PROOF_REPLAY.json").read_text()).get("implicit_seed_roots", []),
        all_proposed_trials_retained=True,
        invalid_AST="null_with_lossless_parameters_and_reason_not_fabricated_AST",
        rejected_candidate_full_proof_DAG_materialized=False,
        changing_prefix_requires_whole_program_rescoring=True,
        live_SQLite_archive_requires_consistent_backup=True,
    )


def _record_identity(folder, count):
    """Bind every proposal and ordered admission, including rejected proposals."""
    trials = [json.loads((folder / ("trial_%05d.json" % i)).read_text()) for i in range(count)]
    admissions = [
        json.loads((folder / "admission" / ("candidate_%05d.json" % i)).read_text())
        for i in range(count)
    ]
    for i, (trial, admission) in enumerate(zip(trials, admissions)):
        if (
            trial["number"] != i
            or admission["trial_number"] != i
            or admission["candidate"] != trial["rule"]
            or admission["standalone"] != trial["standalone"]
        ):
            raise ValueError("proposal/admission identity differs")
    value = dict(count=count, trials_sha256=digest(trials), admissions_sha256=digest(admissions))
    if (folder / "admission/ORDER.json").exists():
        value["admission_order"] = file_record(folder / "admission/ORDER.json")
    return value


def run_discovery(
    *,
    context,
    si,
    patterns,
    relations,
    output,
    input_binding,
    seed,
    budget_per_cluster,
    workers=1,
    initializer_slots=0,
    label_rotation="axis",
    lexical_cache_dir=None,
    resume=False,
    candidate_domain="rich",
    admission_order="standalone_score_desc",
    proposal_strategy="accuracy_guided",
    max_variables=2,
    context_probes=32,
    acquisition_candidates=64,
    expected_body_size=3.0,
    initialization_workers=1,
    initializer_candidate_archive="winners_and_counts",
    admission_policy="strict",
    admission_workers=8,
    admission_batch_size=8,
    gamma_conflict_policy="soft_fit_precision",
    fit_precision_floor=0.9,
):
    """Run or exactly resume a new portable discovery namespace.

    Existing completed experiments are read-only inputs to separate inspection
    tools, not migratable sampler checkpoints. `resume` requires this exact
    source/input/options binding. Partial fit initialization is not resumable.
    """
    clusters = validate_inputs(context, si, patterns, relations, input_binding)
    options = RunOptions(
        seed,
        budget_per_cluster,
        workers,
        initializer_slots,
        label_rotation,
        candidate_domain,
        admission_order,
        proposal_strategy,
        max_variables,
        context_probes,
        acquisition_candidates,
        expected_body_size,
        initialization_workers,
        initializer_candidate_archive,
        admission_policy,
        admission_workers,
        admission_batch_size,
        gamma_conflict_policy,
        fit_precision_floor,
    )
    options.validate(len(clusters))
    if type(resume) is not bool:
        raise TypeError("explicit Boolean resume required")
    out = Path(output).resolve()
    existed = out.exists()
    if existed and not resume:
        raise FileExistsError("output exists; choose new output or explicit exact resume")
    if resume and (not existed or not (out / "INPUTS.json").is_file()):
        raise ValueError("resume requires an existing portable input transaction")
    identity = _run_identity(
        context, si, patterns, relations, input_binding, options, clusters, lexical_cache_dir
    )
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another discovery owns this output directory")
        try:
            immutable_json(out / "INPUTS.json", identity)
            if (out / "COMPLETE.json").exists():
                complete = json.loads((out / "COMPLETE.json").read_text())
                if complete.get("input_identity") != digest(identity):
                    raise ValueError("completed discovery belongs to different inputs")
                for key in ("selection", "proof", "archive"):
                    if file_record(complete[key]["path"]) != complete[key]:
                        raise ValueError("completed artifact changed: " + key)
                archive = json.loads(Path(complete["archive"]["path"]).read_text())
                for key in ("selected", "facts", "proof", "proof_records"):
                    if file_record(archive[key]["path"]) != archive[key]:
                        raise ValueError("completed archive artifact changed: " + key)
                for entry in archive["clusters"]:
                    for key in (
                        "complete",
                        "optuna_database",
                        "sampler_checkpoint",
                        "initialization_complete",
                    ):
                        if file_record(entry[key]["path"]) != entry[key]:
                            raise ValueError("completed BO artifact changed: " + key)
                    folder = Path(entry["complete"]["path"]).parent
                    if _record_identity(folder, entry["trial_count"]) != entry["records"]:
                        raise ValueError("completed proposal/admission archive changed")
                proof = json.loads(Path(complete["proof"]["path"]).read_text())
                if file_record(proof["facts"]["path"]) != proof["facts"]:
                    raise ValueError("completed proof facts changed")
                if file_record(proof["proof_records"]["path"]) != proof["proof_records"]:
                    raise ValueError("completed proof records changed")
                from loris._storage.implicit_seed_proof import validate_proof_roots
                roots = validate_proof_roots(proof, out)
                if archive.get("implicit_seed_roots", []) != roots:
                    raise ValueError("completed archive compact root bindings changed")
                return complete
            return _discover(
                context,
                si,
                patterns,
                relations,
                out,
                input_binding,
                identity,
                options,
                clusters,
                lexical_cache_dir,
            )
        except BaseException as error:
            atomic_json(
                out / "LAST_FAILURE.json",
                dict(
                    type=type(error).__name__,
                    message=str(error),
                    recovery="exact resume only; partial fit initialization requires preserved separate output",
                ),
            )
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _discover(
    context,
    si,
    patterns,
    relations,
    out,
    input_binding,
    identity,
    options,
    clusters,
    lexical_cache_dir,
):
    started = time.monotonic()
    fit = np.asarray(context["fit_indices"])
    gamma_id = gamma_identity(context["gamma"])
    priors = []
    spaces = []
    runner = evaluator = prefilter = None
    try:
        aliases = tuple(dict.fromkeys(a for cluster in clusters for a in si[cluster]))
        shared_fit = (_shared_fit_arrays(context, fit, aliases)
                      if options.proposal_strategy != "random_joint" else None)
        for position, cluster in enumerate(clusters):
            space = build_config_space(
                cluster_id=cluster,
                labels=context["labels"],
                patterns=patterns[cluster],
                models=tuple(
                    ModelSpec(a, True, ParameterDomain(low=0.0, high=1.0)) for a in si[cluster]
                ),
                variables=variable_names(options.max_variables),
                relations=relations,
                candidate_domain=options.candidate_domain,
            )
            if options.proposal_strategy == "random_joint":
                priors.append(None)
                spaces.append(RandomJointSpace(space, out / "initialization" / cluster))
                continue
            prior = build_fit_witness_prior(
                space,
                docs=[context["docs"][int(i)] for i in fit],
                row_ids=[context["row_ids"][int(i)] for i in fit],
                row_roles=[context["row_roles"][int(i)] for i in fit],
                labels=context["labels"],
                gamma_pos=shared_fit['positive'],
                gamma_neg=shared_fit['negative'],
                model_probabilities={
                    a: shared_fit['probabilities'][a]
                    for a in si[cluster]
                },
                model_decisions={a: shared_fit['decisions'][a] for a in si[cluster]},
                attribute_values={
                    a: tuple(v[int(i)] for i in fit) for a, v in context["attribute_values"].items()
                },
                attribute_presence={
                    a: np.asarray(v)[fit] for a, v in context["attribute_presence"].items()
                },
                input_files=context.get("input_files", {}),
                fit_head_labels=shared_fit['head_labels'],
                # Independent workers never open the same lexical SQLite.
                # Serial execution keeps its existing shared-text cache.
                cache_dir=(out / "fit_lexical_cache" / cluster
                           if options.initialization_workers > 1
                           else out / "fit_lexical_cache"),
            )
            priors.append(prior)
            spaces.append(space)
        if options.proposal_strategy != "random_joint":
            spaces = initialize_spaces(spaces, priors, clusters, out, options, started)
        runner, evaluator, admission = evaluator_pair(
            context, out, lexical_cache_dir=lexical_cache_dir,
            gamma_protection=options.gamma_protection, fit_precision_floor=options.precision_floor,
        )
        protocol = dict(
            schema=SCHEMA,
            input_identity=digest(identity),
            options=asdict(options),
            cluster_order=list(clusters),
            initial_sigma=[],
            historical_rules_loaded=False,
            spaces=[s.manifest() for s in spaces],
            priors=[None if p is None else p.binding for p in priors],
            evaluator=evaluator.binding,
            all_legal_zero_candidates_get_line12=True,
            complete_cluster_BO_then_original_order_admission=options.admission_order
            == "proposal_order",
            admission_order=options.admission_order,
            candidate_domain=options.candidate_domain,
            proposal_strategy=options.proposal_strategy,
            variable_proposal_policy="declared finite variable axis; structured fit recipes only for actual axes of at most two variables, full-axis TPE otherwise",
            validation_accuracy_used_for_selection=options.proposal_strategy == "accuracy_guided",
            total_BO_budget=len(clusters) * options.budget_per_cluster,
        )
        immutable_json(out / "learning/protocol.json", protocol)
        evaluator.close()
        runner.close()
        evaluator = runner = None
        previous_attempt_counts = []
        for i in range(len(clusters)):
            attempts = out / "learning" / ("cluster_%02d" % i) / "WORKER_ATTEMPTS.jsonl"
            previous_attempt_counts.append(
                len(attempts.read_text().splitlines()) if attempts.exists() else 0
            )
        all_bo(context, spaces, priors, out, options, lexical_cache_dir, started)
        runner, evaluator, admission = evaluator_pair(
            context, out, lexical_cache_dir=lexical_cache_dir,
            gamma_protection=options.gamma_protection, fit_precision_floor=options.precision_floor,
        )
        merge_worker_scores(evaluator, out, len(clusters))
        selected = []
        cluster_results = []
        if options.admission_policy == "batch_prefilter":
            from .search.parallel_admission import ParallelScorer
            prefilter = ParallelScorer(context, out, options.admission_workers, lexical_cache_dir,
                                       gamma_protection=options.gamma_protection,
                                       fit_precision_floor=options.precision_floor)
        for position, space in enumerate(spaces):
            folder = out / "learning" / ("cluster_%02d" % position)
            records = [
                json.loads((folder / ("trial_%05d.json" % i)).read_text())
                for i in range(options.budget_per_cluster)
            ]

            def progress(record):
                atomic_json(
                    out / "STATUS.json",
                    dict(record, cluster=space.cluster_id, seconds=time.monotonic() - started),
                )

            selected = admit_ordered_cluster(
                order=options.admission_order,
                directory=folder / "admission",
                records=records,
                selected=selected,
                evaluator=admission,
                rule_parser=parse_author_rule,
                on_progress=progress,
                policy=options.admission_policy,
                score_batch=prefilter,
                batch_size=options.admission_batch_size,
            )
            cluster_results.append(
                dict(
                    cluster=space.cluster_id,
                    complete_BO_trials=len(records),
                    admissible=sum(r["rule"] is not None for r in records),
                    rule_form_rejections=trial_rule_form_counts(records),
                    selected_after=len(selected),
                    evaluation=evaluator.evaluate(selected),
                )
            )
        final = dict(
            schema=SCHEMA,
            protocol_id=digest(protocol),
            rules=[r.to_dict() for r in selected],
            rule_form_audit=validate_program_rule_forms(selected),
            rule_form_rejected_candidates=sum(
                c["rule_form_rejections"]["rejected_candidates"] for c in cluster_results
            ),
            proposal_strategy=options.proposal_strategy,
            admission_order=options.admission_order,
            evaluation=evaluator.evaluate(selected),
            clusters=cluster_results,
            complete_BO_trials=len(clusters) * options.budget_per_cluster,
            all_candidates_sequentially_evaluated=options.admission_policy == "strict",
            admission_policy=options.admission_policy,
            admission_batch_size=options.admission_batch_size,
            historical_candidates_added=0,
        )
        immutable_json(out / "learning/SELECTED.json", final)
        final_proof_calls = 0 if (out / "PROOF_REPLAY.json").exists() else 1
        proof = prove(context, selected, final["evaluation"], out, lexical_cache_dir,
                      gamma_protection=options.gamma_protection)
        if gamma_identity(context["gamma"]) != gamma_id:
            raise RuntimeError("Gamma changed during discovery")
        attempts_by_cluster = [
            [
                json.loads(line)
                for line in (out / "learning" / ("cluster_%02d" % i) / "WORKER_ATTEMPTS.jsonl")
                .read_text()
                .splitlines()
            ]
            for i in range(len(clusters))
        ]
        worker_attempts = [r for attempts in attempts_by_cluster for r in attempts]
        current_attempts = [
            r
            for i, attempts in enumerate(attempts_by_cluster)
            for r in attempts[previous_attempt_counts[i] :]
        ]
        current_worker_calls = sum(r["actual_Chase_calls"] for r in current_attempts)
        archive = _archive_index(out, clusters, options)
        immutable_json(out / "BO_ARCHIVE_INDEX.json", archive)
        complete = dict(
            schema=SCHEMA,
            status="complete",
            input_identity=digest(identity),
            proposal_strategy=options.proposal_strategy,
            admission_order=options.admission_order,
            selected_rules=len(selected),
            rule_form_audit=final["rule_form_audit"],
            rule_form_rejected_candidates=final["rule_form_rejected_candidates"],
            complete_BO_trials=len(clusters) * options.budget_per_cluster,
            selection=file_record(out / "learning/SELECTED.json"),
            proof=file_record(out / "PROOF_REPLAY.json"),
            archive=file_record(out / "BO_ARCHIVE_INDEX.json"),
            seconds_this_finishing_attempt=time.monotonic() - started,
            actual_Chase_calls_this_attempt=evaluator.actual_calls,
            actual_Chase_calls_this_attempt_scope="legacy field: parent admission scoring only, not total",
            actual_parent_scoring_Chase_calls_this_attempt=evaluator.actual_calls,
            actual_worker_scoring_Chase_calls_this_attempt=current_worker_calls,
            actual_final_proof_Chase_calls_this_attempt=final_proof_calls,
            actual_prefilter_scoring_Chase_calls_this_attempt=prefilter.calls if prefilter else 0,
            actual_total_Chase_calls_this_attempt=evaluator.actual_calls
            + current_worker_calls
            + final_proof_calls + (prefilter.calls if prefilter else 0),
            actual_worker_scoring_Chase_calls_all_recorded_attempts=sum(
                r["actual_Chase_calls"] for r in worker_attempts
            ),
            current_worker_attempts=current_attempts,
            completed_worker_attempts=[r for r in worker_attempts if r["status"] == "complete"],
            failed_worker_attempts=[r for r in worker_attempts if r["status"] == "failed"],
            lifetime_total_cost_certified=False,
            cost_limitation="attempt counters cannot account for unrecorded work after SIGKILL; prior failed parent admission costs are not accumulated",
            feedback_updates=0,
            new_answers=0,
            test_evaluated=False,
            dev_coverage=proof["dev_coverage"],
            final_positive_metrics=final["evaluation"]["positive_metrics"],
        )
        # If the process stopped between these two final publications, preserve
        # the already committed cost receipt rather than inventing new timings.
        if (out / "learning/COMPLETE.json").exists():
            committed = json.loads((out / "learning/COMPLETE.json").read_text())
            for key in (
                "schema",
                "status",
                "input_identity",
                "selected_rules",
                "complete_BO_trials",
                "selection",
                "proof",
                "archive",
                "final_positive_metrics",
            ):
                if committed.get(key) != complete.get(key):
                    raise ValueError("previous final transaction differs: " + key)
            complete = committed
        immutable_json(out / "learning/COMPLETE.json", complete)
        immutable_json(out / "COMPLETE.json", complete)
        return complete
    finally:
        if prefilter is not None:
            prefilter.close()
        if evaluator is not None:
            evaluator.close()
        if runner is not None:
            runner.close()
        for prior in priors:
            if prior is not None:
                prior.close()
