"""Re-admit an authenticated historical BO pool; this never runs BO.

The public protocol applies the current author rule-form gate and strict
whole-program gain floor. Historical pools remain immutable and readable.
Models, roots, scoring rows, primitives and executable semantics must match.
Prefix-dependent evaluations are recomputed in a new isolated namespace.
"""

import json
import math
from pathlib import Path
import time

from loris._engine.rdl_reference import Rule
from loris._engine.rlearner import atomic_json, digest, immutable_json
from loris._engine.rlearner_config import ModelSpec, ParameterDomain
from loris._learning.artifacts import file_record
from loris.rule_forms import (
    POLICY as RULE_FORM_POLICY,
    inspect_author_rule_form,
    validate_program_rule_forms,
)
from .search.configuration import SCHEMA as CURRENT_CONFIGURATION_SCHEMA
from .archive import status, _verify_cluster, snapshot_sqlite
from .runner import (
    RunOptions,
    validate_inputs,
    evaluator_pair,
    prove,
    build_config_space,
    _run_identity,
)

SCHEMA = "loris_historical_pool_author_rule_form_refilter_v2"
DOMAIN_FIELDS = (
    "cluster_id",
    "labels",
    "variables",
    "models",
    "patterns",
    "relations",
    "head_kinds",
    "features",
    "semantics",
    "search",
)


def read_pool(source, si, patterns, relations, options):
    checked = status(source)
    if checked["status"] != "complete_verified_identity":
        raise ValueError("refilter requires a complete authenticated source discovery")
    run_protocol = json.loads((source / "learning/protocol.json").read_text())
    if (
        run_protocol.get("admission_order", "proposal_order") != "proposal_order"
        or run_protocol.get("candidate_domain", "rich") != "rich"
    ):
        raise ValueError(
            "this refilter protocol accepts only historical rich/proposal_order sources"
        )
    folders = sorted((source / "learning").glob("cluster_*"))
    if len(folders) != len(si):
        raise ValueError("source and current cluster axes differ")
    clusters = []
    prefix = []
    for i, (folder, cluster) in enumerate(zip(folders, si)):
        if folder.name != "cluster_%02d" % i:
            raise ValueError("source cluster sequence is not complete")
        verified, prefix = _verify_cluster(folder, prefix)
        protocol = json.loads((folder / "protocol.json").read_text())
        if (
            protocol["budget"] != options.budget_per_cluster
            or protocol["seed"] != options.seed + 1009 * i
        ):
            raise ValueError(
                "declared BO seed/budget must identify the historical pool; no new BO is run"
            )
        current = build_config_space(
            cluster_id=cluster,
            labels=protocol["space"]["labels"],
            patterns=patterns[cluster],
            models=tuple(
                ModelSpec(a, True, ParameterDomain(low=0.0, high=1.0)) for a in si[cluster]
            ),
            variables=("x", "y"),
            relations=relations,
        ).manifest()
        old = protocol["space"]
        if old.get("schema") not in (
            "paper36_author_free_combination_configuration_v1",
            CURRENT_CONFIGURATION_SCHEMA,
        ):
            raise ValueError(
                "only the author-approved rich grammar is supported for historical refilter"
            )
        if any(old.get(k) != current.get(k) for k in DOMAIN_FIELDS):
            raise ValueError(
                "historical pool has a different model/pattern/relation/parameter domain"
            )
        records = [
            json.loads((folder / ("trial_%05d.json" % n)).read_text())
            for n in range(protocol["budget"])
        ]
        clusters.append(
            dict(
                cluster=cluster,
                position=i,
                folder=folder,
                protocol=protocol,
                records=records,
                source=dict(
                    complete=file_record(folder / "BO_COMPLETE.json"),
                    protocol=file_record(folder / "protocol.json"),
                    proposals_sha256=digest(records),
                    verified=verified,
                ),
            )
        )
    selected_path = Path(json.loads((source / "COMPLETE.json").read_text())["selection"]["path"])
    if prefix != json.loads(selected_path.read_text())["rules"]:
        raise ValueError("source admissions do not reconstruct the completed final program")
    return checked, clusters


def refilter(
    *,
    source,
    output,
    min_gain,
    context,
    si,
    patterns,
    relations,
    input_binding,
    seed,
    budget_per_cluster,
    workers=1,
    initializer_slots=0,
    label_rotation="axis",
    lexical_cache_dir=None,
    candidate_domain="rich",
    admission_order="proposal_order",
    proposal_strategy="accuracy_guided",
    max_variables=2,
    context_probes=32,
    acquisition_candidates=64,
    expected_body_size=3.0,
    initialization_workers=1,
    initializer_candidate_archive="evaluated_with_analytic_skips",
    admission_policy="strict",
    admission_workers=8,
    admission_batch_size=8,
    gamma_conflict_policy=None,
    fit_precision_floor=None,
):
    """Historical pool replay from empty Sigma with delta > max(min_gain,1e-12).

    Source scores are referenced only after exact evaluator binding checks.
    The source score database is snapshotted with SQLite's backup API. Exact
    ordered-program keys can be reused only under identical evaluator binding;
    every unseen prefix gets a complete evaluation or an exact no-op proof.
    This first version intentionally has no refilter-resume option.
    """
    if type(min_gain) not in (float, int) or not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError("min_gain must be finite and nonnegative")
    if (
        admission_policy != "strict"
        or candidate_domain != "rich"
        or admission_order != "proposal_order"
        or proposal_strategy != "accuracy_guided"
        or max_variables != 2
        or context_probes != 32
        or acquisition_candidates != 64
        or expected_body_size != 3.0
    ):
        raise ValueError(
            "this refilter protocol supports original two-variable rich/proposal_order pools and original sampler settings only; use a new discovery for domain/sampler/order variants"
        )
    clusters = validate_inputs(context, si, patterns, relations, input_binding)
    # The source run's Gamma conflict semantics identify its scores; runs
    # written before the option existed were strict. None inherits them.
    source_options = json.loads((Path(source).resolve(strict=True) / "INPUTS.json").read_text()).get("options", {})
    inherited_policy = source_options.get("gamma_conflict_policy", "strict")
    inherited_floor = source_options.get("fit_precision_floor", 0.9)
    if gamma_conflict_policy not in (None, inherited_policy) or fit_precision_floor not in (
            None, inherited_floor):
        raise ValueError("refilter must use the source run's Gamma conflict policy and floor")
    options = RunOptions(seed, budget_per_cluster, workers, initializer_slots, label_rotation,
                         initialization_workers=initialization_workers,
                         initializer_candidate_archive=initializer_candidate_archive,
                         gamma_conflict_policy=inherited_policy,
                         fit_precision_floor=inherited_floor)
    options.validate(len(clusters))
    source = Path(source).resolve(strict=True)
    out = Path(output).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("refilter requires an independent output outside the source run")
    if out.exists():
        raise FileExistsError(
            "refilter output exists; interrupted records are retained, choose a new output"
        )
    checked, pool = read_pool(source, si, patterns, relations, options)
    identity = _run_identity(
        context, si, patterns, relations, input_binding, options, clusters, lexical_cache_dir
    )
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    engine = evaluator = None
    try:
        cache_snapshot = None
        if (source / "program_scores.sqlite").exists():
            cache_snapshot = snapshot_sqlite(
                source / "program_scores.sqlite", out / "program_scores.sqlite"
            )
        engine, evaluator, admission = evaluator_pair(
            context, out, lexical_cache_dir=lexical_cache_dir,
            gamma_protection=options.gamma_protection, fit_precision_floor=options.precision_floor,
        )
        if any(item["protocol"]["evaluator"] != evaluator.binding for item in pool):
            raise ValueError(
                "historical scoring identity differs: data/model/Gamma/eta/text/score/semantics must match"
            )
        epsilon = max(float(min_gain), 1e-12)
        protocol = dict(
            schema=SCHEMA,
            source=file_record(__file__),
            source_run=checked,
            source_input_identity=file_record(source / "INPUTS.json"),
            current_inputs=identity,
            source_pools=[item["source"] for item in pool],
            evaluator=evaluator.binding,
            min_gain=float(min_gain),
            strict_epsilon=epsilon,
            order="original cluster and trial order",
            initial_sigma=[],
            new_BO=0,
            training_updates=0,
            replay_mode="historical_pool_refilter_not_new_search",
            current_public_rule_form=dict(RULE_FORM_POLICY),
            rule_form_change="historical records retained; newly inadmissible candidates explicitly rejected before scoring",
            changed_prefix_requires_new_whole_program_evaluation=True,
            initialization_and_proposal_options_identify_source_only=True,
            grammar_source_relocation="domain/semantics checked; source namespace differs explicitly",
            standalone_scores_referenced_without_rerunning_BO=True,
            cache_snapshot=cache_snapshot,
            cache_policy="identical_binding_and_exact_ordered_program_only",
            resume_supported=False,
        )
        immutable_json(out / "PROTOCOL.json", protocol)
        selected = []
        number = 0
        form_rejections = 0
        for item in pool:
            for record in item["records"]:
                before = len(selected)
                raw = record["rule"]
                rule = None
                if raw is None:
                    decision = dict(
                        line=12,
                        accepted=False,
                        reason="original_invalid:" + record["no_body_reason"],
                    )
                else:
                    rule = Rule.from_dict(raw)
                    form = inspect_author_rule_form(rule)
                    if not form["admissible"]:
                        form_rejections += 1
                        decision = dict(
                            line=12,
                            accepted=False,
                            reason=form["reason"],
                            rule_form=form,
                            scoring_Chase_performed=False,
                        )
                    else:
                        decision = admission.admit(selected, rule, epsilon=epsilon)
                if decision["accepted"]:
                    selected.append(rule)
                receipt = dict(
                    number=number,
                    cluster=item["cluster"],
                    source_trial=record["number"],
                    source_record=file_record(
                        item["folder"] / ("trial_%05d.json" % record["number"])
                    ),
                    candidate=raw,
                    standalone=record["standalone"],
                    decision=decision,
                    selected_before=before,
                    selected_after=len(selected),
                    selected_program_key=evaluator.program_key(selected),
                    strict_epsilon=epsilon,
                )
                immutable_json(out / "decisions" / ("candidate_%05d.json" % number), receipt)
                number += 1
                atomic_json(
                    out / "STATUS.json",
                    dict(
                        phase="historical_pool_sequential_refilter",
                        processed=number,
                        total=checked["BO_trials"],
                        selected_rules=len(selected),
                        new_BO=0,
                        actual_Chase_calls=evaluator.actual_calls,
                    ),
                )
        final = dict(
            schema=SCHEMA,
            protocol_id=digest(protocol),
            rules=[r.to_dict() for r in selected],
            rule_form_audit=validate_program_rule_forms(selected),
            rule_form_rejected_candidates=form_rejections,
            evaluation=evaluator.evaluate(selected),
            source_BO_trials=checked["BO_trials"],
            candidates_processed=number,
            new_BO=0,
            replay_mode="historical_pool_refilter_not_new_search",
        )
        immutable_json(out / "SELECTED.json", final)
        proof = prove(context, selected, final["evaluation"], out, lexical_cache_dir,
                      gamma_protection=options.gamma_protection)
        decisions = [
            json.loads((out / "decisions" / ("candidate_%05d.json" % i)).read_text())
            for i in range(number)
        ]
        complete = dict(
            schema=SCHEMA,
            status="complete",
            new_BO=0,
            complete_BO_trials=0,
            source_BO_trials=checked["BO_trials"],
            candidates_processed=number,
            selected_rules=len(selected),
            rule_form_audit=final["rule_form_audit"],
            rule_form_rejected_candidates=form_rejections,
            selection=file_record(out / "SELECTED.json"),
            proof=file_record(out / "PROOF_REPLAY.json"),
            decisions_sha256=digest(decisions),
            protocol=file_record(out / "PROTOCOL.json"),
            actual_scoring_Chase_calls=evaluator.actual_calls,
            final_proof_Chase_calls=1,
            actual_Chase_calls=evaluator.actual_calls + 1,
            cache_hits=evaluator.cache_hits,
            seconds=time.monotonic() - started,
            dev_coverage=proof["dev_coverage"],
            final_positive_metrics=final["evaluation"]["positive_metrics"],
            new_answers=0,
            training_updates=0,
            method="historical_pool_refilter_not_new_BO",
            no_source_artifacts_changed=True,
        )
        immutable_json(out / "COMPLETE.json", complete)
        return complete
    except BaseException as error:
        atomic_json(
            out / "FAILURE.json",
            dict(
                status="failed",
                type=type(error).__name__,
                reason=str(error),
                new_BO=0,
                resume_supported=False,
                partial_records_preserved=True,
            ),
        )
        raise
    finally:
        if evaluator is not None:
            evaluator.close()
        if engine is not None:
            engine.close()
