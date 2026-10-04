#!/usr/bin/env python
"""Verify the RQ2 fixed-Gamma intervention from actual run outputs."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from loris.discovery.manifest import bound, file_record
from loris._engine.typed_session import seed_identity
from loris.experiments.states import completion_state_evidence, comparable_endpoint


def compare(workspace, variants):
    work = Path(workspace).resolve(strict=True)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("provide distinct actually executed variants")
    shared = None
    records = []
    for variant in variants:
        if not variant.replace("_", "").isalnum():
            raise ValueError("unsafe variant name")
        manifest_path = work / "effective" / ("labeling_" + variant + ".json")
        manifest = json.loads(manifest_path.read_text())
        control_path = manifest_path.with_suffix(".experiment.json")
        control = json.loads(control_path.read_text())
        if (
            control.get("supervision_protocol") != "fixed_gamma"
            or bound(control["effective_manifest"], work) != manifest_path.resolve()
        ):
            raise ValueError("fixed-Gamma effective-manifest identity is not certified")
        loop = manifest["loop"]
        conflict_policy = loop.get("conflict_policy", "halt")
        unknown_semantics = loop.get("unknown_semantics", "full_rebuild")
        if (
            loop["label_budget"] != 0
            or loop["source_review_budget"] != 0
            or conflict_policy not in ("halt", "unknown_arbitrate")
            or manifest["oracle"]["kind"] != "disabled"
        ):
            raise ValueError(
                "fixed stored Gamma requires zero external budgets, disabled provider and no rule-instance reviews"
            )
        signature = {name: manifest[name] for name in ("context", "program", "dynamic", "domains")}
        signature.update(
            conflict_policy=conflict_policy,
            unknown_semantics=unknown_semantics,
            arbitration=manifest.get("arbitration"),
        )
        if shared is None:
            shared = signature
        elif signature != shared:
            raise ValueError(
                "RQ2 variants differ in initial model, program, domains, Gamma or conflict/arbitration protocol"
            )
        context_path = bound(manifest["context"], work)
        with context_path.open("rb") as stream:
            context = pickle.load(stream)  # SHA-bound trusted local experiment asset
        gamma = context["gamma"]
        run = work / "labeling" / variant
        actual = json.loads((run / "EFFECTIVE_MANIFEST.json").read_text())
        for name in ("context", "program", "domains", "strategy", "weighting"):
            if actual[name] != manifest[name]:
                raise ValueError("actual run differs from the fixed-Gamma experiment manifest")
        if actual.get("arbitration") != manifest.get("arbitration"):
            raise ValueError(
                "actual arbitration configuration differs from the shared frozen thresholds"
            )
        if (
            actual["dynamic"]["checkpoint"]["sha256"] != manifest["dynamic"]["checkpoint"]["sha256"]
            or any(
                actual["dynamic"].get(name) != manifest["dynamic"].get(name)
                for name in (
                    "alias",
                    "configuration",
                    "ensemble_dependencies",
                    "probability_lineage",
                )
            )
            or actual["loop"]["label_budget"] != 0
            or actual["loop"]["source_review_budget"] != 0
            or actual["loop"].get("conflict_policy", "halt") != conflict_policy
            or actual["loop"].get("unknown_semantics", "full_rebuild") != unknown_semantics
            or actual["oracle"]["kind"] != "disabled"
        ):
            raise ValueError(
                "actual run changed the declared initial model or supervision intervention"
            )
        completion = json.loads((run / "COMPLETE.json").read_text())
        state_evidence = completion_state_evidence(completion, manifest, run)
        if not comparable_endpoint(completion):
            raise ValueError(
                "fixed-Gamma comparison requires a valid normal endpoint for every variant"
            )
        if (
            completion.get("actual_label_requests") != 0
            or completion.get("actual_source_reviews") != 0
        ):
            raise ValueError("an RQ2 variant actually requested new external supervision")
        roots = json.loads((run / "ROOTS_INITIAL.json").read_text())
        if roots != seed_identity(gamma):
            raise ValueError("actual initial Gamma differs from the shared bound context")
        if completion.get("Gamma") != seed_identity(gamma):
            raise ValueError("actual final Gamma identity changed, including relation roots")
        with np.load(run / "TRUSTED_GAMMA_FINAL.npz", allow_pickle=False) as final:
            if (
                set(final.files) != {"positive", "negative"}
                or not np.array_equal(final["positive"], gamma.pos)
                or not np.array_equal(final["negative"], gamma.neg)
            ):
                raise ValueError("actual final Gamma changed in a fixed-Gamma comparison")
        if manifest["strategy"] == "frozen":
            rounds = completion.get("round_history", [])
            if (
                completion.get("actual_optimizer_updates") != 0
                or completion.get("committed_model_updates") != 0
                or completion.get("checkpoint", {}).get("sha256")
                != manifest["dynamic"]["checkpoint"]["sha256"]
                or not rounds
                or not rounds[0].get("version")
                or rounds[0]["version"] != completion.get("version")
            ):
                raise ValueError(
                    "Frozen changed the initial model/threshold version or performed optimizer updates"
                )
        records.append(
            dict(
                variant=variant,
                effective_manifest=file_record(manifest_path),
                control=file_record(control_path),
                actual_manifest=file_record(run / "EFFECTIVE_MANIFEST.json"),
                completion=file_record(run / "COMPLETE.json"),
                initial_Gamma=file_record(run / "ROOTS_INITIAL.json"),
                final_Gamma=file_record(run / "TRUSTED_GAMMA_FINAL.npz"),
                endpoint=completion["status"],
                actual_label_requests=0,
                actual_source_reviews=0,
                conflict_policy=conflict_policy,
                final_state_kind=state_evidence["final_state_kind"],
                active_quarantine_pairs=state_evidence["active_quarantine_pairs"],
                arbitration_rules=(
                    file_record(run / "MODEL_ARBITRATION_RULES.json")
                    if conflict_policy == "unknown_arbitrate"
                    else None
                ),
            )
        )
    result = dict(
        schema="loris_fixed_gamma_comparison_v1",
        status="verified",
        intervention="all variants preserve the same stored Gamma; external supervision disabled; effective quarantined views may differ",
        same_initial_model_program_domains=True,
        same_initial_and_final_Gamma=True,
        same_stored_Gamma=True,
        same_effective_quarantine_view_claimed=False,
        no_new_external_supervision=True,
        matched_budget_alone_would_not_prove_same_supervision=True,
        variants=records,
    )
    target = work / "FIXED_GAMMA_COMPARISON.json"
    with target.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.workspace, args.variants), indent=2))
