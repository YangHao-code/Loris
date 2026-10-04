"""Run the new protocol through feedback, round transitions and completion.

These synthetic programs exercise the actual public labeling loop; they are
not measurements of dataset quality or an incremental complexity proof.
"""

import json

import numpy as np
import pytest

from loris._engine.rdl_reference import Label, ML, RefGamma
from loris.experiments.states import completion_state_evidence
from loris.labeling.reliability import RETAINED_CHASE_SEMANTICS
from loris.labeling.runner import LabelingRun, Version, facts_id
from tests.test_conflict_protocol_loop import setup, rule, close, forbidden_training
from tests.test_relation_rule_reliability import scored_artifact


def retaining_setup():
    ctx, manifest = setup(labels=("A", "B", "C", "D", "E"))
    manifest["loop"]["unknown_semantics"] = "retain_consequences"
    return ctx, manifest


def chain():
    return (
        rule("A"),
        rule("B", Label("x", "A")),
        rule("C", Label("x", "B")),
        rule("A", Label("x", "B"), sign="neg"),
        rule("D", Label("x", "C")),
        rule("E"),
        rule("D", ML("x", "aux", "D", 0.8)),
        rule("E", ML("x", "aux", "E", 0.8)),
    )


@pytest.mark.parametrize("execution", ["incremental", "full_recompute"])
@pytest.mark.parametrize("weighting", ["uniform", "direct_rule_precision_discount"])
def test_retention_source_gate_and_same_history_Gamma_union_W(tmp_path, execution, weighting):
    ctx, manifest = retaining_setup()
    manifest["loop"]["execution"] = execution
    manifest["weighting"] = weighting
    # This unit fixture binds a known q without independent model estimation;
    # current-model calibration has its own end-to-end tests.
    manifest["feedback"]["rule_reliability_update"] = "frozen_prior"
    rules = chain()
    reliability = scored_artifact(rules, [(8, 7)] * len(rules))
    reliability["chase_semantics"] = RETAINED_CHASE_SEMANTICS
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=forbidden_training,
        reliability=reliability,
    )
    try:
        result, session, _ = run.chase(run.version, phase="initial")
        before = facts_id(result)
        assert result.conflict_unknown[0, 0]
        assert result.positive_facts[0].tolist() == [False, True, True, True, True]
        batch = run.feedback(result, run.version, session=session)
        assert batch.weak_mask[0].tolist() == [False, False, False, False, True]
        assert batch.audit["recorded_source_check"]["extra_Chase_calls"] == 0
        assert batch.audit["rejected_by_consistency_check"] == 0
        assert facts_id(result) == before
        assert not run.gamma.pos.any() and not run.gamma.neg.any()
        if weighting != "uniform":
            assert batch.weights[0, 4] == pytest.approx(0.8 * (1 - 0.5 * 0.8))
        check = json.loads((run.output / "chase_00001/STATE.json").read_text())
        assert check["phase"] == "feedback_check"
        assert check["inference_audit"]["maintenance"]["update_kind"] == "check"
    finally:
        close(run)


def test_new_annotation_round_unlocks_and_rechecks_unchanged_conflict_sources(tmp_path):
    ctx, manifest = retaining_setup()
    rules = chain()
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=forbidden_training,
    )
    try:
        first, session, _ = run.chase(run.version, phase="initial")
        p = run.gamma.pos.copy()
        p[1, 4] = True
        run.gamma = RefGamma(p, run.gamma.neg.copy())
        second, _, _ = run.chase(
            run.version,
            phase="new_verified_answer",
            previous=(first, session),
            update_kind="chaseInc",
        )
        assert second.conflict_unknown[0, 0]
        assert second.positive_facts[0, 1:4].all()
        assert second.positive_facts[1, 4]
        assert second.reference_audit["maintenance"]["unlocked_cells"] == 1
        assert run.inference_counts["chaseInc"] == 1
        assert not run.gamma.pos[0].any()
    finally:
        close(run)


def test_completion_names_history_dependent_state_separately(tmp_path):
    ctx, manifest = retaining_setup()
    manifest["strategy"] = "frozen"
    rules = (rule("A"), rule("A", sign="neg"), rule("A", ML("x", "aux", "A", 0.8)))
    run = LabelingRun(
        context=ctx,
        rules=rules,
        manifest=manifest,
        output=tmp_path / "run",
        trainer=forbidden_training,
    )
    report = run.run()
    assert report["status"] == "conflict_unknown_saturated"
    assert report["current_history_has_quarantined_fixpoint"]
    assert not report["current_Gamma_has_quarantined_fixpoint"]
    assert not report["current_Gamma_has_valid_closure"]
    assert report["effective_execution"] == "retained_incremental"
    assert report["state"]["retained_history"]
    evidence = completion_state_evidence(report, manifest, run.output)
    assert evidence["final_state_kind"] == "retained_quarantined_fixpoint"
    assert (run.output / "chase_00000/HISTORY.json.gz").is_file()


def test_actual_retained_runs_pass_Noinc_history_and_output_comparison(tmp_path):
    from tests.test_revision2_experiment_protocols import script

    (tmp_path / "effective").mkdir()
    for name, execution in (("main", "incremental"), ("noinc", "full_recompute")):
        ctx, manifest = retaining_setup()
        manifest["strategy"] = "frozen"
        manifest["loop"]["execution"] = execution
        (tmp_path / "effective" / ("labeling_" + name + ".json")).write_text(json.dumps(manifest))
        run = LabelingRun(
            context=ctx,
            rules=chain(),
            manifest=manifest,
            output=tmp_path / "labeling" / name,
            trainer=forbidden_training,
        )
        run.run()
    compared = script("check_noinc_equivalence").compare(tmp_path)
    assert compared["status"] == "equivalent"
    assert compared["execution_interpretation"] == "retained_history_incremental_vs_full_rule_scan"
    assert compared["incremental_chase_seconds"] > 0
    # This tiny run checks actual execution accounting, not an assumed speedup.
    assert compared["chase_speedup"] > 0
