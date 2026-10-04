"""A quarantined fixpoint alone never authenticates an LLM proposal as Gamma."""

import json

import numpy as np

from loris._engine.rdl_reference import Label
from loris.labeling.runner import LabelingRun
from tests.test_conflict_protocol_loop import (
    close,
    forbidden_training,
    rule,
    setup,
    temporary_case,
)


def provider(run, directory, targets, kind="llm"):
    directory.mkdir(exist_ok=True, parents=True)
    run.manifest["loop"]["label_budget"] = 1
    run.manifest["loop"]["query_answer_mode"] = "allow_partial"
    run.manifest["oracle"].update(kind="directory_llm_then_human", directory=str(directory))
    (directory / ("00000000." + kind + ".json")).write_text(
        json.dumps(
            dict(
                row_id="r0",
                labels=list(run.labels),
                targets=targets,
                provider=kind,
                provenance="synthetic quarantine consistency test",
            )
        )
    )


def negative_C_run(tmp_path):
    ctx, manifest = setup()
    run = LabelingRun(
        context=ctx,
        rules=(rule("C", sign="neg"), rule("C", Label("x", "B"))),
        manifest=manifest,
        output=tmp_path / "run",
        trainer=forbidden_training,
    )
    before, session, _ = run.chase(run.version, phase="initial_zero_update")
    assert before.negative_facts[0, 2] and not before.conflict_unknown.any()
    return run, before, session


def test_LLM_proposal_with_new_downstream_quarantine_is_not_promoted_to_Gamma(tmp_path):
    run, before, session = negative_C_run(tmp_path)
    try:
        provider(run, tmp_path / "answers", [None, 1, None, None])
        accepted = run.oracle_answer(
            0, "test_proposal", result=before, session=session, version=run.version
        )
        assert not accepted
        assert not run.gamma.pos.any() and not run.gamma.neg.any()
        receipt = json.loads((run.output / "EXTERNAL_00001.json").read_text())
        assert receipt["llm_consistency_checks"] == 1 and receipt["llm_accepted"] == 0
        assert receipt["status"] == "pending_external_answer"
        checked = [
            e
            for e in run.events
            if e["kind"] == "Chase" and e["phase"] == "LLM_proposal_validation"
        ]
        assert len(checked) == 1 and checked[0]["status"] == "fixpoint"
        with np.load(run.output / "chase_00001/FACTS.npz") as facts:
            assert facts["conflict_unknown"][0, 2]
    finally:
        close(run)


def test_LLM_cannot_certify_a_pair_already_quarantined_in_the_current_round(tmp_path):
    ctx, manifest = setup()
    run = LabelingRun(
        context=ctx,
        rules=(rule("A"), rule("A", sign="neg")),
        manifest=manifest,
        output=tmp_path / "run",
        trainer=forbidden_training,
    )
    try:
        before, session, _ = run.chase(run.version, phase="initial_zero_update")
        assert before.conflict_unknown[0, 0]
        provider(run, tmp_path / "answers", [1, None, None, None])
        accepted = run.oracle_answer(
            0, "test_proposal", result=before, session=session, version=run.version
        )
        assert not accepted and not run.gamma.pos.any()
        receipt = json.loads((run.output / "EXTERNAL_00001.json").read_text())
        assert receipt["llm_accepted"] == 0
    finally:
        close(run)


def test_new_quarantine_from_LLM_proposal_uses_human_fallback_without_calling_it_LLM_validation(
    tmp_path,
):
    run, before, session = negative_C_run(tmp_path)
    try:
        directory = tmp_path / "answers"
        provider(run, directory, [None, 1, None, None])
        provider(run, directory, [None, 1, None, None], kind="human")
        accepted = run.oracle_answer(
            0, "test_proposal", result=before, session=session, version=run.version
        )
        assert accepted and run.gamma.pos[0, 1]
        receipt = json.loads((run.output / "EXTERNAL_00001.json").read_text())
        assert receipt["status"] == "accepted_human"
        assert receipt["selected_provider"] == "human"
        assert receipt["llm_accepted"] == 0 and receipt["human_answers"] == 1
    finally:
        close(run)


def test_valid_LLM_proposal_checks_the_same_quarantine_and_temporary_seeds(tmp_path):
    run, version, _, result, session = temporary_case(tmp_path, extra_conflict=True)
    try:
        assert result.conflict_unknown[0, 3]
        assert result.temporary_seed[0][0, 0]
        provider(run, tmp_path / "answers", [None, None, 1, None])
        seen = []
        original = run.chase

        def record(*args, **kwargs):
            if kwargs.get("phase") == "LLM_proposal_validation":
                seen.append(kwargs)
            return original(*args, **kwargs)

        run.chase = record
        assert run.oracle_answer(
            0, "test_proposal", result=result, session=session, version=version
        )
        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0]["initial_unknown"], result.conflict_unknown)
        for sign in (0, 1):
            np.testing.assert_array_equal(
                seen[0]["temporary_seed"][sign], result.temporary_seed[sign]
            )
        assert run.gamma.pos[0, 2]
        assert not run.gamma.pos[0, 0] and not run.gamma.pos[0, 1]
        receipt = json.loads((run.output / "EXTERNAL_00001.json").read_text())
        assert receipt["llm_accepted"] == 1
    finally:
        close(run)
