"""Directory-supplied LLM proposals with consistency checking and human fallback.

This adapter reads already supplied answers. It sends no messages and makes no
model/network call. Provider provenance must distinguish actual external
answers from simulated ones; a filename alone is not evidence of authorship.
"""

import json
from pathlib import Path

from loris.discovery.manifest import file_record


_ALLOWED = {"llm": {"llm", "simulated_llm"}, "human": {"human", "simulated_human"}}


def signed_answer(value, row_id, labels, kind):
    """Validate partial signed labels; JSON null remains unknown, never false."""
    required = {"row_id", "labels", "targets", "provider", "provenance"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("external answer requires row_id, labels, targets, provider, provenance")
    if value["row_id"] != row_id or value["labels"] != list(labels):
        raise ValueError("external answer row/label identity differs")
    y = value["targets"]
    if (
        not isinstance(y, list)
        or len(y) != len(labels)
        or any(v is not None and (type(v) is not int or v not in (0, 1)) for v in y)
    ):
        raise ValueError("external targets must be a full label-axis list of 0, 1, or null")
    if 1 not in y and (not y or any(v is None for v in y)):
        raise ValueError(
            "an annotation without positive labels must explicitly mark every label absent"
        )
    if value["provider"] not in _ALLOWED[kind]:
        raise ValueError(
            "external provider must distinguish actual and simulated " + kind + " answers"
        )
    if not isinstance(value["provenance"], str) or not value["provenance"].strip():
        raise ValueError("external answer source provenance is required")
    return dict(value, labels=list(labels), targets=list(y))


def external_answer(row, row_id, labels, directory, validate_llm):
    """Try one local LLM proposal, then an available local human answer.

    ``validate_llm(answer)`` must test the proposal under the current model,
    rules and trusted Gamma without committing it, returning a bool. False
    triggers human fallback. Exceptions propagate: interrupted/budget-limited
    consistency checks are not successful or completed rejected checks.

    Return ``(answer_or_None, receipt)``. Human answers retain their explicit
    signs and are handed back for trusted-root reconciliation by the caller.
    No consistency claim is made for the human fallback in this adapter.
    """
    if type(row) is not int or row < 0 or not isinstance(row_id, str) or not row_id:
        raise ValueError("external answer needs a nonnegative row and explicit document ID")
    labels = tuple(labels)
    if (
        not labels
        or len(set(labels)) != len(labels)
        or any(not isinstance(x, str) or not x for x in labels)
    ):
        raise ValueError("external answer requires a unique nonempty label axis")
    if not callable(validate_llm):
        raise ValueError("LLM proposals require a current-state consistency callback")
    directory = Path(directory)
    receipt = dict(
        status="pending_external_answer",
        selected_provider=None,
        row=row,
        row_id=row_id,
        llm_proposals=0,
        simulated_llm_proposals=0,
        llm_consistency_checks=0,
        llm_accepted=0,
        human_answers=0,
        simulated_human_answers=0,
        network_calls=0,
        provider_provenance_is_declared=True,
        observations=[],
    )
    proposal = directory / ("%08d.llm.json" % row)
    if proposal.is_file():
        source = file_record(proposal)
        try:
            candidate = signed_answer(json.loads(proposal.read_text()), row_id, labels, "llm")
        except (ValueError, TypeError) as error:
            receipt["observations"].append(
                dict(stage="llm", source=source, status="invalid_proposal", reason=str(error))
            )
        else:
            receipt["llm_proposals"] += 1
            receipt["simulated_llm_proposals"] += int(candidate["provider"] == "simulated_llm")
            receipt["llm_consistency_checks"] += 1
            accepted = validate_llm(candidate)
            if type(accepted) is not bool:
                raise TypeError("LLM consistency callback must return an explicit bool")
            receipt["observations"].append(
                dict(
                    stage="llm", source=source, status="consistent" if accepted else "inconsistent"
                )
            )
            if accepted:
                receipt.update(
                    status="accepted_llm",
                    selected_provider=candidate["provider"],
                    source=source,
                    llm_accepted=1,
                )
                return candidate, receipt
    else:
        receipt["observations"].append(dict(stage="llm", status="proposal_unavailable"))
    return _human_response(row, row_id, labels, directory, receipt)


def _human_response(row, row_id, labels, directory, receipt):
    human_path = directory / ("%08d.human.json" % row)
    if not human_path.is_file():
        receipt["observations"].append(dict(stage="human", status="answer_unavailable"))
        return None, receipt
    human = signed_answer(json.loads(human_path.read_text()), row_id, labels, "human")
    source = file_record(human_path)
    simulated = human["provider"] == "simulated_human"
    receipt.update(
        status="accepted_human",
        selected_provider=human["provider"],
        source=source,
        human_answers=int(not simulated),
        simulated_human_answers=int(simulated),
    )
    receipt["observations"].append(dict(stage="human", status="provided", source=source))
    return human, receipt


def human_answer(row, row_id, labels, directory):
    """Human-only NoLLM ablation; never read or validate an LLM proposal."""
    if type(row) is not int or row < 0 or not isinstance(row_id, str) or not row_id:
        raise ValueError("human answer requires an explicit row/document identity")
    labels = tuple(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("human answer requires a unique nonempty label axis")
    receipt = dict(
        status="pending_external_answer",
        selected_provider=None,
        row=row,
        row_id=row_id,
        llm_proposals=0,
        simulated_llm_proposals=0,
        llm_consistency_checks=0,
        llm_accepted=0,
        human_answers=0,
        simulated_human_answers=0,
        network_calls=0,
        provider_provenance_is_declared=True,
        observations=[],
        llm_stage_skipped=True,
    )
    return _human_response(row, row_id, labels, Path(directory), receipt)
