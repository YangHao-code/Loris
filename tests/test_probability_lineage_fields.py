"""Optional metadata is checked, without requiring a file audit protocol."""

import copy
import json

import pytest

from loris._engine.typed_session import array_identity
from loris.labeling import cli
from tests.test_labeling_reliability_weights import runtime_manifest


def test_plain_source_declaration_needs_no_additional_hash(tmp_path):
    path, _ = runtime_manifest(tmp_path)
    _, _, _, _, receipt = cli.load(path)
    assert receipt["probability_lineage_check"] == dict(
        status="source_declaration_only", checked_fields=[], model_execution_proven=False
    )


@pytest.mark.parametrize("field", ["checkpoint_sha256", "probabilities_sha256", "alias", "row_ids", "labels"])
def test_explicit_source_fields_must_match_loaded_model_and_axes(tmp_path, field):
    path, _ = runtime_manifest(tmp_path)
    manifest, context, _, _, _ = cli.load(path)
    lineage = dict(
        checkpoint_sha256=manifest["dynamic"]["checkpoint"]["sha256"],
        probabilities_sha256=array_identity(context["model_probabilities"][manifest["dynamic"]["alias"]])["sha256"],
        alias=manifest["dynamic"]["alias"],
        row_ids=context["row_ids"],
        labels=context["labels"],
    )
    source = json.loads(path.read_text())
    source["dynamic"]["probability_lineage"] = copy.deepcopy(lineage)
    path.write_text(json.dumps(source))
    receipt = cli.load(path)[-1]
    assert set(receipt["probability_lineage_check"]["checked_fields"]) == set(lineage)
    assert not receipt["probability_lineage_check"]["model_execution_proven"]
    source["dynamic"]["probability_lineage"][field] = ["wrong"] if field in ("row_ids", "labels") else "wrong"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="probability lineage differs"):
        cli.load(path)
