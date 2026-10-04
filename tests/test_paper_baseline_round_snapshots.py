"""Paper Supervised semantics and prediction-only round snapshots."""

import json

import numpy as np
import pytest

from loris._engine.rdl_reference import Doc, Label, Rule, RefGamma
from loris._engine.typed_session import digest
from loris.labeling.runner import LabelingRun
from tests.test_labeling import fixture, synthetic_same_trainer


def covered_fixture():
    context, rules, manifest = fixture()
    rules = rules + (Rule((Doc("x"),), Label("x", "A")),)
    manifest["strategy"] = "supervised"
    return context, rules, manifest


@pytest.mark.parametrize("unknown", [False, True])
def test_supervised_fits_once_even_if_initial_rules_cover_every_document(tmp_path, unknown):
    context, rules, manifest = covered_fixture()
    if unknown:
        manifest["loop"]["conflict_policy"] = "unknown_arbitrate"
        manifest["arbitration"] = dict(alias="aux", default_threshold=0.5, thresholds={})
    calls = []

    def trainer(**kwargs):
        batch = kwargs["batch"]
        calls.append(batch)
        assert not batch.weak_mask.any()
        assert np.array_equal(batch.true_mask, context["gamma"].pos | context["gamma"].neg)
        return synthetic_same_trainer(**kwargs)

    run = LabelingRun(context=context, rules=rules, manifest=manifest, output=tmp_path / "run", trainer=trainer)
    report = run.run()
    assert len(calls) == report["training_attempts"] == 1
    assert report["committed_model_updates"] == report["actual_optimizer_updates"] == 1
    assert report["status"] == "all_documents_resolved"
    assert report["supervised_adaptation"]["status"] == "completed"
    assert report["symbolic_program_digest"] == digest([r.to_dict() for r in rules])
    assert report["actual_label_requests"] == 0
    assert len(report["round_history"]) == 2


def test_supervised_without_Gamma_reports_not_applicable_and_never_requests_for_covered_rows(tmp_path):
    context, rules, manifest = covered_fixture()
    context["gamma"] = RefGamma.empty(4, 2)
    def forbidden(**kwargs):
        pytest.fail("no Gamma supervision exists")
    run = LabelingRun(context=context, rules=rules, manifest=manifest, output=tmp_path / "run", trainer=forbidden)
    report = run.run()
    assert report["status"] == "all_documents_resolved"
    assert report["supervised_adaptation"]["status"] == "no_eligible_Gamma"
    assert report["training_attempts"] == report["actual_label_requests"] == 0


def test_rounds_without_truth_or_receiver_indices_preserve_scoring_axes(tmp_path):
    context, rules, manifest = fixture()
    context.pop("truth")
    context.pop("receiver_indices")
    manifest["strategy"] = "frozen"
    manifest["domains"]["scoring_roles"] = ["dev", "train"]
    run = LabelingRun(context=context, rules=rules, manifest=manifest, output=tmp_path / "run", trainer=None)
    report = run.run()
    with np.load(tmp_path / "run" / report["round_history"][0]["file"]) as snapshot:
        assert snapshot["rows"].tolist() == [1, 2, 3]
        assert snapshot["row_ids"].tolist() == ["r1", "r2", "r3"]
        assert snapshot["labels"].tolist() == ["A", "B"]
        assert snapshot["logical"].shape == (3, 2)
        assert "truth" not in snapshot.files
    with np.load(tmp_path / "run/FINAL_P.npz") as result:
        assert result["row_ids"].tolist() == context["row_ids"]
        assert result["labels"].tolist() == context["labels"]


def test_embedded_truth_joins_snapshot_by_row_when_receiver_order_differs(tmp_path):
    context, rules, manifest = fixture()
    context["receiver_indices"] = np.array([3, 1])
    context["truth"] = np.array([[0, 0], [1, 0]], bool)
    manifest["domains"]["scoring_roles"] = ["dev", "train"]
    manifest["strategy"] = "frozen"
    run = LabelingRun(context=context, rules=rules, manifest=manifest, output=tmp_path / "run", trainer=None)
    report = run.run()
    item = report["round_history"][0]
    assert item["labeling_metrics"]["micro_f1"] == 1
    assert item["logical_metrics"]["micro_f1"] == 1
    assert item["labeling_metrics"]["documents"] == 2


def test_REAL_supervised_already_covered_performs_one_Gamma_optimizer_update(tmp_path):
    from loris.discovery.manifest import file_record
    from loris.labeling.trainer import train_actual
    from tests.test_labeling_actual_aux import assets, Encoder, Tokenizer

    source, parent, configuration = assets(tmp_path)
    context, rules, manifest = covered_fixture()
    manifest["dynamic"].update(checkpoint=file_record(parent), backend_source=source, configuration=configuration)
    def actual(**kwargs):
        return train_actual(**kwargs, tokenizer_factory=lambda _: Tokenizer(), model_factory=lambda _, k: Encoder(k))
    run = LabelingRun(context=context, rules=rules, manifest=manifest, output=tmp_path / "run", trainer=actual)
    report = run.run()
    assert report["status"] == "all_documents_resolved"
    assert report["actual_optimizer_updates"] == report["committed_model_updates"] == 1
    assert report["supervised_adaptation"]["status"] == "completed"
    trace = json.loads((tmp_path / "run/update_00001/ACTUAL_TRAINING_TRACE.json").read_text())
    assert trace["sampled_rows"] == [0]
    assert all(window["weak_mass"] == 0 for window in trace["windows"])
