import json
import numpy as np
import pytest

from loris.chase import GammaSeed
from loris.document import Document
from loris._engine.rdl_reference import Doc, Label, ML, Text, Rule
from loris.predicates import MatchPredicate
from loris.experiments.evaluation import (
    metrics,
    cell_changes,
    calibrate_thresholds,
    model_program,
    transform_program,
    evaluate_program,
)


def test_exact_threshold_and_model_program_semantics(tmp_path):
    from loris._storage.discovery_factory import make_discovery_runner

    p = np.array([[0.5, 0.4999999], [0.2, 0.9], [0.8, 0.3]])
    c = dict(
        docs=[Document(cnt="red") for _ in p],
        labels=["A", "B"],
        model_probabilities={"m": p},
        model_decisions={"m": np.zeros_like(p, bool)},
        attribute_values={},
        attribute_presence={},
        context_id="model-program",
    )
    gamma = GammaSeed(np.zeros_like(p, bool), np.zeros_like(p, bool))
    c["gamma"] = gamma
    runner = make_discovery_runner(c, tmp_path / "cache", subset_direction="standard")
    try:
        result = runner(model_program("m", c["labels"], [0.5, 0.5]), gamma=gamma)
        assert result.status == "fixpoint"
        assert np.array_equal(result.positive_facts, p >= 0.5)
        altered = runner(model_program("m", c["labels"], [0.9, 0.8]), gamma=gamma)
        assert np.array_equal(altered.positive_facts, p >= np.array([0.9, 0.8]))
        neg = np.zeros_like(p, bool)
        neg[0, 0] = True
        invalid = runner(
            model_program("m", c["labels"], [0.5, 0.5]), gamma=GammaSeed(gamma.pos, neg)
        )
        assert invalid.status != "fixpoint"
    finally:
        runner.close()


def test_delete_whole_rule_does_not_widen_body():
    text = Text("x", MatchPredicate("cnt", "red"))
    guarded = Rule((Doc("x"), text, ML("x", "m", "A")), Label("x", "A"))
    simple = Rule((Doc("x"), ML("x", "m", "A")), Label("x", "A"))
    assert transform_program([guarded, simple], "no_text") == ([simple], [0])
    wide, _ = transform_program([guarded], "no_ground")
    assert wide == [simple]


def test_model_threshold_grid_and_cell_attribution():
    y = np.array([[1, 0], [0, 1]], bool)
    p = np.array([[0.3, 0.2], [0.1, 0.4]])
    t = calibrate_thresholds(y, p, [0.2, 0.3, 0.5])
    assert np.array_equal(p >= t, y)
    changes = cell_changes(y, np.array([[0, 1], [0, 1]]), y)
    assert changes == dict(recovered_FN=1, added_FP=0, lost_TP=0, removed_FP=1)
    assert metrics(y, y)["multi_label_completeness"] is None
    assert metrics(np.zeros((2, 2), bool), np.zeros((2, 2), bool))["macro_f1"] == 0


def test_calibration_matches_runtime_float64_boundary():
    p = np.array([[0.9], [0.99]], np.float32)
    y = np.array([[0], [1]], bool)
    # float32(.9) is below the exact float64 grid point .9. The runtime casts
    # the stored probability to float64; numpy's float32 scalar comparison
    # would instead include this false positive and select a different grid.
    t = calibrate_thresholds(y, p, [0.9, 0.95])
    assert t.tolist() == [0.9]
    assert (p.astype(np.float64) >= t).tolist() == y.tolist()


def test_failed_chase_has_no_normal_score(tmp_path, monkeypatch):
    p = np.array([[0.9], [0.8], [0.7]])
    gp = np.zeros_like(p, bool)
    gn = np.zeros_like(p, bool)
    gn[0] = True
    c = dict(
        docs=[Document(cnt="red") for _ in p],
        labels=["A"],
        row_ids=["0", "1", "2"],
        row_roles=["source_fit", "unlabeled_train", "val_ctrl"],
        gamma=GammaSeed(gp, gn),
        fit_indices=np.array([0]),
        receiver_indices=np.array([2]),
        truth=np.array([[1]], bool),
        source_gold={"fit": gp[:1], "val_ctrl": np.array([[1]], bool)},
        model_probabilities={"m": p},
        model_decisions={"m": p >= 0.5},
        attribute_values={},
        attribute_presence={},
        context_id="failed-program",
    )
    monkeypatch.setattr(
        "loris.experiments.evaluation.load_inputs",
        lambda *_: dict(context=c, input_binding={"fixture": True}),
    )
    result = evaluate_program("fixture", None, tmp_path / "failed", alias="m")
    assert result["record"]["status"] != "fixpoint"
    assert result["record"]["metrics"] is None
    assert (tmp_path / "failed/FAILED_PARTIAL_FACTS.npz").exists()
    assert not (tmp_path / "failed/FACTS.npz").exists()


def test_registry_rejects_changed_source(tmp_path):
    from loris.experiments.registry import export_rules

    source = tmp_path / "selected.json"
    source.write_text('{"rules": []}')
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "completed": [
                    {
                        "dataset": "d",
                        "rules": 0,
                        "canonical_ordered_program": {"path": str(source), "sha256": "0" * 64},
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="identity"):
        export_rules(index, tmp_path / "registry")


def test_cluster_routing_is_not_one_combination_on_every_row():
    from loris.experiments.evaluation import routed_vote

    c = dict(
        labels=["A"],
        fit_indices=np.array([0, 1]),
        receiver_indices=np.array([2, 3]),
        attribute_values={"mtd.semantic_cluster_id": ["group-a", "group-b", "group-a", "group-b"]},
        model_decisions={"m1": np.ones((4, 1), bool), "m2": np.zeros((4, 1), bool)},
    )
    pred, mapping = routed_vote(c, {"0": ["m1"], "1": ["m2"]}, np.array([0, 1]))
    assert pred.tolist() == [[True], [False]]
    assert mapping == {"group-a": "0", "group-b": "1"}


def test_matrix_does_not_mark_missing_reliability_dependency_ready(tmp_path):
    from loris.experiments.matrix import prepare, run

    (tmp_path / "label.json").write_text(
        json.dumps(dict(weighting="uniform", loop={"label_budget": 0}, oracle={"kind": "disabled"}))
    )
    assets = dict(
        paper={},
        policy={},
        datasets={
            "fixture": dict(
                discovery_manifest="d.json",
                rules="r.json",
                completion={},
                labeling_manifest="label.json",
                fit_reliability_protocol="fit.json",
            )
        },
    )
    path = tmp_path / "assets.json"
    path.write_text(json.dumps(assets))
    output = tmp_path / "matrix"
    m = prepare(path, output)
    rows = {x["id"]: x for x in m["experiments"]}
    assert rows["fixture/fit_source_reliability"]["complete_Chase_calls"] == 2
    assert rows["fixture/feedback_confidence_only"]["status"] == "blocked_assets"
    assert rows["fixture/labeling_paper_rdg"]["status"] == "blocked_assets"
    assert rows["fixture/labeling_paper_rdg"]["commands"] == []
    assert rows["fixture/feedback_provenance"]["status"] == "blocked_assets"
    with pytest.raises(ValueError, match="not runnable"):
        run(output / "EXPERIMENT_MATRIX.json", "fixture/feedback_provenance", dry_run=True)


def test_reliability_statistics_are_never_reported_as_final_P(tmp_path):
    import csv
    from loris.discovery.manifest import file_record
    from loris.experiments.reporting import collect

    result = tmp_path / "RESULTS.json"
    result.write_text(
        json.dumps(
            dict(
                status="complete", seconds=1.2, costs=[{}, {}], support=[[0, 10]], correct=[[0, 10]]
            )
        )
    )
    complete = tmp_path / "COMPLETE.json"
    complete.write_text(json.dumps(dict(results=file_record(result))))
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            dict(
                experiments=[
                    dict(
                        id="d/fit",
                        dataset="d",
                        output=str(tmp_path / "unused"),
                        kind="fit_source_reliability",
                        status="complete_reused",
                        receipt=file_record(complete),
                    )
                ]
            )
        )
    )
    collect(matrix, tmp_path / "tables")
    rows = list(csv.DictReader((tmp_path / "tables/MAIN_RESULTS.csv").open()))
    assert len(rows) == 1 and rows[0]["macro_f1"] == ""
    assert rows[0]["object"] == "fit_source_reliability_not_final_P"
    costs = list(csv.DictReader((tmp_path / "tables/COSTS.csv").open()))
    assert costs[0]["Chase_calls"] == "2" and costs[0]["updates"] == "0"
