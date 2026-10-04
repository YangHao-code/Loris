"""Quality CLI handoff: fit inputs -> OOF Si -> actual tiny rule search."""
import json

import numpy as np
import pytest

from loris.discovery.__main__ import main
from loris.discovery.manifest import file_record, load_inputs
from loris.discovery.model_selection import verify_selection_output
from loris.discovery.quality_selection import (
    INPUT_SCHEMA, select_quality, load_quality_inputs, predict_quality, load_state,
)
from loris.discovery.quality_scoring import predict
from tests.test_discovery_variants import selection_fixture, discovery_manifest


def sidecar(tmp_path, manifest):
    data = json.loads(manifest.read_text())
    with np.load(data["arrays"]["path"]) as z:
        row_map = {row: i for i, row in enumerate(data["model_interpretation"]["row_ids"])}
        rows = [row_map[row] for row in data["row_ids"]]
        p = np.stack([z[data["model_interpretation"]["probabilities"][a]][rows] for a in data["aliases"]])
    path = tmp_path / "quality.npz"
    np.savez_compressed(path, groups=np.arange(len(rows)), calibrated_probabilities=p,
                        thresholds=np.full((len(data["aliases"]), len(data["labels"])), .5))
    description = dict(schema=INPUT_SCHEMA, selector_manifest=file_record(manifest), arrays=file_record(path),
                       calibration_provenance={"fixture": "identity mapping; thresholds=.5"},
                       group_provenance={"fixture": "independent documents"},
                       **{key:data[key] for key in ("row_ids", "aliases", "labels")})
    result = tmp_path / "quality.json"
    result.write_text(json.dumps(description))
    return result


@pytest.mark.parametrize("utility", ["probability_brier", "threshold_brier"])
@pytest.mark.parametrize("aggregation", ["frequency", "predicted_max"])
@pytest.mark.parametrize("scorer", ["static_global", "static_cluster", "ridge", "residual_mlp"])
def test_quality_receipt_replays_real_scorers_and_aggregations(tmp_path, utility, aggregation, scorer):
    manifest, _, _, _ = selection_fixture(tmp_path)
    quality = sidecar(tmp_path, manifest)
    out = tmp_path / "chosen"
    value = select_quality(manifest, quality, out, utility=utility, scorer=scorer,
                           aggregation=aggregation, folds=2, steps=2, hidden_dim=3, threads=1)
    verified = verify_selection_output(out / "COMPLETE.json")
    assert verified["result"]["selected_aliases_by_cluster"] == value["selected_aliases_by_cluster"]
    assert verify_selection_output(out / "SELECTED.json")["result"] == verified["result"]
    assert verified["result"]["new_BO"] == 0
    assert verified["result"]["selected_aliases_by_cluster"]["0"] == ["z_best", "m_second"] or set(
        verified["result"]["selected_aliases_by_cluster"]["0"]) == {"z_best", "m_second"}
    state = load_state(out / "MODEL.npz")
    assert state["method"] == scorer
    with pytest.raises(FileExistsError):
        select_quality(manifest, quality, out, utility=utility)


def test_cli_text_forward_rejects_truth_and_preserves_float_features(tmp_path):
    manifest, _, _, arrays = selection_fixture(tmp_path)
    quality = sidecar(tmp_path, manifest)
    out = tmp_path / "chosen"
    assert main(["select-quality", "fit", "--manifest", str(manifest), "--quality-inputs", str(quality),
                 "--output", str(out), "--utility", "threshold_brier", "--scorer", "residual_mlp",
                 "--folds", "2", "--steps", "2", "--threads", "1"]) == 0
    query = tmp_path / "text.npz"
    np.savez(query, features=arrays["features"].astype(np.float64), clusters=arrays["cluster_assignments"], row_ids=[10,11])
    scored = tmp_path / "scores.npz"
    assert main(["select-quality", "score", "--model", str(out / "MODEL.npz"),
                 "--inputs", str(query), "--output", str(scored)]) == 0
    with np.load(scored) as z:
        np.testing.assert_allclose(z["scores"], predict(load_state(out / "MODEL.npz"), arrays["features"], arrays["cluster_assignments"]))
    np.savez(query, features=arrays["features"], clusters=arrays["cluster_assignments"], row_ids=[10,11], truth=arrays["truth"])
    with pytest.raises(ValueError, match="text-only"):
        predict_quality(out / "MODEL.npz", query, tmp_path / "bad.npz")


def test_quality_axes_and_thresholds_checked_before_fit(tmp_path):
    manifest, _, _, _ = selection_fixture(tmp_path)
    quality = sidecar(tmp_path, manifest)
    d = json.loads(quality.read_text())
    d["aliases"] = d["aliases"][::-1]
    quality.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="aliases"):
        load_quality_inputs(manifest, quality)
    quality = sidecar(tmp_path, manifest)
    d = json.loads(quality.read_text())
    with np.load(tmp_path / "quality.npz") as z:
        values = {key:z[key] for key in z.files}
    values["thresholds"][:] = 1.
    np.savez(tmp_path / "quality.npz", **values)
    d["arrays"] = file_record(tmp_path / "quality.npz")
    quality.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="thresholds"):
        load_quality_inputs(manifest, quality)


def test_frozen_quality_Si_reaches_actual_BO_and_preserves_expert_probabilities(tmp_path):
    discovery = discovery_manifest(tmp_path)
    selector = tmp_path / "select.manifest.json"
    quality = sidecar(tmp_path, selector)
    out = tmp_path / "chosen"
    select_quality(selector, quality, out, utility="probability_brier", aggregation="predicted_max", folds=2)
    before = load_inputs(discovery)
    d = json.loads(discovery.read_text())
    d.update(model_selection=file_record(out / "COMPLETE.json"), selection_strategy="quality")
    discovery.write_text(json.dumps(d))
    after = load_inputs(discovery)
    for alias in before["context"]["model_probabilities"]:
        np.testing.assert_array_equal(before["context"]["model_probabilities"][alias], after["context"]["model_probabilities"][alias])
    assert after["si"] == json.loads((out / "SELECTED.json").read_text())["selected_aliases_by_cluster"]
    assert after["input_binding"]["selection_control"]["utility"] == "probability_brier"
    with pytest.raises(ValueError, match="actually trained"):
        load_inputs(discovery, {"selection_strategy":"learned"})
    run = tmp_path / "rules"
    assert main(["learn-rules", "--manifest", str(discovery), "--output", str(run), "--model-selection", "quality"]) == 0
    receipt = json.loads((run / "COMPLETE.json").read_text())
    assert receipt["complete_BO_trials"] == 4
    for trial in sorted((run / "learning/cluster_00").glob("trial_*.json")):
        record = json.loads(trial.read_text())
        if record["rule"]:
            from loris._engine.rdl_reference import Rule, ML
            rule = Rule.from_dict(record["rule"])
            assert all(atom.alias in after["si"]["0"] for atom in rule.body if isinstance(atom, ML))


def test_prepare_pack_rejects_reordered_axes_and_cli_binds_valid_pack(tmp_path):
    manifest, _, _, _ = selection_fixture(tmp_path)
    quality = sidecar(tmp_path, manifest)
    data = json.loads(manifest.read_text())
    with np.load(tmp_path / "quality.npz") as z:
        pack = {key:z[key] for key in z.files}
    pack.update({key:np.asarray(data[key]) for key in ("row_ids", "aliases", "labels")})
    source = tmp_path / "source.npz"
    np.savez(source, **pack)
    args = ["select-quality", "prepare", "--manifest", str(manifest), "--source-arrays", str(source),
            "--output", str(tmp_path / "prepared"), "--calibration-description", "identity fixture",
            "--group-description", "independent fixture rows"]
    assert main(args) == 0
    load_quality_inputs(manifest, tmp_path / "prepared/quality_inputs.json")
    pack["aliases"] = pack["aliases"][::-1]
    np.savez(source, **pack)
    args[args.index("--output") + 1] = str(tmp_path / "invalid")
    assert main(args) == 2
    assert not (tmp_path / "invalid").exists()


def test_verify_rejects_full_fit_scores_substituted_for_oof_even_if_receipt_updated(tmp_path):
    manifest, _, _, _ = selection_fixture(tmp_path)
    quality = sidecar(tmp_path, manifest)
    out = tmp_path / "chosen"
    select_quality(manifest, quality, out, utility="probability_brier", folds=2)
    with np.load(out / "SCORES.npz") as z:
        scores = {key:z[key] for key in z.files}
    assert not np.array_equal(scores["oof_scores"], scores["fit_scores"])
    scores["oof_scores"] = scores["fit_scores"]
    np.savez(out / "SCORES.npz", **scores)
    complete = json.loads((out / "COMPLETE.json").read_text())
    complete["scores"] = file_record(out / "SCORES.npz")
    (out / "COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="held-group"):
        verify_selection_output(out / "COMPLETE.json")


def test_comparison_plan_does_not_train_and_execute_binds_all_cases(tmp_path):
    from scripts.discovery.compare_quality import main as comparison_main

    base = discovery_manifest(tmp_path)
    selector = tmp_path / "select.manifest.json"
    quality = sidecar(tmp_path, selector)
    arguments = ["--selector-manifest", str(selector), "--quality-inputs", str(quality),
                 "--base-manifest", str(base), "--output", str(tmp_path / "plan"), "--k", "2",
                 "--folds", "2", "--workers", "1", "--scorers", "static_cluster"]
    assert comparison_main(arguments) == 0
    assert not list((tmp_path / "plan").rglob("MODEL.npz"))
    assert len(json.loads((tmp_path / "plan/PLAN.json").read_text())["cases"]) == 4
    arguments[arguments.index("--output") + 1] = str(tmp_path / "executed")
    assert comparison_main(arguments + ["--execute"]) == 0
    complete = json.loads((tmp_path / "executed/COMPLETE.json").read_text())
    assert complete["stage"] == "selectors_and_checks" and len(complete["cases"]) == 4
    for case in complete["cases"]:
        assert len(load_inputs(case["manifest"])["si"]["0"]) == 2
    assert not list((tmp_path / "executed").rglob("trial_*.json"))


def test_comparison_explicit_BO_uses_two_cluster_workers(tmp_path):
    from scripts.discovery.compare_quality import main as comparison_main
    from loris._engine.rlearner import array_identity

    base = discovery_manifest(tmp_path)
    selector = tmp_path / "select.manifest.json"
    data = json.loads(selector.read_text())
    array_path = tmp_path / "INPUT.npz"
    with np.load(array_path) as z:
        arrays = {key:z[key] for key in z.files}
    arrays["cluster_assignments"] = np.array([0,1], np.int64)
    np.savez(array_path, **arrays)
    data["arrays"] = file_record(array_path)
    selector.write_text(json.dumps(data))
    identity_path = tmp_path / "FIT_IDENTITY.json"
    identity = json.loads(identity_path.read_text())
    identity["clusters"] = array_identity(arrays["cluster_assignments"])
    identity_path.write_text(json.dumps(identity))
    patterns_path = tmp_path / "patterns.json"
    patterns = json.loads(patterns_path.read_text())
    patterns["1"] = patterns["0"]
    patterns_path.write_text(json.dumps(patterns))
    supply_path = tmp_path / "SUPPLY.json"
    supply = json.loads(supply_path.read_text())
    supply["patterns"] = file_record(patterns_path)
    supply_path.write_text(json.dumps(supply))
    base_data = json.loads(base.read_text())
    base_data["pattern_supply"] = file_record(supply_path)
    base.write_text(json.dumps(base_data))
    quality = sidecar(tmp_path, selector)
    out = tmp_path / "parallel"
    assert comparison_main(["--selector-manifest", str(selector), "--quality-inputs", str(quality),
                            "--base-manifest", str(base), "--output", str(out), "--k", "2", "--folds", "2",
                            "--utilities", "probability_brier", "--scorers", "static_cluster",
                            "--aggregations", "frequency", "--workers", "2", "--bo-trials", "2",
                            "--execute", "--execute-bo"]) == 0
    complete = json.loads((out / "COMPLETE.json").read_text())
    assert complete["stage"] == "rule_discovery"
    assert complete["cases"][0]["complete_BO_trials"] == 4
    folder = out / complete["cases"][0]["case"]
    assert load_inputs(folder / "discovery.json")["workers"] == 2
    assert len(list((folder / "rules/learning/cluster_01").glob("trial_*.json"))) == 2
