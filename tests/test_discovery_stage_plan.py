"""The published stage planner creates explicit finite independent experiments."""

import importlib.util
import json
from pathlib import Path

import pytest


def planner():
    path = Path(__file__).resolve().parents[1] / "scripts/discovery/run_stages.py"
    spec = importlib.util.spec_from_file_location("loris_discovery_stage_plan_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path):
    return dict(
        schema="loris_full_inventory_discovery_plan_v1",
        dataset="synthetic_full_axis",
        prepared_manifest=str(tmp_path / "data/MANIFEST.json"),
        historical_exposure=str(tmp_path / "exposure.json"),
        workdir=str(tmp_path / "new_task"),
        seed=42,
        fit_fraction=dict(numerator=1, denominator=8),
        pretrained_model="local/base",
        feature_backbone="local/features",
        backend_source="local/backend",
        gpu_lock="local/lock",
        pool_settings="local/settings.json",
        title_field="title",
        semantic_metadata_fields=[],
        relation_attributes=["cnt", "ttl", "mtd"],
        n_clusters=4,
        workers=2,
        bo=dict(
            budget_per_cluster=6000,
            context_probes=128,
            acquisition_candidates=256,
            expected_body_size=3.0,
        ),
    )


def test_stage_plan_never_runs_or_writes_by_default(tmp_path, capsys, monkeypatch):
    module = planner()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(config(tmp_path)))
    monkeypatch.setattr(module, "execute", lambda *a, **k: pytest.fail("planning must not execute"))
    assert (
        module.main(["--config", str(path), "--stage", "bo", "--variables", "3", "--models", "all"])
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "planned_not_executed"
    assert plan["variant"] == "v3_all_seed42_B6000_I0"
    assert len(plan["commands"]) == 2
    assert plan["commands"][0][3] == "check" and plan["commands"][1][3] == "learn-rules"
    assert not (tmp_path / "new_task").exists()


def test_pattern_stage_really_removes_all_declared_supply_caps(tmp_path):
    module = planner()
    value = config(tmp_path)
    value["pattern_screening"] = "entropy_full_inventory"
    paths = module.layout(value)
    assert paths["patterns"].name == "patterns_full"
    command = module.commands(value, paths, "patterns")[0]
    for option in ("--body-quota", "--title-quota", "--tfidf-terms"):
        assert command[command.index(option) + 1] == "all"
    assert "--no-deduplicate-masks" in command and "--no-exclude-constant-masks" in command
    assert command[command.index("--screening") + 1] == "entropy"
    assert json.loads(command[command.index("--semantic-metadata-fields") + 1]) == []
    context = module.commands(value, paths, "context")[0]
    assert (
        context[context.index("--semantic-metadata-fields") + 1]
        == command[command.index("--semantic-metadata-fields") + 1]
    )
    prewarm = module.commands(value, paths, "prewarm")[0]
    assert prewarm[prewarm.index("--supply") + 1] == str(paths["patterns"])


def test_budget_model_variable_variants_and_resume_are_separate(tmp_path):
    module = planner()
    value = config(tmp_path)
    variants = [
        module.layout(value, **overrides)
        for overrides in (
            {},
            dict(variables=3),
            dict(models="all"),
            dict(budget=12000),
            dict(search_seed=43),
        )
    ]
    assert len({p["run"] for p in variants}) == len(variants)
    assert len({p["manifest"] for p in variants}) == len(variants)
    assert len({p["patterns"] for p in variants}) == 1
    assert module.commands(value, variants[0], "bo", resume=True)[-1][-1] == "--resume"
    assert "--resume" not in module.commands(value, variants[0], "bo")[0]


def test_existing_manifest_is_not_silently_rebound(tmp_path):
    module = planner()
    path = tmp_path / "bound.json"
    module.write_new(path, {"budget": 6000})
    before = path.read_bytes()
    module.write_new(path, {"budget": 6000})
    with pytest.raises(ValueError, match="differs"):
        module.write_new(path, {"budget": 12000})
    assert path.read_bytes() == before


def test_fit_initializer_choice_has_a_separate_run_but_reuses_full_inventory(tmp_path):
    module = planner()
    value = config(tmp_path)
    original = module.layout(value)
    assert original["initializer_slots"] == 0
    value["bo"]["initializer_slots"] = 10
    original_explicit = module.layout(value)
    value["bo"]["initializer_slots"] = 0
    zero = module.layout(value)
    assert zero["initializer_slots"] == 0
    assert zero["variant"] == "v2_learned_seed42_B6000_I0"
    assert zero["manifest"] != original_explicit["manifest"] and zero["run"] != original_explicit["run"]
    assert zero["patterns"] == original_explicit["patterns"]
    assert zero["pool"] == original_explicit["pool"]
    assert module.commands(value, zero, "patterns") == module.commands(value, original_explicit, "patterns")
    assert str(zero["manifest"]) in module.commands(value, zero, "bo")[-1]
    for invalid in (True, -1, 31, 0.5):
        value["bo"]["initializer_slots"] = invalid
        with pytest.raises(ValueError, match="initializer_slots"):
            module.layout(value)
    value["bo"]["initializer_slots"] = 10
    with pytest.raises(ValueError, match="initializer_slots"):
        module.layout(value, budget=9)


def test_capacity_check_distinguishes_raw_npz_budget_from_physical_lower_bound(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    module = planner()
    value = config(tmp_path)
    path = Path(value["prepared_manifest"])
    path.parent.mkdir()
    path.write_text(json.dumps(dict(counts=dict(train=1000, dev=100), labels=["A", "B"])))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
    result = module.disk_preflight(value, module.layout(value))
    assert result["status"] == "insufficient_declared_headroom"
    assert (
        result["remaining_raw_array_budget"]
        > result["guaranteed_uncompressed_array_lower_bound"]
        > 1024
    )
    assert "not a physical lower bound" in result["limits"]
    assert not Path(value["workdir"]).exists()


def test_actual_supervision_stage_uses_the_public_cli_and_saves_receipts(tmp_path):
    from tests.test_discovery_supervision import prepared

    module = planner()
    source = prepared(tmp_path / "source")
    value = config(tmp_path)
    value["prepared_manifest"] = str(source)
    # This tiny CLI integration is not a production capacity estimate. Keep an
    # explicit small reserve so the test also works on a <20 GiB /tmp volume.
    value["disk_reserve_gib"] = 0.01
    exposure = Path(value["historical_exposure"])
    exposure.write_text(json.dumps({"synthetic_fixture_only": True, "prior_models": []}))
    config_path = tmp_path / "stage_config.json"
    config_path.write_text(json.dumps(value))
    assert module.main(["--config", str(config_path), "--stage", "supervision", "--execute"]) == 0
    root = Path(value["workdir"])
    receipt = json.loads((root / "supervision/SUPERVISION_MANIFEST.json").read_text())
    assert receipt["status"] == "complete" and receipt["training_calls"] == receipt["BO_calls"] == 0
    attempts = list((root / "stage_attempts").glob("*/RESULT.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text())["status"] == "stage_completed"
    assert (attempts[0].parent / "command_00.log").stat().st_size > 0
    assert (attempts[0].parent / "DISK_PREFLIGHT.json").is_file()


def test_default_pattern_stage_screens_the_full_inventory_discriminatively(tmp_path):
    module = planner()
    value = config(tmp_path)
    paths = module.layout(value)
    assert paths["patterns"].name == "patterns_discriminative"
    command = module.commands(value, paths, "patterns")[0]
    for option in ("--body-quota", "--title-quota", "--tfidf-terms"):
        assert command[command.index(option) + 1] == "all"
    assert command[command.index("--screening") + 1] == "discriminative"
    assert "--deduplicate-masks" in command and "--exclude-constant-masks" in command
    assert command[command.index("--patterns-per-cluster") + 1] == "512"
    value["pattern_screening"] = "other"
    with pytest.raises(ValueError):
        module.layout(value)


@pytest.mark.parametrize("extra, expected", [
    ({}, ("discriminative", "patterns_discriminative")),
    ({"pattern_screening": "discriminative", "patterns_per_cluster": 256}, ("discriminative", "patterns_discriminative")),
    ({"pattern_screening": "entropy_full_inventory"}, ("entropy", "patterns_full")),
    ({"relation_screening": {"policy": "none"}}, ("discriminative", "patterns_discriminative")),
    ({"relation_screening": {"policy": "discriminative_blocked_equality_v1", "min_groups": 50}},
     ("discriminative", "patterns_discriminative")),
])
def test_plan_entry_point_accepts_declared_screening_fields(tmp_path, capsys, extra, expected):
    module = planner()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(config(tmp_path), **extra)))
    assert module.main(["--config", str(path), "--stage", "patterns"]) == 0
    plan = json.loads(capsys.readouterr().out)
    command = plan["commands"][0]
    assert command[command.index("--screening") + 1] == expected[0]
    assert Path(command[command.index("--output") + 1]).name == expected[1]
    if "patterns_per_cluster" in extra:
        assert command[command.index("--patterns-per-cluster") + 1] == "256"


@pytest.mark.parametrize("extra", [
    {"pattern_screening": "mcc"},
    {"pattern_screening": "entropy_full_inventory", "patterns_per_cluster": 256},
    {"patterns_per_cluster": 0},
    {"relation_screening": {"policy": "other"}},
    {"relation_screening": {"policy": "none", "min_groups": "100"}},
    {"relation_screening": "none"},
    {"unknown_field": 1},
])
def test_plan_entry_point_rejects_invalid_screening_fields(tmp_path, extra):
    module = planner()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(config(tmp_path), **extra)))
    with pytest.raises(ValueError):
        module.main(["--config", str(path), "--stage", "patterns"])
