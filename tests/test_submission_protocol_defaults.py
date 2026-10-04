"""The clean package defaults describe the current formal protocol."""

import inspect
import json
from pathlib import Path
import sys

from loris.discovery import runner
from loris.discovery.search import initialization
from loris.discovery import model_pool
from loris.labeling.query_policy import batch_size


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backends/transformer/src"))
from loris_ssl.config import ExperimentConfig


def test_templates_use_complete_full_train_and_current_search_budget():
    labeling = json.loads((ROOT / "configs/templates/labeling.template.json").read_text())
    discovery = json.loads((ROOT / "configs/templates/discovery.template.json").read_text())
    plan = json.loads((ROOT / "scripts/discovery/plan.template.json").read_text())
    # 2026-10-01: auto = bf16 on compute capability >= 8, fp16 (loss scaling) on V100.
    assert labeling["dynamic"]["configuration"]["amp"] == "auto"
    assert labeling["dynamic"]["amp_init_scale"] == 1024
    assert labeling["dynamic"]["prediction_amp"] == "fp16"
    assert labeling["loop"]["gamma_conflict_policy"] == "validated_gamma_wins"
    assert labeling["loop"]["commit_gate"] == "independent_calibration"
    assert labeling["fallback"]["enabled"] is True
    assert labeling["fallback"]["calibration"] == "independent_calibration"
    assert labeling["calibration"]["cells"] == "trusted_only"
    assert labeling["calibration"]["minimum_support_each_sign"] == 10
    assert labeling["loop"]["query_batch_fraction"] == 0.01
    assert labeling["loop"]["query_batch_min"] == 32
    assert labeling["loop"]["query_answer_mode"] == "complete"
    assert labeling["feedback"]["new_verified_supervision"] == "full_batch"
    assert labeling["loop"]["query_cadence"] == "each_closure"
    assert labeling["loop"]["max_inner_rounds"] == 1
    assert labeling["feedback"]["trusted_replay_fraction"] == 0.05
    assert labeling["feedback"]["rule_feedback_scope"] == "new_or_changed"
    assert discovery["run"]["budget_per_cluster"] == 6000
    assert discovery["run"]["initializer_slots"] == 0
    assert plan["bo"]["budget_per_cluster"] == 6000
    assert plan["bo"]["initializer_slots"] == 0


def test_runtime_defaults_match_templates_and_analytic_archive_is_explicitly_overridable():
    assert batch_size({}, 10000, 10000, 0) == 100
    # 2026-09-30: default is the vectorized exact scan without per-candidate
    # rows; both original archive modes remain explicitly selectable.
    assert runner.RunOptions.__dataclass_fields__["initializer_candidate_archive"].default == "winners_and_counts"
    assert inspect.signature(runner.run_discovery).parameters["initializer_candidate_archive"].default == "winners_and_counts"
    assert inspect.signature(initialization.build_fit_initialized_space).parameters["initializer_candidate_archive"].default == "winners_and_counts"
    assert {"all", "evaluated_with_analytic_skips"} <= set(initialization.ARCHIVE_MODES)
    assert model_pool.model_configs


def test_auxiliary_model_pool_uses_bf16_on_cuda_without_removing_fp16_choice(monkeypatch, tmp_path):
    class FakeBackend:
        class ChaseBackendConfig:
            def __init__(self, **values):
                self.values = values

            def to_dict(self):
                return dict(self.values)

    monkeypatch.setattr(model_pool, "_backend", lambda source: FakeBackend)
    settings = model_pool.PoolSettings(device="cuda")
    configs = model_pool.model_configs(["A"], settings, tmp_path / "encoder", tmp_path / "backend")
    assert configs["aux_dynamic"]["amp"] == "bf16"
    assert ExperimentConfig(amp="fp16").amp == "fp16"
