"""Golden-master configuration for LORIS refactoring verification.

Defines a single, fully-deterministic small run used to capture "golden"
artifacts BEFORE refactoring, then to verify (bit-for-bit where possible) that
each migrated module reproduces the same outputs.

Rationale for the chosen flags (see plan):
  * ``aapd`` has processed CSVs on disk.
  * ``--no_encoder`` drops the heaviest pretrained encoder model.
  * small ``subset_size`` + low ``max_trials`` keep a single run to minutes.
  * ``pattern_mode=sim`` exercises all 9 predicate types + the full Chase path.

torch runs on CPU in this environment (cuda unavailable), which removes the
main source of nondeterminism. CPU model training (TF-IDF/LR/SVM), KMeans,
Optuna (single-process, seeded) and Chase are all deterministic.
"""

from __future__ import annotations

# CLI argument list passed verbatim to the legacy chase pipeline (and later to
# `python -m loris run`). Keep this list authoritative and unchanged across the
# whole refactor so golden artifacts stay comparable.
GOLDEN_ARGV = [
    "--dataset", "aapd",
    "--top_labels", "30",
    "--subset_size", "2000",
    "--max_trials", "20",
    "--no_encoder",
    "--two_val",
    "--rule_strategy", "batch",
    "--cluster_model_selection", "global",
    "--batch_metric_mode", "global_macro",
    "--track1_baseline", "blank",
    "--pattern_mode", "sim",
]

# Where golden artifacts are stored (relative to repo root).
GOLDEN_ARTIFACT_DIR = "tests/golden/artifacts/aapd_small"

# Artifacts captured from the experiment directory after a run, with the
# comparison mode used by compare.py.
#   "json_canonical" -> json.dumps(sort_keys=True) byte-equality
#   "npy_exact"      -> np.array_equal
#   "json_metrics"   -> numeric compare of selected float keys
GOLDEN_ARTIFACTS = {
    "rules.json": "json_canonical",
    "metrics.json": "json_metrics",
    "cluster_0_store.json": "json_canonical",
    "cluster_1_store.json": "json_canonical",
    "cluster_2_store.json": "json_canonical",
    "cluster_3_store.json": "json_canonical",
    "cluster_4_store.json": "json_canonical",
}

# Keys in metrics.json compared numerically (exact on CPU-only path).
GOLDEN_METRIC_KEYS = [
    "baseline_test_micro_f1",
    "baseline_test_macro_f1",
    "final_micro_f1",
    "final_macro_f1",
    "n_rules",
]

# Seeds to force before any run (see plan strategy C).
GOLDEN_SEED = 42
