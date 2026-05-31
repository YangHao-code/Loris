"""run_loris_chase_pipeline.py — MIGRATION SHIM.

The chase pipeline has moved to the :mod:`loris.pipeline` package (Phase 6):
  * rule-discovery steps   -> loris.pipeline.steps
  * shared model/data steps -> loris.pipeline.shared
  * CLI parsing + main      -> loris.pipeline.orchestrator

Run it via ``python -m loris``. This module re-exports the public entry points
(and the model/data steps the golden harness monkeypatches) so existing
imports keep working. Deleted once all callers use ``loris.pipeline`` directly.
"""

from __future__ import annotations

# Orchestration
from loris.pipeline.orchestrator import main, parse_args  # noqa: F401

# Rule-discovery steps
from loris.pipeline.steps import (  # noqa: F401
    run_rule_discovery,
    run_rule_discovery_batch,
    _generate_label_cooccurrence_rules,
    _extract_error_driven_predicates,
)

# Shared model/data steps (the golden entry monkeypatches init_models here)
from loris.pipeline.shared import (  # noqa: F401
    HParams,
    DATASET_REGISTRY,
    load_data,
    configure_logging,
    init_models,
    train_models,
    evaluate_models_on_test,
    run_pattern_abstraction,
    run_dynamic_router,
    register_selected_models,
    _select_top_predicates,
    _select_cluster_models,
    save_and_print_results,
    analyze_rule_corrections,
    _setup_experiment,
    _PredictWrapper,
)

if __name__ == "__main__":
    main()
