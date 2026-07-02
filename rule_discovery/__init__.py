"""
rule_discovery
--------------
Accuracy-Guided Rule Discovery via Meta-learning for the LORIS framework.

Legacy package. RDL/RDLSet now live in :mod:`loris.rules.rdl` (re-bound by
``loris_rule_discovery``); the chase rule learner lives in
:mod:`loris.rules.discovery`. The experimental ``HybridRuleLearner`` and its
standalone pipeline were removed in the Phase 6 cleanup.
"""

from rule_discovery.loris_rule_discovery import (
    RDL,
    RDLSet,
    RuleLearner,
    evaluate_configuration,
)

__all__ = [
    "RDL",
    "RDLSet",
    "RuleLearner",
    "evaluate_configuration",
]
