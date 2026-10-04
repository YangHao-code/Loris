"""Explicit configuration for the author's conflict-quarantine extension.

The arbitration program is declared separately from learned Sigma. Its model
parameters change only between closures; thresholds are explicit configuration,
not inferred from whichever learned rule happens to fire first.
"""

import numpy as np

from .arbitration import paired_rules


POLICY = "unknown_arbitrate"


def arbitration_rules(manifest, labels=None):
    config = manifest.get("arbitration")
    if not isinstance(config, dict) or set(config) != {"alias", "default_threshold", "thresholds"}:
        raise ValueError(
            "unknown_arbitrate requires explicit arbitration alias/default_threshold/thresholds"
        )
    expected_alias = manifest.get("aux_extension", {}).get(
        "source_alias", manifest["dynamic"]["alias"])
    if config["alias"] != expected_alias:
        raise ValueError("arbitration must use the declared dynamic receiver model")
    overrides = config["thresholds"]
    if not isinstance(overrides, dict) or any(not isinstance(k, str) or not k for k in overrides):
        raise ValueError("arbitration thresholds must map label names to explicit thresholds")
    for value in [config["default_threshold"], *overrides.values()]:
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("arbitration thresholds must be finite numbers in [0,1]")
    if labels is None:
        return ()
    if set(overrides) - set(labels):
        raise ValueError("arbitration threshold label outside the complete label axis")
    return tuple(
        rule
        for label in labels
        for rule in paired_rules(
            config["alias"], label, overrides.get(label, config["default_threshold"])
        )
    )


def unknown_mask(result):
    return np.asarray(
        getattr(result, "conflict_unknown", np.zeros_like(result.positive_facts)), bool
    )


def available_predictions(result):
    """Output labels may include arbitration and the undecided-cell fallback.

    Logical feedback, query selection and Gamma never read these additions.
    """
    return (result.positive_facts | getattr(result, "arbitration_positive", False)
            | getattr(result, "fallback_positive", False))


def pending_rows(result, gamma=None):
    # A complete trusted empty label set is an answer, not missing evidence.
    # Derived negatives and temporary model predictions do not certify this.
    needs_label = ~result.positive_facts.any(1)
    if gamma is not None:
        trusted_empty = gamma.neg.all(1) & ~gamma.pos.any(1)
        needs_label &= ~trusted_empty
    # Even a complete trusted answer must not hide an active conflict.
    return needs_label | unknown_mask(result).any(1)


def state_identity(result):
    from loris._engine.typed_session import complete_state_identity, array_identity, digest

    state = complete_state_identity(result)
    if not hasattr(result, "conflict_unknown"):
        return state
    state.pop("sha256")
    state["inference_semantics"] = "author_conflict_quarantine_with_explicit_arbitration"
    if hasattr(result, "history_nodes"):
        from .unknown_incremental import history_digest

        state["inference_semantics"] = "author_retained_unknown_with_explicit_arbitration"
        state["retained_history"] = history_digest(result)
    state["conflict_unknown"] = array_identity(result.conflict_unknown)
    for name in ("arbitration_positive", "arbitration_negative", "remaining_unknown", "fallback_positive"):
        if hasattr(result, name):
            state[name] = array_identity(getattr(result, name))
    if getattr(result, "seed_arbitration", None) is not None:
        state["temporary_seed_rule_evidence"] = digest(result.seed_arbitration)
    state["sha256"] = digest(state)
    return state
