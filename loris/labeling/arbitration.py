"""Explicit model rules for temporarily resolving conflict-unknown pairs.

These are author-added arbitration rules, not ordinary RDL ML predicates.
Each declared (model, label, threshold) has two explicit rules: decision=True
produces a positive label, and decision=False produces a negative label.
Only the caller's conflict_unknown mask enables their instances. The module
does not accept Gamma, run a model, train, or propagate a returned label.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import numbers

import numpy as np


SCHEMA = "loris_conflict_unknown_arbitration_rule_v1"


@dataclass(frozen=True)
class ArbitrationRule:
    alias: str
    label: str
    kappa: float
    decision: bool

    def __post_init__(self):
        for name in ("alias", "label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("arbitration " + name + " must be a nonempty string")
        if (
            isinstance(self.kappa, (bool, np.bool_))
            or not isinstance(self.kappa, numbers.Real)
            or not np.isfinite(self.kappa)
            or not 0 <= self.kappa <= 1
        ):
            raise ValueError("arbitration kappa must be explicit and finite in [0,1]")
        if type(self.decision) is not bool:
            raise ValueError("arbitration decision must be an explicit bool")
        object.__setattr__(self, "kappa", float(self.kappa))

    @property
    def polarity(self):
        return "pos" if self.decision else "neg"

    def to_dict(self):
        return {
            "schema": SCHEMA,
            "body": {
                "conflict_unknown": True,
                "model_decision": {
                    "alias": self.alias,
                    "label": self.label,
                    "kappa": self.kappa,
                    "value": self.decision,
                },
            },
            "head": {"label": self.label, "polarity": self.polarity},
        }

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema", "body", "head"}
            or value["schema"] != SCHEMA
        ):
            raise ValueError("explicit arbitration rule schema required")
        body, head = value["body"], value["head"]
        if (
            not isinstance(body, Mapping)
            or set(body) != {"conflict_unknown", "model_decision"}
            or body["conflict_unknown"] is not True
        ):
            raise ValueError("arbitration rule must explicitly guard conflict_unknown")
        decision = body["model_decision"]
        if not isinstance(decision, Mapping) or set(decision) != {
            "alias",
            "label",
            "kappa",
            "value",
        }:
            raise ValueError("arbitration needs explicit model, label, kappa and decision")
        rule = cls(decision["alias"], decision["label"], decision["kappa"], decision["value"])
        if (
            not isinstance(head, Mapping)
            or set(head) != {"label", "polarity"}
            or head["label"] != rule.label
            or head["polarity"] != rule.polarity
        ):
            raise ValueError("arbitration head must match the declared decision and label")
        return rule

    @property
    def rule_id(self):
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "arbitration:" + hashlib.sha256(raw.encode()).hexdigest()


def paired_rules(alias, label, kappa):
    """Declare both complementary rules; no threshold is inferred or defaulted."""
    return (
        ArbitrationRule(alias, label, kappa, True),
        ArbitrationRule(alias, label, kappa, False),
    )


class AmbiguousArbitrationError(ValueError):
    """All contradictory matched instances are retained for diagnosis."""

    def __init__(self, conflicts):
        self.conflicts = tuple(conflicts)
        super().__init__(
            "ambiguous model arbitration: opposite decisions for %d conflict-unknown pairs"
            % len(self.conflicts)
        )


@dataclass(frozen=True)
class ArbitrationResult:
    positive: np.ndarray
    negative: np.ndarray
    remaining_unknown: np.ndarray
    evidence: tuple
    audit: dict


def _readonly(array):
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _identity(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def arbitrate(
    rules,
    probability_tables,
    conflict_unknown,
    labels,
    *,
    model_state,
    model_version=None,
):
    """Evaluate explicit rules on one frozen pre/post model-table snapshot.

    ``model_state`` identifies the caller's pre/post snapshot; it does not
    request a model update. Tables use the complete ``labels`` column axis.
    The caller must bind aliases to its declared dynamic models and apply its
    Gamma policy before constructing ``conflict_unknown``. An unconfigured
    label stays unknown. Incomplete rule pairs, unavailable model tables and
    ambiguous matched decisions are errors, never implicit .5 predictions.

    Multiple same-sign sources are all recorded. Opposite signs abort the
    whole evaluation with diagnostics, so no partial result is installed.
    Returned masks are temporary consequences, not verified annotation roots.
    """
    if model_state not in ("pre", "post"):
        raise ValueError("model_state must identify the frozen pre or post snapshot")
    if model_version is not None and (
        not isinstance(model_version, str) or not model_version.strip()
    ):
        raise ValueError("model_version must be a nonempty string when supplied")
    labels = tuple(labels)
    if (
        not labels
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("arbitration requires a unique nonempty label axis")
    unknown = np.asarray(conflict_unknown)
    if unknown.dtype != np.bool_ or unknown.ndim != 2 or unknown.shape[1] != len(labels):
        raise ValueError("conflict_unknown must be a boolean document-by-label mask")
    unknown = _readonly(unknown)
    if not isinstance(probability_tables, Mapping):
        raise ValueError("arbitration requires explicitly keyed frozen model tables")
    parsed = tuple(
        rule if isinstance(rule, ArbitrationRule) else ArbitrationRule.from_dict(rule)
        for rule in rules
    )
    if len(set(parsed)) != len(parsed):
        raise ValueError("duplicate arbitration rule")
    parsed = tuple(
        sorted(parsed, key=lambda rule: (rule.alias, rule.label, rule.kappa, rule.decision))
    )
    groups = {}
    for rule in parsed:
        if rule.label not in labels:
            raise ValueError("arbitration rule label absent from the complete label axis")
        groups.setdefault((rule.alias, rule.label, rule.kappa), set()).add(rule.decision)
    if any(decisions != {False, True} for decisions in groups.values()):
        raise ValueError("arbitration requires an explicit complete True/False rule pair")

    tables, identities = {}, {}
    for alias in sorted({rule.alias for rule in parsed}):
        if alias not in probability_tables:
            raise ValueError("arbitration model table unavailable: " + alias)
        raw = np.asarray(probability_tables[alias])
        if raw.dtype.kind not in "iuf" or raw.shape != unknown.shape:
            raise ValueError("arbitration probability table axis/type differs: " + alias)
        table = np.array(raw, dtype=np.float64, copy=True)
        if not np.isfinite(table).all() or np.any(table < 0) or np.any(table > 1):
            raise ValueError("arbitration probabilities must be finite in [0,1]: " + alias)
        tables[alias] = _readonly(table)
        identities[alias] = _identity(tables[alias])

    positive, negative = np.zeros_like(unknown), np.zeros_like(unknown)
    evidence, conflicts = [], []
    by_label = {label: [] for label in labels}
    for rule in parsed:
        by_label[rule.label].append(rule)
    for row, column in np.argwhere(unknown):
        row, column = int(row), int(column)
        label = labels[column]
        sources = []
        for rule in by_label[label]:
            p = float(tables[rule.alias][row, column])
            # Explicit false decision is the strict complement, including at
            # kappa=0 or 1. It is not an RDL negative ML predicate.
            matches = p >= rule.kappa if rule.decision else p < rule.kappa
            if matches:
                sources.append(
                    dict(
                        origin="model_arbitration",
                        rule_id=rule.rule_id,
                        row=row,
                        column=column,
                        label=label,
                        polarity=rule.polarity,
                        alias=rule.alias,
                        kappa=rule.kappa,
                        decision=rule.decision,
                        probability=p,
                        model_state=model_state,
                        model_version=model_version,
                        table_sha256=identities[rule.alias]["sha256"],
                        guard="conflict_unknown",
                        temporary=True,
                    )
                )
        signs = {source["polarity"] for source in sources}
        if len(signs) > 1:
            conflicts.append(dict(row=row, column=column, label=label, sources=sources))
        elif signs:
            (positive if "pos" in signs else negative)[row, column] = True
        evidence.extend(sources)
    if conflicts:
        raise AmbiguousArbitrationError(conflicts)
    remaining = unknown & ~(positive | negative)
    return ArbitrationResult(
        positive=_readonly(positive),
        negative=_readonly(negative),
        remaining_unknown=_readonly(remaining),
        evidence=tuple(evidence),
        audit=dict(
            schema=SCHEMA,
            model_state=model_state,
            model_version=model_version,
            labels=list(labels),
            rules=[rule.to_dict() for rule in parsed],
            frozen_probability_tables=identities,
            input_conflict_unknown=_identity(unknown),
            input_conflict_unknown_pairs=int(unknown.sum()),
            positive_pairs=int(positive.sum()),
            negative_pairs=int(negative.sum()),
            remaining_unknown_pairs=int(remaining.sum()),
            fired_instances=len(evidence),
            positive_comparison="float64 p >= kappa",
            negative_comparison="float64 p < kappa",
            temporary=True,
            writes_Gamma=False,
            model_calls=0,
            training_calls=0,
        ),
    )
