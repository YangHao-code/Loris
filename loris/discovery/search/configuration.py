"""Explicit optional-context RDL space, authorized as an author supplement.

The public author supplement differs from the inherited connected space:
1. instantiate does not require Text/AttrEqual beside every ML atom;
2. startup does not insert such an ML context automatically.
3. cross-document single-label conclusions require a Label body premise.

The original complete feature/parameter/head axes, variable binding, model
semantics and all-used-variable connectivity remain. This is a different
grammar/prior namespace, not an equivalent replay of the old BO study.
"""

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import random

import numpy as np

from loris._engine.compact_storage import ParameterCodec
from loris.predicates import FreqPredicate
from loris._engine.rdl_reference import Doc, Label, LabelCompare, AttrEqual, Text, ML, Rule
from loris._engine.rlearner_config import (
    ConfigurationSpace,
    InvalidConfiguration,
    _canonical,
    _number,
)
from loris._search_space.configuration import (
    ConnectedConfigurationSpace,
    validate_propagation_guard,
)
from loris.rule_forms import (
    POLICY as RULE_FORM_POLICY,
    REJECTION,
    AuthorRuleFormError,
    validate_author_rule_form,
)


SCHEMA = "loris_author_cross_label_configuration_v2"


def _source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def policy_binding():
    """Bind both the supplement and the exact inherited implementations."""
    return {
        "schema": "paper36_author_optional_ML_context_v1",
        "authority": "author authorization 2026-09-16: pure/multiple ML and Label combinations allowed",
        "paper_distinction": "optional ML context is an explicit author supplement, not literal PDF36 context syntax",
        "ML_context": "Text and AttrEqual remain optional features, not mandatory companions of ML",
        "public_rule_form": dict(RULE_FORM_POLICY),
        "startup": "original sparse feature draw without automatic ML-context insertion; retain structural variable-connection repair",
        "unchanged": [
            "original feature/parameter/head axes",
            "explicit signed Label semantics",
            "Doc binding",
            "all substantively used variables connected by body relations",
            "dynamic/static ML semantics",
            "eq/subset/strict_subset meanings",
        ],
        "not_added": [
            "quality/support/F1 grammar filters",
            "model inference",
            "rule admission",
            "historical rules",
        ],
        "equivalent_to_original_BO_namespace": False,
        "sources": [
            _source(__file__),
            _source(inspect.getfile(ConfigurationSpace)),
            _source(inspect.getfile(ConnectedConfigurationSpace)),
            _source(inspect.getfile(Rule)),
            _source(inspect.getfile(validate_author_rule_form)),
        ],
    }


class AuthorFreeCombinationSpace(ConnectedConfigurationSpace):
    """Same executable RDL atoms and axes; ML context is optional."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        inherited = self.manifest()
        parent_space_id = inherited.pop("space_id")
        inherited["schema"] = SCHEMA
        inherited["semantics"][
            "model_context"
        ] = "optional Text/AttrEqual; pure ML and ML+Label permitted"
        inherited["author_combination_policy"] = policy_binding()
        inherited["author_combination_policy"]["parent_connected_space_id"] = parent_space_id
        definition = _canonical(inherited)
        object.__setattr__(self, "_definition", definition)
        object.__setattr__(self, "space_id", hashlib.sha256(definition).hexdigest())
        object.__setattr__(self, "_compact_codec", None)

    def instantiate(self, params):
        """Original explicit construction except the per-ML context check.

        Copied from ConfigurationSpace.instantiate so this ordinary factory
        does not compile AST, monkeypatch classes, or change runtime globals.
        Original all-variable connectivity is checked after construction.
        """
        values = self.include_values(params)
        if values is None:
            return self._instantiate_features(params, self.features)
        # Every include bit is an explicit int 0/1, so the full-axis loop can
        # raise only for ON features. Visiting just the ON features (in axis
        # order) with the OFF bits omitted yields the same body, the same
        # extra/missing sets and therefore the same Rule or the same error.
        axis = self._include_axis()
        on = np.flatnonzero(np.fromiter(values, np.int8, len(values))).tolist()
        features = tuple(self.features[i] for i in on)
        reduced = {key: params[key] for key in params.keys() - axis.key_set}
        reduced.update((feature.include_key, 1) for feature in features)
        return self._instantiate_features(reduced, features)

    def _instantiate_compact(self, value):
        """Internal certified-zero representation; all rule checks are shared.

        Codec validation binds the complete ordered feature axis and rejects
        foreign, duplicate, unordered or out-of-range ON indices, embedded
        include keys and nonfinite parameters. The ordinary factory then checks
        exactly the same active domains, head, variable and cross-rule forms.
        """
        # CapabilitySpace finishes restricting its axes after this parent's
        # constructor. Bind lazily to that final immutable domain, not to the
        # transient rich parent space.
        codec = self._compact_codec
        if codec is None or codec.space_id != self.space_id:
            codec = ParameterCodec.from_space(self)
            object.__setattr__(self, "_compact_codec", codec)
        codec.validate(value)
        features = tuple(self.features[i] for i in value["on_indices"])
        params = dict(value["active_parameters"])
        params.update((feature.include_key, 1) for feature in features)
        return self._instantiate_features(params, features)

    def _instantiate_features(self, params, features):
        if not isinstance(params, Mapping):
            raise TypeError("configuration must be a parameter mapping")
        expected, body, used = set(), [], set()
        head = self._head(params, expected)
        for feature in features:
            key = feature.include_key
            expected.add(key)
            if key not in params or type(params[key]) is not int or params[key] not in (0, 1):
                raise InvalidConfiguration(
                    "every include indicator must be explicit integer 0 or 1: " + key
                )
            if not params[key]:
                continue
            if feature.kind == "pattern":
                spec = self.patterns[feature.index]
                predicate = spec.predicate
                if isinstance(predicate, FreqPredicate):
                    op_key, eta_key = feature.key + ".op", feature.key + ".eta"
                    expected.update((op_key, eta_key))
                    if params.get(
                        op_key
                    ) not in spec.frequency_ops or not spec.frequency_domain.contains(
                        params.get(eta_key)
                    ):
                        raise InvalidConfiguration(
                            "frequency parameters outside the declared domain: " + feature.key
                        )
                    predicate = replace(predicate, op=params[op_key], eta=params[eta_key])
                body.append(Text(feature.var, predicate))
            elif feature.kind == "model":
                spec = self.models[feature.index]
                eta = 0.5
                if spec.dynamic:
                    eta_key = feature.key + ".eta"
                    expected.add(eta_key)
                    if not spec.threshold_domain.contains(params.get(eta_key)):
                        raise InvalidConfiguration(
                            "dynamic threshold outside the declared domain: " + feature.key
                        )
                    eta = params[eta_key]
                body.append(ML(feature.var, spec.alias, feature.label, float(eta), spec.dynamic))
            elif feature.kind == "label":
                body.append(Label(feature.var, feature.label, feature.polarity))
            else:
                body.append(self.relations[feature.index])
        extra, missing = set(params) - expected, expected - set(params)
        if extra or missing:
            raise InvalidConfiguration(
                "configuration fields mismatch: extra=%r missing=%r"
                % (sorted(extra), sorted(missing))
            )
        # The old per-ML Text/AttrEqual check is deliberately absent here.
        for atom in body + [head]:
            if isinstance(atom, (AttrEqual, LabelCompare)):
                used.update((atom.left_var, atom.right_var))
            else:
                used.add(atom.var)
        bindings = [Doc(v) for v in self.variables if v in used]
        rule = Rule(tuple(bindings + body), head)
        validate_propagation_guard(rule)
        try:
            validate_author_rule_form(rule)
        except AuthorRuleFormError as error:
            raise InvalidConfiguration(str(error)) from error
        return rule

    def iter_initial_configurations(
        self, count, *, seed, expected_body_size=3.0, feature_probabilities=None
    ):
        """Original connected sparse startup without inserting ML contexts.

        Connectivity repair may add an existing body relation only to join
        actually disconnected variables. It never adds Text or a relation
        merely because a same-document ML atom lacks context.
        """
        if type(count) is not int or count < 0 or type(seed) is not int:
            raise ValueError("count must be nonnegative and count/seed must be integers")
        if not _number(expected_body_size) or expected_body_size <= 0:
            raise ValueError("expected_body_size must be positive and finite")
        probability = min(1.0, float(expected_body_size) / len(self.features))
        if feature_probabilities is None:
            probabilities = {f.include_key: probability for f in self.features}
        else:
            if not isinstance(feature_probabilities, Mapping) or set(feature_probabilities) != {
                f.include_key for f in self.features
            }:
                raise ValueError(
                    "feature_probabilities must declare every include key exactly once"
                )
            probabilities = dict(feature_probabilities)
            if any(not _number(p) or not 0 < p < 1 for p in probabilities.values()):
                raise ValueError(
                    "explicit feature probabilities require 0 < p < 1; no excluded or forced features"
                )
        rng = random.Random(seed)
        for i in range(count):
            kind = self.head_kinds[i % len(self.head_kinds)]
            for attempt in range(1000):
                if kind in ("add", "remove"):
                    params = self.empty_parameters(
                        kind, label=rng.choice(self.labels), variable=rng.choice(self.variables)
                    )
                else:
                    params = self.empty_parameters(kind, pair=rng.choice(self.pairs))
                for feature in self.features:
                    if rng.random() < probabilities[feature.include_key]:
                        self._enable_random(params, feature, rng)
                # No automatic Text/AttrEqual insertion for an ML feature.
                self._connect_initial_variables(params, rng)
                try:
                    self.instantiate(params)
                except InvalidConfiguration as error:
                    if str(error).startswith(REJECTION):
                        # Keep the original proposal as an explicitly invalid
                        # startup trial; no hidden replacement or Label repair.
                        yield params
                        break
                    continue
                yield params
                break
            else:
                raise InvalidConfiguration(
                    "could not initialize an admissible connected configuration"
                )


def build_config_space(
    *, cluster_id, patterns, models, labels, variables=("x", "y"), relations=None
):
    """Drop-in constructor interface; caller still owns search and scoring."""
    return AuthorFreeCombinationSpace(
        cluster_id=cluster_id,
        patterns=patterns,
        models=models,
        labels=labels,
        variables=variables,
        relations=relations,
    )
