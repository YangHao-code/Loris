"""Direct, scoped RLearner configurations for PDF36 §3 and Appendix B.

This module neither reads rules/candidate banks nor evaluates accuracy. One
instance describes one cluster's current P_i and S_i; every head shares that
same space. The caller owns ONE BO study per cluster, evaluates each configured
rule alone at Figure 6 line 7, then evaluates ordered Sigma + rule at line 12.

Binary indicators have no body-size, dropout, support, gain or beam gate. The
optional sparse initial configurations are reproducible BO starting points,
not rules admitted into Sigma. Parameter domains and the finite variable
inventory are explicit experiment choices, not claims of exhaustive syntax.

The reference grammar retains x/y scopes, signed facts and label comparisons.
Attribute equality is supported by §3 Example 2(6), although absent from the
displayed grammar. Sim, Group, quorum, top-k and semantic text matching are
not silently translated into paper predicates. Static ML/text heads remain
unsupported because §6.1 does not specify their operational consequence.
"""
from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
import operator
import random
import re
from dataclasses import dataclass, replace
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import numpy as np

from loris.predicates import (MatchPredicate, FreqPredicate, CooccurPredicate,
                              BeforePredicate, predicate_to_dict)
from loris._engine.rdl_reference import (Doc, Label, Text, ML, AttrEqual,
                                         LabelCompare, Rule, is_attribute_name)


SCHEMA = 'paper36_scoped_rlearner_configuration_v1'
HEAD_KINDS = ('add', 'remove', 'eq', 'subset', 'strict_subset')
FREQUENCY_OPS = ('==', '!=', '<', '<=', '>', '>=')
PAPER_ATTRIBUTES = ('mtd', 'ttl', 'cnt')
TEXT_TYPES = (MatchPredicate, FreqPredicate, CooccurPredicate, BeforePredicate)


class UnsupportedPaperSyntax(ValueError):
    """An input cannot be represented without extending or losing PDF syntax."""


class InvalidConfiguration(ValueError):
    """Record this BO trial as invalid; do not replace it by another body."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class ParameterDomain:
    """An explicit finite set or closed int/float interval; no hidden bins."""
    values: Tuple[float, ...] = ()
    low: Optional[float] = None
    high: Optional[float] = None
    integer: bool = False

    def __post_init__(self):
        if type(self.integer) is not bool:
            raise ValueError('integer must be an explicit bool')
        object.__setattr__(self, 'values', tuple(self.values))
        if self.values:
            if self.low is not None or self.high is not None:
                raise ValueError('choose values OR an interval')
            if any(not _number(v) for v in self.values) or len(set(self.values)) != len(self.values):
                raise ValueError('domain values must be unique finite numbers')
            if self.integer and any(int(v) != v for v in self.values):
                raise ValueError('integer domain contains a noninteger')
        else:
            if not _number(self.low) or not _number(self.high) or self.low > self.high:
                raise ValueError('interval requires finite low <= high')
            if self.integer and (int(self.low) != self.low or int(self.high) != self.high):
                raise ValueError('integer interval requires integer bounds')

    def to_dict(self):
        return {'values': list(self.values), 'low': self.low,
                'high': self.high, 'integer': self.integer}

    def contains(self, value):
        if not _number(value) or (self.integer and int(value) != value):
            return False
        return value in self.values if self.values else self.low <= value <= self.high

    def suggest(self, trial, name):
        if self.values:
            return trial.suggest_categorical(name, list(self.values))
        if self.integer:
            return trial.suggest_int(name, int(self.low), int(self.high))
        return trial.suggest_float(name, float(self.low), float(self.high))

    def sample(self, rng):
        if self.values:
            return rng.choice(self.values)
        if self.integer:
            return rng.randint(int(self.low), int(self.high))
        return rng.uniform(self.low, self.high)


@dataclass(frozen=True)
class PatternSpec:
    key: str
    predicate: object
    frequency_domain: Optional[ParameterDomain] = None
    frequency_ops: Tuple[str, ...] = FREQUENCY_OPS

    def __post_init__(self):
        object.__setattr__(self, 'frequency_ops', tuple(self.frequency_ops))
        if not isinstance(self.key, str) or not self.key:
            raise ValueError('pattern key must be nonempty')
        if type(self.predicate) not in TEXT_TYPES:
            raise UnsupportedPaperSyntax('P_i accepts only exact match/freq/cooccur/before predicates')
        if self.predicate.attr not in PAPER_ATTRIBUTES or self.predicate.sim:
            raise UnsupportedPaperSyntax('paper text requires mtd/ttl/cnt and lexical/regex semantics')
        for name in ('r', 'r1', 'r2'):
            pattern = getattr(self.predicate, name, None)
            if pattern is not None:
                re.compile(pattern.raw, pattern.flags)
        if isinstance(self.predicate, FreqPredicate):
            if not isinstance(self.frequency_domain, ParameterDomain):
                raise ValueError('freq requires an explicit frequency_domain; no inherited threshold bins')
            if (not self.frequency_ops or len(set(self.frequency_ops)) != len(self.frequency_ops)
                    or not set(self.frequency_ops).issubset(FREQUENCY_OPS)):
                raise UnsupportedPaperSyntax('freq operators must be a nonempty explicit subset of all six paper operators')
        elif self.frequency_domain is not None:
            raise ValueError('frequency_domain would be ignored on a non-frequency pattern')
        elif self.frequency_ops != FREQUENCY_OPS:
            raise ValueError('frequency_ops would be ignored on a non-frequency pattern')

    def to_dict(self):
        return {'key': self.key, 'predicate': predicate_to_dict(self.predicate),
                'frequency_domain': self.frequency_domain.to_dict() if self.frequency_domain else None,
                'frequency_ops': list(self.frequency_ops)}


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    dynamic: bool
    threshold_domain: Optional[ParameterDomain] = None

    def __post_init__(self):
        if not isinstance(self.alias, str) or not self.alias or type(self.dynamic) is not bool:
            raise ValueError('model requires a nonempty alias and explicit dynamic bool')
        if self.dynamic:
            if not isinstance(self.threshold_domain, ParameterDomain):
                raise ValueError('dynamic model requires an explicit calibrated-threshold search domain')
            limits = self.threshold_domain.values or (self.threshold_domain.low, self.threshold_domain.high)
            if min(limits) < 0 or max(limits) > 1:
                raise ValueError('dynamic probability threshold domain must lie in [0, 1]')
        elif self.threshold_domain is not None:
            raise ValueError('static M has fixed decisions; threshold_domain would be silently ignored')

    def to_dict(self):
        return {'alias': self.alias, 'dynamic': self.dynamic,
                'threshold_domain': self.threshold_domain.to_dict() if self.threshold_domain else None,
                'static_semantics': 'separate frozen binary decision table; no optimized eta'}


def paper_relations(variables=('x', 'y'), attributes=PAPER_ATTRIBUTES):
    """Finite primitive inventory, not discovered/selected rules or a bank.

    Equality of document attributes follows the paper example. Label-set
    comparisons are explicit validated relation atoms, not comparisons of
    currently known positive bitsets and not embedding similarity.
    """
    variables, attributes = tuple(variables), tuple(attributes)
    if any(not is_attribute_name(a) for a in attributes):
        raise UnsupportedPaperSyntax('requires paper attributes or explicitly declared mtd.<key> views')
    out = [AttrEqual(x, a, y, a) for x, y in itertools.combinations(variables, 2)
           for a in attributes]
    out.extend(LabelCompare(x, op, y) for x, y in itertools.permutations(variables, 2)
               for op in ('eq', 'subset', 'strict_subset'))
    return tuple(out)


_PARAMS_SENTINEL = '\x01loris exact dense params placeholder\x01'


class _IncludeAxis:
    """Precomputed canonical-JSON template of the all-zero include indicators."""
    def __init__(self, features):
        self.features = features
        self.keys = tuple(f.include_key for f in features)
        self.key_set = frozenset(self.keys)
        self.getter = operator.itemgetter(*self.keys) if self.keys else None
        order = sorted(range(len(self.keys)), key=self.keys.__getitem__)
        self.sorted_keys = [self.keys[i] for i in order]
        self.sorted_position = np.empty(len(self.keys), np.int64)
        self.sorted_position[order] = np.arange(len(self.keys))
        pieces, starts, offsets, cursor = [b'{'], [], [], 1
        for i, key in enumerate(self.sorted_keys):
            entry = _canonical(key) + b':0'
            if i:
                pieces.append(b',')
                cursor += 1
            starts.append(cursor)
            offsets.append(cursor + len(entry) - 1)
            pieces.append(entry)
            cursor += len(entry)
        pieces.append(b'}')
        self.template = b''.join(pieces)
        starts.append(len(self.template))
        self.starts = starts
        self.value_offsets = np.asarray(offsets, np.int64)


@dataclass(frozen=True)
class Feature:
    key: str
    kind: str
    var: Optional[str] = None
    label: Optional[str] = None
    polarity: Optional[str] = None
    index: Optional[int] = None

    def __post_init__(self):
        # The fixed axis is visited by every full-dimensional configuration.
        # Reuse the same immutable parameter name across those mappings instead
        # of allocating another identical string on each property access.
        object.__setattr__(self, '_include_key', self.key + '.include')

    @property
    def include_key(self):
        try:
            return self._include_key
        except AttributeError:
            # A Feature restored from a pre-cache pickle has only the original
            # six fields. Lazily fill the same derived value in that case.
            value = self.key + '.include'
            object.__setattr__(self, '_include_key', value)
            return value

    def to_dict(self):
        # _include_key is an implementation cache, not a manifest field. Keep
        # the original six fields, values and insertion order exactly intact.
        return {'key': self.key, 'kind': self.kind, 'var': self.var,
                'label': self.label, 'polarity': self.polarity,
                'index': self.index}


class ConfigurationSpace:
    """One cluster, one unified study; exact parameters -> one scoped Rule."""
    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('configuration space is frozen; construct a new space for changed inputs')
        object.__setattr__(self, name, value)

    def __init__(self, *, cluster_id, patterns, models, labels,
                 variables=('x', 'y'), relations=None):
        if not isinstance(cluster_id, (str, int)) or isinstance(cluster_id, bool):
            raise ValueError('cluster_id must be a string or integer')
        self.cluster_id = cluster_id
        self.patterns, self.models = tuple(patterns), tuple(models)
        self.labels, self.variables = tuple(labels), tuple(variables)
        if (not self.labels or len(set(self.labels)) != len(self.labels)
                or any(not isinstance(x, str) or not x for x in self.labels)):
            raise ValueError('requires a unique nonempty label axis')
        if (not self.variables or len(set(self.variables)) != len(self.variables)
                or any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', v)
                       for v in self.variables)):
            raise ValueError('declare unique identifier-like document variables explicitly')
        if any(type(p) is not PatternSpec for p in self.patterns):
            raise UnsupportedPaperSyntax('patterns must be current-round PatternSpec objects, not rules or bank rows')
        if len({p.key for p in self.patterns}) != len(self.patterns):
            raise ValueError('duplicate current pattern key')
        if any(type(m) is not ModelSpec for m in self.models):
            raise UnsupportedPaperSyntax('models must be explicit current-round ModelSpec objects')
        if len({m.alias for m in self.models}) != len(self.models):
            raise ValueError('duplicate selected model alias')
        self.relations = paper_relations(self.variables) if relations is None else tuple(relations)
        for atom in self.relations:
            if type(atom) not in (AttrEqual, LabelCompare):
                raise UnsupportedPaperSyntax('relation must be AttrEqual or LabelCompare; Sim/Group/quorum are unsupported')
            if atom.left_var not in self.variables or atom.right_var not in self.variables:
                raise ValueError('relation references an undeclared variable')
            if isinstance(atom, AttrEqual):
                if not is_attribute_name(atom.left_attr) or not is_attribute_name(atom.right_attr):
                    raise UnsupportedPaperSyntax('attribute equality requires paper attributes or declared mtd.<key> views')
            elif atom.op not in ('eq', 'subset', 'strict_subset'):
                raise UnsupportedPaperSyntax('unsupported label comparison')
        if len({_sha(a.to_dict()) for a in self.relations}) != len(self.relations):
            raise ValueError('duplicate relation primitive')
        features = []
        for i, _ in enumerate(self.patterns):
            features.extend(Feature('p%d.%s' % (i, v), 'pattern', v, index=i) for v in self.variables)
        # Independent labels permit multiple predicates of the same model;
        # there is no legacy maximum of two or three selected ML predicates.
        for i, _ in enumerate(self.models):
            for v in self.variables:
                features.extend(Feature('m%d.%s.k%d' % (i, v, k), 'model', v, label, index=i)
                                for k, label in enumerate(self.labels))
        for v in self.variables:
            for k, label in enumerate(self.labels):
                features.extend(Feature('l.%s.k%d.%s' % (v, k, polarity), 'label', v, label, polarity)
                                for polarity in ('pos', 'neg'))
        features.extend(Feature('r%d' % i, 'relation', index=i) for i, _ in enumerate(self.relations))
        self.features = tuple(features)
        self.feature_by_key = MappingProxyType({f.key: f for f in self.features})
        self.head_kinds = HEAD_KINDS if len(self.variables) > 1 else HEAD_KINDS[:2]
        self.pairs = tuple('%s|%s' % pair for pair in itertools.permutations(self.variables, 2))
        definition = {
            'schema': SCHEMA, 'cluster_id': cluster_id, 'variables': list(self.variables),
            'labels': list(self.labels), 'patterns': [p.to_dict() for p in self.patterns],
            'models': [m.to_dict() for m in self.models],
            'relations': [a.to_dict() for a in self.relations],
            'features': [f.to_dict() for f in self.features], 'head_kinds': list(self.head_kinds),
            'semantics': {'negative_premises': 'explicit negative facts; unknown is not negative',
                'shared_variables': 'one valuation shared by the complete body and head',
                'attribute_equality_authority': 'PDF36 section 3 Example 2(6)',
                'extracted_dictionary_key_authority': 'author explicitly authorized declared mtd.key extraction views',
                'extracted_dictionary_key_equality': 'exact complete string value; runtime must bind named values and presence',
                'subset_orientation': 'preserved as written; execution mode belongs to reference session',
                'model_context': 'text on same variable or incident attribute equality',
                'frequency_operators': list(FREQUENCY_OPS),
                'unsupported': ['Sim', 'Group', 'quorum', 'semantic_text_matching', 'TopK',
                                'static_ML_head', 'text_head', 'dynamic_ML_head']},
            'search': {'studies_per_cluster': 1, 'heads_share_study': True,
                'history_inputs': False, 'dropout': False, 'beam': False,
                'body_size_cap': None, 'support_gate': None, 'score_gate': None,
                'line7': 'configured rule alone; scorer is external',
                'line12': 'whole ordered Sigma plus rule; scorer is external'}}
        # Canonical bytes are the immutable receipt. Retaining the enormous
        # parallel tree of per-pattern/per-feature dictionaries serves no
        # operational purpose once the same bytes have been serialized.
        self._definition = _canonical(definition)
        self.space_id = hashlib.sha256(self._definition).hexdigest()
        self._sealed = True

    def manifest(self):
        # Return detached JSON values; callers cannot mutate the stored receipt.
        manifest = json.loads(self._definition)
        manifest['space_id'] = self.space_id
        # The previous canonical JSON round trip sorted all mapping keys,
        # including the new top-level key. Nested keys are already sorted in
        # _definition; sorting only this small mapping preserves that order.
        return {key: manifest[key] for key in sorted(manifest)}

    def _head(self, params, expected):
        kind = params.get('head.kind')
        expected.add('head.kind')
        if kind not in self.head_kinds:
            raise UnsupportedPaperSyntax('head.kind must be one of %r, got %r' % (self.head_kinds, kind))
        if kind in ('add', 'remove'):
            expected.update(('head.var', 'head.label'))
            if params.get('head.var') not in self.variables or params.get('head.label') not in self.labels:
                raise InvalidConfiguration('head variable or label outside the fixed axes')
            return Label(params['head.var'], params['head.label'], 'pos' if kind == 'add' else 'neg')
        expected.add('head.pair')
        if params.get('head.pair') not in self.pairs:
            raise InvalidConfiguration('comparison head requires a declared ordered variable pair')
        left, right = params['head.pair'].split('|')
        return LabelCompare(left, kind, right)

    def instantiate(self, params):
        """Pure construction, without data, support proxies or accuracy gates.

        All include bits are mandatory. Only selected predicates have active
        parameters; unknown or inactive parameters raise instead of being lost.
        Invalid unguarded model configurations must remain visible BO trials.
        """
        if not isinstance(params, Mapping):
            raise TypeError('configuration must be a parameter mapping')
        expected, body, used = set(), [], set()
        head = self._head(params, expected)
        for feature in self.features:
            key = feature.include_key
            expected.add(key)
            if key not in params or type(params[key]) is not int or params[key] not in (0, 1):
                raise InvalidConfiguration('every include indicator must be explicit integer 0 or 1: ' + key)
            if not params[key]:
                continue
            if feature.kind == 'pattern':
                spec = self.patterns[feature.index]
                predicate = spec.predicate
                if isinstance(predicate, FreqPredicate):
                    op_key, eta_key = feature.key + '.op', feature.key + '.eta'
                    expected.update((op_key, eta_key))
                    if params.get(op_key) not in spec.frequency_ops or not spec.frequency_domain.contains(params.get(eta_key)):
                        raise InvalidConfiguration('frequency parameters outside the declared domain: ' + feature.key)
                    predicate = replace(predicate, op=params[op_key], eta=params[eta_key])
                body.append(Text(feature.var, predicate))
            elif feature.kind == 'model':
                spec = self.models[feature.index]
                # The reference uses a distinct binary decision table for M;
                # its common ML record's eta field is an unused fixed sentinel.
                eta = 0.5
                if spec.dynamic:
                    eta_key = feature.key + '.eta'
                    expected.add(eta_key)
                    if not spec.threshold_domain.contains(params.get(eta_key)):
                        raise InvalidConfiguration('dynamic threshold outside the declared domain: ' + feature.key)
                    eta = params[eta_key]
                body.append(ML(feature.var, spec.alias, feature.label, float(eta), spec.dynamic))
            elif feature.kind == 'label':
                body.append(Label(feature.var, feature.label, feature.polarity))
            else:
                body.append(self.relations[feature.index])
        extra, missing = set(params) - expected, expected - set(params)
        if extra or missing:
            raise InvalidConfiguration('configuration fields mismatch: extra=%r missing=%r' % (sorted(extra), sorted(missing)))
        for atom in body:
            if isinstance(atom, ML):
                guarded = any(isinstance(p, Text) and p.var == atom.var for p in body)
                guarded |= any(isinstance(p, AttrEqual) and atom.var in (p.left_var, p.right_var) for p in body)
                if not guarded:
                    raise InvalidConfiguration('ML predicate lacks a textual/attribute context on ' + atom.var)
        for atom in body + [head]:
            if isinstance(atom, (AttrEqual, LabelCompare)):
                used.update((atom.left_var, atom.right_var))
            else:
                used.add(atom.var)
        bindings = [Doc(v) for v in self.variables if v in used]
        return Rule(tuple(bindings + body), head)

    def suggest(self, trial):
        """Ask all indicators, then active predicate parameters, from ONE trial.

        Returns raw parameters even if semantically inadmissible; the caller
        logs InvalidConfiguration when instantiate rejects an unguarded ML.
        No resampling/replacement hides the BO configuration's true outcome.
        """
        kind = trial.suggest_categorical('head.kind', list(self.head_kinds))
        params = {'head.kind': kind}
        if kind in ('add', 'remove'):
            params['head.var'] = trial.suggest_categorical('head.var', list(self.variables))
            params['head.label'] = trial.suggest_categorical('head.label', list(self.labels))
        else:
            params['head.pair'] = trial.suggest_categorical('head.pair', list(self.pairs))
        axis = self._include_axis()
        bulk = getattr(trial, 'include_bits', None)
        values = bulk(axis.keys) if bulk is not None and axis.keys else None
        if values is None:
            for feature in self.features:
                params[feature.include_key] = trial.suggest_categorical(feature.include_key, [0, 1])
                if params[feature.include_key]:
                    self._suggest_active(trial, feature, params)
            return params
        # Exact bulk path: include bits never reach Optuna (CompactTrial reads
        # the same joint proposal/fixed vector); conditional parameters are
        # suggested in the identical feature order, so Optuna/RNG calls match.
        on = np.flatnonzero(np.fromiter(values, np.int8, len(values)))
        adopt = trial.adopt_include_bits
        start = 0
        for i in on.tolist():
            if i > start:
                keys, bits_ = axis.keys[start:i], values[start:i]
                params.update(zip(keys, bits_))
                adopt(keys, bits_)
            feature = self.features[i]
            params[feature.include_key] = trial.suggest_categorical(feature.include_key, [0, 1])
            if params[feature.include_key]:
                self._suggest_active(trial, feature, params)
            start = i + 1
        if start < len(axis.keys):
            keys, bits_ = axis.keys[start:], values[start:]
            params.update(zip(keys, bits_))
            adopt(keys, bits_)
        return params

    def _suggest_active(self, trial, feature, params):
        if feature.kind == 'pattern' and isinstance(self.patterns[feature.index].predicate, FreqPredicate):
            spec = self.patterns[feature.index]
            params[feature.key + '.op'] = trial.suggest_categorical(feature.key + '.op', list(spec.frequency_ops))
            params[feature.key + '.eta'] = spec.frequency_domain.suggest(trial, feature.key + '.eta')
        elif feature.kind == 'model' and self.models[feature.index].dynamic:
            params[feature.key + '.eta'] = self.models[feature.index].threshold_domain.suggest(trial, feature.key + '.eta')

    def _include_axis(self):
        """Cached exact views of the (possibly subclass-restricted) feature axis."""
        cached = self.__dict__.get('_include_axis_cache')
        if cached is None or cached.features is not self.features:
            cached = _IncludeAxis(self.features)
            object.__setattr__(self, '_include_axis_cache', cached)
        return cached

    def include_values(self, params):
        """Include bits in feature order if every one is an explicit int 0/1, else None."""
        axis = self._include_axis()
        if type(params) is not dict or not axis.keys:
            return None
        try:
            values = axis.getter(params)
        except KeyError:
            return None
        if len(axis.keys) == 1:
            values = (values,)
        if set(map(type, values)) != {int} or not set(values) <= {0, 1}:
            return None
        return values

    def params_canonical(self, params):
        """Exact _canonical(dict(params)) bytes, or None to use the plain path."""
        values = self.include_values(params)
        if values is None:
            return None
        axis = self._include_axis()
        active = params.keys() - axis.key_set
        buffer = bytearray(axis.template)
        on = np.flatnonzero(np.fromiter(values, np.int8, len(values)))
        if len(on):
            view = np.frombuffer(buffer, np.uint8)
            view[axis.value_offsets[axis.sorted_position[on]]] = ord('1')
        entries = sorted((bisect.bisect_left(axis.sorted_keys, key), key) for key in active)
        parts, current, n = [], 0, len(axis.sorted_keys)
        for position, key in entries:
            if position < n and axis.sorted_keys[position] == key:
                return None
            if position > current:
                parts.append(bytes(buffer[axis.starts[current]:axis.starts[position] - 1]))
                current = position
            parts.append(_canonical(key) + b':' + _canonical(params[key]))
        if current < n:
            parts.append(bytes(buffer[axis.starts[current]:axis.starts[n] - 1]))
        return b'{' + b','.join(parts) + b'}'

    def empty_parameters(self, head_kind='add', *, label=None, variable=None, pair=None):
        """All features off; convenient for explicit fixtures and BO starts."""
        if head_kind not in self.head_kinds:
            raise UnsupportedPaperSyntax('unsupported or unavailable head kind: ' + str(head_kind))
        params = {feature.include_key: 0 for feature in self.features}
        params['head.kind'] = head_kind
        if head_kind in ('add', 'remove'):
            params.update({'head.label': self.labels[0] if label is None else label,
                           'head.var': self.variables[0] if variable is None else variable})
            if pair is not None:
                raise InvalidConfiguration('pair is not a parameter of a label head')
        else:
            if label is not None or variable is not None:
                raise InvalidConfiguration('label/variable are not parameters of a comparison head')
            params['head.pair'] = self.pairs[0] if pair is None else pair
        return params

    def _enable_random(self, params, feature, rng):
        params[feature.include_key] = 1
        if feature.kind == 'pattern' and isinstance(self.patterns[feature.index].predicate, FreqPredicate):
            spec = self.patterns[feature.index]
            params[feature.key + '.op'] = rng.choice(spec.frequency_ops)
            params[feature.key + '.eta'] = spec.frequency_domain.sample(rng)
        elif feature.kind == 'model' and self.models[feature.index].dynamic:
            params[feature.key + '.eta'] = self.models[feature.index].threshold_domain.sample(rng)

    def initial_configurations(self, count, *, seed, expected_body_size=3.0,
                               feature_probabilities=None):
        return list(self.iter_initial_configurations(count, seed=seed,
            expected_body_size=expected_body_size, feature_probabilities=feature_probabilities))

    def iter_initial_configurations(self, count, *, seed, expected_body_size=3.0,
                                    feature_probabilities=None):
        """Seeded sparse prior, no maximum length, no historic rules or scores.

        Every feature has positive selection probability. Heads are stratified
        across the same study. If a selected ML lacks context, an available
        current-pattern/attribute guard is added, without removing any feature.
        These configurations must consume the normal BO evaluation budget.
        """
        if type(count) is not int or count < 0 or type(seed) is not int:
            raise ValueError('count must be nonnegative and count/seed must be integers')
        if not _number(expected_body_size) or expected_body_size <= 0:
            raise ValueError('expected_body_size must be positive and finite')
        probability = min(1.0, float(expected_body_size) / len(self.features))
        if feature_probabilities is None:
            probabilities = {f.include_key: probability for f in self.features}
        else:
            if (not isinstance(feature_probabilities, Mapping)
                    or set(feature_probabilities) != {f.include_key for f in self.features}):
                raise ValueError('feature_probabilities must declare every include key exactly once')
            probabilities = dict(feature_probabilities)
            if any(not _number(p) or not 0 < p < 1 for p in probabilities.values()):
                raise ValueError('explicit feature probabilities require 0 < p < 1; no excluded or forced features')
        rng = random.Random(seed)
        for i in range(count):
            kind = self.head_kinds[i % len(self.head_kinds)]
            for attempt in range(1000):
                if kind in ('add', 'remove'):
                    params = self.empty_parameters(kind, label=rng.choice(self.labels), variable=rng.choice(self.variables))
                else:
                    params = self.empty_parameters(kind, pair=rng.choice(self.pairs))
                for feature in self.features:
                    if rng.random() < probabilities[feature.include_key]:
                        self._enable_random(params, feature, rng)
                for feature in self.features:
                    if feature.kind != 'model' or not params[feature.include_key]:
                        continue
                    guards = [g for g in self.features if
                              (g.kind == 'pattern' and g.var == feature.var) or
                              (g.kind == 'relation' and isinstance(self.relations[g.index], AttrEqual)
                               and feature.var in (self.relations[g.index].left_var, self.relations[g.index].right_var))]
                    if guards and not any(params[g.include_key] for g in guards):
                        self._enable_random(params, rng.choice(guards), rng)
                try:
                    self.instantiate(params)
                except InvalidConfiguration:
                    continue
                yield params
                break
            else:
                raise InvalidConfiguration('could not initialize an admissible configuration; supply model contexts')

    def configuration_receipt(self, params):
        record = {'schema': SCHEMA, 'space_id': self.space_id, 'cluster_id': self.cluster_id,
                  'params': dict(params)}
        try:
            rule = self.instantiate(params)
        except (InvalidConfiguration, UnsupportedPaperSyntax) as error:
            record.update({'admissible': False, 'rule': None,
                           'invalid_reason': str(error), 'error_type': type(error).__name__})
        else:
            record.update({'admissible': True, 'rule': rule.to_dict(),
                           'invalid_reason': None, 'error_type': None})
        record['configuration_id'] = self._receipt_sha(record, params)
        record['rule_semantic_id'] = _sha(record['rule']) if record['rule'] else None
        return record

    def _receipt_sha(self, record, params):
        encoded = self.params_canonical(params)
        if encoded is None:
            return _sha(record)
        shell = dict(record, params=_PARAMS_SENTINEL)
        raw = _canonical(shell)
        marker = _canonical(_PARAMS_SENTINEL)
        if raw.count(marker) != 1:
            return _sha(record)
        return hashlib.sha256(raw.replace(marker, encoded)).hexdigest()


def build_config_space(*, cluster_id, patterns, models, labels,
                       variables=('x', 'y'), relations=None):
    """No bank/rule/history/beam/dropout arguments exist in this public API."""
    return ConfigurationSpace(cluster_id=cluster_id, patterns=patterns, models=models,
                              labels=labels, variables=variables, relations=relations)
