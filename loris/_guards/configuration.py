"""Author-clarified cross-variable propagation rules and complete search space.

PDF36 p3 requires Doc binding of every variable. Example2(6), p4, relates x/y
with x.mtd=y.mtd before deriving label equality. The additional mandatory body
connection below records the author's 2026-09-10 clarification. It is a grammar
change and a new startup prior, not an equivalent execution optimization.
"""
from collections.abc import Mapping
import copy
import hashlib
from pathlib import Path
import random

from loris._engine.rdl_reference import AttrEqual, LabelCompare, Rule
from loris._engine.rlearner_config import (ConfigurationSpace, InvalidConfiguration,
                                         _canonical, _number)


GUARD_POLICY = {
    'schema': 'paper36_author_cross_variable_propagation_guard_v1',
    'authority': 'author clarification 2026-09-10; PDF36 section3 Example2(6)',
    'pdf_grammar_distinction': 'Doc binds variables; body connectivity is the additional author clarification',
    'binary_head': 'LabelCompare with distinct endpoint variables',
    'required_body_connection': 'head endpoints connected by non-self AttrEqual or LabelCompare body atoms',
    'graph_direction': 'undirected structural dependency; atom semantics retain their original direction',
    'non_connecting_predicates': ['Doc', 'Text', 'ML', 'Label'],
    'single_variable_heads': 'original semantics unchanged',
    'startup_prior': {
        'schema': 'paper36_author_guarded_sparse_startup_v1',
        'head_and_feature_axes': 'all original axes and positive feature probabilities retained',
        'repair': 'if binary endpoints remain disconnected, enable one uniformly chosen existing direct cross-pair relation feature',
        'selected_features_removed': False,
        'later_TPE_proposals_repaired': False,
        'missing_direct_guard_inventory': 'explicit initialization error; never resample heads or drop trials',
        'historical_rule_banks': False,
        'budget': 'initial configurations and all later invalid configurations consume the original study budget',
        'equivalent_prior_to_previous_study': False,
    },
}


def guard_policy_binding():
    return {'policy': copy.deepcopy(GUARD_POLICY),
            'configuration_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


policy_binding = guard_policy_binding


def _connected(head, atoms):
    if not isinstance(head, LabelCompare) or head.left_var == head.right_var:
        return True
    graph = {}
    for atom in atoms:
        if not isinstance(atom, (AttrEqual, LabelCompare)) or atom.left_var == atom.right_var:
            continue
        graph.setdefault(atom.left_var, set()).add(atom.right_var)
        graph.setdefault(atom.right_var, set()).add(atom.left_var)
    seen, pending = {head.left_var}, [head.left_var]
    while pending:
        for other in graph.get(pending.pop(), ()):
            if other == head.right_var: return True
            if other not in seen: seen.add(other); pending.append(other)
    return False


def validate_propagation_guard(rule):
    """Validate body connectivity without changing a rule or its logical atoms."""
    if not isinstance(rule, Rule): raise TypeError('an explicit typed Rule is required')
    if not _connected(rule.head, rule.body):
        raise InvalidConfiguration(
            'binary propagation head requires a cross-variable body guard connecting %s and %s; '
            'Doc bindings and independent Text/ML/Label predicates do not connect documents'
            % (rule.head.left_var, rule.head.right_var))


validate_rule_guard = validate_propagation_guard


class GuardedConfigurationSpace(ConfigurationSpace):
    """Original complete axes with an author-bound grammar and startup prior.

    The old definition is already canonical JSON bytes. Since author_constraints
    sorts before every existing top-level key, prefixing that one canonical
    field preserves exactly the canonical serialization of the full manifest,
    without reparsing its potentially gigabyte-sized dictionary tree.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        binding = guard_policy_binding()
        original = self._definition
        if not original.startswith(b'{') or not original.endswith(b'}'):
            raise RuntimeError('base configuration definition is not a canonical JSON object')
        prefix = b'{"author_constraints":' + _canonical(binding) + (b',' if len(original) > 2 else b'')
        definition = prefix + original[1:]
        object.__setattr__(self, '_definition', definition)
        object.__setattr__(self, 'space_id', hashlib.sha256(definition).hexdigest())
        object.__setattr__(self, '_author_constraints', binding)

    def instantiate(self, params):
        rule = super().instantiate(params)
        validate_propagation_guard(rule)
        return rule

    def _connect_initial_head(self, params, rng):
        if params['head.kind'] in ('add', 'remove'): return
        left, right = params['head.pair'].split('|')
        head = LabelCompare(left, params['head.kind'], right)
        selected = [self.relations[f.index] for f in self.features
                    if f.kind == 'relation' and params[f.include_key]]
        if _connected(head, selected): return
        candidates = [f for f in self.features if f.kind == 'relation'
                      and self.relations[f.index].left_var != self.relations[f.index].right_var
                      and {self.relations[f.index].left_var, self.relations[f.index].right_var} == {left, right}]
        if not candidates:
            raise InvalidConfiguration(
                'cannot initialize binary head %s: no declared direct cross-variable relation primitive; '
                'supply an explicit relation inventory rather than resampling the head' % params['head.pair'])
        # Uniform choice gives every available direct relation positive mass;
        # every originally selected feature remains on. No old rule is loaded.
        self._enable_random(params, rng.choice(candidates), rng)

    def iter_initial_configurations(self, count, *, seed, expected_body_size=3.0,
                                    feature_probabilities=None):
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
            # Preserve the original model-context initialization behavior. The
            # new binary guard is made legal in the first proposal, before the
            # original admissibility check, so it never causes 1000 retries.
            for attempt in range(1000):
                if kind in ('add', 'remove'):
                    params = self.empty_parameters(kind, label=rng.choice(self.labels), variable=rng.choice(self.variables))
                else:
                    params = self.empty_parameters(kind, pair=rng.choice(self.pairs))
                for feature in self.features:
                    if rng.random() < probabilities[feature.include_key]:
                        self._enable_random(params, feature, rng)
                for feature in self.features:
                    if feature.kind != 'model' or not params[feature.include_key]: continue
                    guards = [g for g in self.features if
                        (g.kind == 'pattern' and g.var == feature.var) or
                        (g.kind == 'relation' and isinstance(self.relations[g.index], AttrEqual)
                         and feature.var in (self.relations[g.index].left_var, self.relations[g.index].right_var))]
                    if guards and not any(params[g.include_key] for g in guards):
                        self._enable_random(params, rng.choice(guards), rng)
                self._connect_initial_head(params, rng)
                try: self.instantiate(params)
                except InvalidConfiguration: continue
                yield params
                break
            else:
                raise InvalidConfiguration('could not initialize an admissible configuration; supply model contexts')


def build_config_space(*, cluster_id, patterns, models, labels,
                       variables=('x', 'y'), relations=None):
    return GuardedConfigurationSpace(cluster_id=cluster_id, patterns=patterns, models=models,
                                     labels=labels, variables=variables, relations=relations)
