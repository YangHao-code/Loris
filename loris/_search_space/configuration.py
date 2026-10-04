"""Author-clarified connectivity of every substantively used rule variable.

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

from loris._engine.rdl_reference import AttrEqual, Doc, LabelCompare, Rule, atom_variables
from loris._engine.rlearner_config import (ConfigurationSpace, InvalidConfiguration,
                                         _canonical, _number)


GUARD_POLICY = {
    'schema': 'paper36_author_all_used_variables_connected_v1',
    'authority': 'author clarification 2026-09-10: no disconnected body/head document variables',
    'pdf_grammar_distinction': 'Doc binds variables; body connectivity is the additional author clarification',
    'used_variables': 'union of variables in every non-Doc body atom and the head',
    'required_body_connection': 'all used variables form one component through non-self AttrEqual or LabelCompare body atoms',
    'graph_direction': 'undirected structural dependency; original predicate direction and semantics retained',
    'non_connecting_predicates': ['Doc', 'Text', 'ML', 'Label'],
    'head_scope': 'all Label positive/negative and LabelCompare heads, including same-variable comparison heads',
    'single_document_rules': 'zero or one substantively used variable retains original semantics; Doc-only single-document constant rules remain in scope',
    'unused_Doc_bindings': 'binding declarations only; do not create used variables or connection edges',
    'startup_prior': {
        'schema': 'paper36_author_connected_sparse_startup_v1',
        'head_and_feature_axes': 'all original axes and positive feature probabilities retained',
        'repair': 'join used-variable components by uniformly choosing existing cross-component relation primitives incident to current used variables; unused intermediate variables may be introduced',
        'selected_features_removed': False,
        'head_resampled_for_connectivity': False,
        'later_TPE_proposals_repaired': False,
        'missing_connecting_inventory': 'explicit initialization error before retries; never resample heads or drop trials',
        'historical_rule_banks': False,
        'budget': 'initial configurations and all later invalid configurations consume the original study budget',
        'equivalent_prior_to_previous_study': False,
    },
}


def guard_policy_binding():
    return {'policy': copy.deepcopy(GUARD_POLICY),
            'configuration_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


policy_binding = guard_policy_binding


def _components(used, relations):
    """Return a deterministic partition; self relations never join components."""
    graph = {var: set() for var in used}
    for atom in relations:
        if not isinstance(atom, (AttrEqual, LabelCompare)) or atom.left_var == atom.right_var:
            continue
        graph.setdefault(atom.left_var, set()).add(atom.right_var)
        graph.setdefault(atom.right_var, set()).add(atom.left_var)
    remaining, result = set(used), []
    while remaining:
        seen, pending = {min(remaining)}, [min(remaining)]
        while pending:
            for other in graph.get(pending.pop(), ()):
                if other not in seen:
                    seen.add(other); pending.append(other)
        result.append(frozenset(seen & set(used)))
        remaining.difference_update(seen)
    return tuple(result)


def _used_variables(head, atoms):
    return {var for atom in tuple(atoms) + (head,) if not isinstance(atom, Doc)
            for var in atom_variables(atom)}


def _connected(head, atoms):
    return len(_components(_used_variables(head, atoms), atoms)) <= 1


def validate_propagation_guard(rule):
    """Reject disconnected substantive variables without modifying any atom."""
    if not isinstance(rule, Rule):
        raise TypeError('an explicit typed Rule is required')
    used = _used_variables(rule.head, rule.body)
    parts = _components(used, rule.body)
    if len(parts) > 1:
        raise InvalidConfiguration(
            'all used body/head variables require a connected cross-variable body guard; '
            'disconnected components=%s; Doc bindings and independent Text/ML/Label predicates '
            'do not connect documents' % [sorted(part) for part in parts])

validate_rule_guard = validate_propagation_guard


class ConnectedConfigurationSpace(ConfigurationSpace):
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

    def _connect_initial_variables(self, params, rng):
        head = self._head(params, set())
        used, selected = set(atom_variables(head)), []
        relation_features = []
        for feature in self.features:
            if feature.kind == 'relation':
                atom = self.relations[feature.index]
                relation_features.append(feature)
                if params[feature.include_key]:
                    selected.append(atom)
                    used.update(atom_variables(atom))
            elif params[feature.include_key]:
                used.add(feature.var)
        if len(_components(used, selected)) <= 1:
            return
        # Check the entire declared inventory first. This may connect original
        # variables through a declared but so-far unused intermediate variable.
        reachable = _components(set(self.variables), self.relations)
        enclosing = next((part for part in reachable if used <= part), None)
        if enclosing is None:
            raise InvalidConfiguration(
                'cannot initialize connected rule: no declared relation path joins all '
                'used variables; never resample a head or remove selected features')
        while len(_components(used, selected)) > 1:
            parts = _components(set(self.variables), selected)
            component = {var: index for index, part in enumerate(parts) for var in part}
            candidates = [feature for feature in relation_features
                if not params[feature.include_key]
                and self.relations[feature.index].left_var in enclosing
                and self.relations[feature.index].right_var in enclosing
                and component[self.relations[feature.index].left_var] != component[self.relations[feature.index].right_var]
                and (self.relations[feature.index].left_var in used
                     or self.relations[feature.index].right_var in used)]
            if not candidates:
                raise RuntimeError('connected relation inventory has no joining edge')
            chosen = rng.choice(candidates)
            self._enable_random(params, chosen, rng)
            atom = self.relations[chosen.index]
            selected.append(atom)
            used.update(atom_variables(atom))

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
            # new all-variable guard is made legal in the first proposal, before the
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
                self._connect_initial_variables(params, rng)
                try: self.instantiate(params)
                except InvalidConfiguration: continue
                yield params
                break
            else:
                raise InvalidConfiguration('could not initialize an admissible configuration; supply model contexts')


def build_config_space(*, cluster_id, patterns, models, labels,
                       variables=('x', 'y'), relations=None):
    return ConnectedConfigurationSpace(cluster_id=cluster_id, patterns=patterns, models=models,
                                     labels=labels, variables=variables, relations=relations)
