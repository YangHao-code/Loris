"""Finite, explicitly scoped RDL reference interpreter for PDF36 §§3/6.1.

This is a correctness oracle, not the production corpus executor. It enumerates
all valuations, without a round, document, or grounding cutoff. Every variable
is explicitly bound by Doc; neither document labels nor model outputs seed Γ.

The author confirmed standard subset direction on 2026-09-13: x.lbl ⊆ y.lbl
copies positive labels from x to y. ``standard`` uses this ordinary set
notation. ``paper_p7_reverse`` retains the reversed non-equality comparison
interpretation only for explicit historical compatibility; it is not the
author-confirmed semantics. Every new result records this distinction.

Comparison premises use established comparison facts, never coincidental
containment/equality of partial positive bitmaps. Strict inclusion is retained
as a proper-inclusion constraint, with transitivity and contradiction checks;
no fresh label/witness is invented. As in §6.1, inclusion propagates positive
labels. The text does not prescribe contraposition of negative labels, so this
interpreter does not introduce that additional inference rule.

Text/ML consequences occur in the general §3 grammar, but §6.1 provides no
operational update for them. They fail explicitly rather than silently becoming
ADD. Sim, Group, quorum and unscoped legacy RDLs also fail explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import re
from typing import Mapping

import numpy as np

from loris.chase import ChaseResult, GammaSeed
from loris.document import Document, VALID_ATTRS
from loris.predicates import (MatchPredicate, FreqPredicate, CooccurPredicate,
                              BeforePredicate, predicate_to_dict, predicate_from_dict)


SCHEMA = 'paper36_explicit_variable_rdl_v1'
SEMANTICS = 'paper36_scoped_explicit_facts_reference_v1'
TEXT_TYPES = (MatchPredicate, FreqPredicate, CooccurPredicate, BeforePredicate)


def is_attribute_name(attr):
    """Base document fields or one explicit named metadata key; no legacy groups."""
    return isinstance(attr, str) and (attr in VALID_ATTRS or
        re.fullmatch(r'mtd\.[A-Za-z_][A-Za-z_0-9]*', attr) is not None)


def prepare_attributes(docs, attribute_values=None, attribute_presence=None):
    """Normalize an explicitly declared document view for both execution engines.

    Named values are strings (possibly canonical complete-set keys) or None.
    Equality compares the complete strings, never shared elements or neighbors.
    Missing values cannot support equality or textual guards. Empty strings use
    the Document missing convention unless explicitly certified as observed.
    Field extraction, key normalization, and provenance are caller-owned inputs.
    """
    docs = tuple(docs)
    n = len(docs)
    values = {attr: tuple(doc.get_attr(attr) for doc in docs) for attr in VALID_ATTRS}
    for attr, column in (attribute_values or {}).items():
        if not is_attribute_name(attr) or attr in VALID_ATTRS:
            raise ValueError('attribute_values keys must be declared mtd.<identifier> fields; base fields come from docs')
        if isinstance(column, (str, bytes)):
            raise ValueError('declared attribute column must have one value per document')
        values[attr] = tuple(column)
    for attr, column in values.items():
        if len(column) != n:
            raise ValueError('attribute value document axis differs: ' + attr)
        if any(value is not None and not isinstance(value, str) for value in column):
            raise ValueError('attribute values must be explicit strings or None: ' + attr)
    supplied = attribute_presence or {}
    if set(supplied) - set(values):
        raise ValueError('presence references an undeclared attribute: ' + str(sorted(set(supplied) - set(values))))
    presence = {}
    for attr, column in values.items():
        if attr in supplied:
            mask = np.asarray(supplied[attr])
            if mask.ndim != 1 or mask.shape != (n,):
                raise ValueError('attribute presence document axis differs: ' + attr)
            if not np.isin(mask, [0, 1]).all():
                raise ValueError('attribute presence must be binary: ' + attr)
            mask = np.array(mask, bool, copy=True)
        else:
            mask = np.array([value is not None and value != '' for value in column], bool)
        if any(mask[i] and value is None for i, value in enumerate(column)):
            raise ValueError('None cannot be certified as an observed attribute value: ' + attr)
        mask.setflags(write=False)
        presence[attr] = mask
    return values, presence


def _name(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(field + ' must be a nonempty string')


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Doc:
    var: str

    def __post_init__(self):
        _name(self.var, 'var')

    def to_dict(self):
        return {'type': 'Doc', 'var': self.var}


@dataclass(frozen=True)
class Label:
    var: str
    label: str
    polarity: str = 'pos'

    def __post_init__(self):
        _name(self.var, 'var')
        _name(self.label, 'label')
        if self.polarity not in ('pos', 'neg'):
            raise ValueError('label polarity must be pos or neg')

    def to_dict(self):
        return {'type': 'Label', 'var': self.var, 'label': self.label, 'polarity': self.polarity}


@dataclass(frozen=True)
class Text:
    var: str
    predicate: object

    def __post_init__(self):
        _name(self.var, 'var')
        if type(self.predicate) not in TEXT_TYPES:
            raise TypeError('Text requires a paper Match/Freq/Cooccur/Before predicate')
        if self.predicate.sim:
            raise ValueError('semantic text similarity is not a PDF36 textual primitive')
        for field in ('r', 'r1', 'r2'):
            pattern = getattr(self.predicate, field, None)
            if pattern is not None:
                re.compile(pattern.raw, pattern.flags)

    def to_dict(self):
        return {'type': 'Text', 'var': self.var, 'predicate': predicate_to_dict(self.predicate)}


@dataclass(frozen=True)
class ML:
    var: str
    alias: str
    label: str
    eta: float = 0.5
    dynamic: bool = True

    def __post_init__(self):
        for value, field in ((self.var, 'var'), (self.alias, 'alias'), (self.label, 'label')):
            _name(value, field)
        if not isinstance(self.dynamic, bool):
            raise ValueError('dynamic must be explicit bool')
        if isinstance(self.eta, bool) or not np.isfinite(self.eta) or not 0 <= self.eta <= 1:
            raise ValueError('eta must be finite in [0,1]')
        object.__setattr__(self, 'eta', float(self.eta))

    def to_dict(self):
        return {'type': 'ML', 'var': self.var, 'alias': self.alias, 'label': self.label,
                'eta': self.eta, 'dynamic': self.dynamic}


@dataclass(frozen=True)
class AttrEqual:
    left_var: str
    left_attr: str
    right_var: str
    right_attr: str

    def __post_init__(self):
        _name(self.left_var, 'left_var')
        _name(self.right_var, 'right_var')
        if not is_attribute_name(self.left_attr) or not is_attribute_name(self.right_attr):
            raise ValueError('attribute equality requires cnt/ttl/mtd or declared mtd.key; virtual Group is not this primitive')

    def to_dict(self):
        return {'type': 'AttrEqual', 'left_var': self.left_var, 'left_attr': self.left_attr,
                'right_var': self.right_var, 'right_attr': self.right_attr}


@dataclass(frozen=True)
class LabelCompare:
    left_var: str
    op: str
    right_var: str

    def __post_init__(self):
        _name(self.left_var, 'left_var')
        _name(self.right_var, 'right_var')
        if self.op not in ('eq', 'subset', 'strict_subset'):
            raise ValueError('comparison op must be eq/subset/strict_subset')

    def to_dict(self):
        return {'type': 'LabelCompare', 'left_var': self.left_var, 'op': self.op, 'right_var': self.right_var}


ATOM_TYPES = (Doc, Label, Text, ML, AttrEqual, LabelCompare)


def atom_from_dict(value):
    if not isinstance(value, Mapping):
        raise TypeError('typed atom record must be a mapping')
    value = dict(value)
    kind = value.pop('type', None)
    classes = {cls.__name__: cls for cls in ATOM_TYPES}
    if kind not in classes:
        raise ValueError('unsupported paper atom: ' + str(kind))
    if kind == 'Text':
        value['predicate'] = predicate_from_dict(value['predicate'])
    return classes[kind](**value)


def atom_variables(atom):
    if type(atom) not in ATOM_TYPES:
        raise TypeError('unscoped/unknown predicates require an explicit adapter')
    return (atom.left_var, atom.right_var) if isinstance(atom, (AttrEqual, LabelCompare)) else (atom.var,)


@dataclass(frozen=True)
class Rule:
    body: tuple
    head: object

    def __post_init__(self):
        object.__setattr__(self, 'body', tuple(self.body))
        atoms = self.body + (self.head,)
        for atom in atoms:
            atom_variables(atom)
        bound = {a.var for a in self.body if isinstance(a, Doc)}
        used = {var for atom in atoms for var in atom_variables(atom)}
        if used != bound:
            raise ValueError('all text variables must be bound by Doc in body: ' + str(sorted(used - bound)))
        if not isinstance(self.head, (Label, LabelCompare)):
            raise NotImplementedError('§6.1 only specifies label/comparison state updates; this head is unsupported')

    @property
    def variables(self):
        return tuple(dict.fromkeys(a.var for a in self.body if isinstance(a, Doc)))

    def to_dict(self):
        return {'schema': SCHEMA, 'body': [a.to_dict() for a in self.body], 'head': self.head.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or value.get('schema') != SCHEMA:
            raise ValueError('explicit scoped RDL schema required; legacy RDL is not accepted')
        if set(value) != {'schema', 'body', 'head'}:
            raise ValueError('unknown fields in scoped rule')
        return cls(tuple(atom_from_dict(a) for a in value['body']), atom_from_dict(value['head']))


@dataclass(frozen=True)
class RefGamma:
    """Validated facts; each inclusion pair is (source, target), independent of glyph mode."""
    pos: np.ndarray
    neg: np.ndarray
    subset: tuple = ()
    strict_subset: tuple = ()

    def __post_init__(self):
        seed = GammaSeed(self.pos, self.neg)
        object.__setattr__(self, 'pos', seed.pos)
        object.__setattr__(self, 'neg', seed.neg)
        n = seed.shape[0]
        for field in ('subset', 'strict_subset'):
            edges = []
            for edge in getattr(self, field):
                if len(edge) != 2 or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in edge):
                    raise ValueError('comparison seed must contain integer (source,target) pairs')
                edge = tuple(map(int, edge))
                if min(edge) < 0 or max(edge) >= n:
                    raise ValueError('comparison seed outside document axis')
                edges.append(edge)
            object.__setattr__(self, field, tuple(sorted(set(edges))))

    @property
    def shape(self):
        return self.pos.shape

    @classmethod
    def empty(cls, n, k):
        return cls(np.zeros((n, k), bool), np.zeros((n, k), bool))


def comparison_feasible(pos, neg, inclusion, proper):
    """Check finite-set comparison consistency without inventing label facts.

    Acyclic proper-inclusion constraints alone are insufficient: with one label
    x ⊂ y ⊂ z is impossible. This exact binary feasibility formulation uses
    source[k] <= target[k] for inclusion and |target|-|source| >= 1 for proper
    inclusion. Only a feasibility decision is used; a solver's possible label
    completion must never become observed facts. No time/node limit is imposed.
    """
    if not proper:
        return True
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    nodes = sorted({node for edge in inclusion | proper for node in edge
                    if edge[0] != edge[1]})
    lookup, k = {node: i for i, node in enumerate(nodes)}, pos.shape[1]
    size = len(nodes) * k
    if not size:
        return False
    lower = pos[nodes].astype(np.float64).ravel()
    upper = (~neg[nodes]).astype(np.float64).ravel()
    if np.any(lower > upper):
        return False
    row, column, value, bounds = [], [], [], []
    for source, target in sorted(inclusion | proper):
        if source == target:
            continue
        for label in range(k):
            r = len(bounds)
            row.extend([r, r])
            column.extend([lookup[source] * k + label, lookup[target] * k + label])
            value.extend([1., -1.])
            bounds.append(0.)
    for source, target in sorted(proper):
        if source == target:
            return False
        r = len(bounds)
        for label in range(k):
            row.extend([r, r])
            column.extend([lookup[source] * k + label, lookup[target] * k + label])
            value.extend([1., -1.])
        bounds.append(-1.)
    matrix = coo_matrix((value, (row, column)), shape=(len(bounds), size)).tocsr()
    result = milp(np.zeros(size), integrality=np.ones(size), bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, -np.inf, np.asarray(bounds)),
        options={'mip_rel_gap': 0.0})
    if result.status == 2:
        return False
    if result.status != 0 or result.x is None:
        raise RuntimeError('complete comparison feasibility solver did not certify an outcome: ' + str(result.message))
    integer = np.rint(result.x).astype(np.int64)
    if (np.any(np.abs(result.x - integer) > 1e-6) or np.any(integer < lower)
            or np.any(integer > upper) or np.any(matrix.dot(integer) > np.asarray(bounds))):
        raise RuntimeError('comparison feasibility returned an invalid integer witness')
    return True


class ReferenceChase:
    """Complete finite oracle with explicit scopes and an immutable model table.

    Both fixed and adaptable ML predicates compare confidence against their
    declared kappa. Fixed probabilities can be bound separately across rounds.
    Results retain a first-witness proof DAG for label and comparison facts.
    Scheduler rounds belong to this reference, not the old MultiChase scheduler.
    """
    def __init__(self, labels, model_probabilities=None, model_decisions=None,
                 subset_direction='standard', attribute_presence=None, attribute_values=None,
                 static_model_probabilities=None):
        self.labels = tuple(labels)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError('unique nonempty fixed label axis required')
        for label in self.labels:
            _name(label, 'label')
        if subset_direction not in ('standard', 'paper_p7_reverse'):
            raise ValueError('subset direction must be explicitly standard or paper_p7_reverse')
        self.subset_direction = subset_direction
        self.model_probabilities = self._tables(model_probabilities or {}, False)
        # A missing static override freezes this initial interpretation.  The
        # dictionaries stay independent; their immutable buffers may be shared.
        self.static_model_probabilities = (dict(self.model_probabilities)
            if static_model_probabilities is None
            else self._tables(static_model_probabilities, False))
        self.model_decisions = self._tables(model_decisions or {}, True)
        self._label_index = {label: k for k, label in enumerate(self.labels)}
        self.attribute_values = {}
        for attr, column in (attribute_values or {}).items():
            if not is_attribute_name(attr) or attr in VALID_ATTRS:
                raise ValueError('attribute_values keys must be declared mtd.<identifier> fields')
            if isinstance(column, (str, bytes)):
                raise ValueError('declared attribute column requires a row sequence')
            column = tuple(column)
            if any(value is not None and not isinstance(value, str) for value in column):
                raise ValueError('attribute values must be explicit strings or None')
            self.attribute_values[attr] = column
        self.attribute_presence = {}
        for attr, present in (attribute_presence or {}).items():
            present = np.asarray(present)
            if not is_attribute_name(attr) or present.ndim != 1 or not np.isin(present, [0, 1]).all():
                raise ValueError('attribute_presence requires declared attribute binary row masks')
            present = np.array(present, bool, copy=True)
            present.setflags(write=False)
            self.attribute_presence[attr] = present

    def _tables(self, mapping, binary):
        output = {}
        for alias, value in mapping.items():
            _name(alias, 'model alias')
            value = np.asarray(value)
            if value.ndim != 2 or value.shape[1] != len(self.labels) or not np.isfinite(value).all():
                raise ValueError('model table requires finite [N,L] on the fixed label axis')
            if binary and not np.isin(value, [0, 1]).all():
                raise ValueError('static model decisions must be binary')
            if not binary and ((value < 0).any() or (value > 1).any()):
                raise ValueError('model probabilities must be in [0,1]')
            from .array_cache import _readonly
            output[alias] = _readonly(np.asarray(value, dtype=bool if binary else np.float64))
        return output

    def _edge(self, atom, valuation):
        left, right = valuation[atom.left_var], valuation[atom.right_var]
        if self.subset_direction == 'paper_p7_reverse' and atom.op != 'eq':
            return right, left
        return left, right

    def run(self, docs, rules, gamma):
        docs = tuple(Document(cnt=d.cnt, ttl=d.ttl, mtd=d.mtd) for d in docs)
        rules = tuple(rules)
        n, k = len(docs), len(self.labels)
        if not isinstance(gamma, (GammaSeed, RefGamma)) or gamma.pos.shape != (n, k):
            raise ValueError('explicit Gamma on the complete document/label axes is required')
        if not isinstance(gamma, RefGamma):
            gamma = RefGamma(gamma.pos, gamma.neg)
        for rule in rules:
            if not isinstance(rule, Rule):
                raise TypeError('ReferenceChase requires typed Rule, not legacy/unscoped RDL')
            for atom in rule.body + (rule.head,):
                if isinstance(atom, (Label, ML)) and atom.label not in self._label_index:
                    raise ValueError('rule label outside frozen label axis: ' + atom.label)
        for tables in (self.model_probabilities, self.static_model_probabilities, self.model_decisions):
            if any(table.shape != (n, k) for table in tables.values()):
                raise ValueError('model table document axis differs from full corpus')
        attribute_values, presence = prepare_attributes(docs, self.attribute_values, self.attribute_presence)
        for rule in rules:
            for atom in rule.body:
                if isinstance(atom, AttrEqual) and (atom.left_attr not in attribute_values or atom.right_attr not in attribute_values):
                    raise ValueError('rule references an undeclared named attribute')
        pos, neg = gamma.pos.copy(), gamma.neg.copy()
        inclusion = {(i, i) for i in range(n)}
        proper = set()
        proof = {}
        provenance = {}
        comparison_conflicts = []
        rounds, valuations_checked = 0, 0
        comparison_checks, comparison_check_version = 0, None

        def fact_key(kind, i, j):
            return '%s:%d:%d' % (kind, i, j)

        def remember(kind, i, j, origin, dependencies=(), rule_id=None, valuation=None, static=()):
            key = fact_key(kind, i, j)
            if key not in proof:
                proof[key] = {'kind': kind, 'row': int(i), 'column': int(j), 'origin': origin,
                    'dependencies': list(dependencies), 'rule_id': rule_id,
                    'valuation': dict(valuation or {}), 'static_evidence': list(static)}
                if kind in ('pos', 'neg'):
                    ids = set()
                    if rule_id is not None:
                        ids.add(rule_id)
                    for dependency in dependencies:
                        p = proof[dependency]
                        ids.update(p.get('support_rule_ids', []))
                    proof[key]['support_rule_ids'] = sorted(ids)
                    if ids:
                        provenance[(i, self.labels[j])] = sorted(set(provenance.get((i, self.labels[j]), [])) | ids)
                else:
                    ids = {rule_id} if rule_id is not None else set()
                    for dependency in dependencies:
                        ids.update(proof[dependency].get('support_rule_ids', []))
                    proof[key]['support_rule_ids'] = sorted(ids)
            return key

        for sign, matrix in (('pos', pos), ('neg', neg)):
            for i, j in np.argwhere(matrix):
                remember(sign, int(i), int(j), 'validated_Gamma')
        for i in range(n):
            remember('subset', i, i, 'identity')
        for edge in gamma.subset:
            inclusion.add(edge)
            remember('subset', *edge, origin='validated_Gamma')
        for edge in gamma.strict_subset:
            proper.add(edge)
            remember('strict_subset', *edge, origin='validated_Gamma')

        def invalid():
            nonlocal comparison_checks, comparison_check_version
            if (pos & neg).any():
                return True
            for source, target in sorted(proper):
                if (target, source) in inclusion:
                    comparison_conflicts.append({'kind': 'strict_inclusion_cycle', 'source': source, 'target': target})
                    return True
                # A proper inclusion needs at least one possible tau in target
                # but not source. Fully known evidence can rule all such labels out.
                if not np.any(~pos[source] & ~neg[target]):
                    comparison_conflicts.append({'kind': 'strict_inclusion_has_no_possible_witness',
                                                 'source': source, 'target': target})
                    return True
            # This run is monotone. Every new label/comparison fact inserts one
            # proof key, so proof length identifies its current consistency state
            # without hashing the full arrays at every unchanged valuation.
            if proper and comparison_check_version != len(proof):
                comparison_checks += 1
                if not comparison_feasible(pos, neg, inclusion, proper):
                    comparison_conflicts.append({'kind': 'finite_label_comparisons_unsatisfiable'})
                    return True
                comparison_check_version = len(proof)
            return False

        def close_comparisons():
            changed_any = False
            while True:
                changed = False
                for source, target in sorted(proper):
                    if (source, target) not in inclusion:
                        inclusion.add((source, target))
                        remember('subset', source, target, 'strict_implies_subset',
                                 [fact_key('strict_subset', source, target)])
                        changed = True
                # Relation facts, not coincidental positive-set inclusion, form
                # the transitive closure. Snapshotting keeps proof dependencies older.
                by_source = {}
                for a, b in sorted(inclusion):
                    by_source.setdefault(a, []).append(b)
                for a, b in sorted(inclusion):
                    for c in by_source.get(b, []):
                        deps = [fact_key('subset', a, b), fact_key('subset', b, c)]
                        if (a, c) not in inclusion:
                            inclusion.add((a, c))
                            remember('subset', a, c, 'inclusion_transitivity', deps)
                            changed = True
                        strict_deps = None
                        if (a, b) in proper:
                            strict_deps = [fact_key('strict_subset', a, b), fact_key('subset', b, c)]
                        elif (b, c) in proper:
                            strict_deps = [fact_key('subset', a, b), fact_key('strict_subset', b, c)]
                        if strict_deps is not None and (a, c) not in proper:
                            proper.add((a, c))
                            remember('strict_subset', a, c, 'proper_inclusion_transitivity', strict_deps)
                            changed = True
                for source, target in sorted(inclusion):
                    for label_index in np.flatnonzero(pos[source] & ~pos[target]):
                        label_index = int(label_index)
                        pos[target, label_index] = True
                        remember('pos', target, label_index, 'inclusion_label_copy',
                                 [fact_key('subset', source, target), fact_key('pos', source, label_index)])
                        changed = True
                changed_any |= changed
                if invalid() or not changed:
                    return changed_any

        static_masks = {}
        for rule in rules:
            for atom in rule.body:
                if isinstance(atom, (Text, ML)):
                    key = _canonical(atom.to_dict())
                    if key in static_masks:
                        continue
                    if isinstance(atom, Text):
                        mask = np.array([bool(presence[atom.predicate.attr][i]) and atom.predicate(doc)
                                         for i, doc in enumerate(docs)], bool)
                    else:
                        tables = self.model_probabilities if atom.dynamic else self.static_model_probabilities
                        if atom.alias not in tables:
                            raise ValueError('missing explicit probability table for ' + atom.alias)
                        column = tables[atom.alias][:, self._label_index[atom.label]]
                        mask = column >= atom.eta
                    static_masks[key] = mask

        def body_matches(rule, valuation):
            dependencies, static = [], []
            for atom in rule.body:
                if isinstance(atom, Doc):
                    continue
                if isinstance(atom, Label):
                    i, j = valuation[atom.var], self._label_index[atom.label]
                    if not (pos if atom.polarity == 'pos' else neg)[i, j]:
                        return None
                    dependencies.append(fact_key(atom.polarity, i, j))
                elif isinstance(atom, (Text, ML)):
                    if not static_masks[_canonical(atom.to_dict())][valuation[atom.var]]:
                        return None
                    static.append({'atom': atom.to_dict(), 'row': valuation[atom.var]})
                elif isinstance(atom, AttrEqual):
                    left_i, right_i = valuation[atom.left_var], valuation[atom.right_var]
                    if not presence[atom.left_attr][left_i] or not presence[atom.right_attr][right_i]:
                        return None
                    if attribute_values[atom.left_attr][left_i] != attribute_values[atom.right_attr][right_i]:
                        return None
                    static.append({'atom': atom.to_dict(), 'rows': [valuation[atom.left_var], valuation[atom.right_var]]})
                elif isinstance(atom, LabelCompare):
                    source, target = self._edge(atom, valuation)
                    if atom.op == 'eq':
                        if (source, target) not in inclusion or (target, source) not in inclusion:
                            return None
                        dependencies.extend([fact_key('subset', source, target), fact_key('subset', target, source)])
                    elif atom.op == 'subset':
                        if (source, target) not in inclusion:
                            return None
                        dependencies.append(fact_key('subset', source, target))
                    else:
                        if (source, target) not in proper:
                            return None
                        dependencies.append(fact_key('strict_subset', source, target))
            return dependencies, static

        def apply_head(rule, rule_id, valuation, dependencies, static):
            head, changed = rule.head, False
            if isinstance(head, Label):
                i, j = valuation[head.var], self._label_index[head.label]
                matrix = pos if head.polarity == 'pos' else neg
                if not matrix[i, j]:
                    matrix[i, j] = True
                    remember(head.polarity, i, j, 'rule_application', dependencies, rule_id, valuation, static)
                    changed = True
            else:
                source, target = self._edge(head, valuation)
                kind = 'strict_subset' if head.op == 'strict_subset' else 'subset'
                collection = proper if head.op == 'strict_subset' else inclusion
                edges = [(source, target), (target, source)] if head.op == 'eq' else [(source, target)]
                for a, b in edges:
                    if (a, b) not in collection:
                        collection.add((a, b))
                        remember(kind, a, b, 'rule_application', dependencies, rule_id, valuation, static)
                        changed = True
            if changed:
                close_comparisons()
            return changed

        close_comparisons()
        status = 'conflict' if invalid() else 'fixpoint'
        if status != 'conflict':
            while True:
                rounds += 1
                changed = False
                for rule_id, rule in enumerate(rules):
                    for values in itertools.product(range(n), repeat=len(rule.variables)):
                        valuation = dict(zip(rule.variables, values))
                        valuations_checked += 1
                        matched = body_matches(rule, valuation)
                        if matched is None:
                            continue
                        changed |= apply_head(rule, rule_id, valuation, *matched)
                        if invalid():
                            status = 'conflict'
                            break
                    if status == 'conflict':
                        break
                if status == 'conflict' or not changed:
                    break
        conflicts = [(int(i), self.labels[int(j)]) for i, j in np.argwhere(pos & neg)]
        result = ChaseResult(status=status, predictions=(pos & ~neg).astype(np.float32),
            conflicts=conflicts, n_rounds=rounds, provenance=provenance,
            positive_facts=pos.copy(), negative_facts=neg.copy())
        result.subset_facts = tuple(sorted(inclusion))
        result.strict_subset_facts = tuple(sorted(proper))
        result.comparison_conflicts = comparison_conflicts[:1]
        result.proof_dag = proof
        result.reference_audit = {'semantics': SEMANTICS, 'full_rows': n, 'label_names': list(self.labels),
            'subset_direction': self.subset_direction, 'subset_direction_pending_author_confirmation': False,
            'subset_direction_author_confirmed': self.subset_direction == 'standard',
            'subset_direction_legacy_compatibility': self.subset_direction == 'paper_p7_reverse',
            'proper_inclusion_policy': 'explicit proper constraint; no fresh witness invented; no proper cycles or impossible witness',
            'comparison_consistency': 'complete finite binary feasibility; solver completion never added as facts',
            'comparison_feasibility_checks': comparison_checks,
            'negative_contraposition_added': False, 'model_outputs_unioned_into_Gamma': False,
            'document_lbl_used_as_Gamma': False, 'valuation_count': valuations_checked,
            'missing_attribute_equality': 'requires both observed; default empty string means missing unless explicit presence provided',
            'attribute_presence_sha256': {attr: hashlib.sha256(mask.tobytes()).hexdigest() for attr, mask in sorted(presence.items())},
            'attribute_values_sha256': {attr: _sha(column) for attr, column in sorted(attribute_values.items())},
            'declared_named_attributes': sorted(self.attribute_values),
            'named_attribute_relation': 'complete observed key equality, never shared-element/group/quorum matching',
            'complete_without_cutoff': True, 'program_sha256': _sha([r.to_dict() for r in rules]),
            'comparison_premises_from_partial_bitmaps': False,
            'rounds_are_reference_scheduler_only': True}
        return result
