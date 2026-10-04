"""Fit-witness conditioned proposals with an unchanged full-support TPE branch.

This is a new BO proposal policy, not an equivalent replay of an earlier study.
Only source_fit documents, their explicit Gamma, and frozen model functions are
accepted. Every proposal is still scored once by the original line-7 evaluator;
all completed configurations still reach the original ordered line-12 gate.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, replace
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
from weakref import WeakKeyDictionary

import numpy as np
from optuna.distributions import CategoricalDistribution

from loris.predicates import FreqPredicate
from loris._engine.rdl_reference import AttrEqual, LabelCompare, prepare_attributes
from loris._engine.rlearner_config import _canonical
from loris._engine.sparse_bo import JointBinaryTPESampler
from loris._engine import rlearner as original

SCHEMA = 'paper36_structured_witness_tpe_v1'
DEFAULT_BROAD_PROBABILITY = .25
HEAD_WITNESS_POLICIES = ('body_only', 'complete_fit_labels_v1')
_MASKS = WeakKeyDictionary()


@dataclass(frozen=True)
class _FitDocument:
    cnt: str
    ttl: str
    mtd: str

    def get_attr(self, name):
        if name not in ('cnt','ttl','mtd'): raise ValueError('undeclared text attribute')
        return getattr(self, name)


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _source(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _array(value, shape, *, binary=False):
    arr = np.asarray(value)
    if arr.shape != shape or (binary and (arr.dtype != np.bool_)):
        raise ValueError('fit witness array shape/dtype differs from its explicit axis')
    if not binary and (arr.dtype.kind != 'f' or not np.isfinite(arr).all()
                       or np.any(arr < 0) or np.any(arr > 1)):
        raise ValueError('model probabilities must be finite floating values in [0,1]')
    # Reuse a shared fit table only when its backing storage is immutable.
    # A caller-owned write-protected ndarray still needs a defensive copy.
    from loris._engine.array_cache import _readonly
    return _readonly(arr)


def _array_id(arr):
    return {'shape': list(arr.shape), 'dtype': str(arr.dtype),
            'sha256': hashlib.sha256(arr.tobytes()).hexdigest()}


class FitWitnessPrior:
    """A detached fit-only interpretation; lexical masks are computed lazily.

    There are no labels inferred from unknowns, no corpus-level F1 scores, and no
    supplied candidate/rule bank. Relations use complete observed attribute keys,
    including named mtd.* values. Only selected atoms are evaluated per proposal.
    """
    def __init__(self, space, *, docs, row_ids, row_roles, labels, gamma_pos, gamma_neg,
                 model_probabilities=None, model_decisions=None, attribute_values=None,
                 attribute_presence=None, input_files=None, cache_dir=None,
                 fit_head_labels=None):
        docs = tuple(docs); rows = tuple(row_ids); roles = tuple(row_roles)
        if (not docs or len(rows) != len(docs) or len(set(rows)) != len(rows)
                or any(isinstance(r, bool) or not isinstance(r, (str,int)) or (isinstance(r,str) and not r) for r in rows)
                or roles != ('source_fit',) * len(docs)):
            raise ValueError('witness prior requires unique rows, all explicitly source_fit')
        if tuple(labels) != space.labels:
            raise ValueError('fit witness label identity differs from the full rule space')
        self.rows, self.labels, self.variables = rows, tuple(labels), space.variables
        self.space_id, self.cluster_id = space.space_id, space.cluster_id
        self.features = space.features
        self.patterns, self.models, self.relations = space.patterns, space.models, space.relations
        self.head_kinds, self.pairs = space.head_kinds, space.pairs
        values, presence = prepare_attributes(docs, attribute_values, attribute_presence)
        self.values = {k: tuple(v) for k, v in values.items()}
        self.presence = {k: _array(np.asarray(v, bool), (len(docs),), binary=True)
                         for k, v in presence.items()}
        self.docs = tuple(_FitDocument(cnt=d.get_attr('cnt'), ttl=d.get_attr('ttl'), mtd=d.get_attr('mtd')) for d in docs)
        shape = (len(rows), len(labels))
        self.pos = _array(gamma_pos, shape, binary=True)
        self.neg = _array(gamma_neg, shape, binary=True)
        if np.any(self.pos & self.neg):
            raise ValueError('fit witness Gamma contains a signed contradiction')
        # Optional *complete, labeled fit data* is separate supervision for
        # relation-head proposals. It never fills unknown Gamma/body facts.
        self.head_labels = None if fit_head_labels is None else _array(
            fit_head_labels, shape, binary=True)
        if self.head_labels is not None and (np.any(self.pos & ~self.head_labels)
                or np.any(self.neg & self.head_labels)):
            raise ValueError('complete fit head labels contradict explicit Gamma')
        probabilities, decisions = model_probabilities or {}, model_decisions or {}
        self.probabilities, self.decisions = {}, {}
        for model in self.models:
            table = probabilities if model.dynamic else decisions
            if model.alias not in table:
                raise ValueError('fit witness lacks the declared model interpretation: '+model.alias)
            target = self.probabilities if model.dynamic else self.decisions
            target[model.alias] = _array(table[model.alias], shape, binary=not model.dynamic)
        self.label_index = {v:i for i,v in enumerate(self.labels)}
        self.feature_index = {f.include_key:i for i,f in enumerate(self.features)}
        self.label_features = {(f.var,f.label,f.polarity):i for i,f in enumerate(self.features) if f.kind == 'label'}
        self.pattern_features = {v:tuple(i for i,f in enumerate(self.features) if f.kind == 'pattern' and f.var == v) for v in self.variables}
        self.attr_joins = []
        for i, atom in enumerate(self.relations):
            if not isinstance(atom, AttrEqual) or atom.left_var == atom.right_var:
                continue
            if atom.left_attr not in values or atom.right_attr not in values:
                raise ValueError('fit witness relation uses an undeclared extracted attribute')
            right = {}
            for r,value in enumerate(values[atom.right_attr]):
                if presence[atom.right_attr][r]: right.setdefault(value, []).append(r)
            matching = tuple(r for r,value in enumerate(values[atom.left_attr])
                             if presence[atom.left_attr][r] and value in right
                             and any(other != r for other in right[value]))
            if matching:
                self.attr_joins.append((i, matching, {k:tuple(v) for k,v in right.items()}))
        self.attr_joins = tuple(self.attr_joins)
        self.input_files = dict(input_files or {})
        for path, expected in self.input_files.items():
            if _source(path)['sha256'] != expected:
                raise ValueError('fit witness input file checksum differs')
        # cache_dir is a declared future execution location, never an input bank.
        self.cache_dir = str(Path(cache_dir).resolve()) if cache_dir is not None else None
        self.binding = self._binding()
        self.binding_id = _sha(self.binding)

    def _binding(self):
        return {'schema': SCHEMA+'_fit_interpretation', 'space_id': self.space_id,
                'cluster_id': self.cluster_id, 'row_ids': list(self.rows), 'rows': len(self.rows),
                'row_roles': ['source_fit']*len(self.rows), 'labels': list(self.labels),
                'attributes': {k:_sha(v) for k,v in self.values.items()},
                'attribute_presence': {k:_array_id(v) for k,v in self.presence.items()},
                'gamma_pos': _array_id(self.pos), 'gamma_neg': _array_id(self.neg),
                'complete_fit_head_labels': None if self.head_labels is None else _array_id(self.head_labels),
                'head_supervision_changes_Gamma_or_body_interpretation': False,
                'probabilities': {k:_array_id(v) for k,v in self.probabilities.items()},
                'decisions': {k:_array_id(v) for k,v in self.decisions.items()},
                'input_files': self.input_files, 'unknown_is_negative': False,
                'control_or_test_truth_used': False, 'precomputed_full_pattern_product': False,
                'pattern_domain_sha256': _sha([p.to_dict() for p in self.patterns]),
                'model_domain_sha256': _sha([m.to_dict() for m in self.models]),
                'relation_domain_sha256': _sha([r.to_dict() for r in self.relations]),
                'feature_axis_sha256': _sha([f.to_dict() for f in self.features]),
                'head_kinds': list(self.head_kinds), 'pairs': list(self.pairs),
                'variables': list(self.variables)}

    def validate(self, space=None):
        if self._binding() != self.binding or _sha(self.binding) != self.binding_id:
            raise ValueError('fit witness interpretation changed after certification')
        if space is not None and (space.space_id != self.space_id or
                tuple(f.include_key for f in space.features) != tuple(f.include_key for f in self.features)):
            raise ValueError('fit witness prior belongs to another complete space')
        for path, expected in self.input_files.items():
            if _source(path)['sha256'] != expected:
                raise ValueError('fit witness source file changed')
        return self.binding_id

    def text_holds(self, index, row, active):
        feature = self.features[index]; pred = self.patterns[feature.index].predicate
        if isinstance(pred, FreqPredicate):
            pred = replace(pred, op=active[feature.key+'.op'], eta=active[feature.key+'.eta'])
        if not self.presence[pred.attr][row]: return False
        # Lazy scalar lexical evidence; each atom is evaluated on its actual
        # assigned row, so separate marginal supports cannot fake a joint match.
        if self.cache_dir is not None:
            cache = _MASKS.get(self)
            if cache is None:
                from loris._engine.primitive_masks import TextPrimitiveCache
                cache = TextPrimitiveCache(self.docs, self.cache_dir,
                    attribute_presence={k:self.presence[k] for k in ('cnt','ttl','mtd')})
                _MASKS[self] = cache
            return bool(cache.mask(pred)[row])
        return bool(pred(self.docs[row]))

    def close(self):
        cache = _MASKS.pop(self, None)
        if cache is not None: cache.close()

    def holds(self, index, assignment, active):
        f = self.features[index]
        if f.kind == 'pattern': return self.text_holds(index, assignment[f.var], active)
        if f.kind == 'label':
            return bool((self.pos if f.polarity == 'pos' else self.neg)[assignment[f.var], self.label_index[f.label]])
        if f.kind == 'model':
            spec = self.models[f.index]; row = assignment[f.var]; label = self.label_index[f.label]
            return bool(self.probabilities[spec.alias][row,label] >= active[f.key+'.eta']) if spec.dynamic else bool(self.decisions[spec.alias][row,label])
        atom = self.relations[f.index]; a,b = assignment[atom.left_var],assignment[atom.right_var]
        if isinstance(atom, AttrEqual):
            return bool(self.presence[atom.left_attr][a] and self.presence[atom.right_attr][b]
                        and self.values[atom.left_attr][a] == self.values[atom.right_attr][b])
        # Partial positive sets cannot certify equality/subset of complete labels.
        if a == b: return atom.op != 'strict_subset'
        if not np.all((self.pos | self.neg)[[a,b]]): return False
        left,right = self.pos[a],self.pos[b]
        if atom.op == 'eq': return bool(np.array_equal(left,right))
        subset = bool(np.all(~left | right))
        return subset and (atom.op != 'strict_subset' or bool(np.any(right & ~left)))


def build_fit_witness_prior(space, **kwargs):
    """Explicit constructor: unknown kwargs (including truth) fail before use."""
    return FitWitnessPrior(space, **kwargs)


class StructuredWitnessTPESampler(JointBinaryTPESampler):
    def __init__(self, include_keys, *, seed, prior, broad_probability=DEFAULT_BROAD_PROBABILITY,
                 context_probes=16, head_witness_policy='body_only', **kwargs):
        if not isinstance(prior, FitWitnessPrior): raise TypeError('certified fit witness prior required')
        if not math.isfinite(broad_probability) or not 0 < broad_probability < 1:
            raise ValueError('strictly positive broad and structured proposal probabilities required')
        if type(context_probes) is not int or context_probes < 1:
            raise ValueError('positive explicit lexical context probe count required')
        prior.validate()
        if head_witness_policy not in HEAD_WITNESS_POLICIES:
            raise ValueError('unknown structured relation-head witness policy')
        if head_witness_policy == 'complete_fit_labels_v1' and prior.head_labels is None:
            raise ValueError('relation-head witnesses require complete authenticated fit labels')
        if tuple(include_keys) != tuple(f.include_key for f in prior.features):
            raise ValueError('the complete include axis must be preserved')
        super().__init__(include_keys, seed=seed, **kwargs)
        self.prior = prior
        self.broad_probability = float(broad_probability)
        self.context_probes = context_probes
        self.head_witness_policy = head_witness_policy
        base = self.protocol
        self.protocol = {'schema': SCHEMA, 'base_TPE': base,
            'fit_witness_binding': copy.deepcopy(prior.binding), 'fit_witness_id': prior.binding_id,
            'relation_head_witness_policy': self.head_witness_policy,
            'relation_head_supervision': 'complete labeled fit rows only; never inserted into Gamma or body evidence',
            'broad_probability': self.broad_probability,
            'structured_probability': 1-self.broad_probability,
            'initialization': 'original full-axis connected space initializer: up to 30 enqueued starts, all counted in budget; structured mixture begins after starts',
            'structured': 'condition one original joint TPE draw on one fit valuation, then couple source evidence/head and observed attribute connection',
            'context_probes': context_probes, 'context_probes_are_oracle_queries': False,
            'context_probe_scope': 'at most context_probes original pattern instances on each assigned variable, source first; both sides are best-effort, not a hard textual-context requirement',
            'cross_label_head_witness': 'choose head from source and target explicit same-polarity fit Gamma intersection; no intersection returns the same original draw; target label is evidence only and is omitted from body',
            'cross_fit_witness': 'distinct document rows sharing one observed declared AttrEqual; diagonal-only keys remain in the unmodified broad branch',
            'failed_conditioning': 'restore the pre-attempt conditional TPE delegate state and return the same binary draw, preserving the complete original head/eta/configuration; no replacement trial or score',
            'conditioned_head_literal': 'omit body Label on the same target variable+label (identical head or immediate signed contradiction) and directly identical/equivalent symmetric equality head; an empty resulting body returns the original draw',
            'full_original_parameter_axes': True, 'hard_support_or_F1_pruning': False,
            'negative_and_strict_heads_retained': True, 'historical_rules_imported': False,
            'all_nonnull_candidates_enter_ordered_line12': True,
            'source': _source(__file__), 'original_sparse_bo_source': _source(inspect.getfile(JointBinaryTPESampler)),
            'resume_policy': 'same complete source/input/policy binding and SHA-chained original compact observation journal',
            'equivalent_to_old_proposal_sequence': False}

    def _category(self, study, trial, key, choices):
        return self.delegate.sample_independent(study, trial, key, CategoricalDistribution(choices))

    def _active(self, f, study, trial):
        from optuna.distributions import FloatDistribution, IntDistribution
        out = {}
        if f.kind == 'pattern' and isinstance(self.prior.patterns[f.index].predicate, FreqPredicate):
            spec = self.prior.patterns[f.index]
            out[f.key+'.op'] = self._category(study, trial, f.key+'.op', spec.frequency_ops)
            domain = spec.frequency_domain
        elif f.kind == 'model' and self.prior.models[f.index].dynamic:
            domain = self.prior.models[f.index].threshold_domain
        else: return out
        dist = (CategoricalDistribution(domain.values) if domain.values else
                IntDistribution(int(domain.low),int(domain.high)) if domain.integer else
                FloatDistribution(float(domain.low),float(domain.high)))
        out[f.key+'.eta'] = self.delegate.sample_independent(study, trial, f.key+'.eta', dist)
        return out

    def sample_relative(self, study, trial, search_space):
        broad = super().sample_relative(study, trial, search_space)
        base_diagnostic = copy.deepcopy(self.last_diagnostic)
        self.last_diagnostic = {'schema': SCHEMA, 'trial': trial.number,
            'history': len(self.observations), 'base': base_diagnostic,
            'fit_witness_id': self.prior.binding_id, 'conditional_parameters': {}}
        if self.rng.random_sample() < self.broad_probability or len(self.prior.variables) > 2:
            self.last_diagnostic['branch'] = 'original_full_support_TPE'
            return broad
        # The joint binary draw already exists. A failed conditioning attempt
        # must also restore the conditional TPE state, so the actual Trial gets
        # the original head/eta values instead of silently drawing them again
        # from an advanced delegate. The mixture/assignment RNG is intentionally
        # part of the new proposal policy and is not rewound.
        delegate_before = copy.deepcopy(self.delegate)
        def fallback(reason, **details):
            self.delegate = delegate_before
            self.last_diagnostic.update(branch='original_full_support_TPE',
                conditioning_fallback=reason, conditional_parameters={},
                fallback_conditional_delegate_restored=True, **details)
            return broad
        prior = self.prior; planned = {}
        kind = self._category(study, trial, 'head.kind', prior.head_kinds)
        planned['head.kind'] = kind
        if kind in ('add','remove'):
            target = self._category(study, trial, 'head.var', prior.variables)
            planned['head.var'] = target
            planned['head.label'] = self._category(study, trial, 'head.label', prior.labels)
            cross = len(prior.variables) == 2 and self.rng.random_sample() < .5
        else:
            planned['head.pair'] = self._category(study, trial, 'head.pair', prior.pairs)
            target = planned['head.pair'].split('|')[1]; cross = True
        forced_relation = None
        relation_head_witness = None
        if cross:
            if not prior.attr_joins:
                return fallback('no observed distinct-row attribute join')
            relation_id,left_rows,right_index = prior.attr_joins[self.rng.randint(len(prior.attr_joins))]
            atom = prior.relations[relation_id]
            left = int(left_rows[self.rng.randint(len(left_rows))])
            right_rows = right_index[prior.values[atom.left_attr][left]]
            distinct = tuple(i for i in right_rows if i != left)
            pool = distinct
            if not pool: raise RuntimeError('certified distinct-row attribute join lost its witness')
            if kind not in ('add', 'remove') and self.head_witness_policy == 'complete_fit_labels_v1':
                head_left, head_right = planned['head.pair'].split('|')
                # Filter the complete observed bucket for this already drawn
                # left witness; no pair sampling/truncation or score replaces
                # either F1 filter. The full-support branch is untouched.
                fixed = prior.head_labels[left]
                candidates = prior.head_labels[np.asarray(pool, dtype=np.int64)]
                lhs, rhs = ((fixed, candidates) if head_left == atom.left_var
                            else (candidates, fixed))
                if kind == 'eq':
                    supported = np.all(lhs == rhs, axis=1)
                else:
                    supported = np.all(~lhs | rhs, axis=1)
                    if kind == 'strict_subset':
                        supported &= np.any(rhs & ~lhs, axis=1)
                pool = tuple(r for r, keep in zip(pool, supported) if keep)
                if not pool:
                    return fallback('no complete-fit relation-head witness in observed join',
                        attempted_head_kind=kind, attempted_head_pair=planned['head.pair'],
                        attempted_left_row_id=prior.rows[left])
            right = int(pool[self.rng.randint(len(pool))])
            assignment = {atom.left_var:left,atom.right_var:right}
            if kind not in ('add', 'remove') and self.head_witness_policy == 'complete_fit_labels_v1':
                relation_head_witness = {'kind':kind, 'head_pair':planned['head.pair'],
                    'left_row_id':prior.rows[assignment[head_left]],
                    'right_row_id':prior.rows[assignment[head_right]],
                    'complete_fit_label_identity':prior.binding['complete_fit_head_labels'],
                    'complete_fit_head_holds':True, 'inserted_into_Gamma_or_body':False}
            source = next(v for v in prior.variables if v != target)
            forced_relation = prior.feature_index['r%d.include'%relation_id]
        else:
            source = target; assignment = {source:int(self.rng.randint(len(prior.rows)))}
        row = assignment[source]
        # Only explicit fit facts guide labels. Cross label heads must also be
        # known at the target fit witness; control/test truth and F1 are never
        # consulted, and unknown Gamma entries do not become negative facts.
        head_witness = None
        if cross and kind in ('add','remove'):
            polarity = 'pos' if kind == 'add' else 'neg'
            facts = prior.pos if polarity == 'pos' else prior.neg
            preferred = facts[row] & facts[assignment[target]]
            if not np.any(preferred):
                return fallback('no shared explicit same-polarity fit head label',
                    attempted_head_kind=kind,
                    attempted_witness={v:prior.rows[r] for v,r in assignment.items()})
        else:
            preferred = prior.neg[row] if kind == 'remove' and np.any(prior.neg[row]) else prior.pos[row]
            polarity = 'neg' if kind == 'remove' and np.any(prior.neg[row]) else 'pos'
        choices = np.flatnonzero(preferred)
        source_feature = None
        if len(choices):
            label = planned.get('head.label')
            if label not in prior.label_index or not preferred[prior.label_index[label]]:
                label = prior.labels[int(choices[self.rng.randint(len(choices))])]
            if kind in ('add','remove'): planned['head.label'] = label
            source_feature = prior.label_features[(source,label,polarity)]
            if cross and kind in ('add','remove'):
                head_witness = {'label':label, 'polarity':polarity,
                    'source_variable':source, 'target_variable':target,
                    'source_row_id':prior.rows[row], 'target_row_id':prior.rows[assignment[target]],
                    'both_explicitly_known_in_fit_Gamma':True,
                    'target_fact_used_as_body_literal':False}
        chosen = set(); active = {}
        for index,f in enumerate(prior.features):
            if not broad[f.include_key]: continue
            needed = (f.var,) if f.var is not None else (prior.relations[f.index].left_var,prior.relations[f.index].right_var)
            if any(v not in assignment for v in needed): continue
            values = self._active(f,study,trial)
            if prior.holds(index,assignment,values): chosen.add(index);active.update(values)
        if forced_relation is not None: chosen.add(forced_relation)
        if source_feature is not None: chosen.add(source_feature)
        # Probe each assigned variable against the same fit valuation. Each
        # side has the declared upper bound; a missing textual context does not
        # reject an otherwise supported attribute/label propagation proposal.
        probe_variables = [source] + [v for v in prior.variables if v in assignment and v != source]
        context_probe_receipts = []
        for variable in probe_variables:
            indices = prior.pattern_features[variable]
            attempts = []
            if indices:
                for offset in self.rng.choice(len(indices), size=min(len(indices),self.context_probes), replace=False):
                    index=indices[int(offset)];f=prior.features[index];values=self._active(f,study,trial)
                    holds = prior.holds(index,assignment,values)
                    attempts.append({'include_key':f.include_key, 'parameters':values,
                                     'holds_at_fit_witness':bool(holds)})
                    if holds:
                        chosen.add(index);active.update(values);break
            context_probe_receipts.append({'variable':variable, 'row_id':prior.rows[assignment[variable]],
                'declared_max_probes':self.context_probes, 'actual_probes':len(attempts),
                'attempts':attempts, 'matched':any(a['holds_at_fit_witness'] for a in attempts)})
        # A selected model requires its original textual/attribute context.
        for index in tuple(chosen):
            f=prior.features[index]
            if f.kind != 'model': continue
            contextual = any(prior.features[j].kind == 'pattern' and prior.features[j].var == f.var for j in chosen)
            contextual |= any(prior.features[j].kind == 'relation' and
                isinstance(prior.relations[prior.features[j].index],AttrEqual) and f.var in
                (prior.relations[prior.features[j].index].left_var,prior.relations[prior.features[j].index].right_var) for j in chosen)
            if not contextual: chosen.remove(index)
        # Conditioning must not manufacture a head-implies-itself rule. This
        # concerns only the proposal prior: the unchanged broad branch retains
        # every tautological configuration and every original predicate axis.
        removed_head_literals = []
        for index in tuple(chosen):
            f = prior.features[index]
            tautology = (kind in ('add','remove') and f.kind == 'label'
                and f.var == target and f.label == planned['head.label'])
            if kind not in ('add','remove') and f.kind == 'relation':
                atom = prior.relations[f.index]
                left,right = planned['head.pair'].split('|')
                if isinstance(atom,LabelCompare) and atom.op == kind:
                    tautology = (atom.left_var,atom.right_var) == (left,right)
                    tautology |= kind == 'eq' and (atom.left_var,atom.right_var) == (right,left)
            if tautology:
                chosen.remove(index);removed_head_literals.append(f.include_key)
        if not chosen:
            return fallback('no substantive non-head-literal body at fit witness')
        active = {key:value for key,value in active.items()
                  if any(key.startswith(prior.features[i].key+'.') for i in chosen)}
        if not all(prior.holds(i,assignment,active) for i in chosen):
            raise RuntimeError('structured candidate lost its shared fit witness')
        result = {f.include_key:int(i in chosen) for i,f in enumerate(prior.features)}
        planned.update(active)
        self.last_diagnostic.update(branch='fit_witness_conditioned_TPE',
            conditional_parameters=planned, witness={v:prior.rows[r] for v,r in assignment.items()},
            source_variable=source, head_source_label_coupled=source_feature in chosen,
            head_label_fit_guided=source_feature is not None,
            head_fit_witness=head_witness, context_probes_by_variable=context_probe_receipts,
            relation_head_fit_witness=relation_head_witness,
            removed_conditioned_head_literals=removed_head_literals,
            joint_fit_body_witness_verified=True, selected_body_indicators=len(chosen))
        return result

    def sample_independent(self, study, trial, param_name, param_distribution):
        planned = (self.last_diagnostic or {}).get('conditional_parameters', {})
        if (self.last_diagnostic or {}).get('trial') == trial.number and param_name in planned:
            value=planned[param_name]
            if not param_distribution._contains(param_distribution.to_internal_repr(value)):
                raise ValueError('conditioned parameter escaped its original domain')
            return value
        return super().sample_independent(study,trial,param_name,param_distribution)


def _specialized_cluster_runner():
    """Isolate one import specialization; never patch original module globals."""
    source = textwrap.dedent(inspect.getsource(original.run_cluster_search))
    tree = ast.parse(source); count=0
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module == 'loris._engine.compact_storage':
            if {n.name for n in node.names} != {'STORAGE_MODE','FIXED_ATTR','ParameterCodec','CompactTrial','SamplerCheckpointStore'}:
                raise RuntimeError('upstream compact-storage import contract changed')
            node.module = 'loris._learning.structured_bo_storage';count+=1
    if count != 1: raise RuntimeError('expected exactly one isolated checkpoint import specialization')
    # Round-trip revert proves that the complete scoring, transaction, budget,
    # validation and resume AST has no other change.
    reverted=copy.deepcopy(tree)
    for node in ast.walk(reverted):
        if isinstance(node,ast.ImportFrom) and node.module == 'loris._learning.structured_bo_storage':
            node.module='loris._engine.compact_storage'
    if ast.dump(reverted,include_attributes=False) != ast.dump(ast.parse(source),include_attributes=False):
        raise RuntimeError('specialization changed more than the checkpoint import')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(tree), '<paper36_structured_checkpoint_import_only>', 'exec'),namespace)
    return namespace['run_cluster_search'], {
        'schema':'paper36_isolated_checkpoint_import_specialization_v1',
        'original_function_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'original_module':_source(original.__file__),
        'original_compact_module':_source(Path(original.__file__).with_name('compact_storage.py')),
        'specializer_source':_source(__file__),
        'adapter_source':_source(Path(__file__).with_name('structured_bo_storage.py')),
        'only_AST_change':'one compact-storage ImportFrom module; all imported API names preserved',
        'original_globals_mutated':False}


def run_structured_cluster_search(**kwargs):
    runner,binding=_specialized_cluster_runner()
    declared=kwargs.get('sampler_binding',{})
    if declared.get('execution_specialization') != binding:
        raise ValueError('structured cluster execution source binding differs')
    return runner(**kwargs)


def learn_structured_program(*, directory, spaces, witness_priors, evaluator, seed,
        budget_per_cluster, rule_parser, on_progress=None,
        storage_mode='compact_full_indicators_v1', broad_probability=DEFAULT_BROAD_PROBABILITY,
        sampler_options=None):
    spaces,priors=tuple(spaces),tuple(witness_priors)
    if (not spaces or len(spaces)!=len(priors) or len({s.cluster_id for s in spaces})!=len(spaces)
            or type(seed) is not int or type(budget_per_cluster) is not int or budget_per_cluster<1):
        raise ValueError('complete uniquely identified cluster spaces/priors and full positive budget required')
    if storage_mode not in ('dense_v1','compact_full_indicators_v1'): raise ValueError('unknown storage mode')
    _,execution=_specialized_cluster_runner()
    factories=[];bindings=[]
    for position,(space,prior) in enumerate(zip(spaces,priors)):
        prior.validate(space)
        families={f.include_key:f.kind for f in space.features}
        def factory(local_seed, prior=prior, families=families):
            return StructuredWitnessTPESampler(tuple(families),seed=local_seed,prior=prior,
                broad_probability=broad_probability,feature_families=families,**(sampler_options or {}))
        binding=factory(seed+1009*position).protocol
        binding=dict(binding,execution_specialization=execution)
        factories.append(factory);bindings.append(binding)
    directory=Path(directory)
    protocol={'schema':original.SCHEMA,'seed':seed,'budget_per_cluster':budget_per_cluster,
        'total_BO_budget':len(spaces)*budget_per_cluster,'initial_sigma':[],
        'historical_rules_or_banks_loaded':False,'evaluator':evaluator.binding,
        'spaces':[s.manifest() for s in spaces],'studies_per_cluster':1,
        'algorithm':'cluster BO complete, then sequential whole-program admission',
        'BO_prior':SCHEMA,'sampler_bindings':bindings,'persistence_mode':storage_mode,
        'upstream_scope':'explicit authenticated Pi/Si; proposal prior restricted to cluster source_fit',
        'seed_policy':'one experiment seed, deterministic cluster offset 1009 * cluster_position'}
    protocol_id=original.digest(protocol);original.immutable_json(directory/'protocol.json',protocol)
    selected=[];clusters=[]
    for position,space in enumerate(spaces):
        folder=directory/('cluster_%02d'%position)
        def progress(record):
            payload=dict(record,cluster_position=position,cluster_id=space.cluster_id,
                total_clusters=len(spaces),protocol_id=protocol_id)
            original.atomic_json(directory/'progress.json',payload)
            if on_progress:on_progress(payload)
        records=run_structured_cluster_search(directory=folder,space=space,evaluator=evaluator,
            seed=seed+1009*position,budget=budget_per_cluster,rule_parser=rule_parser,
            sampler_factory=factories[position],sampler_binding=bindings[position],
            on_progress=progress,storage_mode=storage_mode)
        selected=original.admit_cluster(directory=folder/'admission',records=records,
            selected=selected,evaluator=evaluator,rule_parser=rule_parser,on_progress=progress)
        clusters.append({'cluster_id':space.cluster_id,'complete_BO_trials':len(records),
                         'selected_after_cluster':len(selected)})
    result={'schema':original.SCHEMA,'protocol_id':protocol_id,'rules':[original.rule_record(r) for r in selected],
        'evaluation':evaluator.evaluate(selected),'clusters':clusters,
        'complete_BO_trials':len(spaces)*budget_per_cluster,'all_candidates_sequentially_evaluated':True,
        'historical_candidates_added':0,'proposal_policy':SCHEMA}
    original.immutable_json(directory/'SELECTED.json',result)
    original.immutable_json(directory/'COMPLETE.json',{'protocol_id':protocol_id,
        'selected_artifact_sha256':original.digest(result),'selected_rules':len(selected),
        'complete_BO_trials':len(spaces)*budget_per_cluster})
    original.atomic_json(directory/'progress.json', {'phase':'complete', 'protocol_id':protocol_id,
        'selected':len(selected), 'complete_BO_trials':len(spaces)*budget_per_cluster,
        'total_clusters':len(spaces), 'proposal_policy':SCHEMA})
    if on_progress: on_progress({'phase':'complete', 'selected':len(selected),
        'complete_BO_trials':len(spaces)*budget_per_cluster, 'protocol_id':protocol_id})
    return selected
