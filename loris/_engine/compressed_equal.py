"""Exact, opt-in dense equality closure with lazily reconstructed proofs.

This new diagnostic executor does not replace ``typed_runtime`` and never
silently falls back to an unbounded dense run. It certifies either:

* a single Eq rule whose two variables have independent frozen Doc/Text/ML
  domains; or
* a whole positive/non-strict program with an initially supported Eq rule
  whose independent unary domains connect every row and whose initial Gamma
  supplies every label. The resulting all-positive universal relation is a
  reachable fixpoint of the ENTIRE program, including all other rules; or
* the same reachable universal positive prefix in a general program: an
  actually enabled REMOVE/strict head supplies an immediate real conflict,
  while exhaustive scoped absence of such a trigger certifies the fixpoint.

No fact or candidate is removed. Each relation proof is a genuine rule
application or a two-edge inclusion path. Canonical proofs need not be the
first proofs chosen by the indexed/reference scheduler. Negative facts are
never contraposed. Unsupported instances raise before any dense execution.
"""
from collections.abc import Mapping
from collections import Counter
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from loris.chase import ChaseResult
from loris._engine.array_cache import file_sha, sha, _readonly
from loris._engine.rdl_reference import Doc, Label, Text, ML, LabelCompare, Rule, SEMANTICS
from loris._engine.hashing import array_hash
from loris._engine.relation_rows import RelationRows
from loris._engine.typed_runtime import (IndexedChaseRunner, ProofDAG, RelationFacts,
                                       _immutable, _plain, bits, indices)

SCHEMA = 'paper36_exact_compressed_independent_domain_equality_v1'


class UnsupportedDenseEquality(ValueError):
    """The finite certificate does not cover this instance; no Chase was run."""


class ExplicitProofExpansionRequired(ValueError):
    """Use the compact recipe or explicitly request a bounded expansion."""


def _packed(matrix):
    return {'shape': list(matrix.shape), 'bitorder': 'little',
            'packed_hex': np.packbits(matrix.reshape(-1), bitorder='little').tobytes().hex(),
            'array_sha256': array_hash(matrix)}


def _node(kind, row, column, origin, dependencies=(), rule_id=None,
          valuation=None, static=(), support=()):
    return _immutable({'kind': kind, 'row': int(row), 'column': int(column),
        'origin': origin, 'dependencies': list(dependencies), 'rule_id': rule_id,
        'valuation': dict(valuation or {}), 'static_evidence': list(static),
        'support_rule_ids': list(support)})


class EqualityProofDAG(ProofDAG):
    """Complete immutable Mapping; only requested proof nodes are constructed.

    Its identity commits to a reconstructible recipe and the exact frozen
    runner binding. ``iter_records`` explicitly streams every real fact;
    ``to_dict`` refuses accidental eager N-squared expansion.
    """
    def __init__(self, runner, rules, gamma, positive, relation, certificate):
        self._runner = runner
        self._rules = tuple(rules)
        self._gamma_pos, self._gamma_neg = _readonly(gamma.pos), _readonly(gamma.neg)
        self._positive = _readonly(positive)
        self._relation = relation
        self._certificate = _immutable(certificate)
        self._left, self._right = int(certificate['left_domain_hex'], 16), int(certificate['right_domain_hex'], 16)
        self._group = int(certificate['group_hex'], 16)
        self._rule_id = certificate['witness_rule_index']
        self._sources = tuple(next(indices(bits(gamma.pos[:, j]) & self._group), None)
                              for j in range(runner.k))
        self._count = len(relation) + int(positive.sum()) + int(gamma.neg.sum())
        self._descriptor = {'schema': SCHEMA, 'kind': 'complete_lazy_proof_recipe',
            'context_id': runner.context_id, 'runner_binding': runner.binding,
            'executor_source_sha256': file_sha(Path(__file__)),
            'program': [rule.to_dict() for rule in rules],
            'gamma_pos': _packed(gamma.pos), 'gamma_neg': _packed(gamma.neg),
            'gamma_subset': [], 'gamma_strict_subset': [],
            'certificate': _plain(self._certificate), 'count': self._count,
            'relation_identity': relation.identity(), 'positive_sha256': array_hash(positive),
            'negative_sha256': array_hash(gamma.neg),
            'canonical_proof_policy': 'direct_bipartite_edge_else_min_opposite_anchor_two_edges_v1'}
        self._identity_value = {'schema': SCHEMA, 'count': self._count,
                                'recipe_sha256': sha(self._descriptor)}
        self._identity_value['sha256'] = sha(self._identity_value)

    def __len__(self):
        return self._count

    def __iter__(self):
        for sign, matrix in [('pos', self._positive), ('neg', self._gamma_neg)]:
            for i, row in enumerate(matrix):
                for j in np.flatnonzero(row):
                    yield '%s:%d:%d' % (sign, i, j)
        for i, j in self._relation:
            yield 'subset:%d:%d' % (i, j)

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError(key)
        try:
            kind, a, b = key.split(':')
            i, j = int(a), int(b)
        except (ValueError, TypeError):
            raise KeyError(key) from None
        if key != '%s:%d:%d' % (kind, i, j) or not 0 <= i < self._runner.n:
            raise KeyError(key)
        if kind in ('pos', 'neg'):
            matrix = self._positive if kind == 'pos' else self._gamma_neg
            if not 0 <= j < self._runner.k or not matrix[i, j]:
                raise KeyError(key)
            seed = self._gamma_pos if kind == 'pos' else self._gamma_neg
            if seed[i, j]:
                return _node(kind, i, j, 'validated_Gamma')
            source = self._sources[j]
            if source is None:
                raise RuntimeError('derived positive has no real seed witness')
            return _node(kind, i, j, 'inclusion_label_copy',
                ('subset:%d:%d' % (source, i), 'pos:%d:%d' % (source, j)),
                support=(self._rule_id,))
        if kind != 'subset' or (i, j) not in self._relation:
            raise KeyError(key)
        if i == j:
            return _node(kind, i, j, 'identity')
        if self._left & (1 << i) and self._right & (1 << j):
            a, b = i, j
        elif self._left & (1 << j) and self._right & (1 << i):
            a, b = j, i
        else:
            opposite = self._right if self._left & (1 << i) else self._left
            anchor = next(indices(opposite))
            return _node(kind, i, j, 'inclusion_transitivity',
                ('subset:%d:%d' % (i, anchor), 'subset:%d:%d' % (anchor, j)),
                support=(self._rule_id,))
        rule = self._rules[self._rule_id]
        valuation = {rule.head.left_var: a, rule.head.right_var: b}
        dependencies, static = self._runner._evidence(rule, valuation)
        return _node(kind, i, j, 'rule_application', dependencies,
                     self._rule_id, valuation, static, (self._rule_id,))

    def identity(self):
        return copy.deepcopy(self._identity_value)

    def descriptor(self):
        return copy.deepcopy(self._descriptor)

    def to_dict(self):
        raise ExplicitProofExpansionRequired('Complete lazy proof: use descriptor()/iter_records(), or materialize(max_facts=...).')

    def iter_records(self):
        for key in self:
            yield key, _plain(self[key])

    def materialize(self, *, max_facts):
        if isinstance(max_facts, bool) or not isinstance(max_facts, int) or len(self) > max_facts:
            raise ExplicitProofExpansionRequired('Explicit maximum must cover all %d proof facts.' % len(self))
        return dict(self.iter_records())


class RootSourceProofOverlay(ProofDAG):
    """Phase roots over a complete proof without flattening relation proofs.

    The caller must supply every signed root. Identity facts stay identities;
    validated truth, weak corrections and revocable LLM sources remain distinct.
    This helper is not automatically installed into the frozen TypedSession.
    """
    def __init__(self, base, root_sources, *, phase):
        if not isinstance(base, EqualityProofDAG):
            raise TypeError('certified equality recipe required')
        if phase not in ('labeling', 'feedback'):
            raise ValueError('explicit labeling or feedback phase required')
        expected = {'%s:%d:%d' % (sign, i, j)
                    for sign, matrix in [('pos', base._gamma_pos), ('neg', base._gamma_neg)]
                    for i, j in np.argwhere(matrix)}
        if set(root_sources) != expected:
            raise ValueError('every signed root needs its exact source receipt')
        roots = {}
        for key, raw in root_sources.items():
            source = copy.deepcopy(raw)
            if not isinstance(source, dict) or not isinstance(source.get('origin'), str):
                raise ValueError('explicit root origin required')
            if phase == 'feedback':
                if (source['origin'] not in ('feedback_weak_root', 'accepted_weak_root')
                        or not isinstance(source.get('correction_id'), str) or not source['correction_id']):
                    raise ValueError('feedback roots must be weak corrections, never validated truth')
            elif source['origin'] not in ('validated_Gamma', 'ground_truth', 'human', 'llm_assumption'):
                raise ValueError('unsupported labeling root source')
            node = _plain(base[key])
            node.update(origin=source['origin'], root_source=source, logical_phase=phase)
            roots[key] = node
        self._roots, self._base = _immutable(roots), base
        self._phase = phase
        self._identity_value = {'schema': 'paper36_compressed_root_source_overlay_v1',
            'count': len(base), 'phase': phase, 'base_identity': base.identity(),
            'root_sources_sha256': sha(root_sources)}
        self._identity_value['sha256'] = sha(self._identity_value)

    def __len__(self): return len(self._base)
    def __iter__(self): return iter(self._base)
    def __getitem__(self, key):
        return self._roots[key] if key in self._roots else self._base[key]
    def identity(self): return copy.deepcopy(self._identity_value)
    def descriptor(self):
        return {'identity': self.identity(), 'base': self._base.descriptor(),
                'root_overrides': _plain(self._roots)}
    def to_dict(self):
        raise ExplicitProofExpansionRequired('Use descriptor() or explicitly stream iter_records().')
    def iter_records(self):
        for key in self:
            yield key, _plain(self[key])


class EqualityConflictProofDAG(ProofDAG):
    """A real full equality prefix plus the first chosen conflicting head.

    Chase halts on the demonstrated contradiction. Strict edges beyond the
    actual inserted head are not claimed to have reached a later fixpoint.
    """
    def __init__(self, base, derived, contradiction):
        super().__init__(derived, base)
        self._contradiction = _immutable(contradiction)

    def descriptor(self):
        return {'schema': SCHEMA, 'kind': 'complete_lazy_conflict_recipe',
            'context_id': self._base._runner.context_id,
            'base_prefix': self._base.descriptor(), 'derived': _plain(self._derived),
            'contradiction': _plain(self._contradiction), 'identity': self.identity()}

    def to_dict(self):
        raise ExplicitProofExpansionRequired('Use descriptor() or explicitly stream iter_records().')

    def iter_records(self):
        for key in self:
            yield key, _plain(self[key])


class CompressedEqualityRunner:
    """Read-only wrapper with explicit eligibility; never automatic fallback."""
    def __init__(self, runner):
        if not isinstance(runner, IndexedChaseRunner):
            raise TypeError('frozen IndexedChaseRunner required')
        self.runner = runner
        self._source_sha = file_sha(Path(__file__))

    def _certify(self, rules, gamma):
        if gamma.subset or gamma.strict_subset:
            raise UnsupportedDenseEquality('Explicit Gamma comparison roots require the general executor.')
        if np.any(gamma.pos & gamma.neg):
            raise UnsupportedDenseEquality('Initial signed conflict requires a conflict-prefix executor.')
        safe_program = all((isinstance(rule.head, Label) and rule.head.polarity == 'pos')
            or (isinstance(rule.head, LabelCompare) and rule.head.op in ('eq', 'subset'))
            for rule in rules)
        for rule_id, rule in enumerate(rules):
            head = rule.head
            if (not isinstance(head, LabelCompare) or head.op != 'eq'
                    or head.left_var == head.right_var
                    or set(rule.variables) != {head.left_var, head.right_var}
                    or any(not isinstance(atom, (Doc, Text, ML, Label)) for atom in rule.body)):
                continue
            prepared = self.runner._prepare(rule, self.runner._rule(rule)[1])
            domain = dict(prepared['domains'])
            for atom in rule.body:
                if isinstance(atom, Label):
                    source = gamma.pos if atom.polarity == 'pos' else gamma.neg
                    domain[atom.var] &= bits(source[:, self.runner.label_index[atom.label]])
            left, right = domain[head.left_var], domain[head.right_var]
            group = (left | right) if left and right else 0
            static_single = len(rules) == 1 and not any(isinstance(atom, Label) for atom in rule.body)
            total_positive = (not gamma.neg.any() and group == self.runner._all
                              and gamma.pos.any(axis=0).all())
            if not (static_single or total_positive):
                continue
            rows = tuple(indices(group))
            union = gamma.pos[list(rows)].any(axis=0) if rows else np.zeros(self.runner.k, bool)
            if rows and np.any(gamma.neg[list(rows)] & union):
                raise UnsupportedDenseEquality('Equality reaches a signed conflict; conflict-prefix proof is not covered.')
            scope = ('single_static_independent_unary_domains' if static_single else
                     'whole_program_universal_all_positive' if safe_program else
                     'whole_program_universal_positive_prefix_pending_destructive_check')
            return {'scope': scope,
                'witness_rule_index': rule_id, 'left_domain_hex': hex(left), 'right_domain_hex': hex(right),
                'group_hex': hex(group), 'group_rows': len(rows),
                'initial_body_support_only': True, 'all_program_heads_positive_or_non_strict': bool(safe_program),
                'all_labels_initially_supplied': bool(gamma.pos.any(axis=0).all()),
                'complete_fixpoint_theorem': 'complete_bipartite_Eq_connects_union_then_positive_union_v1',
                'full_program_rules': len(rules), 'no_rule_or_fact_removed': True}
        raise UnsupportedDenseEquality('No certified independent-domain Eq complete-fixpoint scope; no dense fallback was run.')

    def __call__(self, rules, gamma=None):
        runner = self.runner
        runner.verify_code()
        if runner.application_scope is not None:
            raise UnsupportedDenseEquality('Approved instance scopes require the full indexed executor.')
        if file_sha(Path(__file__)) != self._source_sha:
            raise RuntimeError('compressed executor source changed during a frozen run')
        rules = tuple(runner._rule(rule)[0] for rule in rules)
        # Validate every selected rule's complete static inputs even when the
        # saturation theorem will make its head redundant. Missing model tables
        # or undeclared attributes cannot be hidden by an already-known head.
        for rule in rules:
            runner._prepare(rule, runner._rule(rule)[1])
        gamma = runner.gamma if gamma is None else runner._gamma(gamma)
        certificate = self._certify(rules, gamma)
        group = int(certificate['group_hex'], 16)
        rows = tuple(indices(group))
        positive = gamma.pos.copy()
        if rows:
            positive[list(rows)] |= gamma.pos[list(rows)].any(axis=0)
        relation_rows = RelationRows(runner.n, reflexive=True)
        for i in rows:
            relation_rows[i] = group
        relation = RelationFacts(relation_rows)
        strict = RelationFacts(RelationRows(runner.n))
        destructive = None
        if certificate['scope'] == 'whole_program_universal_positive_prefix_pending_destructive_check':
            state = SimpleNamespace(out=relation.rows, inc=relation.rows,
                proper=strict.rows, proper_inc=strict.rows,
                label_bits={'pos': [runner._all] * runner.k, 'neg': [0] * runner.k})
            for rule_id, rule in enumerate(rules):
                head = rule.head
                if not ((isinstance(head, Label) and head.polarity == 'neg')
                        or (isinstance(head, LabelCompare) and head.op == 'strict_subset')):
                    continue
                valuation = next(runner._valuations(rule, runner._prepare(rule, runner._rule(rule)[1]), state), None)
                if valuation is not None:
                    destructive = (rule_id, rule, valuation)
                    break
            certificate['destructive_heads_checked_with_full_scoped_body'] = True
            certificate['scope'] = ('whole_program_reachable_universal_positive_prefix_then_conflict'
                if destructive is not None else 'whole_program_universal_all_positive_no_destructive_trigger')
            certificate['destructive_trigger'] = ({'rule_index': destructive[0], 'valuation': destructive[2]}
                                                  if destructive is not None else None)
            if destructive is not None:
                certificate['complete_fixpoint_theorem'] = None
                certificate['conflict_prefix_theorem'] = 'actual_Eq_saturation_then_one_actual_conflicting_head_v1'
        proof = EqualityProofDAG(runner, rules, gamma, positive, relation, certificate)
        provenance = {(int(i), runner.labels[int(j)]): [certificate['witness_rule_index']]
                      for i, j in np.argwhere(positive & ~gamma.pos)}
        changed = len(relation) > runner.n or np.any(positive != gamma.pos)
        result = ChaseResult(status='fixpoint', predictions=positive.astype(np.float32), conflicts=[],
            n_rounds=2 if changed else 1, provenance=provenance,
            positive_facts=_readonly(positive), negative_facts=_readonly(gamma.neg))
        result.subset_facts, result.strict_subset_facts, result.proof_dag = relation, strict, proof
        result.comparison_conflicts = []
        result.reference_audit = {'semantics': SEMANTICS, 'executor': SCHEMA, 'context_id': runner.context_id,
            'full_rows': runner.n, 'label_names': list(runner.labels), 'subset_direction': runner.subset_direction,
            'subset_direction_pending_author_confirmation': False,
            'subset_direction_author_confirmed': runner.subset_direction == 'standard',
            'subset_direction_legacy_compatibility': runner.subset_direction == 'paper_p7_reverse',
            'complete_without_cutoff': True,
            'program_sha256': sha([rule.to_dict() for rule in rules]), 'certificate': certificate,
            'comparison_premises_from_partial_bitmaps': False, 'document_lbl_used_as_Gamma': False,
            'model_outputs_unioned_into_Gamma': False, 'negative_contraposition_added': False,
            'finite_comparison_solver_calls': 0, 'relation_fact_representation': 'complete_immutable_bitset_rows',
            'proof_representation': SCHEMA, 'rounds_and_first_witness_not_claimed_identical_to_reference': True,
            'n_rounds_meaning': 'certified_Eq_saturation_then_full_program_fixpoint_check',
            'automatic_global_runtime_replacement': False}
        if destructive is not None:
            rule_id, rule, valuation = destructive
            dependencies, static = runner._evidence(rule, valuation)
            support = {rule_id}
            for dependency in dependencies:
                support.update(proof[dependency]['support_rule_ids'])
            head = rule.head
            if isinstance(head, Label):
                kind, row, col = 'neg', valuation[head.var], runner.label_index[head.label]
                negative = gamma.neg.copy()
                negative[row, col] = True
                result.negative_facts = _readonly(negative)
                result.predictions = (positive & ~negative).astype(np.float32)
                result.conflicts = [(row, runner.labels[col])]
                contradiction = {'kind': 'positive_negative_label_conflict',
                    'positive_proof': 'pos:%d:%d' % (row, col), 'negative_proof': 'neg:%d:%d' % (row, col)}
                target = (row, runner.labels[col])
                result.provenance[target] = sorted(set(result.provenance.get(target, ())) | support)
            else:
                kind = 'strict_subset'
                row, col = runner._edge(head, valuation)
                proper_rows = [0] * runner.n
                proper_rows[row] = 1 << col
                result.strict_subset_facts = RelationFacts(proper_rows)
                contradiction = {'kind': 'strict_inclusion_cycle', 'source': row, 'target': col,
                    'strict_proof': 'strict_subset:%d:%d' % (row, col),
                    'back_inclusion_proof': 'subset:%d:%d' % (col, row)}
                result.comparison_conflicts = [contradiction]
            key = '%s:%d:%d' % (kind, row, col)
            derived = {key: _node(kind, row, col, 'rule_application', dependencies,
                rule_id, valuation, static, sorted(support))}
            result.proof_dag = EqualityConflictProofDAG(proof, derived, contradiction)
            result.status = 'conflict'
            result.reference_audit.update(conflict_is_actual_reachable_prefix=True,
                conflict_fact_sets_not_claimed_identical_to_other_schedulers=True,
                closure_after_contradiction_not_claimed=True,
                n_rounds_meaning='certified_Eq_saturation_then_actual_conflicting_rule_head')
        return result

    run = __call__


def certified_inverse_rows(result):
    """Exact no-edge-loop incoming relations for this certified symmetric state.

    This is an explicit upper-layer adapter point, not a global claim that
    subset relations are symmetric. Replacing relation fields is detected.
    """
    proof = result.proof_dag
    base = proof._base if isinstance(proof, RootSourceProofOverlay) else proof
    if not isinstance(base, EqualityProofDAG) or result.subset_facts is not base._relation:
        raise TypeError('authentic complete equality result required')
    if result.strict_subset_facts.n != base._runner.n or len(result.strict_subset_facts):
        raise ValueError('certified equality state has no proper relations')
    return result.subset_facts.rows, result.strict_subset_facts.rows


def restore_recipe(runner, descriptor):
    """Rebuild the complete lazy state, recompute the certificate, verify bytes.

    Original frozen corpus/model context is required; a hash does not substitute
    for missing model tables. No expanded relation/proof JSON is needed.
    """
    if descriptor.get('context_id') != runner.context_id or descriptor.get('schema') != SCHEMA:
        raise ValueError('recipe requires its exact frozen corpus/model context')
    target_descriptor = descriptor
    if descriptor.get('kind') == 'complete_lazy_conflict_recipe':
        descriptor = descriptor['base_prefix']
    from loris._engine.rdl_reference import RefGamma
    matrices = []
    for name in ('gamma_pos', 'gamma_neg'):
        packed = descriptor[name]
        if packed.get('shape') != [runner.n, runner.k] or packed.get('bitorder') != 'little':
            raise ValueError('recipe Gamma axes differ')
        data = np.frombuffer(bytes.fromhex(packed['packed_hex']), dtype=np.uint8)
        matrix = np.unpackbits(data, bitorder='little')[:runner.n * runner.k].reshape(runner.n, runner.k).astype(bool)
        if array_hash(matrix) != packed['array_sha256']:
            raise ValueError('recipe Gamma hash differs')
        matrices.append(matrix)
    result = CompressedEqualityRunner(runner)(tuple(Rule.from_dict(rule) for rule in descriptor['program']),
                                             RefGamma(*matrices))
    if result.proof_dag.descriptor() != target_descriptor:
        raise ValueError('recipe is not the exact recomputed complete-proof certificate')
    return result


def save_compressed_result(result, directory):
    """Content-addressed complete proof recipe; never enumerate relation facts."""
    if not isinstance(result.proof_dag, (EqualityProofDAG, EqualityConflictProofDAG)):
        raise TypeError('unmodified certified compressed result required')
    proof = result.proof_dag
    payload = {'schema': 'paper36_recoverable_compressed_result_v1', 'status': 'COMPLETE',
        'proof_identity': proof.identity(), 'proof_recipe': proof.descriptor(),
        'reference_audit': copy.deepcopy(result.reference_audit),
        'restoration': 'restore_recipe(exact_original_IndexedChaseRunner, proof_recipe)'}
    payload['payload_sha256'] = sha(payload)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (proof.identity()['sha256'] + '.json')
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError('different content already occupies this exact proof identity')
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix='.recipe-', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return {'path': str(path.resolve()), 'sha256': file_sha(path),
            'proof_identity': proof.identity(), 'complete_fact_count': len(proof)}


class ExactEqualityDispatch:
    """Optional BO callable: certified compression, otherwise unchanged runner.

    This must be explicitly installed by its caller. Unsupported rules are
    neither deleted nor resampled. Every general fallback is durably logged
    BEFORE execution, including its theoretical dense-relation exposure.
    The existing TwoFilterEvaluator already consumes compact ``identity()``
    methods, so no score, candidate or greedy-selection change is required.
    """
    def __init__(self, runner, *, artifact_directory, on_event=None):
        self.runner = runner
        self.compressed = CompressedEqualityRunner(runner)
        self.directory = Path(artifact_directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.on_event = on_event
        self.counts = Counter()
        self._binding_value = {'schema': 'paper36_exact_equality_dispatch_v1',
            'runtime': runner.binding, 'source_sha256': file_sha(Path(__file__)),
            'compressed_scope': SCHEMA, 'fallback': 'unchanged_complete_IndexedChaseRunner',
            'no_rule_filtering_resampling_or_cutoff': True,
            'compressed_proofs_saved_before_return': True}

    @property
    def binding(self):
        return copy.deepcopy(self._binding_value)

    def _event(self, value):
        record = dict(value, monotonic_seconds=time.monotonic())
        with (self.directory / 'dispatch.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if self.on_event is not None:
            self.on_event(copy.deepcopy(record))

    def __call__(self, rules, gamma=None):
        rules = tuple(rules)
        gamma = self.runner.gamma if gamma is None else self.runner._gamma(gamma)
        key = self.runner.program_key(rules, gamma)
        started = time.monotonic()
        try:
            result = self.compressed(rules, gamma)
        except UnsupportedDenseEquality as error:
            self.counts['general_fallback_calls'] += 1
            has_comparison = any(isinstance(rule.head, LabelCompare) for rule in rules)
            self._event({'stage': 'general_fallback_before_execution', 'program_key': key,
                'program_sha256': sha([rule.to_dict() for rule in rules]), 'n_rules': len(rules),
                'n_rows': self.runner.n, 'reason': str(error),
                'comparison_head_present': has_comparison,
                'possible_subset_fact_upper_bound': self.runner.n ** 2 if has_comparison else self.runner.n,
                'bound_is_risk_description_not_measured_outcome': True,
                'candidate_dropped': False, 'cutoff_added': False})
            result = self.runner(rules, gamma)
            self._event({'stage': 'general_fallback_complete', 'program_key': key,
                'elapsed_seconds': time.monotonic() - started, 'status': result.status,
                'subset_count': len(result.subset_facts), 'proof_count': len(result.proof_dag)})
        else:
            receipt = save_compressed_result(result, self.directory / 'complete_proof_recipes')
            self.counts['certified_compressed_calls'] += 1
            self._event({'stage': 'compressed_complete', 'program_key': key,
                'elapsed_seconds': time.monotonic() - started, 'certificate': result.reference_audit['certificate'],
                'subset_count': len(result.subset_facts), 'proof_count': len(result.proof_dag),
                'recoverable_result': receipt})
        return result

    run = __call__

    def audit(self):
        return {'binding': self.binding, 'counts': dict(self.counts),
                'dispatch_log': str(self.directory / 'dispatch.jsonl'), 'base_runner': self.runner.audit()}
