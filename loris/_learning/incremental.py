"""Event-driven monotone maintenance of the existing finite signed RDL chase.

Inserted facts activate only valuations whose body contains that new fact.
Negative facts are explicit monotone atoms, not negation-as-failure. Retractions,
root provenance changes, rule/model changes, and conflicted states rebuild from
the retained roots; old ML consequences are never recycled into Gamma. This is
an exact addition algorithm plus explicit full-rebuild deletion path, not DRed.
"""
from collections import ChainMap, deque
import copy
import hashlib
from pathlib import Path

import numpy as np

from loris.chase import ChaseResult
from loris._engine.rdl_reference import AttrEqual, Label, LabelCompare, RefGamma
from loris._engine.typed_runtime import _State, ProofDAG, RelationFacts, bits, indices
from loris._engine.relation_rows import RelationRows
from loris._engine.typed_session import TypedChaseSession, _clone, digest, seed_identity, complete_state_identity


class EventState(_State):
    def __init__(self, runner, gamma):
        self.events = []
        super().__init__(runner, gamma)

    def set_label(self, sign, i, j, *args, **kwargs):
        changed = super().set_label(sign, i, j, *args, **kwargs)
        if changed:
            self.events.append((sign, int(i), int(j)))
        return changed

    def _install(self, kind, source, target, *args, **kwargs):
        changed = super()._install(kind, source, target, *args, **kwargs)
        if changed:
            self.events.append((kind, int(source), int(target)))
        return changed


def state_from_result(runner, result):
    state = EventState(runner, RefGamma.empty(runner.n, runner.k))
    state.pos, state.neg = result.positive_facts.copy(), result.negative_facts.copy()
    state.label_bits = {sign: [bits(matrix[:, j]) for j in range(runner.k)]
                        for sign, matrix in [('pos', state.pos), ('neg', state.neg)]}
    state.out = RelationRows.from_rows(result.subset_facts.rows)
    state.proper = RelationRows.from_rows(result.strict_subset_facts.rows)
    state.inc, state.proper_inc = state.out.transpose(), state.proper.transpose()
    state.base_proof, state.proof = result.proof_dag, {}
    state.provenance = copy.deepcopy(result.provenance)
    state.events.clear()
    return state


def event_bindings(atom, event, runner):
    kind, a, b = event
    if isinstance(atom, Label):
        if kind == atom.polarity and b == runner.label_index[atom.label]:
            return ({atom.var: a},)
    elif isinstance(atom, LabelCompare):
        expected = 'strict_subset' if atom.op == 'strict_subset' else 'subset'
        if kind == expected:
            if atom.op != 'eq' and runner.subset_direction != 'standard':
                a, b = b, a
            pairs = ((a, b), (b, a)) if atom.op == 'eq' else ((a, b),)
            return tuple({atom.left_var: x, atom.right_var: y} for x, y in pairs
                         if atom.left_var != atom.right_var or x == y)
    return ()


def valuations(runner, rule, prepared, state, fixed=None, allowed_rows=None, known_head=False,
               rule_index=None, frozen_state=False, round_cache=None, multi_attribute_join=True):
    """Complete original conjunctive valuation, optionally anchored to a delta.

    ``frozen_state=True`` promises that ``state`` is not modified while the
    generator is consumed (a round snapshot). A Label head already known in
    that state is then pruned exactly when its variable is bound, so the
    repeated leaf tests are skipped; a single-variable Label-head rule without
    binary atoms, and a two-variable Label-head rule whose only binary atom is
    one or more AttrEqual atoms between them, are enumerated without per-row big-integer
    joins (value buckets). The yielded valuations, their order and the
    application-scope checks are identical (2026-09-30 optimization).
    ``round_cache`` (a dict owned by the caller for one frozen state and one
    known_head/allowed_rows setting) shares those buckets across anchored jobs.
    """
    assignment = {}; fixed = fixed or {}
    # Static attribute equality can use a later variable's event anchor before
    # that variable is visited.  This only removes branches which the unchanged
    # later fixed-domain check would reject; valuation order stays unchanged.
    # Do not apply this to mutable label-set relations: a yielded valuation can
    # change those relations while the caller resumes this generator.
    # An out-of-axis anchor formerly produced an empty later domain; do not
    # turn that into an early attribute-array indexing error.
    attribute_fixed = {var: row for var, row in fixed.items()
                       if isinstance(row, (int, np.integer)) and 0 <= row < runner.n}
    attribute_assignment = ChainMap(assignment, attribute_fixed) if attribute_fixed else assignment
    variables = rule.variables
    head = rule.head
    # Prepared domains are subsets of the complete row axis runner._all.
    allowed = None if allowed_rows is None else sum(1 << int(i) for i in allowed_rows)
    label_head = isinstance(head, Label)

    def base_domain(var, anchored=True):
        domain = prepared['domains'][var]
        if allowed is not None:
            domain &= allowed
        if anchored and var in fixed:
            domain &= 1 << fixed[var]
        for atom in prepared['unary'][var]:
            domain &= state.label_bits[atom.polarity][runner.label_index[atom.label]]
        if not known_head and head.var == var:
            domain &= ~state.label_bits[head.polarity][runner.label_index[head.label]]
        return domain

    if frozen_state and label_head and len(variables) == 1 and not prepared['binary']:
        var = variables[0]
        for row in indices(base_domain(var)):
            assignment[var] = row
            if runner.application_allowed(rule_index, rule, assignment):
                yield dict(assignment)
        return
    binary = prepared['binary']
    if (frozen_state and label_head and len(variables) == 2 and binary
            and (len(binary) == 1 or multi_attribute_join)
            and all(type(atom) is AttrEqual
                    and {atom.left_var, atom.right_var} == set(variables) for atom in binary)):
        key = ('attribute_join', rule_index)
        join = round_cache.get(key) if round_cache is not None and rule_index is not None else None
        if join is None:
            join_type = _AttributeJoin if len(binary) == 1 else _MultiAttributeJoin
            join = join_type(runner, variables, binary[0] if len(binary) == 1 else binary,
                             base_domain(variables[0], False), base_domain(variables[1], False))
            if round_cache is not None and rule_index is not None:
                round_cache[key] = join
        yield from join.valuations(runner, rule, rule_index, fixed)
        return
    skip_leaf_label_test = frozen_state and label_head
    def visit(depth):
        if depth == len(variables):
            if not runner.application_allowed(rule_index, rule, assignment):
                return
            if not known_head and not skip_leaf_label_test:
                if isinstance(head, Label):
                    if state.label_bits[head.polarity][runner.label_index[head.label]] & (1 << assignment[head.var]):
                        return
                elif runner._relation_domain(head, head.left_var, assignment, state) & (1 << assignment[head.left_var]):
                    return
            yield dict(assignment)
            return
        var = variables[depth]
        domain = prepared['domains'][var]
        if allowed is not None:
            domain &= allowed
        if var in fixed:
            domain &= 1 << fixed[var]
        for atom in prepared['unary'][var]:
            domain &= state.label_bits[atom.polarity][runner.label_index[atom.label]]
        for atom in prepared['binary']:
            if var in (atom.left_var, atom.right_var):
                bound = attribute_assignment if isinstance(atom, AttrEqual) else assignment
                domain &= runner._relation_domain(atom, var, bound, state)
        # Prune already-present heads before expanding the remaining join.
        # This is exactly the leaf test above, pushed to the first point where
        # its bound variables make it decidable. Calibration/source collection
        # explicitly uses known_head=True and must still enumerate all support.
        if not known_head:
            if isinstance(head, Label):
                known = state.label_bits[head.polarity][runner.label_index[head.label]]
                if head.var == var:
                    domain &= ~known
                elif (not skip_leaf_label_test and head.var in assignment
                      and known & (1 << assignment[head.var])):
                    # A frozen state already removed known heads when head.var
                    # was bound, so this test cannot succeed there.
                    return
            elif var in (head.left_var, head.right_var):
                other = head.right_var if var == head.left_var else head.left_var
                if other == var or other in assignment:
                    domain &= ~runner._relation_domain(head, var, assignment, state)
        for row in indices(domain):
            assignment[var] = row
            yield from visit(depth + 1)
        assignment.pop(var, None)
    yield from visit(0)


class _AttributeJoin:
    """Valuations of a two-variable rule joined by one AttrEqual, bucketed by value.

    Built from both variables' domains without anchors; ``valuations`` then
    yields exactly the nested loop's sequence for any anchor: first variable
    ascending over its (anchored) domain, second ascending over its domain &
    runner._relation_domain(...), i.e. rows whose attribute code on the
    second's side equals the first row's code on the other side (an absent
    attribute, code -1, matches nothing).
    """

    def __init__(self, runner, variables, atom, first_domain, second_domain):
        self.first, self.second = variables
        codes = runner._attribute_codes(atom.left_attr, atom.right_attr)
        side = 0 if self.second == atom.left_var else 1
        self.first_codes, self.second_codes = codes[1 - side], codes[side]
        self.first_domain, self.second_domain = first_domain, second_domain
        self._first_mask = self._second_mask = self._buckets = self._first_members = None
        self._full_first = None

    def first_mask(self):
        if self._first_mask is None:
            self._first_mask = _row_mask(self.first_domain, len(self.first_codes))
        return self._first_mask

    def second_mask(self):
        if self._second_mask is None:
            self._second_mask = _row_mask(self.second_domain, len(self.second_codes))
        return self._second_mask

    def buckets(self):
        """code -> ascending second rows of the unanchored second domain."""
        if self._buckets is None:
            rows = np.fromiter(indices(self.second_domain), np.int64)
            self._buckets = _group(rows[self.second_codes[rows] >= 0], self.second_codes)
        return self._buckets

    def first_members(self):
        """code -> ascending rows of the unanchored first domain."""
        if self._first_members is None:
            rows = np.fromiter(indices(self.first_domain), np.int64)
            self._first_members = _group(rows[self.first_codes[rows] >= 0], self.first_codes)
        return self._first_members

    def valuations(self, runner, rule, rule_index, fixed):
        first, second = self.first, self.second
        if not self.first_domain or not self.second_domain:
            return
        if second in fixed:
            y = int(fixed[second])
            if not self.second_mask()[y] or self.second_codes[y] < 0:
                return
            pairs_for = lambda x, ys=(y,): ys
            firsts = self.first_members().get(int(self.second_codes[y]), ())
            if first in fixed:
                firsts = [x for x in firsts if x == fixed[first]]
        else:
            buckets = self.buckets()
            pairs_for = lambda x: buckets[int(self.first_codes[x])]
            if first in fixed:
                x = int(fixed[first])
                firsts = [x] if self.first_mask()[x] and int(self.first_codes[x]) in buckets else ()
            else:
                if self._full_first is None:
                    members = self.first_members()
                    self._full_first = sorted(x for code, rows in members.items() if code in buckets
                                              for x in rows)
                firsts = self._full_first
        scoped = getattr(runner, '_application_scope', True) is not None
        for x in firsts:
            for y in pairs_for(x):
                valuation = {first: x, second: y}
                if scoped and not runner.application_allowed(rule_index, rule, valuation):
                    continue
                yield valuation


class _MultiAttributeJoin(_AttributeJoin):
    """Conjunctive attribute equality, with the same ordered join as one atom.

    Codes from different attribute pairs are never compared to each other:
    their oriented tuple is the join key. A missing component rejects that
    row on its side, including when the other row is missing it as well.
    Joint codes are formed over both full axes before domain filtering, so
    anchors and asymmetric source/receiver domains cannot change equality.
    """

    def __init__(self, runner, variables, atoms, first_domain, second_domain):
        super().__init__(runner, variables, atoms[0], first_domain, second_domain)
        first, second = [], []
        for atom in atoms:
            codes = runner._attribute_codes(atom.left_attr, atom.right_attr)
            side = 0 if self.second == atom.left_var else 1
            first.append(codes[1 - side]); second.append(codes[side])
        rows = np.concatenate((np.column_stack(first), np.column_stack(second)))
        valid = np.all(rows >= 0, axis=1)
        joint = np.full(len(rows), -1, dtype=np.int64)
        if np.any(valid):
            _, joint[valid] = np.unique(rows[valid], axis=0, return_inverse=True)
        self.first_codes = joint[:runner.n]
        self.second_codes = joint[runner.n:]


def _row_mask(domain, n):
    mask = np.zeros(n, bool)
    mask[np.fromiter(indices(domain), np.int64)] = True
    return mask


def _group(rows, codes):
    """code -> ascending Python-int rows (rows ascend already)."""
    if not len(rows):
        return {}
    order = np.argsort(codes[rows], kind='stable')
    sorted_codes = codes[rows][order]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    ends = np.r_[starts[1:], len(sorted_codes)]
    return {int(sorted_codes[a]): rows[order[a:b]].tolist() for a, b in zip(starts, ends)}


def apply_head(runner, rule, ri, valuation, state):
    if not runner.application_allowed(ri, rule, valuation):
        return False
    dependencies, static = runner._evidence(rule, valuation)
    head = rule.head
    if isinstance(head, Label):
        return state.set_label(head.polarity, valuation[head.var], runner.label_index[head.label],
            'rule_application', dependencies, ri, valuation, static)
    a, b = runner._edge(head, valuation)
    kind = 'strict_subset' if head.op == 'strict_subset' else 'subset'
    edges = [(a, b), (b, a)] if head.op == 'eq' else [(a, b)]
    for x, y in edges:
        state.remember(kind, x, y, 'rule_application', dependencies, ri, valuation, static)
    changed = False
    for x, y in edges:
        changed |= state.add_edge(x, y, head.op == 'strict_subset')
    return changed


def validate_static_context_prefix(original, candidate):
    """Certify every non-probabilistic premise on the original document axis.

    Candidate may append evidence rows. The caller separately checks its axes
    and probability tables. Static ML decisions are a distinct interpretation
    and are never implicitly changed by a probability-only theta refinement.
    """
    n = original.n
    if (tuple((d.cnt, d.ttl, d.mtd) for d in original.docs)
            != tuple((d.cnt, d.ttl, d.mtd) for d in candidate.docs[:n])):
        raise ValueError('frozen document cnt/ttl/mtd prefix changed')
    if original.subset_direction != candidate.subset_direction:
        raise ValueError('frozen comparison semantics changed')
    if (set(original.attribute_values) != set(candidate.attribute_values)
            or any(values != candidate.attribute_values[attr][:n]
                   for attr, values in original.attribute_values.items())):
        raise ValueError('frozen attribute values prefix changed')
    for name, label in [('presence', 'attribute presence'), ('decisions', 'static model decisions')]:
        before, after = getattr(original, name), getattr(candidate, name)
        if (set(before) != set(after)
                or any(not np.array_equal(value, after[key][:n]) for key, value in before.items())):
            raise ValueError('frozen ' + label + ' prefix changed')


class IncrementalChase:
    def __init__(self, session):
        if not isinstance(session, TypedChaseSession):
            raise TypeError('TypedChaseSession required')
        self.session = session
        self._result = self._seed = self._sources = self._state = None
        self.audit = []
        self._bind()

    def _bind(self):
        from .rdg import RuleDependencyGraph
        from loris._search_space.configuration import validate_rule_guard
        for rule in self.session.rules:
            validate_rule_guard(rule)
        # The syntax graph depends only on the fixed Sigma. Model updates need
        # fresh static masks, but do not rebuild the paper's reusable RDG.
        if not hasattr(self, 'rdg'):
            self.rdg = RuleDependencyGraph(self.session.rules)
        self.binding = digest(self.session.binding)
        self.prepared = [self.session.runner._prepare(*self.session.runner._rule(rule)) for rule in self.session.rules]

    def replace_session(self, session):
        if not isinstance(session, TypedChaseSession):
            raise TypeError('TypedChaseSession required')
        if session.shape != self.session.shape or session.labels != self.session.labels:
            raise ValueError('replacement session axes differ')
        old = self.session
        if old.program_id != session.program_id:
            raise ValueError('fixed Sigma changed during model refinement')
        validate_static_context_prefix(old.runner, session.runner)
        if set(old.runner.probabilities) != set(session.runner.probabilities):
            raise ValueError('model refinement cannot change model aliases')
        changed_scores = any(not np.array_equal(value,session.runner.probabilities[alias])
                             for alias,value in old.runner.probabilities.items())
        if changed_scores and session.snapshot_id == old.snapshot_id:
            raise ValueError('new probability interpretation requires a new immutable theta identifier')
        self.session = session
        self._result = self._seed = self._sources = self._state = None
        self._bind()
    rebind = replace_session

    def fork(self):
        """Isolate candidate facts while retaining the same frozen Sigma/theta."""
        candidate = object.__new__(IncrementalChase)
        candidate.__dict__ = dict(self.__dict__)
        candidate.audit = copy.deepcopy(self.audit)
        candidate._result = _clone(self._result) if self._result is not None else None
        candidate._seed = self.session.runner._gamma(self._seed) if self._seed is not None else None
        candidate._sources = copy.deepcopy(self._sources)
        candidate._state = (state_from_result(self.session.runner, candidate._result)
            if candidate._result is not None and candidate._result.status == 'fixpoint' else None)
        candidate._parent_state = (complete_state_identity(self._result)['sha256']
            if self._result is not None else None)
        return candidate

    def commit(self, candidate):
        """Commit only a complete successful trial against the unchanged parent."""
        current = complete_state_identity(self._result)['sha256'] if self._result is not None else None
        if (not isinstance(candidate, IncrementalChase) or candidate.session is not self.session
                or candidate.binding != self.binding or candidate._parent_state != current):
            raise ValueError('candidate Chase belongs to another or changed parent state')
        if candidate._result is None or candidate._result.status != 'fixpoint':
            raise ValueError('only a complete consistent candidate Chase may commit')
        installed = candidate.fork()
        self._seed, self._sources = installed._seed, installed._sources
        self._result, self._state, self.audit = installed._result, installed._state, installed.audit
        return self.result

    def _fresh(self, seed, sources, reason):
        result = self.session.chase(seed, phase='labeling', root_sources=sources)
        self._state = state_from_result(self.session.runner, result) if result.status == 'fixpoint' else None
        self.audit.append(dict(mode='full_rebuild', reason=reason, theta=self.session.snapshot_id,
                               seed=digest(seed_identity(seed))))
        return result

    def chase(self, seed, root_sources=None):
        session, runner = self.session, self.session.runner
        if (digest(session.binding) != self.binding or session.snapshot_id != session._snapshot_id
                or session.model_snapshot_id != session._snapshot_id
                or digest([r.to_dict() for r in session.rules]) != session.program_id):
            raise RuntimeError('session binding changed; use replace_session')
        runner.verify_code()
        seed = runner._gamma(seed)
        sources = session._sources(seed, 'labeling', root_sources)
        reason = None
        if self._state is None:
            reason = 'initial_or_previous_conflict_or_model_change'
        elif any(np.any(old & ~new) for old, new in [(self._seed.pos, seed.pos), (self._seed.neg, seed.neg)]):
            reason = 'root_retraction'
        elif not set(self._seed.subset) <= set(seed.subset) or not set(self._seed.strict_subset) <= set(seed.strict_subset):
            reason = 'relation_root_retraction'
        elif any(sources.get(k) != v for k, v in self._sources.items()):
            reason = 'root_provenance_changed'
        elif any(k in self._result.proof_dag for k in set(sources) - set(self._sources)):
            reason = 'derived_fact_promoted_to_explicit_root'
        if reason:
            result = self._fresh(seed, sources, reason)
        else:
            state = self._state
            new_roots = set(sources) - set(self._sources)
            for key in sorted(new_roots):
                kind, a, b = key.split(':'); a, b = int(a), int(b)
                if kind in ('pos', 'neg'):
                    state.set_label(kind, a, b, sources[key]['origin'])
                else:
                    state.remember(kind, a, b, sources[key]['origin'])
                    state.add_edge(a, b, kind == 'strict_subset')
                if key in state.proof:
                    state.proof[key].update(root_source=copy.deepcopy(sources[key]), logical_phase='labeling')
            queue = deque(state.events); state.events.clear()
            visited = set(); fired = 0; seen_events = len(queue)
            conflict = state.invalid()
            while queue and not conflict:
                event = queue.popleft()
                for ri, atom in self.rdg.consumers(event, runner):
                    rule = session.rules[ri]
                    for fixed in event_bindings(atom, event, runner):
                        for valuation in valuations(runner, rule, self.prepared[ri], state, fixed, rule_index=ri):
                            key = (ri, tuple(valuation[v] for v in rule.variables))
                            if key in visited:
                                continue
                            visited.add(key)
                            if apply_head(runner, rule, ri, valuation, state):
                                fired += 1
                            queue.extend(state.events); seen_events += len(state.events); state.events.clear()
                            if state.invalid():
                                conflict = True; break
                        if conflict: break
                    if conflict: break
            if conflict:
                result = self._fresh(seed, sources, 'incremental_conflict_canonical_fresh_diagnostic')
            else:
                result = ChaseResult(status='fixpoint', predictions=(state.pos & ~state.neg).astype(np.float32),
                    conflicts=[], n_rounds=0, provenance=copy.deepcopy(state.provenance),
                    positive_facts=state.pos.copy(), negative_facts=state.neg.copy())
                result.subset_facts, result.strict_subset_facts = RelationFacts(state.out), RelationFacts(state.proper)
                result.proof_dag = ProofDAG(state.proof, state.base_proof)
                result.comparison_conflicts = []
                result.reference_audit = dict(self._result.reference_audit,
                    complete_without_cutoff=True, executor='paper36_monotone_fact_delta_v1',
                    fresh_rounds_or_first_proof_identity_claimed=False)
                result.typed_session_audit = dict(self._result.typed_session_audit,
                    schema='paper36_monotone_fact_delta_v1', phase='labeling',
                    theta=session.snapshot_id, Sigma=session.program_id, context_id=session.context_id,
                    root_sources=copy.deepcopy(sources), root_sources_sha256=digest(sources),
                    seed_sha256=digest(seed_identity(seed)), execution_kind='delta_incremental',
                    full_chase_call_performed_for_this_update=False,
                    previous_complete_state_sha256=complete_state_identity(self._result)['sha256'],
                    inserted_root_sources={k:copy.deepcopy(sources[k]) for k in sorted(new_roots)},
                    execution_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    rdg_source_sha256=hashlib.sha256(Path(__file__).with_name('rdg.py').read_bytes()).hexdigest())
                result.typed_session_audit.pop('exact_result_cache_hit',None)
                result.typed_session_audit['complete_state_sha256'] = complete_state_identity(result)['sha256']
                self.audit.append(dict(mode='incremental', new_roots=len(new_roots), processed_events=seen_events,
                    fired_valuations=fired, theta=session.snapshot_id, only_new_body_fact_anchored_valuations=True))
        self._seed, self._sources, self._result = seed, sources, _clone(result)
        return _clone(result)

    @property
    def result(self):
        return _clone(self._result) if self._result is not None else None
