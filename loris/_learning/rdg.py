"""Rule dependencies, grounded support estimates, and bounded local trust.

RDG reachability alone never claims a body's conjunction is satisfied. Every
activation below checks the entire finite valuation. The influence value is
the paper's union of multi-document support sets, not a submodular guarantee.
"""
from collections import ChainMap, Counter, deque

import numpy as np

from loris._engine.rdl_reference import Label, LabelCompare, RefGamma
from loris._engine.typed_runtime import indices
from .incremental import EventState, state_from_result, event_bindings, valuations, apply_head


class _HypothesisState(EventState):
    @classmethod
    def from_result(cls, runner, before):
        original = state_from_result(runner, before)
        state = object.__new__(cls)
        state.__dict__ = original.__dict__.copy()
        state.provenance = ChainMap({}, original.provenance)
        state._changed_labels = set()
        state._changed_bits = {}
        state._changed_relations = {}
        return state

    def set_label(self, sign, row, label, *args, **kwargs):
        matrix = self.pos if sign == 'pos' else self.neg
        if not matrix[row, label]:
            self._changed_labels.add((sign, row, label))
            self._changed_bits.setdefault((sign, label), self.label_bits[sign][label])
        return super().set_label(sign, row, label, *args, **kwargs)

    def _install(self, kind, source, target, *args, **kwargs):
        outward, inward = ('proper', 'proper_inc') if kind == 'strict_subset' else ('out', 'inc')
        for name, row in ((outward, source), (inward, target)):
            self._changed_relations.setdefault((name, row), getattr(self, name)[row])
        return super()._install(kind, source, target, *args, **kwargs)

    def rollback(self):
        for sign, row, label in self._changed_labels:
            (self.pos if sign == 'pos' else self.neg)[row, label] = False
        for (sign, label), value in self._changed_bits.items():
            self.label_bits[sign][label] = value
        for (name, row), value in self._changed_relations.items():
            getattr(self, name)[row] = value
        self._changed_labels.clear()
        self._changed_bits.clear()
        self._changed_relations.clear()
        self.events.clear()
        self.proof.clear()
        self.provenance.maps[0].clear()
        self.stats = Counter()
        self.version = self.relation_version = self.feasibility_checks = 0
        self.last_valid_version = -1
        self.conflict = None


class RuleDependencyGraph:
    def __init__(self, rules):
        self.rules = tuple(rules)
        self.body_atoms = [(ri, atom) for ri, rule in enumerate(self.rules) for atom in rule.body
                           if isinstance(atom, (Label, LabelCompare))]
        self._consumers = {}
        for ri, atom in self.body_atoms:
            key = (atom.polarity, atom.label) if isinstance(atom,Label) else (
                'strict_subset' if atom.op == 'strict_subset' else 'subset',None)
            self._consumers.setdefault(key,[]).append((ri,atom))
        self.edges = {i: set() for i in range(len(self.rules))}
        for i, rule in enumerate(self.rules):
            head = rule.head
            for j, atom in self.body_atoms:
                if isinstance(head, Label) and isinstance(atom, Label):
                    match = (head.label, head.polarity) == (atom.label, atom.polarity)
                elif isinstance(head, LabelCompare):
                    # Comparison closure also copies already-known positives.
                    match = isinstance(atom, LabelCompare) or (isinstance(atom, Label) and atom.polarity == 'pos')
                else:
                    match = False
                if match:
                    self.edges[i].add(j)

    def consumers(self, event, runner):
        kind, _, label = event
        key = (kind,runner.labels[label] if kind in ('pos','neg') else None)
        yield from self._consumers.get(key,())

    def activation_scope(self, session, before, additions, max_depth=None):
        if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
            raise ValueError('local propagation depth must be nonnegative')
        runner = session.runner
        runner.verify_code()
        state = state_from_result(runner, before)
        prepared = [runner._prepare(*runner._rule(rule)) for rule in self.rules]
        return self._activation_scope(runner, state, prepared, additions, max_depth)

    def _activation_scope(self, runner, state, prepared, additions, max_depth=None):
        for sign, row, label in additions:
            state.set_label(sign, row, label, 'RDG_hypothesis')
        queue = deque((e, 0) for e in state.events)
        state.events.clear()
        active, documents, support, applied = set(), set(), set(), set()
        documents.update(int(row) for _, row, _ in additions)
        support.update(int(row) for _, row, _ in additions)
        activations = []
        while queue:
            event, depth = queue.popleft()
            documents.add(event[1])
            # Inclusion/equality closure can emit positive labels without an
            # explicit Label-headed RDL. These reached documents belong to
            # the same influence support as explicit consequences. Negative
            # events alone do not resolve a document; retain their existing
            # rule-activation behavior without counting them as positive copy.
            if event[0] == 'pos':
                support.add(event[1])
            if event[0] in ('subset', 'strict_subset'):
                documents.add(event[2])
            if max_depth is not None and depth >= max_depth:
                continue
            for ri, atom in self.consumers(event, runner):
                rule = self.rules[ri]
                for fixed in event_bindings(atom, event, runner):
                    for valuation in valuations(runner, rule, prepared[ri], state, fixed, known_head=True, rule_index=ri):
                        key = (ri, tuple(valuation[v] for v in rule.variables))
                        if key in applied:
                            continue
                        applied.add(key); active.add(ri); documents.update(valuation.values())
                        head = rule.head
                        targets = [valuation[head.var]] if isinstance(head, Label) else [valuation[head.left_var], valuation[head.right_var]]
                        support.update(targets)
                        activations.append(dict(rule_index=ri, valuation=dict(valuation), depth=depth + 1))
                        apply_head(runner, rule, ri, valuation, state)
                        queue.extend((e, depth + 1) for e in state.events)
                        state.events.clear()
        return dict(rule_indices=sorted(active), document_indices=sorted(documents),
                    support_documents=sorted(support), activations=activations,
                    max_depth=max_depth, estimated_not_validated_labels=True)

    def estimate(self, session, before, row, label, unlabeled):
        scope = self.activation_scope(session, before, [('pos', row, label)])
        return len(set(scope['support_documents']) & set(unlabeled))

    def estimate_batch(self, session, before, queries, unlabeled):
        """Exact independent hypotheses sharing one fixed closure and rule preparation."""
        runner = session.runner
        runner.verify_code()
        state = _HypothesisState.from_result(runner, before)
        prepared = [runner._prepare(*runner._rule(rule)) for rule in self.rules]
        uncovered = set(unlabeled)
        for row, label in queries:
            runner.verify_code()
            try:
                scope = self._activation_scope(runner, state, prepared, [('pos', row, label)])
                score = len(set(scope['support_documents']) & uncovered)
            finally:
                state.rollback()
            yield score

    def local_check(self, session, before, additions, depth=2):
        """Construct D0/Sigma0 to L, then a true local fixed point within them."""
        scope = self.activation_scope(session, before, additions, max_depth=depth)
        allowed = set(scope['document_indices'])
        runner = session.runner
        pos, neg = np.zeros_like(before.positive_facts), np.zeros_like(before.negative_facts)
        if allowed:
            rows = sorted(allowed); pos[rows] = before.positive_facts[rows]; neg[rows] = before.negative_facts[rows]
        for sign, row, label in additions:
            (pos if sign == 'pos' else neg)[row, label] = True
        if np.any(pos & neg):
            return dict(scope=scope, consistent=False, local_fixed_point_rounds=0,
                local_rule_applications=0, full_program_consistency_claimed=False,
                direct_conflicting_cells=np.argwhere(pos & neg).tolist(), outside_D0_roots_excluded=True)
        relations = []
        for relation in (before.subset_facts, before.strict_subset_facts):
            relations.append(tuple((i, j) for i in allowed for j in indices(relation.rows[i]) if j in allowed))
        state = EventState(runner, RefGamma(pos, neg, *relations))
        selected = scope['rule_indices']
        prepared = {i: runner._prepare(*runner._rule(self.rules[i])) for i in selected}
        rounds, applications = 0, 0
        conflict = state.invalid()
        while not conflict:
            rounds += 1; changed = False
            for ri in selected:
                for valuation in valuations(runner, self.rules[ri], prepared[ri], state, allowed_rows=allowed, rule_index=ri):
                    changed |= apply_head(runner, self.rules[ri], ri, valuation, state)
                    applications += 1
                    if state.invalid():
                        conflict = True; break
                if conflict: break
            if not changed: break
        return dict(scope=scope, consistent=not conflict, local_fixed_point_rounds=rounds,
                    local_rule_applications=applications, full_program_consistency_claimed=False,
                    outside_D0_roots_excluded=True)
