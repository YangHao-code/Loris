"""Round-frozen chase maintenance (paper §6.2 and author supplement).

chaseInc propagates added verified facts. chaseDelta withdraws conclusions of
disabled *instances*, then rederives them under the new frozen interpretation.
The retained first-witness DAG is sufficient for conservative overdeletion;
rederivation considers alternative rules, including unchanged ML predicates.
No old derived fact is promoted to Gamma and no model is trained here.
"""

from collections import defaultdict, deque
import copy

import numpy as np

from loris.chase import ChaseResult
from loris._engine.rdl_reference import Label, LabelCompare, ML
from loris._engine.typed_runtime import ProofDAG, RelationFacts, _plain
from loris._learning.incremental import (
    EventState,
    state_from_result,
    valuations,
    event_bindings,
    apply_head,
    validate_static_context_prefix,
)
from loris._learning.rdg import RuleDependencyGraph


def _roots(gamma, n):
    keys = {"subset:%d:%d" % (i, i) for i in range(n)}
    for sign, matrix in [("pos", gamma.pos), ("neg", gamma.neg)]:
        keys.update("%s:%d:%d" % (sign, i, k) for i, k in np.argwhere(matrix))
    for kind, edges in [("subset", gamma.subset), ("strict_subset", gamma.strict_subset)]:
        keys.update("%s:%d:%d" % (kind, i, j) for i, j in edges)
    return keys


def _head_keys(runner, rule, valuation):
    head = rule.head
    if isinstance(head, Label):
        return ("%s:%d:%d" % (head.polarity, valuation[head.var], runner.label_index[head.label]),)
    a, b = runner._edge(head, valuation)
    kind = "strict_subset" if head.op == "strict_subset" else "subset"
    pairs = ((a, b), (b, a)) if head.op == "eq" else ((a, b),)
    return tuple("%s:%d:%d" % (kind, i, j) for i, j in pairs)


def _result(runner, rules, state, before, mode, audit):
    invalid = state.invalid()
    conflicts = [(int(i), runner.labels[int(k)]) for i, k in np.argwhere(state.pos & state.neg)]
    result = ChaseResult(
        status="conflict" if invalid else "fixpoint",
        predictions=(state.pos & ~state.neg).astype(np.float32),
        conflicts=conflicts,
        n_rounds=0,
        provenance=copy.deepcopy(state.provenance),
        positive_facts=state.pos.copy(),
        negative_facts=state.neg.copy(),
    )
    result.subset_facts = RelationFacts(state.out)
    result.strict_subset_facts = RelationFacts(state.proper)
    proof = {key: _plain(node) for key, node in ProofDAG(state.proof, state.base_proof).items()}
    parents = defaultdict(list)
    pending = {}
    proof_keys = set(proof)
    for key, node in proof.items():
        deps = set(node.get("dependencies", ()))
        if not deps <= proof_keys:
            raise RuntimeError("retained proof depends on a missing fact")
        pending[key] = len(deps)
        for dep in deps:
            parents[dep].append(key)
    queue = deque(key for key, count in pending.items() if not count)
    completed = 0
    while queue:
        key = queue.popleft()
        node = proof[key]
        ids = {node["rule_id"]} if node.get("rule_id") is not None else set()
        for dep in node.get("dependencies", ()):
            ids.update(proof[dep]["support_rule_ids"])
        node["support_rule_ids"] = sorted(ids)
        completed += 1
        for child in parents[key]:
            pending[child] -= 1
            if not pending[child]:
                queue.append(child)
    if completed != len(proof):
        raise RuntimeError("cyclic first-witness proof has no rooted derivation")
    result.proof_dag = ProofDAG(proof)
    # Provenance remains explanatory first-witness ancestry; reliability uses
    # complete threshold closures independently of this representation.
    result.provenance = {}
    for key, node in result.proof_dag.items():
        if node["kind"] in ("pos", "neg") and node.get("support_rule_ids"):
            target = (node["row"], runner.labels[node["column"]])
            result.provenance[target] = sorted(
                set(result.provenance.get(target, ())) | set(node["support_rule_ids"])
            )
    result.comparison_conflicts = (
        [state.conflict]
        if state.conflict and state.conflict["kind"] != "positive_negative_label_conflict"
        else []
    )
    from loris._engine.typed_session import digest

    result.reference_audit = dict(
        before.reference_audit,
        executor=mode,
        context_id=runner.context_id,
        program_sha256=digest([r.to_dict() for r in rules]),
        complete_without_cutoff=True,
        round_model_frozen=True,
        full_chase_call_performed_for_this_update=False,
        maintenance=audit,
    )
    return result


def _propagate(runner, rules, state, prepared, initial_rules=()):
    """Seed changed/rederivation rules, then consume only new fact events."""
    rdg = RuleDependencyGraph(rules)
    visited = set()
    applications = 0

    def fire(ri, fixed=None):
        nonlocal applications
        for valuation in valuations(
            runner, rules[ri], prepared[ri], state, fixed=fixed, rule_index=ri
        ):
            key = (ri, tuple(valuation[v] for v in rules[ri].variables))
            if key in visited:
                continue
            visited.add(key)
            apply_head(runner, rules[ri], ri, valuation, state)
            applications += 1
            if state.invalid():
                return

    if not state.invalid():
        for ri in sorted(initial_rules):
            fire(ri)
            if state.invalid():
                break
    queue = deque(state.events)
    state.events.clear()
    events = 0
    while queue and not state.invalid():
        event = queue.popleft()
        events += 1
        for ri, atom in rdg.consumers(event, runner):
            for fixed in event_bindings(atom, event, runner):
                fire(ri, fixed)
                queue.extend(state.events)
                state.events.clear()
                if state.invalid():
                    break
            if state.invalid():
                break
    return dict(processed_events=events, fired_instances=applications)


def chase_inc(runner, rules, before, old_gamma, gamma):
    """Add Gamma facts without reconsidering unaffected rule applications."""
    if before.status != "fixpoint":
        raise ValueError("chaseInc requires a successful previous closure")
    runner.verify_code()
    gamma = runner._gamma(gamma)
    if (
        np.any(old_gamma.pos & ~gamma.pos)
        or np.any(old_gamma.neg & ~gamma.neg)
        or not set(old_gamma.subset) <= set(gamma.subset)
        or not set(old_gamma.strict_subset) <= set(gamma.strict_subset)
    ):
        raise ValueError("chaseInc only adds verified Gamma facts")
    state = state_from_result(runner, before)
    added = _roots(gamma, runner.n) - _roots(old_gamma, runner.n)
    # A fact newly verified may already have been derived. Re-root its proof
    # explicitly so a later model update can never retract the verified fact.
    proof = {key: _plain(node) for key, node in before.proof_dag.items()}
    for key in added:
        kind, a, b = key.split(":")
        a, b = int(a), int(b)
        proof[key] = dict(
            kind=kind,
            row=a,
            column=b,
            origin="validated_Gamma",
            dependencies=[],
            rule_id=None,
            valuation={},
            static_evidence=[],
            support_rule_ids=[],
        )
    state.base_proof, state.proof = ProofDAG(proof), {}
    for key in sorted(added):
        kind, a, b = key.split(":")
        a, b = int(a), int(b)
        if kind in ("pos", "neg"):
            state.set_label(kind, a, b, "validated_Gamma")
        else:
            state.add_edge(a, b, kind == "strict_subset")
    prepared = [runner._prepare(*runner._rule(rule)) for rule in rules]
    audit = _propagate(runner, rules, state, prepared)
    audit.update(inserted_verified_facts=len(added), Gamma_never_retracted=True)
    return _result(runner, rules, state, before, "chaseInc", audit)


def chase_delta(old_runner, runner, old_rules, rules, before, gamma):
    """Withdraw invalid instance heads and descendants, then rederive now.

    Unchanged rule producers of withdrawn facts are explicitly revisited.
    Unsupported cycles cannot survive: the retained DAG has rooted witnesses,
    and new facts must again have a satisfied complete rule body.
    """
    if before.status != "fixpoint":
        raise ValueError("chaseDelta requires a successful previous closure")
    if (
        len(old_rules) != len(rules)
        or old_runner.labels != runner.labels
        or old_runner.n != runner.n
    ):
        raise ValueError("chaseDelta requires the same rule/document/label axes")
    runner.verify_code()
    old_runner.verify_code()
    validate_static_context_prefix(old_runner, runner)
    if set(old_runner.static_probabilities) != set(runner.static_probabilities) or any(
        not np.array_equal(p, runner.static_probabilities[a])
        for a, p in old_runner.static_probabilities.items()
        if any(
            isinstance(atom, ML) and not atom.dynamic and atom.alias == a
            for rule in rules
            for atom in rule.body
        )
    ):
        raise ValueError("fixed ML probabilities changed during refinement")
    gamma = runner._gamma(gamma)
    old_state = state_from_result(old_runner, before)
    prepared = [runner._prepare(*runner._rule(rule)) for rule in rules]
    changed = set()
    for ri, (old, new) in enumerate(zip(old_rules, rules)):
        if old.head != new.head or len(old.body) != len(new.body):
            raise ValueError("rule structure changed during model refinement")
        for oa, na in zip(old.body, new.body):
            if isinstance(oa, ML) and isinstance(na, ML):
                od, nd = oa.to_dict(), na.to_dict()
                od.pop("eta")
                nd.pop("eta")
                if od != nd or (not oa.dynamic and oa.eta != na.eta):
                    raise ValueError("only dynamic ML thresholds may change")
                if old_runner._static(oa) != runner._static(na):
                    changed.add(ri)
            elif oa != na:
                raise ValueError("non-ML rule structure changed")
    # Exact instance review changes can also invalidate or enable applications.
    if old_runner._application_scope is not None or runner._application_scope is not None:
        changed.update(range(len(rules)))
    roots = _roots(gamma, runner.n)
    withdrawn = set()
    disabled = 0
    for ri in sorted(changed):
        old = old_rules[ri]
        prep = old_runner._prepare(*old_runner._rule(old))
        for val in valuations(old_runner, old, prep, old_state, known_head=True, rule_index=ri):
            valid = runner.application_allowed(ri, rules[ri], val) and all(
                prepared[ri]["domains"][var] & (1 << row) for var, row in val.items()
            )
            if not valid:
                disabled += 1
                withdrawn.update(key for key in _head_keys(runner, old, val) if key not in roots)
    consumers = defaultdict(list)
    for key, node in before.proof_dag.items():
        for dep in node.get("dependencies", ()):
            consumers[dep].append(key)
    queue = deque(withdrawn)
    while queue:
        for key in consumers[queue.popleft()]:
            if key not in roots and key not in withdrawn:
                withdrawn.add(key)
                queue.append(key)
    retained = {key: _plain(node) for key, node in before.proof_dag.items() if key not in withdrawn}
    for node in retained.values():
        if node.get("origin") == "rule_application":
            _, evidence = runner._evidence(rules[node["rule_id"]], node["valuation"])
            node["static_evidence"] = evidence
    # Start from the true roots, preload surviving proof witnesses, and restore
    # retained facts. Reclosing built-in inclusion paths can itself rederive a
    # withdrawn relation or copied label through another surviving path.
    state = EventState(runner, gamma)
    for key, node in retained.items():
        if key not in state.base_proof and key not in state.proof:
            state.proof[key] = node
    for key in retained:
        kind, a, b = key.split(":")
        a, b = int(a), int(b)
        if kind in ("subset", "strict_subset"):
            state.add_edge(a, b, kind == "strict_subset")
    for key in retained:
        kind, a, b = key.split(":")
        a, b = int(a), int(b)
        if kind in ("pos", "neg"):
            state.set_label(kind, a, b, retained[key]["origin"])
    state.events = [e for e in state.events if "%s:%d:%d" % e not in retained]
    types = {(key.split(":")[0], int(key.split(":")[2])) for key in withdrawn}
    rederive = set(changed)
    for ri, rule in enumerate(rules):
        if isinstance(rule.head, Label):
            if (rule.head.polarity, runner.label_index[rule.head.label]) in types:
                rederive.add(ri)
        elif any(kind in ("subset", "strict_subset") for kind, _ in types):
            rederive.add(ri)
    audit = _propagate(runner, rules, state, prepared, rederive)
    present = set(state.base_proof) | set(state.proof)
    audit.update(
        changed_rules=sorted(changed),
        disabled_instances=disabled,
        temporarily_withdrawn=len(withdrawn),
        restored_same_update=len(withdrawn & present),
        finally_deleted=len(withdrawn - present),
        rederivation_rules=sorted(rederive),
        Gamma_never_retracted=True,
        rederive_in_same_model_state=True,
        unchanged_alternative_producers_revisited=True,
    )
    return _result(runner, rules, state, before, "chaseDelta", audit)
