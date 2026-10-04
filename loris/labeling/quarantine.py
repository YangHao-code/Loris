"""Author-approved conflict quarantine under one frozen model interpretation.

This is a new, nonmonotone labeling semantics, not the paper's conflict-failing
Chase or its chaseDelta implementation. Rule candidates stream from a frozen
snapshot while their heads update a separate live state. Each conflicting signed cell is locked
unknown for this invocation, and all facts are rebuilt from the remaining roots.
No old closure, first-conflict partial result, model update, or arbitration is
used here. Verified Gamma is never mutated; only its execution view is masked.
"""

import copy
from types import SimpleNamespace

import numpy as np

from loris.chase import ChaseResult
from loris._engine.rdl_reference import RefGamma
from loris._engine.typed_runtime import ProofDAG, _plain
from loris._engine.typed_session import digest, seed_identity
from loris._learning.incremental import EventState, apply_head, valuations
from .inference import _result


SEMANTICS = "author_conflict_unknown_synchronous_full_recompute_v1"


def _mask(value, shape, name):
    if value is None:
        return np.zeros(shape, dtype=bool)
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(name + " must be a boolean matrix on the complete document/label axes")
    return array.copy()


class _QuarantineState(EventState):
    def __init__(self, runner, execution_roots, unknown, original_gamma):
        self.unknown = unknown
        super().__init__(runner, execution_roots)
        # _State authenticates its roots as Gamma. A temporary arbitration seed
        # is a different kind of root: replace our private proof layer without
        # mutating the runner's cached Gamma proof or the caller's Gamma.
        roots = {key: _plain(node) for key, node in self.base_proof.items()}
        for node in roots.values():
            kind = node["kind"]
            if kind in ("pos", "neg"):
                source = original_gamma.pos if kind == "pos" else original_gamma.neg
                if not source[node["row"], node["column"]]:
                    node["origin"] = "temporary_seed"
                    node["persistent_Gamma"] = False
        self.base_proof = ProofDAG(roots)

    def set_label(self, sign, i, j, *args, **kwargs):
        # This applies to rule heads AND every built-in inclusion label copy.
        # Masked cells have neither sign and therefore cannot satisfy a body.
        if self.unknown[i, j]:
            return False
        return super().set_label(sign, i, j, *args, **kwargs)


def _valuation_snapshot(state):
    """Freeze the indexes read by valuations, without storing its groundings.

    The shared runner, prepared masks, and application scope are immutable.
    Proofs are not queried by valuations. Live application only adds facts in
    this batch, so every premise in the frozen snapshot remains available as
    an authentic proof parent until the whole batch has been consumed.
    """
    snapshot = copy.copy(state)
    snapshot.pos = state.pos.copy()
    snapshot.neg = state.neg.copy()
    snapshot.pos.setflags(write=False)
    snapshot.neg.setflags(write=False)
    snapshot.label_bits = {sign: tuple(rows) for sign, rows in state.label_bits.items()}
    for name in ("out", "inc", "proper", "proper_inc"):
        setattr(snapshot, name, getattr(state, name).copy(frozen=True))
    return snapshot


def quarantine_chase(
    runner,
    rules,
    gamma,
    *,
    initial_positive=None,
    initial_negative=None,
    initial_unknown=None,
) -> ChaseResult:
    """Compute the complete usable closure, isolating conflicting signed cells.

    ``initial_positive/negative`` are explicitly temporary root matrices. They
    are never installed in Gamma. ``initial_unknown`` fixes already quarantined
    cells for this same model round, e.g. an evidence-only feedback replay.
    The unknown mask only grows within this invocation. Independent strict
    relation contradictions still return ``status='conflict'`` with diagnostics.
    """
    runner.verify_code()
    rules = tuple(rules)
    gamma = runner._gamma(gamma)
    gamma_identity = seed_identity(gamma)
    shape = (runner.n, runner.k)
    temporary_pos = _mask(initial_positive, shape, "initial_positive")
    temporary_neg = _mask(initial_negative, shape, "initial_negative")
    unknown = _mask(initial_unknown, shape, "initial_unknown")
    initial_unknown_count = int(unknown.sum())
    root_pos = gamma.pos | temporary_pos
    root_neg = gamma.neg | temporary_neg
    root_conflicts = root_pos & root_neg & ~unknown
    unknown |= root_conflicts
    prepared = [runner._prepare(*runner._rule(rule)) for rule in rules]
    history = []
    if root_conflicts.any():
        history.append(
            dict(
                phase="initial_roots",
                synchronous_round=0,
                newly_quarantined=np.argwhere(root_conflicts).tolist(),
            )
        )
    rebuilds = rounds = applications = 0

    while True:
        execution_roots = RefGamma(
            root_pos & ~unknown,
            root_neg & ~unknown,
            gamma.subset,
            gamma.strict_subset,
        )
        state = _QuarantineState(runner, execution_roots, unknown, gamma)
        rebuilds += 1
        # Seed relation propagation can itself reveal all conflicts at once.
        newly_unknown = state.pos & state.neg & ~unknown
        if newly_unknown.any():
            unknown |= newly_unknown
            history.append(
                dict(
                    phase="seed_relation_closure",
                    synchronous_round=rounds,
                    newly_quarantined=np.argwhere(newly_unknown).tolist(),
                )
            )
            continue

        while True:
            # Exhaust every original complete-body valuation against the SAME
            # frozen pre-application indexes. Stream candidates instead of
            # retaining O(number of groundings) dictionaries. Partial results
            # cannot enable another rule in this batch or stop its enumeration.
            snapshot = _valuation_snapshot(state)
            before_version = state.version
            for ri, rule in enumerate(rules):
                for valuation in valuations(runner, rule, prepared[ri], snapshot, rule_index=ri):
                    apply_head(runner, rule, ri, valuation, state)
                    applications += 1
            rounds += 1
            newly_unknown = state.pos & state.neg & ~unknown
            if newly_unknown.any():
                unknown |= newly_unknown
                history.append(
                    dict(
                        phase="synchronous_rule_batch",
                        synchronous_round=rounds,
                        newly_quarantined=np.argwhere(newly_unknown).tolist(),
                    )
                )
                # Full replay removes downstream unsupported facts/relations
                # and unrooted cycles, while recovering all alternate sources.
                break
            if state.version == before_version:
                if seed_identity(gamma) != gamma_identity:
                    raise RuntimeError("quarantine changed the persistent Gamma record")
                audit = dict(
                    semantics=SEMANTICS,
                    full_recompute=True,
                    synchronous_candidates=True,
                    candidate_storage="streamed_from_frozen_valuation_indexes",
                    original_conflict_failing_chase_called=False,
                    unknown_is_monotone_within_invocation=True,
                    initial_unknown_cells=initial_unknown_count,
                    conflict_unknown_cells=int(unknown.sum()),
                    root_conflict_cells=int(root_conflicts.sum()),
                    rebuilds=rebuilds,
                    synchronous_rounds=rounds,
                    enumerated_candidate_instances=applications,
                    quarantines=history,
                    Gamma_record_unchanged=True,
                    Gamma_execution_quarantined_cells=int(
                        ((gamma.pos | gamma.neg) & unknown).sum()
                    ),
                    temporary_roots_promoted_to_Gamma=False,
                    rule_structure_unchanged=True,
                    model_and_thresholds_frozen=True,
                    strict_relation_failure_policy="conflict_with_diagnostic_not_label_quarantine",
                )
                # This serializer validates all proof parents and the absence
                # of unsupported first-witness cycles. The only remaining
                # invalidity can be an independent finite relation constraint.
                result = _result(
                    runner,
                    rules,
                    state,
                    SimpleNamespace(reference_audit={"semantics": SEMANTICS}),
                    "quarantine_full_recompute",
                    audit,
                )
                result.conflict_unknown = unknown.copy()
                result.n_rounds = rounds
                result.reference_audit.update(
                    semantics=SEMANTICS,
                    full_recompute=True,
                    full_chase_call_performed_for_this_update=True,
                    persistent_Gamma=gamma_identity,
                    persistent_Gamma_sha256=digest(gamma_identity),
                    conflict_unknown_cells=int(unknown.sum()),
                    independent_relation_conflict=result.comparison_conflicts,
                    original_paper_conflict_semantics=False,
                )
                return result
