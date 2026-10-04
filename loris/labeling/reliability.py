"""Independent rule precision, direct-support weights, and proof ablations.

These equations are an author supplement, not equations already in the draft.
The calibration answers are held out of model/rule fitting and all inference.
Historical pooled fit estimates are deliberately not accepted by this module.
"""

import json
from collections import deque
import hashlib
import unicodedata
from pathlib import Path
import pickle
import time

import numpy as np

from loris.discovery.manifest import bound, file_record
from loris.document import Document
from loris._engine.rdl_reference import Label, LabelCompare, RefGamma
from loris._engine.typed_session import array_identity, digest, seed_identity
from loris._storage.discovery_factory import make_discovery_runner

SCHEMA = "loris_independent_rule_reliability_v2"
PROTOCOL_SCHEMA = "loris_independent_rule_calibration_v2"
UNIT = "rule_distinct_checkable_conclusion"
TRANSFER_POLICY = "independent_current_model_scores_recalibrated_before_commit"
FROZEN_POLICY = (
    "frozen_independent_calibration_scores_not_recertified_after_model_or_threshold_updates"
)
ESTIMATOR = {
    "name": "laplace_rule_precision",
    "alpha": 1,
    "beta": 1,
    "formula": "(correct+1)/(support+2)",
    "zero_support": "prior_one_half",
}
RELATION_SCORING = {
    "schema": "loris_relation_propagated_target_labels_v1",
    "unit": "distinct_positive_target_Label_per_rule",
    "source": "all_current_valid_primitive_relation_head_instances",
    "directions": "eq_both; subset_and_strict_follow_declared_direction",
    "self_edges": "no_label_transfer",
    "transitive_path": "last_learned_primitive_relation_rule",
    "trusted_Gamma_transport": "retain_last_learned_source; Gamma_Label_roots_weight_one",
    "known_target": "count_alternative_valid_support_even_if_another_source_fired_first",
}
STRICT_CHASE_SEMANTICS = "strict_halt_v1"
QUARANTINE_CHASE_SEMANTICS = "conflict_pair_quarantine_no_arbitration_v1"
RETAINED_CHASE_SEMANTICS = "conflict_pair_retention_no_arbitration_v1"
# 2026-09-30: retained history where validated Gamma always wins; a derived
# label contradicting Gamma is blocked (never quarantines the Gamma cell).
RETAINED_GAMMA_WINS_SEMANTICS = "conflict_pair_retention_validated_gamma_wins_no_arbitration_v1"
RETAINED_SEMANTICS = (RETAINED_CHASE_SEMANTICS, RETAINED_GAMMA_WINS_SEMANTICS)


def gamma_wins(manifest):
    return manifest.get("loop", {}).get("gamma_conflict_policy", "quarantine") == "validated_gamma_wins"


def relation_scoring(semantics):
    """Bind the changed calibration observation window explicitly."""
    value = dict(RELATION_SCORING)
    if semantics in RETAINED_SEMANTICS:
        value.update(
            source="actual_fired_relation_edges_and_label_transfers",
            observation_window="this_calibration_chase_including_later_quarantined_conclusions",
        )
    return value


def chase_semantics(manifest):
    """Author-added scoring protocol; this is not a PDF-defined formula."""
    return (
        (
            (RETAINED_GAMMA_WINS_SEMANTICS if gamma_wins(manifest) else RETAINED_CHASE_SEMANTICS)
            if manifest.get("loop", {}).get("unknown_semantics") == "retain_consequences"
            else QUARANTINE_CHASE_SEMANTICS
        )
        if manifest.get("loop", {}).get("conflict_policy") == "unknown_arbitrate"
        else STRICT_CHASE_SEMANTICS
    )


def _artifact_chase_semantics(artifact):
    # Historical artifacts were strict. Absence is never a new-protocol claim.
    value = artifact.get("chase_semantics", STRICT_CHASE_SEMANTICS)
    if value not in (STRICT_CHASE_SEMANTICS, QUARANTINE_CHASE_SEMANTICS) + RETAINED_SEMANTICS:
        raise ValueError("unknown reliability chase_semantics")
    return value


def _quarantine_view(closure, gamma):
    """Use active Gamma roots; the persistent verified ledger stays unchanged."""
    marker = getattr(closure, "conflict_unknown", None)
    unknown = np.zeros(gamma.shape, bool) if marker is None else np.asarray(marker)
    if unknown.dtype != np.bool_ or unknown.shape != gamma.shape:
        raise ValueError("conflict_unknown reliability mask differs from signed axes")
    pos, neg = np.asarray(closure.positive_facts), np.asarray(closure.negative_facts)
    if pos.shape != gamma.shape or neg.shape != gamma.shape or np.any((pos | neg) & unknown):
        raise ValueError("reliability requires an arbitration-free quarantined closure")
    active_gamma = RefGamma(
        gamma.pos & ~unknown,
        gamma.neg & ~unknown,
        tuple(getattr(gamma, "subset", ())),
        tuple(getattr(gamma, "strict_subset", ())),
    )
    return unknown, active_gamma


def _closure_chase_semantics(closure):
    if hasattr(closure, "history_nodes"):
        return (RETAINED_GAMMA_WINS_SEMANTICS if getattr(closure, "gamma_wins", False)
                else RETAINED_CHASE_SEMANTICS)
    return (
        QUARANTINE_CHASE_SEMANTICS
        if hasattr(closure, "conflict_unknown")
        else STRICT_CHASE_SEMANTICS
    )


def _validate_chase_semantics(artifact, expected):
    if _artifact_chase_semantics(artifact) != expected:
        raise ValueError("reliability chase_semantics mismatch; recalibration is required")


def rule_score(support, correct):
    """Laplace rule precision: unobserved rules receive the stated 1/2 prior."""
    n, c = np.asarray(support, float), np.asarray(correct, float)
    if (
        n.shape != c.shape
        or not np.isfinite(n).all()
        or not np.isfinite(c).all()
        or np.any(n < 0)
        or np.any(c < 0)
        or np.any(c > n)
        or np.any(n != np.floor(n))
        or np.any(c != np.floor(c))
    ):
        raise ValueError("rule support/correct must be integer counts with 0 <= correct <= support")
    return (c + 1.0) / (n + 2.0)


def source_binding(context, rules, manifest):
    """Initial application snapshot; different calibration rows have own tables."""
    return dict(
        chase_semantics=chase_semantics(manifest),
        context_file_sha256=manifest["context"]["sha256"],
        checkpoint_sha256=manifest["dynamic"]["checkpoint"]["sha256"],
        probability_lineage=manifest["dynamic"]["probability_lineage"],
        model_tables={a: array_identity(p) for a, p in context["model_probabilities"].items()},
        model_decisions={
            a: array_identity(p) for a, p in context.get("model_decisions", {}).items()
        },
        program_id=digest([r.to_dict() for r in rules]),
        initial_Gamma=seed_identity(context["gamma"]),
        row_ids_id=digest(context["row_ids"]),
        labels=list(context["labels"]),
    )


def validated_rule_scores(artifact):
    _artifact_chase_semantics(artifact)
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("source_unit") != UNIT
        or artifact.get("estimator") != ESTIMATOR
        or artifact.get("transfer_policy") not in (TRANSFER_POLICY, FROZEN_POLICY)
    ):
        raise ValueError(
            "independent per-rule calibration schema required; historical fit pooled priors are not independent"
        )
    records = artifact.get("rule_scores")
    if not isinstance(records, list):
        raise ValueError("complete ordered per-rule calibration records required")
    if any(r.get("head_kind") == "LabelCompare" for r in records) and artifact.get(
        "relation_scoring"
    ) != relation_scoring(_artifact_chase_semantics(artifact)):
        raise ValueError(
            "legacy relation whole-set reliability must be recalibrated for propagated target labels"
        )
    values = []
    for i, record in enumerate(records):
        if (
            record.get("rule_id") != i
            or not isinstance(record.get("rule_sha256"), str)
            or not record["rule_sha256"]
        ):
            raise ValueError("rule calibration order or identity mismatch")
        expected = float(rule_score(record["support"], record["correct"]))
        q = record.get("q")
        if isinstance(q, bool) or not isinstance(q, (float, int)) or q != expected:
            raise ValueError("rule q does not match (correct+1)/(support+2)")
        values.append(q)
    return np.asarray(values, float)


def _validate_relation_scoring(rules, artifact):
    relation_ids = [i for i, rule in enumerate(rules) if isinstance(rule.head, LabelCompare)]
    if relation_ids and (
        artifact.get("relation_scoring") != relation_scoring(_artifact_chase_semantics(artifact))
        or any(
            i >= len(artifact["rule_scores"])
            or artifact["rule_scores"][i].get("conclusion_unit")
            != "propagated_distinct_target_Label"
            for i in relation_ids
        )
    ):
        raise ValueError(
            "relation q requires propagated target-label calibration; legacy whole-set "
            "comparison reliability must be recalibrated, not reused"
        )


def _positive_row_bits(positive):
    return [sum(1 << int(col) for col in np.flatnonzero(row)) for row in positive]


def _gamma_successors(gamma):
    outgoing = {}
    for source, target in tuple(getattr(gamma, "subset", ())) + tuple(
        getattr(gamma, "strict_subset", ())
    ):
        source, target = int(source), int(target)
        if source != target:
            outgoing.setdefault(source, set()).add(target)
    return outgoing


def _transport_trusted_relations(
    conclusions, outgoing, positive_bits, blocked_bits, *, allow_partial=False,
    negative_blocked_bits=None,
):
    """Gamma relations transport the existing source without a new learned q.

    This is an explicit operational convention. No negative contraposition or
    new learned provenance is introduced. Monotone bitset union terminates even
    on cycles and retains every alternative source supplied by the caller.
    """
    pending = deque(row for sign, row in conclusions if sign == "pos")
    queued = set(pending)
    while pending:
        row = pending.popleft()
        queued.discard(row)
        labels = conclusions[("pos", row)]
        for target in outgoing.get(row, ()):
            old = conclusions.get(("pos", target), 0)
            new = labels & ~old
            if not new:
                continue
            if new & ~(positive_bits[target] | blocked_bits[target]):
                if not allow_partial:
                    raise ValueError(
                        "trusted relation support is absent from complete current closure"
                    )
                new &= positive_bits[target] | blocked_bits[target]
                if not new:
                    continue
            conclusions[("pos", target)] = old | new
            if target not in queued:
                pending.append(target)
                queued.add(target)
    # A quarantined or Gamma-protected intermediate may lie on an intact
    # transitive relation path. It may carry a temporary source attribution,
    # but its blocked signed Label is never counted as an active conclusion.
    for (sign, row), labels in list(conclusions.items()):
        blocked = (negative_blocked_bits if sign == "neg"
                   and negative_blocked_bits is not None else blocked_bits)
        conclusions[(sign, row)] = labels & ~blocked[row]
    return conclusions


def _relation_source_bits(state, positive_bits, blocked_bits):
    """Include labels reaching a blocked intermediate via intact relations.

    The transitive relation x<=z may copy x.A without using a quarantined y.A
    on x<=y<=z as a Label premise. This preserves attribution to the last
    learned primitive relation edge without reviving y.A as an active fact.
    """
    from loris._engine.typed_runtime import indices

    transported = list(positive_bits)
    # Reflexive rows cannot add anything to their own existing label set.
    # In full-train Gamma almost every row has a protected negative label;
    # visiting just nonidentity rows avoids constructing N-bit singleton
    # integers for all those otherwise untouched documents.
    targets = (state.inc.nonidentity_sources() if
               hasattr(state.inc, "nonidentity_sources") else
               range(len(blocked_bits)))
    for target in targets:
        blocked = blocked_bits[target]
        if blocked:
            sources = (state.inc.columns(target) if
                       hasattr(state.inc, "columns") else
                       indices(state.inc[target]))
            for source in sources:
                transported[target] |= positive_bits[source] & blocked
    return transported


def _direct_conclusions(
    runner,
    rule,
    rule_index,
    state,
    positive_bits,
    gamma_edges,
    *,
    transport_label_heads,
    blocked_bits,
    relation_source_bits,
    allow_partial=False,
    negative_blocked_bits=None,
):
    """All valid immediate rule consequences, represented sparsely by row bits.

    A relation rule produces positive target labels through its own primitive
    head edges, followed only by trusted Gamma edges. Derived transitive edges
    do not acquire invented producers. A later learned edge supplies its own
    q independently, so there is no upstream minimum or first-witness choice.
    """
    from loris._learning.incremental import valuations

    prepared = runner._prepare(*runner._rule(rule))
    head = rule.head
    conclusions = {}
    matches = 0
    for valuation in valuations(
        runner, rule, prepared, state, known_head=True, rule_index=rule_index
    ):
        matches += 1
        if isinstance(head, Label):
            row, col = int(valuation[head.var]), runner.label_index[head.label]
            blocked = (negative_blocked_bits if head.polarity == "neg"
                       and negative_blocked_bits is not None else blocked_bits)
            if blocked[row] & (1 << col):
                continue
            matrix = state.pos if head.polarity == "pos" else state.neg
            if not matrix[row, col]:
                if allow_partial:
                    continue
                raise ValueError(
                    "current valid rule conclusion is absent from the supplied closure"
                )
            key = (head.polarity, row)
            conclusions[key] = conclusions.get(key, 0) | (1 << col)
        elif isinstance(head, LabelCompare):
            source, target = runner._edge(head, valuation)
            edges = ((source, target), (target, source)) if head.op == "eq" else ((source, target),)
            for source, target in edges:
                if source == target:
                    continue  # identity edges perform no cross-document label transfer
                relation = state.proper if head.op == "strict_subset" else state.out
                if not relation[source] & (1 << target):
                    if allow_partial:
                        continue
                    raise ValueError(
                        "current valid relation head is absent from the supplied closure"
                    )
                labels = relation_source_bits[source]
                if labels & ~(positive_bits[target] | blocked_bits[target]):
                    if not allow_partial:
                        raise ValueError(
                            "relation-propagated labels are absent from complete current closure"
                        )
                    labels &= positive_bits[target] | blocked_bits[target]
                if labels:
                    key = ("pos", target)
                    conclusions[key] = conclusions.get(key, 0) | labels
    if isinstance(head, LabelCompare) or transport_label_heads:
        _transport_trusted_relations(
            conclusions, gamma_edges, positive_bits, blocked_bits, allow_partial=allow_partial,
            negative_blocked_bits=negative_blocked_bits,
        )
    return conclusions, matches


def direct_rule_weights(runner, rules, closure, gamma, artifact, *, excluded_rule_ids=()):
    """Maximum q of current direct Label or responsible relation producers.

    All permitted instances are checked on the complete current closure. A
    relation producer is the last learned primitive relation edge on a transfer
    path; subsequent Gamma edges retain that source q. Trusted Label roots have
    weight one, including transport solely through trusted Gamma edges. No
    upstream minimum, inferred negative labels, or first-witness lookup occurs.
    """
    from loris._learning.incremental import state_from_result
    from loris._engine.typed_runtime import indices

    rules = tuple(rules)
    q = validated_rule_scores(artifact)
    if len(q) != len(rules):
        raise ValueError("reliability rule axis differs from current program")
    _validate_relation_scoring(rules, artifact)
    if closure.status != "fixpoint":
        raise ValueError("direct rule weights require a complete successful closure")
    semantics = _closure_chase_semantics(closure)
    _validate_chase_semantics(artifact, semantics)
    unknown, gamma = _quarantine_view(closure, gamma)
    pos, neg = np.asarray(closure.positive_facts, bool), np.asarray(closure.negative_facts, bool)
    if (
        pos.shape != gamma.shape
        or neg.shape != gamma.shape
        or gamma.shape != (runner.n, runner.k)
        or np.any(pos & neg)
        or np.any(gamma.pos & ~pos)
        or np.any(gamma.neg & ~neg)
    ):
        raise ValueError("direct rule weights require matching current program and signed axes")
    excluded = set(excluded_rule_ids)
    partial = getattr(closure, "recorded_source_view", False)
    if any(type(i) is not int or not 0 <= i < len(rules) for i in excluded):
        raise ValueError("excluded rule index outside the current program")
    state = state_from_result(runner, closure)
    positive_bits = _positive_row_bits(pos)
    blocked_bits = _positive_row_bits(unknown)
    negative_blocked_bits = blocked_bits
    if semantics == RETAINED_GAMMA_WINS_SEMANTICS:
        # A complete Gamma-wins closure intentionally omits conclusions that
        # oppose trusted roots. Exclude only that signed support; retain the
        # missing-fact check for every unrelated unprotected conclusion.
        blocked_bits = _positive_row_bits(unknown | gamma.neg)
        negative_blocked_bits = _positive_row_bits(unknown | gamma.pos)
    relation_source_bits = _relation_source_bits(state, positive_bits, blocked_bits)
    gamma_edges = _gamma_successors(gamma)
    weights = np.zeros(gamma.shape, dtype=np.float64)
    support_counts = np.zeros(gamma.shape, dtype=np.int64)
    records = []
    for ri, rule in enumerate(rules):
        if ri in excluded:
            continue
        conclusions, matches = _direct_conclusions(
            runner,
            rule,
            ri,
            state,
            positive_bits,
            gamma_edges,
            transport_label_heads=True,
            blocked_bits=blocked_bits,
            relation_source_bits=relation_source_bits,
            allow_partial=partial,
            negative_blocked_bits=negative_blocked_bits,
        )
        count = 0
        for (sign, row), labels in conclusions.items():
            cols = np.fromiter(indices(labels), dtype=np.int64)
            weights[row, cols] = np.maximum(weights[row, cols], q[ri])
            support_counts[row, cols] += 1
            count += len(cols)
        records.append(
            dict(
                rule_id=ri,
                q=float(q[ri]),
                valid_instances=matches,
                distinct_supported_labels=count,
                head_kind=type(rule.head).__name__,
            )
        )
    roots = {}
    for row, labels in enumerate(_positive_row_bits(gamma.pos)):
        if labels:
            roots[("pos", row)] = labels
    _transport_trusted_relations(
        roots, gamma_edges, positive_bits, blocked_bits, allow_partial=partial,
        negative_blocked_bits=negative_blocked_bits,
    )
    for (_, row), labels in roots.items():
        weights[row, np.fromiter(indices(labels), dtype=np.int64)] = 1.0
    trusted = gamma.pos | gamma.neg
    weights[trusted] = 1.0
    unresolved = (pos | neg) & ~trusted & (weights == 0)
    return weights, dict(
        chase_semantics=semantics,
        conflict_unknown_pairs=int(unknown.sum()),
        arbitration_conclusions_used=False,
        algorithm="all_current_direct_Label_and_relation_transfer_instances",
        formula="max_direct_support_rules(q_rule)",
        multiple_direct_support_policy="maximum_q_author_supplement",
        upstream_rule_scores_used=False,
        first_witness_only=False,
        all_current_instances=True,
        conservative_recorded_source_view=bool(partial),
        duplicate_instances_increase_weight=False,
        closure_calls=0,
        excluded_rule_ids=sorted(excluded),
        relation_head_rule_ids=[i for i, r in enumerate(rules) if isinstance(r.head, LabelCompare)],
        relation_scoring=relation_scoring(semantics),
        relation_only_support_policy="last_learned_primitive_transfer_then_trusted_Gamma_transport",
        trusted_relation_transport="retain_existing_source_q; Gamma_Label_roots_are_one",
        unsupported_nonGamma_labels=int(unresolved.sum()),
        multiply_direct_supported_labels=int((support_counts > 1).sum()),
        rules=records,
        score_transfer_policy=artifact["transfer_policy"],
        current_program_id=digest([r.to_dict() for r in rules]),
        calibrated_program_id=artifact.get("binding", {}).get("program_id"),
        current_Gamma=seed_identity(gamma),
    )


def lineage_scores(chase_subset, rules, gamma, artifact, *, excluded_rule_ids=()):
    """Return rho[N,L] for ALL complete current derivations, including cycles.

    At each distinct rule score t, a fact has a proof with minimum rule score
    at least t exactly when the program restricted to q >= t entails it. Thus
    these complete threshold closures compute max(proofs) min(rule scores)
    without enumerating proofs or depending on a first witness. The callback
    must preserve the current model snapshot and any rule-instance restrictions;
    it receives (selected_rules, gamma), and must account for each Chase call.
    """
    rules = tuple(rules)
    q = validated_rule_scores(artifact)
    if len(q) != len(rules):
        raise ValueError("reliability rule axis differs from current program")
    _validate_relation_scoring(rules, artifact)
    excluded = set(excluded_rule_ids)
    if any(type(i) is not int or not 0 <= i < len(rules) for i in excluded):
        raise ValueError("excluded rule index outside the current program")
    active = [i for i in range(len(rules)) if i not in excluded]
    rho = np.zeros(gamma.shape, dtype=np.float64)
    seen_pos, seen_neg = np.zeros(gamma.shape, bool), np.zeros(gamma.shape, bool)
    calls = []
    fixed_unknown = None
    # Built-in inclusion propagation and Gamma roots have no learned rules,
    # hence the empty minimum is 1. This includes labels copied by Gamma edges.
    levels = [1.0] + sorted({float(q[i]) for i in active if q[i] < 1.0}, reverse=True)
    for threshold in levels:
        chosen = [i for i in active if q[i] >= threshold]
        state = chase_subset(tuple(rules[i] for i in chosen), gamma)
        if state.status != "fixpoint":
            raise ValueError("reliability requires complete successful threshold closures")
        _validate_chase_semantics(artifact, _closure_chase_semantics(state))
        unknown, _ = _quarantine_view(state, gamma)
        if fixed_unknown is None:
            fixed_unknown = unknown.copy()
        elif not np.array_equal(fixed_unknown, unknown):
            raise ValueError("threshold reliability requires the fixed full-program quarantine")
        pos, neg = np.asarray(state.positive_facts, bool), np.asarray(state.negative_facts, bool)
        if pos.shape != rho.shape or neg.shape != rho.shape or np.any(pos & neg):
            raise ValueError("threshold closure has invalid signed label axes")
        if np.any(seen_pos & ~pos) or np.any(seen_neg & ~neg):
            raise ValueError("threshold closures must retain all earlier signed facts")
        if np.any((pos & seen_neg) | (neg & seen_pos)):
            raise ValueError("threshold closures disagree on a signed conclusion")
        fresh = (pos | neg) & ~(seen_pos | seen_neg)
        rho[fresh] = threshold
        seen_pos |= pos
        seen_neg |= neg
        calls.append(
            dict(threshold=threshold, rule_ids=chosen, newly_supported_labels=int(fresh.sum()))
        )
    return rho, dict(
        chase_semantics=_artifact_chase_semantics(artifact),
        quarantine_policy="fixed_full_program_mask_for_all_threshold_subprograms",
        algorithm="complete_threshold_program_closures",
        formula="max_complete_derivations(min_rule_q)",
        all_derivations=True,
        first_witness_only=False,
        excluded_rule_ids=sorted(excluded),
        closure_calls=len(calls),
        thresholds=calls,
        score_transfer_policy=artifact["transfer_policy"],
        current_program_id=digest([r.to_dict() for r in rules]),
        calibrated_program_id=artifact.get("binding", {}).get("program_id"),
        current_Gamma=seed_identity(gamma),
    )


def validate_protocol(protocol, context=None, rules=None, manifest=None):
    required = {
        "schema",
        "context",
        "answers",
        "calibration_row_ids",
        "model_fit_row_ids",
        "rule_fit_row_ids",
        "model_binding",
        "independence_statement",
    }
    if (
        not isinstance(protocol, dict)
        or set(protocol) != required
        or protocol["schema"] != PROTOCOL_SCHEMA
    ):
        raise ValueError(
            "explicit independent calibration protocol required; fit partitions are not calibration"
        )
    for name in ("calibration_row_ids", "model_fit_row_ids", "rule_fit_row_ids"):
        rows = protocol[name]
        if (
            not isinstance(rows, list)
            or len(set(rows)) != len(rows)
            or any(not isinstance(x, str) or not x for x in rows)
        ):
            raise ValueError(name + " must be explicit unique document identities")
    selected = set(protocol["calibration_row_ids"])
    if (
        not selected
        or selected.intersection(protocol["model_fit_row_ids"])
        or selected.intersection(protocol["rule_fit_row_ids"])
    ):
        raise ValueError(
            "calibration rows must be nonempty and disjoint from model and rule fit rows"
        )
    if (
        not isinstance(protocol["independence_statement"], str)
        or not protocol["independence_statement"].strip()
    ):
        raise ValueError("model/rule fit provenance and independence declaration required")
    if context is not None and selected.intersection(context["row_ids"]):
        raise ValueError(
            "independent calibration rows must not occur in the application/discovery context"
        )
    if manifest is not None:
        expected = dict(
            checkpoint_sha256=manifest["dynamic"]["checkpoint"]["sha256"],
            probability_lineage=manifest["dynamic"]["probability_lineage"],
        )
        model_binding = dict(protocol["model_binding"])
        declared_semantics = model_binding.pop("chase_semantics", None)
        if model_binding != expected or (
            declared_semantics is not None and declared_semantics != chase_semantics(manifest)
        ):
            raise ValueError(
                "calibration predictions must bind the same frozen checkpoint/model lineage"
            )
    return protocol


def independent_content_check(target, calibration, selected):
    """Reject renamed cross-context duplicate model inputs, including anchors.

    External historical fitting corpora absent from these bound inputs remain
    covered by the explicit source/fit-ID declaration, not an invented audit.
    """

    def identity(doc):
        value = " ".join(unicodedata.normalize("NFKC", doc.cnt).split())
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    forbidden = {identity(doc) for doc in target["docs"]}
    forbidden.update(identity(doc) for i, doc in enumerate(calibration["docs"]) if not selected[i])
    collisions = [
        calibration["row_ids"][i]
        for i, doc in enumerate(calibration["docs"])
        if selected[i] and identity(doc) in forbidden
    ]
    if collisions:
        raise ValueError(
            "independent calibration content overlaps application/fit or anchor documents despite row IDs: "
            + ", ".join(collisions[:5])
        )
    return dict(
        content_policy="NFKC_whitespace_normalized_model_input_sha256",
        checked_application_documents=len(target["docs"]),
        checked_calibration_documents=int(np.sum(selected)),
        cross_context_content_overlap=0,
        external_fit_content_unavailable_requires_truthful_provenance=True,
    )


def calibration_texts(artifact):
    """Only text enters the trainer's prediction-only additional input set."""
    protocol_path = bound(artifact["protocol"], Path("."))
    protocol = json.loads(protocol_path.read_text())
    with bound(protocol["context"], protocol_path.parent).open("rb") as stream:
        context = pickle.load(stream)
    return [doc.cnt for doc in context["docs"]]


def inference_context(context, selected_rows):
    """Whitelist input fields; strip all scored rows' labels and incident roots.

    All scored labels, including unchecked labels on the same held-out document,
    are removed. Only explicitly unscored anchor rows may retain Gamma facts.
    Embedded Document.lbl, truth banks, and external approvals are not copied.
    """
    n, k = len(context["docs"]), len(context["labels"])
    selected = np.asarray(selected_rows)
    if selected.dtype != np.bool_ or selected.shape != (n,):
        raise ValueError("calibration row mask differs from full context")
    gp, gn = context["gamma"].pos.copy(), context["gamma"].neg.copy()
    if gp.shape != (n, k) or gn.shape != (n, k) or np.any(gp & gn):
        raise ValueError("calibration Gamma requires consistent complete signed axes")
    gp[selected] = False
    gn[selected] = False

    def allowed_edges(name):
        return tuple(
            (int(a), int(b))
            for a, b in getattr(context["gamma"], name, ())
            if not selected[a] and not selected[b]
        )

    gamma = RefGamma(gp, gn, allowed_edges("subset"), allowed_edges("strict_subset"))
    return dict(
        context_id=digest(
            dict(
                parent=context.get("context_id"),
                rows=context["row_ids"],
                masked_Gamma=seed_identity(gamma),
            )
        ),
        docs=[Document(cnt=d.cnt, ttl=d.ttl, mtd=d.mtd) for d in context["docs"]],
        labels=list(context["labels"]),
        row_ids=list(context["row_ids"]),
        row_roles=list(context["row_roles"]),
        gamma=gamma,
        model_probabilities={
            a: np.array(p, copy=True) for a, p in context["model_probabilities"].items()
        },
        static_model_probabilities={
            a: np.array(p, copy=True)
            for a, p in context.get(
                "static_model_probabilities", context["model_probabilities"]
            ).items()
        },
        model_decisions={
            a: np.array(p, copy=True) for a, p in context.get("model_decisions", {}).items()
        },
        attribute_values={a: list(v) for a, v in context.get("attribute_values", {}).items()},
        attribute_presence={
            a: np.array(p, copy=True) for a, p in context.get("attribute_presence", {}).items()
        },
    )


def count_rule_conclusions(
    runner, closure, rules, positive_answers, negative_answers, selected_rows, *, gamma=None
):
    """Distinct checkable target Label conclusions of each current rule.

    Label heads retain their immediate signed-label counting. Relation heads
    count the positive labels their primitive edges transfer (plus trusted
    Gamma transport), using exactly the runtime relation-source attribution.
    Only the target needs an observed answer; unknown labels are never false.
    """
    from loris._learning.incremental import state_from_result
    from loris._engine.typed_runtime import indices

    if closure.status != "fixpoint":
        raise ValueError("a failed/partial Chase cannot calibrate rule reliability")
    unknown, active_gamma = _quarantine_view(closure, runner.gamma if gamma is None else gamma)
    truth_pos, truth_neg = np.asarray(positive_answers), np.asarray(negative_answers)
    shape = (runner.n, runner.k)
    if (
        truth_pos.shape != shape
        or truth_neg.shape != shape
        or truth_pos.dtype != np.bool_
        or truth_neg.dtype != np.bool_
        or np.any(truth_pos & truth_neg)
    ):
        raise ValueError(
            "calibration answers require consistent explicit positive/negative boolean arrays"
        )
    selected = np.asarray(selected_rows)
    if selected.shape != (runner.n,) or selected.dtype != np.bool_:
        raise ValueError("calibration row selection differs from inference context")
    retained = _closure_chase_semantics(closure) in RETAINED_SEMANTICS
    observed = (truth_pos | truth_neg) & selected[:, None]
    scope = getattr(closure, "rule_conclusion_observed", None)
    if scope is not None and (scope.shape != shape or scope.dtype != np.bool_
                              or np.any(observed & ~scope)):
        raise ValueError("calibration observation exceeds the collected receiver scope")
    if not retained:
        observed &= ~unknown
    elif not getattr(closure, "rule_conclusions_enabled", False):
        raise ValueError("retained calibration requires actual fired conclusion records")
    state = state_from_result(runner, closure)
    positive_bits = _positive_row_bits(state.pos)
    blocked_bits = _positive_row_bits(unknown)
    relation_source_bits = _relation_source_bits(state, positive_bits, blocked_bits)
    gamma_edges = _gamma_successors(active_gamma)
    records = []
    for ri, rule in enumerate(rules):
        checked = {}
        if retained:
            conclusions = {}
            for sign, pairs in closure.rule_conclusions.get(ri, {}).items():
                if sign not in ("pos", "neg"):
                    raise ValueError("invalid fired conclusion sign")
                for row, col in pairs:
                    if not 0 <= row < runner.n or not 0 <= col < runner.k:
                        raise ValueError("fired conclusion is outside the calibration axes")
                    key = (sign, row)
                    conclusions[key] = conclusions.get(key, 0) | (1 << col)
            matches = getattr(closure, "rule_conclusion_instances", {}).get(ri)
        else:
            conclusions, matches = _direct_conclusions(
                runner,
                rule,
                ri,
                state,
                positive_bits,
                gamma_edges,
                transport_label_heads=False,
                blocked_bits=blocked_bits,
                relation_source_bits=relation_source_bits,
            )
        for (sign, row), labels in conclusions.items():
            for col in indices(labels):
                if observed[row, col]:
                    checked[(sign, row, col)] = bool(
                        (truth_pos if sign == "pos" else truth_neg)[row, col]
                    )
        n, c = len(checked), sum(checked.values())
        records.append(
            dict(
                rule_id=ri,
                rule_sha256=digest(rule.to_dict()),
                head_kind=rule.head.to_dict()["type"],
                conclusion_unit=(
                    "propagated_distinct_target_Label"
                    if isinstance(rule.head, LabelCompare)
                    else "distinct_signed_Label_head"
                ),
                support=n,
                correct=c,
                q=float(rule_score(n, c)),
                matching_instances=matches,
                conclusion_observation=(
                    "actual_fired_this_chase_including_later_unknown"
                    if retained
                    else "current_valid_instances_in_final_closure"
                ),
                checked_conclusions_id=digest(
                    sorted([list(key) + [value] for key, value in checked.items()])
                ),
            )
        )
        if scope is not None:
            records[-1]["matching_instance_policy"] = (
                "per_batch_firings_omitting_known_unobserved_Label_heads")
    return records


def validate_artifact(artifact, context, rules, manifest):
    values = validated_rule_scores(artifact)
    _validate_relation_scoring(rules, artifact)
    semantics = chase_semantics(manifest)
    _validate_chase_semantics(artifact, semantics)
    # Normalize only genuine historical strict bindings. New quarantine claims
    # require explicit markers in every authenticated evidence layer.
    binding = dict(artifact.get("binding", {}))
    if semantics == STRICT_CHASE_SEMANTICS:
        binding.setdefault("chase_semantics", STRICT_CHASE_SEMANTICS)
    calibration_binding = artifact.get("calibration_binding", {})
    for evidence_binding in (
        binding,
        calibration_binding,
        calibration_binding.get("model_binding", {}),
    ):
        if evidence_binding.get("chase_semantics", STRICT_CHASE_SEMANTICS) != semantics:
            raise ValueError(
                "independent calibration chase_semantics binding missing or incompatible"
            )
    if (
        len(values) != len(rules)
        or artifact.get("labels") != list(context["labels"])
        or binding != source_binding(context, rules, manifest)
        or [r["rule_sha256"] for r in artifact["rule_scores"]]
        != [digest(r.to_dict()) for r in rules]
    ):
        raise ValueError("independent reliability initial context/model/program binding mismatch")
    protocol_path = bound(artifact["protocol"], Path("."))
    protocol = validate_protocol(json.loads(protocol_path.read_text()), context, rules, manifest)
    if protocol["model_binding"].get("chase_semantics", STRICT_CHASE_SEMANTICS) != semantics:
        raise ValueError(
            "independent calibration protocol chase_semantics binding missing or incompatible"
        )
    for name in ("context", "answers"):
        bound(protocol[name], protocol_path.parent)
    receipt = json.loads(bound(artifact["estimation_receipt"], Path(".")).read_text())
    if (
        receipt.get("status") != "complete"
        or receipt.get("chase_semantics", STRICT_CHASE_SEMANTICS) != semantics
        or receipt.get("protocol") != artifact["protocol"]
        or receipt.get("binding_id") != digest(artifact["binding"])
        or receipt.get("rule_scores_id") != digest(artifact["rule_scores"])
        or receipt.get("calibration_binding") != artifact.get("calibration_binding")
        or receipt.get("answers_entered_inference") is not False
        or receipt.get("model_updates") != 0
        or receipt.get("rule_learning_calls") != 0
        or (
            any(isinstance(r.head, LabelCompare) for r in rules)
            and receipt.get("relation_scoring") != relation_scoring(semantics)
        )
        or (
            semantics in (QUARANTINE_CHASE_SEMANTICS,) + RETAINED_SEMANTICS
            and (
                receipt.get("arbitration_conclusions_used") is not False
                or receipt.get("calibration_inference")
                != (
                    "retention_without_arbitration"
                    if semantics in RETAINED_SEMANTICS
                    else "quarantine_without_arbitration"
                )
            )
        )
    ):
        raise ValueError("independent calibration receipt incomplete or incompatible")
    return artifact


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _calibrate(args):
    from .cli import load

    manifest, target, rules, _, load_receipt = load(
        args.manifest, weighting="uniform", ignore_reliability=True
    )
    protocol_path = Path(args.protocol).resolve(strict=True)
    protocol = validate_protocol(json.loads(protocol_path.read_text()), target, rules, manifest)
    context_path = bound(protocol["context"], protocol_path.parent)
    answers_path = bound(protocol["answers"], protocol_path.parent)
    with context_path.open("rb") as stream:
        calibration = pickle.load(stream)  # Bound, trusted-local prepared context.
    from .aux_extension import attach_context

    calibration = attach_context(calibration, manifest)
    n, k = len(calibration["docs"]), len(calibration["labels"])
    if (
        calibration["labels"] != target["labels"]
        or len(calibration["row_ids"]) != n
        or len(set(calibration["row_ids"])) != n
        or len(calibration["row_roles"]) != n
    ):
        raise ValueError("independent calibration full row/label axes differ")
    selected = np.isin(calibration["row_ids"], protocol["calibration_row_ids"])
    if selected.sum() != len(protocol["calibration_row_ids"]):
        raise ValueError("declared calibration row absent from calibration context")
    with np.load(answers_path, allow_pickle=False) as archive:
        if set(archive.files) != {"positive", "negative", "row_ids", "labels"}:
            raise ValueError("calibration NPZ requires exactly positive, negative, row_ids, labels")
        if (
            archive["row_ids"].tolist() != calibration["row_ids"]
            or archive["labels"].tolist() != calibration["labels"]
        ):
            raise ValueError("independent answer axes do not match calibration context")
        pos, neg = archive["positive"].copy(), archive["negative"].copy()
    if (
        pos.shape != (n, k)
        or neg.shape != (n, k)
        or pos.dtype != np.bool_
        or neg.dtype != np.bool_
        or np.any(pos & neg)
        or not np.any((pos | neg)[selected])
    ):
        raise ValueError(
            "calibration answers need observed, consistent positive/negative boolean facts"
        )
    return _estimate(
        manifest,
        target,
        rules,
        protocol,
        calibration,
        pos,
        neg,
        selected,
        Path(args.output),
        context_path,
        answers_path,
        load_receipt,
    )


def _estimate(
    manifest,
    target,
    rules,
    protocol,
    calibration,
    pos,
    neg,
    selected,
    output,
    context_path,
    answers_path,
    load_receipt,
    before_chase=None,
    prior_inference_receipt=None,
):
    n = len(calibration["docs"])
    semantics = chase_semantics(manifest)
    protocol = dict(
        protocol,
        model_binding=dict(protocol["model_binding"], chase_semantics=semantics),
    )
    independence = independent_content_check(target, calibration, selected)
    # The answer bank is outside this inference object and is never a trainer input.
    projected = inference_context(calibration, selected)
    if set(projected["model_probabilities"]) != set(target["model_probabilities"]):
        raise ValueError("independent calibration must contain predictions from every bound model")
    if set(projected["model_decisions"]) != set(target.get("model_decisions", {})):
        raise ValueError("independent calibration static model axis differs")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    binding = source_binding(target, rules, manifest)
    calibration_binding = dict(
        chase_semantics=semantics,
        context=file_record(context_path),
        answers=file_record(answers_path),
        row_ids_id=digest(calibration["row_ids"]),
        calibration_rows=protocol["calibration_row_ids"],
        model_binding=protocol["model_binding"],
        masked_Gamma=seed_identity(projected["gamma"]),
        content_independence=independence,
        static_model_tables={
            a: array_identity(p) for a, p in projected["static_model_probabilities"].items()
        },
        model_tables={a: array_identity(p) for a, p in projected["model_probabilities"].items()},
        model_decisions={a: array_identity(p) for a, p in projected["model_decisions"].items()},
    )
    # Store absolute authenticated references for later validation from any cwd.
    protocol = dict(protocol, context=file_record(context_path), answers=file_record(answers_path))
    _write(output / "PROTOCOL.json", protocol)
    _write(
        output / "INPUTS.json",
        dict(
            manifest=load_receipt,
            binding=binding,
            calibration_binding=calibration_binding,
            independence_is_declared_provenance_not_statistical_proof=True,
        ),
    )
    runner = make_discovery_runner(projected, output / "cache", subset_direction="standard")
    inference_cache = None
    inference_reused = False
    try:
        if before_chase is not None:
            before_chase()
        if semantics in RETAINED_SEMANTICS:
            from .unknown_incremental import retaining_chase
            from . import calibration_inference_cache as inference_cache_module

            inference_binding = inference_cache_module.dependency_binding(
                runner, projected, rules, projected["gamma"], pos, neg, selected, semantics,
            )
            inference_binding["calibration_configuration"] = manifest.get("calibration", {})
            cached = inference_cache_module.load_parent(prior_inference_receipt, inference_binding)
            if cached is None:
                closure = retaining_chase(
                    runner, rules, projected["gamma"], collect_rule_conclusions=True,
                    rule_conclusion_observed=(pos | neg) & selected[:, None],
                    gamma_wins=semantics == RETAINED_GAMMA_WINS_SEMANTICS,
                )
            else:
                closure, records, reused_from = cached
                inference_reused = True
        elif semantics == QUARANTINE_CHASE_SEMANTICS:
            from .quarantine import quarantine_chase

            # Actual current-model Chase, with no arbitration rule or seed.
            # Held-out gold is not available to this call.
            closure = quarantine_chase(runner, rules, projected["gamma"])
        else:
            closure = runner(rules, projected["gamma"])
        if closure.status != "fixpoint":
            result = dict(
                status="conflict_no_reliability_estimate",
                chase_semantics=semantics,
                complete_calibration_denominator_retained=True,
            )
            _write(output / "RESULTS.json", result)
            return 2
        if not inference_reused:
            records = count_rule_conclusions(
                runner, closure, rules, pos, neg, selected, gamma=projected["gamma"]
            )
        fallback_record = _fallback_calibration(manifest, target, projected, closure, pos, neg, selected)
        if semantics in RETAINED_SEMANTICS:
            inference_cache = inference_cache_module.store(
                output, inference_binding, closure, records,
                reused_from=reused_from if inference_reused else None,
            )
    finally:
        runner.close()
    fallback_file = None
    if fallback_record is not None:
        from . import fallback as fallback_module

        fallback_record["binding"] = fallback_module.record_binding(
            manifest, target["labels"], calibration_binding, binding)
        _write(output / "FALLBACK_CALIBRATION.json", fallback_record)
        # RESULTS.json is SHA-bound by RELIABILITY.json, so this pins the record.
        fallback_file = file_record(output / "FALLBACK_CALIBRATION.json")
    result = dict(
        status="complete",
        chase_semantics=semantics,
        calibration_inference=(
            "retention_without_arbitration"
            if semantics in RETAINED_SEMANTICS
            else (
                "quarantine_without_arbitration"
                if semantics == QUARANTINE_CHASE_SEMANTICS
                else "strict_chase"
            )
        ),
        arbitration_conclusions_used=False,
        conflict_unknown_pairs=int(
            np.asarray(
                getattr(closure, "conflict_unknown", np.zeros((n, len(target["labels"])), bool))
            ).sum()
        ),
        protocol=file_record(output / "PROTOCOL.json"),
        binding_id=digest(binding),
        calibration_binding=calibration_binding,
        rule_scores_id=digest(records),
        rule_scores=records,
        relation_scoring=relation_scoring(semantics),
        documents=n,
        calibration_documents=int(selected.sum()),
        checkable_label_cells=int(((pos | neg) & selected[:, None]).sum()),
        answers_entered_inference=False,
        model_updates=0,
        rule_learning_calls=0,
        zero_support_rules=sum(r["support"] == 0 for r in records),
        transfer_policy=TRANSFER_POLICY,
        truth_independence_is_provenance_declaration=True,
        fallback_calibration=fallback_file,
        rule_inference_cache=inference_cache,
        calibration_inference_execution=("verified_dependency_cache" if inference_reused else "fresh_chase"),
        fallback_and_gate_evaluated_for_current_manifest=True,
        conclusion_collection_scope=(
            dict(policy="observed_receivers_and_new_heads", observed=array_identity((pos | neg) & selected[:, None]),
                 non_observed_sources_retained=True, execution_counts_are_all_target=False)
            if semantics in RETAINED_SEMANTICS else None),
    )
    _write(output / "RESULTS.json", result)
    artifact = dict(
        schema=SCHEMA,
        chase_semantics=semantics,
        source_unit=UNIT,
        labels=list(target["labels"]),
        binding=binding,
        calibration_binding=calibration_binding,
        rule_scores=records,
        relation_scoring=relation_scoring(semantics),
        estimator=ESTIMATOR,
        protocol=file_record(output / "PROTOCOL.json"),
        estimation_receipt=file_record(output / "RESULTS.json"),
        transfer_policy=TRANSFER_POLICY,
    )
    validate_artifact(artifact, target, rules, manifest)
    _write(output / "RELIABILITY.json", artifact)
    _write(
        output / "COMPLETE.json",
        dict(status="complete", artifact=file_record(output / "RELIABILITY.json")),
    )
    print(json.dumps(dict(status="complete", artifact=str(output / "RELIABILITY.json")), indent=2))
    return 0


def _fallback_calibration(manifest, target, projected, closure, pos, neg, selected):
    """Undecided-cell fallback thresholds and the final-output gate score.

    Computed on the independent calibration documents from the same closure
    used for rule reliability; calibration answers never enter inference.
    """
    from . import fallback as fallback_module

    config = fallback_module.configuration(manifest)
    gate = manifest.get("loop", {}).get("commit_gate", "none") == "independent_calibration"
    if not gate and (config is None or not config["enabled"]
                     or config["calibration"] != "independent_calibration"):
        return None
    if config is None:
        config = dict(enabled=False, alias=manifest["dynamic"]["alias"],
                      calibration="fixed", default_threshold=1.0, thresholds={},
                      grid=[1.0], minimum_positive_support=1)
    labels = list(target["labels"])
    # V2's learned extension uses the new Aux, but original arbitration and
    # fallback rules retain their old alias and frozen thresholds.
    alias = manifest.get("aux_extension", {}).get(
        "source_alias", manifest["dynamic"]["alias"])
    unknown = np.asarray(getattr(closure, "conflict_unknown", np.zeros(pos.shape, bool)), bool)
    thresholds, score, audit = fallback_module.calibrate(
        positive=np.asarray(closure.positive_facts, bool),
        negative=np.asarray(closure.negative_facts, bool),
        unknown=unknown,
        probabilities=projected["model_probabilities"][alias],
        arbitration_kappa=fallback_module.arbitration_thresholds(manifest, labels),
        truth_positive=pos, truth_negative=neg, rows=selected, config=config, labels=labels,
    )
    if not config["enabled"] or config["calibration"] != "independent_calibration":
        # Gate only: score the final output with the declared (or no) fallback.
        thresholds = None
        score = fallback_module.score_output(
            positive=np.asarray(closure.positive_facts, bool),
            negative=np.asarray(closure.negative_facts, bool), unknown=unknown,
            probabilities=projected["model_probabilities"][alias],
            arbitration_kappa=fallback_module.arbitration_thresholds(manifest, labels),
            truth_positive=pos, truth_negative=neg, rows=selected,
            thresholds=(fallback_module.declared_thresholds(config, labels)
                        if config["enabled"] else None),
        )
        audit = dict(audit, thresholds_source="declared" if config["enabled"] else "no_fallback")
    return dict(schema=fallback_module.SCHEMA, thresholds=thresholds, gate=score, audit=audit,
                calibration_rows=int(np.sum(selected)), answers_entered_inference=False)


def load_fallback_record(artifact_record, artifact, context, rules, manifest, labels):
    """The FALLBACK_CALIBRATION.json written with a reliability artifact, verified.

    ``artifact_record`` is the {path, sha256} of RELIABILITY.json. The record
    must be the file pinned by the artifact's SHA-bound estimation receipt,
    and its binding must equal the one this run would write (2026-10-01).
    """
    from . import fallback as fallback_module

    receipt = json.loads(bound(artifact["estimation_receipt"], Path(".")).read_text())
    reference = receipt.get("fallback_calibration")
    expected_path = Path(artifact_record["path"]).parent / "FALLBACK_CALIBRATION.json"
    if not isinstance(reference, dict):
        raise ValueError("reliability calibration has no bound FALLBACK_CALIBRATION.json (created by "
                         "an older release or without fallback/commit gate); recalibrate: " + str(expected_path))
    path = bound(reference, Path("."))
    if path != expected_path.resolve():
        raise ValueError("bound FALLBACK_CALIBRATION.json is not the one next to the reliability artifact")
    record = json.loads(path.read_text())
    expected = fallback_module.record_binding(
        manifest, labels, artifact.get("calibration_binding", {}), source_binding(context, rules, manifest))
    fallback_module.check_record(record, expected, path)
    return record


def refresh_current(artifact, target, rules, manifest, prediction, output, *, before_chase=None):
    """Calibrate q on the candidate checkpoint/thresholds before installation.

    Every model input is predicted by the same already-fitted backend; gold is
    loaded here only. Static model tables retain the INITIAL calibration
    interpretation, while the Aux and its dependent ensembles are refreshed.
    The returned artifact binds the complete new calibration protocol and snapshot.
    """
    output = Path(output).resolve()
    _validate_chase_semantics(artifact, chase_semantics(manifest))
    if output.exists():
        raise FileExistsError("current-model calibration output already exists")
    protocol_path = bound(artifact["protocol"], Path("."))
    original = json.loads(protocol_path.read_text())
    context_path = bound(original["context"], protocol_path.parent)
    answers_path = bound(original["answers"], protocol_path.parent)
    with context_path.open("rb") as stream:
        calibration = pickle.load(stream)
    from .aux_extension import attach_context

    # The original calibration context has only the frozen source alias. Later
    # snapshots also contain the adaptive alias; preserve those current tables.
    calibration = attach_context(calibration, manifest, allow_existing=True)
    texts = [d.cnt for d in calibration["docs"]]
    if (
        not isinstance(prediction, dict)
        or prediction.get("rows") != len(texts)
        or prediction.get("texts_id") != digest(texts)
        or prediction.get("labels") != list(target["labels"])
        or prediction.get("checkpoint_sha256") != manifest["dynamic"]["checkpoint"]["sha256"]
        or prediction.get("supervision_used") is not False
    ):
        raise ValueError(
            "current-model calibration needs bound prediction-only output from the candidate checkpoint"
        )
    with np.load(bound(prediction["file"], Path(".")), allow_pickle=False) as archive:
        if set(archive.files) != {"probabilities"}:
            raise ValueError("independent prediction archive must contain only probabilities")
        probabilities = np.array(archive["probabilities"], copy=True)
    if (
        probabilities.shape != (len(texts), len(target["labels"]))
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("candidate independent predictions have invalid complete axes")
    selected = np.isin(calibration["row_ids"], original["calibration_row_ids"])
    if int(selected.sum()) != len(original["calibration_row_ids"]):
        raise ValueError("independent calibration row selection changed")
    with np.load(answers_path, allow_pickle=False) as archive:
        if (
            set(archive.files) != {"positive", "negative", "row_ids", "labels"}
            or archive["row_ids"].tolist() != calibration["row_ids"]
            or archive["labels"].tolist() != list(target["labels"])
        ):
            raise ValueError("independent answer identity changed during refinement")
        pos, neg = archive["positive"].copy(), archive["negative"].copy()
    projected = inference_context(calibration, selected)
    projected["model_probabilities"][manifest["dynamic"]["alias"]] = probabilities
    for spec in manifest["dynamic"]["ensemble_dependencies"]:
        projected["model_probabilities"][spec["alias"]] = np.mean(
            [
                np.asarray(projected["model_probabilities"][a], dtype=np.float64)
                >= spec["vote_threshold"]
                for a in spec["dependencies"]
            ],
            axis=0,
        ).astype(np.float32)
    projected["context_id"] = digest(
        dict(
            rows=projected["row_ids"],
            models={a: array_identity(p) for a, p in projected["model_probabilities"].items()},
            static_models={
                a: array_identity(p) for a, p in projected["static_model_probabilities"].items()
            },
            Gamma=seed_identity(projected["gamma"]),
        )
    )
    protocol = dict(
        original,
        model_binding=dict(
            checkpoint_sha256=manifest["dynamic"]["checkpoint"]["sha256"],
            probability_lineage=manifest["dynamic"]["probability_lineage"],
        ),
    )
    validate_protocol(protocol, target, rules, manifest)
    independence = independent_content_check(target, projected, selected)
    # Snapshot contains texts, model predictions and permitted anchor Gamma,
    # not calibration answers. Preserve its static tables through later rounds.
    snapshot = output.parent / (output.name + "_CONTEXT.pkl")
    with snapshot.open("xb") as stream:
        pickle.dump(projected, stream, protocol=pickle.HIGHEST_PROTOCOL)
    protocol["context"] = file_record(snapshot)
    protocol["answers"] = file_record(answers_path)
    status = _estimate(
        manifest,
        target,
        rules,
        protocol,
        projected,
        pos,
        neg,
        selected,
        output,
        snapshot,
        answers_path,
        dict(
            parent_artifact=artifact["estimation_receipt"],
            candidate_prediction=prediction,
            content_independence=independence,
        ),
        before_chase=before_chase,
        prior_inference_receipt=artifact["estimation_receipt"],
    )
    if status != 0:
        raise ValueError(
            "current-model independent calibration conflict; candidate must not be installed"
        )
    path = output / "RELIABILITY.json"
    updated = json.loads(path.read_text())
    validate_artifact(updated, target, rules, manifest)
    return updated, file_record(path)


def main_calibrate(args):
    """Write failure receipts and never overwrite an existing calibration run."""
    output = Path(args.output).resolve()
    existed = output.exists()
    start = time.monotonic()
    if existed:
        raise FileExistsError("calibration output already exists: " + str(output))
    try:
        return _calibrate(args)
    except BaseException as error:
        if output.is_dir():
            _write(
                output / "FAILED.json",
                dict(
                    status="incomplete_no_reliability_result",
                    exception=type(error).__name__,
                    message=str(error),
                    seconds=time.monotonic() - start,
                ),
            )
        raise
