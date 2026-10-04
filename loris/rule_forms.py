"""Author-approved public rule sublanguage; historical Rule decoding stays open.

The 2026-09-22 clarification constrains rule forms, not predicate evaluation.
A cross-document body relation plus a single-label head needs at least one
single-label body fact. With no such premise, a cross rule must conclude a
label-set relation. Same-variable comparisons are not cross-document atoms.
"""

from collections import Counter

from loris._engine.rdl_reference import AttrEqual, Label, LabelCompare, Rule

SCHEMA = "loris_author_cross_label_rule_form_v1"
REJECTION = "author_rule_form:cross_single_label_head_requires_Label_body"
POLICY = {
    "schema": SCHEMA,
    "authority": "author clarification 2026-09-22",
    "cross_body_predicates": "AttrEqual or LabelCompare with distinct variable names",
    "cross_single_label_head": "requires at least one explicit Label premise of either sign",
    "cross_without_Label_premise": "requires LabelCompare head (eq/subset/strict_subset)",
    "Label_premise_location": "any body variable; no extra same-label or source-target condition inferred",
    "same_variable_comparisons": "not cross-document",
    "historical_decoding": "generic Rule/ReferenceChase unchanged; public admission enforces this sublanguage",
    "historical_invalid_program": "reject whole program; never silently delete rules",
}


class AuthorRuleFormError(ValueError):
    """A readable historical rule is outside the current public sublanguage."""

    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def _rule(value):
    if isinstance(value, Rule):
        return value
    if isinstance(value, dict):
        return Rule.from_dict(value)
    raise TypeError("typed Rule or serialized Rule dictionary required")


def inspect_author_rule_form(value):
    rule = _rule(value)
    cross = [
        i
        for i, atom in enumerate(rule.body)
        if isinstance(atom, (AttrEqual, LabelCompare)) and atom.left_var != atom.right_var
    ]
    labels = [i for i, atom in enumerate(rule.body) if isinstance(atom, Label)]
    valid = not cross or bool(labels) or isinstance(rule.head, LabelCompare)
    return dict(
        admissible=valid,
        reason=None if valid else REJECTION,
        cross_body_atom_indices=cross,
        single_label_body_atoms=len(labels),
        single_label_body_atom_indices=labels,
        head_kind=type(rule.head).__name__,
    )


def audit_program_rule_forms(rules):
    records = [dict(rule_index=i, **inspect_author_rule_form(rule)) for i, rule in enumerate(rules)]
    violations = [r for r in records if not r["admissible"]]
    return dict(
        schema=SCHEMA,
        authority=POLICY["authority"],
        policy=dict(POLICY),
        rules_checked=len(records),
        invalid_count=len(violations),
        cross_rules=sum(bool(r["cross_body_atom_indices"]) for r in records),
        violations=violations,
        rules_dropped=0,
    )


def validate_program_rule_forms(rules):
    audit = audit_program_rule_forms(rules)
    if audit["invalid_count"]:
        indices = [r["rule_index"] for r in audit["violations"]]
        raise AuthorRuleFormError(
            REJECTION
            + "; invalid rule indices="
            + repr(indices)
            + "; historical program was not modified; relearn under the declared public grammar",
            audit,
        )
    return audit


def validate_author_rule_form(value):
    rule = _rule(value)
    validate_program_rule_forms((rule,))
    return rule


def parse_author_rule(value):
    return validate_author_rule_form(value)


def trial_rule_form_counts(records):
    """Count explicit invalid BO receipts without dropping/resampling trials."""
    reasons = Counter(
        record.get("no_body_reason")
        for record in records
        if record.get("rule") is None
        and str(record.get("no_body_reason", "")).startswith(REJECTION)
    )
    return dict(
        schema=SCHEMA,
        rejected_candidates=sum(reasons.values()),
        rejection_reasons=dict(reasons),
        invalid_candidates_consume_budget=True,
        hidden_resampling=False,
    )
