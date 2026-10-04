"""Outer-approved exact rule-application scope, immutable during each Chase."""
import json
from types import MappingProxyType

import numpy as np

from .array_cache import sha
from .rdl_reference import prepare_attributes


def scope_corpus_identity(context):
    def plain(value):
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(x) for x in value]
        if isinstance(value, (np.ndarray, np.generic)):
            return plain(value.tolist())
        return value
    docs = tuple(context['docs'])
    values, presence = prepare_attributes(docs, context.get('attribute_values', {}),
                                         context.get('attribute_presence', {}))
    return sha(plain({'docs': [[d.cnt, d.ttl, d.mtd] for d in docs],
        'labels': list(context['labels']), 'attribute_values': values,
        'attribute_presence': {k: v.tolist() for k, v in presence.items()},
        'row_ids': context.get('row_ids'), 'identities': context.get('identities')}))


class FrozenInstanceScope:
    """Whole-instance decisions bound to the exact ordered program and corpus.

    Approval is a caller-owned input, never inferred by this class. Simulation
    provenance remains explicit. A surviving relation retains all consequences.
    """
    __slots__ = ('_json', '_blocked', '_rule_ids', '_corpus', '_id', '_checked_rules')

    def __init__(self, context, rules, decisions):
        rules = tuple(rules)
        rule_ids = tuple(sha(rule.to_dict()) for rule in rules)
        corpus = scope_corpus_identity(context)
        n = len(context['docs'])
        copied = json.loads(json.dumps(list(decisions), ensure_ascii=False, allow_nan=False))
        blocked = {}
        for record in copied:
            if record.get('status') != 'approved' or record.get('action') != 'disable_rule_instance':
                raise ValueError('only explicit approved whole-instance decisions enter a scope')
            for field in ('reason', 'approval_receipt_id', 'reviewer', 'authority_scope', 'parent_conflict_id'):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError('scope decision requires ' + field)
            if record['authority_scope'] != 'exact_instance_same_corpus_program':
                raise ValueError('unsupported scope authority or validity range')
            if record.get('corpus_id') != corpus or record.get('program_id') != sha([r.to_dict() for r in rules]):
                raise ValueError('scope decision corpus or program binding differs')
            index = record.get('rule_index')
            if type(index) is not int or not 0 <= index < len(rules) or record.get('rule_sha256') != rule_ids[index]:
                raise ValueError('scope decision refers to another rule')
            valuation = record.get('valuation')
            if (not isinstance(valuation, dict) or set(valuation) != set(rules[index].variables)
                    or any(type(row) is not int or not 0 <= row < n for row in valuation.values())):
                raise ValueError('scope requires the complete legal rule valuation')
            origin = record.get('provider')
            if origin not in ('human_review', 'expert_review', 'development_truth_review'):
                raise ValueError('source revision requires explicit reviewer provenance')
            if type(record.get('development_only')) is not bool or record['development_only'] != (origin == 'development_truth_review'):
                raise ValueError('simulated review must be declared development-only')
            key = (index, tuple(sorted(valuation.items())))
            if key in blocked:
                raise ValueError('duplicate disabled instance')
            blocked[key] = record['approval_receipt_id']
        body = {'schema': 'paper36_approved_exact_instance_scope_v1', 'corpus_id': corpus,
            'program_id': sha([rule.to_dict() for rule in rules]), 'rule_ids': list(rule_ids),
            'decisions': copied, 'relation_heads_atomic': True, 'inside_chase_scope_mutation': False}
        object.__setattr__(self, '_json', json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
        object.__setattr__(self, '_blocked', MappingProxyType(blocked))
        object.__setattr__(self, '_rule_ids', rule_ids)
        object.__setattr__(self, '_corpus', corpus)
        object.__setattr__(self, '_id', sha(body))
        # rule_index -> Rule object whose identity already matched (Rule is a
        # frozen dataclass); avoids re-hashing the rule for every instance.
        object.__setattr__(self, '_checked_rules', {})

    def __setattr__(self, name, value):
        raise AttributeError('rule application scope is immutable')

    @property
    def scope_id(self):
        return self._id

    def receipt(self):
        return dict(json.loads(self._json), scope_id=self._id)

    def validate(self, context, rules=None):
        if scope_corpus_identity(context) != self._corpus:
            raise ValueError('scope corpus identity differs')
        if rules is not None:
            self.validate_rules(rules)

    def validate_rules(self, rules):
        if tuple(sha(rule.to_dict()) for rule in rules) != self._rule_ids:
            raise ValueError('scope ordered program differs')

    def allows(self, rule_index, rule, valuation):
        if type(rule_index) is not int or not 0 <= rule_index < len(self._rule_ids):
            raise ValueError('scoped application requires an exact rule index')
        if self._checked_rules.get(rule_index) is not rule:
            if sha(rule.to_dict()) != self._rule_ids[rule_index]:
                raise ValueError('scoped application rule identity differs')
            self._checked_rules[rule_index] = rule
        if set(valuation) != set(rule.variables):
            raise ValueError('scoped application requires a full valuation')
        return (rule_index, tuple(sorted(valuation.items()))) not in self._blocked
