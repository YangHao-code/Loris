"""History-retaining conflict-unknown maintenance, an author-added approximation.

All RDL and built-in candidates read a frozen batch. Conflicting label cells
lose both signs; previously derived descendants remain usable. Model-driven
deletion is different: invalid rule-instance/temporary heads and their recorded
descendants are withdrawn and rederived now. A newly false instance whose full
old body was valid triggers a fact-head event covering all non-retired versions
of its head, including versions hidden by unknown. Other recorded-source
invalidations remain version-specific. Both use versioned dependency edges,
Gamma protection and same-update rederivation. Historical witnesses are never
promoted to Gamma. This is not the original paper's monotone closure theorem.

Under the validated-Gamma-wins policy a derivation contradicting Gamma is
blocked instead of quarantined. When an update commits new Gamma cells, every
non-retired recorded version of a fact they refute (active or isolated) is
withdrawn with its recorded descendants. In a same-round check of unvalidated
candidate roots (an LLM proposal) the candidate cells are not protected: a
derivation contradicting one is a conflict and the cell becomes unknown. A
derivation contradicting validated Gamma is blocked; candidate_contradictions
lists the blocked ones that do not already follow from the checked fixed point.
"""

from collections import defaultdict, deque
from collections.abc import Mapping, MutableMapping
import copy
import hashlib
import json
import weakref
from types import MappingProxyType

import numpy as np

from loris._engine.hashing import _immutable_identity
from loris.chase import ChaseResult
from loris._engine.rdl_reference import AttrEqual, Label, LabelCompare, ML, RefGamma
from loris._engine.relation_rows import RelationRows
from loris._engine.typed_runtime import ProofDAG, RelationFacts, _plain, bits, indices, unbits
from loris._engine.typed_session import array_identity, digest, seed_identity, _json_default
from loris._learning.incremental import (
    EventState,
    _AttributeJoin,
    event_bindings,
    valuations,
    validate_static_context_prefix,
)
from loris._learning.rdg import RuleDependencyGraph


SEMANTICS = "author_history_retaining_synchronous_unknown_v1"
HISTORY_SCHEMA = "retained_one_hop_versioned_witnesses_v1"
COMPACT_HISTORY_SCHEMA = "retained_one_hop_implicit_gamma_witnesses_v2"
COMPACT_ROOT_THRESHOLD = 250000


_LABEL_COLUMN_CACHE = {}


def _label_columns(matrix, *, known=None):
    """Exact immutable column-bit cache; callers always own their mutable list."""
    binding = _immutable_identity(matrix)
    key = id(matrix)
    saved = _LABEL_COLUMN_CACHE.get(key) if binding is not None else None
    if (saved is not None and saved[0]() is matrix and
            saved[1][0] is binding[0] and saved[1][1:] == binding[1:]):
        return list(saved[2])
    # known is private, produced together with this exact frozen result matrix.
    columns = tuple(known) if known is not None else tuple(
        bits(matrix[:, j]) for j in range(matrix.shape[1]))
    if binding is not None:
        if len(_LABEL_COLUMN_CACHE) >= 8:
            _LABEL_COLUMN_CACHE.pop(next(iter(_LABEL_COLUMN_CACHE)))
        def expired(ref, key=key):
            current = _LABEL_COLUMN_CACHE.get(key)
            if current is not None and current[0] is ref:
                _LABEL_COLUMN_CACHE.pop(key, None)
        _LABEL_COLUMN_CACHE[key] = (weakref.ref(matrix, expired), binding, columns)
    return list(columns)


def _changed_cells(mask):
    """Skip full coordinate enumeration for an empty Boolean mask, preserving order."""
    return np.argwhere(mask) if np.any(mask) else np.empty((0, mask.ndim), dtype=np.intp)


def _mask(value, shape, name):
    if value is None:
        return np.zeros(shape, bool)
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(name + " requires the complete boolean document/label axis")
    return value.copy()


def _key(kind, row, column):
    return "%s:%d:%d" % (kind, row, column)


def _parts(key):
    kind, row, column = key.split(":")
    return kind, int(row), int(column)


def _string_list_json(values):
    """json.dumps(list of str, ensure_ascii=False, compact) as UTF-8 bytes."""
    if all(type(value) is str for value in values):
        # Printable ASCII without quote/backslash needs no JSON escaping. These
        # are per-character properties, so the concatenation decides them.
        joined = "".join(values)
        if joined.isascii() and joined.isprintable() and '"' not in joined and "\\" not in joined:
            return (('["' + '","'.join(values) + '"]') if values else "[]").encode()
    return json.dumps(values, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def _valuation_json(valuation):
    """Canonical JSON bytes of a rule valuation (same bytes as the json.dumps call)."""
    if type(valuation) is dict and all(type(k) is str and k.isascii() and k.isidentifier()
                                       and type(v) is int for k, v in valuation.items()):
        # Plain identifier keys need no JSON escaping; int rows print as str().
        return ("{" + ",".join('"%s":%d' % (k, valuation[k]) for k in sorted(valuation)) + "}").encode()
    return json.dumps(valuation or {}, default=_json_default, sort_keys=True,
                      ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def _head_keys(runner, rule, valuation):
    if isinstance(rule.head, Label):
        head = rule.head
        return (_key(head.polarity, valuation[head.var], runner.label_index[head.label]),)
    a, b = runner._edge(rule.head, valuation)
    kind = "strict_subset" if rule.head.op == "strict_subset" else "subset"
    pairs = ((a, b), (b, a)) if rule.head.op == "eq" else ((a, b),)
    return tuple(_key(kind, x, y) for x, y in pairs)


def _roots(gamma, n):
    result = {_key("subset", i, i) for i in range(n)}
    for sign, matrix in (("pos", gamma.pos), ("neg", gamma.neg)):
        result.update(_key(sign, i, j) for i, j in np.argwhere(matrix))
    for kind, edges in (("subset", gamma.subset), ("strict_subset", gamma.strict_subset)):
        result.update(_key(kind, i, j) for i, j in edges)
    return result


class _RootMembership:
    """Constant-space membership in Gamma and the implicit identity relation."""

    def __init__(self, gamma):
        self.gamma = gamma
        self.subset = frozenset(gamma.subset)
        self.strict = frozenset(gamma.strict_subset)

    def __contains__(self, key):
        kind, row, column = _parts(key)
        if kind in ("pos", "neg"):
            return bool(getattr(self.gamma, kind)[row, column])
        return ((row == column or (row, column) in self.subset) if kind == "subset"
                else (row, column) in self.strict)


class _Overlay(MutableMapping):
    """Private writes over an immutable baseline, with cheap round snapshots."""

    def __init__(self, base):
        if isinstance(base, _CompactMap):
            self.base, self.delta, self.deleted = base.base, dict(base.delta), set(base.deleted)
        else:
            self.base, self.delta, self.deleted = base, {}, set()
        self.size = len(base)

    def __getitem__(self, key):
        if key in self.delta:
            return self.delta[key]
        if key in self.deleted:
            raise KeyError(key)
        return self.base[key]

    def __setitem__(self, key, value):
        if key not in self:
            self.size += 1
        self.delta[key] = value
        self.deleted.discard(key)

    def __delitem__(self, key):
        self[key]
        self.size -= 1
        self.delta.pop(key, None)
        self.deleted.add(key)

    def __iter__(self):
        for key in self.base:
            if key not in self.deleted:
                yield key
        for key in self.delta:
            if key not in self.base:
                yield key

    def __len__(self):
        return self.size

    def snapshot(self):
        result = _Overlay(self.base)
        result.delta, result.deleted = self.delta.copy(), self.deleted.copy()
        result.size = self.size
        return result


class _CompactMap(Mapping):
    """Immutable explicit overrides of a virtual initial Gamma root map."""

    def __init__(self, overlay):
        self.base = overlay.base
        self.delta = MappingProxyType(dict(overlay.delta))
        self.deleted = frozenset(overlay.deleted)
        self.size = len(overlay)

    __getitem__ = _Overlay.__getitem__
    __iter__ = _Overlay.__iter__
    __len__ = _Overlay.__len__

    def compact(self):
        return dict(roots=self.base.roots.identity(), explicit=self.delta, removed=sorted(self.deleted))


class _ImplicitRoots:
    """Addressable, immutable original Gamma witnesses; no Python object per cell.

    Newly validated cells keep normal versioned nodes. Only the initial roots
    and identity edges use stable ``g:<fact>`` IDs, so changing model contexts
    cannot rewrite historical origins or parent references.
    """

    def __init__(self, gamma, context_id):
        self.gamma, self.context_id = gamma, context_id
        self.membership = _RootMembership(gamma)
        self.count = (gamma.shape[0] + int(gamma.pos.sum()) + int(gamma.neg.sum())
                      + sum(a != b for a, b in gamma.subset) + len(gamma.strict_subset))
        self._identity = dict(schema="implicit_initial_gamma_roots_v1", count=self.count,
                              gamma=seed_identity(gamma), created_context_id=context_id,
                              identity_relation=True, identifier_prefix="g:")

    def identity(self):
        return copy.deepcopy(self._identity)

    def __iter__(self):
        for row in range(self.gamma.shape[0]):
            yield _key("subset", row, row)
        for sign in ("pos", "neg"):
            matrix = getattr(self.gamma, sign)
            # Row iteration bounds scratch memory even for tens of millions of roots.
            for row in np.flatnonzero(matrix.any(1)):
                for column in np.flatnonzero(matrix[row]):
                    yield _key(sign, row, column)
        for kind in ("subset", "strict_subset"):
            for row, column in getattr(self.gamma, kind):
                if kind != "subset" or row != column:
                    yield _key(kind, row, column)

    def node(self, key):
        if key not in self.membership:
            raise KeyError(key)
        kind, row, column = _parts(key)
        return MappingProxyType(dict(
            fact=key, kind=kind, row=row, column=column,
            origin="identity" if kind == "subset" and row == column else "validated_Gamma",
            rule_id=None, valuation=MappingProxyType({}), static_evidence=(), dependencies=(),
            history_dependencies=(), created_context_id=self.context_id))


class _RootMap(Mapping):
    def __init__(self, roots, nodes=False):
        self.roots, self.nodes = roots, nodes

    def __len__(self):
        return self.roots.count

    def __iter__(self):
        for key in self.roots:
            yield "g:" + key if self.nodes else key

    def __getitem__(self, key):
        if self.nodes:
            if not key.startswith("g:"):
                raise KeyError(key)
            return self.roots.node(key[2:])
        if key not in self.roots.membership:
            raise KeyError(key)
        return "g:" + key


class _CompactHistoryDAG(ProofDAG):
    def __init__(self, overlay):
        self.mapping = _CompactMap(overlay)
        # Only explicit nodes require deep immutable conversion and byte caching.
        self.explicit = ProofDAG.adopt(overlay.delta)
        self.mapping.delta = self.explicit._derived
        self._identity = None

    def __getitem__(self, key):
        return self.mapping[key]

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def as_dict(self):
        """Explicit diagnostic expansion, never used by retained maintenance."""
        return dict(self.items())

    def compact(self):
        return self.mapping.compact()

    def identity(self):
        if self._identity is None:
            self._identity = dict(schema=COMPACT_HISTORY_SCHEMA, count=len(self),
                                  roots=self.mapping.base.roots.identity(),
                                  explicit=self.explicit.identity(), removed=sorted(self.mapping.deleted))
            self._identity["sha256"] = digest(self._identity)
        return copy.deepcopy(self._identity)


class _CompactFactDAG(ProofDAG):
    def __init__(self, history, latest):
        self.history, self.latest = history, latest

    def __getitem__(self, key):
        return self.history[self.latest[key]]

    def __iter__(self):
        return iter(self.latest)

    def __len__(self):
        return len(self.latest)

    def as_dict(self):
        return dict(self.items())

    def identity(self):
        body = dict(schema="implicit_gamma_latest_fact_proofs_v2", count=len(self),
                    history=self.history.identity(), latest=digest(self.latest.compact()))
        return dict(body, sha256=digest(body))

    def compact(self):
        return dict(schema="implicit_gamma_latest_fact_proofs_v2",
                    history="HISTORY.json.gz", latest=self.latest.compact())


def _explicit_history_items(nodes):
    """Initial roots never depend on a rule/version and need no DRed scan."""
    if isinstance(nodes, _Overlay) and isinstance(nodes.base, _RootMap):
        return nodes.delta.items()
    if isinstance(nodes, _CompactHistoryDAG):
        return nodes.mapping.delta.items()
    return nodes.items()


class _HistoryState(EventState):
    """Reuse finite comparison checking and adaptive rows, not eager propagation."""

    def __init__(self, runner, before=None, *, overlay=False):
        # The empty immutable seed belongs to this runner/axis. A storage/metadata
        # replacement invalidates it even when ndarray object identity is retained.
        cached = getattr(runner, "_retained_empty_seed", None)
        if cached is None or any(
                (binding := _immutable_identity(matrix)) is None or
                binding[0] is not previous[0] or binding[1:] != previous[1:]
                for matrix, previous in zip((cached[0].pos, cached[0].neg), cached[1])):
            empty = runner._gamma(RefGamma.empty(runner.n, runner.k))
            cached = (empty, tuple(_immutable_identity(m) for m in (empty.pos, empty.neg)))
            runner._retained_empty_seed = cached
        super().__init__(runner, cached[0])
        self.nodes = {}
        self.current = {}
        self.latest = {}
        self.retired = set()
        self.relation_sources = {}
        self.label_relation_sources = {}
        if before is None:
            for key, node in self.base_proof.items():
                self.record(key, dict(_plain(node), history_dependencies=[]))
        else:
            if getattr(before, "history_schema", None) not in (HISTORY_SCHEMA, COMPACT_HISTORY_SCHEMA):
                raise ValueError("retaining update requires an authenticated retaining history")
            nodes = before.history_nodes
            compact = isinstance(nodes, _CompactHistoryDAG)
            overlay = overlay or compact
            self.nodes = (_Overlay(nodes.mapping) if compact else _Overlay(nodes) if overlay else
                          nodes.as_dict() if isinstance(nodes, ProofDAG) else dict(nodes.items()))
            self.current = _Overlay(before.history_current) if overlay else dict(before.history_current)
            self.latest = _Overlay(before.history_latest) if overlay else dict(before.history_latest)
            self.retired = set(getattr(before, "history_retired", ()))
            self.relation_sources = dict(getattr(before, "relation_source_markers", {}))
            self.label_relation_sources = dict(getattr(before, "label_relation_markers", {}))
            self.pos = before.positive_facts.copy()
            self.neg = before.negative_facts.copy()
            self.label_bits = {
                sign: _label_columns(matrix)
                for sign, matrix in (("pos", before.positive_facts), ("neg", before.negative_facts))
            }
            self.out = RelationRows.from_rows(before.subset_facts.rows)
            self.proper = RelationRows.from_rows(before.strict_subset_facts.rows)
            self.inc, self.proper_inc = self.out.transpose(), self.proper.transpose()
        self.events.clear()

    def invalid(self):
        if self.proper.count() == 0:
            if self.last_valid_version != self.version:
                if np.any(self.pos & self.neg):
                    self.conflict = {"kind": "positive_negative_label_conflict"}
                self.last_valid_version = self.version
            return self.conflict is not None
        return super().invalid()

    def record(self, key, candidate):
        node = dict(candidate)
        kind, row, column = _parts(key)
        node.update(fact=key, kind=kind, row=row, column=column)
        node.setdefault("origin", "unknown")
        node.setdefault("rule_id", None)
        node.setdefault("valuation", {})
        node.setdefault("static_evidence", [])
        node.setdefault("dependencies", [])
        node.setdefault("history_dependencies", [])
        for parent in node["history_dependencies"]:
            if parent not in self.nodes:
                raise RuntimeError("candidate history parent is missing")
        # One-hop history is sufficient for event propagation and lazy source
        # validation. Do not duplicate a transitive rule set on every fact.
        node.pop("support_rule_ids", None)
        node["created_context_id"] = self.owner.context_id
        identifier = "h%d" % len(self.nodes)
        self.nodes[identifier] = node
        self.latest[key] = identifier
        self.current[key] = identifier
        return identifier

    def present(self, key):
        kind, row, column = _parts(key)
        if kind in ("pos", "neg"):
            return bool((self.pos if kind == "pos" else self.neg)[row, column])
        relation = self.out if kind == "subset" else self.proper
        return bool(relation[row] & (1 << column))

    def insert(self, key, candidate):
        if self.present(key):
            if candidate["origin"] == "validated_Gamma":
                current = self.nodes[self.current[key]]
                if current["origin"] not in ("validated_Gamma", "identity"):
                    self.record(key, candidate)
            return False
        kind, row, column = _parts(key)
        if kind in ("pos", "neg"):
            (self.pos if kind == "pos" else self.neg)[row, column] = True
            self.label_bits[kind][column] |= 1 << row
        else:
            out, inc = (self.out, self.inc) if kind == "subset" else (self.proper, self.proper_inc)
            out[row] |= 1 << column
            inc[column] |= 1 << row
            self.relation_version += 1
        self.record(key, candidate)
        self.version += 1
        return True

    def remove(self, key):
        if not self.present(key):
            self.current.pop(key, None)
            return False
        kind, row, column = _parts(key)
        if kind in ("pos", "neg"):
            (self.pos if kind == "pos" else self.neg)[row, column] = False
            self.label_bits[kind][column] &= ~(1 << row)
            self.label_relation_sources.pop(key, None)
        else:
            out, inc = (self.out, self.inc) if kind == "subset" else (self.proper, self.proper_inc)
            out[row] &= ~(1 << column)
            inc[column] &= ~(1 << row)
            self.relation_version += 1
            self.relation_sources.pop(key, None)
        self.current.pop(key, None)
        self.version += 1
        return True

    def snapshot(self, *, facts=None):
        state = copy.copy(self)
        state.pos, state.neg = facts if facts is not None else (self.pos.copy(), self.neg.copy())
        state.label_bits = {sign: tuple(value) for sign, value in self.label_bits.items()}
        for name in ("out", "inc", "proper", "proper_inc"):
            setattr(state, name, getattr(self, name).copy(frozen=True))
        state.current = (self.current.snapshot() if isinstance(self.current, _Overlay)
                         else dict(self.current))
        state.relation_sources = dict(self.relation_sources)
        state.label_relation_sources = dict(self.label_relation_sources)
        return state

    def relation_markers(self, key):
        kind, a, b = _parts(key)
        if kind == "subset" and a == b:
            return frozenset((-1,))
        return self.relation_sources.get(key, frozenset())


class _Candidates:
    """One deterministic first witness per new head; no grounding-sized buffer."""

    def __init__(self, state, unknown, rules, ledger):
        self.state, self.unknown, self.rules = state, unknown, rules
        self.ledger = ledger
        # key -> [witness order or None, origin, dependencies, rule_id,
        # valuation, static evidence or None, runner]. The order of a key's
        # first candidate is computed only when a second candidate for the same
        # key arrives; the winner (smallest order, earliest on ties) and its
        # static evidence are exactly those of eager evaluation (2026-09-30).
        self._entries = {}
        self._values = None
        self.relation_updates = defaultdict(set)
        self.label_updates = defaultdict(set)
        self.attempts = 0
        self.rule_sources = tuple(
            b',"source":' + json.dumps(rule.to_dict(), default=_json_default,
                sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
            + b',"valuation":' for rule in rules
        )
        self.dependency_atoms = tuple(
            tuple(atom for atom in rule.body if isinstance(atom, (Label, LabelCompare)))
            for rule in rules
        )
        self.label_dependency_specs = {}
        self.rule_prefixes = {}
        # Sorted plain-identifier variables and the canonical JSON format of a
        # valuation of them with int rows ('{"x":%d,"y":%d}').
        self.valuation_specs = tuple(
            (tuple(sorted(rule.variables)),
             "{" + ",".join('"%s":%%d' % var for var in sorted(rule.variables)) + "}")
            if all(type(var) is str and var.isascii() and var.isidentifier() for var in rule.variables)
            else None
            for rule in rules
        )

    def witness_order(self, rule_id, origin, valuation, dependencies):
        if rule_id is None:
            return digest(dict(source=origin, valuation=valuation or {}, dependencies=dependencies))
        # SHA-256 of the canonical bytes of digest(dict(dependencies=...,
        # source=rule, valuation=...)): sorted keys, compact separators. The
        # hash state after the dependency/rule prefix is reused while a rule's
        # dependencies repeat (always, for rules without Label premises).
        saved = self.rule_prefixes.get(rule_id)
        if saved is None or saved[0] != dependencies:
            prefix = hashlib.sha256(
                b'{"dependencies":' + _string_list_json(dependencies) + self.rule_sources[rule_id])
            self.rule_prefixes[rule_id] = (dependencies, prefix)
        else:
            prefix = saved[1]
        order = prefix.copy()
        order.update(self.valuation_bytes(rule_id, valuation) + b'}')
        return order.hexdigest()

    def valuation_bytes(self, rule_id, valuation):
        """_valuation_json(valuation) via a per-rule format for int rows."""
        spec = self.valuation_specs[rule_id]
        if spec is not None and type(valuation) is dict and len(valuation) == len(spec[0]):
            try:
                values = tuple([valuation[var] for var in spec[0]])
            except KeyError:
                values = None
            if values is not None and all(type(value) is int for value in values):
                return (spec[1] % values).encode()
        return _valuation_json(valuation)

    def rule_dependencies(self, runner, rule_id, valuation):
        dependencies = []
        for atom in self.dependency_atoms[rule_id]:
            if isinstance(atom, Label):
                dependencies.append(_key(atom.polarity, valuation[atom.var], runner.label_index[atom.label]))
            else:
                source, target = runner._edge(atom, valuation)
                dependencies.append(_key('strict_subset' if atom.op == 'strict_subset' else 'subset', source, target))
                if atom.op == 'eq':
                    dependencies.append(_key('subset', target, source))
        return dependencies

    def label_dependencies(self, runner, rule_id, valuation):
        """rule_dependencies() with a precomputed format for Label-only premises."""
        specs = self.label_dependency_specs.get(rule_id, False)
        if specs is False:
            atoms = self.dependency_atoms[rule_id]
            specs = self.label_dependency_specs[rule_id] = (
                tuple((atom.polarity, atom.var, runner.label_index[atom.label]) for atom in atoms)
                if all(isinstance(atom, Label) for atom in atoms) else None
            )
        if specs is None:
            return self.rule_dependencies(runner, rule_id, valuation)
        return ["%s:%d:%d" % (sign, valuation[var], column) for sign, var, column in specs]

    def add_rule_label(self, key, dependencies, rule_id, valuation, runner):
        """add(key, "rule_application", ...) for one Label-head rule instance.

        Only for callers that established add()'s early exits cannot apply: no
        conclusion ledger, the head cell is not unknown, and the head is absent
        from the frozen snapshot (valuations(known_head=False) prunes present
        heads). The recorded candidate and counters are those of add().
        """
        self.attempts += 1
        dependencies = tuple(dict.fromkeys(dependencies))
        current = self.state.current
        for parent in dependencies:
            if parent not in current:
                raise RuntimeError("synchronous candidate has an inactive premise")
        entry = self._entries.get(key)
        if entry is None:
            self._entries[key] = [None, "rule_application", dependencies, rule_id, valuation, None,
                                  runner]
            self._values = None
            return
        if entry[0] is None:
            entry[0] = self.witness_order(entry[3], entry[1], entry[4], entry[2])
        order = self.witness_order(rule_id, "rule_application", valuation, dependencies)
        if entry[0] <= order:
            return
        self._entries[key] = [order, "rule_application", dependencies, rule_id, valuation, None, runner]
        self._values = None

    def add(
        self, key, origin, dependencies=(), rule_id=None, valuation=None, static=(), markers=(),
        runner=None
    ):
        self.attempts += 1
        kind, row, column = _parts(key)
        if kind in ("pos", "neg") and self.unknown[row, column]:
            return
        if self.ledger is not None and rule_id is not None and kind in ("pos", "neg"):
            self.ledger[rule_id][kind].add((row, column))
        if self.ledger is not None and kind in ("subset", "strict_subset"):
            self.relation_updates[key].update(markers)
            if rule_id is not None:
                self.relation_updates[key].add(rule_id)
        if self.state.present(key):
            return
        dependencies = tuple(dict.fromkeys(dependencies))
        if any(parent not in self.state.current for parent in dependencies):
            raise RuntimeError("synchronous candidate has an inactive premise")
        valuation = dict(valuation or {})
        entry = self._entries.get(key)
        if entry is None:
            self._entries[key] = [None, origin, dependencies, rule_id, valuation, static, runner]
            self._values = None
            return
        if entry[0] is None:
            entry[0] = self.witness_order(entry[3], entry[1], entry[4], entry[2])
        order = self.witness_order(rule_id, origin, valuation, dependencies)
        if entry[0] <= order:
            return
        self._entries[key] = [order, origin, dependencies, rule_id, valuation, static, runner]
        self._values = None

    @property
    def values(self):
        """Winning candidate per key, materialized in first-candidate key order."""
        if self._values is None:
            values = {}
            for key, entry in self._entries.items():
                _, origin, dependencies, rule_id, valuation, static, runner = entry
                if static is None:
                    static = entry[5] = runner._evidence(self.rules[rule_id], valuation)[1]
                values[key] = dict(
                    origin=origin,
                    dependencies=list(dependencies),
                    history_dependencies=[self.state.current[parent] for parent in dependencies],
                    rule_id=rule_id,
                    valuation=dict(valuation),
                    static_evidence=list(static),
                )
            self._values = values
        return self._values

    def copy_label(self, target, relation, source):
        _, row, label = _parts(target)
        if self.unknown[row, label]:
            return
        if self.ledger is not None:
            markers = self.state.relation_markers(relation)
            responsible = set(markers) - {-1}
            if -1 in markers:
                responsible.update(self.state.label_relation_sources.get(source, ()))
            self.label_updates[target].update(responsible)
            for ri in responsible:
                self.ledger[ri]["pos"].add((row, label))
        self.add(target, "inclusion_label_copy", (relation, source))


def _head_rule_index(runner, rules):
    indexed = defaultdict(list)
    for ri, rule in enumerate(rules):
        head = rule.head
        if isinstance(head, Label):
            indexed[(head.polarity, runner.label_index[head.label])].append((ri, rule))
    return indexed


def _derivable(runner, rules, prepared, state, key, *, head_rules=None):
    """Whether the signed label fact ``key`` follows in one step from ``state``.

    One application of a rule with this signed Label head whose premises all
    hold in state, or one positive inclusion copy from a subset predecessor
    (strict inclusion implies inclusion in a closed state). Used on the fixed
    point a check starts from: a Gamma-blocked derivation that follows from it
    was blocked there already, independently of the checked candidate.
    """
    kind, row, column = _parts(key)
    relevant = (enumerate(rules) if head_rules is None else
                head_rules.get((kind, column), ()))
    for ri, rule in relevant:
        head = rule.head
        if (not isinstance(head, Label) or head.polarity != kind
                or runner.label_index[head.label] != column):
            continue
        for _ in valuations(runner, rule, prepared[ri], state, fixed={head.var: row},
                            known_head=True, rule_index=ri):
            return True
    if kind == "pos":
        for source in indices(state.inc[row] & ~(1 << row)):
            if state.pos[source, column]:
                return True
    return False


def _compose(left, right):
    # A later learned relation owns the transported conclusion; a Gamma-only
    # suffix transports the already assigned relation source unchanged.
    value = set(right) - {-1}
    if -1 in right:
        value.update(left)
    return value


def _all_events(state):
    # This stream is used only for built-ins. Explicit negatives and positives
    # without a nonidentity outgoing relation cannot trigger a built-in.
    for row in state.out.nonidentity_sources():
        for column in np.flatnonzero(state.pos[row]):
            yield "pos", row, int(column)
        for target in indices(state.out[row] & ~(1 << row)):
            yield "subset", row, target
    for row, targets in enumerate(state.proper):
        for target in indices(targets):
            yield "strict_subset", row, target


def _builtins(state, events, candidates):
    for kind, a, b in events:
        key = _key(kind, a, b)
        if not state.present(key):
            continue
        if kind == "pos":
            for target in indices(state.out[a] & ~(1 << a)):
                candidates.copy_label(_key("pos", target, b), _key("subset", a, target), key)
        elif kind in ("subset", "strict_subset"):
            if kind == "strict_subset":
                candidates.add(
                    _key("subset", a, b),
                    "proper_implies_subset",
                    (key,),
                    markers=state.relation_markers(key),
                )
            for label in np.flatnonzero(state.pos[a]):
                candidates.copy_label(_key("pos", b, label), key, _key("pos", a, label))
            for source in indices(state.inc[a]):
                candidates.add(
                    _key(kind, source, b),
                    (
                        "inclusion_transitivity"
                        if kind == "subset"
                        else "proper_inclusion_transitivity"
                    ),
                    (_key("subset", source, a), key),
                    markers=_compose(
                        state.relation_markers(_key("subset", source, a)),
                        state.relation_markers(key),
                    ),
                )
            for target in indices(state.out[b]):
                candidates.add(
                    _key(kind, a, target),
                    (
                        "inclusion_transitivity"
                        if kind == "subset"
                        else "proper_inclusion_transitivity"
                    ),
                    (key, _key("subset", b, target)),
                    markers=_compose(
                        state.relation_markers(key),
                        state.relation_markers(_key("subset", b, target)),
                    ),
                )
            if kind == "subset":
                for source in indices(state.proper_inc[a]):
                    candidates.add(
                        _key("strict_subset", source, b),
                        "proper_inclusion_transitivity",
                        (_key("strict_subset", source, a), key),
                        markers=_compose(
                            state.relation_markers(_key("strict_subset", source, a)),
                            state.relation_markers(key),
                        ),
                    )
                for target in indices(state.proper[b]):
                    candidates.add(
                        _key("strict_subset", a, target),
                        "proper_inclusion_transitivity",
                        (key, _key("strict_subset", b, target)),
                        markers=_compose(
                            state.relation_markers(key),
                            state.relation_markers(_key("strict_subset", b, target)),
                        ),
                    )


def _commit_markers(state, candidates):
    events = set()
    for updates, saved in (
        (candidates.relation_updates, state.relation_sources),
        (candidates.label_updates, state.label_relation_sources),
    ):
        for key, markers in updates.items():
            if not state.present(key) or not markers:
                continue
            previous = saved.get(key, frozenset())
            new = previous | frozenset(markers)
            if new != previous:
                saved[key] = new
                events.add(_parts(key))
    return events


def _commit(state, values, unknown, protected=None, blocked=None):
    """Insert candidate facts; derived-vs-derived contradictions become unknown.

    ``protected`` = (Gamma positive, Gamma negative) enables validated Gamma
    wins: a derived label contradicting a validated Gamma cell is blocked and
    counted in state.gamma_blocked, so a Gamma cell is never quarantined.
    ``blocked``, when given, also collects the keys blocked by this call.
    """
    events = []
    for key in sorted(values):
        kind, row, column = _parts(key)
        if protected is not None and kind in ("pos", "neg") and (
                protected[1 if kind == "pos" else 0][row, column]):
            state.gamma_blocked_cells.add(key)
            if blocked is not None:
                blocked.add(key)
            continue
        if kind in ("pos", "neg") and unknown[row, column]:
            # Preserve roots for audit without making them execution facts.
            if key not in state.latest:
                state.record(key, values[key])
                state.current.pop(key, None)
            continue
        if state.insert(key, values[key]):
            events.append(_parts(key))
    conflicts = state.pos & state.neg & ~unknown
    for row, column in _changed_cells(conflicts):
        state.remove(_key("pos", row, column))
        state.remove(_key("neg", row, column))
    unknown |= conflicts
    events = [event for event in events if state.present(_key(*event))]
    return events, _changed_cells(conflicts).tolist()


def _root_candidates(gamma, temporary, n, roots=None):
    values = {}
    for sign, matrix in (("pos", temporary[0]), ("neg", temporary[1])):
        for row, column in np.argwhere(matrix):
            values[_key(sign, row, column)] = dict(origin="temporary_seed", persistent_Gamma=False)
    for key in (_roots(gamma, n) if roots is None else roots):
        kind, row, column = _parts(key)
        values[key] = dict(
            origin="identity" if kind == "subset" and row == column else "validated_Gamma"
        )
    return values


def _updated_root_candidates(gamma, temporary, state, previous_gamma):
    """Only missing roots and newly validated sources need root processing.

    Existing Gamma roots already have trusted provenance. A newly validated
    root may already be derived, so it must still replace that provenance.
    Missing old roots are reconsidered after withdrawals or unlocking unknown.
    """
    values = {}
    for sign, matrix, present in (("pos", temporary[0], state.pos),
                                  ("neg", temporary[1], state.neg)):
        for row, column in _changed_cells(matrix & ~present):
            values[_key(sign, row, column)] = dict(origin="temporary_seed", persistent_Gamma=False)
    for sign, present in (("pos", state.pos), ("neg", state.neg)):
        matrix = getattr(gamma, sign)
        changed = matrix & (~getattr(previous_gamma, sign) | ~present)
        for row, column in _changed_cells(changed):
            values[_key(sign, row, column)] = dict(origin="validated_Gamma")
    for kind, edges in (("subset", gamma.subset), ("strict_subset", gamma.strict_subset)):
        previous = set(getattr(previous_gamma, kind))
        for row, column in edges:
            key = _key(kind, row, column)
            if (row, column) not in previous or not state.present(key):
                values[key] = dict(origin="identity" if kind == "subset" and row == column
                                   else "validated_Gamma")
    return values


def _static_instance_valid(runner, rule, prepared, valuation, ri, masks=None):
    """``masks`` optionally caches each prepared domain as a Boolean row mask
    (the same membership test without per-instance big-integer shifts)."""
    if masks is None:
        return (
            set(valuation) == set(rule.variables)
            and runner.application_allowed(ri, rule, valuation)
            and all(prepared["domains"][var] & (1 << int(row)) for var, row in valuation.items())
        )
    if set(valuation) != set(rule.variables) or not runner.application_allowed(ri, rule, valuation):
        return False
    for var, row in valuation.items():
        mask = masks.get((ri, var))
        if mask is None:
            mask = masks[(ri, var)] = unbits(prepared["domains"][var], runner.n)
        row = int(row)
        if row < 0:
            raise ValueError("negative shift count")  # as 1 << row
        # A row beyond the axis has no bit in the domain (same as the shift test).
        if row >= len(mask) or not mask[row]:
            return False
    return True


def _failed_old_instances(old_runner, runner, old_rule, rule, old_prepared, prepared, old_state):
    """(count, head keys) of the old valuations that are no longer statically valid.

    Exactly the result of enumerating valuations(old_runner, old_rule, ...,
    old_state, known_head=True) and testing _static_instance_valid(runner,
    rule, prepared, ...) on each, computed from row sets for Label-head rules
    with one variable and no binary atom, or two variables joined by a single
    AttrEqual, without application scopes. Returns None otherwise.
    """
    if old_runner._application_scope is not None or runner._application_scope is not None:
        return None
    head = old_rule.head
    variables = old_rule.variables
    if not isinstance(head, Label) or rule.variables != variables:
        return None
    binary = old_prepared["binary"]
    column = old_runner.label_index[head.label]

    def enumerated(var):
        # valuations(...) base domain with known_head=True: no head pruning.
        domain = old_prepared["domains"][var]
        for atom in old_prepared["unary"][var]:
            domain &= old_state.label_bits[atom.polarity][old_runner.label_index[atom.label]]
        return domain

    if len(variables) == 1 and not binary:
        var = variables[0]
        failed = enumerated(var) & ~prepared["domains"][var]
        rows = list(indices(failed))
        return len(rows), {_key(head.polarity, row, column) for row in rows}
    if not (len(variables) == 2 and len(binary) == 1 and type(binary[0]) is AttrEqual
            and {binary[0].left_var, binary[0].right_var} == set(variables)):
        return None
    first, second = variables
    join = _AttributeJoin(old_runner, variables, binary[0], enumerated(first), enumerated(second))
    if not join.first_domain or not join.second_domain:
        return 0, set()
    valid_first = unbits(prepared["domains"][first], runner.n).tolist()
    valid_second = unbits(prepared["domains"][second], runner.n).tolist()
    buckets = join.buckets()
    count = 0
    heads = set()
    for code, first_rows in join.first_members().items():
        second_rows = buckets.get(code)
        if not second_rows:
            continue
        good_first = [x for x in first_rows if valid_first[x]]
        good_second = [y for y in second_rows if valid_second[y]]
        count += len(first_rows) * len(second_rows) - len(good_first) * len(good_second)
        # A head row fails iff it has a failing pair: its own row is invalid,
        # or some partner row on the other side is invalid.
        if head.var == first:
            rows = first_rows if len(good_second) < len(second_rows) else \
                [x for x in first_rows if not valid_first[x]]
        else:
            rows = second_rows if len(good_first) < len(first_rows) else \
                [y for y in second_rows if not valid_second[y]]
        heads.update(_key(head.polarity, row, column) for row in rows)
    return count, heads


def _changed_rules(old_runner, runner, old_rules, rules):
    if old_runner is None or old_rules is None:
        raise ValueError("chaseDelta requires old_runner and old_rules")
    if (
        len(old_rules) != len(rules)
        or old_runner.n != runner.n
        or old_runner.labels != runner.labels
    ):
        raise ValueError("model update must preserve rule/document/label axes")
    old_runner.verify_code()
    validate_static_context_prefix(old_runner, runner)
    changed = set()
    for ri, (old, new) in enumerate(zip(old_rules, rules)):
        if old.head != new.head or len(old.body) != len(new.body):
            raise ValueError("model update cannot change rule structure")
        for oa, na in zip(old.body, new.body):
            if isinstance(oa, ML) and isinstance(na, ML):
                od, nd = oa.to_dict(), na.to_dict()
                od.pop("eta")
                nd.pop("eta")
                if od != nd or (not oa.dynamic and oa.eta != na.eta):
                    raise ValueError("only dynamic thresholds may change")
                if not oa.dynamic and not np.array_equal(
                    old_runner.static_probabilities[oa.alias], runner.static_probabilities[na.alias]
                ):
                    raise ValueError("static model probabilities changed")
                if old_runner._static(oa) != runner._static(na):
                    changed.add(ri)
            elif oa != na:
                raise ValueError("model update cannot change non-ML premises")
    if old_runner._application_scope is not None or runner._application_scope is not None:
        changed.update(range(len(rules)))
    return changed


def _withdraw(state, identifiers, protected):
    if not identifiers:
        # Nothing to withdraw; skip building the O(history) consumer map.
        return set(), 0
    consumers = defaultdict(set)
    for identifier, node in _explicit_history_items(state.nodes):
        for parent in node.get("history_dependencies", ()):
            consumers[parent].add(identifier)
    withdrawn = set()
    visited = set()
    queue = deque(identifiers)
    while queue:
        identifier = queue.popleft()
        if identifier in visited or identifier not in state.nodes:
            continue
        visited.add(identifier)
        key = state.nodes[identifier]["fact"]
        if key in protected:
            continue
        state.retired.add(identifier)
        if state.current.get(key) == identifier:
            withdrawn.add(key)
        queue.extend(consumers[identifier])
    removed = sum(state.remove(key) for key in withdrawn)
    return withdrawn, removed


def retaining_chase(
    runner,
    rules,
    gamma,
    *,
    before=None,
    old_runner=None,
    old_rules=None,
    old_gamma=None,
    update_kind=None,
    initial_positive=None,
    initial_negative=None,
    initial_unknown=None,
    execution="incremental",
    collect_rule_conclusions=False,
    rule_conclusion_observed=None,
    gamma_wins=False,
    check_candidates=None,
    decision_only=False,
    root_storage="auto",
):
    """Maintain the same retained history in incremental or full-scan mode.

    ``chaseDelta`` begins a new model round and resets U unless explicitly
    supplied. ``chaseInc``/``check`` retain U and temporary roots by default.
    The caller evaluates explicit arbitration rules and supplies temporary root
    masks; this function never predicts, trains, or changes persistent Gamma.
    Fresh calibration may enable ``collect_rule_conclusions``: this separately
    propagates relation-source markers and records every distinct signed target
    actually produced, including targets later isolated. It is not needed for
    inference and is deliberately disabled on the main incremental path.
    ``rule_conclusion_observed`` optionally scopes calibration's known Label
    heads to observed receiver cells. Unestablished heads and every source
    variable remain unrestricted, so inference and observed conclusion counts
    are unchanged. Non-observed already-known firings and their diagnostics
    are deliberately omitted; this is not a complete all-target ledger.
    ``check_candidates`` = (positive, negative) masks of the Gamma cells under
    validation in a same-round check under validated Gamma wins. They are not
    protected (contradictions with them become conflict_unknown cells), and the
    result's candidate_contradictions lists the Gamma-blocked derivations that
    do not follow from the checked state ``before``.
    """
    if type(collect_rule_conclusions) is not bool:
        raise ValueError("collect_rule_conclusions must be boolean")
    if root_storage not in ("auto", "explicit", "compact"):
        raise ValueError("root_storage must be auto, explicit or compact")
    if type(gamma_wins) is not bool:
        raise ValueError("gamma_wins must be boolean")
    if type(decision_only) is not bool or (decision_only and (
            before is None or update_kind != "check" or collect_rule_conclusions)):
        raise ValueError("decision_only requires a same-round check of a retained baseline")
    if before is not None and bool(getattr(before, "gamma_wins", False)) != gamma_wins:
        raise ValueError("retained history was built under a different Gamma conflict policy")
    if collect_rule_conclusions and before is not None:
        raise ValueError("actual conclusion calibration collection requires a fresh invocation")
    if rule_conclusion_observed is not None and not collect_rule_conclusions:
        raise ValueError("observed conclusion scope requires calibration collection")
    if execution not in ("incremental", "full_recompute"):
        raise ValueError("execution must be incremental or full_recompute")
    if (before is None and update_kind not in (None, "fresh")) or (
        before is not None and update_kind not in ("chaseInc", "chaseDelta", "check")
    ):
        raise ValueError("explicit compatible retaining update kind required")
    runner.verify_code()
    rules = tuple(rules)
    if runner._application_scope is not None:
        runner._application_scope.validate_rules(rules)
    gamma = runner._gamma(gamma)
    compact_roots = bool(isinstance(getattr(before, "history_nodes", None), _CompactHistoryDAG)
                     if before is not None else root_storage == "compact" or (
                         root_storage == "auto" and gamma.pos.sum() + gamma.neg.sum()
                         + runner.n >= COMPACT_ROOT_THRESHOLD))
    gamma_id = seed_identity(gamma)
    shape = (runner.n, runner.k)
    observed_scope = (None if rule_conclusion_observed is None else
                      _mask(rule_conclusion_observed, shape, "rule_conclusion_observed"))
    observed_bits = (None if observed_scope is None else
                     [bits(observed_scope[:, column]) for column in range(runner.k)])
    check_cells = None
    if check_candidates is not None:
        if not gamma_wins or update_kind != "check":
            raise ValueError("check candidates require a same-round check under validated Gamma wins")
        if not isinstance(check_candidates, (tuple, list)) or len(check_candidates) != 2:
            raise ValueError("check candidates must be (positive, negative) masks")
        check_cells = tuple(_mask(value, shape, "check_candidates") for value in check_candidates)
        if np.any(check_cells[0] & ~gamma.pos) or np.any(check_cells[1] & ~gamma.neg):
            raise ValueError("check candidates must be cells of the checked Gamma")
    saved_roots = []

    def gamma_roots():
        # _roots(gamma, n) of this call's frozen Gamma, built once.
        if not saved_roots:
            saved_roots.append(_roots(gamma, runner.n))
        return saved_roots[0]

    if before is not None:
        if before.status != "fixpoint" or before.positive_facts.shape != shape:
            raise ValueError("retaining update requires a successful same-axis prior state")
        old_gamma = runner._gamma(old_gamma if old_gamma is not None else before.persistent_gamma)
        if (np.any(old_gamma.pos & ~gamma.pos) or np.any(old_gamma.neg & ~gamma.neg)
                or not set(old_gamma.subset) <= (set(gamma.subset) | {
                    edge for edge in old_gamma.subset if edge[0] == edge[1]})
                or not set(old_gamma.strict_subset) <= set(gamma.strict_subset)):
            raise ValueError("persistent Gamma may only gain verified facts")
        if update_kind != "chaseDelta":
            if digest([r.to_dict() for r in rules]) != before.reference_audit["program_sha256"]:
                raise ValueError("same-model incremental/check call cannot change the program")
            # Indexed context identity binds data, models and instance scope,
            # but excludes Gamma. Verified additions therefore remain valid.
            if before.reference_audit["context_id"] != runner.context_id:
                raise ValueError(
                    "same-model incremental/check call cannot change the frozen context"
                )
        else:
            old_rules = tuple(old_rules) if old_rules is not None else None
            if old_runner is None or old_rules is None:
                raise ValueError("chaseDelta requires old_runner and old_rules")
            if before.reference_audit[
                "context_id"
            ] != old_runner.context_id or before.reference_audit["program_sha256"] != digest(
                [r.to_dict() for r in old_rules]
            ):
                raise ValueError("chaseDelta old context/program does not match retained history")
    state = _HistoryState(runner, before, overlay=decision_only)
    state.gamma_blocked_cells = set()
    protected = (gamma.pos, gamma.neg) if gamma_wins else None
    base = None
    round_blocked = None
    if check_cells is not None:
        # Candidate cells are checked, not yet validated: a derivation that
        # contradicts one is a conflict (the cell becomes unknown), not a block.
        protected = (gamma.pos & ~check_cells[0], gamma.neg & ~check_cells[1])
        # The committed baseline remains read-only for the whole check. Its
        # fact arrays can be shared for attribution instead of copied again.
        base = state.snapshot(facts=(before.positive_facts, before.negative_facts))
        # Blocks of rule/inclusion derivations; a blocked temporary root (a
        # seed contradicting Gamma) is never a consequence of the candidate.
        round_blocked = set()
    if not collect_rule_conclusions:
        state.relation_sources.clear()
        state.label_relation_sources.clear()
    old_unknown = np.zeros(shape, bool) if before is None else before.conflict_unknown
    if initial_unknown is None:
        unknown = (
            old_unknown.copy() if update_kind in ("chaseInc", "check") else np.zeros(shape, bool)
        )
    else:
        unknown = _mask(initial_unknown, shape, "initial_unknown")
        if update_kind == "check" and np.any(old_unknown & ~unknown):
            raise ValueError("same-round check cannot unlock conflict_unknown")
    if gamma_wins:
        # A validated cell is never unknown (new answers resolve old conflicts).
        unknown = unknown & ~(gamma.pos | gamma.neg)
    old_temporary = (
        (np.zeros(shape, bool), np.zeros(shape, bool)) if before is None else before.temporary_seed
    )
    supplied = initial_positive is not None or initial_negative is not None
    temporary = (
        tuple(
            _mask(value, shape, name)
            for value, name in (
                (initial_positive, "initial_positive"),
                (initial_negative, "initial_negative"),
            )
        )
        if supplied or update_kind not in ("check", "chaseInc")
        else tuple(x.copy() for x in old_temporary)
    )
    prepared = [runner._prepare(*runner._rule(rule)) for rule in rules]
    ledger = (
        {ri: {"pos": set(), "neg": set()} for ri in range(len(rules))}
        if collect_rule_conclusions
        else None
    )
    rule_instances = {ri: 0 for ri in range(len(rules))}
    changed = set()
    withdrawn, removed = set(), 0
    disabled = 0
    refuted_versions = 0
    failed_instance_heads = set()
    failed_head_versions = set()
    if update_kind == "chaseDelta":
        old_rules = tuple(old_rules) if old_rules is not None else None
        changed = _changed_rules(old_runner, runner, old_rules, rules)
        invalid_nodes = set()
        static_masks = {}
        # Historical witnesses include quarantined ancestors that cannot be
        # rediscovered by enumerating just the currently active old closure.
        for identifier, node in _explicit_history_items(state.nodes):
            ri = node.get("rule_id")
            if (
                identifier not in state.retired
                and node.get("origin") == "rule_application"
                and ri in changed
            ):
                if not _static_instance_valid(
                    runner, rules[ri], prepared[ri], node["valuation"], ri, static_masks
                ):
                    invalid_nodes.add(identifier)
                    disabled += 1
        old_state = _HistoryState(old_runner, before)
        old_cache = {}
        for ri in sorted(changed):
            prep = old_runner._prepare(*old_runner._rule(old_rules[ri]))
            counted = _failed_old_instances(
                old_runner, runner, old_rules[ri], rules[ri], prep, prepared[ri], old_state)
            if counted is not None:
                disabled += counted[0]
                failed_instance_heads.update(counted[1])
                continue
            # old_state is private and never modified while enumerating.
            for valuation in valuations(
                old_runner, old_rules[ri], prep, old_state, known_head=True, rule_index=ri,
                frozen_state=True, round_cache=old_cache,
            ):
                if not _static_instance_valid(runner, rules[ri], prepared[ri], valuation, ri,
                                              static_masks):
                    disabled += 1
                    # A currently valid old instance need not be the first
                    # recorded producer. Its newly false body is a fact-head
                    # DRed event, including isolated heads with live descendants
                    # attached to older versions. Seed every non-retired version
                    # of that head, then follow only versioned dependency edges.
                    failed_instance_heads.update(_head_keys(old_runner, old_rules[ri], valuation))
        failed_head_versions = {
            identifier
            for identifier, node in _explicit_history_items(state.nodes)
            if identifier not in state.retired and node["fact"] in failed_instance_heads
        }
        invalid_nodes.update(failed_head_versions)
    else:
        invalid_nodes = set()
    if before is not None:
        if update_kind == "check" and any(
            not np.array_equal(a, b) for a, b in zip(old_temporary, temporary)
        ):
            raise ValueError("same-round check cannot change temporary roots")
        removed_temporary = set()
        for sign, old, new in zip(("pos", "neg"), old_temporary, temporary):
            removed_temporary.update(_key(sign, i, j) for i, j in _changed_cells(old & ~new))
        if removed_temporary:
            invalid_nodes.update(
                identifier
                for identifier, node in _explicit_history_items(state.nodes)
                if node.get("origin") == "temporary_seed" and node["fact"] in removed_temporary
            )
        if gamma_wins:
            # Newly validated Gamma cells win: a derived fact of the opposite
            # sign is withdrawn with its recorded descendants (DRed), exactly
            # like a newly false rule instance. The retained state already
            # respects every Gamma cell it was computed with (contradictions of
            # those were blocked or withdrawn then), so only cells added since
            # can refute recorded facts. For an update that commits new Gamma
            # (not a same-round check) every non-retired version of a refuted
            # fact is withdrawn, including versions isolated as unknown whose
            # recorded descendants are still active (2026-10-01).
            reference = runner._gamma(before.persistent_gamma)
            added_positive = gamma.pos & ~reference.pos
            added_negative = gamma.neg & ~reference.neg
            if check_cells is not None:
                # Unvalidated check candidates never withdraw facts; a
                # contradiction with them is a conflict instead.
                added_positive = added_positive & ~check_cells[0]
                added_negative = added_negative & ~check_cells[1]
            refuted = {_key("neg", int(i), int(j)) for i, j in _changed_cells(added_positive)}
            refuted.update(_key("pos", int(i), int(j)) for i, j in _changed_cells(added_negative))
            invalid_nodes.update(
                state.current[key] for key in refuted if key in state.current and state.present(key))
            recorded = {key for key in refuted if key in state.latest}
            if update_kind != "check" and recorded:
                isolated = {
                    identifier for identifier, node in _explicit_history_items(state.nodes)
                    if identifier not in state.retired and node["fact"] in recorded
                }
                refuted_versions = len(isolated - invalid_nodes)
                invalid_nodes.update(isolated)
        withdrawn, removed = _withdraw(state, invalid_nodes, _RootMembership(gamma))
    # Isolation removes only the pair; it deliberately does not traverse its
    # dependency consumers. Existing consequences and comparison paths survive.
    for row, label in _changed_cells(unknown):
        state.remove(_key("pos", row, label))
        state.remove(_key("neg", row, label))
    if before is None and compact_roots:
        roots = _ImplicitRoots(gamma, runner.context_id)
        state.nodes = _Overlay(_RootMap(roots, nodes=True))
        state.current = _Overlay(_RootMap(roots))
        state.latest = _Overlay(_RootMap(roots))
        state.pos = gamma.pos & ~unknown
        state.neg = gamma.neg & ~unknown
        state.label_bits = {sign: [bits(matrix[:, j]) for j in range(runner.k)]
                            for sign, matrix in (("pos", state.pos), ("neg", state.neg))}
        for sign in ("pos", "neg"):
            for row, column in _changed_cells(getattr(gamma, sign) & unknown):
                state.current.pop(_key(sign, row, column), None)
        for kind, edges in (("subset", gamma.subset), ("strict_subset", gamma.strict_subset)):
            out, inc = (state.out, state.inc) if kind == "subset" else (state.proper, state.proper_inc)
            for row, column in edges:
                out[row] |= 1 << column
                inc[column] |= 1 << row
        # Gamma is already installed; only temporary roots need normal conflict
        # handling. All rules scan once below and all useful built-ins run.
        root_values = _root_candidates(RefGamma.empty(runner.n, runner.k), temporary, runner.n, ())
    else:
        root_values = (_root_candidates(gamma, temporary, runner.n, gamma_roots()) if before is None
                       else _updated_root_candidates(gamma, temporary, state, before.persistent_gamma))
    root_events, root_conflicts = _commit(state, root_values, unknown, protected)
    if ledger is not None:
        for row in range(runner.n):
            state.relation_sources[_key("subset", row, row)] = frozenset((-1,))
        for kind in ("subset", "strict_subset"):
            for row, column in getattr(gamma, kind):
                state.relation_sources[_key(kind, row, column)] = frozenset((-1,))
    initial_rules = set(range(len(rules))) if before is None else set(changed)
    unlocked = old_unknown & ~unknown
    label_types = {
        (_parts(key)[0], _parts(key)[2]) for key in withdrawn if _parts(key)[0] in ("pos", "neg")
    }
    unlocked_labels = set(np.flatnonzero(unlocked.any(0)).tolist())
    relation_withdrawn = any(_parts(key)[0] not in ("pos", "neg") for key in withdrawn)
    for ri, rule in enumerate(rules):
        if isinstance(rule.head, Label):
            k = runner.label_index[rule.head.label]
            if (rule.head.polarity, k) in label_types or k in unlocked_labels:
                initial_rules.add(ri)
        elif relation_withdrawn:
            initial_rules.add(ri)
    # Fresh execution already scans every rule, including relational bodies
    # satisfied by identity. Anchored root/identity jobs would all be skipped
    # below. Avoid checking n identity big integers whose bit widths grow with n.
    rule_events = [] if before is None else list(root_events)
    builtin_events = (
        list(_all_events(state))
        if withdrawn or unlocked.any() or before is None
        else list(root_events)
    )
    rdg = RuleDependencyGraph(rules)
    rule_variables = [rule.variables for rule in rules]
    label_heads = [
        (rule.head.polarity, runner.label_index[rule.head.label], rule.head.var)
        if isinstance(rule.head, Label) else None
        for rule in rules
    ]
    rounds = fired = processed = builtin_processed = candidates_seen = source_events = 0
    quarantines = [dict(batch=0, cells=root_conflicts)] if root_conflicts else []
    while True:
        # All candidate enumeration is synchronous and completes before the
        # single _commit below. No writer runs during this frozen phase, and
        # candidates.values binds parent IDs before _commit changes current.
        snapshot = state
        candidates = _Candidates(snapshot, unknown, rules, ledger)
        jobs = {(ri, ()) for ri in initial_rules}
        if execution == "full_recompute":
            jobs = {(ri, ()) for ri in range(len(rules))}
            builtin_events = list(_all_events(snapshot))
        else:
            for event in rule_events:
                if not snapshot.present(_key(*event)):
                    continue
                processed += 1
                for ri, atom in rdg.consumers(event, runner):
                    if ri in initial_rules:
                        # The same frozen snapshot already has a complete scan
                        # scheduled for this rule. Every anchored job is a
                        # subset of that scan; visited would discard its output.
                        continue
                    for fixed in event_bindings(atom, event, runner):
                        jobs.add((ri, tuple(sorted(fixed.items()))))
        visited = set()
        round_cache = {}
        # A rule scanned by a single job enumerates each valuation once, so its
        # instances need no cross-job deduplication set.
        rule_jobs = defaultdict(int)
        for ri, _ in jobs:
            rule_jobs[ri] += 1
        for ri, fixed in sorted(jobs):
            variables = rule_variables[ri]
            label_head = label_heads[ri]
            deduplicate = rule_jobs[ri] > 1
            instance_prepared = prepared[ri]
            if observed_bits is not None and label_head is not None:
                sign, column, head_var = label_head
                # Never limit sources to calibration rows. Only a head that
                # is already established and not observed is a removable no-op:
                # _Candidates.add would record it in the unused ledger and
                # return before creating any candidate or provenance marker.
                domain = (instance_prepared['domains'][head_var] &
                          (~snapshot.label_bits[sign][column] | observed_bits[column]))
                instance_prepared = dict(instance_prepared, domains=dict(
                    instance_prepared['domains'], **{head_var: domain}))
            enumeration = valuations(
                runner,
                rules[ri],
                instance_prepared,
                snapshot,
                fixed=dict(fixed),
                known_head=ledger is not None,
                rule_index=ri,
                frozen_state=True,
                round_cache=round_cache,
            )
            if label_head is not None and ledger is None:
                # Lean exact path: same instances, candidates and counters.
                sign, column, head_var = label_head
                for valuation in enumeration:
                    if deduplicate:
                        instance = (ri, tuple(valuation[v] for v in variables))
                        if instance in visited:
                            continue
                        visited.add(instance)
                    row = valuation[head_var]
                    if unknown[row, column]:
                        continue
                    candidates.add_rule_label(
                        _key(sign, row, column), candidates.label_dependencies(runner, ri, valuation),
                        ri, valuation, runner)
                    fired += 1
                    rule_instances[ri] += 1
                continue
            for valuation in enumeration:
                if deduplicate:
                    instance = (ri, tuple(valuation[v] for v in variables))
                    if instance in visited:
                        continue
                    visited.add(instance)
                heads = _head_keys(runner, rules[ri], valuation)
                if all(
                    _parts(key)[0] in ("pos", "neg") and unknown[_parts(key)[1:]] for key in heads
                ):
                    continue
                dependencies = candidates.rule_dependencies(runner, ri, valuation)
                for key in heads:
                    candidates.add(key, "rule_application", dependencies, ri, valuation,
                                   static=None, runner=runner)
                fired += 1
                rule_instances[ri] += 1
        builtin_processed += len(builtin_events)
        _builtins(snapshot, builtin_events, candidates)
        rounds += 1
        candidates_seen += candidates.attempts
        events, conflicts = _commit(state, candidates.values, unknown, protected, round_blocked)
        if conflicts:
            quarantines.append(dict(batch=rounds, cells=conflicts))
        marker_events = _commit_markers(state, candidates)
        source_events += len(marker_events)
        if not events and not marker_events:
            break
        rule_events = events
        builtin_events = sorted(set(events) | marker_events)
        initial_rules = set()
    contradictions = None
    if check_cells is not None:
        # Every derivation contradicting validated Gamma is blocked. One that
        # also follows from the checked fixed point existed before the
        # candidate; the others need a fact that only the candidate supplies.
        # This depends only on the two closures, not on the execution mode.
        head_rules = _head_rule_index(runner, rules)
        contradictions = frozenset(
            key for key in round_blocked
            if not _derivable(runner, rules, prepared, base, key, head_rules=head_rules))
    if seed_identity(gamma) != gamma_id:
        raise RuntimeError("retaining Chase changed persistent Gamma")
    state.conflict = None
    state.last_valid_version = -1
    invalid = state.invalid()
    audit = dict(
        semantics=SEMANTICS,
        execution=execution,
        update_kind=update_kind or "fresh",
        full_recompute=execution == "full_recompute",
        same_retained_history_semantics=True,
        synchronous_rules_and_builtin_copies=True,
        changed_rules=sorted(changed),
        processed_events=processed,
        builtin_event_visits=builtin_processed,
        fired_instances=fired,
        candidate_attempts=candidates_seen,
        synchronous_rounds=rounds,
        disabled_recorded_or_active_instances=disabled,
        newly_false_old_valid_instance_heads=len(failed_instance_heads),
        failed_instance_head_history_versions=len(failed_head_versions),
        all_instance_head_events_cover_nonretired_versions=True,
        withdrawal_edges_are_versioned=True,
        temporarily_withdrawn=len(withdrawn),
        active_facts_removed_for_model_change=removed,
        restored_same_update=sum(state.present(key) for key in withdrawn),
        finally_deleted=sum(not state.present(key) for key in withdrawn),
        unlocked_cells=int(unlocked.sum()),
        quarantines=quarantines,
        conflict_unknown_cells=int(unknown.sum()),
        gamma_conflict_policy="validated_gamma_wins" if gamma_wins else "quarantine",
        gamma_blocked_derivations=len(state.gamma_blocked_cells),
        candidate_contradictions=None if contradictions is None else len(contradictions),
        gamma_refuted_isolated_versions_withdrawn=refuted_versions,
        unknown_does_not_retract_descendants=True,
        model_changes_retract_and_rederive=True,
        historical_nodes=len(state.nodes),
        root_candidates_processed=len(root_values),
        decision_only=decision_only,
        baseline_history_shared=decision_only,
        implicit_initial_gamma_roots=compact_roots,
        Gamma_record_unchanged=True,
        temporary_roots_promoted_to_Gamma=False,
        proof_kind="recorded_historical_one_hop_witnesses_not_current_complete_proof",
        rule_conclusions_enabled=collect_rule_conclusions,
        rule_conclusion_policy=(
            "actual_synchronous_firings_distinct_signed_targets_including_later_unknown"
            if observed_scope is None else
            "observed_receivers_and_new_heads_distinct_signed_targets_including_later_unknown"),
        rule_instance_count_policy=(
            "per_batch_firings_with_in_batch_instance_deduplication"
            if observed_scope is None else
            "per_batch_firings_omitting_known_unobserved_Label_heads"),
        calibration_relation_source_event_visits=source_events,
    )
    result = ChaseResult(
        status="conflict" if invalid else "fixpoint",
        predictions=(np.empty((0, 0), np.float32) if decision_only else state.pos.astype(np.float32)),
        conflicts=[],
        n_rounds=rounds,
        provenance={},
        positive_facts=(state.pos if decision_only else
                        np.frombuffer(state.pos.tobytes(), dtype=bool).reshape(shape)),
        negative_facts=(state.neg if decision_only else
                        np.frombuffer(state.neg.tobytes(), dtype=bool).reshape(shape)),
    )
    result.subset_facts = RelationFacts(state.out)
    result.strict_subset_facts = RelationFacts(state.proper)
    if decision_only:
        # No temporary check is adopted as a future history. Keep the actual
        # closure and diagnostics used by acceptance, but never materialize or
        # serialize its million-node shared baseline.
        result.conflict_unknown = unknown
        result.candidate_contradictions = contradictions
        result.comparison_conflicts = [state.conflict] if invalid else []
        result.gamma_wins = gamma_wins
        result.gamma_blocked = len(state.gamma_blocked_cells)
        result.reference_audit = dict(
            semantics=SEMANTICS, context_id=runner.context_id,
            program_sha256=digest([r.to_dict() for r in rules]),
            executor="retaining_" + execution, complete_without_cutoff=True,
            round_model_frozen=True, original_paper_conflict_semantics=False,
            full_chase_call_performed_for_this_update=execution == "full_recompute",
            maintenance=audit, decision_only=True,
        )
        return result
    # These private exact bit tuples were maintained alongside the matrices.
    # Each branch receives fresh mutable lists and fresh writable fact matrices.
    _label_columns(result.positive_facts, known=state.label_bits["pos"])
    _label_columns(result.negative_facts, known=state.label_bits["neg"])
    # Inactive historical ancestors intentionally remain addressable. A node's
    # history_dependencies, not this convenience fact-key map, binds its parents.
    # Nodes inherited from the prior history are already immutable and are
    # reused by identity; the fact-key map shares the same immutable nodes.
    # Contents equal ProofDAG(state.nodes) / ProofDAG({key: _plain(node)}).
    if compact_roots:
        result.history_nodes = _CompactHistoryDAG(state.nodes)
        result.history_current = _CompactMap(state.current)
        result.history_latest = _CompactMap(state.latest)
        result.proof_dag = _CompactFactDAG(result.history_nodes, result.history_latest)
        result.history_schema = COMPACT_HISTORY_SCHEMA
    else:
        history_nodes = ProofDAG.adopt(
            state.nodes, before.history_nodes if before is not None else ())
        latest_nodes = {key: history_nodes[node] for key, node in state.latest.items()}
        result.proof_dag = ProofDAG.adopt(latest_nodes, latest_nodes, encoded_from=history_nodes)
        result.history_schema = HISTORY_SCHEMA
        result.history_nodes = history_nodes
        result.history_current = MappingProxyType(dict(state.current))
        result.history_latest = MappingProxyType(dict(state.latest))
    result.history_retired = frozenset(state.retired)
    result.rule_conclusions = {
        ri: {sign: tuple(sorted(values)) for sign, values in signed.items()}
        for ri, signed in (ledger or {}).items()
    }
    result.rule_conclusion_instances = dict(rule_instances)
    result.rule_conclusions_enabled = collect_rule_conclusions
    if observed_scope is not None:
        result.rule_conclusion_observed = np.frombuffer(
            observed_scope.tobytes(), dtype=bool).reshape(shape)
        audit['rule_conclusion_observed'] = array_identity(observed_scope)
    result.relation_source_markers = MappingProxyType(dict(state.relation_sources))
    result.label_relation_markers = MappingProxyType(dict(state.label_relation_sources))
    result.persistent_gamma = gamma
    result.gamma_wins = gamma_wins
    result.gamma_blocked = int(len(state.gamma_blocked_cells))
    # Gamma-blocked derivations caused by check candidates (None when not requested).
    result.candidate_contradictions = contradictions
    result.conflict_unknown = unknown.copy()
    # A bytes-backed array cannot be made writeable again with setflags(), so
    # cached history identities remain valid for the lifetime of this binding.
    result.temporary_seed = tuple(np.frombuffer(x.tobytes(), dtype=x.dtype).reshape(x.shape)
                                  for x in temporary)
    result.history_identity = history_digest(result)
    result.comparison_conflicts = [state.conflict] if invalid else []
    result.reference_audit = dict(
        semantics=SEMANTICS,
        context_id=runner.context_id,
        program_sha256=digest([r.to_dict() for r in rules]),
        executor="retaining_" + execution,
        complete_without_cutoff=True,
        round_model_frozen=True,
        original_paper_conflict_semantics=False,
        full_chase_call_performed_for_this_update=execution == "full_recompute",
        maintenance=audit,
        history_identity=result.history_identity,
    )
    return result


def export_history(result, *, serialize_nodes=True):
    """Serialize every history field used by maintenance, with explicit roles.

    Active fact arrays/relations and conflict_unknown remain in the normal Chase
    export; dense Gamma/temporary arrays are represented here by compact
    shape/dtype/hash bindings, and must be persisted separately by the caller.
    Calibration source markers are diagnostic only: they never govern
    inference or DRed and collection is supported only on a fresh invocation.
    """
    compact = isinstance(result.history_nodes, _CompactHistoryDAG)
    nodes = (result.history_nodes.compact() if compact else
             _plain(dict(result.history_nodes.items())) if serialize_nodes else result.history_nodes)
    if compact and serialize_nodes:
        nodes = _plain(nodes)
    return dict(
        schema=result.history_schema,
        nodes=nodes,
        current=(_plain(result.history_current.compact()) if compact else dict(result.history_current)),
        latest=(_plain(result.history_latest.compact()) if compact else dict(result.history_latest)),
        retired=sorted(result.history_retired),
        input_gamma_binding=seed_identity(result.persistent_gamma),
        temporary_seed_binding=[array_identity(matrix) for matrix in result.temporary_seed],
        relation_source_markers={
            key: sorted(value) for key, value in result.relation_source_markers.items()
        },
        label_relation_markers={
            key: sorted(value) for key, value in result.label_relation_markers.items()
        },
        rule_conclusions_enabled=result.rule_conclusions_enabled,
    )


def history_digest(result):
    """Same canonical bytes as the export without a redundant deep plain copy.

    The node map of an adopted ProofDAG is assembled from its cached canonical
    node bytes (2026-09-30); every other field is encoded exactly as digest()
    does, and the top-level object is assembled in the same sorted key order.
    Any node that the cached encoder cannot represent falls back to digest().
    """
    # Reuse only truly immutable fields; field replacement invalidates the
    # binding. Generic/mutable callers retain the original rehashing behavior.
    names = ("history_schema", "history_nodes", "history_current", "history_latest",
             "history_retired", "persistent_gamma", "temporary_seed",
             "relation_source_markers", "label_relation_markers", "rule_conclusions_enabled")
    binding = tuple(getattr(result, name) for name in names)
    immutable = (isinstance(result.history_nodes, ProofDAG)
                 and all(isinstance(getattr(result, name), (MappingProxyType, _CompactMap)) for name in (
                     "history_current", "history_latest", "relation_source_markers",
                     "label_relation_markers"))
                 and isinstance(result.history_retired, frozenset)
                 and isinstance(result.temporary_seed, tuple)
                 and not any(matrix.flags.writeable for matrix in (
                     result.persistent_gamma.pos, result.persistent_gamma.neg,
                     *result.temporary_seed)))
    saved = getattr(result, "_history_digest_cache", None)
    if immutable and saved is not None and all(a is b for a, b in zip(binding, saved[0])):
        return saved[1]
    export = export_history(result, serialize_nodes=False)
    nodes = export["nodes"]
    try:
        local = nodes.canonical_local_bytes() if isinstance(nodes, ProofDAG) else None
    except TypeError:
        local = None
    if local is None:
        value = digest(export)
        if immutable:
            result._history_digest_cache = (binding, value)
        return value
    parts = {}
    for name, value in export.items():
        parts[name] = local if name == "nodes" else json.dumps(
            value, default=_json_default, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode()
    body = b"{" + b",".join(
        json.dumps(name, ensure_ascii=False).encode() + b":" + parts[name] for name in sorted(parts)
    ) + b"}"
    value = hashlib.sha256(body).hexdigest()
    if immutable:
        result._history_digest_cache = (binding, value)
    return value


def recorded_feedback_view(runner, rules, result, gamma, *, candidate_mask=None):
    """Filter training use by recorded ancestors only, without alternative search.

    This may conservatively exclude an active fact with an unrecorded valid
    alternative. Invalid history stays usable for inference. Cost is the number
    of visited historical nodes/edges and concrete premise checks, not a claim
    of constant-time or complete-provenance verification.
    """
    if getattr(result, "history_schema", None) not in (HISTORY_SCHEMA, COMPACT_HISTORY_SCHEMA):
        raise ValueError("recorded feedback requires a retaining history")
    rules = tuple(rules)
    gamma = runner._gamma(gamma)
    state = _HistoryState(runner, result)
    unknown = result.conflict_unknown
    selected = (
        np.ones_like(unknown)
        if candidate_mask is None
        else _mask(candidate_mask, unknown.shape, "candidate_mask")
    )
    prepared = [runner._prepare(*runner._rule(rule)) for rule in rules]
    roots = _RootMembership(gamma)
    cache = {}
    visiting = set()
    reasons = defaultdict(int)
    edges = 0

    def local(identifier):
        node = state.nodes.get(identifier)
        if node is None:
            reasons["missing_history"] += 1
            return False, ()
        key = node.get("fact", "")
        if key not in state.current or not state.present(key):
            reasons["inactive_or_unknown_ancestor"] += 1
            return False, ()
        kind, row, column = _parts(key)
        if kind in ("pos", "neg") and unknown[row, column]:
            reasons["quarantined_ancestor"] += 1
            return False, ()
        if key in roots:
            return True, ()
        origin = node.get("origin")
        if origin == "temporary_seed":
            reasons["temporary_seed"] += 1
            return False, ()
        if origin == "rule_application":
            ri = node.get("rule_id")
            valuation = node.get("valuation", {})
            try:
                okay = (
                    type(ri) is int
                    and 0 <= ri < len(rules)
                    and _static_instance_valid(runner, rules[ri], prepared[ri], valuation, ri)
                    and key in _head_keys(runner, rules[ri], valuation)
                    and any(
                        valuations(
                            runner,
                            rules[ri],
                            prepared[ri],
                            state,
                            fixed=valuation,
                            known_head=True,
                            rule_index=ri,
                        )
                    )
                )
            except (ValueError, KeyError, IndexError, TypeError):
                okay = False
            if not okay:
                reasons["invalid_recorded_rule_instance"] += 1
                return False, ()
            required, _ = runner._evidence(rules[ri], valuation)
            if set(required) != set(node.get("dependencies", ())):
                reasons["mismatched_recorded_dependencies"] += 1
                return False, ()
        elif origin not in (
            "inclusion_label_copy",
            "inclusion_transitivity",
            "proper_inclusion_transitivity",
            "proper_implies_subset",
        ):
            reasons["unrecognized_nonroot"] += 1
            return False, ()
        dependencies = node.get("dependencies", ())
        parents = node.get("history_dependencies", ())
        if (
            len(dependencies) != len(parents)
            or any(
                state.nodes.get(parent, {}).get("fact") != dependency
                for dependency, parent in zip(dependencies, parents)
            )
            or (origin != "rule_application" and not parents)
        ):
            reasons["missing_or_mismatched_history_parents"] += 1
            return False, ()
        return None, parents

    def valid(identifier):
        nonlocal edges
        # Iterative postorder also rejects manually corrupted cycles without
        # depending on Python's recursion limit for authentic deep histories.
        stack = [(identifier, False)]
        while stack:
            current, finish = stack.pop()
            if current in cache:
                continue
            if finish:
                parents = state.nodes[current].get("history_dependencies", ())
                cache[current] = all(cache.get(parent, False) for parent in parents)
                visiting.discard(current)
                continue
            if current in visiting:
                reasons["cycle"] += 1
                cache[current] = False
                continue
            value, parents = local(current)
            if value is not None:
                cache[current] = value
                continue
            visiting.add(current)
            edges += len(parents)
            stack.append((current, True))
            stack.extend((parent, False) for parent in reversed(parents))
        return cache[identifier]

    # Active trusted roots need no historical traversal. Visit just selected
    # non-root cells, rather than every cell in the complete training Gamma.
    allowed = selected & ~unknown & ((gamma.pos & state.pos) | (gamma.neg & state.neg))
    for kind, matrix in (("pos", state.pos), ("neg", state.neg)):
        for row, column in np.argwhere(selected & matrix & ~allowed):
            key = _key(kind, row, column)
            if valid(state.current[key]):
                allowed[row, column] = True
    view = copy.copy(result)
    view.positive_facts = result.positive_facts & allowed
    view.negative_facts = result.negative_facts & allowed
    view.predictions = view.positive_facts.astype(np.float32)
    view.reference_audit = copy.deepcopy(result.reference_audit)
    subset = RelationRows(runner.n)
    proper = RelationRows(runner.n)
    for kind, current_relation, relation in (("subset", state.out, subset),
                                              ("strict_subset", state.proper, proper)):
        for row, targets in enumerate(current_relation):
            for column in indices(targets):
                key = _key(kind, row, column)
                if key in roots or valid(state.current[key]):
                    relation[row] |= 1 << column
    view.subset_facts = RelationFacts(subset)
    view.strict_subset_facts = RelationFacts(proper)
    view.recorded_source_view = True
    view.reference_audit["recorded_source_check"] = dict(
        candidate_cells=int(selected.sum()),
        retained_training_cells=int(allowed.sum()),
        visited_history_nodes=len(cache),
        visited_history_edges=edges,
        rejected_reasons=dict(reasons),
        active_inference_facts_unchanged=True,
        extra_Chase_calls=0,
        alternative_derivations_searched=False,
        ordinary_RDL_ML_allowed=True,
        temporary_only_sources_excluded=True,
    )
    return view
