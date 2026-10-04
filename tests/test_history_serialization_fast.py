"""Canonical identities must not change when redundant conversion is removed."""
import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from loris._engine.typed_runtime import ProofDAG
from loris._engine.typed_session import digest, plain
from loris._engine.rdl_reference import RefGamma
from loris.labeling.unknown_incremental import export_history, history_digest


def legacy_digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def test_nested_immutable_and_numpy_have_identical_canonical_identity():
    value = MappingProxyType({'中文': (np.int64(5), np.float32(.25), np.bool_(True)),
        'nested': ProofDAG({'h0': {'dependencies': ['x'], 'value': -0.0,
                                  'other': [None, 4, False]}})})
    assert digest(value) == legacy_digest(value)
    for bad in (float('nan'), np.float64('inf'), {'x': {1, 2}}):
        with pytest.raises((ValueError, TypeError)):
            digest(bad)
        with pytest.raises((ValueError, TypeError)):
            legacy_digest(bad)


def test_history_hash_is_export_hash_and_still_detects_changed_metadata():
    result = SimpleNamespace(history_schema='fixture',
        history_nodes=ProofDAG({'h0': {'fact': 'pos:0:0', 'history_dependencies': []}}),
        history_current={'pos:0:0': 'h0'}, history_latest={'pos:0:0': 'h0'},
        history_retired=frozenset(), persistent_gamma=RefGamma.empty(1, 1),
        temporary_seed=(np.zeros((1, 1), bool), np.zeros((1, 1), bool)),
        relation_source_markers={}, label_relation_markers={}, rule_conclusions_enabled=False)
    before = history_digest(result)
    assert before == legacy_digest(export_history(result))
    result.history_retired = frozenset({'h0'})
    assert history_digest(result) == legacy_digest(export_history(result))
    assert history_digest(result) != before


def test_cached_node_bytes_reproduce_original_history_and_proof_identities(tmp_path):
    """history_digest / ProofDAG.identity of adopted DAGs equal the original
    whole-structure encodings, across fresh, chaseInc and chaseDelta calls."""
    import random
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from loris._engine.array_cache import sha
    from loris._engine.typed_runtime import IndexedChaseRunner, _plain
    from loris.labeling.unknown_incremental import retaining_chase
    from tests.test_unknown_incremental import context
    from loris._engine.rdl_reference import Doc, Label, ML, Rule, Text, AttrEqual
    from loris.document import Document
    from loris.predicates import MatchPredicate

    def check(result):
        assert result.history_identity == digest(export_history(result, serialize_nodes=False))
        for dag in (result.history_nodes, result.proof_dag):
            expected = sha(_plain(dict(dag.items())))
            assert dag.identity()["local_sha256"] == expected

    rng = random.Random(3)
    n = 12
    ctx = dict(docs=[Document(cnt="node%d 文" % i, ttl="", mtd="g%d" % (i % 3)) for i in range(n)],
               labels=["A", "B"], gamma=RefGamma.empty(n, 2),
               model_probabilities={"aux": np.random.default_rng(1).random((n, 2))}, model_decisions={})
    pos = np.zeros((n, 2), bool); neg = np.zeros((n, 2), bool)
    pos[0, 0] = neg[1, 0] = True
    gamma = RefGamma(pos, neg)
    ctx["gamma"] = gamma
    rules = (Rule((Doc("x"), ML("x", "aux", "A", .4, dynamic=True)), Label("x", "A")),
             Rule((Doc("x"), Doc("y"), Label("y", "A"), AttrEqual("x", "mtd", "y", "mtd")), Label("x", "B")),
             Rule((Doc("x"), Text("x", MatchPredicate("cnt", "node1")), Label("x", "B")), Label("x", "A", "neg")))
    runner = IndexedChaseRunner(ctx, tmp_path / "a")
    fresh = retaining_chase(runner, rules, gamma, gamma_wins=True)
    check(fresh)
    pos2 = pos.copy(); pos2[5, 1] = True
    later_gamma = RefGamma(pos2, neg)
    later = IndexedChaseRunner(dict(ctx, gamma=later_gamma), tmp_path / "b")
    inc = retaining_chase(later, rules, later_gamma, before=fresh, old_gamma=gamma,
                          update_kind="chaseInc", gamma_wins=True)
    check(inc)
    assert any(id(v) in inc.history_nodes._node_bytes for v in inc.history_nodes._derived.values())
    new_rules = (Rule((Doc("x"), ML("x", "aux", "A", .6, dynamic=True)), Label("x", "A")),) + rules[1:]
    third = IndexedChaseRunner(dict(ctx, gamma=later_gamma), tmp_path / "c")
    delta = retaining_chase(third, new_rules, later_gamma, before=inc, old_runner=later, old_rules=rules,
                            old_gamma=later_gamma, update_kind="chaseDelta", gamma_wins=True)
    check(delta)
    for r in (runner, later, third):
        r.close()
