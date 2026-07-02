"""Unit tests for Lever B — richer high-precision FN→ADD synthesis.

Guarantees: (1) ``richer=True`` mines a 2-phrase ``CooccurPredicate`` rule where
neither single word clears the precision floor but the conjunction does; (2)
``richer=False`` is unchanged (no such conjunction rule is emitted).
"""
import numpy as np
from loris.document import Document
from loris.predicates import CooccurPredicate
from loris.pipeline.steps import _synthesize_fn_add_rules


def _conjunction_only_corpus():
    """40 docs, label 0 addable everywhere. ``alpha`` and ``beta`` each predict
    label 0 at prec 0.50 (fail 0.70 floor); only ``alpha ∧ beta`` is precise (1.0).
    A shared filler keeps every uni/bigram non-discriminative so the single-phrase
    path cannot sneak in a precise unigram OR bigram."""
    F = "lorem ipsum dolor"               # shared filler (no English stopwords)
    groups = [
        ("alpha %s beta" % F, 1),         # A: both words  → label 0 present
        ("alpha %s gamma" % F, 0),        # B: alpha only
        ("delta %s beta" % F, 0),         # C: beta only
        ("delta %s gamma" % F, 0),        # D: neither
    ]
    docs, y = [], []
    for text, lab in groups:
        for _ in range(10):
            docs.append(Document(cnt=text))
            y.append([lab, 0])            # 2 labels; only label 0 is exercised
    labels = np.asarray(y, dtype=np.float32)
    base_preds = np.zeros_like(labels)    # nothing predicted → all docs addable
    return docs, labels, base_preds, ["L0", "L1"]


def _has_cooccur_rule(rules, label="L0"):
    return [r for r in rules
            if r.consequence == label
            and any(isinstance(p, CooccurPredicate) for p in r.body)]


def test_richer_emits_conjunction_rule():
    docs, labels, base_preds, names = _conjunction_only_corpus()
    rules = _synthesize_fn_add_rules(
        docs, labels, base_preds, names,
        ml_proba_cache=None, ml_guard=False,
        min_prec=0.70, min_fires=3, max_rules_per_label=0, richer=True)
    cooccur = _has_cooccur_rule(rules)
    assert cooccur, "richer=True should mine an alpha∧beta CooccurPredicate rule"
    assert cooccur[0].val_stats.get("val_prec", 0) >= 0.70


def test_not_richer_emits_nothing_for_conjunction_label():
    docs, labels, base_preds, names = _conjunction_only_corpus()
    rules = _synthesize_fn_add_rules(
        docs, labels, base_preds, names,
        ml_proba_cache=None, ml_guard=False,
        min_prec=0.70, min_fires=3, max_rules_per_label=0, richer=False)
    # No single word clears 0.70, so the default path emits no L0 rule and never a Cooccur.
    assert not _has_cooccur_rule(rules), "richer=False must not emit conjunction rules"
    assert not [r for r in rules if r.consequence == "L0"], \
        "richer=False: no single-phrase rule clears the floor for L0"


def test_richer_is_superset_not_replacement():
    """A label with a precise single phrase keeps its single-phrase rule under
    richer (richer adds, never removes)."""
    docs, labels, base_preds, names = _conjunction_only_corpus()
    # make 'beta' precise for L1: present only in true-L1 docs
    for d, row in zip(docs, labels):
        if "beta" in d.cnt:
            row[1] = 1
    plain = _synthesize_fn_add_rules(docs, labels, base_preds, names,
                                     ml_proba_cache=None, ml_guard=False,
                                     min_prec=0.70, min_fires=3,
                                     max_rules_per_label=0, richer=False)
    rich = _synthesize_fn_add_rules(docs, labels, base_preds, names,
                                    ml_proba_cache=None, ml_guard=False,
                                    min_prec=0.70, min_fires=3,
                                    max_rules_per_label=0, richer=True)
    n_plain_l1 = sum(1 for r in plain if r.consequence == "L1")
    n_rich_l1 = sum(1 for r in rich if r.consequence == "L1")
    assert n_plain_l1 >= 1, "single precise phrase should yield an L1 rule"
    assert n_rich_l1 >= n_plain_l1, "richer must not drop existing rules"
