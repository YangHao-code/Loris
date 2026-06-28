"""RuleCleaner rule-refinement baseline (LocalBoost slot) — adapter over the
AUTHORS' code.

LocalBoost (Zhang KDD'23) released no code (cited repo is 404); per the
experiment plan we substitute the newer, code-available, role-matched
**RuleCleaner** (Refining Labeling Functions with Limited Labeled Data, KDD 2025;
repo ``JayLi2018/RuleCleanerKDD25``) and report it under its own name. Both are
weak-supervision rule/LF improvement methods — LocalBoost's slot.

RuleCleaner represents a labeling function as a decision-**TreeRule** over text
predicates (keyword/regex/stem) and *repairs* it against a small set of labeled
"complaints" (instances the rule mislabels) via ``fix_violations`` (Gini /
information-gain node splits and label reversals). We drive the AUTHORS' actual
``TreeRules``/``LFRepair`` code (vendored under ``refs/RuleCleaner``).

Integration (Group C — pattern selection, one-vs-rest binary like the repo):
  1. Build candidate keyword TreeRules from the top-MI patterns (each pattern ->
     a keyword rule voting the label where present).
  2. Populate each rule with the labeled train instances reaching its leaves
     (``populate_violations``) and REPAIR it (``fix_violations``) using the
     labeled set as the "limited labeled data" RuleCleaner refines against.
  3. Score each refined rule by its val macro-F1; keep the top-``top_k`` rules'
     patterns. Those feed the SAME downstream OvR-logistic model + metric as the
     other Group-C methods, so only the selection (rule-refinement) differs.

Module deps (snorkel, textblob, nltk, pulp) are installed; ``psycopg2`` is used
only by the repo's experiment harness (not the algorithm) and is stubbed.

DISCLOSED MODIFICATION. The authors' ``fix_violations`` splits leaves until
purity, which is reachable in their binary spam/ham setting but NOT on noisy
multi-label newswire/academic text — there it grows a degenerate >1000-node tree
and hangs (RecursionError in ``__str__``). We added a single disclosed guard to
the vendored ``LFRepair.fix_violations`` (``treerule.size >= LORIS_RC_MAX_TREE_SIZE``,
default 61) that stops the repair when a tree grows pathologically; well-behaved
rules (which terminate well under the cap) are unaffected. This is the only edit
to the authors' algorithm and is required for the method to run at all on these
datasets.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import types
import warnings
from typing import List

import numpy as np

from loris.baselines import common

_RC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "refs", "RuleCleaner"))

SPAM, HAM, ABSTAIN = 1, 0, -1


def _ensure_rc_importable() -> None:
    if "psycopg2" not in sys.modules:
        sys.modules["psycopg2"] = types.ModuleType("psycopg2")  # harness-only
    if _RC not in sys.path:
        sys.path.insert(0, _RC)


class _Inst:
    """Instance/complaint wrapper RuleCleaner expects.

    Predicates read ``.text`` / ``.stems``; ``populate_violations`` /
    ``redistribute_after_fix`` index it as a dict (``['expected_label']``,
    ``['text']``). Support both. ``id`` is used only for display.
    """
    __slots__ = ("text", "stems", "_d", "id")

    def __init__(self, text: str, expected_label: int, cid: int, stems=None):
        self.text = text
        self.stems = stems if stems is not None else text.lower().split()
        self.id = cid
        self._d = {"text": text, "expected_label": int(expected_label), "cid": cid}

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v):
        self._d[k] = v

    def __contains__(self, k):
        return k in self._d


def _top_keyword_patterns(split, n_patterns: int):
    """Top binary unigram patterns by summed MI (shared Group-C vocabulary)."""
    from loris.baselines.pattern_select import _build_pattern_matrix, _rank_mi
    Xtr, Xval, Xte, vec = _build_pattern_matrix(split, n_patterns=n_patterns)
    vocab = np.array(vec.get_feature_names_out())
    return Xtr, Xval, Xte, vocab, vec


def _rule_predict(tree_rule, insts) -> np.ndarray:
    """Apply a (refined) TreeRule to instances -> {0,1} firing for the label."""
    out = np.zeros(len(insts), dtype=np.int8)
    for i, ins in enumerate(insts):
        try:
            lab = tree_rule.evaluate(ins, "label")
        except Exception:
            lab = ABSTAIN
        out[i] = 1 if lab == SPAM else 0
    return out


def _refine_one_label(vocab, top_idx, train_insts_pos, train_insts_neg,
                      val_insts, repair_strategy):
    """Build + repair keyword TreeRules for ONE binary label; return (rule, val_pred_fn-ready rules).

    Returns a list of repaired TreeRules (one per candidate keyword pattern).
    """
    _ensure_rc_importable()
    from rulecleaner_src.lfs_tree import keyword_labelling_func_builder
    from rulecleaner_src.LFRepair import (populate_violations, fix_violations,
                                          check_tree_purity)

    rules = []
    complaints = train_insts_pos + train_insts_neg
    # the authors' redistribute_after_fix prints debug trees to stdout; silence it
    with contextlib.redirect_stdout(io.StringIO()):
        for j in top_idx:
            kw = str(vocab[j])
            if not kw or " " in kw:   # keyword predicate wants a single token
                continue
            try:
                tr = keyword_labelling_func_builder(keywords=[kw], expected_label=SPAM)
                # populate the rule's leaves with labeled complaints, then repair
                leaf_nodes = []
                seen = set()
                for c in complaints:
                    ln = populate_violations(tr, c)
                    if id(ln) not in seen:
                        seen.add(id(ln))
                        leaf_nodes.append(ln)
                fix_violations(tr, repair_strategy, leaf_nodes)
                rules.append(tr)
            except Exception:
                continue
    return rules

def localboost(split, seed: int = 0, *, top_k: int = 200, n_patterns: int = 1000,
               repair_strategy: str = "naive", max_rules: int = 8,
               max_pos: int = 15, max_neg: int = 15, **kwargs) -> dict:
    """RuleCleaner rule-refinement selection (authors' code); LocalBoost-slot.

    Reported under its own name 'RuleCleaner'. kwargs: top_k, n_patterns,
    repair_strategy ('naive'|'information_gain'), max_rules (candidate patterns
    refined), max_pos/max_neg (labeled "complaints" per label — RuleCleaner is a
    *limited-labeled-data* method, so a small labeled set is on-protocol and
    keeps the O(complaints^2) tree repair tractable). Remaining top_k slots after
    the refined patterns are filled from MI order.
    """
    from loris.baselines.pattern_select import _rank_mi, _fit_score

    Xtr, Xval, Xte, vocab, vec = _top_keyword_patterns(split, n_patterns)
    ytr = np.asarray(split.train_y, dtype=int)
    yval = np.asarray(split.val_y, dtype=int)
    yte = np.asarray(split.test_y, dtype=int)
    n_lbl = ytr.shape[1]
    n_feat = Xtr.shape[1]

    # MI ordering (shared with filter_mi) gives the candidate patterns to refine.
    mi_order = _rank_mi(Xtr, ytr, seed)
    cand = mi_order[: min(max_rules, mi_order.size)]

    # Build train/val instance wrappers once (text reused across labels).
    train_insts_all = [_Inst(split.train_X[i], 0, i) for i in range(len(split.train_X))]
    val_insts = [_Inst(split.val_X[i], 0, i) for i in range(len(split.val_X))]

    # Per-pattern aggregate "refinement value": mean val macro-F1 of the repaired
    # rule across labels for which the pattern is a candidate. Patterns whose
    # refined rules best predict val are ranked first (the RuleCleaner signal).
    pattern_score = np.zeros(n_feat, dtype=np.float64)
    n_refined = 0
    with common.Timer() as t, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lab in range(n_lbl):
            yl = ytr[:, lab]
            if yl.sum() == 0 or yl.sum() == len(yl):
                continue
            pos = [_Inst(split.train_X[i], SPAM, i) for i in np.where(yl > 0)[0]]
            neg = [_Inst(split.train_X[i], HAM, i) for i in np.where(yl == 0)[0]]
            # cap labeled complaints per label (RuleCleaner = limited-labeled-data
            # method; small sets are on-protocol and keep the O(n^2) repair fast)
            rng = np.random.RandomState(seed + lab)
            if len(pos) > max_pos:
                pos = [pos[i] for i in rng.choice(len(pos), max_pos, replace=False)]
            if len(neg) > max_neg:
                neg = [neg[i] for i in rng.choice(len(neg), max_neg, replace=False)]
            rules = _refine_one_label(vocab, cand, pos, neg, val_insts, repair_strategy)
            yval_l = yval[:, lab]
            for rule, j in zip(rules, cand[:len(rules)]):
                pred = _rule_predict(rule, val_insts)
                f1 = common.score(yval_l.reshape(-1, 1), pred.reshape(-1, 1))["macro_f1"]
                pattern_score[j] = max(pattern_score[j], f1)
                n_refined += 1

    # Rank: refined-rule value first, then MI order for the long tail (parity
    # with weshap/localboost downstream feature budget).
    order = np.argsort(-pattern_score)
    refined_cols = [j for j in order if pattern_score[j] > 0]
    tail = [j for j in mi_order if j not in set(refined_cols)]
    ranked = (refined_cols + tail)[: min(top_k, n_feat)]
    cols = np.asarray(ranked, dtype=int)

    metrics = _fit_score(Xtr, ytr, Xval, yval, Xte, yte, cols, seed)
    metrics["n_annotations"] = 0
    metrics["extra"] = {
        "source": "JayLi2018/RuleCleanerKDD25 TreeRule repair",
        "note": "LocalBoost-slot substitute (KDD'25); reported under own name 'RuleCleaner'",
        "top_k": int(len(cols)), "n_patterns": int(n_feat),
        "rules_refined": int(n_refined), "repair_strategy": repair_strategy,
        "wall_sec": float(t.sec),
    }
    return metrics


# Registered under the LocalBoost slot key so the runner/aggregator pick it up;
# the result 'source'/'note' disclose it is RuleCleaner reported under its name.
BASELINES = {"localboost": localboost}
