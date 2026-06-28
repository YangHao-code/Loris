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
               repair_strategy: str = "naive", max_rules: int = 12,
               max_pos: int = 20, max_neg: int = 20, **kwargs) -> dict:
    """RuleCleaner rule-refinement classifier (authors' code); LocalBoost-slot.

    Reported under its own name 'RuleCleaner'. This builds a genuine RULE-BASED
    classifier from the authors' refined TreeRules (NOT a pattern ranker feeding
    a logistic model — that collapsed to filter_mi). Per label: take the top
    ``max_rules`` MI keyword patterns, build + REPAIR a TreeRule for each against
    a small labeled "complaint" set (``fix_violations``), then PREDICT each test
    doc by the refined rules' vote (label on iff the mean rule firing exceeds a
    val-tuned global threshold). This makes RuleCleaner's refinement the thing
    being measured, distinct from the filter/Shapley selectors.

    kwargs: top_k (unused for prediction; kept for CLI parity), n_patterns,
    repair_strategy ('naive'|'information_gain'), max_rules (rules/label),
    max_pos/max_neg (labeled complaints/label — RuleCleaner is a
    limited-labeled-data method, so small sets are on-protocol and keep the
    O(complaints^2) repair tractable).
    """
    from loris.baselines.common import has_raw_text
    # RuleCleaner is a keyword/text TREE-RULE method (predicates match words in
    # the document text). On datasets without raw text (rcv1 = hashed TF-IDF
    # tokens like 'w5215'), keyword rules are meaningless — and the authors'
    # repair path even crashes on the pseudo-tokens. Skip by design, exactly as
    # the transformer/text baselines do on rcv1 (paper footnote).
    if not has_raw_text(split.dataset):
        raise NotImplementedError(
            f"RuleCleaner (localboost) requires raw text; dataset "
            f"'{split.dataset}' ships hashed TF-IDF only.")
    Xtr, Xval, Xte, vocab, vec = _top_keyword_patterns(split, n_patterns)
    ytr = np.asarray(split.train_y, dtype=int)
    yval = np.asarray(split.val_y, dtype=int)
    yte = np.asarray(split.test_y, dtype=int)
    n_lbl = ytr.shape[1]
    n_feat = Xtr.shape[1]

    val_insts = [_Inst(split.val_X[i], 0, i) for i in range(len(split.val_X))]
    test_insts = [_Inst(split.test_X[i], 0, i) for i in range(len(split.test_X))]

    # Per-label candidate keyword patterns: rank by chi2 of the binary pattern
    # vs the label (fast, computed once for all labels via sklearn.chi2). This
    # only picks WHICH patterns become candidate rules; the refinement is the
    # authors' fix_violations.
    from sklearn.feature_selection import chi2 as _chi2
    label_cand = {}
    for lab in range(n_lbl):
        yl = ytr[:, lab]
        if yl.sum() == 0 or yl.sum() == len(yl):
            label_cand[lab] = np.array([], dtype=int)
            continue
        try:
            sc, _ = _chi2(Xtr, yl)
            sc = np.nan_to_num(sc)
            label_cand[lab] = np.argsort(-sc)[: min(max_rules, n_feat)]
        except Exception:
            label_cand[lab] = np.arange(min(max_rules, n_feat))

    # Per-label refined-rule firing scores on val + test (mean over the label's
    # refined rules), then one global threshold tuned on val (LORIS convention).
    val_scores = np.zeros((len(val_insts), n_lbl), dtype=np.float64)
    test_scores = np.zeros((len(test_insts), n_lbl), dtype=np.float64)
    n_refined = 0
    with common.Timer() as t, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lab in range(n_lbl):
            yl = ytr[:, lab]
            cand = label_cand[lab]
            if cand.size == 0:
                continue
            pos = [_Inst(split.train_X[i], SPAM, i) for i in np.where(yl > 0)[0]]
            neg = [_Inst(split.train_X[i], HAM, i) for i in np.where(yl == 0)[0]]
            rng = np.random.RandomState(seed + lab)
            if len(pos) > max_pos:
                pos = [pos[i] for i in rng.choice(len(pos), max_pos, replace=False)]
            if len(neg) > max_neg:
                neg = [neg[i] for i in rng.choice(len(neg), max_neg, replace=False)]
            rules = _refine_one_label(vocab, cand, pos, neg, val_insts, repair_strategy)
            if not rules:
                continue
            n_refined += len(rules)
            # mean firing across the label's refined rules
            vacc = np.zeros(len(val_insts), dtype=np.float64)
            tacc = np.zeros(len(test_insts), dtype=np.float64)
            for rule in rules:
                vacc += _rule_predict(rule, val_insts)
                tacc += _rule_predict(rule, test_insts)
            val_scores[:, lab] = vacc / len(rules)
            test_scores[:, lab] = tacc / len(rules)

    metrics = common.score_from_scores(test_scores, yte, val_scores, yval)
    metrics["n_annotations"] = 0
    metrics["extra"] = {
        "source": "JayLi2018/RuleCleanerKDD25 TreeRule repair (rule-based classifier)",
        "note": "LocalBoost-slot substitute (KDD'25); reported under own name 'RuleCleaner'",
        "rules_refined": int(n_refined), "max_rules": int(max_rules),
        "repair_strategy": repair_strategy, "n_patterns": int(n_feat),
        "wall_sec": float(t.sec),
    }
    return metrics


# Registered under the LocalBoost slot key so the runner/aggregator pick it up;
# the result 'source'/'note' disclose it is RuleCleaner reported under its name.
BASELINES = {"localboost": localboost}
