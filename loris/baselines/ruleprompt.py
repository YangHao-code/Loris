"""RulePrompt baseline (Li et al., WWW'24).

Weakly-supervised text classification via prompting PLMs with self-iterative
*logical rules*.  This is a **first-pass** re-implementation tailored to the
LORIS multi-label baseline harness — validate against the authors' reference
later.

Pipeline
--------
1. **Seed rules.**  For each label, derive the top-``n_keywords`` discriminative
   keywords from the *training* split by a mutual-information / frequency-ratio
   score (presence-of-term vs. label).  These keywords are the human-readable
   "rule" hints for that label.
2. **Prompt.**  For each (sub-sampled) TEST document, prompt the LLM with the
   candidate labels **and** their keyword-rule hints, asking which labels apply.
3. **Iterate** (``iters`` > 1).  Re-derive each label's keyword rules from the
   set of test docs the LLM assigned to that label (pseudo-agreement), blended
   with the train-derived seed keywords, then re-prompt.

Mock mode (``LORIS_LLM_MOCK=1`` / no ``OPENAI_API_KEY``) returns empty
predictions but still executes the full control flow, so the module is
smoke-testable offline.  ``rcv1`` (hashed TF-IDF, no raw text) raises
``NotImplementedError``.

Contract: ``BASELINES = {name: fn}`` with ``fn(split, seed=0, **kwargs) -> dict``.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from loris.baselines.common import has_raw_text, score

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]{2,}")

# Small English stop list — keep keyword rules content-bearing.
_STOP = frozenset("""
a an the and or but if then else for to of in on at by with from as is are was
were be been being this that these those it its their our your his her they them
we you he she i not no nor so than too very can will just dont don't more most
some such only own same out up down over under again further once here there all
any both each few other into through during before after above below between out
""".split())


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _doc_term_sets(texts: List[str]) -> List[set]:
    return [set(_tokenize(t)) for t in texts]


def _derive_keyword_rules(
    term_sets: List[set],
    y: np.ndarray,
    label_names: List[str],
    n_keywords: int,
    min_df: int = 2,
) -> Dict[str, List[str]]:
    """Top-``n_keywords`` discriminative terms per label.

    Score = presence-based mutual-information-style ratio: how much more likely
    a term appears in docs WITH the label vs. its overall base rate, weighted by
    in-label document frequency (so rare-but-pure terms don't dominate).
    """
    n_docs = len(term_sets)
    y = np.asarray(y)
    # Global document frequency per term.
    global_df: Dict[str, int] = {}
    for ts in term_sets:
        for term in ts:
            if term in _STOP:
                continue
            global_df[term] = global_df.get(term, 0) + 1

    rules: Dict[str, List[str]] = {}
    for j, lname in enumerate(label_names):
        pos_idx = np.nonzero(y[:, j] > 0)[0]
        n_pos = len(pos_idx)
        if n_pos == 0:
            rules[lname] = []
            continue
        in_df: Dict[str, int] = {}
        for i in pos_idx:
            for term in term_sets[i]:
                if term in _STOP:
                    continue
                in_df[term] = in_df.get(term, 0) + 1
        base_rate = n_pos / max(n_docs, 1)
        scored = []
        for term, df_in in in_df.items():
            if df_in < min_df:
                continue
            p_term_in = df_in / n_pos                      # P(term | label)
            p_term_all = global_df.get(term, df_in) / n_docs  # P(term)
            # lift over base rate, smoothed; weight by in-label coverage.
            lift = (p_term_in + 1e-6) / (p_term_all + 1e-6)
            score_val = float(np.log(1.0 + lift) * p_term_in)
            # discount terms that are common everywhere relative to in-label.
            purity = df_in / max(global_df.get(term, df_in), 1)
            score_val *= (0.5 + 0.5 * purity)
            # ignore terms not at all enriched for the label.
            if p_term_in <= base_rate:
                continue
            scored.append((score_val, term))
        scored.sort(reverse=True)
        rules[lname] = [t for _, t in scored[:n_keywords]]
    return rules


def _build_user_prompt(text: str, rules: Dict[str, List[str]], max_chars: int) -> str:
    lines = ["Candidate labels and their characteristic keyword rules:"]
    for lname, kws in rules.items():
        hint = ", ".join(kws) if kws else "(no strong keywords)"
        lines.append(f"- {lname}: {hint}")
    lines.append("")
    lines.append("Document:")
    lines.append(text[:max_chars])
    lines.append("")
    lines.append(
        "Using the keyword rules as hints (a label is more likely if the "
        "document matches its keywords), return ONLY a JSON array of the "
        "applicable labels, drawn verbatim from the candidate set."
    )
    return "\n".join(lines)


_SYSTEM = (
    "You are a precise multi-label text classifier guided by per-label keyword "
    "rules. You will be given a document, a fixed set of candidate labels, and "
    "characteristic keywords for each label. Return ONLY a JSON array (possibly "
    "empty) of the labels that apply, drawn verbatim from the candidate set. "
    "No explanation."
)


def ruleprompt(split, seed: int = 0, **kwargs) -> dict:
    """RulePrompt: keyword-rule-guided LLM multi-label classification.

    kwargs
    ------
    n_keywords : int = 8      keywords per label rule
    iters      : int = 1      self-iterative refinement passes (>=1)
    max_test   : int = 1000   test docs to evaluate (sub-sampled w/ seed)
    max_chars  : int = 6000   document truncation for the prompt
    min_df     : int = 2      min in-label doc-freq for a keyword
    """
    if not has_raw_text(split.dataset):
        raise NotImplementedError(
            f"ruleprompt requires raw text; dataset '{split.dataset}' ships "
            "without it (e.g. rcv1 hashed TF-IDF)."
        )

    n_keywords = int(kwargs.get("n_keywords", 8))
    iters = max(1, int(kwargs.get("iters", 1)))
    max_test = int(kwargs.get("max_test", 1000))
    max_chars = int(kwargs.get("max_chars", 6000))
    min_df = int(kwargs.get("min_df", 2))

    from loris.baselines.llm_client import LLMClient

    rng = np.random.RandomState(seed)
    label_names = list(split.label_names)
    n_labels = len(label_names)

    # ── sub-sample test ───────────────────────────────────────────────────────
    n_test_all = len(split.test_X)
    if n_test_all > max_test:
        sel = rng.choice(n_test_all, size=max_test, replace=False)
        sel.sort()
    else:
        sel = np.arange(n_test_all)
    test_X = [split.test_X[i] for i in sel]
    test_y = np.asarray(split.test_y)[sel]
    n_test = len(test_X)

    # ── seed rules from train ─────────────────────────────────────────────────
    train_term_sets = _doc_term_sets(split.train_X)
    seed_rules = _derive_keyword_rules(
        train_term_sets, split.train_y, label_names, n_keywords, min_df
    )
    rules = {l: list(kw) for l, kw in seed_rules.items()}

    cli = LLMClient()
    test_term_sets = _doc_term_sets(test_X)
    name_to_idx = {l: j for j, l in enumerate(label_names)}

    preds = np.zeros((n_test, n_labels), dtype=np.int8)
    for it in range(iters):
        preds = np.zeros((n_test, n_labels), dtype=np.int8)
        for i, text in enumerate(test_X):
            user = _build_user_prompt(text, rules, max_chars)
            reply = cli.chat(_SYSTEM, user)
            applied = LLMClient._parse_label_list(reply, label_names)
            for lab in applied:
                preds[i, name_to_idx[lab]] = 1

        # ── self-iterative rule refinement from pseudo-agreement ─────────────
        if it < iters - 1 and preds.sum() > 0:
            refined = _derive_keyword_rules(
                test_term_sets, preds, label_names, n_keywords, min_df=1
            )
            # blend: keep seed keywords, append newly-discovered ones.
            blended: Dict[str, List[str]] = {}
            for l in label_names:
                merged = list(dict.fromkeys(seed_rules.get(l, []) + refined.get(l, [])))
                blended[l] = merged[: max(n_keywords, len(seed_rules.get(l, [])))]
            rules = blended

    metrics = score(test_y, preds)
    metrics["n_annotations"] = 0  # no human labels; LLM-only weak supervision
    metrics["extra"] = {
        "cost": cli.cost_summary(),
        "rules": rules,
        "n_test_eval": n_test,
        "iters": iters,
        "n_keywords": n_keywords,
        "mock": cli.mock,
    }
    return metrics


BASELINES = {
    "ruleprompt": ruleprompt,
}
