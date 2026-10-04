"""Discriminative, support-aware selection of per-cluster pattern candidates.

Screening policy ``discriminative`` (2026-09-30). The paper retains candidate
predicates "with sufficient coverage and discriminability" (Sec. 5.1) and
leaves final selection to labeling F1. The historical label-entropy screen
gives every single-document pattern the minimum entropy 0, so it kept almost
the whole inventory (60-85% of retained Reuters patterns fired on one fit
document). This policy instead

1. requires cluster support >= max(min_support, ceil(min_support_fraction*|C|))
   and fit-wide coverage <= max_fit_fraction (neither rare nor near-universal);
2. merges candidates with identical fit-wide masks (first by family order);
3. scores a candidate by the binary mutual information between its fit-wide
   mask and each label that has positives in the cluster (support-aware: rare
   and near-universal masks carry little information);
4. keeps, per label, its top-k candidates, then fills the per-cluster budget
   by best score, with a per-family share cap.

Only known-train (fit) text and answers are read.
"""
import math

import numpy as np

FAMILY_ORDER = {"Match": 0, "Cooccur": 1, "Before": 2, "Freq": 3}
ATTRIBUTE_ORDER = {"cnt": 0, "ttl": 1}


def binary_mutual_information(tp, support, positives, n):
    """I(pattern; label) in nats for count arrays (broadcast), exact formula."""
    tp = np.asarray(tp, np.float64)
    support = np.asarray(support, np.float64)
    positives = np.asarray(positives, np.float64)
    n = float(n)
    result = np.zeros(np.broadcast(tp, support, positives).shape)
    cells = (
        (tp, support, positives),
        (support - tp, support, n - positives),
        (positives - tp, n - support, positives),
        (n - support - positives + tp, n - support, n - positives),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        for joint, px, py in cells:
            pj = joint / n
            term = pj * np.log(pj * n * n / (px * py))
            result += np.where(pj > 0, term, 0.0)
    return np.maximum(result, 0.0)


def label_scores(masks, truth, labels):
    """MI of each fit-wide mask with each selected label: (P, len(labels))."""
    if not len(masks) or not len(labels):
        return np.zeros((len(masks), len(labels)))
    m = np.asarray(masks, bool)
    y = np.asarray(truth, bool)[:, labels]
    tp = m.astype(np.float32) @ y.astype(np.float32)
    support = m.sum(axis=1)[:, None]
    positives = y.sum(axis=0)[None, :]
    return binary_mutual_information(np.rint(tp), support, positives, m.shape[1])


def select(items, *, truth, labels, budget, per_label=None, family_share_cap=0.5):
    """Return (chosen, duplicates, outside_budget) item lists.

    items: dicts with keys spec, family, attr, all_fit_mask, support.
    Each chosen item gets 'score', 'score_label' and 'selection'.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("positive per-cluster pattern budget required")
    ordered = sorted(items, key=lambda it: (ATTRIBUTE_ORDER.get(it["attr"], 9),
                                            FAMILY_ORDER.get(it["family"], 9), it["spec"].key))
    unique, duplicates, seen = [], [], {}
    for item in ordered:
        key = np.packbits(item["all_fit_mask"]).tobytes()
        if key in seen:
            duplicates.append((item, seen[key]["spec"].key))
        else:
            seen[key] = item
            unique.append(item)
    labels = [int(x) for x in labels]
    scores = label_scores([it["all_fit_mask"] for it in unique], truth, labels)
    best = scores.max(axis=1) if scores.size else np.zeros(len(unique))
    best_label = scores.argmax(axis=1) if scores.size else np.zeros(len(unique), int)
    for i, item in enumerate(unique):
        item["score"] = float(best[i])
        item["score_label"] = labels[int(best_label[i])] if labels else None
    if per_label is None:
        per_label = max(2, min(32, budget // (2 * max(1, len(labels)))))
    family_cap = max(1, int(math.ceil(family_share_cap * budget)))
    chosen, used, family_counts = [], set(), {}

    def take(i, how, label=None):
        item = unique[i]
        family = item["family"]
        if i in used or family_counts.get(family, 0) >= family_cap or len(chosen) >= budget:
            return False
        used.add(i)
        family_counts[family] = family_counts.get(family, 0) + 1
        item["selection"] = how
        item["selection_label"] = label
        chosen.append(item)
        return True

    tie = np.array([-it["support"] for it in unique], np.float64)
    for column, label in enumerate(labels):
        if len(chosen) >= budget:
            break
        column_scores = scores[:, column]
        order = np.lexsort((np.arange(len(unique)), tie, -column_scores))
        taken = 0
        for i in order:
            if taken >= per_label or column_scores[i] <= 0:
                break
            taken += take(int(i), "per_label_mutual_information", label)
    order = np.lexsort((np.arange(len(unique)), tie, -best))
    for i in order:
        if len(chosen) >= budget:
            break
        take(int(i), "global_mutual_information_fill")
    outside = [unique[i] for i in range(len(unique)) if i not in used]
    return chosen, duplicates, outside
