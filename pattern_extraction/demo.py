"""
pattern_extraction/demo.py
----------------------------
Small self-contained demonstration of the LORIS pattern abstraction pipeline.

Runs two scenarios on synthetic data:
  A) Single-label string labels (auto k-selection via silhouette score)
  B) Multi-hot labels (explicit k=3)

Then demonstrates PatternStore.apply() on held-out documents.

Usage
-----
    # From inside the pattern_extraction/ directory:
    python demo.py

    # From the project root:
    python pattern_extraction/demo.py
"""

from __future__ import annotations

# Allow running as a standalone script from any working directory:
#   python demo.py                          (from inside pattern_extraction/)
#   python pattern_extraction/demo.py       (from project root)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

# _DOCS_FINANCE = [
#     "The stock market rose sharply after the Federal Reserve held interest rates steady.",
#     "Investors poured money into bank shares following strong quarterly earnings.",
#     "The bond market reacted calmly to the latest inflation figures.",
#     "Hedge funds increased their positions in commodities as the dollar weakened.",
#     "The central bank signalled it would raise borrowing costs by 25 basis points.",
#     "Tech stocks led gains on Wall Street amid optimism about AI earnings.",
# ]
_DOCS_FINANCE = [
    "The stock market rose sharply as the market reacted after the Federal Reserve held interest rates steady.",
    "Investors poured money into the market, and the bank shares market rallied following strong quarterly earnings.",
    "The bond market reacted calmly as the market absorbed the latest inflation figures.",
    "Hedge funds increased their positions as the commodities market expanded while the market responded to a weaker dollar.",
    "The central bank signalled it would raise borrowing costs, and the market expected the market reaction to be gradual.",
    "Tech stocks led gains as the market on Wall Street rallied, and the market showed optimism about AI earnings.",
]

# _DOCS_MEDICAL = [
#     "Researchers identified a new gene variant associated with increased cancer risk.",
#     "The clinical trial showed the drug reduced tumour size in 70% of patients.",
#     "Patients with diabetes are advised to monitor blood glucose levels daily.",
#     "A new mRNA vaccine demonstrated efficacy against multiple viral strains.",
#     "The hospital reported three cases of an antibiotic-resistant infection.",
#     "Surgeons performed the first robotic knee replacement using the new system.",
# ]

_DOCS_MEDICAL = [
    "Researchers identified a new gene variant in the medical record of a patient with increased cancer risk.",
    "The clinical trial showed that the medical drug reduced tumour size in 70% of the patient group.",
    "Patients with diabetes are advised by medical staff to monitor blood glucose levels daily as a patient care routine.",
    "A new medical mRNA vaccine demonstrated efficacy in protecting the patient against multiple viral strains.",
    "The hospital reported three cases where a patient developed a medical antibiotic-resistant infection.",
    "Before the medical robotic knee replacement, surgeons prepared the patient using the new system.",
]

_DOCS_TECH = [
    "The software update introduces a new API for third-party developers.",
    "Microchip shortages continue to disrupt global semiconductor supply chains.",
    "The startup raised $50M in Series B funding to expand its cloud platform.",
    "Engineers deployed the model to production after passing all regression tests.",
    "The operating system patch fixes a critical buffer-overflow vulnerability.",
    "Data centres are switching to liquid cooling to improve energy efficiency.",
]

_DOCS_SPORTS = [
    "The home team scored three goals in the final 10 minutes to win the match.",
    "The athlete broke the world record in the 100-metre sprint by 0.02 seconds.",
    "Injuries to key players forced the coach to field a depleted squad.",
    "The tennis player withdrew from the tournament citing a shoulder injury.",
    "The club signed a new striker from a rival league for a record transfer fee.",
    "Bad weather delayed the cycling stage by two hours on Saturday.",
]

ALL_DOCS = _DOCS_FINANCE + _DOCS_MEDICAL + _DOCS_TECH + _DOCS_SPORTS
SINGLE_LABELS = (
    ["finance"] * len(_DOCS_FINANCE)
    + ["medical"] * len(_DOCS_MEDICAL)
    + ["tech"] * len(_DOCS_TECH)
    + ["sports"] * len(_DOCS_SPORTS)
)

# Multi-hot: 4 classes, random binary matrix (reproducible)
rng = np.random.default_rng(0)
MULTI_HOT = rng.integers(0, 2, size=(len(ALL_DOCS), 4)).astype(np.int32)

# Held-out documents for PatternStore.apply() demo
HELD_OUT = [
    "The bank announced a 10% dividend increase for shareholders.",
    "Scientists discovered a protein that blocks tumour growth.",
    "The latest GPU release pushes graphics performance to new heights.",
    "The sprinter qualified for the Olympic final with a personal best.",
]

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    from pattern_extraction import (
        Document,
        PatternAbstractor,
        PatternStore,
    )

    separator = "=" * 70

    # -----------------------------------------------------------------------
    # Scenario A: Single-label, auto k-selection
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("Scenario A — Single-label | auto k-selection")
    print(separator)

    abs_a = PatternAbstractor(
        n_clusters=None,          # auto
        max_auto_k=6,
        tfidf_top_k=15,
        min_coverage=0.05,
        max_coverage=0.85,
        max_entropy_threshold=1.2,
        max_pairs_per_cluster=200,
        random_state=42,
    )
    abs_a.fit(ALL_DOCS, SINGLE_LABELS)

    print(f"\n  Clusters found     : {abs_a.n_clusters_}")
    print(f"  Patterns surviving : {len(abs_a.patterns_)}")
    print(f"  Classes            : {abs_a.label_encoder_.classes_.tolist()}")

    store_a = abs_a.to_store()
    print(f"\n  {store_a}")

    if abs_a.patterns_:
        print("\n  Sample predicates:")
        for pred in abs_a.patterns_:
            print(f"    {pred}")

    # -----------------------------------------------------------------------
    # Scenario B: Multi-hot, explicit k=3
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("Scenario B — Multi-hot labels | explicit k=3")
    print(separator)

    abs_b = PatternAbstractor(
        n_clusters=3,
        tfidf_top_k=15,
        min_coverage=0.05,
        max_coverage=0.85,
        max_entropy_threshold=1.5,
        max_pairs_per_cluster=200,
        random_state=42,
    )
    abs_b.fit(ALL_DOCS, MULTI_HOT)

    print(f"\n  Patterns surviving : {len(abs_b.patterns_)}")
    store_b = abs_b.to_store()
    print(f"  {store_b}")

    # -----------------------------------------------------------------------
    # PatternStore.apply() — apply Scenario A store to held-out docs
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("PatternStore.apply() — applying Scenario A store to held-out docs")
    print(separator)

    if len(store_a) == 0:
        print("  (No patterns in store; skipping apply demo.)")
    else:
        matrix = store_a.apply(HELD_OUT)
        print(f"\n  Feature matrix shape: {matrix.shape}  (n_docs × n_patterns)")
        print(f"  Total True entries  : {int(matrix.sum())}")

        for i, doc_text in enumerate(HELD_OUT):
            matched = store_a.matching_predicates(doc_text)
            print(f"\n  Doc {i}: \"{doc_text[:60]}…\"")
            if matched:
                for pred in matched[:3]:
                    print(f"    ✓ {pred}")
                if len(matched) > 3:
                    print(f"    … and {len(matched) - 3} more")
            else:
                print("    (no patterns matched)")

    # -----------------------------------------------------------------------
    # PatternStore.save() / load() round-trip
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("PatternStore serialization round-trip")
    print(separator)

    import tempfile, os
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w"
    ) as tmp:
        tmp_path = tmp.name

    try:
        store_a.save(tmp_path)
        store_reloaded = PatternStore.load(tmp_path)
        size_kb = os.path.getsize(tmp_path) / 1024
        print(f"\n  Saved  : {len(store_a)} predicates  → {tmp_path} ({size_kb:.1f} KB)")
        print(f"  Loaded : {len(store_reloaded)} predicates  (match={len(store_a)==len(store_reloaded)})")

        # Verify: apply() gives identical results on reloaded store
        if len(store_a) > 0:
            matrix_reloaded = store_reloaded.apply(HELD_OUT)
            identical = (matrix == matrix_reloaded).all()
            print(f"  apply() identical after reload: {identical}")
    finally:
        os.unlink(tmp_path)

    # -----------------------------------------------------------------------
    # Document with mtd / ttl fields
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("Document with mtd/ttl — predicate targeting specific attributes")
    print(separator)

    from pattern_extraction import (
        CooccurPredicate, MatchPredicate, BeforePredicate,
    )
    from pattern_extraction.predicates import _Pattern

    doc = Document(
        cnt="The bank raised interest rates after consulting the Fed.",
        ttl="Finance: rate hike announced",
        mtd="source=Reuters date=2024-03-01",
    )
    p_cnt  = CooccurPredicate("cnt",  _Pattern(r"\bbank\b"),     _Pattern(r"\brate"))
    p_ttl  = MatchPredicate  ("ttl",  _Pattern(r"rate hike"))
    p_mtd  = MatchPredicate  ("mtd",  _Pattern(r"\d{4}-\d{2}-\d{2}"))
    p_miss = BeforePredicate ("cnt",  _Pattern(r"\bFed\b"),      _Pattern(r"\bbank\b"))

    for pred in [p_cnt, p_ttl, p_mtd, p_miss]:
        print(f"  {repr(pred):<55}  →  {pred(doc)}")

    print(f"\n{separator}")
    print("Demo complete — original scenarios.")
    print(separator)

    # ===================================================================
    # Feature 1: Semantic similarity matching (sim=True)
    # ===================================================================
    print(f"\n{separator}")
    print("Feature 1 — Semantic Similarity Matching (sim=True)")
    print(separator)

    from pattern_extraction import MatchPredicate as _MP
    from pattern_extraction import FreqPredicate as _FP

    # A MatchPredicate with exact regex will NOT match a paraphrase…
    p_exact = _MP("cnt", "quarterly earnings report")
    # …but a sim-enabled MatchPredicate will.
    p_sim = _MP("cnt", "quarterly earnings report", sim=True, threshold=0.5)

    doc_bank = Document(cnt="The bank reported record quarterly profits.")
    print(f"\n  Document: \"{doc_bank.cnt}\"")
    print(f"  {repr(p_exact):<55}  → {p_exact(doc_bank)}")
    print(f"  {repr(p_sim):<55}  → {p_sim(doc_bank)}")

    # FreqPredicate sim mode: count semantically similar sentences
    doc_multi = Document(
        cnt="The economy grew steadily. Financial markets rallied. "
            "Investors showed confidence. The weather was sunny."
    )
    p_freq_sim = _FP("cnt", "financial growth", op=">=", eta=2,
                      sim=True, threshold=0.3)
    print(f"\n  Document: \"{doc_multi.cnt[:70]}…\"")
    print(f"  {repr(p_freq_sim):<55}  → {p_freq_sim(doc_multi)}")

    # ===================================================================
    # Feature 2: Text Preprocessing
    # ===================================================================
    print(f"\n{separator}")
    print("Feature 2 — Text Preprocessing (Preprocessor)")
    print(separator)

    from pattern_extraction import Preprocessor

    prep = Preprocessor(
        lemmatize=True,
        normalize_dates=True,
        normalize_numbers=True,
        lowercase=True,
    )
    print(f"\n  {prep}")

    raw_text = "The patients were diagnosed on 2024-03-01 and costs reached $50M."
    processed = prep(raw_text)
    print(f"\n  Before: \"{raw_text}\"")
    print(f"  After : \"{processed}\"")

    # Preprocessor with PatternStore.apply()
    doc_raw = Document(
        cnt="Researchers published findings on January 15, 2024. "
            "The treatment cost $2.5M and helped 85% of patients.",
    )
    prep2 = Preprocessor(normalize_dates=True, normalize_numbers=True)
    doc_pp = prep2.process_document(doc_raw)
    print(f"\n  Raw doc.cnt : \"{doc_raw.cnt[:70]}…\"")
    print(f"  Preprocessed: \"{doc_pp.cnt[:70]}…\"")

    # Show PatternStore.apply() integration with preprocessor
    from pattern_extraction import PatternStore
    from pattern_extraction.predicates import _Pattern
    store_pp = PatternStore(
        patterns=[_MP("cnt", "<DATE>"), _MP("cnt", "<CURRENCY>")],
        n_classes=1,
    )
    matrix_pp = store_pp.apply([doc_raw], preprocessor=prep2)
    print(f"\n  PatternStore.apply() with preprocessor:")
    print(f"    Patterns: {[repr(p) for p in store_pp]}")
    print(f"    Match vector: {matrix_pp[0].tolist()}")

    # ===================================================================
    # Feature 3: MLPredicate & LabelPredicate
    # ===================================================================
    print(f"\n{separator}")
    print("Feature 3 — MLPredicate & LabelPredicate")
    print(separator)

    from pattern_extraction import LabelPredicate, MLPredicate, register_ml_model

    # --- LabelPredicate demos ---
    print("\n  --- LabelPredicate ---")
    doc_lbl = Document(cnt="Some text.", lbl={"finance", "tech"})
    print(f"  Document labels: {doc_lbl.lbl}")

    lp_contains = LabelPredicate("finance", op="contains")
    lp_eq       = LabelPredicate("finance,tech", op="eq")
    lp_subset   = LabelPredicate("finance,tech,medical", op="subset")
    lp_strict   = LabelPredicate("finance,tech,medical", op="strict_subset")

    for lp in [lp_contains, lp_eq, lp_subset, lp_strict]:
        print(f"    {repr(lp):<50}  → {lp(doc_lbl)}")

    # minus operation (side-effect: removes label)
    doc_lbl2 = Document(cnt="Some text.", lbl={"finance", "tech"})
    lp_minus = LabelPredicate("finance", op="minus")
    result_minus = lp_minus(doc_lbl2)
    print(f"    {repr(lp_minus):<50}  → {result_minus}  (labels after: {doc_lbl2.lbl})")

    # --- MLPredicate demo with dummy model ---
    print("\n  --- MLPredicate (dummy model) ---")

    class DummyClassifier:
        """Simple dummy classifier that returns 'positive' for finance-related text."""
        def predict(self, text: str) -> str:
            keywords = {"bank", "stock", "market", "finance", "invest"}
            if any(kw in text.lower() for kw in keywords):
                return "positive"
            return "negative"

    register_ml_model("dummy-finance", DummyClassifier())

    ml_pred = MLPredicate(model_name="dummy-finance", label="positive")
    doc_fin = Document(cnt="The stock market rallied on strong earnings.")
    doc_med = Document(cnt="The patient recovered after surgery.")

    print(f"  {repr(ml_pred)}")
    print(f"    \"{doc_fin.cnt[:50]}…\"  → {ml_pred(doc_fin)}")
    print(f"    \"{doc_med.cnt[:50]}…\"  → {ml_pred(doc_med)}")

    # --- Serialization round-trip for new predicate types ---
    print("\n  --- Serialization round-trip (new predicate types) ---")
    from pattern_extraction import predicate_to_dict, predicate_from_dict

    for pred in [ml_pred, lp_contains, lp_eq]:
        d = predicate_to_dict(pred)
        restored = predicate_from_dict(d)
        print(f"    {repr(pred):<50}  → dict → {repr(restored)}  (match={repr(pred)==repr(restored)})")

    # PatternStore with mixed predicate types
    mixed_store = PatternStore(
        patterns=[
            _MP("cnt", "bank"),
            lp_contains,
            ml_pred,
        ],
        n_classes=2,
    )
    print(f"\n  Mixed PatternStore: {mixed_store}")

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    try:
        mixed_store.save(tmp_path)
        mixed_reloaded = PatternStore.load(tmp_path)
        print(f"  Saved/loaded mixed store: {len(mixed_reloaded)} predicates (match={len(mixed_store)==len(mixed_reloaded)})")
    finally:
        os.unlink(tmp_path)

    print(f"\n{separator}")
    print("All demos complete.")
    print(separator)


def _run_auto_regex_demo() -> None:
    """Feature 4: AutoRegexExtractor demonstration."""
    from pattern_extraction import AutoRegexExtractor, PatternAbstractor

    separator = "=" * 70

    # ===================================================================
    # Feature 4: AutoRegexExtractor
    # ===================================================================
    print(f"\n{separator}")
    print("Feature 4 — AutoRegexExtractor (High-Frequency Structural Patterns)")
    print(separator)

    # --- Synthetic corpus with structural tokens ---
    auto_corpus = [
        "Order ORD-20240301-AB12 was shipped. Contact support with ref ORD-20240301-AB12.",
        "Order ORD-20240302-CD34 failed payment. See error ERR-4501 in the log.",
        "Error ERR-4501 occurred during checkout for order ORD-20240303-EF56.",
        "System version v2.1.0 deployed. Previous version was v2.0.9.",
        "Upgrade to v2.1.0 resolved error ERR-2203 in module auth.",
        "Invoice INV-20240301-001 generated for order ORD-20240304-GH78.",
        "Error ERR-2203 triggered alert. Incident ref INC-0042 opened.",
        "Deployed v2.1.0 to production. Build tag B20240301 confirmed.",
        "Order ORD-20240305-IJ90 completed. Confirmation code CNF-8877 sent.",
        "Error ERR-4501 in payment gateway. Fallback to ERR-2203 handler.",
    ]
    auto_labels = [
        "order", "order", "error", "deploy", "deploy",
        "order", "error", "deploy", "order", "error",
    ]

    # --- Standalone AutoRegexExtractor ---
    print("\n  --- Standalone AutoRegexExtractor ---")
    from pattern_extraction.auto_regex import AutoRegexExtractor as _ARE

    extractor = _ARE(min_freq=2)
    extractor.fit(auto_corpus)

    print(f"\n  {extractor}")
    patterns = extractor.get_patterns()
    print(f"  Discovered {len(patterns)} patterns:")
    for name, pat in patterns:
        print(f"    {name:<35}  regex: {pat.pattern}")

    # Show anchors extracted from the corpus
    anchors = extractor.get_anchors(auto_corpus)
    print(f"\n  Anchors extracted ({len(anchors)}):")
    for a in sorted(anchors)[:15]:
        print(f"    {a}")
    if len(anchors) > 15:
        print(f"    ... and {len(anchors) - 15} more")

    # --- Verify: new tokens matching discovered patterns ---
    print("\n  --- Verification: match new tokens ---")
    test_tokens = ["ORD-20250101-XY99", "ERR-9999", "v3.0.0", "hello"]
    for tok in test_tokens:
        matched_by = [
            name for name, pat in patterns if pat.search(tok)
        ]
        status = "MATCH" if matched_by else "no match"
        print(f"    {tok:<25}  {status}  {matched_by}")

    # --- Integration with PatternAbstractor ---
    print(f"\n  --- PatternAbstractor(auto_regex=True) integration ---")
    abs_auto = PatternAbstractor(
        n_clusters=None,
        max_auto_k=4,
        tfidf_top_k=10,
        min_coverage=0.05,
        max_coverage=0.95,
        max_entropy_threshold=1.5,
        max_pairs_per_cluster=100,
        random_state=42,
        auto_regex=True,
        auto_regex_min_freq=2,
    )
    abs_auto.fit(auto_corpus, auto_labels)

    print(f"\n  Clusters: {abs_auto.n_clusters_}")
    print(f"  Patterns surviving screening: {len(abs_auto.patterns_)}")

    # Show MatchPredicates that came from auto-discovered regex
    from pattern_extraction.predicates import MatchPredicate as _MP2
    match_preds = [p for p in abs_auto.patterns_ if isinstance(p, _MP2)]
    print(f"  MatchPredicates: {len(match_preds)}")
    for p in match_preds[:10]:
        print(f"    {p}")

    print(f"\n{separator}")
    print("Feature 4 demo complete.")
    print(separator)


if __name__ == "__main__":
    run_demo()
    _run_auto_regex_demo()
