# LORIS predicate / rule statistics

Runs aggregated: **7**  ·  total rules: **380**  ·  total predicates: **522**

## Per-experiment breakdown

Each experiment cell (method × dataset) separately: rules, predicate counts, ML-vs-logic split, consequence ops, body length, discovery stages.

| experiment | dataset | rules | preds | ML | logic | add | remove | avg body | stages |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- |
| A_router_aapd_s0 | aapd | 77 | 103 | 103 | 0 | 76 | 1 | 1.338 | stage0_ml:76,stage2_fp_remove:1 |
| A_router_arxiv_s0 | arxiv | 69 | 90 | 89 | 1 | 69 | 0 | 1.304 | stage0_ml:68,stage1_fn_add:1 |
| A_router_bgc_s0 | bgc | 64 | 89 | 85 | 4 | 64 | 0 | 1.391 | stage0_ml:60,stage1_fn_add:4 |
| A_router_goodreads_s0 | goodreads | 46 | 69 | 65 | 4 | 44 | 2 | 1.5 | stage0_ml:42,stage2_fp_remove:2,stage1_fn_add:1,stage3_prop:1 |
| A_router_hupd_s0 | hupd | 57 | 79 | 79 | 0 | 57 | 0 | 1.386 | stage0_ml:57 |
| A_router_pubmed_s0 | pubmed | 34 | 58 | 41 | 17 | 34 | 0 | 1.706 | stage0_ml:23,stage3_prop:6,stage1_fn_add:5 |
| A_router_reuters21578_s0 | reuters21578 | 33 | 34 | 34 | 0 | 33 | 0 | 1.03 | stage0_ml:33 |

Per-experiment predicate-type counts and representative rules are in `rule_stats.json` under `per_run[].predicate_counts_by_type` and `per_run[].representative_rules`.

## Aggregate (all experiments)

## ML vs logic predicates

| class | count | share |
| :-- | --: | --: |
| ML (MLPredicate, MLThreshold) | 496 | 95.0% |
| logic (text/label/graph) | 26 | 5.0% |

## Predicate count by type

| type | role | count |
| :-- | :-- | --: |
| MLThresholdPredicate | ml | 496 |
| MatchPredicate | logic_text | 18 |
| LabelPredicate | label | 7 |
| FreqPredicate | logic_text | 1 |

## Roles

| role | count |
| :-- | --: |
| ml | 496 |
| logic_text | 19 |
| label | 7 |

## Consequence ops · discovery stage

| op | count |  | stage | count |
| :-- | --: | -- | :-- | --: |
| add | 377 |  | stage0_ml | 359 |
| remove | 3 |  | stage1_fn_add | 11 |
|  |  |  | stage3_prop | 7 |
|  |  |  | stage2_fp_remove | 3 |

## Rule body length (arity) distribution

| body_len | #rules |
| --: | --: |
| 1 | 238 |
| 2 | 142 |

## Rules per dataset

| dataset | #rules |
| :-- | --: |
| aapd | 77 |
| arxiv | 69 |
| bgc | 64 |
| hupd | 57 |
| goodreads | 46 |
| pubmed | 34 |
| reuters21578 | 33 |

## Representative rules (top by F1-gain, per dataset)

### aapd

- `ml_thresh('loris_clf_tfidf_svm_unigram_c0', label='cs.it', thresh=0.55) → -cs.ai`  — score=2.6458, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_encoder_mlp_c0', label='cs.it', thresh=0.3) ∧ ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='cs.it', thresh=0.4) → +cs.it`  — score=0.9009, len=2, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='math.it', thresh=0.48) → +math.it`  — score=0.9002, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='quant-ph', thresh=0.36) → +quant-ph`  — score=0.8994, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_encoder_mlp_c0', label='physics.soc-ph', thresh=0.45) → +physics.soc-ph`  — score=0.8111, len=1, stage=stage0_ml

### arxiv

- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='math.NA', thresh=0.47) → +math.NA`  — score=0.8803, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='cs.NA', thresh=0.47) → +cs.NA`  — score=0.8803, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='cs.CV', thresh=0.45) → +cs.CV`  — score=0.8543, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='quant-ph', thresh=0.38) → +quant-ph`  — score=0.8525, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_unigram_c0', label='cs.RO', thresh=0.44) → +cs.RO`  — score=0.8502, len=1, stage=stage0_ml

### bgc

- `ml_thresh('loris_clf_encoder_mlp_c0', label='Nonfiction', thresh=0.54) → +Nonfiction`  — score=0.9469, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='Fiction', thresh=0.4) ∧ ml_thresh('loris_clf_encoder_mlp_c0', label='Fiction', thresh=0.2) → +Fiction`  — score=0.9263, len=2, stage=stage0_ml
- `ml_thresh('loris_clf_encoder_mlp_c0', label='Children’s Books', thresh=0.62) → +Children’s Books`  — score=0.9138, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='Cooking', thresh=0.42) → +Cooking`  — score=0.905, len=1, stage=stage0_ml
- `match(cnt, 'stories') ∧ ml_thresh('loris_clf_encoder_mlp_c0', label='Fiction', thresh=0.3) → +Fiction`  — score=0.8333, len=2, stage=stage1_fn_add

### goodreads

- `ml_thresh('loris_clf_encoder_mlp_c0', label='Science Fiction & Fantasy', thresh=0.25) → -Sports & Recreation`  — score=2.6458, len=1, stage=stage2_fp_remove
- `freq(cnt, '\b(?:woman|women)\b' >= 2.0) ∧ ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='Literature & Fiction', thresh=0.25) → -Sports & Recreation`  — score=1.86, len=2, stage=stage2_fp_remove
- `match(cnt, 'science') ∧ ml_thresh('loris_clf_encoder_mlp_c0', label='Non-Fiction', thresh=0.15) → +Non-Fiction`  — score=0.8571, len=2, stage=stage1_fn_add
- `ml_thresh('loris_clf_encoder_mlp_c0', label='Literature & Fiction', thresh=0.5) ∧ ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='Literature & Fiction', thresh=0.4) → +Literature & Fiction`  — score=0.8407, len=2, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_unigram_c0', label='Food & Cooking', thresh=0.34) → +Food & Cooking`  — score=0.8, len=1, stage=stage0_ml

### hupd

- `ml_thresh('loris_clf_tfidf_svm_unigram_c0', label='H01M', thresh=0.3) ∧ ml_thresh('loris_clf_encoder_mlp_c0', label='H01M', thresh=0.2) → +H01M`  — score=0.8644, len=2, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='H01L', thresh=0.4) ∧ ml_thresh('loris_clf_encoder_mlp_c0', label='H01L', thresh=0.2) → +H01L`  — score=0.8611, len=2, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_unigram_c0', label='E21B', thresh=0.37) → +E21B`  — score=0.8602, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_unigram_c0', label='A61K', thresh=0.4) ∧ ml_thresh('loris_clf_encoder_mlp_c0', label='A61K', thresh=0.2) → +A61K`  — score=0.798, len=2, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='H02J', thresh=0.41) → +H02J`  — score=0.7826, len=1, stage=stage0_ml

### pubmed

- `match(cnt, 'families') ∧ ml_thresh('loris_clf_encoder_mlp_c0', label='D', thresh=0.2) → +D`  — score=1.0, len=2, stage=stage1_fn_add
- `label('Z', contains) ∧ match(cnt, 'nursing') → +M`  — score=1.0, len=2, stage=stage3_prop
- `ml_thresh('loris_clf_encoder_mlp_c0', label='B', thresh=0.3) ∧ ml_thresh('loris_clf_tfidf_svm_unigram_c0', label='B', thresh=0.5) → +B`  — score=0.9807, len=2, stage=stage0_ml
- `ml_thresh('loris_clf_encoder_mlp_c0', label='D', thresh=0.66) → +D`  — score=0.9257, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_encoder_mlp_c0', label='C', thresh=0.6) ∧ ml_thresh('loris_clf_tfidf_svm_unigram_c0', label='C', thresh=0.3) → +C`  — score=0.9029, len=2, stage=stage0_ml

### reuters21578

- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='copper', thresh=0.35) → +copper`  — score=1.0, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_encoder_mlp_c0', label='earn', thresh=0.45) → +earn`  — score=0.991, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_encoder_mlp_c0', label='acq', thresh=0.26) → +acq`  — score=0.9804, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='veg-oil', thresh=0.4) → +veg-oil`  — score=0.9714, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram_c0', label='sugar', thresh=0.45) → +sugar`  — score=0.9688, len=1, stage=stage0_ml

