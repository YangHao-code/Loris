# LORIS predicate / rule statistics

Runs aggregated: **45**  ·  total rules: **2788**  ·  total predicates: **3572**

## Per-experiment breakdown

Each experiment cell (method × dataset) separately: rules, predicate counts, ML-vs-logic split, consequence ops, body length, discovery stages.

| experiment | dataset | rules | preds | ML | logic | add | remove | avg body | stages |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- |
| A_caas_aapd_s0 | aapd | 78 | 98 | 98 | 0 | 76 | 2 | 1.256 | stage0_ml:76,stage2_fp_remove:2 |
| A_hybrid_llm_aapd_s0 | aapd | 81 | 107 | 105 | 2 | 77 | 4 | 1.321 | stage0_ml:76,stage2_fp_remove:4,stage1_fn_add:1 |
| A_indiv_ms_aapd_s0 | aapd | 81 | 107 | 105 | 2 | 77 | 4 | 1.321 | stage0_ml:76,stage2_fp_remove:4,stage1_fn_add:1 |
| A_random_ms_aapd_s0 | aapd | 84 | 102 | 98 | 4 | 79 | 5 | 1.214 | stage0_ml:79,stage2_fp_remove:5 |
| A_router_aapd_s0 | aapd | 85 | 114 | 109 | 5 | 81 | 4 | 1.341 | stage0_ml:80,stage2_fp_remove:4,stage3_prop:1 |
| C_filter_chi2_aapd_s0 | aapd | 84 | 110 | 104 | 6 | 79 | 5 | 1.31 | stage0_ml:79,stage2_fp_remove:5 |
| C_filter_mi_aapd_s0 | aapd | 83 | 109 | 105 | 4 | 79 | 4 | 1.313 | stage0_ml:79,stage2_fp_remove:4 |
| C_localboost_aapd_s0 | aapd | 85 | 112 | 106 | 6 | 79 | 6 | 1.318 | stage0_ml:79,stage2_fp_remove:6 |
| C_weshap_aapd_s0 | aapd | 82 | 107 | 103 | 4 | 79 | 3 | 1.305 | stage0_ml:79,stage2_fp_remove:3 |
| A_caas_arxiv_s0 | arxiv | 70 | 89 | 89 | 0 | 68 | 2 | 1.271 | stage0_ml:68,stage2_fp_remove:2 |
| A_hybrid_llm_arxiv_s0 | arxiv | 79 | 110 | 99 | 11 | 76 | 3 | 1.392 | stage0_ml:68,stage1_fn_add:8,stage2_fp_remove:3 |
| A_indiv_ms_arxiv_s0 | arxiv | 77 | 104 | 96 | 8 | 73 | 4 | 1.351 | stage0_ml:68,stage1_fn_add:5,stage2_fp_remove:4 |
| A_random_ms_arxiv_s0 | arxiv | 68 | 82 | 74 | 8 | 65 | 3 | 1.206 | stage0_ml:61,stage1_fn_add:4,stage2_fp_remove:3 |
| A_router_arxiv_s0 | arxiv | 74 | 98 | 93 | 5 | 71 | 3 | 1.324 | stage0_ml:68,stage1_fn_add:3,stage2_fp_remove:3 |
| C_filter_chi2_arxiv_s0 | arxiv | 71 | 91 | 88 | 3 | 69 | 2 | 1.282 | stage0_ml:68,stage2_fp_remove:2,stage1_fn_add:1 |
| C_filter_mi_arxiv_s0 | arxiv | 72 | 93 | 89 | 4 | 70 | 2 | 1.292 | stage0_ml:68,stage1_fn_add:2,stage2_fp_remove:2 |
| C_localboost_arxiv_s0 | arxiv | 70 | 88 | 87 | 1 | 69 | 1 | 1.257 | stage0_ml:68,stage1_fn_add:1,stage2_fp_remove:1 |
| C_weshap_arxiv_s0 | arxiv | 71 | 90 | 89 | 1 | 69 | 2 | 1.268 | stage0_ml:68,stage2_fp_remove:2,stage1_fn_add:1 |
| A_caas_bgc_s0 | bgc | 71 | 101 | 99 | 2 | 68 | 3 | 1.423 | stage0_ml:66,stage2_fp_remove:3,stage1_fn_add:2 |
| A_hybrid_llm_bgc_s0 | bgc | 65 | 92 | 88 | 4 | 63 | 2 | 1.415 | stage0_ml:60,stage1_fn_add:3,stage2_fp_remove:2 |
| A_indiv_ms_bgc_s0 | bgc | 66 | 94 | 90 | 4 | 64 | 2 | 1.424 | stage0_ml:60,stage1_fn_add:4,stage2_fp_remove:2 |
| A_random_ms_bgc_s0 | bgc | 72 | 107 | 99 | 8 | 70 | 2 | 1.486 | stage0_ml:66,stage1_fn_add:3,stage2_fp_remove:2,stage3_prop:1 |
| A_router_bgc_s0 | bgc | 63 | 86 | 82 | 4 | 63 | 0 | 1.365 | stage0_ml:60,stage1_fn_add:2,stage3_prop:1 |
| C_filter_chi2_bgc_s0 | bgc | 64 | 89 | 88 | 1 | 64 | 0 | 1.391 | stage0_ml:63,stage1_fn_add:1 |
| C_filter_mi_bgc_s0 | bgc | 73 | 106 | 98 | 8 | 70 | 3 | 1.452 | stage0_ml:63,stage1_fn_add:6,stage2_fp_remove:3,stage3_prop:1 |
| C_localboost_bgc_s0 | bgc | 73 | 106 | 101 | 5 | 68 | 5 | 1.452 | stage0_ml:63,stage1_fn_add:5,stage2_fp_remove:5 |
| C_weshap_bgc_s0 | bgc | 69 | 102 | 97 | 5 | 68 | 1 | 1.478 | stage0_ml:63,stage1_fn_add:5,stage2_fp_remove:1 |
| A_caas_rcv1_s0 | rcv1 | 52 | 59 | 57 | 2 | 50 | 2 | 1.135 | stage0_ml:48,stage2_fp_remove:2,stage1_fn_add:1,stage3_prop:1 |
| A_hybrid_llm_rcv1_s0 | rcv1 | 53 | 60 | 54 | 6 | 51 | 2 | 1.132 | stage0_ml:48,stage3_prop:3,stage2_fp_remove:2 |
| A_indiv_ms_rcv1_s0 | rcv1 | 53 | 60 | 54 | 6 | 51 | 2 | 1.132 | stage0_ml:48,stage3_prop:3,stage2_fp_remove:2 |
| A_random_ms_rcv1_s0 | rcv1 | 52 | 60 | 57 | 3 | 50 | 2 | 1.154 | stage0_ml:48,stage2_fp_remove:2,stage1_fn_add:1,stage3_prop:1 |
| A_router_rcv1_s0 | rcv1 | 54 | 67 | 60 | 7 | 51 | 3 | 1.241 | stage0_ml:48,stage2_fp_remove:3,stage3_prop:3 |
| C_filter_chi2_rcv1_s0 | rcv1 | 53 | 65 | 59 | 6 | 51 | 2 | 1.226 | stage0_ml:48,stage3_prop:3,stage2_fp_remove:2 |
| C_filter_mi_rcv1_s0 | rcv1 | 54 | 66 | 59 | 7 | 51 | 3 | 1.222 | stage0_ml:48,stage2_fp_remove:3,stage3_prop:3 |
| C_localboost_rcv1_s0 | rcv1 | 53 | 65 | 59 | 6 | 51 | 2 | 1.226 | stage0_ml:48,stage3_prop:3,stage2_fp_remove:2 |
| C_weshap_rcv1_s0 | rcv1 | 54 | 66 | 60 | 6 | 51 | 3 | 1.222 | stage0_ml:48,stage2_fp_remove:3,stage3_prop:3 |
| A_caas_reuters21578_s0 | reuters21578 | 33 | 33 | 33 | 0 | 33 | 0 | 1.0 | stage0_ml:33 |
| A_hybrid_llm_reuters21578_s0 | reuters21578 | 33 | 33 | 33 | 0 | 33 | 0 | 1.0 | stage0_ml:33 |
| A_indiv_ms_reuters21578_s0 | reuters21578 | 33 | 33 | 33 | 0 | 33 | 0 | 1.0 | stage0_ml:33 |
| A_random_ms_reuters21578_s0 | reuters21578 | 34 | 36 | 35 | 1 | 34 | 0 | 1.059 | stage0_ml:33,stage1_fn_add:1 |
| A_router_reuters21578_s0 | reuters21578 | 33 | 34 | 34 | 0 | 33 | 0 | 1.03 | stage0_ml:33 |
| C_filter_chi2_reuters21578_s0 | reuters21578 | 33 | 34 | 34 | 0 | 33 | 0 | 1.03 | stage0_ml:33 |
| C_filter_mi_reuters21578_s0 | reuters21578 | 34 | 39 | 38 | 1 | 34 | 0 | 1.147 | stage0_ml:33,stage1_fn_add:1 |
| C_localboost_reuters21578_s0 | reuters21578 | 33 | 34 | 34 | 0 | 33 | 0 | 1.03 | stage0_ml:33 |
| C_weshap_reuters21578_s0 | reuters21578 | 33 | 34 | 34 | 0 | 33 | 0 | 1.03 | stage0_ml:33 |

Per-experiment predicate-type counts and representative rules are in `rule_stats.json` under `per_run[].predicate_counts_by_type` and `per_run[].representative_rules`.

## Aggregate (all experiments)

## ML vs logic predicates

| class | count | share |
| :-- | --: | --: |
| ML (MLPredicate, MLThreshold) | 3406 | 95.4% |
| logic (text/label/graph) | 166 | 4.6% |

## Predicate count by type

| type | role | count |
| :-- | :-- | --: |
| MLThresholdPredicate | ml | 3406 |
| MatchPredicate | logic_text | 111 |
| LabelPredicate | label | 27 |
| FreqPredicate | logic_text | 19 |
| CooccurPredicate | logic_text | 6 |
| BeforePredicate | logic_text | 2 |
| SimPredicate | propagation_graph | 1 |

## Roles

| role | count |
| :-- | --: |
| ml | 3406 |
| logic_text | 138 |
| label | 27 |
| propagation_graph | 1 |

## Consequence ops · discovery stage

| op | count |  | stage | count |
| :-- | --: | -- | :-- | --: |
| add | 2690 |  | stage0_ml | 2601 |
| remove | 98 |  | stage2_fp_remove | 98 |
|  |  |  | stage1_fn_add | 62 |
|  |  |  | stage3_prop | 27 |

## Rule body length (arity) distribution

| body_len | #rules |
| --: | --: |
| 1 | 2045 |
| 2 | 718 |
| 3 | 12 |
| 4 | 11 |
| 5 | 1 |
| 6 | 1 |

## Rules per dataset

| dataset | #rules |
| :-- | --: |
| aapd | 743 |
| arxiv | 652 |
| bgc | 616 |
| rcv1 | 478 |
| reuters21578 | 299 |

## Representative rules (top by F1-gain, per dataset)

### aapd

- `ml_thresh('loris_clf_textcnn', label='math.it', thresh=0.55) → -cs.na`  — score=15.8745, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_tfidf_lr_bigram', label='cs.it', thresh=0.65) → -cs.na`  — score=13.5277, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_textcnn', label='math.it', thresh=0.55) ∧ ml_thresh('loris_clf_tfidf_svm_bigram', label='cs.lg', thresh=0.25) → -cs.na`  — score=10.1876, len=2, stage=stage2_fp_remove
- `ml_thresh('loris_clf_textcnn', label='math.it', thresh=0.65) ∧ ml_thresh('loris_clf_tfidf_svm_bigram', label='cs.lg', thresh=0.25) → -cs.na`  — score=9.7095, len=2, stage=stage2_fp_remove
- `cooccur(cnt, '\b(?:timing|timed|times|time)\b', '\b(?:resulting|resulted|results|result)\b') ∧ ml_thresh('loris_clf_tfidf_svm_bigram', label='cs.ne', thresh=0.25) → -cs.na`  — score=8.1607, len=2, stage=stage2_fp_remove

### arxiv

- `match(cnt, '\b(?:models|model)\b') ∧ freq(cnt, '\b(?:imaging|images|image)\b' >= 2.0) ∧ ml_thresh('loris_clf_encoder_mlp', label='cs.CV', thresh=0.25) → -eess.SP`  — score=8.8626, len=3, stage=stage2_fp_remove
- `ml_thresh('loris_clf_textcnn', label='cs.AI', thresh=0.35) → -eess.IV`  — score=5.6569, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_textcnn', label='cs.AI', thresh=0.35) → -cs.CE`  — score=4.5826, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_tfidf_svm_unigram', label='cs.GT', thresh=0.25) → -cs.CE`  — score=4.1231, len=1, stage=stage2_fp_remove
- `match(cnt, '\b(?:imaging|images|image)\b') ∧ match(cnt, '\b(?:models|model)\b') ∧ ml_thresh('loris_clf_tfidf_lr_bigram', label='cs.CV', thresh=0.65) ∧ ml_thresh('loris_clf_tfidf_svm_bigram', label='eess.IV', thresh=0.25) → -cs.NI`  — score=3.8576, len=4, stage=stage2_fp_remove

### bgc

- `ml_thresh('loris_clf_encoder_mlp', label='Children’s Books', thresh=0.65) → -Popular Science`  — score=3.1623, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_encoder_mlp', label='Children’s Books', thresh=0.65) ∧ ml_thresh('loris_clf_tfidf_svm_bigram', label='Nonfiction', thresh=0.25) → -Popular Science`  — score=2.9409, len=2, stage=stage2_fp_remove
- `ml_thresh('loris_clf_tfidf_svm_bigram', label='Historical Fiction', thresh=0.35) → -Contemporary Romance`  — score=2.8284, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_tfidf_svm_unigram', label='Fantasy', thresh=0.45) → -Graphic Novels & Manga`  — score=2.6457, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_tfidf_svm_unigram', label='Nonfiction', thresh=0.55) ∧ ml_thresh('loris_clf_tfidf_svm_unigram', label='Fiction', thresh=0.25) → -Historical Fiction`  — score=2.278, len=2, stage=stage2_fp_remove

### rcv1

- `ml_thresh('loris_clf_tfidf_lr_bigram', label='C15', thresh=0.35) → -C11`  — score=3.0, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_tfidf_svm_unigram', label='C11', thresh=0.35) → -C21`  — score=3.0, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_textcnn', label='C15', thresh=0.45) → -C11`  — score=2.8284, len=1, stage=stage2_fp_remove
- `ml_thresh('loris_clf_tfidf_lr_bigram', label='C15', thresh=0.35) ∧ ml_thresh('loris_clf_tfidf_svm_bigram', label='CCAT', thresh=0.25) → -C11`  — score=2.6304, len=2, stage=stage2_fp_remove
- `ml_thresh('loris_clf_textcnn', label='C15', thresh=0.45) ∧ ml_thresh('loris_clf_tfidf_svm_bigram', label='CCAT', thresh=0.25) → -C11`  — score=2.6304, len=2, stage=stage2_fp_remove

### reuters21578

- `ml_thresh('loris_clf_tfidf_svm_unigram', label='copper', thresh=0.4) → +copper`  — score=1.0, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_tfidf_svm_bigram', label='copper', thresh=0.35) → +copper`  — score=1.0, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_encoder_mlp', label='earn', thresh=0.4) → +earn`  — score=0.991, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_encoder_mlp', label='earn', thresh=0.45) → +earn`  — score=0.991, len=1, stage=stage0_ml
- `ml_thresh('loris_clf_encoder_mlp', label='earn', thresh=0.35) → +earn`  — score=0.9902, len=1, stage=stage0_ml

