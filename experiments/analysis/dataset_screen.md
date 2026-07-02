# LORIS dataset suitability screen

| dataset | verdict | raw text | docs (tr/te) | labels | cardinality | avg words | wordid frac |
| :-- | :-- | :-- | :-- | --: | --: | --: | --: |
| aapd | **PASS** | yes | 53840/1000 | 54 | 2.407 | 163.5 | 0.0 |
| arxiv | **PASS** | yes | 9467/2367 | 40 | 1.956 | 182.4 | 0.0 |
| bgc | **PASS** | yes | 58715/33179 | 146 | 3.007 | 157.4 | 0.0 |
| goodreads | **PASS** | yes | 8903/990 | 18 | 2.543 | 163.1 | 0.0 |
| hupd | **PASS** | yes | 17279/3050 | 60 | 1.332 | 117.1 | 0.0 |
| pubmed | **PASS** | yes | 42193/7446 | 14 | 5.77 | 205.2 | 0.0 |
| rcv1 | **FAIL** | NO | 20000/10000 | 103 | 3.183 | 42.0 | 1.0 |
| reuters21578 | **PASS** | yes | 8301/2076 | 119 | 1.266 | 135.9 | 0.0 |

## Recommended set (text-predicate suitable)

aapd, arxiv, bgc, goodreads, hupd, pubmed, reuters21578

## Excluded

- rcv1 (text looks tokenized (wordid_frac=1.0, alpha_frac=0.0))

