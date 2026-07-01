# Full / million-scale dataset sources (deferred — needs large writable storage)

Current state: the pipeline uses **small slices** (validated end-to-end). Full
versions are tens of GB each and were NOT downloaded (instance had 8.8 GB free,
no large writable disk, `/autodl-pub` read-only). When a large writable disk is
available, download the full sources below and extend each `prepare_<ds>` in
[loris/data/prepare.py](../../loris/data/prepare.py) with a `full=True` path.

| dataset | current slice | full source | full scale | prepare change |
| :-- | :-- | :-- | :-- | :-- |
| **arXiv** | 9k sample CSV (`data/arxiv/processed`) | Kaggle `Cornell-University/arxiv` (`arxiv-metadata-oai-snapshot.json`) | ~2.7M papers, ~4 GB JSON | new `prepare_arxiv(full=True)`: stream JSON, text=title+abstract, labels=categories (subclass) |
| **HUPD** | `data/sample-jan-2016.tar.gz` (26k) | HF `HUPD/hupd` `data/2004..2018.tar.gz` or `all-years.tar` | ~4.5M patents, tens of GB | `prepare_hupd`: accept `years="all"`, loop tars with the existing streaming loop |
| **PubMed MeSH** | HF `owaiskha9654/...MeSH` CSV (50k) | BioASQ Task A (`allMeSH_2022.json`, registration) or PubMed baseline | ~15M abstracts | new `prepare_pubmed(full=True)`: parse BioASQ JSON, labels=MeSH roots |
| **Goodreads** | HF `pszemraj/goodreads-bookgenres` (8.9k) | UCSD Book Graph `goodreads_books.json.gz` + genres (mcauleylab.ucsd.edu, reachable) | ~2.3M books | new `prepare_goodreads(full=True)`: join books+genres, aggregate shelves→genres |
| **MICoL / MAG-CS** | none (deferred) | OpenAlex API (MAG retired) + inverted-index text recovery | large | new `prepare_micol`: map field-of-study→labels, recover abstracts from inverted index |

Notes:
- Every `prepare_*` already emits the standard `text,<label…>` CSV via
  `_save_multi_hot_csv`, and `load_data` caps to top-N labels — so only the
  fetch/parse front-end changes for full scale.
- For million-scale, also raise the per-dataset caps in
  `loris/baselines/common.py::DATASET_DEFAULTS` (currently `subset_size` ≈ 12–20k
  for tractability).
