# Transformer training backend

The six source modules here provide the actual masked weighted BCE training,
model construction, tokenization, checkpoint loading, and prediction used by
LORIS. Pass `--backend-source backends/transformer/src` when building a model
pool. For integration tests set `LORIS_SSL_SOURCE` to that absolute directory.

This is a curated runtime dependency derived from the existing `loris_ssl`
source. Unused experiment trainers were removed and machine-specific defaults
made portable. `SOURCE_MANIFEST.json` records each retained upstream and release
hash. It is a new execution identity, not the unchanged producer of historical
results. Supply pretrained weights separately. No new upstream license is
assigned by this curation.

The head-tail collator also handles zero tail/body budgets explicitly and rejects
lengths smaller than the tokenizer's required special tokens. The ordinary 3:1
head-tail policy remains unchanged; the manifest retains the original upstream
identity and records the corrected release file separately.
