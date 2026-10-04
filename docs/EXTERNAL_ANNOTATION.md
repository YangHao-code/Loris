# External annotation protocol

The labeling loop can consume an externally prepared answer directory. The
directory adapter does not send network requests and never receives an API key.
It authenticates each answer against the current row ID, complete label axis,
declared provider and provenance before the labeling loop applies it.

Configure a manifest with an explicit directory and provider mode:

```json
{
  "oracle": {
    "kind": "directory_llm_then_human",
    "directory": "answers"
  }
}
```

For a queried row, the answer directory may contain
`%08d.llm.json` and `%08d.human.json`. Each file has exactly these fields:

```json
{
  "row_id": "document-7",
  "labels": ["A", "B", "C"],
  "targets": [1, 0, 1],
  "provider": "human",
  "provenance": "Declared annotation source"
}
```

`targets` follows the complete, ordered label axis. `1` is an explicit positive
fact and `0` an explicit negative fact. In `directory_llm_then_human`, `null`
may represent an unresolved label when the manifest allows partial answers; a
complete-answer manifest requires only `0` and `1`. An all-zero vector is a
valid complete answer. A missing or invalid LLM proposal falls back to the
human file after the current Chase consistency check. A failed consistency
check or unavailable human file leaves the row pending; it never invents an
answer. Provider names distinguish real and simulated sources:
`llm`/`human` for declared external sources and `simulated_llm`/
`simulated_human` for experiments.

Use `directory_human` for the NoLLM ablation. It reads only the human file and
does not inspect or validate an LLM proposal.

## Local LLM proposal client

The main template queries after each main closure. Each batch contains 1% of
the initially eligible query pool, rounded up, with a minimum of 32 documents
and capped by remaining eligible documents and the explicit request budget.
LLM and human fallback for the same document count as one request.

Each batch triggers one fine-tuning call from the previous committed
checkpoint. The template uses one epoch, backbone learning rate 3e-6 and head
learning rate 1.5e-5. Training includes complete new answers, new or changed
signed rule targets, and a deterministic 5% sample of older trusted documents.
The first update treats all eligible rule feedback as new; later updates skip
unchanged signed targets. Weight changes alone do not repeat an old target.
Changes of eligibility or target sign do count as new feedback. Current weights
are used for selected targets. Old trusted corrections follow the same replay
sample instead of silently including every old answer.

Set `feedback.rule_feedback_scope` to `new_or_changed`, `trusted_replay` to
`true`, `trusted_replay_fraction` to `0.05`, and `new_verified_supervision`
to `full_batch`. This requires `loop.query_cadence=each_closure` and
`max_inner_rounds=1`. Existing manifests without the scope field retain the
historical all-feedback behavior. Consistency checks and calibration also run
closures; those internal calls do not each request labels or train a model.

`python -m loris.labeling.llm_provider` is a single-request OpenAI-compatible
client. The default wire format is a complete positive-label set and thinking
is disabled:

```bash
python -m loris.labeling.llm_provider \
  --request selected_document.json --output-dir answers \
  --thinking off --answer-format positive_labels
```

The request JSON contains only `row`, `row_id`, `labels`, `text`, and optional
`label_descriptions`; it must not contain truth labels. The model must return
exactly `{"positive_labels": ["..."]}`. Names are checked against the complete
axis, deduplicated, and converted to the canonical ordered binary vector. An
empty positive-label list is valid. The legacy `targets` wire format remains
available with `--answer-format targets` for controlled compatibility.

The API key is read from `SILICONFLOW_API_KEY` by default, or from the variable
selected with `--api-key-env`. Keep that variable in the local process
environment. It is never written to requests, proposals, receipts, logs or
errors. The client requires HTTPS, follows no redirects, makes at most one
request, and rejects truncated or malformed responses. A credential-free API
receipt records status, model identity, token counters and request digest; the
directory adapter records `network_calls: 0` because it performs no network
operation itself.
