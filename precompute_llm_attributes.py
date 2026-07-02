#!/usr/bin/env python3
"""Offline LLM closed-ontology attribute precompute for LORIS Lever C.

Maps each document to a small set of values from a FIXED (closed) ontology using
a local LLM, and writes a cache ``{doc_hash: [values]}`` that the LORIS pipeline
reads via ``compute_llm_attributes`` (``--enable_llm_attrs --llm_attr_cache``).

The LLM runs ONCE here (decoupled from the pipeline) so the heavy model never has
to coexist in-process with LORIS's encoder/embeddings on the GPU. Output is keyed
by ``llm_doc_hash(text)`` (== the pipeline's hash of ``doc.cnt``), so values are
reused verbatim at BO / select / test.

Closed ontology (anti-RoGRAD): the model must pick from a fixed value list; any
out-of-ontology output is dropped. Decoding is greedy (deterministic).

Usage (real LLM, mirrors LBoost/LLMAug Mistral-7B):
    HF_ENDPOINT=https://hf-mirror.com python precompute_llm_attributes.py \
        --dataset aapd --provider transformers \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --ontology_from labels --k 3 --out experiments/llm_attr_aapd.json

Plumbing/smoke (no LLM — deterministic keyword assignment):
    python precompute_llm_attributes.py --dataset aapd --provider dummy \
        --ontology_from labels --out experiments/llm_attr_aapd_dummy.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

# Use the LORIS hash if importable; else define it inline so this script runs in a
# bare LLM env (vLLM/transformers) with no LORIS/numpy/scipy installed. MUST stay
# byte-identical to loris.rules.virtual_attributes.llm_doc_hash so cache keys align.
try:
    from loris.rules.virtual_attributes import llm_doc_hash  # noqa: E402
except Exception:                                            # pragma: no cover
    import hashlib

    def llm_doc_hash(text: str) -> str:
        return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


# ── data ───────────────────────────────────────────────────────────────────────

def _load_texts_and_labels(dataset: str, split: str = "all", texts_file: str = ""):
    """Read the processed CSVs; return (unique_texts, label_names). ``split='test'``
    reads only test.csv (the probe only needs the test split → far fewer LLM calls).
    ``texts_file`` (a JSON list of texts) overrides the text source so we can cover
    EXACTLY the probe's sampled docs — labels are still read from the dataset CSV."""
    import pandas as pd  # noqa: PLC0415
    base = _REPO / "data" / dataset / "processed"
    if texts_file:
        texts = [str(t) for t in json.load(open(texts_file))]
        seen, uniq = set(), []
        for t in texts:
            if t not in seen:
                seen.add(t); uniq.append(t)
        cols = pd.read_csv(base / "test.csv", nrows=1).columns
        return uniq, [c for c in cols if c not in ("text", "title")]
    files = ("test.csv",) if split == "test" else ("train.csv", "test.csv")
    frames = [pd.read_csv(base / f) for f in files if (base / f).exists()]
    if not frames:
        raise FileNotFoundError(f"no processed CSVs under {base}")
    df = pd.concat(frames, ignore_index=True)
    texts = [str(t) for t in df["text"].dropna().tolist()]
    # de-dup while preserving order (cache is keyed by hash anyway)
    seen, uniq = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    label_names = [c for c in df.columns if c != "text" and c != "title"]
    return uniq, label_names


def _norm(s: str) -> str:
    """Normalise for closed-ontology matching: lowercase, unify curly quotes/dashes,
    strip leading list markers ('1.', '- ', '* '), collapse whitespace. Applied to
    BOTH the ontology values and the LLM output so e.g. the label's curly apostrophe
    matches the model's straight one, and '1. manga' matches 'manga'."""
    s = (s or "").lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")):
        s = s.replace(a, b)
    s = re.sub(r"^\s*[\-\*\d]+[.)]?\s*", "", s)   # strip leading list marker
    s = re.sub(r"\s+", " ", s).strip().strip(".").strip()
    return s


def _build_ontology(args, label_names: List[str]) -> List[str]:
    if args.ontology:
        if os.path.exists(args.ontology):
            vals = [l.strip() for l in open(args.ontology) if l.strip()]
        else:
            vals = [v.strip() for v in args.ontology.split(",") if v.strip()]
    elif args.ontology_from == "labels":
        vals = list(label_names)
    else:
        raise SystemExit("provide --ontology <csv|file> or --ontology_from labels")
    # closed + deterministic, normalised so LLM output can match exactly
    out, seen = [], set()
    for v in vals:
        v = _norm(v)
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ── providers ──────────────────────────────────────────────────────────────────

def _parse_to_ontology(raw: str, ontology_set) -> List[str]:
    """Keep only emitted values that are in the closed ontology (drop hallucinations).
    Both sides are _norm-normalised so curly-quote / list-marker mismatches don't
    silently drop valid picks."""
    parts = re.split(r"[,\n;]+", raw or "")
    out, seen = [], set()
    for p in parts:
        p = _norm(p)
        if p in ontology_set and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _prompt(text: str, ontology: List[str], k: int, max_chars: int) -> str:
    topics = ", ".join(ontology)
    body = (text or "")[:max_chars]
    return (
        "[INST] Choose between 1 and %d topics from this FIXED list that best "
        "describe the document. Use ONLY the exact words from the list, comma-"
        "separated. Do not invent topics.\n\nList: %s\n\nDocument: %s\n\n"
        "Topics: [/INST]" % (k, topics, body)
    )


def _run_transformers(texts, ontology, args):
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # decoder-only batched generation needs left pad
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager",
    ).to("cuda").eval()
    ont_set = set(ontology)
    out = {}
    bs = args.batch_size
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        prompts = [_prompt(t, ontology, args.k, args.max_chars) for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_tokens).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for t, full, inp in zip(chunk, gen, enc["input_ids"]):
            new = full[inp.shape[0]:]
            raw = tok.decode(new, skip_special_tokens=True)
            out[llm_doc_hash(t)] = _parse_to_ontology(raw, ont_set)[:args.k]
        del enc, gen
        torch.cuda.empty_cache()                       # free KV cache between batches
        if (i // bs) % 10 == 0:
            print(f"  {i+len(chunk)}/{len(texts)} docs", flush=True)
    return out


def _run_vllm(texts, ontology, args):
    """Batched Mistral-7B via vLLM (mirrors LBoost/LLMAug/infer.py). Greedy/
    deterministic; closed-ontology drop via _parse_to_ontology."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=args.model, dtype="half", enforce_eager=True,
              max_model_len=args.max_tokens,
              gpu_memory_utilization=args.gpu_mem_util)
    sp = SamplingParams(temperature=0, top_p=1, max_tokens=48)
    prompts = [_prompt(t, ontology, args.k, args.max_chars) for t in texts]
    print(f"[vllm] generating {len(prompts)} prompts ...", flush=True)
    outs = llm.generate(prompts, sp)
    ont_set = set(ontology)
    out = {}
    for t, o in zip(texts, outs):
        raw = o.outputs[0].text if o.outputs else ""
        out[llm_doc_hash(t)] = _parse_to_ontology(raw, ont_set)[:args.k]
    return out


def _run_dummy(texts, ontology, args):
    """Deterministic keyword-overlap assignment (NO LLM) — for plumbing/smoke only.
    Assigns the ontology values whose token appears in the text; falls back to the
    single best lexical-overlap value. Not a substitute for the real probe."""
    ont_tokens = {v: set(re.findall(r"[a-z]{3,}", v)) for v in ontology}
    out = {}
    for t in texts:
        words = set(re.findall(r"[a-z]{3,}", (t or "").lower()))
        scored = sorted(
            ((v, len(toks & words)) for v, toks in ont_tokens.items()),
            key=lambda kv: (-kv[1], kv[0]))
        chosen = [v for v, s in scored if s > 0][:args.k]
        out[llm_doc_hash(t)] = chosen
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--provider", choices=["vllm", "transformers", "dummy"], default="dummy")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--split", choices=["all", "test"], default="all",
                    help="'test' = only the test split (probe needs only test docs)")
    ap.add_argument("--texts_file", default="",
                    help="JSON list of texts to process (e.g. the probe's exact TE sample); "
                         "overrides --split. Labels still read from the dataset CSV.")
    ap.add_argument("--ontology", default="", help="comma list OR path to a file (one value/line)")
    ap.add_argument("--ontology_from", choices=["labels", ""], default="")
    ap.add_argument("--k", type=int, default=3, help="max ontology values per doc")
    ap.add_argument("--max_docs", type=int, default=0, help="0 = all")
    ap.add_argument("--max_chars", type=int, default=1500)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    texts, label_names = _load_texts_and_labels(args.dataset, args.split, args.texts_file)
    if args.max_docs > 0:
        texts = texts[:args.max_docs]
    ontology = _build_ontology(args, label_names)
    print(f"[precompute] {len(texts)} docs, {len(ontology)} closed-ontology values, "
          f"provider={args.provider}", flush=True)

    runner = {"vllm": _run_vllm, "transformers": _run_transformers,
              "dummy": _run_dummy}[args.provider]
    cache = runner(texts, ontology, args)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cache, open(args.out, "w"))
    _nonempty = sum(1 for v in cache.values() if v)
    print(f"[precompute] wrote {len(cache)} entries ({_nonempty} non-empty) → {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
