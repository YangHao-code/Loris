"""OpenAI ``gpt-4.1`` client for the LLM-dependent baselines (GPT4, RulePrompt).

Paper setting: ``gpt-4.1-2025-04-14``, ``temperature=0``.  Used by
``gpt4_zeroshot`` and ``ruleprompt``.  Also the concrete backend for
``loris.rill.oracle.LLMOracle`` (currently a stub).

Design:
* lazy ``import openai`` (not a hard dependency of the repo);
* retry with exponential backoff on transient errors;
* strict-ish JSON parsing of the model's label list;
* a **mock mode** (``LORIS_LLM_MOCK=1`` or no ``OPENAI_API_KEY``) that returns
  empty predictions so the pipeline can be smoke-tested offline without cost.
  Mock results are clearly flagged and must NOT be reported as real numbers.

Env:
    OPENAI_API_KEY   required for real calls
    OPENAI_BASE_URL  optional (proxy / Azure-compatible gateway)
    LORIS_LLM_MODEL  override model id (default gpt-4.1-2025-04-14)
    LORIS_LLM_MOCK   '1' ⇒ force offline mock
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional

DEFAULT_MODEL = "gpt-4.1-2025-04-14"


class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.environ.get("LORIS_LLM_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.n_calls = 0
        self.n_prompt_tokens = 0
        self.n_completion_tokens = 0
        self.mock = (
            os.environ.get("LORIS_LLM_MOCK") == "1"
            or not os.environ.get("OPENAI_API_KEY")
        )
        self._client = None
        if not self.mock:
            self._client = self._make_client()

    def _make_client(self):
        try:
            from openai import OpenAI  # openai>=1.0
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package not installed. `pip install openai`, or set "
                "LORIS_LLM_MOCK=1 to run the pipeline offline (mock labels)."
            ) from exc
        kwargs = {}
        base = os.environ.get("OPENAI_BASE_URL")
        if base:
            kwargs["base_url"] = base
        return OpenAI(timeout=self.timeout, **kwargs)

    # ── raw chat ────────────────────────────────────────────────────────────
    def chat(self, system: str, user: str) -> str:
        """Return the model's text reply (empty string in mock mode)."""
        if self.mock:
            self.n_calls += 1
            return ""
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                self.n_calls += 1
                u = getattr(resp, "usage", None)
                if u is not None:
                    self.n_prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
                    self.n_completion_tokens += getattr(u, "completion_tokens", 0) or 0
                return resp.choices[0].message.content or ""
            except Exception as exc:  # transient: backoff
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_exc}")

    # ── multi-label classification helper ────────────────────────────────────
    def classify_multilabel(
        self, text: str, label_names: List[str], max_chars: int = 6000,
    ) -> List[str]:
        """Zero-shot multi-label: return the subset of *label_names* that apply.

        Robust to the model returning prose around a JSON array.
        """
        labels_str = ", ".join(label_names)
        system = (
            "You are a precise multi-label text classifier. You will be given a "
            "document and a fixed set of candidate labels. Return ONLY a JSON "
            "array (possibly empty) of the labels that apply, drawn verbatim "
            "from the candidate set. No explanation."
        )
        user = (
            f"Candidate labels: [{labels_str}]\n\n"
            f"Document:\n{text[:max_chars]}\n\n"
            f"Return a JSON array of the applicable labels."
        )
        reply = self.chat(system, user)
        return self._parse_label_list(reply, label_names)

    @staticmethod
    def _parse_label_list(reply: str, label_names: List[str]) -> List[str]:
        if not reply:
            return []
        valid = {l.lower(): l for l in label_names}
        # try a JSON array first
        m = re.search(r"\[.*\]", reply, re.DOTALL)
        cand: List[str] = []
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    cand = [str(x) for x in arr]
            except Exception:
                cand = []
        if not cand:
            # fallback: substring match
            low = reply.lower()
            cand = [l for l in label_names if l.lower() in low]
        out, seen = [], set()
        for c in cand:
            key = str(c).strip().lower()
            if key in valid and valid[key] not in seen:
                out.append(valid[key])
                seen.add(valid[key])
        return out

    def cost_summary(self) -> dict:
        return {
            "model": self.model,
            "mock": self.mock,
            "n_calls": self.n_calls,
            "prompt_tokens": self.n_prompt_tokens,
            "completion_tokens": self.n_completion_tokens,
        }
