"""LORIS paper baselines (§7 experimental study).

Self-contained adapters that reproduce the comparison methods the paper
evaluates LORIS against, on the SAME data splits and metric as the main
pipeline.  Nothing here touches the LORIS golden path — baselines are only
reached through ``loris.baselines.run_baselines`` (or by importing a module
directly).

Three families (paper §7):

* **Group A — model selection** (``selectors``): Random_MS, Indiv_MS,
  Hybrid_LLM, CAAS.  Swap LORIS's dynamic router; report downstream Macro-F1.
* **Group B — end-to-end labeling**: Snuba (``snuba``), Self-Pretraining
  (``self_pretrain``), RulePrompt (``ruleprompt``), DeBERTa_SVM /
  DeBERTa_XGBoost (``encoder_head``), GPT4 (``gpt4_zeroshot``), BESRA / RAL
  (``hitl``).
* **Group C — pattern-selection ablation** (``pattern_select``): Filter_MI,
  Filter_chi2, WeShap, LocalBoost.

Shared helpers live in ``common`` (data loading + metric + result IO) and
``llm_client`` (OpenAI gpt-4.1 wrapper).
"""

from __future__ import annotations

__all__ = ["common", "llm_client"]
