"""
models/lora_slm_classifier.py
------------------------------
PEFT-tuned Small Language Model (SLM) for multi-label text classification.

Supported backbones: any causal LM with a native sequence-classification
variant in HuggingFace (e.g. Llama-3-8B → ``LlamaForSequenceClassification``,
Mistral-7B → ``MistralForSequenceClassification``).

Two PEFT methods (``peft_method`` parameter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``"lora"`` (default): Low-Rank Adaptation — injects trainable rank-r
  matrices into selected attention projection layers. Most widely used,
  well-tested with HuggingFace PEFT.
* ``"ia3"``: Infused Adapter by Inhibiting and Amplifying Inner Activations
  (Liu et al., 2022). Element-wise rescaling vectors injected into keys,
  values, and feed-forward down-projections. Fewer trainable parameters
  than LoRA; PEFT-native implementation. Corresponds to the "adapter"
  family referenced in Houlsby et al. (2019).

Architecture & design choices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* We use ``AutoModelForSequenceClassification`` with the causal LM backbone.
  The model places a linear ``score`` head on top of the **last token's**
  hidden state — analogous to the ``[CLS]`` token in encoder models.

* 4-bit NF4 quantisation (BitsAndBytes) freezes the base model weights.
  Only PEFT adapter parameters and the classification head are trainable.

* Gradient accumulation compensates for the very small training batch size
  that 4-bit quantisation imposes.

* ``bfloat16`` autocast is preferred over ``float16`` for stability.

Critical initialisation order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Load tokeniser, set ``padding_side="left"`` and ``pad_token = eos_token``.
2. Build ``BitsAndBytesConfig``.
3. Load model with ``quantization_config`` and ``device_map="auto"``.
4. Call ``prepare_model_for_kbit_training(model)``  ← BEFORE get_peft_model.
5. Build ``LoraConfig`` / ``IA3Config`` and call ``get_peft_model``.

Saving / loading
~~~~~~~~~~~~~~~~
``save_lora(output_dir)`` saves the PEFT adapter weights.
The classification head is saved separately because PEFT does not include it.
``load_lora(adapter_dir)`` reconstructs the full model.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.base import BaseDocumentClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

class _SLMDataset(Dataset):
    """Wraps tokenised inputs and optional float labels."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

class LoRASLMClassifier(BaseDocumentClassifier):
    """
    LoRA-tuned causal LM for multi-label text classification.

    Parameters
    ----------
    num_labels : int
        Number of output labels.
    model_name : str
        HuggingFace model identifier (default ``"meta-llama/Meta-Llama-3-8B"``).
        Also supports ``"mistralai/Mistral-7B-v0.1"`` and similar models.
    peft_method : str
        PEFT adapter type: ``"lora"`` (default) or ``"ia3"``.
    max_length : int
        Tokeniser max sequence length (default 512).
    batch_size : int
        Per-step training batch size (default 2; keep tiny for 8B models).
    accumulation_steps : int
        Gradient accumulation steps (default 8; effective batch = 16).
    num_epochs : int
        Training epochs (default 3).
    lr : float
        Learning rate for AdamW (default 2e-4).
    threshold : float
        Sigmoid threshold for binarising predictions (default 0.5).
    lora_r : int
        LoRA rank (default 16). Used only when ``peft_method="lora"``.
    lora_alpha : int
        LoRA scaling factor (default 32). Used only when ``peft_method="lora"``.
    lora_dropout : float
        Dropout within LoRA layers (default 0.05). Used only for LoRA.
    lora_target_modules : Sequence[str]
        Attention projection module names for LoRA (default ``["q_proj", "v_proj"]``).
    use_4bit : bool
        Enable 4-bit NF4 quantisation via BitsAndBytes (default ``True``).
        Set to ``False`` for CPU debugging or when bitsandbytes is unavailable.
    **kwargs
        Forwarded to ``BaseDocumentClassifier.__init__``.

    Attributes
    ----------
    tokenizer : AutoTokenizer
    model : PeftModel (after fit)
    device : torch.device
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        peft_method: str = "lora",
        max_length: int = 512,
        batch_size: int = 2,
        accumulation_steps: int = 8,
        num_epochs: int = 3,
        lr: float = 2e-4,
        threshold: float = 0.5,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = ("q_proj", "v_proj"),
        use_4bit: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels, **kwargs)
        peft_method = peft_method.lower()
        if peft_method not in ("lora", "ia3"):
            raise ValueError(f"peft_method must be 'lora' or 'ia3', got '{peft_method}'")
        self.model_name = model_name
        self.peft_method = peft_method
        self.max_length = max_length
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.threshold = threshold
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = list(lora_target_modules)
        self.use_4bit = use_4bit

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model = None  # PeftModel after fit

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _load_base_model(self):
        """
        Load tokeniser, base model (optionally 4-bit quantised), and
        wrap with PEFT adapters (LoRA or IA³).

        Returns
        -------
        PeftModel
            Adapter-wrapped model with trainable adapters + classification head.
        """
        from peft import get_peft_model, prepare_model_for_kbit_training

        # Step 1 — Tokeniser (left-padding so last token = richest repr)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 2 — Quantisation config
        model_kwargs: Dict[str, Any] = {
            "num_labels": self.num_labels,
            "problem_type": "multi_label_classification",
            "trust_remote_code": True,
        }
        if self.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"
                logger.info("4-bit NF4 quantisation enabled.")
            except ImportError:
                logger.warning(
                    "bitsandbytes not available; falling back to full precision."
                )
                self.use_4bit = False

        if not self.use_4bit:
            model_kwargs["torch_dtype"] = torch.float32

        # Step 3 — Load base model
        logger.info("Loading %s (peft_method=%s) …", self.model_name, self.peft_method)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )

        # Align pad token embedding (Llama/Mistral have no pad token by default)
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Step 4 — Prepare for k-bit training (MUST precede get_peft_model)
        if self.use_4bit:
            base_model = prepare_model_for_kbit_training(
                base_model, use_gradient_checkpointing=True
            )

        # Step 5 — Build PEFT config
        peft_config = self._build_peft_config()

        # Step 6 — Inject adapters
        model = get_peft_model(base_model, peft_config)

        # Ensure classification head is trainable (PEFT may freeze it)
        for name, param in model.named_parameters():
            if "score" in name:
                param.requires_grad_(True)

        model.print_trainable_parameters()
        return model

    def _build_peft_config(self):
        """Return the appropriate PEFT config based on ``self.peft_method``."""
        if self.peft_method == "ia3":
            from peft import IA3Config, TaskType
            # IA³ rescales keys, values, and FFN down-projections
            return IA3Config(
                target_modules=["k_proj", "v_proj", "down_proj"],
                feedforward_modules=["down_proj"],
                task_type=TaskType.SEQ_CLS,
            )
        # lora (default)
        from peft import LoraConfig, TaskType
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
    ) -> "LoRASLMClassifier":
        """
        Initialise the LoRA model and run the fine-tuning loop.

        Uses gradient accumulation and bfloat16 autocast for memory efficiency.

        Parameters
        ----------
        X_train, y_train, X_val, y_val
            See ``BaseDocumentClassifier.fit``.

        Returns
        -------
        LoRASLMClassifier
            ``self``.
        """
        y_train = np.asarray(y_train, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        # 1. Load & wrap model
        self.model = self._load_base_model()

        # Determine device for non-auto-mapped case
        if not self.use_4bit:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
        else:
            # device_map="auto" handles placement; use the score layer's device
            device = next(
                p.device for n, p in self.model.named_parameters() if "score" in n
            )

        # 2. DataLoaders (left-padded tokenisation)
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        # 3. Optimiser — only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        use_bf16 = torch.cuda.is_available()

        best_val_f1 = -1.0
        best_adapter_state: Optional[dict] = None
        best_head_state: Optional[dict] = None

        # 4. Training loop
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(
                    enabled=use_bf16, dtype=torch.bfloat16
                ):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss = criterion(outputs.logits, labels)
                    loss = loss / self.accumulation_steps

                loss.backward()
                epoch_loss += loss.item() * self.accumulation_steps

                if step % self.accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            # 5. Validation
            val_f1 = self._val_micro_f1(val_loader, y_val, device, use_bf16)
            avg_loss = epoch_loss / len(train_loader)
            logger.info(
                "Epoch %d/%d | loss=%.4f | val_micro_f1=%.4f",
                epoch, self.num_epochs, avg_loss, val_f1,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # Save LoRA adapter state and head state separately
                best_adapter_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                    if "lora_" in k
                }
                best_head_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                    if "score" in k
                }

        # 6. Restore best weights
        if best_adapter_state and best_head_state:
            merged = {**best_adapter_state, **best_head_state}
            self.model.load_state_dict(merged, strict=False)

        self.is_fitted = True
        logger.info("Training complete. Best val Micro-F1: %.4f", best_val_f1)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_test: List[str]) -> np.ndarray:
        """Return multi-hot binary predictions, shape ``(n, num_labels)``."""
        proba = self.predict_proba(X_test)
        return (proba >= self.threshold).astype(np.int8)

    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        """
        Return per-label probability estimates, shape ``(n, num_labels)``.

        Processes in mini-batches with ``torch.no_grad()`` and bfloat16 autocast.
        """
        self._check_fitted()
        device = next(
            p.device for n, p in self.model.named_parameters() if "score" in n
        )
        loader = self._make_loader(X_test, labels=None, shuffle=False)
        use_bf16 = device.type == "cuda"

        self.model.eval()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Save / Load LoRA adapters
    # ------------------------------------------------------------------

    def save_lora(self, output_dir: str) -> None:
        """
        Save LoRA adapter weights and classification head.

        ``output_dir/`` will contain:
        - ``adapter_model.bin`` / ``adapter_config.json`` (PEFT standard)
        - ``score_head.pt`` (classification head state dict)

        Parameters
        ----------
        output_dir : str
            Directory path to write weights.
        """
        self._check_fitted()
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)  # LoRA weights only
        # Save classification head separately
        head_state = {
            k: v for k, v in self.model.state_dict().items() if "score" in k
        }
        torch.save(head_state, os.path.join(output_dir, "score_head.pt"))
        logger.info("LoRA adapter + score head saved to %s", output_dir)

    def load_lora(self, adapter_dir: str) -> None:
        """
        Load previously saved LoRA adapter weights and classification head.

        The base model (with 4-bit quantisation if ``use_4bit=True``) must
        already be initialised by calling ``_load_base_model()`` or via
        ``fit()`` first.

        Parameters
        ----------
        adapter_dir : str
            Directory containing files written by ``save_lora()``.
        """
        from peft import PeftModel

        if self.model is None:
            self.model = self._load_base_model()

        self.model = PeftModel.from_pretrained(self.model.base_model.model, adapter_dir)

        head_path = os.path.join(adapter_dir, "score_head.pt")
        if os.path.exists(head_path):
            head_state = torch.load(head_path, map_location="cpu")
            self.model.load_state_dict(head_state, strict=False)
            logger.info("Loaded score head from %s", head_path)

        self.is_fitted = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenise(
        self, texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Tokenise a list of strings with left-padding."""
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _make_loader(
        self,
        texts: List[str],
        labels: Optional[np.ndarray],
        shuffle: bool = False,
    ) -> DataLoader:
        """Tokenise texts and return a DataLoader."""
        enc = self._tokenise(texts)
        label_tensor = (
            torch.from_numpy(labels.astype(np.float32))
            if labels is not None
            else None
        )
        dataset = _SLMDataset(enc["input_ids"], enc["attention_mask"], label_tensor)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _val_micro_f1(
        self,
        loader: DataLoader,
        y_true: np.ndarray,
        device: torch.device,
        use_bf16: bool,
    ) -> float:
        """Compute Micro-F1 on the validation DataLoader."""
        from sklearn.metrics import f1_score

        self.model.eval()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                    logits = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                all_probs.append(probs)

        proba = np.concatenate(all_probs, axis=0)
        preds = (proba >= self.threshold).astype(np.int8)
        return float(
            f1_score(y_true.astype(np.int8), preds, average="micro", zero_division=0)
        )
