from __future__ import annotations

import math
import re
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


def _read_hidden_size(config) -> int:
    for name in ("hidden_size", "d_model", "dim"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("cannot determine encoder hidden size")


class MultiLabelEncoder(nn.Module):
    """Transformer encoder with pooled or label-wise attention prediction.

    Each class has its own query and angular classifier direction.  The latter
    exposes the label-specific feature angles needed by the BBAM auxiliary.
    """

    def __init__(
        self,
        model_name: str,
        n_labels: int,
        head: str = "label_attention",
        dropout: float = 0.1,
        projection_dim: int = 128,
        gradient_checkpointing: bool = False,
        offline: bool = False,
        cache_dir: str = "",
        encoder_config_path: str = "",
        initialize_encoder_from_pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.n_labels = int(n_labels)
        self.head_type = head.replace("_linear", "")
        cache_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        config_source = encoder_config_path or model_name
        config = AutoConfig.from_pretrained(
            config_source,
            local_files_only=(True if encoder_config_path else offline),
            **({} if encoder_config_path else cache_kwargs),
        )
        if initialize_encoder_from_pretrained:
            self.encoder = AutoModel.from_pretrained(
                model_name, config=config, local_files_only=offline, **cache_kwargs,
            )
        else:
            if not encoder_config_path:
                raise ValueError(
                    "encoder_config_path is required when pretrained encoder "
                    "initialisation is disabled"
                )
            # Scoring a fully trained checkpoint should not require fetching or
            # even opening the original pretrained snapshot.  Construct the
            # exact architecture from the run-bound config, then strictly load
            # every trained tensor from the authenticated checkpoint.
            self.encoder = AutoModel.from_config(config)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        self.hidden_size = _read_hidden_size(config)
        self.dropout = nn.Dropout(dropout)

        context_head = self.head_type
        self.linear_classifier = head.endswith("_linear")
        if context_head == "label_attention":
            self.label_queries = nn.Parameter(torch.empty(n_labels, self.hidden_size))
            nn.init.normal_(self.label_queries, std=0.02)
        elif context_head == "pooled":
            self.register_parameter("label_queries", None)
        else:
            raise ValueError("head must be label_attention or pooled")

        self.classifier_weight = nn.Parameter(torch.empty(n_labels, self.hidden_size))
        nn.init.normal_(self.classifier_weight, std=0.02)
        self.classifier_bias = nn.Parameter(torch.zeros(n_labels))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
        )
        self.binary_prototypes = nn.Parameter(torch.empty(n_labels, 2, projection_dim))
        nn.init.normal_(self.binary_prototypes, std=0.02)
        self.prototype_temperature = nn.Parameter(torch.tensor(math.log(1.0 / 0.10)))

    def _masked_mean(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _label_context(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.head_type == "pooled":
            pooled = self._masked_mean(hidden, attention_mask)
            return pooled.unsqueeze(1).expand(-1, self.n_labels, -1)
        queries = F.normalize(self.label_queries, dim=-1)
        # Queries are already unit-normalised.  Dividing once more by sqrt(d)
        # makes the softmax nearly uniform (std ~= 1/sqrt(d)) and destroys the
        # intended label-specific contexts.
        scores = torch.einsum("bsh,lh->bsl", hidden, queries)
        scores = scores.masked_fill(~attention_mask.bool().unsqueeze(-1), -1.0e4)
        attention = torch.softmax(scores, dim=1)
        return torch.einsum("bsl,bsh->blh", attention, hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **_: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        encoded = self.encoder(**encoder_kwargs, return_dict=True)
        hidden = encoded.last_hidden_state
        # A head-independent document representation for downstream, frozen
        # source-role analyses.  This is deliberately derived directly from
        # the encoder's first token: it adds no trainable state and therefore
        # remains compatible with every checkpoint created before this output
        # key existed.
        document_features = F.normalize(hidden[:, 0], dim=-1)
        context = self.dropout(self._label_context(hidden, attention_mask))
        # A second document view aligned with the prediction head.  For the
        # pooled head this is exactly its masked-mean document context; for
        # label attention it averages the per-label contexts.  Like the CLS
        # view above, this is output-only and adds no checkpoint state.
        head_context_features = F.normalize(context.mean(dim=1), dim=-1)
        normal_context = F.normalize(context, dim=-1)
        normal_weight = F.normalize(self.classifier_weight, dim=-1)
        cosine = torch.einsum("blh,lh->bl", normal_context, normal_weight).clamp(-0.9999, 0.9999)
        if self.linear_classifier:
            logits = torch.einsum("blh,lh->bl", context, self.classifier_weight)
            logits = logits + self.classifier_bias
        else:
            scale = self.logit_scale.exp().clamp(1.0, 100.0)
            logits = scale * cosine + self.classifier_bias

        projected = F.normalize(self.projection(context), dim=-1)
        prototypes = F.normalize(self.binary_prototypes, dim=-1)
        prototype_scale = self.prototype_temperature.exp().clamp(1.0, 100.0)
        prototype_logits = prototype_scale * torch.einsum(
            "bld,lcd->blc", projected, prototypes,
        )
        pooled_projection = F.normalize(projected.mean(dim=1), dim=-1)
        return {
            "logits": logits,
            "cosine": cosine,
            "features": projected,
            "pooled_features": pooled_projection,
            "document_features": document_features,
            "head_context_features": head_context_features,
            "prototype_logits": prototype_logits,
        }

    @torch.no_grad()
    def initialize_label_representations(
        self,
        tokenizer,
        label_names: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> None:
        if len(label_names) != self.n_labels:
            raise ValueError("wrong number of label names")
        target_device = device or next(self.encoder.parameters()).device
        descriptions = []
        for name in label_names:
            description = re.sub(r"[._/\-]+", " ", str(name))
            description = re.sub(r"\s+", " ", description).strip()
            descriptions.append(description or str(name))
        tokenized = tokenizer(
            descriptions, padding=True, truncation=True, max_length=32,
            return_tensors="pt",
        )
        tokenized = {key: value.to(target_device) for key, value in tokenized.items()}
        was_training = self.encoder.training
        self.encoder.eval()
        encoded = self.encoder(**tokenized, return_dict=True).last_hidden_state
        mask = tokenized["attention_mask"].to(encoded.dtype).unsqueeze(-1)
        representations = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        representations = F.normalize(representations, dim=-1)
        classifier_init = representations * (0.02 if self.linear_classifier else 1.0)
        self.classifier_weight.copy_(classifier_init.to(self.classifier_weight.dtype))
        if self.label_queries is not None:
            self.label_queries.copy_(representations.to(self.label_queries.dtype))
        self.encoder.train(was_training)

    @torch.no_grad()
    def initialize_class_priors(self, positive_rates: torch.Tensor) -> None:
        rates = positive_rates.to(self.classifier_bias.device).float().clamp(1.0e-4, 1.0 - 1.0e-4)
        self.classifier_bias.copy_(torch.log(rates / (1.0 - rates)))


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    teacher_parameters = dict(teacher.named_parameters())
    for name, student_parameter in student.named_parameters():
        teacher_parameters[name].mul_(decay).add_(student_parameter.detach(), alpha=1.0 - decay)
    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        if name in teacher_buffers:
            teacher_buffers[name].copy_(student_buffer)
