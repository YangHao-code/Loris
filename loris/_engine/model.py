"""Continue parameters from the authenticated current model checkpoint.

The caller loads its explicitly declared backend before importing this module.
Optimizer, scheduler and EMA reset each round; model parameters do not reset.
"""

import hashlib
import json
from typing import Mapping
import torch
from loris_ssl import chase_backend as implementation
from loris_ssl.chase_backend import ChaseTransformerBackend


def state_dict_sha256(state):
    """Bind every parameter/buffer name, shape, dtype and exact tensor bytes."""
    if not isinstance(state, Mapping) or not state:
        raise ValueError("nonempty tensor state_dict required")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict must map names to tensors")
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            [name, str(value.dtype), list(value.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        # Byte view also works with bfloat16 and scalar state buffers.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _ContinuationBackend(ChaseTransformerBackend):
    def __init__(self, *args, expected_initial_state_sha256, parent_generation, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_initial_state_sha256 = expected_initial_state_sha256
        self.parent_generation = parent_generation
        self.actual_initial_state_sha256 = None
        self.actual_initial_head_sha256 = None

    def _optimizer(self, model):
        actual = state_dict_sha256(model.state_dict())
        if actual != self.expected_initial_state_sha256:
            raise RuntimeError(
                "optimizer initialization differs from the complete current checkpoint state"
            )
        self.actual_initial_state_sha256 = actual
        self.actual_initial_head_sha256 = implementation._head_state_sha256(model)
        optimizer = super()._optimizer(model)
        if optimizer.state:
            raise RuntimeError("declared fresh optimizer already contains state")
        return optimizer

    def fit(self, *args, **kwargs):
        raw = dict(super().fit(*args, **kwargs))
        legacy = {
            "previous_chase_round_state_loaded": raw["previous_chase_round_state_loaded"],
            "initialization_lineage": raw["initialization_lineage"],
        }
        lineage = dict(raw["initialization_lineage"])
        lineage.update(
            kind="paper36_current_checkpoint_parameters",
            previous_chase_round_state_loaded=self.parent_generation > 0,
        )
        raw.update(
            initialization_lineage=lineage,
            previous_chase_round_state_loaded=self.parent_generation > 0,
            legacy_backend_initialization_descriptions=legacy,
            initial_full_state_sha256=self.actual_initial_state_sha256,
            optimizer_reset=True,
            scheduler_reset=True,
            EMA_reinitialized_from_current_parameters=True,
        )
        self._fit_audit = raw
        return self.fit_audit
