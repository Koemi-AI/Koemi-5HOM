from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from koemi.configuration.settings import PAD_TOKEN_ID


KOEMI_STATE_FIELDS = (
    "working_state",
    "memory_basis",
    "memory_normalizer",
    "refine_basis",
    "refine_normalizer",
    "local_keys",
    "local_values",
    "local_valid",
    "salient_keys",
    "salient_values",
    "salient_valid",
    "last_token_ids",
)


@dataclass(frozen=True)
class KoemiState:
    working_state: Tensor
    memory_basis: Tensor
    memory_normalizer: Tensor
    refine_basis: Tensor
    refine_normalizer: Tensor
    local_keys: Tensor
    local_values: Tensor
    local_valid: Tensor
    salient_keys: Tensor
    salient_values: Tensor
    salient_valid: Tensor
    last_token_ids: Tensor
    step_index: int

    @classmethod
    def create_static(
        cls,
        batch_size: int,
        embedding_size: int,
        memory_features: int,
        local_memory_size: int,
        salience_memory_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> KoemiState:
        """Create an empty state whose ring buffers already hold their capacity.

        Every ring slot is zero and marked invalid, so the reads produce the same
        values as the growing state created by `create`. The shapes stay constant
        across decode steps, which is what CUDA graph capture requires.
        """
        if local_memory_size < 1 or salience_memory_size < 1:
            raise ValueError("static state requires positive ring capacities")
        return cls(
            working_state=torch.zeros(batch_size, embedding_size, device=device, dtype=dtype),
            memory_basis=torch.zeros(batch_size, embedding_size, memory_features, device=device, dtype=dtype),
            memory_normalizer=torch.zeros(batch_size, memory_features, device=device, dtype=dtype),
            refine_basis=torch.zeros(batch_size, embedding_size, memory_features, device=device, dtype=dtype),
            refine_normalizer=torch.zeros(batch_size, memory_features, device=device, dtype=dtype),
            local_keys=torch.zeros(batch_size, local_memory_size, embedding_size, device=device, dtype=dtype),
            local_values=torch.zeros(batch_size, local_memory_size, embedding_size, device=device, dtype=dtype),
            local_valid=torch.zeros(batch_size, local_memory_size, dtype=torch.bool, device=device),
            salient_keys=torch.zeros(batch_size, salience_memory_size, embedding_size, device=device, dtype=dtype),
            salient_values=torch.zeros(batch_size, salience_memory_size, embedding_size, device=device, dtype=dtype),
            salient_valid=torch.zeros(batch_size, salience_memory_size, dtype=torch.bool, device=device),
            last_token_ids=torch.full((batch_size,), PAD_TOKEN_ID, dtype=torch.long, device=device),
            step_index=0,
        )

    @classmethod
    def create(cls, batch_size: int, embedding_size: int, memory_features: int, device: torch.device) -> KoemiState:
        return cls(
            working_state=torch.zeros(batch_size, embedding_size, device=device),
            memory_basis=torch.zeros(batch_size, embedding_size, memory_features, device=device),
            memory_normalizer=torch.zeros(batch_size, memory_features, device=device),
            refine_basis=torch.zeros(batch_size, embedding_size, memory_features, device=device),
            refine_normalizer=torch.zeros(batch_size, memory_features, device=device),
            local_keys=torch.empty(batch_size, 0, embedding_size, device=device),
            local_values=torch.empty(batch_size, 0, embedding_size, device=device),
            local_valid=torch.empty(batch_size, 0, dtype=torch.bool, device=device),
            salient_keys=torch.empty(batch_size, 0, embedding_size, device=device),
            salient_values=torch.empty(batch_size, 0, embedding_size, device=device),
            salient_valid=torch.empty(batch_size, 0, dtype=torch.bool, device=device),
            last_token_ids=torch.full((batch_size,), PAD_TOKEN_ID, dtype=torch.long, device=device),
            step_index=0,
        )
