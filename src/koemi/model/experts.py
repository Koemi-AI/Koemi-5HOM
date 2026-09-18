from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from koemi.model.layers import GatedFeedForward, RootMeanSquareNorm


TOKEN_HASH_FACTOR = 1_000_003
PREVIOUS_TOKEN_HASH_FACTOR = 97_409
HASH_WIDTH_MASK = 0xFFFFFFFF
HASH_MIX_MULTIPLIER = 0x45D9F3B
HASH_MIX_SHIFT = 16
UNASSIGNED_EXPERT = -1


@dataclass(frozen=True)
class StackedExpertWeights:
    gate_weights: Tensor
    gate_biases: Tensor
    value_weights: Tensor
    value_biases: Tensor
    output_weights: Tensor
    output_biases: Tensor


def content_dispatch_hash(token_ids: Tensor, previous_token_ids: Tensor) -> Tensor:
    value = (
        token_ids * TOKEN_HASH_FACTOR + previous_token_ids * PREVIOUS_TOKEN_HASH_FACTOR
    ) & HASH_WIDTH_MASK
    value = value ^ (value >> HASH_MIX_SHIFT)
    value = (value * HASH_MIX_MULTIPLIER) & HASH_WIDTH_MASK
    value = value ^ (value >> HASH_MIX_SHIFT)
    value = (value * HASH_MIX_MULTIPLIER) & HASH_WIDTH_MASK
    return value ^ (value >> HASH_MIX_SHIFT)


class DeterministicExpertMixture(nn.Module):
    """Content-dispatched experts with a static tensor path for inference.

    Autograd and module hooks retain the reference dispatch path so offload and
    gradient contracts stay bit-exact until a profiled training kernel exists.
    """

    def __init__(
        self,
        embedding_size: int,
        expert_count: int,
        top_k: int = 1,
        dispatch: str = "loop",
    ) -> None:
        super().__init__()
        self.expert_count = expert_count
        self.top_k = top_k
        if expert_count < 0 or top_k < 1 or (expert_count > 0 and top_k > expert_count):
            raise ValueError("top_k must be between one and expert_count")
        if dispatch not in {"loop", "segments"}:
            raise ValueError("dispatch must be loop or segments")
        self.dispatch = dispatch
        self.experts = nn.ModuleList(GatedFeedForward(embedding_size) for _ in range(expert_count))
        self.output_normalizer = RootMeanSquareNorm(embedding_size)
        self._stacked_weights: StackedExpertWeights | None = None

    def cache_stacked_experts(self) -> None:
        """Stack the expert weights once so decoding stops rebuilding them per step.

        The cache is only correct while the weights do not change. Call
        `clear_stacked_experts` before training again, after loading a checkpoint
        or after moving the module to another device. Raises when gradients are
        enabled, because a cached stack would detach the experts from autograd.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("stacked expert weights can only be cached for inference")
        if self.expert_count == 0:
            self._stacked_weights = None
            return
        self._stacked_weights = self._stack_expert_weights()

    def clear_stacked_experts(self) -> None:
        """Drop the cached stack so the next call reads the live expert weights."""
        self._stacked_weights = None

    def _load_from_state_dict(self, *arguments, **keywords) -> None:
        self._stacked_weights = None
        super()._load_from_state_dict(*arguments, **keywords)

    def forward(
        self,
        context: Tensor,
        token_ids: Tensor,
        previous_token_ids: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.expert_count == 0:
            empty = torch.full_like(token_ids, UNASSIGNED_EXPERT)
            return context, empty, empty.unsqueeze(-1)
        assignments = self.assign_top_k(token_ids, previous_token_ids, valid_mask)
        if self._requires_module_dispatch():
            return self._forward_with_module_dispatch(context, assignments, valid_mask)
        if torch.is_grad_enabled():
            if self.dispatch == "segments":
                return self._forward_with_sorted_segments(context, assignments, valid_mask)
            return self._forward_with_module_dispatch(context, assignments, valid_mask)
        return self._forward_with_static_dispatch(context, assignments, valid_mask)

    def _forward_with_static_dispatch(
        self,
        context: Tensor,
        assignments: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        flattened_context = context.reshape(-1, context.shape[-1])
        flattened_assignments = assignments.reshape(-1, self.top_k)
        valid_rows = valid_mask.reshape(-1)
        pair_rows = torch.arange(
            flattened_context.shape[0], device=flattened_context.device
        ).repeat_interleave(self.top_k)
        pair_experts = flattened_assignments.reshape(-1)
        pair_valid = (
            pair_experts.ge(0)
            & pair_experts.lt(self.expert_count)
            & valid_rows.repeat_interleave(self.top_k)
        )
        pair_valid = pair_valid & self._first_assignment_occurrence(flattened_assignments).reshape(-1)
        safe_pair_experts = pair_experts.clamp(0, self.expert_count - 1)
        pair_context = flattened_context.index_select(0, pair_rows)
        pair_context = torch.where(
            pair_valid.unsqueeze(-1),
            pair_context,
            torch.zeros((), dtype=pair_context.dtype, device=pair_context.device),
        )
        dispatched_context = self._apply_batched_experts(pair_context, safe_pair_experts)
        dispatched_context = dispatched_context * (
            pair_valid.to(dispatched_context.dtype).unsqueeze(-1) / self.top_k
        )
        expert_updates = torch.zeros_like(flattened_context)
        expert_updates.index_add_(0, pair_rows, dispatched_context)
        updated_context = self.output_normalizer(flattened_context + expert_updates)
        mixed_context = torch.where(valid_rows.unsqueeze(-1), updated_context, flattened_context)
        return mixed_context.reshape_as(context), assignments[:, :, 0], assignments

    def _forward_with_module_dispatch(
        self,
        context: Tensor,
        assignments: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        flattened_context = context.reshape(-1, context.shape[-1])
        flattened_assignments = assignments.reshape(-1, self.top_k)
        expert_updates = torch.zeros_like(flattened_context)
        for expert_index, expert in enumerate(self.experts):
            row_indices = torch.nonzero(
                (flattened_assignments == expert_index).any(dim=1), as_tuple=False
            ).squeeze(-1)
            if row_indices.numel() == 0:
                continue
            expert_context = expert(flattened_context.index_select(0, row_indices))
            expert_updates.index_add_(
                0,
                row_indices,
                (expert_context / self.top_k).to(dtype=expert_updates.dtype),
            )
        valid_rows = valid_mask.reshape(-1)
        updated_context = self.output_normalizer(flattened_context + expert_updates)
        mixed_context = torch.where(valid_rows.unsqueeze(-1), updated_context, flattened_context)
        return mixed_context.reshape_as(context), assignments[:, :, 0], assignments

    def _forward_with_sorted_segments(
        self,
        context: Tensor,
        assignments: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        flattened_context = context.reshape(-1, context.shape[-1])
        flattened_assignments = assignments.reshape(-1, self.top_k)
        pair_count = flattened_assignments.numel()
        device = flattened_context.device
        pair_rows = torch.arange(flattened_context.shape[0], device=device).repeat_interleave(self.top_k)
        pair_experts = flattened_assignments.reshape(-1)
        pair_valid = (
            pair_experts.ge(0)
            & pair_experts.lt(self.expert_count)
            & valid_mask.reshape(-1).repeat_interleave(self.top_k)
            & self._first_assignment_occurrence(flattened_assignments).reshape(-1)
        )
        sortable_experts = torch.where(
            pair_valid, pair_experts, pair_experts.new_full((), self.expert_count)
        )
        counts = torch.zeros(self.expert_count + 1, dtype=torch.long, device=device)
        counts.scatter_add_(0, sortable_experts, torch.ones_like(sortable_experts))
        order = torch.argsort(sortable_experts, stable=True)
        sorted_rows = pair_rows.index_select(0, order)
        sorted_context = flattened_context.index_select(0, sorted_rows)
        segment_sizes = counts[: self.expert_count].tolist()
        expert_updates = torch.zeros_like(flattened_context)
        offset = 0
        for expert_index, segment_size in enumerate(segment_sizes):
            if segment_size == 0:
                continue
            end = offset + segment_size
            expert_context = self.experts[expert_index](sorted_context[offset:end])
            expert_updates.index_add_(
                0,
                sorted_rows[offset:end],
                (expert_context / self.top_k).to(dtype=expert_updates.dtype),
            )
            offset = end
        if offset > pair_count:
            raise RuntimeError("sorted expert segments covered more pairs than the dispatch produced")
        valid_rows = valid_mask.reshape(-1)
        updated_context = self.output_normalizer(flattened_context + expert_updates)
        mixed_context = torch.where(valid_rows.unsqueeze(-1), updated_context, flattened_context)
        return mixed_context.reshape_as(context), assignments[:, :, 0], assignments

    def _stack_expert_weights(self) -> StackedExpertWeights:
        return StackedExpertWeights(
            gate_weights=torch.stack(tuple(expert.gate_projection.weight for expert in self.experts)),
            gate_biases=torch.stack(tuple(expert.gate_projection.bias for expert in self.experts)),
            value_weights=torch.stack(tuple(expert.value_projection.weight for expert in self.experts)),
            value_biases=torch.stack(tuple(expert.value_projection.bias for expert in self.experts)),
            output_weights=torch.stack(tuple(expert.output_projection.weight for expert in self.experts)),
            output_biases=torch.stack(tuple(expert.output_projection.bias for expert in self.experts)),
        )

    def _apply_batched_experts(self, values: Tensor, expert_indices: Tensor) -> Tensor:
        if values.shape[0] == 0:
            return values
        stacked = self._stacked_weights
        if (
            stacked is None
            or stacked.gate_weights.device != values.device
            or stacked.gate_weights.dtype != self.experts[0].gate_projection.weight.dtype
        ):
            stacked = self._stack_expert_weights()
        selected_gate_weights = stacked.gate_weights.index_select(0, expert_indices)
        selected_gate_biases = stacked.gate_biases.index_select(0, expert_indices)
        selected_value_weights = stacked.value_weights.index_select(0, expert_indices)
        selected_value_biases = stacked.value_biases.index_select(0, expert_indices)
        selected_output_weights = stacked.output_weights.index_select(0, expert_indices)
        selected_output_biases = stacked.output_biases.index_select(0, expert_indices)
        gate_values = torch.vmap(functional.linear)(
            values,
            selected_gate_weights,
            selected_gate_biases,
        )
        value_values = torch.vmap(functional.linear)(
            values,
            selected_value_weights,
            selected_value_biases,
        )
        gated_values = functional.silu(gate_values) * value_values
        return torch.vmap(functional.linear)(
            gated_values,
            selected_output_weights,
            selected_output_biases,
        )

    def _first_assignment_occurrence(self, assignments: Tensor) -> Tensor:
        if self.top_k == 1:
            return torch.ones_like(assignments, dtype=torch.bool)
        slot_indices = torch.arange(self.top_k, device=assignments.device)
        earlier_slot = slot_indices.unsqueeze(0) < slot_indices.unsqueeze(1)
        repeated_assignment = assignments.unsqueeze(-1) == assignments.unsqueeze(-2)
        return ~(repeated_assignment & earlier_slot).any(dim=-1)

    def _requires_module_dispatch(self) -> bool:
        return any(
            parameter.numel() == 0
            for expert in self.experts
            for parameter in expert.parameters()
        ) or any(
            bool(module._forward_pre_hooks or module._forward_hooks)
            for expert in self.experts
            for module in expert.modules()
        )

    def assign(self, token_ids: Tensor, previous_token_ids: Tensor, valid_mask: Tensor) -> Tensor:
        return self.assign_top_k(token_ids, previous_token_ids, valid_mask)[:, :, 0]

    def assign_top_k(self, token_ids: Tensor, previous_token_ids: Tensor, valid_mask: Tensor) -> Tensor:
        if self.expert_count == 0:
            return torch.full((*token_ids.shape, self.top_k), UNASSIGNED_EXPERT, dtype=torch.long, device=token_ids.device)
        context_hash = content_dispatch_hash(token_ids, previous_token_ids)
        offsets = torch.arange(self.top_k, device=token_ids.device, dtype=context_hash.dtype)
        assignments = (context_hash.unsqueeze(-1) + offsets * 0x9E3779B9).remainder(self.expert_count)
        return assignments.masked_fill(~valid_mask.unsqueeze(-1), UNASSIGNED_EXPERT)
