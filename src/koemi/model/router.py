from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from koemi.model.experts import DeterministicExpertMixture, UNASSIGNED_EXPERT


@dataclass(frozen=True)
class RouterStatistics:
    probability_mass: Tensor
    assignment_counts: Tensor
    valid_count: Tensor

    @property
    def balance_loss(self) -> Tensor:
        count = self.valid_count.clamp_min(1)
        importance = self.probability_mass / count
        load = self.assignment_counts / self.assignment_counts.sum().clamp_min(1)
        return self.probability_mass.numel() * (importance * load).sum()


def merge_router_statistics(statistics: list[RouterStatistics]) -> RouterStatistics | None:
    if not statistics:
        return None
    return RouterStatistics(
        sum(item.probability_mass for item in statistics),
        sum(item.assignment_counts for item in statistics),
        sum(item.valid_count for item in statistics),
    )


class LearnedExpertMixture(DeterministicExpertMixture):
    def __init__(self, embedding_size: int, expert_count: int, top_k: int = 1,
                 dispatch: str = "loop") -> None:
        if expert_count < 1:
            raise ValueError("learned routing requires at least one expert")
        super().__init__(embedding_size, expert_count, top_k, dispatch)
        self.gate = nn.Linear(embedding_size, expert_count, bias=False)

    def forward(self, context: Tensor, token_ids: Tensor, previous_token_ids: Tensor,
                valid_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, RouterStatistics]:
        gate_context = context.masked_fill(~valid_mask.unsqueeze(-1), 0)
        probabilities = self.gate(gate_context).float().softmax(dim=-1)
        weights, assignments = probabilities.topk(self.top_k, dim=-1)
        assignments = assignments.masked_fill(~valid_mask.unsqueeze(-1), UNASSIGNED_EXPERT)
        weights = weights.masked_fill(~valid_mask.unsqueeze(-1), 0)
        counts = torch.zeros(self.expert_count, device=context.device, dtype=torch.float32)
        counts.scatter_add_(0, assignments.clamp_min(0).reshape(-1),
                            valid_mask.unsqueeze(-1).expand_as(assignments).reshape(-1).float())
        statistics = RouterStatistics(
            (probabilities * valid_mask.unsqueeze(-1)).sum(dim=(0, 1)),
            counts,
            valid_mask.sum().float(),
        )
        mixed = self.combine(context, assignments, weights, valid_mask)
        return mixed, assignments[..., 0], assignments, statistics

    def combine(self, context: Tensor, assignments: Tensor, weights: Tensor,
                valid_mask: Tensor) -> Tensor:
        flat_context = context.reshape(-1, context.shape[-1])
        pair_rows = torch.arange(flat_context.shape[0], device=context.device).repeat_interleave(self.top_k)
        pair_experts = assignments.reshape(-1)
        pair_weights = weights.reshape(-1)
        updates = torch.zeros_like(flat_context)
        hooked = self._requires_module_dispatch()
        if not torch.is_grad_enabled() and not hooked:
            values = flat_context.index_select(0, pair_rows)
            values = values.masked_fill(pair_experts.lt(0).unsqueeze(-1), 0)
            values = self._apply_batched_experts(values, pair_experts.clamp_min(0))
            updates.index_add_(0, pair_rows, (values * pair_weights.unsqueeze(-1)).to(updates.dtype))
        elif self.dispatch == "segments" and not hooked:
            sortable = pair_experts.masked_fill(pair_experts.lt(0), self.expert_count)
            order = sortable.argsort(stable=True)
            rows = pair_rows.index_select(0, order)
            sorted_context = flat_context.index_select(0, rows)
            sorted_weights = pair_weights.index_select(0, order)
            sizes = torch.bincount(sortable, minlength=self.expert_count + 1)[:self.expert_count].tolist()
            offset = 0
            for expert, size in zip(self.experts, sizes, strict=True):
                if size:
                    end = offset + size
                    values = expert(sorted_context[offset:end]) * sorted_weights[offset:end, None]
                    updates.index_add_(0, rows[offset:end], values.to(updates.dtype))
                    offset = end
        else:
            for index, expert in enumerate(self.experts):
                pairs = torch.nonzero(pair_experts == index, as_tuple=False).squeeze(-1)
                if pairs.numel():
                    rows = pair_rows.index_select(0, pairs)
                    values = expert(flat_context.index_select(0, rows))
                    values = values * pair_weights.index_select(0, pairs).unsqueeze(-1)
                    updates.index_add_(0, rows, values.to(updates.dtype))
        mixed = self.output_normalizer(flat_context + updates).reshape_as(context)
        return torch.where(valid_mask.unsqueeze(-1), mixed, context)
