from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from koemi.data.tokenizer import TextTokenizer
from koemi.model.network import KoemiModel
from koemi.runtime.fast_decode import (
    BatchDecoder,
    SamplingPolicy,
    TokenSampler,
    clone_state,
)


MINIMUM_DRAFT_PROBABILITY = 1e-12


@dataclass(frozen=True)
class DraftProposal:
    """Tokens a drafter proposes plus the distribution each one was drawn from.

    `token_ids` has shape `[draft_length]`. `probabilities` has shape
    `[draft_length, vocabulary]` and row `i` must give a non-zero mass to
    `token_ids[i]`, otherwise the acceptance ratio is undefined.
    """

    token_ids: Tensor
    probabilities: Tensor

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 1 or self.token_ids.dtype != torch.long:
            raise ValueError("draft token ids must be a one-dimensional long tensor")
        if self.probabilities.ndim != 2 or self.probabilities.shape[0] != self.token_ids.shape[0]:
            raise ValueError("draft probabilities must carry one row per proposed token")

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])


@dataclass(frozen=True)
class AcceptanceResult:
    """Outcome of verifying one draft block against the target distribution.

    `token_ids` are the tokens the target model commits to. `accepted_draft_tokens`
    counts how many of them came from the draft unchanged; the remaining token is
    either the residual resample after a rejection or the bonus token drawn when
    every draft token survived.
    """

    token_ids: tuple[int, ...]
    accepted_draft_tokens: int
    used_bonus: bool


class Drafter(Protocol):
    """Source of speculative continuations for one sequence."""

    def begin(self, context_ids: Sequence[int]) -> None: ...

    def propose(self, draft_length: int) -> DraftProposal | None: ...

    def commit(self, token_ids: Sequence[int]) -> None: ...


def accept_draft_tokens(
    proposal: DraftProposal,
    target_probabilities: Tensor,
    generator: torch.Generator | None = None,
) -> AcceptanceResult:
    """Apply speculative rejection sampling to one draft block.

    `target_probabilities` has shape `[draft_length + 1, vocabulary]`: row `i`
    is the target distribution for proposed token `i`, and the last row is the
    bonus distribution used when every proposed token is accepted. The committed
    tokens are distributed exactly as tokens drawn one at a time from the target
    distribution, up to the floating-point difference between a windowed forward
    and a step-by-step forward.
    """
    draft_length = len(proposal)
    if draft_length == 0:
        raise ValueError("an empty proposal cannot be verified")
    if target_probabilities.ndim != 2 or target_probabilities.shape[0] != draft_length + 1:
        raise ValueError("target probabilities must carry one row per proposal position plus a bonus row")
    if target_probabilities.shape[1] != proposal.probabilities.shape[1]:
        raise ValueError("draft and target must share one vocabulary")
    selection = proposal.token_ids.unsqueeze(1)
    target_mass = target_probabilities[:draft_length].gather(1, selection).squeeze(1)
    draft_mass = proposal.probabilities.gather(1, selection).squeeze(1)
    ratio = (target_mass / draft_mass.clamp_min(MINIMUM_DRAFT_PROBABILITY)).clamp(max=1.0)
    uniform = torch.rand(
        draft_length,
        device=target_probabilities.device,
        dtype=ratio.dtype,
        generator=generator,
    )
    leading_accepts = (uniform < ratio).long().cumprod(dim=0)
    accepted_count = int(leading_accepts.sum())
    kept = tuple(int(token_id) for token_id in proposal.token_ids[:accepted_count].tolist())
    if accepted_count == draft_length:
        bonus = torch.multinomial(target_probabilities[draft_length], 1, generator=generator)
        return AcceptanceResult(kept + (int(bonus),), draft_length, True)
    target_row = target_probabilities[accepted_count]
    residual = (target_row - proposal.probabilities[accepted_count]).clamp_min(0.0)
    residual_mass = float(residual.sum())
    distribution = target_row if residual_mass <= 0.0 else residual / residual_mass
    replacement = torch.multinomial(distribution, 1, generator=generator)
    return AcceptanceResult(kept + (int(replacement),), accepted_count, False)


class NgramDrafter:
    """Propose the continuation that followed the same recent suffix earlier on.

    The proposal is a point mass, so a token is accepted with exactly the
    probability the target model assigns to it. There is no second model and no
    extra parameter memory, which makes this the cheapest draft source for
    repetitive text such as code, tables and quoted context.
    """

    def __init__(
        self,
        vocabulary_size: int,
        device: str | torch.device = "cpu",
        *,
        maximum_order: int = 8,
        minimum_order: int = 2,
    ) -> None:
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        if minimum_order < 1 or maximum_order < minimum_order:
            raise ValueError("maximum_order must be at least minimum_order and both must be positive")
        self.vocabulary_size = vocabulary_size
        self.device = torch.device(device)
        self.maximum_order = maximum_order
        self.minimum_order = minimum_order
        self._context: list[int] = []
        self._positions: dict[int, dict[tuple[int, ...], int]] = {
            order: {} for order in range(minimum_order, maximum_order + 1)
        }
        self._indexed_length = 0

    def begin(self, context_ids: Sequence[int]) -> None:
        self._context = [int(token_id) for token_id in context_ids]
        self._positions = {order: {} for order in range(self.minimum_order, self.maximum_order + 1)}
        self._indexed_length = 0
        self._index_new_positions()

    def commit(self, token_ids: Sequence[int]) -> None:
        self._context.extend(int(token_id) for token_id in token_ids)
        self._index_new_positions()

    def propose(self, draft_length: int) -> DraftProposal | None:
        if draft_length < 1:
            raise ValueError("draft_length must be at least 1")
        context_length = len(self._context)
        highest_order = min(self.maximum_order, context_length)
        for order in range(highest_order, self.minimum_order - 1, -1):
            pattern = tuple(self._context[context_length - order :])
            end = self._positions[order].get(pattern)
            if end is None:
                continue
            candidate = self._context[end : end + draft_length]
            if candidate:
                return self._point_mass_proposal(candidate)
        return None

    def _point_mass_proposal(self, candidate: Sequence[int]) -> DraftProposal:
        token_ids = torch.tensor(list(candidate), dtype=torch.long, device=self.device)
        probabilities = torch.zeros(
            (token_ids.shape[0], self.vocabulary_size), dtype=torch.float32, device=self.device
        )
        probabilities.scatter_(1, token_ids.unsqueeze(1), 1.0)
        return DraftProposal(token_ids, probabilities)

    def _index_new_positions(self) -> None:
        context_length = len(self._context)
        for order, table in self._positions.items():
            first_end = max(order, self._indexed_length)
            for end in range(first_end, context_length):
                table[tuple(self._context[end - order : end])] = end
        self._indexed_length = context_length


class ModelDrafter:
    """Propose tokens from a smaller model that shares the target vocabulary.

    The draft model keeps its own recurrent state. Every proposal is rolled back
    before the committed tokens are replayed, so the draft state always follows
    the tokens the target model actually accepted.
    """

    def __init__(
        self,
        model: KoemiModel,
        *,
        policy: SamplingPolicy | None = None,
        generator: torch.Generator | None = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        model.eval()
        self.decoder = BatchDecoder(model, autocast_dtype=autocast_dtype)
        self.sampler = TokenSampler(policy or SamplingPolicy(), self.decoder.device, generator)
        self._pending_logits: Tensor | None = None

    def begin(self, context_ids: Sequence[int]) -> None:
        prompt = torch.tensor(
            [[int(token_id) for token_id in context_ids]],
            dtype=torch.long,
            device=self.decoder.device,
        )
        self._pending_logits = self.decoder.prefill(prompt)

    def commit(self, token_ids: Sequence[int]) -> None:
        if self._pending_logits is None:
            raise RuntimeError("begin must run before commit")
        committed = torch.tensor(
            [[int(token_id) for token_id in token_ids]],
            dtype=torch.long,
            device=self.decoder.device,
        )
        self._pending_logits = self.decoder.extend(committed)[:, -1]

    def propose(self, draft_length: int) -> DraftProposal | None:
        if self._pending_logits is None:
            raise RuntimeError("begin must run before propose")
        if draft_length < 1:
            raise ValueError("draft_length must be at least 1")
        saved_state = clone_state(self.decoder.state)
        saved_logits = self._pending_logits
        logits = self._pending_logits
        token_rows: list[Tensor] = []
        probability_rows: list[Tensor] = []
        for _ in range(draft_length):
            token, distribution = self.sampler.sample_with_probabilities(logits)
            token_rows.append(token)
            probability_rows.append(distribution)
            logits = self.decoder.step(token)
        self.decoder.adopt(saved_state)
        self._pending_logits = saved_logits
        return DraftProposal(
            torch.cat(token_rows, dim=0).reshape(-1),
            torch.cat(probability_rows, dim=0),
        )


@dataclass(frozen=True)
class SpeculativeStatistics:
    """Counters that decide whether speculation is paying for itself.

    `target_forward_calls` is the number the baseline decoder would keep at
    `committed_tokens`; speculation only wins when it stays clearly below that.
    """

    rounds: int
    proposed_tokens: int
    accepted_draft_tokens: int
    committed_tokens: int
    target_forward_calls: int

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.proposed_tokens

    @property
    def tokens_per_target_call(self) -> float:
        if self.target_forward_calls == 0:
            return 0.0
        return self.committed_tokens / self.target_forward_calls


@dataclass(frozen=True)
class SpeculativeGeneration:
    token_ids: tuple[int, ...]
    text: str
    statistics: SpeculativeStatistics


def speculative_generate(
    model: KoemiModel,
    tokenizer: TextTokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    drafter: Drafter,
    device: str | torch.device = "cpu",
    policy: SamplingPolicy | None = None,
    generator: torch.Generator | None = None,
    draft_length: int = 4,
    autocast_dtype: torch.dtype | None = None,
    prompt_token_ids: Sequence[int] | None = None,
) -> SpeculativeGeneration:
    """Generate one sequence by verifying whole draft blocks instead of single tokens.

    Each round runs one windowed forward over the proposed block and one forward
    over the tokens that survived verification, so a block of `m` accepted tokens
    costs two target forwards instead of `m`. Speculation loses when acceptance
    collapses; `SpeculativeStatistics` reports the ratio that decides it.
    """
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if draft_length < 1:
        raise ValueError("draft_length must be at least 1")
    resolved_device = torch.device(device)
    model.eval()
    decoder = BatchDecoder(model, autocast_dtype=autocast_dtype)
    sampler = TokenSampler(policy or SamplingPolicy(), resolved_device, generator)
    prompt_ids = (
        tuple(tokenizer.encode(prompt))
        if prompt_token_ids is None
        else tuple(int(token_id) for token_id in prompt_token_ids)
    )
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")
    pending_logits = decoder.prefill(
        torch.tensor([list(prompt_ids)], dtype=torch.long, device=resolved_device)
    )
    drafter.begin(prompt_ids)
    generated: list[int] = []
    rounds = 0
    proposed_tokens = 0
    accepted_draft_tokens = 0
    target_forward_calls = 1
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        proposal = drafter.propose(min(draft_length, remaining))
        if proposal is None or len(proposal) == 0:
            token = sampler(pending_logits)
            committed = (int(token),)
            generated.extend(committed)
            drafter.commit(committed)
            if len(generated) >= max_new_tokens:
                break
            pending_logits = decoder.step(token)
            target_forward_calls += 1
            continue
        rounds += 1
        proposed_tokens += len(proposal)
        saved_state = clone_state(decoder.state)
        verification_logits = decoder.extend(proposal.token_ids.reshape(1, -1))
        target_forward_calls += 1
        target_rows = torch.cat((pending_logits, verification_logits[0]), dim=0)
        result = accept_draft_tokens(proposal, sampler.probabilities(target_rows), generator)
        accepted_draft_tokens += result.accepted_draft_tokens
        generated.extend(result.token_ids)
        drafter.commit(result.token_ids)
        if len(generated) >= max_new_tokens:
            break
        if result.used_bonus:
            bonus = torch.tensor(
                [[result.token_ids[-1]]], dtype=torch.long, device=resolved_device
            )
            pending_logits = decoder.step(bonus)
        else:
            decoder.adopt(saved_state)
            committed = torch.tensor(
                [list(result.token_ids)], dtype=torch.long, device=resolved_device
            )
            pending_logits = decoder.extend(committed)[:, -1]
        target_forward_calls += 1
    token_ids = tuple(generated[:max_new_tokens])
    statistics = SpeculativeStatistics(
        rounds=rounds,
        proposed_tokens=proposed_tokens,
        accepted_draft_tokens=accepted_draft_tokens,
        committed_tokens=len(token_ids),
        target_forward_calls=target_forward_calls,
    )
    return SpeculativeGeneration(token_ids, tokenizer.decode(token_ids), statistics)


__all__ = [
    "AcceptanceResult",
    "DraftProposal",
    "Drafter",
    "MINIMUM_DRAFT_PROBABILITY",
    "ModelDrafter",
    "NgramDrafter",
    "SpeculativeGeneration",
    "SpeculativeStatistics",
    "accept_draft_tokens",
    "speculative_generate",
]
