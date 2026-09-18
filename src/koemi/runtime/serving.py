"""Continuous-batching serving engine for Koemi.

`InferenceBatchScheduler` owns the queue and never runs a model; `BatchDecoder`
runs a model and assumes one batch stays together from prefill to finish. Real
serving needs both: requests arrive at any time, join a decode batch already in
flight, stream their tokens, and leave without restarting anyone else.

Joining mid-flight is exact here because no computation in the model reads
`KoemiState.step_index`; `D-022` removed absolute position from the expert hash,
so a row that started 200 tokens ago and a row that started this step decode
identically side by side.

This module is the engine, not a server. It exposes a Python API and no
transport, because a network surface is a separate decision.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Callable

import torch
from torch import Tensor

from koemi.configuration.settings import PAD_TOKEN_ID
from koemi.model.network import KoemiModel
from koemi.model.state import KoemiState
from koemi.runtime.fast_decode import (
    BatchDecoder,
    SamplingPolicy,
    TokenSampler,
    select_state_rows,
    stack_states,
    to_static_state,
)


DEFAULT_MAX_CONCURRENT_SEQUENCES = 32
DEFAULT_MAX_PROMPT_TOKENS = 8_192
DEFAULT_MAX_NEW_TOKENS = 1_024
DEFAULT_PREFILL_BUCKET_SIZE = 16

CANCELLED = "cancelled"
COMPLETED = "completed"
DEADLINE_EXCEEDED = "deadline_exceeded"
STOPPED = "stopped"
TOKEN_LIMIT = "token_limit"


@dataclass(frozen=True)
class ServingLimits:
    """Bounds every accepted request is checked against.

    These are the admission control for the engine. A prompt is rejected at
    submit, never part way through execution, so a caller cannot occupy the
    batch and then fail.
    """

    max_concurrent_sequences: int = DEFAULT_MAX_CONCURRENT_SEQUENCES
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    prefill_bucket_size: int = DEFAULT_PREFILL_BUCKET_SIZE

    def __post_init__(self) -> None:
        for value, name in (
            (self.max_concurrent_sequences, "max_concurrent_sequences"),
            (self.max_prompt_tokens, "max_prompt_tokens"),
            (self.max_new_tokens, "max_new_tokens"),
            (self.prefill_bucket_size, "prefill_bucket_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class GenerationRequest:
    """One caller's generation, validated before it can occupy the batch."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int = 64
    policy: SamplingPolicy = field(default_factory=SamplingPolicy)
    stop_token_ids: tuple[int, ...] = ()
    deadline_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if isinstance(self.max_new_tokens, bool) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(self.policy, SamplingPolicy):
            raise TypeError("policy must be a SamplingPolicy")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0.0:
            raise ValueError("deadline_seconds must be positive when given")


@dataclass(frozen=True)
class GenerationOutcome:
    """Terminal record for one request, delivered after the stream closes."""

    request_id: str
    token_ids: tuple[int, ...]
    reason: str
    queue_wait_seconds: float
    time_to_first_token_seconds: float | None
    latency_seconds: float
    decode_steps: int


class TokenStream:
    """Ordered, bounded token channel for one request.

    Iterating yields token ids as the engine produces them and stops when the
    request reaches a terminal state. `outcome` is available once the stream has
    closed, so a consumer always learns why generation ended.
    """

    _SENTINEL = object()

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._tokens: queue.SimpleQueue = queue.SimpleQueue()
        self._outcome: GenerationOutcome | None = None
        self._closed = threading.Event()

    def _emit(self, token_id: int) -> None:
        self._tokens.put(int(token_id))

    def _close(self, outcome: GenerationOutcome) -> None:
        self._outcome = outcome
        self._tokens.put(self._SENTINEL)
        self._closed.set()

    def __iter__(self) -> Iterator[int]:
        while True:
            item = self._tokens.get()
            if item is self._SENTINEL:
                return
            yield item

    def wait(self, timeout: float | None = None) -> GenerationOutcome:
        """Block until the request finishes and return its outcome."""
        if not self._closed.wait(timeout):
            raise TimeoutError(f"request {self.request_id} did not finish within {timeout} seconds")
        assert self._outcome is not None
        return self._outcome

    @property
    def outcome(self) -> GenerationOutcome | None:
        return self._outcome

    @property
    def closed(self) -> bool:
        return self._closed.is_set()


@dataclass
class _Sequence:
    request: GenerationRequest
    stream: TokenStream
    state: KoemiState
    next_token: Tensor
    generated: list[int]
    submitted_at: float
    started_at: float
    first_token_at: float | None = None
    decode_steps: int = 0
    cancelled: bool = False

    @property
    def request_id(self) -> str:
        return self.request.request_id

    def terminal_reason(self, now: float) -> str | None:
        if self.cancelled:
            return CANCELLED
        if self.generated and self.generated[-1] in self.request.stop_token_ids:
            return STOPPED
        if len(self.generated) >= self.request.max_new_tokens:
            return TOKEN_LIMIT
        if self.request.deadline_seconds is not None:
            if now - self.submitted_at >= self.request.deadline_seconds:
                return DEADLINE_EXCEEDED
        return None


@dataclass(frozen=True)
class ServingMetrics:
    """Counters a caller can use to decide whether the engine keeps up."""

    admitted_requests: int
    completed_requests: int
    rejected_requests: int
    decode_steps: int
    generated_tokens: int
    active_sequences: int
    waiting_requests: int
    peak_active_sequences: int

    @property
    def tokens_per_decode_step(self) -> float:
        if self.decode_steps == 0:
            return 0.0
        return self.generated_tokens / self.decode_steps


class ServingEngine:
    """Run many generations together, admitting and retiring rows every step.

    The engine is driven by `step`, which does exactly one unit of work: admit
    what fits, prefill the admissions, decode every active row once, emit the
    tokens and retire whatever finished. A caller that wants a background loop
    uses `run_until_idle` or drives `step` from its own thread; keeping the unit
    explicit is what makes the continuous-batching equivalence testable.

    CUDA graph capture is deliberately not used: the batch width changes whenever
    a row joins or leaves, and a captured graph is fixed to one shape.
    """

    def __init__(
        self,
        model: KoemiModel,
        *,
        limits: ServingLimits | None = None,
        autocast_dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if model.training:
            raise RuntimeError("the serving engine requires a model in evaluation mode")
        self.model = model
        self.limits = limits or ServingLimits()
        self.decoder = BatchDecoder(model, autocast_dtype=autocast_dtype)
        self.device = self.decoder.device
        self._generator = generator
        self._clock = clock or time.monotonic
        self._samplers: dict[SamplingPolicy, TokenSampler] = {}
        self._waiting: list[tuple[GenerationRequest, TokenStream, float]] = []
        self._active: list[_Sequence] = []
        self._lock = threading.Lock()
        self._admitted_requests = 0
        self._completed_requests = 0
        self._rejected_requests = 0
        self._decode_steps = 0
        self._generated_tokens = 0
        self._peak_active_sequences = 0

    def submit(self, request: GenerationRequest) -> TokenStream:
        """Validate and queue one request, returning its token stream."""
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")
        self._validate_against_limits(request)
        stream = TokenStream(request.request_id)
        with self._lock:
            if any(request.request_id == waiting.request_id for waiting, _, _ in self._waiting):
                raise ValueError(f"request {request.request_id} is already queued")
            if any(request.request_id == sequence.request_id for sequence in self._active):
                raise ValueError(f"request {request.request_id} is already running")
            self._waiting.append((request, stream, self._clock()))
        return stream

    def cancel(self, request_id: str) -> bool:
        """Stop a request within one decode step, whether queued or running."""
        with self._lock:
            for index, (waiting, stream, submitted_at) in enumerate(self._waiting):
                if waiting.request_id == request_id:
                    del self._waiting[index]
                    now = self._clock()
                    stream._close(
                        GenerationOutcome(
                            request_id=request_id,
                            token_ids=(),
                            reason=CANCELLED,
                            queue_wait_seconds=now - submitted_at,
                            time_to_first_token_seconds=None,
                            latency_seconds=now - submitted_at,
                            decode_steps=0,
                        )
                    )
                    return True
            for sequence in self._active:
                if sequence.request_id == request_id:
                    sequence.cancelled = True
                    return True
        return False

    def step(self) -> int:
        """Run one admission, one decode pass and one retirement. Return active rows."""
        self._admit()
        self._decode_once()
        self._retire()
        with self._lock:
            return len(self._active)

    def run_until_idle(self, max_steps: int | None = None) -> int:
        """Drive `step` until nothing is waiting or running. Return the step count."""
        steps = 0
        while True:
            with self._lock:
                idle = not self._waiting and not self._active
            if idle:
                return steps
            if max_steps is not None and steps >= max_steps:
                raise RuntimeError(f"serving did not drain within {max_steps} steps")
            self.step()
            steps += 1

    def generate(self, requests: Sequence[GenerationRequest]) -> dict[str, GenerationOutcome]:
        """Submit every request, drain the engine and return the outcomes by id."""
        streams = {request.request_id: self.submit(request) for request in requests}
        budget = sum(request.max_new_tokens for request in requests) + len(requests) + 1
        self.run_until_idle(max_steps=budget)
        return {request_id: stream.wait(0.0) for request_id, stream in streams.items()}

    def metrics(self) -> ServingMetrics:
        with self._lock:
            return ServingMetrics(
                admitted_requests=self._admitted_requests,
                completed_requests=self._completed_requests,
                rejected_requests=self._rejected_requests,
                decode_steps=self._decode_steps,
                generated_tokens=self._generated_tokens,
                active_sequences=len(self._active),
                waiting_requests=len(self._waiting),
                peak_active_sequences=self._peak_active_sequences,
            )

    def _validate_against_limits(self, request: GenerationRequest) -> None:
        if len(request.prompt_token_ids) > self.limits.max_prompt_tokens:
            self._rejected_requests += 1
            raise ValueError(
                f"prompt of {len(request.prompt_token_ids)} tokens exceeds "
                f"max_prompt_tokens={self.limits.max_prompt_tokens}"
            )
        if request.max_new_tokens > self.limits.max_new_tokens:
            self._rejected_requests += 1
            raise ValueError(
                f"max_new_tokens={request.max_new_tokens} exceeds "
                f"max_new_tokens={self.limits.max_new_tokens}"
            )
        vocabulary_size = self.model.settings.vocabulary_size
        for token_id in request.prompt_token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                self._rejected_requests += 1
                raise TypeError("prompt_token_ids must contain integers")
            if not 0 <= token_id < vocabulary_size or token_id == PAD_TOKEN_ID:
                self._rejected_requests += 1
                raise ValueError(f"prompt token id {token_id} is outside the content vocabulary")

    def _sampler_for(self, policy: SamplingPolicy) -> TokenSampler:
        sampler = self._samplers.get(policy)
        if sampler is None:
            sampler = TokenSampler(policy, self.device, self._generator)
            self._samplers[policy] = sampler
        return sampler

    def _admit(self) -> None:
        with self._lock:
            capacity = self.limits.max_concurrent_sequences - len(self._active)
            if capacity <= 0 or not self._waiting:
                return
            admitted = self._waiting[:capacity]
            del self._waiting[: len(admitted)]
        now = self._clock()
        sequences = []
        for group in self._prefill_groups(admitted):
            sequences.extend(self._prefill_group(group, now))
        with self._lock:
            self._active.extend(sequences)
            self._admitted_requests += len(sequences)
            self._peak_active_sequences = max(self._peak_active_sequences, len(self._active))

    def _prefill_groups(
        self, admitted: Sequence[tuple[GenerationRequest, TokenStream, float]]
    ) -> list[list[tuple[GenerationRequest, TokenStream, float]]]:
        """Split one admission into length buckets, longest bucket first.

        Every group is prefilled inside the same `step`, so nothing waits for a
        partner and time to first token is unchanged. The point is only to stop a
        long prompt from setting the padded width of every short one beside it:
        measured on a lognormal spread of 32 prompts, one group padded 82,64% of
        its tokens and buckets of 16 padded 8,17%.
        """
        bucket_size = self.limits.prefill_bucket_size
        groups: dict[int, list[tuple[GenerationRequest, TokenStream, float]]] = {}
        for entry in admitted:
            key = (len(entry[0].prompt_token_ids) - 1) // bucket_size
            groups.setdefault(key, []).append(entry)
        return [groups[key] for key in sorted(groups, reverse=True)]

    def _prefill_group(
        self,
        group: Sequence[tuple[GenerationRequest, TokenStream, float]],
        now: float,
    ) -> list[_Sequence]:
        width = max(len(request.prompt_token_ids) for request, _, _ in group)
        padded = torch.full((len(group), width), PAD_TOKEN_ID, dtype=torch.long, device=self.device)
        for row, (request, _, _) in enumerate(group):
            prompt = request.prompt_token_ids
            padded[row, width - len(prompt) :] = torch.tensor(
                prompt, dtype=torch.long, device=self.device
            )
        logits = self.decoder.prefill(padded)
        settings = self.model.settings
        batch_state = to_static_state(
            self.decoder.state, settings.local_memory_size, settings.salience_memory_size
        )
        tokens = self._sample_rows([request.policy for request, _, _ in group], logits)
        return [
            _Sequence(
                request=request,
                stream=stream,
                state=select_state_rows(batch_state, (row,), step_index=len(request.prompt_token_ids)),
                next_token=tokens[row],
                generated=[],
                submitted_at=submitted_at,
                started_at=now,
            )
            for row, (request, stream, submitted_at) in enumerate(group)
        ]

    def _sample_rows(self, policies: Sequence[SamplingPolicy], logits: Tensor) -> list[Tensor]:
        """Sample one token per row, one sampler call per distinct policy."""
        rows_by_policy: dict[SamplingPolicy, list[int]] = {}
        for row, policy in enumerate(policies):
            rows_by_policy.setdefault(policy, []).append(row)
        sampled: list[Tensor | None] = [None] * len(policies)
        for policy, rows in rows_by_policy.items():
            index = torch.tensor(rows, dtype=torch.long, device=logits.device)
            drawn = self._sampler_for(policy)(logits.index_select(0, index))
            for position, row in enumerate(rows):
                sampled[row] = drawn[position : position + 1]
        return [token for token in sampled if token is not None]

    def _decode_once(self) -> None:
        with self._lock:
            active = list(self._active)
        if not active:
            return
        now = self._clock()
        pending = torch.cat([sequence.next_token for sequence in active], dim=0)
        token_ids = pending.reshape(-1).tolist()
        for sequence, token_id in zip(active, token_ids, strict=True):
            sequence.generated.append(int(token_id))
            self._generated_tokens += 1
            if sequence.first_token_at is None:
                sequence.first_token_at = now
            sequence.stream._emit(int(token_id))
        self._decode_steps += 1
        surviving = [sequence for sequence in active if sequence.terminal_reason(now) is None]
        if not surviving:
            return
        rows = [row for row, sequence in enumerate(active) if sequence.terminal_reason(now) is None]
        self.decoder.adopt(stack_states([sequence.state for sequence in surviving]))
        index = torch.tensor(rows, dtype=torch.long, device=pending.device)
        logits = self.decoder.step(pending.index_select(0, index))
        next_state = self.decoder.state
        sampled = self._sample_rows([sequence.request.policy for sequence in surviving], logits)
        for row, sequence in enumerate(surviving):
            sequence.state = select_state_rows(
                next_state,
                (row,),
                step_index=len(sequence.request.prompt_token_ids) + len(sequence.generated),
            )
            sequence.next_token = sampled[row]
            sequence.decode_steps += 1

    def _retire(self) -> None:
        now = self._clock()
        with self._lock:
            finished = [sequence for sequence in self._active if sequence.terminal_reason(now) is not None]
            if not finished:
                return
            finished_ids = {sequence.request_id for sequence in finished}
            self._active = [
                sequence for sequence in self._active if sequence.request_id not in finished_ids
            ]
            self._completed_requests += len(finished)
        for sequence in finished:
            reason = sequence.terminal_reason(now) or COMPLETED
            sequence.stream._close(
                GenerationOutcome(
                    request_id=sequence.request_id,
                    token_ids=tuple(sequence.generated),
                    reason=reason,
                    queue_wait_seconds=sequence.started_at - sequence.submitted_at,
                    time_to_first_token_seconds=(
                        None
                        if sequence.first_token_at is None
                        else sequence.first_token_at - sequence.submitted_at
                    ),
                    latency_seconds=now - sequence.submitted_at,
                    decode_steps=sequence.decode_steps,
                )
            )


__all__ = [
    "CANCELLED",
    "COMPLETED",
    "DEADLINE_EXCEEDED",
    "DEFAULT_MAX_CONCURRENT_SEQUENCES",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MAX_PROMPT_TOKENS",
    "DEFAULT_PREFILL_BUCKET_SIZE",
    "GenerationOutcome",
    "GenerationRequest",
    "STOPPED",
    "TOKEN_LIMIT",
    "ServingEngine",
    "ServingLimits",
    "ServingMetrics",
    "TokenStream",
]
