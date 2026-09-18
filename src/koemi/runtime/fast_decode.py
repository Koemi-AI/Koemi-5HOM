from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from koemi.configuration.settings import PAD_TOKEN_ID
from koemi.data.tokenizer import TextTokenizer
from koemi.model.network import KoemiModel
from koemi.model.state import KOEMI_STATE_FIELDS, KoemiState


GRAPH_WARMUP_STEPS = 3


@dataclass(frozen=True)
class SamplingPolicy:
    """Token selection rule applied to one decode step.

    `temperature` scales the logits and must be positive. `top_k` keeps only the
    highest scoring ids before sampling. `greedy` takes the argmax and ignores
    both. The padding id is always removed before selection.
    """

    temperature: float = 1.0
    top_k: int | None = None
    greedy: bool = False

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None:
            if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
                raise ValueError("top_k must be a positive integer")


class TokenSampler:
    """Select decode tokens without reading a tensor value back to the host."""

    def __init__(
        self,
        policy: SamplingPolicy,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        self.policy = policy
        self.generator = generator
        self._padding_index = torch.tensor([PAD_TOKEN_ID], dtype=torch.long, device=device)

    def content_scores(self, logits: Tensor) -> Tensor:
        """Return float scores with the padding id masked out."""
        return logits.float().index_fill(-1, self._padding_index, float("-inf"))

    def probabilities(self, logits: Tensor) -> Tensor:
        """Return the distribution this policy draws from, for `[batch, vocabulary]` logits."""
        scores = self.content_scores(logits)
        if self.policy.greedy:
            selected = scores.argmax(dim=-1, keepdim=True)
            return torch.zeros_like(scores).scatter_(-1, selected, 1.0)
        return torch.softmax(self._restrict(scores) / self.policy.temperature, dim=-1)

    def sample_with_probabilities(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        """Return `[batch, 1]` sampled ids together with the distribution they came from."""
        distribution = self.probabilities(logits)
        if self.policy.greedy:
            return distribution.argmax(dim=-1, keepdim=True), distribution
        return torch.multinomial(distribution, num_samples=1, generator=self.generator), distribution

    def __call__(self, logits: Tensor) -> Tensor:
        """Return `[batch, 1]` sampled ids for `[batch, vocabulary]` logits."""
        if logits.ndim != 2:
            raise ValueError("sampling requires logits with shape [batch, vocabulary]")
        scores = self.content_scores(logits)
        if self.policy.greedy:
            return scores.argmax(dim=-1, keepdim=True)
        probabilities = torch.softmax(self._restrict(scores) / self.policy.temperature, dim=-1)
        return torch.multinomial(probabilities, num_samples=1, generator=self.generator)

    def _restrict(self, scores: Tensor) -> Tensor:
        if self.policy.top_k is None:
            return scores
        kept = min(self.policy.top_k, scores.shape[-1])
        threshold = torch.topk(scores, kept, dim=-1).values[..., -1:]
        return scores.masked_fill(scores < threshold, float("-inf"))


def to_static_state(state: KoemiState, local_memory_size: int, salience_memory_size: int) -> KoemiState:
    """Return the same state with both ring buffers held at their capacity.

    Empty slots stay zero and invalid, so every read produces the value the
    growing state would produce. Constant shapes are what CUDA graph capture and
    allocation-free decoding require.
    """
    if local_memory_size < 1 or salience_memory_size < 1:
        raise ValueError("static state requires positive ring capacities")
    local_keys, local_values, local_valid = _resize_ring(
        state.local_keys, state.local_values, state.local_valid, local_memory_size
    )
    salient_keys, salient_values, salient_valid = _resize_ring(
        state.salient_keys, state.salient_values, state.salient_valid, salience_memory_size
    )
    return replace(
        state,
        local_keys=local_keys,
        local_values=local_values,
        local_valid=local_valid,
        salient_keys=salient_keys,
        salient_values=salient_values,
        salient_valid=salient_valid,
    )


def clone_state(state: KoemiState) -> KoemiState:
    """Return a state whose tensors are private to the caller."""
    cloned = {field_name: getattr(state, field_name).clone() for field_name in KOEMI_STATE_FIELDS}
    return replace(state, **cloned)


def select_state_rows(state: KoemiState, rows: Sequence[int], step_index: int | None = None) -> KoemiState:
    """Return the state restricted to `rows`, preserving field order."""
    index = torch.tensor(tuple(rows), dtype=torch.long, device=state.working_state.device)
    selected = {
        field_name: getattr(state, field_name).index_select(0, index)
        for field_name in KOEMI_STATE_FIELDS
    }
    return replace(
        state,
        **selected,
        step_index=state.step_index if step_index is None else step_index,
    )


def stack_states(states: Sequence[KoemiState]) -> KoemiState:
    """Join per-sequence states into one batched state, preserving row order.

    Every state must already sit at ring capacity, which `to_static_state`
    guarantees, so the rings concatenate without reshaping. `step_index` is
    carried as the largest of the inputs: no computation in the model reads it,
    so rows that sit at different positions still decode exactly.
    """
    if not states:
        raise ValueError("stack_states requires at least one state")
    joined = {}
    for field_name in KOEMI_STATE_FIELDS:
        tensors = [getattr(state, field_name) for state in states]
        shapes = {tuple(tensor.shape[1:]) for tensor in tensors}
        if len(shapes) != 1:
            raise ValueError(f"states disagree on the shape of {field_name}; pad rings to capacity first")
        joined[field_name] = torch.cat(tensors, dim=0)
    return replace(states[0], **joined, step_index=max(state.step_index for state in states))


class BatchDecoder:
    """Fixed-shape recurrent decoding for a batch of sequences.

    The decoder keeps the recurrent state at ring capacity so every step runs the
    same shapes, marks the decode inputs trusted so the model skips its host-side
    validation, and never moves a tensor value to the host. `capture_graph`
    replays one captured CUDA graph per step; it requires CUDA, a model without
    forward hooks and no warm token cache, and it fails at construction when CUDA
    is absent instead of silently running the eager path. Capture turns the
    autocast weight cache off, because a cast recorded once and replayed many
    times is the documented hazard of combining the two; that combination has not
    been executed on a CUDA device from this repository.
    """

    def __init__(
        self,
        model: KoemiModel,
        *,
        capture_graph: bool = False,
        autocast_dtype: torch.dtype | None = None,
        cache_expert_weights: bool = True,
    ) -> None:
        if model.training:
            raise RuntimeError("the decoder requires a model in evaluation mode")
        device = next(model.parameters()).device
        if capture_graph and device.type != "cuda":
            raise RuntimeError("CUDA graph capture requires a model on a CUDA device")
        self.model = model
        self.device = device
        self._capture_graph = capture_graph
        self._autocast_dtype = autocast_dtype
        self._graph: torch.cuda.CUDAGraph | None = None
        self._state: KoemiState | None = None
        self._static_input_ids: Tensor | None = None
        self._static_logits: Tensor | None = None
        self._step_index = 0
        if cache_expert_weights and model.settings.expert_count > 0:
            with torch.no_grad():
                model.cache_inference_weights()

    @property
    def state(self) -> KoemiState:
        """Return the current recurrent state carrying the real step index."""
        if self._state is None:
            raise RuntimeError("prefill must run before the state is available")
        return replace(self._state, step_index=self._step_index)

    def adopt(self, state: KoemiState, step_index: int | None = None) -> None:
        """Continue from an existing state instead of prefilling a prompt."""
        settings = self.model.settings
        self._state = to_static_state(state, settings.local_memory_size, settings.salience_memory_size)
        self._step_index = state.step_index if step_index is None else step_index
        self._reset_graph()

    def prefill(self, input_ids: Tensor) -> Tensor:
        """Evaluate right-aligned padded prompts and return `[batch, vocabulary]` logits."""
        self._validate_prompt(input_ids)
        trusted = not bool((input_ids == PAD_TOKEN_ID).any().item())
        with torch.no_grad(), self._autocast():
            output = self.model(input_ids, trusted_inputs=trusted)
        settings = self.model.settings
        self._state = to_static_state(
            output.state, settings.local_memory_size, settings.salience_memory_size
        )
        self._step_index = input_ids.shape[1]
        self._reset_graph()
        return output.logits[:, -1]

    def step(self, token_ids: Tensor) -> Tensor:
        """Advance one token per row and return `[batch, vocabulary]` logits."""
        if self._state is None:
            raise RuntimeError("prefill must run before step")
        decode_ids = self._normalize_step_input(token_ids)
        if self._capture_graph:
            return self._graph_step(decode_ids)
        with torch.no_grad(), self._autocast():
            output = self.model(decode_ids, self._state, trusted_inputs=True)
        self._state = output.state
        self._step_index += 1
        return output.logits[:, -1]

    def extend(self, token_ids: Tensor) -> Tensor:
        """Advance several tokens per row in one forward and return every logit row.

        The result has shape `[batch, length, vocabulary]`: position `i` holds the
        distribution for the token that follows `token_ids[:, i]`.
        """
        if self._state is None:
            raise RuntimeError("prefill must run before extend")
        if not isinstance(token_ids, Tensor) or token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("extend requires token ids with shape [batch, length]")
        if token_ids.shape[0] != self._state.working_state.shape[0]:
            raise ValueError("extend token ids must carry one row per decoded sequence")
        with torch.no_grad(), self._autocast():
            output = self.model(token_ids, self._state, trusted_inputs=True)
        settings = self.model.settings
        self._state = to_static_state(
            output.state, settings.local_memory_size, settings.salience_memory_size
        )
        self._step_index += token_ids.shape[1]
        self._reset_graph()
        return output.logits

    def _graph_step(self, decode_ids: Tensor) -> Tensor:
        if self._graph is None:
            self._allocate_static_buffers(decode_ids)
            self._capture()
        self._static_input_ids.copy_(decode_ids)
        self._graph.replay()
        self._step_index += 1
        return self._static_logits[:, -1]

    def _allocate_static_buffers(self, decode_ids: Tensor) -> None:
        self._static_input_ids = decode_ids.clone()
        self._state = clone_state(self._state)
        with torch.no_grad(), self._autocast():
            probe = self.model(self._static_input_ids, self._state, trusted_inputs=True)
        self._static_logits = probe.logits.clone()

    def _capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.no_grad(), self._autocast():
                for _ in range(GRAPH_WARMUP_STEPS):
                    self.model(self._static_input_ids, self._state, trusted_inputs=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad(), self._autocast():
                output = self.model(self._static_input_ids, self._state, trusted_inputs=True)
            self._static_logits.copy_(output.logits)
            for field_name in KOEMI_STATE_FIELDS:
                getattr(self._state, field_name).copy_(getattr(output.state, field_name))
        self._graph = graph

    def _reset_graph(self) -> None:
        self._graph = None
        self._static_input_ids = None
        self._static_logits = None

    def _autocast(self):
        if self._autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self._autocast_dtype,
            cache_enabled=not self._capture_graph,
        )

    def _normalize_step_input(self, token_ids: Tensor) -> Tensor:
        if not isinstance(token_ids, Tensor):
            raise TypeError("step token ids must be a torch.Tensor")
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(-1)
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("step requires exactly one token id per row")
        if token_ids.shape[0] != self._state.working_state.shape[0]:
            raise ValueError("step token ids must carry one row per decoded sequence")
        if token_ids.dtype != torch.long:
            raise TypeError("step token ids must use torch.long")
        if token_ids.device != self.device:
            raise ValueError("step token ids must live on the model device")
        return token_ids

    def _validate_prompt(self, input_ids: Tensor) -> None:
        if not isinstance(input_ids, Tensor):
            raise TypeError("prompt ids must be a torch.Tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("prompt ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise TypeError("prompt ids must use torch.long")
        if input_ids.device != self.device:
            raise ValueError("prompt ids must live on the model device")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= self.model.settings.vocabulary_size:
            raise ValueError("prompt ids contain values outside the model vocabulary")
        if not _padding_is_left_aligned(input_ids):
            raise ValueError("prompt padding must be a prefix of every row")


@dataclass(frozen=True)
class BatchGeneration:
    """Result of one batched generation call.

    `token_ids` holds the generated ids per row, already cut at `stop_token_id`.
    `texts` decodes those ids. `generated_tokens` counts what survived the cut,
    which is the denominator a throughput number must use.
    """

    token_ids: tuple[tuple[int, ...], ...]
    texts: tuple[str, ...]
    prompt_tokens: int
    generated_tokens: int


def generate_batch(
    model: KoemiModel,
    tokenizer: TextTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    *,
    device: str | torch.device = "cpu",
    policy: SamplingPolicy | None = None,
    generator: torch.Generator | None = None,
    capture_graph: bool = False,
    autocast_dtype: torch.dtype | None = None,
    stop_token_id: int | None = None,
    prompt_token_ids: Sequence[Sequence[int]] | None = None,
) -> BatchGeneration:
    """Generate continuations for several prompts in one recurrent decode loop.

    Every prompt is right-aligned inside one padded batch, so the recurrent tail
    a row carries is its own suffix. Sampling stays on the device and the only
    host transfer is the single read of the generated ids at the end.
    `prompt_token_ids` overrides the encoding of `prompts`, which is what a
    span-by-span prompt needs to match the tokenization used during training.
    """
    if not prompts:
        raise ValueError("at least one prompt is required")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    resolved_policy = policy or SamplingPolicy()
    resolved_device = torch.device(device)
    if prompt_token_ids is None:
        encoded_prompts = tuple(tuple(tokenizer.encode(prompt)) for prompt in prompts)
    else:
        encoded_prompts = tuple(tuple(int(token_id) for token_id in row) for row in prompt_token_ids)
        if len(encoded_prompts) != len(prompts):
            raise ValueError("prompt_token_ids must carry one row per prompt")
    if any(not prompt_ids for prompt_ids in encoded_prompts):
        raise ValueError("every prompt must encode to at least one token")
    padded_prompts = _right_align(encoded_prompts, resolved_device)
    model.eval()
    decoder = BatchDecoder(model, capture_graph=capture_graph, autocast_dtype=autocast_dtype)
    sampler = TokenSampler(resolved_policy, resolved_device, generator)
    generated = torch.empty(
        (len(encoded_prompts), max_new_tokens), dtype=torch.long, device=resolved_device
    )
    logits = decoder.prefill(padded_prompts)
    for index in range(max_new_tokens):
        next_tokens = sampler(logits)
        generated[:, index : index + 1].copy_(next_tokens)
        if index + 1 == max_new_tokens:
            break
        logits = decoder.step(next_tokens)
    cut_rows = tuple(_cut_at_stop(tuple(row), stop_token_id) for row in generated.tolist())
    return BatchGeneration(
        token_ids=cut_rows,
        texts=tuple(tokenizer.decode(row) for row in cut_rows),
        prompt_tokens=sum(len(prompt_ids) for prompt_ids in encoded_prompts),
        generated_tokens=sum(len(row) for row in cut_rows),
    )


def _cut_at_stop(row: tuple[int, ...], stop_token_id: int | None) -> tuple[int, ...]:
    if stop_token_id is None or stop_token_id not in row:
        return row
    return row[: row.index(stop_token_id)]


def _right_align(encoded_prompts: tuple[tuple[int, ...], ...], device: torch.device) -> Tensor:
    maximum_length = max(len(prompt_ids) for prompt_ids in encoded_prompts)
    padded = torch.full(
        (len(encoded_prompts), maximum_length), PAD_TOKEN_ID, dtype=torch.long, device=device
    )
    for row_index, prompt_ids in enumerate(encoded_prompts):
        start = maximum_length - len(prompt_ids)
        padded[row_index, start:] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    return padded


def _padding_is_left_aligned(input_ids: Tensor) -> bool:
    padding = input_ids == PAD_TOKEN_ID
    if not bool(padding.any()):
        return True
    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    first_content = torch.where(padding, torch.full_like(positions, length), positions).amin(dim=1)
    last_padding = torch.where(padding, positions, torch.full_like(positions, -1)).amax(dim=1)
    return bool((last_padding < first_content).all())


def _resize_ring(
    keys: Tensor, values: Tensor, valid: Tensor, capacity: int
) -> tuple[Tensor, Tensor, Tensor]:
    length = keys.shape[1]
    if length == capacity:
        return keys, values, valid
    if length > capacity:
        return keys[:, -capacity:], values[:, -capacity:], valid[:, -capacity:]
    missing = capacity - length
    key_padding = keys.new_zeros((keys.shape[0], missing, keys.shape[2]))
    value_padding = values.new_zeros((values.shape[0], missing, values.shape[2]))
    valid_padding = torch.zeros((valid.shape[0], missing), dtype=torch.bool, device=valid.device)
    return (
        torch.cat((key_padding, keys), dim=1),
        torch.cat((value_padding, values), dim=1),
        torch.cat((valid_padding, valid), dim=1),
    )


__all__ = [
    "BatchDecoder",
    "BatchGeneration",
    "GRAPH_WARMUP_STEPS",
    "SamplingPolicy",
    "TokenSampler",
    "clone_state",
    "generate_batch",
    "select_state_rows",
    "to_static_state",
]
