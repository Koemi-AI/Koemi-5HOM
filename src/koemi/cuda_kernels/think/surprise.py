from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor


__all__ = [
    "CudaThinkBenchmark",
    "CudaThinkUnavailableError",
    "CudaThinkValidationError",
    "benchmark_causal_surprise",
    "causal_surprise",
]


_SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16, torch.float32)


class CudaThinkUnavailableError(RuntimeError):
    """Raised when the opt-in native think kernel cannot run without CUDA."""


class CudaThinkValidationError(ValueError):
    """Raised when the causal-surprise tensors violate the native kernel contract."""


@dataclass(frozen=True)
class CudaThinkBenchmark:
    """Synchronized native-kernel timing for one causal-surprise shape."""

    elapsed_seconds: float
    latency_milliseconds: float
    positions_per_second: float
    position_count: int
    warmup_iterations: int
    measured_iterations: int


class _CausalSurpriseFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        context: torch.autograd.function.FunctionCtx,
        prior_states: Tensor,
        content_weight: Tensor,
        content_bias: Tensor,
        target_ids: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        extension = _load_extension()
        contiguous_inputs = (
            prior_states.contiguous(),
            content_weight.contiguous(),
            content_bias.contiguous(),
            target_ids.contiguous(),
            valid_mask.contiguous(),
        )
        surprise = extension.forward(*contiguous_inputs)
        context.save_for_backward(*contiguous_inputs)
        return surprise

    @staticmethod
    def backward(
        context: torch.autograd.function.FunctionCtx,
        gradient: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None, None]:
        prior_states, content_weight, content_bias, target_ids, valid_mask = context.saved_tensors
        extension = _load_extension()
        gradient_prior, gradient_weight, gradient_bias = extension.backward(
            prior_states,
            content_weight,
            content_bias,
            target_ids,
            valid_mask,
            gradient.contiguous(),
        )
        return (
            gradient_prior.to(dtype=prior_states.dtype),
            gradient_weight.to(dtype=content_weight.dtype),
            gradient_bias.to(dtype=content_bias.dtype),
            None,
            None,
        )


def causal_surprise(
    prior_states: Tensor,
    content_weight: Tensor,
    content_bias: Tensor,
    target_ids: Tensor,
    valid_mask: Tensor,
    *,
    validate_targets: bool = True,
) -> Tensor:
    """Compute normalized causal surprise with a native CUDA forward/backward.

    Parameters use the Koemi shapes ``[batch, sequence, width]``, ``[content_vocab,
    width]``, ``[content_vocab]``, ``[batch, sequence]`` and ``[batch, sequence]``.
    ``prior_states[:, position]`` must already be the state before that position;
    the kernel never reads another sequence position. Invalid positions return zero
    and produce no gradient. ``validate_targets=False`` is an explicit trusted-input
    mode that skips a host-side range synchronization for already checked ids.

    Raises ``CudaThinkValidationError`` for an invalid contract and
    ``CudaThinkUnavailableError`` when CUDA is unavailable.
    """
    _validate_inputs(
        prior_states,
        content_weight,
        content_bias,
        target_ids,
        valid_mask,
        validate_targets,
    )
    if prior_states.shape[1] == 0:
        return prior_states.sum(dim=-1)[:, :0]
    return _CausalSurpriseFunction.apply(
        prior_states,
        content_weight,
        content_bias,
        target_ids,
        valid_mask,
    )


def benchmark_causal_surprise(
    prior_states: Tensor,
    content_weight: Tensor,
    content_bias: Tensor,
    target_ids: Tensor,
    valid_mask: Tensor,
    *,
    warmup_iterations: int = 10,
    measured_iterations: int = 50,
    validate_targets: bool = True,
) -> CudaThinkBenchmark:
    """Measure synchronized native forward latency and positions per second."""
    if isinstance(warmup_iterations, bool) or not isinstance(warmup_iterations, int) or warmup_iterations < 0:
        raise CudaThinkValidationError("warmup_iterations must be a non-negative integer")
    if isinstance(measured_iterations, bool) or not isinstance(measured_iterations, int) or measured_iterations <= 0:
        raise CudaThinkValidationError("measured_iterations must be a positive integer")
    _validate_inputs(
        prior_states,
        content_weight,
        content_bias,
        target_ids,
        valid_mask,
        validate_targets,
    )
    with torch.no_grad():
        for _ in range(warmup_iterations):
            causal_surprise(
                prior_states,
                content_weight,
                content_bias,
                target_ids,
                valid_mask,
                validate_targets=False,
            )
        torch.cuda.synchronize(prior_states.device)
        started_at = perf_counter()
        for _ in range(measured_iterations):
            causal_surprise(
                prior_states,
                content_weight,
                content_bias,
                target_ids,
                valid_mask,
                validate_targets=False,
            )
        torch.cuda.synchronize(prior_states.device)
        elapsed_seconds = perf_counter() - started_at
    position_count = prior_states.shape[0] * prior_states.shape[1]
    average_seconds = elapsed_seconds / measured_iterations
    return CudaThinkBenchmark(
        elapsed_seconds=elapsed_seconds,
        latency_milliseconds=average_seconds * 1000.0,
        positions_per_second=position_count / average_seconds if average_seconds > 0.0 else math.inf,
        position_count=position_count,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )


def _validate_inputs(
    prior_states: Tensor,
    content_weight: Tensor,
    content_bias: Tensor,
    target_ids: Tensor,
    valid_mask: Tensor,
    validate_targets: bool,
) -> int:
    tensors = {
        "prior_states": prior_states,
        "content_weight": content_weight,
        "content_bias": content_bias,
        "target_ids": target_ids,
        "valid_mask": valid_mask,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise CudaThinkValidationError(f"{name} must be a torch.Tensor")
    if prior_states.ndim != 3:
        raise CudaThinkValidationError("prior_states must have shape [batch, sequence, width]")
    if content_weight.ndim != 2:
        raise CudaThinkValidationError("content_weight must have shape [content_vocab, width]")
    if content_bias.ndim != 1:
        raise CudaThinkValidationError("content_bias must have shape [content_vocab]")
    expected_sequence_shape = prior_states.shape[:2]
    if target_ids.shape != expected_sequence_shape or valid_mask.shape != expected_sequence_shape:
        raise CudaThinkValidationError("target_ids and valid_mask must match [batch, sequence]")
    batch_size, _, width = prior_states.shape
    content_vocab = content_weight.shape[0]
    if content_weight.shape[1] != width or content_bias.shape[0] != content_vocab:
        raise CudaThinkValidationError("content head dimensions must match prior_states width")
    if batch_size < 1 or content_vocab < 2 or width < 1:
        raise CudaThinkValidationError("batch, content_vocab and width must be positive")
    if width > 4096:
        raise CudaThinkValidationError("width must be at most 4096 for the shared-memory backward kernel")
    if prior_states.dtype not in _SUPPORTED_DTYPES:
        raise CudaThinkValidationError("prior_states dtype must be float16, bfloat16 or float32")
    if content_weight.dtype != prior_states.dtype or content_bias.dtype != prior_states.dtype:
        raise CudaThinkValidationError("prior_states, content_weight and content_bias must share a dtype")
    if target_ids.dtype != torch.long:
        raise CudaThinkValidationError("target_ids must use torch.int64")
    if valid_mask.dtype != torch.bool:
        raise CudaThinkValidationError("valid_mask must use torch.bool")
    if isinstance(validate_targets, bool) is False:
        raise CudaThinkValidationError("validate_targets must be a boolean")
    if not torch.cuda.is_available():
        raise CudaThinkUnavailableError("native think CUDA kernel is unavailable: torch.cuda.is_available() is False")
    device_tensors = tuple(tensors.values())
    non_cuda = [str(tensor.device) for tensor in device_tensors if tensor.device.type != "cuda"]
    if non_cuda:
        raise CudaThinkValidationError(f"all think tensors must be on CUDA; found {', '.join(non_cuda)}")
    devices = {tensor.device for tensor in device_tensors}
    if len(devices) != 1:
        raise CudaThinkValidationError(f"all think tensors must use one CUDA device; found {', '.join(sorted(map(str, devices)))}")
    if validate_targets and target_ids.numel() > 0:
        minimum_target = int(target_ids.amin().item())
        maximum_target = int(target_ids.amax().item())
        if minimum_target < 0 or maximum_target >= content_vocab:
            raise CudaThinkValidationError(
                f"target_ids must be in [0, {content_vocab}); found [{minimum_target}, {maximum_target}]"
            )
    return content_vocab


@lru_cache(maxsize=1)
def _load_extension():
    from torch.utils.cpp_extension import load

    source_directory = Path(__file__).resolve().parent
    return load(
        name="koemi_think_surprise_cuda_v1",
        sources=[
            str(source_directory / "surprise_bindings.cpp"),
            str(source_directory / "surprise_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
