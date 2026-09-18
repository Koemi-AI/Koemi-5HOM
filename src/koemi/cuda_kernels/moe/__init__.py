from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class CudaMoeUnavailableError(RuntimeError):
    """Raised when the isolated MoE CUDA extension cannot be built or loaded."""


class CudaMoeValidationError(ValueError):
    """Raised when an isolated MoE CUDA operation receives an unsafe contract."""


_EXTENSION_LOCK = threading.Lock()
_EXTENSION: Any | None = None


def load_native_extension() -> Any:
    """Build and load the native MoE extension on a CUDA host.

    Returns the loaded PyTorch extension. The build is cached by PyTorch's
    extension cache and is performed only on the first call in this process.
    """

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise CudaMoeUnavailableError(
            "Koemi MoE CUDA kernels require torch.cuda.is_available() == True"
        )
    try:
        from torch.utils.cpp_extension import CUDA_HOME, load
    except Exception as error:
        raise CudaMoeUnavailableError(
            "PyTorch C++/CUDA extension support is unavailable"
        ) from error
    if CUDA_HOME is None:
        raise CudaMoeUnavailableError(
            "Koemi MoE CUDA kernels require a CUDA toolkit with nvcc"
        )

    source_directory = Path(__file__).resolve().parent
    sources = [
        str(source_directory / "moe_kernels.cpp"),
        str(source_directory / "moe_kernels.cu"),
    ]
    extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]
    with _EXTENSION_LOCK:
        if _EXTENSION is None:
            try:
                _EXTENSION = load(
                    name="koemi_moe_cuda_v1",
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                    with_cuda=True,
                    verbose=os.environ.get("KOEMI_MOE_VERBOSE_BUILD") == "1",
                )
            except Exception as error:
                raise CudaMoeUnavailableError(
                    "Koemi MoE CUDA extension failed to compile"
                ) from error
    return _EXTENSION


def _require_cuda_tensor(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise CudaMoeValidationError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise CudaMoeValidationError(f"{name} must be contiguous")


def _require_same_cuda_device(tensors: dict[str, Tensor]) -> None:
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        rendered = ", ".join(sorted(str(device) for device in devices))
        raise CudaMoeValidationError(f"all MoE tensors must use one CUDA device, found {rendered}")


def _require_supported_context_dtype(tensor: Tensor, name: str) -> None:
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise CudaMoeValidationError(
            f"{name} must use float32, float16, or bfloat16; found {tensor.dtype}"
        )


def _require_positive_bounded(value: int, name: str, upper_bound: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CudaMoeValidationError(f"{name} must be a positive integer")
    if value > upper_bound:
        raise CudaMoeValidationError(f"{name} exceeds the CUDA MoE limit of {upper_bound}")
    return value


def _require_inference_only(tensors: dict[str, Tensor]) -> None:
    gradient_tensors = [name for name, tensor in tensors.items() if tensor.requires_grad]
    if gradient_tensors:
        names = ", ".join(gradient_tensors)
        raise RuntimeError(
            "the native MoE path is inference-only; tensors require gradients: " + names
        )


def _require_no_duplicate_valid_assignments(
    assignments: Tensor,
    valid_mask: Tensor,
    *,
    error_message: str,
) -> None:
    top_k = assignments.shape[-1]
    if top_k < 2:
        return
    repeated = assignments.unsqueeze(-1).eq(assignments.unsqueeze(-2))
    repeated = torch.triu(repeated, diagonal=1).any(dim=-1)
    if bool((repeated & valid_mask.unsqueeze(-1)).any().item()):
        raise CudaMoeValidationError(error_message)


def route_topk_valid(
    token_ids: Tensor,
    previous_token_ids: Tensor,
    valid_mask: Tensor,
    *,
    expert_count: int,
    top_k: int,
) -> Tensor:
    """Route tokens with HERM's content hash and reject duplicate valid slots.

    Parameters are contiguous CUDA tensors with equal shape. The return shape is
    ``token_ids.shape + (top_k,)`` and invalid rows contain ``-1``.
    """

    tensors = {
        "token_ids": token_ids,
        "previous_token_ids": previous_token_ids,
        "valid_mask": valid_mask,
    }
    for name, tensor in tensors.items():
        _require_cuda_tensor(tensor, name)
    _require_same_cuda_device(tensors)
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise CudaMoeValidationError("token_ids must use int32 or int64")
    if previous_token_ids.dtype != token_ids.dtype:
        raise CudaMoeValidationError("token_ids and previous_token_ids must use one dtype")
    if valid_mask.dtype != torch.bool:
        raise CudaMoeValidationError("valid_mask must use boolean dtype")
    if token_ids.shape != previous_token_ids.shape or token_ids.shape != valid_mask.shape:
        raise CudaMoeValidationError("routing tensors must have equal shapes")
    expert_count = _require_positive_bounded(expert_count, "expert_count", 1 << 20)
    top_k = _require_positive_bounded(top_k, "top_k", 256)
    if top_k > expert_count:
        raise CudaMoeValidationError("top_k must not exceed expert_count")

    extension = load_native_extension()
    assignments = extension.route_topk(
        token_ids,
        previous_token_ids,
        valid_mask,
        expert_count,
        top_k,
    ).reshape(*token_ids.shape, top_k)
    _require_no_duplicate_valid_assignments(
        assignments,
        valid_mask,
        error_message="MoE routing produced duplicate experts for a valid token",
    )
    return assignments


def permute_topk(
    context: Tensor,
    assignments: Tensor,
    valid_mask: Tensor,
    *,
    expert_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sort valid token-expert pairs into stable expert segments.

    Returns sorted context, source token rows, sorted expert ids, and an
    exclusive segment-offset tensor of length ``expert_count + 1``. Invalid
    pairs are placed after all expert segments.
    """

    tensors = {"context": context, "assignments": assignments, "valid_mask": valid_mask}
    for name, tensor in tensors.items():
        _require_cuda_tensor(tensor, name)
    _require_same_cuda_device(tensors)
    _require_supported_context_dtype(context, "context")
    if context.ndim < 2 or context.shape[-1] < 1:
        raise CudaMoeValidationError("context must have at least two dimensions and a non-empty width")
    if assignments.ndim != 2 or assignments.shape[0] != context.numel() // context.shape[-1]:
        raise CudaMoeValidationError("assignments must have shape [token_count, top_k]")
    if assignments.dtype not in (torch.int32, torch.int64):
        raise CudaMoeValidationError("assignments must use int32 or int64")
    if valid_mask.ndim != 1 or valid_mask.shape[0] != assignments.shape[0]:
        raise CudaMoeValidationError("valid_mask must have shape [token_count]")
    expert_count = _require_positive_bounded(expert_count, "expert_count", 1 << 20)
    top_k = _require_positive_bounded(int(assignments.shape[1]), "top_k", 256)
    if top_k > expert_count:
        raise CudaMoeValidationError("top_k must not exceed expert_count")

    flattened_context = context.reshape(-1, context.shape[-1])
    valid_assignments = valid_mask.unsqueeze(-1)
    out_of_range = ((assignments < 0) | (assignments >= expert_count)) & valid_assignments
    if bool(out_of_range.any().item()):
        raise CudaMoeValidationError("valid MoE assignments must be in [0, expert_count)")
    _require_no_duplicate_valid_assignments(
        assignments,
        valid_mask,
        error_message="MoE permutation received duplicate experts for a valid token",
    )
    return tuple(
        load_native_extension().permute_topk(
            flattened_context,
            assignments,
            valid_mask,
            expert_count,
        )
    )


def grouped_expert_mlp_inference(
    sorted_context: Tensor,
    sorted_experts: Tensor,
    segment_offsets: Tensor,
    gate_weights: Tensor,
    gate_biases: Tensor,
    value_weights: Tensor,
    value_biases: Tensor,
    output_weights: Tensor,
    output_biases: Tensor,
) -> Tensor:
    """Run the fused grouped gated MLP for sorted MoE pairs.

    The operation is inference-only. Weights use shapes ``[E, H, D]``,
    ``[E, H]``, ``[E, H, D]``, ``[E, H]``, ``[E, D, H]`` and ``[E, D]``.
    """

    tensors = {
        "sorted_context": sorted_context,
        "sorted_experts": sorted_experts,
        "segment_offsets": segment_offsets,
        "gate_weights": gate_weights,
        "gate_biases": gate_biases,
        "value_weights": value_weights,
        "value_biases": value_biases,
        "output_weights": output_weights,
        "output_biases": output_biases,
    }
    for name, tensor in tensors.items():
        _require_cuda_tensor(tensor, name)
    _require_same_cuda_device(tensors)
    _require_supported_context_dtype(sorted_context, "sorted_context")
    _require_inference_only(tensors)
    if sorted_context.ndim != 2:
        raise CudaMoeValidationError("sorted_context must have shape [pair_count, width]")
    if sorted_experts.ndim != 1 or sorted_experts.shape[0] != sorted_context.shape[0]:
        raise CudaMoeValidationError("sorted_experts must have shape [pair_count]")
    if sorted_experts.dtype != torch.int64:
        raise CudaMoeValidationError("sorted_experts must use int64")
    if segment_offsets.dtype != torch.int64 or segment_offsets.ndim != 1:
        raise CudaMoeValidationError("segment_offsets must be a one-dimensional int64 tensor")
    if gate_weights.ndim != 3 or value_weights.ndim != 3 or output_weights.ndim != 3:
        raise CudaMoeValidationError("expert weights must be three-dimensional")
    expert_count, hidden_width, input_width = gate_weights.shape
    if input_width != sorted_context.shape[1]:
        raise CudaMoeValidationError("gate weight input width does not match sorted_context")
    if value_weights.shape != (expert_count, hidden_width, input_width):
        raise CudaMoeValidationError("value weights do not match gate weight shapes")
    if output_weights.shape[0] != expert_count or output_weights.shape[1] != input_width:
        raise CudaMoeValidationError("output weights do not match expert and output widths")
    if output_weights.shape[2] != hidden_width:
        raise CudaMoeValidationError("output weight hidden width does not match gate weights")
    if gate_biases.shape != (expert_count, hidden_width):
        raise CudaMoeValidationError("gate biases do not match gate weights")
    if value_biases.shape != (expert_count, hidden_width):
        raise CudaMoeValidationError("value biases do not match value weights")
    if output_biases.shape != (expert_count, input_width):
        raise CudaMoeValidationError("output biases do not match output weights")
    if segment_offsets.shape[0] != expert_count + 1:
        raise CudaMoeValidationError("segment_offsets must have expert_count + 1 entries")
    if any(tensor.dtype != sorted_context.dtype for tensor in tensors.values() if tensor is not sorted_experts and tensor is not segment_offsets):
        raise CudaMoeValidationError("all grouped MLP floating tensors must use one dtype")
    valid_pair_count = int(segment_offsets[-1].item())
    if valid_pair_count < 0 or valid_pair_count > sorted_context.shape[0]:
        raise CudaMoeValidationError("segment_offsets final value is outside pair_count")
    if valid_pair_count:
        selected_experts = sorted_experts[:valid_pair_count]
        if bool((selected_experts < 0).any().item()) or bool(
            (selected_experts >= expert_count).any().item()
        ):
            raise CudaMoeValidationError("sorted expert ids are outside the expert catalog")
    return load_native_extension().grouped_expert_mlp(
        sorted_context,
        sorted_experts,
        segment_offsets,
        gate_weights,
        gate_biases,
        value_weights,
        value_biases,
        output_weights,
        output_biases,
    )


def combine_topk(
    expert_outputs: Tensor,
    sorted_rows: Tensor,
    segment_offsets: Tensor,
    *,
    token_count: int,
    top_k: int,
) -> Tensor:
    """Combine sorted expert outputs into token rows using ``1 / top_k``."""

    tensors = {
        "expert_outputs": expert_outputs,
        "sorted_rows": sorted_rows,
        "segment_offsets": segment_offsets,
    }
    for name, tensor in tensors.items():
        _require_cuda_tensor(tensor, name)
    _require_same_cuda_device(tensors)
    _require_supported_context_dtype(expert_outputs, "expert_outputs")
    _require_inference_only(tensors)
    if expert_outputs.ndim != 2:
        raise CudaMoeValidationError("expert_outputs must have shape [pair_count, width]")
    if sorted_rows.dtype != torch.int64 or sorted_rows.ndim != 1:
        raise CudaMoeValidationError("sorted_rows must be a one-dimensional int64 tensor")
    if sorted_rows.shape[0] != expert_outputs.shape[0]:
        raise CudaMoeValidationError("sorted_rows must match expert_outputs pair_count")
    if segment_offsets.dtype != torch.int64 or segment_offsets.ndim != 1:
        raise CudaMoeValidationError("segment_offsets must be a one-dimensional int64 tensor")
    token_count = _require_positive_bounded(token_count, "token_count", 1 << 31)
    top_k = _require_positive_bounded(top_k, "top_k", 256)
    valid_pair_count = int(segment_offsets[-1].item())
    if valid_pair_count < 0 or valid_pair_count > expert_outputs.shape[0]:
        raise CudaMoeValidationError("segment_offsets final value is outside pair_count")
    if valid_pair_count:
        selected_rows = sorted_rows[:valid_pair_count]
        if bool((selected_rows < 0).any().item()) or bool(
            (selected_rows >= token_count).any().item()
        ):
            raise CudaMoeValidationError("sorted rows are outside token_count")
    return load_native_extension().combine_topk(
        expert_outputs,
        sorted_rows,
        segment_offsets,
        token_count,
        top_k,
    )


__all__ = [
    "CudaMoeUnavailableError",
    "CudaMoeValidationError",
    "combine_topk",
    "grouped_expert_mlp_inference",
    "load_native_extension",
    "permute_topk",
    "route_topk_valid",
]
