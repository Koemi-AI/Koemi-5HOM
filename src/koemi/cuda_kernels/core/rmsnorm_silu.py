from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from koemi.cuda_kernels.loader import (
    CudaExtensionValidationError,
    load_cuda_extension,
    register_cuda_extension,
)


__all__ = ["fused_rmsnorm_silu", "load_core_extension"]


_SUPPORTED_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
_CORE_EXTENSION_SPEC = register_cuda_extension(
    "core",
    "rmsnorm_silu",
    (
        Path(__file__).with_name("rmsnorm_silu.cpp"),
        Path(__file__).with_name("rmsnorm_silu.cu"),
    ),
)


def load_core_extension(*, device: torch.device | str | int | None = None, verbose: bool = False) -> Any:
    """Compile or load the namespaced native RMSNorm+SiLU extension."""

    return load_cuda_extension(_CORE_EXTENSION_SPEC, device=device, verbose=verbose)


def fused_rmsnorm_silu(input_tensor: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Apply RMSNorm and SiLU in one CUDA forward with an autograd backward."""

    _validate_inputs(input_tensor, weight, eps)
    extension = load_core_extension(device=input_tensor.device)
    return _FusedRMSNormSiLU.apply(input_tensor, weight, float(eps), extension)


class _FusedRMSNormSiLU(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        input_tensor: Tensor,
        weight: Tensor,
        eps: float,
        extension: Any,
    ) -> Tensor:
        output, inverse_rms = extension.rmsnorm_silu_forward(input_tensor, weight, eps)
        context.save_for_backward(input_tensor, weight, inverse_rms)
        context.eps = eps
        context.extension = extension
        return output

    @staticmethod
    def backward(context: Any, gradient_output: Tensor) -> tuple[Tensor | None, Tensor | None, None, None]:
        input_tensor, weight, inverse_rms = context.saved_tensors
        contiguous_gradient = gradient_output.contiguous()
        gradient_input, gradient_weight = context.extension.rmsnorm_silu_backward(
            contiguous_gradient,
            input_tensor,
            weight,
            inverse_rms,
            context.eps,
        )
        return gradient_input, gradient_weight, None, None


def _validate_inputs(input_tensor: Tensor, weight: Tensor, eps: float) -> None:
    if not isinstance(input_tensor, Tensor):
        raise TypeError("input_tensor must be a torch.Tensor")
    if not isinstance(weight, Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if input_tensor.ndim != 2:
        raise CudaExtensionValidationError("input_tensor must have shape [rows, hidden]")
    if input_tensor.shape[1] == 0:
        raise CudaExtensionValidationError("input_tensor hidden dimension must be positive")
    if weight.ndim != 1 or weight.shape[0] != input_tensor.shape[1]:
        raise CudaExtensionValidationError(
            f"weight must have shape {(input_tensor.shape[1],)}, got {tuple(weight.shape)}"
        )
    if input_tensor.dtype not in _SUPPORTED_DTYPES or weight.dtype not in _SUPPORTED_DTYPES:
        raise CudaExtensionValidationError("input_tensor and weight must use float16, bfloat16 or float32")
    if input_tensor.dtype != weight.dtype:
        raise CudaExtensionValidationError("input_tensor and weight must have the same dtype")
    if input_tensor.device != weight.device:
        raise CudaExtensionValidationError("input_tensor and weight must use the same device")
    if input_tensor.device.type != "cuda":
        raise CudaExtensionValidationError(f"input_tensor must be on CUDA, got {input_tensor.device}")
    if not input_tensor.is_contiguous() or not weight.is_contiguous():
        raise CudaExtensionValidationError("input_tensor and weight must be contiguous")
    if isinstance(eps, bool) or not isinstance(eps, (float, int)) or not math.isfinite(float(eps)):
        raise CudaExtensionValidationError("eps must be a finite positive number")
    if float(eps) <= 0.0:
        raise CudaExtensionValidationError("eps must be a finite positive number")
