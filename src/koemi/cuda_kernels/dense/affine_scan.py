from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.cpp_extension import CUDA_HOME, load


__all__ = [
    "DenseCudaBuildError",
    "DenseCudaUnavailableError",
    "DenseCudaValidationError",
    "dense_affine_scan",
    "native_cuda_available",
]


_SUPPORTED_CUDA_DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_EXTENSION_NAME = "koemi_dense_affine_scan_v1"
_EXTENSION_LOCK = threading.Lock()
_EXTENSION: Any | None = None


class DenseCudaUnavailableError(RuntimeError):
    """Raised when the native dense kernel cannot run in the current process."""


class DenseCudaBuildError(DenseCudaUnavailableError):
    """Raised when the native CUDA extension cannot be compiled or loaded."""


class DenseCudaValidationError(ValueError):
    """Raised when tensors violate the native dense affine-scan contract."""


def native_cuda_available() -> bool:
    """Return whether CUDA tensors and a local CUDA toolkit are available."""

    return bool(torch.cuda.is_available() and torch.version.cuda and CUDA_HOME)


def dense_affine_scan(
    retention: Tensor,
    increment: Tensor,
    initial: Tensor | None = None,
) -> Tensor:
    """Run the opt-in native CUDA affine scan with first-order autograd support.

    `retention` and `increment` have shape `[batch, sequence, ...]`, and
    `retention` may broadcast over state dimensions. `initial` has shape
    `[batch, ...]`; omitted initial state is zero. All tensors must share one
    CUDA device, dtype, and the supported floating-point dtype set. The
    function never falls back to CPU or changes the legacy Python path.
    """

    _validate_inputs(retention, increment, initial)
    if not native_cuda_available():
        raise DenseCudaUnavailableError(
            "native dense CUDA scan requires torch CUDA support, a CUDA device, and nvcc"
        )

    normalized_initial = initial
    if normalized_initial is None:
        normalized_initial = increment.new_zeros(
            (increment.shape[0],) + tuple(increment.shape[2:])
        )

    normalized_increment = increment.contiguous()
    return _DenseAffineScanFunction.apply(
        retention,
        normalized_increment,
        normalized_initial,
    )


class _DenseAffineScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        context: torch.autograd.function.FunctionCtx,
        retention: Tensor,
        increment: Tensor,
        initial: Tensor,
    ) -> Tensor:
        normalized_retention = _normalize_retention(retention, increment)
        extension = _load_extension()
        states = extension.forward(normalized_retention, increment, initial)
        context.save_for_backward(normalized_retention, initial, states)
        context.retention_shape = retention.shape
        return states

    @staticmethod
    def backward(
        context: torch.autograd.function.FunctionCtx,
        gradient_states: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        retention, initial, states = context.saved_tensors
        extension = _load_extension()
        gradient_retention_expanded, gradient_increment, gradient_initial = extension.backward(
            retention,
            initial,
            states,
            gradient_states.contiguous(),
        )
        gradient_retention = _reduce_to_shape(
            gradient_retention_expanded,
            context.retention_shape,
        )
        return gradient_retention, gradient_increment, gradient_initial


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        source_path = Path(__file__).with_name("affine_scan.cu")
        if not source_path.is_file():
            raise DenseCudaBuildError(f"native CUDA source is missing: {source_path}")
        try:
            _EXTENSION = load(
                name=_EXTENSION_NAME,
                sources=[str(source_path)],
                extra_cuda_cflags=["-O3"],
                extra_cflags=[] if os.name == "nt" else ["-O3"],
                verbose=os.environ.get("KOEMI_DENSE_CUDA_BUILD_VERBOSE") == "1",
                with_cuda=True,
            )
        except Exception as error:
            raise DenseCudaBuildError(
                f"failed to build native dense CUDA extension: {error}"
            ) from error
    return _EXTENSION


def _validate_inputs(
    retention: Tensor,
    increment: Tensor,
    initial: Tensor | None,
) -> None:
    _validate_tensor("retention", retention)
    _validate_tensor("increment", increment)
    if initial is not None:
        _validate_tensor("initial", initial)

    if retention.ndim < 2 or increment.ndim < 2:
        raise DenseCudaValidationError(
            "retention and increment must have at least two dimensions"
        )
    if retention.ndim != increment.ndim:
        raise DenseCudaValidationError(
            "retention and increment must have the same rank"
        )
    if retention.shape[:2] != increment.shape[:2]:
        raise DenseCudaValidationError(
            "retention and increment must have the same batch and sequence dimensions"
        )
    try:
        broadcast_shape = torch.broadcast_shapes(retention.shape, increment.shape)
    except RuntimeError as error:
        raise DenseCudaValidationError(
            "retention must broadcast to increment's shape"
        ) from error
    if broadcast_shape != increment.shape:
        raise DenseCudaValidationError(
            "retention must broadcast to increment's shape"
        )

    tensor_values = (retention, increment) if initial is None else (retention, increment, initial)
    tensor_dtypes = {value.dtype for value in tensor_values}
    unsupported_dtypes = tensor_dtypes.difference(_SUPPORTED_CUDA_DTYPES)
    if unsupported_dtypes:
        names = ", ".join(sorted(str(dtype) for dtype in unsupported_dtypes))
        raise DenseCudaValidationError(f"unsupported CUDA dtype(s): {names}")
    if len(tensor_dtypes) != 1:
        raise DenseCudaValidationError(
            "retention, increment and initial must have the same dtype"
        )

    if initial is not None:
        expected_initial_shape = (increment.shape[0],) + tuple(increment.shape[2:])
        if initial.shape != expected_initial_shape:
            raise DenseCudaValidationError(
                f"initial must have shape {expected_initial_shape}, got {tuple(initial.shape)}"
            )

    if not torch.cuda.is_available():
        raise DenseCudaUnavailableError(
            "native dense CUDA scan is unavailable: torch.cuda.is_available() is False"
        )
    scan_tensors = (retention, increment) if initial is None else (retention, increment, initial)
    non_cuda_tensors = [value for value in scan_tensors if value.device.type != "cuda"]
    if non_cuda_tensors:
        devices = ", ".join(str(value.device) for value in non_cuda_tensors)
        raise DenseCudaValidationError(
            f"all dense scan tensors must be on CUDA; found {devices}"
        )
    scan_devices = {value.device for value in scan_tensors}
    if len(scan_devices) != 1:
        devices = ", ".join(sorted(str(device) for device in scan_devices))
        raise DenseCudaValidationError(
            f"all dense scan tensors must use one CUDA device; found {devices}"
        )
    if not native_cuda_available():
        raise DenseCudaUnavailableError(
            "native dense CUDA scan requires a CUDA toolkit with nvcc"
        )


def _validate_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _normalize_retention(retention: Tensor, increment: Tensor) -> Tensor:
    if retention.shape == increment.shape:
        return retention.contiguous()
    if all(dimension == 1 for dimension in retention.shape[2:]):
        return retention.contiguous()
    return retention.expand_as(increment).contiguous()


def _reduce_to_shape(gradient: Tensor, target_shape: torch.Size) -> Tensor:
    if gradient.shape == target_shape:
        return gradient
    if len(target_shape) > gradient.ndim:
        raise DenseCudaValidationError(
            f"native gradient rank {gradient.ndim} is smaller than target rank {len(target_shape)}"
        )
    reduced = gradient
    while reduced.ndim > len(target_shape):
        reduced = reduced.sum(dim=-1)
    for dimension in range(reduced.ndim - 1, 1, -1):
        if target_shape[dimension] == 1 and reduced.shape[dimension] != 1:
            reduced = reduced.sum(dim=dimension, keepdim=True)
        elif target_shape[dimension] != reduced.shape[dimension]:
            raise DenseCudaValidationError(
                f"native gradient dimension {dimension} has shape {reduced.shape[dimension]}, "
                f"expected {target_shape[dimension]}"
            )
    if reduced.shape != target_shape:
        raise DenseCudaValidationError(
            f"native gradient shape {tuple(reduced.shape)} does not match {tuple(target_shape)}"
        )
    return reduced
