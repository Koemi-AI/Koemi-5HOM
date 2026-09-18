from __future__ import annotations

import unittest

import torch

from koemi.cuda_kernels.dense import (
    DenseCudaBuildError,
    DenseCudaUnavailableError,
    DenseCudaValidationError,
    dense_affine_scan,
    native_cuda_available,
)
from koemi.cuda_kernels.dense.affine_scan import _reduce_to_shape
from koemi.model.scan import affine_scan


CUDA_SKIP_REASON = "CUDA device and nvcc are unavailable"


def sequential_affine_scan(
    retention: torch.Tensor,
    increment: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    state = initial
    states: list[torch.Tensor] = []
    for position in range(increment.shape[1]):
        state = retention[:, position] * state + increment[:, position]
        states.append(state)
    if not states:
        return increment[:, :0]
    return torch.stack(states, dim=1)


class DenseCudaContractTests(unittest.TestCase):
    def test_cpu_input_is_rejected_without_fallback(self) -> None:
        retention = torch.ones((1, 2, 4), dtype=torch.float32)
        increment = torch.ones((1, 2, 4), dtype=torch.float32)
        if torch.cuda.is_available():
            with self.assertRaisesRegex(DenseCudaValidationError, "must be on CUDA"):
                dense_affine_scan(retention, increment)
        else:
            with self.assertRaisesRegex(DenseCudaUnavailableError, "torch.cuda.is_available"):
                dense_affine_scan(retention, increment)

    def test_invalid_shape_is_rejected_before_native_build(self) -> None:
        retention = torch.ones((1, 2, 4))
        increment = torch.ones((1, 2, 5))
        with self.assertRaisesRegex(DenseCudaValidationError, "broadcast"):
            dense_affine_scan(retention, increment)

    def test_invalid_initial_shape_is_rejected_before_native_build(self) -> None:
        retention = torch.ones((1, 2, 4))
        increment = torch.ones((1, 2, 4))
        initial = torch.ones((1, 5))
        with self.assertRaisesRegex(DenseCudaValidationError, "initial must have shape"):
            dense_affine_scan(retention, increment, initial)

    def test_broadcast_gradient_reduction_handles_missing_feature_dimensions(self) -> None:
        expanded_gradient = torch.ones((2, 3, 4), dtype=torch.float32)
        reduced = _reduce_to_shape(expanded_gradient, torch.Size((2, 3)))
        self.assertEqual(torch.Size((2, 3)), reduced.shape)
        self.assertTrue(torch.equal(reduced, torch.full((2, 3), 4.0)))


@unittest.skipUnless(native_cuda_available(), CUDA_SKIP_REASON)
class DenseCudaExecutionTests(unittest.TestCase):
    device = torch.device("cuda")

    def test_exact_retention_forward_matches_reference(self) -> None:
        torch.manual_seed(101)
        retention = torch.rand((2, 37, 64), device=self.device) * 0.9 + 0.05
        increment = torch.randn((2, 37, 64), device=self.device)
        initial = torch.randn((2, 64), device=self.device)
        expected = sequential_affine_scan(retention, increment, initial)
        obtained = dense_affine_scan(retention, increment, initial)
        self.assertTrue(torch.allclose(obtained, expected, atol=1e-5, rtol=1e-5))

    def test_broadcast_retention_forward_matches_reference(self) -> None:
        torch.manual_seed(102)
        retention = torch.rand((2, 29, 1, 1), device=self.device) * 0.9 + 0.05
        increment = torch.randn((2, 29, 4, 8), device=self.device)
        initial = torch.randn((2, 4, 8), device=self.device)
        expected = sequential_affine_scan(retention, increment, initial)
        obtained = dense_affine_scan(retention, increment, initial)
        self.assertTrue(torch.allclose(obtained, expected, atol=1e-5, rtol=1e-5))

    def test_partial_broadcast_retention_forward_matches_reference(self) -> None:
        torch.manual_seed(103)
        retention = torch.rand((2, 17, 1, 3), device=self.device) * 0.9 + 0.05
        increment = torch.randn((2, 17, 4, 3), device=self.device)
        initial = torch.randn((2, 4, 3), device=self.device)
        expected = sequential_affine_scan(retention, increment, initial)
        obtained = dense_affine_scan(retention, increment, initial)
        self.assertTrue(torch.allclose(obtained, expected, atol=1e-5, rtol=1e-5))

    def test_supported_low_precision_forward_matches_reference(self) -> None:
        supported_dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        for dtype in supported_dtypes:
            with self.subTest(dtype=dtype):
                torch.manual_seed(104)
                retention = torch.rand((1, 13, 32), device=self.device, dtype=dtype) * 0.8 + 0.1
                increment = torch.randn((1, 13, 32), device=self.device, dtype=dtype)
                initial = torch.randn((1, 32), device=self.device, dtype=dtype)
                expected = sequential_affine_scan(retention, increment, initial)
                obtained = dense_affine_scan(retention, increment, initial)
                self.assertTrue(torch.allclose(obtained, expected, atol=5e-3, rtol=5e-3))

    def test_forward_and_first_order_gradients_match_reference(self) -> None:
        torch.manual_seed(105)
        retention_values = torch.rand((2, 19, 1, 1), device=self.device) * 0.8 + 0.1
        increment_values = torch.randn((2, 19, 4, 8), device=self.device)
        initial_values = torch.randn((2, 4, 8), device=self.device)
        upstream_gradient = torch.randn_like(increment_values)

        native_retention = retention_values.detach().requires_grad_()
        native_increment = increment_values.detach().requires_grad_()
        native_initial = initial_values.detach().requires_grad_()
        native_states = dense_affine_scan(native_retention, native_increment, native_initial)
        native_states.backward(upstream_gradient)
        native_gradients = (
            native_retention.grad.detach(),
            native_increment.grad.detach(),
            native_initial.grad.detach(),
        )

        reference_retention = retention_values.detach().requires_grad_()
        reference_increment = increment_values.detach().requires_grad_()
        reference_initial = initial_values.detach().requires_grad_()
        reference_states = sequential_affine_scan(
            reference_retention,
            reference_increment,
            reference_initial,
        )
        reference_states.backward(upstream_gradient)
        reference_gradients = (
            reference_retention.grad.detach(),
            reference_increment.grad.detach(),
            reference_initial.grad.detach(),
        )

        self.assertTrue(torch.allclose(native_states, reference_states, atol=1e-5, rtol=1e-5))
        for obtained, expected in zip(native_gradients, reference_gradients):
            self.assertTrue(torch.allclose(obtained, expected, atol=2e-5, rtol=2e-5))

    def test_cuda_benchmark_reports_without_claiming_speedup(self) -> None:
        torch.manual_seed(106)
        retention = torch.rand((8, 512, 128), device=self.device) * 0.8 + 0.1
        increment = torch.randn((8, 512, 128), device=self.device)
        initial = torch.randn((8, 128), device=self.device)
        for _ in range(5):
            affine_scan(retention, increment, initial)
            dense_affine_scan(retention, increment, initial)
        torch.cuda.synchronize(self.device)

        def measure(function: object) -> float:
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            for _ in range(20):
                function(retention, increment, initial)
            finished.record()
            finished.synchronize()
            return started.elapsed_time(finished) / 20.0

        baseline_milliseconds = measure(affine_scan)
        native_milliseconds = measure(dense_affine_scan)
        print(
            "dense_affine_scan benchmark "
            f"baseline_ms={baseline_milliseconds:.4f} "
            f"native_ms={native_milliseconds:.4f} "
            f"ratio={native_milliseconds / baseline_milliseconds:.4f}"
        )
        self.assertGreater(baseline_milliseconds, 0.0)
        self.assertGreater(native_milliseconds, 0.0)


if __name__ == "__main__":
    unittest.main()
