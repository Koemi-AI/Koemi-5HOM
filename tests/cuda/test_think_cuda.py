from __future__ import annotations

import math
import unittest

import torch

from koemi.cuda_kernels.think import (
    CudaThinkUnavailableError,
    CudaThinkValidationError,
    benchmark_causal_surprise,
    causal_surprise,
)


CUDA_SKIP_REASON = "CUDA unavailable: torch.cuda.is_available() is False"


def reference_causal_surprise(
    prior_states: torch.Tensor,
    content_weight: torch.Tensor,
    content_bias: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    logits = torch.matmul(prior_states.float(), content_weight.float().transpose(0, 1))
    logits = logits + content_bias.float()
    log_partition = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    negative_log_likelihood = log_partition - target_logits
    surprise = 1.0 - torch.exp(-negative_log_likelihood / math.log(content_weight.shape[0]))
    return torch.where(valid_mask, surprise, torch.zeros_like(surprise)).to(prior_states.dtype)


class ThinkCudaContractTests(unittest.TestCase):
    def build_cpu_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            torch.randn(2, 5, 8),
            torch.randn(17, 8),
            torch.randn(17),
            torch.randint(0, 17, (2, 5), dtype=torch.long),
            torch.ones(2, 5, dtype=torch.bool),
        )

    def test_cpu_execution_reports_unavailable_without_fallback(self) -> None:
        inputs = self.build_cpu_inputs()
        if torch.cuda.is_available():
            with self.assertRaisesRegex(CudaThinkValidationError, "all think tensors must be on CUDA"):
                causal_surprise(*inputs)
        else:
            with self.assertRaisesRegex(CudaThinkUnavailableError, r"torch.cuda.is_available\(\) is False"):
                causal_surprise(*inputs)

    def test_shape_and_dtype_contracts_are_rejected_before_dispatch(self) -> None:
        prior_states, content_weight, content_bias, target_ids, valid_mask = self.build_cpu_inputs()
        with self.assertRaisesRegex(CudaThinkValidationError, "target_ids and valid_mask"):
            causal_surprise(prior_states, content_weight, content_bias, target_ids[:, :-1], valid_mask)
        with self.assertRaisesRegex(CudaThinkValidationError, "share a dtype"):
            causal_surprise(prior_states, content_weight.double(), content_bias.double(), target_ids, valid_mask)
        with self.assertRaisesRegex(CudaThinkValidationError, "target_ids must use torch.int64"):
            causal_surprise(prior_states, content_weight, content_bias, target_ids.int(), valid_mask)

    def test_benchmark_options_are_validated_without_dispatch(self) -> None:
        inputs = self.build_cpu_inputs()
        with self.assertRaisesRegex(CudaThinkValidationError, "non-negative integer"):
            benchmark_causal_surprise(*inputs, warmup_iterations=-1)
        with self.assertRaisesRegex(CudaThinkValidationError, "positive integer"):
            benchmark_causal_surprise(*inputs, measured_iterations=0)


@unittest.skipUnless(torch.cuda.is_available(), CUDA_SKIP_REASON)
class ThinkCudaExecutionTests(unittest.TestCase):
    device = torch.device("cuda:0")

    def build_inputs(self, *, requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(17)
        prior_states = torch.randn(2, 7, 8, device=self.device, requires_grad=requires_grad)
        content_weight = torch.randn(19, 8, device=self.device, requires_grad=requires_grad)
        content_bias = torch.randn(19, device=self.device, requires_grad=requires_grad)
        target_ids = torch.randint(0, 19, (2, 7), device=self.device, dtype=torch.long)
        valid_mask = torch.ones(2, 7, device=self.device, dtype=torch.bool)
        valid_mask[0, -1] = False
        return prior_states, content_weight, content_bias, target_ids, valid_mask

    def test_forward_matches_the_stable_pytorch_reference(self) -> None:
        inputs = self.build_inputs()
        obtained = causal_surprise(*inputs)
        expected = reference_causal_surprise(*inputs)
        torch.testing.assert_close(obtained, expected, rtol=3e-4, atol=3e-4)
        self.assertTrue(torch.equal(obtained[0, -1], torch.zeros((), device=self.device)))

    def test_backward_matches_the_reference_for_all_differentiable_inputs(self) -> None:
        prior_states, content_weight, content_bias, target_ids, valid_mask = self.build_inputs(requires_grad=True)
        native_output = causal_surprise(prior_states, content_weight, content_bias, target_ids, valid_mask)
        native_loss = native_output.square().sum()
        native_gradients = torch.autograd.grad(
            native_loss,
            (prior_states, content_weight, content_bias),
            retain_graph=True,
        )

        reference_prior = prior_states.detach().clone().requires_grad_()
        reference_weight = content_weight.detach().clone().requires_grad_()
        reference_bias = content_bias.detach().clone().requires_grad_()
        reference_output = reference_causal_surprise(
            reference_prior,
            reference_weight,
            reference_bias,
            target_ids,
            valid_mask,
        )
        reference_loss = reference_output.square().sum()
        reference_gradients = torch.autograd.grad(
            reference_loss,
            (reference_prior, reference_weight, reference_bias),
        )

        torch.testing.assert_close(native_output, reference_output, rtol=3e-4, atol=3e-4)
        for native_gradient, reference_gradient in zip(native_gradients, reference_gradients):
            torch.testing.assert_close(native_gradient, reference_gradient, rtol=2e-3, atol=2e-3)
        self.assertTrue(torch.equal(native_gradients[0][0, -1], torch.zeros(8, device=self.device)))

    def test_future_rows_cannot_change_an_earlier_result(self) -> None:
        inputs = self.build_inputs()
        baseline = causal_surprise(*inputs)
        changed_prior = inputs[0].clone()
        changed_prior[:, 4:] = torch.randn_like(changed_prior[:, 4:]) * 100.0
        changed_targets = inputs[3].clone()
        changed_targets[:, 4:] = (changed_targets[:, 4:] + 7) % 19
        changed = causal_surprise(changed_prior, inputs[1], inputs[2], changed_targets, inputs[4])
        torch.testing.assert_close(changed[:, :4], baseline[:, :4], rtol=0.0, atol=0.0)

    def test_invalid_positions_are_zero_and_have_no_gradient(self) -> None:
        prior_states, content_weight, content_bias, target_ids, valid_mask = self.build_inputs(requires_grad=True)
        output = causal_surprise(prior_states, content_weight, content_bias, target_ids, valid_mask)
        output.sum().backward()
        self.assertTrue(torch.equal(output[0, -1], torch.zeros((), device=self.device)))
        self.assertTrue(torch.equal(prior_states.grad[0, -1], torch.zeros(8, device=self.device)))

    def test_benchmark_reports_synchronized_latency_and_throughput(self) -> None:
        inputs = self.build_inputs()
        measurement = benchmark_causal_surprise(
            *inputs,
            warmup_iterations=2,
            measured_iterations=5,
        )
        self.assertEqual(14, measurement.position_count)
        self.assertEqual(2, measurement.warmup_iterations)
        self.assertEqual(5, measurement.measured_iterations)
        self.assertGreater(measurement.elapsed_seconds, 0.0)
        self.assertGreater(measurement.latency_milliseconds, 0.0)
        self.assertGreater(measurement.positions_per_second, 0.0)
        print(
            "THINK_CUDA_BENCHMARK "
            f"latency_ms={measurement.latency_milliseconds:.4f} "
            f"positions_per_second={measurement.positions_per_second:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
