from __future__ import annotations

import unittest

import torch
from torch.nn import functional

from koemi.cuda_kernels.moe import (
    CudaMoeUnavailableError,
    CudaMoeValidationError,
    combine_topk,
    grouped_expert_mlp_inference,
    load_native_extension,
    permute_topk,
    route_topk_valid,
)
from koemi.model.experts import content_dispatch_hash


CUDA_SKIP_REASON = "CUDA device and nvcc are unavailable"


class MoeCudaContractTests(unittest.TestCase):
    def test_route_rejects_cpu_without_fallback(self) -> None:
        token_ids = torch.tensor([[1, 2]], dtype=torch.long)
        previous = torch.tensor([[0, 1]], dtype=torch.long)
        valid = torch.ones_like(token_ids, dtype=torch.bool)
        with self.assertRaisesRegex(CudaMoeValidationError, "must be a CUDA tensor"):
            route_topk_valid(token_ids, previous, valid, expert_count=4, top_k=2)

    def test_extension_load_reports_unavailable_without_cuda(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA is available")
        with self.assertRaisesRegex(CudaMoeUnavailableError, "torch.cuda.is_available"):
            load_native_extension()

    def test_permute_rejects_cpu_without_fallback(self) -> None:
        context = torch.ones((2, 4))
        assignments = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        valid = torch.ones(2, dtype=torch.bool)
        with self.assertRaisesRegex(CudaMoeValidationError, "must be a CUDA tensor"):
            permute_topk(context, assignments, valid, expert_count=2)


@unittest.skipUnless(torch.cuda.is_available(), CUDA_SKIP_REASON)
class MoeCudaExecutionTests(unittest.TestCase):
    device = torch.device("cuda")

    def test_route_matches_content_hash_reference(self) -> None:
        token_ids = torch.tensor([[1, 2, 99], [7, 3, 4]], device=self.device, dtype=torch.long)
        previous = torch.tensor([[0, 1, 2], [6, 7, 8]], device=self.device, dtype=torch.long)
        valid = torch.tensor([[True, True, False], [True, False, True]], device=self.device)
        obtained = route_topk_valid(token_ids, previous, valid, expert_count=8, top_k=2)
        hashed = content_dispatch_hash(token_ids, previous)
        offsets = torch.arange(2, device=self.device, dtype=hashed.dtype)
        expected = (hashed.unsqueeze(-1) + offsets * 0x9E3779B9).remainder(8)
        expected = expected.masked_fill(~valid.unsqueeze(-1), -1)
        torch.testing.assert_close(obtained, expected)

    def test_permute_and_combine_preserve_stable_pairs(self) -> None:
        context = torch.arange(12, device=self.device, dtype=torch.float32).reshape(3, 4)
        assignments = torch.tensor([[1, 0], [0, 1], [1, 0]], device=self.device, dtype=torch.long)
        valid = torch.tensor([True, True, False], device=self.device)
        sorted_context, sorted_rows, sorted_experts, offsets = permute_topk(
            context, assignments, valid, expert_count=2
        )
        valid_pairs = int(offsets[-1].item())
        self.assertEqual(4, sorted_context.shape[0])
        self.assertEqual(2, valid_pairs)
        self.assertTrue(torch.equal(sorted_experts[:valid_pairs], torch.tensor([0, 1], device=self.device)))
        outputs = sorted_context + 10.0
        combined = combine_topk(
            outputs,
            sorted_rows,
            offsets,
            token_count=context.shape[0],
            top_k=2,
        )
        expected = torch.zeros_like(context)
        for pair in range(valid_pairs):
            expected[sorted_rows[pair]] += outputs[pair] / 2.0
        torch.testing.assert_close(combined, expected)

    def test_grouped_expert_mlp_matches_reference(self) -> None:
        expert_count, input_width, hidden_width, pair_count = 2, 4, 3, 3
        sorted_context = torch.randn(pair_count, input_width, device=self.device)
        sorted_experts = torch.tensor([0, 0, 1], device=self.device, dtype=torch.long)
        offsets = torch.tensor([0, 2, 3], device=self.device, dtype=torch.long)
        gate_weights = torch.randn(expert_count, hidden_width, input_width, device=self.device)
        gate_biases = torch.randn(expert_count, hidden_width, device=self.device)
        value_weights = torch.randn_like(gate_weights)
        value_biases = torch.randn_like(gate_biases)
        output_weights = torch.randn(expert_count, input_width, hidden_width, device=self.device)
        output_biases = torch.randn(expert_count, input_width, device=self.device)
        obtained = grouped_expert_mlp_inference(
            sorted_context,
            sorted_experts,
            offsets,
            gate_weights,
            gate_biases,
            value_weights,
            value_biases,
            output_weights,
            output_biases,
        )
        expected_rows = []
        for row, expert in enumerate(sorted_experts.tolist()):
            gate = functional.linear(sorted_context[row], gate_weights[expert], gate_biases[expert])
            value = functional.linear(sorted_context[row], value_weights[expert], value_biases[expert])
            hidden = functional.silu(gate) * value
            expected_rows.append(functional.linear(hidden, output_weights[expert], output_biases[expert]))
        expected = torch.stack(expected_rows)
        torch.testing.assert_close(obtained, expected, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
