from __future__ import annotations

import collections
import math
import unittest

import torch
from torch.nn import functional

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.execution import ExecutionMode
from koemi.model.memory import LocalKeyValueMemory, NEGATIVE_INFINITY
from koemi.model.network import KoemiModel


def unfold_read_window(memory, carried_keys, carried_values, carried_valid, keys, values, valid_mask, queries):
    """The strided-view read this module replaced, kept as the equivalence oracle."""
    batch_size, length, width = queries.shape
    window = memory.local_memory_size
    carried_length = carried_keys.shape[1]
    all_keys = torch.cat((carried_keys, keys), dim=1)
    all_values = torch.cat((carried_values, values), dim=1)
    all_valid = torch.cat((carried_valid, valid_mask), dim=1)
    leading_keys = all_keys.new_zeros(batch_size, window, width)
    leading_values = all_values.new_zeros(batch_size, window, width)
    leading_valid = torch.zeros(batch_size, window, dtype=torch.bool, device=queries.device)
    padded_keys = torch.cat((leading_keys, all_keys), dim=1)
    padded_values = torch.cat((leading_values, all_values), dim=1)
    padded_valid = torch.cat((leading_valid, all_valid), dim=1)
    start = carried_length
    key_windows = padded_keys.unfold(1, window, 1)[:, start : start + length]
    value_windows = padded_values.unfold(1, window, 1)[:, start : start + length]
    valid_windows = padded_valid.unfold(1, window, 1)[:, start : start + length]
    scores = torch.einsum("btdw,btd->btw", key_windows, queries) / math.sqrt(width)
    masked_scores = scores.masked_fill(~valid_windows, NEGATIVE_INFINITY)
    any_slot = valid_windows.any(dim=-1, keepdim=True)
    weights = torch.softmax(masked_scores, dim=-1)
    weights = torch.where(any_slot, weights, torch.zeros_like(weights))
    local_value = torch.einsum("btdw,btw->btd", value_windows, weights)
    normalized_queries = functional.normalize(queries, dim=-1)
    normalized_keys = functional.normalize(key_windows, dim=2)
    similarity = torch.einsum("btdw,btd->btw", normalized_keys, normalized_queries)
    similarity = similarity.masked_fill(~valid_windows, NEGATIVE_INFINITY)
    highest = similarity.amax(dim=-1)
    novelty = torch.where(any_slot.squeeze(-1), 1.0 - highest, torch.ones_like(highest))
    return local_value, novelty.clamp(0.0, 1.0)


def saved_activation_bytes(model: KoemiModel, input_ids: torch.Tensor) -> int:
    seen: set[int] = set()
    total = [0]

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        pointer = tensor.data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            total[0] += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
        output.logits.float().square().mean().backward()
    return total[0]


class SlidingWindowReadTest(unittest.TestCase):
    cases = ((4, 0, 2, 7, 16), (4, 4, 2, 7, 16), (8, 3, 3, 11, 24), (16, 16, 2, 20, 32), (6, 2, 1, 1, 16))

    def test_it_matches_the_strided_window_read(self) -> None:
        torch.manual_seed(4)
        for window, carried, batch, length, width in self.cases:
            with self.subTest(window=window, carried=carried, length=length):
                memory = LocalKeyValueMemory(width, window)
                carried_keys = torch.randn(batch, carried, width)
                carried_values = torch.randn(batch, carried, width)
                carried_valid = torch.rand(batch, carried) > 0.3
                keys = torch.randn(batch, length, width)
                values = torch.randn(batch, length, width)
                queries = torch.randn(batch, length, width)
                valid_mask = torch.rand(batch, length) > 0.2
                expected = unfold_read_window(
                    memory, carried_keys, carried_values, carried_valid, keys, values, valid_mask, queries
                )
                obtained = memory.read_window(
                    carried_keys, carried_values, carried_valid, keys, values, valid_mask, queries
                )
                torch.testing.assert_close(obtained[0], expected[0], rtol=0.0, atol=1e-5)
                torch.testing.assert_close(obtained[1], expected[1], rtol=0.0, atol=1e-5)

    def test_the_mask_selects_the_window_entries_strictly_before_each_query(self) -> None:
        all_valid = torch.ones(1, 7, dtype=torch.bool)
        mask = LocalKeyValueMemory.sliding_window_mask(all_valid, carried_length=3, length=4, window=2)
        self.assertEqual(mask[0, 0].tolist(), [False, True, True, False, False, False, False])
        self.assertEqual(mask[0, 3].tolist(), [False, False, False, False, True, True, False])

    def test_an_invalid_source_is_never_eligible(self) -> None:
        all_valid = torch.ones(1, 5, dtype=torch.bool)
        all_valid[0, 1] = False
        mask = LocalKeyValueMemory.sliding_window_mask(all_valid, carried_length=2, length=3, window=2)
        self.assertFalse(bool(mask[0, :, 1].any()))

    def test_a_query_with_no_predecessor_reads_nothing(self) -> None:
        memory = LocalKeyValueMemory(8, 4)
        empty = torch.empty(1, 0, 8)
        empty_valid = torch.empty(1, 0, dtype=torch.bool)
        keys = torch.randn(1, 1, 8)
        values = torch.randn(1, 1, 8)
        queries = torch.randn(1, 1, 8)
        local_value, novelty = memory.read_window(
            empty, empty, empty_valid, keys, values, torch.ones(1, 1, dtype=torch.bool), queries
        )
        self.assertTrue(torch.equal(local_value, torch.zeros_like(local_value)))
        self.assertTrue(torch.equal(novelty, torch.ones_like(novelty)))


class ActivationCheckpointingTest(unittest.TestCase):
    def settings(self, checkpointing: bool) -> ModelSettings:
        return ModelSettings(
            embedding_size=32,
            memory_features=8,
            local_memory_size=4,
            salience_memory_size=4,
            expert_count=8,
            expert_top_k=3,
            scan_chunk=8,
            activation_checkpointing=checkpointing,
            ablation="no_refine",
        )

    def build_pair(self) -> tuple[KoemiModel, KoemiModel, torch.Tensor]:
        torch.manual_seed(5)
        plain = KoemiModel(self.settings(False))
        checkpointed = KoemiModel(self.settings(True))
        checkpointed.load_state_dict(plain.state_dict())
        plain.train()
        checkpointed.train()
        input_ids = torch.randint(0, PAD_TOKEN_ID, (2, 32))
        input_ids[0, -3:] = PAD_TOKEN_ID
        return plain, checkpointed, input_ids

    def test_it_agrees_on_logits_state_and_counters(self) -> None:
        plain, checkpointed, input_ids = self.build_pair()
        plain_output = plain(input_ids, execution_mode=ExecutionMode.PARALLEL)
        checkpointed_output = checkpointed(input_ids, execution_mode=ExecutionMode.PARALLEL)
        self.assertTrue(torch.equal(plain_output.logits, checkpointed_output.logits))
        self.assertTrue(torch.equal(plain_output.state.working_state, checkpointed_output.state.working_state))
        self.assertTrue(torch.equal(plain_output.state.memory_basis, checkpointed_output.state.memory_basis))
        self.assertTrue(torch.equal(plain_output.expert_indices, checkpointed_output.expert_indices))
        self.assertEqual(plain_output.token_count, checkpointed_output.token_count)
        self.assertEqual(plain_output.expert_count, checkpointed_output.expert_count)
        self.assertEqual(plain_output.state.step_index, checkpointed_output.state.step_index)

    def test_it_agrees_on_gradients(self) -> None:
        plain, checkpointed, input_ids = self.build_pair()
        plain(input_ids, execution_mode=ExecutionMode.PARALLEL).logits.square().sum().backward()
        checkpointed(input_ids, execution_mode=ExecutionMode.PARALLEL).logits.square().sum().backward()
        plain_gradient = torch.cat([parameter.grad.reshape(-1) for parameter in plain.parameters()])
        checkpointed_gradient = torch.cat(
            [parameter.grad.reshape(-1) for parameter in checkpointed.parameters()]
        )
        relative = (plain_gradient - checkpointed_gradient).norm() / plain_gradient.norm()
        self.assertLess(float(relative), float(torch.finfo(torch.float32).eps))

    def test_it_saves_activation_memory(self) -> None:
        plain, checkpointed, input_ids = self.build_pair()
        plain_bytes = saved_activation_bytes(plain, input_ids)
        checkpointed_bytes = saved_activation_bytes(checkpointed, input_ids)
        self.assertLess(checkpointed_bytes, plain_bytes)

    def test_it_stays_off_under_inference_mode(self) -> None:
        _, checkpointed, input_ids = self.build_pair()
        checkpointed.eval()
        with torch.no_grad():
            output = checkpointed(input_ids, execution_mode=ExecutionMode.PARALLEL)
        self.assertFalse(output.logits.requires_grad)

    def test_the_default_keeps_checkpointing_off(self) -> None:
        self.assertFalse(ModelSettings().activation_checkpointing)

    def test_it_rejects_a_non_boolean_flag(self) -> None:
        with self.assertRaises(TypeError):
            ModelSettings(activation_checkpointing="yes")


if __name__ == "__main__":
    unittest.main()
