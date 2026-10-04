from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel
from koemi.model.router import LearnedExpertMixture
from koemi.training.checkpoints import CheckpointStore
from koemi.training.objective import calculate_training_objective


class LearnedRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.settings = ModelSettings(embedding_size=16, memory_features=4,
                                      local_memory_size=3, salience_memory_size=3,
                                      expert_count=4, expert_top_k=2, expert_routing="learned")
        self.tokens = torch.tensor([[65, 66, 67, 68, PAD_TOKEN_ID], [70, 71, 72, 73, 74]])

    def test_top_k_is_unique_padding_preserved_and_balance_finite(self) -> None:
        mixture = LearnedExpertMixture(16, 4, 3)
        context = torch.randn(2, 5, 16)
        valid = self.tokens != PAD_TOKEN_ID
        mixed, _, assignments, stats = mixture(context, self.tokens, self.tokens, valid)
        self.assertTrue(torch.equal(mixed[~valid], context[~valid]))
        self.assertTrue(assignments[~valid].eq(-1).all())
        for row in assignments[valid].tolist():
            self.assertEqual(len(set(row)), 3)
        self.assertEqual(int(stats.assignment_counts.sum()), int(valid.sum()) * 3)
        self.assertTrue(torch.isfinite(stats.balance_loss))

    def test_task_loss_trains_top_one_gate_without_balance(self) -> None:
        model = KoemiModel(replace(self.settings, expert_top_k=1, expert_load_balance_weight=0))
        output = model(self.tokens)
        objective = calculate_training_objective(output, self.tokens.masked_fill(self.tokens.eq(PAD_TOKEN_ID), -100),
                                                torch.zeros_like(self.tokens, dtype=torch.bool), 1)
        objective.task_loss.backward()
        self.assertIsNotNone(model.experts.gate.weight.grad)
        self.assertGreater(float(model.experts.gate.weight.grad.abs().sum()), 0)

    def test_validation_objective_excludes_auxiliary_router_loss(self) -> None:
        model = KoemiModel(self.settings)
        output = model(self.tokens)
        targets = self.tokens.masked_fill(self.tokens.eq(PAD_TOKEN_ID), -100)
        thinking = torch.zeros_like(self.tokens, dtype=torch.bool)
        train = calculate_training_objective(output, targets, thinking, 1)
        validation = calculate_training_objective(output, targets, thinking, 1, include_router_loss=False)
        torch.testing.assert_close(train.total_loss - validation.total_loss, output.router_loss)
        torch.testing.assert_close(validation.total_loss, validation.task_loss)

    def test_loop_segments_and_static_agree_with_gradients_and_autocast(self) -> None:
        for top_k in (1, 3):
            loop = LearnedExpertMixture(16, 4, top_k)
            segments = LearnedExpertMixture(16, 4, top_k, dispatch="segments")
            segments.load_state_dict(loop.state_dict())
            contexts = [torch.randn(2, 5, 16, requires_grad=True)]
            contexts.append(contexts[0].detach().clone().requires_grad_(True))
            valid = self.tokens != PAD_TOKEN_ID
            first = loop(contexts[0], self.tokens, self.tokens, valid)[0]
            second = segments(contexts[1], self.tokens, self.tokens, valid)[0]
            torch.testing.assert_close(first, second)
            first.square().sum().backward()
            second.square().sum().backward()
            torch.testing.assert_close(contexts[0].grad, contexts[1].grad)
            for a, b in zip(loop.parameters(), segments.parameters(), strict=True):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad)
            loop.eval()
            with torch.no_grad():
                loop.cache_stacked_experts()
                static = loop(contexts[0], self.tokens, self.tokens, valid)[0]
                torch.testing.assert_close(first, static)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertTrue(torch.isfinite(segments(contexts[1], self.tokens, self.tokens, valid)[0]).all())

    def test_empty_mask_has_zero_loss_and_preserves_context(self) -> None:
        mixture = LearnedExpertMixture(16, 4, 2)
        context = torch.randn(2, 5, 16)
        mixed, _, _, stats = mixture(context, self.tokens, self.tokens, torch.zeros_like(self.tokens, dtype=torch.bool))
        self.assertTrue(torch.equal(mixed, context))
        self.assertEqual(float(stats.balance_loss.detach()), 0)

    def test_parallel_sequential_chunked_checkpointed_loss_and_gradients_agree(self) -> None:
        reference = KoemiModel(self.settings)
        expected = reference(self.tokens)
        expected_loss = expected.logits.square().mean() + expected.router_loss
        expected_loss.backward()
        for chunk, checkpointing, mode in ((2, False, ExecutionMode.PARALLEL),
                                           (2, True, ExecutionMode.PARALLEL),
                                           (128, False, ExecutionMode.SEQUENTIAL)):
            model = KoemiModel(replace(self.settings, scan_chunk=chunk, activation_checkpointing=checkpointing))
            model.load_state_dict(reference.state_dict())
            actual = model(self.tokens, execution_mode=mode)
            torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(actual.router_loss, expected.router_loss)
            (actual.logits.square().mean() + actual.router_loss).backward()
            for (name, a), (_, b) in zip(reference.named_parameters(), model.named_parameters(), strict=True):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4, msg=name)

    def test_routing_is_causal(self) -> None:
        model = KoemiModel(self.settings)
        modified = self.tokens.clone()
        modified[:, 3:] = 100
        first, second = model(self.tokens), model(modified)
        self.assertTrue(torch.equal(first.active_expert_indices[:, :3], second.active_expert_indices[:, :3]))
        torch.testing.assert_close(first.logits[:, :3], second.logits[:, :3])

    def test_checkpoint_round_trip_and_legacy_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learned.pt"
            model = KoemiModel(self.settings).eval()
            CheckpointStore().save(path, model)
            loaded = CheckpointStore().load(path).model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(self.tokens).logits, loaded(self.tokens).logits)
            legacy = KoemiModel(replace(self.settings, expert_routing="hash"))
            payload = {"format_version": 7, "model_state": legacy.state_dict(), "model_settings": legacy.settings.to_dict()}
            del payload["model_settings"]["expert_routing"]
            del payload["model_settings"]["expert_load_balance_weight"]
            torch.save(payload, path)
            recovered = CheckpointStore().load(path).model
            torch.testing.assert_close(legacy(self.tokens).logits, recovered(self.tokens).logits, atol=0, rtol=0)

    def test_settings_reject_invalid_modes_and_weights(self) -> None:
        for settings in ({"expert_routing": "missing"}, {"expert_routing": "learned"},
                         {"expert_load_balance_weight": float("nan")},
                         {"expert_count": 4, "expert_routing": "learned", "ablation": "affine"}):
            with self.assertRaises(ValueError):
                ModelSettings(**settings)

    def test_expert_hooks_are_called_in_segments_mode(self) -> None:
        mixture = LearnedExpertMixture(16, 4, 4, dispatch="segments")
        calls = []
        handle = mixture.experts[0].register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            with torch.no_grad():
                mixture(torch.randn(2, 5, 16), self.tokens, self.tokens, self.tokens.ne(PAD_TOKEN_ID))
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
