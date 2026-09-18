from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.execution import ExecutionMode
from koemi.model.experts import DeterministicExpertMixture
from koemi.model.network import KoemiModel


def build_case(expert_count: int, top_k: int, width: int = 24, batch: int = 3, length: int = 11):
    torch.manual_seed(11)
    loop = DeterministicExpertMixture(width, expert_count, top_k=top_k, dispatch="loop")
    segments = DeterministicExpertMixture(width, expert_count, top_k=top_k, dispatch="segments")
    segments.load_state_dict(loop.state_dict())
    token_ids = torch.randint(0, PAD_TOKEN_ID, (batch, length))
    token_ids[0, -1] = PAD_TOKEN_ID
    token_ids[1, 0] = PAD_TOKEN_ID
    previous_token_ids = torch.randint(0, PAD_TOKEN_ID, (batch, length))
    valid_mask = token_ids != PAD_TOKEN_ID
    context = torch.randn(batch, length, width)
    return loop, segments, context, token_ids, previous_token_ids, valid_mask


class SegmentDispatchEquivalenceTest(unittest.TestCase):
    def test_segments_match_the_loop_on_values_for_every_top_k(self) -> None:
        for expert_count, top_k in ((4, 1), (8, 3), (16, 6), (7, 5)):
            with self.subTest(expert_count=expert_count, top_k=top_k):
                loop, segments, context, tokens, previous, valid = build_case(expert_count, top_k)
                assignments = loop.assign_top_k(tokens, previous, valid)
                expected = loop._forward_with_module_dispatch(context, assignments, valid)
                obtained = segments._forward_with_sorted_segments(context, assignments, valid)
                self.assertTrue(torch.equal(expected[0], obtained[0]))
                self.assertTrue(torch.equal(expected[1], obtained[1]))
                self.assertTrue(torch.equal(expected[2], obtained[2]))

    def test_segments_match_the_loop_on_every_expert_gradient(self) -> None:
        loop, segments, context, tokens, previous, valid = build_case(16, 6)
        loop_context = context.clone().requires_grad_(True)
        segment_context = context.clone().requires_grad_(True)
        assignments = loop.assign_top_k(tokens, previous, valid)
        loop._forward_with_module_dispatch(loop_context, assignments, valid)[0].square().sum().backward()
        segments._forward_with_sorted_segments(segment_context, assignments, valid)[0].square().sum().backward()
        for loop_parameter, segment_parameter in zip(loop.parameters(), segments.parameters(), strict=True):
            self.assertIsNotNone(loop_parameter.grad)
            self.assertTrue(torch.equal(loop_parameter.grad, segment_parameter.grad))

    def test_the_input_gradient_differs_only_by_accumulation_rounding(self) -> None:
        loop, segments, context, tokens, previous, valid = build_case(16, 6)
        loop_context = context.clone().requires_grad_(True)
        segment_context = context.clone().requires_grad_(True)
        assignments = loop.assign_top_k(tokens, previous, valid)
        loop._forward_with_module_dispatch(loop_context, assignments, valid)[0].square().sum().backward()
        segments._forward_with_sorted_segments(segment_context, assignments, valid)[0].square().sum().backward()
        difference = (loop_context.grad - segment_context.grad).abs().max()
        resolution = torch.finfo(torch.float32).eps * loop_context.grad.abs().max()
        self.assertLess(float(difference), float(resolution))

    def test_every_padded_row_keeps_its_context_untouched(self) -> None:
        loop, segments, context, tokens, previous, valid = build_case(8, 3)
        assignments = loop.assign_top_k(tokens, previous, valid)
        obtained = segments._forward_with_sorted_segments(context, assignments, valid)[0]
        padded = ~valid
        self.assertTrue(padded.any())
        self.assertTrue(torch.equal(obtained[padded], context[padded]))

    def test_segment_dispatch_issues_no_nonzero_call(self) -> None:
        loop, segments, context, tokens, previous, valid = build_case(16, 6)
        assignments = loop.assign_top_k(tokens, previous, valid)
        calls = {"nonzero": 0}
        original_nonzero = torch.nonzero

        def counted_nonzero(*arguments, **keywords):
            calls["nonzero"] += 1
            return original_nonzero(*arguments, **keywords)

        torch.nonzero = counted_nonzero
        try:
            segments._forward_with_sorted_segments(context, assignments, valid)
            segment_calls = calls["nonzero"]
            calls["nonzero"] = 0
            loop._forward_with_module_dispatch(context, assignments, valid)
            loop_calls = calls["nonzero"]
        finally:
            torch.nonzero = original_nonzero
        self.assertEqual(segment_calls, 0)
        self.assertEqual(loop_calls, 16)


class SegmentDispatchModelTest(unittest.TestCase):
    def settings(self, dispatch: str) -> ModelSettings:
        return ModelSettings(
            embedding_size=32,
            memory_features=8,
            local_memory_size=4,
            salience_memory_size=4,
            expert_count=8,
            expert_top_k=3,
            expert_dispatch=dispatch,
            scan_chunk=8,
            ablation="no_refine",
        )

    def test_whole_model_agrees_in_logits_state_and_gradients(self) -> None:
        torch.manual_seed(5)
        loop_model = KoemiModel(self.settings("loop"))
        segment_model = KoemiModel(self.settings("segments"))
        segment_model.load_state_dict(loop_model.state_dict())
        input_ids = torch.randint(0, PAD_TOKEN_ID, (2, 24))
        input_ids[0, -3:] = PAD_TOKEN_ID
        loop_output = loop_model(input_ids, execution_mode=ExecutionMode.PARALLEL)
        segment_output = segment_model(input_ids, execution_mode=ExecutionMode.PARALLEL)
        self.assertTrue(torch.equal(loop_output.logits, segment_output.logits))
        self.assertTrue(torch.equal(loop_output.state.working_state, segment_output.state.working_state))
        self.assertTrue(torch.equal(loop_output.state.memory_basis, segment_output.state.memory_basis))
        loop_output.logits.square().sum().backward()
        segment_output.logits.square().sum().backward()
        loop_gradient = torch.cat([parameter.grad.reshape(-1) for parameter in loop_model.parameters()])
        segment_gradient = torch.cat([parameter.grad.reshape(-1) for parameter in segment_model.parameters()])
        relative_difference = (loop_gradient - segment_gradient).norm() / loop_gradient.norm()
        self.assertLess(float(relative_difference), float(torch.finfo(torch.float32).eps))

    def test_settings_reject_an_unknown_dispatch(self) -> None:
        with self.assertRaises(ValueError):
            ModelSettings(expert_count=4, expert_dispatch="grouped")

    def test_loop_remains_the_default(self) -> None:
        self.assertEqual(ModelSettings().expert_dispatch, "loop")


if __name__ == "__main__":
    unittest.main()
