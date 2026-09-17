from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.network import KoemiModel
from koemi.model.state import KOEMI_STATE_FIELDS, KoemiState
from koemi.runtime.fast_decode import (
    BatchDecoder,
    SamplingPolicy,
    TokenSampler,
    generate_batch,
    select_state_rows,
    to_static_state,
)


SYNCHRONIZING_METHODS = ("item", "tolist", "__int__", "__float__", "__bool__")


class HostSynchronizationGuard:
    """Fail as soon as the guarded block reads a tensor value back to the host."""

    def __init__(self) -> None:
        self.saved: dict[str, object] = {}

    def __enter__(self) -> HostSynchronizationGuard:
        for name in SYNCHRONIZING_METHODS:
            self.saved[name] = getattr(torch.Tensor, name)
            setattr(torch.Tensor, name, self._forbid(name))
        return self

    def __exit__(self, *exception) -> None:
        for name, original in self.saved.items():
            setattr(torch.Tensor, name, original)

    @staticmethod
    def _forbid(name: str):
        def guarded(*arguments, **keywords):
            raise AssertionError(f"the decode path called torch.Tensor.{name}")

        return guarded


def build_model(**overrides) -> KoemiModel:
    settings = dict(
        embedding_size=16,
        memory_features=4,
        local_memory_size=4,
        salience_memory_size=3,
        expert_count=2,
        scan_chunk=8,
    )
    settings.update(overrides)
    model = KoemiModel(ModelSettings(**settings))
    model.eval()
    return model


def greedy_reference(model: KoemiModel, prompt_ids: list[int], steps: int) -> list[int]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    generated: list[int] = []
    with torch.no_grad():
        output = model(input_ids)
        state = output.state
        logits = output.logits[:, -1]
        for _ in range(steps):
            scores = logits.float().clone()
            scores[:, PAD_TOKEN_ID] = float("-inf")
            token = scores.argmax(dim=-1, keepdim=True)
            generated.append(int(token))
            output = model(token, state)
            state = output.state
            logits = output.logits[:, -1]
    return generated


class StaticStateTests(unittest.TestCase):
    def test_static_rings_reproduce_the_growing_state_logits(self) -> None:
        torch.manual_seed(3)
        model = build_model()
        prompt = torch.tensor([[70, 73, 70, 79, 32]], dtype=torch.long)
        follow_up = torch.tensor([[105, 115]], dtype=torch.long)
        with torch.no_grad():
            prefill = model(prompt)
            dynamic = model(follow_up, prefill.state)
            static_state = to_static_state(prefill.state, 4, 3)
            static = model(follow_up, static_state)
        self.assertTrue(torch.allclose(dynamic.logits, static.logits, atol=1e-5))
        for field_name in ("working_state", "memory_basis", "memory_normalizer"):
            self.assertTrue(
                torch.allclose(
                    getattr(dynamic.state, field_name),
                    getattr(static.state, field_name),
                    atol=1e-5,
                ),
                field_name,
            )

    def test_static_rings_keep_one_shape_across_steps(self) -> None:
        torch.manual_seed(4)
        model = build_model()
        decoder = BatchDecoder(model)
        decoder.prefill(torch.tensor([[65, 66, 67]], dtype=torch.long))
        shapes = tuple(getattr(decoder.state, name).shape for name in KOEMI_STATE_FIELDS)
        for _ in range(5):
            decoder.step(torch.tensor([[68]], dtype=torch.long))
            self.assertEqual(
                shapes, tuple(getattr(decoder.state, name).shape for name in KOEMI_STATE_FIELDS)
            )

    def test_static_state_refuses_a_zero_capacity(self) -> None:
        state = KoemiState.create(1, 8, 2, torch.device("cpu"))
        with self.assertRaises(ValueError):
            to_static_state(state, 0, 4)

    def test_state_rows_can_be_selected(self) -> None:
        state = KoemiState.create(3, 8, 2, torch.device("cpu"))
        state.working_state[1].fill_(2.0)
        selected = select_state_rows(state, (1,))
        self.assertEqual((1, 8), tuple(selected.working_state.shape))
        self.assertEqual(2.0, float(selected.working_state[0, 0]))


class BatchDecoderTests(unittest.TestCase):
    def test_decode_matches_the_reference_greedy_loop(self) -> None:
        torch.manual_seed(5)
        model = build_model()
        prompt_ids = [70, 73, 70, 79, 32, 109, 101, 97, 110, 115]
        expected = greedy_reference(model, prompt_ids, 8)
        result = generate_batch(
            model,
            ByteTokenizer(),
            ("FIFO means",),
            8,
            policy=SamplingPolicy(greedy=True),
        )
        self.assertEqual(tuple(expected), result.token_ids[0])

    def test_extend_agrees_with_repeated_steps(self) -> None:
        torch.manual_seed(6)
        model = build_model()
        stepwise = BatchDecoder(model)
        stepwise.prefill(torch.tensor([[65, 66, 67]], dtype=torch.long))
        block = [68, 69, 70]
        stepwise_logits = [stepwise.step(torch.tensor([[token]], dtype=torch.long)) for token in block]
        batched = BatchDecoder(model)
        batched.prefill(torch.tensor([[65, 66, 67]], dtype=torch.long))
        extended = batched.extend(torch.tensor([block], dtype=torch.long))
        for index, logits in enumerate(stepwise_logits):
            self.assertTrue(
                torch.allclose(logits, extended[:, index], atol=1e-5),
                f"position {index} differs by {float((logits - extended[:, index]).abs().max())}",
            )
        self.assertTrue(
            torch.allclose(stepwise.state.working_state, batched.state.working_state, atol=1e-5)
        )

    def test_the_decode_loop_never_synchronizes_with_the_host(self) -> None:
        torch.manual_seed(7)
        model = build_model()
        decoder = BatchDecoder(model)
        sampler = TokenSampler(SamplingPolicy(temperature=0.8), torch.device("cpu"))
        logits = decoder.prefill(torch.tensor([[65, 66, 67, 68]], dtype=torch.long))
        with HostSynchronizationGuard():
            for _ in range(4):
                token = sampler(logits)
                logits = decoder.step(token)
        self.assertEqual((1, model.settings.vocabulary_size), tuple(logits.shape))

    def test_the_guard_catches_the_validating_forward(self) -> None:
        torch.manual_seed(8)
        model = build_model()
        input_ids = torch.tensor([[65]], dtype=torch.long)
        with self.assertRaises(AssertionError):
            with HostSynchronizationGuard():
                with torch.no_grad():
                    model(input_ids)

    def test_trusted_inputs_counts_every_token_without_a_mask_reduction(self) -> None:
        torch.manual_seed(9)
        model = build_model()
        input_ids = torch.tensor([[65, 66], [67, 68]], dtype=torch.long)
        with torch.no_grad():
            validated = model(input_ids)
            trusted = model(input_ids, trusted_inputs=True)
        self.assertEqual(4, validated.token_count)
        self.assertEqual(4, trusted.token_count)
        self.assertTrue(torch.equal(validated.logits, trusted.logits))

    def test_a_training_model_is_refused(self) -> None:
        model = build_model()
        model.train()
        with self.assertRaises(RuntimeError):
            BatchDecoder(model)

    def test_cuda_graph_capture_requires_cuda(self) -> None:
        model = build_model()
        with self.assertRaises(RuntimeError):
            BatchDecoder(model, capture_graph=True)

    def test_right_aligned_padding_is_required(self) -> None:
        model = build_model()
        decoder = BatchDecoder(model)
        with self.assertRaises(ValueError):
            decoder.prefill(torch.tensor([[65, PAD_TOKEN_ID, 66]], dtype=torch.long))

    def test_step_before_prefill_is_refused(self) -> None:
        model = build_model()
        decoder = BatchDecoder(model)
        with self.assertRaises(RuntimeError):
            decoder.step(torch.tensor([[65]], dtype=torch.long))


class BatchGenerationTests(unittest.TestCase):
    def test_a_batch_matches_one_prompt_at_a_time(self) -> None:
        torch.manual_seed(10)
        model = build_model()
        tokenizer = ByteTokenizer()
        prompts = ("queue", "a first in first out queue")
        batched = generate_batch(model, tokenizer, prompts, 6, policy=SamplingPolicy(greedy=True))
        for index, prompt in enumerate(prompts):
            single = generate_batch(model, tokenizer, (prompt,), 6, policy=SamplingPolicy(greedy=True))
            self.assertEqual(single.token_ids[0], batched.token_ids[index], prompt)

    def test_a_stop_token_cuts_the_row(self) -> None:
        torch.manual_seed(11)
        model = build_model()
        result = generate_batch(
            model,
            ByteTokenizer(),
            ("abc",),
            6,
            policy=SamplingPolicy(greedy=True),
        )
        row = result.token_ids[0]
        stop_token_id = row[0]
        cut = generate_batch(
            model,
            ByteTokenizer(),
            ("abc",),
            6,
            policy=SamplingPolicy(greedy=True),
            stop_token_id=stop_token_id,
        )
        self.assertEqual((), cut.token_ids[0])
        self.assertEqual(0, cut.generated_tokens)
        absent_token_id = next(token_id for token_id in range(PAD_TOKEN_ID) if token_id not in row)
        kept = generate_batch(
            model,
            ByteTokenizer(),
            ("abc",),
            6,
            policy=SamplingPolicy(greedy=True),
            stop_token_id=absent_token_id,
        )
        self.assertEqual(row, kept.token_ids[0])

    def test_an_empty_prompt_list_is_refused(self) -> None:
        model = build_model()
        with self.assertRaises(ValueError):
            generate_batch(model, ByteTokenizer(), (), 4)


class SamplingTests(unittest.TestCase):
    def test_the_padding_token_is_never_selected(self) -> None:
        sampler = TokenSampler(SamplingPolicy(temperature=1.0), torch.device("cpu"))
        logits = torch.full((1, PAD_TOKEN_ID + 1), -20.0)
        logits[0, PAD_TOKEN_ID] = 40.0
        logits[0, 65] = 0.0
        self.assertEqual(65, int(sampler(logits)))

    def test_top_k_removes_everything_below_the_threshold(self) -> None:
        sampler = TokenSampler(SamplingPolicy(temperature=1.0, top_k=1), torch.device("cpu"))
        logits = torch.zeros((1, PAD_TOKEN_ID + 1))
        logits[0, 12] = 5.0
        for _ in range(8):
            self.assertEqual(12, int(sampler(logits)))

    def test_greedy_probabilities_are_a_point_mass(self) -> None:
        sampler = TokenSampler(SamplingPolicy(greedy=True), torch.device("cpu"))
        logits = torch.zeros((1, PAD_TOKEN_ID + 1))
        logits[0, 7] = 3.0
        distribution = sampler.probabilities(logits)
        self.assertEqual(1.0, float(distribution[0, 7]))
        self.assertEqual(1.0, float(distribution.sum()))

    def test_an_invalid_policy_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            SamplingPolicy(temperature=0.0)
        with self.assertRaises(ValueError):
            SamplingPolicy(top_k=0)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class CudaGraphDecodeTests(unittest.TestCase):
    def test_a_replayed_graph_matches_the_eager_loop(self) -> None:
        torch.manual_seed(20)
        model = build_model().cuda()
        prompt = torch.tensor([[65, 66, 67, 68]], dtype=torch.long, device="cuda")
        token = torch.tensor([[69]], dtype=torch.long, device="cuda")
        eager = BatchDecoder(model)
        eager.prefill(prompt)
        graphed = BatchDecoder(model, capture_graph=True)
        graphed.prefill(prompt)
        for step_index in range(4):
            expected = eager.step(token)
            produced = graphed.step(token)
            self.assertTrue(
                torch.allclose(expected, produced, atol=1e-4),
                f"step {step_index} differs by {float((expected - produced).abs().max())}",
            )

    def test_a_replayed_graph_advances_the_state_once_per_step(self) -> None:
        torch.manual_seed(21)
        model = build_model().cuda()
        prompt = torch.tensor([[65, 66, 67, 68]], dtype=torch.long, device="cuda")
        token = torch.tensor([[69]], dtype=torch.long, device="cuda")
        graphed = BatchDecoder(model, capture_graph=True)
        graphed.prefill(prompt)
        for expected_step in range(5, 9):
            graphed.step(token)
            self.assertEqual(expected_step, graphed.state.step_index)

    def test_graph_decoding_generates_the_same_text_as_the_eager_path(self) -> None:
        torch.manual_seed(22)
        model = build_model().cuda()
        tokenizer = ByteTokenizer()
        eager = generate_batch(
            model, tokenizer, ("FIFO means",), 8, device="cuda", policy=SamplingPolicy(greedy=True)
        )
        graphed = generate_batch(
            model,
            tokenizer,
            ("FIFO means",),
            8,
            device="cuda",
            policy=SamplingPolicy(greedy=True),
            capture_graph=True,
        )
        self.assertEqual(eager.token_ids, graphed.token_ids)


class ExpertWeightCacheTests(unittest.TestCase):
    def test_cached_stacks_do_not_change_the_output(self) -> None:
        torch.manual_seed(12)
        model = build_model(expert_count=4, expert_top_k=2)
        input_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
        with torch.no_grad():
            uncached = model(input_ids)
            model.cache_inference_weights()
            cached = model(input_ids)
        self.assertTrue(torch.equal(uncached.logits, cached.logits))

    def test_training_mode_releases_the_cache(self) -> None:
        torch.manual_seed(13)
        model = build_model(expert_count=4, expert_top_k=2)
        with torch.no_grad():
            model.cache_inference_weights()
        self.assertIsNotNone(model.experts._stacked_weights)
        model.train()
        self.assertIsNone(model.experts._stacked_weights)

    def test_caching_under_gradients_is_refused(self) -> None:
        model = build_model(expert_count=2)
        with self.assertRaises(RuntimeError):
            model.cache_inference_weights()

    def test_loading_a_state_dictionary_releases_the_cache(self) -> None:
        torch.manual_seed(14)
        model = build_model(expert_count=4, expert_top_k=2)
        with torch.no_grad():
            model.cache_inference_weights()
        model.load_state_dict(model.state_dict())
        self.assertIsNone(model.experts._stacked_weights)

    def test_new_weights_reach_the_output_after_a_reload(self) -> None:
        torch.manual_seed(15)
        model = build_model(expert_count=4, expert_top_k=2)
        other = build_model(expert_count=4, expert_top_k=2)
        input_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
        with torch.no_grad():
            model.cache_inference_weights()
            model(input_ids)
            model.load_state_dict(other.state_dict())
            reloaded = model(input_ids)
            expected = other(input_ids)
        self.assertTrue(torch.equal(expected.logits, reloaded.logits))


if __name__ == "__main__":
    unittest.main()
