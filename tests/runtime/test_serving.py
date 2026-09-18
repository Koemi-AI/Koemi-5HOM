from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.network import KoemiModel
from koemi.model.state import KOEMI_STATE_FIELDS
from koemi.runtime.fast_decode import (
    BatchDecoder,
    SamplingPolicy,
    generate_batch,
    select_state_rows,
    stack_states,
    to_static_state,
)
from koemi.runtime.serving import (
    CANCELLED,
    DEADLINE_EXCEEDED,
    STOPPED,
    TOKEN_LIMIT,
    GenerationRequest,
    ServingEngine,
    ServingLimits,
)


def build_model(seed: int = 7) -> KoemiModel:
    torch.manual_seed(seed)
    settings = ModelSettings(
        embedding_size=32,
        memory_features=8,
        local_memory_size=4,
        salience_memory_size=4,
        expert_count=8,
        expert_top_k=3,
        scan_chunk=8,
        ablation="no_refine",
    )
    return KoemiModel(settings).eval()


def greedy() -> SamplingPolicy:
    return SamplingPolicy(greedy=True)


def prompt(length: int, offset: int = 0) -> tuple[int, ...]:
    return tuple((offset + index * 7) % PAD_TOKEN_ID or 1 for index in range(length))


def reference_greedy(model: KoemiModel, prompt_ids: tuple[int, ...], count: int) -> tuple[int, ...]:
    """Decode one sequence alone, which is the oracle every batched run must match."""
    decoder = BatchDecoder(model)
    logits = decoder.prefill(torch.tensor([list(prompt_ids)], dtype=torch.long))
    generated: list[int] = []
    for _ in range(count):
        token = logits.float().index_fill(-1, torch.tensor([PAD_TOKEN_ID]), float("-inf")).argmax(
            dim=-1, keepdim=True
        )
        generated.append(int(token))
        logits = decoder.step(token)
    return tuple(generated)


class StackStatesTest(unittest.TestCase):
    def test_stacking_then_selecting_returns_the_original_rows(self) -> None:
        model = build_model()
        decoder = BatchDecoder(model)
        decoder.prefill(torch.tensor([prompt(6), prompt(6, 3)], dtype=torch.long))
        batch_state = to_static_state(decoder.state, 4, 4)
        rows = [select_state_rows(batch_state, (index,)) for index in range(2)]
        restacked = stack_states(rows)
        for field_name in KOEMI_STATE_FIELDS:
            self.assertTrue(torch.equal(getattr(restacked, field_name), getattr(batch_state, field_name)))

    def test_it_carries_the_largest_step_index(self) -> None:
        model = build_model()
        decoder = BatchDecoder(model)
        decoder.prefill(torch.tensor([prompt(6)], dtype=torch.long))
        state = to_static_state(decoder.state, 4, 4)
        first = select_state_rows(state, (0,), step_index=3)
        second = select_state_rows(state, (0,), step_index=91)
        self.assertEqual(stack_states([first, second]).step_index, 91)

    def test_it_rejects_rings_that_were_never_padded_to_capacity(self) -> None:
        model = build_model()
        short = model.initial_state(1, torch.device("cpu"))
        padded = to_static_state(short, 4, 4)
        with self.assertRaises(ValueError):
            stack_states([short, padded])

    def test_it_rejects_an_empty_sequence(self) -> None:
        with self.assertRaises(ValueError):
            stack_states([])


class StepIndexIndependenceTest(unittest.TestCase):
    def test_a_row_position_never_reaches_the_computation(self) -> None:
        from dataclasses import replace

        model = build_model()
        input_ids = torch.randint(1, PAD_TOKEN_ID, (2, 6))
        with torch.no_grad():
            base = model(input_ids)
            shifted = model(input_ids, replace(model.initial_state(2, input_ids.device), step_index=9_999))
        self.assertTrue(torch.equal(base.logits, shifted.logits))
        for field_name in KOEMI_STATE_FIELDS:
            self.assertTrue(torch.equal(getattr(base.state, field_name), getattr(shifted.state, field_name)))


class ContinuousBatchingEquivalenceTest(unittest.TestCase):
    def test_every_request_matches_decoding_it_alone(self) -> None:
        model = build_model()
        prompts = {"a": prompt(5), "b": prompt(9, 11), "c": prompt(3, 40)}
        expected = {name: reference_greedy(model, ids, 6) for name, ids in prompts.items()}
        engine = ServingEngine(model)
        outcomes = engine.generate(
            [
                GenerationRequest(name, ids, max_new_tokens=6, policy=greedy())
                for name, ids in prompts.items()
            ]
        )
        for name in prompts:
            self.assertEqual(outcomes[name].token_ids, expected[name], f"request {name} diverged")

    def test_a_request_joining_a_running_batch_still_matches(self) -> None:
        model = build_model()
        early_prompt = prompt(7)
        late_prompt = prompt(4, 23)
        expected_early = reference_greedy(model, early_prompt, 8)
        expected_late = reference_greedy(model, late_prompt, 3)
        engine = ServingEngine(model)
        early = engine.submit(GenerationRequest("early", early_prompt, max_new_tokens=8, policy=greedy()))
        for _ in range(4):
            engine.step()
        late = engine.submit(GenerationRequest("late", late_prompt, max_new_tokens=3, policy=greedy()))
        engine.run_until_idle(max_steps=40)
        self.assertEqual(early.wait(0.0).token_ids, expected_early)
        self.assertEqual(late.wait(0.0).token_ids, expected_late)

    def test_a_row_leaving_early_does_not_disturb_the_rest(self) -> None:
        model = build_model()
        prompts = {"short": prompt(5), "long": prompt(6, 17)}
        expected_long = reference_greedy(model, prompts["long"], 9)
        engine = ServingEngine(model)
        outcomes = engine.generate(
            [
                GenerationRequest("short", prompts["short"], max_new_tokens=2, policy=greedy()),
                GenerationRequest("long", prompts["long"], max_new_tokens=9, policy=greedy()),
            ]
        )
        self.assertEqual(len(outcomes["short"].token_ids), 2)
        self.assertEqual(outcomes["long"].token_ids, expected_long)

    def test_it_agrees_with_generate_batch_on_a_shared_prompt(self) -> None:
        model = build_model()
        shared = prompt(6)
        batched = generate_batch(
            model,
            ByteTokenizer(),
            ["ignored", "ignored"],
            max_new_tokens=5,
            policy=greedy(),
            prompt_token_ids=[list(shared), list(shared)],
        )
        engine = ServingEngine(model)
        outcomes = engine.generate(
            [GenerationRequest("one", shared, max_new_tokens=5, policy=greedy())]
        )
        self.assertEqual(outcomes["one"].token_ids, tuple(batched.token_ids[0]))


class StopConditionTest(unittest.TestCase):
    def test_the_token_limit_closes_the_stream(self) -> None:
        engine = ServingEngine(build_model())
        outcomes = engine.generate([GenerationRequest("a", prompt(4), max_new_tokens=3, policy=greedy())])
        self.assertEqual(outcomes["a"].reason, TOKEN_LIMIT)
        self.assertEqual(len(outcomes["a"].token_ids), 3)

    def test_a_stop_token_ends_generation_early(self) -> None:
        model = build_model()
        prompt_ids = prompt(5)
        expected = reference_greedy(model, prompt_ids, 6)
        engine = ServingEngine(model)
        outcomes = engine.generate(
            [
                GenerationRequest(
                    "a", prompt_ids, max_new_tokens=6, policy=greedy(), stop_token_ids=(expected[1],)
                )
            ]
        )
        self.assertEqual(outcomes["a"].reason, STOPPED)
        self.assertEqual(outcomes["a"].token_ids, expected[:2])

    def test_a_queued_request_can_be_cancelled(self) -> None:
        engine = ServingEngine(build_model())
        stream = engine.submit(GenerationRequest("a", prompt(4), max_new_tokens=50, policy=greedy()))
        self.assertTrue(engine.cancel("a"))
        self.assertEqual(stream.wait(0.0).reason, CANCELLED)
        self.assertEqual(stream.wait(0.0).token_ids, ())

    def test_a_running_request_stops_within_one_step(self) -> None:
        engine = ServingEngine(build_model())
        stream = engine.submit(GenerationRequest("a", prompt(4), max_new_tokens=50, policy=greedy()))
        engine.step()
        engine.step()
        self.assertTrue(engine.cancel("a"))
        engine.step()
        self.assertTrue(stream.closed)
        self.assertEqual(stream.wait(0.0).reason, CANCELLED)

    def test_a_deadline_retires_the_request_with_a_reason(self) -> None:
        ticks = iter([0.0] + [float(index) for index in range(1, 200)])
        engine = ServingEngine(build_model(), clock=lambda: next(ticks))
        outcomes = engine.generate(
            [GenerationRequest("a", prompt(4), max_new_tokens=50, policy=greedy(), deadline_seconds=3.0)]
        )
        self.assertEqual(outcomes["a"].reason, DEADLINE_EXCEEDED)
        self.assertLess(len(outcomes["a"].token_ids), 50)

    def test_cancelling_an_unknown_request_reports_false(self) -> None:
        self.assertFalse(ServingEngine(build_model()).cancel("missing"))


class StreamingTest(unittest.TestCase):
    def test_the_stream_yields_every_token_in_order_then_stops(self) -> None:
        model = build_model()
        prompt_ids = prompt(5)
        expected = reference_greedy(model, prompt_ids, 4)
        engine = ServingEngine(model)
        stream = engine.submit(GenerationRequest("a", prompt_ids, max_new_tokens=4, policy=greedy()))
        engine.run_until_idle(max_steps=20)
        self.assertEqual(tuple(stream), expected)

    def test_tokens_appear_before_the_request_finishes(self) -> None:
        engine = ServingEngine(build_model())
        stream = engine.submit(GenerationRequest("a", prompt(5), max_new_tokens=6, policy=greedy()))
        engine.step()
        engine.step()
        self.assertFalse(stream.closed)
        self.assertIsNone(stream.outcome)

    def test_waiting_past_the_timeout_raises(self) -> None:
        engine = ServingEngine(build_model())
        stream = engine.submit(GenerationRequest("a", prompt(5), max_new_tokens=6, policy=greedy()))
        with self.assertRaises(TimeoutError):
            stream.wait(0.01)


class AdmissionControlTest(unittest.TestCase):
    def test_an_oversized_prompt_is_rejected_at_submit(self) -> None:
        engine = ServingEngine(build_model(), limits=ServingLimits(max_prompt_tokens=4))
        with self.assertRaises(ValueError):
            engine.submit(GenerationRequest("a", prompt(9), max_new_tokens=2))
        self.assertEqual(engine.metrics().rejected_requests, 1)
        self.assertEqual(engine.metrics().waiting_requests, 0)

    def test_an_oversized_token_budget_is_rejected_at_submit(self) -> None:
        engine = ServingEngine(build_model(), limits=ServingLimits(max_new_tokens=4))
        with self.assertRaises(ValueError):
            engine.submit(GenerationRequest("a", prompt(4), max_new_tokens=99))

    def test_a_token_outside_the_content_vocabulary_is_rejected(self) -> None:
        engine = ServingEngine(build_model())
        for bad_prompt in ((PAD_TOKEN_ID,), (-1,), (99_999,)):
            with self.subTest(prompt=bad_prompt):
                with self.assertRaises(ValueError):
                    engine.submit(GenerationRequest("a", bad_prompt, max_new_tokens=2))

    def test_concurrency_is_capped_and_the_rest_waits(self) -> None:
        engine = ServingEngine(build_model(), limits=ServingLimits(max_concurrent_sequences=2))
        for index in range(5):
            engine.submit(GenerationRequest(f"r{index}", prompt(4, index), max_new_tokens=3, policy=greedy()))
        engine.step()
        self.assertEqual(engine.metrics().active_sequences, 2)
        self.assertEqual(engine.metrics().waiting_requests, 3)
        engine.run_until_idle(max_steps=60)
        self.assertEqual(engine.metrics().completed_requests, 5)
        self.assertLessEqual(engine.metrics().peak_active_sequences, 2)

    def test_a_duplicate_request_id_is_refused(self) -> None:
        engine = ServingEngine(build_model())
        engine.submit(GenerationRequest("a", prompt(4), max_new_tokens=2, policy=greedy()))
        with self.assertRaises(ValueError):
            engine.submit(GenerationRequest("a", prompt(4), max_new_tokens=2, policy=greedy()))

    def test_a_training_model_is_refused(self) -> None:
        model = build_model()
        model.train()
        with self.assertRaises(RuntimeError):
            ServingEngine(model)

    def test_limits_reject_non_positive_values(self) -> None:
        for field_name in ("max_concurrent_sequences", "max_prompt_tokens", "max_new_tokens"):
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    ServingLimits(**{field_name: 0})


class IsolationTest(unittest.TestCase):
    def test_a_long_running_neighbour_cannot_change_another_result(self) -> None:
        model = build_model()
        alone = reference_greedy(model, prompt(5), 4)
        engine = ServingEngine(model)
        outcomes = engine.generate(
            [
                GenerationRequest("target", prompt(5), max_new_tokens=4, policy=greedy()),
                GenerationRequest("noisy", prompt(12, 31), max_new_tokens=20, policy=greedy()),
            ]
        )
        self.assertEqual(outcomes["target"].token_ids, alone)

    def test_metrics_count_every_generated_token(self) -> None:
        engine = ServingEngine(build_model())
        engine.generate(
            [
                GenerationRequest("a", prompt(4), max_new_tokens=3, policy=greedy()),
                GenerationRequest("b", prompt(4, 5), max_new_tokens=5, policy=greedy()),
            ]
        )
        metrics = engine.metrics()
        self.assertEqual(metrics.generated_tokens, 8)
        self.assertEqual(metrics.completed_requests, 2)
        self.assertGreater(metrics.tokens_per_decode_step, 1.0)


if __name__ == "__main__":
    unittest.main()
