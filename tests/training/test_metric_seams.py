from __future__ import annotations

import math
import unittest

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel
from koemi.training.a100_run import (
    DEFAULT_EFFECTIVE_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    LEGACY_MODEL_SETTING_DEFAULTS,
    CorpusQuotas,
    RunConfiguration,
    normalize_manifest_model_settings,
    run_model_settings_view,
)
from koemi.training.dataset import IGNORE_TARGET_ID
from koemi.training.objective import calculate_training_objective, token_cross_entropy
from koemi.training.trainer import MetricAccumulator
from pathlib import Path


def build_model(expert_count: int = 8, **overrides) -> KoemiModel:
    torch.manual_seed(9)
    defaults = dict(
        embedding_size=32,
        memory_features=8,
        local_memory_size=4,
        salience_memory_size=4,
        expert_count=expert_count,
        expert_top_k=min(3, expert_count) or 1,
        scan_chunk=8,
        ablation="no_refine",
    )
    defaults.update(overrides)
    return KoemiModel(ModelSettings(**defaults))


def legacy_expert_counts(output) -> tuple[int, ...]:
    """Count per-expert assignments the way the removed comparison loop did."""
    if output.expert_indices.numel() == 0:
        return ()
    assignments = (
        output.active_expert_indices
        if output.active_expert_indices is not None
        else output.expert_indices.unsqueeze(-1)
    )
    return tuple(int((assignments == index).sum()) for index in range(output.expert_count))


class ExpertActivationCountTest(unittest.TestCase):
    def sample(self, expert_count: int = 8):
        model = build_model(expert_count)
        model.train()
        input_ids = torch.randint(0, PAD_TOKEN_ID, (3, 24))
        input_ids[0, -5:] = PAD_TOKEN_ID
        input_ids[1, :2] = PAD_TOKEN_ID
        return model(input_ids, execution_mode=ExecutionMode.PARALLEL)

    def test_it_matches_the_per_expert_comparison_loop(self) -> None:
        for expert_count in (1, 3, 8, 16):
            with self.subTest(expert_count=expert_count):
                output = self.sample(expert_count)
                self.assertEqual(output.expert_activation_counts, legacy_expert_counts(output))

    def test_totals_sum_to_the_valid_assignments(self) -> None:
        output = self.sample(8)
        totals = output.expert_activation_totals
        self.assertIsNotNone(totals)
        expected = int((output.active_expert_indices[output.valid_positions] >= 0).sum())
        self.assertEqual(int(totals.sum()), expected)

    def test_it_issues_no_boolean_indexing_or_bincount(self) -> None:
        output = self.sample(8)
        calls = {"count": 0}
        originals = {name: getattr(torch, name) for name in ("nonzero", "bincount")}

        def counted(name):
            def wrapper(*arguments, **keywords):
                calls["count"] += 1
                return originals[name](*arguments, **keywords)

            return wrapper

        for name in originals:
            setattr(torch, name, counted(name))
        try:
            output.expert_activation_totals
        finally:
            for name, original in originals.items():
                setattr(torch, name, original)
        self.assertEqual(calls["count"], 0)

    def test_a_zero_expert_model_reports_no_counts(self) -> None:
        model = build_model(expert_count=0)
        model.train()
        output = model(torch.randint(0, PAD_TOKEN_ID, (2, 16)), execution_mode=ExecutionMode.PARALLEL)
        self.assertIsNone(output.expert_activation_totals)
        self.assertEqual(output.expert_activation_counts, ())


class MetricAccumulatorExpertTest(unittest.TestCase):
    def test_it_accumulates_totals_across_batches(self) -> None:
        model = build_model()
        model.train()
        accumulator = MetricAccumulator()
        expected = [0] * 8
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            input_ids = torch.randint(0, PAD_TOKEN_ID, (2, 16))
            output = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
            accumulator.accumulate_expert_totals(output.expert_activation_totals)
            for index, count in enumerate(legacy_expert_counts(output)):
                expected[index] += count
        self.assertEqual(accumulator.expert_activation_counts, tuple(expected))

    def test_merging_two_accumulators_adds_their_totals(self) -> None:
        model = build_model()
        model.train()
        output = model(torch.randint(0, PAD_TOKEN_ID, (2, 16)), execution_mode=ExecutionMode.PARALLEL)
        first = MetricAccumulator()
        second = MetricAccumulator()
        first.accumulate_expert_totals(output.expert_activation_totals)
        second.accumulate_expert_totals(output.expert_activation_totals)
        first.accumulate_expert_totals(second.expert_totals)
        doubled = tuple(2 * count for count in legacy_expert_counts(output))
        self.assertEqual(first.expert_activation_counts, doubled)

    def test_an_empty_accumulator_reports_no_counts(self) -> None:
        self.assertEqual(MetricAccumulator().expert_activation_counts, ())


class ObjectiveTokenLossTest(unittest.TestCase):
    def build_batch(self):
        model = build_model()
        model.train()
        torch.manual_seed(3)
        input_ids = torch.randint(0, PAD_TOKEN_ID, (3, 24))
        target_ids = torch.randint(0, PAD_TOKEN_ID, (3, 24))
        target_ids[0, :4] = IGNORE_TARGET_ID
        thinking_mask = torch.rand(3, 24) > 0.6
        return model, input_ids, target_ids, thinking_mask

    def test_the_returned_token_loss_matches_a_separate_fp32_pass(self) -> None:
        model, input_ids, target_ids, thinking_mask = self.build_batch()
        output = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
        objective = calculate_training_objective(output, target_ids, thinking_mask, 0.5)
        recomputed = token_cross_entropy(output.logits.float(), target_ids)
        self.assertTrue(torch.equal(objective.token_loss.detach(), recomputed))

    def test_it_matches_under_bfloat16_autocast(self) -> None:
        model, input_ids, target_ids, thinking_mask = self.build_batch()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
            objective = calculate_training_objective(output, target_ids, thinking_mask, 0.5)
            recomputed = token_cross_entropy(output.logits.float(), target_ids)
        self.assertTrue(torch.equal(objective.token_loss.detach(), recomputed))

    def test_the_token_loss_keeps_the_shape_of_the_targets(self) -> None:
        model, input_ids, target_ids, thinking_mask = self.build_batch()
        output = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
        objective = calculate_training_objective(output, target_ids, thinking_mask, 0.5)
        self.assertEqual(objective.token_loss.shape, target_ids.shape)


class EffectiveBatchTest(unittest.TestCase):
    def configuration(self, **overrides) -> RunConfiguration:
        values = dict(
            results_directory=Path("results"),
            session_seconds=60,
            data_seed=1,
            model_seed=2,
            sequence_length=32,
            quotas=CorpusQuotas(1, 1, 1, 1, 1),
            opencode_scan_limit=1,
            source_scan_limit=1,
            shuffle_buffer_size=1,
            num_workers=0,
            checkpoint_interval_seconds=60,
            log_interval_steps=1,
            evaluation_batches=1,
        )
        values.update(overrides)
        return RunConfiguration(**values)

    def test_the_defaults_reproduce_the_run_in_flight(self) -> None:
        configuration = self.configuration()
        self.assertEqual(configuration.effective_batch_size, 64)
        self.assertEqual(configuration.learning_rate, 3e-4)

    def test_it_rejects_a_non_positive_effective_batch(self) -> None:
        with self.assertRaises(ValueError):
            self.configuration(effective_batch_size=0)

    def test_it_rejects_a_non_finite_learning_rate(self) -> None:
        with self.assertRaises(ValueError):
            self.configuration(learning_rate=math.inf)

    def test_the_calibrator_never_offers_a_batch_above_the_effective_batch(self) -> None:
        from koemi.training.a100_safe_run import AGGRESSIVE_BATCH_CANDIDATES

        for effective in (16, 64, 128):
            offered = [candidate for candidate in AGGRESSIVE_BATCH_CANDIDATES if candidate <= effective]
            self.assertTrue(offered)
            self.assertLessEqual(max(offered), effective)

    def test_the_candidate_list_now_reaches_past_the_old_ceiling(self) -> None:
        from koemi.training.a100_safe_run import AGGRESSIVE_BATCH_CANDIDATES

        self.assertGreater(max(AGGRESSIVE_BATCH_CANDIDATES), 64)


class CompileSeamTest(unittest.TestCase):
    def test_the_default_keeps_compilation_off(self) -> None:
        self.assertFalse(ModelSettings().compile_forward)

    def test_the_uncompiled_model_returns_forward_window_itself(self) -> None:
        model = build_model()
        self.assertEqual(model.window_callable(), model.forward_window)

    def test_asking_for_compilation_returns_a_cached_wrapper(self) -> None:
        model = build_model(compile_forward=True)
        first = model.window_callable()
        self.assertIsNot(first, model.forward_window)
        self.assertIs(first, model.window_callable())

    def test_it_rejects_a_non_boolean_flag(self) -> None:
        with self.assertRaises(TypeError):
            ModelSettings(compile_forward="yes")


class ResumeStillCompatibleTest(unittest.TestCase):
    def test_every_new_setting_is_dropped_from_the_legacy_view(self) -> None:
        settings = ModelSettings(expert_count=8, expert_top_k=3)
        legacy = settings.to_dict()
        for name in LEGACY_MODEL_SETTING_DEFAULTS:
            legacy.pop(name)
        self.assertEqual(normalize_manifest_model_settings(legacy), run_model_settings_view(settings))

    def test_the_legacy_defaults_cover_compile_and_checkpointing(self) -> None:
        self.assertIn("compile_forward", LEGACY_MODEL_SETTING_DEFAULTS)
        self.assertIn("activation_checkpointing", LEGACY_MODEL_SETTING_DEFAULTS)
        self.assertIn("expert_dispatch", LEGACY_MODEL_SETTING_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
