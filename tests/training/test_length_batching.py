from __future__ import annotations

import unittest
from pathlib import Path

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.data.contracts import DatasetRecord
from koemi.training.a100_run import (
    DEFAULT_LENGTH_BUCKET_SIZE,
    LEGACY_BATCHING,
    LEGACY_MODEL_SETTING_DEFAULTS,
    DeterministicBatchSampler,
    LengthBucketedBatchSampler,
    MaterializedCausalByteDataset,
    RunConfiguration,
    collate_materialized_chunks,
    create_loader,
    fingerprint,
    normalize_manifest_model_settings,
    run_batching_view,
    run_model_settings_view,
)
from koemi.training.a100_run import CorpusQuotas


def build_dataset(sequence_length: int = 32) -> MaterializedCausalByteDataset:
    records = []
    for index in range(60):
        body_length = 8 + (index * 37) % 190
        records.append(
            DatasetRecord(
                identifier=f"record-{index}",
                input_text=f"question {index}",
                thinking_text="reason " * (1 + index % 3),
                output_text="x" * body_length,
                metadata={},
                system_text="You are precise.",
            )
        )
    return MaterializedCausalByteDataset(records, sequence_length)


def padding_fraction(dataset: MaterializedCausalByteDataset, batches) -> float:
    padded = 0
    real = 0
    for batch in batches:
        chunks = [dataset[index] for index in batch]
        collated = collate_materialized_chunks(chunks)
        padded += collated["input_ids"].numel()
        real += int((collated["input_ids"] != PAD_TOKEN_ID).sum())
    return (padded - real) / padded


class LengthBucketedSamplerTest(unittest.TestCase):
    def test_it_cuts_padding_against_the_index_sampler(self) -> None:
        dataset = build_dataset()
        index_batches = list(DeterministicBatchSampler(len(dataset), 8, 7, 0, 0))
        length_batches = list(LengthBucketedBatchSampler(dataset.chunk_lengths(), 8, 7, 0, 0, 16))
        index_padding = padding_fraction(dataset, index_batches)
        length_padding = padding_fraction(dataset, length_batches)
        self.assertLess(length_padding, index_padding)

    def test_it_covers_every_chunk_exactly_once(self) -> None:
        dataset = build_dataset()
        batches = list(LengthBucketedBatchSampler(dataset.chunk_lengths(), 8, 7, 0, 0, 16))
        visited = [index for batch in batches for index in batch]
        self.assertEqual(sorted(visited), list(range(len(dataset))))

    def test_the_plan_is_reproducible_for_the_same_seed_and_epoch(self) -> None:
        lengths = build_dataset().chunk_lengths()
        first = list(LengthBucketedBatchSampler(lengths, 8, 7, 1, 0, 16))
        second = list(LengthBucketedBatchSampler(lengths, 8, 7, 1, 0, 16))
        self.assertEqual(first, second)

    def test_a_different_epoch_reorders_the_batches(self) -> None:
        lengths = build_dataset().chunk_lengths()
        first = list(LengthBucketedBatchSampler(lengths, 8, 7, 0, 0, 16))
        second = list(LengthBucketedBatchSampler(lengths, 8, 7, 1, 0, 16))
        self.assertNotEqual(first, second)

    def test_a_resume_starts_on_the_same_batch_the_full_plan_holds(self) -> None:
        lengths = build_dataset().chunk_lengths()
        full = list(LengthBucketedBatchSampler(lengths, 8, 7, 0, 0, 16))
        resumed = list(LengthBucketedBatchSampler(lengths, 8, 7, 0, 5, 16))
        self.assertEqual(resumed, full[5:])

    def test_no_batch_crosses_its_length_bucket(self) -> None:
        dataset = build_dataset()
        lengths = dataset.chunk_lengths()
        bucket_size = 16
        for batch in LengthBucketedBatchSampler(lengths, 8, 7, 0, 0, bucket_size):
            keys = {(lengths[index] - 1) // bucket_size for index in batch}
            self.assertEqual(len(keys), 1)

    def test_a_start_index_past_the_plan_is_rejected(self) -> None:
        lengths = build_dataset().chunk_lengths()
        count = len(LengthBucketedBatchSampler(lengths, 8, 7, 0, 0, 16))
        with self.assertRaises(ValueError):
            LengthBucketedBatchSampler(lengths, 8, 7, 0, count + 1, 16)


class ChunkLengthTest(unittest.TestCase):
    def test_chunk_lengths_match_the_materialized_inputs(self) -> None:
        dataset = build_dataset()
        lengths = dataset.chunk_lengths()
        self.assertEqual(len(lengths), len(dataset))
        for index in range(len(dataset)):
            self.assertEqual(lengths[index], len(dataset[index][0]))


class LoaderSelectionTest(unittest.TestCase):
    def test_the_loader_defaults_to_the_index_sampler(self) -> None:
        dataset = build_dataset()
        loader = create_loader(dataset, 8, 7, 0, 0, 0)
        self.assertIsInstance(loader.batch_sampler, DeterministicBatchSampler)

    def test_the_loader_selects_the_length_sampler_on_request(self) -> None:
        dataset = build_dataset()
        loader = create_loader(dataset, 8, 7, 0, 0, 0, "length", 16)
        self.assertIsInstance(loader.batch_sampler, LengthBucketedBatchSampler)

    def test_an_unknown_batching_is_rejected(self) -> None:
        dataset = build_dataset()
        with self.assertRaises(ValueError):
            create_loader(dataset, 8, 7, 0, 0, 0, "sorted", 16)


class ResumeCompatibilityTest(unittest.TestCase):
    def legacy_settings_dict(self) -> dict:
        values = ModelSettings(expert_count=8, expert_top_k=3).to_dict()
        for name in LEGACY_MODEL_SETTING_DEFAULTS:
            values.pop(name)
        return values

    def test_a_manifest_written_before_expert_dispatch_still_matches(self) -> None:
        settings = ModelSettings(expert_count=8, expert_top_k=3)
        self.assertEqual(
            normalize_manifest_model_settings(self.legacy_settings_dict()),
            run_model_settings_view(settings),
        )

    def test_turning_segments_on_stops_matching_a_legacy_manifest(self) -> None:
        settings = ModelSettings(expert_count=8, expert_top_k=3, expert_dispatch="segments")
        self.assertNotEqual(
            normalize_manifest_model_settings(self.legacy_settings_dict()),
            run_model_settings_view(settings),
        )

    def test_the_run_signature_is_unchanged_for_a_legacy_run(self) -> None:
        settings = ModelSettings(expert_count=8, expert_top_k=3)
        legacy_contract = {"model_settings": self.legacy_settings_dict()}
        current_contract = {
            "model_settings": run_model_settings_view(settings),
            **run_batching_view(LEGACY_BATCHING, DEFAULT_LENGTH_BUCKET_SIZE),
        }
        self.assertEqual(fingerprint(legacy_contract), fingerprint(current_contract))

    def test_length_batching_changes_the_run_signature(self) -> None:
        settings = ModelSettings(expert_count=8, expert_top_k=3)
        legacy_contract = {"model_settings": self.legacy_settings_dict()}
        length_contract = {
            "model_settings": run_model_settings_view(settings),
            **run_batching_view("length", 64),
        }
        self.assertNotEqual(fingerprint(legacy_contract), fingerprint(length_contract))


class RunConfigurationTest(unittest.TestCase):
    def configuration(self, **overrides):
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

    def test_it_defaults_to_index_batching(self) -> None:
        self.assertEqual(self.configuration().batching, LEGACY_BATCHING)

    def test_it_rejects_an_unknown_batching(self) -> None:
        with self.assertRaises(ValueError):
            self.configuration(batching="sorted")

    def test_it_rejects_a_non_positive_bucket_size(self) -> None:
        with self.assertRaises(ValueError):
            self.configuration(batching="length", length_bucket_size=0)


if __name__ == "__main__":
    unittest.main()
