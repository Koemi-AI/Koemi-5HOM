from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from koemi.cli import main
from koemi.data.hybrid_tokenizer import MINIMUM_VOCABULARY_SIZE, HybridVocabulary
from koemi.training.checkpoints import CheckpointStore


RECORDS = tuple(
    {
        "id": f"queue-{index:03d}",
        "input": "Explain the queue in one sentence.",
        "thinking": "A queue preserves arrival order.",
        "output": "A queue is first in first out, so the queue keeps arrival order.",
        "metadata": {"source": "test"},
    }
    for index in range(12)
)

MODEL_FLAGS = (
    "--embedding-size",
    "16",
    "--memory-features",
    "4",
    "--local-memory-size",
    "4",
    "--salience-memory-size",
    "3",
    "--scan-chunk",
    "8",
)


def write_dataset(workspace: Path) -> Path:
    dataset_path = workspace / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in RECORDS) + "\n", encoding="utf-8"
    )
    return dataset_path


def capture_stdout():
    output = io.BytesIO()
    stdout = type("BufferedStdout", (), {"buffer": output})()
    return output, patch("koemi.cli.sys.stdout", stdout)


class HybridVocabularyCommandTests(unittest.TestCase):
    def test_a_vocabulary_is_built_trained_and_generated_from(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dataset_path = write_dataset(workspace)
            vocabulary_path = workspace / "vocabulary.json"
            build_status = main(
                [
                    "build-vocabulary",
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(vocabulary_path),
                    "--vocabulary-size",
                    str(MINIMUM_VOCABULARY_SIZE + 48),
                ]
            )
            self.assertEqual(0, build_status)
            vocabulary = HybridVocabulary.load(vocabulary_path)
            self.assertGreater(len(vocabulary.merges), 0)

            checkpoint_path = workspace / "model.pt"
            train_status = main(
                [
                    "train",
                    "--dataset",
                    str(dataset_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--vocabulary",
                    str(vocabulary_path),
                    "--epochs",
                    "1",
                    "--sequence-length",
                    "32",
                    *MODEL_FLAGS,
                ]
            )
            self.assertEqual(0, train_status)
            loaded = CheckpointStore().load(checkpoint_path)
            self.assertEqual(vocabulary.vocabulary_size, loaded.model_settings.vocabulary_size)
            self.assertEqual(vocabulary.merges, loaded.vocabulary.merges)

            output, stdout_patch = capture_stdout()
            with stdout_patch:
                generate_status = main(
                    [
                        "generate",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--prompt",
                        "Explain the queue in one sentence.",
                        "--max-new-bytes",
                        "6",
                    ]
                )
            self.assertEqual(0, generate_status)
            self.assertTrue(output.getvalue())

    def test_a_byte_checkpoint_is_expanded_to_a_hybrid_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dataset_path = write_dataset(workspace)
            vocabulary_path = workspace / "vocabulary.json"
            self.assertEqual(
                0,
                main(
                    [
                        "build-vocabulary",
                        "--dataset",
                        str(dataset_path),
                        "--output",
                        str(vocabulary_path),
                        "--vocabulary-size",
                        str(MINIMUM_VOCABULARY_SIZE + 32),
                    ]
                ),
            )
            byte_checkpoint = workspace / "byte.pt"
            self.assertEqual(
                0,
                main(
                    [
                        "train",
                        "--dataset",
                        str(dataset_path),
                        "--checkpoint",
                        str(byte_checkpoint),
                        "--epochs",
                        "1",
                        "--sequence-length",
                        "32",
                        *MODEL_FLAGS,
                    ]
                ),
            )
            hybrid_checkpoint = workspace / "hybrid.pt"
            output, stdout_patch = capture_stdout()
            with stdout_patch:
                expand_status = main(
                    [
                        "expand-vocabulary",
                        "--checkpoint",
                        str(byte_checkpoint),
                        "--vocabulary",
                        str(vocabulary_path),
                        "--output",
                        str(hybrid_checkpoint),
                    ]
                )
            self.assertEqual(0, expand_status)
            report = json.loads(output.getvalue().decode("utf-8"))
            self.assertEqual(257, report["previous_vocabulary_size"])
            self.assertEqual(MINIMUM_VOCABULARY_SIZE + 32, report["vocabulary_size"])
            migrated = CheckpointStore().load(hybrid_checkpoint)
            self.assertIsNotNone(migrated.vocabulary)

            generated, generated_patch = capture_stdout()
            with generated_patch:
                generate_status = main(
                    [
                        "generate",
                        "--checkpoint",
                        str(hybrid_checkpoint),
                        "--prompt",
                        "Explain the queue in one sentence.",
                        "--max-new-bytes",
                        "4",
                    ]
                )
            self.assertEqual(0, generate_status)
            self.assertTrue(generated.getvalue())

    def test_building_over_an_existing_file_needs_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dataset_path = write_dataset(workspace)
            vocabulary_path = workspace / "vocabulary.json"
            vocabulary_path.write_text("{}", encoding="utf-8")
            status = main(
                [
                    "build-vocabulary",
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(vocabulary_path),
                ]
            )
            self.assertEqual(2, status)


class FastDecodeCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        workspace = Path(self.directory.name)
        dataset_path = write_dataset(workspace)
        self.checkpoint_path = workspace / "model.pt"
        self.assertEqual(
            0,
            main(
                [
                    "train",
                    "--dataset",
                    str(dataset_path),
                    "--checkpoint",
                    str(self.checkpoint_path),
                    "--epochs",
                    "1",
                    "--sequence-length",
                    "32",
                    *MODEL_FLAGS,
                ]
            ),
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def generate(self, *extra: str) -> tuple[int, bytes]:
        output, stdout_patch = capture_stdout()
        with stdout_patch:
            status = main(
                [
                    "generate",
                    "--checkpoint",
                    str(self.checkpoint_path),
                    "--prompt",
                    "Explain the queue in one sentence.",
                    "--max-new-bytes",
                    "8",
                    *extra,
                ]
            )
        return status, output.getvalue()

    def test_fast_decode_generates(self) -> None:
        status, output = self.generate("--fast-decode")
        self.assertEqual(0, status)
        self.assertTrue(output)

    def test_greedy_fast_decode_is_repeatable(self) -> None:
        first = self.generate("--fast-decode", "--greedy")
        second = self.generate("--fast-decode", "--greedy")
        self.assertEqual(0, first[0])
        self.assertEqual(first[1], second[1])

    def test_ngram_speculation_generates(self) -> None:
        status, output = self.generate("--ngram-draft", "--greedy", "--draft-length", "3")
        self.assertEqual(0, status)
        self.assertTrue(output)

    def test_model_drafting_generates(self) -> None:
        status, output = self.generate(
            "--draft-checkpoint", str(self.checkpoint_path), "--greedy", "--draft-length", "2"
        )
        self.assertEqual(0, status)
        self.assertTrue(output)

    def test_speculation_matches_the_greedy_fast_path(self) -> None:
        baseline = self.generate("--fast-decode", "--greedy")
        speculative = self.generate("--ngram-draft", "--greedy", "--draft-length", "4")
        self.assertEqual(baseline[1], speculative[1])

    def test_a_cuda_graph_without_fast_decode_is_refused(self) -> None:
        status, _ = self.generate("--cuda-graph")
        self.assertEqual(2, status)

    def test_two_draft_sources_are_refused(self) -> None:
        status, _ = self.generate(
            "--ngram-draft", "--draft-checkpoint", str(self.checkpoint_path)
        )
        self.assertEqual(2, status)

    def test_a_prefix_cache_with_fast_decode_is_refused(self) -> None:
        status, _ = self.generate(
            "--fast-decode",
            "--bulk-prefix-cache",
            str(Path(self.directory.name) / "bulk"),
            "--bulk-prefix-cache-namespace",
            "test",
        )
        self.assertEqual(2, status)


if __name__ == "__main__":
    unittest.main()
