from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.data.hybrid_tokenizer import FIRST_MERGE_ID, HybridTokenizer, HybridVocabulary
from koemi.model.network import KoemiModel
from koemi.training.checkpoints import CheckpointStore, expand_model_vocabulary


def build_model(vocabulary_size: int = PAD_TOKEN_ID + 1) -> KoemiModel:
    model = KoemiModel(
        ModelSettings(
            vocabulary_size=vocabulary_size,
            embedding_size=16,
            memory_features=4,
            local_memory_size=4,
            salience_memory_size=3,
            expert_count=2,
            scan_chunk=8,
        )
    )
    model.eval()
    return model


def build_vocabulary() -> HybridVocabulary:
    return HybridVocabulary(((113, 117), (FIRST_MERGE_ID, 101)))


class VocabularyExpansionTests(unittest.TestCase):
    def test_the_existing_rows_are_copied_without_change(self) -> None:
        torch.manual_seed(41)
        model = build_model()
        vocabulary = build_vocabulary()
        expanded = expand_model_vocabulary(model, vocabulary)
        carried = model.settings.vocabulary_size
        self.assertEqual(vocabulary.vocabulary_size, expanded.settings.vocabulary_size)
        self.assertTrue(torch.equal(model.embedding.weight, expanded.embedding.weight[:carried]))
        self.assertTrue(
            torch.equal(model.token_predictor.weight, expanded.token_predictor.weight[:carried])
        )
        self.assertTrue(
            torch.equal(model.token_predictor.bias, expanded.token_predictor.bias[:carried])
        )
        self.assertTrue(
            torch.equal(model.recurrent_state.retention_projection.weight, expanded.recurrent_state.retention_projection.weight)
        )

    def test_the_existing_ids_keep_their_logits_without_surprise(self) -> None:
        torch.manual_seed(41)
        model = KoemiModel(
            ModelSettings(
                vocabulary_size=PAD_TOKEN_ID + 1,
                embedding_size=16,
                memory_features=4,
                local_memory_size=4,
                salience_memory_size=3,
                expert_count=2,
                scan_chunk=8,
                ablation="no_surprise",
            )
        )
        model.eval()
        vocabulary = build_vocabulary()
        expanded = expand_model_vocabulary(model, vocabulary)
        input_ids = torch.tensor([[113, 117, 101, 117, 101]], dtype=torch.long)
        with torch.no_grad():
            original = model(input_ids)
            migrated = expanded(input_ids)
        self.assertTrue(
            torch.allclose(
                original.logits, migrated.logits[..., : model.settings.vocabulary_size], atol=1e-5
            )
        )

    def test_surprise_changes_when_the_vocabulary_grows(self) -> None:
        torch.manual_seed(48)
        model = build_model()
        expanded = expand_model_vocabulary(model, build_vocabulary())
        input_ids = torch.tensor([[113, 117, 101, 117, 101]], dtype=torch.long)
        with torch.no_grad():
            original = model(input_ids)
            migrated = expanded(input_ids)
        self.assertFalse(torch.allclose(original.surprise_values, migrated.surprise_values, atol=1e-6))

    def test_a_new_row_starts_at_the_mean_of_its_bytes(self) -> None:
        torch.manual_seed(42)
        model = build_model()
        vocabulary = build_vocabulary()
        expanded = expand_model_vocabulary(model, vocabulary)
        expected = model.embedding.weight[torch.tensor([113, 117])].mean(dim=0)
        self.assertTrue(torch.allclose(expected, expanded.embedding.weight[FIRST_MERGE_ID], atol=1e-6))
        expected_head = model.token_predictor.weight[torch.tensor([113, 117, 101])].mean(dim=0)
        self.assertTrue(
            torch.allclose(expected_head, expanded.token_predictor.weight[FIRST_MERGE_ID + 1], atol=1e-6)
        )

    def test_the_padding_row_stays_zero(self) -> None:
        torch.manual_seed(43)
        expanded = expand_model_vocabulary(build_model(), build_vocabulary())
        self.assertTrue(torch.all(expanded.embedding.weight[PAD_TOKEN_ID] == 0.0))

    def test_a_smaller_vocabulary_is_refused(self) -> None:
        model = build_model(PAD_TOKEN_ID + 40)
        with self.assertRaises(ValueError):
            expand_model_vocabulary(model, HybridVocabulary(()))

    def test_an_equal_vocabulary_returns_the_same_model(self) -> None:
        vocabulary = build_vocabulary()
        model = build_model(vocabulary.vocabulary_size)
        self.assertIs(model, expand_model_vocabulary(model, vocabulary))

    def test_the_expanded_model_runs_on_hybrid_ids(self) -> None:
        torch.manual_seed(44)
        vocabulary = build_vocabulary()
        expanded = expand_model_vocabulary(build_model(), vocabulary)
        tokenizer = HybridTokenizer(vocabulary)
        token_ids = tokenizer.encode("queue")
        self.assertTrue(any(token_id >= FIRST_MERGE_ID for token_id in token_ids))
        with torch.no_grad():
            output = expanded(torch.tensor([token_ids], dtype=torch.long))
        self.assertEqual(vocabulary.vocabulary_size, output.logits.shape[-1])
        self.assertTrue(torch.isfinite(output.logits).all())


class CheckpointVocabularyTests(unittest.TestCase):
    def test_a_checkpoint_carries_the_vocabulary(self) -> None:
        torch.manual_seed(45)
        vocabulary = build_vocabulary()
        model = build_model(vocabulary.vocabulary_size)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            CheckpointStore().save(path, model, vocabulary=vocabulary)
            loaded = CheckpointStore().load(path)
        self.assertIsNotNone(loaded.vocabulary)
        self.assertEqual(vocabulary.merges, loaded.vocabulary.merges)

    def test_a_byte_checkpoint_loads_without_a_vocabulary(self) -> None:
        torch.manual_seed(46)
        model = build_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            CheckpointStore().save(path, model)
            loaded = CheckpointStore().load(path)
        self.assertIsNone(loaded.vocabulary)
        self.assertEqual(PAD_TOKEN_ID + 1, loaded.model_settings.vocabulary_size)

    def test_a_vocabulary_that_does_not_match_the_head_is_refused(self) -> None:
        torch.manual_seed(47)
        model = build_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            with self.assertRaises(ValueError):
                CheckpointStore().save(path, model, vocabulary=build_vocabulary())


if __name__ == "__main__":
    unittest.main()
