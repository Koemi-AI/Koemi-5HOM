from __future__ import annotations

import math
import unittest

import torch
from torch.nn import functional

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel


def build_model(vocabulary_size: int, **overrides) -> KoemiModel:
    settings = dict(
        vocabulary_size=vocabulary_size,
        embedding_size=16,
        memory_features=4,
        local_memory_size=4,
        salience_memory_size=3,
        expert_count=2,
        scan_chunk=6,
        ablation="herm",
    )
    settings.update(overrides)
    model = KoemiModel(ModelSettings(**settings))
    model.eval()
    return model


class HybridVocabularySettingsTests(unittest.TestCase):
    def test_a_vocabulary_below_the_byte_range_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ModelSettings(vocabulary_size=PAD_TOKEN_ID)

    def test_a_larger_vocabulary_is_accepted(self) -> None:
        self.assertEqual(4096, ModelSettings(vocabulary_size=4096).vocabulary_size)


class SurpriseTests(unittest.TestCase):
    def test_the_byte_path_keeps_its_original_surprise(self) -> None:
        torch.manual_seed(51)
        model = build_model(PAD_TOKEN_ID + 1)
        input_ids = torch.randint(0, 255, (2, 7), dtype=torch.long)
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        prior_states = torch.randn(2, 7, model.settings.embedding_size)
        with torch.no_grad():
            produced = model.calculate_surprise(prior_states, input_ids, valid_mask)
            content_logits = model.predict_tokens(prior_states)[..., :PAD_TOKEN_ID]
            token_nll = functional.cross_entropy(
                content_logits.reshape(-1, PAD_TOKEN_ID),
                input_ids.clamp(max=PAD_TOKEN_ID - 1).reshape(-1),
                reduction="none",
            ).view_as(input_ids)
            expected = 1.0 - torch.exp(-token_nll / math.log(PAD_TOKEN_ID))
        self.assertTrue(torch.equal(produced, expected))

    def test_the_hybrid_path_drops_only_the_padding_column(self) -> None:
        torch.manual_seed(52)
        model = build_model(PAD_TOKEN_ID + 9)
        logits = torch.randn(1, 3, model.settings.vocabulary_size)
        content = model.content_logits(logits)
        self.assertEqual(model.settings.vocabulary_size - 1, content.shape[-1])
        self.assertTrue(torch.equal(logits[..., :PAD_TOKEN_ID], content[..., :PAD_TOKEN_ID]))
        self.assertTrue(torch.equal(logits[..., PAD_TOKEN_ID + 1 :], content[..., PAD_TOKEN_ID:]))

    def test_a_hybrid_id_maps_to_its_content_column(self) -> None:
        model = build_model(PAD_TOKEN_ID + 9)
        input_ids = torch.tensor([[0, 255, PAD_TOKEN_ID + 1, PAD_TOKEN_ID + 8]], dtype=torch.long)
        self.assertEqual(
            [[0, 255, PAD_TOKEN_ID, PAD_TOKEN_ID + 7]],
            model.content_token_ids(input_ids).tolist(),
        )

    def test_surprise_stays_bounded_for_a_hybrid_vocabulary(self) -> None:
        torch.manual_seed(53)
        model = build_model(PAD_TOKEN_ID + 9)
        input_ids = torch.tensor([[65, PAD_TOKEN_ID + 3, 66, PAD_TOKEN_ID + 8]], dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids)
        self.assertTrue(torch.all(output.surprise_values >= 0.0))
        self.assertTrue(torch.all(output.surprise_values < 1.0))


class HybridExecutionTests(unittest.TestCase):
    def test_the_paths_agree_on_hybrid_ids(self) -> None:
        torch.manual_seed(54)
        model = build_model(PAD_TOKEN_ID + 17)
        input_ids = torch.tensor(
            [[65, PAD_TOKEN_ID + 4, 66, PAD_TOKEN_ID + 16, 67, 68, PAD_TOKEN_ID + 1, 70]],
            dtype=torch.long,
        )
        with torch.no_grad():
            parallel = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
            sequential = model(input_ids, execution_mode=ExecutionMode.SEQUENTIAL)
        self.assertTrue(torch.allclose(parallel.logits, sequential.logits, atol=1e-4))
        self.assertTrue(torch.allclose(parallel.surprise_values, sequential.surprise_values, atol=1e-4))
        self.assertTrue(
            torch.allclose(parallel.state.memory_basis, sequential.state.memory_basis, atol=1e-4)
        )

    def test_padding_stays_excluded_with_a_hybrid_vocabulary(self) -> None:
        torch.manual_seed(55)
        model = build_model(PAD_TOKEN_ID + 17)
        input_ids = torch.tensor([[65, PAD_TOKEN_ID + 4, PAD_TOKEN_ID, PAD_TOKEN_ID]], dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids)
        self.assertEqual(2, output.token_count)
        self.assertEqual([True, True, False, False], output.valid_positions[0].tolist())
        self.assertEqual([0.0, 0.0], output.surprise_values[0, 2:].tolist())

    def test_an_id_above_the_vocabulary_is_refused(self) -> None:
        model = build_model(PAD_TOKEN_ID + 9)
        with self.assertRaises(ValueError):
            model(torch.tensor([[PAD_TOKEN_ID + 9]], dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
