from __future__ import annotations

import copy
import logging
import unittest

import torch
from torch import nn

from koemi.model.decisions import typed_decisions
from koemi.training.decisions import (DecisionTrainingSettings, decision_loss,
                                      fit_decision_temperature, train_decisions)


class TinyDecisionModel(nn.Module):
    def __init__(self, transformer_head: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Embedding(8, 8)
        self.head = (nn.TransformerEncoder(
            nn.TransformerEncoderLayer(8, 2, 16, dropout=0, batch_first=True, norm_first=True),
            2, enable_nested_tensor=False) if transformer_head else nn.Identity())
        self.scorer = nn.Linear(8, 1)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        values = self.encoder(input_ids)
        if isinstance(self.head, nn.TransformerEncoder):
            values = self.head(values, src_key_padding_mask=~attention_mask.bool())
        else:
            values = self.head(values)
        markers = marker_pos.clamp_min(0).unsqueeze(-1).expand(-1, -1, values.shape[-1])
        return self.scorer(values.gather(1, markers)).squeeze(-1), None


def batch(targets=((0.85, 0.15), (0.15, 0.85), (0.7, 0.3))):
    count = len(targets)
    return {"input_ids": torch.tensor([[1, 2]] * count),
            "attention_mask": torch.ones(count, 2, dtype=torch.long),
            "marker_pos": torch.tensor([[0, 1]] * count),
            "marker_mask": torch.ones(count, 2, dtype=torch.bool),
            "qtype": torch.zeros(count, dtype=torch.long),
            "target": torch.tensor(targets)}


class DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)

    def test_cross_entropy_matches_known_soft_target_gradient_and_masks(self) -> None:
        logits = torch.tensor([[0., 0., float("nan")]], requires_grad=True)
        target = torch.tensor([[0.8, 0.2, 0.]])
        mask = torch.tensor([[True, True, False]])
        loss = decision_loss(logits, target, mask, torch.tensor([0]))
        self.assertAlmostEqual(float(loss.detach()), float(torch.log(torch.tensor(2.))), places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[-0.3, 0.3, 0.]]))

    def test_ordinal_loss_applies_only_to_score(self) -> None:
        logits = torch.tensor([[2., 0., -2.]])
        gold = torch.tensor([[0., 0., 1.]])
        mask = torch.ones_like(gold, dtype=torch.bool)
        choice = decision_loss(logits, gold, mask, torch.tensor([0]))
        score = decision_loss(logits, gold, mask, torch.tensor([1]))
        self.assertGreater(float(score), float(choice))

    def test_invalid_targets_types_and_sparse_masks_fail(self) -> None:
        mask = torch.tensor([[True, True, False]])
        for target in ([[0., 0., 0.]], [[0.5, 0.4, 0.1]], [[-0.1, 1.1, 0.]], [[float("nan"), 1., 0.]]):
            with self.assertRaises(ValueError):
                decision_loss(torch.zeros(1, 3), torch.tensor(target), mask, torch.tensor([0]))
        with self.assertRaises(ValueError):
            decision_loss(torch.zeros(1, 3), torch.tensor([[.5, 0., .5]]),
                          torch.tensor([[True, False, True]]), torch.tensor([1]))
        with self.assertRaises(ValueError):
            decision_loss(torch.zeros(1, 3), torch.tensor([[1., 0., 0.]]),
                          torch.ones(1, 3, dtype=torch.bool), torch.tensor([2]))

    def test_partial_accumulation_equals_full_batch_with_uneven_microbatches(self) -> None:
        full_model = TinyDecisionModel()
        accumulated_model = copy.deepcopy(full_model)
        data = batch()
        micro = [{name: value[:1] for name, value in data.items()},
                 {name: value[1:] for name, value in data.items()}]
        train_decisions(full_model, [data], DecisionTrainingSettings(epochs=1))
        result = train_decisions(accumulated_model, micro, DecisionTrainingSettings(epochs=1, accumulation_steps=4))
        self.assertEqual(result.optimizer_steps, 1)
        self.assertEqual(result.decisions_seen, 3)
        for a, b in zip(full_model.parameters(), accumulated_model.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_frozen_encoder_and_two_layer_head_learn_with_logs(self) -> None:
        model = TinyDecisionModel(transformer_head=True)
        original_encoder = model.encoder.weight.detach().clone()
        data = batch(((0.85, 0.15),) * 4)
        logger = logging.getLogger("koemi.decision-test")
        with self.assertLogs(logger, level="INFO") as captured:
            result = train_decisions(model, [data], DecisionTrainingSettings(
                epochs=15, learning_rate=0.02, freeze_encoder=True), logger)
        self.assertLess(result.epoch_losses[-1], result.epoch_losses[0])
        torch.testing.assert_close(original_encoder, model.encoder.weight, atol=0, rtol=0)
        self.assertIsNone(model.encoder.weight.grad)
        self.assertEqual(result.optimizer_steps, 15)
        self.assertIn("decision_epoch_completed", captured.output[-1])

    def test_invalid_marker_and_padding_are_rejected_before_training(self) -> None:
        for marker in (2, -1):
            data = batch()
            data["marker_pos"][0, 0] = marker
            with self.assertRaisesRegex(ValueError, "markers"):
                train_decisions(TinyDecisionModel(), [data], DecisionTrainingSettings())
        data = batch()
        data["attention_mask"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "padded"):
            train_decisions(TinyDecisionModel(), [data], DecisionTrainingSettings())

    def test_temperature_fitting_reduces_held_out_nll_and_preserves_argmax(self) -> None:
        held_out_logits = torch.tensor([[4., 0.], [0., 4.]] * 10)
        held_out_targets = torch.tensor([[0.7, 0.3], [0.3, 0.7]] * 10)
        mask = torch.ones_like(held_out_logits, dtype=torch.bool)
        types = torch.zeros(20, dtype=torch.long)
        temperature = fit_decision_temperature(held_out_logits, held_out_targets, mask)
        before = decision_loss(held_out_logits, held_out_targets, mask, types)
        after = decision_loss(held_out_logits / temperature, held_out_targets, mask, types)
        self.assertLess(float(after), float(before))
        self.assertTrue(torch.equal(held_out_logits.argmax(-1), (held_out_logits / temperature).argmax(-1)))
        self.assertGreater(temperature, 1)

    def test_typed_answers_choice_score_and_noul(self) -> None:
        answers = typed_decisions(torch.tensor([[0., 1., float("nan")], [0., 0., 0.], [0., 0., 100.]]),
                                  [("a", "b"), ("low", "medium", "high"), ("false", "true")],
                                  ["choice", "score", "noul"])
        self.assertEqual(answers[0].value, "b")
        self.assertAlmostEqual(answers[1].value, 1., places=6)
        self.assertAlmostEqual(answers[2].value, .5)
        for answer in answers:
            self.assertAlmostEqual(sum(answer.probabilities), 1., places=6)

    def test_single_option_and_invalid_labels(self) -> None:
        answer = typed_decisions(torch.tensor([[3.]]), [("only",)], ["choice"])[0]
        self.assertEqual(answer.value, "only")
        self.assertEqual(answer.max_probability, 1)
        for labels, kind in ((("true", "false"), "noul"), (("a", "a"), "choice"), (("", "b"), "choice")):
            with self.assertRaises(ValueError):
                typed_decisions(torch.zeros(1, 2), [labels], [kind])


if __name__ == "__main__":
    unittest.main()
