from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from koemi.training.decisions import DecisionTrainingSettings, decision_forward, train_decisions
from koemi.training.laya_decisions import calibrate, main, make_batches, read_rows, split_rows, state_key


class LayaAdapterTests(unittest.TestCase):
    def test_split_keeps_duplicate_states_and_all_questions_together(self) -> None:
        rows = [{"state": {"text": str(i)}, "questions": {}, "gold": {}} for i in range(10)]
        rows.append({"state": {"text": "0"}, "questions": {"second": {}}, "gold": {}})
        train, held = split_rows(rows, .2, 12)
        self.assertEqual(len(train) + len(held), len(rows))
        self.assertFalse({state_key(row) for row in train} & {state_key(row) for row in held})
        self.assertEqual((train, held), split_rows(rows, .2, 12))

    def test_invalid_json_and_missing_gold_fail_with_line_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('{"broken":\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON.*:1"):
                read_rows(path)
            path.write_text(json.dumps({"state": "a", "questions": {"q": {}}, "gold": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gold"):
                read_rows(path)

    def test_split_rejects_one_state_and_invalid_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct states"):
            split_rows([{"state": "same"}] * 4, .2, 0)
        for fraction in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                split_rows([{"state": "a"}, {"state": "b"}], fraction, 0)


@unittest.skipUnless(importlib.util.find_spec("laya"), "optional laya dependency unavailable")
class ActualLayaArchitectureTests(unittest.TestCase):
    def test_actual_laya_model_trains_and_calibrates_offline(self) -> None:
        from laya.common import DecisionModel

        class TinyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=16)
                self.embedding = nn.Embedding(12, 16)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        torch.manual_seed(7)
        model = DecisionModel(TinyEncoder(), head_layers=2, dropout=0)
        items = [{"ids": [1, 2, 3, 4], "markers": [1, 2], "qtype": 0,
                  "target": [.85, .15]} for _ in range(6)]
        batches = make_batches(items, 0, 2)
        initial = model.encoder.embedding.weight.detach().clone()
        result = train_decisions(model, batches, DecisionTrainingSettings(
            epochs=10, learning_rate=.02, freeze_encoder=True, accumulation_steps=2))
        self.assertLess(result.epoch_losses[-1], result.epoch_losses[0])
        self.assertEqual(result.optimizer_steps, 20)
        torch.testing.assert_close(initial, model.encoder.embedding.weight, atol=0, rtol=0)
        temperatures, metrics = calibrate(model, make_batches([
            {"ids": [5, 6, 7, 8], "markers": [1, 2], "qtype": 0, "target": [.6, .4]}
            for _ in range(4)], 0, 2), "cpu")
        self.assertEqual(len(temperatures), 3)
        self.assertLessEqual(metrics["0"]["nll_after"], metrics["0"]["nll_before"] + 1e-6)
        with torch.no_grad():
            self.assertEqual(tuple(decision_forward(model, batches[0]).shape), (2, 2))
        self.assertIsNone(model.act_head[0].weight.grad)

    def test_calibration_fits_within_the_pinned_laya_runtime_bounds(self) -> None:
        from laya.common import TEMP_MIN, TEMP_MAX

        class OverconfidentModel(nn.Module):
            def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
                return torch.tensor([[4., 0.]]).expand(input_ids.shape[0], -1), None

        data = make_batches([{"ids": [1, 2], "markers": [0, 1], "qtype": 0,
                              "target": [.6, .4]}] * 10, 0, 2)
        temperatures, metrics = calibrate(OverconfidentModel(), data, "cpu")
        self.assertGreaterEqual(temperatures[0], TEMP_MIN)
        self.assertLessEqual(temperatures[0], TEMP_MAX)
        self.assertLess(metrics["0"]["nll_after"], metrics["0"]["nll_before"])

    @unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("safetensors"),
                         "full optional decision dependencies unavailable")
    def test_full_command_exports_a_checkpoint_the_public_laya_loader_reads(self) -> None:
        import laya
        from laya.common import DecisionModel
        from safetensors.torch import save_file
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

        words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "billing", "technical",
                 "payments", "refunds", "software", "bugs", "Which", "department", "handles", "this", "choice"]
        vocabulary = {word: index for index, word in enumerate(words)}
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                           cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
        model = DecisionModel(BertModel(BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                              num_attention_heads=2, intermediate_size=32, max_position_embeddings=128)),
                              head_layers=2)
        config = {"encoder": "offline-tiny-bert", "head_layers": 2, "max_len": 128, "head_max_len": 64,
                  "act_costs": {"review": 1}, "temperature": [1, 1, 1],
                  "temperature_by_options": {"choice:2": 8.0}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original"
            source.mkdir()
            save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()},
                      str(source / "model.safetensors"))
            model.encoder.config.save_pretrained(source / "encoder")
            tokenizer.save_pretrained(source / "tokenizer")
            (source / "rl_agent_config.json").write_text(json.dumps(config), encoding="utf-8")
            output = Path(directory) / "trained"
            dataset = Path(__file__).resolve().parents[2] / "examples" / "typed_decisions.jsonl"
            arguments = ["--model-directory", str(source), "--dataset", str(dataset), "--output", str(output),
                         "--epochs", "2", "--batch-size", "2", "--accumulation-steps", "2", "--freeze-encoder"]
            self.assertEqual(main(arguments), 0)
            saved_config = json.loads((output / "rl_agent_config.json").read_text(encoding="utf-8"))
            self.assertNotIn("temperature_by_options", saved_config)
            self.assertEqual(json.loads((source / "rl_agent_config.json").read_text(encoding="utf-8")), config)
            report = json.loads((output / "koemi_training_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["training_states"], 3)
            self.assertEqual(report["calibration_states"], 1)
            self.assertFalse(report["act_head_trained"])
            agent = laya.load(str(output), device="cpu")
            self.assertEqual(list(agent.temperature), saved_config["temperature"])
            row = read_rows(dataset)[0]
            result = agent.predict(row["state"], row["questions"])
            self.assertIn("department", result["answers"])
            with self.assertRaises(FileExistsError):
                main(arguments)


if __name__ == "__main__":
    unittest.main()
