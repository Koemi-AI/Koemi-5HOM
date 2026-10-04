from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import tempfile
from pathlib import Path
from collections.abc import Sequence
from importlib.metadata import version

import torch
from torch import Tensor

from koemi.training.decisions import (DecisionTrainingSettings, decision_forward, decision_loss,
                                      fit_decision_temperature, train_decisions)


LAYA_VERSION = "0.3.27"


def read_rows(path: Path, maximum_rows: int = 10_000) -> list[dict]:
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("decision dataset must be a local file no larger than 64 MiB")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if (not isinstance(row, dict) or not isinstance(row.get("state"), (str, list, dict)) or
                    not isinstance(row.get("questions"), dict) or not row["questions"] or
                    not isinstance(row.get("gold"), dict) or set(row["gold"]) != set(row["questions"])):
                raise ValueError(f"row {line_number} requires state, questions and gold for every question")
            rows.append(row)
            if len(rows) > maximum_rows:
                raise ValueError(f"decision dataset exceeds {maximum_rows} rows")
    if not rows:
        raise ValueError("decision dataset has no rows")
    return rows


def state_key(row: dict) -> str:
    state = json.dumps(row["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def split_rows(rows: Sequence[dict], calibration_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not math.isfinite(calibration_fraction) or not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between zero and one")
    keys = sorted({state_key(row) for row in rows},
                  key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    if len(keys) < 2:
        raise ValueError("training and calibration require at least two distinct states")
    count = max(1, min(len(keys) - 1, int(len(keys) * calibration_fraction)))
    held = set(keys[:count])
    return ([row for row in rows if state_key(row) not in held],
            [row for row in rows if state_key(row) in held])


def encode_rows(rows: Sequence[dict], tokenizer, max_len: int, head_max_len: int,
                allow_state_truncation: bool = False) -> list[dict]:
    from laya.agent import Agent
    from laya.common import QTYPES, build_sequence, render_options

    items = []
    for row_number, row in enumerate(rows, 1):
        for qid, question in row["questions"].items():
            Agent._check_question(qid, question)
            internal = Agent._to_internal(question)
            internal.pop("option_order", None)
            kind = internal["t"]
            keys = (list(internal["crit"]) if kind == "choice" else
                    ["false", "true"] if kind == "noul" else
                    [str(index) for index in range(len(internal["crit"]))])
            gold = row["gold"][qid]
            if not isinstance(gold, dict) or not isinstance(gold.get("probabilities"), dict):
                raise ValueError(f"row {row_number} question {qid}: gold requires probabilities")
            probabilities = gold["probabilities"]
            if set(probabilities) != set(keys):
                raise ValueError(f"row {row_number} question {qid}: probabilities must name every option exactly")
            target = [probabilities[key] for key in keys]
            if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0 for value in target) or
                    not math.isclose(sum(target), 1, abs_tol=1e-6, rel_tol=1e-5)):
                raise ValueError(f"row {row_number} question {qid}: invalid probability distribution")
            ids, markers, stats, truncation = build_sequence(
                tokenizer, row["state"], internal, max_len, head_max_len,
                return_stats=True, return_truncation_stats=True)
            if len(markers) != len(keys) or stats["options_distinct"] != len(render_options(internal)):
                raise ValueError(f"row {row_number} question {qid}: option text collapsed under the token budget")
            if truncation["truncated"] and not allow_state_truncation:
                raise ValueError(f"row {row_number} question {qid}: state truncated; increase max_len or explicitly allow truncation")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[kind], "target": target})
    return items


def make_batches(items: Sequence[dict], pad_id: int, batch_size: int) -> list[dict[str, Tensor]]:
    from laya.common import collate_items

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "target")
    batches = []
    for start in range(0, len(items), batch_size):
        collated = collate_items([items[start:start + batch_size]], pad_id)
        batches.append({key: collated[key] for key in keys})
    return batches


def calibrate(model, batches: Sequence[dict[str, Tensor]], device: str) -> tuple[list[float], dict]:
    from laya.common import TEMP_MIN, TEMP_MAX

    records: dict[int, list[tuple[Tensor, Tensor]]] = {kind: [] for kind in range(3)}
    model.eval()
    with torch.no_grad():
        for batch in batches:
            inputs = {key: value.to(device) for key, value in batch.items() if isinstance(value, Tensor)}
            logits = decision_forward(model, inputs).cpu()
            for index, kind in enumerate(batch["qtype"].tolist()):
                count = int(batch["marker_mask"][index].sum())
                records[kind].append((logits[index, :count], batch["target"][index, :count]))
    temperatures = []
    metrics = {}
    for kind, rows in records.items():
        if not rows:
            temperatures.append(1.)
            continue
        width = max(logits.numel() for logits, _ in rows)
        logits = torch.zeros(len(rows), width)
        targets = torch.zeros_like(logits)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for index, (values, target) in enumerate(rows):
            count = values.numel()
            logits[index, :count], targets[index, :count], mask[index, :count] = values, target, True
        temperature = fit_decision_temperature(logits, targets, mask,
                                               minimum_temperature=TEMP_MIN, maximum_temperature=TEMP_MAX)
        qtypes = torch.full((len(rows),), kind, dtype=torch.long)
        before = float(decision_loss(logits, targets, mask, qtypes, ordinal_weight=0))
        after = float(decision_loss(logits / temperature, targets, mask, qtypes, ordinal_weight=0))
        temperatures.append(temperature)
        metrics[str(kind)] = {"decisions": len(rows), "nll_before": before, "nll_after": after}
    return temperatures, metrics


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised Laya decision training with a held-out state split")
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--ordinal-weight", type=float, default=1.)
    parser.add_argument("--calibration-fraction", type=float, default=.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--head-max-len", type=int)
    parser.add_argument("--allow-state-truncation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = DecisionTrainingSettings(
        epochs=arguments.epochs, learning_rate=arguments.learning_rate,
        encoder_learning_rate=arguments.encoder_learning_rate,
        accumulation_steps=arguments.accumulation_steps, freeze_encoder=arguments.freeze_encoder,
        ordinal_weight=arguments.ordinal_weight, device=arguments.device, seed=arguments.seed)
    source = arguments.model_directory.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    required = ("rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/tokenizer_config.json")
    if any(not (source / name).is_file() for name in required):
        raise ValueError("model-directory must contain a complete local Laya checkpoint")
    rows = read_rows(arguments.dataset)
    training_rows, calibration_rows = split_rows(rows, arguments.calibration_fraction, settings.seed)
    if version("laya") != LAYA_VERSION:
        raise ValueError(f"this adapter requires laya=={LAYA_VERSION}")
    from laya.common import build_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    cfg = json.loads((source / "rl_agent_config.json").read_text(encoding="utf-8"))
    max_len = arguments.max_len if arguments.max_len is not None else cfg.get("max_len", 512)
    head_max_len = arguments.head_max_len if arguments.head_max_len is not None else cfg.get("head_max_len", 192)
    if not isinstance(max_len, int) or not isinstance(head_max_len, int) or not 8 <= head_max_len < max_len:
        raise ValueError("token budgets require 8 <= head_max_len < max_len")
    tokenizer = AutoTokenizer.from_pretrained(str(source / "tokenizer"), local_files_only=True, trust_remote_code=False)
    training_items = encode_rows(training_rows, tokenizer, max_len, head_max_len, arguments.allow_state_truncation)
    calibration_items = encode_rows(calibration_rows, tokenizer, max_len, head_max_len, arguments.allow_state_truncation)
    model = build_model(cfg, encoder_dir=str(source / "encoder"))
    model.load_state_dict(load_file(str(source / "model.safetensors")), strict=True)
    model.float()
    result = train_decisions(model, make_batches(training_items, tokenizer.pad_token_id, arguments.batch_size), settings)
    temperatures, metrics = calibrate(model, make_batches(calibration_items, tokenizer.pad_token_id, arguments.batch_size), settings.device)
    cfg.update(max_len=max_len, head_max_len=head_max_len, temperature=temperatures)
    cfg.pop("temperature_by_options", None)
    report = {"kind": "supervised_decisions", "laya_version": LAYA_VERSION,
              "training_states": len({state_key(row) for row in training_rows}),
              "calibration_states": len({state_key(row) for row in calibration_rows}),
              "epoch_losses": result.epoch_losses, "optimizer_steps": result.optimizer_steps,
              "decisions_seen": result.decisions_seen, "calibration_fit": metrics,
              "temperatures": temperatures, "evaluated_on_independent_test_data": False,
              "act_head_trained": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="koemi-decisions-", dir=output.parent) as temporary:
        staged = Path(temporary) / "checkpoint"
        staged.mkdir()
        save_file({name: value.detach().float().cpu().contiguous() for name, value in model.state_dict().items()},
                  str(staged / "model.safetensors"))
        model.encoder.config.save_pretrained(staged / "encoder")
        tokenizer.save_pretrained(staged / "tokenizer")
        (staged / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (staged / "koemi_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staged.rename(output)
    logging.info("decision_checkpoint_saved output=%s train_states=%s calibration_states=%s act_head_trained=false",
                 output, report["training_states"], report["calibration_states"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
