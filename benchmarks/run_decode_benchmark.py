from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.data import create_tokenizer
from koemi.data.tokenizer import TextTokenizer
from koemi.model.network import KoemiModel
from koemi.runtime.fast_decode import SamplingPolicy, generate_batch
from koemi.runtime.speculative import NgramDrafter, ModelDrafter, speculative_generate
from koemi.training.checkpoints import CheckpointStore
from koemi.training.generation import generate_text


DEFAULT_PROMPT = (
    "def queue_pop(queue):\n"
    "    if not queue:\n"
    "        raise IndexError('pop from an empty queue')\n"
    "    return queue.pop(0)\n"
    "def queue_pop(queue):\n"
)


@dataclass(frozen=True)
class DecodeMeasurement:
    """One timed decode run.

    `tokens_per_second` divides the tokens the run actually committed by the wall
    time of the loop, warm-up excluded. `target_forward_calls` is only reported by
    the speculative paths, where it is the number that decides whether the block
    verification paid for itself.
    """

    path: str
    generated_tokens: int
    seconds: float
    tokens_per_second: float
    target_forward_calls: int | None = None
    acceptance_rate: float | None = None


def build_model(arguments: argparse.Namespace) -> tuple[KoemiModel, TextTokenizer]:
    if arguments.checkpoint is not None:
        loaded = CheckpointStore().load(arguments.checkpoint, arguments.device)
        loaded.model.eval()
        return loaded.model, create_tokenizer(loaded.vocabulary)
    torch.manual_seed(arguments.seed)
    settings = ModelSettings(
        embedding_size=arguments.embedding_size,
        memory_features=arguments.memory_features,
        local_memory_size=arguments.local_memory_size,
        salience_memory_size=arguments.salience_memory_size,
        expert_count=arguments.expert_count,
        expert_top_k=arguments.expert_top_k,
        scan_chunk=arguments.scan_chunk,
    )
    model = KoemiModel(settings).to(arguments.device)
    model.eval()
    return model, create_tokenizer(None)


def build_draft_model(arguments: argparse.Namespace, vocabulary_size: int) -> KoemiModel | None:
    if arguments.draft_checkpoint is not None:
        loaded = CheckpointStore().load(arguments.draft_checkpoint, arguments.device)
        loaded.model.eval()
        return loaded.model
    if not arguments.synthetic_draft:
        return None
    torch.manual_seed(arguments.seed + 1)
    settings = ModelSettings(
        vocabulary_size=vocabulary_size,
        embedding_size=max(8, arguments.embedding_size // 4),
        memory_features=arguments.memory_features,
        local_memory_size=arguments.local_memory_size,
        salience_memory_size=arguments.salience_memory_size,
        scan_chunk=arguments.scan_chunk,
    )
    draft_model = KoemiModel(settings).to(arguments.device)
    draft_model.eval()
    return draft_model


def synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def measure(path: str, device: str, run) -> DecodeMeasurement:
    run()
    synchronize(device)
    started_at = time.perf_counter()
    result = run()
    synchronize(device)
    seconds = time.perf_counter() - started_at
    generated_tokens, forward_calls, acceptance_rate = result
    return DecodeMeasurement(
        path=path,
        generated_tokens=generated_tokens,
        seconds=seconds,
        tokens_per_second=generated_tokens / seconds if seconds > 0 else 0.0,
        target_forward_calls=forward_calls,
        acceptance_rate=acceptance_rate,
    )


def run_baseline(model, tokenizer, arguments) -> tuple[int, None, None]:
    generate_text(
        model,
        tokenizer,
        arguments.prompt,
        arguments.max_new_tokens,
        arguments.temperature,
        arguments.device,
    )
    return arguments.max_new_tokens, None, None


def run_fast(model, tokenizer, arguments, capture_graph: bool) -> tuple[int, None, None]:
    result = generate_batch(
        model,
        tokenizer,
        (arguments.prompt,) * arguments.batch_size,
        arguments.max_new_tokens,
        device=arguments.device,
        policy=SamplingPolicy(temperature=arguments.temperature, greedy=arguments.greedy),
        capture_graph=capture_graph,
    )
    return result.generated_tokens, None, None


def run_speculative(model, tokenizer, arguments, drafter_factory) -> tuple[int, int, float]:
    result = speculative_generate(
        model,
        tokenizer,
        arguments.prompt,
        arguments.max_new_tokens,
        drafter=drafter_factory(),
        device=arguments.device,
        policy=SamplingPolicy(temperature=arguments.temperature, greedy=arguments.greedy),
        draft_length=arguments.draft_length,
    )
    return (
        result.statistics.committed_tokens,
        result.statistics.target_forward_calls,
        result.statistics.acceptance_rate,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_decode_benchmark",
        description="Compare the decode paths of a Koemi-4HCM checkpoint on one device",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to measure; omit to use random weights")
    parser.add_argument("--draft-checkpoint", default=None, help="Smaller checkpoint used as the drafter")
    parser.add_argument(
        "--synthetic-draft",
        action="store_true",
        help="Build a narrower random draft model when no draft checkpoint is given",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--draft-length", type=int, default=4)
    parser.add_argument("--ngram-maximum-order", type=int, default=8)
    parser.add_argument("--ngram-minimum-order", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--memory-features", type=int, default=16)
    parser.add_argument("--local-memory-size", type=int, default=16)
    parser.add_argument("--salience-memory-size", type=int, default=16)
    parser.add_argument("--expert-count", type=int, default=0)
    parser.add_argument("--expert-top-k", type=int, default=1)
    parser.add_argument("--scan-chunk", type=int, default=128)
    parser.add_argument("--report", default=None)
    return parser


def main(argument_values: list[str] | None = None) -> int:
    arguments = create_parser().parse_args(argument_values)
    model, tokenizer = build_model(arguments)
    draft_model = build_draft_model(arguments, model.settings.vocabulary_size)
    measurements = [
        measure("baseline", arguments.device, lambda: run_baseline(model, tokenizer, arguments)),
        measure("fast_decode", arguments.device, lambda: run_fast(model, tokenizer, arguments, False)),
        measure(
            "speculative_ngram",
            arguments.device,
            lambda: run_speculative(
                model,
                tokenizer,
                arguments,
                lambda: NgramDrafter(
                    model.settings.vocabulary_size,
                    arguments.device,
                    maximum_order=arguments.ngram_maximum_order,
                    minimum_order=arguments.ngram_minimum_order,
                ),
            ),
        ),
    ]
    if torch.device(arguments.device).type == "cuda":
        measurements.append(
            measure("fast_decode_cuda_graph", arguments.device, lambda: run_fast(model, tokenizer, arguments, True))
        )
    if draft_model is not None:
        measurements.append(
            measure(
                "speculative_model",
                arguments.device,
                lambda: run_speculative(
                    model,
                    tokenizer,
                    arguments,
                    lambda: ModelDrafter(
                        draft_model,
                        policy=SamplingPolicy(
                            temperature=arguments.temperature, greedy=arguments.greedy
                        ),
                    ),
                ),
            )
        )
    payload = {
        "architecture": "Koemi-4HCM",
        "device": arguments.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "vocabulary_size": model.settings.vocabulary_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": arguments.batch_size,
        "max_new_tokens": arguments.max_new_tokens,
        "draft_length": arguments.draft_length,
        "padding_token_id": PAD_TOKEN_ID,
        "measurements": [asdict(measurement) for measurement in measurements],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.report:
        Path(arguments.report).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
