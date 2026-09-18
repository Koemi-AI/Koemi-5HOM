from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel


def build_model(settings: ModelSettings, device: torch.device, model_seed: int) -> KoemiModel:
    torch.manual_seed(model_seed)
    return KoemiModel(settings).to(device)


def measure_forward_backward(
    model: KoemiModel,
    input_ids: torch.Tensor,
    device: torch.device,
    repeats: int,
    autocast_dtype: torch.dtype | None,
) -> dict[str, Any]:
    model.train()

    def one_pass() -> None:
        model.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.enable_grad()
        )
        with context:
            output = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
            loss = output.logits.float().square().mean()
        loss.backward()

    one_pass()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    for _ in range(repeats):
        one_pass()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = (time.perf_counter() - started_at) / repeats
    report: dict[str, Any] = {
        "seconds_per_forward_backward": elapsed_seconds,
        "tokens_per_second": int(input_ids.numel()) / elapsed_seconds,
    }
    if device.type == "cuda":
        report["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return report


def count_nonzero_calls(model: KoemiModel, input_ids: torch.Tensor) -> int:
    calls = {"count": 0}
    original_nonzero = torch.nonzero

    def counted_nonzero(*arguments, **keywords):
        calls["count"] += 1
        return original_nonzero(*arguments, **keywords)

    model.train()
    torch.nonzero = counted_nonzero
    try:
        model(input_ids, execution_mode=ExecutionMode.PARALLEL)
    finally:
        torch.nonzero = original_nonzero
    return calls["count"]


def compare_dispatch(
    embedding_size: int,
    expert_count: int,
    expert_top_k: int,
    scan_chunk: int,
    batch_size: int,
    sequence_length: int,
    repeats: int,
    device_name: str,
    model_seed: int,
) -> dict[str, Any]:
    device = torch.device(device_name)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    torch.manual_seed(model_seed)
    input_ids = torch.randint(0, PAD_TOKEN_ID, (batch_size, sequence_length), device=device)
    report: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "autocast_dtype": str(autocast_dtype),
        "embedding_size": embedding_size,
        "expert_count": expert_count,
        "expert_top_k": expert_top_k,
        "scan_chunk": scan_chunk,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "repeats": repeats,
        "windows_per_forward": -(-sequence_length // scan_chunk),
        "paths": {},
    }
    for dispatch in ("loop", "segments"):
        settings = ModelSettings(
            embedding_size=embedding_size,
            memory_features=16,
            local_memory_size=32,
            salience_memory_size=32,
            expert_count=expert_count,
            expert_top_k=expert_top_k,
            expert_dispatch=dispatch,
            scan_chunk=scan_chunk,
            ablation="no_refine",
        )
        model = build_model(settings, device, model_seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        path_report = measure_forward_backward(model, input_ids, device, repeats, autocast_dtype)
        path_report["nonzero_calls_per_forward"] = count_nonzero_calls(model, input_ids)
        report["paths"][dispatch] = path_report
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    loop_seconds = report["paths"]["loop"]["seconds_per_forward_backward"]
    segment_seconds = report["paths"]["segments"]["seconds_per_forward_backward"]
    report["segments_speedup"] = loop_seconds / segment_seconds
    return report


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the expert dispatch paths on one device")
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--expert-count", type=int, default=128)
    parser.add_argument("--expert-top-k", type=int, default=6)
    parser.add_argument("--scan-chunk", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-seed", type=int, default=1337)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    report = compare_dispatch(
        arguments.embedding_size,
        arguments.expert_count,
        arguments.expert_top_k,
        arguments.scan_chunk,
        arguments.batch_size,
        arguments.sequence_length,
        arguments.repeats,
        arguments.device,
        arguments.model_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
