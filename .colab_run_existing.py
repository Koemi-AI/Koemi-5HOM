from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


SOURCE_FILES = {
    "core/rmsnorm_silu.cpp": "src/koemi/cuda_kernels/core/rmsnorm_silu.cpp",
    "core/rmsnorm_silu.cu": "src/koemi/cuda_kernels/core/rmsnorm_silu.cu",
    "dense/affine_scan.cu": "src/koemi/cuda_kernels/dense/affine_scan.cu",
    "think/surprise_bindings.cpp": "src/koemi/cuda_kernels/think/surprise_bindings.cpp",
    "think/surprise_kernel.cu": "src/koemi/cuda_kernels/think/surprise_kernel.cu",
    "moe/moe_kernels.cpp": "src/koemi/cuda_kernels/moe/moe_kernels.cpp",
    "moe/moe_kernels.cu": "src/koemi/cuda_kernels/moe/moe_kernels.cu",
}


REMOTE_TEMPLATE = r'''import json
import os
import tempfile
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F


SOURCES = __SOURCE_PAYLOAD__


def emit(marker, payload):
    print(marker + " " + json.dumps(payload, sort_keys=True), flush=True)


def assert_close(actual, expected, *, rtol, atol, name):
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except Exception as error:
        raise AssertionError(f"{name}: {error}") from error


def cuda_time(function, warmup=5, iterations=20):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(iterations):
        function()
    finished.record()
    finished.synchronize()
    return started.elapsed_time(finished) / iterations


def write_sources(root):
    paths = {}
    for relative, source in SOURCES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        paths[relative] = str(path)
    return paths


def build_extensions(paths):
    from torch.utils.cpp_extension import load

    os.environ["MAX_JOBS"] = "1"
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    build_root = Path(tempfile.mkdtemp(prefix="koemi_cuda_build_"))

    import shutil
    if shutil.which("ninja") is None:
        import subprocess
        import sys
        emit("KOEMI_DEPENDENCY", {"package": "ninja", "action": "install"})
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", "ninja"]
        )
    if shutil.which("ninja") is None:
        raise RuntimeError("ninja installation completed without an executable on PATH")

    def build(name, source_names):
        extension_build_directory = build_root / name
        extension_build_directory.mkdir(parents=True, exist_ok=True)
        extension = load(
            name=name,
            sources=[paths[item] for item in source_names],
            build_directory=str(extension_build_directory),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
        emit("KOEMI_COMPILED", {"extension": name})
        return extension

    return (
        build("koemi_core_cuda_a100_v1", ["core/rmsnorm_silu.cpp", "core/rmsnorm_silu.cu"]),
        build("koemi_dense_cuda_a100_v1", ["dense/affine_scan.cu"]),
        build("koemi_think_cuda_a100_v1", ["think/surprise_bindings.cpp", "think/surprise_kernel.cu"]),
        build("koemi_moe_cuda_a100_v1", ["moe/moe_kernels.cpp", "moe/moe_kernels.cu"]),
    )


def test_core(extension, device):
    torch.manual_seed(101)
    rows, hidden = 8, 128
    values = torch.randn(rows, hidden, device=device, dtype=torch.float32)
    weight = torch.randn(hidden, device=device, dtype=torch.float32)
    eps = 1e-6
    native, inverse = extension.rmsnorm_silu_forward(values, weight, eps)
    expected_inverse = torch.rsqrt(values.square().mean(dim=-1) + eps)
    expected = F.silu(values * expected_inverse[:, None] * weight)
    assert_close(inverse, expected_inverse, rtol=2e-5, atol=2e-6, name="core inverse rms")
    assert_close(native, expected, rtol=2e-5, atol=2e-5, name="core forward")

    upstream = torch.randn_like(native)
    native_input_grad, native_weight_grad = extension.rmsnorm_silu_backward(
        upstream.contiguous(), values, weight, inverse, eps
    )
    reference_values = values.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference_inverse = torch.rsqrt(reference_values.square().mean(dim=-1) + eps)
    reference_output = F.silu(reference_values * reference_inverse[:, None] * reference_weight)
    reference_input_grad, reference_weight_grad = torch.autograd.grad(
        reference_output, (reference_values, reference_weight), upstream
    )
    assert_close(native_input_grad, reference_input_grad, rtol=4e-4, atol=4e-4, name="core input gradient")
    assert_close(native_weight_grad, reference_weight_grad, rtol=4e-4, atol=4e-4, name="core weight gradient")

    benchmark_values = torch.randn(32, 512, device=device)
    benchmark_weight = torch.randn(512, device=device)

    def native_call():
        extension.rmsnorm_silu_forward(benchmark_values, benchmark_weight, eps)

    def reference_call():
        inverse_value = torch.rsqrt(benchmark_values.square().mean(dim=-1) + eps)
        F.silu(benchmark_values * inverse_value[:, None] * benchmark_weight)

    native_ms = cuda_time(native_call)
    reference_ms = cuda_time(reference_call)
    emit("KOEMI_BENCH", {"kernel": "core_rmsnorm_silu", "native_ms": native_ms, "reference_ms": reference_ms, "ratio": native_ms / reference_ms})
    emit("KOEMI_PASS", {"kernel": "core_rmsnorm_silu", "checks": ["forward", "backward", "benchmark"]})


def sequential_scan(retention, increment, initial):
    state = initial
    states = []
    for position in range(increment.shape[1]):
        state = retention[:, position] * state + increment[:, position]
        states.append(state)
    return torch.stack(states, dim=1)


def test_dense(extension, device):
    torch.manual_seed(102)
    batch, sequence, features = 3, 31, 64
    retention = torch.rand(batch, sequence, features, device=device) * 0.8 + 0.1
    increment = torch.randn(batch, sequence, features, device=device)
    initial = torch.randn(batch, features, device=device)
    native = extension.forward(retention.contiguous(), increment.contiguous(), initial.contiguous())
    expected = sequential_scan(retention, increment, initial)
    assert_close(native, expected, rtol=2e-5, atol=2e-5, name="dense forward")

    upstream = torch.randn_like(native)
    native_gradients = extension.backward(
        retention.contiguous(), initial.contiguous(), native.contiguous(), upstream.contiguous()
    )
    reference_retention = retention.detach().clone().requires_grad_()
    reference_increment = increment.detach().clone().requires_grad_()
    reference_initial = initial.detach().clone().requires_grad_()
    reference_states = sequential_scan(reference_retention, reference_increment, reference_initial)
    reference_gradients = torch.autograd.grad(
        reference_states, (reference_retention, reference_increment, reference_initial), upstream
    )
    for index, (actual, expected_gradient) in enumerate(zip(native_gradients, reference_gradients)):
        assert_close(actual, expected_gradient, rtol=4e-5, atol=4e-5, name=f"dense gradient {index}")

    benchmark_retention = torch.rand(8, 256, 128, device=device) * 0.8 + 0.1
    benchmark_increment = torch.randn(8, 256, 128, device=device)
    benchmark_initial = torch.randn(8, 128, device=device)
    native_ms = cuda_time(lambda: extension.forward(benchmark_retention, benchmark_increment, benchmark_initial), iterations=10)
    reference_ms = cuda_time(lambda: sequential_scan(benchmark_retention, benchmark_increment, benchmark_initial), iterations=10)
    emit("KOEMI_BENCH", {"kernel": "dense_affine_scan", "native_ms": native_ms, "reference_ms": reference_ms, "ratio": native_ms / reference_ms})
    emit("KOEMI_PASS", {"kernel": "dense_affine_scan", "checks": ["forward", "backward", "benchmark"]})


def surprise_reference(prior, weight, bias, targets, valid):
    logits = prior.float() @ weight.float().transpose(0, 1) + bias.float()
    log_partition = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = log_partition - target_logits
    surprise = 1.0 - torch.exp(-nll / torch.log(torch.tensor(float(weight.shape[0]), device=prior.device)))
    return torch.where(valid, surprise, torch.zeros_like(surprise)).to(prior.dtype)


def test_think(extension, device):
    torch.manual_seed(103)
    batch, sequence, width, vocab = 2, 7, 64, 257
    prior = torch.randn(batch, sequence, width, device=device)
    weight = torch.randn(vocab, width, device=device)
    bias = torch.randn(vocab, device=device)
    targets = torch.randint(0, vocab, (batch, sequence), device=device, dtype=torch.long)
    valid = torch.ones(batch, sequence, device=device, dtype=torch.bool)
    valid[0, -1] = False
    native = extension.forward(prior.contiguous(), weight.contiguous(), bias.contiguous(), targets.contiguous(), valid.contiguous())
    expected = surprise_reference(prior, weight, bias, targets, valid)
    assert_close(native, expected, rtol=4e-4, atol=4e-4, name="think forward")

    upstream = torch.randn_like(native)
    native_gradients = extension.backward(
        prior.contiguous(), weight.contiguous(), bias.contiguous(), targets.contiguous(), valid.contiguous(), upstream.contiguous()
    )
    reference_prior = prior.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()
    reference_output = surprise_reference(reference_prior, reference_weight, reference_bias, targets, valid)
    reference_gradients = torch.autograd.grad(
        reference_output, (reference_prior, reference_weight, reference_bias), upstream
    )
    for index, (actual, expected_gradient) in enumerate(zip(native_gradients, reference_gradients)):
        assert_close(actual, expected_gradient, rtol=3e-3, atol=3e-3, name=f"think gradient {index}")
    if not torch.equal(native[0, -1], torch.zeros((), device=device)):
        raise AssertionError("think invalid position was not zero")

    benchmark_prior = torch.randn(4, 64, 128, device=device)
    benchmark_weight = torch.randn(1024, 128, device=device)
    benchmark_bias = torch.randn(1024, device=device)
    benchmark_targets = torch.randint(0, 1024, (4, 64), device=device, dtype=torch.long)
    benchmark_valid = torch.ones(4, 64, device=device, dtype=torch.bool)
    native_ms = cuda_time(lambda: extension.forward(benchmark_prior, benchmark_weight, benchmark_bias, benchmark_targets, benchmark_valid), iterations=10)
    reference_ms = cuda_time(lambda: surprise_reference(benchmark_prior, benchmark_weight, benchmark_bias, benchmark_targets, benchmark_valid), iterations=10)
    emit("KOEMI_BENCH", {"kernel": "think_causal_surprise", "native_ms": native_ms, "reference_ms": reference_ms, "ratio": native_ms / reference_ms})
    emit("KOEMI_PASS", {"kernel": "think_causal_surprise", "checks": ["forward", "backward", "causal_mask", "benchmark"]})


def content_hash(token_ids, previous_token_ids):
    value = (token_ids.to(torch.int64) * 1000003 + previous_token_ids.to(torch.int64) * 97409) & 0xFFFFFFFF
    value = value ^ (value >> 16)
    value = (value * 0x45D9F3B) & 0xFFFFFFFF
    value = value ^ (value >> 16)
    value = (value * 0x45D9F3B) & 0xFFFFFFFF
    return value ^ (value >> 16)


def test_moe(extension, device):
    torch.manual_seed(104)
    token_count, width, expert_count, top_k, hidden = 12, 8, 4, 2, 6
    token_ids = torch.tensor([1, 2, 99, 7, 3, 4, 11, 12, 13, 14, 15, 16], device=device, dtype=torch.long)
    previous = torch.tensor([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14], device=device, dtype=torch.long)
    valid = torch.ones(token_count, device=device, dtype=torch.bool)
    valid[-1] = False
    assignments = extension.route_topk(token_ids, previous, valid, expert_count, top_k)
    hashed = content_hash(token_ids, previous)
    offsets = torch.arange(top_k, device=device, dtype=hashed.dtype)
    expected_assignments = (hashed[:, None] + offsets[None, :] * 0x9E3779B9) % expert_count
    expected_assignments = expected_assignments.masked_fill(~valid[:, None], -1)
    if not torch.equal(assignments, expected_assignments):
        raise AssertionError("MoE route differs from content hash reference")

    context = torch.randn(token_count, width, device=device)
    sorted_context, sorted_rows, sorted_experts, segment_offsets = extension.permute_topk(
        context.contiguous(), assignments.contiguous(), valid.contiguous(), expert_count
    )
    valid_pairs = int(segment_offsets[-1].item())
    if valid_pairs <= 0 or valid_pairs > token_count * top_k:
        raise AssertionError(f"invalid MoE valid pair count: {valid_pairs}")
    if not torch.equal(sorted_context[valid_pairs:], torch.zeros_like(sorted_context[valid_pairs:])):
        raise AssertionError("MoE invalid sorted context rows are not zero")

    gate_weights = torch.randn(expert_count, hidden, width, device=device)
    gate_biases = torch.randn(expert_count, hidden, device=device)
    value_weights = torch.randn_like(gate_weights)
    value_biases = torch.randn_like(gate_biases)
    output_weights = torch.randn(expert_count, width, hidden, device=device)
    output_biases = torch.randn(expert_count, width, device=device)
    expert_outputs = extension.grouped_expert_mlp(
        sorted_context.contiguous(), sorted_experts.contiguous(), segment_offsets.contiguous(),
        gate_weights.contiguous(), gate_biases.contiguous(), value_weights.contiguous(), value_biases.contiguous(),
        output_weights.contiguous(), output_biases.contiguous()
    )
    expected_rows = []
    for pair in range(valid_pairs):
        expert = int(sorted_experts[pair].item())
        gate = F.linear(sorted_context[pair], gate_weights[expert], gate_biases[expert])
        value = F.linear(sorted_context[pair], value_weights[expert], value_biases[expert])
        expected_rows.append(F.linear(F.silu(gate) * value, output_weights[expert], output_biases[expert]))
    expected_expert_outputs = torch.zeros_like(expert_outputs)
    expected_expert_outputs[:valid_pairs] = torch.stack(expected_rows)
    assert_close(expert_outputs, expected_expert_outputs, rtol=5e-4, atol=5e-4, name="MoE grouped MLP")

    combined = extension.combine_topk(
        expert_outputs.contiguous(), sorted_rows.contiguous(), segment_offsets.contiguous(), token_count, top_k
    )
    expected_combined = torch.zeros(token_count, width, device=device, dtype=torch.float32)
    for pair in range(valid_pairs):
        expected_combined[int(sorted_rows[pair].item())] += expert_outputs[pair].float() / top_k
    assert_close(combined, expected_combined.to(combined.dtype), rtol=5e-4, atol=5e-4, name="MoE combine")

    emit("KOEMI_BENCH", {"kernel": "moe_route_permute_group_combine", "tokens": token_count, "top_k": top_k, "valid_pairs": valid_pairs})
    emit("KOEMI_PASS", {"kernel": "moe_route_permute_group_combine", "checks": ["route", "permute", "grouped_mlp", "combine"]})


def main():
    properties = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
    }
    emit("KOEMI_PREFLIGHT", properties)
    if not torch.cuda.is_available():
        emit("KOEMI_NO_CUDA", {"reason": "torch.cuda.is_available() is False"})
        return
    device = torch.device("cuda:0")
    root = Path(tempfile.mkdtemp(prefix="koemi_cuda_sources_"))
    try:
        paths = write_sources(root)
        core, dense, think, moe = build_extensions(paths)
        test_core(core, device)
        test_dense(dense, device)
        test_think(think, device)
        test_moe(moe, device)
        emit("KOEMI_REMOTE_DONE", {"status": "pass", "device": properties["device"]})
    except Exception as error:
        emit("KOEMI_FAIL", {"type": type(error).__name__, "error": str(error)})
        traceback.print_exc()
        raise


main()
'''


def build_remote_code() -> str:
    payload = {
        relative: Path(path).read_text(encoding="utf-8")
        for relative, path in SOURCE_FILES.items()
    }
    return REMOTE_TEMPLATE.replace("__SOURCE_PAYLOAD__", json.dumps(payload))


class RemoteHarnessError(RuntimeError):
    pass


def transport() -> StdioTransport:
    return StdioTransport(
        command=r"C:\Users\Brenno\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\uv.exe",
        args=[
            "tool",
            "run",
            "--from",
            "git+https://github.com/googlecolab/colab-mcp",
            "colab-mcp",
        ],
    )


async def run_attempt(attempt: int, remote_code: str) -> None:
    async with Client(transport(), timeout=1800) as client:
        connection = await client.call_tool("open_colab_browser_connection", {})
        print(f"MCP_CONNECTED attempt={attempt} result={connection!r}", flush=True)
        names = []
        for poll in range(36):
            await asyncio.sleep(5)
            names = [tool.name for tool in await client.list_tools()]
            if "get_cells" in names and "update_cell" in names and "run_code_cell" in names:
                break
            if poll % 3 == 0:
                print(f"MCP_WAIT attempt={attempt} poll={poll + 1} tools={names!r}", flush=True)
        print(json.dumps({"attempt": attempt, "tools": names}, indent=2), flush=True)
        if "get_cells" not in names or "update_cell" not in names or "run_code_cell" not in names:
            raise RuntimeError("Colab notebook tools did not become available")
        cells_result = await client.call_tool("get_cells", {"includeOutputs": True})
        cells = cells_result.structured_content["cells"]
        selected = next((cell for cell in cells if cell.get("cell_type") == "code"), None)
        if selected is None:
            raise RuntimeError("connected Colab notebook has no code cell")
        print(json.dumps({"attempt": attempt, "cell_id": selected["id"], "remote_chars": len(remote_code)}, indent=2), flush=True)
        update = await client.call_tool(
            "update_cell",
            {"cellId": selected["id"], "content": remote_code},
        )
        print("UPDATE_RESULT " + repr(update), flush=True)
        result = await client.call_tool("run_code_cell", {"cellId": selected["id"]})
        print("RUN_RESULT " + repr(result), flush=True)
        result_blob = json.dumps(getattr(result, "structured_content", None), default=str)
        if "KOEMI_FAIL" in result_blob:
            raise RemoteHarnessError("remote CUDA harness reported KOEMI_FAIL")
        if "KOEMI_REMOTE_DONE" not in result_blob and "KOEMI_NO_CUDA" not in result_blob:
            raise RemoteHarnessError("remote CUDA harness returned no terminal marker")


async def main() -> None:
    remote_code = build_remote_code()
    max_attempts = max(1, int(os.environ.get("KOEMI_MCP_RETRIES", "30")))
    base_delay = max(1.0, float(os.environ.get("KOEMI_MCP_RETRY_DELAY", "5")))
    max_delay = max(base_delay, float(os.environ.get("KOEMI_MCP_RETRY_MAX_DELAY", "30")))
    for attempt in range(1, max_attempts + 1):
        try:
            await run_attempt(attempt, remote_code)
            print(f"MCP_DONE attempt={attempt}", flush=True)
            return
        except RemoteHarnessError:
            raise
        except Exception as error:
            if attempt >= max_attempts:
                print(f"MCP_RETRY_EXHAUSTED attempts={attempt} error={error!r}", flush=True)
                raise
            delay = min(max_delay, base_delay * (2 ** min(attempt - 1, 4)))
            print(
                f"MCP_RETRY attempt={attempt} next_attempt={attempt + 1} "
                f"delay_seconds={delay:g} error={error!r}",
                flush=True,
            )
            await asyncio.sleep(delay)


if __name__ == "__main__":
    asyncio.run(main())
