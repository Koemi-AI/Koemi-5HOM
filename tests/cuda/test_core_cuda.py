from __future__ import annotations

import unittest
from pathlib import Path

import torch

from koemi.cuda_kernels.core import fused_rmsnorm_silu, load_core_extension
from koemi.cuda_kernels.loader import (
    CudaExtensionUnavailableError,
    CudaExtensionValidationError,
    cuda_extension_available,
    detect_cuda_toolchain,
    register_cuda_extension,
)


CUDA_SKIP_REASON = "CUDA runtime and nvcc are unavailable for native extension execution"


class CudaLoaderContractTests(unittest.TestCase):
    def test_toolchain_status_reports_real_local_prerequisites(self) -> None:
        status = detect_cuda_toolchain()
        self.assertIsInstance(status.runtime_available, bool)
        self.assertIsInstance(status.compiler_available, bool)
        self.assertIsInstance(status.ready, bool)
        self.assertTrue(status.reason)
        self.assertEqual(status.ready, cuda_extension_available())

    def test_core_registration_is_namespaced_and_content_addressed(self) -> None:
        core_directory = Path(__file__).resolve().parents[2] / "src" / "koemi" / "cuda_kernels" / "core"
        sources = (core_directory / "rmsnorm_silu.cpp", core_directory / "rmsnorm_silu.cu")
        first = register_cuda_extension("agent_one", "activation", sources)
        second = register_cuda_extension("agent_two", "activation", sources)
        changed_flags = register_cuda_extension("agent_one", "activation", sources, extra_cuda_cflags=("-lineinfo",))
        self.assertNotEqual(first.qualified_name, second.qualified_name)
        self.assertNotEqual(first.qualified_name, changed_flags.qualified_name)
        self.assertTrue(first.qualified_name.startswith("koemi_cuda_agent_one_activation_"))
        self.assertEqual(first.abi_version, 1)

    def test_invalid_registration_is_rejected_before_compilation(self) -> None:
        source = Path(__file__).resolve().parents[2] / "src" / "koemi" / "cuda_kernels" / "core" / "rmsnorm_silu.cu"
        with self.assertRaises(CudaExtensionValidationError):
            register_cuda_extension("bad namespace", "activation", (source,))

    def test_cpu_or_missing_toolchain_never_fakes_kernel_execution(self) -> None:
        if cuda_extension_available():
            self.skipTest("native CUDA extension is available; execution is covered by the CUDA suite")
        input_tensor = torch.ones((2, 8), dtype=torch.float32)
        weight = torch.ones((8,), dtype=torch.float32)
        with self.assertRaises(CudaExtensionValidationError):
            fused_rmsnorm_silu(input_tensor, weight)


@unittest.skipUnless(cuda_extension_available(), CUDA_SKIP_REASON)
class NativeRMSNormSiLUTests(unittest.TestCase):
    device = torch.device("cuda")

    def test_extension_compiles_and_exports_the_declared_abi(self) -> None:
        extension = load_core_extension(device=self.device)
        self.assertEqual(1, extension.KOEMI_CUDA_ABI_VERSION)
        self.assertTrue(callable(extension.rmsnorm_silu_forward))
        self.assertTrue(callable(extension.rmsnorm_silu_backward))

    def test_forward_matches_torch_reference(self) -> None:
        torch.manual_seed(11)
        input_tensor = torch.randn((5, 257), device=self.device, dtype=torch.float32)
        weight = torch.randn((257,), device=self.device, dtype=torch.float32)
        epsilon = 1e-6
        expected_normalized = input_tensor * torch.rsqrt(input_tensor.square().mean(dim=-1, keepdim=True) + epsilon)
        expected = torch.nn.functional.silu(expected_normalized * weight)
        obtained = fused_rmsnorm_silu(input_tensor, weight, epsilon)
        self.assertTrue(torch.allclose(obtained, expected, atol=2e-5, rtol=2e-5))

    def test_backward_matches_torch_reference(self) -> None:
        torch.manual_seed(12)
        source_input = torch.randn((3, 129), device=self.device, dtype=torch.float32)
        source_weight = torch.randn((129,), device=self.device, dtype=torch.float32)
        upstream = torch.randn_like(source_input)
        epsilon = 1e-6

        fused_input = source_input.detach().requires_grad_()
        fused_weight = source_weight.detach().requires_grad_()
        fused_output = fused_rmsnorm_silu(fused_input, fused_weight, epsilon)
        (fused_output * upstream).sum().backward()

        reference_input = source_input.detach().requires_grad_()
        reference_weight = source_weight.detach().requires_grad_()
        normalized = reference_input * torch.rsqrt(reference_input.square().mean(dim=-1, keepdim=True) + epsilon)
        reference_output = torch.nn.functional.silu(normalized * reference_weight)
        (reference_output * upstream).sum().backward()

        self.assertTrue(torch.allclose(fused_output, reference_output, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(fused_input.grad, reference_input.grad, atol=3e-4, rtol=3e-4))
        self.assertTrue(torch.allclose(fused_weight.grad, reference_weight.grad, atol=3e-4, rtol=3e-4))

    def test_shape_dtype_device_and_contiguity_contracts_are_enforced(self) -> None:
        input_tensor = torch.ones((2, 8), device=self.device, dtype=torch.float32)
        weight = torch.ones((8,), device=self.device, dtype=torch.float32)
        with self.assertRaisesRegex(CudaExtensionValidationError, "hidden dimension"):
            fused_rmsnorm_silu(torch.ones((2, 0), device=self.device), torch.ones((0,), device=self.device))
        with self.assertRaisesRegex(CudaExtensionValidationError, "same dtype"):
            fused_rmsnorm_silu(input_tensor, weight.to(torch.float16))
        with self.assertRaisesRegex(CudaExtensionValidationError, "contiguous"):
            fused_rmsnorm_silu(input_tensor[:, ::2], weight[:4])


if __name__ == "__main__":
    unittest.main()
