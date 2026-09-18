from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

import torch
from torch.utils import cpp_extension


__all__ = [
    "CUDA_EXTENSION_ABI_VERSION",
    "CudaExtensionLoadError",
    "CudaExtensionSpec",
    "CudaExtensionUnavailableError",
    "CudaExtensionValidationError",
    "CudaToolchainStatus",
    "cuda_extension_available",
    "detect_cuda_toolchain",
    "load_cuda_extension",
    "register_cuda_extension",
]


CUDA_EXTENSION_ABI_VERSION = 1
_CUDA_SOURCE_SUFFIXES = frozenset({".cu", ".cuh", ".cpp", ".cc", ".cxx"})
_VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_EXTENSION_LOCK = threading.RLock()
_LOADED_EXTENSIONS: dict[str, ModuleType] = {}


class CudaExtensionUnavailableError(RuntimeError):
    """Raised when the local CUDA runtime or compiler cannot load an extension."""


class CudaExtensionValidationError(ValueError):
    """Raised when an extension registration violates the loader contract."""


class CudaExtensionLoadError(RuntimeError):
    """Raised when a registered extension fails to compile or has a wrong ABI."""


@dataclass(frozen=True)
class CudaToolchainStatus:
    """Snapshot of the CUDA runtime and native-extension build prerequisites."""

    runtime_available: bool
    torch_cuda_version: str | None
    cuda_home: Path | None
    nvcc_path: Path | None
    reason: str

    @property
    def compiler_available(self) -> bool:
        return self.torch_cuda_version is not None and self.cuda_home is not None and self.nvcc_path is not None

    @property
    def ready(self) -> bool:
        return self.runtime_available and self.compiler_available


@dataclass(frozen=True)
class CudaExtensionSpec:
    """Namespaced, content-addressed contract for one native CUDA extension."""

    namespace: str
    name: str
    sources: tuple[Path, ...]
    extra_cflags: tuple[str, ...]
    extra_cuda_cflags: tuple[str, ...]
    source_digest: str
    abi_version: int = CUDA_EXTENSION_ABI_VERSION

    @property
    def qualified_name(self) -> str:
        return f"koemi_cuda_{self.namespace}_{self.name}_{self.source_digest[:16]}"

    def build_name(self, architecture_key: str) -> str:
        build_material = "|".join(
            (
                self.qualified_name,
                architecture_key,
                sys.version.split()[0],
                torch.__version__,
            )
        )
        build_digest = hashlib.sha256(build_material.encode("utf-8")).hexdigest()[:16]
        return f"koemi_cuda_{self.namespace}_{self.name}_{build_digest}"


def detect_cuda_toolchain() -> CudaToolchainStatus:
    """Inspect the CUDA runtime, PyTorch CUDA build, CUDA_HOME and `nvcc`."""

    runtime_available = bool(torch.cuda.is_available())
    torch_cuda_version = torch.version.cuda
    raw_cuda_home = getattr(cpp_extension, "CUDA_HOME", None)
    cuda_home = Path(raw_cuda_home).expanduser().resolve() if raw_cuda_home else None
    nvcc_candidates: list[Path] = []
    path_nvcc = shutil.which("nvcc")
    if path_nvcc:
        nvcc_candidates.append(Path(path_nvcc).expanduser().resolve())
    if cuda_home:
        nvcc_candidates.extend((cuda_home / "bin" / "nvcc", cuda_home / "bin" / "nvcc.exe"))
    nvcc_path = next((candidate for candidate in nvcc_candidates if candidate.is_file()), None)

    missing: list[str] = []
    if not runtime_available:
        missing.append("torch.cuda.is_available() is False")
    if torch_cuda_version is None:
        missing.append("the installed PyTorch has no CUDA build")
    if cuda_home is None:
        missing.append("CUDA_HOME is unset")
    if nvcc_path is None:
        missing.append("nvcc was not found")
    reason = "CUDA runtime and native compiler are available" if not missing else "; ".join(missing)
    return CudaToolchainStatus(
        runtime_available=runtime_available,
        torch_cuda_version=torch_cuda_version,
        cuda_home=cuda_home,
        nvcc_path=nvcc_path,
        reason=reason,
    )


def cuda_extension_available() -> bool:
    """Return whether this process can compile and execute a CUDA extension."""

    return detect_cuda_toolchain().ready


def register_cuda_extension(
    namespace: str,
    name: str,
    sources: Iterable[str | os.PathLike[str]],
    *,
    extra_cflags: Sequence[str] = (),
    extra_cuda_cflags: Sequence[str] = (),
) -> CudaExtensionSpec:
    """Register a content-addressed CUDA extension under a collision-safe name.

    Every native binding must export `KOEMI_CUDA_ABI_VERSION` with the value in
    `CUDA_EXTENSION_ABI_VERSION`. Namespaces are owned by individual features;
    source contents and compiler flags are part of the cache identity.
    """

    _validate_identifier("namespace", namespace)
    _validate_identifier("name", name)
    source_paths = _normalize_sources(sources)
    cflags = _normalize_flags("extra_cflags", extra_cflags)
    cuda_cflags = _normalize_flags("extra_cuda_cflags", extra_cuda_cflags)
    source_digest = _calculate_source_digest(namespace, name, source_paths, cflags, cuda_cflags)
    return CudaExtensionSpec(
        namespace=namespace,
        name=name,
        sources=source_paths,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        source_digest=source_digest,
    )


def load_cuda_extension(
    specification: CudaExtensionSpec,
    *,
    device: torch.device | str | int | None = None,
    verbose: bool = False,
) -> ModuleType:
    """Compile or load one registered extension from a stable local cache."""

    status = detect_cuda_toolchain()
    if not status.ready:
        raise CudaExtensionUnavailableError(
            "CUDA extension is unavailable: "
            f"{status.reason}. Install a CUDA-enabled PyTorch and expose a matching nvcc."
        )

    architecture_flags, architecture_key = _resolve_architecture_flags(device, specification.extra_cuda_cflags)
    module_name = specification.build_name(architecture_key)
    with _EXTENSION_LOCK:
        cached_module = _LOADED_EXTENSIONS.get(module_name)
        if cached_module is not None:
            return cached_module

        build_directory = _extension_build_directory(specification, module_name)
        build_directory.mkdir(parents=True, exist_ok=True)
        cxx_flags = list(_default_cxx_flags()) + list(specification.extra_cflags)
        cuda_flags = list(_default_cuda_flags()) + list(architecture_flags) + list(specification.extra_cuda_cflags)
        try:
            native_module = cpp_extension.load(
                name=module_name,
                sources=[str(source) for source in specification.sources],
                extra_cflags=cxx_flags,
                extra_cuda_cflags=cuda_flags,
                build_directory=str(build_directory),
                with_cuda=True,
                is_python_module=True,
                verbose=verbose,
                keep_intermediates=True,
            )
        except (OSError, RuntimeError) as error:
            raise CudaExtensionLoadError(
                f"Could not compile CUDA extension {module_name}. "
                f"nvcc={status.nvcc_path}; cache={build_directory}; error={error}"
            ) from error

        actual_abi_version = getattr(native_module, "KOEMI_CUDA_ABI_VERSION", None)
        if actual_abi_version != specification.abi_version:
            raise CudaExtensionLoadError(
                f"CUDA extension {module_name} exported ABI {actual_abi_version!r}; "
                f"expected {specification.abi_version}."
            )
        _LOADED_EXTENSIONS[module_name] = native_module
        return native_module


def _validate_identifier(field_name: str, value: str) -> None:
    if not isinstance(value, str) or _VALID_IDENTIFIER.fullmatch(value) is None:
        raise CudaExtensionValidationError(
            f"{field_name} must contain only letters, numbers and underscores and start with a letter"
        )


def _normalize_sources(sources: Iterable[str | os.PathLike[str]]) -> tuple[Path, ...]:
    if isinstance(sources, (str, bytes, os.PathLike)):
        source_values = (sources,)
    else:
        source_values = tuple(sources)
    if not source_values:
        raise CudaExtensionValidationError("sources must contain at least one native source")

    normalized: list[Path] = []
    for source in source_values:
        if isinstance(source, bytes):
            raise CudaExtensionValidationError("source paths must be text paths")
        path = Path(source).expanduser().resolve()
        if path.suffix.lower() not in _CUDA_SOURCE_SUFFIXES:
            raise CudaExtensionValidationError(f"unsupported native source suffix: {path.name}")
        if not path.is_file():
            raise CudaExtensionValidationError(f"native source does not exist: {path}")
        if path in normalized:
            raise CudaExtensionValidationError(f"duplicate native source: {path}")
        normalized.append(path)
    if not any(path.suffix.lower() == ".cu" for path in normalized):
        raise CudaExtensionValidationError("a CUDA extension must include at least one .cu source")
    return tuple(normalized)


def _normalize_flags(field_name: str, flags: Sequence[str]) -> tuple[str, ...]:
    if isinstance(flags, (str, bytes)):
        raise CudaExtensionValidationError(f"{field_name} must be a sequence of compiler flags")
    normalized = tuple(flags)
    if any(not isinstance(flag, str) or not flag or "\x00" in flag for flag in normalized):
        raise CudaExtensionValidationError(f"{field_name} contains an invalid compiler flag")
    return normalized


def _calculate_source_digest(
    namespace: str,
    name: str,
    sources: Sequence[Path],
    cflags: Sequence[str],
    cuda_cflags: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"abi={CUDA_EXTENSION_ABI_VERSION}\nnamespace={namespace}\nname={name}\n".encode("utf-8"))
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    for flag in (*cflags, *cuda_cflags):
        digest.update(flag.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_architecture_flags(
    device: torch.device | str | int | None,
    custom_cuda_flags: Sequence[str],
) -> tuple[tuple[str, ...], str]:
    if any(flag.startswith(("-gencode", "--generate-code", "-arch=")) for flag in custom_cuda_flags):
        return (), "custom-architecture"
    architecture_override = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if architecture_override:
        return (), f"torch-cuda-arch-list:{architecture_override}"
    device_index = _device_index(device)
    major, minor = torch.cuda.get_device_capability(device_index)
    architecture = f"{major}{minor}"
    return (f"-gencode=arch=compute_{architecture},code=sm_{architecture}",), f"sm_{architecture}"


def _device_index(device: torch.device | str | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise CudaExtensionValidationError(f"device must be CUDA, got {resolved_device}")
    return torch.cuda.current_device() if resolved_device.index is None else resolved_device.index


def _extension_build_directory(specification: CudaExtensionSpec, module_name: str) -> Path:
    configured_root = os.environ.get("KOEMI_CUDA_CACHE_DIR", "").strip()
    cache_root = Path(configured_root).expanduser() if configured_root else Path.home() / ".cache" / "koemi" / "cuda_extensions"
    return cache_root.resolve() / specification.namespace / specification.name / module_name


def _default_cxx_flags() -> tuple[str, ...]:
    return ("/O2",) if os.name == "nt" else ("-O3",)


def _default_cuda_flags() -> tuple[str, ...]:
    return ("-O3", "--expt-relaxed-constexpr")
