from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from koemi.model.experts import content_dispatch_hash
from koemi.model.identity import ModelIdentity


MANIFEST_FILENAME = "manifest.json"
MANIFEST_DIGEST_FILENAME = "manifest.json.sha256"
SUBMAPPING_FORMAT = "koemi-moe-submapping"
SUBMAPPING_VERSION = 1
ROUTING_KIND = "content_dispatch_hash"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SubmappingError(RuntimeError):
    """Base error for the isolated MoE submapping store."""


class ManifestError(SubmappingError):
    """The store manifest is missing, malformed, or not MoE-only."""


class ManifestIntegrityError(ManifestError):
    """The manifest JSON does not match its SHA-256 sidecar."""


class BlockIntegrityError(ManifestError):
    """A selected block does not match its manifest digest or tensor contract."""


class UnsafePathError(ManifestError):
    """A manifest path would escape the layout directory."""


class UnknownExpertError(SubmappingError):
    """A MoE call selected an expert outside the published catalog."""


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    parsed = _nonnegative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _safe_component(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{name} must be a safe single path component")
    if value in {".", ".."}:
        raise ValueError(f"{name} cannot be a traversal component")
    return value


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer: int
    expert: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.layer, "layer")
        _nonnegative_int(self.expert, "expert")

    def as_tuple(self) -> tuple[int, int]:
        return self.layer, self.expert


@dataclass(frozen=True, order=True)
class ExpertTensorKey:
    layer: int
    expert: int
    tensor: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.layer, "layer")
        _nonnegative_int(self.expert, "expert")
        _safe_component(self.tensor, "tensor")

    @property
    def expert_key(self) -> ExpertKey:
        return ExpertKey(self.layer, self.expert)


@dataclass(frozen=True)
class LayoutPublication:
    directory: Path
    manifest_path: Path
    manifest_sha256: str
    block_count: int
    expert_count: int


@dataclass(frozen=True)
class ReadMetrics:
    requested: int
    unique: int
    cache_hits: int
    bytes_read: int
    read_seconds: float
    evictions: int

    @property
    def cache_hit(self) -> int:
        return self.cache_hits

    def to_dict(self) -> dict[str, int | float]:
        return {
            "requested": self.requested,
            "unique": self.unique,
            "cache_hits": self.cache_hits,
            "bytes_read": self.bytes_read,
            "read_seconds": self.read_seconds,
            "evictions": self.evictions,
        }


@dataclass
class SubmappingMetrics:
    requested: int = 0
    unique: int = 0
    cache_hits: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0
    evictions: int = 0

    @property
    def cache_hit(self) -> int:
        return self.cache_hits

    def to_dict(self) -> dict[str, int | float]:
        return {
            "requested": self.requested,
            "unique": self.unique,
            "cache_hits": self.cache_hits,
            "bytes_read": self.bytes_read,
            "read_seconds": self.read_seconds,
            "evictions": self.evictions,
        }


@dataclass(frozen=True)
class _BlockRecord:
    path: str
    offset: int
    length: int
    sha256: str
    serialized_bytes: int


@dataclass(frozen=True)
class _TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    blocks: tuple[_BlockRecord, ...]


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def route_moe_experts(
    token_ids: Tensor,
    previous_token_ids: Tensor,
    valid_mask: Tensor,
    *,
    expert_count: int,
    top_k: int,
) -> Tensor:
    if not isinstance(token_ids, Tensor) or not isinstance(previous_token_ids, Tensor) or not isinstance(valid_mask, Tensor):
        raise TypeError("MoE routing inputs must be tensors")
    if token_ids.shape != previous_token_ids.shape or token_ids.shape != valid_mask.shape:
        raise ValueError("MoE routing inputs must have equal shapes")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must have boolean dtype")
    expert_count = _positive_int(expert_count, "expert_count")
    top_k = _positive_int(top_k, "top_k")
    if top_k > expert_count:
        raise ValueError("top_k must not exceed expert_count")
    context_hash = content_dispatch_hash(token_ids, previous_token_ids)
    offsets = torch.arange(top_k, device=token_ids.device, dtype=context_hash.dtype)
    assignments = (context_hash.unsqueeze(-1) + offsets * 0x9E3779B9).remainder(expert_count)
    duplicate_assignments = assignments.unsqueeze(-1) == assignments.unsqueeze(-2)
    duplicate_assignments = torch.triu(duplicate_assignments, diagonal=1).any(dim=-1)
    if bool((duplicate_assignments & valid_mask.unsqueeze(-1)).any().item()):
        raise ValueError("MoE routing produced duplicate experts for a valid token")
    return assignments.masked_fill(~valid_mask.unsqueeze(-1), -1)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path: Path, tensor: Tensor) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(tensor, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _coerce_tensor_key(value: object) -> ExpertTensorKey:
    if isinstance(value, ExpertTensorKey):
        return value
    if isinstance(value, tuple) and len(value) == 3:
        return ExpertTensorKey(value[0], value[1], value[2])
    raise TypeError(
        "expert tensors must be keyed by ExpertTensorKey or (layer, expert, tensor)"
    )


def _normalise_counts(
    layer_count: object,
    experts_per_layer: int | Mapping[int, int],
) -> dict[int, int]:
    layers = _positive_int(layer_count, "layer_count")
    if isinstance(experts_per_layer, bool):
        raise ValueError("experts_per_layer must be positive")
    if isinstance(experts_per_layer, int):
        count = _positive_int(experts_per_layer, "experts_per_layer")
        return {layer: count for layer in range(layers)}
    if not isinstance(experts_per_layer, Mapping):
        raise TypeError("experts_per_layer must be an integer or a layer mapping")
    counts = {
        _nonnegative_int(layer, "layer"): _positive_int(count, "expert count")
        for layer, count in experts_per_layer.items()
    }
    expected_layers = set(range(layers))
    if set(counts) != expected_layers:
        raise ValueError("experts_per_layer must define every layer exactly once")
    return counts


def _normalise_tensor_catalog(
    expert_tensors: Mapping[object, Tensor] | Iterable[tuple[object, Tensor]],
    counts: Mapping[int, int],
) -> dict[ExpertKey, dict[str, Tensor]]:
    items = expert_tensors.items() if isinstance(expert_tensors, Mapping) else expert_tensors
    catalog: dict[ExpertKey, dict[str, Tensor]] = {}
    for raw_key, tensor in items:
        key = _coerce_tensor_key(raw_key)
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{key.tensor} must be a torch.Tensor")
        if tensor.layout != torch.strided or tensor.numel() == 0:
            raise ValueError("expert tensors must be non-empty strided tensors")
        if key.layer not in counts or key.expert >= counts[key.layer]:
            raise ValueError(f"expert tensor {key} is outside the declared MoE layout")
        expert_catalog = catalog.setdefault(key.expert_key, {})
        if key.tensor in expert_catalog:
            raise ValueError(f"duplicate expert tensor: {key}")
        expert_catalog[key.tensor] = tensor

    expected = {
        ExpertKey(layer, expert)
        for layer, count in counts.items()
        for expert in range(count)
    }
    missing = sorted(expected - set(catalog))
    extra = sorted(set(catalog) - expected)
    if missing or extra:
        raise ValueError(f"MoE catalog does not match declared experts: missing={missing}, extra={extra}")
    for key, tensors in catalog.items():
        if not tensors:
            raise ValueError(f"expert {key} has no tensors")
    for layer in counts:
        names = {
            frozenset(catalog[ExpertKey(layer, expert)])
            for expert in range(counts[layer])
        }
        if len(names) != 1:
            raise ValueError(f"all experts in layer {layer} must expose the same tensor names")
    return catalog


def publish_moe_layout(
    directory: str | Path,
    expert_tensors: Mapping[object, Tensor] | Iterable[tuple[object, Tensor]],
    *,
    layer_count: int,
    experts_per_layer: int | Mapping[int, int],
    block_elements: int = 1 << 20,
    model_name: str = "koemi-moe",
    identity: ModelIdentity | None = None,
) -> LayoutPublication:
    """Publish a complete MoE expert catalog as atomically-created tensor blocks."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if identity is not None and not isinstance(identity, ModelIdentity):
        raise TypeError("identity must be a ModelIdentity instance")
    block_elements = _positive_int(block_elements, "block_elements")
    counts = _normalise_counts(layer_count, experts_per_layer)
    catalog = _normalise_tensor_catalog(expert_tensors, counts)

    final_directory = Path(directory).expanduser().resolve()
    if not final_directory.name:
        raise ValueError("directory must name a layout directory")
    if os.path.lexists(final_directory):
        raise FileExistsError(f"layout directory already exists: {final_directory}")
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = final_directory.parent / (
        f".{final_directory.name}.staging-{uuid.uuid4().hex}"
    )
    staging_directory.mkdir()

    block_count = 0
    manifest_catalog: list[dict[str, object]] = []
    try:
        for expert_key in sorted(catalog):
            tensor_entries: list[dict[str, object]] = []
            for tensor_name, source_tensor in sorted(catalog[expert_key].items()):
                cpu_tensor = source_tensor.detach().to(device="cpu").contiguous()
                flat_tensor = cpu_tensor.reshape(-1)
                block_entries: list[dict[str, object]] = []
                for block_index, offset in enumerate(range(0, flat_tensor.numel(), block_elements)):
                    block = flat_tensor[offset : offset + block_elements].clone()
                    relative_path = (
                        f"blocks/layer-{expert_key.layer:06d}/"
                        f"expert-{expert_key.expert:06d}/"
                        f"tensor-{tensor_name}/block-{block_index:06d}.pt"
                    )
                    block_path = staging_directory.joinpath(*relative_path.split("/"))
                    block_path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_torch_save(block_path, block)
                    serialized = block_path.read_bytes()
                    block_entries.append(
                        {
                            "path": relative_path,
                            "offset": offset,
                            "length": block.numel(),
                            "sha256": hashlib.sha256(serialized).hexdigest(),
                            "serialized_bytes": len(serialized),
                        }
                    )
                    block_count += 1
                tensor_entries.append(
                    {
                        "name": tensor_name,
                        "shape": [int(dimension) for dimension in cpu_tensor.shape],
                        "dtype": str(cpu_tensor.dtype),
                        "numel": cpu_tensor.numel(),
                        "blocks": block_entries,
                    }
                )
            manifest_catalog.append(
                {
                    "layer": expert_key.layer,
                    "expert": expert_key.expert,
                    "tensors": tensor_entries,
                }
            )

        manifest: dict[str, object] = {
            "format": SUBMAPPING_FORMAT,
            "version": SUBMAPPING_VERSION,
            "model_type": "moe",
            "is_moe": True,
            "model_name": model_name,
            "model_identity": None if identity is None else identity.to_payload(),
            "routing": {
                "kind": ROUTING_KIND,
            },
            "moe": {
                "layer_count": len(counts),
                "experts_per_layer": {str(layer): count for layer, count in sorted(counts.items())},
            },
            "layout": {
                "kind": "expert-blocks",
                "block_elements": block_elements,
                "tensor_axis": "flattened",
                "immutable_blocks": True,
            },
            "block_count": block_count,
            "catalog": manifest_catalog,
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        _atomic_write_bytes(staging_directory / MANIFEST_FILENAME, manifest_bytes)
        _atomic_write_bytes(
            staging_directory / MANIFEST_DIGEST_FILENAME,
            f"{manifest_digest}\n".encode("ascii"),
        )
        os.replace(staging_directory, final_directory)
    except BaseException:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise

    return LayoutPublication(
        directory=final_directory,
        manifest_path=final_directory / MANIFEST_FILENAME,
        manifest_sha256=manifest_digest,
        block_count=block_count,
        expert_count=sum(counts.values()),
    )


def _safe_manifest_path(root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path or "\x00" in raw_path:
        raise UnsafePathError("manifest block path is not a safe relative path")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise UnsafePathError(f"manifest block path is unsafe: {raw_path}")
    if not relative.parts or relative.parts[0] != "blocks":
        raise UnsafePathError("manifest block path must remain below blocks/")
    candidate = root.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise UnsafePathError(f"manifest block path escapes the layout: {raw_path}") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise UnsafePathError(f"manifest block path uses a symlink: {raw_path}")
    if not candidate.is_file():
        raise ManifestError(f"manifest block is missing: {raw_path}")
    return candidate


def _manifest_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"manifest field {name} must be an integer")
    return value


def _parse_manifest(
    root: Path,
    manifest: object,
) -> dict[ExpertKey, dict[str, _TensorRecord]]:
    if not isinstance(manifest, dict):
        raise ManifestError("manifest root must be a JSON object")
    if manifest.get("format") != SUBMAPPING_FORMAT or manifest.get("version") != SUBMAPPING_VERSION:
        raise ManifestError("unsupported submapping manifest format")
    if manifest.get("model_type") != "moe" or manifest.get("is_moe") is not True:
        raise ManifestError("only MoE manifests are supported")
    routing = manifest.get("routing")
    if not isinstance(routing, dict) or routing.get("kind") != ROUTING_KIND:
        raise ManifestError("only content_dispatch_hash MoE routing is supported")
    raw_identity = manifest.get("model_identity")
    if raw_identity is not None:
        try:
            ModelIdentity.from_payload(raw_identity)
        except (TypeError, ValueError) as error:
            raise ManifestError("manifest model identity is invalid") from error

    moe = manifest.get("moe")
    if not isinstance(moe, dict):
        raise ManifestError("manifest does not declare an MoE layout")
    layer_count = _manifest_int(moe.get("layer_count"), "moe.layer_count")
    if layer_count <= 0:
        raise ManifestError("moe.layer_count must be positive")
    raw_counts = moe.get("experts_per_layer")
    if not isinstance(raw_counts, dict):
        raise ManifestError("moe.experts_per_layer must be an object")
    expected_layer_names = {str(layer) for layer in range(layer_count)}
    if set(raw_counts) != expected_layer_names:
        raise ManifestError("moe.experts_per_layer must define every layer")
    counts = {
        layer: _manifest_int(raw_counts[str(layer)], f"moe.experts_per_layer.{layer}")
        for layer in range(layer_count)
    }
    if any(count <= 0 for count in counts.values()):
        raise ManifestError("every MoE layer must contain at least one expert")

    layout = manifest.get("layout")
    if (
        not isinstance(layout, dict)
        or layout.get("kind") != "expert-blocks"
        or layout.get("immutable_blocks") is not True
    ):
        raise ManifestError("manifest does not declare expert blocks")
    if layout.get("tensor_axis") != "flattened":
        raise ManifestError("unsupported expert block axis")
    _manifest_int(layout.get("block_elements"), "layout.block_elements")
    if layout["block_elements"] <= 0:
        raise ManifestError("layout.block_elements must be positive")

    raw_catalog = manifest.get("catalog")
    if not isinstance(raw_catalog, list):
        raise ManifestError("manifest catalog must be a list")
    catalog: dict[ExpertKey, dict[str, _TensorRecord]] = {}
    seen_paths: set[str] = set()
    block_count = 0
    for raw_expert in raw_catalog:
        if not isinstance(raw_expert, dict):
            raise ManifestError("each catalog expert must be an object")
        layer = _manifest_int(raw_expert.get("layer"), "catalog.layer")
        expert = _manifest_int(raw_expert.get("expert"), "catalog.expert")
        if layer < 0 or layer >= layer_count:
            raise ManifestError(f"catalog expert is outside the declared MoE layout: layer={layer}")
        if expert < 0 or expert >= counts[layer]:
            raise ManifestError(
                f"catalog expert is outside the declared MoE layout: layer={layer}, expert={expert}"
            )
        key = ExpertKey(layer, expert)
        if key in catalog:
            raise ManifestError(f"duplicate catalog expert: {key}")
        raw_tensors = raw_expert.get("tensors")
        if not isinstance(raw_tensors, list) or not raw_tensors:
            raise ManifestError(f"catalog expert has no tensors: {key}")
        tensor_catalog: dict[str, _TensorRecord] = {}
        for raw_tensor in raw_tensors:
            if not isinstance(raw_tensor, dict):
                raise ManifestError("each catalog tensor must be an object")
            name = raw_tensor.get("name")
            try:
                _safe_component(name, "catalog tensor name")
            except ValueError as error:
                raise UnsafePathError(str(error)) from error
            if name in tensor_catalog:
                raise ManifestError(f"duplicate tensor in expert {key}: {name}")
            raw_shape = raw_tensor.get("shape")
            if not isinstance(raw_shape, list) or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in raw_shape
            ):
                raise ManifestError(f"invalid shape for {key}/{name}")
            shape = tuple(raw_shape)
            numel = _manifest_int(raw_tensor.get("numel"), f"{key}/{name}.numel")
            if numel <= 0 or math.prod(shape) != numel:
                raise ManifestError(f"tensor element count does not match shape for {key}/{name}")
            dtype = raw_tensor.get("dtype")
            if not isinstance(dtype, str) or not re.fullmatch(r"torch\.[A-Za-z0-9_]+", dtype):
                raise ManifestError(f"invalid tensor dtype for {key}/{name}")
            raw_blocks = raw_tensor.get("blocks")
            if not isinstance(raw_blocks, list) or not raw_blocks:
                raise ManifestError(f"tensor has no blocks for {key}/{name}")
            blocks: list[_BlockRecord] = []
            expected_offset = 0
            for raw_block in raw_blocks:
                if not isinstance(raw_block, dict):
                    raise ManifestError(f"invalid block record for {key}/{name}")
                raw_path = raw_block.get("path")
                path = _safe_manifest_path(root, raw_path)
                if raw_path in seen_paths:
                    raise ManifestError(f"block path is listed more than once: {raw_path}")
                seen_paths.add(raw_path)
                offset = _manifest_int(raw_block.get("offset"), f"{raw_path}.offset")
                length = _manifest_int(raw_block.get("length"), f"{raw_path}.length")
                serialized_bytes = _manifest_int(
                    raw_block.get("serialized_bytes"), f"{raw_path}.serialized_bytes"
                )
                digest = raw_block.get("sha256")
                if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                    raise ManifestError(f"invalid block digest for {raw_path}")
                if offset != expected_offset or length <= 0 or serialized_bytes <= 0:
                    raise ManifestError(f"non-contiguous block range for {key}/{name}")
                blocks.append(
                    _BlockRecord(
                        path=raw_path,
                        offset=offset,
                        length=length,
                        sha256=digest,
                        serialized_bytes=serialized_bytes,
                    )
                )
                expected_offset += length
                block_count += 1
            if expected_offset != numel:
                raise ManifestError(f"block ranges do not cover {key}/{name}")
            tensor_catalog[name] = _TensorRecord(
                name=name,
                shape=shape,
                dtype=dtype,
                numel=numel,
                blocks=tuple(blocks),
            )
        catalog[key] = tensor_catalog

    expected_experts = {
        ExpertKey(layer, expert)
        for layer, count in counts.items()
        for expert in range(count)
    }
    if set(catalog) != expected_experts:
        raise ManifestError("manifest catalog is incomplete or contains unexpected experts")
    for layer, count in counts.items():
        tensor_names = {
            frozenset(catalog[ExpertKey(layer, expert)])
            for expert in range(count)
        }
        if len(tensor_names) != 1:
            raise ManifestError(f"experts in layer {layer} expose different tensor names")
    if _manifest_int(manifest.get("block_count"), "block_count") != block_count:
        raise ManifestError("manifest block_count does not match the catalog")
    return catalog


class MoeSubmappingStore:
    """MoE-only block store with bounded RAM residency and exact execution."""

    def __init__(
        self,
        directory: Path,
        manifest: dict[str, object],
        catalog: dict[ExpertKey, dict[str, _TensorRecord]],
        *,
        cache_bytes: int,
        prefetch_limit: int,
    ) -> None:
        self.directory = directory
        self._manifest = manifest
        self._catalog = catalog
        self.cache_bytes = _nonnegative_int(cache_bytes, "cache_bytes")
        self.prefetch_limit = _nonnegative_int(prefetch_limit, "prefetch_limit")
        self._cache: OrderedDict[ExpertKey, Mapping[str, Tensor]] = OrderedDict()
        self._cache_sizes: dict[ExpertKey, int] = {}
        self._resident_bytes = 0
        self._metrics = SubmappingMetrics()
        self._last_metrics = ReadMetrics(0, 0, 0, 0, 0.0, 0)
        self._last_prefetch_metrics = self._last_metrics

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        cache_bytes: int = 0,
        prefetch_limit: int = 8,
    ) -> MoeSubmappingStore:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise ManifestError(f"layout directory is missing: {root}")
        manifest_path = root / MANIFEST_FILENAME
        digest_path = root / MANIFEST_DIGEST_FILENAME
        for path, label in ((manifest_path, "manifest"), (digest_path, "manifest digest")):
            if path.is_symlink() or not path.is_file():
                raise ManifestError(f"{label} file is missing or unsafe")
        manifest_bytes = manifest_path.read_bytes()
        expected_digest = digest_path.read_text(encoding="ascii").strip()
        if not _SHA256.fullmatch(expected_digest):
            raise ManifestIntegrityError("manifest SHA-256 sidecar is invalid")
        actual_digest = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_digest != expected_digest:
            raise ManifestIntegrityError("manifest SHA-256 does not match manifest JSON")
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ManifestError("manifest JSON is invalid") from error
        catalog = _parse_manifest(root, manifest)
        return cls(
            root,
            manifest,
            catalog,
            cache_bytes=cache_bytes,
            prefetch_limit=prefetch_limit,
        )

    @property
    def manifest(self) -> dict[str, object]:
        return deepcopy(self._manifest)

    @property
    def metrics(self) -> SubmappingMetrics:
        return replace(self._metrics)

    @property
    def last_metrics(self) -> ReadMetrics:
        return self._last_metrics

    @property
    def last_prefetch_metrics(self) -> ReadMetrics:
        return self._last_prefetch_metrics

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def cached_experts(self) -> tuple[ExpertKey, ...]:
        return tuple(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_sizes.clear()
        self._resident_bytes = 0

    def _coerce_expert_key(self, value: object) -> ExpertKey:
        if isinstance(value, ExpertKey):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            return ExpertKey(value[0], value[1])
        raise TypeError("expert selections must be ExpertKey or (layer, expert)")

    def _validate_selected(self, keys: Sequence[ExpertKey]) -> None:
        for key in keys:
            if key not in self._catalog:
                raise UnknownExpertError(
                    f"expert (layer={key.layer}, expert={key.expert}) is outside the MoE catalog"
                )

    def _cache_get(self, key: ExpertKey) -> Mapping[str, Tensor] | None:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
        return cached

    def _cache_put(self, key: ExpertKey, tensors: Mapping[str, Tensor]) -> int:
        size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if size > self.cache_bytes or self.cache_bytes == 0:
            return 0
        evictions = 0
        while self._cache and self._resident_bytes + size > self.cache_bytes:
            evicted_key, _ = self._cache.popitem(last=False)
            self._resident_bytes -= self._cache_sizes.pop(evicted_key)
            evictions += 1
        self._cache[key] = tensors
        self._cache_sizes[key] = size
        self._resident_bytes += size
        return evictions

    def _load_tensor(self, record: _TensorRecord) -> tuple[Tensor, int]:
        pieces: list[Tensor] = []
        bytes_read = 0
        for block in record.blocks:
            block_path = self.directory.joinpath(*PurePosixPath(block.path).parts)
            serialized = block_path.read_bytes()
            actual_digest = hashlib.sha256(serialized).hexdigest()
            if actual_digest != block.sha256:
                raise BlockIntegrityError(f"SHA-256 mismatch for selected block: {block.path}")
            if len(serialized) != block.serialized_bytes:
                raise BlockIntegrityError(f"serialized size mismatch for selected block: {block.path}")
            try:
                loaded = torch.load(
                    io.BytesIO(serialized),
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            except Exception as error:
                raise BlockIntegrityError(f"selected block cannot be loaded: {block.path}") from error
            if not isinstance(loaded, Tensor) or loaded.layout != torch.strided:
                raise BlockIntegrityError(f"selected block is not a dense tensor: {block.path}")
            if loaded.numel() != block.length or str(loaded.dtype) != record.dtype:
                raise BlockIntegrityError(f"selected block tensor contract is invalid: {block.path}")
            pieces.append(loaded.detach().to(device="cpu").contiguous().reshape(-1))
            bytes_read += len(serialized)
        flattened = torch.cat(pieces, dim=0)
        if flattened.numel() != record.numel:
            raise BlockIntegrityError(f"selected tensor length is invalid: {record.name}")
        try:
            restored = flattened.reshape(record.shape)
        except RuntimeError as error:
            raise BlockIntegrityError(f"selected tensor shape is invalid: {record.name}") from error
        if str(restored.dtype) != record.dtype:
            raise BlockIntegrityError(f"selected tensor dtype is invalid: {record.name}")
        return restored, bytes_read

    def _load_expert(self, key: ExpertKey) -> tuple[Mapping[str, Tensor], int, float]:
        started = time.perf_counter()
        tensors: dict[str, Tensor] = {}
        bytes_read = 0
        for name, record in self._catalog[key].items():
            tensor, tensor_bytes = self._load_tensor(record)
            tensors[name] = tensor
            bytes_read += tensor_bytes
        return MappingProxyType(tensors), bytes_read, time.perf_counter() - started

    @staticmethod
    def _execution_view(tensors: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Give an injected executor exact values without exposing cached storage."""

        return MappingProxyType({name: tensor.detach().clone() for name, tensor in tensors.items()})

    def _read_selected(self, keys: Sequence[ExpertKey]) -> dict[ExpertKey, Mapping[str, Tensor]]:
        self._validate_selected(keys)
        unique_keys = list(dict.fromkeys(keys))
        cache_hits = 0
        bytes_read = 0
        read_seconds = 0.0
        evictions = 0
        result: dict[ExpertKey, Mapping[str, Tensor]] = {}
        for key in unique_keys:
            cached = self._cache_get(key)
            if cached is not None:
                cache_hits += 1
                result[key] = cached
                continue
            tensors, loaded_bytes, elapsed = self._load_expert(key)
            bytes_read += loaded_bytes
            read_seconds += elapsed
            evictions += self._cache_put(key, tensors)
            result[key] = tensors
        call_metrics = ReadMetrics(
            requested=len(keys),
            unique=len(unique_keys),
            cache_hits=cache_hits,
            bytes_read=bytes_read,
            read_seconds=read_seconds,
            evictions=evictions,
        )
        self._last_metrics = call_metrics
        self._metrics.requested += call_metrics.requested
        self._metrics.unique += call_metrics.unique
        self._metrics.cache_hits += call_metrics.cache_hits
        self._metrics.bytes_read += call_metrics.bytes_read
        self._metrics.read_seconds += call_metrics.read_seconds
        self._metrics.evictions += call_metrics.evictions
        return result

    def read_experts(
        self,
        selections: Iterable[ExpertKey | tuple[int, int]],
    ) -> dict[ExpertKey, Mapping[str, Tensor]]:
        keys = [self._coerce_expert_key(selection) for selection in selections]
        return self._read_selected(keys)

    def _ids_for_layer(self, layer: int, expert_ids: int | Iterable[int]) -> list[ExpertKey]:
        layer = _nonnegative_int(layer, "layer")
        if isinstance(expert_ids, bool):
            raise ValueError("expert_ids must contain integers")
        if isinstance(expert_ids, int):
            raw_ids = [expert_ids]
        elif isinstance(expert_ids, Tensor):
            if expert_ids.ndim != 1:
                raise ValueError("expert_ids tensor must be one-dimensional")
            raw_ids = expert_ids.detach().to(device="cpu").tolist()
        else:
            raw_ids = list(expert_ids)
        return [ExpertKey(layer, _nonnegative_int(expert_id, "expert_id")) for expert_id in raw_ids]

    def prefetch(self, layer: int, expert_ids: int | Iterable[int]) -> ReadMetrics:
        """Synchronously cache selected experts; it never executes or changes outputs."""

        keys = self._ids_for_layer(layer, expert_ids)
        if len(keys) > self.prefetch_limit:
            raise ValueError(
                f"prefetch is limited to {self.prefetch_limit} expert selections per call"
            )
        self._read_selected(keys)
        self._last_prefetch_metrics = self._last_metrics
        return self._last_prefetch_metrics

    def execute_moe(
        self,
        layer: int,
        expert_ids: int | Iterable[int],
        inputs: Tensor,
        expert_executor: Callable[[Tensor, Mapping[str, Tensor]], Tensor],
        *,
        gate_weights: Sequence[float] | Tensor | None = None,
    ) -> Tensor:
        """Run only the selected experts and combine their injected outputs.

        Duplicate top-k IDs are loaded and executed once, while their supplied
        weights are summed so the returned value remains equivalent to a
        reference that evaluates every requested slot.
        """

        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if not callable(expert_executor):
            raise TypeError("expert_executor must be callable")
        keys = self._ids_for_layer(layer, expert_ids)
        if not keys:
            raise ValueError("a MoE call must select at least one expert")
        loaded = self._read_selected(keys)
        unique_keys = list(dict.fromkeys(keys))

        if gate_weights is None:
            raw_weights = [1.0 / len(keys)] * len(keys)
        elif isinstance(gate_weights, Tensor):
            if gate_weights.ndim != 1:
                raise ValueError("gate_weights tensor must be one-dimensional")
            raw_weights = gate_weights.detach().to(device="cpu").tolist()
        else:
            raw_weights = list(gate_weights)
        if len(raw_weights) != len(keys):
            raise ValueError("gate_weights must contain one value per selected expert")
        weights: list[float] = []
        for weight in raw_weights:
            if isinstance(weight, bool):
                raise ValueError("gate weights must be finite numbers")
            try:
                parsed = float(weight)
            except (TypeError, ValueError) as error:
                raise ValueError("gate weights must be finite numbers") from error
            if not math.isfinite(parsed):
                raise ValueError("gate weights must be finite numbers")
            weights.append(parsed)
        combined_weights = {key: 0.0 for key in unique_keys}
        for key, weight in zip(keys, weights):
            combined_weights[key] += weight

        outputs: dict[ExpertKey, Tensor] = {}
        expected_shape: torch.Size | None = None
        expected_device: torch.device | None = None
        for key in unique_keys:
            output = expert_executor(inputs, self._execution_view(loaded[key]))
            if not isinstance(output, Tensor):
                raise TypeError("expert_executor must return a torch.Tensor")
            if expected_shape is None:
                expected_shape = output.shape
                expected_device = output.device
            elif output.shape != expected_shape or output.device != expected_device:
                raise ValueError("all expert outputs must have the same shape and device")
            outputs[key] = output

        result: Tensor | None = None
        for key in unique_keys:
            term = outputs[key] * combined_weights[key]
            result = term if result is None else result + term
        if result is None:
            raise RuntimeError("MoE execution produced no output")
        return result

    run_moe = execute_moe


SubmappingStore = MoeSubmappingStore


__all__ = [
    "BlockIntegrityError",
    "ExpertKey",
    "ExpertTensorKey",
    "LayoutPublication",
    "ManifestError",
    "ManifestIntegrityError",
    "MoeSubmappingStore",
    "ReadMetrics",
    "ROUTING_KIND",
    "SubmappingError",
    "SubmappingMetrics",
    "SubmappingStore",
    "UnknownExpertError",
    "UnsafePathError",
    "publish_moe_layout",
    "route_moe_experts",
]
