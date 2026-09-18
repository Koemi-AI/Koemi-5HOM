"""Immutable, hash-verified checkpoint generations.

This catalog is deliberately separate from the legacy checkpoint writers.  A
published generation is never overwritten; recovery scans generations and
loads only payloads that pass path, size, digest, manifest, and
``weights_only`` validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from koemi.model.identity import ModelIdentity


FORMAT_VERSION = 1
PAYLOAD_FILENAME = "payload.pt"
MANIFEST_FILENAME = "manifest.json"
LATEST_FILENAME = "latest.json"
_GENERATION_PATTERN = re.compile(r"^generation-(?P<number>[0-9]{20})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CheckpointCatalogError(RuntimeError):
    """Base error for catalog publication and recovery failures."""


class CheckpointPathError(CheckpointCatalogError):
    """The catalog path or a manifest path is unsafe or unusable."""


class CheckpointPayloadError(CheckpointCatalogError):
    """The payload cannot be published as a weights-only artifact."""


class CheckpointManifestError(CheckpointCatalogError):
    """A generation manifest or its payload failed validation."""


@dataclass(frozen=True)
class CheckpointManifest:
    format_version: int
    generation: int
    payload_filename: str
    payload_sha256: str
    payload_size_bytes: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CheckpointPublication:
    generation: int
    generation_path: Path
    manifest_path: Path
    payload_path: Path
    retained_generations: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointRecord:
    generation: int
    generation_path: Path
    manifest_path: Path
    payload_path: Path
    manifest: CheckpointManifest
    payload: Any


@dataclass(frozen=True)
class CheckpointRecovery:
    checkpoint: CheckpointRecord | None
    pointer_generation: int | None
    rejected_generations: tuple[int, ...]
    diagnostics: tuple[str, ...]

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint is not None

    @property
    def generation(self) -> int | None:
        return None if self.checkpoint is None else self.checkpoint.generation

    @property
    def payload(self) -> Any:
        return None if self.checkpoint is None else self.checkpoint.payload


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointManifestError("manifest metadata must be finite JSON") from exc
    return (encoded + "\n").encode("utf-8")


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise CheckpointCatalogError(f"could not atomically write {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_generation_number(generation: int) -> int:
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise CheckpointManifestError("generation must be a positive integer")
    return generation


def _validate_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise CheckpointManifestError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_payload_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise CheckpointManifestError("payload_filename must be a single relative filename")
    if value != PAYLOAD_FILENAME:
        raise CheckpointManifestError(f"payload_filename must be {PAYLOAD_FILENAME!r}")
    return value


class CheckpointCatalog:
    """Publish and recover immutable, weights-only checkpoint generations."""

    def __init__(self, root: str | os.PathLike[str], *, retention: int = 3) -> None:
        if isinstance(retention, bool) or not isinstance(retention, int) or retention < 1:
            raise ValueError("retention must be a positive integer")
        self.root = Path(root)
        if self.root.exists() and not self.root.is_dir():
            raise CheckpointPathError(f"catalog path is not a directory: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / LATEST_FILENAME
        self.retention = retention

    def publish(
        self,
        payload: Any,
        *,
        identity: ModelIdentity | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointPublication:
        """Atomically append one generation after weights-only validation."""
        if identity is not None and not isinstance(identity, ModelIdentity):
            raise TypeError("identity must be a ModelIdentity instance")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        manifest_metadata = dict(metadata or {})
        if identity is not None:
            manifest_metadata["model_identity"] = identity.to_payload()
        elif "model_identity" in manifest_metadata:
            raw_identity = manifest_metadata["model_identity"]
            if not isinstance(raw_identity, Mapping):
                raise CheckpointManifestError("metadata.model_identity must be a mapping")
            try:
                ModelIdentity.from_payload(raw_identity)
            except (TypeError, ValueError) as exc:
                raise CheckpointManifestError("metadata.model_identity is invalid") from exc
        _json_bytes(manifest_metadata)

        generation = self._next_generation()
        staging = self.root / f".generation-{generation:020d}-{uuid.uuid4().hex}.tmp"
        final_path = self._generation_path(generation)
        staging.mkdir()
        payload_path = staging / PAYLOAD_FILENAME
        manifest_path = staging / MANIFEST_FILENAME
        try:
            try:
                torch.save(payload, payload_path)
                _fsync_file(payload_path)
                torch.load(payload_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise CheckpointPayloadError(
                    "payload is not readable with torch.load(weights_only=True)"
                ) from exc

            payload_size = payload_path.stat().st_size
            payload_digest = _sha256_file(payload_path)
            manifest = {
                "format_version": FORMAT_VERSION,
                "generation": generation,
                "payload_filename": PAYLOAD_FILENAME,
                "payload_sha256": payload_digest,
                "payload_size_bytes": payload_size,
                "metadata": manifest_metadata,
            }
            manifest_bytes = _json_bytes(manifest)
            with manifest_path.open("wb") as stream:
                stream.write(manifest_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging, final_path)
            staging = final_path
            latest = {
                "format_version": FORMAT_VERSION,
                "generation": generation,
                "manifest_sha256": _sha256_bytes(manifest_bytes),
            }
            _atomic_write_bytes(self.latest_path, _json_bytes(latest))
        except CheckpointCatalogError:
            self._remove_tree(staging)
            raise
        except Exception as exc:
            self._remove_tree(staging)
            raise CheckpointCatalogError("checkpoint publication failed") from exc

        retained = self._apply_retention(generation)
        return CheckpointPublication(
            generation=generation,
            generation_path=final_path,
            manifest_path=final_path / MANIFEST_FILENAME,
            payload_path=final_path / PAYLOAD_FILENAME,
            retained_generations=retained,
        )

    def load_generation(self, generation: int) -> CheckpointRecord:
        """Load one generation, failing closed on any integrity mismatch."""
        generation = _validate_generation_number(generation)
        generation_path = self._generation_path(generation)
        if generation_path.is_symlink() or not generation_path.is_dir():
            raise CheckpointManifestError(f"generation directory is missing: {generation}")
        manifest_path = generation_path / MANIFEST_FILENAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise CheckpointManifestError(f"generation {generation} manifest is missing or unsafe")
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CheckpointManifestError(f"generation {generation} has an invalid manifest") from exc
        manifest = self._parse_manifest(raw_manifest, generation)
        payload_path = generation_path / manifest.payload_filename
        if payload_path.is_symlink() or not payload_path.is_file():
            raise CheckpointManifestError(f"generation {generation} payload is missing")
        try:
            actual_size = payload_path.stat().st_size
        except OSError as exc:
            raise CheckpointManifestError(f"generation {generation} payload cannot be stat-ed") from exc
        if actual_size != manifest.payload_size_bytes:
            raise CheckpointManifestError(
                f"generation {generation} payload size mismatch: "
                f"expected {manifest.payload_size_bytes}, got {actual_size}"
            )
        actual_digest = _sha256_file(payload_path)
        if actual_digest != manifest.payload_sha256:
            raise CheckpointManifestError(f"generation {generation} payload SHA-256 mismatch")
        try:
            payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointManifestError(
                f"generation {generation} payload cannot be loaded with weights_only=True"
            ) from exc
        return CheckpointRecord(
            generation=generation,
            generation_path=generation_path,
            manifest_path=manifest_path,
            payload_path=payload_path,
            manifest=manifest,
            payload=payload,
        )

    def load_latest(self) -> CheckpointRecovery:
        """Recover the greatest valid generation and report rejected candidates."""
        diagnostics: list[str] = []
        pointer_generation: int | None = None
        try:
            pointer = json.loads(self.latest_path.read_text(encoding="utf-8"))
            if not isinstance(pointer, Mapping):
                raise ValueError("pointer must be an object")
            pointer_generation = _validate_generation_number(pointer.get("generation"))
            pointer_manifest = _validate_sha256(pointer.get("manifest_sha256"), "manifest_sha256")
            pointer_manifest_path = self._generation_path(pointer_generation) / MANIFEST_FILENAME
            if not pointer_manifest_path.is_file():
                raise CheckpointManifestError("pointer generation manifest is missing")
            if _sha256_file(pointer_manifest_path) != pointer_manifest:
                raise CheckpointManifestError("pointer manifest SHA-256 mismatch")
        except FileNotFoundError:
            pass
        except Exception as exc:
            pointer_generation = None
            diagnostics.append(f"latest_pointer_invalid: {exc}")

        valid: list[CheckpointRecord] = []
        rejected: list[int] = []
        for generation in self._generation_numbers():
            try:
                valid.append(self.load_generation(generation))
            except Exception as exc:
                rejected.append(generation)
                diagnostics.append(f"generation_{generation}_rejected: {exc}")
        valid.sort(key=lambda checkpoint: checkpoint.generation, reverse=True)
        checkpoint = valid[0] if valid else None
        return CheckpointRecovery(
            checkpoint=checkpoint,
            pointer_generation=pointer_generation,
            rejected_generations=tuple(rejected),
            diagnostics=tuple(diagnostics),
        )

    def recover_latest(self) -> CheckpointRecovery:
        """Compatibility spelling for :meth:`load_latest`."""
        return self.load_latest()

    def _parse_manifest(self, raw: Any, expected_generation: int) -> CheckpointManifest:
        if not isinstance(raw, Mapping):
            raise CheckpointManifestError("manifest must be an object")
        if raw.get("format_version") != FORMAT_VERSION:
            raise CheckpointManifestError("unsupported checkpoint manifest format")
        generation = _validate_generation_number(raw.get("generation"))
        if generation != expected_generation:
            raise CheckpointManifestError("manifest generation does not match its directory")
        payload_filename = _validate_payload_filename(raw.get("payload_filename"))
        payload_sha256 = _validate_sha256(raw.get("payload_sha256"), "payload_sha256")
        payload_size = raw.get("payload_size_bytes")
        if isinstance(payload_size, bool) or not isinstance(payload_size, int) or payload_size < 1:
            raise CheckpointManifestError("payload_size_bytes must be a positive integer")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise CheckpointManifestError("metadata must be an object")
        metadata_copy = dict(metadata)
        if "model_identity" in metadata_copy:
            raw_identity = metadata_copy["model_identity"]
            if not isinstance(raw_identity, Mapping):
                raise CheckpointManifestError("metadata.model_identity must be a mapping")
            try:
                ModelIdentity.from_payload(raw_identity)
            except (TypeError, ValueError) as exc:
                raise CheckpointManifestError("metadata.model_identity is invalid") from exc
        _json_bytes(metadata_copy)
        return CheckpointManifest(
            format_version=FORMAT_VERSION,
            generation=generation,
            payload_filename=payload_filename,
            payload_sha256=payload_sha256,
            payload_size_bytes=payload_size,
            metadata=metadata_copy,
        )

    def _next_generation(self) -> int:
        numbers = self._generation_numbers()
        return (max(numbers) + 1) if numbers else 1

    def _generation_numbers(self) -> list[int]:
        numbers: list[int] = []
        for path in self.root.iterdir():
            match = _GENERATION_PATTERN.fullmatch(path.name)
            if match and path.is_dir():
                numbers.append(int(match.group("number")))
        return sorted(numbers)

    def _generation_path(self, generation: int) -> Path:
        return self.root / f"generation-{generation:020d}"

    def _apply_retention(self, newest: int) -> tuple[int, ...]:
        numbers = self._generation_numbers()
        protected = set(numbers[-self.retention :])
        protected.add(newest)
        for generation in numbers:
            if generation in protected:
                continue
            self._remove_tree(self._generation_path(generation))
        return tuple(sorted(self._generation_numbers()))

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


__all__ = [
    "CheckpointCatalog",
    "CheckpointCatalogError",
    "CheckpointManifest",
    "CheckpointManifestError",
    "CheckpointPathError",
    "CheckpointPayloadError",
    "CheckpointPublication",
    "CheckpointRecord",
    "CheckpointRecovery",
]
