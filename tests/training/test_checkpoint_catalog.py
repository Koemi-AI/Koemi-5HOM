from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from koemi.training.checkpoint_catalog import (
    CheckpointCatalog,
    CheckpointManifestError,
    CheckpointPathError,
    CheckpointPayloadError,
)


def _mark_unsafe_load(marker_path: str) -> None:
    Path(marker_path).write_text("executed", encoding="utf-8")


class UnsafePayload:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _mark_unsafe_load, (str(self.marker_path),)


class CheckpointCatalogTests(unittest.TestCase):
    def payload(self, step: int) -> dict[str, object]:
        return {
            "optimizer_step": step,
            "model_state": {"weight": torch.tensor([float(step), 2.0])},
            "training_state": {"epoch_index": step // 2, "tokens_seen": step * 10},
            "metrics": [0.5, step],
        }

    def test_publication_writes_manifest_and_verified_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = CheckpointCatalog(Path(directory) / "catalog")
            publication = catalog.publish(self.payload(1))

            self.assertEqual(1, publication.generation)
            self.assertTrue(publication.generation_path.is_dir())
            self.assertEqual(publication.generation_path / "manifest.json", publication.manifest_path)
            self.assertEqual(publication.generation_path / "payload.pt", publication.payload_path)
            manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(1, manifest["format_version"])
            self.assertEqual(1, manifest["generation"])
            self.assertEqual("payload.pt", manifest["payload_filename"])
            self.assertEqual(
                hashlib.sha256(publication.payload_path.read_bytes()).hexdigest(),
                manifest["payload_sha256"],
            )
            self.assertEqual(publication.payload_path.stat().st_size, manifest["payload_size_bytes"])
            self.assertTrue(catalog.latest_path.is_file())

    def test_latest_returns_the_greatest_valid_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = CheckpointCatalog(Path(directory) / "catalog")
            catalog.publish(self.payload(1))
            catalog.publish(self.payload(2))

            recovery = catalog.recover_latest()

            self.assertTrue(recovery.has_checkpoint)
            self.assertEqual(2, recovery.generation)
            self.assertEqual(2, recovery.payload["optimizer_step"])
            self.assertEqual(2, recovery.pointer_generation)
            self.assertEqual((), recovery.rejected_generations)

    def test_corrupt_newest_payload_recovers_deterministically_to_older_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = CheckpointCatalog(Path(directory) / "catalog")
            first = catalog.publish(self.payload(1))
            second = catalog.publish(self.payload(2))
            second.payload_path.write_bytes(b"corrupt payload")

            recovery = catalog.load_latest()

            self.assertEqual(first.generation, recovery.generation)
            self.assertEqual(1, recovery.payload["optimizer_step"])
            self.assertEqual((second.generation,), recovery.rejected_generations)
            self.assertTrue(
                any(
                    "payload size" in diagnostic or "SHA-256" in diagnostic
                    for diagnostic in recovery.diagnostics
                )
            )

    def test_corrupt_latest_pointer_falls_back_to_generation_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = CheckpointCatalog(Path(directory) / "catalog")
            catalog.publish(self.payload(1))
            catalog.publish(self.payload(2))
            catalog.latest_path.write_text("not-json", encoding="utf-8")

            recovery = catalog.load_latest()

            self.assertEqual(2, recovery.generation)
            self.assertIsNone(recovery.pointer_generation)
            self.assertTrue(any("latest_pointer_invalid" in diagnostic for diagnostic in recovery.diagnostics))

    def test_retention_removes_generations_older_than_the_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "catalog"
            catalog = CheckpointCatalog(store_path, retention=2)
            catalog.publish(self.payload(1))
            catalog.publish(self.payload(2))
            publication = catalog.publish(self.payload(3))

            generation_names = sorted(path.name for path in store_path.glob("generation-*") if path.is_dir())

            self.assertEqual(
                ["generation-00000000000000000002", "generation-00000000000000000003"],
                generation_names,
            )
            self.assertEqual((2, 3), publication.retained_generations)
            self.assertEqual(3, catalog.load_latest().generation)

    def test_manifest_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = CheckpointCatalog(Path(directory) / "catalog")
            publication = catalog.publish(self.payload(1))
            manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
            manifest["payload_sha256"] = "0" * 64
            publication.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            recovery = catalog.load_latest()

            self.assertFalse(recovery.has_checkpoint)
            self.assertEqual((1,), recovery.rejected_generations)
            self.assertTrue(any("SHA-256" in diagnostic for diagnostic in recovery.diagnostics))

    def test_invalid_manifest_payload_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = CheckpointCatalog(Path(directory) / "catalog")
            publication = catalog.publish(self.payload(1))
            manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
            manifest["payload_filename"] = "../outside.pt"
            publication.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(CheckpointManifestError):
                catalog.load_generation(1)
            recovery = catalog.load_latest()

            self.assertFalse(recovery.has_checkpoint)
            self.assertEqual((1,), recovery.rejected_generations)

    def test_existing_file_cannot_be_used_as_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid_store = Path(directory) / "store-file"
            invalid_store.write_text("not a directory", encoding="utf-8")

            with self.assertRaises(CheckpointPathError):
                CheckpointCatalog(invalid_store)

    def test_unsafe_payload_is_rejected_and_temporary_files_are_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "catalog"
            catalog = CheckpointCatalog(store_path)

            with self.assertRaises(CheckpointPayloadError):
                catalog.publish(object())

            self.assertEqual([], list(store_path.glob("generation-*")))
            self.assertEqual([], list(store_path.glob("*.tmp")))
            self.assertEqual([], list(store_path.glob(".generation-*")))

    def test_recovery_never_executes_an_unsafe_serialized_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "catalog"
            catalog = CheckpointCatalog(store_path)
            generation_path = store_path / "generation-00000000000000000001"
            generation_path.mkdir()
            payload_path = generation_path / "payload.pt"
            marker_path = Path(directory) / "unsafe-load-marker"
            torch.save(UnsafePayload(marker_path), payload_path)
            manifest = {
                "format_version": 1,
                "generation": 1,
                "payload_filename": "payload.pt",
                "payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                "payload_size_bytes": payload_path.stat().st_size,
            }
            manifest_path = generation_path / "manifest.json"
            manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
            manifest_path.write_bytes(manifest_bytes)
            catalog.latest_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "generation": 1,
                        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            recovery = catalog.load_latest()

            self.assertFalse(recovery.has_checkpoint)
            self.assertFalse(marker_path.exists())
            self.assertEqual((1,), recovery.rejected_generations)


if __name__ == "__main__":
    unittest.main()
