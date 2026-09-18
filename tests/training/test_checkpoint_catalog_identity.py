from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import torch

from koemi.model.identity import ModelIdentity, ModelLicensing
from koemi.training.checkpoint_catalog import CheckpointCatalog, CheckpointManifestError


def build_identity() -> ModelIdentity:
    return ModelIdentity(
        organization="Koemi Labs",
        model_name="Koemi HERM",
        version="0.4.0",
        heading="Koemi HERM by Koemi Labs",
        licensing=ModelLicensing(
            source_license="MIT",
            source_availability="open",
            weights_availability="closed",
            data_availability="unknown",
        ),
    )


class CheckpointCatalogIdentityTests(unittest.TestCase):
    def test_manifest_carries_model_identity_without_entering_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            catalog = CheckpointCatalog(Path(temporary_directory) / "catalog")
            publication = catalog.publish(
                {"model_state": {"weight": torch.ones(2)}},
                identity=build_identity(),
                metadata={"run_signature": "run-1", "optimizer_step": 7},
            )
            recovery = catalog.load_latest()

        assert recovery.checkpoint is not None
        restored_identity = ModelIdentity.from_payload(
            recovery.checkpoint.manifest.metadata["model_identity"]
        )
        self.assertEqual(build_identity(), restored_identity)
        self.assertEqual("run-1", recovery.checkpoint.manifest.metadata["run_signature"])
        self.assertNotIn("model_identity", recovery.payload)
        self.assertEqual(publication.generation, recovery.generation)

    def test_manifest_identity_schema_is_validated_on_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            catalog = CheckpointCatalog(Path(temporary_directory) / "catalog")
            publication = catalog.publish({"model_state": {"weight": torch.ones(1)}}, identity=build_identity())
            manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
            manifest["metadata"]["model_identity"]["model_name"] = ""
            publication.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(CheckpointManifestError):
                catalog.load_generation(publication.generation)


if __name__ == "__main__":
    unittest.main()
