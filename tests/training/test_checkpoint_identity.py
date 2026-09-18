from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from koemi.configuration.settings import ModelSettings
from koemi.model.identity import ModelIdentity, ModelLicensing, get_model_identity
from koemi.model.network import KoemiModel
from koemi.training.checkpoints import CheckpointStore


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


def build_model() -> KoemiModel:
    return KoemiModel(
        ModelSettings(
            embedding_size=8,
            memory_features=2,
            local_memory_size=1,
            salience_memory_size=1,
        )
    )


class CheckpointIdentityTests(unittest.TestCase):
    def test_checkpoint_round_trip_binds_identity_to_loaded_model(self) -> None:
        model = build_model()
        identity = build_identity()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.pt"
            CheckpointStore().save(path, model, identity=identity)
            loaded = CheckpointStore().load(path)

        self.assertEqual(identity, loaded.identity)
        self.assertEqual(identity, get_model_identity(loaded.model))

    def test_legacy_checkpoint_without_identity_stays_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.pt"
            CheckpointStore().save(path, build_model())
            loaded = CheckpointStore().load(path)

        self.assertIsNone(loaded.identity)
        self.assertIsNone(get_model_identity(loaded.model))

    def test_invalid_identity_payload_is_rejected_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.pt"
            CheckpointStore().save(path, build_model(), identity=build_identity())
            raw = torch.load(path, map_location="cpu", weights_only=True)
            raw["identity"]["unexpected"] = True
            torch.save(raw, path)

            with self.assertRaisesRegex(ValueError, "unexpected schema"):
                CheckpointStore().load(path)


if __name__ == "__main__":
    unittest.main()
