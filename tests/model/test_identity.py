from __future__ import annotations

import unittest

from koemi.configuration.settings import ModelSettings
from koemi.model.identity import (
    ModelIdentity,
    ModelLicensing,
    attach_model_identity,
    get_model_identity,
)
from koemi.model.network import KoemiModel


def open_source_only_licensing() -> ModelLicensing:
    return ModelLicensing(
        source_license="MIT",
        source_availability="open",
        weights_availability="closed",
        data_availability="unknown",
    )


class ModelIdentityTests(unittest.TestCase):
    def test_custom_header_is_metadata_not_prompt_text(self) -> None:
        identity = ModelIdentity(
            organization="Acme AI",
            model_name="Aurora",
            version="1.2.0",
            architecture="HERM",
            heading="Aurora speaks for Acme",
            licensing=open_source_only_licensing(),
        )

        header = identity.render_header()

        self.assertIn("heading=Aurora speaks for Acme", header)
        self.assertIn("architecture=HERM", header)
        self.assertIn("source=open:MIT", header)
        self.assertIn("weights=closed:", header)
        self.assertNotIn("<|system|>", header)

    def test_default_heading_and_model_id_are_deterministic(self) -> None:
        first = ModelIdentity("Acme AI", "Aurora", "1.2.0", licensing=open_source_only_licensing())
        second = ModelIdentity("Acme AI", "Aurora", "1.2.0", licensing=open_source_only_licensing())

        self.assertEqual("Acme AI Aurora (1.2.0)", first.heading)
        self.assertEqual("acme-ai/aurora:1-2-0", first.model_id)
        self.assertEqual(first.identity_digest, second.identity_digest)

    def test_payload_round_trip_preserves_license_policy(self) -> None:
        identity = ModelIdentity(
            organization="Koemi Labs",
            model_name="Koemi HERM",
            version="0.4.0",
            variant="moe-submapping",
            description="A HERM model artifact.",
            licensing=ModelLicensing(
                source_license="MIT",
                weights_license="Apache-2.0",
                data_license="CC-BY-4.0",
                source_availability="open",
                weights_availability="open",
                data_availability="restricted",
                license_url="https://example.invalid/licenses/koemi",
                attribution="Koemi contributors",
            ),
        )

        restored = ModelIdentity.from_payload(identity.to_payload())

        self.assertEqual(identity, restored)
        self.assertEqual(identity.model_id, restored.model_id)

    def test_open_artifact_requires_license(self) -> None:
        with self.assertRaisesRegex(ValueError, "open source availability requires a license"):
            ModelLicensing(source_availability="open")

    def test_header_fields_reject_control_characters(self) -> None:
        with self.assertRaisesRegex(ValueError, "control character"):
            ModelIdentity("Acme\nInjected", "Aurora", "1.0.0")

    def test_payload_schema_is_strict(self) -> None:
        identity = ModelIdentity("Acme AI", "Aurora", "1.0.0")
        payload = identity.to_payload()
        payload["unexpected"] = True

        with self.assertRaisesRegex(ValueError, "unexpected schema"):
            ModelIdentity.from_payload(payload)

    def test_model_binding_does_not_add_identity_to_state_dict(self) -> None:
        model = KoemiModel(ModelSettings(embedding_size=8, memory_features=2, local_memory_size=1, salience_memory_size=1))
        identity = ModelIdentity("Acme AI", "Aurora", "1.0.0")

        self.assertIs(attach_model_identity(model, identity), model)
        self.assertEqual(identity, get_model_identity(model))
        self.assertNotIn("model_identity", model.state_dict())


if __name__ == "__main__":
    unittest.main()
