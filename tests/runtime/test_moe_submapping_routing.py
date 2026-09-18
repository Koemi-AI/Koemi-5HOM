from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import torch

from koemi.model.identity import ModelIdentity
from koemi.runtime.moe_submapping import (
    ManifestError,
    publish_moe_layout,
    route_moe_experts,
    MoeSubmappingStore,
)


def _tensors() -> dict[tuple[int, int, str], torch.Tensor]:
    return {
        (0, 0, "weight"): torch.tensor([1.0]),
        (0, 1, "weight"): torch.tensor([2.0]),
        (0, 2, "weight"): torch.tensor([3.0]),
    }


class MoeSubmappingRoutingTests(unittest.TestCase):
    def test_route_preserves_valid_rows_and_masks_padding(self) -> None:
        assignments = route_moe_experts(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[True, False, True]]),
            expert_count=4,
            top_k=2,
        )

        self.assertEqual((1, 3, 2), tuple(assignments.shape))
        self.assertTrue(torch.equal(assignments[0, 1], torch.tensor([-1, -1])))
        self.assertEqual(2, assignments[0, 0].unique().numel())
        self.assertEqual(2, assignments[0, 2].unique().numel())

    def test_duplicate_top_k_is_rejected_for_valid_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate experts"):
            route_moe_experts(
                torch.tensor([[1]]),
                torch.tensor([[0]]),
                torch.ones((1, 1), dtype=torch.bool),
                expert_count=3,
                top_k=3,
            )

    def test_duplicate_padding_is_not_a_valid_route(self) -> None:
        assignments = route_moe_experts(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[False, False, False]]),
            expert_count=3,
            top_k=3,
        )
        self.assertTrue(torch.equal(assignments, torch.full((1, 3, 3), -1, dtype=torch.long)))

    def test_non_moe_and_shape_contracts_are_rejected(self) -> None:
        inputs = torch.tensor([[1, 2]])
        previous = torch.tensor([[0, 1]])
        valid = torch.ones((1, 2), dtype=torch.bool)
        with self.assertRaises(ValueError):
            route_moe_experts(inputs, previous, valid, expert_count=0, top_k=1)
        with self.assertRaises(ValueError):
            route_moe_experts(inputs, previous[:, :1], valid, expert_count=4, top_k=1)
        with self.assertRaises(ValueError):
            route_moe_experts(inputs, previous, torch.ones((1, 2)), expert_count=4, top_k=1)

    def test_identity_is_carried_by_the_moe_manifest(self) -> None:
        identity = ModelIdentity(
            organization="Koemi Labs",
            model_name="HERM MoE",
            version="0.1.0",
            heading="Koemi HERM MoE",
        )
        with tempfile.TemporaryDirectory() as temporary:
            layout = Path(temporary) / "layout"
            publish_moe_layout(
                layout,
                _tensors(),
                layer_count=1,
                experts_per_layer=3,
                identity=identity,
            )
            store = MoeSubmappingStore.open(layout)
            self.assertEqual(identity.to_payload(), store.manifest["model_identity"])

            manifest = store.manifest
            manifest["model_identity"]["model_name"] = ""
            manifest_path = layout / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            (layout / "manifest.json.sha256").write_text(
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                encoding="ascii",
            )
            with self.assertRaises(ManifestError):
                MoeSubmappingStore.open(layout)

    def test_negative_catalog_indices_are_rejected_as_manifest_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = Path(temporary) / "layout"
            publish_moe_layout(
                layout,
                _tensors(),
                layer_count=1,
                experts_per_layer=3,
            )
            manifest_path = layout / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["catalog"][0]["layer"] = -1
            manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            manifest_path.write_bytes(manifest_bytes)
            (layout / "manifest.json.sha256").write_text(
                hashlib.sha256(manifest_bytes).hexdigest(),
                encoding="ascii",
            )

            with self.assertRaises(ManifestError):
                MoeSubmappingStore.open(layout)


if __name__ == "__main__":
    unittest.main()
