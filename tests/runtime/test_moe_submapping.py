from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from koemi.runtime import moe_submapping
from koemi.runtime.moe_submapping import (
    BlockIntegrityError,
    ManifestError,
    ManifestIntegrityError,
    MoeSubmappingStore,
    UnknownExpertError,
    UnsafePathError,
    publish_moe_layout,
)


def make_expert_tensors(
    expert_count: int = 3,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[tuple[int, int, str], torch.Tensor]:
    tensors: dict[tuple[int, int, str], torch.Tensor] = {}
    for expert in range(expert_count):
        tensors[(0, expert, "scale")] = torch.tensor([expert + 1.0], dtype=dtype)
        tensors[(0, expert, "bias")] = torch.tensor([expert * 10.0], dtype=dtype)
    return tensors


def publish_test_layout(directory: Path, *, expert_count: int = 3) -> Path:
    layout = directory / "layout"
    publish_moe_layout(
        layout,
        make_expert_tensors(expert_count),
        layer_count=1,
        experts_per_layer=expert_count,
        block_elements=1,
        model_name="test-moe",
    )
    return layout


def rewrite_manifest(layout: Path, mutate, *, update_digest: bool) -> None:
    manifest_path = layout / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_path.write_bytes(payload)
    if update_digest:
        digest = hashlib.sha256(payload).hexdigest()
        (layout / "manifest.json.sha256").write_text(digest + "\n", encoding="ascii")


def apply_expert(inputs: torch.Tensor, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return inputs * tensors["scale"] + tensors["bias"]


class MoeSubmappingPublicationTests(unittest.TestCase):
    def test_manifest_declares_moe_blocks_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            publication = publish_moe_layout(
                Path(temporary) / "layout",
                make_expert_tensors(),
                layer_count=1,
                experts_per_layer=3,
                block_elements=1,
            )
            manifest_bytes = publication.manifest_path.read_bytes()
            sidecar_digest = (publication.directory / "manifest.json.sha256").read_text(
                encoding="ascii"
            ).strip()
            manifest = json.loads(manifest_bytes)

            self.assertEqual(publication.manifest_sha256, hashlib.sha256(manifest_bytes).hexdigest())
            self.assertEqual(publication.manifest_sha256, sidecar_digest)
            self.assertEqual("moe", manifest["model_type"])
            self.assertTrue(manifest["is_moe"])
            self.assertEqual("content_dispatch_hash", manifest["routing"]["kind"])
            self.assertEqual("expert-blocks", manifest["layout"]["kind"])
            self.assertTrue(manifest["layout"]["immutable_blocks"])
            self.assertEqual(6, manifest["block_count"])
            self.assertEqual(3, publication.expert_count)

    def test_atomic_publish_cleans_staging_after_a_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            target = parent / "layout"
            with patch.object(moe_submapping.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    publish_moe_layout(
                        target,
                        make_expert_tensors(),
                        layer_count=1,
                        experts_per_layer=3,
                    )
            self.assertFalse(target.exists())
            self.assertEqual([], list(parent.glob(".layout.staging-*")))
            self.assertEqual([], list(parent.rglob("*.tmp")))


class MoeSubmappingExecutionTests(unittest.TestCase):
    def test_tensor_dtype_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = Path(temporary) / "layout"
            source = make_expert_tensors(dtype=torch.float64)
            publish_moe_layout(
                layout,
                source,
                layer_count=1,
                experts_per_layer=3,
                block_elements=1,
            )
            restored = MoeSubmappingStore.open(layout).read_experts([(0, 2)])

            self.assertEqual(torch.float64, restored[next(iter(restored))]["scale"].dtype)
            torch.testing.assert_close(restored[next(iter(restored))]["scale"], source[(0, 2, "scale")])

    def test_selected_expert_combination_matches_the_in_memory_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publish_test_layout(Path(temporary))
            store = MoeSubmappingStore.open(layout, cache_bytes=128)
            inputs = torch.tensor([[2.0], [3.0]], dtype=torch.float32)

            measured = store.execute_moe(
                0,
                [0, 2],
                inputs,
                apply_expert,
                gate_weights=[0.25, 0.75],
            )
            source_tensors = make_expert_tensors()
            reference = 0.25 * apply_expert(
                inputs,
                {
                    "scale": source_tensors[(0, 0, "scale")],
                    "bias": source_tensors[(0, 0, "bias")],
                },
            ) + 0.75 * apply_expert(
                inputs,
                {
                    "scale": source_tensors[(0, 2, "scale")],
                    "bias": source_tensors[(0, 2, "bias")],
                },
            )

            torch.testing.assert_close(measured, reference)
            self.assertGreater(store.metrics.bytes_read, 0)
            self.assertEqual(2, store.last_metrics.unique)

    def test_unselected_expert_is_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publish_test_layout(Path(temporary))
            store = MoeSubmappingStore.open(layout, cache_bytes=0)
            manifest = store.manifest
            unselected = next(
                expert
                for expert in manifest["catalog"]
                if expert["expert"] == 1
            )
            unselected_path = layout / Path(
                unselected["tensors"][0]["blocks"][0]["path"]
            )
            unselected_path.write_bytes(unselected_path.read_bytes() + b"corruption")

            result = store.execute_moe(0, [0, 2], torch.ones(1, 1), apply_expert)

            self.assertEqual((1, 1), tuple(result.shape))
            self.assertEqual(2, store.last_metrics.unique)

    def test_lru_cache_hits_and_eviction_are_bounded_by_tensor_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publish_test_layout(Path(temporary))
            store = MoeSubmappingStore.open(layout, cache_bytes=8)

            store.read_experts([(0, 0)])
            first = store.last_metrics
            store.read_experts([(0, 0)])
            hit = store.last_metrics
            store.read_experts([(0, 1)])
            eviction = store.last_metrics
            store.read_experts([(0, 0)])
            miss_after_eviction = store.last_metrics

            self.assertEqual(0, first.cache_hits)
            self.assertEqual(1, hit.cache_hits)
            self.assertEqual(1, eviction.evictions)
            self.assertEqual(0, miss_after_eviction.cache_hits)
            self.assertLessEqual(store.resident_bytes, 8)

    def test_duplicate_top_k_ids_are_read_and_executed_once_but_keep_weighting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publish_test_layout(Path(temporary))
            store = MoeSubmappingStore.open(layout, cache_bytes=0)
            calls: list[float] = []

            def executor(inputs: torch.Tensor, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
                calls.append(float(tensors["bias"].item()))
                return apply_expert(inputs, tensors)

            inputs = torch.ones(2, 1)
            measured = store.execute_moe(0, [0, 0, 1], inputs, executor)
            expected = (
                apply_expert(inputs, {"scale": torch.tensor([1.0]), "bias": torch.tensor([0.0])})
                + apply_expert(inputs, {"scale": torch.tensor([1.0]), "bias": torch.tensor([0.0])})
                + apply_expert(inputs, {"scale": torch.tensor([2.0]), "bias": torch.tensor([10.0])})
            ) / 3.0

            torch.testing.assert_close(measured, expected)
            self.assertEqual([0.0, 10.0], calls)
            self.assertEqual(3, store.last_metrics.requested)
            self.assertEqual(2, store.last_metrics.unique)
            self.assertEqual(0, store.last_metrics.cache_hits)

    def test_out_of_catalog_expert_is_refused_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = MoeSubmappingStore.open(publish_test_layout(Path(temporary)))
            with self.assertRaises(UnknownExpertError):
                store.execute_moe(0, [3], torch.ones(1, 1), apply_expert)


class MoeSubmappingIntegrityTests(unittest.TestCase):
    def test_selected_block_digest_corruption_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publish_test_layout(Path(temporary))
            store = MoeSubmappingStore.open(layout)
            block_path = layout / Path(
                store.manifest["catalog"][0]["tensors"][0]["blocks"][0]["path"]
            )
            block_path.write_bytes(block_path.read_bytes() + b"corruption")

            with self.assertRaises(BlockIntegrityError):
                store.read_experts([(0, 0)])

    def test_manifest_digest_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publish_test_layout(Path(temporary))
            rewrite_manifest(layout, lambda manifest: manifest.update(model_name="changed"), update_digest=False)

            with self.assertRaises(ManifestIntegrityError):
                MoeSubmappingStore.open(layout)

    def test_dense_manifest_is_refused_even_with_a_valid_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publish_test_layout(Path(temporary))
            rewrite_manifest(layout, lambda manifest: manifest.update(is_moe=False), update_digest=True)

            with self.assertRaisesRegex(ManifestError, "only MoE"):
                MoeSubmappingStore.open(layout)

    def test_manifest_traversal_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publish_test_layout(Path(temporary))

            def add_traversal(manifest: dict[str, object]) -> None:
                manifest["catalog"][0]["tensors"][0]["blocks"][0]["path"] = "../outside.pt"

            rewrite_manifest(layout, add_traversal, update_digest=True)

            with self.assertRaises(UnsafePathError):
                MoeSubmappingStore.open(layout)

    def test_unsupported_routing_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publish_test_layout(Path(temporary))
            rewrite_manifest(
                layout,
                lambda manifest: manifest["routing"].update(kind="dense_gate"),
                update_digest=True,
            )

            with self.assertRaisesRegex(ManifestError, "content_dispatch_hash"):
                MoeSubmappingStore.open(layout)


class MoeSubmappingMetricsAndPrefetchTests(unittest.TestCase):
    def test_metrics_expose_requested_unique_cache_bytes_and_read_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = MoeSubmappingStore.open(publish_test_layout(Path(temporary)), cache_bytes=128)
            store.read_experts([(0, 0), (0, 0), (0, 1)])
            metrics = store.last_metrics

            self.assertEqual(3, metrics.requested)
            self.assertEqual(2, metrics.unique)
            self.assertEqual(0, metrics.cache_hits)
            self.assertGreater(metrics.bytes_read, 0)
            self.assertGreaterEqual(metrics.read_seconds, 0.0)
            self.assertEqual(metrics.to_dict(), store.metrics.to_dict())

    def test_prefetch_is_bounded_and_output_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = publish_test_layout(Path(temporary))
            baseline_store = MoeSubmappingStore.open(layout, cache_bytes=128)
            prefetched_store = MoeSubmappingStore.open(
                layout,
                cache_bytes=128,
                prefetch_limit=2,
            )
            inputs = torch.tensor([[2.0], [4.0]])
            baseline = baseline_store.execute_moe(0, [0, 1], inputs, apply_expert)
            prefetch_metrics = prefetched_store.prefetch(0, [0, 1])
            measured = prefetched_store.execute_moe(0, [0, 1], inputs, apply_expert)

            torch.testing.assert_close(measured, baseline)
            self.assertEqual(2, prefetch_metrics.unique)
            self.assertEqual(2, prefetched_store.last_metrics.cache_hits)
            self.assertEqual(0, prefetched_store.last_metrics.bytes_read)
            with self.assertRaisesRegex(ValueError, "limited"):
                prefetched_store.prefetch(0, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
