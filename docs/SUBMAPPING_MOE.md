# MoE Submapping storage

`koemi.runtime.moe_submapping` is an isolated, MoE-only storage component. It
does not replace or modify the normal Koemi runner. The first slice is a
CPU-testable artifact format that makes expert residency explicit before a
GPU/NVMe runner is introduced.

## Contract

`publish_moe_layout()` accepts a complete catalog of expert tensors and writes:

- an immutable JSON manifest;
- a SHA-256 sidecar for that manifest;
- per-expert, per-tensor blocks under `blocks/layer-XXXXXX/expert-XXXXXX/`;
- block sizes, dtypes, shapes and SHA-256 digests.

`MoeSubmappingStore.open()` rejects non-MoE manifests, unsupported routing,
unsafe paths, missing blocks and digest mismatches. Reads use
`torch.load(..., weights_only=True)` and are admitted to a byte-bounded LRU
cache only after the tensor contract is checked.

The routing helper mirrors HERM's current `content_dispatch_hash` route. The
fork rejects duplicate valid top-k assignments instead of silently reducing
`k`; padding is never routed. This is deliberately stricter than the legacy
dispatch path, which remains unchanged.

```python
from koemi.runtime.moe_submapping import MoeSubmappingStore

store = MoeSubmappingStore.open("artifacts/moe-layout", cache_bytes=64 * 1024**2)
result = store.execute_moe(
    0,
    [2, 7],
    inputs,
    lambda values, expert: values @ expert["weight"],
)
print(store.last_metrics.to_dict())
```

## Boundary and measurements

The format is a submapping/block vertical slice, not a production cube/mmap
engine. It has no direct `KoemiModel` integration, no asynchronous I/O, no VRAM
staging, and no measured NVMe or throughput result. The current tests prove
integrity, bounded cache behavior, selected-expert reads, deterministic output
and identity transport on the local CPU only.

The intended next runner must compare resident versus cold-cache versus warm-
cache inference, keep routing and dtype identical, measure logical and physical
bytes separately, and fail closed on a missing or corrupt expert. Colibri's
hierarchical VRAM/RAM/NVMe design is an architectural reference, not evidence
of Koemi performance.
