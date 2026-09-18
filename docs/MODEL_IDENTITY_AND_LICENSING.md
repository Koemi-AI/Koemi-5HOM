# Model identity and licensing

Koemi stores a model's public identity as artifact metadata. The identity can
include a custom heading, organization, model name, version, HERM architecture,
variant, description and a separate policy for source, weights and data.

The metadata is not a system prompt. It does not change the token stream, the
loss, or the `forward` pass. It gives loaders, registries and tools a stable
answer to “which model is this?” A model can only verbalize that identity
reliably when the training data and evaluation measure that behavior.

```python
from koemi.model.identity import ModelIdentity, ModelLicensing

identity = ModelIdentity(
    organization="Koemi Labs",
    model_name="Koemi HERM",
    version="0.4.0",
    heading="Koemi HERM — a model by Koemi Labs",
    licensing=ModelLicensing(
        source_license="MIT",
        source_availability="open",
        weights_availability="closed",
        data_availability="unknown",
    ),
)

print(identity.model_id)
print(identity.render_header())
```

`source_availability="open"` with `weights_availability="closed"` describes a
source-only release. An open artifact must name its license. The metadata does
not grant rights that the owner has not granted; source, weights, datasets and
trademarks remain separate legal objects and require legal review for a public
release.

The identity is now carried by both checkpoint paths. `CheckpointStore` keeps
it in the legacy payload for backward-compatible model export, while
`CheckpointCatalog` keeps it in the immutable manifest beside the hashed,
weights-only payload. The MoE Submapping manifest can carry the same payload
without putting identity text into the token stream.

The normal A100/training runner is not migrated to `CheckpointCatalog` in this
phase. Existing checkpoints without identity remain loadable, and the catalog
is opt-in until a runner adapter has its own migration and regression tests.

## Verification boundary

The identity schema, header rendering, licensing validation, payload round trip
and model binding are covered by `tests/model/test_identity.py`. Checkpoint
manifest transport is covered by `tests/training/test_checkpoint_identity.py`
and `tests/training/test_checkpoint_catalog_identity.py`; MoE manifest transport
is covered by `tests/runtime/test_moe_submapping_routing.py`. These tests do not
claim that generated text will self-identify; that requires a trained-model
behavior test.
