# Koemi-4HCM release dossier

Koemi-4HCM means **Koemi-4 HERM Consolidation Model**. It is the consolidation
release of the HERM model, runtime and training infrastructure that started as
Koemi-3HIP (Koemi-3 HERM Initial Phase).

This document is the release record for the repository state on 2026-09-18. It
separates shipped code, hardware measurements and work that is still opt-in or
unproven.

## Release position

| Item | Koemi-4HCM state |
| --- | --- |
| Python package | `0.4.0` |
| Source license | MIT, as declared by `LICENSE` |
| Weights | No weights are shipped in this repository |
| Dataset rights | Separate from source and weights; each corpus revision needs its own review |
| Default runner | Compatibility path remains unchanged |
| Native CUDA | Four opt-in operator slices; compiled and exercised on an A100 |
| MoE storage | Isolated MoE-only Submapping slice; not the normal runner |
| Public model identity | Artifact metadata with custom heading and licensing fields |

The name change describes consolidation, not a claim that every experimental
path has become the default model execution path. A legacy 3HIP checkpoint can
still be loaded where its existing format and settings are supported.

## What changed from 3HIP to 4HCM

| Area | Koemi-3HIP | Koemi-4HCM |
| --- | --- | --- |
| HERM core | Initial bounded recurrent and associative-memory implementation | Same causal model family, with the runtime, memory, caching and training boundaries documented as one release |
| Model identity | No stable release identity contract | `ModelIdentity` carries organization, model name, version, custom heading, architecture, variant and description |
| Licensing metadata | Not carried with every artifact path | Source, weights and data availability are declared separately in checkpoint and catalog metadata |
| Checkpoints | Legacy checkpoint store and rotating training checkpoints | Immutable `CheckpointCatalog` with payload hashes, `weights_only=True` recovery and corruption fallback; legacy paths remain supported |
| Training | Causal byte training, thinking masks and deterministic experts | Length-aware batches, explicit A100 safe/aggressive profiles, rotating recovery, separate answer/thinking metrics and measured expert occupancy |
| Inference | Recurrent generation and exact prefix reuse | Prefill/decode batching, fast decode, speculative paths, exact RAM/SSD block reuse and parameter offload tiers, all at explicit boundaries |
| Tokenization | Byte-first path | Hybrid byte/BPE vocabulary with checkpoint migration and span-marker protection |
| MoE | Deterministic content-hash dispatch in the model path | Strict isolated MoE Submapping storage fork plus native MoE route/permute/group/combine kernels; no change to the normal runner |
| CUDA | PyTorch CUDA seams without a measured native-kernel record | Native `.cu` slices for core, dense, think and MoE, with A100 compilation and contract evidence |
| Operations | Notebook and local runner paths | Bounded Colab MCP retryer for CPU-to-A100 runtime transitions, with explicit retry and remote-failure markers |

## HERM architecture in 4HCM

HERM remains a causal model, not a Transformer replacement claim. The execution
chain is:

```text
byte or hybrid token
    -> embedding and RMSNorm
    -> bounded recurrent state h_t
    -> linear causal preview and surprise s_t
    -> fast associative memory
    -> optional slow residual/refine memory
    -> exact local ring + exact salient ring
    -> typed fusion
    -> deterministic content-hash experts, when enabled
    -> byte or hybrid vocabulary head
```

The recurrent state is bounded by a token-local affine update. The associative
memory keeps fast and slow rank-one factors with normalizers. The read is causal:
the state before the current token is used for its prediction and write. Exact
local and salient rings provide bounded retrieval without pretending that a
similar prompt is an exact cache hit.

Surprise is a normalized negative-log-likelihood signal. It scales memory-write
influence; it does not choose a hidden execution branch. Visible `thinking`
spans have their own supervised mask and loss weight. They are training data,
not a claim that generated text is a private reasoning trace.

When enabled, the MoE route is deterministic and content-based. The current
training configuration uses 128 experts with six active experts per valid byte.
It has no learned router, router loss or semantic routing claim. The separate
Submapping fork stores selected expert tensors as validated blocks and refuses
non-MoE, unsafe, corrupt or duplicate-route artifacts.

## Native CUDA slices

The code lives under `src/koemi/cuda_kernels/` and is covered by conditional
tests under `tests/cuda/`. The normal `KoemiModel` runner was not modified to
silently select these extensions.

| Slice | Work | A100 evidence |
| --- | --- | --- |
| `core` | Fused RMSNorm + SiLU forward/backward | `0.01280 ms` native vs `0.09078 ms` reference, ratio `0.1410` |
| `dense` | Affine scan forward/backward | `0.06277 ms` native vs `8.47729 ms` sequential reference, ratio `0.0074` |
| `think` | Causal surprise forward/backward | Forward `0.04321 ms` vs `0.37970 ms`, ratio `0.1138`; backward `0.09400 ms` vs `0.56842 ms`, ratio `0.1654` |
| `moe` | Content route, stable permutation, grouped MLP contract and combine | Compiled and contract-checked; no end-to-end MoE throughput claim and no backward path |

The remote run reported an `NVIDIA A100-SXM4-40GB`, compute capability `8.0`,
PyTorch `2.11.0+cu128` and CUDA `12.8`. The think path now uses CUDA GEMM
through ATen plus custom warp-shuffle reductions and a coalesced logits-gradient
kernel. This is a measured operator improvement, not proof of the same gain for
the whole model. It also materializes a float logits matrix and therefore is not
a standalone fully fused custom GEMM.

## Training evidence in the supplied log

Source: `C:\Users\Brenno\Documents\LOGS IMPORTANTES TREINANDO DA IA.txt`.
The log contains snapshots from the aggressive A100 profile in
`koemi-a100-aggressive-v1`. The profile is documented as approximately
1.035B parameters, width 1,152, sequence length 512, 128 experts, top-6
dispatch, 200,000 bounded records and 7.5-hour resumable sessions within a
188-hour budget.

The final snapshots in the supplied log are optimizer steps `18,700` through
`18,800`. They show movement rather than a monotonic curve:

| Metric | Observed range in steps 18,700–18,800 | Last snapshot |
| --- | ---: | ---: |
| Tokens seen | `456,175,358`–`458,615,359` | `458,615,359` |
| Loss (nats) | `0.5128`–`0.5844` | `0.5844` |
| Perplexity | `1.6700`–`1.7939` | `1.7939` |
| Answer BPB | `0.6136`–`0.6580` | `0.6136` |
| Thinking BPB | `1.0821`–`1.1901` | `1.1901` |
| Supervised tokens/s | `23,484`–`30,211` | `30,211` |
| Peak GPU allocation | `33,727,177,216` bytes | `33,727,177,216` bytes |

The 128-expert bank was occupied in every snapshot. Normalized routing entropy
was `0.9810`–`0.9856`, and load Gini was `0.2068`–`0.2371`. Those values are
evidence of measured routing distribution, not evidence that the experts have
learned semantic specializations.

The log also contains the export procedure that loads the newest rotating
checkpoint and writes an inference checkpoint named from the optimizer step.
The existence and integrity of the exported file are not independently verified
in this repository, so the release does not publish it as a shipped weight.

## Identity, headers and licensing

4HCM artifacts can carry a stable identity without putting a system prompt into
the model input:

```python
from koemi.model.identity import ModelIdentity, ModelLicensing

identity = ModelIdentity(
    organization="Koemi Labs",
    model_name="Koemi-4HCM",
    version="0.4.0",
    heading="Koemi-4HCM by Koemi Labs",
    architecture="HERM",
    licensing=ModelLicensing(
        source_license="MIT",
        source_availability="open",
        weights_availability="closed",
        data_availability="unknown",
    ),
)
```

The identity is stored beside model payloads and can be transported through the
legacy checkpoint store, immutable catalog and MoE manifest. It does not alter
the token stream, train the model to say its name or grant rights over weights,
datasets or trademarks. Self-identification requires training examples and a
behavior evaluation. A public release must choose the actual weights and data
policy instead of copying the example above.

## Verification boundary

Verified in this release preparation:

- the full local suite: `548` tests passed with `16` conditional CUDA skips;
- local Python compilation and `git diff --check`;
- all four native extensions compiled on the A100 and emitted
  `KOEMI_COMPILED`, `KOEMI_PASS` and `KOEMI_REMOTE_DONE` markers;
- think forward and backward equivalence against explicit references on the
  A100 harness;
- Colab MCP reconnection after a runtime transition through the retrying
  launcher.

Not a 4HCM claim yet:

- end-to-end model throughput or quality improvement from the native kernels;
- native CUDA integration into the default runner;
- production NVMe prefetch, cube/mmap storage or GPU/SSD overlap for MoE;
- learned semantic routing, distributed training or Transformer-level quality;
- an open-weights or source-only legal release of a trained model;
- a trained behavior test proving that the model identifies itself in generated
  text.

## Launch artifacts

The source release is prepared in the local repository. The following remain
deliberate release decisions rather than implicit defaults:

1. publish source under MIT;
2. choose whether a specific checkpoint's weights are closed, shared or open;
3. document the exact training-data revisions and their licenses;
4. publish only measured A100 operator numbers with their tested shapes;
5. keep the normal runner and legacy 3HIP artifact names available for
   compatibility until a migration is explicitly announced.

No tag, remote push, weight upload or dataset redistribution is performed by
this preparation commit.
