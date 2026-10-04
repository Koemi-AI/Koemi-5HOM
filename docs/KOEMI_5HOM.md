# Koemi-5HOM: HERM Optimized Model

## Implementation contract

Goal: evolve the Koemi source to version 0.5.0 with a trainable, causal MoE
router and a simpler supervised training path for typed decision models.
MDT means **Modelo de Decisão Tipada** (Typed Decision Model): decisions with
defined output types, including option choices, ordinal scores and yes probabilities.
An expert gate chooses computation; a typed decision head chooses an answer.
They have different losses and must not be treated as the same mechanism.

Scope: optional learned top-k routing after HERM fusion, probability-weighted
dropless dispatch, whole-sequence balance statistics, router loss/occupancy
logs, CLI/checkpoint integration, and a standalone supervised decision trainer
compatible with Laya's five-input/two-output forward contract. The trainer
accepts encoded batches and supports encoder freezing, separate learning
rates, partial accumulation groups, ordinal loss and held-out temperature
fitting. It does not download models or execute remote Python.

Acceptance criteria, verified by execution:

- Hash checkpoints without the new settings load with identical logits.
- Learned routing assigns distinct experts to each valid top-k slot, ignores
  padding and sends a task gradient into the gate even at top-k one.
- Loop and sorted dispatch agree on outputs and parameter/input gradients.
- Parallel, sequential, chunked and activation-checkpointed HERM execution
  agree on logits and router balance loss within floating-point tolerance.
- A short CLI train saves/loads the learned gate and generates a continuation.
- A tiny offline decision model with Laya's architecture/forward contract
  learns a soft target; frozen encoders stay fixed; uneven accumulation gives
  the same update as the corresponding full batch.
- Temperature fitting uses a separate calibration batch, does not increase
  its NLL, and cannot be described as a generalization result.
- The complete local test suite, compileall and diff whitespace check pass.

Assumptions: hash remains the compatibility default; learned routing is opt-in;
local execution is CPU; labeled decisions or teacher distributions are
available to the caller. Supervised cross-entropy is a proposed alternative
to noisy policy-gradient training when those distributions are known, not a
reimplementation of TypeSafe's proprietary RLCD.

Out of scope: claiming superior quality, training pretrained Laya weights in
this local session, replacing HERM with ModernBERT, running paid GPU training,
FP8 quantization, integrating DeepGEMM CUDA kernels or distributing weights.

## Reference study

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) separates dense/grouped
  GEMM execution from route selection. Its contiguous MoE layout groups token
  rows by expert; masked layouts address decode with CUDA graphs. Current
  requirements name SM90/SM100 and CUDA 12.9+. The previously measured Koemi
  A100 is SM80, so installing DeepGEMM is not an A100 acceleration plan.
- [Jev documentation](https://docs.typesafe.ai/introduction) defines typed
  questions and probability outputs. Public API documentation does not expose
  enough of the proprietary architecture/training to reproduce Jev.
- [Laya source](https://github.com/NandhaKishorM/laya/tree/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2)
  was inspected at revision `8a6e1328cce2460a0e5aa348ad465bb1b5821cd2`.
  `laya/common.py` defines a bidirectional encoder, transformer decision head,
  option-marker scorer and separate act head. `laya/train.py` already offers
  soft cross-entropy as an alternative to RLCD. Therefore, simply adding
  supervised learning is not a novel research claim. Koemi's contribution is
  the independent encoded-batch trainer and its explicit contracts/tests.

Laya's published model card reports option-budget collapse, overconfidence
and an act-head signal that can be uninformative. These are reasons to retain
option masks, validate targets, calibrate on held-out states, and avoid
presenting softmax confidence as measured correctness.

## Routing design

The gate reads only the fused context at the current causal position. Its
softmax is evaluated in FP32. Top-k indices are unique by construction;
selected probabilities retain their dense-softmax mass instead of being
renormalized, so the task loss trains the gate at top-k one too. The residual
expert update is a weighted sum. No token is dropped for capacity reasons.

Balance uses additive probability mass, assignment counts and valid-token
count, merged over the entire forward before forming the auxiliary term.
Changing scan chunk size cannot redefine the balance objective. Assignment
counts are detached; probability mass retains its gate gradient.

Sorted dispatch groups rows by expert, inspired by DeepGEMM's contiguous
layout. It uses ordinary PyTorch autograd and one MLP call per nonempty expert;
it is not a native grouped GEMM. Its count transfer synchronizes CUDA during
training. Inference retains the existing static tensor dispatch and hook-aware
reference fallback. No speedup is claimed without a matched benchmark.

## Usage and verification

For a step-by-step introduction to supervised typed decisions, see
[Introdução ao treinamento de MDT](MDT_TRAINING.md).

Train a learned HERM gate while preserving hash as the default:

```bash
python -m koemi train --dataset examples/canonical.jsonl \
  --checkpoint artifacts/koemi-5hom.pt --expert-count 8 --expert-top-k 2 \
  --expert-routing learned --expert-dispatch segments
```

The local CPU smoke used width 16, four experts/top-2, scan chunk 8, two
epochs and a record-level validation holdout. It completed four optimizer
steps, saved/reloaded the gate and generated a continuation. Validation NLL
was 5.719637 then 5.699242. This is a pipeline smoke with a tiny corpus, not
a quality comparison or a throughput benchmark.

To specialize a local Laya checkpoint without the policy-gradient term:

```bash
python -m pip install -e ".[decisions]"
python -m koemi.training.laya_decisions \
  --model-directory /path/to/local/laya-checkpoint \
  --dataset examples/typed_decisions.jsonl \
  --output artifacts/laya-supervised --freeze-encoder
```

Each JSONL row contains `state`, `questions`, and `gold`. Every question must
have a gold probability for each choice key, score level index (`"0"`, `"1"`,
...) or noul label (`"false"`, `"true"`). Targets must be nonnegative, finite,
sum to one and assign no probability to padding. The encoder/transformer
decision head scores option markers; it does not generate answer strings.

The adapter preserves Laya's checkpoint layout with safetensors, encoder
configuration and tokenizer. It fits per-type temperatures inside the
installed Laya runtime's bounds (0.5 to 5 in pinned 0.3.27), removes inherited
option-bucket temperatures and reports calibration NLL before/after. The
generic MDT fitter supports configurable bounds; its default is 0.1 to 10.
No calibration result is described as an independent test-set measurement.
The act/escalate head is preserved and explicitly reported as untrained.

The source study found newer `laya/train.py` on the inspected GitHub revision;
the pinned PyPI 0.3.27 wheel does not contain that module. It also omits
`laya.backends`, so `laya.load(..., backend="eager")` fails in that wheel.
The tested loading form is `laya.load(local_directory, device="cpu")` using
its default backend. Koemi's training adapter uses local model construction
and does not call the missing backend API.

Verification commands:

```bash
python -m unittest discover -s tests -p "test*.py"
python -m compileall -q src benchmarks tests
python -m pip check
git diff --check
```

Optional integration tests construct the real Laya `DecisionModel` around a
tiny encoder, train it offline, then exercise the complete command with a
tiny bidirectional BERT/tokenizer fixture. The exported artifact is reloaded
by the public Laya loader and used to predict a typed answer. No pretrained
Laya weights, remote model code, or paid GPU session are used by these tests.

On 2026-10-04, the complete suite passed 573 tests in 152.084 seconds, with
16 CUDA skips on the CPU host. All 25 new tests ran, including optional Laya
integration. Compileall, package dependency validation and the whitespace
check also passed. These checks establish local behavior and compatibility;
they do not establish improved model quality or GPU speed.
