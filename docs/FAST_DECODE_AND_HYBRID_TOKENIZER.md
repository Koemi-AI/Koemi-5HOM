# Fast decode and the hybrid tokenizer

Two subsystems, one problem: a byte-level recurrent model emits one token per
forward, and a byte is a small unit of text. The first subsystem makes each
forward cheaper and lets one forward commit several tokens. The second makes a
token carry more text.

They are independent. Fast decode works on the current byte checkpoint without
retraining. The hybrid tokenizer changes the vocabulary, so it needs either a new
run or the migration described at the end.

## Where the time goes in HERM decode

HERM is recurrent. Its decode step is already constant work per token: there is
no growing attention matrix, and the state is bounded. So the per-token cost at
batch size one is dominated by fixed overhead, not by arithmetic:

- Six host synchronizations per token. `validate_input_ids` reads `min()` and
  `max()`, `forward_window` reads `valid_mask.sum()`, `generate_text` reads the
  sampled id with `.item()`, and the prefill/decode request validators read two
  more reductions.
- One stack of every expert weight per forward. `_apply_batched_experts` called
  `torch.stack` over the whole expert bank on each step, so a 128-expert model
  copied 128 expert weight matrices to select six rows.
- Ring buffers that change shape while they fill, which blocks CUDA graph
  capture and forces a fresh allocation per step.

Each of those is fixed cost per token. None of them shrinks when the model is
small, and all of them are paid 128 times to write 128 bytes.

## `koemi.runtime.fast_decode`

`BatchDecoder` removes the three costs above.

**Trusted inputs.** `KoemiModel.forward` accepts `trusted_inputs=True`. The
caller promises the batch carries real token ids and no padding; the model then
skips `validate_input_ids` and takes the token count from the tensor size instead
of reducing the mask. The promise is checked once, on the prompt, by
`BatchDecoder.prefill`.

**Static rings.** `to_static_state` pads `local_*` and `salient_*` to their
capacity with zeroed, invalid slots. An invalid slot contributes nothing to any
read, so the values match the growing state, and `tail`/`salient_tail` preserve
the capacity from then on. `tests/runtime/test_fast_decode.py` asserts both: the
logits agree within `1e-5`, and every state field keeps one shape across steps.

**CUDA graph capture.** With static shapes, no host synchronization and no
data-dependent control flow, one decode step is capturable.
`BatchDecoder(model, capture_graph=True)` warms up on a side stream, captures the
forward plus the copies back into the state buffers, and replays the graph once
per step. Sampling stays outside the graph. The constructor fails on a CPU model
rather than silently running the eager path.

**Cached expert stacks.** `KoemiModel.cache_inference_weights()` stacks the
expert bank once. `train()` releases it, and caching is refused while gradients
are enabled, so a stale stack cannot reach a training step.

**Batched generation.** `generate_batch` right-aligns the prompts inside one
padded batch, samples on the device, writes each column into a preallocated
tensor and reads the ids back once at the end.

Verification on this CPU-only host, `tests/runtime/test_fast_decode.py`: a guard
replaces `Tensor.item`, `tolist`, `__int__`, `__float__` and `__bool__` with a
function that raises, and the decode loop runs four steps inside it without
tripping. The same guard trips immediately on the validating forward, which is
what makes the first assertion meaningful.

## Speculative decoding

`koemi.runtime.speculative` verifies whole blocks instead of single tokens.

One round from state `S` after `n` tokens, with `pending_logits` giving the
distribution for token `n + 1`:

1. The drafter proposes `d_1 … d_k` with the distribution each one came from.
2. One windowed forward over the block returns the target distribution for
   `d_2 … d_k` plus the bonus distribution after `d_k`. The distribution for
   `d_1` is `pending_logits`, already in hand.
3. `accept_draft_tokens` applies rejection sampling: token `i` is accepted with
   probability `min(1, q_i(d_i) / p_i(d_i))`; the first rejection is replaced by
   a draw from the normalized `max(0, q - p)`; a fully accepted block draws one
   bonus token from the last row.
4. A fully accepted block keeps the state the verification forward produced and
   advances one token over the bonus. A rejected block restores the saved state
   and replays only the tokens that survived.

So a block of `m` committed tokens costs two target forwards instead of `m`.
When acceptance collapses it costs two forwards for one token, which is worse
than the baseline. `SpeculativeStatistics` reports `acceptance_rate` and
`tokens_per_target_call` precisely so that case is visible rather than assumed.

Two drafters ship:

- `NgramDrafter` searches the context for the most recent earlier occurrence of
  the current suffix and proposes what followed it. The proposal is a point mass,
  which makes the acceptance probability exactly the target mass on that token.
  There is no second model and no extra parameter memory. It suits code, tables
  and quoted context, where the continuation has appeared before.
- `ModelDrafter` runs a smaller checkpoint with its own recurrent state. Each
  proposal is rolled back before the committed tokens are replayed, so the draft
  state follows what the target accepted.

Both feed the same acceptance core, so the committed tokens are distributed as if
they were drawn one at a time from the target distribution, up to the
floating-point difference between a windowed forward and a step-by-step forward.
That difference is real: `tests/model/test_execution.py` compares the two paths at
`atol=1e-4`, not at equality.

## Measured on CPU

Random weights, 1,041,555 parameters, 8 experts top-2, 48 generated tokens,
greedy, `torch 2.14.0+cpu`, no CUDA device:

| Path | tokens/s | Target forwards | Acceptance |
| --- | --- | --- | --- |
| `baseline` (`generate_text`) | 42.1 | 48 | — |
| `fast_decode` | 55.6 | 48 | — |
| `speculative_ngram` | 49.6 | 48 | 0.50 |
| `speculative_model` (random draft) | 12.2 | 96 | 0.00 |

Reproduce with:

```bash
.koemi-venv/Scripts/python.exe benchmarks/run_decode_benchmark.py \
  --embedding-size 128 --expert-count 8 --expert-top-k 2 \
  --max-new-tokens 48 --greedy --synthetic-draft --draft-length 4
```

Read those numbers for what they are. The model has random weights, so the
n-gram drafter half-fails and the random draft model fails completely; the last
row is the cost of speculation without acceptance, measured rather than argued.
The CUDA graph path is absent because this host has no CUDA device. Nothing here
predicts an A100 result: the overhead fast decode removes is larger on a GPU
than on a CPU, and the speculation ratio depends on a trained model, so both
numbers must be measured again on the target device.

## Hybrid tokenizer

`koemi.data.hybrid_tokenizer` is byte-level BPE with the span markers reserved.

The id layout keeps the byte checkpoint meaningful:

| Range | Content |
| --- | --- |
| `0 … 255` | one byte each |
| `256` | padding, unchanged |
| `257 … 260` | `<\|system\|>`, `<\|input\|>`, `<\|thinking\|>`, `<\|output\|>` |
| `261 …` | learned merges |

Because the alphabet is the full byte range, coverage is total: an unseen
character does not become an unknown token, it decodes to its own bytes.
`decode(encode(text)) == text` for every text the tests exercise, including
emoji, CJK, control bytes and the full `bytes(range(256))` payload through
`encode_bytes`/`decode_bytes`.

**Markers are atomic.** `serialize_record_tokens` encodes each span separately,
so no merge can straddle a marker and shift the supervised mask.
`encode_prompt` builds an inference prompt through the same spans, so the
tokenization at inference matches the one training wrote. The test asserts the
property directly: the ids the supervised mask selects decode back to exactly the
record's output text.

**Training.** `train_hybrid_vocabulary` counts pretokenizer pieces, then learns
merges with an incremental pair index. The pretokenizer groups `_` with letters,
so `queue_push` stays one piece, which matters for code. Ties are broken by the
smallest id pair, so the same corpus produces the same vocabulary.

```bash
.venv/bin/python -m koemi build-vocabulary \
  --dataset data/corpus.jsonl --output artifacts/vocabulary.json \
  --vocabulary-size 8192
.venv/bin/python -m koemi train \
  --dataset data/corpus.jsonl --vocabulary artifacts/vocabulary.json \
  --checkpoint artifacts/koemi-hybrid.pt
```

The vocabulary travels inside the checkpoint, so `generate` needs no flag: it
builds the tokenizer the checkpoint declares. A checkpoint without a vocabulary
stays a byte checkpoint, and the checkpoint format version does not change.

## Migrating a byte checkpoint

```bash
.venv/bin/python -m koemi expand-vocabulary \
  --checkpoint artifacts/koemi-3hip.pt \
  --vocabulary artifacts/vocabulary.json \
  --output artifacts/koemi-hybrid.pt
```

`expand_model_vocabulary` copies every existing embedding and head row unchanged
and initializes each new row at the mean of the rows of the bytes it expands to.
Everything outside the vocabulary parameters is loaded as is.

One consequence is not cosmetic and is asserted by a test: **surprise changes.**
Surprise is the causal NLL of the observed token normalized by `log(content
vocabulary size)`, so growing the vocabulary changes both the cross entropy and
the normalizer. The memory write weights depend on surprise, so the forward moves
even on byte-only input. Expansion preserves the weights; it does not preserve the
function. Treat the migrated checkpoint as a warm start that still needs
training, not as the same model with a bigger head.

## What is not covered

- `MaterializedCausalByteDataset` in `src/koemi/training/a100_run.py` stores its
  token streams as `bytes`, so the A100 runner is byte-only. The hybrid path runs
  through the CLI trainer. Moving the A100 runner would mean changing its stream
  storage to a wider integer array, and that file drives a paid run.
- No CUDA measurement exists for any of this. The CUDA graph tests are written
  and skip on a host without a device; the benchmark reports the graph path only
  when CUDA is present.
- Graph capture combined with BF16 autocast is written but unexecuted. Capture
  disables the autocast weight cache, because a cast recorded once and replayed
  many times is the documented hazard; run the CUDA graph tests on the target
  device before trusting that combination.
- The vocabulary is learned from pretokenizer pieces, not from a quality
  ablation. A vocabulary size is a hypothesis until bits per byte is compared at
  matched compute.
- Speculative decoding is single-sequence. Batched speculation needs ragged
  acceptance handling that this implementation does not attempt.
