# Fast Whisper — Adaptive Speculative Inference

Whisper-Tiny drafts tokens; Whisper-Large-v3 verifies a block at a time. The
project studies decoder execution overhead, KV-cache rollback and the conditions
under which speculation pays for its extra encoder and draft work.

**Demo:** upload a short English speech clip and compare transcripts, exact token
agreement, latency, acceptance and the draft-length schedule across four paths.

**GPU notebook:** [Open in Colab](https://colab.research.google.com/github/AarushRathore1729/Simplismart_Assign/blob/feat/whisper-adaptive-profile/notebooks/gpu_upgrade.ipynb).
The same notebook runs in a Lightning AI CUDA environment.

| Path | Implementation |
|---|---|
| `baseline` | Target-only greedy decoding with a persistent KV cache |
| `speculative` | Existing cached fixed-k draft/verify decoder |
| `optimized` | Fixed-k with preallocated token buffers, device-side token mapping and deferred draft-cache catch-up |
| `adaptive` | Buffered decoder with acceptance-based k adjustment and target-only fallback |

The buffered decoder avoids per-proposal token transfers to the host and sends
one small acceptance/EOS control packet per verification block. Fully accepted
blocks defer the final draft-cache extension: the next draft call processes the
two uncached tokens together. Rejection crops self-attention state while keeping
the encoder cross-attention cache intact.

Only token workspaces are preallocated; KV tensors remain dynamically managed
by Transformers. This project uses PyTorch's CUDA kernels and SDPA, with no
claim of a hand-written CUDA kernel.

## Run on Lightning AI or Colab

Select a CUDA GPU first. The full target/draft pair is intended for a GPU with
at least 16 GB VRAM. The notebook preserves the environment's installed PyTorch.

```bash
git clone --branch feat/whisper-adaptive-profile --single-branch \
  https://github.com/AarushRathore1729/Simplismart_Assign.git
cd Simplismart_Assign
python -m pip install -e '.[dev,benchmark,demo]' 'transformers==4.57.1'
python -m pytest
```

First check two real clips. This is a smoke check, not the résumé evaluation:

```bash
specvoice-compare --audio audio1.wav audio2.wav --max-samples 2 \
  --repetitions 1 --warmup-runs 1 --max-new-tokens 128 \
  --output results/gpu_smoke.json
```

Then collect all four paths on the same LibriSpeech inputs:

```bash
specvoice-compare --max-samples 10 --repetitions 3 --warmup-runs 1 \
  --max-new-tokens 128 --profile --output results/comparison.json
```

Add `--verify-generate` for an additional untimed comparison with Hugging Face
`generate()`. Every measured run already requires exact agreement with the
shared target greedy decoder. If tokens differ, the runner saves mismatch
evidence and refuses to report a speedup. Near-tied logits can behave differently
in different floating-point block shapes; an observed mismatch requires
investigation before an equivalence claim.

If any `truncated_runs` value is nonzero, repeat with `--max-new-tokens 256`
before reporting WER. The default dataset is a small test subset; report its
actual sample count rather than calling it a full LibriSpeech evaluation.

## Audio-upload demo

```bash
specvoice-demo --port 7860
```

Open port 7860 in Lightning AI. The notebook can launch the same demo inline.
`--share` enables a public temporary Gradio link when explicitly requested.

## What the benchmark records

- Warmup and randomized measurement order, with repeated measurements.
- p50/p95 request latency across per-utterance repetition medians.
- Separate feature preparation, target encoder, draft encoder and decoder time.
- Serial throughput, draft acceptance, model calls, k history and fallback tokens.
- Peak allocated CUDA memory and incremental peak above resident allocations.
- Exact token agreement and normalized WER when reference text is available.
- GPU, dtype, software versions, model revisions and source commit.

Request latency includes waveform resampling, feature preparation, both required
encoders, decoding and text conversion. It excludes model loading, file I/O and
optional oracle checks. Component inference latency is reported separately.
All variants have both models resident; memory is not a target-only deployment
comparison. Small-sample p95 is descriptive, and throughput is serial batch size
one, not a concurrent-serving result.

## Profiling

`--profile` runs separately from the headline latency measurement and exports:

- `comparison.trace.json`: CPU/CUDA timeline, shapes and memory events.
- `comparison.profile.txt`: operator statistics.
- `comparison.stages.json`: CPU and CUDA stream spans for draft, verification,
  cache cropping, control and fallback.

Open the trace in a local Chrome trace viewer or Perfetto. CUDA stream spans can
include host launch gaps; they are not sums of kernel execution time.
NVTX ranges also support an optional Nsight Systems capture when installed:

```bash
nsys profile --trace=cuda,nvtx,osrt -o results/whisper_nsys \
  specvoice-transcribe audio1.wav --mode adaptive --profile-stages
```

## Validation and scope

Greedy English short-form audio, batch size one, at most 30 seconds. No streaming,
training, distributed serving or top-p sampling is part of this upgrade.

The CPU checks pass for cache-sensitive acceptance/rejection, EOS inside a block,
token-budget boundaries, adaptive fallback, actual Transformers cache objects
and pretrained Tiny inference. See [validation](docs/validation.md).
Large-v3 GPU improvements must be measured before being quoted.

The earlier assignment code remains under `fast_whisper/`; its historical
README and results are in [legacy_assignment.md](docs/legacy_assignment.md).
Those sampling results are a different implementation and are not evidence
for the new greedy decoder. Use the `specvoice-*` commands above for this upgrade.
