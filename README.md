# Fast Whisper — GPU Inference Optimization

Whisper-Tiny drafts tokens and Whisper-Large-v3 verifies a block at a time.
This project measures the cost of drafting, target verification, cache rollback,
token processing and host–GPU control in autoregressive speech inference.

**Recorded result: 1.71× p50 request speedup on a Tesla T4**, from 833 to 488 ms,
with exact target-token agreement and unchanged normalized WER on ten test
utterances. Fixed drafting was faster than the adaptive heuristic, so buffered
fixed k=4 is the default.

![Recorded T4 benchmark](docs/assets/t4_latency.svg)

## Recorded benchmark

Large-v3/Tiny, FP16, SDPA, batch size one; the first ten utterances from
`hf-internal-testing/librispeech_asr_dummy` (`clean`, `validation`), one warmup
and three repeats. Each mode is timed in randomized order.

| Path | p50 request | p95 request | p50 speedup | Draft acceptance | Normalized WER |
|---|---:|---:|---:|---:|---:|
| Target only | 833 ms | 2503 ms | 1.00× | — | 3.94% |
| Cached fixed k=4 | 494 ms | 1411 ms | 1.69× | 73.4% | 3.94% |
| Buffered fixed k=4 | 488 ms | 1381 ms | 1.71× | 72.1% | 3.94% |
| Adaptive k=1–8 | 586 ms | 1724 ms | 1.42× | 68.6% | 3.94% |

All measured token comparisons passed; no output reached the token limit. This
is a small test-subset result, with descriptive latency quantiles. It does not
establish performance on full LibriSpeech or concurrent serving.

The buffered path is only 1.2% lower in p50 request latency than the existing
cached fixed path in this run. The headline 1.71× compares against target-only
inference. Deferred draft-cache catch-up reduces average draft forward calls
from 38.8 to 34.4. Adaptive fallback made this workload slower than fixed drafting.

Raw reports, profiler operator statistics, component spans and an audit are
committed in [benchmarks/t4_20261013](benchmarks/t4_20261013). The recorded source
is commit `90b883a`. The optional mask-caching optimization added afterward has
its own notebook evaluation; its performance is not included in this table.

## Quick start

Use a CUDA GPU. The tested T4 has 16 GB VRAM. The notebook preserves the
runtime's installed CUDA PyTorch and pins the tested Transformers release.

[Open GPU notebook in Colab](https://colab.research.google.com/github/AarushRathore1729/Simplismart_Assign/blob/main/notebooks/gpu_upgrade.ipynb)
· [Notebook source](notebooks/gpu_upgrade.ipynb) · [Decoder design](docs/design.md)

```bash
git clone --branch feat/whisper-adaptive-profile --single-branch   https://github.com/AarushRathore1729/Simplismart_Assign.git
cd Simplismart_Assign
python -m pip install -e '.[dev,benchmark,demo,api]' 'transformers==4.57.1'
python -m pytest
```

First run a two-clip smoke check:

```bash
specvoice-compare --audio audio1.wav audio2.wav --max-samples 2   --repetitions 1 --warmup-runs 1 --output results/gpu_smoke.json
```

Reproduce the recorded workload and audit the result:

```bash
specvoice-compare --max-samples 10 --repetitions 3 --warmup-runs 1   --max-new-tokens 128 --profile --output results/comparison.json
specvoice-audit results/comparison.json --output results/comparison.audit.json
```

## Demo and REST API

```bash
specvoice-demo --port 7860
specvoice-serve --port 8000
```

Launch the desired interface. The demo accepts an English audio upload, displays
matching target/fixed-draft transcripts, compares all four paths and exports the
measured result. Open forwarded port 7860 in Lightning AI; the notebook also
supports an inline Gradio demo. `--share` explicitly enables a temporary public
Gradio link. See the [one-minute demo guide](docs/demo.md).

The REST API loads one engine and provides `POST /transcribe`, `GET /health`
and `/docs`, with JSON text, tokens and component timings.

```bash
curl -X POST 'http://127.0.0.1:8000/transcribe?mode=optimized&max_new_tokens=128'   -F 'file=@audio1.wav'
```

## Implementations

| Mode | What it does |
|---|---|
| `baseline` | Target-only greedy decoding with persistent KV state |
| `speculative` | Existing cached fixed-k draft/verify decoder |
| `optimized` | Preallocated token workspaces, device-side ID mapping and deferred draft-cache catch-up |
| `adaptive` | Buffered decoder with acceptance-based k adjustment and target-only fallback |

Rejection commits the accepted prefix plus the target correction and crops only
self-attention KV state. Fully accepted blocks consume the two uncached draft
inputs together on the next iteration. Draft proposals stay on device; one small
acceptance/EOS packet is transferred per verification block. Transformers still
manages dynamic KV allocations. Implementation: Python, PyTorch SDPA/CUDA,
PyTorch Profiler and NVTX.

## Profile-guided follow-up

The uploaded T4 trace exposed repeated `arange`/`isin`/`unique` work in token
suppression. `--cached-suppression` builds boolean masks once, preserving the
shared target policy. FP16/FP32 score-equivalence tests and a pretrained CPU
oracle check pass. The GPU notebook includes this separate wider evaluation:

```bash
specvoice-compare --modes baseline optimized --max-samples 30   --sample-strategy evenly-spaced --repetitions 3 --max-new-tokens 256   --cached-suppression --verify-generate --profile --output results/expanded.json
specvoice-audit results/expanded.json --output results/expanded.audit.json
```

Selected clips above 30 seconds are recorded and skipped. Report the actual
completed sample count. If any output is truncated, use a larger permitted token
budget before reporting WER. The target greedy oracle continues using the
original Hugging Face suppression processors.

## Measurement and validation

- Request latency includes resampling, feature preparation, required encoders,
  decoding and text conversion. It excludes model loading, file I/O and oracle checks.
- Component encoder/decoder times are synchronized CUDA measurements. Serial
  throughput and peak allocated CUDA memory are recorded separately.
- Both models are resident in every path; memory is not a target-only deployment
  comparison. CUDA stream spans can include host launch gaps.
- Profiling runs separately and exports `.trace.json`, `.profile.txt` and
  `.stages.json`, with ranges for draft, verification, control, cache and fallback.
- Every measured run requires exact tokens against the target decoder. New
  reports include per-run token digests and recomputable summary metrics.
- 84 tests cover cache-sensitive rejection, EOS, budget limits, actual Transformers
  cache objects, suppression equivalence, uploaded audio, REST responses and audit failures.

An additional `--verify-generate` check compares the target path with Hugging Face
`generate()` outside timing. A numerical token mismatch stops publication of
speedup evidence and saves the discrepancy. [Validation record](docs/validation.md).

Scope: greedy English transcription, batch size one, audio up to 30 seconds.

## Repository map

- `src/specvoice/asr/`: decoders, cache rollback, Whisper policy and transcription.
- `src/specvoice/compare.py`, `audit.py`: measured comparison and result verification.
- `src/specvoice/demo.py`, `web_api.py`: upload demonstration and REST interface.
- `notebooks/gpu_upgrade.ipynb`: Colab/Lightning execution and evidence export.
- `benchmarks/`: committed measured evidence with source and hardware provenance.
- `scripts/render_benchmark.py`: reproducible SVG/PNG benchmark figure.
- `tests/`: offline correctness, API and benchmark validation.

The original assignment remains under `fast_whisper/`; its [historical README](docs/legacy_assignment.md)
and sampling results are a separate implementation. Use the `specvoice-*` commands
for the verified greedy inference project.
