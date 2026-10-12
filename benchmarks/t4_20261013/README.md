# T4 evaluation · 13 October 2026

Recorded source: `90b883aa549051f0996ae92a314441296a8dcae5`.
Models: Whisper-Large-v3 and Whisper-Tiny, automatic CUDA FP16, SDPA.
Hardware: Tesla T4 (compute capability 7.5), CUDA runtime 13.0.
Software: Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 4.57.1.

Ten unique first-ten test-subset utterances, three repetitions per mode,
randomized path order, one warmup. Four paths; all token comparisons passed;
zero truncation. The evaluation records request latency and decoder latency
separately. Exact tokens are checked by the benchmark against its shared greedy
target policy. This run did not enable the additional Hugging Face oracle check.

- `comparison.json`: original full evaluation, unchanged.
- `gpu_smoke.json`: original two-file GPU integration check, unchanged.
- `comparison.profile.txt`: operator statistics from a separate adaptive profile.
- `comparison.stages.json`: component spans for that instrumented clip.
- `audit.json`: independently checked structure/counts and recomputed summaries.

The original uploaded archive has SHA-256
`434157a5b8188dc2b1d07c2ab825f38457887dc1b773e9a576ac9aaed63904ca`.
The large trace remains in that archive; Git contains the compact operator and
component evidence. New runs export traces and token digests locally.

The profile's 1.04-second request time is instrumented and is excluded from the
headline latency table. The `aten::isin` count and related mask construction
motivated the optional cached-suppression follow-up; these initial GPU numbers
predate it.

Recompute the audit:

```bash
specvoice-audit benchmarks/t4_20261013/comparison.json
python scripts/render_benchmark.py benchmarks/t4_20261013/comparison.json
```
