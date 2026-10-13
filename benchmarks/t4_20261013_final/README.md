# Final T4 evaluation · 13 October 2026

Recorded source: `1bed58a422c42bb83751dfdcdd7aeb2985a171a8` (clean tree).
Large-v3/Tiny, FP16, SDPA, Tesla T4, CUDA runtime 13.0.
Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 4.57.1.

The final `expanded.json` evaluates 30 unique evenly spaced test-subset utterances,
three repetitions, randomized order and one warmup. Inputs span one speaker and
three chapters. The independent Hugging Face generate() oracle and all 90
optimized/target token digests pass; no outputs were truncated.

| Metric | Target-only cached greedy | Buffered k=4 + cached masks |
|---|---:|---:|
| p50 request | 820.4 ms | 512.1 ms |
| p95 request | 1793.6 ms | 1047.1 ms |
| p50 decoder | 655.6 ms | 328.0 ms |
| Normalized WER | 3.22% | 3.22% |
| Serial generated tokens/s | 23.29 | 37.03 |

Request speedup is 1.602×; decoder speedup is 1.998×. Request timing excludes
file I/O, model loading and the additional correctness oracle. Both model weights
remain resident in every path; these are serial batch-size-one measurements.

- `expanded.json`: final raw measurements, unchanged.
- `expanded.audit.json`: provided audit, unchanged.
- `expanded.recomputed.audit.json`: audit recomputed after upload.
- `comparison.json` and `comparison.audit.json`: separate uncached ten-utterance
  four-path rerun, retained for transparency.
- `*.profile.txt` and `*.stages.json`: separate instrumented profile evidence.
- `profile_comparison.json`: event counts from both full traces. The profiled
  clip produces identical token IDs, five target calls and 20 draft calls.

The comparison trace contains 90 isin and 90 unique events; the cached-mask
trace contains zero of each. Runtime cudaLaunchKernel calls fall 8427→6578,
and driver cuLaunchKernel calls stay at 1354. Profile timing is excluded from
headline latency; its counts establish removed work on this one clip.

Archive SHA-256: `af0843334d95f426e63fd5b176f6139a1fc4e3683b0773768f49fe3fc97a2a3e`.
Large raw traces remain in that archive. Compact reports are committed here.
The initial run remains in `../t4_20261013`; do not combine its timing statistics
with this evaluation, because selection and conditions differ.

```bash
specvoice-audit benchmarks/t4_20261013_final/expanded.json
python scripts/render_benchmark.py benchmarks/t4_20261013_final/expanded.json
```
