# Resume wording

**Fast Whisper — GPU Inference Optimization** · Python, PyTorch, CUDA profiling

- Accelerated Whisper-Large-v3 using Whisper-Tiny draft–verify decoding; reduced
  p50 request latency from 833 to 488 ms (41.4%, 1.71×) on an
  NVIDIA T4 with exact target-token agreement and unchanged normalized WER across
  10 LibriSpeech test utterances and three repeated runs.
- Implemented persistent KV-cache rollback, preallocated token workspaces and
  deferred draft-cache catch-up; reduced average draft forward calls by 11.3%
  against the existing fixed-k decoder and compared fixed/adaptive scheduling
  through CPU/CUDA profiling, reproducible benchmarks and an upload demo.

Evidence: `benchmarks/t4_20261013/comparison.json`, recorded source `90b883a`.
The 1.71× compares with target-only inference; the buffered-vs-existing-fixed
p50 difference is 1.2% in this run. Do not attribute the whole gain to buffering.
No GPU speedup is claimed yet for the optional cached-suppression follow-up.
