# Resume wording

Project title: **Fast Whisper — Adaptive Speculative Inference**

Implementation statements supported by the code and correctness tests:

- Implemented greedy Whisper draft–verify inference with persistent KV-cache
  rollback, preallocated token buffers, acceptance-based draft scheduling and
  target-only fallback; verified token equivalence across acceptance, rejection
  and EOS cases.
- Built a reproducible CUDA-synchronized benchmark and audio-upload demo comparing
  target-only, fixed and adaptive decoding; exports request latency, throughput,
  memory, WER and separate CPU/CUDA profiling traces.

Replace the second bullet with actual measured findings after the GPU run:

> Reduced [request or decoder] p50 latency from [A] to [B] ms ([C]% reduction)
> on [GPU, dtype, N utterances], with [D]% draft acceptance and exact token
> agreement with the target greedy decoder; profiled encoder, draft, verification
> and cache overhead.

Use request latency for an end-to-end inference statement; use decoder latency
only when explicitly saying decoder. Quote p95 only with sample count and workload
scope. A ten-utterance run supports a small evaluation, not a production claim.
Do not write CUDA C++ kernels, Nsight analysis or a GPU speedup unless those were
actually implemented or run and supported by saved evidence.
