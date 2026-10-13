# Resume wording

**Fast Whisper — GPU Inference Optimization** · Python, PyTorch, CUDA profiling

- Accelerated Whisper-Large-v3 with Whisper-Tiny draft–verify decoding; reduced
  p50 request latency from 820 to 512 ms (37.6%, 1.60×) and achieved 2.00× decoder
  speedup on an NVIDIA T4 with exact target-token agreement across 30 test
  utterances and three repeated runs.
- Implemented KV-cache rollback, buffered token workspaces and cached suppression
  masks; eliminated repeated mask construction and reduced `cudaLaunchKernel`
  runtime calls by 21.9% on a profiled 23-token clip, with reproducible benchmarks,
  an audio-upload demo and FastAPI.

The 30-utterance evaluation passed the Hugging Face generate() oracle and all
90 per-run optimized/target digest comparisons; normalized WER was unchanged
at 3.22%. Evidence: `benchmarks/t4_20261013_final`, recorded source `1bed58a`.

The request speedup includes required encoders and feature/text processing;
model loading and file I/O are excluded. The decoder speedup is measured
separately. The launch-call result belongs to one instrumented clip and applies
to the CUDA runtime API; driver launch calls are unchanged. The subset has one
speaker across three chapters, rather than a full LibriSpeech evaluation.
