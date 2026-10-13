# Resume wording

**Fast Whisper — GPU Inference Optimization** · Python, PyTorch, CUDA profiling

- Accelerated Whisper-Large-v3 with Whisper-Tiny draft–verify decoding, cutting
  p50 request latency by 37.6% (820 → 512 ms) and achieving 2.00× decoder speedup
  on NVIDIA T4, with exact target-token agreement across 30 utterances and three runs.
- Implemented KV-cache rollback, buffered token workspaces and cached suppression
  masks; reduced CUDA runtime launch calls by 21.9% on a profiled clip and built
  reproducible GPU benchmarks, a Gradio demo and FastAPI inference endpoint.

[Website](https://fast-whisper-aarush.socialhill32.chatgpt.site)
· [GitHub](https://github.com/AarushRathore1729/fast-whisper)

The 30-utterance evaluation passed the Hugging Face generate() oracle and all
90 per-run optimized/target digest comparisons; normalized WER was unchanged
at 3.22%. Evidence: `benchmarks/t4_20261013_final`, recorded source `1bed58a`.

The request speedup includes required encoders and feature/text processing;
model loading and file I/O are excluded. The decoder speedup is measured
separately. The launch-call result belongs to one instrumented clip and applies
to the CUDA runtime API; driver launch calls are unchanged. The subset has one
speaker across three chapters, rather than a full LibriSpeech evaluation.
