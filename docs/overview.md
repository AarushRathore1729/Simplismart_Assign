# Implementation overview

The cached greedy foundation has developed into the GPU inference project
specified in the [README](../README.md), with [decoder design](design.md),
[measured T4 evidence](../benchmarks/t4_20261013_final) and [validation](validation.md).

The project includes cache-safe greedy draft/verify decoding, shared Whisper
prompt/suppression policy, semantic tokenizer mapping, buffered and adaptive
implementations, synchronized GPU benchmarks, profiling, audio upload and REST.

The earlier sampling implementation is retained in [archive/prototype](../archive/prototype).
Current speedup and exact-token claims use greedy decoding. Streaming/VAD,
LLM/TTS orchestration and distributed serving are outside this project release.
