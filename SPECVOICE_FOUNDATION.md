# SpecVoice foundation

The cached greedy foundation has developed into the GPU inference project
specified in the [README](README.md), with [decoder design](docs/design.md),
[measured T4 evidence](benchmarks/t4_20261013) and [validation](docs/validation.md).

The project includes cache-safe greedy draft/verify decoding, shared Whisper
prompt/suppression policy, semantic tokenizer mapping, buffered and adaptive
implementations, synchronized GPU benchmarks, profiling, audio upload and REST.

The earlier sampling prototype remains a historical assignment implementation.
Current speedup and exact-token claims use greedy decoding. Streaming/VAD,
LLM/TTS orchestration and distributed serving are outside this project release.
