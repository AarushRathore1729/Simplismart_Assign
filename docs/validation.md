# Validation record

Validation performed on 13 October 2026 (India time):

- 64 tests passed, including the audio-upload callback and report download;
  Ruff source and test checks passed.
- Python 3.12, PyTorch 2.14.1+cpu, Transformers 4.57.1.
- No CUDA GPU is available in the implementation environment.
- Real Transformers Whisper models with randomly initialized small dimensions
  validate encoder-decoder KV-cache integration without model downloads.
- Cache-sensitive toy models validate rejection, full acceptance, catch-up,
  EOS, budget limits and target-only fallback. Their predictions depend on
  the complete cache history, not just the most recent token.
- Pretrained `openai/whisper-tiny` as both target and draft, using `audio1.wav`,
  64 maximum new tokens, all four paths: exact target token agreement passed.
- The pretrained target-only path also matched Hugging Face `generate()`.
- CPU profiling exported a timeline, operator table and component spans.
- Two LibriSpeech dummy-test utterances, two repeats per mode: all four paths
  matched the target tokens and had identical normalized WER (0.0741).
- The actual Large-v3/Tiny vocabularies and English decoder prompt were checked.
  The single target-only language token is handled as an unmappable draft
  conditioning token; target verification still decides all committed output.

The pretrained Tiny/Tiny smoke check tests model integration and correctness;
it does not test the Large-v3/Tiny performance hypothesis. CPU speculation was
slower than target-only decoding in this check. No GPU speedup is established
by these results. The GPU notebook contains the remaining evaluation steps.

After a CUDA evaluation, keep the generated comparison JSON, Markdown table
and trace with the exact model, GPU, dtype, sample count and repetition count.
Do not reuse the legacy top-p benchmark numbers for this decoder.

## GPU evidence now verified

The provided T4 archive evaluates Large-v3/Tiny on ten unique test utterances,
three repetitions, with all four paths matching target tokens and no truncation.
The raw evidence and recomputed audit are in `benchmarks/t4_20261013`.
Buffered fixed drafting reached 488.3 ms p50 request latency versus 833.2 ms
for target-only (1.706×); normalized WER is 0.03937 in every path.

The current extension adds cached token-suppression masks, wider deterministic
sample selection, per-run token digests, a report audit and the REST API.
84 tests pass. Cached-mask FP16/FP32 scores match Hugging Face, and pretrained
Tiny audio with the additional generate() oracle check passes on CPU.
The added mask-caching option has a separate GPU notebook experiment and is
not included in the recorded 1.71× result.
