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
These initial ten-utterance numbers are preserved separately from the final
30-utterance evaluation below.


## Final GPU validation

The final uploaded archive records source `1bed58a`, Tesla T4, FP16, PyTorch
2.11.0+cu130 and Transformers 4.57.1. Thirty unique evenly spaced utterances,
three repetitions, one speaker and three chapters: the Hugging Face generate()
oracle passed on all inputs, all 90 token-digest comparisons passed, and no
outputs were truncated. Summary metrics were recomputed from the raw rows.

- p50 request: 820.4 to 512.1 ms (37.6% reduction, 1.602× speedup).
- p95 request: 1793.6 to 1047.1 ms (41.6% reduction).
- p50 decoder: 655.6 to 328.0 ms (1.998× speedup).
- Normalized WER: 0.032197 for both paths.

The full traces were checked on the same 23-generated-token clip: token IDs,
five target calls and 20 draft calls are identical. The cached-mask trace removes
all 90 isin and unique events. Runtime cudaLaunchKernel calls fall from 8427 to
6578, while driver cuLaunchKernel calls remain 1354. These are instrumented
trace counts, not a corpus-wide latency attribution.

Version 0.3.1 enables the already GPU-validated cached-mask path by default;
--no-cached-suppression preserves the uncached comparison. The original
benchmark data remain unchanged and the final evidence has its own directory.
