# SpecVoice foundation

This branch rebuilds the original assignment prototype as a correctness-first
research and engineering project.

## Current milestone

The foundation implements:

- a cached greedy target decoder;
- greedy speculative decoding with batched target verification;
- persistent and rollback-safe target/draft KV-cache handling;
- exact token-equivalence tests covering acceptance and rejection;
- synchronized GPU timing helpers;
- an offline benchmark that refuses to report speedup if tokens differ;
- structured per-component latency and acceptance-rate output.

Sampling is intentionally excluded. The previous top-p implementation did not
apply the rejection correction required to preserve the target distribution.
It will only return after distribution-level correctness tests are included.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,benchmark]'
pytest
```

## Run the offline benchmark

```bash
specvoice-benchmark \
  --target-model openai/whisper-large-v3 \
  --draft-model openai/whisper-tiny \
  --dataset hf-internal-testing/librispeech_asr_dummy \
  --dataset-config clean \
  --split validation \
  --max-samples 10 \
  --draft-k 4 \
  --warmup-runs 1 \
  --repetitions 3 \
  --output results/foundation.json
```

The benchmark performs an explicit warm-up, synchronizes GPU measurements and
reports medians across repeated runs. It records target-encoder, draft-encoder,
baseline-decoder and speculative-decoder time separately. Baseline total latency
includes the target encoder. Speculative total latency includes both encoders.

## Correctness boundary

Both paths currently consume raw model logits. Hugging Face `generate()` can
apply Whisper-specific suppression and timestamp processors, so it is not used
as the token-equivalence oracle. A shared Whisper logits-processor adapter is
the next ASR milestone.

## Next milestones

1. **Implemented on `feat/whisper-integration`:** shared Whisper prompt,
   suppression and timestamp policy plus an end-to-end short-form audio CLI.
2. Larger LibriSpeech evaluation with repeated runs and confidence intervals.
3. Streaming audio buffer, VAD and local-agreement transcription.
4. FastAPI WebSocket session and browser microphone client.
5. Streaming LLM/TTS, verified tool gate and barge-in cancellation.
