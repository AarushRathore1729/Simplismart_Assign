# Sample audio

Two bundled WAV clips for smoke checks and the audio-upload demonstration.
From the repository root:

```bash
specvoice-compare --audio samples/audio1.wav samples/audio2.wav \
  --max-samples 2 --repetitions 1 --warmup-runs 1 \
  --max-new-tokens 128 --output results/gpu_smoke.json
```

The 30-utterance benchmark uses the LibriSpeech dummy clean validation subset
and is recorded separately in `benchmarks/t4_20261013_final`.
