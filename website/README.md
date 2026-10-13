# Fast Whisper benchmark showcase

[Public website](https://fast-whisper-aarush.socialhill32.chatgpt.site)

A static, responsive project page with measured aggregate comparisons, all 30
utterances, repeated timings, exact transcript/token checks, raw report downloads,
and an explicitly illustrative draft–verify walkthrough. No GPU is needed to view
the showcase. Its live-transcription button opens the GPU notebook.

Build and preview from the repository root:

```bash
python website/build.py
python -m http.server 8080 --directory website/dist
```

The builder reads the committed final GPU evidence directly; it does not rerun or
modify benchmark measurements. Generated `dist/` output is ignored by Git.

JavaScript syntax and interactions were checked, including all 30 clip selections,
three latency metrics, all four walkthrough steps and all 90 token hashes.
Browser visual QA was unavailable in the build environment.
