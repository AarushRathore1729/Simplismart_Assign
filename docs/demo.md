# One-minute demo

1. Launch `specvoice-demo` on the GPU used for the benchmark.
2. Upload `audio1.wav` or a short English clip. Click **Transcribe and compare**.
3. Show the target-only and buffered fixed-draft transcripts and exact agreement.
4. Point out the measured request and decoder times for all four paths.
5. Open the raw measurements and show draft calls, target calls, draft lengths
   and fallback. Download the result.

Suggested explanation:

“Tiny proposes several tokens, and Large-v3 checks the block in one pass. A
mismatch commits the target correction and rolls back speculative cache state.
In my T4 test, fixed drafting cut median request latency from 833 to 488 ms.
Adaptive drafting was slower, so fixed k=4 is the default. I kept both paths
and the profiler output to explain the tradeoff.”

For the uploaded clip, use the latency shown by that actual run. The 833/488 ms
numbers belong to the recorded ten-utterance benchmark, not every possible clip.

## REST upload

```bash
specvoice-serve --port 8000
curl -X POST 'http://127.0.0.1:8000/transcribe?mode=optimized&max_new_tokens=128' \
  -F 'file=@audio1.wav'
```

`GET /health` reports readiness. `/docs` exposes the request schema. The response
includes text, token IDs, encoder/decoder/request times and decode statistics.
Run one server worker per GPU for this demo.
