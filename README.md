# Fast Whisper Speculative Decoding

This project implements speculative decoding for OpenAI's Whisper models to accelerate inference. It uses a small draft model (Whisper Tiny) to propose tokens and a large target model (Whisper Large V3) to verify them.

## Functionality

- **Speculative Decoding**: Uses a draft-verify mechanism where a small model generates candidate tokens that are verified in parallel by the large model.
- **Greedy & Nucleus Sampling**: Supports both greedy decoding and top-p (nucleus) sampling.
- **KV Caching**: Optimized with Key-Value caching for both draft and target models.
- **Benchmarking**: Tools to compare baseline vs. speculative performance (WER and Speedup).
- **Web API**: FastAPI server for remote transcription.

## Installation

### Prerequisites
- Python 3.8+
- `ffmpeg` installed on your system.

### Setup
```bash
# Clone the repository
git clone https://github.com/AarushRathore1729/Simplismart_Assign.git
cd Simplismart_Assign

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torchcodec  # Required for audio decoding
```

## Usage

### 1. Run Benchmark (Local or Colab)

**Google Colab:**
Use the provided notebook `colab_runner.ipynb` to run the benchmark on a free T4 GPU. This avoids large downloads and disk usage issues by using a dummy dataset.

**Local Run:**
Run a single inference pass with specific parameters:

```bash
# Run with Dummy dataset (Fast, no large download)
python run.py --target-model openai/whisper-large-v3 --draft-model openai/whisper-tiny --dataset hf-internal-testing/librispeech_asr_dummy --split validation --max-samples 10

# Run with Full LibriSpeech (Requires ~30GB disk space)
python run.py --dataset librispeech_asr --split validation_clean --max-samples 10
```

### 2. Web API

Start the FastAPI server for remote transcription. You can set the models via environment variables.

```bash
# Start server (defaults to Large-V3 target, Tiny draft)
export TARGET_MODEL=openai/whisper-large-v3
export DRAFT_MODEL=openai/whisper-tiny
uvicorn fast_whisper.web_api:app --port 8000
```

**Test the API:**
```bash
# Create a dummy audio file
ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -y test.wav

# Send request
curl -X POST -F "file=@test.wav" http://127.0.0.1:8000/transcribe
```

**Response:**
```json
{
  "text": ["...transcribed text..."],
  "duration": 0.5
}
```

## Results

I compared the performance of Speculative Decoding (Large-V3 + Tiny) vs Baseline (Large-V3 only).

| Environment | Device | Speedup | WER (Baseline) | WER (Speculative) | Notes |
|------------|--------|---------|----------------|-------------------|-------|
| **Google Colab** | T4 GPU | **1.76x** | 0.0945 | 0.0945 | Best config: `top_p=0.4`, `draft_k=8`. |
| **Local Machine** | CPU | 0.61x | 0.0861 | 0.0861 | Slower due to CPU overhead and serial execution. |

*Note: Speculative decoding requires a GPU to be effective. On CPU, the overhead of running the draft model sequentially outweighs the verification gains.*

### Hyperparameter Tuning (Experiment)
I ran a grid search on Google Colab (T4 GPU) to find the optimal `top_p` and `draft_k` values.

**Top 3 Configurations:**
1. **Speedup: 1.76x** | `top_p=0.4`, `draft_k=8` | WER: 0.0945 (Best Speedup)
2. **Speedup: 1.75x** | `top_p=0.2`, `draft_k=6` | WER: 0.0945 (Best WER tie)
3. **Speedup: 1.74x** | `top_p=0.2`, `draft_k=8` | WER: 0.1063

The experiment shows that a higher `draft_k` (8) combined with a moderate `top_p` (0.4) yields the best speedup on this dataset.


