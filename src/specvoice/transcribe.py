"""Command-line entry point for real audio transcription."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .asr.transcriber import SpeculativeWhisperTranscriber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with SpecVoice")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--target-model", default="openai/whisper-large-v3")
    parser.add_argument("--draft-model", default="openai/whisper-tiny")
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    parser.add_argument("--mode", choices=("baseline", "speculative"), default="speculative")
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0 or cpu")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--return-timestamps", action="store_true")
    parser.add_argument(
        "--verify-generate",
        action="store_true",
        help="Require exact tokens from Hugging Face generate() before returning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    transcriber = SpeculativeWhisperTranscriber.from_pretrained(
        target_model_name=args.target_model,
        draft_model_name=args.draft_model,
        language=args.language,
        task=args.task,
        draft_k=args.draft_k,
        return_timestamps=args.return_timestamps,
        device=args.device,
        dtype=dtype,
    )
    result = transcriber.transcribe_file(
        args.audio,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        verify_generate=args.verify_generate,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
