"""Reproducible offline benchmark for the SpecVoice foundation.

This command intentionally benchmarks greedy decoding first. Sampling-based
speculation is excluded until its rejection correction is implemented and
distribution-level tests are available.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .asr.decoder import GreedyDecoder, SpeculativeGreedyDecoder
from .asr.whisper import build_whisper_policy, validate_whisper_pair


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    synchronize(device)
    start = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize(device)
    return result, time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark greedy speculative Whisper decoding")
    parser.add_argument("--target-model", default="openai/whisper-large-v3")
    parser.add_argument("--draft-model", default="openai/whisper-tiny")
    parser.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--dataset-config", default="clean")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/foundation.json"))
    return parser.parse_args()


def _encode(model: Any, model_inputs: dict[str, torch.Tensor]) -> Any:
    return model.get_encoder()(**model_inputs)


def _to_device(inputs: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype):
    return {
        key: value.to(device, dtype=dtype if value.dtype.is_floating_point else value.dtype)
        for key, value in inputs.items()
    }


def main() -> None:
    # Optional heavyweight imports stay out of unit tests and package import.
    from datasets import load_dataset
    from jiwer import wer
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, __version__

    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    target_processor = AutoProcessor.from_pretrained(args.target_model)
    draft_processor = AutoProcessor.from_pretrained(args.draft_model)
    target_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.target_model, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device).eval()
    draft_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.draft_model, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device).eval()

    validate_whisper_pair(target_model, draft_model, target_processor, draft_processor)

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    policy = build_whisper_policy(
        target_processor,
        target_model,
        language=args.language,
        task=args.task,
        return_timestamps=False,
        device=device,
    )
    prefix = policy.prefix_ids

    baseline_decoder = GreedyDecoder(
        target_model,
        target_model.config.eos_token_id,
        logits_processor=policy.logits_processor,
    )
    speculative_decoder = SpeculativeGreedyDecoder(
        target_model,
        draft_model,
        target_model.config.eos_token_id,
        draft_k=args.draft_k,
        logits_processor=policy.logits_processor,
    )

    if args.warmup_runs < 0:
        raise ValueError("warmup-runs cannot be negative")
    if args.repetitions < 1:
        raise ValueError("repetitions must be at least 1")

    if len(dataset) and args.warmup_runs:
        warmup_audio = dataset[0]["audio"]
        warmup_target_inputs = _to_device(
            target_processor(
                warmup_audio["array"],
                sampling_rate=warmup_audio["sampling_rate"],
                return_tensors="pt",
            ),
            device,
            dtype,
        )
        warmup_draft_inputs = _to_device(
            draft_processor(
                warmup_audio["array"],
                sampling_rate=warmup_audio["sampling_rate"],
                return_tensors="pt",
            ),
            device,
            dtype,
        )
        for _ in range(args.warmup_runs):
            warmup_target_encoder = _encode(target_model, warmup_target_inputs)
            warmup_draft_encoder = _encode(draft_model, warmup_draft_inputs)
            baseline_decoder.decode(warmup_target_encoder, prefix, args.max_new_tokens)
            speculative_decoder.decode(
                warmup_target_encoder,
                warmup_draft_encoder,
                prefix,
                args.max_new_tokens,
            )
        synchronize(device)

    samples = []
    baseline_times = []
    speculative_times = []
    baseline_predictions = []
    speculative_predictions = []
    references = []

    for index, sample in enumerate(dataset):
        audio = sample["audio"]
        target_inputs = _to_device(
            target_processor(
                audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
            ),
            device,
            dtype,
        )
        draft_inputs = _to_device(
            draft_processor(
                audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
            ),
            device,
            dtype,
        )

        repetitions = []
        for repetition in range(args.repetitions):
            target_encoder, target_encoder_seconds = timed_call(
                device, _encode, target_model, target_inputs
            )
            draft_encoder, draft_encoder_seconds = timed_call(
                device, _encode, draft_model, draft_inputs
            )
            baseline, baseline_decode_seconds = timed_call(
                device,
                baseline_decoder.decode,
                target_encoder,
                prefix,
                args.max_new_tokens,
            )
            speculative, speculative_decode_seconds = timed_call(
                device,
                speculative_decoder.decode,
                target_encoder,
                draft_encoder,
                prefix,
                args.max_new_tokens,
            )

            exact_match = torch.equal(baseline.token_ids, speculative.token_ids)
            if not exact_match:
                raise RuntimeError(
                    f"Token mismatch on sample {index}, repetition {repetition}; "
                    "refusing to report speedup"
                )
            repetitions.append(
                {
                    "target_encoder": target_encoder_seconds,
                    "draft_encoder": draft_encoder_seconds,
                    "baseline_decoder": baseline_decode_seconds,
                    "speculative_decoder": speculative_decode_seconds,
                    "baseline_total": target_encoder_seconds + baseline_decode_seconds,
                    "speculative_total": target_encoder_seconds
                    + draft_encoder_seconds
                    + speculative_decode_seconds,
                }
            )

        baseline_text = target_processor.batch_decode(
            baseline.token_ids, skip_special_tokens=True
        )[0].strip()
        speculative_text = target_processor.batch_decode(
            speculative.token_ids, skip_special_tokens=True
        )[0].strip()
        reference = sample["text"].strip()
        median_latency = {
            name: statistics.median(repetition[name] for repetition in repetitions)
            for name in repetitions[0]
        }
        baseline_total = median_latency["baseline_total"]
        speculative_total = median_latency["speculative_total"]
        baseline_times.append(baseline_total)
        speculative_times.append(speculative_total)
        baseline_predictions.append(baseline_text)
        speculative_predictions.append(speculative_text)
        references.append(reference)
        samples.append(
            {
                "index": index,
                "reference": reference,
                "prediction": speculative_text,
                "exact_token_match": exact_match,
                "median_latency_seconds": median_latency,
                "repetitions": repetitions,
                "speculative_stats": asdict(speculative.stats),
                "acceptance_rate": speculative.stats.acceptance_rate,
            }
        )

    baseline_median = statistics.median(baseline_times)
    speculative_median = statistics.median(speculative_times)
    report = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": str(dtype),
        },
        "config": vars(args) | {"output": str(args.output)},
        "summary": {
            "samples": len(samples),
            "all_exact_token_matches": True,
            "baseline_wer": wer(references, baseline_predictions),
            "speculative_wer": wer(references, speculative_predictions),
            "baseline_median_seconds": baseline_median,
            "speculative_median_seconds": speculative_median,
            "median_speedup": baseline_median / speculative_median,
            "mean_acceptance_rate": statistics.mean(
                sample["acceptance_rate"] for sample in samples
            ),
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved full report to {args.output}")


if __name__ == "__main__":
    main()
