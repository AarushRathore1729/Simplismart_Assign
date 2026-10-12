"""Compare target-only, existing fixed, buffered fixed and adaptive inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .asr.transcriber import SpeculativeWhisperTranscriber, load_audio, resample_audio
from .benchmark import _decode_audio

MODES = ("baseline", "speculative", "optimized", "adaptive")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="openai/whisper-large-v3")
    parser.add_argument("--draft-model", default="openai/whisper-tiny")
    parser.add_argument("--audio", nargs="+", type=Path, help="Local clips; omit for LibriSpeech")
    parser.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--dataset-config", default="clean")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--sample-strategy", choices=("first", "evenly-spaced"), default="first")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--max-draft-k", type=int, default=8)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument(
        "--verify-generate", action="store_true", help="Extra untimed HF oracle check"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Export a separate instrumented trace"
    )
    parser.add_argument("--profile-mode", choices=("optimized", "adaptive"), default="optimized")
    parser.add_argument("--cached-suppression", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/comparison.json"))
    args = parser.parse_args(argv)
    if args.max_samples < 1 or args.repetitions < 1 or args.warmup_runs < 0:
        parser.error("Use positive max-samples/repetitions and nonnegative warmup-runs")
    if "baseline" not in args.modes:
        parser.error("Include baseline: it is the required token-equivalence oracle")
    args.modes = list(dict.fromkeys(args.modes))
    return args


def environment(device: torch.device) -> dict[str, Any]:
    import transformers

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_capability": list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


def load_samples(args):
    if args.audio:
        samples = []
        for path in args.audio[: args.max_samples]:
            audio, rate = load_audio(path)
            samples.append({"id": str(path), "array": audio, "rate": rate, "reference": None})
        return samples, {"source": "local_audio"}

    from datasets import Audio, load_dataset

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    fingerprint = dataset._fingerprint
    if args.sample_strategy == "first":
        indices = list(range(min(args.max_samples, len(dataset))))
    else:
        indices = evenly_spaced_indices(len(dataset), args.max_samples)
    samples = []
    skipped = []
    for index in indices:
        row = dataset[index]
        audio = _decode_audio(row["audio"])
        if len(audio["array"]) / audio["sampling_rate"] > 30:
            skipped.append(str(row.get("id", index)))
            continue
        # Resample once so the quality evaluation has a consistent 16kHz input.
        array = resample_audio(audio["array"], audio["sampling_rate"])
        samples.append(
            {
                "id": str(row.get("id", index)),
                "array": array,
                "rate": 16000,
                "reference": row["text"].strip(),
            }
        )
    return samples, {
        "source": args.dataset,
        "fingerprint": fingerprint,
        "selection_indices": indices,
        "skipped_over_30s": skipped,
    }


def evenly_spaced_indices(size: int, limit: int) -> list[int]:
    """Deterministic wider coverage; no duplicated input indices."""
    if size < 0 or limit < 1:
        raise ValueError("size must be nonnegative and limit positive")
    if size == 0:
        return []
    return np.linspace(0, size - 1, min(size, limit), dtype=int).tolist()


def measure(engine, sample, mode, max_new_tokens):
    device = engine.device
    resident = peak = extra = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        resident = torch.cuda.memory_allocated(device) / 2**20
        torch.cuda.reset_peak_memory_stats(device)
    result = engine.transcribe_array(
        sample["array"], sample["rate"], mode=mode, max_new_tokens=max_new_tokens
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        extra = max(0.0, peak - resident)
    row = {
        "request_seconds": result.request_seconds,
        "inference_seconds": result.total_seconds,
        "preprocessing_seconds": result.preprocessing_seconds,
        "target_encoder_seconds": result.target_encoder_seconds,
        "draft_encoder_seconds": result.draft_encoder_seconds,
        "decoder_seconds": result.decoder_seconds,
        "resident_cuda_mib": resident,
        "peak_cuda_mib": peak,
        "extra_peak_cuda_mib": extra,
        "stats": asdict(result.decode_stats),
        "acceptance_rate": result.decode_stats.acceptance_rate,
        "token_sha256": hashlib.sha256(
            np.asarray(result.token_ids, dtype="<i8").tobytes()
        ).hexdigest(),
        "length_limit_reached": (
            result.decode_stats.generated_tokens == max_new_tokens
            and result.token_ids[-1] != engine.target_model.config.eos_token_id
        ),
    }
    return result, row


def summarize(samples, modes):
    """Latency quantiles are across per-utterance repetition medians."""
    from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, wer

    normalize = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    summary = {}
    for mode in modes:
        latencies = [
            statistics.median(r["request_seconds"] for r in s["runs"][mode]) for s in samples
        ]
        inference = [
            statistics.median(r["inference_seconds"] for r in s["runs"][mode]) for s in samples
        ]
        decoders = [
            statistics.median(r["decoder_seconds"] for r in s["runs"][mode]) for s in samples
        ]
        runs = [r for s in samples for r in s["runs"][mode]]
        elapsed = sum(r["request_seconds"] for r in runs)
        proposed = sum(r["stats"]["draft_tokens"] for r in runs)
        accepted = sum(r["stats"]["accepted_draft_tokens"] for r in runs)
        peaks = [r["peak_cuda_mib"] for r in runs if r["peak_cuda_mib"] is not None]
        extra = [r["extra_peak_cuda_mib"] for r in runs if r["extra_peak_cuda_mib"] is not None]
        references = [s["reference"] for s in samples]
        quality = None
        if all(reference is not None for reference in references):
            quality = wer(
                [normalize(r) for r in references],
                [normalize(s["predictions"][mode]) for s in samples],
            )
        summary[mode] = {
            "p50_request_ms": float(np.percentile(latencies, 50) * 1000),
            "p95_request_ms": float(np.percentile(latencies, 95) * 1000),
            "p50_inference_ms": float(np.percentile(inference, 50) * 1000),
            "p50_decoder_ms": float(np.percentile(decoders, 50) * 1000),
            "generated_tokens_per_second": sum(r["stats"]["generated_tokens"] for r in runs)
            / elapsed,
            "audio_seconds_per_second": sum(
                s["audio_seconds"] * len(s["runs"][mode]) for s in samples
            )
            / elapsed,
            "acceptance_rate": accepted / proposed if proposed else None,
            "target_calls_per_run": statistics.mean(r["stats"]["target_calls"] for r in runs),
            "draft_calls_per_run": statistics.mean(r["stats"]["draft_calls"] for r in runs),
            "fallback_tokens_total": sum(r["stats"]["fallback_tokens"] for r in runs),
            "truncated_runs": sum(r["length_limit_reached"] for r in runs),
            "max_peak_cuda_mib": max(peaks) if peaks else None,
            "max_extra_peak_cuda_mib": max(extra) if extra else None,
            "normalized_wer": quality,
        }
    baseline = summary["baseline"]
    for mode, values in summary.items():
        values["p50_request_speedup"] = baseline["p50_request_ms"] / values["p50_request_ms"]
        values["p50_decoder_speedup"] = baseline["p50_decoder_ms"] / values["p50_decoder_ms"]
    return summary


def markdown_table(summary):
    lines = [
        "| Mode | p50 request (ms) | p95 request (ms) | Speedup | Acceptance | WER |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, row in summary.items():
        acceptance = "—" if row["acceptance_rate"] is None else f"{row['acceptance_rate']:.1%}"
        quality = "—" if row["normalized_wer"] is None else f"{row['normalized_wer']:.4f}"
        lines.append(
            f"| {mode} | {row['p50_request_ms']:.1f} | {row['p95_request_ms']:.1f} | "
            f"{row['p50_request_speedup']:.2f}× | {acceptance} | {quality} |"
        )
    return "\n".join(lines)


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def profile_one(engine, sample, mode, max_new_tokens, output):
    decoder = engine.adaptive_decoder if mode == "adaptive" else engine.optimized_decoder
    original = decoder.profile_stages
    decoder.profile_stages = True
    activities = [torch.profiler.ProfilerActivity.CPU]
    if engine.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True
        ) as prof:
            result = engine.transcribe_array(
                sample["array"], sample["rate"], mode=mode, max_new_tokens=max_new_tokens
            )
        prof.export_chrome_trace(str(output.with_suffix(".trace.json")))
        output.with_suffix(".profile.txt").write_text(
            prof.key_averages().table(sort_by="cpu_time_total", row_limit=40), encoding="utf-8"
        )
        write_report(output.with_suffix(".stages.json"), result.to_dict())
    finally:
        decoder.profile_stages = original


def run_comparison(
    engine,
    samples,
    *,
    modes=MODES,
    repetitions=3,
    warmup_runs=1,
    max_new_tokens=128,
    seed=42,
    report=None,
    output=None,
):
    if not samples:
        raise ValueError("No audio samples to benchmark")
    for _ in range(warmup_runs):
        for mode in modes:
            measure(engine, samples[0], mode, max_new_tokens)
    rng = random.Random(seed)
    completed = []
    report = report if report is not None else {}
    for index, sample in enumerate(samples):
        runs = {mode: [] for mode in modes}
        predictions = {}
        for repetition in range(repetitions):
            order = list(modes)
            rng.shuffle(order)
            results = {}
            for mode in order:
                result, row = measure(engine, sample, mode, max_new_tokens)
                row["repetition"] = repetition
                row["measurement_order"] = order
                runs[mode].append(row)
                results[mode] = result
                predictions[mode] = result.text
            for mode in modes:
                if results[mode].token_ids != results["baseline"].token_ids:
                    report.update(
                        status="token_mismatch",
                        samples=completed,
                        mismatch={
                            "sample_id": sample["id"],
                            "repetition": repetition,
                            "mode": mode,
                            "baseline_tokens": results["baseline"].token_ids,
                            "variant_tokens": results[mode].token_ids,
                        },
                    )
                    if output:
                        write_report(output, report)
                    raise RuntimeError(
                        f"Token mismatch in {mode} on {sample['id']}; no speedup reported"
                    )
        completed.append(
            {
                "id": sample["id"],
                "audio_seconds": len(sample["array"]) / sample["rate"],
                "reference": sample.get("reference"),
                "predictions": predictions,
                "exact_token_match": True,
                "runs": runs,
            }
        )
        report.update(status="in_progress", samples=completed)
        if output:
            write_report(output, report)
        print(f"[{index + 1}/{len(samples)}] token agreement passed: {sample['id']}", flush=True)
    report.update(
        status="complete",
        all_exact_token_matches=True,
        sample_count=len(completed),
        summary=summarize(completed, modes),
    )
    if output:
        write_report(output, report)
        output.with_suffix(".md").write_text(
            markdown_table(report["summary"]) + "\n", encoding="utf-8"
        )
    return report


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    engine = SpeculativeWhisperTranscriber.from_pretrained(
        target_model_name=args.target_model,
        draft_model_name=args.draft_model,
        draft_k=args.draft_k,
        max_draft_k=args.max_draft_k,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attention,
        cached_suppression=args.cached_suppression,
    )
    samples, source = load_samples(args)
    print(f"Device: {engine.device}; samples: {len(samples)}; modes: {args.modes}", flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["audio"] = [str(p) for p in args.audio] if args.audio else None
    config["resolved_dtype"] = str(engine.dtype)
    report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(engine.device),
        "config": config,
        "source": source,
        "model_revisions": {
            "target": getattr(engine.target_model.config, "_commit_hash", None),
            "draft": getattr(engine.draft_model.config, "_commit_hash", None),
        },
        "methodology": {
            "request_latency": "resampling + feature preparation + encoders + decoding + text conversion; excludes model loading, file I/O and oracle checks",
            "inference_latency": "target encoder + draft encoder when used + decoder; includes synchronization at stage boundaries",
            "quantiles": "across per-utterance medians of repeated measurements; descriptive on small samples",
            "throughput": "serial batch-size-1 replay; generated tokens include EOS; not concurrent serving throughput",
            "memory": "all paths have both models resident; extra peak is above resident allocated CUDA memory",
            "quality": "exact tokens vs shared target greedy policy; WER lowercases and removes punctuation",
            "profiling": "separate instrumented run, excluded from headline benchmark",
        },
    }
    if args.verify_generate:
        for sample in samples:
            engine.transcribe_array(
                sample["array"],
                sample["rate"],
                mode="baseline",
                max_new_tokens=args.max_new_tokens,
                verify_generate=True,
            )
        report["hf_generate_oracle_passed"] = True
    report = run_comparison(
        engine,
        samples,
        modes=args.modes,
        repetitions=args.repetitions,
        warmup_runs=args.warmup_runs,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        report=report,
        output=args.output,
    )
    print(markdown_table(report["summary"]))
    print(f"Saved {args.output}", flush=True)
    if args.profile:
        profile_one(engine, samples[0], args.profile_mode, args.max_new_tokens, args.output)
        print(f"Saved trace and stage timings beside {args.output}", flush=True)


if __name__ == "__main__":
    main()
