"""Audit recorded benchmark counts, token digests and derived metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .compare import summarize


def audit_report(report):
    if report.get("status") != "complete" or not report.get("all_exact_token_matches"):
        raise ValueError("Only completed, token-equivalent reports can be published")
    samples = report.get("samples", [])
    if not samples or len(samples) != report.get("sample_count"):
        raise ValueError("Sample count does not match the recorded samples")
    ids = [sample["id"] for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    modes = report["config"]["modes"]
    repetitions = report["config"]["repetitions"]
    digest_checks = 0
    for sample in samples:
        if not sample.get("exact_token_match") or not 0 < sample["audio_seconds"] <= 30:
            raise ValueError("Invalid sample duration or token-match status")
        if any(sample["predictions"][mode] != sample["predictions"]["baseline"] for mode in modes):
            raise ValueError("Transcript disagreement")
        for mode in modes:
            runs = sample["runs"][mode]
            if len(runs) != repetitions:
                raise ValueError("Repetition count mismatch")
            for index, row in enumerate(runs):
                if row["repetition"] != index or row["request_seconds"] <= 0:
                    raise ValueError("Invalid run index or latency")
                component_sum = sum(
                    row[key]
                    for key in (
                        "target_encoder_seconds",
                        "draft_encoder_seconds",
                        "decoder_seconds",
                    )
                )
                if not math.isclose(component_sum, row["inference_seconds"], abs_tol=1e-8):
                    raise ValueError("Component latency sum mismatch")
                stats = row["stats"]
                if not 0 <= stats["accepted_draft_tokens"] <= stats["draft_tokens"]:
                    raise ValueError("Invalid accepted draft-token count")
                if row["length_limit_reached"]:
                    raise ValueError("Truncated output: rerun with a larger token budget")
                digest = row.get("token_sha256")
                baseline_digest = sample["runs"]["baseline"][index].get("token_sha256")
                if digest is not None and baseline_digest is not None:
                    if digest != baseline_digest:
                        raise ValueError("Token digest mismatch")
                    if mode != "baseline":
                        digest_checks += 1
    computed = summarize(samples, modes)
    for mode in modes:
        for field, expected in computed[mode].items():
            actual = report["summary"][mode][field]
            if expected is None:
                if actual is not None:
                    raise ValueError(f"Summary mismatch: {mode}.{field}")
            elif not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"Summary mismatch: {mode}.{field}")
    return {
        "status": "passed",
        "samples": len(samples),
        "repetitions": repetitions,
        "modes": modes,
        "summary_recomputed": True,
        "token_digest_checks": digest_checks,
        "exact_match_evidence": "per-run digests"
        if digest_checks
        else "source benchmark gate; original report predates token digests",
        "source_commit": report["environment"]["git_commit"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = audit_report(json.loads(raw))
    result["report_sha256"] = hashlib.sha256(raw).hexdigest()
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
