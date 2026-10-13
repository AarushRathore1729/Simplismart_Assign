"""Build the static showcase from the committed, audited GPU report."""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "benchmarks" / "t4_20261013_final"
DESTINATION = HERE / "dist"


def main():
    report = json.loads((SOURCE / "expanded.json").read_text(encoding="utf-8"))
    data = {"summary": report["summary"], "samples": []}
    for sample in report["samples"]:
        item = {
            key: sample[key]
            for key in ("id", "audio_seconds", "predictions", "reference", "exact_token_match")
        }
        item["runs"] = {
            mode: [
                {
                    "request_ms": run["request_seconds"] * 1000,
                    "decoder_ms": run["decoder_seconds"] * 1000,
                    "token_sha256": run["token_sha256"],
                    "target_calls": run["stats"]["target_calls"],
                    "draft_calls": run["stats"]["draft_calls"],
                    "accepted": run["stats"]["accepted_draft_tokens"],
                    "draft_tokens": run["stats"]["draft_tokens"],
                }
                for run in runs
            ]
            for mode, runs in sample["runs"].items()
        }
        data["samples"].append(item)
    DESTINATION.mkdir(exist_ok=True)
    for filename in ("index.html", "styles.css", "app.js"):
        shutil.copy2(HERE / filename, DESTINATION / filename)
    (DESTINATION / "data.json").write_text(
        json.dumps(data, separators=(",", ":")), encoding="utf-8"
    )
    shutil.copy2(SOURCE / "expanded.json", DESTINATION / "benchmark.json")
    shutil.copy2(SOURCE / "profile_comparison.json", DESTINATION / "profile.json")
    print(f"Built {len(data['samples'])}-utterance showcase in {DESTINATION}")


if __name__ == "__main__":
    main()
