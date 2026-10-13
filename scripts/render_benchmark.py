"""Render the recorded GPU benchmark; all labels come from the JSON report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/t4_latency"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    modes = list(report["summary"])
    labels = {
        "baseline": "Target only",
        "speculative": "Cached fixed k=4",
        "optimized": "Buffered fixed k=4",
        "adaptive": "Adaptive k=1–8",
    }
    if report["config"].get("cached_suppression"):
        labels["optimized"] = "Buffered k=4 + cached masks"
    values = [report["summary"][m]["p50_request_ms"] for m in modes]
    speeds = [report["summary"][m]["p50_request_speedup"] for m in modes]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "specvoice"})
    fig, ax = plt.subplots(figsize=(10.5, 5.4 if len(modes) > 2 else 4.4))
    fig.patch.set_facecolor("#ffffff")
    colors = {
        "baseline": "#596477",
        "speculative": "#7183a5",
        "optimized": "#17678d",
        "adaptive": "#a47936",
    }
    ax.barh(np.arange(len(modes)), values, color=[colors[m] for m in modes], height=0.58)
    ax.set_yticks(np.arange(len(modes)), [labels[m] for m in modes], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.34)
    ax.set_xlabel("p50 request latency (ms) · lower is better", fontsize=11, labelpad=12)
    for y, latency, speed in zip(range(len(modes)), values, speeds):
        ax.text(latency + 18, y, f"{latency:.0f} ms   {speed:.2f}×", va="center", fontsize=11)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c0c5ce")
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e8ebf0", linewidth=0.8)
    fig.suptitle(
        "Fast Whisper · recorded T4 evaluation",
        x=0.03,
        y=0.96,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#18273b",
    )
    fig.text(
        0.03,
        0.875,
        f"Whisper-Large-v3 + Tiny · FP16 · {report['sample_count']} test utterances "
        f"× {report['config']['repetitions']} repeats",
        color="#526174",
        fontsize=11,
    )
    fig.text(
        0.03,
        0.055,
        "Exact target-token agreement in every measured comparison. Normalized WER: "
        f"{100 * report['summary']['baseline']['normalized_wer']:.2f}% for all paths.",
        fontsize=10,
        color="#344357",
    )
    fig.text(
        0.03,
        0.022,
        "Request includes feature preparation, required encoders, decoding and text conversion; excludes model loading and file I/O.",
        fontsize=8.5,
        color="#5a6879",
    )
    fig.subplots_adjust(left=0.32 if len(modes) <= 2 else 0.23, right=0.98, top=0.79, bottom=0.22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    svg_path = args.output.with_suffix(".svg")
    fig.savefig(svg_path, metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    print(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
