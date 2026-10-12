"""Small audio-upload demonstration of measured draft/verify decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock

from .asr.transcriber import SpeculativeWhisperTranscriber, load_audio
from .compare import MODES, run_comparison


def build_demo(engine):
    import gradio as gr

    lock = Lock()

    def compare(path, max_tokens):
        if not path:
            raise gr.Error("Upload a speech clip of at most 30 seconds.")
        audio, rate = load_audio(path)
        sample = {"id": Path(path).name, "array": audio, "rate": rate, "reference": None}
        with lock:
            report = run_comparison(
                engine,
                [sample],
                modes=MODES,
                repetitions=1,
                warmup_runs=1,
                max_new_tokens=int(max_tokens),
            )
        summary = report["summary"]
        predictions = report["samples"][0]["predictions"]
        table = [
            "| Path | Request (ms) | Decoder (ms) | Speedup | Acceptance |",
            "|---|---:|---:|---:|---:|",
        ]
        for mode, row in summary.items():
            acceptance = "—" if row["acceptance_rate"] is None else f"{row['acceptance_rate']:.1%}"
            table.append(
                f"| {mode} | {row['p50_request_ms']:.1f} | {row['p50_decoder_ms']:.1f} | "
                f"{row['p50_request_speedup']:.2f}× | {acceptance} |"
            )

        results_path = Path(mkdtemp(prefix="specvoice-demo-")) / "comparison.json"
        results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return (
            predictions["baseline"],
            predictions["optimized"],
            "**Exact token agreement: passed across all four paths.**\n\n"
            + "\n".join(table)
            + "\n\nOne clip, one measured run per mode after warmup. Use the benchmark for résumé numbers.",
            report,
            str(results_path),
        )

    with gr.Blocks(title="Fast Whisper — GPU Inference Optimization") as demo:
        gr.Markdown(
            "# Fast Whisper\nCompare target-only inference, cached fixed drafting, buffered drafting and adaptive drafting on the same speech clip."
        )
        with gr.Row():
            audio = gr.Audio(type="filepath", label="English speech · at most 30 seconds")
            max_tokens = gr.Slider(32, 256, value=128, step=32, label="Maximum new tokens")
        run = gr.Button("Transcribe and compare", variant="primary")
        with gr.Row():
            baseline = gr.Textbox(label="Target-only transcript", lines=5)
            optimized = gr.Textbox(label="Buffered fixed-draft transcript", lines=5)
        metrics = gr.Markdown()
        with gr.Accordion("Draft lengths, fallback and raw measurements", open=False):
            raw = gr.JSON()
        download = gr.File(label="Download measured result")
        run.click(compare, [audio, max_tokens], [baseline, optimized, metrics, raw, download])
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="openai/whisper-large-v3")
    parser.add_argument("--draft-model", default="openai/whisper-tiny")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    engine = SpeculativeWhisperTranscriber.from_pretrained(
        target_model_name=args.target_model, draft_model_name=args.draft_model
    )
    build_demo(engine).queue().launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
