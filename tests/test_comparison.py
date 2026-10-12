import json
from dataclasses import replace

import numpy as np
import pytest
from test_transcriber import make_transcriber

from specvoice.compare import parse_args, run_comparison


def test_comparison_reports_all_four_paths():
    engine = make_transcriber()
    sample = {
        "id": "toy",
        "array": np.zeros(800, dtype=np.float32),
        "rate": 16000,
        "reference": "2 3 4",
    }
    report = run_comparison(engine, [sample], repetitions=2, warmup_runs=0)
    assert report["status"] == "complete"
    assert report["all_exact_token_matches"]
    assert set(report["summary"]) == {"baseline", "speculative", "optimized", "adaptive"}
    assert all(row["normalized_wer"] == 0 for row in report["summary"].values())


def test_mismatch_saves_evidence_and_refuses_speedup(tmp_path):
    engine = make_transcriber()
    original = engine.transcribe_array

    def wrong(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["mode"] == "optimized":
            return replace(result, token_ids=(1, 9, 0))
        return result

    engine.transcribe_array = wrong
    output = tmp_path / "result.json"
    sample = {
        "id": "toy",
        "array": np.zeros(800, dtype=np.float32),
        "rate": 16000,
        "reference": "2 3 4",
    }
    with pytest.raises(RuntimeError, match="no speedup"):
        run_comparison(
            engine,
            [sample],
            modes=("baseline", "optimized"),
            repetitions=1,
            warmup_runs=0,
            output=output,
        )
    report = json.loads(output.read_text())
    assert report["status"] == "token_mismatch"
    assert "summary" not in report


def test_cli_requires_baseline_and_positive_sample_count():
    with pytest.raises(SystemExit):
        parse_args(["--modes", "adaptive"])
    with pytest.raises(SystemExit):
        parse_args(["--max-samples", "0"])
