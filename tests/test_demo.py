import numpy as np
import pytest
import soundfile as sf
from test_transcriber import make_transcriber

from specvoice.demo import build_demo

pytest.importorskip("gradio")


def test_upload_demo_has_four_path_metrics_and_download(tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(path, np.zeros(800, dtype=np.float32), 16000)
    demo = build_demo(make_transcriber())
    callback = next(block.fn for block in demo.fns.values() if block.fn.__name__ == "compare")
    baseline, adaptive, metrics, report, result_path = callback(str(path), 128)
    assert baseline == adaptive == "2 3 4"
    assert "Exact token agreement" in metrics
    assert set(report["summary"]) == {"baseline", "speculative", "optimized", "adaptive"}
    assert result_path.endswith("comparison.json")
    demo.close()


def test_upload_demo_requires_audio():
    demo = build_demo(make_transcriber())
    callback = next(block.fn for block in demo.fns.values() if block.fn.__name__ == "compare")
    with pytest.raises(Exception, match="Upload a speech clip"):
        callback(None, 128)
    demo.close()
