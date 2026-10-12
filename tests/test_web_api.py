from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from test_transcriber import make_transcriber

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from specvoice.web_api import create_app


def test_upload_api_returns_verified_engine_transcript():
    encoded = BytesIO()
    sf.write(encoded, np.zeros(800, dtype=np.float32), 16000, format="WAV")
    with TestClient(create_app(make_transcriber())) as client:
        assert client.get("/health").json()["ready"]
        response = client.post("/transcribe", files={"file": ("speech.wav", encoded.getvalue())})
        assert response.status_code == 200
        result = response.json()
        assert result["text"] == "2 3 4"
        assert result["mode"] == "optimized"
        assert result["decode_stats"]["generated_tokens"] == 4


@pytest.mark.parametrize("audio", [b"", b"not audio"])
def test_upload_api_rejects_invalid_audio(audio):
    with TestClient(create_app(make_transcriber())) as client:
        response = client.post("/transcribe", files={"file": ("bad.wav", audio)})
        assert response.status_code == 422
