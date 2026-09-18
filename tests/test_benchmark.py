from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from specvoice.benchmark import _decode_audio


def test_decode_audio_bytes_without_torchcodec():
    source = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    encoded = BytesIO()
    sf.write(encoded, source, 16_000, format="WAV", subtype="FLOAT")

    decoded = _decode_audio({"bytes": encoded.getvalue(), "path": None})

    assert decoded["sampling_rate"] == 16_000
    np.testing.assert_allclose(decoded["array"], source)


def test_decode_audio_requires_bytes_or_path():
    with pytest.raises(ValueError, match="neither bytes nor a path"):
        _decode_audio({"bytes": None, "path": None})
