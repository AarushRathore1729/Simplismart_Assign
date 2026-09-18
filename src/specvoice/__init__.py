"""SpecVoice: low-latency voice-agent components."""

from .asr.decoder import DecodeResult, DecodeStats, GreedyDecoder, SpeculativeGreedyDecoder

__all__ = [
    "DecodeResult",
    "DecodeStats",
    "GreedyDecoder",
    "SpeculativeGreedyDecoder",
]

__version__ = "0.1.0"

