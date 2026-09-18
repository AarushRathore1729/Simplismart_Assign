"""SpecVoice: low-latency voice-agent components."""

from .asr.decoder import DecodeResult, DecodeStats, GreedyDecoder, SpeculativeGreedyDecoder
from .asr.transcriber import SpeculativeWhisperTranscriber, TranscriptionResult

__all__ = [
    "DecodeResult",
    "DecodeStats",
    "GreedyDecoder",
    "SpeculativeGreedyDecoder",
    "SpeculativeWhisperTranscriber",
    "TranscriptionResult",
]

__version__ = "0.2.0"
