"""SpecVoice: Whisper draft-verify inference and GPU benchmarking."""

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

__version__ = "0.3.0"
