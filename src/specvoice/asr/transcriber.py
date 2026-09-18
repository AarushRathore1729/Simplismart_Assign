"""End-to-end short-form Whisper transcription."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import soundfile as sf
import torch

from .decoder import DecodeStats, GreedyDecoder, SpeculativeGreedyDecoder
from .whisper import build_whisper_policy, validate_whisper_pair

DecoderMode = Literal["baseline", "speculative"]


@dataclass(frozen=True)
class TranscriptionResult:
    """Text, exact token sequence and component-level timings."""

    text: str
    token_ids: tuple[int, ...]
    mode: DecoderMode
    target_encoder_seconds: float
    draft_encoder_seconds: float
    decoder_seconds: float
    total_seconds: float
    decode_stats: DecodeStats
    generate_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        return payload


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file and downmix it to mono float32."""

    audio, sampling_rate = sf.read(Path(path), dtype="float32", always_2d=False)
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Audio must contain at least one mono or stereo sample")
    return array, int(sampling_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int = 16_000) -> np.ndarray:
    """Deterministically resample mono audio without requiring torchaudio."""

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sampling rates must be positive")
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Expected a non-empty mono waveform")
    if source_rate == target_rate:
        return values

    output_length = max(1, round(values.size * target_rate / source_rate))
    source_positions = np.arange(values.size, dtype=np.float64)
    target_positions = np.arange(output_length, dtype=np.float64) * source_rate / target_rate
    target_positions = np.minimum(target_positions, values.size - 1)
    return np.interp(target_positions, source_positions, values).astype(np.float32)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function: Any, *args: Any) -> tuple[Any, float]:
    _synchronize(device)
    start = perf_counter()
    result = function(*args)
    _synchronize(device)
    return result, perf_counter() - start


def _move_inputs(
    inputs: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, dtype=dtype if value.dtype.is_floating_point else value.dtype)
        for key, value in inputs.items()
    }


def _encode(model: Any, inputs: dict[str, torch.Tensor]) -> Any:
    return model.get_encoder()(**inputs)


def _trim_padding(tokens: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    if pad_token_id is None:
        return tokens
    values = tokens[0].tolist()
    while values and values[-1] == pad_token_id:
        values.pop()
    return torch.tensor([values], device=tokens.device, dtype=tokens.dtype)


class SpeculativeWhisperTranscriber:
    """Run baseline or correctness-equivalent speculative Whisper decoding."""

    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        target_processor: Any,
        draft_processor: Any,
        *,
        language: str = "en",
        task: str = "transcribe",
        draft_k: int = 4,
        return_timestamps: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.target_processor = target_processor
        self.draft_processor = draft_processor
        self.language = language
        self.task = task
        self.return_timestamps = return_timestamps
        self.device = torch.device(device or getattr(target_model, "device", "cpu"))
        self.dtype = dtype or getattr(target_model, "dtype", torch.float32)
        self.vocabulary_map = validate_whisper_pair(
            target_model,
            draft_model,
            target_processor,
            draft_processor,
            device=self.device,
        )
        self.policy = build_whisper_policy(
            target_processor,
            target_model,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            device=self.device,
        )
        self.draft_policy = build_whisper_policy(
            draft_processor,
            draft_model,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            device=self.device,
        )
        mapped_prefix = self.vocabulary_map.target_to_draft(self.policy.prefix_ids)
        if not torch.equal(mapped_prefix, self.draft_policy.prefix_ids):
            raise ValueError("Draft and target Whisper prompts are not semantically equivalent")
        eos_token_id = int(target_model.config.eos_token_id)
        self.baseline_decoder = GreedyDecoder(
            target_model,
            eos_token_id,
            logits_processor=self.policy.logits_processor,
        )
        self.speculative_decoder = SpeculativeGreedyDecoder(
            target_model,
            draft_model,
            eos_token_id,
            draft_k=draft_k,
            logits_processor=self.policy.logits_processor,
            draft_logits_processor=self.draft_policy.logits_processor,
            target_to_draft=self.vocabulary_map.target_to_draft,
            draft_to_target=self.vocabulary_map.draft_to_target,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        target_model_name: str = "openai/whisper-large-v3",
        draft_model_name: str = "openai/whisper-tiny",
        language: str = "en",
        task: str = "transcribe",
        draft_k: int = 4,
        return_timestamps: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SpeculativeWhisperTranscriber:
        """Load a compatible target/draft pair from Hugging Face."""

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        resolved_dtype = dtype or (
            torch.float16 if resolved_device.type == "cuda" else torch.float32
        )
        model_kwargs = {
            "torch_dtype": resolved_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        target_processor = AutoProcessor.from_pretrained(target_model_name)
        draft_processor = AutoProcessor.from_pretrained(draft_model_name)
        target_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            target_model_name, **model_kwargs
        ).to(resolved_device).eval()
        draft_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            draft_model_name, **model_kwargs
        ).to(resolved_device).eval()
        return cls(
            target_model,
            draft_model,
            target_processor,
            draft_processor,
            language=language,
            task=task,
            draft_k=draft_k,
            return_timestamps=return_timestamps,
            device=resolved_device,
            dtype=resolved_dtype,
        )

    def _prepare(self, processor: Any, audio: np.ndarray, sampling_rate: int):
        inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        return _move_inputs(dict(inputs), self.device, self.dtype)

    @torch.inference_mode()
    def transcribe_array(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        *,
        mode: DecoderMode = "speculative",
        max_new_tokens: int = 128,
        verify_generate: bool = False,
    ) -> TranscriptionResult:
        """Transcribe an in-memory waveform of at most 30 seconds."""

        if mode not in ("baseline", "speculative"):
            raise ValueError("mode must be 'baseline' or 'speculative'")
        waveform = resample_audio(audio, sampling_rate, 16_000)
        target_inputs = self._prepare(self.target_processor, waveform, 16_000)
        target_encoder, target_encoder_seconds = _timed(
            self.device, _encode, self.target_model, target_inputs
        )

        draft_encoder_seconds = 0.0
        if mode == "baseline":
            decoded, decoder_seconds = _timed(
                self.device,
                self.baseline_decoder.decode,
                target_encoder,
                self.policy.prefix_ids,
                max_new_tokens,
            )
        else:
            draft_inputs = self._prepare(self.draft_processor, waveform, 16_000)
            draft_encoder, draft_encoder_seconds = _timed(
                self.device, _encode, self.draft_model, draft_inputs
            )
            decoded, decoder_seconds = _timed(
                self.device,
                self.speculative_decoder.decode,
                target_encoder,
                draft_encoder,
                self.policy.prefix_ids,
                max_new_tokens,
            )

        generate_verified = False
        if verify_generate:
            reference = self.target_model.generate(
                **target_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                language=self.language,
                task=self.task,
                return_timestamps=self.return_timestamps,
                force_unique_generate_call=True,
            )
            reference_tokens = reference.sequences if hasattr(reference, "sequences") else reference
            generated_tokens = decoded.token_ids[:, self.policy.begin_index :]
            pad_token_id = getattr(self.target_model.generation_config, "pad_token_id", None)
            reference_tokens = _trim_padding(reference_tokens, pad_token_id)
            generated_tokens = _trim_padding(generated_tokens, pad_token_id)
            if not torch.equal(reference_tokens, generated_tokens):
                raise RuntimeError(
                    "Custom greedy decoding differs from transformers.generate(); "
                    "refusing to report this transcription as verified"
                )
            generate_verified = True

        text = self.target_processor.batch_decode(
            decoded.token_ids, skip_special_tokens=True
        )[0].strip()
        return TranscriptionResult(
            text=text,
            token_ids=tuple(int(token) for token in decoded.token_ids[0].tolist()),
            mode=mode,
            target_encoder_seconds=target_encoder_seconds,
            draft_encoder_seconds=draft_encoder_seconds,
            decoder_seconds=decoder_seconds,
            total_seconds=target_encoder_seconds + draft_encoder_seconds + decoder_seconds,
            decode_stats=decoded.stats,
            generate_verified=generate_verified,
        )

    def transcribe_file(
        self,
        path: str | Path,
        *,
        mode: DecoderMode = "speculative",
        max_new_tokens: int = 128,
        verify_generate: bool = False,
    ) -> TranscriptionResult:
        audio, sampling_rate = load_audio(path)
        return self.transcribe_array(
            audio,
            sampling_rate,
            mode=mode,
            max_new_tokens=max_new_tokens,
            verify_generate=verify_generate,
        )
