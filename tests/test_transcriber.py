from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from specvoice.asr.transcriber import SpeculativeWhisperTranscriber, resample_audio


@dataclass
class ToyCache:
    length: int

    def crop(self, max_length: int) -> None:
        self.length = min(self.length, max_length)


class ToyEncoder:
    def __call__(self, input_features):
        return SimpleNamespace(last_hidden_state=input_features)


class ToySpeechModel:
    def __init__(self, transitions):
        self.transitions = transitions
        self.config = SimpleNamespace(
            vocab_size=16,
            decoder_start_token_id=1,
            eos_token_id=0,
        )
        self.generation_config = SimpleNamespace(
            suppress_tokens=[],
            begin_suppress_tokens=[],
            no_timestamps_token_id=15,
            eos_token_id=0,
            bos_token_id=1,
            pad_token_id=0,
        )
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def get_encoder(self):
        return ToyEncoder()

    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1
        return torch.tensor([[2, 3, 4, 0]], dtype=torch.long)

    def __call__(
        self,
        decoder_input_ids,
        encoder_outputs,
        past_key_values,
        use_cache,
        return_dict,
    ):
        del encoder_outputs, use_cache, return_dict
        logits = torch.full((1, decoder_input_ids.shape[1], 16), -1000.0)
        for position, token in enumerate(decoder_input_ids[0].tolist()):
            logits[0, position, self.transitions[token]] = 0.0
        previous = 0 if past_key_values is None else past_key_values.length
        return SimpleNamespace(
            logits=logits,
            past_key_values=ToyCache(previous + decoder_input_ids.shape[1]),
        )


class ToyProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(get_vocab=lambda: {str(index): index for index in range(16)})

    def get_decoder_prompt_ids(self, language, task, no_timestamps):
        del language, task, no_timestamps
        return []

    def __call__(self, audio, sampling_rate, return_tensors):
        assert sampling_rate == 16_000
        assert return_tensors == "pt"
        return {"input_features": torch.tensor(audio[None, None, :], dtype=torch.float32)}

    def batch_decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens
        values = [token for token in token_ids[0].tolist() if token not in (0, 1)]
        return [" ".join(map(str, values))]


def make_transcriber(draft_transitions=None):
    target = {1: 2, 2: 3, 3: 4, 4: 0, 0: 0, 9: 0}
    draft = draft_transitions or target
    return SpeculativeWhisperTranscriber(
        ToySpeechModel(target),
        ToySpeechModel(draft),
        ToyProcessor(),
        ToyProcessor(),
        draft_k=3,
    )


def test_end_to_end_array_baseline_and_speculative_tokens_match():
    transcriber = make_transcriber({1: 2, 2: 9, 9: 0, 3: 4, 4: 0, 0: 0})
    audio = np.zeros(800, dtype=np.float32)

    baseline = transcriber.transcribe_array(audio, 8_000, mode="baseline")
    speculative = transcriber.transcribe_array(
        audio, 8_000, mode="speculative", verify_generate=True
    )

    assert speculative.token_ids == baseline.token_ids == (1, 2, 3, 4, 0)
    assert speculative.text == baseline.text == "2 3 4"
    assert speculative.generate_verified
    assert 0.0 < speculative.decode_stats.acceptance_rate < 1.0
    assert speculative.draft_encoder_seconds >= 0.0


def test_resample_audio_changes_length_and_validates_inputs():
    audio = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    assert resample_audio(audio, 8_000, 16_000).shape == (16,)
    assert resample_audio(audio, 8_000, 8_000) is audio
    with pytest.raises(ValueError, match="Sampling rates"):
        resample_audio(audio, 0)
