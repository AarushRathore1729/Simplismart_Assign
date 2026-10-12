"""Output and cache correctness tests; no pretrained model downloads."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from specvoice.asr.decoder import GreedyDecoder, SpeculativeGreedyDecoder
from specvoice.asr.optimized import AdaptiveDraftSchedule, BufferedSpeculativeDecoder


@dataclass
class HistoryCache:
    tokens: list[int]

    def crop(self, length):
        self.tokens = self.tokens[:length]


class HistoryModel:
    """Predictions depend on all cached tokens, exposing incorrect rollback."""

    def __init__(self, offset=0, stop_after=25):
        self.offset = offset
        self.stop_after = stop_after
        self.input_lengths = []

    def __call__(self, decoder_input_ids, encoder_outputs, past_key_values, **kwargs):
        history = [] if past_key_values is None else list(past_key_values.tokens)
        self.input_lengths.append(decoder_input_ids.shape[1])
        logits = torch.full((1, decoder_input_ids.shape[1], 16), -100.0)
        for index, token in enumerate(decoder_input_ids[0].tolist()):
            history.append(token)
            prediction = (sum((i + 1) * v for i, v in enumerate(history)) + self.offset) % 14 + 1
            if len(history) >= self.stop_after:
                prediction = 0
            logits[0, index, prediction] = 0
        return SimpleNamespace(logits=logits, past_key_values=HistoryCache(history))


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("offset", [0, 3, 9])
def test_cache_sensitive_exact_equivalence(adaptive, k, offset):
    prefix = torch.tensor([[1, 2, 3]])
    baseline = GreedyDecoder(HistoryModel(), 0).decode(None, prefix, 40)
    decoder = BufferedSpeculativeDecoder(
        HistoryModel(), HistoryModel(offset), 0, draft_k=k, max_draft_k=8, adaptive=adaptive
    )
    result = decoder.decode(None, None, prefix, 40)
    assert torch.equal(result.token_ids, baseline.token_ids)
    assert (
        result.stats.control_transfers
        == result.stats.verification_blocks + result.stats.fallback_tokens
    )
    if adaptive and offset:
        assert result.stats.fallback_tokens > 0


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5, 8, 12])
def test_no_bonus_or_proposal_overruns_budget(max_tokens):
    prefix = torch.tensor([[1]])
    baseline = GreedyDecoder(HistoryModel(stop_after=100), 0).decode(None, prefix, max_tokens)
    result = BufferedSpeculativeDecoder(
        HistoryModel(stop_after=100), HistoryModel(stop_after=100), 0, draft_k=4
    ).decode(None, None, prefix, max_tokens)
    assert torch.equal(result.token_ids, baseline.token_ids)
    assert result.stats.generated_tokens == max_tokens


@pytest.mark.parametrize("stop_after", [1, 2, 3, 4, 5, 7])
def test_stops_at_first_accepted_eos_inside_block(stop_after):
    prefix = torch.tensor([[1]])
    baseline = GreedyDecoder(HistoryModel(stop_after=stop_after), 0).decode(None, prefix, 20)
    result = BufferedSpeculativeDecoder(
        HistoryModel(stop_after=stop_after), HistoryModel(stop_after=stop_after), 0, draft_k=8
    ).decode(None, None, prefix, 20)
    assert torch.equal(result.token_ids, baseline.token_ids)
    assert result.token_ids[0, -1] == 0
    assert (result.token_ids == 0).sum() == 1


def test_fully_accepted_blocks_defer_draft_cache_extension():
    prefix = torch.tensor([[1]])
    old_draft = HistoryModel(stop_after=100)
    new_draft = HistoryModel(stop_after=100)
    old = SpeculativeGreedyDecoder(HistoryModel(stop_after=100), old_draft, 0, draft_k=3).decode(
        None, None, prefix, 20
    )
    new = BufferedSpeculativeDecoder(HistoryModel(stop_after=100), new_draft, 0, draft_k=3).decode(
        None, None, prefix, 20
    )
    assert torch.equal(old.token_ids, new.token_ids)
    assert new.stats.draft_calls < old.stats.draft_calls
    assert 2 in new_draft.input_lengths


def test_schedule_increases_decreases_and_falls_back():
    schedule = AdaptiveDraftSchedule()
    schedule.observe(4, 4)
    assert schedule.k == 5
    schedule.observe(0, 5)
    assert schedule.k == 2
    schedule.observe(0, 2)
    schedule.observe(0, 1)
    assert schedule.target_only


def test_cpu_stage_instrumentation_is_optional():
    result = BufferedSpeculativeDecoder(
        HistoryModel(), HistoryModel(), 0, profile_stages=True
    ).decode(None, None, torch.tensor([[1]]), 10)
    assert set(result.stats.stage_cpu_seconds) == {"draft", "verify", "control", "cache_crop"}
    assert result.stats.stage_cuda_seconds == {}


@pytest.mark.parametrize("adaptive", [False, True])
def test_real_transformers_whisper_cache_without_downloads(adaptive):
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    torch.manual_seed(17)
    config = WhisperConfig(
        vocab_size=16,
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=8,
        max_target_positions=64,
        decoder_start_token_id=1,
        eos_token_id=0,
        pad_token_id=0,
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    target = WhisperForConditionalGeneration(config).eval()
    draft = WhisperForConditionalGeneration(config).eval()
    with torch.inference_mode():
        features = torch.randn(1, 8, 16)
        target_encoder = target.get_encoder()(features)
        draft_encoder = draft.get_encoder()(features)
        prefix = torch.tensor([[1, 2]])
        baseline = GreedyDecoder(target, 0).decode(target_encoder, prefix, 12)
        actual = BufferedSpeculativeDecoder(target, draft, 0, adaptive=adaptive, draft_k=4).decode(
            target_encoder, draft_encoder, prefix, 12
        )
    assert torch.equal(actual.token_ids, baseline.token_ids)
