from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from specvoice.asr.decoder import GreedyDecoder, SpeculativeGreedyDecoder


@dataclass
class ToyCache:
    length: int

    def crop(self, max_length: int) -> None:
        self.length = min(self.length, max_length)


@dataclass
class ToyOutput:
    logits: torch.Tensor
    past_key_values: ToyCache


class ToyModel:
    def __init__(self, transitions: dict[int, int], vocab_size: int = 16):
        self.transitions = transitions
        self.vocab_size = vocab_size
        self.input_lengths = []

    def __call__(
        self,
        decoder_input_ids,
        encoder_outputs,
        past_key_values,
        use_cache,
        return_dict,
    ):
        del encoder_outputs, use_cache, return_dict
        self.input_lengths.append(decoder_input_ids.shape[1])
        logits = torch.full(
            (1, decoder_input_ids.shape[1], self.vocab_size),
            -1000.0,
            dtype=torch.float32,
        )
        for position, token in enumerate(decoder_input_ids[0].tolist()):
            logits[0, position, self.transitions[token]] = 0.0
        previous_length = 0 if past_key_values is None else past_key_values.length
        return ToyOutput(logits, ToyCache(previous_length + decoder_input_ids.shape[1]))


def decode_pair(target_transitions, draft_transitions, draft_k=3, max_new_tokens=8):
    target_for_baseline = ToyModel(target_transitions)
    target_for_speculation = ToyModel(target_transitions)
    draft = ToyModel(draft_transitions)
    prefix = torch.tensor([[1]], dtype=torch.long)
    baseline = GreedyDecoder(target_for_baseline, eos_token_id=0).decode(
        None, prefix, max_new_tokens
    )
    speculative = SpeculativeGreedyDecoder(
        target_for_speculation,
        draft,
        eos_token_id=0,
        draft_k=draft_k,
    ).decode(None, None, prefix, max_new_tokens)
    return baseline, speculative, target_for_speculation, draft


def test_exact_match_when_every_draft_token_is_accepted():
    transitions = {1: 2, 2: 3, 3: 4, 4: 5, 5: 0, 0: 0}
    baseline, speculative, target, draft = decode_pair(transitions, transitions)

    assert torch.equal(speculative.token_ids, baseline.token_ids)
    assert speculative.token_ids.tolist() == [[1, 2, 3, 4, 5, 0]]
    assert speculative.stats.acceptance_rate == 1.0
    assert speculative.stats.target_calls < baseline.stats.target_calls
    assert max(target.input_lengths[1:], default=1) <= 4
    assert max(draft.input_lengths[1:], default=1) == 1


def test_rejection_rolls_back_and_uses_target_token():
    target = {1: 2, 2: 3, 3: 4, 4: 5, 5: 0, 9: 8, 8: 0, 0: 0}
    draft = {1: 2, 2: 9, 9: 8, 3: 4, 4: 5, 5: 0, 0: 0, 8: 0}
    baseline, speculative, _, _ = decode_pair(target, draft)

    assert torch.equal(speculative.token_ids, baseline.token_ids)
    assert speculative.token_ids.tolist() == [[1, 2, 3, 4, 5, 0]]
    assert 0.0 < speculative.stats.acceptance_rate < 1.0


def test_respects_max_new_tokens_without_bonus_overrun():
    transitions = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0, 0: 0}
    baseline, speculative, _, _ = decode_pair(
        transitions, transitions, draft_k=4, max_new_tokens=3
    )

    assert torch.equal(speculative.token_ids, baseline.token_ids)
    assert speculative.token_ids.tolist() == [[1, 2, 3, 4]]
    assert speculative.stats.generated_tokens == 3


def test_rejects_invalid_configuration_and_batch_size():
    model = ToyModel({1: 0, 0: 0})
    with pytest.raises(ValueError, match="draft_k"):
        SpeculativeGreedyDecoder(model, model, eos_token_id=0, draft_k=0)

    decoder = GreedyDecoder(model, eos_token_id=0)
    with pytest.raises(ValueError, match="batch size 1"):
        decoder.decode(None, torch.tensor([[1], [1]]), 2)


def test_supports_distinct_draft_and_target_token_id_spaces():
    target_transitions = {1: 2, 2: 3, 3: 4, 4: 0, 0: 0}
    draft_transitions = {6: 7, 7: 8, 8: 9, 9: 5, 5: 5}
    target_to_draft_table = torch.tensor([5, 6, 7, 8, 9])
    draft_to_target_table = torch.tensor([-1, -1, -1, -1, -1, 0, 1, 2, 3, 4])

    baseline = GreedyDecoder(ToyModel(target_transitions), eos_token_id=0).decode(
        None, torch.tensor([[1]]), 8
    )
    speculative = SpeculativeGreedyDecoder(
        ToyModel(target_transitions),
        ToyModel(draft_transitions),
        eos_token_id=0,
        draft_k=3,
        target_to_draft=lambda ids: target_to_draft_table[ids],
        draft_to_target=lambda ids: draft_to_target_table[ids],
    ).decode(None, None, torch.tensor([[1]]), 8)

    assert torch.equal(speculative.token_ids, baseline.token_ids)
    assert speculative.token_ids.tolist() == [[1, 2, 3, 4, 0]]
