from types import SimpleNamespace

import pytest
import torch

from specvoice.asr.whisper import build_whisper_policy, validate_whisper_pair


class FakeProcessor:
    def __init__(self, vocab=None):
        self.tokenizer = SimpleNamespace(get_vocab=lambda: vocab or {"token": 1})

    def get_decoder_prompt_ids(self, language, task, no_timestamps):
        assert language == "en"
        assert task == "transcribe"
        return [(1, 3), (2, 4), (3, 5)] if no_timestamps else [(1, 3), (2, 4)]


def fake_model(**overrides):
    config = {
        "vocab_size": 12,
        "decoder_start_token_id": 1,
        "eos_token_id": 0,
    }
    config.update(overrides)
    generation_config = SimpleNamespace(
        suppress_tokens=[8],
        begin_suppress_tokens=[9],
        no_timestamps_token_id=5,
        eos_token_id=0,
        bos_token_id=1,
        max_initial_timestamp_index=50,
    )
    return SimpleNamespace(
        config=SimpleNamespace(**config),
        generation_config=generation_config,
        device=torch.device("cpu"),
    )


def test_builds_prompt_and_applies_shared_suppression_policy():
    policy = build_whisper_policy(FakeProcessor(), fake_model())

    assert policy.prefix_ids.tolist() == [[1, 3, 4, 5]]
    assert policy.begin_index == 4

    scores = torch.zeros((1, 12))
    first = policy.logits_processor(policy.prefix_ids, scores)
    later = policy.logits_processor(
        torch.cat((policy.prefix_ids, torch.tensor([[2]])), dim=1), scores
    )
    assert torch.isneginf(first[0, 8])
    assert torch.isneginf(first[0, 9])
    assert torch.isneginf(later[0, 8])
    assert not torch.isneginf(later[0, 9])


def test_timestamp_prompt_excludes_no_timestamps_token():
    policy = build_whisper_policy(
        FakeProcessor(), fake_model(), return_timestamps=True
    )
    assert policy.prefix_ids.tolist() == [[1, 3, 4]]
    assert policy.begin_index == 3


def test_rejects_non_contiguous_prompt_positions():
    processor = FakeProcessor()
    processor.get_decoder_prompt_ids = lambda **_: [(2, 3)]
    with pytest.raises(ValueError, match="contiguous"):
        build_whisper_policy(processor, fake_model())


def test_validates_model_and_tokenizer_compatibility():
    target = fake_model()
    draft = fake_model(vocab_size=13)
    with pytest.raises(ValueError, match="vocab_size"):
        validate_whisper_pair(target, draft, FakeProcessor(), FakeProcessor())

    with pytest.raises(ValueError, match="tokenizers"):
        validate_whisper_pair(
            target,
            fake_model(),
            FakeProcessor({"a": 1}),
            FakeProcessor({"a": 2}),
        )
