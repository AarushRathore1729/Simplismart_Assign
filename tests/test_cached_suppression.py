import pytest
import torch
from test_whisper_policy import FakeProcessor, fake_model

from specvoice.asr.whisper import build_whisper_policy
from specvoice.compare import evenly_spaced_indices


@pytest.mark.parametrize("length", [4, 5, 8, 12])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cached_masks_match_the_hugging_face_policy(length, dtype):
    standard = build_whisper_policy(FakeProcessor(), fake_model())
    cached = build_whisper_policy(FakeProcessor(), fake_model(), cached_suppression=True)
    torch.manual_seed(19)
    scores = torch.randn(1, 12, dtype=dtype)
    context = torch.ones(1, length, dtype=torch.long)
    expected = standard.logits_processor(context, scores)
    actual = cached.logits_processor(context, scores)
    assert torch.equal(actual, expected)
    assert torch.equal(cached.prefix_ids, standard.prefix_ids)


def test_cached_suppression_keeps_standard_timestamp_rules():
    cached = build_whisper_policy(
        FakeProcessor(), fake_model(), return_timestamps=True, cached_suppression=True
    )
    assert cached.return_timestamps
    assert type(cached.logits_processor).__name__ == "LogitsProcessorList"


@pytest.mark.parametrize("size,limit", [(73, 30), (3, 10), (1, 1), (0, 5)])
def test_evenly_spaced_selection_has_no_duplicates(size, limit):
    indices = evenly_spaced_indices(size, limit)
    assert len(indices) == len(set(indices)) == min(size, limit)
    assert all(0 <= index < size for index in indices)
    if size and limit > 1:
        assert indices[0] == 0 and indices[-1] == size - 1
