import torch

from specvoice.asr.cache import crop_cache


def test_crops_only_legacy_self_attention_state():
    self_key = torch.zeros(1, 2, 7, 4)
    self_value = torch.zeros(1, 2, 7, 4)
    cross_key = torch.zeros(1, 2, 20, 4)
    cross_value = torch.zeros(1, 2, 20, 4)
    cache = ((self_key, self_value, cross_key, cross_value),)

    cropped = crop_cache(cache, 3)

    assert cropped[0][0].shape[-2] == 3
    assert cropped[0][1].shape[-2] == 3
    assert cropped[0][2].shape[-2] == 20
    assert cropped[0][3].shape[-2] == 20

