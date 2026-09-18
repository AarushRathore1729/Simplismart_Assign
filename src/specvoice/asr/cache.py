"""Compatibility helpers for cropping decoder KV caches.

The speculative decoder keeps the following invariant between verification
blocks: a cache contains every committed token except the final token. The
final committed token is passed into the next model call, which produces the
logits for the next token and extends the cache by one position.
"""

from __future__ import annotations

from typing import Any


def crop_cache(cache: Any, max_length: int) -> Any:
    """Crop self-attention state to ``max_length`` tokens.

    Supports recent Transformers cache objects and legacy encoder-decoder
    tuples. Cross-attention keys and values are intentionally left untouched.
    """

    if cache is None:
        return None
    if max_length < 0:
        raise ValueError("max_length must be non-negative")

    self_attention_cache = getattr(cache, "self_attention_cache", None)
    if self_attention_cache is not None and hasattr(self_attention_cache, "crop"):
        self_attention_cache.crop(max_length)
        return cache

    if hasattr(cache, "crop"):
        cache.crop(max_length)
        return cache

    if isinstance(cache, (tuple, list)):
        cropped_layers = []
        for layer in cache:
            if not isinstance(layer, (tuple, list)):
                raise TypeError("Unsupported legacy cache layer")

            states = list(layer)
            # Encoder-decoder legacy caches store self K/V first and cross K/V
            # afterwards. Only the self-attention sequence dimension is cropped.
            for index in range(min(2, len(states))):
                state = states[index]
                if state is not None:
                    states[index] = state[..., :max_length, :]
            cropped_layers.append(type(layer)(states))
        return type(cache)(cropped_layers)

    raise TypeError(f"Unsupported cache type: {type(cache)!r}")

