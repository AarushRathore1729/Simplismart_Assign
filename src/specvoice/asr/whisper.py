"""Whisper-specific prompt and logits-processing policy.

The baseline and speculative decoders must apply the *same* token policy.  This
module builds that policy from a Hugging Face Whisper checkpoint without
depending on private ``generate()`` internals.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class WhisperDecodingPolicy:
    """Prompt tokens and processors shared by both decoding paths."""

    prefix_ids: torch.Tensor
    logits_processor: Any
    begin_index: int
    return_timestamps: bool


def build_whisper_policy(
    processor: Any,
    model: Any,
    *,
    language: str = "en",
    task: str = "transcribe",
    return_timestamps: bool = False,
    device: torch.device | str | None = None,
) -> WhisperDecodingPolicy:
    """Build the short-form greedy policy used by Hugging Face Whisper.

    The policy covers the language/task prompt, ``<|notimestamps|>`` selection,
    globally suppressed tokens, tokens suppressed only at the first generated
    position, and Whisper's timestamp constraints when timestamps are enabled.
    """

    from transformers.generation.logits_process import (
        LogitsProcessorList,
        SuppressTokensAtBeginLogitsProcessor,
        SuppressTokensLogitsProcessor,
        WhisperTimeStampLogitsProcessor,
    )

    resolved_device = torch.device(device or getattr(model, "device", "cpu"))
    generation_config = deepcopy(model.generation_config)
    generation_config.return_timestamps = return_timestamps

    prompt_pairs = processor.get_decoder_prompt_ids(
        language=language,
        task=task,
        no_timestamps=not return_timestamps,
    )
    prefix = [int(model.config.decoder_start_token_id)]
    for position, token_id in prompt_pairs:
        if int(position) != len(prefix):
            raise ValueError(
                "Whisper decoder prompt positions must be contiguous; "
                f"expected {len(prefix)}, received {position}"
            )
        if token_id is None:
            raise ValueError("Whisper decoder prompt cannot contain an undefined token")
        prefix.append(int(token_id))

    begin_index = len(prefix)
    processors = LogitsProcessorList()
    if return_timestamps:
        if not hasattr(generation_config, "no_timestamps_token_id"):
            raise ValueError("The checkpoint does not define Whisper timestamp tokens")
        processors.append(
            WhisperTimeStampLogitsProcessor(generation_config, begin_index=begin_index)
        )

    suppress_tokens = getattr(generation_config, "suppress_tokens", None)
    if suppress_tokens:
        processors.append(SuppressTokensLogitsProcessor(suppress_tokens, device=resolved_device))

    begin_suppress_tokens = getattr(generation_config, "begin_suppress_tokens", None)
    if begin_suppress_tokens:
        processors.append(
            SuppressTokensAtBeginLogitsProcessor(
                begin_suppress_tokens,
                begin_index=begin_index,
                device=resolved_device,
            )
        )

    return WhisperDecodingPolicy(
        prefix_ids=torch.tensor([prefix], dtype=torch.long, device=resolved_device),
        logits_processor=processors,
        begin_index=begin_index,
        return_timestamps=return_timestamps,
    )


def validate_whisper_pair(
    target_model: Any,
    draft_model: Any,
    target_processor: Any,
    draft_processor: Any,
) -> None:
    """Reject draft/target pairs that cannot safely share token IDs."""

    fields = ("vocab_size", "decoder_start_token_id", "eos_token_id")
    for field in fields:
        target_value = getattr(target_model.config, field, None)
        draft_value = getattr(draft_model.config, field, None)
        if target_value != draft_value:
            raise ValueError(
                f"Draft and target {field} must match: {draft_value!r} != {target_value!r}"
            )

    target_tokenizer = getattr(target_processor, "tokenizer", None)
    draft_tokenizer = getattr(draft_processor, "tokenizer", None)
    if (
        target_tokenizer is not None
        and draft_tokenizer is not None
        and hasattr(target_tokenizer, "get_vocab")
        and hasattr(draft_tokenizer, "get_vocab")
        and target_tokenizer.get_vocab() != draft_tokenizer.get_vocab()
    ):
        raise ValueError("Draft and target tokenizers must map every token to the same ID")
