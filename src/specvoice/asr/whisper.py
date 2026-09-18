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

# OpenAI renamed the no-speech control token when producing the large-v3
# tokenizer.  The checkpoints use the token for the same semantic event, but
# matching tokenizer strings literally would incorrectly reject Tiny/Large-v3.
_SEMANTIC_TOKEN_ALIASES = {
    "<|nocaptions|>": "<|nospeech|>",
    "<|nospeech|>": "<|nocaptions|>",
}


@dataclass(frozen=True)
class WhisperDecodingPolicy:
    """Prompt tokens and processors shared by both decoding paths."""

    prefix_ids: torch.Tensor
    logits_processor: Any
    begin_index: int
    return_timestamps: bool


@dataclass(frozen=True)
class WhisperVocabularyMap:
    """Bidirectional token-ID mapping between compatible Whisper tokenizers."""

    target_to_draft_table: torch.Tensor
    draft_to_target_table: torch.Tensor

    def target_to_draft(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._map(token_ids, self.target_to_draft_table, "target", "draft")

    def draft_to_target(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._map(token_ids, self.draft_to_target_table, "draft", "target")

    @staticmethod
    def _map(
        token_ids: torch.Tensor,
        table: torch.Tensor,
        source_name: str,
        destination_name: str,
    ) -> torch.Tensor:
        if token_ids.numel() == 0:
            return token_ids.clone()
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= table.numel():
            raise ValueError(f"{source_name} token ID is outside the tokenizer vocabulary")
        mapped = table[token_ids.to(device=table.device, dtype=torch.long)]
        if bool((mapped < 0).any()):
            missing = sorted(set(token_ids[mapped.to(token_ids.device) < 0].tolist()))
            raise ValueError(
                f"Cannot map {source_name} token IDs {missing} into the {destination_name} vocabulary"
            )
        return mapped.to(token_ids.device)


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
    *,
    device: torch.device | str | None = None,
) -> WhisperVocabularyMap:
    """Build and validate the semantic ID map for a draft/target pair."""

    target_tokenizer = getattr(target_processor, "tokenizer", None)
    draft_tokenizer = getattr(draft_processor, "tokenizer", None)
    if target_tokenizer is None or draft_tokenizer is None:
        raise ValueError("Both Whisper processors must expose their tokenizers")

    target_vocab = target_tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()
    target_size = int(target_model.config.vocab_size)
    draft_size = int(draft_model.config.vocab_size)
    if max(target_vocab.values(), default=-1) >= target_size:
        raise ValueError("Target tokenizer contains IDs outside the target model vocabulary")
    if max(draft_vocab.values(), default=-1) >= draft_size:
        raise ValueError("Draft tokenizer contains IDs outside the draft model vocabulary")

    resolved_device = torch.device(device or getattr(target_model, "device", "cpu"))
    target_to_draft = torch.full((target_size,), -1, dtype=torch.long, device=resolved_device)
    draft_to_target = torch.full((draft_size,), -1, dtype=torch.long, device=resolved_device)
    for token, draft_id in draft_vocab.items():
        target_id = target_vocab.get(token)
        if target_id is None:
            alias = _SEMANTIC_TOKEN_ALIASES.get(token)
            target_id = target_vocab.get(alias) if alias is not None else None
        if target_id is None:
            raise ValueError(f"Draft token {token!r} is absent from the target tokenizer")
        draft_to_target[int(draft_id)] = int(target_id)
        target_to_draft[int(target_id)] = int(draft_id)

    mapping = WhisperVocabularyMap(target_to_draft, draft_to_target)
    for field in ("decoder_start_token_id", "eos_token_id"):
        target_id = int(getattr(target_model.config, field))
        draft_id = int(getattr(draft_model.config, field))
        mapped = int(mapping.draft_to_target(torch.tensor([draft_id], device=resolved_device))[0])
        if mapped != target_id:
            raise ValueError(f"Draft and target {field} tokens are not semantically equivalent")
    return mapping
