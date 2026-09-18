"""Correctness-first greedy and speculative greedy decoders.

Both decoders use the same raw model logits. This is deliberate: equality can
be tested token-for-token without comparing a custom loop against
``transformers.generate()``, which may apply additional model-specific logits
processors. A production Whisper adapter can supply the same logits processor
to both paths in a later milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import torch

from .cache import crop_cache


class LogitsProcessor(Protocol):
    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor: ...


def _identity_processor(_input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return logits


def _model_forward(
    model: Any,
    decoder_input_ids: torch.Tensor,
    encoder_outputs: Any,
    cache: Any,
) -> Any:
    return model(
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )


@dataclass(frozen=True)
class DecodeStats:
    generated_tokens: int
    draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    draft_calls: int = 0
    target_calls: int = 0
    elapsed_seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.draft_tokens


@dataclass(frozen=True)
class DecodeResult:
    token_ids: torch.Tensor
    stats: DecodeStats


class GreedyDecoder:
    """Autoregressive target-model decoder with a persistent KV cache."""

    def __init__(
        self,
        model: Any,
        eos_token_id: int,
        logits_processor: LogitsProcessor | None = None,
    ) -> None:
        self.model = model
        self.eos_token_id = eos_token_id
        self.logits_processor = logits_processor or _identity_processor

    @torch.inference_mode()
    def decode(
        self,
        encoder_outputs: Any,
        prefix_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> DecodeResult:
        _validate_inputs(prefix_ids, max_new_tokens)
        tokens = prefix_ids.clone()
        cache = None
        target_calls = 0
        start = perf_counter()

        for _ in range(max_new_tokens):
            model_input = tokens if cache is None else tokens[:, -1:]
            output = _model_forward(self.model, model_input, encoder_outputs, cache)
            target_calls += 1
            cache = output.past_key_values
            logits = self.logits_processor(tokens, output.logits[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
            if int(next_token.item()) == self.eos_token_id:
                break

        return DecodeResult(
            token_ids=tokens,
            stats=DecodeStats(
                generated_tokens=tokens.shape[1] - prefix_ids.shape[1],
                target_calls=target_calls,
                elapsed_seconds=perf_counter() - start,
            ),
        )


class SpeculativeGreedyDecoder:
    """Greedy draft/verify decoding that exactly matches greedy target decoding.

    The verifier scores a block of draft tokens in one forward pass. On a
    mismatch, the matching prefix is committed followed by the verifier token.
    If all proposals match, one bonus verifier token is committed.
    """

    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        eos_token_id: int,
        draft_k: int = 4,
        logits_processor: LogitsProcessor | None = None,
    ) -> None:
        if draft_k < 1:
            raise ValueError("draft_k must be at least 1")
        self.target_model = target_model
        self.draft_model = draft_model
        self.eos_token_id = eos_token_id
        self.draft_k = draft_k
        self.logits_processor = logits_processor or _identity_processor

    @torch.inference_mode()
    def decode(
        self,
        target_encoder_outputs: Any,
        draft_encoder_outputs: Any,
        prefix_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> DecodeResult:
        _validate_inputs(prefix_ids, max_new_tokens)
        tokens = prefix_ids.clone()
        target_cache = None
        draft_cache = None
        target_calls = 0
        draft_calls = 0
        proposed_total = 0
        accepted_total = 0
        start = perf_counter()

        while tokens.shape[1] - prefix_ids.shape[1] < max_new_tokens:
            remaining = max_new_tokens - (tokens.shape[1] - prefix_ids.shape[1])
            proposal_limit = min(self.draft_k, remaining)
            proposals = []

            for _ in range(proposal_limit):
                draft_input = tokens if draft_cache is None else tokens[:, -1:]
                if proposals:
                    draft_input = proposals[-1]

                output = _model_forward(
                    self.draft_model,
                    draft_input,
                    draft_encoder_outputs,
                    draft_cache,
                )
                draft_calls += 1
                draft_cache = output.past_key_values
                draft_context = torch.cat((tokens, *proposals), dim=1) if proposals else tokens
                logits = self.logits_processor(draft_context, output.logits[:, -1, :])
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                proposals.append(next_token)
                if int(next_token.item()) == self.eos_token_id:
                    break

            proposed = torch.cat(proposals, dim=1)
            proposed_total += proposed.shape[1]

            if target_cache is None:
                verifier_input = torch.cat((tokens, proposed), dim=1)
                prediction_start = tokens.shape[1] - 1
            else:
                verifier_input = torch.cat((tokens[:, -1:], proposed), dim=1)
                prediction_start = 0

            target_output = _model_forward(
                self.target_model,
                verifier_input,
                target_encoder_outputs,
                target_cache,
            )
            target_calls += 1
            target_cache = target_output.past_key_values

            verification_logits = target_output.logits[
                :, prediction_start : prediction_start + proposed.shape[1], :
            ]
            target_predictions = []
            for index in range(proposed.shape[1]):
                context = torch.cat((tokens, proposed[:, :index]), dim=1)
                processed = self.logits_processor(context, verification_logits[:, index, :])
                target_predictions.append(torch.argmax(processed, dim=-1, keepdim=True))
            target_predictions_tensor = torch.cat(target_predictions, dim=1)

            matches = proposed.eq(target_predictions_tensor)
            mismatch_positions = (~matches[0]).nonzero(as_tuple=False)

            if mismatch_positions.numel() > 0:
                mismatch_index = int(mismatch_positions[0].item())
                accepted_total += mismatch_index
                correction = target_predictions_tensor[:, mismatch_index : mismatch_index + 1]
                tokens = torch.cat(
                    (tokens, proposed[:, :mismatch_index], correction),
                    dim=1,
                )
                desired_cache_length = tokens.shape[1] - 1
                target_cache = crop_cache(target_cache, desired_cache_length)
                draft_cache = crop_cache(draft_cache, desired_cache_length)
                if int(correction.item()) == self.eos_token_id:
                    break
                continue

            accepted_total += proposed.shape[1]
            tokens = torch.cat((tokens, proposed), dim=1)

            if int(proposed[0, -1].item()) == self.eos_token_id:
                target_cache = crop_cache(target_cache, tokens.shape[1] - 1)
                break

            generated = tokens.shape[1] - prefix_ids.shape[1]
            if generated >= max_new_tokens:
                target_cache = crop_cache(target_cache, tokens.shape[1] - 1)
                break

            # Process the final proposal in the draft model so its cache covers
            # every committed token except the bonus verifier token.
            draft_extension = _model_forward(
                self.draft_model,
                proposed[:, -1:],
                draft_encoder_outputs,
                draft_cache,
            )
            draft_calls += 1
            draft_cache = draft_extension.past_key_values

            bonus_logits = target_output.logits[:, prediction_start + proposed.shape[1], :]
            bonus_logits = self.logits_processor(tokens, bonus_logits)
            bonus = torch.argmax(bonus_logits, dim=-1, keepdim=True)
            tokens = torch.cat((tokens, bonus), dim=1)
            if int(bonus.item()) == self.eos_token_id:
                break

        return DecodeResult(
            token_ids=tokens,
            stats=DecodeStats(
                generated_tokens=tokens.shape[1] - prefix_ids.shape[1],
                draft_tokens=proposed_total,
                accepted_draft_tokens=accepted_total,
                draft_calls=draft_calls,
                target_calls=target_calls,
                elapsed_seconds=perf_counter() - start,
            ),
        )


def _validate_inputs(prefix_ids: torch.Tensor, max_new_tokens: int) -> None:
    if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
        raise ValueError("Foundation decoder currently supports batch size 1")
    if prefix_ids.shape[1] == 0:
        raise ValueError("prefix_ids cannot be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
