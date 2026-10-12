"""Buffered greedy speculation with acceptance-based draft scheduling.

The persistent self-attention cache covers committed tokens except the last
one. After a fully accepted block, the next draft call consumes two uncached
tokens together, avoiding a separate draft-cache extension call. Drafting does
not read token IDs back to the CPU. Verification returns one small control
packet per block, including the first EOS in the committed prefix.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from .cache import crop_cache
from .decoder import (
    DecodeResult,
    DecodeStats,
    SpeculativeGreedyDecoder,
    _model_forward,
    _validate_inputs,
)


@dataclass
class AdaptiveDraftSchedule:
    """Small deterministic heuristic; state is reset for each utterance.

    Increase k after high acceptance, halve it after low acceptance, and use
    the target alone after three low-acceptance blocks. This is a heuristic,
    not a claim that adaptive speculation is optimal on every GPU.
    """

    initial_k: int = 4
    max_k: int = 8
    alpha: float = 0.5
    fallback_patience: int = 3

    def __post_init__(self):
        if not 1 <= self.initial_k <= self.max_k:
            raise ValueError("Require 1 <= initial_k <= max_k")
        self.k = self.initial_k
        self.ewma = 0.75
        self.low_blocks = 0
        self.target_only = False

    def observe(self, accepted: int, proposed: int) -> None:
        rate = accepted / max(proposed, 1)
        self.ewma = self.alpha * rate + (1 - self.alpha) * self.ewma
        self.low_blocks = self.low_blocks + 1 if rate < 0.35 else 0
        if self.low_blocks >= self.fallback_patience:
            self.target_only = True
        elif rate == 1.0 and self.ewma >= 0.8:
            self.k = min(self.max_k, self.k + 1)
        elif rate < 0.5:
            self.k = max(1, self.k // 2)


class StageRecorder:
    """Opt-in CPU spans, CUDA stream spans, profiler ranges and NVTX ranges.

    CUDA events are resolved once at the end, never between stages. GPU spans
    may include host launch gaps; they are not sums of kernel execution time.
    Keep this instrumentation disabled for headline latency measurements.
    """

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = enabled
        self.device = device
        self.cpu: dict[str, float] = {}
        self.cuda: dict[str, float] = {}
        self.events: list[tuple[str, Any, Any]] = []

    @contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        start = perf_counter()
        use_cuda = self.device.type == "cuda"
        if use_cuda:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            torch.cuda.nvtx.range_push(f"specvoice.{name}")
        try:
            with torch.profiler.record_function(f"specvoice.{name}"):
                yield
        finally:
            if use_cuda:
                torch.cuda.nvtx.range_pop()
                end.record()
                self.events.append((name, begin, end))
            self.cpu[name] = self.cpu.get(name, 0.0) + perf_counter() - start

    def finish(self):
        if self.events:
            torch.cuda.synchronize(self.device)
            for name, begin, end in self.events:
                self.cuda[name] = self.cuda.get(name, 0.0) + begin.elapsed_time(end) / 1000


class BufferedSpeculativeDecoder(SpeculativeGreedyDecoder):
    """Fixed or adaptive draft/verify using preallocated token workspaces.

    Only token and verification workspaces are preallocated; Transformers
    still manages dynamic KV tensors. Exactly matches the shared greedy target
    policy mathematically. Real GPU token equality remains a benchmark gate.
    """

    def __init__(
        self,
        *args,
        adaptive: bool = False,
        max_draft_k: int = 8,
        profile_stages: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if max_draft_k < self.draft_k:
            raise ValueError("max_draft_k cannot be smaller than draft_k")
        self.adaptive = adaptive
        self.max_draft_k = max_draft_k if adaptive else self.draft_k
        self.profile_stages = profile_stages

    @torch.inference_mode()
    def decode(self, target_encoder_outputs, draft_encoder_outputs, prefix_ids, max_new_tokens):
        _validate_inputs(prefix_ids, max_new_tokens)
        start = perf_counter()
        prefix_length = prefix_ids.shape[1]
        length = prefix_length
        buffer = prefix_ids.new_empty((1, prefix_length + max_new_tokens + self.max_draft_k + 1))
        buffer[:, :length].copy_(prefix_ids)
        predictions = prefix_ids.new_empty((1, self.max_draft_k + 1))
        positions = torch.arange(self.max_draft_k + 1, device=prefix_ids.device)
        target_cache = draft_cache = None
        draft_cache_length = 0
        target_calls = draft_calls = proposed_total = accepted_total = 0
        fallback_tokens = control_transfers = 0
        draft_lengths = []
        schedule = AdaptiveDraftSchedule(self.draft_k, self.max_draft_k)
        recorder = StageRecorder(self.profile_stages, prefix_ids.device)

        while length - prefix_length < max_new_tokens:
            remaining = max_new_tokens - (length - prefix_length)
            if self.adaptive and schedule.target_only:
                with recorder.span("target_fallback"):
                    model_input = (
                        buffer[:, :length]
                        if target_cache is None
                        else buffer[:, length - 1 : length]
                    )
                    output = _model_forward(
                        self.target_model, model_input, target_encoder_outputs, target_cache
                    )
                    target_cache = output.past_key_values
                    scores = self.logits_processor(buffer[:, :length], output.logits[:, -1, :])
                    next_token = scores.argmax(dim=-1, keepdim=True)
                    buffer[:, length : length + 1].copy_(next_token)
                target_calls += 1
                fallback_tokens += 1
                length += 1
                control_transfers += 1
                if int(next_token.item()) == self.eos_token_id:
                    break
                continue

            k = min(schedule.k if self.adaptive else self.draft_k, remaining)
            draft_lengths.append(k)
            with recorder.span("draft"):
                for index in range(k):
                    context_length = length + index
                    model_input = self.target_to_draft(buffer[:, draft_cache_length:context_length])
                    output = _model_forward(
                        self.draft_model, model_input, draft_encoder_outputs, draft_cache
                    )
                    draft_cache = output.past_key_values
                    draft_cache_length = context_length
                    scores = self.draft_logits_processor(
                        self.target_to_draft(buffer[:, :context_length]), output.logits[:, -1, :]
                    )
                    proposal = self.draft_to_target(scores.argmax(dim=-1, keepdim=True))
                    buffer[:, context_length : context_length + 1].copy_(proposal)
                    draft_calls += 1

            with recorder.span("verify"):
                model_input = (
                    buffer[:, : length + k]
                    if target_cache is None
                    else buffer[:, length - 1 : length + k]
                )
                prediction_start = length - 1 if target_cache is None else 0
                output = _model_forward(
                    self.target_model, model_input, target_encoder_outputs, target_cache
                )
                target_cache = output.past_key_values
                target_calls += 1
                # Every prediction uses its own prefix. This matters for
                # first-position suppression and history-dependent policies.
                for index in range(k + 1):
                    scores = self.logits_processor(
                        buffer[:, : length + index], output.logits[:, prediction_start + index, :]
                    )
                    predictions[:, index : index + 1].copy_(scores.argmax(dim=-1, keepdim=True))

            with recorder.span("control"):
                proposed = buffer[:, length : length + k]
                matches = proposed.eq(predictions[:, :k])
                accepted_tensor = matches.long().cumprod(dim=1).sum()
                candidates = predictions[:, : k + 1].clone()
                candidates[:, :k] = torch.where(
                    positions[:k] < accepted_tensor, proposed, predictions[:, :k]
                )
                commit_limit = torch.clamp(accepted_tensor + 1, max=remaining)
                first_eos = torch.where(
                    candidates[0].eq(self.eos_token_id) & (positions[: k + 1] < commit_limit),
                    positions[: k + 1] + 1,
                    k + 2,
                ).amin()
                commit_tensor = torch.minimum(commit_limit, first_eos)
                # One control packet transfer; no per-proposal .item() calls.
                accepted, committed, finished = torch.stack(
                    (accepted_tensor, commit_tensor, (first_eos <= commit_limit).long())
                ).tolist()
                control_transfers += 1
                buffer[:, length : length + committed].copy_(candidates[:, :committed])
                length += committed
                proposed_total += k
                accepted_total += min(accepted, committed)

            with recorder.span("cache_crop"):
                desired = length - 1
                target_cache = crop_cache(target_cache, desired)
                draft_cache = crop_cache(draft_cache, desired)
                draft_cache_length = min(draft_cache_length, desired)
            if self.adaptive:
                schedule.observe(min(accepted, committed), k)
            if finished:
                break

        recorder.finish()
        return DecodeResult(
            token_ids=buffer[:, :length].clone(),
            stats=DecodeStats(
                generated_tokens=length - prefix_length,
                draft_tokens=proposed_total,
                accepted_draft_tokens=accepted_total,
                draft_calls=draft_calls,
                target_calls=target_calls,
                elapsed_seconds=perf_counter() - start,
                verification_blocks=len(draft_lengths),
                fallback_tokens=fallback_tokens,
                draft_lengths=tuple(draft_lengths),
                control_transfers=control_transfers,
                stage_cpu_seconds=recorder.cpu,
                stage_cuda_seconds=recorder.cuda,
            ),
        )
