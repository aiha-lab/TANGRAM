# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep-decision logic for non-uniform KV cache compression.

Owns the per-request score buffers and the global-top-K keep decision.
KV writes and block-table updates live in the FlashAttention backend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.v1.attention.compression.gate import CompressionGate, load_gates

logger = init_logger(__name__)


@dataclass
class KeepDecision:
    """Cross-layer global threshold for one chunk: ``score > threshold ⇒ keep``.

    The eval region is the workspace slice ``[eval_start, eval_end)``;
    sink / locked / window positions outside it are auto-kept.
    """
    threshold: float
    sink_size: int
    win_size: int
    adjusted_ratio: float
    eval_start: int = 0
    eval_end: int = 0


@dataclass
class _LayerCompressState:
    # [num_kv_heads, win_size]: window-region scores from the previous chunk.
    prior_window_scores: torch.Tensor | None = None
    # [G]: tokens already promoted to "kept" by prior compress steps.
    locked_count_per_group: torch.Tensor | None = None
    # [G]: kept length after the most recent compress
    # (= sink + locked + win, clamped to total_seen).
    valid_lengths_per_group: torch.Tensor | None = None
    # [num_kv_heads, chunk_len]: this step's gate score.
    pending_score: torch.Tensor | None = None


@dataclass
class _RequestCompressState:
    layer_states: dict[int, _LayerCompressState] = field(default_factory=dict)
    cross_layer_decision: KeepDecision | None = None
    # [L, num_kv_heads, win + chunk]: grow-only workspace
    # laid out as [prior_window | pending_score].
    score_workspace: torch.Tensor | None = None
    workspace_size: int = 0
    # [L, G, win + chunk]: workspace after head-group amax; rebuilt each chunk.
    cached_group_scores: torch.Tensor | None = None
    cached_sorted_indices: torch.Tensor | None = None    # [L, G, eval_len]
    cached_k_new_cpu: np.ndarray | None = None           # [L, G]
    locked_count_cpu: np.ndarray | None = None           # [L, G]
    # [L, G] int32: post-evict kept_lengths. Under TP the runner
    # cross-rank MAX-reduces this for block-pool consistency.
    cached_kept_lengths_cpu: np.ndarray | None = None


class KVCompressor:
    """One instance per model; per-request state held in ``req_state``.

    ``compress_active`` is flipped by the ModelRunner around the compress
    forward pass and read by the per-layer pre-hook.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        page_group_size: int,
        head_size: int,
        hidden_dim: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        assert num_kv_heads % page_group_size == 0, (
            f"num_kv_heads ({num_kv_heads}) must be divisible by "
            f"page_group_size ({page_group_size}).")

        self.num_layers = num_layers
        self.num_kv_heads_per_layer = num_kv_heads
        self.page_group_size = page_group_size
        self.num_head_groups_per_layer = num_kv_heads // page_group_size
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) \
            else device

        # Populated by ``load_gate_checkpoint``; kept separate so unit tests
        # can exercise compress() without a checkpoint.
        self.gates: list[CompressionGate] = []

        self.req_state: dict[str, _RequestCompressState] = {}

        # ``pending_req_offsets`` is a list of ``(req_id, start, end)``
        # triples giving each compression-active request's token range in
        # the batch's hidden_states. Tokens outside any triple are skipped.
        self.compress_active: bool = False
        self.pending_req_offsets: list[tuple[str, int, int]] | None = None

    def load_gate_checkpoint(
        self,
        model_name: str,
        gate_path: str,
        num_kv_heads_total: int,
        tp_rank: int,
    ) -> None:
        """Load per-layer gates from a Fast-KVzip checkpoint.

        Under TP, ``num_kv_heads_total`` is the model-global KV-head count
        the checkpoint was trained against; the loader shards it to this
        rank's slice ``[tp_rank * per_rank, (tp_rank + 1) * per_rank)``.
        """
        self.gates = load_gates(
            model_name=model_name,
            gate_path=gate_path,
            num_layers=self.num_layers,
            num_kv_heads_per_rank=self.num_kv_heads_per_layer,
            num_kv_heads_total=num_kv_heads_total,
            tp_rank=tp_rank,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def begin_request(self, req_id: str) -> None:
        if req_id in self.req_state:
            raise RuntimeError(
                f"KVCompressor.begin_request: '{req_id}' already active.")
        self.req_state[req_id] = _RequestCompressState()

    def end_request(self, req_id: str) -> None:
        # Idempotent for worker shutdown paths.
        self.req_state.pop(req_id, None)

    def receive_hidden_score(
        self,
        req_id: str,
        layer_idx: int,
        score: torch.Tensor,
    ) -> None:
        """Stash the gate-produced ``[num_kv_heads_per_layer, chunk_len]``
        score for the next compress() call (consumed same-step)."""
        if score.shape[0] != self.num_kv_heads_per_layer:
            raise ValueError(
                f"score head dim {score.shape[0]} != "
                f"num_kv_heads_per_layer {self.num_kv_heads_per_layer}.")
        if req_id not in self.req_state:
            raise RuntimeError(
                f"KVCompressor.receive_hidden_score: '{req_id}' not "
                "begin_request'd.")
        layer_state = self.req_state[req_id].layer_states.setdefault(
            layer_idx, _LayerCompressState())
        layer_state.pending_score = score

    def prepare_keep_decision(
        self,
        req_id: str,
        prev_seq_lens_per_layer: torch.Tensor,
        chunk_len: int,
        ratio: float,
        window_size: int,
        n_sink_tokens: int,
        total_prompt_tokens: int,
    ) -> KeepDecision:
        """Cross-layer global threshold over the eval region;
        ``score > threshold ⇒ keep``. Caches per-(layer, group) sorted
        indices for the executor. Enforces the once-only invariant:
        ``prev_seq_lens_per_layer`` matches the last
        ``valid_lengths_per_group`` (or is all-zero on the first chunk)."""
        if req_id not in self.req_state:
            raise RuntimeError(
                f"prepare_keep_decision: '{req_id}' not begin_request'd.")
        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"prepare_keep_decision: ratio must be in (0, 1], got {ratio}.")

        req = self.req_state[req_id]
        num_layers = self.num_layers
        num_groups = self.num_head_groups_per_layer
        page_group_size = self.page_group_size
        num_kv_heads = self.num_kv_heads_per_layer

        # Sink / window sized by the smallest (layer, group) total length.
        prev_lens = prev_seq_lens_per_layer.to(dtype=torch.long).cpu()
        if prev_lens.shape != (num_layers, num_groups):
            raise ValueError(
                f"prev_seq_lens shape {tuple(prev_lens.shape)} != "
                f"({num_layers}, {num_groups}).")
        min_total = int((prev_lens + chunk_len).min().item())
        sink_size = min(n_sink_tokens, min_total)
        win_size = min(window_size, max(0, min_total - sink_size))

        self._assert_once_only(req, prev_lens, num_layers, num_groups)

        # ``adjusted_ratio`` mirrors baseline FastKVzip
        # (wrapper.py:188-194); we hold ``win_size`` fixed, so the
        # window-shrink branch collapses to zero.
        eff_prompt = max(0, int(total_prompt_tokens) - sink_size)
        if ratio >= 1.0 or eff_prompt <= win_size:
            adjusted_ratio = 1.0
        elif ratio * eff_prompt < win_size:
            adjusted_ratio = 0.0
        else:
            adjusted_ratio = max(0.0, min(1.0,
                (ratio * eff_prompt - win_size) / (eff_prompt - win_size)))

        # First chunk skips the locked [sink, sink+win) prefix; subsequent
        # chunks evaluate the whole fresh chunk.
        is_first_chunk = not any(
            ls.valid_lengths_per_group is not None
            for ls in req.layer_states.values())
        eval_start = win_size + sink_size if is_first_chunk else 0
        eval_end = chunk_len
        eval_len = max(0, eval_end - eval_start)

        # Workspace dtype/device follow the gate output (may stay float32
        # even when self.dtype == bfloat16) for byte-equivalence.
        first_score = next(
            (ls.pending_score for ls in req.layer_states.values()
             if ls.pending_score is not None), None)
        if first_score is None:
            raise RuntimeError(
                f"prepare_keep_decision({req_id}): no pending_score "
                "— receive_hidden_score must run first.")
        dtype, device = first_score.dtype, first_score.device
        neg_inf = torch.finfo(dtype).min

        workspace_need = win_size + chunk_len
        if (req.score_workspace is None
                or req.workspace_size < workspace_need
                or req.score_workspace.dtype != dtype
                or req.score_workspace.device != device):
            req.score_workspace = torch.empty(
                num_layers, num_kv_heads, workspace_need,
                dtype=dtype, device=device)
            req.workspace_size = workspace_need
        workspace = req.score_workspace
        workspace_alloc = req.workspace_size
        workspace.fill_(neg_inf)

        pending, prior, prev_locked = self._collect_layer_tensors(
            req, num_layers, num_groups, num_kv_heads,
            win_size, chunk_len, dtype, device, neg_inf)
        if win_size > 0:
            workspace[:, :, :win_size] = prior
        workspace[:, :, win_size:win_size + chunk_len] = pending

        max_locked = (
            prev_lens.to(device) + chunk_len - sink_size - win_size
        ).clamp_min(0)
        locked = torch.minimum(prev_locked, max_locked)
        for layer_idx in range(num_layers):
            req.layer_states[layer_idx].locked_count_per_group = (
                locked[layer_idx])

        # Save next chunk's prior_window from this chunk's tail.
        if adjusted_ratio < 1.0:
            if win_size > 0 and chunk_len >= win_size:
                new_prior = pending[
                    :, :, chunk_len - win_size:].detach().clone()
                for layer_idx in range(num_layers):
                    req.layer_states[layer_idx].prior_window_scores = (
                        new_prior[layer_idx])
            elif win_size == 0:
                empty = torch.empty(
                    num_kv_heads, 0, dtype=dtype, device=device)
                for layer_idx in range(num_layers):
                    req.layer_states[layer_idx].prior_window_scores = empty
            # win > chunk_len is degenerate; leave prior unchanged.

        # amax over heads-in-group → per-(layer, group) scores.
        group_scores = workspace.view(
            num_layers, num_groups, page_group_size,
            workspace_alloc).amax(dim=2)
        req.cached_group_scores = group_scores

        # topk(n+1).min is O(N) vs full sort O(NlogN). Under TP, all-gather
        # group_scores along the head-group axis so the threshold spans
        # every rank's KV heads (same global semantic as single-process
        # FastKVzip). Each rank's sorted_idx / k_new below stay per-rank.
        if eval_len <= 0 or adjusted_ratio >= 1.0:
            threshold = float('-inf')
        elif adjusted_ratio <= 0.0:
            threshold = float('inf')
        else:
            tp_world_size = get_tensor_model_parallel_world_size()
            eval_slice = group_scores[:, :, eval_start:eval_end]
            if tp_world_size > 1:
                eval_slice_contiguous = eval_slice.contiguous()
                gathered = [
                    torch.empty_like(eval_slice_contiguous)
                    for _ in range(tp_world_size)
                ]
                torch.distributed.all_gather(
                    gathered, eval_slice_contiguous,
                    group=get_tp_group().device_group)
                flat = torch.cat(gathered, dim=1).reshape(-1)
            else:
                flat = eval_slice.reshape(-1)
            n = max(int(flat.numel() * adjusted_ratio) - 1, 0)
            threshold = float(torch.topk(flat, k=n + 1).values.min().item())

        # Per-(layer, group) sort + k_new cache for the executor; skipped
        # on the fast / zero paths to avoid a needless sort.
        if eval_len > 0 and 0.0 < adjusted_ratio < 1.0:
            sorted_vals, sorted_idx = group_scores[
                :, :, eval_start:eval_end].sort(dim=-1, descending=True)
            k_new = (sorted_vals > threshold).sum(dim=-1)
            req.cached_sorted_indices = sorted_idx
            req.cached_k_new_cpu = k_new.cpu().numpy().astype(np.int64)
        else:
            req.cached_sorted_indices = None
            req.cached_k_new_cpu = None
        req.locked_count_cpu = locked.cpu().numpy().astype(np.int64)

        decision = KeepDecision(
            threshold=threshold,
            sink_size=int(sink_size),
            win_size=int(win_size),
            adjusted_ratio=float(adjusted_ratio),
            eval_start=int(eval_start),
            eval_end=int(eval_end),
        )
        req.cross_layer_decision = decision
        req.cached_kept_lengths_cpu = None
        return decision

    def compute_kept_lengths_per_rank(
        self,
        req_id: str,
        eff_seq_lens_row: np.ndarray,
        chunk_len: int,
        floor_min: int,
    ) -> np.ndarray:
        """Per-(layer, group) post-evict kept_lengths for this chunk.

        Mirrors :py:meth:`CompressionExecutor.run_request` arithmetic
        without touching KV cache. Result is cached on
        ``req.cached_kept_lengths_cpu`` so the executor can read it back;
        under TP the caller may cross-rank MAX-reduce the cache before
        ``run_request`` to keep the block pool consistent."""
        req = self.req_state.get(req_id)
        if req is None or req.cross_layer_decision is None:
            raise RuntimeError(
                f"compute_kept_lengths_per_rank({req_id}): "
                "cross_layer_decision missing — prepare_keep_decision "
                "must run first.")
        keep_dec = req.cross_layer_decision
        sink_size = keep_dec.sink_size
        win_size = keep_dec.win_size
        adjusted_ratio = keep_dec.adjusted_ratio
        eval_len = max(0, keep_dec.eval_end - keep_dec.eval_start)

        num_layers = self.num_layers
        num_groups = self.num_head_groups_per_layer
        block_size = self.block_size
        floor_min_int = int(floor_min)

        prev_lens = eff_seq_lens_row.astype(
            np.int64, copy=False).reshape(num_layers, num_groups)
        total_seen = prev_lens + chunk_len

        # adjusted_ratio >= 1: keep every position in the eval region.
        if adjusted_ratio >= 1.0:
            kept_lengths = total_seen.astype(np.int32)
            req.cached_kept_lengths_cpu = kept_lengths
            return kept_lengths

        locked_cpu = (
            req.locked_count_cpu
            if req.locked_count_cpu is not None
            else np.zeros((num_layers, num_groups), dtype=np.int64))
        k_new_cpu = req.cached_k_new_cpu

        kept_lengths = np.zeros(
            (num_layers, num_groups), dtype=np.int32)
        for layer_idx in range(num_layers):
            for group_idx in range(num_groups):
                total_seen_g = int(total_seen[layer_idx, group_idx])
                locked_count = int(locked_cpu[layer_idx, group_idx])
                if eval_len > 0:
                    # adjusted_ratio == 0 ⇒ no sort cached, keep none.
                    k_new = (int(k_new_cpu[layer_idx, group_idx])
                             if k_new_cpu is not None else 0)
                    kept_now = (
                        sink_size + locked_count + k_new + win_size)
                    target_floor = min(floor_min_int, total_seen_g)
                    if kept_now < target_floor:
                        extra = min(
                            target_floor - kept_now,
                            eval_len - k_new)
                        if extra > 0:
                            k_new += extra
                    k_aligned = (
                        ((k_new + block_size - 1) // block_size)
                        * block_size)
                    k_aligned = min(k_aligned, eval_len)
                else:
                    k_aligned = 0
                new_locked = locked_count + k_aligned
                kept_length = sink_size + new_locked + win_size
                if kept_length > total_seen_g:
                    kept_length = total_seen_g
                kept_lengths[layer_idx, group_idx] = kept_length
        req.cached_kept_lengths_cpu = kept_lengths
        return kept_lengths

    def _assert_once_only(
        self,
        req: "_RequestCompressState",
        prev_lens: torch.Tensor,
        num_layers: int,
        num_groups: int,
    ) -> None:
        """Compression must run only on chunked-prefill: ``prev_lens`` must
        match the prior ``valid_lengths_per_group`` (or be all-zero on the
        first chunk)."""
        ref = next(
            (ls for ls in req.layer_states.values()
             if ls.valid_lengths_per_group is not None
             or ls.locked_count_per_group is not None), None)

        if ref is None:
            if (prev_lens != 0).any():
                bad = int((prev_lens != 0).any(dim=1).long().argmax())
                raise RuntimeError(
                    f"once-only violated: layer {bad} "
                    f"prev_lens={prev_lens[bad].tolist()} but no prior state.")
            return

        device = (ref.valid_lengths_per_group
                  if ref.valid_lengths_per_group is not None
                  else ref.locked_count_per_group).device
        valid = torch.zeros(
            num_layers, num_groups, dtype=torch.long, device=device)
        for layer_idx in range(num_layers):
            ls = req.layer_states.get(layer_idx)
            if ls is not None and ls.valid_lengths_per_group is not None:
                valid[layer_idx] = ls.valid_lengths_per_group.to(torch.long)
        valid_cpu = valid.cpu()
        if not torch.equal(valid_cpu, prev_lens):
            bad = int((valid_cpu != prev_lens).any(dim=1).long().argmax())
            raise RuntimeError(
                f"once-only violated: layer {bad} "
                f"prev_lens={prev_lens[bad].tolist()} "
                f"valid_lens={valid_cpu[bad].tolist()}.")

    def _collect_layer_tensors(
        self,
        req: "_RequestCompressState",
        num_layers: int,
        num_groups: int,
        num_kv_heads: int,
        win_size: int,
        chunk_len: int,
        dtype: torch.dtype,
        device: torch.device,
        neg_inf: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack per-layer pending / prior / prev_locked into [L, ...]
        tensors. Consumes pending_score on every layer."""
        empty_prior = torch.full(
            (num_kv_heads, win_size), neg_inf, dtype=dtype, device=device)
        zero_locked = torch.zeros(num_groups, dtype=torch.long, device=device)
        pending_list: list[torch.Tensor] = []
        prior_list: list[torch.Tensor] = []
        locked_list: list[torch.Tensor] = []
        for layer_idx in range(num_layers):
            state = req.layer_states.get(layer_idx)
            if state is None or state.pending_score is None:
                raise RuntimeError(
                    f"layer {layer_idx}: no pending_score "
                    "— receive_hidden_score must run first.")
            fresh = state.pending_score
            state.pending_score = None
            if fresh.shape[1] != chunk_len:
                raise ValueError(
                    f"layer {layer_idx}: pending_score chunk_len "
                    f"{fresh.shape[1]} != {chunk_len}.")
            if fresh.dtype != dtype or fresh.device != device:
                raise RuntimeError(
                    f"layer {layer_idx}: pending_score dtype/device "
                    f"mismatch (got {fresh.dtype}/{fresh.device}, "
                    f"expected {dtype}/{device}).")
            pending_list.append(fresh)

            prior = state.prior_window_scores
            prior_ok = (
                prior is not None and win_size > 0
                and prior.shape[1] == win_size
                and prior.dtype == dtype
                and prior.device == device)
            prior_list.append(prior if prior_ok else empty_prior)

            locked_list.append(
                state.locked_count_per_group.to(torch.long)
                if state.locked_count_per_group is not None
                else zero_locked)

        pending = torch.stack(pending_list, dim=0)
        prior = (torch.stack(prior_list, dim=0) if win_size > 0
                 else torch.empty(
                     num_layers, num_kv_heads, 0,
                     dtype=dtype, device=device))
        prev_locked = torch.stack(locked_list, dim=0)
        return pending, prior, prev_locked

    def attach_to_attention_layers(
        self,
        attention_layers: list[nn.Module],
    ) -> None:
        """Wire each layer's gate + forward pre-hook.

        Hook fires only when ``compress_active`` is True and
        ``pending_req_offsets`` is non-empty; otherwise zero overhead.
        For each ``(req_id, start, end)`` triple it scores
        ``hidden_states[start:end]`` and stashes via
        ``receive_hidden_score``. ``attention_layers[i]`` must correspond
        to ``gates[i]`` — caller owns the ordering."""
        if len(attention_layers) != len(self.gates):
            raise ValueError(
                f"attach_to_attention_layers: got "
                f"{len(attention_layers)} layers but "
                f"{len(self.gates)} gates.")

        for layer_idx, (layer, gate) in enumerate(
                zip(attention_layers, self.gates)):
            layer.compression_gate = gate

            def pre_hook(module, args, kwargs, _idx=layer_idx, _gate=gate):
                if not self.compress_active:
                    return None
                offsets = self.pending_req_offsets
                if not offsets:
                    return None
                hidden_states = args[0] if args else kwargs.get(
                    "hidden_states")
                if hidden_states is None:
                    return None

                # Concatenate all compression-active req slices into a
                # single gate forward per layer to amortise kernel launch.
                valid = [(req, start, end)
                         for req, start, end in offsets if end > start]
                if not valid:
                    return None

                with torch.no_grad():
                    if len(valid) == 1:
                        req, start, end = valid[0]
                        score = _gate(hidden_states[start:end])
                        self.receive_hidden_score(req, _idx, score)
                    else:
                        slices = [hidden_states[start:end]
                                  for _, start, end in valid]
                        full = torch.cat(slices, dim=0)
                        # [num_kv_heads, sum(end - start)]
                        score_full = _gate(full)
                        cursor = 0
                        for req, start, end in valid:
                            length = end - start
                            self.receive_hidden_score(
                                req, _idx,
                                score_full[:, cursor:cursor + length],
                            )
                            cursor += length
                return None

            layer.register_forward_pre_hook(
                pre_hook, with_kwargs=True)
