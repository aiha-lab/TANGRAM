# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request KV cache reordering after compression.

Runs after ``model.forward``. For each (layer, head-group) it gathers
the cached KV, assembles ``keep_idx = sink ∪ locked ∪ topk ∪ window``
from the caches ``prepare_keep_decision`` stashed, and scatters the
kept K/V back in block-aligned form. Backend-agnostic: touches only
KV tensors and the block_table."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.attention.compression.compressor import KVCompressor
from vllm.v1.worker.block_table import BlockTable


@dataclass
class CompressionMetadata:
    """Per-(request, step) compression info passed to ``run_request``.

    ``floor_min`` is the per-(layer, group) ``kept_lengths`` absolute
    floor; 0 disables it. All other run-shape config lives on the
    compressor under ``req_state[req_id]``.
    """
    req_id: str
    row_idx: int
    chunk_len: int
    floor_min: int


class CompressionExecutor:
    """One instance per ModelRunner; stateless across requests."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads_per_layer: int,
        page_group_size: int,
        head_size: int,
        block_size: int,
    ) -> None:
        assert num_kv_heads_per_layer % page_group_size == 0, (
            f"num_kv_heads_per_layer ({num_kv_heads_per_layer}) must be "
            f"divisible by page_group_size ({page_group_size})."
        )
        self.num_layers = num_layers
        self.num_kv_heads_per_layer = num_kv_heads_per_layer
        self.page_group_size = page_group_size
        self.num_head_groups_per_layer = (
            num_kv_heads_per_layer // page_group_size
        )
        self.head_size = head_size
        self.block_size = block_size
        # Reused arange slabs; sink/win sizes are KeepDecision-uniform.
        self._sink_idx_cache: torch.Tensor | None = None
        self._win_idx_cache: torch.Tensor | None = None

    def run_request(
        self,
        layer_kv_caches: list[torch.Tensor],
        block_table: BlockTable,
        eff_seq_lens_cpu: np.ndarray,
        compressor: KVCompressor,
        compression_metadata: CompressionMetadata,
    ) -> np.ndarray:
        """Apply the keep decision to every layer in one call.

        Slots ``[0, kept_lengths[layer, group])`` of each cache are
        overwritten with the kept KV (block-aligned write). The
        ``block_table`` is not mutated; the caller invokes
        ``compact_after_compress_all_layers`` afterwards. Returns
        ``[num_layers, num_groups]`` int32 post-evict lengths.

        Requires ``prepare_keep_decision`` to have populated
        ``req.cross_layer_decision`` and the per-(layer, group) caches.
        """
        assert block_table.head_grouped, (
            "CompressionExecutor.run_request requires a head-grouped "
            "BlockTable."
        )
        assert len(layer_kv_caches) == self.num_layers

        num_layers = self.num_layers
        num_groups = self.num_head_groups_per_layer
        page_group_size = self.page_group_size
        head_size = self.head_size
        block_size = self.block_size
        metadata = compression_metadata
        device = layer_kv_caches[0].device

        req = compressor.req_state.get(metadata.req_id)
        if req is None or req.cross_layer_decision is None:
            raise RuntimeError(
                f"CompressionExecutor.run_request({metadata.req_id}): "
                "cross_layer_decision missing — prepare_keep_decision "
                "must run before run_request.")
        keep_dec = req.cross_layer_decision
        sink_size = keep_dec.sink_size
        win_size = keep_dec.win_size
        adjusted_ratio = keep_dec.adjusted_ratio
        eval_start = keep_dec.eval_start
        eval_end = keep_dec.eval_end
        eval_len = max(0, eval_end - eval_start)

        # Under TP the runner cross-rank MAX-reduces kept_lengths before
        # reaching us.
        if req.cached_kept_lengths_cpu is None:
            raise RuntimeError(
                f"CompressionExecutor.run_request({metadata.req_id}): "
                "cached_kept_lengths_cpu missing — "
                "compute_kept_lengths_per_rank must run before "
                "run_request.")
        kept_lengths_all = req.cached_kept_lengths_cpu

        locked_cpu = (
            req.locked_count_cpu
            if req.locked_count_cpu is not None
            else np.zeros((num_layers, num_groups), dtype=np.int64))
        sorted_idx = req.cached_sorted_indices
        group_scores = req.cached_group_scores

        if (self._sink_idx_cache is None
                or self._sink_idx_cache.numel() < sink_size
                or self._sink_idx_cache.device != device):
            self._sink_idx_cache = torch.arange(
                max(sink_size, 64), device=device, dtype=torch.long)
        sink_idx_full = self._sink_idx_cache[:sink_size]
        if (self._win_idx_cache is None
                or self._win_idx_cache.numel() < win_size
                or self._win_idx_cache.device != device):
            self._win_idx_cache = torch.arange(
                max(win_size, 4096), device=device, dtype=torch.long)
        win_idx_base = self._win_idx_cache[:win_size]

        block_table_gpu = block_table.block_table.gpu
        new_locked_all = np.zeros((num_layers, num_groups), dtype=np.int64)
        chunk_len = metadata.chunk_len
        row_idx = metadata.row_idx

        for layer_idx in range(num_layers):
            kv_cache = layer_kv_caches[layer_idx]
            layer_first = layer_idx * num_groups

            prev_seq_lens_np = eff_seq_lens_cpu[
                row_idx, layer_first : layer_first + num_groups
            ].astype(np.int64, copy=True)
            total_seen_per_group = prev_seq_lens_np + chunk_len
            total_seen_max = int(total_seen_per_group.max())
            if total_seen_max == 0:
                raise RuntimeError(
                    f"CompressionExecutor.run_request(layer={layer_idx}"
                    f", req={metadata.req_id}): total_seen=0 (prev=0, "
                    f"chunk_len={chunk_len}). Skip the compression step "
                    "instead of calling run_request.")

            # Fast path: keep everything → only refresh new_locked.
            if adjusted_ratio >= 1.0:
                for group_idx in range(num_groups):
                    locked = int(locked_cpu[layer_idx, group_idx])
                    kept_length = int(
                        kept_lengths_all[layer_idx, group_idx])
                    k_aligned = max(
                        0,
                        kept_length - sink_size - locked - win_size)
                    new_locked_all[layer_idx, group_idx] = (
                        locked + k_aligned)
                continue

            for group_idx in range(num_groups):
                total_seen = int(total_seen_per_group[group_idx])
                locked = int(locked_cpu[layer_idx, group_idx])
                kept_lo = sink_size + locked
                win_lo = total_seen - win_size

                # Under TP this rank may need to extend its top-k up to
                # the cross-rank-MAX-reduced kept_length; sorted_idx
                # already holds eval_len positions so the slice is safe.
                kept_length = int(kept_lengths_all[layer_idx, group_idx])
                k_aligned = max(
                    0, kept_length - sink_size - locked - win_size)
                if k_aligned > eval_len:
                    k_aligned = eval_len
                new_locked_all[layer_idx, group_idx] = locked + k_aligned

                if kept_length == 0:
                    continue

                n_blocks = (total_seen + block_size - 1) // block_size
                block_ids = block_table_gpu[
                    row_idx, layer_first + group_idx, :n_blocks
                ].long()
                slab_kv = kv_cache[:, block_ids].reshape(
                    2, -1, page_group_size, head_size)[:, :total_seen]

                pieces: list[torch.Tensor] = []
                if sink_size > 0:
                    pieces.append(sink_idx_full)
                if locked > 0:
                    pieces.append(torch.arange(
                        sink_size, sink_size + locked,
                        device=device, dtype=torch.long))
                if k_aligned > 0:
                    if sorted_idx is not None:
                        topk_local = sorted_idx[
                            layer_idx, group_idx, :k_aligned]
                    else:
                        scores = group_scores[
                            layer_idx, group_idx, eval_start:eval_end]
                        topk_local = torch.topk(
                            scores, k=k_aligned).indices
                    topk_local, _ = topk_local.sort()
                    pieces.append(topk_local + kept_lo)
                if win_size > 0:
                    pieces.append(win_idx_base + win_lo)
                keep_idx = (
                    torch.cat(pieces) if pieces
                    else torch.empty(
                        0, dtype=torch.long, device=device))

                kept_kv = slab_kv.index_select(1, keep_idx)

                # Scatter back; pad trailing partial block with zeros.
                n_blocks_write = (
                    kept_length + block_size - 1) // block_size
                padded_size = n_blocks_write * block_size
                if kept_length < padded_size:
                    pad = torch.zeros(
                        2, padded_size - kept_length,
                        page_group_size, head_size,
                        dtype=kept_kv.dtype, device=device)
                    kept_kv = torch.cat([kept_kv, pad], dim=1)
                kv_cache[:, block_ids[:n_blocks_write]] = kept_kv.view(
                    2, n_blocks_write, block_size,
                    page_group_size, head_size)

        # Batched H2D: collapses L per-layer state writes into two.
        new_locked_gpu = torch.from_numpy(new_locked_all).to(device)
        valid_lengths_gpu = torch.from_numpy(
            kept_lengths_all.astype(np.int64)).to(device)
        for layer_idx in range(num_layers):
            state = req.layer_states.get(layer_idx)
            if state is None:
                continue
            state.locked_count_per_group = new_locked_gpu[layer_idx]
            state.valid_lengths_per_group = valid_lengths_gpu[layer_idx]

        return kept_lengths_all
