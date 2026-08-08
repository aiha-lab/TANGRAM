# KeyDiff use case

Source: [`vllm/v1/attention/compression/keydiff.py`](../../vllm/v1/attention/compression/keydiff.py) ·
Paper: [arXiv:2504.15364](https://arxiv.org/abs/2504.15364)

## How it works

The scorer reads the model's post-RoPE keys and nothing else:

```python
k = key.reshape(chunk_len, self.num_kv_heads, self.head_size).float()
anchor = F.normalize(k, p=2, dim=-1).mean(dim=0, keepdim=True)  # chunk mean direction
score = -F.cosine_similarity(k, anchor, dim=-1)                 # [T, H], higher = keep
```

Keys pointing near the chunk's mean direction are the least distinctive, so negating the
cosine similarity ranks the redundant tokens lowest and evicts them first. Two
consequences follow: KeyDiff is **gate-free** (no checkpoint to load, unlike FastKVzip)
and **query-independent** (it never reads queries, so chunked prefill carries no extra
state across chunks).

The `[num_kv_heads, chunk_len]` score it returns is the contract every Tangram scorer
shares, so `compression_level` stays an orthogonal knob.

## Speedup

<p align="center">
  <img src="../assets/speedup/speedup_keydiff.png" alt="Tangram end-to-end speedup, KeyDiff scorer" width="100%"/>
</p>

## Accuracy

KeyDiff is the second column.

<p align="center">
  <img src="../assets/accuracy/accuracy_scbench.png" alt="SCBench accuracy comparison, Tangram vs PyTorch across four compression methods" width="100%"/>
</p>

## How to run

Speedup:

```bash
cd benchmarks/tangram/speedup
SCORERS=keydiff RATIOS="1.0 0.5 0.25 0.1" ./run_speedup.sh
```

SCBench accuracy:

```bash
cd benchmarks/tangram
SCORER=keydiff LEVEL=perlayer_cluster DATASET=mid RATIOS="1.0 0.5 0.25 0.1" \
bash benchmark_scbench.sh
```

Override `MODEL=` for another model.

### Knobs

`SCORER` (or `SCORERS` in `run_speedup.sh`, which takes a list) picks the importance
scorer, and **it must be `keydiff` for anything on this page to apply** — the scripts
otherwise default to `snapkv`, and `run_speedup.sh` to `snapkv fastkvzip`.

`LEVEL` is the *selection level*: the scope a KV budget is shared over, which decides
whether heads may keep different numbers of tokens.

| `LEVEL` | Budget scope |
| ------- | ------------ |
| `uniform` | Every (layer, head group) keeps the same token count; only *which* tokens are kept differs. Needs no cluster map. |
| `perlayer_cluster` | Each layer gets an equal budget, spread non-uniformly across the heads in that layer. Needs the `_perlayer` cluster map. |
| `crosslayer_cluster` | One global budget spread across all layers and heads, so an important head in any layer can keep more. Needs the cross-layer cluster map. |

The scripts also accept `perlayer_head` and `crosslayer_head`, which apply the same two
scopes with a head-calibrated threshold instead of a cluster-calibrated one. KeyDiff
cluster maps for every verified model already ship under
[`tools/head_group_clustering/cluster_maps/keydiff/`](../../tools/head_group_clustering/cluster_maps/keydiff),
so the cluster levels work with no extra step.
