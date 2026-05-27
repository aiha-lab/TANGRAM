<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <img src="docs/assets/logos/tangram-logo-text.png" alt="Tangram" width="480"/>
</p>

<h3 align="center">
Tangram: Unlocking Non-Uniform KV Cache for Efficient Multi-turn LLM Serving
</h3>

<p align="center">
| <a href="https://aiha-lab.github.io/tangram-page/"><b>Project Page</b></a> | <a href="https://aiha-lab.github.io/tangram-page/#"><b>Paper</b></a> |
</p>

**Tangram** is a serving system that makes non-uniform KV cache compression
practical for multi-turn LLM serving. It is built on top of
[vLLM](https://github.com/vllm-project/vllm).

**Highlights**

- **Up to ~5× memory savings** — head group page reclaims fragmented KV cache memory
- **Up to 2.3× throughput** — static budget allocation removes scheduling overhead
- **Minimal accuracy loss** — non-uniform KV compression preserves the heads that matter

**Core techniques**

1. **Deterministic Budget Allocation** — static per-head memory footprint, no runtime scheduling overhead
2. **Head Group Page** — clusters heads by retention demand with independent, vectorized page tables
3. **Ahead-of-Time (AOT) Load Balancing** — offline workload partitioning for uniform SM utilization

---

## Built on vLLM

Tangram is built on top of [vLLM](https://github.com/vllm-project/vllm), a fast
and easy-to-use library for LLM inference and serving. See the
[vLLM documentation](https://docs.vllm.ai/en/latest/) for the underlying engine
and supported models.

Install from source:

```bash
pip install -e .
```

## Quick Start

Head Group paging is on by default; add `--enable-compression` for non-uniform compression.

```bash
vllm serve /path/to/Qwen2.5-7B-Instruct-1M \
    --enable-compression \
    --compression-ratio 0.3
```

By default `--max-model-len` follows the model's `max_position_embeddings`
and the server listens on port `8000`; override with the standard vLLM
flags if you need a different value.

- `--enable-compression` / `--compression-ratio R` — non-uniform KV cache compression with retention fraction `R`.
- `--page-group-size N` — head group page (default `4`).

`enforce_eager`, `max_num_batched_tokens`, and `enable_prefix_caching` are
auto-set when these features are on. Compression sizing
(`--compression-chunk-size`, `--compression-window-size`,
`--compression-n-sink-tokens`) can be tuned if needed.

## Supported Models

- Qwen3-4B
- Qwen2.5-7B (Remove DCA Config on config.json)
- Qwen3-32B
- Llama-3.1-8B-Instruct

# Supported Attention Backend
- FlashAttention

Test a completion:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "/path/to/Qwen2.5-7B-Instruct-1M",
         "prompt": "Tangram is",
         "max_tokens": 32,
         "temperature": 0}'
```

## Citation

