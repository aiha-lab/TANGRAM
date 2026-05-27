#!/usr/bin/env bash
# Sweep SCBench across compression ratios for Tangram on vLLM.
#
# Per-model context-length reference (LLaMA3 tokenizer, approximate):
#   Qwen2.5-7B-Instruct-1M (1M)    → all SCBench datasets fit
#   Qwen3-4B-Instruct-2507 (262K)  → most datasets fit; use _short/_mid for mf, kv
#   Qwen2.5-32B            (131K)  → use _short/_mid for mf, kv
#   LLaMA-3.1-8B-Instruct  (128K)  → use _short/_mid for mf, kv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# vLLM V1's engine core is launched via fork by default. CUDA being touched in
# the benchmark driver (e.g. via the vllm import) makes the forked subprocess
# raise "Cannot re-initialize CUDA". Force spawn so the engine core starts
# from a clean Python process.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# ---- Model preset (edit as needed) ---------------------------------------
GPU_ID=${GPU_ID:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct-1M}
MAX_LEN=${MAX_LEN:-200000}
EXTRA_ARGS=()

# Python interpreter: uses the active environment. Activate your venv/conda env
# first (with vllm + requirements/cuda.txt installed), or override with
# PYTHON=/path/to/python.
PYTHON=${PYTHON:-python3}

# Multi-GPU example:
#   GPU_ID=0,1,2,3
#   MODEL=Qwen/Qwen2.5-32B-Instruct
#   MAX_LEN=131072
#   EXTRA_ARGS=(--tensor-parallel-size 4 --disable-custom-all-reduce \
#               --gpu-memory-utilization 0.85)

# ---- Sweep configuration -------------------------------------------------
DATASETS=(scbench_kv)
RATIOS=(1.0)
NUM=${NUM:-100}
PAGE_GROUP_SIZE=${PAGE_GROUP_SIZE:-4}
MAX_TOKENS=${MAX_TOKENS:-512}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/results_scbench"}

# ---- Run -----------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        echo "===== dataset=${DATASET}  ratio=${RATIO} ====="
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON" "${SCRIPT_DIR}/benchmark_scbench.py" \
            -d "${DATASET}" \
            --num "${NUM}" \
            --ratio "${RATIO}" \
            --page-group-size "${PAGE_GROUP_SIZE}" \
            -m "${MODEL}" \
            --max-model-len "${MAX_LEN}" \
            --max-tokens "${MAX_TOKENS}" \
            --single-turn \
            --force-exact-tokens \
            --output-dir "${OUTPUT_DIR}" \
            "${EXTRA_ARGS[@]}"
    done
done
