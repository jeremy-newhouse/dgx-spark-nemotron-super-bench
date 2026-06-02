#!/usr/bin/env bash
#
# Launch vLLM serving nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 on a single
# DGX Spark (GB10, sm_121a) with the built-in MTP speculative-decoding head.
#
# Usage:
#   1) Pre-launch on the Spark (avoids a unified-memory loader OOM):
#        sudo sysctl vm.drop_caches=3
#   2) ./launch_server.sh
#   3) Wait for "Application startup complete" (first load takes several minutes).
#
# Override any value via env, e.g.:  MAX_MODEL_LEN=8192 SPEC_TOKENS=3 ./launch_server.sh
set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"  # HF id or local path
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # decode rate is insensitive to this; NVIDIA's guide uses 1048576
SPEC_TOKENS="${SPEC_TOKENS:-3}"           # MTP draft length (k); 3 is the canonical recipe
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Model:  ${MODEL}"
echo "Image:  ${IMAGE}"
echo "Recipe: marlin dense GEMM + auto-pick CUTLASS MoE + MTP k=${SPEC_TOKENS}, fp8 KV, batch=1"

# --- preflight: check free memory (does NOT run any privileged command) ---
# The Spark shares one 128 GB pool between GPU and host, so an over-committed launch can
# OOM the HOST, not just the GPU. This does not drop the page cache for you (that needs
# sudo) -- it confirms there's enough free memory and tells you to drop the cache if not.
# (vm.drop_caches is a write-only trigger that always reads back 0, so MemAvailable is the
# meaningful readiness signal.)
if [ -r /proc/meminfo ]; then
  meminfo_gib() { awk -v k="$1:" '$1==k{printf "%d",$2/1048576}' /proc/meminfo; }
  have="$(meminfo_gib MemAvailable)"; cached="$(meminfo_gib Cached)"; need="${MIN_AVAIL_GIB:-90}"
  echo "Preflight: MemAvailable ${have} GiB, page cache ${cached} GiB (want MemAvailable >= ${need} GiB for ~75 GiB weights + headroom)."
  if [ "${have:-0}" -lt "$need" ] && [ -z "${FORCE:-}" ]; then
    echo "STOP: only ${have} GiB available -- not enough to launch safely on this unified-memory box."
    echo "  Drop the Linux page cache first (it can OOM the loader), then re-run this script:"
    echo "      sudo sysctl vm.drop_caches=3"
    echo "  If that doesn't free enough, close other processes (or try GPU_MEM_UTIL=0.80)."
    echo "  Override this check with FORCE=1 (not recommended on a remote Spark)."
    exit 1
  fi
fi

docker run --rm --gpus all \
  -p "${PORT}:8000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --oom-score-adj 1000 \
  "${IMAGE}" \
  --model "${MODEL}" \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8 \
  --mamba_ssm_cache_dtype float32 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-seqs 1 \
  --speculative_config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS},\"moe_backend\":\"triton\"}"

# Notes:
# - Do NOT add a Marlin MoE override (e.g. forcing moe to marlin at the model level).
#   Leaving the MoE backend to vLLM's auto-pick (native CUTLASS NVFP4) is the fast,
#   stable path on v0.22.0; forcing a Marlin MoE repack can OOM the unified-memory box.
# - The deployment recipe also adds `--reasoning-parser nemotron_v3`. It does not
#   affect decode throughput, so it is omitted here to reduce startup failure surface.
# - --max-num-seqs is 1 for a clean single-stream measurement; NVIDIA's Spark guide
#   recommends 4 for stability. At batch=1 it does not change the decode rate.
