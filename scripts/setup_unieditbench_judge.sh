#!/usr/bin/env bash
set -euo pipefail

# Build a Worker-local UniEditBench environment, merge the complete video LoRA,
# and serve it on an OpenAI-compatible endpoint. Keep large caches on Worker
# NVMe because the Merlin home filesystem is small.

AURORA_USER_DIR="${AURORA_USER_DIR:-/mlx_devbox/users/jieyu.li}"
UNIEDIT_REPO="${UNIEDIT_REPO:-${AURORA_USER_DIR}/external/UniEditBench}"
JUDGE_BASE="${JUDGE_BASE:-${AURORA_USER_DIR}/models/Qwen3-VL-4B-Instruct}"
JUDGE_ADAPTER="${JUDGE_ADAPTER:-${AURORA_USER_DIR}/models/UniEditBench_models/sft_image_video_lora_4b}"
JUDGE_RUNTIME="${JUDGE_RUNTIME:-/tmp/aurora-unieditbench}"
JUDGE_PORT="${JUDGE_PORT:-8005}"
UV_BIN="${UV_BIN:-/home/tiger/.local/bin/uv}"
JUDGE_VENV="${JUDGE_RUNTIME}/.venv"
MERGED_MODEL="${JUDGE_RUNTIME}/Qwen3-VL-4B-SFT-Image-Video-merged"

for required_path in "$UNIEDIT_REPO/requirements.txt" "$JUDGE_BASE/config.json" "$JUDGE_ADAPTER/adapter_config.json"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required file: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$JUDGE_RUNTIME" /tmp/aurora-uv-cache /tmp/aurora-xdg-cache /tmp/aurora-vllm-cache /tmp/aurora-torchinductor-cache

if [[ ! -x "$JUDGE_VENV/bin/swift" ]]; then
  UV_CACHE_DIR=/tmp/aurora-uv-cache "$UV_BIN" venv "$JUDGE_VENV" --python 3.10
  UV_CACHE_DIR=/tmp/aurora-uv-cache "$UV_BIN" pip install \
    --python "$JUDGE_VENV/bin/python" \
    -r "$UNIEDIT_REPO/requirements.txt" \
    decord
fi

if [[ ! -f "$MERGED_MODEL/config.json" ]]; then
  XDG_CACHE_HOME=/tmp/aurora-xdg-cache "$JUDGE_VENV/bin/swift" export \
    --model "$JUDGE_BASE" \
    --adapters "$JUDGE_ADAPTER" \
    --merge_lora true \
    --attn_impl eager \
    --output_dir "$MERGED_MODEL"
fi

exec env \
  CUDA_VISIBLE_DEVICES=0 \
  XDG_CACHE_HOME=/tmp/aurora-xdg-cache \
  VLLM_CACHE_ROOT=/tmp/aurora-vllm-cache \
  TORCHINDUCTOR_CACHE_DIR=/tmp/aurora-torchinductor-cache \
  "$JUDGE_VENV/bin/swift" deploy \
    --model "$MERGED_MODEL" \
    --device_map balanced \
    --vllm_tensor_parallel_size 1 \
    --attn_impl eager \
    --infer_backend vllm \
    --port "$JUDGE_PORT" \
    --vllm_max_model_len 16384 \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_enforce_eager \
    --max_new_tokens 2048 \
    --served_model_name Qwen3-VL-4B-SFT-Image+Video-Merged
