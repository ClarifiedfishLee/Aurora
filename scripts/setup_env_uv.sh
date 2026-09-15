#!/usr/bin/env bash
# Aurora environment setup using uv (training + inference + evaluation).
# Validated target: Python 3.10, PyTorch 2.5.0 + CUDA 12.4, flash-attn 2.7.3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${AURORA_VENV_DIR:-${REPO_DIR}/.venv}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_DIR}/../.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${AURORA_UV_PYTHON_DIR:-${REPO_DIR}/../.uv-python}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

echo "=== Aurora uv environment setup ==="
echo "Repository: ${REPO_DIR}"
echo "Environment: ${VENV_DIR}"
echo "uv cache: ${UV_CACHE_DIR}"

uv python install 3.10
uv venv --python 3.10 "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "--- Installing PyTorch 2.5.0 (cu124) ---"
uv pip install --python "${PYTHON_BIN}" \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0

echo "--- Installing build and core dependencies ---"
uv pip install --python "${PYTHON_BIN}" \
    "setuptools<70" wheel ninja packaging \
    accelerate pyyaml modelscope imageio imageio-ffmpeg einops wandb \
    safetensors sentencepiece protobuf pandas peft lmdb datasets \
    huggingface-hub==0.34

# modelscope may select a different transformers build, so pin it afterwards.
uv pip install --python "${PYTHON_BIN}" transformers==5.3.0

echo "--- Installing prebuilt flash-attn 2.7.3 wheel ---"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
uv pip install --python "${PYTHON_BIN}" "${FLASH_ATTN_WHEEL}"

echo "--- Installing DeepSpeed and evaluation dependencies ---"
# DeepSpeed 0.19.x uses custom-op schemas that PyTorch 2.5 cannot infer.
CUDA_HOME="${CUDA_HOME}" uv pip install --python "${PYTHON_BIN}" deepspeed==0.15.4
uv pip install --python "${PYTHON_BIN}" \
    decord openai ftfy opencv-python-headless "diffusers>=0.36"

# Dependencies are installed explicitly above. --no-deps prevents optional
# platform extras from affecting uv's resolution of this editable package.
uv pip install --python "${PYTHON_BIN}" --no-deps -e "${REPO_DIR}"

echo "--- Verifying imports ---"
"${PYTHON_BIN}" - <<'PY'
import cv2
import decord
import deepspeed
import diffusers
import flash_attn
import imageio
import modelscope
import openai
import peft
import torch
import transformers

assert torch.__version__.startswith("2.5.0"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert transformers.__version__ == "5.3.0", transformers.__version__
assert flash_attn.__version__ == "2.7.3", flash_attn.__version__
print(f"OK: python environment is ready")
print(f"    torch={torch.__version__} cuda={torch.version.cuda}")
print(f"    transformers={transformers.__version__} flash_attn={flash_attn.__version__}")
print(f"    deepspeed={deepspeed.__version__} diffusers={diffusers.__version__}")
print(f"    decord={decord.__version__} openai={openai.__version__} cv2={cv2.__version__}")
PY

echo "=== Setup complete ==="
echo "Activate with: source ${VENV_DIR}/bin/activate"
