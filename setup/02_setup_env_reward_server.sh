#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_vars.sh"
source_conda

if env_exists "$AWM_REWARD_ENV_NAME"; then
  if [[ "${AWM_UPDATE_ENV:-0}" != 1 ]]; then
    log "Environment $AWM_REWARD_ENV_NAME exists; leaving it unchanged. Set AWM_UPDATE_ENV=1 to update."
    exit 0
  fi
else
  conda create -n "$AWM_REWARD_ENV_NAME" python="$PYTHON_VERSION" --yes
fi
conda activate "$AWM_REWARD_ENV_NAME"
# Do not inherit main-environment constraints into the older detector stack.
export PIP_CONSTRAINT="$PROJECT_ROOT/reward-server/requirements-geneval.txt"
python -m pip install pip==24.3.1 setuptools==60.2.0 wheel
python -m pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$PIP_CONSTRAINT"

mkdir -p "$PROJECT_ROOT/third_party"
ensure_checkout https://github.com/open-mmlab/mmcv.git v1.7.2 \
  4c01b026f0afa5a91a5f54aea313788da1e40f95 "$PROJECT_ROOT/third_party/mmcv"
ensure_checkout https://github.com/open-mmlab/mmdetection.git v2.28.2 \
  e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1 "$PROJECT_ROOT/third_party/mmdetection"

# Defaults to H100. For A100 set TORCH_CUDA_ARCH_LIST=8.0; other GPUs need
# the appropriate architecture and a toolkit compatible with Torch cu121.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ]]; then
  log "MMCV needs a CUDA toolkit/compiler. Set CUDA_HOME and PATH before installation."
  exit 1
fi
MMCV_WITH_OPS=1 python -m pip install --no-build-isolation -v -e "$PROJECT_ROOT/third_party/mmcv"
python -m pip install --no-build-isolation -v -e "$PROJECT_ROOT/third_party/mmdetection"
# v1.7.2 satisfies v2.28.2's original MMCV check; no source patch is needed.

REWARD_SERVER_DIR="$PROJECT_ROOT/reward-server"
MODEL_FILE="$REWARD_SERVER_DIR/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
if [[ ! -f "$MODEL_FILE" ]]; then
  download="$(mktemp "$REWARD_SERVER_DIR/.mask2former-download.XXXXXX")"
  wget -O "$download" https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
  mv -- "$download" "$MODEL_FILE"
fi
python -m pip check
conda deactivate
log "Geneval reward environment setup finished."
