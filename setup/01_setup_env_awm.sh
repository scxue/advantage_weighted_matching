#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_vars.sh"
source_conda

if env_exists "$AWM_ENV_NAME"; then
  if [[ "${AWM_UPDATE_ENV:-0}" != 1 ]]; then
    log "Environment $AWM_ENV_NAME exists; leaving it unchanged. Set AWM_UPDATE_ENV=1 to update."
    exit 0
  fi
else
  # Channel terms, if applicable, must be accepted explicitly by the user.
  conda create -n "$AWM_ENV_NAME" python="$PYTHON_VERSION" --yes
fi
conda activate "$AWM_ENV_NAME"
cd "$PROJECT_ROOT/advantage_weighted_matching"
export PIP_CONSTRAINT="$PROJECT_ROOT/advantage_weighted_matching/requirements.txt"
python -m pip install -e .

# Optional CUDA build: not required by the default SD3 DDP recipe.
if [[ "${AWM_INSTALL_FLASH_ATTN:-0}" == 1 ]]; then
  python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
fi
# OCR's Paddle stack is separate from the validated Geneval dependency set.
if [[ "${AWM_INSTALL_OCR:-0}" == 1 ]]; then
  python -m pip install paddlepaddle-gpu==2.5.2 paddleocr==2.9.1 python-Levenshtein
  python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)"
fi
if [[ "${AWM_INSTALL_IMAGEREWARD:-0}" == 1 ]]; then
  python -m pip install image-reward timm==1.0.13
fi
python -m pip check
conda deactivate
log "Main environment setup finished."
