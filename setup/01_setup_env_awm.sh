#!/bin/bash

# Source the common variables and functions
source "$(dirname "$0")/common_vars.sh"
source_conda

ENV_NAME="awm"
log "Step 1: Setting up Conda environment: ${ENV_NAME}"

# Check if environment already exists
if conda info --envs | grep -q "^${ENV_NAME}\s"; then
  log "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
  log "Accepting Conda Terms of Service..."
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

  log "Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" --yes

  log "Activating environment and installing packages for '${ENV_NAME}'..."
  conda activate "$ENV_NAME"

  # Navigate to the correct directory
  cd "${PROJECT_ROOT}/advantage_weighted_matching"

  pip install -e .
  pip install flash-attn==2.7.4.post1 --no-build-isolation
  pip install ipdb

  # handle deepspeed zero2 version issue
  pip install deepspeed==0.17.2
  pip install accelerate==1.9.0
  pip install transformers==4.54.0

  # --- PaddleOCR Installation ---
  log "Installing PaddleOCR and its dependencies..."
  pip install paddlepaddle-gpu==2.5.2
  pip install paddleocr==2.9.1
  pip install python-Levenshtein

  # --- Pre-download PaddleOCR Model ---
  log "Pre-downloading the PaddleOCR model..."
  python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)"
  log "PaddleOCR model download complete."

  log "Installing ImageReward and CLIP..."
  pip install image-reward
  pip install git+https://github.com/openai/CLIP.git

  # --- Upgrade timm ---
  log "Upgrading timm..."
  pip install timm==1.0.13

  conda deactivate
  log "Environment '${ENV_NAME}' setup complete."
fi