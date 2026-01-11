#!/bin/bash

# Source the common variables and functions
source "$(dirname "$0")/common_vars.sh"
source_conda

ENV_NAME="reward_server"
log "Step 2: Setting up Conda environment: ${ENV_NAME}"

# Check if environment already exists
if conda info --envs | grep -q "^${ENV_NAME}\s"; then
  log "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
  log "Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" --yes

  log "Activating environment and installing packages for '${ENV_NAME}'..."
  conda activate "$ENV_NAME"
  log "Downgrading pip to a version compatible with editable installs..."
  conda install -y "pip<25"

  # Navigate to the project directory
  REWARD_SERVER_DIR="${PROJECT_ROOT}/reward-server"
  cd "$REWARD_SERVER_DIR"

  log "Installing PyTorch and other dependencies..."
  pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

  pip install transformers==4.38.2 gunicorn==23.0.0 openmim==0.3.9 open-clip-torch==2.31.0 numpy==1.26.0 opencv-python==4.11.0.86 clip-benchmark==1.6.1 flask==3.1.0

  log "Installing MMCV and MMDetection..."
  # For Ampere arch, derictly use 'mim install mmcv-full mmengine'
  # For Hopper arch, install mmcv with the following command
  git clone -b v1.7.2 --depth=1 https://github.com/open-mmlab/mmcv.git
  cd mmcv
  MMCV_WITH_OPS=1 MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .
  cd ..

  if [ ! -d "mmdetection" ]; then
    git clone https://github.com/open-mmlab/mmdetection.git
  fi
  cd mmdetection
  git checkout 2.x

  # Modify max version in init to avoid version conflicts
  log "Patching mmdetection version requirement..."
  sed -i "s/mmcv_maximum_version = '.*'/mmcv_maximum_version = '2.3.0'/" mmdet/__init__.py
  # For Ampere arch, derictly use 'pip install -e .'
  # For Hopper arch, install mmdetection with the following command
  MMCV_CUDA_ARGS="-arch=sm_90" pip install --no-build-isolation -v -e .
  cd "$REWARD_SERVER_DIR"

  log "Downloading Mask2Former model weights..."
  MODEL_FILE="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
  if [ ! -f "$MODEL_FILE" ]; then
    wget -O "$MODEL_FILE" https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
  else
    log "Model weights already downloaded."
  fi

  conda deactivate
  log "Environment '${ENV_NAME}' setup complete."
fi