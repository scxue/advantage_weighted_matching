#!/bin/bash

# Source the common variables and functions
source "$(dirname "$0")/common_vars.sh"
source_conda

ENV_NAME="edm"
log "Step 1: Setting up Conda environment: ${ENV_NAME}"

# Check if environment already exists
if conda info --envs | grep -q "^${ENV_NAME}\s"; then
  log "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
  log "Accepting Conda Terms of Service..."
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

  log "Creating Conda environment '${ENV_NAME}'"
  cd "${PROJECT_ROOT}/collections/experimental/edm"
  conda env create -f environment.yml -n "$ENV_NAME" --yes

  log "Activating environment and installing packages for '${ENV_NAME}'..."
  conda activate "$ENV_NAME"

  # Create directories if they don't exist
  log "Creating data directories..."
  mkdir -p "${PROJECT_ROOT}/collections/experimental/edm/downloads/cifar10"
  mkdir -p "${PROJECT_ROOT}/collections/experimental/edm/datasets"
  mkdir -p "${PROJECT_ROOT}/collections/experimental/edm/fid-refs"

  # Define file paths
  CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
  CIFAR_DOWNLOAD_PATH="${PROJECT_ROOT}/collections/experimental/edm/downloads/cifar10/cifar-10-python.tar.gz"
  CIFAR_ZIP_PATH="${PROJECT_ROOT}/collections/experimental/edm/datasets/cifar10-32x32.zip"
  CIFAR_FID_REF_PATH="${PROJECT_ROOT}/collections/experimental/edm/fid-refs/cifar10-32x32.npz"

  # Download CIFAR-10 dataset if it doesn't exist
  if [ -f "$CIFAR_DOWNLOAD_PATH" ]; then
    log "CIFAR-10 dataset already downloaded. Skipping download."
  else
    log "Downloading CIFAR-10 dataset..."
    wget -O "$CIFAR_DOWNLOAD_PATH" "$CIFAR_URL"
  fi

  # Run dataset_tool.py
  log "Running dataset_tool.py..."
  python dataset_tool.py --source="$CIFAR_DOWNLOAD_PATH" --dest="$CIFAR_ZIP_PATH"

  # Run fid.py
  log "Running fid.py to generate reference..."
  python fid.py ref --data="$CIFAR_ZIP_PATH" --dest="$CIFAR_FID_REF_PATH"

  # # Define file paths current imagenet preprocessing is not working; it will get stuck
  # IMAGENET_DOWNLOAD_PATH="/sensei-fs-3/users/cge/zslbp/data/imagenet/train"
  # IMAGENET_ZIP_PATH="${PROJECT_ROOT}/collections/experimental/edm/datasets/imagenet-64x64.zip"
  # IMAGENET_FID_REF_PATH="${PROJECT_ROOT}/collections/experimental/edm/fid-refs/imagenet-64x64.npz"

  # # Run dataset_tool.py
  # log "Running dataset_tool.py..."
  # python dataset_tool.py --source="$IMAGENET_DOWNLOAD_PATH" --dest="$IMAGENET_ZIP_PATH"

  # # Run fid.py
  # log "Running fid.py to generate reference..."
  # python fid.py ref --data="$IMAGENET_ZIP_PATH" --dest="$IMAGENET_FID_REF_PATH"


  conda deactivate
  log "Environment '${ENV_NAME}' setup complete."
fi