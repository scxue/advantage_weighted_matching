#!/bin/bash

# Source the common variables and functions
source "$(dirname "$0")/common_vars.sh"

log "Step 0: Installing Miniconda"

# Check if conda is already installed
if [ -d "$CONDA_PATH" ]; then
  log "Miniconda already installed at ${CONDA_PATH}. Skipping installation."
else
  log "Downloading Miniconda..."
  cd "$BASE_DIR"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh

  log "Installing Miniconda to ${CONDA_PATH}..."
  bash "${BASE_DIR}/Miniconda3.sh" -b -p "$CONDA_PATH"

  log "Cleaning up Miniconda installer..."
  rm "${BASE_DIR}/Miniconda3.sh"

  log "Miniconda installation complete."
fi