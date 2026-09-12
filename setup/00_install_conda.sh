#!/bin/bash

# Source the common variables and functions
source "$(dirname "$0")/common_vars.sh"

log "Step 0: Installing Miniconda"

# Check if conda is already installed
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
  log "Miniconda already installed at ${CONDA_PATH}. Skipping installation."
else
  log "Downloading Miniconda..."
  if [ -e "$CONDA_PATH" ]; then
    log "CONDA_PATH exists but is not a Conda installation; refusing to overwrite it."
    exit 1
  fi
  INSTALLER="$(mktemp "$TMPDIR/awm-miniconda.XXXXXX.sh")"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$INSTALLER"

  log "Installing Miniconda to ${CONDA_PATH}..."
  bash "$INSTALLER" -b -p "$CONDA_PATH"

  log "Cleaning up Miniconda installer..."
  rm -- "$INSTALLER"

  log "Miniconda installation complete."
fi
