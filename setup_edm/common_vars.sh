#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Shared Variables ---

# Base directory for all operations
export BASE_DIR="/mnt/localssd"

# Project root directory
export PROJECT_ROOT="${BASE_DIR}/colligo/contrib/Mori/mori"

# Miniconda installation path
export CONDA_PATH="/home/colligo/miniconda3"

# Python version for the environments
export PYTHON_VERSION="3.10.16"

export HUGGINGFACE_TOKEN="your_hf_token"

# --- Helper Functions ---

# A simple logging function to make output clearer
log() {
  echo "--------------------------------------------------"
  echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] $1"
  echo "--------------------------------------------------"
}

# Function to source conda profile if it exists
source_conda() {
  if [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
    log "Sourcing conda profile..."
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
  else
    log "ERROR: Conda profile not found at ${CONDA_PATH}/etc/profile.d/conda.sh"
    exit 1
  fi
}