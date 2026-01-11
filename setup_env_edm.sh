#!/bin/bash

# This is the main script to set up the entire environment.
# It calls other scripts in a specific order.

# Exit on any error
set -e

SCRIPT_DIR="$(dirname "$0")/setup_edm"

# Make all scripts in the setup directory executable
chmod +x ${SCRIPT_DIR}/*.sh

# Run the setup scripts in order
${SCRIPT_DIR}/00_install_conda.sh
${SCRIPT_DIR}/01_setup_env_edm.sh

echo ""
echo "=========================================="
echo "All setup scripts executed successfully."
echo "=========================================="