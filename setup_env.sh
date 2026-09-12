#!/usr/bin/env bash
set -euo pipefail
AWM_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/setup" && pwd)"
bash "$AWM_SETUP_DIR/00_install_conda.sh"
bash "$AWM_SETUP_DIR/01_setup_env_awm.sh"
bash "$AWM_SETUP_DIR/02_setup_env_reward_server.sh"
echo "Setup complete. Existing environments were not modified unless AWM_UPDATE_ENV=1."
