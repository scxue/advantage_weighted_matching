#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Source Common Variables & Functions ---
# Assuming this script is run from the project root
source "$(dirname "$0")/../setup/common_vars.sh"

# --- MINIMAL MODIFICATION START: Log into services ---
# Activate a base environment to use the login CLIs.
source_conda
conda activate awm
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
fi
# if [ -n "$WANDB_API_KEY" ]; then
#     # The WANDB_API_KEY env var is used automatically by the login command.
#     wandb login "$WANDB_API_KEY" --relogin --host="$WANDB_HOST"
# fi
conda deactivate
# --- MINIMAL MODIFICATION END ---

# --- Log Directory with Timestamp ---
# MODIFICATION: Generate a timestamp for this specific run.
TIMESTAMP=$(date +'%Y%m%d_%H%M%S') # Format: YearMonthDay_HourMinuteSecond, e.g., 20250903_021824

# MODIFICATION: Create a unique log directory for this run.
LOG_DIR="${BASE_DIR}/logs/run_${TIMESTAMP}" 
mkdir -p "$LOG_DIR"
REWARD_SERVER_LOG="${LOG_DIR}/reward_server.log"
TRAINING_LOG="${LOG_DIR}/training.log"

# --- Announce Log Location ---
# ADDITION: Inform the user where to find the logs for this run.
log "This run's logs will be saved in: ${LOG_DIR}"


# # --- Cleanup Function ---
# # This function will be called when the script receives an exit signal.
# cleanup() {
#     log "Caught signal... Cleaning up background processes."
#     # Check if the REWARD_SERVER_PID variable is set
#     if [ -n "$REWARD_SERVER_PID" ]; then
#         # Kill the process group to ensure all child processes of gunicorn are also terminated
#         log "Stopping Reward Server (PID: $REWARD_SERVER_PID)..."
#         kill -SIGTERM -- "-$REWARD_SERVER_PID" || true # The 'true' prevents the script from exiting if the process is already dead
#         wait "$REWARD_SERVER_PID" 2>/dev/null
#     fi
#     log "Cleanup complete. Exiting."
#     exit 0
# }

# # --- Trap Exit Signals ---
# # It's crucial to trap signals to run the cleanup function.
# # SIGINT is for Ctrl+C, SIGTERM is for the 'kill' command.
# trap cleanup SIGINT SIGTERM EXIT

# # --- Process 1: Start Reward Server in the Background ---
# log "Starting Reward Server in the background..."
# log "Output will be logged to: ${REWARD_SERVER_LOG}"

# # Use a subshell to activate conda env and run the server.
# # 'setsid' creates a new session, making the process the group leader, which helps in killing it cleanly.
# (
#   set -e
#   source_conda
#   conda activate reward_server
#   cd "${PROJECT_ROOT}/collections/experimental/reward-server/"
#   # Use setsid to ensure we can kill the whole process group
#   setsid gunicorn "app_geneval:create_app()"
# ) > "$REWARD_SERVER_LOG" 2>&1 &

# REWARD_SERVER_PID=$!
# log "Reward Server started with PID: ${REWARD_SERVER_PID}"
# # Give it a moment to initialize
# sleep 5

# --- Process 2: Start Training in the Foreground ---
log "Starting Accelerate training process..."
log "Output will be logged to: ${TRAINING_LOG}"

# This command runs in the foreground. The script will wait here until it completes or is terminated.
(
  set -e
  source_conda
  conda activate awm
  # Assuming the paths in the command are relative to the project root
  cd "${PROJECT_ROOT}/advantage_weighted_matching"
  echo "RANK: $RANK"
  echo "MASTER_ADDR: $MASTER_ADDR"
  echo "MASTER_PORT: $MASTER_PORT"
  accelerate launch \
    --config_file "${PROJECT_ROOT}/advantage_weighted_matching/scripts/accelerate_configs/multi_node.yaml" \
    --num_machines 4 --num_processes 32 \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    "${PROJECT_ROOT}/advantage_weighted_matching/scripts/train_sd3_awm.py" \
    --config "${PROJECT_ROOT}/advantage_weighted_matching/config/dgx_awm.py:pickscore_sd3_no_cfg_4nodes"
) 2>&1 | tee "$TRAINING_LOG"

# The 'trap' will handle cleanup when the script exits after this point.