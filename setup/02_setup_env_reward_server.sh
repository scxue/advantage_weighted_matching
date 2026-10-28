#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_vars.sh"
source_conda

configure_cuda_build() {
  local nvcc compiler_root directory includes="" libraries="" jobs runtime_header=0 crt_header=0
  if [[ -z "${CUDA_HOME:-}" ]]; then
    nvcc=$(command -v nvcc) || { log "Set CUDA_HOME to a CUDA toolkit with nvcc."; return 1; }
    export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$nvcc")")")"
  fi
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    log "CUDA_HOME has no executable bin/nvcc: $CUDA_HOME"
    return 1
  fi
  export CUDA_HOME="$(cd "$CUDA_HOME" && pwd -P)"
  nvcc=$(readlink -f "$CUDA_HOME/bin/nvcc")
  compiler_root=$(dirname "$(dirname "$nvcc")")
  # Conda may split runtime headers and compiler CRT headers. Also handle a
  # CUDA_HOME whose bin/ is a symlink into a separate toolkit prefix.
  for directory in "$CUDA_HOME/include" "$CUDA_HOME/targets/x86_64-linux/include" \
      "$compiler_root/include" "$compiler_root/targets/x86_64-linux/include"; do
    if [[ -d "$directory" && ":$includes:" != *":$directory:"* ]]; then
      includes="${includes:+$includes:}$directory"
      if [[ -f "$directory/cuda_runtime_api.h" ]]; then runtime_header=1; fi
      if [[ -f "$directory/crt/host_defines.h" ]]; then crt_header=1; fi
    fi
  done
  for directory in "$CUDA_HOME/lib64" "$CUDA_HOME/targets/x86_64-linux/lib" \
      "$compiler_root/lib64" "$compiler_root/targets/x86_64-linux/lib"; do
    if [[ -d "$directory" && ":$libraries:" != *":$directory:"* ]]; then
      libraries="${libraries:+$libraries:}$directory"
    fi
  done
  if (( !runtime_header || !crt_header )); then
    log "CUDA runtime/CRT headers missing under $CUDA_HOME or $compiler_root; checked: $includes"
    return 1
  fi
  export PATH="$CUDA_HOME/bin:$PATH"
  export CPATH="$includes${CPATH:+:$CPATH}"
  if [[ -n "$libraries" ]]; then
    export LIBRARY_PATH="$libraries${LIBRARY_PATH:+:$LIBRARY_PATH}"
  fi
  # Leave LD_LIBRARY_PATH untouched: these settings are for compilation only.
  log "CUDA_HOME=$CUDA_HOME; compiler=$nvcc; include paths=$includes"
  "$CUDA_HOME/bin/nvcc" --version || return 1
  if ! printf '#include <cuda_runtime_api.h>\n' | "${CXX:-c++}" -x c++ -fsyntax-only -; then
    log "CUDA C++ header check failed; check CUDA_HOME/CPATH and the C++ compiler."
    return 1
  fi
  ninja --version || { log "ninja is required to build MMCV."; return 1; }
  if [[ -z "${MAX_JOBS:-}" ]]; then
    jobs=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
    if (( jobs > 16 )); then jobs=16; fi
    export MAX_JOBS="$jobs"
  fi
  if [[ ! "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    log "MAX_JOBS must be a positive integer."
    return 1
  fi
}

if env_exists "$AWM_REWARD_ENV_NAME"; then
  if [[ "${AWM_UPDATE_ENV:-0}" != 1 ]]; then
    log "Environment $AWM_REWARD_ENV_NAME exists; leaving it unchanged. Set AWM_UPDATE_ENV=1 to update."
    exit 0
  fi
else
  conda create -n "$AWM_REWARD_ENV_NAME" python="$PYTHON_VERSION" --yes
fi
conda activate "$AWM_REWARD_ENV_NAME"
# Do not inherit main-environment constraints into the older detector stack.
export PIP_CONSTRAINT="$PROJECT_ROOT/reward-server/requirements-geneval.txt"
python -m pip install pip==24.3.1 setuptools==60.2.0 wheel
python -m pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$PIP_CONSTRAINT"

configure_cuda_build
python - <<'PY'
import os
import re
import subprocess
import torch

output = subprocess.check_output([os.path.join(os.environ["CUDA_HOME"], "bin/nvcc"), "--version"], text=True)
match = re.search(r"release (\d+)\.(\d+)", output)
if not match or not torch.version.cuda:
    raise SystemExit("Cannot determine nvcc/Torch CUDA versions; a CUDA-enabled Torch is required.")
compiler = match.group(1, 2)
runtime = tuple(torch.version.cuda.split(".")[:2])
if compiler[0] != runtime[0]:
    raise SystemExit(f"CUDA major version mismatch: nvcc={'.'.join(compiler)}, Torch={torch.version.cuda}")
if compiler != runtime:
    print(f"Warning: nvcc={'.'.join(compiler)}, Torch={torch.version.cuda}; matching versions are recommended.")
PY

mkdir -p "$PROJECT_ROOT/third_party"
ensure_checkout https://github.com/open-mmlab/mmcv.git v1.7.2 \
  4c01b026f0afa5a91a5f54aea313788da1e40f95 "$PROJECT_ROOT/third_party/mmcv"
ensure_checkout https://github.com/open-mmlab/mmdetection.git v2.28.2 \
  e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1 "$PROJECT_ROOT/third_party/mmdetection"

# Defaults to H100. For A100 set TORCH_CUDA_ARCH_LIST=8.0; other GPUs need
# the appropriate architecture and a toolkit compatible with Torch cu121.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
MMCV_WITH_OPS=1 python -m pip install --no-build-isolation -v -e "$PROJECT_ROOT/third_party/mmcv"
python -m pip install --no-build-isolation -v -e "$PROJECT_ROOT/third_party/mmdetection"
# v1.7.2 satisfies v2.28.2's original MMCV check; no source patch is needed.

REWARD_SERVER_DIR="$PROJECT_ROOT/reward-server"
MODEL_FILE="$REWARD_SERVER_DIR/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
if [[ ! -f "$MODEL_FILE" ]]; then
  download="$(mktemp "$REWARD_SERVER_DIR/.mask2former-download.XXXXXX")"
  wget -O "$download" https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
  mv -- "$download" "$MODEL_FILE"
fi
python -m pip check
conda deactivate
log "Geneval reward environment setup finished."
