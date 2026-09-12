#!/usr/bin/env bash
set -e

# Locate this checkout independently of the caller's working directory.
AWM_DETECTED_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${BASE_DIR:-$AWM_DETECTED_ROOT}}"
export BASE_DIR="${BASE_DIR:-$PROJECT_ROOT}"
if [[ -z "${CONDA_PATH:-}" ]]; then
  if [[ -n "${CONDA_EXE:-}" && -f "$CONDA_EXE" ]]; then
    CONDA_PATH="$(cd -- "$(dirname -- "$CONDA_EXE")/.." && pwd)"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_PATH="$(conda info --base)"
  else
    CONDA_PATH="${HOME}/miniconda3"
  fi
fi
export CONDA_PATH
export PYTHON_VERSION="${PYTHON_VERSION:-3.10.16}"
export AWM_ENV_NAME="${AWM_ENV_NAME:-awm}"
export AWM_REWARD_ENV_NAME="${AWM_REWARD_ENV_NAME:-reward_server}"
# Do not replace the user's credentials with a placeholder.
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/.cache/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/.cache/pip}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

log() {
  echo ">> $*"
}

source_conda() {
  if [[ ! -f "$CONDA_PATH/etc/profile.d/conda.sh" ]]; then
    log "Conda profile missing: set CONDA_PATH or run setup/00_install_conda.sh"
    return 1
  fi
  source "$CONDA_PATH/etc/profile.d/conda.sh"
}

env_exists() {
  conda env list | awk -v name="$1" '$1 == name {found=1} END {exit !found}'
}

# Reuse only an unmodified checkout of the exact expected commit. Never reset,
# checkout or patch an existing dependency directory.
ensure_checkout() {
  local url="$1" tag="$2" revision="$3" destination="$4"
  if [[ ! -e "$destination" ]]; then
    git clone --branch "$tag" --depth=1 "$url" "$destination"
  fi
  if [[ ! -d "$destination/.git" ]] ||
     [[ "$(git -C "$destination" rev-parse HEAD)" != "$revision" ]] ||
     [[ -n "$(git -C "$destination" status --porcelain --untracked-files=normal)" ]]; then
    log "Dependency checkout is not a clean $tag at $revision: $destination"
    log "Move it aside or resolve its changes yourself; nothing has been overwritten."
    return 1
  fi
}
