#!/bin/bash
set -euo pipefail

########################################
# 0. 基本配置
########################################

BASE_DIR="/opt/tiger/helloworld"
ENV_ROOT="/opt/tiger/envs"
HDFS_BASE="/mnt/hdfs/__MERLIN_USER_DIR__"

CONDA_PACK_DIR="${HDFS_BASE}/conda_packs"
MODELS_HDFS_DIR="${HDFS_BASE}/models"

FLOW_ENV_NAME="flow_grpo"
REWARD_ENV_NAME="reward_server"

MASK2FORMER_NAME="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
MASK2FORMER_HDFS_PATH="${MODELS_HDFS_DIR}/mask2former/${MASK2FORMER_NAME}"
MASK2FORMER_LOCAL_PATH="${BASE_DIR}/reward-server/${MASK2FORMER_NAME}"

PADDLEOCR_HDFS_DIR="${MODELS_HDFS_DIR}/paddleocr"
PADDLEOCR_LOCAL_DIR="${HOME}/.paddleocr"

log() {
  echo "[`date +"%Y-%m-%d %H:%M:%S"`] $*"
}

########################################
# 1. 网络设置（如果有的话）
########################################

NETWORK_SH="${BASE_DIR}/network.sh"
if [ -f "${NETWORK_SH}" ]; then
  log "Sourcing network.sh: ${NETWORK_SH}"
  # shellcheck disable=SC1090
  source "${NETWORK_SH}"
else
  log "network.sh not found at ${NETWORK_SH}, skipping."
fi

########################################
# 2. 解压 Conda 环境 + conda-unpack
########################################

setup_env_from_pack() {
  local env_name="$1"
  local env_dir="${ENV_ROOT}/${env_name}"
  local tar_path="${CONDA_PACK_DIR}/${env_name}.tar.gz"

  log "========== Setting up env: ${env_name} =========="

  if [ ! -f "${tar_path}" ]; then
    log "ERROR: Conda pack tar not found: ${tar_path}"
    exit 1
  fi

  # 如果 env 目录下已经有 python，就认为已经解压过
  if [ -x "${env_dir}/bin/python" ]; then
    log "Env ${env_name} already exists at ${env_dir}, skipping tar extract."
  else
    log "Creating env directory: ${env_dir}"
    mkdir -p "${env_dir}"

    log "Extracting ${tar_path} to ${env_dir} ..."
    tar xzf "${tar_path}" -C "${env_dir}"

    log "Running conda-unpack in ${env_dir} ..."
    "${env_dir}/bin/conda-unpack"
  fi

  # 简单测试一下
  log "Testing Python in env ${env_name} ..."
  "${env_dir}/bin/python" -c "import sys; print('Python OK in ${env_name}', sys.version)" || {
    log "ERROR: Python test failed in env ${env_name}"
    exit 1
  }

  log "Env ${env_name} setup done."
}

# flow_grpo
setup_env_from_pack "${FLOW_ENV_NAME}"

# reward_server
setup_env_from_pack "${REWARD_ENV_NAME}"

########################################
# 3. 同步 Mask2Former 模型到原始路径
########################################

log "========== Syncing Mask2Former model =========="

if [ ! -f "${MASK2FORMER_HDFS_PATH}" ]; then
  log "ERROR: Mask2Former ckpt not found in HDFS: ${MASK2FORMER_HDFS_PATH}"
  log "Please copy it first, e.g.:"
  log "  mkdir -p ${MODELS_HDFS_DIR}/mask2former"
  log "  cp /opt/tiger/helloworld/reward-server/${MASK2FORMER_NAME} ${MODELS_HDFS_DIR}/mask2former/"
  exit 1
fi

mkdir -p "$(dirname "${MASK2FORMER_LOCAL_PATH}")"

if [ -f "${MASK2FORMER_LOCAL_PATH}" ]; then
  log "Local Mask2Former exists at ${MASK2FORMER_LOCAL_PATH}, skipping copy."
else
  log "Copying Mask2Former from HDFS mount to local path ..."
  cp "${MASK2FORMER_HDFS_PATH}" "${MASK2FORMER_LOCAL_PATH}"
fi

ls -lh "${MASK2FORMER_LOCAL_PATH}" || true
log "Mask2Former sync done."

########################################
# 4. 同步 PaddleOCR 缓存到原始路径 (~/.paddleocr)
########################################

log "========== Syncing PaddleOCR cache =========="

if [ ! -d "${PADDLEOCR_HDFS_DIR}" ]; then
  log "ERROR: PaddleOCR cache dir not found in HDFS: ${PADDLEOCR_HDFS_DIR}"
  log "Please copy it first, e.g.:"
  log "  mkdir -p ${MODELS_HDFS_DIR}/paddleocr"
  log "  cp -r ~/.paddleocr/* ${MODELS_HDFS_DIR}/paddleocr/"
  exit 1
fi

mkdir -p "${PADDLEOCR_LOCAL_DIR}"

log "Copying PaddleOCR cache from HDFS mount to ${PADDLEOCR_LOCAL_DIR} ..."
cp -r "${PADDLEOCR_HDFS_DIR}/"* "${PADDLEOCR_LOCAL_DIR}/" || true

log "PaddleOCR local structure:"
ls -R "${PADDLEOCR_LOCAL_DIR}" || true

log "PaddleOCR sync done."

########################################
# 5. 同步 mmcv 和 mmdet 压缩包到 reward_server 目录并解压
########################################

log "========== Syncing mmcv and mmdet tarballs to reward_server =========="

# 指定的存放路径
REWARD_SERVER_DIR="${BASE_DIR}/reward-server"

# mmcv 和 mmdet 文件的 HDFS 路径
MMCV_HDFS_PATH="${HDFS_BASE}/mm_series_built/mmcv_built.tar.gz"
MMDET_HDFS_PATH="${HDFS_BASE}/mm_series_built/mmdet_built.tar.gz"

# reward_server 目标目录
MMCV_LOCAL_PATH="${REWARD_SERVER_DIR}/mmcv_built.tar.gz"
MMDET_LOCAL_PATH="${REWARD_SERVER_DIR}/mmdet_built.tar.gz"

# 创建目标目录
mkdir -p "${REWARD_SERVER_DIR}"

# 同步 mmcv 和 mmdet 文件
log "Copying mmcv_built.tar.gz to ${REWARD_SERVER_DIR} ..."
cp "${MMCV_HDFS_PATH}" "${MMCV_LOCAL_PATH}" || {
  log "ERROR: Failed to copy mmcv_built.tar.gz"
  exit 1
}

log "Copying mmdet_built.tar.gz to ${REWARD_SERVER_DIR} ..."
cp "${MMDET_HDFS_PATH}" "${MMDET_LOCAL_PATH}" || {
  log "ERROR: Failed to copy mmdet_built.tar.gz"
  exit 1
}

# 解压 mmcv 和 mmdet 压缩包
log "Extracting mmcv_built.tar.gz ..."
tar xzf "${MMCV_LOCAL_PATH}" -C "${REWARD_SERVER_DIR}" || {
  log "ERROR: Failed to extract mmcv_built.tar.gz"
  exit 1
}

log "Extracting mmdet_built.tar.gz ..."
tar xzf "${MMDET_LOCAL_PATH}" -C "${REWARD_SERVER_DIR}" || {
  log "ERROR: Failed to extract mmdet_built.tar.gz"
  exit 1
}

log "mmcv and mmdet sync and extraction done."

########################################
# 6. 完成
########################################

echo ""
log "=========================================="
log "Node setup complete."
log "flow_grpo env:      ${ENV_ROOT}/${FLOW_ENV_NAME}"
log "reward_server env:  ${ENV_ROOT}/${REWARD_ENV_NAME}"
log "Mask2Former ckpt:   ${MASK2FORMER_LOCAL_PATH}"
log "PaddleOCR cache:    ${PADDLEOCR_LOCAL_DIR}"
log "mmcv tarball:       ${MMCV_LOCAL_PATH}"
log "mmdet tarball:      ${MMDET_LOCAL_PATH}"
log "=========================================="
echo ""