# Installation and portable paths

The setup targets Linux x86-64, Python 3.10 and NVIDIA GPUs. It creates separate
`awm` (training) and `reward_server` (Geneval detector) Conda environments.
The scripts locate this checkout from their own location, not the working directory.

## Setup

```bash
# Optional: point at an existing Conda installation.
export CONDA_PATH=/path/to/miniconda3
# CUDA compiler matching the reward environment's cu121 Torch is needed for MMCV.
export CUDA_HOME=/path/to/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0  # H100; use 8.0 for A100
bash setup_env.sh
```

If Conda is absent, the bootstrap downloads the Linux x86-64 Miniconda installer
and installs it at CONDA_PATH (default: ~/miniconda3). Existing directories are
never overwritten. If channel terms require acceptance, resolve that explicitly;
the scripts do not accept terms on your behalf.

Existing environments are left unchanged by default. To deliberately update:

```bash
AWM_UPDATE_ENV=1 bash setup_env.sh
```

Do not update an environment while a training/reward job uses it. For a clean
validation use new environment names:
```bash
AWM_ENV_NAME=awm_clean AWM_REWARD_ENV_NAME=reward_clean bash setup_env.sh
```

The training scripts also honor these environment-name variables.
Optional components are opt-in:

- `AWM_INSTALL_OCR=1`: install PaddleOCR/Paddle GPU and download its English model.
- `AWM_INSTALL_FLASH_ATTN=1`: compile FlashAttention; not needed by default DDP.
- `AWM_INSTALL_IMAGEREWARD=1`: install the optional ImageReward scorer.

PickScore uses the core Transformers dependencies and downloads its model when used.

## Dependency sources

- `advantage_weighted_matching/requirements.txt` is the single source for main
  direct dependencies. setup.py reads it; optional installations use it as
  constraints. Do not add later version overrides in shell scripts.
- Main core: Torch 2.6.0, TorchVision 0.21.0, TorchAudio 2.6.0,
  Accelerate 1.9.0, Transformers 4.54.0, Tokenizers 0.21.4,
  Hub 0.36.2, Diffusers 0.33.1, PEFT 0.10.0, W&B 0.29.0.
- xformers 0.0.29.post3 requires Torch 2.6.0. FlashAttention/MMCV source builds
  also need compatible CUDA compilers.
- `reward-server/requirements-geneval.txt` describes the independent detector
  stack: Torch 2.1.2+cu121, Transformers 4.38.2, Tokenizers 0.15.2.
  Do not install training requirements into this environment.
- The old broad `reward-server/requirements.txt` is a legacy environment export,
  not the supported Geneval installation entry point.

MMCV v1.7.2 and MMDetection v2.28.2 are cloned into third_party at fixed commits.
An existing checkout must match and be clean; the installer stops instead of
resetting it. Both directories are ignored by the parent repository. No MMCV
version-guard patch is needed with this pair.

## Runtime paths

| Variable | Default | Meaning |
| --- | --- | --- |
| AWM_OUTPUT_DIR | checkout/outputs | Checkpoints under task/node-count/timestamp; tracker logs under logs |
| AWM_DATA_DIR | checkout/advantage_weighted_matching/dataset | Root containing geneval, ocr and pickscore |
| CONDA_PATH | Detected Conda base, otherwise ~/miniconda3 | Conda installation |
| AWM_ENV_NAME | awm | Training environment name |
| AWM_REWARD_ENV_NAME | reward_server | Geneval environment name |
| TMPDIR | checkout/.cache/tmp | Build/download temporary files |
| PIP_CACHE_DIR | checkout/.cache/pip | pip cache |

Dataset/output roots expand ~ and are resolved to absolute paths when configs
are constructed. Relative environment overrides are relative to the caller's
working directory. Defaults are independent of that directory.

```bash
export AWM_OUTPUT_DIR=/your/storage/awm
export AWM_DATA_DIR=/your/datasets
```

Explicit `--config.save_dir=/path/to/run` and `--config.dataset=/path/to/task`
still override config defaults. Full resume restores the original save directory
from checkpoint metadata; changing AWM_OUTPUT_DIR does not relocate an old run.
Credentials are optional environment inputs and are never replaced with dummy values.

## Validation

After activating the main environment:

```bash
python -m pip check
cd advantage_weighted_matching
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v
```

Offline W&B tests need local IPC sockets but no cloud credentials.
