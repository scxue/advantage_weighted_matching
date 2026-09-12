# reward-server

Serves reward inference using an HTTP server.

## Geneval installation

Use a separate environment from AWM training:

```bash
# From the repository root; see setup/README.md for Conda/CUDA configuration.
bash setup/02_setup_env_reward_server.sh
```

The script uses `requirements-geneval.txt`, Torch 2.1.2+cu121, MMCV v1.7.2,
and MMDetection v2.28.2. The two detector repositories are checked out at fixed
commits under `third_party/`; existing mismatched/modified checkouts are not
overwritten. They are local dependencies, not submodules tracked by the parent
repository. No version-check patch is needed.

MMCV compilation needs a compatible CUDA toolkit. H100 is the default target
(`TORCH_CUDA_ARCH_LIST=9.0`); use `8.0` for A100.
Existing environments are skipped unless `AWM_UPDATE_ENV=1` is explicitly set.
The older `requirements.txt` is a legacy broad export, not this installation path.

Defaults locate the detector config under `third_party/mmdetection/configs/`
and its downloaded checkpoint in `reward-server/`. Override without editing code:

```bash
export GENEVAL_CONFIG_PATH=/path/to/mask2former_config.py
export GENEVAL_CKPT_PATH=/directory/containing/detector/weights
```

## Usage

### Geneval

```bash
conda activate "${AWM_REWARD_ENV_NAME:-reward_server}"
cd reward-server
GENEVAL_NUM_DEVICES=8 gunicorn --config gunicorn.conf.py "app_geneval:create_app()"
```

After the workers have loaded their models:

```bash
python test/test_geneval.py
```

The detector checkpoint filename is
`mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`. The setup script downloads
it only when absent, using a temporary file before publishing the final name.

### DeQA
If there's an error, please refer to [DeQA](https://github.com/zhiyuanyou/DeQA-Score ) to install DeQA's dependencies.
Start the server side:

```bash
cd reward-server/
conda deactivate
conda activate reward_server
gunicorn "app_deqa:create_app()"
```

After starting, you can run the client for testing:

```bash
python test/test_deqa.py
```
