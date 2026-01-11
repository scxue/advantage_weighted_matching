# reward-server

Serves reward inference using an HTTP server.

## Install

### GenEval

```bash
# First
conda create -n reward_server python=3.10.16
# Then
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
# Then
pip install -r requirements.txt
```

Then install mmdet:

```bash
mim install mmcv-full mmengine
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
# Modify mmdet/__init__.py: set mmcv_maximum_version = '2.3.0'
pip install -e .
```

Then download mask2former:

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth -O "$1/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
```

Modify `MY_CONFIG_PATH` and `MY_CKPT_PATH` in `reward-server/reward_server/gen_eval.py` to your own paths.

## Usage

### GenEval

Start the server side:

```bash
cd reward-server/
conda deactivate
conda activate reward_server
gunicorn "app_geneval:create_app()"
```

You must modify `gunicorn.conf.py` to change the number of GPUs.

After starting, you can run the client for testing:

```bash
python test/test_geneval.py
```

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
