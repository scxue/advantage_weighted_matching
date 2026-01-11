set -x

DATA_PATH=/mnt/localssd/imagenet-64x64.zip
cp /sensei-fs-3/users/cge/data/imagenet-64x64.zip $DATA_PATH

torchrun --standalone --nproc_per_node=8 /mnt/localssd/colligo/contrib/Mori/mori/collections/experimental/edm/train.py --outdir=/mnt/localssd/edm_training-runs_bsl \
    --data=$DATA_PATH --cond=1 --arch=adm --duration=2500 --batch=1024 --lr=1e-4 --ema=50 --dropout=0.10 --augment=0 --fp16=1 --ls=100 --tick=200 --snap=10 --precond=edm_x_s