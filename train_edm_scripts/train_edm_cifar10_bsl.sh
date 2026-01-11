# Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
torchrun --standalone --nproc_per_node=8 /mnt/localssd/colligo/contrib/Mori/mori/collections/experimental/edm/train.py --outdir=/mnt/localssd/edm_training-runs_bsl \
    --data=/mnt/localssd/colligo/contrib/Mori/mori/collections/experimental/edm/datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --snap=10