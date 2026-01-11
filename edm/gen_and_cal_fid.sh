# Generate 50000 images and save them as fid-tmp/*/*.png

source /home/colligo/miniconda3/etc/profile.d/conda.sh && conda activate edm

rm -rf /mnt/localssd/fid-tmp

ITER_NUM_LIST=(000502 001004 001506 002007 002509 003011 003512 004014)

for ITER_NUM in ${ITER_NUM_LIST[@]}; do
    cd /mnt/localssd/colligo/contrib/Mori/mori/collections/experimental/edm/ && 
    torchrun --standalone --nproc_per_node=8 generate.py --outdir=/mnt/localssd/fid-${ITER_NUM} --seeds=0-49999 --subdirs \
        --network=/mnt/localssd/edm_training-runs_bsl/00000-cifar10-32x32-cond-ddpmpp-edm-gpus8-batch512-fp32/network-snapshot-${ITER_NUM}.pkl 

    # Calculate FID
    cd /mnt/localssd/colligo/contrib/Mori/mori/collections/experimental/edm/ && 
    torchrun --standalone --nproc_per_node=8 fid.py calc --images=/mnt/localssd/fid-${ITER_NUM} \
        --ref=/mnt/localssd/colligo/contrib/Mori/mori/collections/experimental/edm/fid-refs/cifar10-32x32.npz \
        &> ${ITER_NUM}_fid_log.log
done


