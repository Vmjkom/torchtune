#!/bin/bash
source /opt/miniconda3/bin/activate pytorch
# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export TRITON_CACHE_DIR="/tmp/$(whoami)-triton-cache-$SLURM_NODEID" #This will otherwise be directed to your home directory in /.triton
# Start conda environment inside the container
$WITH_CONDA
# Set interfaces to be used by RCCL.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3


# The usual PyTorch initialisations (also needed on NVIDIA)
# Note that since we fix the port ID it is not possible to run, e.g., two
# instances via this script using half a node each.
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_GPUS_ON_NODE
export OMP_NUM_THREADS=1

tune run \
    --nproc-per-node=$SLURM_GPUS_ON_NODE \
    --standalone \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    full_finetune_distributed \
    --config ./recipes/configs/gemma2/27B_full.yaml