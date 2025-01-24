#!/bin/bash
#SBATCH -J torchtune
#SBATCH --cpus-per-task=7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --account=project_462000615
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eox pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -f -s logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

#MODULES
module purge
module load LUMI


#DISTRIBUTED
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

#LOGGING
export PYTHONWARNINGS="ignore"

#export HF_TOKEN="TOKEN HERE FROM HF" #This is commented out because i have the token in my home directory
export CONFIG=apps/main/configs/debug.yaml
echo "CONF" $CONFIG

#Debugging
#export NCCL_DEBUG=INFO
#export TRITON_DEBUG=1
export NCCL_DEBUG_FILE=logs/nccl-debug-$SLURM_JOB_ID.log
#export TORCH_SHOW_CPP_STACKTRACES=1

#Masks for binding cpu cores to right numa nodes and therefor to right gpu's
#c=fe
#MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

#SINGULARITY
CONTAINER=/appl/local/containers/tested-containers/lumi-pytorch-rocm-6.2.3-python-3.12-pytorch-v2.5.1-dockerhash-4d68a29c872e.sif
echo "CONTAINER" $CONTAINER
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export PWD=(`pwd -P`)

srun --label \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    ./slurm_scripts/launch.sh


echo "END $SLURM_JOBID: $(date)"