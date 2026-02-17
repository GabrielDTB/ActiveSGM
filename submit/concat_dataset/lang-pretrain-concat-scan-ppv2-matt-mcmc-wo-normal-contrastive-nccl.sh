#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_h100
#SBATCH -N 4
#SBATCH --ntasks-per-node=4   
#SBATCH --cpus-per-task=16    
#SBATCH --gpus-per-node=4
#SBATCH --time=120:00:00

# conda activation
source ~/.bashrc
micromamba activate scene_splat

module purge
module load 2023
module load CUDA/12.4.0

echo "Running on $(hostname) | $(date)"
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR=${MASTER_NODE}.local.snellius.surf.nl
MASTER_PORT=29501
echo "Master node: $MASTER_NODE"
echo "Master address: $MASTER_ADDR"

# NCCL configuration
export NCCL_DEBUG=WARN       # INFO, for debugging
export NCCL_DEBUG_SUBSYS=INIT,COLL   # ALL, for debugging
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eno2np0
export NCCL_SOCKET_TIMEOUT=1800

export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_BLOCKING_WAIT=1            
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=1800  # 30 min, in case

WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

cd /path/to/workspace/SceneSplat
export PYTHONPATH=./   

gpu_num=$((4*$SLURM_NNODES))
batch_size=$((2*gpu_num))
batch_size_val=$((1*gpu_num))
batch_size_test=$((1*gpu_num))
num_worker=$((8*gpu_num))  

# Important! let srun handle task distribution
srun python tools/train.py \
        --config-file configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
        --options save_path=exp_runs/lang_pretrainer/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive \
        batch_size=$batch_size batch_size_val=$batch_size_val \
        batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
        --num-gpus $gpu_num \
        --multi_node

