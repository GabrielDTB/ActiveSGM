#!/bin/bash
#SBATCH --output=logs/%A.log  # Output log file
#SBATCH --error=logs/%A.log   # Error log file
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=120:00:00

# conda activation
source ~/.bashrc
micromamba activate scene_splat

echo "Running on $(hostname)"
cd /path/to/workspace/SceneSplat
export PYTHONPATH=./

gpu_num=$((4*$SLURM_NNODES))
batch_size=$((2*gpu_num))
batch_size_val=$((2*gpu_num))
batch_size_test=$((1*gpu_num))
num_worker=$((8*gpu_num))

python tools/train.py \
  --config-file configs/scannet/lang-pretrain-scannet-mcmc-wo-normal-contrastive.py \
  --options \
    save_path=exp_runs/lang_pretrainer/lang-pretrain-scannet-mcmc-wo-normal-contrastive \
    batch_size=$batch_size batch_size_val=$batch_size_val \
    batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
  --num-gpus $gpu_num \