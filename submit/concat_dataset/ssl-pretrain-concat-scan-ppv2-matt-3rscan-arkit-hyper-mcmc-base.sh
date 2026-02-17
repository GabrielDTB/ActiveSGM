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
conda activate sonata
echo "Running on $(hostname)"
cd /insait/qimaqi/workspace/SceneSplat_release/
export PYTHONPATH=./

gpu_num=1
batch_size=$((12*gpu_num))
batch_size_val=$((1*gpu_num))
batch_size_test=$((1*gpu_num))
num_worker=$((8*gpu_num))

python tools/ssl_pretrain.py \
  --config-file configs/concat_dataset/ssl-pretrain-concat-scan-ppv2-matt-3rscan-arkit-hyper-mcmc-base.py \
  --options \
    save_path=exp_runs/ssl_pretrainer/ssl-pretrain-scannet-all-base \
    batch_size=$batch_size batch_size_val=$batch_size_val \
    batch_size_test=$batch_size_test num_worker=$num_worker gpu_nums=$gpu_num \
  --num-gpus $gpu_num \