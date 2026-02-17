#!/bin/bash

#python scripts/preprocess_gs.py \
#       --input /mnt/Data2/liyan/ActiveSGM/results/MP3D_5k/YmJkqBEsHnH/ActiveSem/run_0/splatam/rgb_GS_4900.ply \
#       --output /mnt/Data3/liyan/preprocess/MP3D/YmJkqBEsHnH/ActiveSGM/




#python -u preprocess_matterport3d_gs.py \
#    --pc_root  /path/to/ptv3_preprocessed/matterport3d \
#    --gs_root  /path/to/gaussian_world/matterport3d_region_mcmc_3dgs \
#    --output_root /path/to/gaussian_world/preprocessed/matterport3d_region_mcmc_3dgs \
#    --num_workers 8 \
#    --feat_root /path/to/gaussian_world/matterport3d_region_mcmc_3dgs/language_features_siglip2
#
#export PYTHONPATH=$PWD:$PYTHONPATH
#export CUDA_HOME="$CONDA_PREFIX"
#export PATH="$CUDA_HOME/bin:$PATH"
#export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"
#
#export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
#export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
#export CUDAHOSTCXX="$CXX"
#export TORCH_CUDA_ARCH_LIST="80;86;89;90"
#export FLASH_ATTENTION_FORCE_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"


#python tools/train.py \
#    --config-file configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
#    --options skip_eval=True \
#    save_feat=True\
#    save_path=/mnt/Data4/scene_splat_7k/eval/MP3D/ \
#    weight=/mnt/Data4/scene_splat_7k/checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth \
#    test_only=True \
#    --num-gpus 1