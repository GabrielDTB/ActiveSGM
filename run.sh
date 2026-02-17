#!/bin/bash

export CMAKE_GENERATOR=Ninja
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

repo_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#python scripts/encode_labels.py \
#--labels_file ${repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/mp3d40.txt \
#--output ${repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/mp3d40_text_embeddings_siglip2.pt \
#--no_prefix

#scenes=( GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH)
#scenes=(office0 office1 office2 office3 office4 room0 room1 room2)


############################################
##   run evaluation need pc_segment.npy and pc_coord.npy
##   before running, rename the pc_segment_{label}.npy to pc_segment.npy, depends on which label you want to use
##   also need to match the config file for text_embedding in the zero_shot segmentation dict
#############################################



python tools/train.py \
    --config-file configs/concat_dataset/custom.py \
    --options skip_eval=False \
    save_feat=True \
    save_path=./eval/preprocess/MP3D/ActiveSGM_pred \
    weight=/mnt/Data4/scene_splat_7k/checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth \
    test_only=True \
    --num-gpus 1
