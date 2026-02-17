#!/bin/bash

export CMAKE_GENERATOR=Ninja
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

repo_root='/mnt/Data2/liyan/SceneSplat'

DATASET_PATH='/mnt/Data3/liyan/matterport/'
DATASET_NAME='matterport'

#python train.py --dataset_path ${DATASET_PATH} --dataset_name ${DATASET_NAME}
python test.py --dataset_path ${DATASET_PATH} --dataset_name ${DATASET_NAME}