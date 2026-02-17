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

python -m scripts.test_gsbki \
  --data_dir /mnt/Data3/liyan/preprocess/MP3D/ActiveSGM \
  --scene GdvgFV5R1Z5 \
  --embedding_model ${repo_root}/autoencoder/ckpt/matterport/best_ckpt.pth \
  --text_embeddings ${repo_root}/pointcept/datasets/preprocessing/matterport3d/meta_data/matterport21_text_embeddings_siglip2.pt \
  --latent_dim 16 \
  --device cuda \
  --downsample 0


