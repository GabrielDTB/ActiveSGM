#!/bin/bash

set -euo pipefail

ROOT=${PWD}

### create conda environment ###
conda create -n activegamer python=3.8 cmake=3.14.0

### activate conda environment ###
conda activate activegamer

# ### Setup habitat-sim ###
cd ${ROOT}/third_parties
if [ ! -d habitat_sim ]; then
    git clone git@github.com:Huangying-Zhan/habitat-sim.git habitat_sim
fi
cd habitat_sim
pip install -r requirements.txt
python setup.py install --headless --bullet

# ### Install thir parties ###
### need to change torch/cuda version to adapt your system cuda version
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 -f https://download.pytorch.org/whl/cu117/torch_stable.html

######### Need to package these (all setup.py).

pip install git+https://github.com/facebookresearch/pytorch3d.git@05cbea115acbbcbea77999c03d55155b23479991
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110

############ Meed to figure out WTF to do with these deps

###### install avtivemap
pip install charset_normalizer==2.0.4
pip install mmengine==0.7.3
pip install -r ./envs/requirements_activemap.txt
pip install -r ./envs/requirements_semantic.txt

############ Probably just package this as is.
#            It is very silly, though.

### CoSLAM installation ###
cd ${ROOT}/third_parties/coslam
git checkout 3bb904e
cd external/NumpyMarchingCubes
python setup.py install

############### Should be straightforward.

#### install torch_sparse
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

############### God idk where to put this, but I don't really have to worry about it rn.

## dowload replica data
bash scripts/data/replica_download.sh data/replica_v1
bash scripts/data/replica_update.sh data/replica_v1
bash scripts/data/replica_slam_download.sh

### create soft link if you did not download to the current dir
#ln -s /mnt/Data2/slam_datasets/replica_v1 ./data/replica_v1
#ln -s /mnt/Data2/slam_datasets/Replica ./data/Replica
#ln -s /mnt/Data2/slam_datasets/replica_sim_nvs ./data/replica_sim_nvs
#ln -s /mnt/Data4/slam_datasets/mp3d_sim_nvs ./data/mp3d_sim_nvs
