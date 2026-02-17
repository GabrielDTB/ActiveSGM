#!/bin/bash

export CMAKE_GENERATOR=Ninja
export PYTHONPATH=$(pwd)
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

DATA_ROOT=/mnt/Data3/liyan/preprocess/MP3D/
REPO_ROOT=/mnt/Data2/liyan/SceneSplat/
RAW_DATA=/mnt/Data4/slam_datasets/MP3D/v1/scans/

scenes=( GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH )

############ process GS
#for selected_scene in ${scenes[@]}
#do
#python scripts/preprocess_gs_v2.py \
#       --input /mnt/Data2/liyan/ActiveSGM/results/MP3D/${selected_scene}/ActiveSem/run_0/splatam/final/ \
#       --transform \
#       --output /mnt/Data3/liyan/preprocess/MP3D/ActiveSGM_v2/${selected_scene}/
#
#done

############ process GT segmentation

for selected_scene in ${scenes[@]}
do
python scripts/preprocess_pc.py \
       --input ${DATA_ROOT}/ActiveSGM/${selected_scene}/semantic_clean.ply    \
       --category_mapping ${REPO_ROOT}/pointcept/datasets/preprocessing/matterport3d/meta_data/category_mapping.tsv \
       --label_type mp3d21 \
       --scene_seg_info ${RAW_DATA}/${selected_scene}/${selected_scene}/house_segmentations/${selected_scene}.semseg.json \
       --label_txt ${REPO_ROOT}/pointcept/datasets/preprocessing/matterport3d/meta_data/mp3d40.txt \
       --output ${DATA_ROOT}/ActiveSGM/${selected_scene}/

done

############################################
##   run evaluation need pc_segment.npy and pc_coord.npy
##   before running, rename the pc_segment_{label}.npy to pc_segment.npy, depends on which label you want to use
##   also need to match the config file for text_embedding in the zero_shot segmentation dict
#############################################

###
#
#
# scenes=(office0 office1 office2 office3 office4 room0 room1 room2)
#
#for selected_scene in ${scenes[@]}
#do
#
#python scripts/preprocess_gs_v2.py \
#       --input /mnt/Data2/liyan/ActiveSGM/results/Replica/${selected_scene}/ActiveSem/run_0/splatam/final/params.npz \
#       --output /mnt/Data3/liyan/preprocess/Replica/ActiveSGM/${selected_scene}/
#
#done
