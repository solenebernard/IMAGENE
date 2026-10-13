#!/bin/bash

# Runs all the steps of this folder sequentially: bash 1_bf_analysis/run.sh
# Comment out the steps already done. All the parameters are in config.yaml

# Stop at the first step that fails
set -e

# Run from the folder code/
cd "$(dirname "$0")/.."

# Parameters read from config.yaml
eval "$(python3 -m tools.config_to_shell)"

mkdir -p ${ROOTPATH}${EXP_PATH}live_cell_features

### 1.1 ### Stitch the tiles (only if the images were exported unmerged)
python3 -m 1_bf_analysis.1_stitch_blend

### M1 ### Segment each non-merged tile, at every time frame, with the finetuned cellpose model
mkdir -p ${LIVE}/non_merged/finetune_z2
cp ${LIVE}/non_merged/*_z${Z_FINETUNE}_ch00.tif ${LIVE}/non_merged/finetune_z2/
cellpose \
  --use_gpu \
  --pretrained_model ${MODEL_FINETUNE} \
  --dir ${LIVE}/non_merged/finetune_z2 \
  --savedir ${LIVE}/non_merged/finetune_z2 \
  --save_png \
  --no_npy \
  --verbose

### 1.2 ### Merge the masks of the tiles into one global mask per time frame
python3 -m 1_bf_analysis.2_merge_masks

### 1.3 ### Track the cells across the time frames
python3 -m 1_bf_analysis.3_track_cells

### 1.4 ### Create the videos of individual cells
python3 -m 1_bf_analysis.4_create_live_dataset
