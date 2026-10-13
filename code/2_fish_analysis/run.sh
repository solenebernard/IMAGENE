#!/bin/bash

# Runs all the steps of this folder sequentially: bash 2_fish_analysis/run.sh
# Comment out the steps already done. All the parameters are in config.yaml

# Stop at the first step that fails
set -e

# Run from the folder code/
cd "$(dirname "$0")/.."

# Parameters read from config.yaml
eval "$(python3 -m tools.config_to_shell)"

mkdir -p ${FISH}/mip ${FISH}/masks/${CONDITION} ${FISH}/bigfish/${CONDITION} ${FISH}/clumps_detection

### 2.1 ### Maximum intensity projection of each tile
python3 -m 2_fish_analysis.1_fish_mip

### M3 ### Cellpose segmentation of the MIP of each tile
# To correct the masks by hand: stop here, correct them in the cellpose GUI, and save them as png in masks/CONDITION/
for s in $(seq -w 0 $((FISH_N_TILES-1))); do
    cellpose --use_gpu --model cpsam --image_path ${FISH}/mip/${CONDITION}_round_*_s${s}_mip.tif --save_png --verbose
done
mv ${FISH}/mip/${CONDITION}_round_*_mip_cp_masks.png ${FISH}/masks/${CONDITION}/

### 2.2 ### Merge the masks and give a unique ID to each cell
python3 -m 2_fish_analysis.2_cell_reassignment

### M4 ### Manual: the thresholds of detection must be saved before 2.3
# in ${FISH}/bigfish/${CONDITION}/${CONDITION}_thresh_BIGFISH.npy (see details.md)

### 2.3 ### Detect the spots with bigfish: one call per tile
for s in $(seq 0 $((FISH_N_TILES-1))); do
    python3 -m 2_fish_analysis.3_bigfish_detection --loop_i $s
done

### 2.4 ### Manual, interactive: the thresholds of the clumps must be saved before 2.5
# streamlit run 2_fish_analysis/4_clumps_detector_streamlit.py

### 2.5 ### Masks of the clumps
python3 -m 2_fish_analysis.5_clumps_detector

### 2.6 ### Gene expression of each cell
python3 -m 2_fish_analysis.6_gene_expression
