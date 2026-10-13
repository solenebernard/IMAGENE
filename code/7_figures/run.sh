#!/bin/bash

# Generates all the figures: bash 7_figures/run.sh
# Comment out the figures that are not needed. The parameters are in config.yaml and 7_figures/config_figures.yaml
# The figures are saved in 7_figures/output/

# Stop at the first figure that fails
set -e

# Run from the folder code/
cd "$(dirname "$0")/.."

### Step 1: live cells
python3 -m 7_figures.1_segmentation

### Step 2: smFISH
python3 -m 7_figures.2_fish_rna
python3 -m 7_figures.3_clumps

### Step 3: matching
python3 -m 7_figures.4_matching
python3 -m 7_figures.15_cell_flow
python3 -m 7_figures.16_size_expression

### Step 4: features of the live cells
python3 -m 7_figures.5_live_features
python3 -m 7_figures.6_fourier_features

### Step 5: model
python3 -m 7_figures.7_gene_thresholds
python3 -m 7_figures.8_umap
python3 -m 7_figures.9_correlations
python3 -m 7_figures.10_model_performance
# The two next ones need the folders of the pipeline (data_dir: null in config_figures.yaml)
# python3 -m 7_figures.11_example_cells
# python3 -m 7_figures.12_contours_trajectories

### Video
# Needs the whole field of view: only with the folders of the pipeline (data_dir: null in config_figures.yaml)
# python3 -m 7_figures.13_video

### Video of a BF tile with the masks and the ID of the cells
python3 -m 7_figures.14_video_tile
