#!/bin/bash

# Main figure on the prediction of the gene expression: bash 6_exp_figures/run.sh
# Draws its panels, then assembles them. The parameters are in config.yaml and 6_exp_figures/config_figures.yaml
# Inputs (in ../exp/): the tables of step 5 in results/, the tables of step 4.1 in live_cell_features/ and the thumbnails
# of the cells in thumbnails/
# The figures are saved in ../exp/results/figures/ (main figure: main/main_figure_prediction.png and .pdf)

# Stop at the first figure that fails
set -e

# Run from the folder code/
cd "$(dirname "$0")/.."

### Panels g and h: size of the dataset, performance of the models, and performance with each group of features
python3 -m 5_POC_activation.4_figures_performances
python3 -m 5_POC_activation.6_figures_feature_groups

### Panel f: thresholds of the genes
python3 -m 6_exp_figures.1_gene_thresholds
### Panels a to e: UMAP of the features of the live cells and of the gene expression
python3 -m 6_exp_figures.2_umap
### Panel i: trajectories and contours of the cells with a low and a high expression of a few genes
python3 -m 6_exp_figures.3_contours_trajectories_grid

### Main figure, assembled from the panels above
python3 -m 6_exp_figures.4_main_figure_prediction
