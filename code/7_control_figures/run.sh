#!/bin/bash

# Main figure of the control experiment: bash 7_control_figures/run.sh
# Draws its panels, then assembles them. The inputs are in ../control/ (control.exp_path of config.yaml): the tables,
# masks and images of the control experiment needed by the panels (figures_input/), the alignment of the movie on the
# barcode after fixation (alignment/), and the mask of the fixed cells (BarcodeAfterUV/, for the size of its field of view)
# The figures are saved in ../control/results/figures/ (main figure: main_figure_control.png and .pdf)

# Stop at the first figure that fails
set -e

# Run from the folder code/
cd "$(dirname "$0")/.."

### Panel b: fraction of the cells that stay put after UV (high density)
python3 -m 7_control_figures.1_sticking_fraction_after_uv --canal 2
### Panel c: trajectories and contours of the cells before and after UV (high density)
python3 -m 7_control_figures.2_trajectories_contours_after_uv --canal 2
### Panel d: search of the translation aligning the last frame of the movie on the barcode after fixation
python3 -m 7_control_figures.3_alignment_search --canal 2
### Panel e: segmentation of the live and fixed cells, and pairs passing the quality control
python3 -m 7_control_figures.4_matching_live_fixed --canal 2
### Panel f: confusion matrices of the barcodes, live against fixed cells
python3 -m 7_control_figures.5_confusion_matrices
### Panel g: balanced accuracy as a function of the quality control of the matching
python3 -m 7_control_figures.6_accuracy_vs_qc_ratio
### Panel h: flux diagram of the cells, from the movie to the final dataset (high density)
python3 -m 7_control_figures.7_flux_diagram_cells --canal 2

### Main figure, assembled from the panels above
python3 -m 7_control_figures.8_main_figure_control
