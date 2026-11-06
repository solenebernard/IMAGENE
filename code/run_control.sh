#!/bin/bash

# Analysis of the control experiment: bash run_control.sh (from the folder code/)
#
# In the control experiment, the cells carry a colour barcode (two dyes, ch01 and ch02), read before the movie and
# after fixation, so that the matching of the live and fixed cells can be checked. Two densities of cells are
# imaged (folders canal_highdensity and canal_lowdensity). The folder of the experiment is control.exp_path of
# config.yaml (config_confidential.yaml if it exists); it holds the original images:
#     BarcodeBeforeTimelapse/   colour barcode before the movie (4 x 4 tiles; ch00 brightfield, ch01, ch02)
#     Live/                     movie, 21 frames (4 x 4 tiles, 4 planes)
#     BarcodeAfterUV/           colour barcode after fixation (8 x 8 tiles for the high density, 7 x 8 for the low one)
#     LongMovieAfterUV/         long movie after UV, 172 frames (1 tile, high density only)
#
# The images are analysed with the scripts of steps 1 and 3, as the main experiment, in the folder analysis/ of the
# experiment (one CONDITION per acquisition, see 7_control_figures/analysis/1_link_images.py):
#     - each acquisition: segmentation of the tiles (M1) and merging of the tiles into a global mask (1.2)
#     - the movies: tracking (1.3)
#     - barcode before the movie <-> first frame of the movie: tracking (1.3) of these two frames
#     - last frame of the movie <-> barcode after fixation: matching of step 3.1, as for the smFISH
#     - last frame of the movie <-> first frame of the long movie after UV: matching of step 3.1, with a scale of 1
# Then 7_control_figures/analysis/ and 7_control_figures/pipeline/ prepare the inputs of the figures (figures_input/ of
# the experiment), measure the barcodes and the motion of the cells, and 7_control_figures/run.sh makes the main figure
# (saved in results/figures/ of the experiment).
#
# The images of the control experiment are not in this repository: this script is provided for information. The
# repository contains the part of figures_input/ read by the main figure, in control/ (control.exp_path of config.yaml).
#
# Each step can be switched off below (false) once it has been done. The script stops at the first step that fails.

# Densities of cells analysed
DENSITIES="highdensity lowdensity"

# Steps to run: true or false
SEGMENTATION=false   # M1: cellpose on every tile of every acquisition (GPU). false: masks already computed, linked by 1_link_images.py
MERGING=true         # 1.2: global masks
TRACKING=true        # 1.3: tracking of the movies, and of the barcode before the movie with the first frame
STITCHING=false      # 1.1: stitched images of the movie, high density only (used only by a video, not by the main figure)
MATCHING=true        # 3.1: movie <-> barcode after fixation, and movie <-> long movie after UV
FIGURES=true         # inputs of the figures, barcodes, motion of the cells, figures

# Stop at the first step that fails
set -e

# Run from the folder code/
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

# Folder of the experiment (CONTROL) and finetuned cellpose model (MODEL_FINETUNE), from config.yaml
eval "$(python3 -m tools.config_to_shell)"
ANALYSIS=${CONTROL}analysis/
ANALYSIS_UV=${CONTROL}analysis_uv/

# Runs a script of steps 1 and 3 on one acquisition of the control experiment:
# imagene CONDITION N_FRAMES N_ROWS N_COLS MODULE [EXP_PATH] [OTHER PARAMETERS]
# The parameters of the acquisition replace the ones of config.yaml for this run only (CONFIG_SET, see tools/read_yaml.py)
imagene() {
    local condition=$1 frames=$2 rows=$3 cols=$4 module=$5 exp=${6:-${ANALYSIS}} other=$7
    CONDITION=${condition} CONFIG_SET="paths.rootpath='' experiment.exp_path=${exp} \
live.n_frames=${frames} live.n_rows=${rows} live.n_cols=${cols} live.z_finetune=0 ${other}" python3 -m ${module}
}

# Number of frames and tile grid of each acquisition: frames rows columns
geometry() {
    case $1 in
        step0_*) echo "1 4 4" ;;
        step1_*) echo "21 4 4" ;;
        afterUV_highdensity) echo "1 8 8" ;;
        afterUV_lowdensity) echo "1 7 8" ;;
        step3_*) echo "172 1 1" ;;
    esac
}

# Images of the experiment linked in analysis/Live/CONDITION/non_merged/, with the names read by the scripts
python3 -m 7_control_figures.analysis.1_link_images

for DENSITY in ${DENSITIES}
do
    echo "########## ${DENSITY}"
    ACQUISITIONS="step0_${DENSITY} step1_${DENSITY} afterUV_${DENSITY}"
    if [ -d "${ANALYSIS}Live/step3_${DENSITY}" ]; then
        ACQUISITIONS="${ACQUISITIONS} step3_${DENSITY}"
    fi

    for ACQUISITION in ${ACQUISITIONS}
    do
        LIVE=${ANALYSIS}Live/${ACQUISITION}
        read FRAMES ROWS COLS <<< "$(geometry ${ACQUISITION})"

        ### M1 ### Segmentation of the brightfield tiles with the finetuned cellpose model
        if ${SEGMENTATION}; then
            mkdir -p ${LIVE}/non_merged/finetune_z2
            # Links to the masks already computed (1_link_images.py) removed: cellpose would write into them
            find ${LIVE}/non_merged/finetune_z2 -type l -name '*_cp_masks.png' -delete
            # Links to the images, not copies
            for image in ${LIVE}/non_merged/*_z00_ch00.tif; do
                ln -sf "../$(basename ${image})" "${LIVE}/non_merged/finetune_z2/$(basename ${image})"
            done
            cellpose \
              --use_gpu \
              --pretrained_model ${MODEL_FINETUNE} \
              --dir ${LIVE}/non_merged/finetune_z2 \
              --savedir ${LIVE}/non_merged/finetune_z2 \
              --save_png \
              --no_npy \
              --verbose
        fi

        ### 1.2 ### Global mask of each frame
        if ${MERGING}; then
            imagene ${ACQUISITION} ${FRAMES} ${ROWS} ${COLS} 1_bf_analysis.2_merge_masks
        fi

        ### 1.3 ### Tracking of the movies
        if ${TRACKING} && [ ${FRAMES} -gt 1 ]; then
            imagene ${ACQUISITION} ${FRAMES} ${ROWS} ${COLS} 1_bf_analysis.3_track_cells
        fi
    done

    ### 1.3 ### Barcode before the movie <-> first frame of the movie: tracking of these two frames
    if ${TRACKING}; then
        LINK=${ANALYSIS}Live/link_${DENSITY}
        mkdir -p ${LINK}/global_masks
        ln -sf ../../step0_${DENSITY}/global_masks/global_mask_t00.png ${LINK}/global_masks/global_mask_t00.png
        ln -sf ../../step1_${DENSITY}/global_masks/global_mask_t00.png ${LINK}/global_masks/global_mask_t01.png
        imagene link_${DENSITY} 2 4 4 1_bf_analysis.3_track_cells
    fi

    ### 1.1 ### Stitched images of the movie (high density only; used only by a video, not by the main figure)
    if ${STITCHING} && [ ${DENSITY} = highdensity ]; then
        imagene step1_${DENSITY} 21 4 4 1_bf_analysis.1_stitch_blend
    fi

    ### 3.1 ### Last frame of the movie <-> barcode after fixation, as for the smFISH (scale 40/63 of config.yaml)
    if ${MATCHING}; then
        mkdir -p ${ANALYSIS}Fish/masks/step1_${DENSITY} ${ANALYSIS}alignment
        python3 -c "
import numpy as np, tifffile; from PIL import Image; Image.MAX_IMAGE_PIXELS = None
tifffile.imwrite('${ANALYSIS}Fish/masks/step1_${DENSITY}/mosaic_mask_reassigned.tiff',
                 np.asarray(Image.open('${ANALYSIS}Live/afterUV_${DENSITY}/global_masks/global_mask_t00.png')), compression='zlib')"
        imagene step1_${DENSITY} 21 4 4 3_match_bf_fish.1_live_fish_matching

        ### 3.1 ### Last frame of the movie <-> first frame of the long movie after UV (same objective: scale 1).
        # The long movie is one tile somewhere in the field of view of the movie: wider translation search
        if [ -d "${ANALYSIS}Live/step3_${DENSITY}" ]; then
            mkdir -p ${ANALYSIS_UV}Live ${ANALYSIS_UV}Fish/masks/step1_${DENSITY} ${ANALYSIS_UV}alignment
            ln -sfn ../../analysis/Live/step1_${DENSITY} ${ANALYSIS_UV}Live/step1_${DENSITY}
            python3 -c "
import numpy as np, tifffile; from PIL import Image
tifffile.imwrite('${ANALYSIS_UV}Fish/masks/step1_${DENSITY}/mosaic_mask_reassigned.tiff',
                 np.asarray(Image.open('${ANALYSIS}Live/step3_${DENSITY}/global_masks/global_mask_t00.png')), compression='zlib')"
            imagene step1_${DENSITY} 21 4 4 3_match_bf_fish.1_live_fish_matching ${ANALYSIS_UV} \
                "constants.scale='1' matching.translation_delta=6000"
        fi
    fi
done


##### FIGURES (inputs in figures_input/ of the experiment, figures in its results/figures/)
if ${FIGURES}; then
    FIGURES_INPUT=${CONTROL}figures_input/

    ### Inputs of the figures, from the outputs above (folder figures_input/ of the experiment)
    python3 -m 7_control_figures.analysis.2_prepare_figure_inputs

    ### Barcode of each cell (90th percentile of ch01 and ch02 in its mask) and motion of the cells
    python3 7_control_figures/pipeline/barcode_intensities.py --path_folder ${FIGURES_INPUT}canal2/step0
    python3 7_control_figures/pipeline/barcode_intensities.py --path_folder ${FIGURES_INPUT}canal05/step0
    python3 7_control_figures/pipeline/barcode_intensities.py --path_folder ${FIGURES_INPUT}canal2/step5_afterPFA \
        --tilescan_order "[[0,1,2,3,4,5,6,7],[15,14,13,12,11,10,9,8],[16,17,18,19,20,21,22,23],[31,30,29,28,27,26,25,24],[32,33,34,35,36,37,38,39],[47,46,45,44,43,42,41,40],[48,49,50,51,52,53,54,55],[63,62,61,60,59,58,57,56]]"
    python3 7_control_figures/pipeline/barcode_intensities.py --path_folder ${FIGURES_INPUT}canal05/step7_afterPFA \
        --tilescan_order "[[0,1,2,3,4,5,6,7],[15,14,13,12,11,10,9,8],[16,17,18,19,20,21,22,23],[31,30,29,28,27,26,25,24],[32,33,34,35,36,37,38,39],[47,46,45,44,43,42,41,40],[48,49,50,51,52,53,54,55]]"
    python3 7_control_figures/pipeline/track_dynamics.py --input_dir ${FIGURES_INPUT}canal2/step1
    python3 7_control_figures/pipeline/track_dynamics.py --input_dir ${FIGURES_INPUT}canal05/step1
    if [ -d "${FIGURES_INPUT}canal2/step3" ]; then
        python3 7_control_figures/pipeline/track_dynamics.py --input_dir ${FIGURES_INPUT}canal2/step3
        # Successive windows of 21 frames (7 min) of the long movie after UV, every 10 frames, and the last one
        for START in $(seq 0 10 150) 151; do
            python3 7_control_figures/pipeline/track_dynamics.py --input_dir ${FIGURES_INPUT}canal2/step3 --n_frames 21 --frame_start ${START}
        done
    fi

    ### Main figure of the control experiment (7_control_figures/run.sh)
    bash 7_control_figures/run.sh
fi
