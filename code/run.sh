#!/bin/bash

# Runs the whole pipeline: bash run.sh (from the folder code/)
#   - steps 1 to 4 for each condition of CONDITIONS
#   - step 5 once, on all the conditions together, then the two main figures (steps 6 and 7)
# All the parameters are in config.yaml (config_confidential.yaml if it exists) and 6_exp_figures/config_figures.yaml.
# Each step can be switched off below (false) once it has been done. The script stops at the first step that fails.
#
# Steps 1 to 4 need the raw images and the finetuned Cellpose model, which are not in this repository: they are off.
# With the files of this repository, steps 5.1 to 5.6 and the figures can be run (STEP_5, STEP_6, STEP_7).

# Conditions processed by steps 1 to 4. They replace experiment.condition of config.yaml
CONDITIONS="3-24H 4-24H"

# Steps to run: true or false
STEP_1=false         # live cells: stitching, segmentation, merging, tracking, single-cell movies (raw images)
STITCH_TILES=false   # step 1.1, only if the microscope exported unmerged tiles
STEP_2=false         # smFISH: projection, segmentation, spot detection, aggregates, gene expression (raw images)
STEP_3=false         # matching of the live and fixed cells (masks of steps 1 and 2)
STEP_4=false         # features of the live cells (masks of step 1)
STEP_5=true          # dataset, models and their figures (about 1 h 15 min on 16 cores)
STEP_6=true          # main figure of the prediction
STEP_7=true          # main figure of the control experiment

# Stop at the first step that fails
set -e

# Run from the folder code/
cd "$(dirname "$0")"


##### STEPS 1 TO 4: ONE CONDITION AT A TIME
for CONDITION in ${CONDITIONS}
do
    # The condition is read by all the python scripts (tools/read_yaml.py) and by tools.config_to_shell
    export CONDITION
    # Paths and parameters of this condition: ROOTPATH, EXP_PATH, LIVE, FISH, MODEL_FINETUNE, Z_FINETUNE, FISH_N_TILES
    eval "$(python3 -m tools.config_to_shell)"
    echo "########## Condition ${CONDITION}"

    ##### STEP 1: LIVE CELLS
    if ${STEP_1}; then
        mkdir -p ${ROOTPATH}${EXP_PATH}live_cell_features

        ### 1.1 ### Stitch the tiles (only if the images were exported unmerged)
        if ${STITCH_TILES}; then
            python3 -m 1_bf_analysis.1_stitch_blend
        fi

        # ### M1 ### Segment each non-merged tile, at every time frame, with the finetuned cellpose model
        # mkdir -p ${LIVE}/non_merged/finetune_z2
        # cp ${LIVE}/non_merged/*_z${Z_FINETUNE}_ch00.tif ${LIVE}/non_merged/finetune_z2/
        # cellpose \
        #   --use_gpu \
        #   --pretrained_model ${MODEL_FINETUNE} \
        #   --dir ${LIVE}/non_merged/finetune_z2 \
        #   --savedir ${LIVE}/non_merged/finetune_z2 \
        #   --save_png \
        #   --no_npy \
        #   --verbose

        ### 1.2 ### Merge the masks of the tiles into one global mask per time frame
        python3 -m 1_bf_analysis.2_merge_masks
        ### 1.3 ### Track the cells across the time frames
        python3 -m 1_bf_analysis.3_track_cells
        ### 1.4 ### Create the videos of individual cells
        # python3 -m 1_bf_analysis.4_create_live_dataset
    fi

    ##### STEP 2: SMFISH
    if ${STEP_2}; then
        mkdir -p ${FISH}/mip ${FISH}/masks/${CONDITION} ${FISH}/bigfish/${CONDITION} ${FISH}/clumps_detection

        ### 2.1 ### Maximum intensity projection of each tile
        python3 -m 2_fish_analysis.1_fish_mip

        ### M3 ### Cellpose segmentation of the MIP of each tile
        # The masks are used without correction. To correct them by hand: set STEP_2=false after this command,
        # correct the masks in the cellpose GUI, save them as png in masks/CONDITION/, then run the rest of step 2
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

        ### 2.4 ### Manual, interactive: the thresholds of the aggregates must be saved before 2.5
        # streamlit run 2_fish_analysis/4_clumps_detector_streamlit.py

        ### 2.5 ### Masks of the aggregates
        python3 -m 2_fish_analysis.5_clumps_detector
        ### 2.6 ### Gene expression of each cell
        python3 -m 2_fish_analysis.6_gene_expression
    fi

    ##### STEP 3: MATCHING OF THE LIVE AND FIXED CELLS
    if ${STEP_3}; then
        mkdir -p ${ROOTPATH}${EXP_PATH}alignment
        ### 3.1 ### Match the live and FISH cells
        python3 -m 3_match_bf_fish.1_live_fish_matching
    fi

    ##### STEP 4: FEATURES OF THE LIVE CELLS
    if ${STEP_4}; then
        ### 4.1 ### Features from the finetuned masks
        python3 -m 4_extract_features_ML.1_hand_features_fourier_finetune
        ### 4.2 ### Dynamics and neighbors
        python3 -m 4_extract_features_ML.2_dynamics_features
    fi
done
unset CONDITION


##### STEP 5: DATASET AND PREDICTION OF THE GENE EXPRESSION (all the conditions of learning.conditions in config.yaml)
if ${STEP_5}; then
    ### 5.1 ### Table of features
    python3 -m 5_POC_activation.1_concatenate_features
    ### 5.2 ### Table of all gene profiles
    python3 -m 5_POC_activation.2_concatenate_gene_expr
    ### 5.3 ### Training and testing of the models
    python3 -m 5_POC_activation.3_model
    ### 5.4 ### Figures of the performance
    python3 -m 5_POC_activation.4_figures_performances
    ### 5.5 ### Models with each group of features
    python3 -m 5_POC_activation.5_model_feature_groups
    ### 5.6 ### Figures of the groups of features
    python3 -m 5_POC_activation.6_figures_feature_groups
fi


##### STEP 6: MAIN FIGURE OF THE PREDICTION (saved in ../exp/results/figures/)
if ${STEP_6}; then
    bash 6_exp_figures/run.sh
fi


##### STEP 7: MAIN FIGURE OF THE CONTROL EXPERIMENT (saved in ../control/results/figures/)
if ${STEP_7}; then
    bash 7_control_figures/run.sh
fi
