#!/bin/sh

#SBATCH -t 04:00:00
##SBATCH --partition=gpu
##SBATCH --qos=gpu  
##SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
##SBATCH --array=0-24


module load Python/3.11.5
module load gcc/9.2.0
module load cuda/12.2  

# Load Apptainer if needed (depends on your cluster environment)
module load apptainer

### 1 ### BRIGHTFIELD
python3 -m 1_bf_analysis.stitch_blend
python3 -m 1_bf_analysis.prepare_live_images
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 TORCH_CUDA_ARCH_LIST="8.0" python3 -m 1_bf_analysis.segment_video --loop_i $SLURM_ARRAY_TASK_ID
python3 -m 1_bf_analysis.clean_mask --loop_t $SLURM_ARRAY_TASK_ID
python3 -m 1_bf_analysis.clean_mask_traj
python3 -m 1_bf_analysis.create_live_dataset

### 2 ### FISH
python3 -m 2_fish_analysis.bigfish_detection --loop_i $SLURM_ARRAY_TASK_ID
python3 -m 2_fish_analysis.cell_reassignment
python3 -m 2_fish_analysis.gene_expression

### 3 ### MATCHING
python3 -m 3_match_bf_fish.live_fish_matching.py

### 4 ### MACHINE LEARNING
python3 -m 4_extract_features_ML.hand_features_fourier
python3 -m 4_extract_features_ML.sobel_filter
python3 -m 4_extract_features_ML.cell_feats_scikit

### 5 ### LEARNING
python3 -m 5_POC_activation.concatenate_features.py
python3 -m 5_POC_activation.concatenate_gene_expr.py
python3 -m 5_POC_activation.model_24H.py

