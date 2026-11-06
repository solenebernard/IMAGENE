#!/bin/bash

# Runs all the steps of this folder sequentially: bash 4_extract_features_ML/run.sh
# Comment out the steps already done. All the parameters are in config.yaml

# Stop at the first step that fails
set -e

# Run from the folder code/
cd "$(dirname "$0")/.."

### 4.1 ### Features from the finetuned masks
python3 -m 4_extract_features_ML.1_hand_features_fourier_finetune
### 4.2 ### Dynamics and neighbors
python3 -m 4_extract_features_ML.2_dynamics_features
