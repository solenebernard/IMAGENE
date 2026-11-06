#!/bin/bash

# Runs all the steps of this folder sequentially: bash 5_POC_activation/run.sh
# Comment out the steps already done. All the parameters are in config.yaml

# Stop at the first step that fails
set -e

# Run from the folder code/
cd "$(dirname "$0")/.."

### 5.1 ### Table of features
python3 -m 5_POC_activation.1_concatenate_features
### 5.2 ### Table of all gene profiles
python3 -m 5_POC_activation.2_concatenate_gene_expr
### 5.3 ### Training and testing of the models
python3 -m 5_POC_activation.3_model
### 5.4 ### Figures of the performance
python3 -m 5_POC_activation.4_figures_performances
### 5.5 ### Models with the shape only, the motility only, and both
python3 -m 5_POC_activation.5_model_feature_groups
### 5.6 ### Figures of the groups of features
python3 -m 5_POC_activation.6_figures_feature_groups
