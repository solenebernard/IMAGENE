#!/bin/bash

# Runs all the steps of this folder sequentially: bash 3_match_bf_fish/run.sh
# Comment out the steps already done. All the parameters are in config.yaml

# Stop at the first step that fails
set -e

# Run from the folder code/
cd "$(dirname "$0")/.."

# Parameters read from config.yaml
eval "$(python3 -m tools.config_to_shell)"

### 3.1 ### Match the live and FISH cells
mkdir -p ${ROOTPATH}${EXP_PATH}alignment
python3 -m 3_match_bf_fish.1_live_fish_matching
