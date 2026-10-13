#!/bin/bash

# Installs the libraries required by the pipeline, in the current python environment: bash install.sh
# It is recommended to create an environment first, for example:
#   conda create -n imagene python=3.10
#   conda activate imagene

# Stop at the first error
set -e

# Run from the folder code/
cd "$(dirname "$0")"

### Python libraries
python3 -m pip install -r requirements.txt

### Check that all the libraries are found
python3 - <<PYTHON
import importlib
libraries = ['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'yaml', 'tifffile', 'PIL', 'imageio', 'cv2',
             'skimage', 'sklearn', 'cellpose', 'bigfish', 'streamlit', 'pyefd', 'numexpr', 'shapely', 'powerlaw',
             'umap', 'statsmodels']
missing = []
for name in libraries:
    try:
        importlib.import_module(name)
    except Exception as error:
        missing.append(f'{name} ({error})')
if missing:
    raise SystemExit('Libraries not found: ' + ', '.join(missing))
print(f'{len(libraries)} python libraries found')
PYTHON

### Tools that are not python libraries
# ffmpeg with the H.264 encoder: for the videos of 7_figures (else the videos are saved in MPEG-4)
if command -v ffmpeg > /dev/null && ffmpeg -hide_banner -encoders 2> /dev/null | grep -q libx264; then
    echo "ffmpeg with libx264 found"
else
    echo "WARNING: ffmpeg with libx264 not found. Install it with: conda install -c conda-forge ffmpeg"
fi

# Detection of the spots (step 2.3): the repository pipeline_smfish-main must be next to the folder code/
if [ -d "../pipeline_smfish-main/src" ]; then
    echo "pipeline_smfish-main found"
else
    echo "WARNING: ../pipeline_smfish-main/src not found. It is needed for the detection of the spots (step 2.3)"
fi
