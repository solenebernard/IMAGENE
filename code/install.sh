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
