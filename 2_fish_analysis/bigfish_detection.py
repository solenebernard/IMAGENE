'''
Optimized spot detection
From already chosen thresholds
'''

import sys
sys.path.append('../pipeline_smfish-main/src/')

# Import local tools
from tools.read_yaml import *
from tools.imports import *

# Import fish tools
from tools.tools_fish import *

from detection_fish.detect_fish_spots import DetectionPipeline

import argparse
## FOR MULTIPROCESSING
parser = argparse.ArgumentParser()
parser.add_argument("--loop_i", type=int)
args = parser.parse_args()


# Initialize variables
folder_tif = ROOTPATH + EXP_PATH + 'Fish/' + CONDITION + '/'
folder_mask = ROOTPATH + EXP_PATH + 'Fish/masks/' + CONDITION + '/'
folder_save_loc = ROOTPATH + EXP_PATH + 'Fish/bigfish/' + CONDITION + '/'
device = DEVICE

dt = DetectionPipeline()

list_files = list(pathlib.Path(folder_tif).glob('*t*'))

# Get all possible s values
pattern = r"_s(\d{2})_"
# Extract unique values
s_values = set()
for filename in list_files:
    match = re.search(pattern, str(filename))
    if match:
        s_values.add(match.group(1))
# Convert to sorted list
s_values = sorted(s_values)

#print(s_values)

# Run BIGFISH 

# voxel_size_nm = (300, 103, 103)
voxel_size_nm = (399.52483870967737, 103.09496824621396, 103.09496824621396)
object_radius_nm = (350, 150, 150)

thresholds_dict = np.load(folder_save_loc + CONDITION + '_thresh_BIGFISH.npy', allow_pickle=True).item()

# Extract unique values depending on the thresholds we have
# Get all possible t values
pattern = r"t(\d{2})_"
t_values = set()
for filename in thresholds_dict.keys():
    match = re.search(pattern, str(filename))
    if match:
        t_values.add(match.group(1))
# Convert to sorted list
t_values = sorted(t_values)


# Loop through the tiles
for s_ in s_values[args.loop_i:args.loop_i+1]:
# for s_ in s_values[:]:
    mask_name = '*_s{}_*.png'.format(s_)
    # Loop through the rounds
    for t_ in t_values:
        # Loop through the channel:
        for ch in range(2): ######################## TO CHANGE
            ch_ = str(ch).zfill(2)
            name_file = '*t{}_s{}_z*_ch{}.tif'.format(t_, s_, ch_) # Take all z
            frame_names = list(pathlib.Path(folder_tif).glob(name_file))
            frame_names.sort()
            print(name_file, len(frame_names))
            rna_stack = assemble_z_stack(frame_names)
            rna_mip = np.max(rna_stack, axis = 0)

            pattern = f"t{t_}_.*?_ch{ch_}"
            # Get threshold from the dict
            thresh=None
            for x in thresholds_dict.keys():
                match = re.search(pattern, x)
                if match:
                    print(x)
                    thresh = thresholds_dict[x]

            # If we found a threshold in the dictionnary
            if thresh:
                detected_spots = dt.spot_bigfish(
                    rna_stack,
                    voxel_size_nm=voxel_size_nm,
                    object_radius_nm=object_radius_nm,
                    thresh=thresh,
                )
                
                name_save = 't{}_s{}_ch{}.npy'.format(t_, s_, ch_)
                np.save(folder_save_loc+name_save, detected_spots)

