'''
Step 2.3: detection of the RNA spots of one smFISH tile with Big-FISH, with the thresholds chosen beforehand (step M4).

For each round and FISH channel with a threshold, the spots are detected in 3D on the z stack of the tile.
The detection itself is tools.tools_fish.detect_spots_bigfish.

Input:  ROOTPATH/EXP_PATH/Fish/CONDITION/*t{round}_s{s}_z*_ch{ch}.tif
        ROOTPATH/EXP_PATH/Fish/bigfish/CONDITION/CONDITION_thresh_BIGFISH.npy: the thresholds of detection
Output: ROOTPATH/EXP_PATH/Fish/bigfish/CONDITION/t{round}_s{s}_ch{ch}.npy: coordinates (z, row, column) of the spots

Run: python3 -m 2_fish_analysis.3_bigfish_detection --loop_i S   (S: index of the tile, one call per tile, see run.sh)
'''


# Import local tools
import argparse
import numpy as np
import pathlib
import re
from tools.read_yaml import CONDITION, EXP_PATH, FISH_N_CHANNELS, OBJECT_RADIUS_NM, ROOTPATH, VOXEL_SIZE_NM
from tools.tools_fish import assemble_z_stack, detect_spots_bigfish


def main():
    '''Detect the spots of the tile given by --loop_i'''
    ## FOR MULTIPROCESSING
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop_i", type=int)
    args = parser.parse_args()

    # Initialize variables
    folder_tif = ROOTPATH + EXP_PATH + 'Fish/' + CONDITION + '/'
    folder_save_loc = ROOTPATH + EXP_PATH + 'Fish/bigfish/' + CONDITION + '/'

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

    # Run BIGFISH
    voxel_size_nm = VOXEL_SIZE_NM
    object_radius_nm = OBJECT_RADIUS_NM

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
    for s_ in s_values[args.loop_i:args.loop_i + 1]:
        # Loop through the rounds
        for t_ in t_values:
            # Loop through the channel:
            for ch in range(FISH_N_CHANNELS):
                ch_ = str(ch).zfill(2)
                name_file = '*t{}_s{}_z*_ch{}.tif'.format(t_, s_, ch_)  # Take all z
                frame_names = list(pathlib.Path(folder_tif).glob(name_file))
                frame_names.sort()
                print(name_file, len(frame_names))
                rna_stack = assemble_z_stack(frame_names)

                pattern = f"t{t_}_.*?_ch{ch_}"
                # Get threshold from the dict
                thresh = None
                for x in thresholds_dict.keys():
                    match = re.search(pattern, x)
                    if match:
                        print(x)
                        thresh = thresholds_dict[x]

                # If we found a threshold in the dictionnary
                if thresh:
                    detected_spots = detect_spots_bigfish(
                        rna_stack,
                        voxel_size_nm=voxel_size_nm,
                        object_radius_nm=object_radius_nm,
                        thresh=thresh,
                    )

                    name_save = 't{}_s{}_ch{}.npy'.format(t_, s_, ch_)
                    np.save(folder_save_loc + name_save, detected_spots)


if __name__ == "__main__":
    main()
