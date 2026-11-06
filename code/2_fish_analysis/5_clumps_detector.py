'''
Step 2.5: mask of the clumps of each round and FISH channel, on the whole smFISH mosaic.

Each tile is thresholded with the threshold chosen in step 2.4 (the largest one over the tiles, for this round and
channel), the tiles are assembled into the mosaic, and the regions smaller than fish.clumps_min_size are removed.
In step 2.6, the cells that overlap a clump get a missing value for this gene.

Input:  ROOTPATH/EXP_PATH/Fish/CONDITION/ (z stacks), ROOTPATH/EXP_PATH/Fish/clumps_detection/CONDITION_thresholds.csv
Output: ROOTPATH/EXP_PATH/Fish/clumps_detection/CONDITION_mask_total_t{round}_ch{ch}.png

Run: python3 -m 2_fish_analysis.5_clumps_detector
'''


# Import local tools
import numpy as np
import pandas as pd
from tools.read_yaml import (
    CLUMPS_MIN_SIZE, CONDITION, EXP_PATH, FISH_DEFAULT_PATTERN, FISH_N_CHANNELS, FISH_N_TILES, FISH_OVERLAP,
    FISH_TILE_SIZE, ROOTPATH, ROUNDS
)
from tools.tools_fish import show_mip, load_snake_pattern, remove_small_regions, merge_snake_tiles
import imageio.v2 as imageio


def main():
    '''Masks of the clumps of every round and channel'''
    # Version with the merged mask
    OVERLAP = FISH_OVERLAP
    IM_SIZE = FISH_TILE_SIZE
    MINSIZE = CLUMPS_MIN_SIZE

    path_clumps = f'{ROOTPATH}{EXP_PATH}Fish/clumps_detection/'
    thresh_csv = pd.read_csv(path_clumps + f'{CONDITION}_thresholds.csv',
                             index_col=['s', 't', 'ch'])

    pattern_snake = load_snake_pattern(ROOTPATH, EXP_PATH, CONDITION, FISH_DEFAULT_PATTERN)

    # Create all masks
    for t in ROUNDS:
        for ch in range(FISH_N_CHANNELS):
            list_mips = []
            for s in range(FISH_N_TILES):
                mip = show_mip(ROOTPATH, EXP_PATH, CONDITION, s, t, ch)
                list_mips.append(mip)
            thresh = thresh_csv.loc[:, t, ch, :].max().values[0]
            t_, ch_ = str(t).zfill(2), str(ch).zfill(2)

            # Mosaic of the thresholded tiles
            total_image, _ = merge_snake_tiles([mip > thresh for mip in list_mips], pattern_snake, IM_SIZE, OVERLAP)

            total_image = total_image > 0
            total_image = remove_small_regions(total_image, MINSIZE)
            imageio.imwrite(path_clumps + f'{CONDITION}_mask_total_t{t_}_ch{ch_}.png', total_image.astype(np.uint8))


if __name__ == "__main__":
    main()
