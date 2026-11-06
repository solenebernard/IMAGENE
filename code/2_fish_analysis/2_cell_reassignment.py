'''
Step 2.2: merge the cellpose masks of the smFISH tiles into one mosaic, with a unique ID for each cell.

The tiles are placed following the snake pattern of the microscope: ROOTPATH/EXP_PATH/Fish/pattern_snake/CONDITION.npy
if it exists, else fish.default_pattern (5x5). The ID of a cell of the tile s becomes 1000*s + its ID in the tile.
The cells touching the border of their tile are removed, and where two tiles overlap, the first tile placed has
the priority.

Input:  ROOTPATH/EXP_PATH/Fish/masks/CONDITION/*_s{s}_*.png (cellpose masks of step M3; empty if a tile is missing)
Output: ROOTPATH/EXP_PATH/Fish/masks/CONDITION/mosaic_mask_reassigned.tiff: the mosaic
        ROOTPATH/EXP_PATH/Fish/masks/CONDITION/mosaic_mask_reassigned_s{s}.tiff: each tile, as placed in the mosaic

Run: python3 -m 2_fish_analysis.2_cell_reassignment
'''


# Import local tools
import numpy as np
import pathlib
import tifffile
from PIL import Image
from tools.read_yaml import CONDITION, EXP_PATH, FISH_DEFAULT_PATTERN, FISH_OVERLAP, FISH_TILE_SIZE, ROOTPATH
from tools.tools_fish import load_snake_pattern, merge_snake_tiles


def main():
    '''Build and save the mosaic of the masks'''
    OVERLAP = FISH_OVERLAP
    IM_SIZE = FISH_TILE_SIZE

    pattern_snake = load_snake_pattern(ROOTPATH, EXP_PATH, CONDITION, FISH_DEFAULT_PATTERN)
    folder_mask = ROOTPATH + EXP_PATH + "Fish/masks/" + CONDITION + "/"

    masks = []

    # Merge all the segmentation
    # + change the ID
    # two first number from 00 to 24
    # Three last numbers: from 000 to 999

    # Loop through the tiles
    n_tiles = int(np.max(pattern_snake[pattern_snake != None]) + 1)
    for s in range(n_tiles):
        s_ = str(s).zfill(2)
        mask_name = '*_s{}_*.png'.format(s_)
        try:
            mask_path = list(pathlib.Path(folder_mask).glob(mask_name))[0]
            mask = np.asarray(Image.open(mask_path))
        except IndexError:  # no mask for this tile: empty tile
            mask = np.zeros((IM_SIZE, IM_SIZE))
        masks.append(mask)

    # Mosaic: the cells touching the border of their tile are removed, and the first tile has the priority where two tiles overlap
    total_image, placed_tiles = merge_snake_tiles(masks, pattern_snake, IM_SIZE, OVERLAP, remove_border=True)

    # Save the submask of each tile, as placed in the mosaic
    for s, im in placed_tiles.items():
        s_ = str(s).zfill(2)
        tifffile.imwrite(folder_mask + 'mosaic_mask_reassigned_s{}.tiff'.format(s_), im.astype(np.int16))

    tifffile.imwrite(folder_mask + 'mosaic_mask_reassigned.tiff', total_image.astype(np.int16))


if __name__ == "__main__":
    main()
