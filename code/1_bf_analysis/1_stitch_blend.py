'''
Step 1.1 (optional): stitch the brightfield tiles into one image per time frame and z slice.
Only needed when the microscope exported the tiles separately (folder non_merged/).

The tiles follow a snake pattern of live.n_rows x live.n_cols tiles, with a fractional overlap live.overlap.
In the overlaps, the tiles are blended with linear weights.

Input:  ROOTPATH/EXP_PATH/Live/CONDITION/non_merged/*_t{t}_s{s}_z{z}_ch00.tif
Output: ROOTPATH/EXP_PATH/Live/CONDITION/BF_custom_stitched_merged_t{t}_z{z}_ch00.tif

Run: python3 -m 1_bf_analysis.1_stitch_blend
'''


# Import local tools
import numpy as np
import pathlib
import re
import tifffile
from tools.read_yaml import CONDITION, EXP_PATH, LIVE_N_COLS, LIVE_N_ROWS, LIVE_OVERLAP, ROOTPATH


def stitch(image_folder, save_folder, match_s, t_, z_, rows, cols, overlap):
    '''
    Stitch the tiles match_s of the time frame t_ and the z slice z_ (grid of rows x cols tiles in a snake pattern,
    fractional overlap), and save the stitched image in save_folder
    '''

    # Get all tiles
    file_dict = [list(pathlib.Path(image_folder).glob('*_t{}_s{}_z{}_ch00.tif'.format(t_, s_, z_))) for s_ in match_s]

    # === Load sample image to get dimensions ===
    sample = tifffile.imread(file_dict[0])
    H, W = sample.shape
    overlap_x = int(W * overlap)
    overlap_y = int(H * overlap)
    step_x = W - overlap_x
    step_y = H - overlap_y

    # === Create canvas and weight map ===
    stitched = np.zeros((step_y * (rows - 1) + H, step_x * (cols - 1) + W), dtype=np.float32)
    weight_map = np.zeros_like(stitched, dtype=np.float32)

    # === Row/col index with snake pattern ===
    def get_row_col(index, num_cols):
        row = index // num_cols
        col = index % num_cols if row % 2 == 0 else num_cols - 1 - (index % num_cols)
        return row, col

    # === Stitching with blending ===
    for idx in range(rows * cols):
        fname = file_dict[idx]
        img = tifffile.imread(fname).astype(np.float32)
        row, col = get_row_col(idx, cols)

        y = row * step_y
        x = col * step_x

        h, w = img.shape

        # Create blending weights
        wy = np.ones(h)
        wx = np.ones(w)

        if row > 0:
            wy[:overlap_y] = np.linspace(0, 1, overlap_y)
        if row < rows - 1:
            wy[-overlap_y:] = np.linspace(1, 0, overlap_y)

        if col > 0:
            wx[:overlap_x] = np.linspace(0, 1, overlap_x)
        if col < cols - 1:
            wx[-overlap_x:] = np.linspace(1, 0, overlap_x)

        weight = np.outer(wy, wx)

        # Place image and weights into stitched canvas
        stitched[y:y + h, x:x + w] += img * weight
        weight_map[y:y + h, x:x + w] += weight

    # === Normalize by accumulated weights to finish blending ===
    stitched_final = (stitched / np.maximum(weight_map, 1e-6)).astype(np.uint16)

    # === Save ===
    output_path = save_folder + "BF_custom_stitched_merged_t{}_z{}_ch00.tif".format(t_, z_)
    tifffile.imwrite(output_path, stitched_final)


def main():
    '''Stitch all the time frames and z slices'''
    # === PARAMETERS ===
    image_folder = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/non_merged/'
    save_folder = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/'
    rows, cols = LIVE_N_ROWS, LIVE_N_COLS                # Grid size
    overlap = LIVE_OVERLAP

    print(image_folder, save_folder)

    # Scan all all t frames
    # Get all possible z
    frame_names_z = list(pathlib.Path(image_folder).glob('*_t00_s00_z*_ch00.tif'))
    pattern = r"_z(\d{2})_"
    match_z = [re.search(pattern, str(filename)) for filename in frame_names_z]
    match_z = [m.group(1) for m in match_z if m]
    match_z.sort()
    max_z_ = str(len(match_z) - 1).zfill(2)

    # Get all possible t
    frame_names_t = list(pathlib.Path(image_folder).glob('*_t*_s00_z{}_ch00.tif'.format(max_z_)))
    pattern = r"_t(\d{2})_"
    match_t = [re.search(pattern, str(filename)) for filename in frame_names_t]
    match_t = [m.group(1) for m in match_t if m]
    match_t.sort()

    # Get all possible tiles s
    frame_names_s = list(pathlib.Path(image_folder).glob('*_t00_s*_z00_ch00.tif'))
    pattern = r"_s(\d{2})_"
    match_s = [re.search(pattern, str(filename)) for filename in frame_names_s]
    match_s = [m.group(1) for m in match_s if m]
    match_s.sort()

    for t_ in match_t:
        for z_ in match_z:
            stitch(image_folder, save_folder, match_s, t_, z_, rows, cols, overlap)


if __name__ == "__main__":
    main()
