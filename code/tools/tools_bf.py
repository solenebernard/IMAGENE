
import numpy as np
import pandas as pd
import tifffile
import pathlib
from PIL import Image
import re
import cv2

def compute_snake_tile_coords(tile_size, overlap, n_rows, n_cols):
    """
    Position (x0, x1, y0, y1) of each tile in the merged image.
    The tiles follow a snake pattern: left to right for even rows, right to left for odd rows.
    overlap: in pixels
    """
    coords = [None] * (n_rows*n_cols)
    step = tile_size - overlap
    tile_id = 0
    for row in range(n_rows):
        if row % 2 == 0:
            col_order = range(n_cols)
        else:
            col_order = reversed(range(n_cols))
        for col in col_order:
            x0, y0 = col * step, row * step
            coords[tile_id] = (x0, x0 + tile_size, y0, y0 + tile_size)
            tile_id += 1
    return coords


def load_masks_tile_cellpose(rootpath, exp_path, condition, t, n_tiles=16, z=2):
    """
    Load the cellpose masks of all the tiles for the time frame t. Dimension: (s,x,y)
    """
    data_path_raw = f'{rootpath}{exp_path}/Live/{condition}/non_merged/finetune_z2/'
    path_tiles = pathlib.Path(data_path_raw)
    t_, z_ = str(t).zfill(2), str(z).zfill(2)
    tile_masks_path = [list(path_tiles.glob(f'*t{t_}_s{str(s).zfill(2)}_z{z_}_ch00_cp_masks.png'))[0]\
                  for s in range(n_tiles)]
    tile_masks = np.asarray([Image.open(x) for x in tile_masks_path])
    return(tile_masks)


##### TRACKED MASKS
# The masks with the track ID are not saved: they are built from the global masks (step 1.2) or from the
# cellpose masks of the tiles, and from the table of the tracks (step 1.3)

def load_tracks(rootpath, exp_path, condition):
    """
    Table of the tracks (step 1.3), one row per cell and time frame:
    unique_id (ID of the track), t, global_id (ID in the global mask of the frame), centroid_x, centroid_y, area
    """
    return pd.read_csv(f'{rootpath}{exp_path}/Live/{condition}/track/tracks.csv')


def load_global_id_map(rootpath, exp_path, condition):
    """
    Table of the cells of the global masks (step 1.2), one row per cell and time frame:
    t, global_id, s (tile of the cell), local_id (ID in the cellpose mask of the tile)
    """
    return pd.read_csv(f'{rootpath}{exp_path}/Live/{condition}/global_masks/global_id_map.csv')


def relabel(mask, old_id, new_id):
    """Replace the IDs old_id of the mask by new_id. The other IDs are set to 0"""
    lut = np.zeros(max(int(mask.max()), int(np.max(old_id, initial=0))) + 1, dtype=np.uint16)
    lut[np.asarray(old_id, dtype=np.int64)] = new_id
    return lut[mask]


def load_global_mask(rootpath, exp_path, condition, t):
    """Global mask of the time frame t (step 1.2): the ID are not tracked"""
    path = f'{rootpath}{exp_path}/Live/{condition}/global_masks/global_mask_t{str(t).zfill(2)}.png'
    return np.asarray(Image.open(path))


def load_tracked_mask(rootpath, exp_path, condition, t, tracks):
    """Global mask of the time frame t, with the track ID of each cell"""
    tracks_t = tracks[tracks['t']==t]
    mask = load_global_mask(rootpath, exp_path, condition, t)
    return relabel(mask, tracks_t['global_id'].values, tracks_t['unique_id'].values)


def load_tracked_masks(rootpath, exp_path, condition, t_max):
    """Global masks of all the time frames, with the track ID of each cell. Dimension: (t,x,y)"""
    tracks = load_tracks(rootpath, exp_path, condition)
    return np.asarray([load_tracked_mask(rootpath, exp_path, condition, t, tracks) for t in range(t_max)])


def load_tracked_tile_mask(rootpath, exp_path, condition, t, s, tracks, id_map, z=2):
    """
    Cellpose mask of the tile s for the time frame t, with the track ID of each cell.
    Each cell is in one tile only: the tile chosen when merging (step 1.2)
    """
    folder = pathlib.Path(f'{rootpath}{exp_path}/Live/{condition}/non_merged/finetune_z2/')
    t_, s_, z_ = str(t).zfill(2), str(s).zfill(2), str(z).zfill(2)
    mask = np.asarray(Image.open(list(folder.glob(f'*t{t_}_s{s_}_z{z_}_ch00_cp_masks.png'))[0]))
    # Cells of the tile: ID in the tile -> ID in the global mask -> track ID
    cells = id_map[(id_map['t']==t)&(id_map['s']==s)].merge(tracks[tracks['t']==t], on=['t', 'global_id'])
    return relabel(mask, cells['local_id'].values, cells['unique_id'].values)


def load_tracked_tile_masks(rootpath, exp_path, condition, t_max, n_tiles=16, z=2):
    """Cellpose masks of all the tiles and time frames, with the track ID of each cell. Dimension: (t,s,x,y)"""
    tracks = load_tracks(rootpath, exp_path, condition)
    id_map = load_global_id_map(rootpath, exp_path, condition)
    return np.asarray([[load_tracked_tile_mask(rootpath, exp_path, condition, t, s, tracks, id_map, z=z) \
                        for s in range(n_tiles)] for t in range(t_max)])


def load_bf_images(rootpath, exp_path, condition, t_max = None):
    data_path_raw = rootpath + exp_path + 'Live/' + condition + '/'

    # Scan all all t frames 
    # Get all possible z
    frame_names_z = list(pathlib.Path(data_path_raw).glob('*_t00_z*_ch00.tif'))
    pattern = r"_z(\d{2})_"
    match_z = [re.search(pattern, str(filename)) for filename in frame_names_z]
    match_z = [m.group(1) for m in match_z if m]
    match_z.sort()
    max_z_ = str(len(match_z)-1).zfill(2)

    # Get all frames
    frame_names_t = list(pathlib.Path(data_path_raw).glob('*_t*_z{}_ch00.tif'.format(max_z_)))
    pattern = r"_t(\d{2})_"
    match_t = [re.search(pattern, str(filename)) for filename in frame_names_t]
    match_t = [m.group(1) for m in match_t if m]
    match_t.sort()

    if t_max is not None:
        match_t = np.arange(t_max)
        match_t = [str(t).zfill(2) for t in match_t]
    # Extract all frame names: all t and all z
    frame_names = [[list(pathlib.Path(data_path_raw).glob('*_t{}_z{}_ch00.tif'.format(t_,z_)))[0] \
                    for z_ in match_z] for t_ in match_t] 

    # Load all the masks (for each time frame) and all brightfield images
    list_tif_images = []

    for i, frames in enumerate(frame_names):
        im = [tifffile.imread(frame) for frame in frames] # all z for a given t
        list_tif_images.append(np.asarray(im))

    list_tif_images = np.asarray(list_tif_images) # Dimension: (t,z,x,y)
    
    return(list_tif_images)

def load_images_and_masks(rootpath, exp_path, condition, t_max):
    list_tif_images = load_bf_images(rootpath, exp_path, condition, t_max=t_max)
    list_masks = load_tracked_masks(rootpath, exp_path, condition, t_max)
    return(list_tif_images, list_masks)


def remove_border(im):
    image = np.copy(im)
    # Crop the border with 0 values
    _,x,y = np.where(image>0)
    if len(x)>0:
        xmin, xmax, ymin, ymax = x.min(),x.max(), y.min(), y.max()
        image = image[:, xmin:xmax,ymin:ymax]
    return(image)

def pick_z_laplacian(image):
    """
    Choose the image (z,h,w) to (h,w) via laplacian metric to measure the focus
    """
    metrics = []
    image = remove_border(image)

    for j in range(image.shape[0]):
        m = cv2.Laplacian(image[j], cv2.CV_64F).var()
        metrics.append(m)
    metrics = np.asarray(metrics)
    # Best metric
    j = np.argmax(metrics)
    return(j)
