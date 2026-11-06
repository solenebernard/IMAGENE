'''
Tools for the smFISH images of step 2: z stacks and maximum intensity projections of the tiles, snake pattern of the
tiles, mosaic of the tiles, removal of the small regions and of the cells on the border of a tile, detection of the
RNA spots with Big-FISH
'''

import tifffile
import numpy as np
import pathlib
import bigfish.detection as detection
import bigfish.stack as stack


def assemble_z_stack(frame_names):
    z_stack = []
    for image_path in frame_names:
        image = tifffile.imread(image_path)
        z_stack.append(np.array(image))
    z_stack_array = np.stack(z_stack, axis=0)
    return z_stack_array


def detect_spots_bigfish(rna, voxel_size_nm, object_radius_nm, thresh=None):
    '''Detects the RNA spots of a z stack (z, row, column) with Big-FISH: LoG filter, local maxima, then threshold.
    thresh: the threshold of detection (step M4); if None, Big-FISH chooses it automatically.
    Returns an array (n_spots, 3) of the coordinates (z, row, column) of the spots, in pixels.'''
    spot_radius_px = detection.get_object_radius_pixel(
        voxel_size_nm=voxel_size_nm, object_radius_nm=object_radius_nm, ndim=len(object_radius_nm))
    rna_log = stack.log_filter(rna, sigma=spot_radius_px)
    mask = detection.local_maximum_detection(rna_log, min_distance=spot_radius_px)
    if thresh is None:
        thresh = detection.automated_threshold_setting(rna_log, mask)
    spots, _ = detection.spots_thresholding(rna_log, mask, thresh)
    return spots


def show_tile(rootpath, exp_path, condition, s, t, ch):
    '''z stack of a FISH tile, for the round t and the channel ch. Dimension: (z,x,y)'''
    folder_tif = rootpath + exp_path + 'Fish/' + condition + '/'
    s_, t_, ch_ = str(s).zfill(2), str(t).zfill(2), str(ch).zfill(2)
    name_tif = '*_t{}_s{}_z*_ch{}.tif'.format(t_, s_, ch_)
    frame_names = list(pathlib.Path(folder_tif).glob(name_tif))
    images = assemble_z_stack(frame_names)
    return (images)


def show_mip(rootpath, exp_path, condition, s, t, ch):
    '''Maximum intensity projection of a FISH tile, for the round t and the channel ch'''
    return np.max(show_tile(rootpath, exp_path, condition, s, t, ch), axis=0)


##### MOSAIC OF THE FISH TILES

def load_snake_pattern(rootpath, exp_path, condition, default_pattern):
    '''
    Position of each tile in the mosaic: 2D array with the tile index s at each position (None where there is no tile).
    The pattern of the condition (EXP_PATH/Fish/pattern_snake/CONDITION.npy) if it exists, else the default one (5x5)
    '''
    pattern_snake_path = rootpath + exp_path + "Fish/pattern_snake/" + condition + ".npy"
    if pathlib.Path(pattern_snake_path).exists():
        return np.load(pattern_snake_path, allow_pickle=True)
    return np.load(default_pattern)


def remove_small_regions(mask, min_size):
    """
    Set to 0 any connected region in the mask with fewer than `min_size` pixels.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (2D array).
    min_size : int
        Minimum number of pixels a region must have to be kept.

    Returns
    -------
    np.ndarray
        Cleaned binary mask.
    """
    from scipy import ndimage
    # Label connected components
    labeled, num = ndimage.label(mask)

    # Count pixels in each component
    sizes = np.bincount(labeled.ravel())

    # Mask out small components
    remove = sizes < min_size
    remove_mask = remove[labeled]

    cleaned = mask.copy()
    cleaned[remove_mask] = 0
    return cleaned


def remove_cell_border(im):
    '''Remove the cells (labels) touching the border of the image. The image is modified in place and returned'''
    # Look where are the cells
    x, y = np.where(im > 0)
    # Get the index of the cells
    values_pos = im[im > 0]
    # Get the shape of the image
    X_shape, Y_shape = im.shape[0] - 1, im.shape[1] - 1
    # Check if there are cells at the border of the image
    bool_pos_lim = np.any(((x == 0), (y == 0), (x == X_shape), (y == Y_shape)), axis=0)
    # Get which index of cells are at the border
    ind_lim = values_pos[bool_pos_lim]
    # Remove the cells with those indices
    bool_im = np.any(np.asarray([im == i for i in ind_lim]), axis=0)
    im[bool_im] = 0
    return (im)


def merge_snake_tiles(tiles, pattern_snake, im_size, overlap, remove_border=False):
    '''
    Mosaic of the FISH tiles, placed following pattern_snake.
    The labels of the tile s are shifted by 1000*s, so that each label is unique in the mosaic.
    Where two tiles overlap, the first tile placed has the priority: the labels of the next tile that overlap
    the mosaic are erased.
    tiles: list of the 2D images of the tiles, indexed by s
    remove_border: remove the cells touching the border of their tile before placing it
    Returns the mosaic, and the tiles as placed in the mosaic {s: tile}
    '''
    n_rows, n_cols = pattern_snake.shape
    total_image = np.zeros((im_size * n_rows - (n_rows - 1) * overlap,
                            im_size * n_cols - (n_cols - 1) * overlap))
    placed = {}

    # Do the snake
    for i in range(n_rows):
        for j in range(n_cols):
            id_tile = pattern_snake[i, j]
            if id_tile is None:
                continue

            im = np.copy(tiles[id_tile])
            im[im > 0] = im[im > 0] + 1000 * id_tile
            if remove_border:
                im = remove_cell_border(im)

            # Rows from top to bottom, columns from left to right
            xmin, xmax = i * (im_size - overlap), i * (im_size - overlap) + im_size
            ymin, ymax = j * (im_size - overlap), j * (im_size - overlap) + im_size

            # Fix where the two indices overlay
            overlap_mask = (total_image[xmin:xmax, ymin:ymax] > 0) & (im > 0)  # get the mask for overlay
            index_overlap_1 = np.unique(im[overlap_mask])  # get indices in the first image

            # Priority to the first image: erase indices in the second image
            for x in index_overlap_1:
                im[im == x] = 0

            placed[id_tile] = im
            total_image[xmin:xmax, ymin:ymax] += im

    return total_image, placed
