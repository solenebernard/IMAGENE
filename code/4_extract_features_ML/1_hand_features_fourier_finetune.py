'''
Step 4.1: shape features of each live cell at each time frame, from the masks with the track ID (steps 1.2 and 1.3).

Saved in ROOTPATH/EXP_PATH/live_cell_features/:
    - fourier_finetune_CONDITION.csv: Fourier coefficients (a, b, c, d) of the contour of the cell, normalised for
      rotation and starting point (features.fourier_order harmonics), from the cellpose mask of its tile
    - finetune_tile_CONDITION.csv: tile of each cell at each time frame
    - finetune_scikit_CONDITION.csv: region properties of scikit-image (area, perimeter, axes, ...) and displacement
      of the centroid to the next time frame
    - finetune_iou_CONDITION.csv: IoU of the mask with the mask of the next time frame, before (iou) and after
      (iou_abs) aligning the centroids, and time between the two acquisitions (dt, in frames, from the order of the tiles)
    - finetune_area_explored_CONDITION.csv: area of the union of the masks of the cell over the movie, and of its
      convex hull
    - is_on_edge_CONDITION.csv: cells touching the border of the field of view, at each time frame

Run: python3 -m 4_extract_features_ML.1_hand_features_fourier_finetune
'''


# Import local tools
import numpy as np
import pandas as pd
from tools.read_yaml import CONDITION, EXP_PATH, FOURIER_ORDER, LIVE_N_TILES, N_FRAMES, ROOTPATH, Z_FINETUNE
from tools.tools_bf import compute_normalized_efd, get_contour, load_tracked_masks, load_tracked_tile_masks

# To compute hand-crafted features (get_contour and compute_normalized_efd: tools/tools_bf.py)
import itertools
from skimage.measure import regionprops_table


order = FOURIER_ORDER


def compute_dt(s0, s1):
    return (1 + (s1 - s0) / LIVE_N_TILES)


def fourier_features(rootpath, exp_path, condition, masks_tiles):
    name_cols = list(itertools.chain.from_iterable([[f'a_{i}', f'b_{i}', f'c_{i}', f'd_{i}'] for i in range(order)]))

    df = pd.DataFrame()
    df_tile = pd.DataFrame()
    for t in range(masks_tiles.shape[0]):
        for s in range(masks_tiles.shape[1]):
            m = masks_tiles[t, s]
            unique_id = np.unique(m[m > 0])

            for id in unique_id[:]:
                mask_idx = m == id
                contour = get_contour(mask_idx)
                coeffs, locus = compute_normalized_efd(contour, order=order, enforce_d1_nonneg=True)
                dict_feats = {l: x for l, x in zip(name_cols, coeffs.flatten())}

                index = pd.MultiIndex.from_arrays([[id], [t]], names=["cell_bf", "t"])
                df_t_s = pd.DataFrame(dict_feats, index=index)
                df_t_s_tile = pd.DataFrame({'s': s}, index=index)

                df = pd.concat((df, df_t_s))
                df_tile = pd.concat((df_tile, df_t_s_tile))

    df.to_csv(rootpath + exp_path + f'live_cell_features/fourier_finetune_{condition}.csv', index=True)
    df_tile.to_csv(rootpath + exp_path + f'live_cell_features/finetune_tile_{condition}.csv', index=True)


def coords_to_hashes(coords, W):
    return coords[0] * W + coords[1]  # unique pixel index


def is_on_edge_feat(rootpath, exp_path, condition, masks_merged):
    T, H, W = masks_merged.shape
    border = np.stack((masks_merged[:, 0, :], masks_merged[:, -1, :],
                       masks_merged[:, :, 0], masks_merged[:, :, -1]), axis=1)
    x = [np.unique(x[x > 0]) for x in border]
    rows = []
    for t, cells in enumerate(x):
        for cell in cells:
            rows.append([t, cell])
    is_one_edge = pd.DataFrame(rows, columns=['t', 'cell_bf'])
    is_one_edge.to_csv(rootpath + exp_path + f'live_cell_features/is_on_edge_{condition}.csv', index=False)


def total_area_explored_feats(rootpath, exp_path, condition, masks_merged):
    '''
    For each cell, from the union of its masks over the time frames:
        - area_explored_total: number of pixels of the union
        - convex_hull_area: area of the convex hull of the union
        - count_t: number of time frames of the cell
    '''
    from scipy.spatial import ConvexHull, QhullError
    unique_id = np.unique(masks_merged[masks_merged > 0])

    index_pos_mask = np.where(masks_merged > 0)
    value_pos_mask = masks_merged[index_pos_mask]

    rows = []
    for id in unique_id[:]:
        sub_pos_t = index_pos_mask[0][value_pos_mask == id]
        sub_pos = np.stack((index_pos_mask[1][value_pos_mask == id],
                            index_pos_mask[2][value_pos_mask == id]), axis=0)
        unique_sub_pos = np.unique(sub_pos, axis=1)
        size_explored = unique_sub_pos.shape[1]
        n_t = len(np.unique(sub_pos_t))
        # Area of the convex hull of the union (for a 2D hull, scipy calls the area "volume" and the perimeter "area")
        try:
            hull_area = ConvexHull(unique_sub_pos.T).volume
        except (QhullError, ValueError):
            # Fewer than 3 pixels, or all the pixels on a line
            hull_area = np.nan
        rows.append([id, size_explored, hull_area, n_t])

    # Build df once at end
    df = pd.DataFrame(rows, columns=["cell_bf", "area_explored_total", "convex_hull_area", 'count_t'])
    df = df.set_index(["cell_bf"])
    df.to_csv(rootpath + exp_path + f'live_cell_features/finetune_area_explored_{condition}.csv', index=True)


def iou_feats(rootpath, exp_path, condition, masks_merged):
    '''
    Efficient code to compute iou
    '''
    df_tile = pd.read_csv(rootpath + exp_path + f'live_cell_features/finetune_tile_{condition}.csv', index_col=['cell_bf', 't'])
    unique_id = np.unique(masks_merged[masks_merged > 0])

    index_pos_mask = np.where(masks_merged > 0)
    value_pos_mask = masks_merged[index_pos_mask]

    rows = []
    W = masks_merged.shape[2]
    index_tile = set(df_tile.index)
    for t0 in range(masks_merged.shape[0] - 1):
        print(t0)
        t1 = t0 + 1

        index_pos_t0 = [x[index_pos_mask[0] == t0] for x in index_pos_mask[1:]]
        index_pos_t1 = [x[index_pos_mask[0] == t1] for x in index_pos_mask[1:]]
        val_pos_t0 = value_pos_mask[index_pos_mask[0] == t0]
        val_pos_t1 = value_pos_mask[index_pos_mask[0] == t1]

        for id in unique_id:

            if ((id, t0) in index_tile) and ((id, t1) in index_tile):
                s0, s1 = df_tile.loc[(id, t0), 's'], df_tile.loc[(id, t1), 's']

                m0_index = np.array([x[val_pos_t0 == id] for x in index_pos_t0])
                m1_index = np.array([x[val_pos_t1 == id] for x in index_pos_t1])

                h0 = coords_to_hashes(m0_index, W)
                h1 = coords_to_hashes(m1_index, W)

                inter = len(np.intersect1d(h0, h1))
                union = len(np.union1d(h0, h1))

                if (union > 0):
                    iou = inter / union

                # Now iou_abs
                # Remove the centroid in both cases
                centroid0 = np.round(m0_index.mean(axis=1))
                centroid1 = np.round(m1_index.mean(axis=1))

                m0_index -= centroid0[:, None].astype(np.int64)
                m1_index -= centroid1[:, None].astype(np.int64)

                h0 = coords_to_hashes(m0_index, W)
                h1 = coords_to_hashes(m1_index, W)

                inter = len(np.intersect1d(h0, h1))
                union = len(np.union1d(h0, h1))

                if (union > 0):
                    iou_abs = inter / union

                # Store row as tuple
                rows.append((id, t0, iou, iou_abs, compute_dt(s0, s1)))

    # Build df once at end
    df = pd.DataFrame(rows, columns=["cell_bf", "t", "iou", "iou_abs", "dt"])
    df = df.set_index(["cell_bf", "t"])
    df.to_csv(rootpath + exp_path + f'live_cell_features/finetune_iou_{condition}.csv', index=True)


def feats_scikit(list_masks, rootpath, exp_path, condition):
    '''
    - Load the nucleus and whole cell masks
    - Relabel the nucleus mask to match the whole cell id
    - Compute: whole cell area, nucleus area, and distance between centroids
    '''
    n_t = len(list_masks)

    # Compute the features
    # Ratio
    props_all = []
    for t in range(n_t):
        props_cells = regionprops_table(
            list_masks[t],
            properties=('label', 'area', 'centroid', 'bbox_area', 'perimeter',
                        'eccentricity', 'equivalent_diameter_area', 'extent',
                        'feret_diameter_max', 'major_axis_length', 'minor_axis_length',
                        'orientation', 'solidity', 'moments_hu', 'moments_central',
                        'moments_normalized', 'inertia_tensor', 'inertia_tensor_eigvals')
        )
        df_cells = pd.DataFrame(props_cells)
        df_cells['t'] = t

        props_all.append(df_cells)
    # Concatenate all frames
    df_cells_all = pd.concat([c for c in props_all])
    df_cells_all = df_cells_all.rename(columns={'label': 'cell_bf'})
    df_cells_all = df_cells_all.set_index(['cell_bf', 't'])
    # Shift x and y **within each cell_bf**
    df_cells_all['centroid-0_next'] = df_cells_all.groupby(level='cell_bf')['centroid-0'].shift(-1)
    df_cells_all['centroid-1_next'] = df_cells_all.groupby(level='cell_bf')['centroid-1'].shift(-1)
    # Compute Euclidean distance
    df_cells_all['delta_distance'] = np.sqrt((df_cells_all['centroid-0_next'] - df_cells_all['centroid-0'])**2 +
                                             (df_cells_all['centroid-1_next'] - df_cells_all['centroid-1'])**2)
    # Drop helper columns
    df_cells_all = df_cells_all.drop(columns=['centroid-0_next', 'centroid-1_next'])
    df_cells_all.to_csv(rootpath + exp_path + 'live_cell_features/finetune_scikit_' + condition + '.csv', index=True)


def main():
    '''Compute and save all the features'''
    rootpath, exp_path, condition = ROOTPATH, EXP_PATH, CONDITION
    # Cellpose masks of the tiles and global masks, with the track ID of each cell (steps 1.2 and 1.3)
    masks_tiles = load_tracked_tile_masks(rootpath, exp_path, condition, t_max=N_FRAMES, n_tiles=LIVE_N_TILES, z=Z_FINETUNE)
    masks_merged = load_tracked_masks(rootpath, exp_path, condition, t_max=N_FRAMES)

    fourier_features(rootpath, exp_path, condition, masks_tiles)
    feats_scikit(masks_merged, rootpath, exp_path, condition)
    iou_feats(rootpath, exp_path, condition, masks_merged)
    total_area_explored_feats(rootpath, exp_path, condition, masks_merged)
    is_on_edge_feat(rootpath, exp_path, condition, masks_merged)


if __name__ == "__main__":
    main()
