'''
Step 3.1: match the live cells (last time frame of the movie) with the smFISH cells.

1. The smFISH mosaic mask is rescaled to the pixel size of the BF images (constants.scale). Objects smaller than
   500 pixels, or whose centroid falls outside their mask, are removed.
2. The translation between the two fields of view is found by brute force on the centroids of the cells
   (matching.translation_*), with the masks of the whole cells or of the nuclei (matching.dapi_mask).
3. Each live cell is paired with the smFISH cell that is its reciprocal closest neighbour (closer than
   constants.max_dist). The distance to the second closest cell, on each side, is saved for the quality control of
   the pairs in step 5.

Input:  ROOTPATH/EXP_PATH/Live/CONDITION/track/tracks.csv (step 1.3)
        ROOTPATH/EXP_PATH/Fish/masks/CONDITION/mosaic_mask_reassigned.tiff (step 2.2)
Output: ROOTPATH/EXP_PATH/alignment/CONDITION-alignment.csv: the pairs of cells
        ROOTPATH/EXP_PATH/alignment/CONDITION-overlay.npy: BF mask and aligned smFISH mask, on the BF field of view
        ROOTPATH/EXP_PATH/alignment/CONDITION-matching.png: illustration of the matching

Run: python3 -m 3_match_bf_fish.1_live_fish_matching
'''


# Import local tools
import numpy as np
import tifffile
from PIL import Image
from tools.read_yaml import (
    CONDITION, DAPI_MASK, EXP_PATH, MAX_DIST_MATCHING, N_FRAMES, RATIO_QC, ROOTPATH, SCALE, TRANSLATION_DELTA,
    TRANSLATION_DELTA_REFINED, TRANSLATION_N
)
from tools.tools_bf import load_tracks, load_tracked_mask
from tools.matching import build_overlay, save_matching_illustration, get_translation, match_cells

import cv2


# Fast version of extract_centroid_coords
def extract_centroid_coords(mask, unique_values=None):
    '''
    Extracts the centroid coordinates of the labeled regions in a mask
    :param mask: 2D numpy array with labeled regions
    :param unique_values (optional): list/1D array. if None, loop over all strictly positive values of mask. If not, loop on unique_values
    :return: x_coords, y_coords: lists of x and y coordinates of the centroids
    '''
    # Initialize lists to store the x and y coordinates of the centroids
    x_coords = []
    y_coords = []
    areas = []
    # Filter out the background (label 0)
    x_pos, y_pos = np.where(mask > 0)
    # Take the labels
    mask_values = mask[x_pos, y_pos]
    # Loop through the unique labels
    if unique_values is None:
        unique_values = np.unique(mask_values)
    for i in unique_values:
        ind = np.where(mask_values == i)[0]
        y, x = x_pos[ind], y_pos[ind]
        # Compute centroid (= the average of the x and y coordinates)
        x_coords.append(x.mean())
        y_coords.append(y.mean())
        # Compute area
        area = len(x)
        areas.append(area)
    x_coords, y_coords = np.asarray(x_coords), np.asarray(y_coords)
    coords = np.stack((x_coords, y_coords))
    return unique_values, coords.T, np.asarray(areas)


def filter_segmentation(mask):
    '''
    The mask could have some segmentation issue (SAM2)
    A way to detect error is to check if the centroid falls inside a cell
    If not: remove the segmentation!
    Also remove small cells!
    '''
    labeled_im = mask.copy()
    # Detect problematic cells
    unique_labels = np.unique(labeled_im)
    unique_labels = unique_labels[unique_labels > 0]
    # Compute the centroids of cells
    unique_values, centroids, areas = extract_centroid_coords(mask)
    centroids = np.asarray(np.floor(centroids), dtype=np.int32)  # Cast as an index
    # If the value of the mask is 0 at the centroid position: the centroid is outside
    values = labeled_im[centroids[:, 1], centroids[:, 0]]
    id_val = np.where(values == 0)[0]  # If it falls outside a cell
    problematic_id = unique_labels[id_val]

    # Iterate over the problematic id
    # And remove the cells
    for x in problematic_id:
        bool_x = labeled_im == x
        labeled_im[bool_x] = 0

    # Filter size with size smaller than 500
    problematic_id = unique_values[np.where(areas < 500)[0]]
    print(problematic_id)
    for x in problematic_id:
        bool_x = labeled_im == x
        labeled_im[bool_x] = 0

    return (labeled_im)


def main():
    '''Align the two fields of view and match the cells'''
    # Constants
    scale = SCALE
    max_dist = MAX_DIST_MATCHING  # Do not pick a custom threshold: make it high, and save the distance for each matching, so we can filter after! (do not filter right now)

    # Do you we the DAPI mask or whole cells fish masks for alignment?
    DAPI_mask = DAPI_MASK

    # Initialize variables
    save_path_file = ROOTPATH + EXP_PATH + "alignment/" + CONDITION + "-alignment.csv"

    # BF cells: ID and centroid of the tracked cells at the last time frame, from the table of the tracks (step 1.3)
    tracks = load_tracks(ROOTPATH, EXP_PATH, CONDITION)
    tracks_last = tracks[tracks['t'] == N_FRAMES - 1]
    mask_values_bf = tracks_last['unique_id'].values
    centroids_bf = tracks_last[['centroid_x', 'centroid_y']].values

    # Fish segmentation
    FISH_segmentation = np.asarray(tifffile.imread(ROOTPATH + EXP_PATH + 'Fish/masks/' + CONDITION + "/mosaic_mask_reassigned.tiff"))  # segmentation of FISH
    # Realign according to the scaling factor
    FISH_segmentation = cv2.resize(FISH_segmentation, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    FISH_segmentation = filter_segmentation(FISH_segmentation)

    if DAPI_mask:
        fish_match = np.asarray(tifffile.imread(ROOTPATH + EXP_PATH + 'Fish/masks/' + CONDITION + "/DAPI_masks/mosaic_mask_reassigned.tiff"))
        fish_match = cv2.resize(fish_match, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        fish_match = filter_segmentation(fish_match)
    else:
        fish_match = FISH_segmentation

    # Extract centroids
    mask_values_fish, centroids_fish, areas_fish = extract_centroid_coords(FISH_segmentation)
    mask_values_fish_match, centroids_fish_match, areas_fish_match = extract_centroid_coords(fish_match)

    # Translate
    x_trans, y_trans, error_mat = get_translation(centroids_fish_match, centroids_bf, n=TRANSLATION_N, delta=TRANSLATION_DELTA, delta_refined=TRANSLATION_DELTA_REFINED)
    print(f'Translation: {x_trans:.1f}, {y_trans:.1f}')
    centroids_bf = centroids_bf + np.array([x_trans, y_trans])

    # Match with closest neighbour, and save the table of the pairs
    save_frame = match_cells(mask_values_fish, centroids_fish, mask_values_bf, centroids_bf, max_dist)
    save_frame.to_csv(save_path_file, index=False)

    # SAVE THE OVERLAY: on the whole BF field of view
    # First channel: BF mask of the last time frame with the track ID. Second channel: FISH mask, placed with the alignment
    Image.MAX_IMAGE_PIXELS = 2000000000
    bf_segmentation = load_tracked_mask(ROOTPATH, EXP_PATH, CONDITION, N_FRAMES - 1, tracks)
    overlay_save = build_overlay(bf_segmentation, FISH_segmentation, x_trans, y_trans)
    np.save(ROOTPATH + EXP_PATH + "alignment/" + CONDITION + '-overlay.npy', overlay_save)

    # ILLUSTRATION: global alignment and pairs kept after the quality control
    save_matching_illustration(overlay_save, save_frame, RATIO_QC, ROOTPATH + EXP_PATH + "alignment/" + CONDITION + '-matching.png', title=CONDITION)


if __name__ == "__main__":
    main()
