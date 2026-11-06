'''
Prepares the inputs of the figures of the control experiment, from the outputs of run_control.sh.

The figure scripts of the control experiment read one folder per
density in figures_input/ (canal2: high density, canal05: low density), with the layout described in _common.py.
This script builds this layout from the outputs of the scripts of steps 1 and 3, run by run_control.sh in
control/analysis/ (and control/analysis_uv/ for the long movie after UV). Images and masks are linked, not copied;
the tables are converted:

    figures_input/canal{2,05}/
        step0/                     colour barcode before the movie: tiles, tile masks (M1), global mask (1.2)
        step1/                     movie: global masks (1.2), stitched images (1.1), tracks (1.3)
        step{5,7}_afterPFA/        colour barcode after fixation: tiles, tile masks (M1), global mask (1.2)
        step3/                     long movie after UV (high density): images, masks (1.2) and tracks (1.3)
        match_step_0_1/            barcode before the movie <-> first frame of the movie: tracking (1.3) of the
                                   two frames
        match_step_1_{5,7}/        last frame of the movie <-> barcode after fixation: matching of step 3.1
        match_step_1_3/            last frame of the movie <-> first frame of the movie after UV: matching of
                                   step 3.1 (scale 1)

Run: python3 -m 7_control_figures.analysis.2_prepare_figure_inputs   (from code/)
'''

import contextlib
import io
import json
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # code/, for tools/
with contextlib.redirect_stdout(io.StringIO()):
    from tools.read_yaml import ROOTPATH, SCALE, config

Image.MAX_IMAGE_PIXELS = None
CONTROL = pathlib.Path(ROOTPATH + config['control']['exp_path'])
ANALYSIS, ANALYSIS_UV, OUTPUT = CONTROL / 'analysis', CONTROL / 'analysis_uv', CONTROL / 'figures_input'
SCALE_AFTER_FIXATION = SCALE   # 40/63: objective of the fixed cells

# density -> (folder of the figures, folder of the barcode after fixation, original folder names)
DENSITIES = {
    'highdensity': ('canal2', 'step5_afterPFA'),
    'lowdensity': ('canal05', 'step7_afterPFA'),
}


def link(source, target):
    '''Link target -> source (relative), replacing an older link'''
    target = pathlib.Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    os.symlink(os.path.relpath(source, target.parent), target)


def raw_prefix(folder, pattern):
    '''Prefix of the original images of a folder (the part of the name before _t / _s)'''
    for path in sorted(pathlib.Path(folder).iterdir()):
        match = re.match(pattern, path.name)
        if match and not path.name.startswith('.'):
            return match.group('prefix')
    raise FileNotFoundError(f'No image matching {pattern} in {folder}')


def live(condition):
    return ANALYSIS / 'Live' / condition


def global_mask(condition, t, analysis=ANALYSIS):
    return analysis / 'Live' / condition / 'global_masks' / f'global_mask_t{t:02d}.png'


def tile_masks(condition):
    '''{s: path} of the cellpose masks of the tiles (M1) of a single-frame acquisition'''
    masks = {}
    for path in (live(condition) / 'non_merged' / 'finetune_z2').glob('*_cp_masks.png'):
        match = re.search(r'_t00_s(\d+)_z00_ch00_cp_masks\.png$', path.name)
        if match and not path.name.startswith('.'):
            masks[int(match.group(1))] = path
    return masks


def barcode_step(condition, original, target, prefix):
    '''Tiles of all the channels, tile masks, global mask and table of the tiles of the global cells'''
    for path in sorted(pathlib.Path(original).glob(f'{prefix}_s*_ch*.tif')):
        if not path.name.startswith('.') and not path.name.endswith('_cp_masks.png'):
            link(path, target / path.name)
    for s, path in tile_masks(condition).items():
        link(path, target / f'{prefix}_s{s:02d}_ch00_cp_masks.png')
    link(global_mask(condition, 0), target / 'global_masks' / f'{prefix}_ch00_global_mask.png')
    id_map = pd.read_csv(live(condition) / 'global_masks' / 'global_id_map.csv')
    id_map[['global_id', 'local_id', 's']].to_csv(target / 'global_masks' / f'{prefix}_ch00_global_id_map.csv', index=False)


def cell_ids(condition, t):
    '''Identifiers of the cells of the global mask of a frame (1.2)'''
    id_map = pd.read_csv(live(condition) / 'global_masks' / 'global_id_map.csv')
    return id_map.loc[id_map['t'] == t, 'global_id'].values


def with_unmatched(match, ids_1, ids_2):
    '''Adds a row for each cell of the two images without a pair, as read by the figures
    (match_type unmatched_1: cell of the first image, unmatched_2: cell of the second image)'''
    alone_1 = np.setdiff1d(ids_1, match['global_id_1'])
    alone_2 = np.setdiff1d(ids_2, match['global_id_2'])
    rows = [pd.DataFrame({'global_id_1': alone_1, 'match_type': 'unmatched_1'}),
            pd.DataFrame({'global_id_2': alone_2, 'match_type': 'unmatched_2'})]
    return pd.concat([match] + rows, ignore_index=True)[match.columns]


def second_nearest(points, others):
    '''Distance of each point to its second nearest point of others'''
    distances, _ = cKDTree(others).query(points, k=2)
    return distances[:, 1]


def iou_of_pairs(mask_1, mask_2, pairs):
    '''Intersection over union of the cells (id_1, id_2) of two masks of the same size'''
    m1, m2 = mask_1.ravel().astype(np.int64), mask_2.ravel().astype(np.int64)
    area_1, area_2 = np.bincount(m1), np.bincount(m2)
    both = (m1 > 0) & (m2 > 0)
    codes = m1[both] * (m2.max() + 1) + m2[both]
    overlap = pd.Series(codes).value_counts()
    values = []
    for a, b in pairs:
        inter = overlap.get(a * (m2.max() + 1) + b, 0)
        values.append(inter / (area_1[a] + area_2[b] - inter) if inter else 0.0)
    return np.array(values)


def match_from_link(density, target, prefix_0, prefix_1):
    '''match_step_0_1: pairs of the tracking (1.3) of the two frames [barcode before the movie, first frame of the movie]'''
    tracks = pd.read_csv(live(f'link_{density}') / 'track' / 'tracks.csv')
    frame_0, frame_1 = tracks[tracks['t'] == 0], tracks[tracks['t'] == 1]
    pairs = frame_0.merge(frame_1, on='unique_id', suffixes=('_1', '_2'))
    xy_1, xy_2 = pairs[['centroid_x_1', 'centroid_y_1']].values, pairs[['centroid_x_2', 'centroid_y_2']].values
    iou = iou_of_pairs(np.asarray(Image.open(global_mask(f'step0_{density}', 0))),
                       np.asarray(Image.open(global_mask(f'step1_{density}', 0))),
                       zip(pairs['global_id_1'], pairs['global_id_2']))
    match = pd.DataFrame({
        'global_id_1': pairs['global_id_1'], 'global_id_2': pairs['global_id_2'], 'iou': iou,
        'centroid_distance_px': np.hypot(*(xy_1 - xy_2).T),
        'second_nn_distance_1': second_nearest(xy_1, frame_1[['centroid_x', 'centroid_y']].values),
        'second_nn_distance_2': second_nearest(xy_2, frame_0[['centroid_x', 'centroid_y']].values),
        'match_type': np.where(iou > 0, 'overlap', 'distance'),
    })
    n_pairs = len(match)
    match = with_unmatched(match, cell_ids(f'step0_{density}', 0), cell_ids(f'step1_{density}', 0))
    folder = target / 'match_step_0_1'
    folder.mkdir(parents=True, exist_ok=True)
    match.to_csv(folder / f'{prefix_0}_x_{prefix_1}_match.csv', index=False)
    return n_pairs


def match_from_alignment(alignment_csv, tracks, last_frame, folder, prefix_1, prefix_2, scale, shape_1, ids_1, ids_2):
    '''match_step_1_N: pairs of the matching of step 3.1, with the alignment (offset of the second image in the first)'''
    alignment = pd.read_csv(alignment_csv)
    last = tracks[tracks['t'] == last_frame].set_index('unique_id')
    bf = last.loc[alignment['BF']]
    # Step 3.1 adds a translation to the centroids of the movie to align them with the other image:
    # the other image is placed at minus this translation in the frame of the movie
    x_trans = float(np.median(alignment['BF_center_x'].values - bf['centroid_x'].values))
    y_trans = float(np.median(alignment['BF_center_y'].values - bf['centroid_y'].values))
    match = pd.DataFrame({
        'global_id_1': bf['global_id'].values, 'global_id_2': alignment['Fish'].values, 'iou': np.nan,
        'centroid_distance_px': alignment['Distance'].values,
        'second_nn_distance_1': alignment['Distance_2nd_bf_to_fish'].values,
        'second_nn_distance_2': alignment['Distance_2nd_fish_to_bf'].values,
        'match_type': 'distance',
    })
    n_pairs = len(match)
    match = with_unmatched(match, ids_1, ids_2)
    folder.mkdir(parents=True, exist_ok=True)
    match.to_csv(folder / f'{prefix_1}_x_{prefix_2}_match.csv', index=False)
    params = {'folder_2_channel': 'ch00', 'scale': scale, 'dx': -x_trans, 'dy': -y_trans, 'folder_1_shape': list(shape_1)}
    with open(folder / f'{prefix_1}_x_{prefix_2}_alignment_params.json', 'w') as f:
        json.dump(params, f)
    return n_pairs, params


for density, (canal, after_folder) in DENSITIES.items():
    if not live(f'step1_{density}').is_dir():
        print(f'{density}: no outputs of run_control.sh, skipped')
        continue
    target = OUTPUT / canal
    original_0 = CONTROL / 'BarcodeBeforeTimelapse' / f'canal_{density}'
    original_1 = CONTROL / 'Live' / f'canal_{density}'
    original_after = CONTROL / 'BarcodeAfterUV' / f'canal_{density}'
    prefix_0 = raw_prefix(original_0, r'^(?P<prefix>.+)_s\d+_ch00\.tif$')
    prefix_1 = raw_prefix(original_1, r'^(?P<prefix>.+)_t\d+_s\d+_z\d+_ch00\.tif$')
    prefix_after = raw_prefix(original_after, r'^(?P<prefix>.+)_s\d+_ch00\.tif$')

    # Colour barcodes: before the movie (step0) and after fixation
    barcode_step(f'step0_{density}', original_0, target / 'step0', prefix_0)
    barcode_step(f'afterUV_{density}', original_after, target / after_folder, prefix_after)

    # Movie (step1): global masks, stitched images, tracks
    tracks_1 = pd.read_csv(live(f'step1_{density}') / 'track' / 'tracks.csv')
    frames = sorted(tracks_1['t'].unique())
    for t in frames:
        link(global_mask(f'step1_{density}', t), target / 'step1' / 'global_masks' / f'{prefix_1}_t{t:02d}_ch00_global_mask.png')
        stitched = live(f'step1_{density}') / f'BF_custom_stitched_merged_t{t:02d}_z00_ch00.tif'
        if stitched.exists():
            link(stitched, target / 'step1' / 'global_masks' / f'{prefix_1}_t{t:02d}_ch00_stitched.tif')
    (target / 'step1' / 'track').mkdir(parents=True, exist_ok=True)
    tracks_1[['unique_id', 't', 'global_id']].to_csv(target / 'step1' / 'track' / f'{prefix_1}_ch00_tracks.csv', index=False)
    shape_1 = np.asarray(Image.open(global_mask(f'step1_{density}', frames[-1]))).shape

    # Matching: barcode before the movie <-> movie, movie <-> barcode after fixation
    n_01 = match_from_link(density, target, prefix_0, prefix_1)
    n_after, params = match_from_alignment(ANALYSIS / 'alignment' / f'step1_{density}-alignment.csv', tracks_1, frames[-1],
                                           target / f'match_step_1_{after_folder[4]}', prefix_1, prefix_after,
                                           SCALE_AFTER_FIXATION, shape_1,
                                           cell_ids(f'step1_{density}', frames[-1]), cell_ids(f'afterUV_{density}', 0))
    print(f'{canal} ({density}): {len(frames)} frames, {tracks_1["unique_id"].nunique()} tracks | '
          f'barcode -> movie: {n_01} pairs | movie -> after fixation: {n_after} pairs, offset ({params["dx"]:.0f}, {params["dy"]:.0f})')

    # Long movie after UV (step3), if it was analysed
    if not live(f'step3_{density}').is_dir():
        continue
    original_3 = CONTROL / 'LongMovieAfterUV' / f'canal_{density}'
    prefix_3 = raw_prefix(original_3, r'^(?P<prefix>.+)_t\d+_z\d+_ch00\.tif$')
    tracks_3 = pd.read_csv(live(f'step3_{density}') / 'track' / 'tracks.csv')
    for t in sorted(tracks_3['t'].unique()):
        image = next(original_3.glob(f'{prefix_3}_t{t:03d}_z02_ch00.tif'))
        link(image, target / 'step3' / 'z_slice' / image.name)
        # The global mask of the single tile has the identifiers of the tracks table
        link(global_mask(f'step3_{density}', t), target / 'step3' / 'z_slice' / f'{prefix_3}_t{t:03d}_z02_ch00_cp_masks.png')
    (target / 'step3' / 'track').mkdir(parents=True, exist_ok=True)
    tracks_3[['unique_id', 't', 'global_id']].to_csv(target / 'step3' / 'track' / f'{prefix_3}_ch00_tracks.csv', index=False)
    n_13, params = match_from_alignment(ANALYSIS_UV / 'alignment' / f'step1_{density}-alignment.csv', tracks_1, frames[-1],
                                        target / 'match_step_1_3', prefix_1, prefix_3, 1.0, shape_1,
                                        cell_ids(f'step1_{density}', frames[-1]), cell_ids(f'step3_{density}', 0))
    print(f'{canal} ({density}): long movie after UV: {tracks_3["t"].nunique()} frames, {tracks_3["unique_id"].nunique()} tracks | '
          f'movie -> movie after UV: {n_13} pairs, offset ({params["dx"]:.0f}, {params["dy"]:.0f})')
