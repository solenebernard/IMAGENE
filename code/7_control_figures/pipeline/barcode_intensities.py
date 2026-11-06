'''
Step 0 - colors.

For every tile mask (<prefix>_s<NN>_ch00_cp_masks.png, segmented on the BF
channel), measure the signal of every segmented object in the other channels
(ch01, ch02, ch03) and call each object positive/negative in each channel.
A single-FOV (untiled) input is also auto-detected --
<prefix>_ch00_cp_masks.png with no _s<NN> at all - and treated as one tile
(s=0).

The per-tile (local_id) measurements are then linked to the global_id
of the merging of the tiles (step 1.2, path_folder/global_masks/*_global_id_map.csv),
so that objects can be tracked across the stitched mosaic. Single-FOV input
skips this: there's only ever one tile and nothing was stitched, so
global_id is just local_id.

Positive/negative calling uses each object's 90th-percentile pixel intensity
(robust to a few saturated/noisy pixels, unlike a plain mean). q95/q99/q100 are also measured
and saved (extra descriptive stats only - thresholding/positive-calling
below always uses q90). The threshold applied is picked per (tile, channel)
with this priority:
  1. the exact threshold saved for this tile/channel in
     path_folder/global_masks/<prefix>_manual_thresholds.csv, if present
  2. otherwise, the saved threshold of the spatially closest tile that has
     one for this channel (grid distance, via --tilescan_order)
  3. otherwise, the pooled-Otsu threshold (on log1p intensity, since raw
     intensities are strongly right-skewed) computed across all 16 tiles --
     this assumes the cell population in that channel is bimodal (on vs
     off), which is the expected behaviour for these color-labeled cells

Output: path_folder/global_masks/<prefix>_signal_per_object.csv
    s, local_id, global_id, area,
    ch01_q90, ch01_q95, ch01_q99, ch01_q100, ch01_threshold, ch01_positive,
    ...(same for ch02, ch03)

Usage:
    python barcode_intensities.py --path_folder path/to/experiment/Canal2/step0_colors/
'''

# Import local tools
import argparse
import numpy as np
import pandas as pd
import re
import tifffile
from PIL import Image
from pathlib import Path

import ast
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

parser = argparse.ArgumentParser(
    description='Measure per-object signal in ch01/ch02/ch03 for every BF tile mask '
                 'and link it to the global_id of the merged mask (step 1.2).'
)
parser.add_argument(
    '--path_folder', type=Path, required=True,
    help='Folder with tile masks (<prefix>_s<NN>_ch00_cp_masks.png) and channel '
         'tifs (<prefix>_s<NN>_ch0X.tif). Must contain a global_masks/ subfolder '
         'with the *_global_id_map.csv of the merging of the tiles (step 1.2).',
)
parser.add_argument(
    '--tilescan_order', type=str,
    default='[[0,1,2,3],[7,6,5,4],[8,9,10,11],[15,14,13,12]]',
    help='2D Python list: tilescan_order[row][col] = s-index at that grid '
         'position. Used to '
         'find the spatially closest tile with a manually saved threshold.',
)
args = parser.parse_args()

path_folder = args.path_folder
path_global_masks = path_folder / 'global_masks'
channels = ['ch01', 'ch02']

_MASK_RE = re.compile(r'^(?P<prefix>.+)_s(?P<s>\d+)_ch00_cp_masks\.png$')
_SINGLE_FOV_MASK_RE = re.compile(r'^(?P<prefix>.+)_ch00_cp_masks\.png$')


def discover_tile_masks(path_folder):
    '''Scan path_folder for BF tile masks. Two layouts are auto-detected:
      tiled (multi-FOV):  <prefix>_s<NN>_ch00_cp_masks.png
      single FOV:         <prefix>_ch00_cp_masks.png (no _s<NN> at all,
      never tiled/stitched) - treated as one tile, s=0.

    Returns ({s_idx: path}, prefix, is_tiled).'''
    tiles = {}
    prefix = ''
    for f in sorted(path_folder.glob('*_s*_ch00_cp_masks.png')):
        if f.name.startswith('.'):
            continue
        m = _MASK_RE.match(f.name)
        if not m:
            continue
        tiles[int(m.group('s'))] = f
        if not prefix:
            prefix = m.group('prefix')
    if tiles:
        return tiles, prefix, True

    for f in sorted(path_folder.glob('*_ch00_cp_masks.png')):
        if f.name.startswith('.'):
            continue
        m = _SINGLE_FOV_MASK_RE.match(f.name)
        if not m:
            continue
        tiles[0] = f
        prefix = m.group('prefix')
        break
    return tiles, prefix, False


def build_s_to_grid(tilescan_order):
    '''s-index -> (row, col) in the tilescan grid.'''
    return {
        s: (r, c)
        for r, row in enumerate(tilescan_order)
        for c, s in enumerate(row)
    }


def load_manual_thresholds(path_global_masks, prefix):
    '''Load the manual thresholds <prefix>_manual_thresholds.csv, if it exists.

    Returns {(s, channel): threshold}, empty if the file doesn't exist yet.
    '''
    path = path_global_masks / f'{prefix}_manual_thresholds.csv'
    if not path.exists():
        return {}
    manual = pd.read_csv(path)
    return {(int(row.s), row.channel): float(row.threshold) for row in manual.itertuples(index=False)}


def closest_manual_threshold(manual_thresholds, s_to_grid, s, ch):
    '''Saved threshold of the spatially closest other tile for this channel.

    Distance is grid distance (row/col in the tilescan), not |s - s'| --
    s-indices follow a snake scan order, not physical adjacency.
    Returns (threshold, source_s) or None if no other tile has one saved.
    '''
    candidates = [(s2, thr) for (s2, ch2), thr in manual_thresholds.items()
                  if ch2 == ch and s2 != s]
    if not candidates:
        return None
    r1, c1 = s_to_grid.get(s, (0, 0))

    def grid_dist(s2):
        r2, c2 = s_to_grid.get(s2, (0, 0))
        return (r1 - r2) ** 2 + (c1 - c2) ** 2

    s2_best, thr_best = min(candidates, key=lambda sc: grid_dist(sc[0]))
    return thr_best, s2_best


tile_masks, prefix, is_tiled = discover_tile_masks(path_folder)
if not tile_masks:
    raise FileNotFoundError(f'No *_ch00_cp_masks.png tiles found in {path_folder}')
print(f"Prefix: '{prefix}'  |  {'tiled' if is_tiled else 'single-FOV (untiled)'}  |  "
      f'{len(tile_masks)} tile(s): {sorted(tile_masks)}')

# Per-object intensity quantiles in each channel, one row per (s, local_id).
# q90 is the one used for positive/negative calling below (robust to a few
# saturated/noisy pixels); q95/q99/q100 are extra descriptive stats only.
QUANTILES = {'q90': 0.90, 'q95': 0.95, 'q99': 0.99, 'q100': 1.00}
records = []
for s, mask_path in sorted(tile_masks.items()):
    s_ = str(s).zfill(2)
    mask = np.asarray(Image.open(mask_path)).astype(np.uint32)

    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        continue

    area = ndi.sum(np.ones_like(mask, dtype=np.float64), labels=mask, index=labels)

    channel_quantiles = {}
    for ch in channels:
        tif_name = f'{prefix}_s{s_}_{ch}.tif' if is_tiled else f'{prefix}_{ch}.tif'
        image = tifffile.imread(path_folder / tif_name)
        # one np.quantile call per object computes all 4 percentiles at once
        channel_quantiles[ch] = np.array([
            np.quantile(image[mask == lbl], list(QUANTILES.values())) for lbl in labels
        ])

    for i, lbl in enumerate(labels):
        row = {'s': s, 'local_id': int(lbl), 'area': float(area[i])}
        for ch in channels:
            row.update({
                f'{ch}_{qname}': float(channel_quantiles[ch][i, j])
                for j, qname in enumerate(QUANTILES)
            })
        records.append(row)

df = pd.DataFrame(records)
print(f'Measured {len(df)} object(s) across {len(tile_masks)} tile(s)')

# Positive/negative call per (tile, channel): manual threshold if saved for this
# exact tile, else the closest tile's manual threshold, else pooled-Otsu (on
# log1p q90 intensity).
manual_thresholds = load_manual_thresholds(path_global_masks, prefix)
s_to_grid = build_s_to_grid(ast.literal_eval(args.tilescan_order))
if manual_thresholds:
    print(f'Loaded {len(manual_thresholds)} manually saved threshold(s) from '
          f"{path_global_masks / f'{prefix}_manual_thresholds.csv'}")

for ch in channels:
    log_values = np.log1p(df[f'{ch}_q90'].to_numpy())
    otsu_threshold = float(np.expm1(threshold_otsu(log_values)))

    n_manual = n_closest = n_otsu = 0
    per_s_threshold = {}
    for s in tile_masks:
        if (s, ch) in manual_thresholds:
            per_s_threshold[s] = manual_thresholds[(s, ch)]
            n_manual += 1
            continue
        closest = closest_manual_threshold(manual_thresholds, s_to_grid, s, ch)
        if closest is not None:
            per_s_threshold[s], _ = closest
            n_closest += 1
        else:
            per_s_threshold[s] = otsu_threshold
            n_otsu += 1

    df[f'{ch}_threshold'] = df['s'].map(per_s_threshold)
    df[f'{ch}_positive'] = df[f'{ch}_q90'] > df[f'{ch}_threshold']
    n_pos = int(df[f'{ch}_positive'].sum())
    print(f'  {ch}: {n_manual} tile(s) manual, {n_closest} tile(s) from closest '
          f'neighbor, {n_otsu} tile(s) pooled-Otsu={otsu_threshold:.1f}  '
          f'->  {n_pos}/{len(df)} positive')

# Link to the global_id of the merged mask (step 1.2) - not
# needed for single-FOV input, since there's only ever one tile (s=0) and
# nothing was stitched, so global_id is just local_id.
if not is_tiled:
    df['global_id'] = df['local_id']
else:
    id_map_candidates = list(path_global_masks.glob(f'{prefix}_*_global_id_map.csv'))
    if not id_map_candidates:
        raise FileNotFoundError(
            f'No *_global_id_map.csv found in {path_global_masks}. '
            f'Run step 1.2 (1_bf_analysis/2_merge_masks.py), then 7_control_figures/analysis/2_prepare_figure_inputs.py (see run_control.sh).'
        )
    id_map_path = id_map_candidates[0]
    id_map = pd.read_csv(id_map_path)[['s', 'local_id', 'global_id']]
    df = df.merge(id_map, on=['s', 'local_id'], how='left')
    n_missing = int(df['global_id'].isna().sum())
    if n_missing:
        print(f'  Note: {n_missing} object(s) have no global_id (redundant/cropped '
              f'tile duplicates dropped during stitching)')

columns = ['s', 'local_id', 'global_id', 'area']
for ch in channels:
    columns += [f'{ch}_{qname}' for qname in QUANTILES] + [f'{ch}_threshold', f'{ch}_positive']
df = df[columns]

path_global_masks.mkdir(parents=True, exist_ok=True)
output_path = path_global_masks / f'{prefix}_signal_per_object.csv'
df.to_csv(output_path, index=False)
print(f'Saved -> {output_path}  ({len(df)} rows)')
