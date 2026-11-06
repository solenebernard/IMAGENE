'''
Helper functions used by the figures and by the modules of this folder (the tracking and the matching themselves are
done by the scripts of steps 1 and 3 of the pipeline):
    - global masks:     pick_mask, discover_global_masks, extract_centroids
    - alignment:        rescale_mask, build_aligned_canvas
    - timelapse masks:  discover_timelapse_masks (and the functions it calls), load_mask
'''

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None  # the global masks of the whole field of view are up to ~220 million pixels


##### FILES

def find_one(folder, pattern, required=True):
    '''The first file matching pattern in folder ('._*' macOS metadata files skipped).'''
    candidates = [f for f in sorted(folder.glob(pattern)) if not f.name.startswith('.')]
    if not candidates:
        if required:
            raise FileNotFoundError(f'No {pattern} in {folder}')
        return None
    return candidates[0]


##### GLOBAL MASKS

_GMASK_RE = re.compile(
    r'^(?P<prefix>.+?)(?:_t(?P<t>\d+))?_(?P<channel>ch\d+)_global_mask\.png$'
)


def discover_global_masks(global_masks_dir, channel):
    '''Scan global_masks_dir for one channel's stitched mask(s).

    Returns (single_path, t_paths, prefix):
      - single_path is set (t_paths empty) for a single-timepoint folder.
      - t_paths = {t: path} is set (single_path is None) for a timelapse.
    '''
    single_path = None
    t_paths: dict[int, Path] = {}
    prefix = ''
    for f in sorted(global_masks_dir.glob(f'*_{channel}_global_mask.png')):
        if f.name.startswith('.'):
            continue
        m = _GMASK_RE.match(f.name)
        if not m or m.group('channel') != channel:
            continue
        prefix = m.group('prefix')
        if m.group('t') is not None:
            t_paths[int(m.group('t'))] = f
        else:
            single_path = f
    return single_path, t_paths, prefix


def pick_mask(folder, channel, endpoint):
    '''Load the relevant global mask (uint32 labels) for one folder.

    endpoint: 'first' or 'last' - which time frame to use if the folder is
    a timelapse; ignored for single-timepoint folders.

    Returns (mask, t_used, prefix). t_used is None for single-timepoint folders.
    '''
    global_masks_dir = folder / 'global_masks'
    single_path, t_paths, prefix = discover_global_masks(global_masks_dir, channel)
    if not prefix:
        raise FileNotFoundError(
            f'No *_{channel}_global_mask.png found in {global_masks_dir}. '
            f'Run step 1.2 (1_bf_analysis/2_merge_masks.py), then 7_control_figures/analysis/2_prepare_figure_inputs.py (see run_control.sh).'
        )
    if t_paths:
        t_used = max(t_paths) if endpoint == 'last' else min(t_paths)
        path = t_paths[t_used]
    else:
        t_used, path = None, single_path

    mask = np.asarray(Image.open(path)).astype(np.uint32)
    return mask, t_used, prefix


def extract_centroids(mask):
    '''Labels and their (x, y) = (col, row) centroids, one per label > 0.'''
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return labels, np.zeros((0, 2))
    centroids_rc = np.asarray(
        ndi.center_of_mass(np.ones_like(mask, dtype=np.float64), labels=mask, index=labels)
    ).reshape(-1, 2)
    return labels, centroids_rc[:, [1, 0]]


##### ALIGNMENT

def rescale_mask(mask, scale):
    '''Nearest-neighbor rescale (order=0) so integer labels aren't corrupted.'''
    return ndi.zoom(mask, zoom=scale, order=0)


def build_aligned_canvas(mask_1_shape, mask_2, dx, dy):
    '''Place mask_2 into a canvas the size of mask_1, shifted by (dx, dy).'''
    H, W = mask_1_shape
    h2, w2 = mask_2.shape
    x0, y0 = int(round(dx)), int(round(dy))

    canvas = np.zeros((H, W), dtype=mask_2.dtype)
    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    w = min(w2 - src_x0, W - dst_x0)
    h = min(h2 - src_y0, H - dst_y0)
    if w > 0 and h > 0:
        canvas[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = mask_2[src_y0:src_y0 + h, src_x0:src_x0 + w]
    return canvas


##### TIMELAPSE MASKS

_TIMELAPSE_GMASK_RE = re.compile(r'^(?P<prefix>.+?)_t(?P<t>\d+)_(?P<channel>ch\d+)_global_mask\.png$')


_ZSLICE_MASK_RE = re.compile(
    r'^(?P<prefix>.+?)_t(?P<t>\d+)_z(?P<z>\d+)_(?P<channel>ch\d+)_cp_masks\.png$'
)


_TILED_HINT_RE = re.compile(r'_s\d+_t\d+|_t\d+_s\d+', re.IGNORECASE)


def _pick_z(z_to_path, preferred_z):
    '''Select one z-slice path: preferred_z if available, otherwise smallest z.'''
    if preferred_z is not None and preferred_z in z_to_path:
        return z_to_path[preferred_z]
    return z_to_path[min(z_to_path)]


def discover_multi_fov_masks(global_masks_dir, channel):
    '''Scan global_masks_dir for one channel's per-timepoint stitched masks.'''
    t_paths: dict[int, Path] = {}
    prefix = ''
    for f in sorted(global_masks_dir.glob(f'*_t*_{channel}_global_mask.png')):
        if f.name.startswith('.'):
            continue
        m = _TIMELAPSE_GMASK_RE.match(f.name)
        if not m or m.group('channel') != channel:
            continue
        t_paths[int(m.group('t'))] = f
        prefix = m.group('prefix')
    return t_paths, prefix


def discover_single_fov_masks(z_slice_dir, channel, preferred_z):
    '''Scan z_slice_dir for one channel's per-timepoint (untiled) Cellpose masks.'''
    tz_paths: dict[int, dict[int, Path]] = defaultdict(dict)
    prefix = ''
    for f in sorted(z_slice_dir.glob(f'*_{channel}_cp_masks.png')):
        if f.name.startswith('.'):
            continue
        m = _ZSLICE_MASK_RE.match(f.name)
        if not m or m.group('channel') != channel:
            continue
        t, z = int(m.group('t')), int(m.group('z'))
        tz_paths[t][z] = f
        prefix = m.group('prefix')
    t_paths = {t: _pick_z(z_dict, preferred_z) for t, z_dict in tz_paths.items()}
    return t_paths, prefix


def discover_timelapse_masks(input_dir, channel, preferred_z):
    '''Find one mask per timepoint for input_dir, trying multi-FOV then single-FOV.

    Returns (t_paths, prefix, mode) where mode describes which layout was used.
    '''
    global_masks_dir = input_dir / 'global_masks'
    if global_masks_dir.is_dir():
        t_paths, prefix = discover_multi_fov_masks(global_masks_dir, channel)
        if t_paths:
            return t_paths, prefix, 'multi-FOV (global_masks/)'

    z_slice_dir = input_dir / 'z_slice'
    if z_slice_dir.is_dir():
        t_paths, prefix = discover_single_fov_masks(z_slice_dir, channel, preferred_z)
        if t_paths:
            return t_paths, prefix, 'single-FOV (z_slice/)'
        if any(_TILED_HINT_RE.search(f.name) for f in z_slice_dir.glob(f'*_{channel}_cp_masks.png')):
            raise FileNotFoundError(
                f'{input_dir} looks tiled (multi-FOV) but has no global_masks/*_{channel}_global_mask.png. '
                f'Run step 1.2 (1_bf_analysis/2_merge_masks.py), then 7_control_figures/analysis/2_prepare_figure_inputs.py (see run_control.sh).'
            )

    raise FileNotFoundError(
        f"No timelapse masks found for channel '{channel}' in {input_dir} "
        f'(looked for global_masks/*_{channel}_global_mask.png '
        f'and z_slice/*_{channel}_cp_masks.png).'
    )


def load_mask(path):
    '''Load a Cellpose/stitched PNG mask as uint32 labels.'''
    return np.asarray(Image.open(path)).astype(np.uint32)
