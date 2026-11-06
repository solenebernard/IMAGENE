'''
Shared code for the cell-attachment figures (panels b and c of the main figure): how cells slow down and
attach during the 1-hour brightfield timelapse that follows UV (step3),
compared with the timelapse before UV (step1).

Only channels with a tracked step3 timelapse and a match_step_1_3/ folder can
be used (canal2).

Data (folder figures_input/, see _common.py):
    step1/track/*_track_per_frame.csv, step3/track/*_track_per_frame.csv
        one row per cell and frame: centroid, step_distance (px moved since the
        previous frame), ... (track_dynamics.py)
    step3/track/*_track_dynamics_n21_t<a>-<b>.csv
        per-track metrics over successive 21-frame windows of step3
        (track_dynamics.py --n_frames 21 --frame_start <a>)
    match_step_1_3/*_match.csv
        step1 last frame <-> step3 first frame (step 3.1)

"Matched cells" = cells present at the end of step1 AND at the start of step3,
so the same cell can be followed before and after UV.
"Net displacement" = straight-line distance between a cell's position in the
first and last frame of a 21-frame window (px in the tables, shown in µm:
1 px = 0.1625 µm); small = the cell stays put.
'''

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import find_contours

import _common as C

Image.MAX_IMAGE_PIXELS = None  # step1's stitched masks are ~57M px

N_FRAMES = 21          # frames per window (= the length of the step1 timelapse)
STEP3_DT_S = C.FRAME_DT_S          # seconds between two step3 frames
PIXEL_SIZE_UM = C.PIXEL_SIZE_UM    # µm per pixel of the step1 / step3 images
WINDOW_MIN = C.minutes(N_FRAMES - 1)  # duration of a 21-frame window: 7 min
STICKING_UM = [5.0, 10.0]  # figure 22: one curve per net-displacement threshold (µm), cells at or below it

# Net-displacement classes and their colours: one blue, light = small. The
# cut-offs are set in µm and converted to px, the unit of the tracking tables.
DISPLACEMENT_BINS_UM = [0, 5, 10, 20, np.inf]
DISPLACEMENT_BINS = [b / PIXEL_SIZE_UM for b in DISPLACEMENT_BINS_UM]  # px: 0, 30.8, 61.5, 123.1, inf
_UM = [f'{b:g}' for b in DISPLACEMENT_BINS_UM[1:-1]]  # 5, 10, 20
DISPLACEMENT_LABELS = ([f'< {_UM[0]} µm'] + [f'{a}–{b} µm' for a, b in zip(_UM[:-1], _UM[1:])]
                       + [f'≥ {_UM[-1]} µm'])
DISPLACEMENT_COLORS = ['#86b6ef', '#2a78d6', '#1c5cab', '#104281']
NOT_TRACKED, NOT_TRACKED_COLOR = 'out of the frame', '#b4b2ab'  # track absent from the window (or a single frame)
CLASS_COLOR = dict(zip(DISPLACEMENT_LABELS, DISPLACEMENT_COLORS)) | {NOT_TRACKED: NOT_TRACKED_COLOR}


##### TABLES

def load_per_frame(canal):
    '''(step1 per-frame table, step3 per-frame table, matched pairs).

    Columns of the two per-frame tables are suffixed _step1 / _step3. The match
    file links step1's label at its LAST frame to step3's label at its FIRST
    frame; pairs translates both into track ids: unique_id_step1, unique_id_step3.'''
    if not (canal.dir / 'match_step_1_3').is_dir():
        raise FileNotFoundError(f'No match_step_1_3/ in {canal.dir} -- the attachment figures need a tracked '
                                f'step3 timelapse matched to step1: run step 3.1 (3_match_bf_fish/1_live_fish_matching.py, movie <-> movie after UV) '
                                f'and 7_control_figures/analysis/2_prepare_figure_inputs.py (see run_control.sh)')
    df1 = pd.read_csv(C.find_one(canal.dir / 'step1' / 'track', '*_ch00_track_per_frame.csv')).add_suffix('_step1')
    df3 = pd.read_csv(C.find_one(canal.dir / 'step3' / 'track', '*_ch00_track_per_frame.csv')).add_suffix('_step3')
    match = pd.read_csv(C.find_one(canal.dir / 'match_step_1_3', '*_match.csv'))

    last_1 = df1[df1['t_step1'] == df1['t_step1'].max()]
    first_3 = df3[df3['t_step3'] == df3['t_step3'].min()]
    pairs = (match[match['match_type'] == 'distance'].astype({'global_id_1': int, 'global_id_2': int})
             .merge(last_1[['global_id_step1', 'unique_id_step1']], left_on='global_id_1', right_on='global_id_step1')
             .merge(first_3[['global_id_step3', 'unique_id_step3']], left_on='global_id_2', right_on='global_id_step3')
             [['unique_id_step1', 'unique_id_step3', 'centroid_distance_px', 'second_nn_distance_1', 'second_nn_distance_2']])
    print(f'canal{canal.key}: {len(pairs)} cell(s) matched between the end of step1 and the start of step3')
    return df1, df3, pairs


##### MASKS AND CONTOURS

def step1_mask_path(canal, t):
    return C.find_one(canal.dir / 'step1' / 'global_masks', f'*_t{t:02d}_ch00_global_mask.png')


def step3_mask_path(canal, t):
    return C.find_one(canal.dir / 'step3' / 'z_slice', f'*_t{t:03d}_z*_ch00_cp_masks.png')


def step3_image_path(canal, t):
    return C.find_one(canal.dir / 'step3' / 'z_slice', f'*_t{t:03d}_z*_ch00.tif')


def step3_offset(canal):
    '''(dx, dy) placing step3's single tile inside step1's field of view.'''
    with open(C.find_one(canal.dir / 'match_step_1_3', '*_alignment_params.json')) as f:
        align = json.load(f)
    if align['scale'] != 1.0:
        raise ValueError('step3 was rescaled to match step1 -- its contours would need rescaling too')
    return align['dx'], align['dy']


def cell_contours(mask_path, gid_to_uid, offset=(0.0, 0.0)):
    '''{track id: [(N, 2) contour in (x, y)]} for the labels in gid_to_uid
    ({label in this frame: track id}), shifted by offset = (dx, dy). Contours
    are traced on each cell's own bounding box, not the full frame.'''
    mask = np.asarray(Image.open(mask_path))
    slices = ndi.find_objects(mask)
    out = {}
    for gid, uid in gid_to_uid.items():
        if gid > len(slices) or slices[gid - 1] is None:
            continue
        sl = slices[gid - 1]
        crop = np.pad(mask[sl] == gid, 1)  # pad so contours close at the box edge
        out[uid] = [np.column_stack((c[:, 1] + sl[1].start - 1 + offset[0], c[:, 0] + sl[0].start - 1 + offset[1]))
                    for c in find_contours(crop.astype(float), 0.5)]
    return out


def plot_trajectories(ax, df, step, ids, t_values, mask_path, offset, title, relative_time=False, colorbar=True):
    '''Centroid trajectories and per-frame contours of the tracks `ids` over
    the frames t_values, both coloured by time. step = 'step1' or 'step3'.
    Positions are drawn in µm (step1's pixel coordinates x PIXEL_SIZE_UM) and
    the colour scale is in minutes (frame number x 21 s).
    relative_time: colour by the time since the window's first frame, so that
    several windows share one colour scale (then draw a single colorbar from
    the returned ScalarMappable, with colorbar=False here).'''
    um = PIXEL_SIZE_UM
    uid_col, t_col = f'unique_id_{step}', f't_{step}'
    sub = df[df[uid_col].isin(ids) & df[t_col].isin(t_values)].sort_values([uid_col, t_col])
    t_ref = min(t_values) if relative_time else 0  # colour value = frame number - t_ref
    norm = plt.Normalize(min(t_values) - t_ref, max(t_values) - t_ref)
    cmap = plt.cm.viridis

    for t in t_values:
        frame = sub[sub[t_col] == t]
        gid_to_uid = dict(zip(frame[f'global_id_{step}'].astype(int), frame[uid_col]))
        for contours in cell_contours(mask_path(t), gid_to_uid, offset).values():
            for c in contours:
                ax.plot(c[:, 0] * um, c[:, 1] * um, color=cmap(norm(t - t_ref)), linewidth=0.4, alpha=0.6)

    for _, track in sub.groupby(uid_col):
        x, y = (track[f'centroid_x_{step}'] + offset[0]) * um, (track[f'centroid_y_{step}'] + offset[1]) * um
        ax.plot(x, y, linewidth=1, c='black', alpha=0.5)
        ax.scatter(x, y, c=track[t_col] - t_ref, cmap=cmap, norm=norm, s=6, zorder=3)

    when = 'time after UV (min)' if step == 'step3' else 'time (min)'
    ax.set_title(f'{title}\n({C.minutes(min(t_values)):.1f} to {C.minutes(max(t_values)):.1f} min'
                 f"{' after UV' if step == 'step3' else ''})")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    if colorbar:
        cbar = plt.colorbar(mappable, ax=ax, label=when, shrink=0.7)
        cbar.formatter = FuncFormatter(lambda frame, _: f'{C.minutes(frame):.0f}')  # frame number -> minutes
        cbar.update_ticks()
    return mappable
