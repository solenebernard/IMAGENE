'''
Matching of the live cells with the fixed cells (panel e of the main figure of the control experiment).

Three panels over the same field of view, in step1's pixel frame:
  left    segmentation of the last frame of the live-cell timelapse (step1), green
  middle  segmentation of the fixed cells (after PFA), red; the after-PFA scan
          is rescaled (40x -> 63x) and shifted into step1's frame with the
          alignment found by step 3.1
  right   overlay of the two: every cell dim, the cells of the pairs passing
          the match QC bright (overlapping live + fixed pixels look yellow),
          each such pair joined by a line between the two centroids
Pairs come from the step1 <-> after-PFA match (mutual nearest centroid); a pair
passes the QC if 1 - distance / distance to the second-nearest cell >= --ratio_qc
on both sides (0.6 by default).

The field of view is the part of step1 covered by the after-PFA scan, or a
square of --window px at its centre (default 2000 px = 325 µm; 0 = all of it).

Output: control/results/figures/canal<N>/4_matching_live_fixed.png

Usage:
    python3 -m 7_control_figures.4_matching_live_fixed --canal 2
    python3 -m 7_control_figures.4_matching_live_fixed --canal 2 --window 0 --ratio_qc 0.7
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this folder, for _common.py and the libraries
import _common as C
import _illustration_lib as L

DIM, BRIGHT = 110, 255          # mask intensity of the other cells / of the QC-passing pairs
LINE_COLOR = '#4da3ff'          # centroid-to-centroid lines
SCALE_BAR_UM = 50

parser = C.base_parser(__doc__)
parser.add_argument('--ratio_qc', type=float, default=C.RATIO_QC, help='Match QC in [0, 1], larger = stricter (default: control.ratio_qc of config.yaml).')
parser.add_argument('--window', type=int, default=2000,
                    help='Side (px, step1 frame) of the square shown, at the centre of the region covered by the '
                         'after-PFA scan (default 2000; 0 = the whole region).')
args = parser.parse_args()
canal = C.get_canal(args.root, args.canal)
L.configure(canal.dir)

# --- Masks: step1's last frame, and the after-PFA mask aligned into its frame
mask_live, _, _ = L.pick_mask(L.CANAL_DIR / L.STEP1_FOLDER, 'ch00', endpoint='last')
mask_live = mask_live.astype(np.uint32)
mask_fixed_raw, _, _ = L.pick_mask(L.CANAL_DIR / L.STEP7_FOLDER, 'ch00', endpoint='first')
alignment = L.load_alignment_params(L.CANAL_DIR / L.MATCH_1_7_DIR)
mask_fixed = L.build_aligned_canvas(mask_live.shape, L.rescale_mask(mask_fixed_raw, alignment['scale']),
                                    alignment['dx'], alignment['dy']).astype(np.uint32)

# --- Field of view: the region covered by the after-PFA scan (or a square at its centre)
rows, cols = np.nonzero(mask_fixed)
r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
if args.window:
    rc, cc, half = (r0 + r1) // 2, (c0 + c1) // 2, args.window // 2
    r0, r1 = max(r0, rc - half), min(r1, rc + half)
    c0, c1 = max(c0, cc - half), min(c1, cc + half)
live, fixed = mask_live[r0:r1, c0:c1], mask_fixed[r0:r1, c0:c1]


def centroids(mask):
    '''Centroid (row, col) of each label of the mask, indexed by label.'''
    r, c = np.nonzero(mask)
    labels = mask[r, c].astype(np.int64)
    area = np.bincount(labels)
    ids = np.nonzero(area)[0]
    return pd.DataFrame({'row': np.bincount(labels, weights=r)[ids] / area[ids],
                         'col': np.bincount(labels, weights=c)[ids] / area[ids]}, index=ids)


cent_live, cent_fixed = centroids(live), centroids(fixed)

# --- Pairs passing the QC, with both cells in the field of view
match = pd.read_csv(C.find_one(L.CANAL_DIR / L.MATCH_1_7_DIR, '*_match.csv'))
match = match[match['match_type'] == 'distance']
qc = C.passes_ratio_qc(match['centroid_distance_px'], match['second_nn_distance_1'],
                       match['second_nn_distance_2'], args.ratio_qc)
pairs = match[qc & match['global_id_1'].isin(cent_live.index) & match['global_id_2'].isin(cent_fixed.index)]
id_live, id_fixed = pairs['global_id_1'].astype(int), pairs['global_id_2'].astype(int)
print(f'Field of view: rows {r0}-{r1}, cols {c0}-{c1} | {len(cent_live)} live, {len(cent_fixed)} fixed cells, '
      f'{len(pairs)} pair(s) passing ratio QC >= {args.ratio_qc} (of {int(qc.sum())} in the whole field)')

# --- Images
rgb_live = np.zeros(live.shape + (3,), dtype=np.uint8)
rgb_live[..., 1] = (live > 0) * 255
rgb_fixed = np.zeros(fixed.shape + (3,), dtype=np.uint8)
rgb_fixed[..., 0] = (fixed > 0) * 255
overlay = np.zeros(live.shape + (3,), dtype=np.uint8)
overlay[..., 1] = np.where(np.isin(live, id_live), BRIGHT, (live > 0) * DIM)
overlay[..., 0] = np.where(np.isin(fixed, id_fixed), BRIGHT, (fixed > 0) * DIM)
p_live, p_fixed = cent_live.loc[id_live].to_numpy(), cent_fixed.loc[id_fixed].to_numpy()

# Size of panel e of the main figure: 3 units wide, 1 high (tools/style.py)
fig, axes = plt.subplots(1, 3, figsize=(3 * C.PANEL_UNIT, C.PANEL_UNIT))
for ax, img in zip(axes, (rgb_live, rgb_fixed, overlay)):
    ax.imshow(img, interpolation='nearest')
axes[2].plot([p_live[:, 1], p_fixed[:, 1]], [p_live[:, 0], p_fixed[:, 0]], color=LINE_COLOR, linewidth=0.8,
             marker='.', markersize=2.5)
bar_px, pad = SCALE_BAR_UM / C.PIXEL_SIZE_UM, 0.04 * live.shape[1]
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, live.shape[1])
    ax.set_ylim(live.shape[0], 0)
    ax.plot([pad, pad + bar_px], [live.shape[0] - pad] * 2, color='white', linewidth=3, solid_capstyle='butt')
    ax.text(pad + bar_px / 2, live.shape[0] - 1.4 * pad, f'{SCALE_BAR_UM} µm', color='white', ha='center',
            va='bottom')
axes[0].set_title(f'last frame live cell timelapse\nimaging segmentation\n({len(cent_live)} cells)')
axes[1].set_title(f'fixed cell image\nsegmentation\n({len(cent_fixed)} cells)')
axes[2].set_title(f'pairwise matching after alignment\nand quality control, QC ratio = {args.ratio_qc:g}\n({len(pairs)} pairs)')
fig.tight_layout()
C.save_figure(fig, C.figure_path(args, __file__, canal), args.dpi)
