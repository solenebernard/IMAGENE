'''
Search of the translation aligning the last frame of the movie on the barcode after fixation (panel d of the main
figure of the control experiment).

The search is recomputed from the centroids saved in alignment/<density>_barcode_afterUV/ of the control experiment:
  - the centroids of the fixed cells are rescaled to the pixels of the movie (scale of the saved alignment, 40/63)
  - for each translation of a grid, the mean distance from each live cell to the nearest translated fixed cell,
    counting only the cells, on both sides, in the overlap of the two fields of view at this translation
  - a coarse search (200 x 200 translations, +- 2000 px around the offset between the centres of mass of the two sets
    of cells), then a refined one (100 x 100, +- 300 px around the best coarse translation)
Two heat maps (no colour bar): the mean distance over the coarse and over the refined grid of translations, dark for
short distances. The translation found
is checked against the one saved in this folder.

Output: control/results/figures/canal<N>/3_alignment_search.png

Usage:
    python3 -m 7_control_figures.3_alignment_search --canal 2
'''

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this folder, for _common.py and the libraries
import _common as C

Image.MAX_IMAGE_PIXELS = None
DENSITY = {'2': 'highdensity', '05': 'lowdensity'}  # canal key -> density of the folders of the experiment
SEARCH_N, SEARCH_DELTA = 200, 2000.0                # coarse search: grid size, half-width (px)
REFINE_N, REFINE_DELTA = 100, 300.0                 # refined search

parser = C.base_parser(__doc__)
args = parser.parse_args()
C.larger_text()  # larger text in the main figure
canal = C.get_canal(args.root, args.canal)
experiment = Path(args.root).parent   # folder of the control experiment (figures_input/ is in it)
density = DENSITY[canal.key]

# Centroids saved in alignment/, as (x, y): live cells of the last frame, fixed cells
folder = experiment / 'alignment' / f'{density}_barcode_afterUV'
params = json.loads(C.find_one(folder, '*_alignment_params.json').read_text())
live = pd.read_csv(folder / 'step1_t20_centroids.csv')
fixed = pd.read_csv(C.find_one(folder, 'step[0-9]_centroids.csv'))
p0 = live[['centroid_x', 'centroid_y']].to_numpy()
p1 = fixed[['centroid_x', 'centroid_y']].to_numpy() * params['scale']

# Fields of view: the mask of the movie, and the mask after fixation rescaled
shape_0 = tuple(params['folder_1_shape'])
mask_fixed = next(p for p in (experiment / 'BarcodeAfterUV' / f'canal_{density}' / 'global_masks').glob('*_ch00_global_mask.png')
                  if not p.name.startswith('.'))
width, height = Image.open(mask_fixed).size
shape_1 = (round(height * params['scale']), round(width * params['scale']))


def mean_nn_distance(dx, dy):
    '''Mean distance from each live cell to the nearest fixed cell translated by (dx, dy), in the overlap only'''
    (h0, w0), (h1, w1) = shape_0, shape_1
    x0, x1, y0, y1 = max(0.0, dx), min(float(w0), dx + w1), max(0.0, dy), min(float(h0), dy + h1)
    if x1 <= x0 or y1 <= y0:
        return np.inf
    q1 = p1 + [dx, dy]
    inside = lambda p: (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
    a, b = p0[inside(p0)], q1[inside(q1)]
    if len(a) == 0 or len(b) == 0:
        return np.inf
    return float(cKDTree(b).query(a)[0].mean())


def search(cx, cy, n, delta):
    '''Mean distance over a n x n grid of translations around (cx, cy), and the best translation'''
    dxs, dys = np.linspace(cx - delta, cx + delta, n), np.linspace(cy - delta, cy + delta, n)
    distance = np.array([[mean_nn_distance(dx, dy) for dy in dys] for dx in dxs])
    i, j = np.unravel_index(np.argmin(distance), distance.shape)
    return distance, dxs, dys, dxs[i], dys[j]


guess = p0.mean(axis=0) - p1.mean(axis=0)
coarse = search(guess[0], guess[1], SEARCH_N, SEARCH_DELTA)
refined = search(coarse[3], coarse[4], REFINE_N, REFINE_DELTA)
print(f'Translation: dx = {refined[3]:.1f}, dy = {refined[4]:.1f} px '
      f"(saved: dx = {params['dx']:.1f}, dy = {params['dy']:.1f}); minimum {refined[0].min():.1f} px")

# Size of panel d of the main figure: 2 units wide, 1 high (tools/style.py)
fig, axes = plt.subplots(1, 2, figsize=(2 * C.PANEL_UNIT, C.PANEL_UNIT))
for ax, (title, (distance, dxs, dys, *_)) in zip(axes, [('Coarse search', coarse), ('Refined search', refined)]):
    ax.imshow(distance.T, origin='lower', extent=[dxs.min(), dxs.max(), dys.min(), dys.max()], aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('dx (px)')
    ax.set_ylabel('dy (px)')
fig.tight_layout()
C.save_figure(fig, C.figure_path(args, __file__, canal), args.dpi)
