'''
Fraction of cells that stay put over time after UV, two thresholds.

For each cell and window of 21 frames (7 min, one frame every 21 s), the net
displacement is the straight-line distance between its positions in the first
and last frame of the window. Two curves: the percentage of cells whose net
displacement is at most 5 µm, and at most 10 µm (--thresholds_um; 1 px =
0.1625 µm).
Each curve has one point every 10 min after UV (10, 20, ...), computed over
the 21 frames of step3 centred on that time (the 10 before it, that frame, and
the 10 after; --every_min, --window_frames), plus one point for the first 21
frames of step3 (frames 0-20, plotted at their centre, 3.5 min). Times whose
window does not fit in the timelapse (0 min, and the very end) are not plotted.
The point left of the UV line ("live cell imaging") is the same cells before
UV, over the last 21 frames of step1 (the whole timelapse). x axis: time after
photo-immobilization (UV), in min.
The SAME cells are counted at every point: the cells matched between step1 and
step3 that are measured before UV (present in the first and last frame of the
step1 window; 83 in canal2). Every percentage is out of that fixed number: a
cell that is not tracked in a window (it left the field of view, or the
tracking lost it) counts as above the threshold.

Output: control/results/figures/canal<N>/1_sticking_fraction_after_uv.png

Usage:
    python3 -m 7_control_figures.1_sticking_fraction_after_uv --canal 2
    python3 -m 7_control_figures.1_sticking_fraction_after_uv --canal 2 --thresholds_um 5 10
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this folder, for _common.py and the libraries
import _common as C
import _attachment_lib as A

STEP1_X = -8  # min: where the before-UV points are drawn, left of the time axis
CURVE_COLORS = ['#1c5cab', '#6aa5ea', '#0d3a73', '#a9cdf5']  # one per threshold, in --thresholds_um order

parser = C.base_parser(__doc__)
parser.add_argument('--thresholds_um', type=float, nargs='+', default=A.STICKING_UM,
                    help='Net-displacement thresholds (µm), one curve each: cells at or below it (default 5 10).')
parser.add_argument('--every_min', type=float, default=10.0, help='One point every this many minutes after UV (default 10).')
parser.add_argument('--window_frames', type=int, default=21, help='Frames per window, centred on each time point (default 21).')
args = parser.parse_args()
thresholds_px = [t / A.PIXEL_SIZE_UM for t in args.thresholds_um]  # the tracking tables are in px
canal = C.get_canal(args.root, args.canal)
df1, df3, pairs = A.load_per_frame(canal)
n_window = args.window_frames
window_min = C.minutes(n_window - 1)  # duration between the first and last frame of a window


def net_displacement(df, step, ids, t_first, t_last):
    '''Distance (px) between each cell's position at frames t_first and t_last
    (only the cells of `ids` present in both), indexed by the cell's track id.'''
    df = df[df[f'unique_id_{step}'].isin(ids)]
    xy = [df[df[f't_{step}'] == t].set_index(f'unique_id_{step}')[[f'centroid_x_{step}', f'centroid_y_{step}']]
          for t in (t_first, t_last)]
    delta = (xy[1] - xy[0]).dropna()
    return pd.Series(np.hypot(delta.iloc[:, 0], delta.iloc[:, 1]), index=delta.index)


def row(minutes, t0, t1, net):
    '''One point: the number of cells tracked and, per threshold, the % of the fixed n_cells at or below it.'''
    out = {'minutes': minutes, 't_start': t0, 't_end': t1, 'tracked': len(net)}
    for um, px in zip(args.thresholds_um, thresholds_px):
        out[f'pct_le_{um:g}um'] = 100 * int((net <= px).sum()) / n_cells
    return out


# step1: the cells before UV, over its last window_frames frames. The cells
# measured here are the fixed population of every point of the figure.
t1_last = int(df1['t_step1'].max())
net_1 = net_displacement(df1, 'step1', pairs['unique_id_step1'], t1_last - n_window + 1, t1_last)
cells = pairs[pairs['unique_id_step1'].isin(net_1.index)]
n_cells = len(cells)
before = row(STEP1_X, t1_last - n_window + 1, t1_last, net_1)
print(f'{n_cells} cell(s) counted at every point (of {len(pairs)} matched)')

# step3: the first window_frames frames, then one point every --every_min, over the window centred on it
t3_min, t3_max = int(df3['t_step3'].min()), int(df3['t_step3'].max())
rows = []
first_minutes = C.minutes(t3_min + (n_window - 1) / 2)
for minutes in [first_minutes, *np.arange(args.every_min, C.minutes(t3_max) + 1e-9, args.every_min)]:
    centre = int(round(minutes * 60 / C.FRAME_DT_S))            # frame at that time
    t0, t1 = centre - n_window // 2, centre - n_window // 2 + n_window - 1
    if t0 < t3_min or t1 > t3_max:
        print(f'{minutes:g} min: window t{t0}-{t1} does not fit in the timelapse (t{t3_min}-{t3_max}), skipped')
        continue
    rows.append(row(minutes, t0, t1, net_displacement(df3, 'step3', cells['unique_id_step3'], t0, t1)))
table = pd.DataFrame([before | {'minutes': 'before UV'}] + rows)
print(table.round(1).to_string(index=False))
after = pd.DataFrame(rows).sort_values('minutes')

# Size of panel b of the main figure: 1 unit wide and high (tools/style.py)
fig, ax = plt.subplots(figsize=(C.PANEL_UNIT, C.PANEL_UNIT))
for k, (um, color) in enumerate(zip(args.thresholds_um, CURVE_COLORS)):
    col = f'pct_le_{um:g}um'
    ax.plot(after['minutes'], after[col], color=color, linewidth=2, marker='o', markersize=6,
            markeredgecolor='white', markeredgewidth=1.5, label=f'net displacement ≤ {um:g} µm')
    ax.plot(STEP1_X, before[col], linestyle='none', color=color, marker='o', markersize=7,
            markeredgecolor='white', markeredgewidth=1.5)
    # label the before-UV, first and last values; above the highest curve, below the others
    dy = 10 if k == int(np.argmax(args.thresholds_um)) else -16
    for x, y in [(STEP1_X, before[col]), *after.iloc[[0, -1]][['minutes', col]].itertuples(index=False)]:
        ax.annotate(f'{y:.0f}%', (x, y), textcoords='offset points', xytext=(0, dy), ha='center', color=color)
ax.axvline(0, color='#999', linewidth=0.8, linestyle='--', zorder=0)
ax.text(0.5, 3, 'UV', color='#777', ha='left', va='bottom')
ticks = np.arange(0, after['minutes'].max() + 1, args.every_min)
ax.set_xticks([STEP1_X, *ticks], ['live cell\nimaging', *[f'{int(v)}' for v in ticks]])
ax.set_xlim(STEP1_X - 4, after['minutes'].max() + 3)
ax.set_ylim(0, 100)
# Labels and legend larger than the default of tools/style.py
# On two lines: on one line, the label is wider than the panel and is cut on the right
ax.set_xlabel('time after\nphoto-immobilization (min)', fontsize=C.FONTS['title'])
ax.set_ylabel(f'% of the {n_cells} cells', fontsize=C.FONTS['title'])
ax.legend(frameon=False, loc='lower right', fontsize=C.FONTS['title'] - 2)
ax.grid(axis='y', color='#e5e5e5', linewidth=0.8)
ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
fig.tight_layout()
C.save_figure(fig, C.figure_path(args, __file__, canal), args.dpi)
