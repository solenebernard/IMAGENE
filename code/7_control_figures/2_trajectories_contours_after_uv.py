'''
Trajectories and shapes of the matched cells, before and after UV.

Three panels over the same field of view (step3's tile; positions in µm, from
step1's pixel coordinates with 1 px = 0.1625 µm), each a window of 21 frames
(7 min, one frame every 21 s):
  "live cells"                        the 21 frames of step1, before UV
  "right after immobilization"        the first 21 frames of step3
  "~30min after immobilization"       the 21 frames of step3 centred on 30 min
                                      after UV (frames 76-96; --late_min)
For each matched cell: the path of its centroid (black line, dots coloured by
time) and its outline in every frame (coloured by time). The colour is the time
since the start of the window, so the three panels share ONE colour scale
(colorbar on the right).
Moving cells leave long tracks and scattered outlines; attached cells leave a
dot and stacked outlines. No axes: each panel has a 50 µm scale bar, in a
margin under the field of view.

Output: control/results/figures/canal<N>/2_trajectories_contours_after_uv.png

Usage:
    python3 -m 7_control_figures.2_trajectories_contours_after_uv --canal 2
'''

import tifffile
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this folder, for _common.py and the libraries
import _common as C
import _attachment_lib as A

SCALE_BAR_UM = 50

parser = C.base_parser(__doc__)
parser.add_argument('--late_min', type=float, default=30.0,
                    help="Time after UV (min) the third panel's window is centred on (default 30).")
args = parser.parse_args()
C.larger_text()  # larger text in the main figure
canal = C.get_canal(args.root, args.canal)
df1, df3, pairs = A.load_per_frame(canal)

dx, dy = A.step3_offset(canal)      # where step3's tile sits in step1's frame
t1 = sorted(df1['t_step1'].unique())[:A.N_FRAMES]
t3 = sorted(df3['t_step3'].unique())
late_centre = int(round(args.late_min * 60 / C.FRAME_DT_S))    # frame at --late_min
late = [t for t in t3 if late_centre - A.N_FRAMES // 2 <= t <= late_centre + A.N_FRAMES // 2]
tile = tifffile.imread(A.step3_image_path(canal, t3[0])).shape[0]  # step3 tile size (px)
print(f'late window: frames {late[0]}-{late[-1]} ({C.minutes(late[0]):.1f}-{C.minutes(late[-1]):.1f} min after UV)')

panels = [  # (per-frame table, step, frames, mask finder, offset, title)
    (df1, 'step1', t1, lambda t: A.step1_mask_path(canal, t), (0.0, 0.0), 'live cells'),
    (df3, 'step3', t3[:A.N_FRAMES], lambda t: A.step3_mask_path(canal, t), (dx, dy), 'right after immobilization'),
    (df3, 'step3', late, lambda t: A.step3_mask_path(canal, t), (dx, dy), f'~{args.late_min:g}min after immobilization'),
]

# Size of panel c of the main figure: 3 units wide, 1 high (tools/style.py)
fig, axes = plt.subplots(1, len(panels), figsize=(3 * C.PANEL_UNIT, C.PANEL_UNIT), sharex=True, sharey=True, layout='constrained')
for ax, (df, step, frames, mask_path, offset, title) in zip(axes, panels):
    mappable = A.plot_trajectories(ax, df, step, set(pairs[f'unique_id_{step}']), frames, mask_path, offset, title,
                                   relative_time=True, colorbar=False)
    ax.set_title(title)
# one colorbar for the three panels: time since the start of each window
cbar = fig.colorbar(mappable, ax=list(axes), shrink=0.7, label='time within the window (min)')
cbar.formatter = FuncFormatter(lambda frame, _: f'{C.minutes(frame):.0f}')  # frames -> minutes
cbar.update_ticks()

# every panel zoomed on step3's tile, image convention (y pointing down)
um = A.PIXEL_SIZE_UM
axes[0].set_xlim(dx * um, (dx + tile) * um)
# a margin under the field of view (8 % of its height) holds the scale bar
axes[0].set_ylim((dy + 1.08 * tile) * um, dy * um)
x_bar, y_bar = dx * um, (dy + 1.045 * tile) * um  # under the bottom-left corner
for ax in axes:
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.plot([x_bar, x_bar + SCALE_BAR_UM], [y_bar, y_bar], color='black', linewidth=4, solid_capstyle='butt',
            zorder=6, clip_on=False)
    ax.text(x_bar + SCALE_BAR_UM + 0.015 * tile * um, y_bar, f'{SCALE_BAR_UM} µm', ha='left', va='center',
            zorder=6, clip_on=False)
C.save_figure(fig, C.figure_path(args, __file__, canal), args.dpi)
