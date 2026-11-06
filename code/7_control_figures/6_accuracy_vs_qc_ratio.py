'''
Balanced accuracy as a function of the match QC threshold, both canals.

For each ratio_qc from 0.02 to 0.90, the matched live/fixed pairs with a
colour on both sides and passing the QC are kept, and the balanced accuracy
of their barcode classes is computed (solid lines, left axis); the dotted
lines (right axis) are the number of pairs passing the QC. One colour per
canal, named by its cell density: canal2 = high density (2 M cells/mL),
canal05 = low density (0.5 M cells/mL).
Shows the trade-off: a stricter QC is more accurate but keeps fewer cells.

Output: control/results/figures/6_accuracy_vs_qc_ratio.png

Usage:
    python3 -m 7_control_figures.6_accuracy_vs_qc_ratio
'''

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this folder, for _common.py and the libraries
import _common as C
import _illustration_lib as L

CANAL_COLORS = ['#e8743b', '#2a78d6', '#3a9e5c', '#8e5bc4']  # one per canal, in C.all_canals() order
CANAL_NAMES = {'2': 'High density', '05': 'Low density'}       # legend name of each canal

parser = C.base_parser(__doc__, canal=False)
C.add_matched_cells_options(parser)
args = parser.parse_args()
sweep = np.round(np.arange(0.02, 0.901, 0.02), 2)

# Size of panel g of the main figure: 1 unit wide, 1.5 high (tools/style.py). Two plots sharing the QC ratio:
# the balanced accuracy (top), the number of pairs passing QC (bottom)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(C.PANEL_UNIT, 1.5 * C.PANEL_UNIT), sharex=True)
for canal, color in zip(C.all_canals(args.root), CANAL_COLORS):
    L.configure(canal.dir)
    df, excluded = L.prepare(args.label_mode, args.positive_fraction, args.dead_cell_removal, args.iou_split_threshold)
    pairs = L.matched_pairs_table(df, *excluded)
    n_colored = int(pairs['both_colored'].sum())
    results = [L.balanced_accuracy_at_threshold(pairs, qc) for qc in sweep]  # (balanced accuracy, pairs kept)
    accuracy = np.array([r[0] for r in results]) * 100
    n_kept = np.array([r[1] for r in results])

    name = CANAL_NAMES.get(canal.key, f'canal{canal.key}')
    ax.plot(sweep, accuracy, color=color, linewidth=2)
    ax2.plot(sweep, n_kept, color=color, linewidth=2, label=name)  # the number of pairs before QC is the start of the curve
    print(f'canal{canal.key}: ' + ', '.join(f'QC {qc}: {a:.1f}% ({n})' for qc, a, n in zip(sweep, accuracy, n_kept)))

ax.set_title('Balanced accuracy vs QC ratio\n(pairs with a colour on both sides)')
# Labels and legend larger than the default of tools/style.py
ax.set_ylabel('balanced\naccuracy (%)', fontsize=C.FONTS['title'])
ax2.set_ylabel('pairs\npassing QC', fontsize=C.FONTS['title'])
ax2.set_xlabel('QC ratio', fontsize=C.FONTS['title'])
ax2.set_ylim(bottom=0)
for a in (ax, ax2):
    a.grid(alpha=0.3)
ax2.legend(loc='upper right', frameon=False, fontsize=C.FONTS['title'] - 2)
fig.tight_layout()
C.save_figure(fig, C.figure_path(args, __file__), args.dpi)
