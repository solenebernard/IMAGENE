'''
Confusion matrices of the combinatorial stain, live vs fixed cells,
at the two cell densities and three levels of match QC.

Rows of panels: low density (canal05) on top, high density (canal2) below.
Columns of panels: no QC, then only the pairs whose match passes QC ratio
>= 0.3, then >= 0.6 (--qc_values).
In each matrix, rows = the stain class assigned to the live cell (measured at
step0, carried through the step1 tracking), columns = the class of the fixed
cell it is matched to after PFA; only matched pairs with a class on both sides
are counted. The classes are in the fixed order (ch01-, ch02-), (ch02 only),
(ch01 only), (ch01+, ch02+) on both axes (no tick labels). Each tile shows the
row-normalised percentage and the number of pairs. Each panel's title gives
the QC ratio, the balanced accuracy (mean of the per-class recalls), the
agreement expected by chance (fixed-cell classes shuffled 1000 times) and the
number of pairs.

Output: control/results/figures/5_confusion_matrices.png

Usage:
    python3 -m 7_control_figures.5_confusion_matrices
    python3 -m 7_control_figures.5_confusion_matrices --qc_values 0.3 0.6
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this folder, for _common.py and the libraries
import _common as C
import _illustration_lib as L

DENSITY = {'05': 'Low density', '2': 'High density'}  # canal key -> title of the row, drawn top to bottom in this order

parser = C.base_parser(__doc__, canal=False)
C.add_matched_cells_options(parser)
parser.add_argument('--qc_values', type=float, nargs='+', default=[0.3, 0.6],
                    help='QC ratio of the panels after the first (no QC) one (default 0.3 0.6).')
args = parser.parse_args()
C.larger_text()  # larger text in the main figure
canals = {c.key: c for c in C.all_canals(args.root)}
class_order = list(L.CLASS_COLORS.values())


def draw_matrix(ax, live, fixed, qc_label):
    '''Row-normalised confusion matrix of one set of pairs, no ticks, four-line title.'''
    counts = (pd.crosstab(pd.Series(live, name='live'), pd.Series(fixed, name='fixed'))
              .reindex(index=class_order, columns=class_order, fill_value=0).to_numpy())
    n = int(counts.sum())
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums != 0)
    has_row = row_sums.ravel() > 0
    balanced = float(np.diag(normalized)[has_row].mean()) if has_row.any() else float('nan')
    baseline = L.random_label_baseline(list(live), list(fixed)) if n else float('nan')

    ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(class_order)):
        for j in range(len(class_order)):
            ax.text(j, i, f'{normalized[i, j] * 100:.0f}%\n(n={counts[i, j]})', ha='center', va='center',
                    fontsize=C.FONTS['title'] - 2, color='white' if normalized[i, j] > 0.5 else 'black')
    ax.set_xticks([])
    ax.set_yticks([])
    # Title: the quality control in bold, then the scores, smaller (with the size of the titles, the lines of two
    # neighbouring matrices overlap)
    ax.set_title(f'balanced accuracy = {balanced * 100:.1f}%\nrandom baseline = {baseline * 100:.1f}%\nn = {n} pairs',
                 fontsize=C.FONTS['title'] - 3, pad=4)
    ax.annotate(qc_label, xy=(0.5, 1), xycoords=ax.title, xytext=(0, 2), textcoords='offset points',
                ha='center', va='bottom', fontsize=C.FONTS['title'], fontweight='bold')
    print(f'  {qc_label}: balanced accuracy {balanced * 100:.1f}%, random baseline {baseline * 100:.1f}%, n = {n}')


qc_panels = [('no QC', None)] + [(f'QC ratio = {qc:g}', qc) for qc in args.qc_values]
rows = [key for key in DENSITY if key in canals]
# Size of panel f of the main figure: 2 x 2 units (tools/style.py). One row of matrices per density, with its title
fig = plt.figure(figsize=(2 * C.PANEL_UNIT, 2 * C.PANEL_UNIT))
row_figures = np.atleast_1d(fig.subfigures(len(rows), 1))
for r, key in enumerate(rows):
    # Placed by hand: the top of each row holds its title, then the four-line titles of the matrices
    top, bottom = 0.70, 0.03  # of the matrices, in the row
    row_figures[r].subplots_adjust(left=0.08, right=0.99, top=top, bottom=bottom, wspace=0.06)
    row_figures[r].suptitle(DENSITY[key], y=0.99, fontsize=C.FONTS['title'], fontweight='bold')
    # Centred on the matrices, not on the row (which also holds the titles)
    row_figures[r].supylabel('live cell combinatorial\nstain assignment', x=0.01, y=(top + bottom) / 2,
                             multialignment='center')
    row_axes = np.atleast_1d(row_figures[r].subplots(1, len(qc_panels)))
    print(f'{DENSITY[key]} (canal{key})')
    L.configure(canals[key].dir)
    df, excluded = L.prepare(args.label_mode, args.positive_fraction, args.dead_cell_removal, args.iou_split_threshold)
    pairs = L.matched_pairs_table(df, *excluded)
    colored = pairs['both_colored'].to_numpy()
    live, fixed = pairs['color1'].to_numpy(), pairs['color7'].to_numpy()
    for c, (qc_label, qc) in enumerate(qc_panels):
        keep = colored if qc is None else colored & L.qc_pass_mask(pairs, qc)
        draw_matrix(row_axes[c], live[keep], fixed[keep], qc_label)
fig.supxlabel('fixed cell combinatorial stain assignment', y=0.0)
C.save_figure(fig, C.figure_path(args, __file__), args.dpi)
