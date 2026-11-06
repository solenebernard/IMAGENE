'''
Main figure of the control experiment, assembled from the figures of this folder saved as PNG:
    (no panel a: the letters start at b)
    b) fraction of the cells that stay put over time after UV (high density)            1_sticking_fraction_after_uv.py
    c) trajectories and contours of the cells before and after UV (high density)        2_trajectories_contours_after_uv.py
    d) search of the translation aligning the last frame of the movie on the barcode     3_alignment_search.py
    e) segmentation of the live and fixed cells, and pairs passing the quality control  4_matching_live_fixed.py
    f) confusion matrices of the barcodes, live against fixed cells                     5_confusion_matrices.py
    g) balanced accuracy as a function of the quality control of the matching           6_accuracy_vs_qc_ratio.py
    h) flux diagram of the cells, from the movie to the final dataset (high density)    7_flux_diagram_cells.py
Run the scripts above first. Each panel is drawn by its script with the size of its cells of the grid (PANEL_UNIT of
tools/style.py), so that its text has the same size in the figure (see tools/figures.compose_grid).

Output: control/results/figures/main_figure_control.png and .pdf, and a copy of the figure and of its panels in
control/results/figures/FIGURE 2 CONTROL/

Usage:
    python3 -m 7_control_figures.8_main_figure_control
'''

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))   # code/, for tools/
from tools.figures import compose_grid

import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from tools.read_yaml import CONTROL_PATH
output = Path(CONTROL_PATH) / 'results' / 'figures'
high = output / 'canal2'
# Grid of square cells (tools/style.py, PANEL_UNIT): (letter, images, column, row, width, height)
#   b c c c d d
#   e e e f f g   (g: 1.5 cells high, centred vertically next to f)
#   h h h f f
panels = [
    ('b', [high / '1_sticking_fraction_after_uv.png'], 0, 0, 1, 1),
    ('c', [high / '2_trajectories_contours_after_uv.png'], 1, 0, 3, 1),
    ('d', [high / '3_alignment_search.png'], 4, 0, 2, 1),
    ('e', [high / '4_matching_live_fixed.png'], 0, 1, 3, 1),
    ('f', [output / '5_confusion_matrices.png'], 3, 1, 2, 2),
    ('g', [output / '6_accuracy_vs_qc_ratio.png'], 5, 1.25, 1, 1.5),  # centred next to f
    ('h', [high / '7_flux_diagram_cells.png'], 0, 2, 3, 1),
]
compose_grid(panels, output / 'main_figure_control', panels_dir=output / 'FIGURE 2 CONTROL')
