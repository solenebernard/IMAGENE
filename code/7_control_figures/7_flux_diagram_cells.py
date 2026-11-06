'''
Flux diagram: how many tracked cells end up in the final dataset.

Starting from the live cells tracked through the whole step1 timelapse, each
column splits the cells kept by the previous one (heights = cell counts):

  Fully tracked in step1      present in all 21 frames
  Alive / Assumed dead        dead = footprint barely changes over the movie
                              (iou_all_frames >= --iou_split_threshold)
  Barcoded / Not barcoded     matched to a step0 cell with a colour barcode
  Paired / Unpaired           matched to a fixed cell (mutually closest
                              neighbour) with a colour, after PFA
  Final dataset / Filtered    the match with the fixed cell passes the quality
                              threshold (--ratio_qc, 0.6 by default)
  Accurate / Inaccurate       same / different barcode class (ch01, ch02
                              positivity) at step0 and after fixation

The counts of every stage are also printed. --end step4 compares with the
second colour scan (step4) instead of the scan after fixation.

Output: control/results/figures/canal<N>/7_flux_diagram_cells.png

Usage:
    python3 -m 7_control_figures.7_flux_diagram_cells --canal 2
    python3 -m 7_control_figures.7_flux_diagram_cells --canal 2 --no-dead_cell_removal --label_mode threshold
'''

import argparse

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this folder, for _common.py and the libraries
import _common as C
import tracking_accuracy_step0_step4 as T  # the pipeline script that defines the chain and the diagram

parser = C.base_parser(__doc__)
parser.add_argument('--end', choices=['after_pfa', 'step4'], default='after_pfa',
                    help='Step compared with step0: the scan after fixation (default) or step4.')
parser.add_argument('--ratio_qc', type=float, default=C.RATIO_QC,
                    help='Match QC in [0, 1], larger = stricter (default: control.ratio_qc of config.yaml), applied to the match '
                         'with the end step (as in 5_confusion_matrices.py and 6_accuracy_vs_qc_ratio.py).')
parser.add_argument('--label_mode', choices=['quantile', 'threshold'], default='quantile',
                    help="Positive/negative barcode call: 'quantile' = top --positive_fraction of each channel "
                         "and step; 'threshold' = the thresholds saved in *_signal_per_object.csv.")
parser.add_argument('--positive_fraction', type=float, default=0.5,
                    help='--label_mode quantile: fraction of cells (highest q90) called positive (default 0.5).')
parser.add_argument('--iou_split_threshold', type=float, default=C.IOU_STABLE,
                    help='step1 iou_all_frames at or above which a cell is assumed dead (default 0.6).')
parser.add_argument('--dead_cell_removal', action=argparse.BooleanOptionalAction, default=True,
                    help='Split off the assumed-dead cells (on by default; --no-dead_cell_removal skips that column).')
args = parser.parse_args()
canal = C.get_canal(args.root, args.canal)

# Tell the pipeline module which step is compared with step0
T.END_STEP, T.END_FOLDER, T.END_MATCH_DIR = T.resolve_end_step(canal.dir, args.end)

# Cells tracked through the whole step1 timelapse (tracks have no gaps)
tracks = pd.read_csv(C.find_one(canal.dir / 'step1' / 'track', '*_ch00_tracks.csv'))
n_frames = tracks['t'].nunique()
frames_per_track = tracks.groupby('unique_id')['t'].nunique()
full_tracks = set(frames_per_track.index[frames_per_track == n_frames])

# Final dataset: cells followed step0 -> step1 -> end step whose match with the end step passes the QC
chain = T.add_labels(T.build_chain(canal.dir, args.ratio_qc), args.label_mode, args.positive_fraction, canal.dir)

# Assumed-dead cells are set aside before anything else
dead = T.load_dead_tracks(canal.dir, tracks, full_tracks, args.iou_split_threshold) if args.dead_cell_removal else set()
chain = chain[~chain['unique_id'].isin(dead)]

# Intermediate stages, counted before the QC (the QC only removes matches)
alive = full_tracks - dead
barcoded = T.barcoded_tracks(canal.dir) & alive                 # matched to step0
paired = set(T.build_chain(canal.dir)['unique_id']) - dead      # also matched to the end step

subtitle = (f'labels: top {args.positive_fraction:.0%} q90 per step' if args.label_mode == 'quantile'
            else "labels: each step's saved thresholds")
subtitle += f'  |  ratio QC {args.ratio_qc}'
if args.dead_cell_removal:
    subtitle += f'  |  dead cells removed (iou_all_frames >= {args.iou_split_threshold})'

path = C.figure_path(args, __file__, canal)
counts = T.plot_tracking_flux(len(full_tracks), n_frames, len(barcoded), len(paired), chain, path, subtitle,
                              n_dead=len(dead) if args.dead_cell_removal else None,
                              iou_threshold=args.iou_split_threshold, ratio_qc=args.ratio_qc)
print(f'Saved -> {path}')
print('  ' + '  |  '.join(f'{name}: {n}' for name, n in counts.items()))
