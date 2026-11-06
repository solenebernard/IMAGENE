'''
Shared helpers for the figure scripts in this folder: where the data is, the
command-line options every script has, and the tables several figures share.

Folder layout read by the figures: figures_input/ of the control experiment (control.exp_path of config.yaml), the part of the outputs of the analysis of the
control experiment (steps 1 and 3 of the pipeline) needed by the figures:

    figures_input/canal05/, figures_input/canal2/      one folder per density (low, high)
        step0/, step4/, step<N>_afterPFA/              colour barcode steps
            global_masks/*_signal_per_object.csv       per-cell barcode intensities
        step1/, step3/                                 brightfield timelapses
            track/*_ch00_tracks.csv                    cell tracks
            track/*_ch00_track_dynamics*.csv           per-track motion metrics
        match_step_<i>_<j>/*_match.csv                 cell matches between steps

The after-PFA step is step5 in canal2 and step7 in canal05: it is detected,
never hard-coded.

Match QC (ratio_qc, in [0, 1], larger = stricter): a match is trusted if, on
both sides,
    1 - centroid distance / distance to the second-nearest candidate >= ratio_qc
(0.7 = the match is at most 0.3x as far as the next candidate).
'''

import sys
import csv
import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # figures are saved to files, never shown
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent   # code/7_control_figures/
PIPELINE_DIR = SCRIPT_DIR / 'pipeline'         # the modules of the original control analysis used by the figures
sys.path.insert(0, str(PIPELINE_DIR))          # so that they can be imported
sys.path.insert(1, str(SCRIPT_DIR.parent))     # code/, for tools/
from helpers import find_one                   # noqa: E402
from tools.matching import passes_ratio_qc     # noqa: E402,F401 (used by the scripts as C.passes_ratio_qc)

# The folder of the control experiment (control.exp_path of config.yaml): figures_input/ (tables, masks and images
# needed by the figures), alignment/ and BarcodeAfterUV/ (read by 3_alignment_search.py); the figures are saved in
# results/figures/ of it
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from tools.read_yaml import CONTROL_PATH, CONTROL_RATIO_QC, PIXEL_SIZE
CONTROL_DIR = Path(CONTROL_PATH)
DEFAULT_ROOT = CONTROL_DIR / 'figures_input'
OUTPUT_DIR = CONTROL_DIR / 'results' / 'figures'

CHANNELS = ['ch01', 'ch02']             # the two barcode dyes
RATIO_QC = CONTROL_RATIO_QC  # control.ratio_qc of config.yaml
# Same font as the figures of the main experiment (tools/style.py), set for every figure of this folder
from tools.style import FONTS, PANEL_UNIT, set_style
set_style()


def larger_text():
    '''Text larger than the default of tools/style.py, for the panels with small text in the main figure: labels at
    the size of the titles, the rest of the text 2 pt smaller'''
    plt.rcParams.update({'font.size': FONTS['title'] - 2, 'axes.labelsize': FONTS['title'],
                         'figure.labelsize': FONTS['title'], 'xtick.labelsize': FONTS['title'] - 2,
                         'ytick.labelsize': FONTS['title'] - 2, 'legend.fontsize': FONTS['title'] - 2})


IOU_STABLE = 0.6                        # iou_all_frames above which a track's footprint is "stable"
PIXEL_SIZE_UM = PIXEL_SIZE              # µm per pixel of the live images (step1 / step3), constants.pixel_size_um
FRAME_DT_S = 21                         # seconds between two frames of a timelapse (step1 and step3)
UM_PER_MIN = PIXEL_SIZE_UM / (FRAME_DT_S / 60)  # 1 px per frame, in µm per minute


def minutes(frames):
    '''Duration of `frames` frame intervals (or time of frame number `frames`), in minutes.'''
    return frames * FRAME_DT_S / 60


##### CANALS, COMMAND LINE, SAVING

@dataclass
class Canal:
    key: str            # '2', '05'
    dir: Path           # .../canal2
    final_folder: str   # 'step5_afterPFA'
    final: str          # 'step5' - used as column suffix
    final_number: str   # '5'


def get_canal(root, key):
    canal_dir = Path(root) / f'canal{key}'
    folders = [d.name for d in sorted(canal_dir.glob('step*_afterPFA')) if d.is_dir()]
    if len(folders) != 1:
        raise FileNotFoundError(f'Expected exactly one step<N>_afterPFA folder in {canal_dir}, found {folders}')
    number = folders[0].removeprefix('step').split('_')[0]
    return Canal(key, canal_dir, folders[0], f'step{number}', number)


def all_canals(root):
    return [get_canal(root, d.name.removeprefix('canal')) for d in sorted(Path(root).glob('canal*')) if d.is_dir()]


def base_parser(description, canal=True):
    '''Options common to every figure script.'''
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT,
                        help='Folder holding canal05/, canal2/ (default: figures_input/ of the control experiment).')
    if canal:
        parser.add_argument('--canal', default='2', help="Channel to analyse: '2' or '05' (default 2).")
    parser.add_argument('--output_dir', type=Path, default=OUTPUT_DIR,
                        help='Where figures are saved (default: results/figures/ of the control experiment).')
    parser.add_argument('--dpi', type=int, default=200)
    return parser


def add_matched_cells_options(parser, dead_cell_option=True):
    '''Options of the matched-cells figures (00-06). Defaults: quantile labels
    with the top 50% positive, ratio QC 0.7, and dead cells (step1
    iou_all_frames >= 0.6) removed.'''
    parser.add_argument('--ratio_qc', type=float, default=RATIO_QC,
                        help='Match QC in [0, 1], larger = stricter (default: control.ratio_qc of config.yaml): keep a step1 <-> after-PFA '
                             'match if 1 - distance / second-nearest distance >= this on both sides.')
    parser.add_argument('--label_mode', choices=['quantile', 'fixed'], default='quantile',
                        help="Positive/negative barcode call: 'quantile' = top --positive_fraction of each "
                             "channel and step; 'fixed' = the FIXED_THRESHOLDS in _illustration_lib.py.")
    parser.add_argument('--positive_fraction', type=float, default=0.5,
                        help='--label_mode quantile: fraction of cells (highest q90) called positive (default 0.5).')
    parser.add_argument('--iou_split_threshold', type=float, default=IOU_STABLE,
                        help="step1 iou_all_frames at or above which a track's footprint is 'stable' (default 0.6).")
    if dead_cell_option:
        parser.add_argument('--dead_cell_removal', action=argparse.BooleanOptionalAction, default=True,
                            help='Drop step1 cells with iou_all_frames >= --iou_split_threshold (assumed dead). '
                                 'On by default; --no-dead_cell_removal keeps them.')


def figure_path(args, script_file, canal=None, suffix=''):
    '''<output_dir>/[canal<key>/]<script name><suffix>.png'''
    folder = args.output_dir / (canal.dir.name if canal else '')
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f'{Path(script_file).stem}{suffix}.png'


def save_figure(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {path}')


##### TABLES: BARCODE INTENSITIES, MATCHES, THE STEP0 -> STEP1 -> AFTER-PFA CHAIN

def load_colors(canal, step_folder, step):
    '''One step's per-cell barcode table, every column suffixed _<step>. Tile
    objects without a global_id (dropped when tiles were stitched) are removed.'''
    df = pd.read_csv(find_one(canal.dir / step_folder / 'global_masks', '*_signal_per_object.csv'))
    df = df.dropna(subset=['global_id']).astype({'global_id': int})
    return df.rename(columns={c: f'{c}_{step}' for c in df.columns})


def load_match(canal, match_dir, id_1, id_2, hop):
    '''Accepted matches of one match_step_<i>_<j> folder (None if the folder
    doesn't exist), ids renamed to id_1 / id_2 and the other columns suffixed
    with hop (e.g. '1_5').'''
    path = find_one(canal.dir / match_dir, '*_match.csv', required=False)
    if path is None:
        return None
    df = pd.read_csv(path)
    df = df[df['match_type'].isin(['overlap', 'distance'])]  # unmatched rows have no partner
    df = df.rename(columns={'global_id_1': id_1, 'global_id_2': id_2, 'iou': f'iou_step_{hop}',
                            'centroid_distance_px': f'centroid_distance_px_{hop}',
                            'second_nn_distance_1': f'second_nn_distance_1_{hop}',
                            'second_nn_distance_2': f'second_nn_distance_2_{hop}',
                            'match_type': f'match_type_{hop}'})
    return df.astype({id_1: int, id_2: int})


def load_excluded_ids(canal, step_folder, channel):
    '''Cells excluded by hand, in <prefix>_excluded_objects_<channel>.csv (empty set if none).'''
    path = find_one(canal.dir / step_folder / 'global_masks', f'*_excluded_objects_{channel}.csv', required=False)
    if path is None:
        return set()
    with open(path, newline='') as f:
        return {int(row['global_id']) for row in csv.DictReader(f)}

