'''
Accuracy of step1's tracking, measured through the colors: each cell is
followed from step0 (colors) to an END step (colors) THROUGH the step1 movie,

    step0 --match_step_0_1--> step1 t_first --track (unique_id)--> step1 t_last --match_step_1_<N>--> END

and its (ch01, ch02) positivity class at step0 is compared with the class of
the END object it ends up matched to. --end picks the END step:
  step4     : step4/, matched via match_step_1_4/
  after_pfa : the step<N>_afterPFA/ folder (step5 in canal2, step7 in canal05,
              detected), matched via match_step_1_<N>/
"step4" below means whichever END step was picked.

If every hop (both matches AND the
step1 tracking in between) were correct, a cell would keep its class --
up to the color measurement's own noise - so the confusion matrix and its
accuracy are an end-to-end check of the chain, the step1 tracking being the
main source of possible identity swaps (21 frames of moving cells vs. two
single-frame matches).

Labels (--label_mode):
  quantile  : per step and channel, the top --positive_fraction of objects
              (by ch0X_q90) are positive. The cut-off is taken over a fixed
              population (step0: the barcoded, fully
              tracked cells; END: every segmented cell of that step), NOT over
              the cells that survive the filters, so it doesn't move when
              --ratio_qc or --dead_cell_removal change which cells are kept.
  threshold : each step's own saved threshold (ch0X_positive column of its
              *_signal_per_object.csv).

The accuracy is compared with a random-label baseline (step4 classes
shuffled among the same cells, averaged over N_RANDOM_TRIALS), i.e. the
agreement expected from the class frequencies alone with no real tracking.

Outputs (in ./<canal_dir's own folder name>/):
(file names use the END step's name, e.g. step0_step4 or step0_step7)
    tracking_accuracy_step0_step4_pairs.csv     one row per chained cell
    tracking_accuracy_step0_step4_confusion.png row-normalized confusion matrix
    tracking_accuracy_step0_step4_flux.png      flux diagram: step1 cells tracked
        through the WHOLE movie (present in every frame) ->
        [with --dead_cell_removal: alive vs assumed dead, then only the alive ones ->]
        barcoded (matched to step0, with a color) or not ->
        paired with END (mutually closest neighbour, with a color) or unpaired ->
        [with --ratio_qc: final dataset (the step1 -> END match passes) vs filtered out ->]
        accurate (barcode match) vs inaccurate (barcode mismatch)
        matched to BOTH step0 and step4 (with a color at both) or not ->
        among the matched ones, same vs different color class
    tracking_accuracy_step0_step4_movie.gif     the step1 movie (stitched
        step1/global_masks/*_t<NN>_ch00_stitched.tif), every chained cell's contour in
        GREEN if its step0 and step4 classes match, RED if not, and every
        other step1 cell (not matched to a color at step0 and/or step4) in
        light GRAY. Downsampled by
        --gif_downsample; skip with --no_gif.

The END step's match_step_1_<N>/ folder must exist.

--dead_cell_removal: a fully tracked step1 cell whose footprint barely changes
over the whole movie (iou_all_frames >= --iou_split_threshold, from
track_dynamics.py's step1 *_ch00_track_dynamics.csv) is considered dead and excluded from the matching --
it's not in the pairs CSV / confusion matrix, gets its own node in the flux
diagram and a dark contour in the GIF.

Usage:
    python tracking_accuracy_step0_step4.py --canal_dir path/to/control/figures_input/canal2
    python tracking_accuracy_step0_step4.py --canal_dir path/to/control/figures_input/canal2 \\
        --label_mode threshold
    python tracking_accuracy_step0_step4.py --canal_dir path/to/control/figures_input/canal05 \\
        --end after_pfa --label_mode threshold --ratio_qc 0.7
'''

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from helpers import find_one  # pipeline/helpers.py (this folder)

Image.MAX_IMAGE_PIXELS = None  # stitched step1 frames/masks are ~57M px

CHANNELS = ('ch01', 'ch02')
# (ch01, ch02) positivity -> class name, fixed order for the matrix
CLASSES = [(False, False), (False, True), (True, False), (True, True)]
CLASS_NAMES = [f'({int(a)},{int(b)})' for a, b in CLASSES]

N_RANDOM_TRIALS = 1000
RANDOM_SEED = 0

MATCH_RGB, MISMATCH_RGB = (40, 200, 70), (230, 40, 40)  # contour colors in the GIF
UNMATCHED_RGB = (215, 215, 215)  # cells with no step0 and/or END color (light, to stand out on the BF)
DEAD_RGB = (25, 25, 25)  # --dead_cell_removal: cells excluded as dead

# The END step (set in main() from --end): its display name (e.g. "step4",
# "step7"), its folder and its match_step_1_<N> folder. Internally its
# columns use the suffix "_end"; the saved CSV renames them to END_STEP.
END_STEP, END_FOLDER, END_MATCH_DIR = 'step4', 'step4', 'match_step_1_4'


def resolve_end_step(canal_dir, end):
    '''(display name, folder, match folder) for --end 'step4' or 'after_pfa'.'''
    if end == 'step4':
        name, folder = 'step4', 'step4'
    else:
        folders = [d.name for d in sorted(canal_dir.glob('step*_afterPFA')) if d.is_dir()]
        if len(folders) != 1:
            raise FileNotFoundError(f'Expected exactly one step<N>_afterPFA folder in {canal_dir}, found {folders}')
        folder = folders[0]
        name = 'step' + folder.removeprefix('step').split('_')[0]
    match_dir = f"match_step_1_{name.removeprefix('step')}"
    if not (canal_dir / match_dir).is_dir():
        if end == 'step4':
            raise FileNotFoundError(f'No {match_dir}/ in {canal_dir} -- no script of this repository prepares it: '
                                    f'use --end after_pfa.')
        raise FileNotFoundError(f'No {match_dir}/ in {canal_dir} -- run step 3.1 (3_match_bf_fish/1_live_fish_matching.py) '
                                f'for step1 -> {folder}, then 7_control_figures/analysis/2_prepare_figure_inputs.py (see run_control.sh).')
    return name, folder, match_dir


GIF_FRAME_MS = 300


##### LOADING + CHAINING

def load_match(path, id_1, id_2, hop, ratio_qc=None):
    '''Accepted matches only (unmatched rows have NaN ids that would join with
    each other). With ratio_qc (in [0, 1], larger = stricter), also only
    matches with 1 - centroid distance / second-nearest-neighbour distance
    >= ratio_qc on BOTH sides, e.g. 0.7 -> the match at most 0.3x as far as
    the second candidate (mutual-nearest
    match CSVs).'''
    df = pd.read_csv(path)
    df = df[df['match_type'].isin(['overlap', 'distance'])]
    if ratio_qc is not None:
        if not {'second_nn_distance_1', 'second_nn_distance_2'}.issubset(df.columns):
            raise ValueError(f'{path.name} has no second_nn_distance columns -- re-run 7_control_figures/analysis/2_prepare_figure_inputs.py '
                             f'(it writes them from the pairs of steps 1.3 and 3.1) to use --ratio_qc')
        n_before = len(df)
        df = df[(1 - df['centroid_distance_px'] / df['second_nn_distance_1'] >= ratio_qc) &
                (1 - df['centroid_distance_px'] / df['second_nn_distance_2'] >= ratio_qc)]
        print(f'  {path.parent.name}: {len(df)}/{n_before} match(es) pass ratio QC {ratio_qc}')
    df = df.rename(columns={'global_id_1': id_1, 'global_id_2': id_2, 'iou': f'iou_{hop}',
                            'centroid_distance_px': f'centroid_distance_px_{hop}', 'match_type': f'match_type_{hop}'})
    return df[[id_1, id_2, f'iou_{hop}', f'centroid_distance_px_{hop}', f'match_type_{hop}']].astype({id_1: int, id_2: int})


def load_colors(canal_dir, folder, step):
    '''<folder>/global_masks/*_signal_per_object.csv, objects without a global_id
    (dropped when stitching) removed, columns suffixed _<step>.'''
    df = pd.read_csv(find_one(canal_dir / folder / 'global_masks', '*_signal_per_object.csv'))
    df = df.dropna(subset=['global_id']).astype({'global_id': int})
    cols = ['global_id'] + [f'{ch}_{x}' for ch in CHANNELS for x in ('q90', 'positive')]
    return df[cols].rename(columns={c: f'{c}_{step}' for c in cols})


def build_chain(canal_dir, ratio_qc=None):
    '''One row per cell followed step0 -> step1 (tracked) -> END, with both steps' colors.

    ratio_qc filters the step1 -> END match only:
    the step0 -> step1 match is always kept.'''
    m01 = load_match(find_one(canal_dir / 'match_step_0_1', '*_match.csv'), 'global_id_step0', 'global_id_step1_t0', '0_1')
    m14 = load_match(find_one(canal_dir / END_MATCH_DIR, '*_match.csv'), 'global_id_step1_tlast', 'global_id_end', '1_end', ratio_qc)

    tracks = pd.read_csv(find_one(canal_dir / 'step1' / 'track', '*_ch00_tracks.csv'))
    t_first, t_last = tracks['t'].min(), tracks['t'].max()
    track = (tracks[tracks['t'] == t_first].rename(columns={'global_id': 'global_id_step1_t0'})[['unique_id', 'global_id_step1_t0']]
             .merge(tracks[tracks['t'] == t_last].rename(columns={'global_id': 'global_id_step1_tlast'})[['unique_id', 'global_id_step1_tlast']],
                    on='unique_id'))
    print(f'step1 tracks spanning t{t_first} -> t{t_last}: {len(track)}')

    chain = (m01.merge(track, on='global_id_step1_t0')
                .merge(m14, on='global_id_step1_tlast')
                .merge(load_colors(canal_dir, 'step0', 'step0'), on='global_id_step0')
                .merge(load_colors(canal_dir, END_FOLDER, 'end'), on='global_id_end'))
    print(f'matches: step0->step1 {len(m01)}, step1->{END_STEP} {len(m14)}  |  chained step0 -> {END_STEP}: {len(chain)}')
    return chain


def barcoded_tracks(canal_dir):
    '''Step1 tracks spanning the whole movie whose first-frame cell is matched
    to a step0 cell with a color (no ratio QC) - the "barcoded" cells.'''
    m01 = load_match(find_one(canal_dir / 'match_step_0_1', '*_match.csv'), 'global_id_step0', 'global_id_step1_t0', '0_1')
    m01 = m01.merge(load_colors(canal_dir, 'step0', 'step0')[['global_id_step0']], on='global_id_step0')
    tracks = pd.read_csv(find_one(canal_dir / 'step1' / 'track', '*_ch00_tracks.csv'))
    first = tracks[tracks['t'] == tracks['t'].min()].rename(columns={'global_id': 'global_id_step1_t0'})
    return set(first.merge(m01, on='global_id_step1_t0')['unique_id'])


def load_dead_tracks(canal_dir, tracks, full_tracks, iou_threshold):
    '''Fully tracked step1 tracks with iou_all_frames >= iou_threshold (track_dynamics.py).
    Refuses a dynamics CSV computed from other tracks than tracks.csv (e.g. re-tracked since).'''
    dynamics = pd.read_csv(find_one(canal_dir / 'step1' / 'track', '*_ch00_track_dynamics.csv'))
    frames_per_track = tracks.groupby('unique_id').size()
    n_frames_dyn = dynamics.set_index('unique_id')['n_frames']
    if set(n_frames_dyn.index) != set(frames_per_track.index) or not (n_frames_dyn.reindex(frames_per_track.index) == frames_per_track).all():
        raise ValueError(f"step1's *_ch00_track_dynamics.csv doesn't match its *_ch00_tracks.csv (re-tracked since?) -- "
                         f"re-run track_dynamics.py --input_dir {canal_dir / 'step1'} before using --dead_cell_removal.")
    iou = dynamics.set_index('unique_id')['iou_all_frames']
    return {uid for uid in full_tracks if iou.get(uid, np.nan) >= iou_threshold}


##### LABELS + ACCURACY

def quantile_cutoffs(canal_dir, positive_fraction):
    '''{(channel, 'step0' | 'end'): q90 cut-off} above which a cell is positive,
    from fixed reference populations:
      step0 - the step0 cells of the barcoded step1 tracks spanning the whole
               movie (matched to step0, no ratio QC)
      end   - every segmented cell of the END step
    Independent of --ratio_qc / --dead_cell_removal, so the same cell gets the
    same label whichever filters are on.'''
    m01 = load_match(find_one(canal_dir / 'match_step_0_1', '*_match.csv'), 'global_id_step0', 'global_id_step1_t0', '0_1')
    tracks = pd.read_csv(find_one(canal_dir / 'step1' / 'track', '*_ch00_tracks.csv'))
    first = tracks[tracks['t'] == tracks['t'].min()].rename(columns={'global_id': 'global_id_step1_t0'})
    spanning = set(tracks.loc[tracks['t'] == tracks['t'].max(), 'unique_id'])
    step0 = (first[first['unique_id'].isin(spanning)].merge(m01, on='global_id_step1_t0')
             .merge(load_colors(canal_dir, 'step0', 'step0'), on='global_id_step0'))
    end = load_colors(canal_dir, END_FOLDER, 'end')
    return {(ch, step): float(table[f'{ch}_q90_{step}'].quantile(1 - positive_fraction))
            for ch in CHANNELS for step, table in (('step0', step0), ('end', end))}


def add_labels(chain, label_mode, positive_fraction, canal_dir):
    '''label_<ch>_<step> booleans, then class_<step> as an index into CLASSES.

    'quantile': positive = q90 above quantile_cutoffs()'s cut-off for that
    channel and step. 'threshold': the step's saved ch0X_positive column.'''
    chain = chain.copy()
    cutoffs = quantile_cutoffs(canal_dir, positive_fraction) if label_mode == 'quantile' else {}
    for step in ('step0', 'end'):
        for ch in CHANNELS:
            if label_mode == 'quantile':
                chain[f'label_{ch}_{step}'] = chain[f'{ch}_q90_{step}'] > cutoffs[(ch, step)]
            else:
                chain[f'label_{ch}_{step}'] = chain[f'{ch}_positive_{step}'].astype(bool)
        chain[f'class_{step}'] = [CLASSES.index((bool(a), bool(b)))
                                  for a, b in zip(chain[f'label_ch01_{step}'], chain[f'label_ch02_{step}'])]
    return chain


def random_baseline(true, pred):
    '''Mean accuracy after shuffling pred among the cells (permutation null).'''
    rng = np.random.default_rng(RANDOM_SEED)
    return float(np.mean([np.mean(true == rng.permutation(pred)) for _ in range(N_RANDOM_TRIALS)]))


def plot_confusion(chain, output_path, subtitle):
    true, pred = chain['class_step0'].to_numpy(), chain['class_end'].to_numpy()
    counts = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(counts, (true, pred), 1)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums != 0)

    has_row = row_sums.ravel() > 0
    stats = {
        'n': len(chain),
        'accuracy': float(np.mean(true == pred)),
        'balanced_accuracy': float(np.diag(normalized)[has_row].mean()),
        'random_baseline': random_baseline(true, pred),
        **{f'{ch}_agreement': float(np.mean(chain[f'label_{ch}_step0'] == chain[f'label_{ch}_end'])) for ch in CHANNELS},
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASSES)), CLASS_NAMES)
    ax.set_xlabel(f'{END_STEP} (ch01,ch02)')
    ax.set_ylabel('step0 (ch01,ch02)')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f'{normalized[i, j] * 100:.1f}%\n(n={counts[i, j]})', ha='center', va='center',
                    color='white' if normalized[i, j] > 0.5 else 'black', fontsize=9)
    ax.set_title(f'step0 -> step1 tracking -> {END_STEP}: color confusion (row-normalized)\n'
                 f"accuracy={stats['accuracy'] * 100:.1f}%  balanced={stats['balanced_accuracy'] * 100:.1f}%  "
                 f"(random-label baseline {stats['random_baseline'] * 100:.1f}%)\n"
                 f"n={stats['n']} cell(s)  |  {subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return stats


##### FLUX DIAGRAM: STEP1 TRACKS -> MATCHED TO BOTH STEPS -> SAME / DIFFERENT COLOR

def _hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def plot_tracking_flux(n_full, n_frames, n_barcoded, n_paired, chain,
                       output_path, subtitle, n_dead=None,
                       iou_threshold=None, ratio_qc=None):
    '''Flux (alluvial) diagram of the tracking's efficiency, node heights
    proportional to cell counts, each column splitting the first node of the
    previous one:
        fully tracked step1 cells (present in all n_frames frames)
        [-> alive / assumed dead (iou_all_frames >= iou_threshold), if n_dead is given]
        -> barcoded (matched to step0, with a color) / not barcoded
        -> paired with END (mutually closest neighbour, with a color) / unpaired
        [-> final dataset (the step1 -> END match passes the ratio QC) / filtered out, if ratio_qc is given]
        -> accurate (barcode match: same step0/END class) / inaccurate (mismatch).
    Same green/red/light-gray (and dark for dead) as the movie GIF.'''
    n_final = len(chain)
    n_same = int((chain['class_step0'] == chain['class_end']).sum())

    def pct(n, of):
        return f'{n / of:.0%}' if of else '-'

    # name: (column, count, color, label); children: name -> [child names], stacked in that order
    nodes = {'tracked': (0, n_full, '#5b5b5b',
                         f'Live cells tracked\nacross all {n_frames} frames\n{n_full} cells (100%)')}
    children = {}
    col, parent, pool = 0, 'tracked', n_full

    def split(kept, dropped):
        '''Add a column: `kept` (continues to the next column) on top, `dropped` below.'''
        nonlocal col, parent, pool
        col += 1
        for name, n, color, label in (kept, dropped):
            nodes[name] = (col, n, color, f'{label}\n{n} ({pct(n, pool)})')
        children[parent] = [kept[0], dropped[0]]
        parent, pool = kept[0], kept[1]

    if n_dead is not None:
        split(('alive', n_full - n_dead, '#5b8fd6', 'Live cells'),
              ('dead', n_dead, _hex(DEAD_RGB), 'Cells assumed\ndead based\non IoU'))
    split(('barcoded', n_barcoded, '#3d84d9', 'Barcoded'),
          ('not_barcoded', pool - n_barcoded, _hex(UNMATCHED_RGB), 'Not barcoded'))
    paired_with = 'cell after fixation' if END_FOLDER.endswith('_afterPFA') else END_STEP
    split(('paired', n_paired, '#2a6fc4', f'Paired with\n{paired_with}\n(mutually closest\nneighbour)'),
          ('unpaired', pool - n_paired, _hex(UNMATCHED_RGB), 'Unpaired'))
    if ratio_qc is not None:
        split(('final', n_final, '#1c5cab', f'Final dataset\n(QC ratio = {ratio_qc:g})'),
              ('filtered', pool - n_final, '#a3a3a3', f'Excluded\n(QC ratio = {ratio_qc:g})'))
    split(('accurate', n_same, _hex(MATCH_RGB), 'Fixed and live cell\nbarcode match'),
          ('inaccurate', n_final - n_same, _hex(MISMATCH_RGB), "Barcodes\ndon't match"))
    n_cols = col + 1

    # Layout: each column stacked top-down in insertion order, 1 cell = 1 y unit
    gap = 0.04 * n_full
    node_y, column_y = {}, {}
    for name, (x, n, _, _) in nodes.items():
        y = column_y.get(x, 0.0)
        node_y[name] = (x, y, y + n)
        column_y[x] = y + n + gap

    node_w = 0.06

    def ribbon(src, dst, src_offset, n, color):
        x0, y0, _ = node_y[src]
        x1, y1, _ = node_y[dst]
        xs = np.linspace(x0 + node_w / 2, x1 - node_w / 2, 60)
        u = (xs - xs[0]) / (xs[-1] - xs[0])
        u = u * u * (3 - 2 * u)  # smoothstep
        top = (y0 + src_offset) + (y1 - (y0 + src_offset)) * u
        ax.fill_between(xs, top, top + n, color=color, alpha=0.35, linewidth=0)

    # same font as all the figures (tools/style.py); the figure is sized so that the text stays large relative to
    # the diagram
    # Size of panel h of the main figure of the control experiment: 3 units wide, 1 high
    from tools.style import FONTS, PANEL_UNIT
    # Text as large as in the other panels of the main figure (larger_text of _common.py: the titles, 2 pt smaller)
    font_size, fig_h = FONTS['title'] - 2, PANEL_UNIT
    fig, ax = plt.subplots(figsize=(3 * PANEL_UNIT, fig_h))
    for src, dsts in children.items():
        offset = 0
        for dst in dsts:
            n = nodes[dst][1]
            if n > 0:
                ribbon(src, dst, offset, n, nodes[dst][2])
            offset += n

    y_max = max(y1 for _, _, y1 in node_y.values()) + gap
    # height of one text line in y units (cells), to keep the labels of a column from overlapping
    line_h = font_size * 1.3 / 72 / (fig_h * 0.85) * (y_max + gap)
    label_bottom = {}  # column -> y under the last label drawn in it
    for name, (x, y0, y1) in node_y.items():
        if y1 <= y0:
            continue
        ax.add_patch(Rectangle((x - node_w / 2, y0), node_w, y1 - y0, facecolor=nodes[name][2],
                               edgecolor='white', linewidth=1.5, zorder=3))
        ha, dx = ('right', -node_w) if x == 0 else ('left', node_w)
        half = (nodes[name][3].count('\n') + 1) * line_h / 2
        y_text = max((y0 + y1) / 2, label_bottom.get(x, -np.inf) + half)  # pushed down if it would overlap
        label_bottom[x] = y_text + half
        ax.text(x + dx, y_text, nodes[name][3], ha=ha, va='center', fontsize=font_size)

    ax.set_xlim(-0.85, n_cols - 0.2)
    ax.set_ylim(max(y_max, max(label_bottom.values())), -gap)  # top-down
    ax.axis('off')
    # No title (panel h of the main figure of the control experiment): the settings are printed instead
    print(f'step1 tracking efficiency: step0 -> step1 (tracked) -> {END_STEP} | {subtitle}')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return {name: n for name, (_, n, _, _) in nodes.items()}


##### GIF OF THE STEP1 MOVIE, CONTOURS COLORED BY STEP0 VS STEP4 AGREEMENT

def save_tracking_gif(canal_dir, chain, output_path, downsample, dead=None):
    '''Every step1 frame (stitched mosaic), downsampled by `downsample`, with
    each chained track's contour at that frame in green (same step0/step4
    class) or red (different), and every other cell of that frame (not
    matched to a color at step0 and/or step4, or not tracked at all) in
    light gray - or, with --dead_cell_removal, near-black for the tracks
    excluded as dead. The track -> per-frame label lookup comes from step1's
    tracks.csv, the same one build_chain used.'''
    dead = dead or set()
    frames_dir = canal_dir / 'step1' / 'global_masks'
    tracks = pd.read_csv(find_one(canal_dir / 'step1' / 'track', '*_ch00_tracks.csv'))
    match_by_uid = dict(zip(chain['unique_id'], chain['class_step0'] == chain['class_end']))
    t_values = sorted(tracks['t'].unique())

    lo = hi = None
    frames = []
    for t in t_values:
        image = tifffile.imread(find_one(frames_dir, f'*_t{t:02d}_ch00_stitched.tif'))[::downsample, ::downsample]
        if lo is None:  # one intensity window for the whole movie, so it doesn't flicker
            lo, hi = np.percentile(image, [0.5, 99.5])
        gray = (np.clip((image.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)

        # label -> color lookup for this frame: chained tracks green/red, every other cell gray
        mask = np.asarray(Image.open(find_one(frames_dir, f'*_t{t:02d}_ch00_global_mask.png')))[::downsample, ::downsample]
        lut = np.zeros((int(mask.max()) + 1, 3), dtype=np.uint8)
        lut[1:] = UNMATCHED_RGB
        frame_tracks = tracks[tracks['t'] == t]
        for gid, uid in zip(frame_tracks['global_id'], frame_tracks['unique_id']):
            if gid >= len(lut):
                continue
            if uid in match_by_uid:
                lut[gid] = MATCH_RGB if match_by_uid[uid] else MISMATCH_RGB
            elif uid in dead:
                lut[gid] = DEAD_RGB

        # contours traced at output resolution, thickened to 2 px
        edges = ndi.binary_dilation(find_boundaries(mask, mode='inner'))
        edge_labels = ndi.grey_dilation(mask, size=3)
        draw_px = edges & (edge_labels > 0)
        rgb[draw_px] = lut[edge_labels[draw_px]]

        frame = Image.fromarray(rgb)
        ImageDraw.Draw(frame).text((10, 10), f'step 1  t = {t:02d}', fill=(255, 255, 255))
        frames.append(frame)

    # legend strip under the movie
    n_match, n_total = int(sum(match_by_uid.values())), len(match_by_uid)
    n_unmatched = tracks['unique_id'].nunique() - n_total - len(dead)
    width = frames[0].width
    legend = Image.new('RGB', (width, 22), (90, 90, 90))  # dark, so the light-gray swatch shows
    draw = ImageDraw.Draw(legend)
    x = 10
    for rgb_color, text in ((MATCH_RGB, f'same color step0 vs {END_STEP} ({n_match})'),
                            (MISMATCH_RGB, f'different color ({n_total - n_match})'),
                            (UNMATCHED_RGB, f'no step0 and/or {END_STEP} color ({n_unmatched} track(s) + untracked cells)'),
                            *([(DEAD_RGB, f'dead, excluded ({len(dead)})')] if dead else [])):
        draw.rectangle((x, 6, x + 12, 16), fill=rgb_color)
        draw.text((x + 18, 5), text, fill=(255, 255, 255))
        x += 18 + int(draw.textlength(text)) + 30

    frames = [Image.fromarray(np.vstack([np.asarray(f), np.asarray(legend)])) for f in frames]
    # Fixed palette: exact red/green/light gray/legend colors + evenly spaced grays for the brightfield
    fixed = [MATCH_RGB, MISMATCH_RGB, UNMATCHED_RGB, DEAD_RGB, (90, 90, 90), (255, 255, 255)]
    grays = [(v, v, v) for v in np.linspace(0, 255, 256 - len(fixed)).astype(int)]
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette([v for c in fixed + grays for v in c])
    frames = [f.quantize(palette=palette_img, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=GIF_FRAME_MS, loop=0, optimize=True)


##### MAIN

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--canal_dir', type=Path, required=True,
                        help='e.g. path/to/control/figures_input/canal2 (needs match_step_0_1/, '
                             "step1/track/, the END step's match_step_1_<N>/, step0/ and END signal_per_object.csv).")
    parser.add_argument('--end', choices=['step4', 'after_pfa'], default='step4',
                        help='END step compared with step0: step4, or the after-PFA step (step5 in canal2, '
                             'step7 in canal05, detected).')
    parser.add_argument('--label_mode', choices=['quantile', 'threshold'], default='quantile')
    parser.add_argument('--ratio_qc', type=float, default=None,
                        help='In [0, 1]: keep only step1->END matches with 1 - centroid distance / '
                             'second-nearest-neighbour distance >= ratio_qc on both sides of the match '
                             '(0.7 = the match at most 0.3x as far as the second candidate; larger = '
                             'stricter; default: no QC). The step0->step1 match is not filtered.')
    parser.add_argument('--positive_fraction', type=float, default=0.5,
                        help='--label_mode quantile only: fraction of cells (highest q90) labeled positive.')
    parser.add_argument('--gif_downsample', type=int, default=4,
                        help='Movie GIF: keep every N-th pixel of the stitched step1 frames (default 4: '
                             '7577 px -> 1895 px).')
    parser.add_argument('--no_gif', action='store_true', help='Skip the step1 movie GIF.')
    parser.add_argument('--dead_cell_removal', action='store_true',
                        help='Exclude fully tracked step1 cells with iou_all_frames >= --iou_split_threshold '
                             '(a near-static footprint over the whole movie = dead cell) from the matching.')
    parser.add_argument('--iou_split_threshold', type=float, default=0.6,
                        help="--dead_cell_removal cutoff on step1's iou_all_frames (default 0.6).")
    args = parser.parse_args()
    if args.ratio_qc is not None and not 0 <= args.ratio_qc <= 1:
        parser.error(f'--ratio_qc must be in [0, 1], got {args.ratio_qc} (e.g. 0.7; larger = stricter)')

    output_dir = Path(args.canal_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Canal dir: {args.canal_dir}  |  saving outputs -> {output_dir}/')

    global END_STEP, END_FOLDER, END_MATCH_DIR
    END_STEP, END_FOLDER, END_MATCH_DIR = resolve_end_step(args.canal_dir, args.end)
    print(f'END step: {END_STEP} ({END_FOLDER}, matched via {END_MATCH_DIR})')
    out = f'tracking_accuracy_step0_{END_STEP}'

    # Fully tracked = present in every frame (tracks have no gaps, so this is
    # the same as spanning the first to the last frame)
    tracks = pd.read_csv(find_one(args.canal_dir / 'step1' / 'track', '*_ch00_tracks.csv'))
    n_frames = tracks['t'].nunique()
    frames_per_track = tracks.groupby('unique_id')['t'].nunique()
    full_tracks = set(frames_per_track.index[frames_per_track == n_frames])

    chain = add_labels(build_chain(args.canal_dir, args.ratio_qc), args.label_mode, args.positive_fraction, args.canal_dir)
    assert set(chain['unique_id']) <= full_tracks, 'chained cells should all be fully tracked'
    subtitle = (f'labels: top {args.positive_fraction:.0%} q90 per step' if args.label_mode == 'quantile'
                else "labels: each step's saved thresholds")
    if args.ratio_qc is not None:
        subtitle += f'  |  ratio QC {args.ratio_qc}'

    dead = set()
    if args.dead_cell_removal:
        dead = load_dead_tracks(args.canal_dir, tracks, full_tracks, args.iou_split_threshold)
        n_before = len(chain)
        chain = chain[~chain['unique_id'].isin(dead)]
        subtitle += f'  |  dead cells removed (iou_all_frames >= {args.iou_split_threshold})'
        print(f'--dead_cell_removal: {len(dead)}/{len(full_tracks)} fully tracked cell(s) dead '
              f'(iou_all_frames >= {args.iou_split_threshold})  ->  {len(chain)}/{n_before} chained cell(s) kept')

    # Barcoded = matched to step0; paired = also matched to END (both before the
    # ratio QC, which only drops links). Every stage is a subset of the previous one.
    alive_full = full_tracks - dead
    barcoded = barcoded_tracks(args.canal_dir) & alive_full
    if args.ratio_qc is None:
        paired = set(chain['unique_id'])
    else:
        print('Counting pairs before the ratio QC:')
        paired = set(build_chain(args.canal_dir)['unique_id']) - dead
    assert set(chain['unique_id']) <= paired <= barcoded <= alive_full

    pairs_path = output_dir / f'{out}_pairs.csv'
    chain.rename(columns=lambda c: c.replace('_end', f'_{END_STEP}')).to_csv(pairs_path, index=False)
    print(f'Saved -> {pairs_path}  ({len(chain)} chained cell(s))')

    confusion_path = output_dir / f'{out}_confusion.png'
    stats = plot_confusion(chain, confusion_path, subtitle)
    print(f'Saved -> {confusion_path}')
    print(f"  accuracy {stats['accuracy'] * 100:.1f}%  |  balanced {stats['balanced_accuracy'] * 100:.1f}%  |  "
          f"random-label baseline {stats['random_baseline'] * 100:.1f}%  |  "
          + '  '.join(f"{ch} agreement {stats[f'{ch}_agreement'] * 100:.1f}%" for ch in CHANNELS))

    flux_path = output_dir / f'{out}_flux.png'
    counts = plot_tracking_flux(len(full_tracks), n_frames, len(barcoded), len(paired), chain, flux_path, subtitle,
                                n_dead=len(dead) if args.dead_cell_removal else None,
                                iou_threshold=args.iou_split_threshold, ratio_qc=args.ratio_qc)
    print(f'Saved -> {flux_path}')
    print(f"  {counts['tracked']} of {len(frames_per_track)} step1 track(s) span all {n_frames} frames"
          + (f"  |  {counts['dead']} assumed dead, {counts['alive']} alive" if args.dead_cell_removal else '')
          + f"  |  {counts['barcoded']} barcoded, {counts['not_barcoded']} not"
          + f"  |  {counts['paired']} paired with {END_STEP}, {counts['unpaired']} unpaired"
          + (f"  |  {counts['final']} final dataset, {counts['filtered']} filtered out" if args.ratio_qc is not None else '')
          + f"  |  {counts['accurate']} accurate, {counts['inaccurate']} inaccurate")

    if not args.no_gif:
        gif_path = output_dir / f'{out}_movie.gif'
        print(f'Building the step1 movie GIF (downsample x{args.gif_downsample})...')
        save_tracking_gif(args.canal_dir, chain, gif_path, args.gif_downsample, dead)
        print(f'Saved -> {gif_path}  ({gif_path.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
