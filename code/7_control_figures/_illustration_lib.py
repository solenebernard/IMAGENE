'''
Shared drawing code for the "matched cells" figures (panels e, f and g of the main figure): everything that
compares each live cell's colour barcode (step0, carried through the step1
tracking) with the colour of the fixed cell it is matched to after PFA.

Entry points used by the figure scripts, at the bottom of this file:
    configure(canal_dir)   point the module at one canal
    prepare(args)          joined table + barcode labels + manual exclusions
    load_geometry(df)      masks, contours and centroids for the drawings

Naming: the after-PFA step is called "step7" in column names and variables
whatever its real number (step5 in canal2, step7 in canal05).

Barcode classes (ch01, ch02 positivity) and their colours:
    (-,-) black   (+,-) yellow (ch01 only)   (-,+) blue (ch02 only)   (+,+) green
    gray  = no colour data, or dead cell when the dead cells are marked (mark_dead)

Match QC (ratio_qc, in [0, 1], larger = stricter): a step1 <-> after-PFA match
is trusted if, on both sides,
    1 - centroid distance / distance to the second-nearest candidate >= ratio_qc
'''

import csv
import json
import sys
from pathlib import Path

# Loading and alignment of the masks: pipeline/helpers.py
sys.path.insert(0, str(Path(__file__).resolve().parent / 'pipeline'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
from skimage.measure import regionprops, find_contours

from helpers import find_one, pick_mask, rescale_mask, build_aligned_canvas

CANAL_DIR = Path('figures_input/canal2/')  # overwritten from --canal_dir in main()
STEP0_FOLDER = 'step0'
STEP1_FOLDER = 'step1'
# The after-PFA step is step5 in canal2 but step7 in canal05: both are
# detected from --canal_dir in main() (detect_after_pfa). Internally it is
# still called "step7" (column suffixes, variable names) whatever its number.
STEP7_FOLDER = 'step5_afterPFA'
MATCH_1_7_DIR = 'match_step_1_5'
MATCH_0_1_DIR = 'match_step_0_1'

CLASS_COLORS = {
    (False, False): '#000000',  # black: negative in both channels
    (False, True): '#286EFF',   # blue: ch02 only
    (True, False): '#FFD700',   # yellow: ch01 only
    (True, True): '#00C846',    # green: both channels
}

# CLASS_COLORS = {
#     (False, False, False): "pink",
#     (False, False, True): "purple",
#     (False, True, False): "yellow",
#     (False, True, True): "orange",
#     (True, False, False): "green",
#     (True, False, True): "lightblue",
#     (True, True, False): "darkblue",
#     (True, True, True): "blue",
# }


EXCLUDED_COLOR = 'gray'  # no ch01/ch02 q90 data at all, or dead cell (see mark_dead) - nothing to color by
# global_id appears in the <prefix>_excluded_objects_<channel>.csv of its step, if present.
# Only a marker that keeps these objects out of the statistics: it is drawn
# as a gray too and has no legend entry (must differ from every CLASS_COLORS
# value and from EXCLUDED_COLOR).
MANUAL_EXCLUDED_COLOR = 'dimgray'
# The two cells of a matched pair that passes the match-QC are drawn at full
# opacity; every other cell (unmatched, or matched but failing the QC) is
# faded: every alpha of its outline / fill / hatch is multiplied by FADED_ALPHA.
FADED_ALPHA = 0.25
LINK_COLOR = '0.35'  # centroid-to-centroid line of those pairs
LINK_LINEWIDTH = 0.6
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from tools.read_yaml import PIXEL_SIZE
PIXEL_SIZE_UM = PIXEL_SIZE  # µm per pixel of step1's frame (the frame of the map), constants.pixel_size_um
SCALE_BAR_UM = 100      # length of the map's scale bar

BACKGROUND_ALPHA = 0.2  # faint translucent fill behind both outline and hatch
HATCH_PATTERN = '//////////'  # repeat count controls density (more repeats = tighter lines)
HATCH_LINEWIDTH = 1.4  # matplotlib.rcParams['hatch.linewidth'] - global, set once in main()

# Random-label baseline (see random_label_baseline): how many shuffles to
# average the chance same-color rate over, and a fixed seed so re-running
# on the same data reproduces the same title number.
N_RANDOM_TRIALS = 1000
RANDOM_SEED = 0


##### LOADING + JOINING

def _one(pattern_dir, pattern):
    candidates = [f for f in sorted(pattern_dir.glob(pattern)) if not f.name.startswith('.')]
    if not candidates:
        raise FileNotFoundError(f'No file matching {pattern} in {pattern_dir}')
    return candidates[0]


def _left_merge(left, right, on):
    '''merge(how='left') that never matches NaN keys against each other
    (pandas treats NaN==NaN as a match by default, unlike SQL).'''
    return left.merge(right.dropna(subset=[on]), on=on, how='left')


def load_excluded_ids(step_folder, channel):
    '''Manually-excluded global_ids for this step+channel, from
    <prefix>_excluded_objects_<channel>.csv --
    empty set if that step/channel combination was never annotated.'''
    path = find_one(CANAL_DIR / step_folder / 'global_masks', f'*_excluded_objects_{channel}.csv', required=False)
    if path is None:
        return set()
    with open(path, newline='') as f:
        return {int(row['global_id']) for row in csv.DictReader(f)}


def detect_after_pfa(canal_dir):
    '''(after-PFA step folder, its match_step_1_<N> folder) in canal_dir, e.g.
    ('step7_afterPFA', 'match_step_1_7') for canal05, ('step5_afterPFA',
    'match_step_1_5') for canal2.'''
    folders = [d.name for d in sorted(canal_dir.glob('step*_afterPFA')) if d.is_dir()]
    if len(folders) != 1:
        raise FileNotFoundError(f'Expected exactly one step<N>_afterPFA folder in {canal_dir}, found {folders}')
    step_number = folders[0].removeprefix('step').split('_')[0]
    match_dir = f'match_step_1_{step_number}'
    if not (canal_dir / match_dir).is_dir():
        raise FileNotFoundError(f'No {match_dir}/ in {canal_dir} -- run step 3.1 (3_match_bf_fish/1_live_fish_matching.py) '
                                f'for step1 -> {folders[0]}, then 7_control_figures/analysis/2_prepare_figure_inputs.py (see run_control.sh).')
    return folders[0], match_dir


def load_alignment_params(match_dir):
    candidates = [f for f in sorted(match_dir.glob('*_alignment_params.json')) if not f.name.startswith('.')]
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


def load_step1_dynamics():
    '''track_dynamics.py's per-track dynamics CSV (unique_id, n_frames,
    centroid_total_distance, ..., iou_all_frames, ...) for step1_bf_movie,
    columns prefixed step1_ so
    e.g. step1_iou_all_frames lines up with the rest of this file's
    step<N>_ naming. None (with a warning) if track_dynamics.py hasn't been
    run yet - everything else in this file still works without it, just
    without the iou-split outputs in main().'''
    candidates = [f for f in sorted((CANAL_DIR / STEP1_FOLDER / 'track').glob('*_track_dynamics.csv'))
                  if not f.name.startswith('.')]
    if not candidates:
        print('  Warning: no *_track_dynamics.csv in step1_bf_movie/track/ -- '
              'run track_dynamics.py first to get step1_iou_all_frames etc.')
        return None
    dynamics = pd.read_csv(candidates[0])
    return dynamics.rename(columns={c: (c if c == 'unique_id' else f'step1_{c}') for c in dynamics.columns})


def load_and_join():
    df = pd.read_csv(str(_one(CANAL_DIR / MATCH_1_7_DIR, '*_match.csv')))
    df_match_0_1 = pd.read_csv(str(_one(CANAL_DIR / MATCH_0_1_DIR, '*_match.csv')))
    df_track_movie = pd.read_csv(str(_one(CANAL_DIR / STEP1_FOLDER / 'track', '*_tracks.csv')))

    # Bridge step1's t0 (matched to step0) and t20 (matched to step7) via
    # the movie's persistent unique_id - see the module docstring.
    # (first/last frame of the movie - t0/t20 for a 21-frame step1)
    t_first, t_last = df_track_movie['t'].min(), df_track_movie['t'].max()
    df_track0 = df_track_movie[df_track_movie['t'] == t_first].rename(columns={'global_id': 'global_id_step1_t0'})
    df_track20 = df_track_movie[df_track_movie['t'] == t_last].rename(columns={'global_id': 'global_id_step1_t20'})
    df_track = df_track0.merge(df_track20, on='unique_id')

    df = df.rename(columns={
        'global_id_1': 'global_id_step1_t20', 'global_id_2': 'global_id_step7',
        'iou': 'iou_step_1_7', 'centroid_distance_px': 'centroid_distance_px_1_7', 'match_type': 'match_type_1_7',
    })
    # match_step_0_1 now also has second_nn_distance_1/_2 (mutual-nearest matching):
    # suffixed so they don't collide with match_step_1_7's, which passes_match_qc reads
    df_match_0_1 = df_match_0_1.rename(columns={
        'global_id_1': 'global_id_step0', 'global_id_2': 'global_id_step1_t0',
        'second_nn_distance_1': 'second_nn_distance_1_0_1', 'second_nn_distance_2': 'second_nn_distance_2_0_1',
    })

    colors0 = pd.read_csv(str(_one(CANAL_DIR / STEP0_FOLDER / 'global_masks', '*_signal_per_object.csv')))
    colors0 = colors0.rename(columns=lambda c: f'{c}_step0')
    colors7 = pd.read_csv(str(_one(CANAL_DIR / STEP7_FOLDER / 'global_masks', '*_signal_per_object.csv')))
    colors7 = colors7.rename(columns=lambda c: f'{c}_step7')

    df_match_0_1 = _left_merge(df_match_0_1, colors0, on='global_id_step0')
    df = _left_merge(df, colors7, on='global_id_step7')
    df = _left_merge(df, df_track, on='global_id_step1_t20')
    df = _left_merge(df, df_match_0_1, on='global_id_step1_t0')

    dynamics = load_step1_dynamics()
    if dynamics is not None:
        df = _left_merge(df, dynamics, on='unique_id')
    return df


def with_quantile_split_labels(df, positive_fraction=0.5):
    '''Overwrite ch0i_positive_stepj (for ch01/ch02 x step0/step7) with a
    quantile split of ch0i_q90_stepj instead of the pipeline's own
    Otsu/manual-threshold call: the top positive_fraction of objects (by
    q90) are labeled positive, the rest negative - so positive/negative
    always splits the same way by construction (positive_fraction=0.5 is
    the original median split), rather than however many objects happened
    to clear a saved cutoff. class_for() below reads
    these same column names, so nothing downstream needs to change to pick
    this up.'''
    df = df.copy()
    for step in ('step0', 'step7'):
        for ch in ('ch01', 'ch02'):
            q90_col, label_col = f'{ch}_q90_{step}', f'{ch}_positive_{step}'
            cutoff = df[q90_col].quantile(1 - positive_fraction)
            # plain bool dtype can't hold NaN - pandas' nullable "boolean"
            # dtype can, so a missing q90 correctly stays missing here
            # instead of silently becoming a (wrong) False.
            df[label_col] = (df[q90_col] > cutoff).astype('boolean')
            df.loc[df[q90_col].isna(), label_col] = pd.NA
    return df


# Alternative to with_median_split_labels() below: fixed, manually-picked
# cutoffs instead of a per-run 50/50
# split - select via --label_mode fixed.
FIXED_THRESHOLDS = {
    'ch01_threshold_step0': 120,
    'ch02_threshold_step0': 131,
    'ch01_threshold_step7': 98,
    'ch02_threshold_step7': 106,
}


def with_fixed_threshold_labels(df, thresholds):
    '''Overwrite ch0i_positive_stepj (for ch01/ch02 x step0/step7) using
    FIXED thresholds (q90 > thresholds[f"{ch}_threshold_{step}"]) instead
    of with_median_split_labels()'s per-run median - same column names
    either way, so class_for() below needs no changes to pick this up.'''
    df = df.copy()
    for step in ('step0', 'step7'):
        for ch in ('ch01', 'ch02'):
            q90_col, label_col = f'{ch}_q90_{step}', f'{ch}_positive_{step}'
            threshold = thresholds[f'{ch}_threshold_{step}']
            df[label_col] = (df[q90_col] > threshold).astype('boolean')
            df.loc[df[q90_col].isna(), label_col] = pd.NA
    return df


def class_for(row, suffix, excluded_ch01, excluded_ch02):
    '''suffix: 'step0' (-> step1's outline color) or 'step7' (-> step7's
    hatch color). A manual exclusion (this object's own global_id in
    EITHER channel's exclusion list) takes priority
    over everything else - it's a deliberate human call. Otherwise this
    always returns the object's own ch01/ch02 class color; gray is only
    a fallback for objects with no q90 data at all, or rows flagged
    dead by mark_dead(). Match-pairing reliability is NOT considered
    here any more - see passes_match_qc, used only to pick the pairs
    render_illustration compares.'''
    is_dead = row.get('is_dead')
    if pd.notna(is_dead) and bool(is_dead):
        return EXCLUDED_COLOR
    obj_id = row.get(f'global_id_{suffix}')
    if pd.notna(obj_id) and (int(obj_id) in excluded_ch01 or int(obj_id) in excluded_ch02):
        return MANUAL_EXCLUDED_COLOR

    p1, p2 = row.get(f'ch01_positive_{suffix}'), row.get(f'ch02_positive_{suffix}')
    if pd.isna(p1) or pd.isna(p2):
        return EXCLUDED_COLOR
    return CLASS_COLORS[(bool(p1), bool(p2))]
    # p1, p2, p3 = row.get(f"ch01_positive_{suffix}"), row.get(f"ch02_positive_{suffix}"), row.get(f"ch03_positive_{suffix}")
    # if pd.isna(p1) or pd.isna(p2) or pd.isna(p3):
    #     return EXCLUDED_COLOR
    # return CLASS_COLORS[(bool(p1), bool(p2),  bool(p3))]


def passes_match_qc(row, ratio_qc):
    '''Whether a matched pair's pairing itself is trustworthy enough to
    draw at full opacity and compare: its centroid distance must be small
    relative to BOTH sides' second-nearest neighbour distance,
        1 - centroid_distance_px_1_7 / second_nn_distance_1 >= ratio_qc
        1 - centroid_distance_px_1_7 / second_nn_distance_2 >= ratio_qc
    (ratio_qc in [0, 1], larger = stricter: e.g. 0.7 = the match at most
    0.3x as far as the second candidate)
    - a second candidate almost as close as the match itself makes the
    pairing ambiguous, independent of either side's own color.
    (centroid_distance_px_1_7 is the step1 <-> after-PFA hop's distance,
    whatever that step's number - renamed in load_and_join.) A match with
    no second_nn columns (e.g. an IoU-based hop) can't be judged, so
    it passes.'''
    has_second_nn = 'second_nn_distance_1' in row.index and pd.notna(row.get('second_nn_distance_1'))
    if not has_second_nn:
        return True
    dist = row['centroid_distance_px_1_7']
    return bool(1 - dist / row['second_nn_distance_1'] >= ratio_qc
                and 1 - dist / row['second_nn_distance_2'] >= ratio_qc)


def random_label_baseline(colors_1, colors_7, n_trials=N_RANDOM_TRIALS, seed=RANDOM_SEED):
    '''Chance-agreement baseline for the real same-color match rate: keep
    colors_1 (step1's ground-truth color for each highlighted pair) fixed,
    repeatedly shuffle colors_7 (breaking the actual step1<->step7
    correspondence while preserving step7's own color frequencies), and
    average the resulting same-color fraction over n_trials shuffles. This
    is the classic permutation-test null - how much apparent agreement
    you'd see from the marginal class frequencies alone, with no real
    pairing at all - to contrast against the real (paired) rate.'''
    if not colors_1:
        return float('nan')
    arr1, arr7 = np.array(colors_1), np.array(colors_7)
    rng = np.random.default_rng(seed)
    fractions = [np.mean(arr1 == rng.permutation(arr7)) for _ in range(n_trials)]
    return float(np.mean(fractions))


def matched_pairs_table(df, excluded_step0_ch01, excluded_step0_ch02, excluded_step7_ch01, excluded_step7_ch02):
    '''One row per matched (both step1 AND step7 sides present) pair in
    df, with just what an accuracy-vs-QC-threshold sweep needs: color1,
    color7, both_colored, and the two raw QC distances. Colors and manual
    exclusion don't depend on the QC thresholds at all, so this is
    computed ONCE per (sub)population and reused across every ratio_qc
    balanced_accuracy_at_threshold call in the sweep, instead of re-deriving them
    per threshold like render_illustration's own loop does per drawing
    pass.'''
    matched = df.dropna(subset=['global_id_step1_t20', 'global_id_step7']).copy()
    color1 = matched.apply(lambda row: class_for(row, 'step0', excluded_step0_ch01, excluded_step0_ch02), axis=1)
    color7 = matched.apply(lambda row: class_for(row, 'step7', excluded_step7_ch01, excluded_step7_ch02), axis=1)
    manual = (color1 == MANUAL_EXCLUDED_COLOR) | (color7 == MANUAL_EXCLUDED_COLOR)
    color1, color7 = color1.mask(manual, MANUAL_EXCLUDED_COLOR), color7.mask(manual, MANUAL_EXCLUDED_COLOR)
    both_colored = ~color1.isin([EXCLUDED_COLOR, MANUAL_EXCLUDED_COLOR]) & ~color7.isin([EXCLUDED_COLOR, MANUAL_EXCLUDED_COLOR])

    second_nn_1 = matched['second_nn_distance_1'] if 'second_nn_distance_1' in matched.columns else np.nan
    second_nn_2 = matched['second_nn_distance_2'] if 'second_nn_distance_2' in matched.columns else np.nan
    return pd.DataFrame({
        'color1': color1.to_numpy(), 'color7': color7.to_numpy(), 'both_colored': both_colored.to_numpy(),
        'centroid_distance_px_1_7': matched['centroid_distance_px_1_7'].to_numpy(),
        'second_nn_distance_1': second_nn_1, 'second_nn_distance_2': second_nn_2,
    })


def qc_pass_mask(pairs, ratio_qc):
    '''Vectorized passes_match_qc over matched_pairs_table's rows.'''
    dist = pairs['centroid_distance_px_1_7'].to_numpy(dtype='float64')
    second_1 = pairs['second_nn_distance_1'].to_numpy(dtype='float64')
    second_2 = pairs['second_nn_distance_2'].to_numpy(dtype='float64')
    has_second_nn = ~np.isnan(second_1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return ~has_second_nn | ((1 - dist / second_1 >= ratio_qc) & (1 - dist / second_2 >= ratio_qc))


def balanced_accuracy_at_threshold(pairs, ratio_qc):
    '''(balanced_accuracy, n_selected) among matched_pairs_table's rows
    that pass match-QC at this ratio_qc - same pass/fail rule
    as passes_match_qc, vectorized over the whole table at once instead of
    row-by-row, so a sweep over many threshold values stays cheap.

    balanced_accuracy is mean per-class recall (only over color1 classes
    actually present among the selected rows), same convention as
    sklearn.metrics.balanced_accuracy_score / plot_confusion_matrix's
    balanced_accuracy. NaN (not 0% or 100%) when n_selected == 0 - no
    data at all to compute a rate from at that threshold, so there's
    nothing to plot there rather than a misleading edge value.'''
    selected = pairs['both_colored'].to_numpy() & qc_pass_mask(pairs, ratio_qc)
    n_selected = int(selected.sum())
    if n_selected == 0:
        return float('nan'), 0

    c1, c7 = pairs['color1'].to_numpy()[selected], pairs['color7'].to_numpy()[selected]
    recalls = [float(np.mean(c7[c1 == cls] == cls)) for cls in np.unique(c1)]
    return float(np.mean(recalls)), n_selected


##### CONTOUR EXTRACTION
# Per object, from its own small bbox crop (not the whole mask), kept at full resolution

def object_polygons(mask, labels):
    '''({label: [contour (N,2) arrays in (row, col) full-resolution coords]},
    {label: (row, col) centroid}) - centroids come along for free from the
    same regionprops pass used for the contours' bbox crops, so the fine
    connecting line below doesn't need a second pass over the mask.'''
    contours_by_label: dict[int, list[np.ndarray]] = {}
    centroids: dict[int, tuple[float, float]] = {}
    for rp in regionprops(mask.astype(np.int32)):
        if rp.label not in labels:
            continue
        centroids[rp.label] = rp.centroid
        r0, c0, r1, c1 = rp.bbox
        pad = 1
        r0p, c0p = max(0, r0 - pad), max(0, c0 - pad)
        crop = mask[r0p:r1 + pad, c0p:c1 + pad] == rp.label
        contours = find_contours(crop.astype(np.float32), level=0.5)
        contours_by_label[rp.label] = [contour + np.array([r0p, c0p]) for contour in contours]
    return contours_by_label, centroids


def render_illustration(
    df,
    mask1, contours1, centroids1,
    mask7, contours7, centroids7,
    excluded_step0_ch01, excluded_step0_ch02,
    excluded_step7_ch01, excluded_step7_ch02,
    ratio_qc,
    output_path, dpi, title_suffix='',
):
    '''Draw one illustration PNG for the given (already filtered) rows of
    df, reusing masks/contours/centroids/exclusions computed once for the
    whole dataset in main() - so main() can call this more than once (the
    full df, then an iou-split subset) without redoing any of that work.

    The two cells of a matched pair that passes match-QC are drawn at full
    opacity, joined by a thin centroid-to-centroid line. Every other cell
    (unmatched, or matched but failing the QC) is faded (FADED_ALPHA),
    without a line.

    Returns (paired_colors_1, paired_colors_7): the step1/step7 color of
    every compared (matched, QC-passing, both-colored) pair drawn here
    - so the caller can feed the exact same pairs into
    plot_confusion_matrix without recomputing which pairs qualify.'''
    fig, ax = plt.subplots(figsize=(14, 14))
    n1_drawn = n7_drawn = n_boxes = n_boxes_same = n_links = 0
    n_pairs_drawn = n_step1_only = n_step7_only = n_qc_pairs = 0
    paired_colors_1: list[str] = []
    paired_colors_7: list[str] = []
    for _, row in df.iterrows():
        gid1_raw, gid7_raw = row.get('global_id_step1_t20'), row.get('global_id_step7')
        has1, has7 = pd.notna(gid1_raw), pd.notna(gid7_raw)
        if not has1 and not has7:
            continue
        if has1 and not has7:
            n_step1_only += 1
        elif has7 and not has1:
            n_step7_only += 1

        gid1 = int(gid1_raw) if has1 else None
        gid7 = int(gid7_raw) if has7 else None
        color1 = class_for(row, 'step0', excluded_step0_ch01, excluded_step0_ch02) if has1 else None
        color7 = class_for(row, 'step7', excluded_step7_ch01, excluded_step7_ch02) if has7 else None
        opacity = FADED_ALPHA  # 1.0 for the two cells of a matched pair passing match-QC

        if has1 and has7:
            # Manual exclusion is a deliberate human call about one
            # specific object - if it applies to either side of a
            # matched pair, the pairing itself is suspect, so it's
            # propagated to both. Missing q90 data (gray) is
            # NOT propagated - it's a per-object data-availability fact,
            # unrelated to the other side, so it stays wherever it
            # actually applies.
            if color1 == MANUAL_EXCLUDED_COLOR or color7 == MANUAL_EXCLUDED_COLOR:
                color1 = color7 = MANUAL_EXCLUDED_COLOR
            n_pairs_drawn += 1

            # Pair passing match-QC (a close second-nearest neighbour
            # makes a pairing ambiguous): both cells at full opacity,
            # whatever their colors, plus a fine centroid-to-centroid
            # line showing which two shapes go together. zorder=5 keeps
            # it above the cell patches regardless of draw order.
            # The pair is COMPARED (title statistics, returned colors)
            # only if both sides also have a real class color (not gray).
            both_colored = (color1 not in (EXCLUDED_COLOR, MANUAL_EXCLUDED_COLOR)
                            and color7 not in (EXCLUDED_COLOR, MANUAL_EXCLUDED_COLOR))
            if passes_match_qc(row, ratio_qc):
                pts1, pts7 = contours1.get(gid1, []), contours7.get(gid7, [])
                if pts1 and pts7:
                    opacity = 1.0
                    n_qc_pairs += 1
                    if both_colored:
                        n_boxes += 1
                        n_boxes_same += color1 == color7
                        paired_colors_1.append(color1)
                        paired_colors_7.append(color7)

                    c1_, c7_ = centroids1.get(gid1), centroids7.get(gid7)
                    if c1_ is not None and c7_ is not None:
                        ax.plot([c1_[1], c7_[1]], [c1_[0], c7_[0]], color=LINK_COLOR,
                                linewidth=LINK_LINEWIDTH, alpha=0.8, zorder=5)
                        n_links += 1

        if has1:
            for contour in contours1.get(gid1, []):
                xy = contour[:, ::-1]
                # faint translucent fill first, then the class-colored
                # outline on top - alpha on a single patch would dim the
                # outline itself too, so these are two separate patches
                # sharing the same contour instead of one patch doing both.
                ax.add_patch(Polygon(xy, closed=True, facecolor=color1, edgecolor='none',
                                     alpha=BACKGROUND_ALPHA * opacity))
                ax.add_patch(Polygon(xy, closed=True, fill=False, edgecolor=color1, linewidth=1.2, alpha=opacity))
                n1_drawn += 1
        if has7:
            for contour in contours7.get(gid7, []):
                xy = contour[:, ::-1]
                ax.add_patch(Polygon(xy, closed=True, facecolor=color7, edgecolor='none',
                                     alpha=BACKGROUND_ALPHA * opacity))
                ax.add_patch(Polygon(xy, closed=True, facecolor='none', edgecolor=color7,
                                     hatch=HATCH_PATTERN, linewidth=0.4, alpha=opacity))
                n7_drawn += 1

    ax.set_xlim(0, mask1.shape[1])
    ax.set_ylim(mask1.shape[0], 0)  # image convention: row 0 at top
    ax.set_aspect('equal')
    ax.axis('off')
    # scale bar, bottom-left corner
    bar_px, pad = SCALE_BAR_UM / PIXEL_SIZE_UM, 0.02 * mask1.shape[1]
    y_bar = mask1.shape[0] - pad
    ax.plot([pad, pad + bar_px], [y_bar, y_bar], color='black', linewidth=3, solid_capstyle='butt', zorder=6)
    ax.text(pad + bar_px / 2, y_bar - 0.6 * pad, f'{SCALE_BAR_UM} µm', ha='center', va='bottom', fontsize=10, zorder=6)
    pct_same = 100 * n_boxes_same / n_boxes if n_boxes else float('nan')
    random_pct = 100 * random_label_baseline(paired_colors_1, paired_colors_7)
    title = (f'step1 (outline) x step7 (hatched)  --  {n_pairs_drawn} matched pair(s), '
             f'{n_step1_only} unmatched step1, {n_step7_only} unmatched step7\n'
             f'{n_qc_pairs} pair(s) pass the match QC (bright); {pct_same:.1f}% of those with a barcode on both '
             f'sides ({n_boxes_same}/{n_boxes}) have the same barcode'
             f'  (random-label baseline: {random_pct:.1f}%)')
    if title_suffix:
        title += f'\n{title_suffix}'
    ax.set_title(title)

    class_names = {(False, False): 'ch01-, ch02-', (True, False): 'ch01+ only',
                   (False, True): 'ch02+ only', (True, True): 'ch01+, ch02+'}
    legend_handles = [
        Patch(facecolor='none', edgecolor=CLASS_COLORS[k], linewidth=1.2, label=class_names[k])
        for k in [(False, False), (True, False), (False, True), (True, True)]
    ]
    gray_label = 'no barcode, or dead cell' if 'is_dead' in df.columns else 'no barcode'
    legend_handles.append(Patch(facecolor='none', edgecolor=EXCLUDED_COLOR, linewidth=1.2, label=gray_label))
    legend_handles.append(Patch(facecolor='none', edgecolor='0.3', linewidth=1.2, label='live cell (step1, last frame)'))
    legend_handles.append(Patch(facecolor='none', edgecolor='0.3', hatch=HATCH_PATTERN, linewidth=0.4,
                                label='fixed cell (after PFA)'))
    legend_handles.append(Line2D([], [], color=LINK_COLOR, linewidth=LINK_LINEWIDTH, label='pair passing the match QC (bright)'))
    legend_handles.append(Patch(facecolor='none', edgecolor='0.3', linewidth=1.2, alpha=FADED_ALPHA,
                                label='faded: unmatched, or match failing the QC'))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {output_path}  '
          f'({n1_drawn} step1 outline(s), {n7_drawn} step7 hatch region(s), '
          f'{n_qc_pairs} QC-passing pair(s) of which {n_boxes} compared, {n_links} link line(s))')
    return paired_colors_1, paired_colors_7


def plot_confusion_matrix(colors_1, colors_7, output_path, dpi, title_suffix=''):
    '''Class-confusion matrix (step1 color vs step7 color) for the SAME
    compared pairs render_illustration drew (see its
    paired_colors_1/_7 return) - diagonal = same-color matches. Cell
    color encodes the ROW-normalized fraction (i.e. given step1 landed in
    that row's class, what fraction of its step7 matches fell in each
    column - sklearn's normalize='true' convention), while each cell's
    text shows both that percentage AND the raw count underneath it.
    Title states the resulting accuracy (== render_illustration's own
    same-color-match rate) against random_label_baseline's chance rate
    (same comparison as the illustration's own title, just as a matrix
    instead of counted boxes), the balanced accuracy (mean per-class
    recall - the row-normalized diagonal, averaged only over classes
    step1 actually has - catches a high plain accuracy that's really
    just one dominant class matching itself, same convention as
    sklearn.metrics.balanced_accuracy_score), and the total pair count.'''
    color_order = list(CLASS_COLORS.values())  # fixed order, matches the data
    tick_labels = [f'({int(k[0])},{int(k[1])})' for k in CLASS_COLORS]  # (ch01,ch02), same order as color_order
    if not colors_1:
        print(f'  Warning: no highlighted pairs -- skipping confusion matrix {output_path.name}')
        return

    counts = pd.crosstab(
        pd.Series(colors_1, name='step1'), pd.Series(colors_7, name='step7'),
    ).reindex(index=color_order, columns=color_order, fill_value=0)
    counts_arr = counts.to_numpy()
    n_total = int(counts_arr.sum())

    row_sums = counts_arr.sum(axis=1, keepdims=True)
    normalized = np.divide(counts_arr, row_sums, out=np.zeros_like(counts_arr, dtype=float), where=row_sums != 0)

    accuracy = float(np.trace(counts_arr)) / n_total
    random_accuracy = random_label_baseline(colors_1, colors_7)

    # Balanced accuracy: mean per-class recall (the row-normalized
    # diagonal), averaged only over classes step1 actually has - same
    # convention as sklearn.metrics.balanced_accuracy_score, which skips
    # classes absent from the true labels rather than counting them as 0.
    has_row = row_sums.flatten() > 0
    balanced_accuracy = float(np.diag(normalized)[has_row].mean()) if has_row.any() else float('nan')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(tick_labels)))
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('step7 (ch01,ch02)')
    ax.set_ylabel('step1 (ch01,ch02)')
    for i in range(len(tick_labels)):
        for j in range(len(tick_labels)):
            text_color = 'white' if normalized[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{normalized[i, j] * 100:.1f}%\n(n={counts_arr[i, j]})',
                    ha='center', va='center', color=text_color, fontsize=9)

    title = (f'Confusion matrix: step1 vs step7 color  (row-normalized)\n'
             f'accuracy={accuracy * 100:.1f}%  balanced_accuracy={balanced_accuracy * 100:.1f}%  '
             f'(random-label baseline: {random_accuracy * 100:.1f}%)\n'
             f'n={n_total} pair(s) total')
    if title_suffix:
        title += f'\n{title_suffix}'
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f'Saved -> {output_path}  (confusion matrix, {n_total} pair(s), '
          f'accuracy={accuracy * 100:.1f}%, balanced_accuracy={balanced_accuracy * 100:.1f}%)')


##### ENTRY POINTS FOR THE FIGURE SCRIPTS

def configure(canal_dir):
    '''Point this module at one canal folder (sets the after-PFA step and its
    match folder: step5 / match_step_1_5 in canal2, step7 / match_step_1_7 in canal05).'''
    global CANAL_DIR, STEP7_FOLDER, MATCH_1_7_DIR
    CANAL_DIR = Path(canal_dir)
    STEP7_FOLDER, MATCH_1_7_DIR = detect_after_pfa(CANAL_DIR)
    print(f'Canal dir: {CANAL_DIR}  |  after-PFA step: {STEP7_FOLDER} (matched via {MATCH_1_7_DIR})')


def prepare(label_mode='quantile', positive_fraction=0.5, dead_cell_removal=False, iou_split_threshold=0.6):
    '''Joined table (one row per step1 and/or after-PFA object) with its
    barcode labels, and the manual exclusion sets.

    label_mode 'quantile': per step and channel, the top positive_fraction of
    objects (by q90 intensity) are positive. 'fixed': FIXED_THRESHOLDS.
    dead_cell_removal: drop step1 cells whose footprint barely changes over the
    movie (step1_iou_all_frames >= iou_split_threshold).

    Returns (df, excluded) with excluded = (step0_ch01, step0_ch02, step7_ch01, step7_ch02).
    '''
    df = load_and_join()
    if label_mode == 'fixed':
        df = with_fixed_threshold_labels(df, FIXED_THRESHOLDS)
    else:
        df = with_quantile_split_labels(df, positive_fraction=positive_fraction)

    if dead_cell_removal:
        if 'step1_iou_all_frames' not in df.columns:
            raise FileNotFoundError("dead_cell_removal needs step1's *_track_dynamics.csv (run track_dynamics.py)")
        dead = df['step1_iou_all_frames'] >= iou_split_threshold  # NaN -> False: those rows are kept
        print(f'  dead-cell removal: {int(dead.sum())} row(s) with step1_iou_all_frames >= {iou_split_threshold} dropped')
        df = df[~dead].copy()

    excluded = (load_excluded_ids(STEP0_FOLDER, 'ch01'), load_excluded_ids(STEP0_FOLDER, 'ch02'),
                load_excluded_ids(STEP7_FOLDER, 'ch01'), load_excluded_ids(STEP7_FOLDER, 'ch02'))
    n_pairs = df.dropna(subset=['global_id_step1_t20', 'global_id_step7']).shape[0]
    print(f"  {len(df)} row(s): {n_pairs} matched pair(s), {df['global_id_step1_t20'].notna().sum()} step1 object(s), "
          f"{df['global_id_step7'].notna().sum()} after-PFA object(s)")
    return df, excluded


def mark_dead(df, iou_split_threshold=0.6):
    '''Flag (column is_dead) instead of dropping the step1 cells whose footprint
    barely changes over the movie (step1_iou_all_frames >= iou_split_threshold).
    class_for() then gives such a row gray, on both the live and the fixed side,
    which also keeps it out of the compared pairs - same pairs as with
    prepare(dead_cell_removal=True).'''
    if 'step1_iou_all_frames' not in df.columns:
        raise FileNotFoundError("marking dead cells needs step1's *_track_dynamics.csv (run track_dynamics.py)")
    df = df.copy()
    df['is_dead'] = df['step1_iou_all_frames'] >= iou_split_threshold  # NaN -> False
    print(f"  dead cells: {int(df['is_dead'].sum())} row(s) with step1_iou_all_frames >= {iou_split_threshold} drawn in gray")
    return df


def load_geometry(df):
    '''(mask1, contours1, centroids1, mask7, contours7, centroids7): step1's last
    frame in its own pixel frame, and the after-PFA mask rescaled + shifted into
    that frame with the alignment saved in match_step_1_<N>/*_alignment_params.json.'''
    mask1, _, _ = pick_mask(CANAL_DIR / STEP1_FOLDER, 'ch00', endpoint='last')
    mask1 = mask1.astype(np.uint32)
    mask7_raw, _, _ = pick_mask(CANAL_DIR / STEP7_FOLDER, 'ch00', endpoint='first')
    params = load_alignment_params(CANAL_DIR / MATCH_1_7_DIR)
    if params is not None:
        mask7 = build_aligned_canvas(mask1.shape, rescale_mask(mask7_raw, params['scale']),
                                     params['dx'], params['dy']).astype(np.uint32)
    else:
        mask7 = mask7_raw.astype(np.uint32)
    contours1, centroids1 = object_polygons(mask1, set(df['global_id_step1_t20'].dropna().astype(int)))
    contours7, centroids7 = object_polygons(mask7, set(df['global_id_step7'].dropna().astype(int)))
    matplotlib.rcParams['hatch.linewidth'] = HATCH_LINEWIDTH  # global rcParam, no per-patch override
    return mask1, contours1, centroids1, mask7, contours7, centroids7
