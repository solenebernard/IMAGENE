'''
Per-track motion + morphology dynamics over a timelapse (e.g.
step1_bf_movie/) - consumes the already-computed tracking results
(global_masks/*_t<NN>_<channel>_global_mask.png + the saved
track/<prefix>_<channel>_tracks.csv) rather than re-deriving them: no
linking happens here, only relabeling each frame's own per-frame global_id
by the track's persistent unique_id (a plain lookup from the tracks.csv
already on disk), then measuring how each track moves and deforms across
the whole timelapse.

Optimised: centroids/bboxes come from
one regionprops pass per frame, and all crop-based work (union area,
multi-frame IoU, consecutive-frame IoU/boundary-distance) happens on each
track's own small bounding-box crop, never the full multi-thousand-pixel
frame.

--n_frames caps how many timepoints dynamics are computed over: by default
(None) every available timepoint is used, same as before. Given a smaller
value, a *contiguous* window of that many frames is picked (one window,
shared by every track, not a per-track choice) instead of the full
timelapse - e.g. to see how much of the signal survives with a shorter
movie. The window's start defaults to random; --frame_start pins it to a
specific timepoint instead (e.g. to compare the same window across runs).

Output: <input_dir>/track/<prefix>_<channel>_track_dynamics.csv (or
..._n<K>_t<start>-<end>.csv when --n_frames is set, so it doesn't clobber
the full-timelapse run)
    unique_id, n_frames, centroid_total_distance, centroid_mean_speed,
    centroid_net_displacement,
    area_explored_px, iou_all_frames, mean_iou_consecutive,
    std_iou_consecutive, mean_contour_deformation, std_contour_deformation,
    mean_area, std_area, mean_perimeter, std_perimeter, mean_eccentricity,
    std_eccentricity (the last six: per-frame shape features, averaged over
    every frame the track is present in - not just consecutive-pair deltas
    like the dynamics above)

Also: <input_dir>/track/<prefix>_<channel>_track_per_frame.csv (same suffix)
-- the per-frame values the table above aggregates, one row per (track, t):
    unique_id, t, global_id (that frame's own label in its global mask),
    centroid_y, centroid_x (px, in the stitched whole-slide global mask),
    area, perimeter, eccentricity,
    dt_prev, step_distance (vs the track's previous present frame, NaN on
    its first), iou_prev, contour_deformation_prev (vs frame t-1 only, NaN
    if the track is absent at t-1 - same rule as the consecutive means)

Usage:
    python track_dynamics.py --input_dir path/to/experiment/Canal2/step1_bf_movie/
    python track_dynamics.py --input_dir ...step1_bf_movie/ --n_frames 21
    python track_dynamics.py --input_dir ...step1_bf_movie/ --n_frames 21 --frame_start 5
'''

import csv
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt

from helpers import discover_timelapse_masks, load_mask


##### TRACK DYNAMICS (MOTION + SHAPE FEATURES)

def _frame_props_full(mask):
    '''Centroid + bbox + shape features for every label in a frame, in one
    regionprops pass, with area/perimeter/
    eccentricity since compute_track_dynamics needs those too and a second
    regionprops pass over the same mask would be wasted work.'''
    props = {}
    for rp in regionprops(mask.astype(np.int32)):
        r0, c0, r1, c1 = rp.bbox
        props[int(rp.label)] = {
            'cy': rp.centroid[0], 'cx': rp.centroid[1], 'bbox': (r0, r1, c0, c1),
            'area': rp.area, 'perimeter': rp.perimeter, 'eccentricity': rp.eccentricity,
        }
    return props


def compute_track_dynamics(frame_masks, t_indices):
    '''
    Compute per-track motion and morphology dynamics over the full timelapse.

    Returns (dynamics, per_frame): dynamics is {track_id: aggregated metrics}
    (keys below); per_frame is one dict per (track, frame) with the values
    those aggregates are computed from (see the module docstring).

    Optimised: centroids/bboxes come from a single regionprops pass per frame;
    EDT is run on the cropped cell bbox, not the full image.

    Per-track output keys:
      centroid_total_distance  - total path length of the centroid (px)
      centroid_mean_speed      - mean displacement per frame (px / frame)
      centroid_net_displacement - straight-line centroid distance between the
                                  track's first and last present frame (px)
      area_explored_px         - union of all pixels occupied across all frames
      mean_iou_consecutive     - mean IoU between mask at t and t+1
      std_iou_consecutive      - std of that IoU
      mean_contour_deformation - mean symmetric boundary distance (px) t->t+1
      std_contour_deformation  - std of boundary distance
      mean_area, std_area              - cell area (px^2) over every present frame
      mean_perimeter, std_perimeter    - cell perimeter (px) over every present frame
      mean_eccentricity, std_eccentricity - shape eccentricity over every present frame
    '''
    # One regionprops pass per frame - cheap, avoids per-track full-image scans
    frame_prop_list = [_frame_props_full(m) for m in frame_masks]

    all_track_ids = sorted({tid for fp in frame_prop_list for tid in fp})
    nan = float('nan')
    dynamics: dict[int, dict] = {}
    per_frame: list[dict] = []

    H, W = frame_masks[0].shape
    MARGIN = 5

    for tid in all_track_ids:
        present_idx = [i for i, fp in enumerate(frame_prop_list) if tid in fp]

        # Area explored: union of crops within the global bounding box for this track
        bboxes = [frame_prop_list[i][tid]['bbox'] for i in present_idx]
        gr0 = max(0, min(b[0] for b in bboxes) - MARGIN)
        gr1 = min(H, max(b[1] for b in bboxes) + MARGIN)
        gc0 = max(0, min(b[2] for b in bboxes) - MARGIN)
        gc1 = min(W, max(b[3] for b in bboxes) + MARGIN)
        union_crop = np.zeros((gr1 - gr0, gc1 - gc0), dtype=bool)
        for i, (r0, r1, c0, c1) in zip(present_idx, bboxes):
            union_crop[r0 - gr0: r1 - gr0, c0 - gc0: c1 - gc0] |= (
                frame_masks[i][r0:r1, c0:c1] == tid
            )
        area_explored = int(union_crop.sum())

        # Multi-frame IoU: intersection of all T masks / union of all T masks.
        # Computed on the global bbox crop so it scales with cell size.
        intersection_px = union_crop.copy()  # start full (all True where any mask was)
        for i in present_idx:
            r0b, r1b, c0b, c1b = frame_prop_list[i][tid]['bbox']
            cell_crop = np.zeros(union_crop.shape, dtype=bool)
            cell_crop[r0b - gr0: r1b - gr0, c0b - gc0: c1b - gc0] = (
                frame_masks[i][r0b:r1b, c0b:c1b] == tid
            )
            intersection_px &= cell_crop
        iou_all_frames = (
            float(intersection_px.sum()) / float(union_crop.sum())
            if union_crop.sum() > 0 else nan
        )

        # Shape features: per-frame area/perimeter/eccentricity, averaged
        # over every frame the track is present in - unlike the motion
        # dynamics below these don't need a second (consecutive) frame, so
        # they're computed here (available even for single-frame tracks).
        areas = [frame_prop_list[i][tid]['area'] for i in present_idx]
        perimeters = [frame_prop_list[i][tid]['perimeter'] for i in present_idx]
        eccentricities = [frame_prop_list[i][tid]['eccentricity'] for i in present_idx]
        shape_stats = {
            'mean_area': float(np.mean(areas)), 'std_area': float(np.std(areas)),
            'mean_perimeter': float(np.mean(perimeters)), 'std_perimeter': float(np.std(perimeters)),
            'mean_eccentricity': float(np.mean(eccentricities)), 'std_eccentricity': float(np.std(eccentricities)),
        }

        frame_rows = []
        for k, i in enumerate(present_idx):
            fp = frame_prop_list[i][tid]
            prev = frame_prop_list[present_idx[k - 1]][tid] if k > 0 else None
            frame_rows.append({
                'unique_id': tid, 't': t_indices[i],
                'centroid_y': fp['cy'], 'centroid_x': fp['cx'],
                'area': fp['area'], 'perimeter': fp['perimeter'], 'eccentricity': fp['eccentricity'],
                'dt_prev': t_indices[i] - t_indices[present_idx[k - 1]] if prev else nan,
                'step_distance': float(np.hypot(fp['cy'] - prev['cy'], fp['cx'] - prev['cx'])) if prev else nan,
                'iou_prev': nan, 'contour_deformation_prev': nan,
            })
        per_frame.extend(frame_rows)

        if len(present_idx) < 2:
            dynamics[tid] = {
                'centroid_total_distance': nan,
                'centroid_mean_speed': nan,
                'centroid_net_displacement': nan,
                'area_explored_px': area_explored,
                'iou_all_frames': iou_all_frames,
                'mean_iou_consecutive': nan,
                'std_iou_consecutive': nan,
                'mean_contour_deformation': nan,
                'std_contour_deformation': nan,
                **shape_stats,
            }
            continue

        # Centroids from regionprops (already computed, no full-image scan)
        centroids = [(frame_prop_list[i][tid]['cy'],
                      frame_prop_list[i][tid]['cx'])
                     for i in present_idx]

        step_distances = [
            float(np.hypot(centroids[k + 1][0] - centroids[k][0],
                           centroids[k + 1][1] - centroids[k][1]))
            for k in range(len(centroids) - 1)
        ]
        total_distance = float(np.sum(step_distances))
        mean_speed = float(np.mean(step_distances))
        net_displacement = float(np.hypot(centroids[-1][0] - centroids[0][0],
                                          centroids[-1][1] - centroids[0][1]))

        # Per-consecutive-pair: IoU and contour deformation
        # All work is done on small bbox crops - no full-image operations here
        ious: list[float] = []
        contour_defs: list[float] = []
        for k in range(len(present_idx) - 1):
            i, j = present_idx[k], present_idx[k + 1]
            if t_indices[j] != t_indices[i] + 1:
                continue

            r0_i, r1_i, c0_i, c1_i = frame_prop_list[i][tid]['bbox']
            r0_j, r1_j, c0_j, c1_j = frame_prop_list[j][tid]['bbox']
            r0 = max(0, min(r0_i, r0_j) - MARGIN)
            r1 = min(H, max(r1_i, r1_j) + MARGIN)
            c0 = max(0, min(c0_i, c0_j) - MARGIN)
            c1 = min(W, max(c1_i, c1_j) + MARGIN)

            crop_t = frame_masks[i][r0:r1, c0:c1] == tid
            crop_t1 = frame_masks[j][r0:r1, c0:c1] == tid

            inter = int((crop_t & crop_t1).sum())
            union = int((crop_t | crop_t1).sum())
            ious.append(inter / union if union > 0 else 1.0)
            frame_rows[k + 1]['iou_prev'] = ious[-1]

            bd_t = find_boundaries(crop_t, mode='inner')
            bd_t1 = find_boundaries(crop_t1, mode='inner')
            if bd_t.any() and bd_t1.any():
                d = (distance_transform_edt(~bd_t1)[bd_t].mean()
                     + distance_transform_edt(~bd_t)[bd_t1].mean()) / 2
                contour_defs.append(float(d))
                frame_rows[k + 1]['contour_deformation_prev'] = contour_defs[-1]

        dynamics[tid] = {
            'centroid_total_distance': total_distance,
            'centroid_mean_speed': mean_speed,
            'centroid_net_displacement': net_displacement,
            'area_explored_px': area_explored,
            'iou_all_frames': iou_all_frames,
            'mean_iou_consecutive': float(np.mean(ious)) if ious else nan,
            'std_iou_consecutive': float(np.std(ious)) if ious else nan,
            'mean_contour_deformation': float(np.mean(contour_defs)) if contour_defs else nan,
            'std_contour_deformation': float(np.std(contour_defs)) if contour_defs else nan,
            **shape_stats,
        }

    return dynamics, per_frame


##### MAIN

def load_track_mapping(track_csv):
    '''{t: {global_id: unique_id}} from the saved tracks.csv - a plain
    lookup, no re-linking.'''
    mapping: dict[int, dict[int, int]] = defaultdict(dict)
    with open(track_csv, newline='') as f:
        for row in csv.DictReader(f):
            mapping[int(row['t'])][int(row['global_id'])] = int(row['unique_id'])
    return mapping


def relabel_by_track(mask, gid_to_uid):
    '''Relabel one frame's own per-frame global_id labels by each track's
    persistent unique_id (vectorized lookup-table remap). Any label not in
    gid_to_uid becomes 0 - e.g. a track dropped by the tracking.'''
    max_gid = int(mask.max())
    lut = np.zeros(max_gid + 1, dtype=np.int64)
    for gid, uid in gid_to_uid.items():
        if 0 < gid <= max_gid:
            lut[gid] = uid
    return lut[np.clip(mask, 0, max_gid)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='e.g. path/to/experiment/Canal2/step1_bf_movie/')
    parser.add_argument('--channel', type=str, default='ch00',
                        help='Mask channel to track (default: ch00 = BF).')
    parser.add_argument('--z', type=int, default=None,
                        help='Preferred z-slice for single-FOV inputs (default: smallest available).')
    parser.add_argument(
        '--n_frames', type=int, default=None,
        help='Compute dynamics over a CONSECUTIVE window of this many '
             'timepoints instead of every available one (default: use all). '
             'One window is picked, shared by every track -- random unless '
             '--frame_start is also given.',
    )
    parser.add_argument(
        '--frame_start', type=int, default=None,
        help='Only used with --n_frames: which timepoint (t) to start the '
             'window at, instead of a random start. Must be an available '
             'timepoint with at least --n_frames timepoints from there on.',
    )
    parser.add_argument('--output_dir', type=Path, default=None,
                        help='Default: <input_dir>/track/')
    args = parser.parse_args()

    if args.frame_start is not None and args.n_frames is None:
        raise ValueError('--frame_start requires --n_frames.')

    # Discovery only - no linking here.
    t_paths, prefix, mode = discover_timelapse_masks(args.input_dir, args.channel, args.z)
    if len(t_paths) < 2:
        raise ValueError(f'Found only {len(t_paths)} timepoint(s) in {args.input_dir} -- nothing to compute.')

    t_indices = sorted(t_paths)
    print(f"Prefix: '{prefix}'  |  mode: {mode}  |  {len(t_indices)} timepoint(s): {t_indices}")

    if args.n_frames is not None:
        if args.n_frames > len(t_indices):
            raise ValueError(f'--n_frames {args.n_frames} exceeds the {len(t_indices)} available timepoint(s).')
        if args.n_frames < len(t_indices):
            if args.frame_start is not None:
                if args.frame_start not in t_indices:
                    raise ValueError(f'--frame_start {args.frame_start} is not an available timepoint '
                                     f'(available: {t_indices[0]}..{t_indices[-1]}).')
                start = t_indices.index(args.frame_start)
                if start + args.n_frames > len(t_indices):
                    raise ValueError(f'--frame_start {args.frame_start} + --n_frames {args.n_frames} '
                                     f'exceeds the available timepoints (last available: {t_indices[-1]}).')
                window_kind = 'fixed'
            else:
                start = random.randint(0, len(t_indices) - args.n_frames)
                window_kind = 'random'
            t_indices = t_indices[start:start + args.n_frames]
            print(f'  {window_kind} window: t={t_indices[0]}..{t_indices[-1]} ({args.n_frames} frame(s))')
        elif args.frame_start is not None:
            print(f'  --frame_start {args.frame_start} ignored -- --n_frames '
                  f'{args.n_frames} already covers every available timepoint')

    track_csv = args.input_dir / 'track' / f'{prefix}_{args.channel}_tracks.csv'
    if not track_csv.exists():
        raise FileNotFoundError(
            f'{track_csv} not found -- run step 1.3 (1_bf_analysis/3_track_cells.py), then 7_control_figures/analysis/2_prepare_figure_inputs.py '
            f'(see run_control.sh).'
        )
    print(f'Loading track mapping from {track_csv} ...')
    track_mapping = load_track_mapping(track_csv)

    print('Loading masks + relabeling by track (no re-linking)...')
    tracked_masks = [
        relabel_by_track(load_mask(t_paths[t]), track_mapping.get(t, {}))
        for t in t_indices
    ]

    n_frames_by_tid: dict[int, int] = defaultdict(int)
    for mask in tracked_masks:
        for tid in np.unique(mask[mask > 0]):
            n_frames_by_tid[int(tid)] += 1

    print(f'Computing dynamics for {len(n_frames_by_tid)} track(s)...')
    dynamics, per_frame = compute_track_dynamics(tracked_masks, t_indices)

    output_dir = args.output_dir or (args.input_dir / 'track')
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'_n{args.n_frames}_t{t_indices[0]:02d}-{t_indices[-1]:02d}' if args.n_frames is not None else ''
    csv_path = output_dir / f'{prefix}_{args.channel}_track_dynamics{suffix}.csv'

    fieldnames = [
        'unique_id', 'n_frames',
        'centroid_total_distance', 'centroid_mean_speed', 'centroid_net_displacement',
        'area_explored_px',
        'iou_all_frames', 'mean_iou_consecutive', 'std_iou_consecutive',
        'mean_contour_deformation', 'std_contour_deformation',
        'mean_area', 'std_area', 'mean_perimeter', 'std_perimeter',
        'mean_eccentricity', 'std_eccentricity',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tid in sorted(dynamics):
            row = {'unique_id': tid, 'n_frames': n_frames_by_tid[tid]}
            row.update(dynamics[tid])
            writer.writerow(row)

    print(f'Saved -> {csv_path}  ({len(dynamics)} track(s))')

    # Per-frame values behind the aggregates above, plus each frame's own
    # global_id so rows can be traced back to that frame's global mask
    uid_to_gid = {t: {uid: gid for gid, uid in track_mapping.get(t, {}).items()} for t in t_indices}
    per_frame_path = output_dir / f'{prefix}_{args.channel}_track_per_frame{suffix}.csv'
    per_frame_fieldnames = [
        'unique_id', 't', 'global_id', 'centroid_y', 'centroid_x',
        'area', 'perimeter', 'eccentricity',
        'dt_prev', 'step_distance', 'iou_prev', 'contour_deformation_prev',
    ]
    with open(per_frame_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_frame_fieldnames)
        writer.writeheader()
        for row in sorted(per_frame, key=lambda r: (r['unique_id'], r['t'])):
            writer.writerow({**row, 'global_id': uid_to_gid[row['t']].get(row['unique_id'])})

    print(f'Saved -> {per_frame_path}  ({len(per_frame)} track-frame row(s))')


if __name__ == '__main__':
    main()
