'''
Track the cells across the time frames, from the global masks of 2_merge_masks.
Adapted from track_timelapse.py

Consecutive frames are matched with the Hungarian algorithm (optimal one-to-one assignment).
The cost is the IoU where the masks overlap, and the distance between centroids where they do not.
A cell that is not linked starts a new track.

Saved in ROOTPATH/EXP_PATH/Live/CONDITION/track/tracks.csv, one row per track and time frame:
    - unique_id: ID of the track
    - t: time frame
    - global_id: ID of the cell in the global mask of this time frame
    - centroid_x, centroid_y: centroid of the cell in the global mask (x: column, y: row)
    - area: number of pixels of the cell
The masks are not saved again: the mask of a track at the time frame t is the cell global_id of the global mask.
'''

# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

from collections import defaultdict
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(mask_a, mask_b):
    """
    IoU between every pair of labels across two frames.
    Encode overlapping pixels as (a_id, b_id) pairs and count them with np.bincount in a single pass.
    """
    flat_a = mask_a.ravel().astype(np.int64)
    flat_b = mask_b.ravel().astype(np.int64)

    ids_a = [int(i) for i in np.unique(flat_a) if i > 0]
    ids_b = [int(i) for i in np.unique(flat_b) if i > 0]
    if not ids_a or not ids_b:
        return ids_a, ids_b, np.zeros((len(ids_a), len(ids_b)), dtype=np.float32)

    max_a = int(flat_a.max())
    max_b = int(flat_b.max())

    # Per-label pixel counts (areas)
    area_a = np.bincount(flat_a, minlength=max_a + 1)
    area_b = np.bincount(flat_b, minlength=max_b + 1)

    # Count overlapping pixels per (a, b) pair by encoding the pair as a single int
    ov = (flat_a > 0) & (flat_b > 0)
    stride = max_b + 1
    pair_codes = flat_a[ov] * stride + flat_b[ov]
    overlap_counts = np.bincount(pair_codes, minlength=max_a * stride + max_b + 1)

    idx_a = {v: i for i, v in enumerate(ids_a)}
    idx_b = {v: i for i, v in enumerate(ids_b)}

    iou = np.zeros((len(ids_a), len(ids_b)), dtype=np.float32)
    for code in np.where(overlap_counts > 0)[0]:
        a, b = int(code // stride), int(code % stride)
        inter = int(overlap_counts[code])
        union = int(area_a[a]) + int(area_b[b]) - inter
        if union > 0:
            iou[idx_a[a], idx_b[b]] = inter / union

    return ids_a, ids_b, iou


def frame_props(mask):
    """Centroid (row, column) and area of every label of a frame: {id: (cy, cx, area)}"""
    return {int(rp.label): (rp.centroid[0], rp.centroid[1], rp.area) for rp in regionprops(mask.astype(np.int32))}


def centroid_distance_matrix(ids_a, ids_b, props_a, props_b):
    """Pairwise Euclidean distance between the centroids of the labels of two frames."""
    if not ids_a or not ids_b:
        return np.zeros((len(ids_a), len(ids_b)), dtype=np.float32)
    c_a = np.array([props_a[i][:2] for i in ids_a])
    c_b = np.array([props_b[i][:2] for i in ids_b])
    return np.hypot(c_a[:, None, 0] - c_b[None, :, 0], c_a[:, None, 1] - c_b[None, :, 1]).astype(np.float32)


def link_across_time(frame_masks, props, iou_threshold=0, max_centroid_distance=100):
    """
    Assign unique track IDs to the cells across all the time frames.
    Returns id_map: {(t, id in frame_masks[t]): track ID}
    """
    track_id = 1
    id_map = {}

    for local in np.unique(frame_masks[0]):
        if local > 0:
            id_map[(0, int(local))] = track_id
            track_id += 1

    diag = float(np.hypot(*frame_masks[0].shape))

    for t in range(1, len(frame_masks)):
        ids_prev, ids_curr, iou = compute_iou_matrix(frame_masks[t - 1], frame_masks[t])

        matched_curr = set()
        if ids_prev and ids_curr:
            dist = centroid_distance_matrix(ids_prev, ids_curr, props[t - 1], props[t])
            # Any mask overlap (cost in [0,1)) always beats no overlap at all
            # (cost >= 1, ranked by how close the centroids are).
            cost = np.where(iou > 0, 1.0 - iou, 1.0 + dist / diag)
            # Optimal one-to-one assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            for ia, ib in zip(row_ind, col_ind):
                pid, cid = int(ids_prev[ia]), int(ids_curr[ib])
                if iou[ia, ib] > 0:
                    accepted = iou[ia, ib] >= iou_threshold
                else:
                    accepted = dist[ia, ib] <= max_centroid_distance
                if accepted:
                    id_map[(t, cid)] = id_map[(t - 1, pid)]
                    matched_curr.add(cid)

        # Not linked: new track
        for cid in ids_curr:
            if int(cid) not in matched_curr:
                id_map[(t, int(cid))] = track_id
                track_id += 1
        print(f't={t}: {len(matched_curr)}/{len(ids_curr)} cells linked')

    return id_map


def main(rootpath, exp_path, condition):
    data_path_live = f'{rootpath}{exp_path}Live/{condition}/'
    path_output = data_path_live + 'track/'
    os.makedirs(path_output, exist_ok=True)

    frame_masks = [load_global_mask(rootpath, exp_path, condition, t) for t in range(N_FRAMES)]
    props = [frame_props(m) for m in frame_masks]

    id_map = link_across_time(frame_masks, props, iou_threshold=IOU_THRESHOLD, \
                              max_centroid_distance=MAX_CENTROID_DISTANCE)

    # One row per track and time frame
    rows = [{'unique_id': unique_id, 't': t, 'global_id': global_id, \
             'centroid_x': props[t][global_id][1], 'centroid_y': props[t][global_id][0], \
             'area': props[t][global_id][2]} for (t, global_id), unique_id in id_map.items()]
    tracks = pd.DataFrame(rows).sort_values(['unique_id', 't'])

    # Remove the tracks present in fewer than MIN_FRAMES time frames
    n_frames = tracks.groupby('unique_id')['t'].transform('count')
    tracks = tracks[n_frames >= MIN_FRAMES]

    tracks.to_csv(path_output + 'tracks.csv', index=False)
    n_frames = tracks.groupby('unique_id').size()
    print(f'{len(n_frames)} tracks, {(n_frames == N_FRAMES).sum()} present in all the time frames')
    if tracks['unique_id'].max() > np.iinfo(np.uint16).max:
        print('WARNING: more than 65535 tracks, the track ID do not fit in the 16-bit masks')


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = 2000000000
    main(ROOTPATH, EXP_PATH, CONDITION)
