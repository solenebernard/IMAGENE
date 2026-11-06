'''
Matching between the live (BF) and the smFISH cells (step 3): translation between the two fields of view, pairs of
reciprocal closest neighbours, quality control of the pairs, overlay of the two masks and illustration of the matching
'''


def passes_ratio_qc(distance, distance_2nd_1, distance_2nd_2, ratio_qc):
    '''
    A pair is kept if it is much closer than the second closest candidate, on both sides:
        1 - distance / distance to the second closest cell >= ratio_qc
    ratio_qc is in [0, 1], larger = stricter.
    distance: distance between the two cells of the pair
    distance_2nd_1, distance_2nd_2: distance to the second closest cell, from each side of the pair
    Returns a boolean (False if a distance is missing)
    '''
    return (1 - distance / distance_2nd_1 >= ratio_qc) & (1 - distance / distance_2nd_2 >= ratio_qc)


def passes_ratio_qc_table(df, ratio_qc):
    '''Same for a table with the columns of the alignment (step 3)'''
    return passes_ratio_qc(df['Distance'], df['Distance_2nd_bf_to_fish'], df['Distance_2nd_fish_to_bf'], ratio_qc)


def build_overlay(bf_mask, fish_mask, x_trans, y_trans):
    '''
    Overlay of the two segmentations on the whole BF field of view. Dimension: (x, y, 3)
    First channel: BF mask (not cropped, not translated). Second channel: FISH mask, placed on the BF field of
    view with the alignment (the part outside the BF field of view is cut). Third channel: empty.
    (x_trans, y_trans): translation added to the BF centroids (x: column, y: row) to align them with the FISH cells
    '''
    import numpy as np
    # Position of the FISH image in the BF image
    x0, y0 = -int(round(x_trans)), -int(round(y_trans))
    height, width = bf_mask.shape
    fish_height, fish_width = fish_mask.shape
    y_min, y_max = max(y0, 0), min(y0 + fish_height, height)
    x_min, x_max = max(x0, 0), min(x0 + fish_width, width)
    fish_aligned = np.zeros((height, width), dtype=np.int32)
    fish_aligned[y_min:y_max, x_min:x_max] = fish_mask[y_min - y0:y_max - y0, x_min - x0:x_max - x0]
    print(f'Overlay: FISH field of view at row {y0}, column {x0} of the BF field of view')
    return np.stack((bf_mask.astype(np.int32), fish_aligned, np.zeros((height, width), dtype=np.int32)), axis=-1)


def mask_centroids(mask):
    '''Centroid (row, column) of each label of the mask: {label: (row, column)}'''
    import numpy as np
    x_pos, y_pos = np.where(mask > 0)
    values = mask[x_pos, y_pos].astype(np.int64)
    areas = np.bincount(values)
    labels = np.where(areas > 0)[0]
    rows, cols = np.bincount(values, weights=x_pos)[labels] / areas[labels], np.bincount(values, weights=y_pos)[labels] / areas[labels]
    return {int(l): (r, c) for l, r, c in zip(labels, rows, cols)}


def save_matching_illustration(overlay, alignment, ratio_qc, path, title='', downsample=4):
    '''
    Illustration of the matching, saved as png:
    - left: global alignment. BF cells in red, FISH cells in green, and a blue line between the two cells of each pair
    - right: pairs kept after the quality control (ratio_qc). The cells of the pairs kept are bright, the others dark
    overlay: output of build_overlay. alignment: table of the pairs (step 3)
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    centroids_bf, centroids_fish = mask_centroids(overlay[:, :, 0]), mask_centroids(overlay[:, :, 1])
    # The pairs with their two cells in the overlay
    pairs = alignment[alignment['BF'].isin(list(centroids_bf)) & alignment['Fish'].isin(list(centroids_fish))]
    pairs_qc = pairs[passes_ratio_qc_table(pairs, ratio_qc)]

    # Smaller image for the figure
    small = overlay[::downsample, ::downsample]
    rgb_all = np.zeros(small.shape, dtype=np.uint8)
    rgb_all[small > 0] = 255
    rgb_qc = np.zeros(small.shape, dtype=np.uint8)
    rgb_qc[small > 0] = 100
    rgb_qc[:, :, 0][np.isin(small[:, :, 0], pairs_qc['BF'])] = 255
    rgb_qc[:, :, 1][np.isin(small[:, :, 1], pairs_qc['Fish'])] = 255

    def plot_pairs(ax, image, table):
        ax.imshow(image, origin='lower')
        c_bf = np.array([centroids_bf[i] for i in table['BF']]).reshape(-1, 2) / downsample
        c_fish = np.array([centroids_fish[i] for i in table['Fish']]).reshape(-1, 2) / downsample
        ax.plot([c_bf[:, 1], c_fish[:, 1]], [c_bf[:, 0], c_fish[:, 0]], c='deepskyblue', linewidth=1, marker='.', ms=1)
        ax.set_xticks([])
        ax.set_yticks([])

    n_bf, n_fish = len(centroids_bf), len(centroids_fish)
    fig, axes = plt.subplots(1, 2, figsize=(24, 13))
    plot_pairs(axes[0], rgb_all, pairs)
    axes[0].set_title(f'Global alignment: {n_bf} BF cells (red), {n_fish} FISH cells (green), {len(pairs)} pairs\n'
                      f'Pairs: {len(pairs)/max(n_bf, 1)*100:.1f} % of BF cells, {len(pairs)/max(n_fish, 1)*100:.1f} % of FISH cells', fontsize=16)
    plot_pairs(axes[1], rgb_qc, pairs_qc)
    axes[1].set_title(f'Pairs kept after quality control (ratio_qc = {ratio_qc})\n'
                      f'{len(pairs_qc)} pairs, {len(pairs_qc)/max(len(pairs), 1)*100:.1f} % of the pairs', fontsize=16)
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'Saved {path}')


##### MATCHING OF THE BF AND FISH CELLS (step 3)

def compute_distance(p0, p1):
    '''
    For two sets of points p0 and p1
    Compute the mean of the distances between each point of p0
    to its closest neighbour in p1
    '''
    import numpy as np
    from scipy.spatial.distance import cdist
    mat_dist = cdist(p0, p1)
    arg_min_dist = np.argmin(mat_dist, axis=1)
    min_dist = np.min(mat_dist, axis=1)
    return (np.mean(min_dist), arg_min_dist)


def brute_force(p0, p1, dx_trans_min, dy_trans_min, dx_trans_max, dy_trans_max, n):
    '''
    Brute force: test all translations between two extreme translation for each axis
    with a granularity n
    '''
    import numpy as np
    # Get all possible translation for each axis
    dx_trans = np.linspace(dx_trans_min, dx_trans_max, n)
    dy_trans = np.linspace(dy_trans_min, dy_trans_max, n)

    error_mat = np.zeros([n, n])
    for i, x_trans in enumerate(dx_trans):
        for j, y_trans in enumerate(dy_trans):
            p1_trans = p1 + np.array([x_trans, y_trans])

            error, _ = compute_distance(p0, p1_trans)
            error_mat[i, j] = error

    # Get the translation that minimizes the distance
    i, j = np.argmin(error_mat) // n, np.argmin(error_mat) % n
    x_trans, y_trans = dx_trans[i], dy_trans[j]

    return (x_trans, y_trans, error_mat)


def get_translation(p0, p1, n=40, delta=1500, delta_refined=300, plot=False):
    '''
    Translation to add to p1 to align it with p0: a first search by brute force on a large range of translations,
    then a refined search around the best translation of the first search.
    plot: show the error of each translation tested (both searches)
    Returns the translation (x, y) and the errors of the first search
    '''
    import matplotlib.pyplot as plt

    p0_xmin, p0_xmax, p0_ymin, p0_ymax = p0[:, 0].min(), p0[:, 0].max(), \
    p0[:, 1].min(), p0[:, 1].max()

    p1_xmin, p1_xmax, p1_ymin, p1_ymax = p1[:, 0].min(), p1[:, 0].max(), \
        p1[:, 1].min(), p1[:, 1].max()

    dx_trans_max = (p0_xmax - p0_xmin) - (p1_xmax - p1_xmin) + delta
    dy_trans_max = (p0_ymax - p0_ymin) - (p1_ymax - p1_ymin) + delta
    dx_trans_min = - delta
    dy_trans_min = - delta

    x_trans, y_trans, error_mat = brute_force(p0, p1, dx_trans_min, dy_trans_min, dx_trans_max, dy_trans_max, n)
    if plot:
        plt.imshow((error_mat.T), origin='lower')
        plt.colorbar()
        plt.show()

    # Add one leval of dichotomy
    dx_trans_max = x_trans + delta_refined
    dy_trans_max = y_trans + delta_refined
    dx_trans_min = x_trans - delta_refined
    dy_trans_min = y_trans - delta_refined

    x_trans, y_trans, error_mat2 = brute_force(p0, p1, dx_trans_min, dy_trans_min, dx_trans_max, dy_trans_max, n)
    if plot:
        plt.imshow((error_mat2.T), origin='lower')
        plt.colorbar()
        plt.show()

    return (x_trans, y_trans, error_mat)


def match_closest_neighbour(p0, p1, max_dist, plot=False):
    '''
    Reciprocal closest neighbours between the points p0 and p1, closer than max_dist,
    and the second closest neighbour of each matched point (not reciprocal).
    plot: show the matching
    Returns the matrix of distances, the matrix of matching (1: matched),
    and the matrices of the second closest neighbours from p0 to p1 and from p1 to p0 (2: second closest)
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist

    # Get closest neighbour
    mat_dist = cdist(p0, p1)

    # Initialize matrix of matching from p0 to p1
    mat_match0 = np.zeros_like(mat_dist)
    arg_min_dist0 = np.argsort(mat_dist, axis=1)[:, 0]
    mat_match0[np.arange(len(p0)), arg_min_dist0] = 1
    # Set threshold of maximum distance of max_dist
    min_dist0 = np.min(mat_dist, axis=1)
    mat_match0[np.arange(len(p0))[min_dist0 > max_dist], arg_min_dist0[min_dist0 > max_dist]] = 0

    # Initialize matrix of matching from p1 to p0
    mat_match1 = np.zeros_like(mat_dist)
    arg_min_dist1 = np.argsort(mat_dist, axis=0)[0]
    mat_match1[arg_min_dist1, np.arange(len(p1))] = 1
    # Set threshold of maximum distance of max_dist
    min_dist1 = np.min(mat_dist, axis=0)
    mat_match1[arg_min_dist1[min_dist1 > max_dist], np.arange(len(p1))[min_dist1 > max_dist]] = 0

    # Total matching matrix is the multiplication of both: it's reciprocal
    mat_match = mat_match0 * mat_match1

    # Now: second closest neighbour
    # It's not reciprocal
    mat_match_scd_0 = np.zeros_like(mat_match)
    mat_match_scd_1 = np.zeros_like(mat_match)

    # Each matched point of p0 and p1 is linked to its closest neighbour
    ind_matched0 = np.where(mat_match == 1)[0]
    ind_matched1 = np.where(mat_match == 1)[1]

    arg_2min_dist0 = np.argsort(mat_dist, axis=1)[:, 1]
    mat_match_scd_0[ind_matched0, arg_2min_dist0[ind_matched0]] = 2

    arg_2min_dist1 = np.argsort(mat_dist, axis=0)[1]
    mat_match_scd_1[arg_2min_dist1[ind_matched1], ind_matched1] = 2

    if plot:
        # Vizualize matching
        plt.figure(figsize=(20, 20))
        plt.scatter(p0[:, 0], p0[:, 1], s=3, c='green', marker='.')
        plt.scatter(p1[:, 0], p1[:, 1], s=3, c='red', marker='.')
        ind_p0, ind_p1 = np.where(mat_match == 1)
        plt.scatter(p0[ind_p0][:, 0], p0[ind_p0][:, 1], s=8, c='green')
        plt.scatter(p1[ind_p1][:, 0], p1[ind_p1][:, 1], s=8, c='red')
        plt.plot([p0[ind_p0][:, 0], p1[ind_p1][:, 0]], [p0[ind_p0][:, 1], p1[ind_p1][:, 1]], c='blue', linewidth=0.5)

        ind_p0, ind_p1 = np.where(mat_match_scd_0 == 2)
        plt.plot([p0[ind_p0][:, 0], p1[ind_p1][:, 0]], [p0[ind_p0][:, 1], p1[ind_p1][:, 1]], c='green', linewidth=0.5, alpha=0.5, linestyle='--')
        ind_p0, ind_p1 = np.where(mat_match_scd_1 == 2)
        plt.plot([p0[ind_p0][:, 0], p1[ind_p1][:, 0]], [p0[ind_p0][:, 1], p1[ind_p1][:, 1]], c='red', linewidth=0.5, alpha=0.5, linestyle='--')
        plt.show()

    return (mat_dist, mat_match, mat_match_scd_0, mat_match_scd_1)


def match_cells(mask_values_fish, centroids_fish, mask_values_bf, centroids_bf, max_dist, plot=False):
    '''
    Table of the pairs of BF and FISH cells (reciprocal closest neighbours, see match_closest_neighbour),
    with the distance of each pair and the distance to the second closest cell from each side.
    centroids_bf must already be translated to align with the FISH cells.
    plot: show the matching
    '''
    import numpy as np
    import pandas as pd

    mat_dist, mat_match, mat_match_scd_fish, mat_match_scd_bf = match_closest_neighbour(centroids_fish, centroids_bf, max_dist, plot=plot)

    # Closest neighbour
    ind_p0, ind_p1 = np.where(mat_match == 1)
    matched_fish_id, matched_bf_id = mask_values_fish[ind_p0], mask_values_bf[ind_p1]
    matched_dist = mat_dist[ind_p0, ind_p1]
    matched_fish_centroid, matched_bf_centroid = centroids_fish[ind_p0], centroids_bf[ind_p1]

    # Second closest neigbour from p0 to p1
    ind2_p1 = np.asarray([np.where(mat_match_scd_fish[x] == 2)[0][0] for x in ind_p0])  # keep the order
    matched_bf_id_scd = mask_values_bf[ind2_p1]
    matched_dist_2_fish = mat_dist[ind_p0, ind2_p1]

    # Second closest neigbour from p1 to p0
    ind2_p0 = np.asarray([np.where(mat_match_scd_bf[:, x] == 2)[0][0] for x in ind_p1])  # keep the order
    matched_fish_id_scd = mask_values_fish[ind2_p0]
    matched_dist_2_bf = mat_dist[ind2_p0, ind_p1]

    return pd.DataFrame({"BF": matched_bf_id, "Fish": matched_fish_id,
                        "BF_center_x": matched_bf_centroid[:, 0], "BF_center_y": matched_bf_centroid[:, 1],
                         "Fish_center_x": matched_fish_centroid[:, 0], "Fish_center_y": matched_fish_centroid[:, 1],
                         "Distance": matched_dist,
                         'Distance_2nd_bf_to_fish': matched_dist_2_bf, 'Distance_2nd_fish_to_bf': matched_dist_2_fish,
                         'Id_2nd_bf_to_fish': matched_fish_id_scd, 'Id_2nd_fish_to_bf': matched_bf_id_scd})
