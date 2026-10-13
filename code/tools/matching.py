'''
Quality control of the matching between the live (BF) and the FISH cells
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
    y_min, y_max = max(y0, 0), min(y0+fish_height, height)
    x_min, x_max = max(x0, 0), min(x0+fish_width, width)
    fish_aligned = np.zeros((height, width), dtype=np.int32)
    fish_aligned[y_min:y_max, x_min:x_max] = fish_mask[y_min-y0:y_max-y0, x_min-x0:x_max-x0]
    print(f'Overlay: FISH field of view at row {y0}, column {x0} of the BF field of view')
    return np.stack((bf_mask.astype(np.int32), fish_aligned, np.zeros((height, width), dtype=np.int32)), axis=-1)


def mask_centroids(mask):
    '''Centroid (row, column) of each label of the mask: {label: (row, column)}'''
    import numpy as np
    x_pos, y_pos = np.where(mask > 0)
    values = mask[x_pos, y_pos].astype(np.int64)
    areas = np.bincount(values)
    labels = np.where(areas > 0)[0]
    rows, cols = np.bincount(values, weights=x_pos)[labels]/areas[labels], np.bincount(values, weights=y_pos)[labels]/areas[labels]
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
    pairs = alignment[alignment['BF'].isin(list(centroids_bf))&alignment['Fish'].isin(list(centroids_fish))]
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
        c_bf = np.array([centroids_bf[i] for i in table['BF']]).reshape(-1, 2)/downsample
        c_fish = np.array([centroids_fish[i] for i in table['Fish']]).reshape(-1, 2)/downsample
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
