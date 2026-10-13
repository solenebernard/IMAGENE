'''
Matching of the live cells with the smFISH cells (step 3)
- matching: segmentation of the last BF frame, smFISH segmentation, and pairs kept after the quality control
- alignment: mean distance to the closest cell as a function of the translation, and overlay of the two segmentations
- influence_ratio_qc: number of pairs as a function of the quality control ratio
- cell_counts.csv: number of cells at each step of the matching

Quality control: a pair is kept if 1 - distance / distance to the second closest cell >= ratio_qc, on both sides

Input: overlay and table of step 3 (alignment/), 6_results/dataset.csv
'''

from scipy.spatial.distance import cdist

from tools.figures import *

params = FIG['matching']
condition = FIG_CONDITION

# First channel: BF mask after alignment. Second channel: FISH mask
overlay = load_overlay(condition)
alignment = load_alignment(condition)
bf_mask, fish_mask = overlay[:,:,0], overlay[:,:,1]


def centroids(mask):
    '''Centroid (row, column) of each label of the mask: Series indexed by the label'''
    x_pos, y_pos = np.where(mask>0)
    values = mask[x_pos, y_pos].astype(np.int64)
    areas = np.bincount(values)
    ids = np.where(areas>0)[0]
    coords = np.stack((np.bincount(values, weights=x_pos)[ids], np.bincount(values, weights=y_pos)[ids]), axis=1)
    return pd.DataFrame(coords/areas[ids, None], index=ids, columns=['x', 'y'])

centroids_bf, centroids_fish = centroids(bf_mask), centroids(fish_mask)


##### PAIRS KEPT AFTER THE QUALITY CONTROL
pairs = alignment[passes_ratio_qc_table(alignment, RATIO_QC)]
pairs = pairs[pairs['BF'].isin(centroids_bf.index)&pairs['Fish'].isin(centroids_fish.index)]

# All the cells in dark, the matched cells in bright
overlay_plot = np.zeros(overlay.shape, dtype=np.uint8)
overlay_plot[overlay>0] = 128
overlay_plot[:,:,0][np.isin(bf_mask, pairs['BF'])] = 255
overlay_plot[:,:,1][np.isin(fish_mask, pairs['Fish'])] = 255
centroid_bf_plot = centroids_bf.loc[pairs['BF']].values
centroid_fish_plot = centroids_fish.loc[pairs['Fish']].values

rgb_bf = np.zeros(overlay.shape, dtype=np.uint8)
rgb_bf[:,:,0] = (bf_mask>0)*255
rgb_fish = np.zeros(overlay.shape, dtype=np.uint8)
rgb_fish[:,:,1] = (fish_mask>0)*255

fig, axes = plt.subplots(1, 3, figsize=(12,4))
axes[0].imshow(rgb_bf, origin='lower')
axes[1].imshow(rgb_fish, origin='lower')
axes[2].imshow(overlay_plot, origin='lower')
axes[2].plot([centroid_bf_plot[:,1], centroid_fish_plot[:,1]], \
             [centroid_bf_plot[:,0], centroid_fish_plot[:,0]], c='blue', linewidth=1, marker='.', ms=3)
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, params['window'])
    ax.set_ylim(0, params['window'])
axes[0].set_title('Last frame of live cell \n timelapse segmentation')
axes[1].set_title('smFISH segmentation')
axes[2].set_title(f'Pairwise matching after alignment \n and quality control (ratio = {RATIO_QC})')
fig.tight_layout()
save_figure(fig, 'matching/matching')


##### ALIGNMENT
# Mean distance between each FISH cell and its closest BF cell, for all the translations of the BF cells
# around the alignment found in step 3 (center of the map)
n, delta = params['translation_n'], params['translation_delta']
translations = np.linspace(-delta, delta, n)
error_mat = np.zeros((n, n))
for i, dx in enumerate(translations):
    for j, dy in enumerate(translations):
        mat_dist = cdist(centroids_fish.values, centroids_bf.values + np.array([dx, dy]))
        error_mat[i, j] = np.mean(np.min(mat_dist, axis=1))

fig, axes = plt.subplots(1, 2, figsize=(7,4))
axes[0].imshow(error_mat, cmap='plasma')
axes[1].imshow((overlay[::10,::10]>0).astype(float))
for ax in axes:
    ax.axis('off')
axes[0].set_title('Find global minimum of distance \n for alignment of fields of view')
axes[1].set_title('Overlay of the two segmentations \n after alignment')
fig.tight_layout()
save_figure(fig, 'matching/alignment')


##### INFLUENCE OF THE QUALITY CONTROL RATIO
df = pd.read_csv('6_results/dataset.csv', index_col=['exp_path', 'condition', 'cell_bf'])
bool_keep = (df['is_on_edge']==False)&(df['count']==N_FRAMES)
list_ratio_qc = np.linspace(0, 1, 101)
n_pairs = [(passes_ratio_qc_table(df, r)&bool_keep).sum() for r in list_ratio_qc]

fig, ax = plt.subplots(1, 1, figsize=(3,2.5))
ax.plot(list_ratio_qc, n_pairs, linewidth=2)
ax.axvline(RATIO_QC, c='black', linewidth=1, linestyle='--')
ax.set_xlabel('Quality control ratio')
ax.set_ylabel('Number of pairs')
fig.tight_layout()
save_figure(fig, 'matching/influence_ratio_qc')


##### NUMBER OF CELLS AT EACH STEP
df_condition = df.xs(condition, level='condition')
bool_keep = df_condition['Fish'].notna()&(df_condition['is_on_edge']==False)&(df_condition['count']==N_FRAMES)
bool_qc = passes_ratio_qc_table(df_condition, RATIO_QC)
counts = {
    'BF cells in the overlay': len(centroids_bf),
    'FISH cells': len(centroids_fish),
    'Pairs': len(alignment),
    'Pairs not on the edge and in all the time frames': bool_keep.sum(),
    f'Pairs with also quality control ratio >= {RATIO_QC}': (bool_qc&bool_keep).sum(),
    'Pairs with also CD8 positive': (bool_qc&bool_keep&(df_condition['Gene_CD8']>1)).sum(),
}
counts = pd.Series(counts, name=condition)
path_counts = pathlib.Path(FIG['output_dir']) / 'matching/cell_counts.csv'
counts.to_csv(path_counts, header=True)
print(counts)
