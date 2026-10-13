'''
Detection of the clumps of fluorescence in the smFISH images (steps 2.4 and 2.5)
- remove_clumps: projection of a gene, and segmentation of the cells and of the clumps.
  The cells overlapping a clump (red) are removed for this gene, the others (yellow) are kept.

Input: FISH tile and tile mask of step 2, thresholds of the clumps of step 2.4
'''

from scipy import ndimage
from skimage.measure import label

from tools.figures import *

params = FIG['clumps']
condition = FIG_CONDITION
s, t, ch, size = params['tile'], params['round'], params['channel'], params['crop']
gene = GENES[2*t + ch]


def remove_small_regions(mask, min_size):
    '''Set to 0 any connected region of the mask with fewer than min_size pixels'''
    labeled, _ = ndimage.label(mask)
    sizes = np.bincount(labeled.ravel())
    cleaned = mask.copy()
    cleaned[(sizes < min_size)[labeled]] = 0
    return cleaned


thresh = load_clumps_thresholds(condition)
# Same threshold for all the tiles: the maximum of the selected thresholds
thresh = thresh.loc[:, t, ch].max().values[0]

mip = load_fish_mip(condition, s, t, ch)[:size, :size]
mask = load_fish_tile_mask(condition, s)[:size, :size]
mask_clumps = remove_small_regions(label(mip>thresh, connectivity=1), CLUMPS_MIN_SIZE)

fig, axes = plt.subplots(1, 2, figsize=(16,9))
for ax in axes:
    ax.imshow(mip, vmin=np.quantile(mip, 0.001), vmax=np.quantile(mip, 0.998), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
for id in np.unique(mask[mask>0]):
    is_on_clump = np.any(mask_clumps[mask==id]>0)
    plot_contour(axes[1], mask==id, alpha=1, linewidth=3, c='red' if is_on_clump else 'yellow')
for id in np.unique(mask_clumps[mask_clumps>0]):
    plot_contour(axes[1], mask_clumps==id, alpha=1, linewidth=3, c='blue')
axes[0].set_title(f'Maximum intensity projection of {gene}', fontdict={'fontsize':30})
axes[1].set_title('Segmentation of cells and clumps', fontdict={'fontsize':30})
fig.tight_layout()
save_figure(fig, 'clumps/remove_clumps')
