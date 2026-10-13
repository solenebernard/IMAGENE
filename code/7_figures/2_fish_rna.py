'''
smFISH images (step 2), for a tile of the non-activated condition and a tile of the activated condition
- {condition}_fish_mip_genes: maximum intensity projection of each gene
- {condition}_fish_spots_mip_genes: same with the detected spots, the contour of the cells and the number of spots per cell
- {condition}_fish_segmentation: projection of all the genes and DAPI, with the contour of the cells

Input: FISH tiles, detected spots and tile masks of step 2
'''

from tools.figures import *

params = FIG['fish_rna']
list_quantiles_max = params['quantiles_max']
n_genes = len(list_quantiles_max)
genes = GENES[:n_genes]
n_cols = 4
n_rows = int(np.ceil(n_genes/n_cols))

# The two conditions: name, tile, crop, color
conditions = [(FIG['condition_control'], params['tile_control'], params['crop_control'], 'red'),
              (FIG['conditions_activated'][0], params['tile_activated'], params['crop_activated'], 'yellow')]


def load_crop(condition, s, crop, t, ch):
    '''Crop of the projection and spots of the crop (z,x,y), for the round t and the channel ch'''
    x0, y0, size = crop
    mip, spots = load_fish_tile(condition, s, t, ch)
    spots = spots - np.array([0, x0, y0])
    spots = spots[(spots[:,1]>=0)&(spots[:,1]<size)&(spots[:,2]>=0)&(spots[:,2]<size)]
    return mip[x0:x0+size, y0:y0+size], spots


# Projection of each gene, without and with the spots and the contour of the cells
masks = [load_fish_tile_mask(condition, s)[crop[0]:crop[0]+crop[2], crop[1]:crop[1]+crop[2]] \
         for condition, s, crop, _ in conditions]
figs = [plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*6)) for _ in conditions]
figs_spots = [plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*6)) for _ in conditions]
for i in range(n_genes):
    t, ch = i//2, i%2
    crops = [load_crop(condition, s, crop, t, ch) for condition, s, crop, _ in conditions]
    # Same contrast for the two conditions
    all_pixels = np.concatenate([mip.flatten() for mip, _ in crops])
    vmax = np.quantile(all_pixels, list_quantiles_max[i])
    for (mip, spots), mask, (_, axes), (_, axes_spots), (_, _, _, color) in zip(crops, masks, figs, figs_spots, conditions):
        ax, ax_spots = axes.flatten()[i], axes_spots.flatten()[i]
        ax.imshow(mip, vmin=np.quantile(all_pixels, 0.0), vmax=vmax, cmap='gray')
        ax_spots.imshow(mip, vmin=np.quantile(all_pixels, 0.01), vmax=vmax, cmap='gray')
        ax_spots.scatter(spots[:,2], spots[:,1], s=6, c=color)
        for id in np.unique(mask[mask>0]):
            if plot_contour(ax_spots, mask==id, alpha=1, linewidth=3, c=color):
                x, y = np.where(mask==id)
                count = np.sum(mask[spots[:,1], spots[:,2]]==id)
                ax_spots.text(y.mean(), x.max()+35, count, c=color, fontsize=30)
        for a in [ax, ax_spots]:
            a.set_title(genes[i], fontdict={'fontsize':30})
for (fig, axes), (fig_spots, axes_spots), (condition, _, _, _) in zip(figs, figs_spots, conditions):
    for ax in np.concatenate((axes.flatten(), axes_spots.flatten())):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig_spots.tight_layout()
    save_figure(fig, f'fish_rna/{condition}_fish_mip_genes')
    save_figure(fig_spots, f'fish_rna/{condition}_fish_spots_mip_genes')


# Segmentation of the FISH cells: projection of all the genes (green) and DAPI (blue)
def normalize(image):
    image = np.clip(image, a_min=np.quantile(image, 0.001), a_max=np.quantile(image, 0.999))
    return (image-image.min())/(image.max()-image.min())

condition, s = conditions[1][:2]
H, W = params['segmentation_crop']
mip_dapi = normalize(load_fish_dapi_mip(condition, s)[:H,:W])
mip_total = np.max([load_fish_mip(condition, s, t, ch)[:H,:W] \
                    for t in range(params['segmentation_rounds']) for ch in range(FISH_N_CHANNELS)], axis=0)
mip_total = normalize(mip_total)
rgb_image = np.stack((np.zeros_like(mip_total), mip_total, mip_dapi), axis=-1)

mask = load_fish_tile_mask(condition, s)[:H,:W]
fig, axes = plt.subplots(1, 2, figsize=(8,4))
for id in np.unique(mask[mask>0]):
    plot_contour(axes[1], mask==id, alpha=1, linewidth=1, c='red')
for ax in axes:
    ax.imshow(rgb_image)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
save_figure(fig, f'fish_rna/{condition}_fish_segmentation')
