'''
Segmentation and tracking of the live cells on a BF tile (step 1)
- segmentation_bf_all_t_frames: contour of the tracked cells for all the time frames
- segmentation_bf_subset_frames: same for a few time frames
- segmentation_and_prediction: contour colored by the prediction of the model for a gene (needs step 5.3)

Input: BF tiles and tile masks and tracks of step 1, 6_results/prediction_{gene}.csv
'''

from tools.figures import *

params = FIG['segmentation']
x0, y0, size = params['crop']
cmap = matplotlib.colormaps['plasma']

image_bf_tile = load_bf_tiles(FIG_CONDITION, params['tile'])
masks_tile = load_tile_masks(FIG_CONDITION, params['tile'])
image_bf = image_bf_tile[:, x0:x0+size, y0:y0+size]
masks = masks_tile[:, x0:x0+size, y0:y0+size]
unique_id = np.unique(masks[masks>0])
vmin, vmax = np.quantile(image_bf, 0.00), np.quantile(image_bf, 0.999)


def plot_frame(ax, im, ma, linewidth):
    '''BF image with the contour and the ID of each cell'''
    ax.imshow(im, vmin=vmin, vmax=vmax, cmap='gray')
    for i, id in enumerate(unique_id):
        color = cmap(i/(len(unique_id)*2))
        if plot_contour(ax, ma==id, alpha=1, linewidth=linewidth, c=color):
            x, y = np.where(ma==id)
            ax.text(y.mean(), x.min()-10, id, fontsize=20, c=color)
    ax.set_xticks([])
    ax.set_yticks([])


# All the time frames
n_cols = 7
n_rows = int(np.ceil(N_FRAMES/n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*6))
for t, ax in enumerate(axes.flatten()):
    if t < N_FRAMES:
        plot_frame(ax, image_bf[t], masks[t], linewidth=2)
        ax.set_title(f't={t}')
    else:
        ax.axis('off')
fig.tight_layout()
save_figure(fig, 'segmentation/segmentation_bf_all_t_frames')

# Subset of the time frames
list_t = params['frames_subset']
fig, axes = plt.subplots(1, len(list_t), figsize=(len(list_t)*6, 6.5))
for ax, t in zip(axes.flatten(), list_t):
    plot_frame(ax, image_bf[t], masks[t], linewidth=3)
    ax.axis('off')
    ax.set_title(f't={t}', fontdict={'fontsize':30})
fig.tight_layout()
save_figure(fig, 'segmentation/segmentation_bf_subset_frames')


# Contour of the cells with the prediction of the model
# Color: predicted low (blue) or high (red). Line: correct (solid), wrong (dashed) or no ground truth (dotted)
gene = params['prediction_gene']
path_prediction = pathlib.Path(f'6_results/prediction_{gene}.csv')
if path_prediction.exists():
    pred_df = pd.read_csv(path_prediction, index_col=['exp_path', 'condition', 'cell_bf'])
    pred_df = pred_df.xs(FIG_CONDITION, level='condition').droplevel('exp_path')

    # First time frame of a tile inside the field of view of the smFISH image, where the cells have a number of RNA
    image_prediction = load_bf_tiles(FIG_CONDITION, params['prediction_tile'], list_t=[0])[0]
    mask_prediction = load_tile_masks(FIG_CONDITION, params['prediction_tile'], list_t=[0])[0]

    # Part of the tile that is shown
    if params['prediction_window'] is not None:
        xmin, xmax, ymin, ymax = params['prediction_window']
    else:
        # The square where the most cells have a number of RNA, and the fewest have none
        size = params['prediction_window_size']
        cells = pd.DataFrame(measure.regionprops_table(mask_prediction, properties=('label', 'bbox'))).set_index('label')
        cells = cells[cells.index.isin(pred_df.index)]
        has_count = pred_df.loc[cells.index, f'Gene_{gene}'].notna().values
        best_score = -np.inf
        for y0 in range(0, mask_prediction.shape[0]-size+1, 50):
            for x0 in range(0, mask_prediction.shape[1]-size+1, 50):
                # Cells entirely inside the square
                inside = ((cells['bbox-0']>y0)&(cells['bbox-1']>x0)&(cells['bbox-2']<y0+size)&(cells['bbox-3']<x0+size)).values
                score = np.sum(inside&has_count) - np.sum(inside&~has_count)
                if score > best_score:
                    best_score, xmin, ymin = score, x0, y0
        xmax, ymax = xmin+size, ymin+size
        print(f'Part of the tile shown: columns {xmin}-{xmax}, rows {ymin}-{ymax}')
    im, ma = image_prediction[ymin:ymax, xmin:xmax], mask_prediction[ymin:ymax, xmin:xmax]
    vmin, vmax = np.quantile(image_prediction, 0.0001), np.quantile(image_prediction, 0.999)
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.imshow(im, vmin=vmin, vmax=vmax, cmap='gray')
    for id in np.unique(ma[ma>0]):
        if id not in pred_df.index:
            continue
        x, y = np.where(ma==id)
        # Skip the cells cut by the border of the part that is shown
        if x.min()==0 or y.min()==0 or x.max()==ma.shape[0]-1 or y.max()==ma.shape[1]-1:
            continue
        row = pred_df.loc[id]
        pred, true, val = row[f'pred_{gene}'], row[f'Gene_bin_{gene}'], row[f'Gene_{gene}']
        color = 'red' if pred==1 else 'blue'
        if np.isnan(true):
            linestyle = ':'
        elif true==pred:
            linestyle = '-'
        else:
            linestyle = '--'
        plot_contour(ax, ma==id, alpha=1, linewidth=3, c=color, linestyle=linestyle)
        if not np.isnan(val):
            # Number of RNA of the gene, above the cell (below if another cell is just above)
            above = ma[max(x.min()-45, 0):x.min(), max(int(y.mean())-30, 0):int(y.mean())+30]
            if np.any((above>0)&(above!=id)):
                ax.text(y.mean(), x.max()+3, int(val), fontsize=20, c=color, ha='center', va='top', clip_on=True)
            else:
                ax.text(y.mean(), x.min()-3, int(val), fontsize=20, c=color, ha='center', va='bottom', clip_on=True)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    save_figure(fig, 'segmentation/segmentation_and_prediction', formats=('png',))
else:
    print(f'{path_prediction} not found (step 5.3): the figure with the prediction is skipped')
