'''
UMAP of the cells (step 5)
- umap_color_features: UMAP of the features of the live cells, colored by four features
- umap_color_genes: same UMAP, colored by the expression of each gene
- umap_thumbnails: same UMAP, each cell shown by its first BF frame
- umap_genes: UMAP of the gene expression of the CD8 positive cells, colored by the expression of each gene
- umap_genes_color_feat: same UMAP, colored by four features of the live cells

Input: 6_results/dataset.csv, single-cell movies of step 1.6 (for umap_thumbnails)
'''

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from tools.figures import *

size_p = 5
df = load_dataset()
df = add_umap_features(df, select_features(df))
df = add_umap_genes(df)
index_cd8 = cd8_positive(df)

# The four features: column, title, limits of the colors
list_feats = [('area_mean', 'Average cell area', 1000, 6000),
              ('avg_speed', 'Average cell speed', None, 35),
              ('vor_area_mean', 'Average voronoi area', None, 130000),
              ('iou_mean', 'Average IoU', 0.2, 0.7)]
list_feats = [x for x in list_feats if x[0] in df.columns]


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')


def plot_genes(col_x, col_y, name):
    '''
    One UMAP per gene, colored by the expression of the gene in the CD8 positive cells.
    Returns the aspect ratio of the panels: length on the figure of one unit of y divided by the length of one unit of x
    '''
    n_cols = 5
    n_rows = int(np.ceil(len(LEARNING_GENES)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2.5))
    for ax in axes.flatten():
        clean_axis(ax)
    for ax, gene in zip(axes.flatten(), LEARNING_GENES):
        y_gene = df.loc[index_cd8, f'Gene_{gene}']
        sc = ax.scatter(df.loc[index_cd8, col_x], df.loc[index_cd8, col_y], c=y_gene, vmax=np.nanpercentile(y_gene, 90), s=size_p)
        ax.set_title(gene)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    # Aspect ratio of the first panel, once the figure is drawn
    fig.canvas.draw()
    box = axes.flatten()[0].get_window_extent()
    x_min, x_max = axes.flatten()[0].get_xlim()
    y_min, y_max = axes.flatten()[0].get_ylim()
    aspect = (box.height/(y_max-y_min)) / (box.width/(x_max-x_min))
    save_figure(fig, name)
    return aspect


##### UMAP OF THE FEATURES OF THE LIVE CELLS
fig, axes = plt.subplots(2, 2, figsize=(4,5))
for ax in axes.flatten():
    clean_axis(ax)
for ax, (feat, name, vmin, vmax) in zip(axes.flatten(), list_feats):
    sc = ax.scatter(df['umap_0'], df['umap_1'], c=df[feat], s=size_p, vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=ax)
    ax.set_title(name)
fig.tight_layout()
save_figure(fig, 'model/umap_color_features')

aspect_umap = plot_genes('umap_0', 'umap_1', 'model/umap_color_genes')


# Each cell shown by its first BF frame
def show_bf_image(exp_path, condition, id_cell_bf):
    '''First frame of the movie of the cell, as a RGB image. None if the movie is not available'''
    frame_folder = pathlib.Path(ROOTPATH + exp_path + 'Live/' + condition + f'/live_dataset/cell_{id_cell_bf}/')
    frame_paths = sorted(frame_folder.glob('t*.tiff'))
    if len(frame_paths) == 0:
        return None
    img = tifffile.imread(frame_paths[0])/2**16
    first_frame = img - np.median(img)
    vmin, vmax = np.quantile(first_frame, 0.0001), np.quantile(first_frame, 0.9999)
    first_frame = np.round(np.clip((first_frame-vmin)/(vmax-vmin)*2**8, 0, 255)).astype(np.uint8)
    return np.tile(first_frame[:,:,None], (1,1,3))

# Same aspect ratio between the two axes as in umap_color_genes. The figure keeps the same area (30 x 20 inches),
# so that the thumbnails overlap as little as before, and its shape follows from the aspect ratio
height_over_width = aspect_umap * (df['umap_1'].max()-df['umap_1'].min()) / (df['umap_0'].max()-df['umap_0'].min())
width = np.sqrt(30*20/height_over_width)
fig, ax = plt.subplots(figsize=(width, width*height_over_width))
ax.set_aspect(aspect_umap)
n_images = 0
for exp_path, condition, id_cell_bf in df.index:
    img = show_bf_image(exp_path, condition, id_cell_bf)
    if img is None:
        continue
    x, y = df.loc[(exp_path, condition, id_cell_bf), ['umap_0', 'umap_1']]
    ax.add_artist(AnnotationBbox(OffsetImage(img[50:-50,50:-50], zoom=0.5, interpolation=None), (x, y), frameon=False))
    ax.scatter(x, y)
    n_images += 1
ax.axis('off')
fig.tight_layout()
if n_images > 0:
    save_figure(fig, 'model/umap_thumbnails')
else:
    plt.close(fig)
    print('No single-cell movie found (step 1.6): umap_thumbnails is skipped')


##### UMAP OF THE GENE EXPRESSION
plot_genes('umap2_1', 'umap2_0', 'model/umap_genes')

fig, axes = plt.subplots(2, 2, figsize=(4,5))
for ax in axes.flatten():
    clean_axis(ax)
for ax, (feat, name, _, _) in zip(axes.flatten(), list_feats):
    c_feat = df.loc[index_cd8, feat]
    sc = ax.scatter(df.loc[index_cd8, 'umap2_1'], df.loc[index_cd8, 'umap2_0'], c=c_feat, vmax=np.nanpercentile(c_feat, 90), s=size_p)
    fig.colorbar(sc, ax=ax)
    ax.set_title(name)
fig.tight_layout()
save_figure(fig, 'model/umap_genes_color_feat')
