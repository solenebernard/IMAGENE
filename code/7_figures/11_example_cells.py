'''
Two example cells: one with a high and one with a low expression of a gene
- example_video_cell_{id}: BF images of the cell over time, with its contour
- example_umap: position of the two cells in the UMAP of the features of the live cells
- example_violin_plot: distribution of the features over all the cells, with the values of the two cells

Input: 6_results/dataset.csv, merged BF images, global masks and tracks of step 1
'''

from tools.figures import *

params = FIG['example_cells']
condition = FIG_CONDITION
gene = params['gene']
z = params['z']
dx, dy = params['crop']
list_colors = ['red', 'blue']

df = load_dataset()
df = add_umap_features(df, select_features(df))
# The cells of the condition, indexed by the ID of the BF cell
df_condition = df.xs(condition, level='condition').droplevel('exp_path')

list_cells = params['cells']
if list_cells is None:
    # The CD8 positive cells with the highest and the lowest expression of the gene
    expression = df_condition.loc[df_condition['Gene_CD8']>1, f'Gene_{gene}'].dropna().sort_values()
    list_cells = [expression.index[-1], expression.index[0]]
print(df_condition.loc[list_cells, [f'Gene_{g}' for g in LEARNING_GENES]])


##### BF IMAGES OF THE CELLS OVER TIME
t_range = np.unique(np.concatenate((np.arange(N_FRAMES)[::3], [N_FRAMES-1])))
masks = np.asarray([load_merged_mask(condition, t) for t in t_range])
# Crop centered on the cell (center of its bounding box over time)
windows = []
for id_cell in list_cells:
    _, x, y = np.where(masks == id_cell)
    xmean, ymean = (x.min()+x.max()+1)//2, (y.min()+y.max()+1)//2
    windows.append((slice(max(xmean-dx//2, 0), xmean+dx//2), slice(max(ymean-dy//2, 0), ymean+dy//2)))
# Load the merged images one by one and keep only the crops
crops = [[] for _ in list_cells]
for t in t_range:
    image = load_merged_bf(condition, t, z)
    for i, window in enumerate(windows):
        crops[i].append(image[window])

for id_cell, color, window, images in zip(list_cells, list_colors, windows, crops):
    images = np.asarray(images)
    vmin, vmax = np.quantile(images, 0.001), np.quantile(images, 0.999)
    fig, axes = plt.subplots(1, len(t_range), figsize=(8,1.8))
    for ax, t, image, mask in zip(axes, t_range, images, masks):
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        plot_contour(ax, mask[window]==id_cell, alpha=1, linewidth=0.7, c=color)
        ax.set_title(f't={t}')
        ax.axis('off')
    fig.tight_layout()
    save_figure(fig, f'model/example_video_cell_{id_cell}', formats=('png',), transparent=True)


##### POSITION IN THE UMAP
fig, ax = plt.subplots(1, 1, figsize=(4.5,3))
ax.scatter(df['umap_0'], df['umap_1'], s=0.5, c='black')
for id_cell, color in zip(list_cells, list_colors):
    ax.scatter(df_condition.loc[id_cell, 'umap_0'], df_condition.loc[id_cell, 'umap_1'], c=color, s=30)
ax.axis('off')
save_figure(fig, 'model/example_umap', formats=('png',), transparent=True)


##### VIOLIN PLOT OF THE FEATURES
plot_features = [x for x in params['features'] + ['n_neigh_thresh_100_mean', 'n_neigh_thresh_200_mean'] if x in df.columns]
plot_features_name = [x.replace('n_neigh_thresh_', 'n_neigh_').replace('_mean', '') if x.startswith('n_neigh') else x \
                      for x in plot_features]

# Normalize each feature between 0 and 1 over all the cells
data = df[plot_features]
data = (data - data.min()) / (data.max() - data.min() + 1e-9)
data = data.xs(condition, level='condition').droplevel('exp_path')
positions = np.arange(len(plot_features))

fig, ax = plt.subplots(figsize=(11,2.5), facecolor="white")
violins = ax.violinplot([data[col].dropna().values for col in plot_features], positions=positions, widths=0.8, \
                        showmeans=False, showmedians=False, showextrema=False)
for body in violins["bodies"]:
    body.set_facecolor("#457B9D")
    body.set_alpha(0.5)
    body.set_edgecolor("none")
# All the cells, with a jitter
rng = np.random.default_rng(0)
for i, col in enumerate(plot_features):
    y = data[col].values
    ax.scatter(rng.normal(i, 0.08, size=len(y)), y, s=10, color="black", alpha=0.1, linewidths=0, zorder=1)
# The two cells
for id_cell, color in zip(list_cells, list_colors):
    ax.scatter(positions, data.loc[id_cell].values, s=80, color=color, edgecolor="white", linewidth=1.2, zorder=3)
ax.set_xticks(positions)
ax.set_xticklabels(plot_features_name, rotation=30, ha="right", fontsize=11)
ax.set_ylim(0, 1)
ax.set_ylabel("Normalized value", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.15)
fig.tight_layout()
save_figure(fig, 'model/example_violin_plot', formats=('png',), bbox_inches="tight", transparent=True)
