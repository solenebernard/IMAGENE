'''
Illustration of the features of the live cells (step 4)
- voronoi: Voronoi diagram of the cells of a BF frame, colored by the area of the Voronoi cell
- kneighbours: number of neighbors of a cell within several distances
- overlap_trajectory_cell_{id}: trajectory, path length, total distance and area explored by a cell
- trajectory_cell_{id}: the cell and its contour at each time frame

Input: merged BF image and merged mask of a frame, BF tiles and tile masks and tracks of step 1
'''

import matplotlib.colors as colors
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage.measure import regionprops_table

from tools.figures import *

params = FIG['live_features']
condition = FIG_CONDITION
t, z = params['frame'], params['z']
w_min, w_max = params['window']

# Part of the field of view: the window, and a margin for the cells around
size = w_max + 1000
image_bf = load_merged_bf(condition, t, z)[:size, :size]
mask = load_merged_mask(condition, t)[:size, :size]
vmin, vmax = np.quantile(image_bf, 0.001), np.quantile(image_bf, 0.999)

cells = pd.DataFrame(regionprops_table(mask, properties=('label', 'area', 'centroid'))).set_index('label')
cells = cells[cells['area']>MIN_CELL_AREA]
points = cells[['centroid-0', 'centroid-1']].values


##### VORONOI
vor = Voronoi(points)
# Finite Voronoi cells and their area
polygons = [vor.vertices[region] for region in vor.regions if len(region)>0 and -1 not in region]
areas = np.array([0.5*np.abs(np.dot(p[:,0], np.roll(p[:,1], 1)) - np.dot(p[:,1], np.roll(p[:,0], 1))) for p in polygons])
in_window = np.array([np.all((p>=w_min)&(p<=w_max)) for p in polygons])
norm = colors.Normalize(vmin=areas[in_window].min(), vmax=areas[in_window].max())

# The first axis of the image is the horizontal axis of the figure
fig, ax = plt.subplots(1, 1, figsize=(10,10))
voronoi_plot_2d(vor, ax, point_size=1, show_vertices=False)
ax.imshow(image_bf.T, vmin=vmin, vmax=vmax, cmap='gray')
for id in cells.index:
    plot_contour(ax, mask==id, transpose=True, alpha=1, linewidth=2, c='red')
ax.add_collection(PolyCollection(polygons, facecolor=matplotlib.colormaps['viridis'](norm(areas)), edgecolor='none', alpha=0.1))
ax.set_xlim(w_min, w_max)
ax.set_ylim(w_min, w_max)
ax.axis('off')
fig.tight_layout()
save_figure(fig, 'live_features/voronoi')


##### NUMBER OF NEIGHBORS
center = params['center_cell']
if center is None:
    # The cell of the window with the most neighbors within the largest distance
    from scipy.spatial.distance import cdist
    n_neighbors = pd.Series(np.sum(cdist(points, points)<=params['radii'][-1], axis=1), index=cells.index)
    margin = 500
    n_neighbors[np.any((points<w_min+margin)|(points>w_max-margin), axis=1)] = 0
    center = n_neighbors.idxmax()
centroid_center = cells.loc[center, ['centroid-0', 'centroid-1']].values
distance = pd.Series(np.linalg.norm(points - centroid_center, axis=1), index=cells.index)
radii = params['radii']
list_colors = ['blue', 'mediumblue', 'darkblue']

fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.imshow(image_bf.T, vmin=np.quantile(image_bf, 0.0001), vmax=np.quantile(image_bf, 0.9999), cmap='gray')
plot_contour(ax, mask==center, transpose=True, alpha=1, linewidth=2, c='black')
ax.scatter(points[:,0], points[:,1], c='red')
ax.scatter(centroid_center[0], centroid_center[1], alpha=1, c='black')
radius_prev = 0
for c, radius in zip(list_colors, radii):
    ax.add_patch(patches.Circle(centroid_center, radius, edgecolor=c, facecolor=c, linewidth=2, alpha=0.1))
    n = np.sum(distance<=radius)-1
    ax.text(centroid_center[0]-60, centroid_center[1]+radius+5, f'$r={radius}$ \n $n={n}$', c=c, fontsize=25)
    # The cells between the previous radius and this one
    for id in distance.index[(distance<=radius)&(distance>radius_prev)]:
        if id != center:
            contour = measure.find_contours(mask==id)[0]
            ax.plot(contour[:,0], contour[:,1], c=c)
            ax.fill(contour[:,0], contour[:,1], linewidth=2, alpha=0.4, c=c)
            ax.scatter(*cells.loc[id, ['centroid-0', 'centroid-1']], c=c)
    radius_prev = radius
# The other cells in red
for id in distance.index[distance>radii[-1]]:
    plot_contour(ax, mask==id, transpose=True, alpha=1, linewidth=2, c='red')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(centroid_center[0]-500, centroid_center[0]+500)
ax.set_ylim(centroid_center[1]-500, centroid_center[1]+500)
fig.tight_layout()
save_figure(fig, 'live_features/kneighbours')


##### TRAJECTORY
s, delta = params['trajectory_tile'], params['trajectory_margin']
image_bf = load_bf_tiles(condition, s)
list_masks = load_tile_masks(condition, s)

list_cells = params['trajectory_cells']
if list_cells is None:
    # The cell of the tile, present in all the time frames, with the longest trajectory
    props = pd.concat([pd.DataFrame(regionprops_table(m, properties=('label', 'centroid'))).assign(t=i) \
                       for i, m in enumerate(list_masks)]).set_index(['label', 't'])
    path_lengths = {}
    for id, traj in props.groupby(level='label'):
        if len(traj) == len(list_masks):
            path_lengths[id] = np.linalg.norm(np.diff(traj.values, axis=0), axis=1).sum()
    list_cells = [max(path_lengths, key=path_lengths.get)]

for id_cell in list_cells:
    mask_traj = (list_masks==id_cell)
    _, idx, idy = np.where(mask_traj)
    minx, maxx = max(idx.min()-delta, 0), idx.max()+1+delta
    miny, maxy = max(idy.min()-delta, 0), idy.max()+1+delta
    images = image_bf[:, minx:maxx, miny:maxy]
    mask_traj = mask_traj[:, minx:maxx, miny:maxy]
    ratio = images.shape[1]/images.shape[2]

    # Trajectory of the centroid
    centroids = np.array([np.stack(np.where(m)).mean(axis=1) for m in mask_traj if m.any()])
    total_distance = np.linalg.norm(centroids[-1]-centroids[0])
    path_length = np.linalg.norm(np.diff(centroids, axis=0), axis=1).sum()
    # Total explored area
    total_mask = mask_traj.sum(axis=0)>0

    fig, ax = plt.subplots(1, 1, figsize=(9,9*ratio))
    ax.imshow(mask_traj.sum(axis=0), cmap='gray')
    ax.plot(centroids[:,1], centroids[:,0], color='#00BFFF', linewidth=3, marker='.', ms=4)
    ax.plot([centroids[0,1], centroids[-1,1]], [centroids[0,0], centroids[-1,0]], color='red', linewidth=3, ms=4)
    plot_contour(ax, total_mask, alpha=1, linewidth=3, c='#ffffff')
    ax.set_xticks([])
    ax.set_yticks([])
    y_text = mask_traj.shape[1]-30
    ax.text(10, y_text, f'total distance = {np.round(total_distance, 2)}', color='red', fontsize=22)
    ax.text(10, y_text-20, f'path length = {np.round(path_length, 2)}', color='#00BFFF', fontsize=22)
    ax.text(10, y_text-40, f'total area explored = {total_mask.sum()}', color='#ffffff', fontsize=22)
    fig.tight_layout()
    save_figure(fig, f'live_features/overlap_trajectory_cell_{id_cell}')

    # The cell at each time frame
    n_cols = 7
    n_rows = int(np.ceil(len(images)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4*ratio))
    for ax in axes.flatten():
        ax.axis('off')
    for ax, im, m in zip(axes.flatten(), images, mask_traj):
        ax.imshow(im, vmin=np.quantile(images, 0.001), vmax=np.quantile(images, 0.999), cmap='gray')
        plot_contour(ax, m, alpha=1, linewidth=1, c='red')
    fig.tight_layout()
    save_figure(fig, f'live_features/trajectory_cell_{id_cell}')
