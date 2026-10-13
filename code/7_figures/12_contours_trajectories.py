'''
Contour and trajectory of the cells, by condition and by level of expression of each gene
- contour_condition, trajectory_condition: contours (first time frame) and trajectories of cells of the
  non-activated and of the activated conditions
- contour_low_high_{gene}, trajectory_low_high_{gene}: same for the cells with the lowest and the highest
  expression of the gene

Input: 6_results/dataset.csv, tables of features of step 4.1 (fourier_finetune, finetune_scikit)
'''

import pyefd

from tools.figures import *

params = FIG['contours_trajectories']
path_feats = DATA_PATH + 'live_cell_features/'


def load_features(name, conditions):
    '''Table of features of step 4.1 for the conditions that are available, indexed by (condition, cell_bf, t)'''
    tables = {}
    for condition in conditions:
        path = pathlib.Path(path_feats + name.format(condition=condition))
        if path.exists():
            tables[condition] = pd.read_csv(path, index_col=['cell_bf', 't'])
    return pd.concat(tables, names=['condition']).sort_index()

all_conditions = [FIG['condition_control']] + FIG['conditions_activated']
df_fourier = load_features('fourier_finetune_{condition}.csv', all_conditions)
df_scikit = load_features('finetune_scikit_{condition}.csv', all_conditions)


def plot_cell_contour(ax, condition, cell_bf):
    '''Contour of the cell at its first time frame, reconstructed from its Fourier coefficients'''
    coeffs = np.asarray(df_fourier.loc[(condition, cell_bf)].iloc[0]).reshape((FOURIER_ORDER, 4))
    reconstruction = pyefd.reconstruct_contour(coeffs, np.array([0,0]), 128)
    ax.plot(reconstruction[:,0], reconstruction[:,1], alpha=0.4)


def plot_cell_trajectory(ax, condition, cell_bf, **kwargs):
    '''Trajectory of the centroid of the cell, starting from the origin'''
    traj = df_scikit.loc[(condition, cell_bf), ['centroid-0', 'centroid-1']]
    traj = traj - traj.iloc[0]
    # Skip the cells seen in a single time frame, and the trajectories with a tracking error
    if len(traj) > 1 and np.linalg.norm(np.diff(traj, axis=0), axis=1).max() < MAX_SPEED:
        ax.plot(traj['centroid-0'], traj['centroid-1'], alpha=0.4, **kwargs)


def clean_axes(axes, limit=None):
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if limit is not None:
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.axis('off')


##### BY CONDITION
conditions = [c for c in [FIG['condition_control'], FIG['conditions_activated'][0]] \
              if c in df_fourier.index.get_level_values('condition')]
fig_contour, axes_contour = plt.subplots(1, len(conditions), figsize=(7.5*len(conditions), 5), sharex=True, sharey=True, squeeze=False)
fig_traj, axes_traj = plt.subplots(1, len(conditions), figsize=(4*len(conditions), 5), sharex=True, sharey=True, squeeze=False)
for ax_contour, ax_traj, condition in zip(axes_contour[0], axes_traj[0], conditions):
    # The first cells tracked over all the time frames
    n_frames = df_fourier.loc[condition].groupby(level='cell_bf').size()
    for cell_bf in n_frames.index[n_frames==N_FRAMES][:params['n_cells']]:
        plot_cell_contour(ax_contour, condition, cell_bf)
        plot_cell_trajectory(ax_traj, condition, cell_bf)
    ax_contour.set_title(condition)
    ax_traj.set_title(condition)
clean_axes(axes_contour[0])
clean_axes(axes_traj[0])
save_figure(fig_contour, 'model/contour_condition', formats=('pdf',))
save_figure(fig_traj, 'model/trajectory_condition', formats=('pdf',))


##### BY LEVEL OF EXPRESSION OF EACH GENE
df = load_dataset()
# CD8 positive cells of the activated conditions
df = df.loc[cd8_positive(df)]
df = df[df.index.get_level_values('condition').isin(FIG['conditions_activated'])]

for gene in LEARNING_GENES:
    if gene == 'CD8':
        continue
    expression = df[f'Gene_{gene}'].dropna()
    # The cells with the lowest and the highest expression, on each side of the threshold
    low = expression[expression<GENE_THRESHOLDS[gene]].sort_values(ascending=True)
    high = expression[expression>=GENE_THRESHOLDS[gene]].sort_values(ascending=False)

    fig_contour, axes_contour = plt.subplots(1, 2, figsize=(4,2.5), sharex=True, sharey=True)
    fig_traj, axes_traj = plt.subplots(1, 2, figsize=(4,2.5), sharex=True, sharey=True)
    for ax_contour, ax_traj, cells, name in zip(axes_contour, axes_traj, [low, high], ['Low', 'High']):
        for _, condition, cell_bf in cells.index[:params['n_cells_gene']]:
            plot_cell_contour(ax_contour, condition, cell_bf)
            plot_cell_trajectory(ax_traj, condition, cell_bf, marker='.', linewidth=2, ms=2)
        ax_contour.set_title(f'{name} {gene}')
        ax_traj.set_title(f'{name} {gene}')
    clean_axes(axes_contour, limit=params['contour_limit'])
    clean_axes(axes_traj, limit=params['trajectory_limit'])
    fig_contour.tight_layout()
    fig_traj.tight_layout()
    save_figure(fig_contour, f'model/contour_low_high_{gene}', formats=('png',))
    save_figure(fig_traj, f'model/trajectory_low_high_{gene}', formats=('png',))
