'''
Trajectories and contours of the cells with a low and a high expression of a few genes, in one grid
- contours_trajectories_grid: one row per gene (contours_trajectories.grid_genes in config_figures.yaml), four columns:
  trajectories of the cells with a low expression, of the cells with a high expression, then their contours

The cells are the CD8 positive cells of the activated conditions with the
lowest expression of the gene below its threshold, and with the highest expression above it
(contours_trajectories.n_cells_gene cells of each). Each trajectory starts at the origin, and each contour is the one
of the first time frame, reconstructed from its Fourier coefficients. The panels of the trajectories, and those of the
contours, have the same limits, so that they can be compared; by default (contours_trajectories.trajectory_limit and
contour_limit: null) the limits are set so that all the cells fit in the panels.

Input: exp/results/dataset.csv, tables of features of step 4.1 (fourier_finetune, finetune_scikit)
'''

import matplotlib.pyplot as plt
import numpy as np
from tools.figures import cd8_positive, cell_contour, cell_trajectory, FIG, load_dataset, load_features, save_figure
from tools.read_yaml import GENE_THRESHOLDS, MAX_SPEED, PIXEL_SIZE
from tools.style import FONTS, PANEL_UNIT




MARGIN = 1.12  # Automatic limits of the panels: largest extent of the cells shown, plus 12 %


def cell_contour_um(df_fourier, condition, cell_bf):
    '''Contour of the cell at its first time frame, reconstructed from its Fourier coefficients, in um: (N, 2)'''
    return np.asarray(cell_contour(df_fourier, condition, cell_bf) * PIXEL_SIZE)


def cell_trajectory_um(df_scikit, condition, cell_bf):
    '''Trajectory of the centroid of the cell, starting from the origin, in um: (N, 2). None for the cells seen in a
    single time frame and the trajectories with a tracking error'''
    traj = cell_trajectory(df_scikit, condition, cell_bf) * PIXEL_SIZE
    if len(traj) > 1 and np.linalg.norm(np.diff(traj, axis=0), axis=1).max() < MAX_SPEED * PIXEL_SIZE:
        return traj[['centroid-0', 'centroid-1']].to_numpy()
    return None


def main():
    '''Draw and save the figures'''
    params = FIG['contours_trajectories']

    conditions = FIG['conditions_activated']
    df_fourier = load_features('fourier_finetune_{condition}.csv', conditions)
    df_scikit = load_features('finetune_scikit_{condition}.csv', conditions)

    df = load_dataset()
    # CD8 positive cells of the activated conditions
    df = df.loc[cd8_positive(df)]
    df = df[df.index.get_level_values('condition').isin(conditions)]

    genes = params['grid_genes']
    columns = ['Lowest\nexpressing', 'Highest\nexpressing'] * 2
    # Trajectories and contours of each panel (row: gene, column: low and high trajectories, low and high contours)
    curves = {}
    for row, gene in enumerate(genes):
        expression = df[f'Gene_{gene}'].dropna()
        # The cells with the lowest and the highest expression, on each side of the threshold
        low = expression[expression < GENE_THRESHOLDS[gene]].sort_values(ascending=True)
        high = expression[expression >= GENE_THRESHOLDS[gene]].sort_values(ascending=False)
        for col, cells in enumerate([low, high, low, high]):
            cells = cells.index[:params['n_cells_gene']]
            if col < 2:
                curves[row, col] = [cell_trajectory_um(df_scikit, condition, cell_bf) for _, condition, cell_bf in cells]
            else:
                curves[row, col] = [cell_contour_um(df_fourier, condition, cell_bf) for _, condition, cell_bf in cells]
            curves[row, col] = [c for c in curves[row, col] if c is not None]

    # Same limits for all the panels of trajectories, and for all the panels of contours, so that they can be compared:
    # contours_trajectories.trajectory_limit and contour_limit (pixels), or, if null, the largest extent of the cells
    def limit(cols, name):
        if params.get(name) is not None:
            return params[name] * PIXEL_SIZE
        return MARGIN * max(np.abs(c).max() for (_, col), cs in curves.items() if col in cols for c in cs)
    limit_trajectories, limit_contours = limit((0, 1), 'trajectory_limit'), limit((2, 3), 'contour_limit')
    limits = [limit_trajectories] * 2 + [limit_contours] * 2

    # Size of panel i of the main figure: 1.3 units wide, 1 high (tools/style.py)
    fig, axes = plt.subplots(len(genes), 4, figsize=(1.3 * PANEL_UNIT, PANEL_UNIT), squeeze=False)
    for row, gene in enumerate(genes):
        for col in range(4):
            ax = axes[row, col]
            for c in curves[row, col]:
                if col < 2:
                    ax.plot(c[:, 0], c[:, 1], alpha=0.5, marker='.', linewidth=1.5, ms=2)
                else:
                    ax.plot(c[:, 0], c[:, 1], alpha=0.4, linewidth=1)
            ax.set_xlim(-limits[col], limits[col])
            ax.set_ylim(-limits[col], limits[col])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col])
            if col == 0:
                ax.set_ylabel(gene)
        # Scale bars: 20 um for the trajectories, 5 um for the contours
        for col, (length, text) in zip([0, 2], [(20, '20 µm'), (5, '5 µm')]):
            ax = axes[row, col]
            x0, y0 = -limits[col] * 0.85, -limits[col] * 0.85
            ax.plot([x0, x0 + length], [y0, y0], color='black', linewidth=2)
            # Label to the right of the bar
            ax.text(x0 + length + limits[col] * 0.05, y0, text, ha='left', va='center', fontsize=FONTS['annotation'])
    # Room at the top for the titles of the two pairs of columns
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    # Just above the titles of the columns (two lines)
    y_title = axes[0, 0].get_position().y1 + 2.6 * FONTS['title'] / 72 / fig.get_figheight()
    for (first, last), text in [((0, 1), 'Trajectories'), ((2, 3), 'Contours')]:
        left, right = axes[0, first].get_position().x0, axes[0, last].get_position().x1
        fig.text((left + right) / 2, y_title, text, ha='center', va='bottom', fontsize=FONTS['title'], fontweight='bold')
    save_figure(fig, 'model/contours_trajectories_grid')


if __name__ == "__main__":
    main()
