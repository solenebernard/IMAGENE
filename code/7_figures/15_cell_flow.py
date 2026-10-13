'''
Flow (Sankey) diagram of the number of live cells kept at each step of the matching, for each activated condition.
The width of each flow is proportional to the number of cells; the cells lost at a step leave downwards.
1. live cells tracked over the whole movie
2. of which in the field of view of the smFISH image
3. of which matched to a smFISH cell
4. of which passing the quality control of the matching (ratio_qc)
5. of which with enough RNA: at least min_mean_rna RNA per gene on average (the others have almost no RNA in any gene)
6. of which CD8 positive: the cells used by the models

- cell_flow: the diagram
- cell_flow.csv: the numbers

Input: 6_results/dataset.csv, overlay of step 3
'''

from tools.figures import *
from tools.matching import mask_centroids
from tools.model import mean_rna_per_gene

conditions = FIG['conditions_activated']
steps = ['Live cells tracked\nover the whole movie',
         'In the field of view\nof the smFISH image',
         'Matched to a smFISH cell',
         f'Passing the quality control\n(ratio_qc = {RATIO_QC})',
         f'At least {MIN_MEAN_RNA} RNA per gene\non average',
         'CD8 positive']

df = pd.read_csv('6_results/dataset.csv', index_col=['exp_path', 'condition', 'cell_bf'])


def count_cells(condition):
    '''Number of live cells of the condition kept at each step'''
    cells = df.xs(condition, level='condition').droplevel('exp_path')
    # 1. Tracked over the whole movie
    cells = cells[cells['count']==N_FRAMES]
    counts = [len(cells)]

    # 2. In the field of view of the smFISH image: centroid of the cell (last time frame) inside the smFISH mosaic
    overlay = load_overlay(condition)
    is_fish = overlay[:,:,1] > 0
    rows, cols = np.where(is_fish.any(axis=1))[0], np.where(is_fish.any(axis=0))[0]
    in_fov = [id_cell for id_cell, (r, c) in mask_centroids(overlay[:,:,0]).items() \
              if rows.min() <= r <= rows.max() and cols.min() <= c <= cols.max()]
    cells = cells[cells.index.isin(in_fov)]
    counts.append(len(cells))

    # 3. Matched to a smFISH cell
    cells = cells[cells['Fish'].notna()]
    counts.append(len(cells))

    # 4. Passing the quality control of the matching
    cells = cells[passes_ratio_qc_table(cells, RATIO_QC)]
    counts.append(len(cells))

    # 5. Enough RNA: total number of RNA of the genes divided by the number of genes with a value
    cells = cells[mean_rna_per_gene(cells) >= MIN_MEAN_RNA]
    counts.append(len(cells))

    # 6. CD8 positive: at least two RNA of CD8
    counts.append(int((cells['Gene_CD8']>1).sum()))
    return counts

counts = pd.DataFrame({condition: count_cells(condition) for condition in conditions}, \
                      index=[step.replace('\n', ' ') for step in steps])
path_csv = pathlib.Path(FIG['output_dir']) / 'matching/cell_flow.csv'
path_csv.parent.mkdir(parents=True, exist_ok=True)
counts.to_csv(path_csv)
print(counts)


##### FLOW (SANKEY) DIAGRAM: one row per condition
# The height of each bar and the width of each flow are proportional to the number of cells.
# At each step, the cells that are kept continue to the right and the cells that are lost leave downwards.
losses = ['outside the smFISH\nfield of view', 'no smFISH cell', 'ambiguous match', 'almost no RNA', 'CD8 negative']
colors = matplotlib.colormaps['Blues'](np.linspace(0.45, 0.85, len(steps)))
bar_width, step_x = 0.16, 1.0
x_curve = np.linspace(0, 1, 50)
smooth = 3*x_curve**2 - 2*x_curve**3 # from 0 to 1, flat at both ends

fig, axes = plt.subplots(len(conditions), 1, figsize=(15.5, 4.3*len(conditions)), squeeze=False)
for ax, condition in zip(axes[:,0], conditions):
    n_cells = counts[condition].values
    n_max = n_cells[0]
    drop = 0.45*n_max # how far down the lost cells leave
    for j, (step, n) in enumerate(zip(steps, n_cells)):
        x = j*step_x
        # Bar of the step: from the top (0) down to -n
        ax.fill_between([x, x+bar_width], 0, -n, color=colors[j], zorder=3)
        ax.text(x+bar_width/2, 0.04*n_max, f'{step}\n{n}', ha='center', va='bottom', fontsize=10, linespacing=1.3)
        if j == len(steps)-1:
            break
        n_next = n_cells[j+1]
        x0, x1 = x+bar_width, x+step_x
        # Cells kept: straight flow to the next bar
        ax.fill_between([x0, x1], 0, -n_next, color=colors[j], alpha=0.35, zorder=1)
        ax.text((x0+x1)/2, -n_next/2, f'{n_next/n*100:.0f} %' if n > 0 else '', ha='center', va='center', fontsize=10, zorder=4)
        # Cells lost: flow bending downwards
        n_lost = n - n_next
        if n_lost > 0:
            x_lost = x0 + x_curve*(step_x-bar_width)*0.75
            ax.fill_between(x_lost, -n_next - smooth*drop, -n - smooth*drop, color='lightgray', alpha=0.9, zorder=2)
            ax.text(x_lost[-1]+0.02, -(n_next+n)/2 - drop, f'{n_lost}\n{losses[j]}', ha='left', va='center', fontsize=9, color='dimgray')
    ax.set_title(condition, fontsize=15, loc='left', pad=40)
    ax.set_xlim(-0.05, (len(steps)-1)*step_x+bar_width+0.35)
    ax.set_ylim(-n_max-drop-0.03*n_max, 0.02*n_max)
    ax.axis('off')
fig.tight_layout()
save_figure(fig, 'matching/cell_flow')
