'''
Distribution of the expression of each gene and threshold between low and high expression (step 5.3)
- thresholds: histogram of the number of RNA per CD8 positive cell, for the non-activated and the activated
  conditions, with the threshold of the gene

Input: 6_results/all_gene_profile.csv
'''

from tools.figures import *

exp_path = FIG['exp_path_gene_profile']
list_gene = [gene for gene in LEARNING_GENES if gene != 'CD8']

all_genes = pd.read_csv('6_results/all_gene_profile.csv', index_col=['exp_path', 'condition', 'fish'], low_memory=False)
available_exp = all_genes.index.unique('exp_path')
if exp_path not in available_exp:
    raise KeyError(f"exp_path_gene_profile '{exp_path}' (7_figures/config_figures.yaml) is not in 6_results/all_gene_profile.csv. "
                   f"Experiments in the table: {list(available_exp)}")
all_genes = all_genes.loc[exp_path]
# CD8 positive cells
all_genes = all_genes[all_genes['Gene_CD8']>1]

# The conditions that are available: name of the legend and conditions
conditions = {'NA': [FIG['condition_control']], '24H': FIG['conditions_activated']}
available = all_genes.index.get_level_values('condition')

n_cols = 3
n_rows = int(np.ceil(len(list_gene)/n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
for ax in axes.flatten():
    ax.axis('off')
for ax, gene in zip(axes.flatten(), list_gene):
    ax.axis('on')
    # Same range for all the conditions
    vmax = int(np.floor(np.nanquantile(all_genes[f'Gene_{gene}'], q=0.96)))
    nbins = vmax if vmax<60 else vmax//3
    for name, list_conditions in conditions.items():
        gene_val = all_genes.loc[available.isin(list_conditions), f'Gene_{gene}']
        if len(gene_val) > 0:
            ax.hist(gene_val, bins=nbins, range=(0, vmax), alpha=0.5, density=True, label=name)
    ax.axvline(GENE_THRESHOLDS[gene], c='black', linewidth=2)
    ax.set_title(gene)
    ax.set_yticks([])
axes.flatten()[len(list_gene)-1].legend()
fig.tight_layout()
save_figure(fig, 'model/thresholds')
