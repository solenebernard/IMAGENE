'''
Distribution of the expression of each gene and threshold between low and high expression (step 5.3)
- thresholds: histogram of the number of RNA per CD8 positive cell, for the non-activated and the activated
  conditions, with the threshold of the gene

Input: exp/results/all_gene_profile.csv
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools.figures import FIG, save_figure
from tools.read_yaml import EXP_PATH, GENE_THRESHOLDS, LEARNING_GENES, RESULTS_PATH
from tools.style import PANEL_UNIT


def main():
    '''Draw and save the figures'''
    # The experiment of config.yaml, unless another one is given in config_figures.yaml
    exp_path = FIG['exp_path_gene_profile'] or EXP_PATH
    list_gene = [gene for gene in LEARNING_GENES if gene != 'CD8']

    all_genes = pd.read_csv(RESULTS_PATH + 'all_gene_profile.csv', index_col=['exp_path', 'condition', 'fish'], low_memory=False)
    available_exp = all_genes.index.unique('exp_path')
    if exp_path not in available_exp:
        raise KeyError(f"exp_path_gene_profile '{exp_path}' (6_exp_figures/config_figures.yaml) is not in exp/results/all_gene_profile.csv. "
                       f"Experiments in the table: {list(available_exp)}")
    all_genes = all_genes.loc[exp_path]
    # CD8 positive cells
    all_genes = all_genes[all_genes['Gene_CD8'] > 1]

    # The conditions that are available: name of the legend and conditions
    conditions = {'NA': [FIG['condition_control']], '24H': FIG['conditions_activated']}
    available = all_genes.index.get_level_values('condition')

    n_cols = 3
    n_rows = int(np.ceil(len(list_gene) / n_cols))
    # Size of panel f of the main figure: 1 unit wide and high (tools/style.py)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(PANEL_UNIT, PANEL_UNIT))
    for ax in axes.flatten():
        ax.axis('off')
    for ax, gene in zip(axes.flatten(), list_gene):
        ax.axis('on')
        # Same range for all the conditions
        vmax = int(np.floor(np.nanquantile(all_genes[f'Gene_{gene}'], q=0.96)))
        nbins = vmax if vmax < 60 else vmax // 3
        for name, list_conditions in conditions.items():
            gene_val = all_genes.loc[available.isin(list_conditions), f'Gene_{gene}']
            if len(gene_val) > 0:
                ax.hist(gene_val, bins=nbins, range=(0, vmax), alpha=0.5, density=True, label=name)
        ax.axvline(GENE_THRESHOLDS[gene], c='black', linewidth=2)
        ax.set_title(gene)
        ax.set_yticks([])
    axes.flatten()[len(list_gene) - 1].legend()
    fig.tight_layout()
    save_figure(fig, 'model/thresholds')


if __name__ == "__main__":
    main()
