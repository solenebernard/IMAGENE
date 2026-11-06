'''
Figures of the performance of the models, from the tables saved by 3_model.py

Saved in exp/results/:
    - dataset_size.png: number of cells with a low and a high expression of each gene
    - performance.png: balanced accuracy of each gene (nested cross-validation), for all the cells and for the extreme cells.
      Star: significantly better than chance (permutation test of 3_model.py)

Input: exp/results/performances_24H.csv, exp/results/performances_24H_extreme.csv

Run: python3 -m 5_POC_activation.4_figures_performances
     python3 -m 5_POC_activation.4_figures_performances --normalize_target   (results of 3_model.py --normalize_target)
'''

# Import local tools
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tools.read_yaml import RESULTS_PATH

from tools.style import FONTS, PANEL_UNIT, set_style


def plot_bars(ax, x, width, table, column, **kwargs):
    '''Bars of a balanced accuracy (in %), with its confidence interval and its value above'''
    values = table[column].values * 100
    low, high = table[f'{column}_ci_low'].values * 100, table[f'{column}_ci_high'].values * 100
    ax.bar(x, values, width, yerr=np.abs(np.stack((low, high)) - values), capsize=3, **kwargs)
    for xi, value, top in zip(x, values, high):
        if not np.isnan(value):
            ax.text(xi, top + 1, f'{value:.1f}', ha='center', va='bottom', fontsize=FONTS['annotation'])


def finish_axis(ax, x, genes, title):
    ax.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(genes)
    ax.set_ylabel('Balanced accuracy (%)')
    # Room above 100 % for the stars and the legend, under the title
    ax.set_ylim(0, 140)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title(title)
    # Legend inside the frame, above the bars: same width as the figure of the size of the dataset
    ax.legend(loc='upper center', ncol=2)


def plot_stars(ax, x, table):
    '''Star above the bars of the genes predicted better than chance (permutation test)'''
    for xi, significant, high in zip(x, table['significant'], table['Balanced_Accuracy_ci_high']):
        if significant == True:
            ax.text(xi, high * 100 + 4.5, '*', ha='center', va='bottom', fontsize=FONTS['symbol'])


def main():
    '''Draw and save the figures'''
    # Same font on all the figures (tools/style.py)
    set_style()

    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize_target', action='store_true', help='Figures of the results of 3_model.py --normalize_target (suffix _normalized_target)')
    args = parser.parse_args()
    suffix = '_normalized_target' if args.normalize_target else ''

    results_avg = pd.read_csv(f'{RESULTS_PATH}performances_24H{suffix}.csv', index_col='gene')
    results_avg_extreme = pd.read_csv(f'{RESULTS_PATH}performances_24H_extreme{suffix}.csv', index_col='gene')

    ##### FIGURE: size of the dataset of each gene (all the cells, not the extreme values)
    # Number of cells with a low and with a high expression: the cells of the training and test folds together
    n_low = (results_avg['nN_train'] + results_avg['nN_test']).round().astype(int)
    n_high = (results_avg['nP_train'] + results_avg['nP_test']).round().astype(int)
    x = np.arange(len(results_avg))
    # The two figures, one above the other, make panel g of the main figure: 1.7 units wide, 1 high (tools/style.py)
    fig, ax = plt.subplots(1, 1, figsize=(1.7 * PANEL_UNIT, PANEL_UNIT / 2))
    for offset, values, label in [(-0.2, n_low, 'Low expression'), (0.2, n_high, 'High expression')]:
        bars = ax.bar(x + offset, values, 0.4, label=label)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=FONTS['annotation'])
    ax.set_xticks(x)
    ax.set_xticklabels(results_avg.index)
    ax.set_ylabel('Number of cells')
    # Room above the bars for the legend
    ax.set_ylim(0, 1.7 * max(n_low.max(), n_high.max()))
    ax.set_yticks(range(0, int(max(n_low.max(), n_high.max())) + 200, 200))
    ax.set_title('Dataset size per gene')
    ax.legend(ncol=2, loc='upper right')
    fig.tight_layout()
    os.makedirs(RESULTS_PATH + 'figures', exist_ok=True)
    fig.savefig(f'{RESULTS_PATH}figures/dataset_size{suffix}.png', format='png', dpi=300)
    plt.close(fig)

    results_avg_extreme = results_avg_extreme.reindex(results_avg.index)

    x = np.arange(len(results_avg))
    genes = results_avg.index

    ##### FIGURE: nested cross-validation, for all the cells and for the extreme values
    fig, ax = plt.subplots(1, 1, figsize=(1.7 * PANEL_UNIT, PANEL_UNIT / 2))
    for offset, table, label in [(-0.2, results_avg, 'All samples'), (0.2, results_avg_extreme, 'Extreme')]:
        plot_bars(ax, x + offset, 0.4, table, 'Balanced_Accuracy', label=label)
        plot_stars(ax, x + offset, table)
    finish_axis(ax, x, genes, 'Effect of the training set on balanced accuracy')
    fig.tight_layout()
    fig.savefig(f'{RESULTS_PATH}figures/performance{suffix}.png', format='png', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
