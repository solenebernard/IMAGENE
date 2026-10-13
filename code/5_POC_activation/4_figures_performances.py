'''
Figures of the performance of the models, from the tables saved by 3_model.py

Saved in 6_results/:
    - dataset_size.png: number of cells with a low and a high expression of each gene
    - performance.png: balanced accuracy of each gene (nested cross-validation), for all the cells and for the extreme cells.
      Star: significantly better than chance (permutation test of 3_model.py)

Input: 6_results/performances_24H.csv, 6_results/performances_24H_extreme.csv

Run: python3 -m 5_POC_activation.4_figures_performances
     python3 -m 5_POC_activation.4_figures_performances --normalize_target   (results of 3_model.py --normalize_target)
'''

# Import local tools
from tools.read_yaml import *
from tools.imports import *

from matplotlib import rc
rc('font', family='sans-serif')

parser = argparse.ArgumentParser()
parser.add_argument('--normalize_target', action='store_true', help='Figures of the results of 3_model.py --normalize_target (suffix _normalized_target)')
args = parser.parse_args()
suffix = '_normalized_target' if args.normalize_target else ''

results_avg = pd.read_csv(f'6_results/performances_24H{suffix}.csv', index_col='gene')
results_avg_extreme = pd.read_csv(f'6_results/performances_24H_extreme{suffix}.csv', index_col='gene')


##### FIGURE: size of the dataset of each gene (all the cells, not the extreme values)
# Number of cells with a low and with a high expression: the cells of the training and test folds together
n_low = (results_avg['nN_train'] + results_avg['nN_test']).round().astype(int)
n_high = (results_avg['nP_train'] + results_avg['nP_test']).round().astype(int)
x = np.arange(len(results_avg))
fig, ax = plt.subplots(1, 1, figsize=(max(1.1*len(x)+2, 8), 4.5))
for offset, values, label in [(-0.2, n_low, 'Low expression'), (0.2, n_high, 'High expression')]:
    bars = ax.bar(x+offset, values, 0.4, label=label)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f'{gene}\n(n = {n})' for gene, n in zip(results_avg.index, n_low+n_high)], rotation=45, ha='right')
ax.set_ylabel('Number of cells')
ax.set_ylim(0, 1.15*max(n_low.max(), n_high.max()))
ax.set_title('Size of the dataset of each gene: matched CD8 positive cells with a low and a high expression', fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(f'6_results/dataset_size{suffix}.png', format='png', dpi=300)
plt.close(fig)


def plot_bars(ax, x, width, table, column, **kwargs):
    '''Bars of a balanced accuracy (in %), with its confidence interval and its value above'''
    values = table[column].values*100
    low, high = table[f'{column}_ci_low'].values*100, table[f'{column}_ci_high'].values*100
    ax.bar(x, values, width, yerr=np.abs(np.stack((low, high)) - values), capsize=3, **kwargs)
    for xi, value, top in zip(x, values, high):
        if not np.isnan(value):
            ax.text(xi, top+1, f'{value:.1f}', ha='center', va='bottom', fontsize=8)


def finish_axis(ax, x, genes, title):
    ax.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha='right')
    ax.set_ylabel('Balanced accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10)
    # Legend under the figure, not to hide the bars
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=1, fontsize=9, frameon=False)


results_avg_extreme = results_avg_extreme.reindex(results_avg.index)
# Tables saved by an older version of 3_model.py: corrected p-value instead of the column significant
for table in (results_avg, results_avg_extreme):
    if 'significant' not in table.columns:
        table['significant'] = table['q_value'] < 0.05


def plot_stars(ax, x, table):
    '''Star above the bars of the genes predicted better than chance (permutation test)'''
    for xi, significant, high in zip(x, table['significant'], table['Balanced_Accuracy_ci_high']):
        if significant == True:
            ax.text(xi, high*100+4.5, '*', ha='center', va='bottom', fontsize=14)

x = np.arange(len(results_avg))
genes = results_avg.index

##### FIGURE: nested cross-validation, for all the cells and for the extreme values
fig, ax = plt.subplots(1, 1, figsize=(max(1.2*len(x)+2, 8), 5.2))
for offset, table, label in [(-0.2, results_avg, 'All samples'), (0.2, results_avg_extreme, 'Extreme')]:
    plot_bars(ax, x+offset, 0.4, table, 'Balanced_Accuracy', label=label)
    plot_stars(ax, x+offset, table)
finish_axis(ax, x, genes, 'Nested cross-validation. Error bar: 95 % confidence interval. *: significantly better than chance')
fig.tight_layout()
fig.savefig(f'6_results/performance{suffix}.png', format='png', dpi=300)
plt.close(fig)
