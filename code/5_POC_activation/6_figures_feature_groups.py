'''
Figure of the models trained with different groups of features, from the table saved by 5_model_feature_groups.py

Saved in 6_results/:
    - performance_feature_groups.png: balanced accuracy of each gene with each group of features alone (size, shape,
      motility, overlap, neighbourhood) and with their combinations. Star: significantly better than chance

Run: python3 -m 5_POC_activation.6_figures_feature_groups
'''

# Import local tools
from tools.read_yaml import *
from tools.imports import *

from matplotlib import rc
rc('font', family='sans-serif')

results = pd.read_csv('6_results/performances_feature_groups.csv', index_col=['gene', 'features'])
genes = list(results.index.get_level_values('gene').unique())
x = np.arange(len(genes))


def plot_bars(ax, x, width, table, column, **kwargs):
    '''Bars of a score (in %), with its confidence interval and its value above'''
    values = table[column].values*100
    low, high = table[f'{column}_ci_low'].values*100, table[f'{column}_ci_high'].values*100
    ax.bar(x, values, width, yerr=np.abs(np.stack((low, high)) - values), capsize=1.5, error_kw={'elinewidth': 0.8}, **kwargs)
    for xi, value, top in zip(x, values, np.maximum(high, values)):
        ax.text(xi, top+0.8, f'{value:.1f}', ha='center', va='bottom', fontsize=6, rotation=90)
    # Star above the models significantly better than chance (permutation test)
    if 'significant' in table.columns:
        for xi, top, significant in zip(x, np.maximum(high, values), table['significant']):
            if significant == True:
                ax.text(xi, top+6.5, '*', ha='center', va='bottom', fontsize=9)


##### FIGURE: balanced accuracy of each set of features. Groups alone in plain colors, combinations hatched
feature_sets = {'size': ('Size', 'lightgray', None), 'shape': ('Shape', 'tab:blue', None), 'motility': ('Motility', 'tab:orange', None), \
                'overlap': ('Overlap', 'tab:brown', None), 'neighbourhood': ('Neighbourhood', 'tab:purple', None), \
                'size + motility': ('Size + motility', 'navajowhite', '//'), 'size + shape': ('Size + shape', 'lightblue', '//'), \
                'size + motility + shape': ('Size + motility + shape', 'lightgreen', '//'), 'all': ('All', 'tab:green', None)}
# Only the sets of features that were trained
feature_sets = {name: value for name, value in feature_sets.items() if name in results.index.get_level_values('features')}
width = 0.85/len(feature_sets)
fig, ax = plt.subplots(1, 1, figsize=(max(2.6*len(genes)+2, 8), 5.5))
for i, (name, (label, color, hatch)) in enumerate(feature_sets.items()):
    table = results.xs(name, level='features').loc[genes]
    plot_bars(ax, x+(i-(len(feature_sets)-1)/2)*width, width, table, 'Balanced_Accuracy', color=color, hatch=hatch, \
              edgecolor='white', linewidth=0.3, label=f"{label} ({int(table['n_features'].iloc[0])})")
ax.axhline(50, color='black', linestyle='--', linewidth=0.8, label='Chance')
ax.set_xticks(x)
ax.set_xticklabels(genes, rotation=45, ha='right')
ax.set_ylabel('Balanced accuracy (%)')
ax.set_ylim(0, 100)
ax.set_title('Prediction with each group of features (nested cross-validation, same cells and folds)\nError bars: 95 % confidence interval. *: significantly better than chance', fontsize=10)
ax.legend(fontsize=8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.18), title='Features (number of features)', title_fontsize=8)
fig.tight_layout()
fig.savefig('6_results/performance_feature_groups.png', format='png', dpi=300)
plt.close(fig)
