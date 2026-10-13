'''
Performance of the prediction of the expression of each gene (step 5.3)
- performance: number of cells with high and low expression, true positive rate, true negative rate and
  balanced accuracy on the test folds, for all the cells and for the cells with extreme expression
- performance_test: true positive rate, true negative rate and balanced accuracy on the test folds

Input: 6_results/performances_24H.csv, 6_results/performances_24H_extreme.csv
'''

from tools.figures import *


def load_performance(path):
    '''Performance averaged over the folds, with the balanced accuracy and the number of cells of each class'''
    results = pd.read_csv(path, index_col='gene')
    results = results[results.index != 'CD8']
    results['balanced_accuracy_test'] = (results['TPR_test']+results['TNR_test'])/2
    results['Number_negative'] = results['nN_train']+results['nN_test']
    results['Number_positive'] = results['nP_train']+results['nP_test']
    return results

results_avg = load_performance('6_results/performances_24H.csv')
results_avg_extreme = load_performance('6_results/performances_24H_extreme.csv').reindex(results_avg.index)
genes = results_avg.index
x = np.arange(len(genes))


def write_values(ax, bars, fmt, fontsize):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, fmt.format(height), ha='center', va='bottom', fontsize=fontsize)


# All the cells and the cells with extreme expression
metrics = ['Number_positive', 'Number_negative', 'TPR_test', 'TNR_test', 'balanced_accuracy_test']
name_metrics = ['Number Of Positive', 'Number Of Negative', 'True Positive Rate', 'True Negative Rate', 'Balanced Accuracy']
fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
for i, (ax, metric, name) in enumerate(zip(axes, metrics, name_metrics)):
    is_rate = i >= 2
    if is_rate: # in percentage
        values, values_extreme, fmt = results_avg[metric]*100, results_avg_extreme[metric]*100, '{:.1f}'
    else:
        values, values_extreme, fmt = np.floor(results_avg[metric]), np.floor(results_avg_extreme[metric]), '{:.0f}'
    bars = ax.bar(x-0.2, values, 0.4, label='All samples')
    bars_extreme = ax.bar(x+0.2, values_extreme, 0.4, label='Extreme')
    write_values(ax, bars, fmt, fontsize=7)
    write_values(ax, bars_extreme, fmt, fontsize=7)
    ax.set_title(name, fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=14)
    if is_rate:
        ax.set_ylim(0, 110)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
axes[0].legend()
fig.tight_layout()
save_figure(fig, 'model/performance', formats=('png',))

# Performance on the test folds
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, metric, name in zip(axes, metrics[2:], name_metrics[2:]):
    bars = ax.bar(x, results_avg[metric], 0.6, label='Test')
    write_values(ax, bars, '{:.2f}', fontsize=9)
    ax.set_title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
fig.tight_layout()
save_figure(fig, 'model/performance_test', formats=('pdf',))
