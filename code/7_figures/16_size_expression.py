'''
Size of the live cell against the expression of each gene, for the cells used by the models
(reliably matched, CD8 positive)
- size_expression: mean area of the cell against its number of RNA, one panel per gene
- size_expression_normalized: mean area of the cell against its number of RNA normalized by the size of the cell
  (number of RNA divided by the mean area of the cell relative to the median area, as with 3_model.py --normalize_target)

In each panel: Spearman correlation and its p-value, and the threshold between low and high expression (dashed line).

Input: 6_results/dataset.csv
'''

from scipy.stats import spearmanr

from tools.figures import *

df = load_dataset()
df = df.loc[cd8_positive(df)]
genes = [gene for gene in LEARNING_GENES if gene != 'CD8']

# Mean area of the cell over the movie, in um2, and relative to the median area of the cells
pixel_size = 0.1625 # um, BF image (40x)
area = df['area_mean'] * pixel_size**2
relative_area = df['area_mean'] / df['area_mean'].median()


def plot_size_expression(normalized, name):
    '''One scatter plot per gene: area of the cell (x) against its expression (y)'''
    n_cols = 3
    n_rows = int(np.ceil(len(genes)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.6*n_rows), squeeze=False)
    for ax in axes.flatten()[len(genes):]:
        ax.axis('off')
    for ax, gene in zip(axes.flatten(), genes):
        expression = df[f'Gene_{gene}'].dropna()
        if normalized:
            expression = expression / relative_area.loc[expression.index]
        x = area.loc[expression.index]
        ax.scatter(x, expression, s=6, alpha=0.5, linewidths=0)
        ax.axhline(GENE_THRESHOLDS[gene], color='gray', linestyle='--', linewidth=0.8)
        r, p_value = spearmanr(x, expression)
        p_text = 'p < 0.001' if p_value < 0.001 else f'p = {p_value:.3f}'
        ax.set_title(f'{gene}\nSpearman r = {r:.2f}, {p_text}, n = {len(expression)}', fontsize=10)
        ax.set_xlabel('Mean area of the cell (µm²)')
        ax.set_ylabel('Number of RNA / relative area' if normalized else 'Number of RNA')
        # The few cells with the highest expression do not set the scale
        ax.set_ylim(-0.03*np.percentile(expression, 99.5), 1.05*np.percentile(expression, 99.5))
        ax.spines[['top', 'right']].set_visible(False)
    title = 'Size of the cell against the expression normalized by the size of the cell' if normalized \
            else 'Size of the cell against the expression of each gene'
    fig.suptitle(title + '. Dashed line: threshold between low and high expression', fontsize=11)
    fig.tight_layout()
    save_figure(fig, name)


plot_size_expression(normalized=False, name='model/size_expression')
plot_size_expression(normalized=True, name='model/size_expression_normalized')
