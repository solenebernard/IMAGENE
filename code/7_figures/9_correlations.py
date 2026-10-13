'''
Correlation between the features of the live cells and the expression of each gene (step 5)
- correlation_matrix: Spearman correlation for all the matched cells
- correlation_matrix_extreme: same for the cells with the lowest and the highest expression of a gene
The correlations that are not significant (false discovery rate above 5 %) are set to 0.
The features are the ones used by the models (relative to the size of the cell, see tools/model.py). They are ordered
by family (size, shape, motility, overlap, neighbourhood), shown by the colored band above the matrix; inside a family,
the summaries of the same measure (mean, std, max, min) are next to each other. The genes are clustered.

Input: 6_results/dataset.csv
'''

from scipy.stats import spearmanr
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from matplotlib.patches import Patch

from tools.figures import *
from tools.model import select_feature_columns, feature_groups, normalize_features

df = load_dataset()
# Features of the models, relative to the size of the cell
df_features = normalize_features(df[select_feature_columns(df)])
df = pd.concat((df.drop(columns=[c for c in df_features.columns if c in df.columns]), df_features), axis=1)

# Features ordered by family. In the shape family, the descriptors of the mask come before the Fourier features
groups = feature_groups(df_features.columns)
is_fourier = lambda c: c.startswith(('magnitude_', 'elliptic_area_'))
groups['shape'] = [c for c in groups['shape'] if not is_fourier(c)] + [c for c in groups['shape'] if is_fourier(c)]
families = {'Size': ('size', 'gray'), 'Shape': ('shape', 'tab:blue'), 'Motility': ('motility', 'tab:orange'), \
            'Overlap': ('overlap', 'tab:brown'), 'Neighbourhood': ('neighbors', 'tab:purple')}
families = {name: (groups[key], color) for name, (key, color) in families.items() if len(groups[key]) > 0}
features = sum((columns for columns, _ in families.values()), [])
family_colors = pd.Series({feature: color for columns, color in families.values() for feature in columns}, name='Family')
list_gene = LEARNING_GENES


def correlation_matrix(df, name):
    '''Matrix of the significant Spearman correlations between the features (ordered by family) and the genes (clustered)'''
    corr_mat = pd.DataFrame(index=features, columns=list_gene, dtype=float)
    pval_mat = pd.DataFrame(index=features, columns=list_gene, dtype=float)
    for gene in list_gene:
        for feat in features:
            feature_v, gene_v = df[feat], df[f'Gene_{gene}']
            bool_nan = (gene_v.isna()==False)&(feature_v.isna()==False)
            feature_v, gene_v = feature_v[bool_nan], gene_v[bool_nan]
            if feature_v.nunique() < 2 or gene_v.nunique() < 2:
                # The correlation is not defined for a constant feature or gene: not significant
                r, p = 0, 1
            else:
                r, p = spearmanr(feature_v, gene_v)
            corr_mat.loc[feat, gene] = r
            pval_mat.loc[feat, gene] = p

    # Benjamini-Hochberg correction
    _, qvals, _, _ = multipletests(pval_mat.values.flatten(), method="fdr_bh")
    qval_mat = pd.DataFrame(qvals.reshape(pval_mat.shape), index=pval_mat.index, columns=pval_mat.columns)
    corr_mat = corr_mat.mask(qval_mat > 0.05).fillna(0)

    g = sns.clustermap(
        corr_mat.T,
        cmap="coolwarm", center=0,
        cbar_kws={'label': 'Spearman correlation'},
        linewidths=0.5,
        linecolor="lightgray",
        col_cluster=False, # the features keep the order of the families
        row_cluster=True,
        col_colors=family_colors,
        figsize=(max(0.16*len(features), 20), 8),
        cbar_pos=(0, 0.5, 0.01, 0.2)
    )
    # Show the name of all the features and genes
    order_genes = g.dendrogram_row.reordered_ind
    g.ax_heatmap.set_xticks(np.arange(len(features)) + 0.5)
    g.ax_heatmap.set_yticks(np.arange(len(list_gene)) + 0.5)
    g.ax_heatmap.set_xticklabels(corr_mat.index, rotation=90, fontsize=6)
    g.ax_heatmap.set_yticklabels(corr_mat.columns[order_genes], rotation=0, fontsize=6)
    # Line between two families, and legend of the families
    position = 0
    for columns, _ in list(families.values())[:-1]:
        position += len(columns)
        g.ax_heatmap.axvline(position, color='black', linewidth=1.5)
    g.ax_col_colors.legend(handles=[Patch(color=color, label=f'{name} ({len(columns)})') for name, (columns, color) in families.items()], \
                           ncol=len(families), loc='lower center', bbox_to_anchor=(0.5, 1.2), frameon=False, fontsize=9)
    save_figure(g.fig, name, formats=('png',))


# All the matched cells, missing features replaced by the median as for the model
df_filled = df.copy()
df_filled[features] = df[features].fillna(df[features].median())
correlation_matrix(df_filled, 'model/correlation_matrix')

# The cells with the lowest and the highest expression of at least one gene, among the CD8 positive cells
index_cd8 = cd8_positive(df)
index_extreme = pd.Index([])
for gene in list_gene:
    y = df.loc[index_cd8, f'Gene_{gene}'].dropna().sort_values()
    index_extreme = index_extreme.union(y.index[:N_EXTREME]).union(y.index[-N_EXTREME:])
correlation_matrix(df.loc[index_extreme], 'model/correlation_matrix_extreme')
