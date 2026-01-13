# Import local tools
from tools.read_yaml import *
from tools.imports import *

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.inspection import permutation_importance
import umap
from kneed import KneeLocator
from scipy.signal import savgol_filter


from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv('6_results/dataset.csv', index_col=['exp_path', 'condition', 'cell_bf'])
df = df.loc[('2025-10-13_Olga6/', slice(None), slice(None))]

list_gene = ['GZMB', 'CD69', 'CCR7', 'GBP4', 'ACTB', 'CORO1A'] # Genes of high quality in olga6

##### FILTER DATA
# Bool
distance_tresh, distance_2nd_tresh = 60,80
# Filter on distance
distance = df['Distance']<distance_tresh
distance_2nd = (df['Distance_2nd_bf_to_fish']>distance_2nd_tresh)&\
                (df['Distance_2nd_fish_to_bf']>distance_2nd_tresh)
# Is not on edge
bool_edge = df['is_on_edge'] == False
# With 21 time frames
bool_count = df['count'] == 21
bool_match = distance&distance_2nd&bool_edge&bool_count
df.loc[~bool_match, 'Fish'] = None
for col in list_gene:
    df.loc[~bool_match, f'Gene_{col}'] = None


####### SELECT FEATURE

# Remove some columns
columns_gene = [c for c in df.columns \
                if c.startswith(('Gene_', 'delta_z_', 'Fish', \
                    'Distance', 'Id_2nd', 'is_on_edge', 'count', 'BF_center_', 'traj_', 'centroid-'))]

select_features = [x for x in df.columns if x not in columns_gene]

# OLGA6 
dict_thresholds = {'CORO1A': 40,
                'TNF': 3,
                'GZMB': 11,
                'ACTB': 100,
                'CD69': 10,
                'IRF8': 5,
                'CCR7': 30,
                'GBP4': 4,
                'XP01': 5,
                'IFNG': 3,
                'IL2RA': 3}

# Correlation between genes?

# Initialize correlation & pval matrices
corr_mat = pd.DataFrame(index=list_gene, columns=list_gene, dtype=float)
pval_mat = pd.DataFrame(index=list_gene, columns=list_gene, dtype=float)

dict_n_rows = {}
# Compute correlations
for gene1 in list_gene:
    dict_n_rows[gene1] = int((df[f'Gene_{gene1}'].isna()==False).sum())
    for gene2 in list_gene:
        gene_v1, gene_v2 = df[f'Gene_{gene1}'], df[f'Gene_{gene2}']
        # Binary ?
        gene_v1 = gene_v1>dict_thresholds[gene1]
        gene_v2 = gene_v2>dict_thresholds[gene2]
        bool_nan = (gene_v1.isna()==False)&(gene_v2.isna()==False)
        r, p = spearmanr(gene_v1[bool_nan],gene_v2[bool_nan])
        if np.isnan(r):
            r,p = 0,0
        corr_mat.loc[gene1, gene2] = r
        pval_mat.loc[gene1, gene2] = p
# Optional: adjust p-values for multiple testing (FDR across all feature–gene pairs)
flat_pvals = pval_mat.values.flatten()
_, qvals, _, _ = multipletests(flat_pvals, method="fdr_bh")
qval_mat = pd.DataFrame(qvals.reshape(pval_mat.shape), index=pval_mat.index, columns=pval_mat.columns)

# Mask nonsignificant correlations (e.g., q > 0.05)
corr_mat = corr_mat.mask(qval_mat.abs() > 0.05)
corr_mat = corr_mat.fillna(0)
# Mask nonsignificant correlations (e.g., q > 0.05)
corr_mat = corr_mat[corr_mat.abs().max(axis=1)>0.2]
# Plot heatmap of correlations
g = sns.clustermap(
    corr_mat,
    cmap="coolwarm", center=0,
    cbar_kws={'label': 'Spearman correlation'},
    linewidths=0.5,
    linecolor="lightgray",
    col_cluster=True,
    row_cluster=True
)
row_order = g.dendrogram_row.reordered_ind
col_order = g.dendrogram_col.reordered_ind

# Reorder original matrix
reordered = corr_mat.iloc[row_order, col_order]

# Show all row and column labels
g.ax_heatmap.set_xticks(np.arange(len(reordered.columns)) + 0.5)
g.ax_heatmap.set_yticks(np.arange(len(reordered.index)) + 0.5)
g.ax_heatmap.set_xticklabels(reordered.columns, rotation=90, fontsize=6)
g.ax_heatmap.set_yticklabels(reordered.index, rotation=0, fontsize=6)

###### FEATURE DIMENSION REDUCTION : SELECTION OF CORRELATED FEATURES


# Initialize correlation & pval matrices
corr_mat = pd.DataFrame(index=select_features, columns=list_gene, dtype=float)
pval_mat = pd.DataFrame(index=select_features, columns=list_gene, dtype=float)

dict_n_rows = {}
# Compute correlations
for gene in list_gene:
    dict_n_rows[gene] = int((df[f'Gene_{gene}'].isna()==False).sum())
    for feat in select_features:
        feature_v, gene_v = df[feat], df[f'Gene_{gene}']
        # Binary ?
        # gene_v = gene_v>dict_thresholds[gene]
        bool_nan = (gene_v.isna()==False)&(feature_v.isna()==False)
        r, p = spearmanr(feature_v[bool_nan],gene_v[bool_nan])
        if np.isnan(r):
            r,p = 0,0
        corr_mat.loc[feat, gene] = r
        pval_mat.loc[feat, gene] = p
        

# Optional: adjust p-values for multiple testing (FDR across all feature–gene pairs)
flat_pvals = pval_mat.values.flatten()
_, qvals, _, _ = multipletests(flat_pvals, method="fdr_bh")
qval_mat = pd.DataFrame(qvals.reshape(pval_mat.shape), index=pval_mat.index, columns=pval_mat.columns)

# Mask nonsignificant correlations (e.g., q > 0.05)
corr_mat = corr_mat.mask(qval_mat.abs() > 0.05)
corr_mat = corr_mat.fillna(0)
# Mask nonsignificant correlations (e.g., q > 0.05)
corr_mat = corr_mat[corr_mat.abs().max(axis=1)>0.2]
# Get clustered order of indices

# Plot heatmap of correlations

g = sns.clustermap(
    corr_mat,
    cmap="coolwarm", center=0,
    cbar_kws={'label': 'Spearman correlation'},
    linewidths=0.5,
    linecolor="lightgray",
    col_cluster=True,
    row_cluster=True,
    figsize=(10, 20)
)

row_order = g.dendrogram_row.reordered_ind
col_order = g.dendrogram_col.reordered_ind

# Reorder original matrix
reordered = corr_mat.iloc[row_order, col_order]

# Show all row and column labels
g.ax_heatmap.set_xticks(np.arange(len(reordered.columns)) + 0.5)
g.ax_heatmap.set_yticks(np.arange(len(reordered.index)) + 0.5)
g.ax_heatmap.set_xticklabels(reordered.columns, rotation=90, fontsize=6)
g.ax_heatmap.set_yticklabels(reordered.index, rotation=0, fontsize=6)






###### MODEL

best_params = {'n_estimators': 100,
 'min_samples_split': 10,
 'min_samples_leaf': 4,
 'max_features': 'sqrt',
 'max_depth': 5,
 'bootstrap': False}

scorer = make_scorer(balanced_accuracy_score)

results = pd.DataFrame()
result_importance = pd.DataFrame()

for i,gene in enumerate(list_gene):

    gene_tresh = dict_thresholds[gene]
    

    X = df[select_features].dropna(axis=1, how='any')
    X = (X - X.mean()) / X.std()
    y = df.loc[X.index, f'Gene_{gene}']
    
    # Filter NAN gene and binary target
    gene_isna = y.isna()
    y_bin = (y[gene_isna==False]>gene_tresh).astype(int)
    X = X.loc[y_bin.index]
    
    if (len(X)>10)&(len(np.unique(y_bin))==2):
        
        # Stratified K-Fold setup
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
      
        class_weights = [1.0, np.sum(y_bin == 0)/np.sum(y_bin == 1)]
        
        rf_model = RandomForestClassifier(
            class_weight='balanced',
            n_jobs=-1
        )

            
        # K FOLDS

        metrics = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_bin), 1):
            print(f"\n Gene {gene} 🧩 Fold {fold}/{n_splits}")
            
            # Use .iloc to subset by row positions
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
        
            model = RandomForestClassifier(class_weight='balanced', **best_params)
            
            # Train model
            model.fit(X_train, y_train)
                
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)[:, 1]
            
            # --- Metrics ---
            auc = roc_auc_score(y_test, y_proba_test)
            acc = balanced_accuracy_score(y_test, y_pred_test)
            
            # Confusion matrices
            cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
            cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
            
            tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
            tn_test,  fp_test,  fn_test,  tp_test  = cm_test.ravel()
            
            # True Positive / Negative Rates
            tpr_train = tp_train / (tp_train + fn_train + 1e-8)  # sensitivity
            tnr_train = tn_train / (tn_train + fp_train + 1e-8)  # specificity
            tpr_test  = tp_test  / (tp_test  + fn_test  + 1e-8)
            tnr_test  = tn_test  / (tn_test  + fp_test  + 1e-8)
            
            metrics.append({
                'fold': fold,
                'AUC': auc,
                'Balanced_Accuracy': acc,
                'TPR_train': tpr_train,
                'TNR_train': tnr_train,
                'TPR_test': tpr_test,
                'TNR_test': tnr_test,
                'nN_train': np.sum(y_train==0),
                'nP_train': np.sum(y_train==1),
                'nN_test': np.sum(y_test==0),
                'nP_test': np.sum(y_test==1),
            })
            
            print(f"AUC: {auc:.3f}, BalAcc: {acc:.3f}")
            print(f"TPR_train: {tpr_train:.3f}, TNR_train: {tnr_train:.3f}")
            print(f"TPR_test : {tpr_test:.3f},  TNR_test : {tnr_test:.3f}")
            
            importances = pd.DataFrame({'importance': model.feature_importances_, 'feature':X_train.columns})
            importances['fold'] = fold
            importances['gene'] = gene
            result_importance =  pd.concat((result_importance, importances),axis=0)
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df['gene'] = gene
        metrics_df.set_index(['gene', 'fold'], inplace=True)
        
        results = pd.concat((results, metrics_df))

print('Train and test on olga3 (average of 5-stratified kfolds)')
results_avg = results.groupby('gene').agg('mean')
# results_avg.to_csv('./model_performance.csv')

results_avg['balanced_accuracy_train'] = (results_avg['TPR_train']+results_avg['TNR_train'])/2
results_avg['balanced_accuracy_test'] = (results_avg['TPR_test']+results_avg['TNR_test'])/2
results_avg['Percentage_Negative_train'] = results_avg['nN_train']/(results_avg['nN_train']+results_avg['nP_train'])
results_avg['Percentage_Positive_train'] = results_avg['nP_train']/(results_avg['nN_train']+results_avg['nP_train'])
results_avg['Percentage_Negative_test'] = results_avg['nN_test']/(results_avg['nN_test']+results_avg['nP_test'])
results_avg['Percentage_Positive_test'] = results_avg['nP_test']/(results_avg['nN_test']+results_avg['nP_test'])
metrics = ['Percentage_Positive', 'Percentage_Negative', 'TPR', 'TNR', 'balanced_accuracy']
n_metrics = len(metrics)
n_genes = results_avg.shape[0]
fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=True)
if n_metrics == 1:
    axes = [axes]

for ax, metric in zip(axes, metrics):
    train_col = f"{metric}_train"
    test_col  = f"{metric}_test"
    
    if (train_col not in results_avg.columns) or (test_col not in results_avg.columns):
        print(f"⚠️ Skipping {metric} — missing train/test columns")
        continue
    
    # Bar positions
    x = np.arange(n_genes)
    width = 0.35
    
    bars_train = ax.bar(x - width/2, results_avg[train_col], width, label='Train')
    bars_test = ax.bar(x + width/2, results_avg[test_col],  width, label='Test')

    # Add value labels on top of each bar
    for bars in [bars_train, bars_test]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, height + 0.01,
                f"{height:.2f}", ha='center', va='bottom', fontsize=9
            )
    
    ax.set_title(metric.replace('_', ' ').title())
    ax.set_xticks(x)
    ax.set_xticklabels(results_avg.index, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
axes[0].legend()
fig.suptitle("Performance metrics per gene", fontsize=14)
plt.tight_layout()
plt.savefig('6_results/performance.png')
plt.show()

result_importance_avg = result_importance.set_index(['gene', 'feature', 'fold']).groupby(['gene', 'feature']).agg('mean')
result_importance_avg


df_heat = result_importance_avg.unstack('feature')['importance']  # pivot so features are columns
# df_heat = df_heat.loc[['ACTB', 'CORO1A', 'GZMB', 'IL2RA', 'XP01']]
df_heat = df_heat.loc[:,df_heat.abs().max(axis=0)>0.01]

g = sns.clustermap(
    df_heat.T,
    cmap="coolwarm", center=0,
    cbar_kws={'label': 'Spearman correlation'},
    linewidths=0.5,
    linecolor="lightgray",
    col_cluster=True,
    row_cluster=True,
    figsize=(10, 10)
)
row_order = g.dendrogram_row.reordered_ind
col_order = g.dendrogram_col.reordered_ind

# Reorder original matrix
reordered = df_heat.iloc[col_order, row_order]

# Show all row and column labels
g.ax_heatmap.set_yticks(np.arange(len(reordered.columns)) + 0.5)
g.ax_heatmap.set_xticks(np.arange(len(reordered.index)) + 0.5)
g.ax_heatmap.set_yticklabels(reordered.columns, rotation=0, fontsize=6)
g.ax_heatmap.set_xticklabels(reordered.index, rotation=90, fontsize=6)

