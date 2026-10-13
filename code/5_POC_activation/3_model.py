'''
Prediction of the expression of each gene (low or high) from the features of the live cells, with a random forest.

For each gene, two analyses:
    - all the cells: low or high expression, on each side of the threshold of the gene
    - extreme cells: the N_EXTREME cells with the lowest expression against the N_EXTREME cells with the highest

Evaluation:
    - nested cross-validation: the hyperparameters are selected on the training cells of each fold only
      (inner cross-validation), and the model is evaluated on the cells of the fold that were never used.
      The missing values and the scaling of the features are also computed on the training cells only.
    - confidence interval (95 %) of the scores, by bootstrap of the cells
    - permutation test: the same cross-validation with shuffled labels gives the scores expected by chance,
      and a p-value for each gene (corrected for the number of genes)

Saved in 6_results/:
    - performances_24H.csv, performances_24H_extreme.csv: scores of each gene
    - prediction_{gene}.csv: prediction for each live cell
    - feature_importances.csv: importance of each feature for each gene
The figures are generated from these tables by 4_figures_performances.py

The trainings are independent, so they are run at the same time on all the cores of the computer
(learning.n_jobs in config.yaml). The results do not depend on the number of cores.

Run: python3 -m 5_POC_activation.3_model
     python3 -m 5_POC_activation.3_model --normalize_target   (predict the number of RNA divided by the area of the cell; suffix _normalized_target)
     python3 -m 5_POC_activation.3_model --quick       (fast run to check the figures, not for the final results)
'''

# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.matching import passes_ratio_qc_table
from tools.model import select_feature_columns, normalize_features, make_model, fit_and_predict, mean_rna_per_gene

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
from statsmodels.stats.multitest import multipletests

# Options (can be combined):
# --normalize_target: predict the expression adjusted for the size of the cell, instead of the number of RNA
# --quick: only to check that the pipeline and the figures work
# The scores are less reliable and the p-values are not meaningful. Run without --quick for the final results
parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='Fewer hyperparameter combinations, shuffles and resamplings')
parser.add_argument('--normalize_target', action='store_true', help='Predict the expression adjusted for the size of the cell (count divided by area). Saved with the suffix _normalized_target')
args = parser.parse_args()
if args.quick:
    N_ITER, N_PERMUTATIONS, N_BOOTSTRAP = 3, 20, 100
    print('QUICK MODE: 3 hyperparameter combinations, 20 shuffles, 100 resamplings. Not for the final results')

df = pd.read_csv('6_results/dataset.csv', index_col=['exp_path', 'condition', 'cell_bf'])
list_gene = LEARNING_GENES


##### FILTER DATA
# Quality control of the matching: the pair is much closer than the second closest candidate
bool_qc = passes_ratio_qc_table(df, RATIO_QC)
# Is not on edge
bool_edge = df['is_on_edge'] == False
# With all the time frames
bool_count = df['count'] == N_FRAMES
# Enough RNA: the cells with almost no RNA in every gene are removed
bool_rna = mean_rna_per_gene(df) >= MIN_MEAN_RNA
print(f'{int((bool_qc&bool_edge&bool_count&~bool_rna).sum())} matched cells removed: fewer than {MIN_MEAN_RNA} RNA per gene')
bool_match = bool_qc&bool_edge&bool_count&bool_rna
df.loc[~bool_match, 'Fish'] = None
for col in list_gene:
    df.loc[~bool_match, f'Gene_{col}'] = None

####### SELECT FEATURE
select_features = select_feature_columns(df)


# Table of the features of all the live cells
df_features = df[select_features]
# Features relative to the size of the cell: the size is carried by the mean area only (see normalize_features)
df_features = normalize_features(df_features)
suffix = ''
select_features = list(df_features.columns)
if args.normalize_target:
    suffix += '_normalized_target'
    print(f'NORMALIZED TARGET: number of RNA divided by the area of the cell. Results saved with the suffix {suffix}')

# Thresholds between low and high expressing genes
dict_thresholds = GENE_THRESHOLDS
# Hyperparameters of the random forest explored by the random search
param_dist = {f'rf__{key}': value for key, value in PARAM_DIST.items()}


##### MODEL AND EVALUATION

def rates(y_true, y_pred):
    '''True positive rate (sensitivity) and true negative rate (specificity)'''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn + 1e-8), tn / (tn + fp + 1e-8)


def bootstrap_interval(y_true, y_pred, y_proba, n_bootstrap, rng):
    '''Confidence interval (95 %) of the balanced accuracy and of the AUC, by bootstrap of the cells'''
    y_true, y_pred, y_proba = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_proba)
    scores = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[index])) == 2:
            scores.append([balanced_accuracy_score(y_true[index], y_pred[index]), roc_auc_score(y_true[index], y_proba[index])])
    return np.percentile(scores, [2.5, 97.5], axis=0) # rows: low and high. columns: balanced accuracy and AUC


def outer_folds(y_bin):
    '''Folds of the nested cross-validation: list of (indices of the training cells, indices of the test cells)'''
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(outer_cv.split(np.zeros(len(y_bin)), y_bin))


def trainings(X, y_bin):
    '''
    Trainings needed to evaluate one gene: {name of the training: (training features, training labels, test features)}
    One per fold of the nested cross-validation
    '''
    tasks = {}
    for fold, (train_idx, test_idx) in enumerate(outer_folds(y_bin), 1):
        tasks[('fold', fold)] = (X.iloc[train_idx], y_bin.iloc[train_idx], X.iloc[test_idx])
    return tasks


def evaluate(X, y_bin, name, fitted):
    '''
    Nested cross-validation, bootstrap and permutation test for one gene.
    X: features of the cells with a label. y_bin: label (0: low, 1: high)
    fitted: results of the trainings of the gene (see trainings)
    Returns: scores (dict), prediction of each cell when it was in the test fold, importance of the features
    '''
    rng = np.random.default_rng(RANDOM_STATE)
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Nested cross-validation: the hyperparameters of each fold were selected on its training cells only
    folds, list_params, importances = [], [], []
    y_pred = pd.Series(np.nan, index=X.index)
    y_proba = pd.Series(np.nan, index=X.index)
    for fold, (train_idx, test_idx) in enumerate(outer_folds(y_bin), 1):
        y_train, y_test = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
        result = fitted[('fold', fold)]
        list_params.append(result['params'])
        importances.append(result['importances'])

        y_pred.iloc[test_idx] = result['pred_test']
        y_proba.iloc[test_idx] = result['proba_test']
        tpr_train, tnr_train = rates(y_train, result['pred_train'])
        tpr_test, tnr_test = rates(y_test, y_pred.iloc[test_idx])
        folds.append({
            'AUC': roc_auc_score(y_test, y_proba.iloc[test_idx]),
            'Balanced_Accuracy': balanced_accuracy_score(y_test, y_pred.iloc[test_idx]),
            'TPR_train': tpr_train, 'TNR_train': tnr_train, 'TPR_test': tpr_test, 'TNR_test': tnr_test,
            'nN_train': np.sum(y_train==0), 'nP_train': np.sum(y_train==1),
            'nN_test': np.sum(y_test==0), 'nP_test': np.sum(y_test==1),
        })
        print(f"{name} fold {fold}/{N_SPLITS}: AUC {folds[-1]['AUC']:.3f}, balanced accuracy {folds[-1]['Balanced_Accuracy']:.3f}")
    scores = pd.DataFrame(folds).mean().to_dict()

    # Confidence interval, from the predictions of all the test folds
    interval = bootstrap_interval(y_bin, y_pred, y_proba, N_BOOTSTRAP, rng)
    scores.update({'Balanced_Accuracy_ci_low': interval[0, 0], 'Balanced_Accuracy_ci_high': interval[1, 0], \
                   'AUC_ci_low': interval[0, 1], 'AUC_ci_high': interval[1, 1]})

    # Permutation test: same cross-validation with shuffled labels.
    # The hyperparameters are fixed (the ones selected the most often), for the true and for the shuffled labels
    params = pd.Series([str(sorted(p.items())) for p in list_params]).mode()[0]
    params = {key.replace('rf__', ''): value for key, value in next(p for p in list_params if str(sorted(p.items())) == params).items()}
    score, permuted_scores, p_value = permutation_test_score(make_model(**params), X, y_bin, cv=outer_cv, scoring='balanced_accuracy', \
                                                             n_permutations=N_PERMUTATIONS, n_jobs=N_JOBS, random_state=RANDOM_STATE)
    scores.update({'Balanced_Accuracy_fixed_parameters': score, 'Balanced_Accuracy_permuted': permuted_scores.mean(), \
                   'Balanced_Accuracy_permuted_95': np.percentile(permuted_scores, 95), 'p_value': p_value})

    importances = pd.Series(np.mean(importances, axis=0), index=X.columns)
    return scores, y_pred, importances


def add_significance(results):
    '''
    q_value: p-value of the permutation test corrected for the number of genes (Benjamini-Hochberg)
    significant: True if the model of the gene is better than chance, that is if its q_value is below 0.05
    '''
    results['q_value'] = multipletests(results['p_value'], method='fdr_bh')[1]
    results['significant'] = results['q_value'] < 0.05
    return results


##### ALL THE CELLS, THEN ONLY THE EXTREME VALUES
# CD8 positive cells: at least two RNA of CD8
is_cd8 = df['Gene_CD8'] > 1
# Mean area of the live cell over the movie, relative to the median area of the cells used by the model
relative_area = df['area_mean'] / df.loc[is_cd8, 'area_mean'].median()

# 1. Data of each analysis: {(gene, 'all' or 'extreme'): (features, labels)}, and the list of all the trainings
datasets, expression, tasks = {}, {}, {}
for gene in list_gene:
    y = df.loc[is_cd8, f'Gene_{gene}'].dropna()
    if args.normalize_target:
        # Expression adjusted for the size of the cell: number of RNA per unit of area, expressed as the number of RNA
        # that the cell would have with the median area. The threshold of the gene can then be used unchanged
        y = y / relative_area.loc[y.index]

    # All the cells: low or high expression, on each side of the threshold of the gene
    y_bin = (y >= dict_thresholds[gene]).astype(int)
    if min(np.sum(y_bin==0), np.sum(y_bin==1)) < 2*N_SPLITS:
        print(f'{gene}: not enough cells with low or high expression ({np.sum(y_bin==0)} low, {np.sum(y_bin==1)} high), skipped')
        continue
    expression[gene] = y
    datasets[(gene, 'all')] = (df_features.loc[y_bin.index], y_bin)
    # Model trained on all the cells with a label, to predict the live cells without a label
    tasks[(gene, 'all', 'final')] = (df_features.loc[y_bin.index], y_bin, df_features)

    # Extreme values: the cells with the lowest expression against the cells with the highest expression
    n_extreme = min(N_EXTREME, len(y)//2)
    y_sorted = y.sort_values()
    y_bin = pd.Series(np.concatenate((np.zeros(n_extreme), np.ones(n_extreme))).astype(int), \
                      index=y_sorted.index[:n_extreme].append(y_sorted.index[-n_extreme:]))
    if n_extreme >= 2*N_SPLITS:
        datasets[(gene, 'extreme')] = (df_features.loc[y_bin.index], y_bin)

for key, (X, y_bin) in datasets.items():
    for name, task in trainings(X, y_bin).items():
        tasks[key + name] = task

# 2. All the trainings (selection of the hyperparameters included), at the same time on the cores of the computer
print(f'{len(tasks)} trainings on {N_JOBS if N_JOBS > 0 else os.cpu_count()} cores')
outputs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(fit_and_predict)(*task, param_dist, N_ITER) for task in tasks.values())
fitted = dict(zip(tasks.keys(), outputs))

# 3. Scores of each gene
results, results_extreme, result_importance = {}, {}, {}
for (gene, analysis), (X, y_bin) in datasets.items():
    fitted_gene = {key[2:]: value for key, value in fitted.items() if key[:2] == (gene, analysis)}
    if analysis == 'extreme':
        results_extreme[gene], _, _ = evaluate(X, y_bin, f'{gene} (extreme)', fitted_gene)
        continue
    results[gene], y_pred, result_importance[gene] = evaluate(X, y_bin, gene, fitted_gene)

    # Prediction for all the live cells, with the true value when available. Used by the figures.
    # Cells with a label: prediction when the cell was in the test fold. Other cells: model trained on all the cells with a label
    pred_df = pd.DataFrame({f'pred_{gene}': fitted[(gene, 'all', 'final')]['pred_test'], \
                            f'Gene_{gene}': expression[gene].reindex(df.index), \
                            f'Gene_bin_{gene}': y_bin.reindex(df.index)}, index=df.index)
    pred_df.loc[y_pred.index, f'pred_{gene}'] = y_pred
    pred_df.to_csv(f'6_results/prediction_{gene}{suffix}.csv')

results_avg = add_significance(pd.DataFrame(results).T.rename_axis('gene'))
results_avg_extreme = add_significance(pd.DataFrame(results_extreme).T.rename_axis('gene'))

# SAVE THE RESULTS
results_avg.to_csv(f'6_results/performances_24H{suffix}.csv')
results_avg_extreme.to_csv(f'6_results/performances_24H_extreme{suffix}.csv')
pd.DataFrame(result_importance).rename_axis('feature').to_csv(f'6_results/feature_importances{suffix}.csv')
print(results_avg[['AUC', 'Balanced_Accuracy', 'Balanced_Accuracy_ci_low', 'Balanced_Accuracy_ci_high', \
                   'Balanced_Accuracy_permuted', 'p_value', 'q_value', 'significant']])
