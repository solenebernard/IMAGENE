'''
What does each group of features predict? Same prediction as 3_model.py (low or high expression of each gene,
random forest, nested cross-validation), trained with different groups of features:
    - size:           the mean area of the cell
    - shape:          shape descriptors and contour (Fourier) of the cell, relative to its size
    - motility:       trajectory of the centroid of the cell only (distances, straightness, turning angle, alpha)
    - overlap:        overlap of the cell with itself between time frames (iou) and area explored relative to its size:
                      they mix the displacement and the changes of shape of the cell
    - neighbourhood:  Voronoi area, distance to the boundary and number of neighbours within several distances
and with combinations of them: size + motility, size + shape, size + motility + shape, and all the groups together.

As in 3_model.py, the shape features are expressed relative to the size of the cell (normalize_features in
tools/model.py), so that the size and the shape are separate groups. The groups are defined in tools/model.py

The models of a gene are tested on the same cells and the same folds, and each score is given with a confidence
interval (95 %) by bootstrap of the cells.

Each model is also compared with chance, as in 3_model.py: permutation test (same cross-validation with shuffled
labels, with the hyperparameters selected the most often), corrected for the number of genes. Columns p_value,
q_value (corrected p-value) and significant (q_value below 0.05).

The trainings are independent, so they are run at the same time on all the cores of the computer
(learning.n_jobs in config.yaml). The results do not depend on the number of cores.

Saved in 6_results/:
    - performances_feature_groups.csv: scores of each gene and each group of features
The figure is generated from these tables by 6_figures_feature_groups.py

Run: python3 -m 5_POC_activation.5_model_feature_groups
     python3 -m 5_POC_activation.5_model_feature_groups --quick   (fast run to check the figures, not for the final results)
'''

# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.matching import passes_ratio_qc_table
from tools.model import select_feature_columns, feature_groups, normalize_features, fit_and_predict, mean_rna_per_gene, \
                        most_frequent_params, permutation_p_value

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='Fewer hyperparameter combinations, shuffles and resamplings')
args = parser.parse_args()
if args.quick:
    N_ITER, N_PERMUTATIONS, N_BOOTSTRAP = 3, 20, 100
    print('QUICK MODE: 3 hyperparameter combinations, 20 shuffles, 100 resamplings. Not for the final results')

df = pd.read_csv('6_results/dataset.csv', index_col=['exp_path', 'condition', 'cell_bf'])

##### FILTER DATA: same cells as 3_model.py
# Quality control of the matching, not on the edge, with all the time frames, enough RNA (not a cell with almost no
# RNA in every gene), CD8 positive (at least two RNA of CD8)
is_used = passes_ratio_qc_table(df, RATIO_QC) & (df['is_on_edge'] == False) & (df['count'] == N_FRAMES) \
          & (mean_rna_per_gene(df) >= MIN_MEAN_RNA) & (df['Gene_CD8'] > 1)
df = df[is_used]

##### GROUPS OF FEATURES
# Features relative to the size of the cell: the size is carried by the mean area only
df_features = normalize_features(df[select_feature_columns(df)])
groups = feature_groups(df_features.columns)
groups = {'size': groups['size'], 'shape': groups['shape'], 'motility': groups['motility'], \
          'overlap': groups['overlap'], 'neighbourhood': groups['neighbors']}
# A group without any feature in the dataset is skipped (for example the neighbourhood, if its table was not computed)
groups = {name: columns for name, columns in groups.items() if len(columns) > 0}
# The models: each group alone, then combinations
combinations = [[name] for name in groups] + [['size', 'motility'], ['size', 'shape'], ['size', 'motility', 'shape'], list(groups)]
feature_sets = {' + '.join(names) if len(names) < len(groups) else 'all': sum((groups[name] for name in names), []) for names in combinations}
# Features of each set in the order of the dataset, as in 3_model.py: the random forest depends on the order of the
# features, so that the set with all the families gives exactly the model of 3_model.py
feature_sets = {name: [c for c in df_features.columns if c in columns] for name, columns in feature_sets.items()}
for name, columns in feature_sets.items():
    print(f'{name}: {len(columns)} features')

param_dist = {f'rf__{key}': value for key, value in PARAM_DIST.items()}


def scores(y_true, y_pred, y_proba):
    return np.array([balanced_accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_proba)])


##### 1. LABELS OF EACH GENE AND LIST OF ALL THE TRAININGS
# One training per gene, per set of features and per fold of the nested cross-validation.
# The folds are the same for all the sets of features of a gene.
labels, tasks = {}, {}
for gene in LEARNING_GENES:
    y = df[f'Gene_{gene}'].dropna()
    y_bin = (y >= GENE_THRESHOLDS[gene]).astype(int)
    n_low, n_high = int(np.sum(y_bin==0)), int(np.sum(y_bin==1))
    if min(n_low, n_high) < 2*N_SPLITS:
        print(f'{gene}: not enough cells with low or high expression ({n_low} low, {n_high} high), skipped')
        continue
    labels[gene] = y_bin
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y_bin)), y_bin)):
        for name, columns in feature_sets.items():
            X = df_features.loc[y_bin.index, columns]
            tasks[(gene, name, fold)] = (test_idx, (X.iloc[train_idx], y_bin.iloc[train_idx], X.iloc[test_idx]))

##### 2. ALL THE TRAININGS, AT THE SAME TIME ON THE CORES OF THE COMPUTER
# The hyperparameters are selected on the training cells of each fold only
print(f'{len(tasks)} trainings on {N_JOBS if N_JOBS > 0 else os.cpu_count()} cores')
outputs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(fit_and_predict)(*task, param_dist, N_ITER) for _, task in tasks.values())

# Prediction of each cell when it is in the test fold: label and probability of a high expression
predictions = {(gene, name): (np.zeros(len(y_bin)), np.zeros(len(y_bin))) for gene, y_bin in labels.items() for name in feature_sets}
for (gene, name, fold), (test_idx, _), output in zip(tasks.keys(), tasks.values(), outputs):
    predictions[(gene, name)][0][test_idx] = output['pred_test']
    predictions[(gene, name)][1][test_idx] = output['proba_test']

##### 3. COMPARISON WITH CHANCE: PERMUTATION TEST OF EACH MODEL, AT THE SAME TIME ON THE CORES OF THE COMPUTER
# The hyperparameters are fixed: the ones selected the most often among the folds of the model
params = {key: [] for key in predictions}
for (gene, name, fold), output in zip(tasks.keys(), outputs):
    params[(gene, name)].append(output['params'])
print(f'{len(params)} permutation tests of {N_PERMUTATIONS} shuffles')
p_values = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(permutation_p_value)( \
    df_features.loc[labels[gene].index, feature_sets[name]], labels[gene], most_frequent_params(params[(gene, name)]), N_PERMUTATIONS) \
    for gene, name in params)
p_values = dict(zip(params.keys(), p_values))

##### 4. SCORES OF EACH GENE
rows = []
for gene, y_bin in labels.items():
    y_true = y_bin.values
    n_low, n_high = int(np.sum(y_true==0)), int(np.sum(y_true==1))

    # Bootstrap of the cells
    rng = np.random.default_rng(RANDOM_STATE)
    resampled = {name: [] for name in feature_sets}
    for _ in range(N_BOOTSTRAP):
        index = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[index])) == 2:
            for name in feature_sets:
                y_pred, y_proba = predictions[(gene, name)]
                resampled[name].append(scores(y_true[index], y_pred[index], y_proba[index]))
    resampled = {name: np.array(values) for name, values in resampled.items()}

    # Scores of each set of features. Columns of the arrays: balanced accuracy, AUC
    values = {name: scores(y_true, *predictions[(gene, name)]) for name in feature_sets}
    for name, columns in feature_sets.items():
        low, high = np.percentile(resampled[name], [2.5, 97.5], axis=0)
        rows.append({'gene': gene, 'features': name, 'n_features': len(columns), 'n_low': n_low, 'n_high': n_high,
                     'Balanced_Accuracy': values[name][0], 'Balanced_Accuracy_ci_low': low[0], 'Balanced_Accuracy_ci_high': high[0],
                     'AUC': values[name][1], 'AUC_ci_low': low[1], 'AUC_ci_high': high[1], 'p_value': p_values[(gene, name)]})

# SAVE THE RESULTS
results = pd.DataFrame(rows).set_index(['gene', 'features'])
# Better than chance. q_value: p-value corrected for the number of genes (Benjamini-Hochberg), for each set of features
results['q_value'] = results.groupby(level='features')['p_value'].transform(lambda p: multipletests(p, method='fdr_bh')[1])
results['significant'] = results['q_value'] < 0.05
results.to_csv('6_results/performances_feature_groups.csv')
pd.set_option('display.width', 250)
print('Balanced accuracy (%)')
print((results['Balanced_Accuracy'].unstack('features')[list(feature_sets)]*100).round(1))
print('Significantly better than chance')
print(results['significant'].unstack('features')[list(feature_sets)].replace({True: '*', False: ''}))
