'''
Features and model shared by the scripts of 5_POC_activation
'''

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, permutation_test_score

from tools.read_yaml import RANDOM_STATE, N_SPLITS, LEARNING_GENES

# Features of the motility of the cell: only the ones computed from the trajectory of its centroid
MOTILITY_FEATURES = ('net_displacement', 'straightness', 'mean_turning_angle', 'delta_distance_', 'alpha')
# Features of the overlap of the cell with itself over time: overlap between consecutive time frames (iou), area
# explored (union of the masks of the cell) and area of the convex hull of this union.
# They mix the displacement of the cell, its changes of shape and its size, so they are a group of their own
OVERLAP_FEATURES = ('iou_', 'area_explored_total', 'convex_hull_area')
# Features of the neighbourhood of the cell: neither its shape nor its motility
NEIGHBOR_FEATURES = ('vor_area_', 'distance_boundary_', 'n_neigh_')


def mean_rna_per_gene(df):
    '''
    Mean number of RNA per gene of each cell: total number of RNA of the genes of the model divided by the number
    of these genes with a value for the cell. Missing if the cell has no value for any gene.
    A matched cell with almost no RNA in every gene (below learning.min_mean_rna) is not used by the models.
    '''
    genes = df[[f'Gene_{gene}' for gene in LEARNING_GENES]]
    n_genes = genes.notna().sum(axis=1)
    return genes.sum(axis=1) / n_genes.where(n_genes > 0)


def select_feature_columns(df):
    '''Columns of the dataset used as features by the models'''
    # Not features: genes, matching, position of the cell in the field of view
    columns_remove = [c for c in df.columns \
                      if c.startswith(('Gene_', 'delta_z_', 'Fish', \
                          'Distance', 'Id_2nd', 'is_on_edge', 'count', 'BF_center_', 'traj_', 'centroid-'))]
    columns_remove += [c for c in df.columns if c.startswith(('moments_', 'inertia_'))]
    columns_remove += ['avg_speed', 'max_speed', 'speed_std', 'elongation']
    # Same information as delta_distance_mean: for a cell present in all the time frames, the length of the
    # trajectory is the mean displacement multiplied by the number of displacements
    columns_remove += ['total_distance']
    # Features without any value are also removed
    return [c for c in df.columns if c not in columns_remove and df[c].notna().any()]


def feature_groups(columns):
    '''
    Split of the features into groups: {'size': [...], 'shape': [...], 'motility': [...], 'overlap': [...], 'neighbors': [...]}
    size: the mean area of the cell. shape: shape descriptors and contour (Fourier) of the cell, summarised over the time frames
    '''
    motility = [c for c in columns if c.startswith(MOTILITY_FEATURES)]
    overlap = [c for c in columns if c.startswith(OVERLAP_FEATURES)]
    neighbors = [c for c in columns if c.startswith(NEIGHBOR_FEATURES)]
    size = [c for c in columns if c == 'area_mean']
    shape = [c for c in columns if c not in size + motility + overlap + neighbors]
    return {'size': size, 'shape': shape, 'motility': motility, 'overlap': overlap, 'neighbors': neighbors}


def normalize_features(features):
    '''
    Features expressed relative to the size of the cell, so that the size is carried by one feature only (area_mean).
    Always applied before training.
        - shape: the lengths (perimeter, axes) are divided by the mean diameter of the cell, and the areas (bounding
          box, variations of the area) by its mean area. The Fourier features are already relative to the size of
          the cell in the dataset (1_concatenate_features.py)
        - overlap: the area explored and the area of the convex hull of the masks of the cell are divided by the mean
          area of the cell. The area explored is the union of the masks of the cell divided by the number of time
          frames, so the result is between 1 / number of time frames (the cell did not move) and 1 (the cell never
          overlaps with itself).
        - motility: the distances travelled are not changed
        - the mean area is kept unchanged: it is the size of the cell. The mean diameter is removed (same information)
        - alpha: the value 0, given by older versions of the dataset when the fit is not reliable, is replaced by a missing value
    '''
    features = features.copy()
    diameter, area = features['equivalent_diameter_area_mean'], features['area_mean']
    lengths = ('perimeter_', 'feret_diameter_max_', 'major_axis_length_', 'minor_axis_length_', 'equivalent_diameter_area_')
    areas = ('area_', 'bbox_area_', 'convex_hull_area') # area_ includes area_explored_total
    for column in features.columns:
        if column == 'area_mean':
            continue
        if column.startswith(lengths):
            features[column] = features[column]/diameter
        elif column.startswith(areas):
            features[column] = features[column]/area
    if 'alpha' in features.columns:
        features['alpha'] = features['alpha'].where(features['alpha'] != 0)
    return features.drop(columns=['equivalent_diameter_area_mean'])


def make_model(**params):
    '''
    Median of the missing values, scaling and random forest.
    In a pipeline: the median and the scaling are computed on the training cells only.
    '''
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scaler', StandardScaler()),
                     ('rf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, **params))])


def search_model(X, y, param_dist, n_iter, n_jobs=1):
    '''Random search of the hyperparameters by cross-validation on (X, y), then fit of the best model on (X, y)'''
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(estimator=make_model(), param_distributions=param_dist, scoring='balanced_accuracy', \
                                n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=RANDOM_STATE)
    search.fit(X, y)
    return search


def fit_and_predict(X_train, y_train, X_test, param_dist, n_iter):
    '''
    Selects the hyperparameters and trains the model on the training cells, then predicts the test cells.
    Runs on one core: the scripts run many of these trainings at the same time, one per core.
    '''
    search = search_model(X_train, y_train, param_dist, n_iter)
    model = search.best_estimator_
    return {'params': search.best_params_, 'importances': model.named_steps['rf'].feature_importances_,
            'pred_train': model.predict(X_train), 'pred_test': model.predict(X_test),
            'proba_test': model.predict_proba(X_test)[:, 1]}


def most_frequent_params(list_params):
    '''The combination of hyperparameters selected the most often among the folds, as arguments of the random forest'''
    keys = [str(sorted(params.items())) for params in list_params]
    params = list_params[keys.index(max(set(keys), key=keys.count))]
    return {key.replace('rf__', ''): value for key, value in params.items()}


def permutation_p_value(X, y, params, n_permutations):
    '''
    Permutation test: p-value of the balanced accuracy of the cross-validation, compared with the same
    cross-validation with shuffled labels. The hyperparameters are fixed. Runs on one core.
    '''
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    _, _, p_value = permutation_test_score(make_model(**params), X, y, cv=cv, scoring='balanced_accuracy', \
                                           n_permutations=n_permutations, n_jobs=1, random_state=RANDOM_STATE)
    return p_value
