# Import local tools
from tools.read_yaml import *
from tools.imports import *

from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import shap

from matplotlib import rc
rc('font', family='Helvetica')

df = pd.read_csv('6_results/dataset.csv', index_col=['exp_path', 'condition', 'cell_bf'])
list_gene = ['GZMB', 'CD69', 'CCR7', 'GBP4', 'ACTB', 'CORO1A', 'IRF8', 'IL2RA', 'TNF', 'CD8']


##### FILTER DATA
# Bool
distance_tresh, distance_2nd_tresh = 100,50
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

columns_remove_features = [c for c in df.columns \
                if c.startswith(('moments_', 'inertia_'))]
select_features = [x for x in select_features if x not in columns_remove_features]
select_features = [x for x in select_features if x not in ['avg_speed', 'max_speed','speed_std','elongation']]


n_components = 2
X = df[select_features].fillna(df[select_features].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = umap.UMAP(n_components=n_components)
x_transform = pca.fit_transform(X_scaled)
df_2 = pd.DataFrame(x_transform, columns=[f'umap_{i}' for i in range(n_components)], index=X.index)
df = pd.concat((df, df_2), axis=1)


pca2 = umap.UMAP(n_components=n_components)
Y = df[[f'Gene_{gene}' for gene in list_gene]].dropna(axis=0, how='any')
Y = Y[Y['Gene_CD8']>1]   # Filter CD8 negative! 
scaler_y = StandardScaler()
Y_scaled = scaler_y.fit_transform(Y)
x_transform2 = pca2.fit_transform(Y_scaled)
df_2 = pd.DataFrame(x_transform2, columns=[f'umap2_{i}' for i in range(n_components)], index=Y.index)
df = pd.concat((df, df_2), axis=1)


# Thresholds between low and high expressing genes
dict_thresholds = {'CORO1A': 40,
                'TNF': 5,
                'GZMB': 11,
                'ACTB': 80,
                'CD69': 4,
                'IRF8': 4,
                'CCR7': 30,
                'GBP4': 5,
                'XP01': 3,
                'IFNG': 2,
                'IL2RA': 5,
                'CD8':2}

param_dist = {
    'n_estimators': [100],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}
scorer = make_scorer(balanced_accuracy_score)


results = pd.DataFrame()
result_importance = pd.DataFrame()

for i,gene in enumerate(list_gene):

    gene_tresh = dict_thresholds[gene]
    
    # Prepare data for training
    X = df[select_features].fillna(df[select_features].median())
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled,columns=X.columns, index=X.index)
    y = df.loc[X.index, f'Gene_{gene}']

    # Filter CD8 negative! 
    y_CD8 = df.loc[X.index, f'Gene_CD8']
    filter_CD8 = y_CD8[y_CD8.isna()==False]>1 # at least two value of CD8 RNA
    filter_CD8 = filter_CD8[filter_CD8]
    X,y = X.loc[filter_CD8.index], y.loc[filter_CD8.index]

    # Filter NAN gene and binarize target
    gene_isna = y.isna()
    y_bin = (y[gene_isna==False]>=gene_tresh).astype(int)
    X = X.loc[y_bin.index]

  
    if (len(X)>10)&(len(np.unique(y_bin))==2):
        
        # Stratified K-Fold setup
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        class_weights = [1.0, np.sum(y_bin == 0)/np.sum(y_bin == 1)]
        class_weights = {i:class_weights[i] for i in range(2)}

        
        rf_model = RandomForestClassifier(
            class_weight='balanced',
            # class_weight=class_weights,
            n_jobs=-1
        )

        search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_dist,
            scoring=scorer,   # balanced accuracy
            n_iter=30,        # number of random combinations
            cv=skf,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X, y_bin)
        best_params = search.best_params_
        print(best_params)
            
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


            # --- ROC curve on test set ---
            fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
            auc_test = roc_auc_score(y_test, y_proba_test)
            
            metrics.append({
                'fold': fold,
                'AUC_Test': auc_test,
                'AUC': auc,
                'FPR': fpr,
                'TPR': tpr,
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
            
            # # Feature importance
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            interaction_values = explainer.shap_interaction_values(X_train)
            sv = shap_values[1]               # SHAP values for positive class
            iv = interaction_values[1]        # interaction values for positive class

        
            importances = pd.DataFrame({'importance': model.feature_importances_, 'feature':X_train.columns})
            importances['fold'] = fold
            importances['gene'] = gene
            result_importance =  pd.concat((result_importance, importances),axis=0)
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df['gene'] = gene
        metrics_df.set_index(['gene', 'fold'], inplace=True)
        
        results = pd.concat((results, metrics_df))




##### NOW ONLY EXTREME VALUES: most and least expression gene expression

results_extreme = pd.DataFrame()
result_importance = pd.DataFrame()

for i,gene in enumerate(list_gene):

    gene_tresh = dict_thresholds[gene]
    
    # Prepare data
    X = df[select_features].fillna(df[select_features].median())
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled,columns=X.columns, index=X.index)
    y = df.loc[X.index, f'Gene_{gene}']

    # Filter CD8 negative! 
    y_CD8 = df.loc[X.index, f'Gene_CD8']
    filter_CD8 = y_CD8[y_CD8.isna()==False]>1
    filter_CD8 = filter_CD8[filter_CD8]
    X,y = X.loc[filter_CD8.index], y.loc[filter_CD8.index]

    # # Keep extreme values only
    n_extreme = 100
    y = pd.concat((y[~y.isna()].sort_values().iloc[:n_extreme], y[~y.isna()].sort_values().iloc[-n_extreme:]))

    # Filter NAN gene and binary target
    gene_isna = y.isna()
    y_bin = pd.Series(np.concatenate((np.zeros(n_extreme), np.ones(n_extreme))), index=y.index)
    X = X.loc[y_bin.index]

    print(gene, (y_bin==0).sum(),(y_bin==1).sum())
    
    if (len(X)>10)&(len(np.unique(y_bin))==2):
        
        # Stratified K-Fold setup
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        class_weights = [1.0, np.sum(y_bin == 0)/np.sum(y_bin == 1)]
        class_weights = {i:class_weights[i] for i in range(2)}

        
        rf_model = RandomForestClassifier(
            class_weight='balanced',
            # class_weight=class_weights,
            n_jobs=-1
        )

        search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_dist,
            scoring=scorer,   # balanced accuracy
            n_iter=30,        # number of random combinations
            cv=skf,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X, y_bin)
        best_params = search.best_params_
        print(best_params)
            
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


            # --- ROC curve on test set ---
            fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
            auc_test = roc_auc_score(y_test, y_proba_test)
            
            metrics.append({
                'fold': fold,
                'AUC_Test': auc_test,
                'AUC': auc,
                'FPR': fpr,
                'TPR': tpr,
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
            
            # # Feature importance
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            interaction_values = explainer.shap_interaction_values(X_train)
            sv = shap_values[1]               # SHAP values for positive class
            iv = interaction_values[1]        # interaction values for positive class

            importances = pd.DataFrame({'importance': model.feature_importances_, 'feature':X_train.columns})
            importances['fold'] = fold
            importances['gene'] = gene
            result_importance =  pd.concat((result_importance, importances),axis=0)
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df['gene'] = gene
        metrics_df.set_index(['gene', 'fold'], inplace=True)
        
        results_extreme = pd.concat((results_extreme, metrics_df))


results_avg = results[['AUC', 'Balanced_Accuracy', 'TPR_train', 'TNR_train', 'TPR_test', 'TNR_test', 'nN_train', 'nP_train', 'nN_test', 'nP_test']].groupby('gene').agg('mean')
results_avg_extreme = results_extreme[['AUC', 'Balanced_Accuracy', 'TPR_train', 'TNR_train', 'TPR_test', 'TNR_test', 'nN_train', 'nP_train', 'nN_test', 'nP_test']].groupby('gene').agg('mean')


# SAVE BOTH RESULTS
results_avg.to_csv('6_results/performances_24H.csv')
results_avg_extreme.to_csv('6_results/performances_extreme_24H.csv')


### LOAD RESULTS
results_avg = pd.read_csv('6_results/performances_24H.csv', index_col='gene')
results_avg_extreme = pd.read_csv('6_results/performances_24H_extreme.csv', index_col='gene')

results_avg['balanced_accuracy_train'] = (results_avg['TPR_train']+results_avg['TNR_train'])/2
results_avg['balanced_accuracy_test'] = (results_avg['TPR_test']+results_avg['TNR_test'])/2
results_avg['Percentage_Negative_train'] = results_avg['nN_train']/(results_avg['nN_train']+results_avg['nP_train'])
results_avg['Percentage_Positive_train'] = results_avg['nP_train']/(results_avg['nN_train']+results_avg['nP_train'])
results_avg['Percentage_Negative_test'] = results_avg['nN_test']/(results_avg['nN_test']+results_avg['nP_test'])
results_avg['Percentage_Positive_test'] = results_avg['nP_test']/(results_avg['nN_test']+results_avg['nP_test'])
results_avg['Number_negative'] = results_avg['nN_train']+results_avg['nN_test']
results_avg['Number_positive'] = results_avg['nP_train']+results_avg['nP_test']


results_avg_extreme['balanced_accuracy_train'] = (results_avg_extreme['TPR_train']+results_avg_extreme['TNR_train'])/2
results_avg_extreme['balanced_accuracy_test'] = (results_avg_extreme['TPR_test']+results_avg_extreme['TNR_test'])/2
results_avg_extreme['Percentage_Negative_train'] = results_avg_extreme['nN_train']/(results_avg_extreme['nN_train']+results_avg_extreme['nP_train'])
results_avg_extreme['Percentage_Positive_train'] = results_avg_extreme['nP_train']/(results_avg_extreme['nN_train']+results_avg_extreme['nP_train'])
results_avg_extreme['Percentage_Negative_test'] = results_avg_extreme['nN_test']/(results_avg_extreme['nN_test']+results_avg_extreme['nP_test'])
results_avg_extreme['Percentage_Positive_test'] = results_avg_extreme['nP_test']/(results_avg_extreme['nN_test']+results_avg_extreme['nP_test'])
results_avg_extreme['Number_negative'] = results_avg_extreme['nN_train']+results_avg_extreme['nN_test']
results_avg_extreme['Number_positive'] = results_avg_extreme['nP_train']+results_avg_extreme['nP_test']

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
plt.show()



results_avg_extreme = results_avg_extreme[results_avg_extreme.index!='CD8']

metrics = ['Number_positive', 'Number_negative', 'TPR_test', 'TNR_test', 'balanced_accuracy_test']
name_metrics = ['Number of positive', 'Number of negative', 'True positive rate', 'True negative rate', 'Balanced accuracy']
fig, axes = plt.subplots(1, 5, figsize=(5 * n_metrics, 5), sharey=False)
if n_metrics == 1:
    axes = [axes]

for i,(ax, metric, metric_name) in enumerate(zip(axes, metrics[:],name_metrics)):
    # Bar positions
    x = np.arange(n_genes)
    width = 0.4
    
    if i>=2:
        # bars_train = ax.bar(x - width/2, results_avg[train_col], width, label='Train')
        bars_test = ax.bar(x-0.2, results_avg[metric]*100,  width, label='All samples')
        bars_test_extreme = ax.bar(x+0.2, results_avg_extreme[metric]*100,  width, label='Extreme')
    else:
        # bars_train = ax.bar(x - width/2, results_avg[train_col], width, label='Train')
        bars_test = ax.bar(x-0.2, (results_avg[metric]).astype('int'),  width, label='All samples')
        bars_test_extreme = ax.bar(x+0.2, (results_avg_extreme[metric]).astype('int'),  width, label='Extreme')

    # Add value labels on top of each bar
    for bars in [bars_test,bars_test_extreme]:
        for bar in bars:
            height = bar.get_height()
            if i>=2:
                text = f"{height:.1f}"
            else:
                text = height
            ax.text(
                bar.get_x() + bar.get_width()/2, height + 0.02,
                text, ha='center', va='bottom', fontsize=7
            )

    ax.set_title(metric_name.replace('_', ' ').title())
    ax.set_xticks(x)
    ax.set_xticklabels(results_avg.index, rotation=45, ha='right')
for ax in axes[-3:]:
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
# axes[0].legend()
# fig.suptitle("Performance metrics per gene", fontsize=14)
plt.tight_layout()
plt.savefig('.6_results/performance.png', format='png', dpi=300)
plt.show()





