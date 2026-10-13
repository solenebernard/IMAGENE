# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

from scipy.spatial import ConvexHull
import seaborn as sns
import umap

def get_features_efd(efd, order):
    """
    Compute magnitudes and elliptic areas from EFDs.
    
    Parameters:
        efd: np.ndarray of shape (n, 4)
            Each row: [a_n, b_n, c_n, d_n]
    
    Returns:
        M_norm: np.ndarray of shape (n,)
            Magnitude for each harmonic
        A_norm: np.ndarray of shape (n,)
            Elliptic area for each harmonic
    """
    # Extract coefficients
    a = efd[f'a_{order}']
    b = efd[f'b_{order}']
    c = efd[f'c_{order}']
    d = efd[f'd_{order}']
    
    # Magnitude of each harmonic
    M = np.sqrt(a**2 + b**2 + c**2 + d**2)
    
    # Semi-axes of the ellipse for each harmonic
    term1 = a**2 + b**2 + c**2 + d**2
    term2 = np.sqrt((a**2 + b**2 - c**2 - d**2)**2 + 4*(a*c + b*d)**2)
    
    lambda1 = np.sqrt(0.5 * (term1 + term2))
    lambda2 = np.sqrt(0.5 * np.clip(term1 - term2, a_min = 0, a_max=1e8)) # safety clip
    # print(term1 - term2)
    
    # Elliptic area
    A = np.pi * lambda1 * lambda2
    
    return M, A


def compute_metrics_trajectory_fourier(traj):
    if len(traj)==1:
        return {
            "total_distance": None,
            "net_displacement": None,
            "straightness": None,
            "avg_speed": None,
            "max_speed": None,
            "speed_std": None,
            "mean_turning_angle": None,
            "elongation": None
        }
    
    elif len(traj)>=2:
        deltas = np.diff(traj, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        
        # Kinematic
        total_dist = distances.sum()
        net_disp = np.linalg.norm(traj[-1] - traj[0])
       
        speeds = distances
        avg_speed = speeds.mean()
        max_speed = speeds.max()
        
    
        # Geometric
        if len(traj) >= 3:
            # Angular change
            v1 = deltas[:-1]
            v2 = deltas[1:]
            angles = np.arccos(
                np.clip(
                    np.sum(v1 * v2, axis=1) / 
                    (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)), 
                    -1.0, 1.0)
            )
            mean_turning_angle = np.mean(np.abs(angles))
            speed_std = speeds.std()
            straightness = net_disp / total_dist if total_dist != 0 else 0
        else:
            mean_turning_angle = None
            speed_std = None
            straightness = None
        
        # PCA for elongation
        centered = traj - traj.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        elongation = eigvals[-1] / eigvals[0] if eigvals[0] > 0 else 0
        
        return {
            "total_distance": total_dist,
            "net_displacement": net_disp,
            "straightness": straightness,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "speed_std": speed_std,
            "mean_turning_angle": mean_turning_angle,
            "elongation": elongation
        }


def reduce_features_fourier_finetune(df_fourier, order=10):
    list_feats = [get_features_efd(df_fourier, o) for o in range(order)]
    df_magnitude = pd.DataFrame({f'magnitude_{i}':x[0] for i,x in enumerate(list_feats)})
    df_elliptic_area = pd.DataFrame({f'elliptic_area_{i}':x[1] for i,x in enumerate(list_feats)})
    df_fourier = df_fourier.join(df_magnitude).join(df_elliptic_area)
    
    df_group = df_fourier.groupby('cell_bf')
    new_df = df_group.aggregate(['mean', 'std', 'max', 'min'])
    new_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in new_df.columns]
    for o in range(order):
        new_df = new_df.drop(columns=[c for c in new_df.columns \
            if c.startswith((f'a_{o}', f'b_{o}', f'c_{o}', f'd_{o}'))])
    return(new_df)  

def reduce_features_cell(df_cell):
    # Circularity of the cell at each time frame: 4 pi area / perimeter^2. 1 for a disc, lower for an elongated
    # or irregular cell, independent of the size of the cell (missing value if the perimeter is 0)
    df_cell = df_cell.copy()
    df_cell['circularity'] = 4*np.pi*df_cell['area']/df_cell['perimeter'].where(df_cell['perimeter']>0)**2
    df_group = df_cell.groupby('cell_bf')
    new_df = df_group.aggregate(['mean', 'std', 'max', 'min'])
    new_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in new_df.columns]
    # Compute trajectory characteristics
    new_df2 = pd.DataFrame()
    rows = []
    for _, traj in df_group[['centroid-0', 'centroid-1']]:
        new_dict = compute_metrics_trajectory_fourier(np.asarray(traj))
        rows.append(new_dict)
    new_df2 = pd.DataFrame(rows, index=df_group.groups.keys())
    new_df2.index.name = 'cell_bf'
    new_df = new_df.join(new_df2)
    return(new_df)
    
    

def normalize_fourier(all_feats):
    '''
    Fourier features relative to the size of the cell, on the summaries of each cell (mean, std, max, min):
    the magnitudes (lengths) are divided by the mean equivalent diameter of the cell over the movie,
    and the elliptic areas by its mean area
    '''
    diameter, area = all_feats['equivalent_diameter_area_mean'], all_feats['area_mean']
    for column in all_feats.columns:
        if column.startswith('magnitude_'):
            all_feats[column] = all_feats[column]/diameter
        elif column.startswith('elliptic_area_'):
            all_feats[column] = all_feats[column]/area
    return all_feats


def reduce_neighbors(df_neigh):
    df_group = df_neigh.groupby('cell_bf')
    new_df = df_group.aggregate(['mean', 'std', 'max', 'min'])
    new_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in new_df.columns]
    return(new_df)     

list_exp_path, list_condition = LEARNING_EXP_PATHS, LEARNING_CONDITIONS
filter_error = FILTER_ERROR

all_features = pd.DataFrame()

for exp_path, l_cond in zip(list_exp_path, list_condition):
    print(exp_path)
    for condition in l_cond:

        all_feats_mean = pd.DataFrame()
        gene_expr = pd.read_csv(ROOTPATH + exp_path + 'Fish/' + condition + '.csv', index_col='Key')
        alignment = pd.read_csv(ROOTPATH + exp_path + 'alignment/' + condition + '-alignment.csv', index_col='BF')

        # FINETUNE MASKS
        path_feats = ROOTPATH + exp_path + 'live_cell_features/'
        # CONTOUR
        order = FOURIER_ORDER
        df_fourier_finetune = pd.read_csv(path_feats + f'fourier_finetune_{condition}.csv', index_col=['cell_bf', 't'])
        all_feats_mean = reduce_features_fourier_finetune(df_fourier_finetune, order=order)
        all_feats_mean['count'] = df_fourier_finetune.groupby('cell_bf').size()

        # IOU
        df_iou_finetune = pd.read_csv(path_feats + f'finetune_iou_{condition}.csv', index_col=['cell_bf', 't'])
        all_feats_mean = all_feats_mean.join(reduce_neighbors(df_iou_finetune[['iou', 'iou_abs']]))

        # SCIKITMAGE FEATURES AND TRAJECTORY
        df_scikit_finetune = pd.read_csv(path_feats + f'finetune_scikit_{condition}.csv', index_col=['cell_bf', 't'])
        all_feats_mean = all_feats_mean.join(reduce_features_cell(df_scikit_finetune))
        # Fourier features relative to the size of the cell
        all_feats_mean = normalize_fourier(all_feats_mean)

        # IS ON EDGE
        is_on_edge = pd.read_csv(path_feats + f'is_on_edge_{condition}.csv')
        all_feats_mean['is_on_edge'] = all_feats_mean.index.isin(is_on_edge['cell_bf'])

        # AREA EXPLORED
        df_area_explored = pd.read_csv(path_feats + f'finetune_area_explored_{condition}.csv', index_col=['cell_bf'])
        df_area_explored['area_explored_total'] /= df_area_explored['count_t']
        all_feats_mean = all_feats_mean.join(df_area_explored[['area_explored_total']])
        # Area of the convex hull of the union of the masks of the cell (not divided by the number of time frames)
        if 'convex_hull_area' in df_area_explored.columns:
            all_feats_mean = all_feats_mean.join(df_area_explored[['convex_hull_area']])
        else:
            print(f'{condition}: no convex_hull_area in finetune_area_explored_{condition}.csv, run step 4.1 again to compute it')
        
        # DYNAMICS FEATURES
        path_dynamics = ROOTPATH + exp_path + 'live_cell_features/dynamics_feats_'+ condition + '.csv'
        if pathlib.Path(path_dynamics).exists():
            df_dyna = pd.read_csv(path_dynamics, index_col=['cell_bf'])
            # The fit is not reliable: missing value
            df_dyna.loc[df_dyna['std_err']>ALPHA_MAX_STD_ERR, 'alpha'] = np.nan
            all_feats_mean = all_feats_mean.join(df_dyna[['alpha']])
        
        # NEIGHBORS FEATURES
        path_neigh = ROOTPATH + exp_path + 'live_cell_features/neighbors_feats_'+ condition + '.csv'
        if pathlib.Path(path_neigh).exists():
            df_neigh = pd.read_csv(path_neigh, index_col=['cell_bf', 't']).drop(columns=['area', 'centroid-0', 'centroid-1'])
            df_neigh_mean = reduce_neighbors(df_neigh)
            all_feats_mean = all_feats_mean.join(df_neigh_mean)
            
        # MATCHING AND GENE EXPRESSION
        # All the live cells are kept: the cells without a FISH cell have no value for the matching and the genes
        alignment = alignment.reset_index().set_index('Fish').join(gene_expr).reset_index().set_index('BF')
        all_feats_mean = all_feats_mean.join(alignment)

        all_feats_mean['exp_path'] = exp_path
        all_feats_mean['condition'] = condition
        all_feats_mean = all_feats_mean.reset_index().set_index(['exp_path', 'condition', 'cell_bf'])
        
        # Filter the error
        if filter_error:
            all_feats_mean = all_feats_mean[all_feats_mean['max_speed']<MAX_SPEED]
            all_feats_mean = all_feats_mean[all_feats_mean['is_on_edge']==False]

        all_features = pd.concat((all_features, all_feats_mean),axis=0)
    

all_features.to_csv('6_results/dataset.csv')
