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
            "convex_hull_area": None,
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
            hull_area = ConvexHull(traj).area
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
            hull_area = 0.0
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
            "convex_hull_area": hull_area,
            "elongation": elongation
        }



def reduce_features_fourier(df_fourier, order=10):
    list_feats = [get_features_efd(df_fourier, o) for o in range(order)]
    df_magnitude = pd.DataFrame({f'magnitude_{i}':x[0] for i,x in enumerate(list_feats)})
    df_elliptic_area = pd.DataFrame({f'elliptic_area_{i}':x[1] for i,x in enumerate(list_feats)})
    df_fourier = df_fourier.join(df_magnitude).join(df_elliptic_area)
    
    df_group = df_fourier.groupby('Cell bf')
    new_df = df_group.aggregate(['mean', 'std', 'max', 'min'])
    new_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in new_df.columns]
    new_df = pd.concat((new_df, pd.DataFrame({'is_on_edge':df_group['is_on_edge'].any()})),axis=1)
    new_df = pd.concat((new_df, pd.DataFrame({'count':df_group['a_0'].count()})),axis=1)

    # Compute trajectory characteristics
    new_df2 = pd.DataFrame()
    rows = []
    for _, traj in df_group[['traj_x', 'traj_y']]:
        new_dict = compute_metrics_trajectory_fourier(np.asarray(traj))
        rows.append(new_dict)
    new_df2 = pd.DataFrame(rows, index=df_group.groups.keys())
    new_df2.index.name = 'Cell bf'

    new_df = new_df.drop(columns=['is_on_edge_mean', 'is_on_edge_std']).join(new_df2)

    return(df_fourier, new_df)  

def reduce_features_cell(df_cell):
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
    
    

def reduce_features_nucleus(df_nucleus):
    df_nucleus['ratio_area_nucleus_cell'] = df_nucleus['area_nucleus']/df_nucleus['area_cell']
    df_nucleus['ratio_bbox_area_nucleus_cell'] = df_nucleus['bbox_area_nucleus']/df_nucleus['bbox_area_cell']
    df_nucleus = df_nucleus.drop(columns=['area_cell',  'centroid-0_cell', 'centroid-1_cell', \
        'centroid-0_nucleus', 'centroid-1_nucleus'])
    # Ignore zeros (it might means the nucleus segmentation is missing)
    df_clean = df_nucleus.replace(0, np.nan)
    df_group = df_clean.groupby('cell_bf')
    new_df = df_group.aggregate(['mean', 'std', 'max', 'min'])
    new_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in new_df.columns]
    return(new_df)


def reduce_neighbors(df_neigh):
    df_group = df_neigh.groupby('cell_bf')
    new_df = df_group.aggregate(['mean', 'std', 'max', 'min'])
    new_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in new_df.columns]
    return(new_df)


def reduce_features_lamellipodia(df_lamellipodia):
    df_lamellipodia = df_lamellipodia.drop(columns=['area_cell',  'centroid-0_cell', 'centroid-1_cell', \
        'centroid-0_inner', 'centroid-1_inner', 'centroid-0_outer', 'centroid-1_outer'])
    df_lamellipodia = df_lamellipodia.droplevel('t')
    return(df_lamellipodia)

                                
# list_exp_path, list_condition = ['2025-10-13_Olga6/'], \
#                                 [['1-NA', '2-MIX', '3-24H', '4-24H']]
                                

list_exp_path, list_condition = ["2025-03-04_Proof-of-concept-Olga/" , \
                            '2025-03-31_Proof-of-concept-Olga2/', \
                            "2025-04-13_Proof-of-concept-Olga3/", \
                            '2025-10-13_Olga6/'], \
                            [['1-NA', '2-4H-Mael', '3-4H-Nico', '4-24H-Nico'], \
                            ['1-NA', '2-4H', '3-24H', '4-MIX'], \
                            ['1-NA', '2-4H', '3-24H', '4-MIX'], \
                            ['1-NA', '2-MIX', '3-24H', '4-24H']]
        
filter_error = True

all_features = pd.DataFrame()

for exp_path, l_cond in zip(list_exp_path, list_condition):
    print(exp_path)
    for condition in l_cond:

        all_feats_mean = pd.DataFrame()
        gene_expr = pd.read_csv(ROOTPATH + exp_path + 'Fish/' + condition + '.csv', index_col='Key')
        alignment = pd.read_csv(ROOTPATH + exp_path + 'alignment/' + condition + '-alignment.csv', index_col='BF')

        # TEXTURE
        texture = pd.read_csv(ROOTPATH + exp_path + 'live_cell_features/texture_'+condition+'.csv', index_col=['cell_bf', 't', 'z'])
        all_feats_mean = texture.groupby(['cell_bf', 'z']).agg('mean').unstack(level='z')
        all_feats_mean.columns = [f"{col[0]}_z{col[1]}" for col in all_feats_mean.columns]

        # CONTOUR
        order = 10
        fourier = pd.read_csv(ROOTPATH + exp_path + 'live_cell_features/'+condition+'_fourier2.csv', index_col=['Cell bf', 't'])
        fourier, fourier_mean = reduce_features_fourier(fourier, order=order)
        for o in range(order):
            fourier_mean = fourier_mean.drop(columns=[c for c in fourier_mean.columns \
                if c.startswith((f'a_{o}', f'b_{o}', f'c_{o}', f'd_{o}'))])
        if len(all_feats_mean>1):
            all_feats_mean = all_feats_mean.join(fourier_mean)
        else:
            all_feats_mean = fourier_mean
            all_feats_mean.index.name = 'cell_bf'
        
        # DYNAMICS FEATURES
        path_dynamics = ROOTPATH + exp_path + 'live_cell_features/dynamics_feats_'+ condition + '.csv'
        if pathlib.Path(path_dynamics).exists():
            df_dyna = pd.read_csv(path_dynamics, index_col=['cell_bf'])
            df_dyna.loc[df_dyna['std_err']>0.03, 'alpha'] = 0
            all_feats_mean = all_feats_mean.join(df_dyna[['alpha']])
        
        # NEIGHBORS FEATURES
        path_neigh = ROOTPATH + exp_path + 'live_cell_features/neighbors_feats_'+ condition + '.csv'
        if pathlib.Path(path_dynamics).exists():
            df_neigh = pd.read_csv(path_neigh, index_col=['cell_bf', 't']).drop(columns=['area', 'centroid-0', 'centroid-1'])
            df_neigh_mean = reduce_neighbors(df_neigh)
            all_feats_mean = all_feats_mean.join(df_neigh_mean)
            
        all_feats_mean = all_feats_mean.join(alignment)
        all_feats_mean = all_feats_mean.reset_index().set_index('Fish')
        all_feats_mean = all_feats_mean[~all_feats_mean.index.get_level_values('Fish').isna()]
        all_feats_mean = all_feats_mean.join(gene_expr)

        all_feats_mean['exp_path'] = exp_path
        all_feats_mean['condition'] = condition
        all_feats_mean = all_feats_mean.reset_index().set_index(['exp_path', 'condition', 'cell_bf'])
        all_feats_mean = all_feats_mean.astype({'Fish': 'int64'})
        
        # Filter the error
        if filter_error:
            all_feats_mean = all_feats_mean[all_feats_mean['max_speed']<200]
            all_feats_mean = all_feats_mean[all_feats_mean['is_on_edge']==False]

        all_features = pd.concat((all_features, all_feats_mean),axis=0)
    

all_features.to_csv('6_results/dataset.csv')
