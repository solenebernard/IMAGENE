# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

# To compute hand-crafted features
import matplotlib
import numexpr as ne
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import pyefd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import splprep, splev
import itertools
from matplotlib.path import Path

save_path_model = ROOTPATH + 'learning/'


pick_cond = CONDITION
pick_exp_path = EXP_PATH

'''
Compute Fourier features
'''


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



# 1. Speed (assuming uniform time step = 1)
def compute_speed(traj):
    diffs = np.diff(traj, axis=0)
    speed = np.linalg.norm(diffs, axis=1)
    return speed

# 2. Turning angles (change in direction)
def compute_turning_angles(traj):
    v1 = np.diff(traj[:-1], axis=0)
    v2 = np.diff(traj[1:], axis=0)
    dot = np.einsum('ij,ij->i', v1, v2)
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos_theta = np.clip(dot / norms, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    return angles

# 3. Path length
def compute_path_length(traj):
    return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

# 4. Net displacement
def compute_net_displacement(traj):
    return np.linalg.norm(traj[-1] - traj[0])

# 5. Tortuosity
def compute_tortuosity(traj):
    path_length = compute_path_length(traj)
    net_disp = compute_net_displacement(traj)
    return path_length / net_disp if net_disp != 0 else np.inf

def compute_metrics_trajectory(traj):
    '''
    From trajectory with shape (n,3)
    '''
    speed = compute_speed(traj[:,:])
    angles = compute_turning_angles(traj[:,:])
    path_length = compute_path_length(traj[:,:])
    net_displacement = compute_net_displacement(traj[:,:])
    tortuosity = compute_tortuosity(traj[:,:])
    return(speed, angles, path_length, net_displacement, tortuosity)


def reconstruct_simple_contour(coeffs, locus, number_of_points):
    # number_of_points = contour.shape[0]
    # locus = pyefd.calculate_dc_coefficients(contour)
    # coeffs = pyefd.elliptic_fourier_descriptors(contour, order=order)
    reconstruction = pyefd.reconstruct_contour(coeffs, locus, number_of_points)
    # diff, _, _ = directed_hausdorff(reconstruction, contour) 
    return(reconstruction)


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

def get_contour(mask):
    contours, _ = cv2.findContours((mask>0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Step 2: Choose the largest contour (assuming single cell per mask)
    contour = max(contours, key=cv2.contourArea)
    # Step 3: Reshape to Nx2 format (x, y) as expected
    contour = contour[:, 0, :]  # Shape: (N, 2)
    # Step 4: Ensure contour is in float
    contour = contour.astype(float)
    
    # Find the starting point
    cx, cy = contour.mean(axis=0)
    # Compute angles from centroid to each point
    angles = np.arctan2(contour[:,1] - cy, contour[:,0] - cx)  # arctan2(y,y)
    # Find point closest to angle 0
    start_idx = np.argmin(np.abs(angles))
    contour_aligned = np.roll(contour, -start_idx, axis=0)
    return(contour_aligned)

# def fourier_descriptor(contour, order=100):
#     coeffs = pyefd.elliptic_fourier_descriptors(contour, order=order, normalize=True)
#     locus = pyefd.calculate_dc_coefficients(contour)
#     return(coeffs, locus)

def elliptic_fourier_descriptors_angle(contour, order=10, normalize=False):
    """
    Compute EFD coefficients using uniform-angle FFT (not arc-length).
    
    Parameters
    ----------
    contour : (N,2) ndarray
        Contour points (x,y) sampled uniformly in angle θ ∈ [0,2π).
    order : int
        Number of harmonics to return.
    normalize : bool
        If True, normalize coefficients to be translation, rotation,
        and scale invariant.
    
    Returns
    -------
    coeffs : (order,4) ndarray
        EFD coefficients [a_n, b_n, c_n, d_n] for n=1..order
    locus : (2,) ndarray
        DC offsets (A0, C0).
    """
    contour = np.asarray(contour)
    N = contour.shape[0]
    x, y = contour[:,0], contour[:,1]
    
    # FFT (complex coefficients)
    X = np.fft.fft(x) / N
    Y = np.fft.fft(y) / N
    
    coeffs = []
    for n in range(1, order+1):
        Xn = X[n]
        Yn = Y[n]
        a_n =  2*np.real(Xn)
        b_n = -2*np.imag(Xn)
        c_n =  2*np.real(Yn)
        d_n = -2*np.imag(Yn)
        coeffs.append([a_n, b_n, c_n, d_n])
    
    coeffs = np.array(coeffs)
    locus = np.array([np.real(X[0]), np.real(Y[0])])  # translation
    
    if normalize:
        coeffs = pyefd.normalize_efd(np.array(coeffs), size_invariant=True, return_transformation=False)
    
    return coeffs, locus

def resample_contour(contour, n_points=128):
    x, y = contour[:, 0], contour[:, 1]
    if (x[0]!=x[-1])or(y[0]!=y[-1]):
        x,y = np.concatenate((x, x[:1])), np.concatenate((y, y[:1]))
    tck, u = splprep([x, y], s=0, per=True)
    u_new = np.linspace(0, 1, n_points)
    x_new, y_new = splev(u_new, tck)
    return np.stack((x_new, y_new), axis=-1)

def compute_normalized_efd(contour, order=10, n_points=300, enforce_d1_nonneg=True):
    
    efd, locus = elliptic_fourier_descriptors_angle(contour, normalize=False, order=order)
    efd_norm = pyefd.normalize_efd(efd)  # pyefd normalization (scale/rotation/phase)
    # efd_norm shape: (order, 4) -> columns [a_n, b_n, c_n, d_n]
    if enforce_d1_nonneg:
        if efd_norm[0,3] < 0:  # d1 < 0 -> flip sine terms b_n and d_n
            efd_norm[:,1] *= -1
            efd_norm[:,3] *= -1
    # flatten to 1D descriptor if needed
    return efd_norm, locus


def mask_binary_is_on_edge(mask, x_min, y_min, mask_shape, delta=10):
    idx_x, idx_y = np.where(mask>0)
    idx_x, idx_y = idx_x+x_min, idx_y+y_min
    size_x, size_y = mask_shape
    if len(idx_x)>0:
        if idx_x.min()<delta or idx_y.min()<delta or size_x-idx_x.max()<delta or size_y-idx_y.max()<delta:
            return(True)
    return(False)


def contour_to_mask(contour, shape=(100, 100)):
    path = matplotlib.path.Path(contour)
    x, y = np.meshgrid(np.arange(-shape[1]//2, shape[1]//2), \
                       np.arange(-shape[0]//2, shape[0]//2))
    coords = np.vstack((x.ravel(), y.ravel())).T
    mask_flat = path.contains_points(coords)
    return mask_flat.reshape(shape)


def compute_iou(contour1, contour2, shape=(100, 100)):
    mask1 = contour_to_mask(contour1, shape)
    mask2 = contour_to_mask(contour2, shape)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union != 0 else 0
    return iou, intersection, union

def process_cell_fourier(exp_path, condition, order=10, n_points=128):

    list_masks = load_masks(ROOTPATH, exp_path, condition, clean=True)
    unique_bf_idx, bf_counts = np.unique(list_masks[list_masks>0], return_counts=True)

    # Remove bf cell id with very few pixels
    unique_bf_idx = unique_bf_idx[bf_counts>1000]
    mask_shape = (list_masks.shape[1], list_masks.shape[2])
    
    # Flatten list of name of EFD
    name_cols = list(itertools.chain.from_iterable([[f'a_{i}', f'b_{i}', f'c_{i}', f'd_{i}'] for i in range(order)]))
    df_final_dtypes = {'area': 'int64',
            'iou': 'float64',
            'iou_abs': 'float64',
            'is_on_edge':'bool',
            'traj_x':'float64',
            'traj_y': 'float64'} | \
            {l:'float64' for l in name_cols}

    df_final = pd.DataFrame()

    for count_bf, idx_bf in enumerate(unique_bf_idx):

        is_unique_large_cell = []

        df = pd.DataFrame()

        # To improve accuracy: crop the big mask
        sub_mask = ne.evaluate("list_masks == idx_bf")
        proj_mask = np.any(sub_mask, axis=0)
        # Collapse time axis using `np.any()` to get where the object exists in space
        proj_mask = np.any(sub_mask, axis=0)  # shape (H, W), True where object appears in any frame
        idx_x, idx_y = np.nonzero(proj_mask)
        x_min, x_max, y_min, y_max = idx_x.min(),idx_x.max()+1, idx_y.min(),idx_y.max()+1
        new_mask = sub_mask[:, x_min:x_max, y_min:y_max]

        visu_mask = np.zeros_like(new_mask[0], dtype=np.int16)
        
        for i,mask_idx in enumerate(new_mask):

            labeled_mask = label(mask_idx)
            unique_labeled = np.unique(labeled_mask[labeled_mask>0])
            unique_labeled.sort()
            if len(unique_labeled)>1: # Meaning there is more than one cell
                # Keep the largest ?
                count_unique = np.asarray([len(np.where(labeled_mask==x)[0]) for x in unique_labeled])
                ratio = count_unique/count_unique.max()
                # Check if the largest is the unique big cell: the second largest must be at least 2 times smaller
                list_second_ratio = ratio[ratio<1]
                if len(list_second_ratio)>0:
                    second_ratio = list_second_ratio.max()
                    if second_ratio>0.5:
                        is_unique_large_cell.append(False)
                    else:
                        is_unique_large_cell.append(True)
                else:
                    is_unique_large_cell.append(False)
                print(idx_bf, i, unique_labeled, count_unique, second_ratio)
                new_label = unique_labeled[np.argmax(count_unique)]
                mask_idx = labeled_mask==new_label
            else:
                is_unique_large_cell.append(True)
            
            if np.sum(mask_idx)>500:
                visu_mask += mask_idx.astype(np.int16)

                c_x, c_y = np.where(mask_idx>0)
                centroid = [c_x.mean(), c_y.mean()]

                bool_edge = mask_binary_is_on_edge(mask_idx, x_min, y_min, mask_shape) # keep track if at some t, the cell is at the edge of the FOV
                contour = get_contour(mask_idx)
                # contour = resample_contour(contour, n_points=n_points)
                coeffs, locus = compute_normalized_efd(contour, order=order, n_points=n_points, enforce_d1_nonneg=True)
                # M, A = get_features_efd(coeffs)
                contour_recon = reconstruct_simple_contour(coeffs, locus, number_of_points=n_points) - locus
                area = np.sum(mask_idx)

                # plt.plot(contour[:,0], contour[:,1])
                # plt.plot(contour_recon[:,0], contour_recon[:,1])
                # plt.show()

                if i>0:
                    # Get IOU
                    intersection = np.logical_and(mask_idx>0, prev_mask>0).sum()
                    union = np.logical_or(mask_idx>0, prev_mask>0).sum()
                    iou_abs = intersection / union if union != 0 else 0

                    if prev_contour_recon is not None:
                        try:
                            iou, _, _ = compute_iou(contour_recon, prev_contour_recon, shape=(256, 256))
                        except:
                            iou = None
                    else:
                        iou = None

                    # plt.imshow((mask_idx.astype(np.int8)+prev_mask.astype(np.int8)))
                    # plt.show()
                    # plt.plot(contour_recon[:,0],contour_recon[:,1])
                    # plt.plot(prev_contour_recon[:,0],prev_contour_recon[:,1])
                    # plt.show()

                else:
                    iou_abs, iou = None, None
                
                feats = pd.DataFrame([{'Cell bf':idx_bf,'t':i, 'area': area, 'iou':iou, 'iou_abs':iou_abs, \
                                'is_on_edge':bool_edge, 'traj_x':centroid[0],'traj_y':centroid[1]} | \
                                    {l:x for l,x in zip(name_cols,coeffs.flatten())}])
                feats.set_index(['Cell bf', 't'], inplace=True)
                feats = feats.astype(df_final_dtypes)
                            
                df = pd.concat((df, feats))
                
                
            else:
                contour_recon = None

            prev_mask = np.copy(mask_idx)
            prev_contour_recon = np.copy(contour_recon)
        
        cell_dir = ROOTPATH + exp_path + 'Live/' + condition + f'/live_dataset/cell_{idx_bf}/'
        if not os.path.exists(cell_dir):
            os.makedirs(cell_dir)

        try:
            traj = np.asarray(df[['traj_x','traj_y']])
            _, ax = plt.subplots()
            ind_pos = np.where(visu_mask[:]>0)
            xmin, xmax, ymin, ymax = ind_pos[0].min(), ind_pos[0].max(), ind_pos[1].min(), ind_pos[1].max()
            ax.imshow(visu_mask[xmin:xmax+1, ymin:ymax+1].T,origin='lower')
            ax.plot(traj[:,0]-xmin,traj[:,1]-ymin,c='black',linewidth=0.5, marker='.')
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(cell_dir + 'trajectory.png')
            plt.show()
            plt.close()
        except:
            pass
        
        if len(df)>1 and (np.all(is_unique_large_cell)): # If there is something to add and the mask is valid
            df_final = pd.concat((df_final, df))
    
        if count_bf%50==0:
            df_final.to_csv(ROOTPATH + exp_path + 'live_cell_features/' + condition + '_fourier2.csv', index=True)

    return(df_final)



def reduce_features_fourier(df_fourier, order=10, filter_error=True):
    list_feats = [get_features_efd(df_fourier, o) for o in range(order)]
    df_magnitude = pd.DataFrame({f'magnitude_{i}':x[0] for i,x in enumerate(list_feats)})
    df_elliptic_area = pd.DataFrame({f'elliptic_area_{i}':x[1] for i,x in enumerate(list_feats)})
    df_fourier = df_fourier.join(df_magnitude).join(df_elliptic_area)
    
    df_group = df_fourier.groupby('Cell bf')
    new_df = df_group.aggregate(['mean', 'std'])
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

    # Filter the error
    if filter_error:
        new_df = new_df[new_df['max_speed']<200]
        new_df = new_df[new_df['is_on_edge']==False]

    return(df_fourier, new_df)  

def join_gene_expression(list_exp_path, list_condition, gene_list):

    all_df = pd.DataFrame()
    for exp_path, l_cond in zip(list_exp_path, list_condition):
        for condition in l_cond:
            path = ROOTPATH + exp_path + 'live_cell_features/' + condition + '_fourier2.csv'
            df_fourier = pd.read_csv(path, index_col=['Cell bf', 't'])
            df_fourier, new_df = reduce_features_fourier(df_fourier, order=10)
            new_df['condition']=condition
            new_df['exp_path']=exp_path
            new_df.index.name = 'cell_bf'
            new_df = new_df.set_index(['exp_path', 'condition'], append=True)

            # Join gene expression if possible
            try:
                gene_expr = pd.read_csv(ROOTPATH + exp_path + 'Fish/' + condition +'.csv', index_col='Key')
                # Get the matching indices
                alignement = pd.read_csv(ROOTPATH + exp_path + 'alignment/' + condition +'-alignment.csv', index_col='Fish')
                
                # Join gene to the matching
                for gene in gene_list:
                    # If gene exists
                    if f'Gene_{gene}' in gene_expr.columns:
                        alignement = alignement.join(gene_expr[[f'Gene_{gene}', f'delta_z_l_{gene}', f'delta_z_r_{gene}']])
                    else:
                        alignement[f'Gene_{gene}'] = None
                        alignement[f'delta_z_l_{gene}'] = None
                        alignement[f'delta_z_r_{gene}'] = None
                
                # alignement = alignement.join(gene_expr[[f'Gene_{gene}' for gene in gene_list] + \
                #                                        [f'delta_z_l_{gene}' for gene in gene_list] + \
                #                                        [f'delta_z_r_{gene}' for gene in gene_list]])
                alignement = alignement.rename(columns={'BF':'cell_bf'})
                alignement = alignement.reset_index().set_index(['cell_bf']) # Keep Fish id
                alignement['condition'] = condition
                alignement['exp_path'] = exp_path
                alignement = alignement.set_index(['exp_path', 'condition'], append=True)

                # Join the fourier features
                new_df = new_df.join(alignement)

                print(exp_path, condition, new_df.shape, all_df.shape)

            except:
                pass
            
            all_df = pd.concat((all_df, new_df))

    return(all_df)


# # Compute Fourier Features from BF masks
process_cell_fourier(EXP_PATH, CONDITION)
