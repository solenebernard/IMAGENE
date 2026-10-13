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
from skimage.measure import regionprops_table

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
    efd_norm = pyefd.normalize_efd(efd, size_invariant=False)  # pyefd normalization (scale/rotation/phase)
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


def contour_area(contour):
    """
    contour: array of shape (n, 2), where each row is [x, y]
             The contour is assumed to form a closed polygon.

    Returns:
        absolute area (float)
    """
    x = contour[:, 0]
    y = contour[:, 1]

    # close the polygon explicitly
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    return 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def contour_perimeter(contour, closed=True):
    """
    Compute the perimeter of a 2D contour.

    Parameters
    ----------
    contour : ndarray of shape (n,2)
        The contour points [[x0,y0], [x1,y1], ...].
    closed : bool
        Whether to treat the contour as closed (connect last point to first).

    Returns
    -------
    perimeter : float
        Total length of the contour.
    """
    contour = np.asarray(contour)
    diffs = np.diff(contour, axis=0)
    segment_lengths = np.sqrt((diffs[:,0])**2 + (diffs[:,1])**2)
    perimeter = segment_lengths.sum()

    if closed:
        # add distance between last and first point
        dx = contour[0,0] - contour[-1,0]
        dy = contour[0,1] - contour[-1,1]
        perimeter += np.sqrt(dx**2 + dy**2)

    return perimeter

def compute_dt(s0, s1):
    return(1 + (s1-s0)/LIVE_N_TILES)

def mask_intersection_equal(m0, m1):
    """
    Returns a mask m2 where:
        m2[i,j] = m1[i,j] if m1[i,j] == m0[i,j]
        m2[i,j] = 0 otherwise
    
    m0, m1 must be same shape.
    """
    m0 = np.asarray(m0)
    m1 = np.asarray(m1)

    m2 = np.zeros_like(m1)
    mask = (m0 == m1)
    m2[mask] = m1[mask]
    return m2


def fourier_features(rootpath, exp_path, condition, masks_tiles):
    dict_feats = {}
    name_cols = list(itertools.chain.from_iterable([[f'a_{i}', f'b_{i}', f'c_{i}', f'd_{i}'] for i in range(order)]))

    df = pd.DataFrame()
    df_tile = pd.DataFrame()
    for t in range(masks_tiles.shape[0]):
        for s in range(masks_tiles.shape[1]):
            m = masks_tiles[t,s]
            unique_id = np.unique(m[m>0])
            
            # fig, axes = plt.subplots(1,2,figsize=(10,4))
            for id in unique_id[:]:
                mask_idx = m==id
                contour = get_contour(mask_idx)
                # contour = resample_contour(contour, n_points=n_points)
                coeffs, locus = compute_normalized_efd(contour, order=order, n_points=n_points, enforce_d1_nonneg=True)
                contour_recon = reconstruct_simple_contour(coeffs, locus, number_of_points=n_points) - locus
                dict_feats = {l:x for l,x in zip(name_cols,coeffs.flatten())}

                # axes[0].plot(contour[:,0]-locus[0], contour[:,1]-locus[1])
                # axes[1].plot(contour_recon[:,0], contour_recon[:,1])
               
                index = pd.MultiIndex.from_arrays([[id], [t]], names=["cell_bf", "t"])
                df_t_s = pd.DataFrame(dict_feats, index=index)
                df_t_s_tile = pd.DataFrame({'s': s}, index=index)

                df = pd.concat((df, df_t_s))
                df_tile = pd.concat((df_tile, df_t_s_tile))

            # plt.show()

    df.to_csv(rootpath + exp_path + f'live_cell_features/fourier_finetune_{condition}.csv', index=True)
    df_tile.to_csv(rootpath + exp_path + f'live_cell_features/finetune_tile_{condition}.csv', index=True)


def coords_to_hashes(coords, W):
    # xy = coords.reshape(-1, 2)
    return coords[0] * W + coords[1]  # unique pixel index


def is_on_edge_feat(rootpath, exp_path, condition, masks_merged):
    T,H,W = masks_merged.shape
    border = np.stack((masks_merged[:,0,:], masks_merged[:,-1,:], \
            masks_merged[:,:,0], masks_merged[:,:,-1]),axis=1) 
    x = [np.unique(x[x>0]) for x in border]
    rows = []
    for t,cells in enumerate(x):
        for cell in cells:
            rows.append([t,cell])
    is_one_edge = pd.DataFrame(rows, columns=['t', 'cell_bf'])
    is_one_edge.to_csv(rootpath + exp_path + f'live_cell_features/is_on_edge_{condition}.csv', index=False)
    
  
def total_area_explored_feats(rootpath, exp_path, condition, masks_merged):
    '''
    For each cell, from the union of its masks over the time frames:
        - area_explored_total: number of pixels of the union
        - convex_hull_area: area of the convex hull of the union
        - count_t: number of time frames of the cell
    '''
    from scipy.spatial import ConvexHull, QhullError
    df_tile = pd.read_csv(rootpath + exp_path + f'live_cell_features/finetune_tile_{condition}.csv', index_col=['cell_bf', 't'])
    unique_id = np.unique(masks_merged[masks_merged>0])

    index_pos_mask = np.where(masks_merged>0)
    value_pos_mask = masks_merged[index_pos_mask]

    rows = []
    for id in unique_id[:]:
        sub_pos_t = index_pos_mask[0][value_pos_mask==id]
        sub_pos = np.stack((index_pos_mask[1][value_pos_mask==id], \
            index_pos_mask[2][value_pos_mask==id]),axis=0)
        unique_sub_pos = np.unique(sub_pos, axis=1)
        size_explored = unique_sub_pos.shape[1]
        n_t = len(np.unique(sub_pos_t))
        # Area of the convex hull of the union (for a 2D hull, scipy calls the area "volume" and the perimeter "area")
        try:
            hull_area = ConvexHull(unique_sub_pos.T).volume
        except (QhullError, ValueError):
            # Fewer than 3 pixels, or all the pixels on a line
            hull_area = np.nan
        rows.append([id, size_explored, hull_area, n_t])
            
    # Build df once at end
    df = pd.DataFrame(rows, columns=["cell_bf", "area_explored_total", "convex_hull_area", 'count_t'])
    df = df.set_index(["cell_bf"])
    df.to_csv(rootpath + exp_path + f'live_cell_features/finetune_area_explored_{condition}.csv', index=True)


  
def iou_feats(rootpath, exp_path, condition, masks_merged):
    '''
    Efficient code to compute iou
    '''
    df_tile = pd.read_csv(rootpath + exp_path + f'live_cell_features/finetune_tile_{condition}.csv', index_col=['cell_bf', 't'])
    unique_id = np.unique(masks_merged[masks_merged>0])

    index_pos_mask = np.where(masks_merged>0)
    value_pos_mask = masks_merged[index_pos_mask]

    rows = []
    W = masks_merged.shape[2]
    index_tile = set(df_tile.index)
    for t0 in range(masks_merged.shape[0]-1):
        print(t0)
        t1 = t0+1
        # m0, m1 = masks_merged[t0],masks_merged[t1]

        index_pos_t0 = [x[index_pos_mask[0]==t0] for x in index_pos_mask[1:]]
        index_pos_t1 = [x[index_pos_mask[0]==t1] for x in index_pos_mask[1:]]
        val_pos_t0 = value_pos_mask[index_pos_mask[0]==t0]
        val_pos_t1 = value_pos_mask[index_pos_mask[0]==t1]
        
        for id in unique_id:
            
            if ((id,t0) in index_tile) and ((id,t1) in index_tile):
                s0, s1 = df_tile.loc[(id,t0),'s'],df_tile.loc[(id,t1),'s']

                m0_index = np.array([x[val_pos_t0==id] for x in index_pos_t0])
                m1_index = np.array([x[val_pos_t1==id] for x in index_pos_t1])

                h0 = coords_to_hashes(m0_index, W)
                h1 = coords_to_hashes(m1_index, W)

                inter = len(np.intersect1d(h0, h1))
                union = len(np.union1d(h0, h1))

                if (union>0):
                    iou = inter/union
                
                # Now iou_abs
                # Remove the centroid in both cases
                centroid0 = np.round(m0_index.mean(axis=1))
                centroid1 = np.round(m1_index.mean(axis=1))
                
                m0_index -= centroid0[:,None].astype(np.int64)
                m1_index -= centroid1[:,None].astype(np.int64)

                h0 = coords_to_hashes(m0_index, W)
                h1 = coords_to_hashes(m1_index, W)

                inter = len(np.intersect1d(h0, h1))
                union = len(np.union1d(h0, h1))

                if (union>0):
                    iou_abs = inter/union

                # Store row as tuple
                rows.append((id, t0, iou, iou_abs, compute_dt(s0, s1)))

    # Build df once at end
    df = pd.DataFrame(rows, columns=["cell_bf", "t", "iou", "iou_abs", "dt"])
    df = df.set_index(["cell_bf", "t"])
    df.to_csv(rootpath + exp_path + f'live_cell_features/finetune_iou_{condition}.csv', index=True)


def feats_scikit(list_masks, rootpath, exp_path, condition):
    '''
    - Load the nucleus and whole cell masks
    - Relabel the nucleus mask to match the whole cell id
    - Compute: whole cell area, nucleus area, and distance between centroids
    '''
    n_t = len(list_masks)
    
    # Compute the features
    # Ratio 
    props_all = []
    for t in range(n_t):
        props_cells = regionprops_table(
            list_masks[t],
            properties=('label', 'area', 'centroid', 'bbox_area', 'perimeter', \
                'eccentricity','equivalent_diameter_area', 'extent', \
                'feret_diameter_max', 'major_axis_length', 'minor_axis_length', \
                'orientation', 'solidity', 'moments_hu', 'moments_central', \
                'moments_normalized', 'inertia_tensor', 'inertia_tensor_eigvals')
        )
        df_cells = pd.DataFrame(props_cells)
        df_cells['t'] = t
        
        props_all.append(df_cells)
    # Concatenate all frames
    df_cells_all = pd.concat([c for c in props_all])
    df_cells_all = df_cells_all.rename(columns={'label':'cell_bf'})
    df_cells_all = df_cells_all.set_index(['cell_bf', 't'])
    # Shift x and y **within each cell_bf**
    df_cells_all['centroid-0_next'] = df_cells_all.groupby(level='cell_bf')['centroid-0'].shift(-1)
    df_cells_all['centroid-1_next'] = df_cells_all.groupby(level='cell_bf')['centroid-1'].shift(-1)
    # Compute Euclidean distance
    df_cells_all['delta_distance'] = np.sqrt((df_cells_all['centroid-0_next'] - df_cells_all['centroid-0'])**2 + \
        (df_cells_all['centroid-1_next'] - df_cells_all['centroid-1'])**2)
    # Optional: drop helper columns
    df_cells_all = df_cells_all.drop(columns=['centroid-0_next', 'centroid-1_next'])
    # df_merged['distance_centroids'] = np.sqrt(
    #     (df_merged['centroid-0_cell'] - df_merged['centroid-0_nucleus'])**2 +
    #     (df_merged['centroid-1_cell'] - df_merged['centroid-1_nucleus'])**2
    # )
    df_cells_all.to_csv(rootpath + exp_path + 'live_cell_features/finetune_scikit_' + condition + '.csv', index=True)
    


# Temporary
rootpath, exp_path, condition = ROOTPATH, EXP_PATH, CONDITION
# condition = '4-24H'
# Cellpose masks of the tiles and global masks, with the track ID of each cell (steps 1.2 and 1.3)
masks_tiles = load_tracked_tile_masks(rootpath, exp_path, condition, t_max=N_FRAMES, n_tiles=LIVE_N_TILES, z=Z_FINETUNE)
masks_merged = load_tracked_masks(rootpath, exp_path, condition, t_max=N_FRAMES)
order = FOURIER_ORDER
n_points = FOURIER_N_POINTS

fourier_features(rootpath, exp_path, condition, masks_tiles)
feats_scikit(masks_merged, rootpath, exp_path, condition)
iou_feats(rootpath, exp_path, condition, masks_merged)
total_area_explored_feats(rootpath, exp_path, condition, masks_merged)
is_on_edge_feat(rootpath, exp_path, condition, masks_merged)



