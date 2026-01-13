# Import local tools
from tools.read_yaml import *
from tools.imports import *

import itertools
import time
import numexpr as ne
import pywt
import mahotas

# To compute hand-crafted features
import cv2
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor

from scipy.stats import skew, kurtosis
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff

from shapely.geometry import Polygon

import pyefd
from pyefd import elliptic_fourier_descriptors, reconstruct_contour

from sklearn.decomposition import PCA
import umap

save_path_model = ROOTPATH + 'learning/'

def load_masks(exp_path, condition):
    data_path_raw = ROOTPATH + exp_path + 'Live/' + condition + '/'
    path_masks_sam2 = data_path_raw + 'results/'

    # Get all frames
    frame_names_t = list(pathlib.Path(data_path_raw).glob('*_t*_z{}_ch00.tif'.format('00')))
    pattern = r"_t(\d{2})_"
    match_t = [re.search(pattern, str(filename)) for filename in frame_names_t]
    match_t = [m.group(1) for m in match_t if m]
    match_t.sort()

    list_masks = []
    for i in range(len(match_t)):
        mask = np.asarray(Image.open(path_masks_sam2 + 'seg_movie_merged_t{}_cp_mask.png'.format(str(i).zfill(2)))) # The mask
        list_masks.append(mask)
    list_masks = np.asarray(list_masks) # Dimension: (t,x,y)
    return(list_masks)

def load_bf_images(exp_path, condition):
    data_path_raw = ROOTPATH + exp_path + 'Live/' + condition + '/'

    # Scan all all t frames 
    # Get all possible z
    frame_names_z = list(pathlib.Path(data_path_raw).glob('*_t00_z*_ch00.tif'))
    pattern = r"_z(\d{2})_"
    match_z = [re.search(pattern, str(filename)) for filename in frame_names_z]
    match_z = [m.group(1) for m in match_z if m]
    match_z.sort()
    max_z_ = str(len(match_z)-1).zfill(2)

    # Get all frames
    frame_names_t = list(pathlib.Path(data_path_raw).glob('*_t*_z{}_ch00.tif'.format(max_z_)))
    pattern = r"_t(\d{2})_"
    match_t = [re.search(pattern, str(filename)) for filename in frame_names_t]
    match_t = [m.group(1) for m in match_t if m]
    match_t.sort()

    # Extract all frame names: all t and all z
    frame_names = [[list(pathlib.Path(data_path_raw).glob('*_t{}_z{}_ch00.tif'.format(t_,z_)))[0] \
                    for z_ in match_z] for t_ in match_t] 

    # Load all the masks (for each time frame) and all brightfield images
    list_tif_images = []

    for i, frames in enumerate(frame_names):
        im = [tifffile.imread(frame) for frame in frames] # all z for a given t
        list_tif_images.append(np.asarray(im))

    list_tif_images = np.asarray(list_tif_images) # Dimension: (t,z,x,y)
    
    return(list_tif_images)

def load_images_and_masks(exp_path, condition):
    list_tif_images = load_bf_images(exp_path, condition)
    list_masks = load_masks(exp_path, condition)
    return(list_tif_images, list_masks)


def remove_border(im):
    image = np.copy(im)
    # Crop the border with 0 values
    _,x,y = np.where(image>0)
    if len(x)>0:
        xmin, xmax, ymin, ymax = x.min(),x.max(), y.min(), y.max()
        image = image[:, xmin:xmax,ymin:ymax]
    return(image)

def pick_z_laplacian(image):
    """
    Choose the image (z,h,w) to (h,w) via laplacian metric to measure the focus
    """
    metrics = []
    image = remove_border(image)

    for j in range(image.shape[0]):
        m = cv2.Laplacian(image[j], cv2.CV_64F).var()
        metrics.append(m)
    metrics = np.asarray(metrics)
    # Best metric
    j = np.argmax(metrics)
    return(j)


# Get trajectory information through 5 functions

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

def compute_metrics_trajectory_fourier2(traj):
    speed = compute_speed(traj[:,:])
    angles = compute_turning_angles(traj[:,:])
    speed = np.concatenate(([None], speed))
    angles = np.concatenate(([None], angles,[None]))
    return(speed[:len(traj)], angles[:len(traj)])

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
    
    else:
        deltas = np.diff(traj, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        
        # Kinematic
        total_dist = distances.sum()
        net_disp = np.linalg.norm(traj[-1] - traj[0])
        straightness = net_disp / total_dist if total_dist != 0 else 0
        speeds = distances
        avg_speed = speeds.mean()
        max_speed = speeds.max()
        speed_std = speeds.std()
        
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
        
        # Geometric
        if len(traj) >= 3:
            hull_area = ConvexHull(traj).area
        else:
            hull_area = 0.0
        
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



def extract_cell_features(image_zdim, mask):
    """
    Extracts morphological, intensity, and texture features from a brightfield image with multiple z
    using a binary mask for the cell. 
    Outputs a dataframe, with a row for each z
    """
    features = {}
    
    # Ensure mask is binary
    mask = mask.astype(bool)

    # -----------------------
    # Morphological Features
    # -----------------------
    labeled_mask = label(mask)
    props = regionprops(labeled_mask, intensity_image=image_zdim[0])

    if len(props) == 0:
        raise ValueError("No cell detected in the mask.")

    prop = props[0]  # Assuming one cell per image

    features['z'] = [z for z in range(len(image_zdim))]
    features['area'] = [prop.area for _ in range(len(image_zdim))]
    features['perimeter'] = [prop.perimeter for _ in range(len(image_zdim))]
    features['eccentricity'] = [prop.eccentricity for _ in range(len(image_zdim)) ]
    features['solidity'] = [prop.solidity for _ in range(len(image_zdim))]
    features['extent'] = [prop.extent for _ in range(len(image_zdim))]
    features['major_axis_length'] = [prop.major_axis_length for _ in range(len(image_zdim))]
    features['minor_axis_length'] = [prop.minor_axis_length for _ in range(len(image_zdim))]
    features['circularity'] = [(4 * np.pi * prop.area) / (prop.perimeter**2 + 1e-6) for _ in range(len(image_zdim))]
    
    # -----------------------
    # Position Features
    # -----------------------
    features['centroid_x'] = [prop.centroid[0] for _ in range(len(image_zdim))]
    features['centroid_y'] = [prop.centroid[1] for _ in range(len(image_zdim))]
    
    # -----------------------
    # Intensity Features
    # -----------------------
    cell_pixels = image_zdim[:,mask]
    features['mean_intensity'] = np.mean(cell_pixels,axis=1)
    features['std_intensity'] = np.std(cell_pixels,axis=1)
    features['min_intensity'] = np.min(cell_pixels,axis=1)
    features['max_intensity'] = np.max(cell_pixels,axis=1)
    features['median_intensity'] = np.median(cell_pixels,axis=1)
    features['skewness_intensity'] = skew(cell_pixels,axis=1)
    features['kurtosis_intensity'] = kurtosis(cell_pixels,axis=1)

    # -----------------------
    # Texture Features (GLCM)
    # -----------------------
    # Quantize image to 8-bit (256 levels)
    quantized = (image_zdim / 255).astype(np.uint8)

    # Apply mask to zero out background
    masked_img = quantized * mask.astype(np.uint8)

    # Crop to bounding box of cell to compute GLCM
    minr, minc, maxr, maxc = prop.bbox
    cropped = masked_img[:, minr:maxr, minc:maxc]

    # GLCM Computation
    glcm_list = [graycomatrix(c, distances=[1], angles=[0], levels=2**8, symmetric=True, normed=True) for c in cropped]
    features['texture_contrast'] = [graycoprops(glcm, 'contrast')[0, 0] for glcm in glcm_list]
    features['texture_dissimilarity'] = [graycoprops(glcm, 'dissimilarity')[0, 0] for glcm in glcm_list]
    features['texture_homogeneity'] = [graycoprops(glcm, 'homogeneity')[0, 0] for glcm in glcm_list]
    features['texture_energy'] = [graycoprops(glcm, 'energy')[0, 0] for glcm in glcm_list]
    features['texture_correlation'] = [graycoprops(glcm, 'correlation')[0, 0] for glcm in glcm_list]
    features['texture_ASM'] = [graycoprops(glcm, 'ASM')[0, 0] for glcm in glcm_list]

    return pd.DataFrame(features)


# For Fourier features

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
    return(contour)

def fourier_descriptor(contour, order=100):
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=order)
    locus = pyefd.calculate_dc_coefficients(contour)
    return(coeffs, locus)

def resample_contour(contour, n_points=128):
    x, y = contour[:, 0], contour[:, 1]
    if (x[0]!=x[-1])or(y[0]!=y[-1]):
        x,y = np.concatenate((x, x[:1])), np.concatenate((y, y[:1]))
    tck, u = splprep([x, y], s=0, per=True)
    u_new = np.linspace(0, 1, n_points)
    x_new, y_new = splev(u_new, tck)
    return np.stack((x_new, y_new), axis=-1)

def mask_binary_is_on_edge(mask, x_min, y_min, mask_shape, delta=10):
    idx_x, idx_y = np.where(mask>0)
    idx_x, idx_y = idx_x+x_min, idx_y+y_min
    size_x, size_y = mask_shape
    if len(idx_x)>0:
        if idx_x.min()<delta or idx_y.min()<delta or size_x-idx_x.max()<delta or size_y-idx_y.max()<delta:
            return(True)
    return(False)


def contour_to_mask(contour, shape=(100, 100)):
    mask = np.zeros(shape, dtype=bool)
    path = Path(contour)
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

##### EFD FUNCTIONS
def compute_efd_hausdorff_distance(contour, recon):
    """
    Compute the Hausdorff distance between a contour and its EFD reconstruction.
    Returns:
        float: Symmetric Hausdorff distance.
    """
    # Compute directed Hausdorff distances in both directions
    d1 = directed_hausdorff(contour, recon)[0]
    d2 = directed_hausdorff(recon, contour)[0]

    return max(d1, d2)

def eccentricity_from_efd(coeffs):
    a1, b1, c1, d1 = coeffs[0]
    major = np.sqrt(a1**2 + c1**2)
    minor = np.sqrt(b1**2 + d1**2)
    if major < minor:
        major, minor = minor, major
    return np.sqrt(1 - (minor / major)**2)

def aspect_ratio_from_efd(coeffs):
    a1, b1, c1, d1 = coeffs[0]
    major = np.sqrt(a1**2 + c1**2)
    minor = np.sqrt(b1**2 + d1**2)
    return max(major, minor) / min(major, minor)

def complexity_ratio(coeffs):
    base_energy = np.sum(coeffs[0]**2)
    higher_energy = np.sum(coeffs[1:]**2)
    return higher_energy / base_energy if base_energy != 0 else np.inf

def fourier_entropy(coeffs):
    power = np.sum(coeffs**2, axis=1)
    total = np.sum(power)
    if total == 0:
        return 0
    probs = power / total
    return -np.sum(probs * np.log(probs + 1e-10))  # add epsilon to avoid log(0)

def symmetry_ratio(coeffs):
    power = np.sum(coeffs**2, axis=1)
    even = np.sum(power[1::2])  # 2nd, 4th, ...
    odd = np.sum(power[0::2])   # 1st, 3rd, ...
    return even / odd if odd != 0 else np.inf

def roughness_index(contour):
    poly = Polygon(contour)
    if not poly.is_valid or poly.area == 0:
        return np.inf
    perimeter = poly.length
    area = poly.area
    return (perimeter ** 2) / (4 * np.pi * area)

def extract_efd_features(coeffs_normed):
    # coeffs = elliptic_fourier_descriptors(contour, order=order, normalize=True)
    return {
        "eccentricity": eccentricity_from_efd(coeffs_normed),
        "aspect_ratio": aspect_ratio_from_efd(coeffs_normed),
        "complexity_ratio": complexity_ratio(coeffs_normed),
        "fourier_entropy": fourier_entropy(coeffs_normed),
        "symmetry_ratio": symmetry_ratio(coeffs_normed)
        # "hausdorff_error": hausdorff_error(contour, coeffs),
        # "roughness_index": roughness_index(contour),
    }

def extract_other_cell_features(contour):
    return {
        "area": cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.float32)),
        "roughness_index": roughness_index(contour)}


####  Extract texture features
def extract_texture_features(image, mask=None, distances=[1], angles=[0], lbp_radius=1, lbp_points=8):
    """
    Extracts a set of texture features from a grayscale image.
    
    Parameters:
        image (ndarray): 2D grayscale image.
        mask (ndarray): Optional binary mask to extract features within.
        distances (list): Pixel pair distance(s) for GLCM.
        angles (list): Angles in radians for GLCM.
        lbp_radius (int): Radius for LBP.
        lbp_points (int): Number of circularly symmetric neighbor set points for LBP.
    
    Returns:
        dict: A dictionary of scalar texture features.
    """
    if mask is not None:
        image = image * mask

    features = {}

    # === GLCM ===
    levels = 256 if image.max() > 128 else 8  # reduce levels if needed
    image_uint8 = np.uint8((image / image.max()) * (levels - 1))  # normalize
    glcm = graycomatrix(image_uint8, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    for p in props:
        stat = graycoprops(glcm, p)
        features[f'glcm_{p}_mean'] = stat.mean()
        features[f'glcm_{p}_std'] = stat.std()

    # === Haralick features === (mahotas requires uint8)
    haralick_feats = mahotas.features.haralick(image_uint8, ignore_zeros=True).mean(axis=0)
    for i, val in enumerate(haralick_feats):
        features[f'haralick_{i}'] = val

    # === LBP ===
    lbp = local_binary_pattern(image, lbp_points, lbp_radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_points + 3), density=True)
    for i, val in enumerate(lbp_hist):
        features[f'lbp_{i}'] = val

    # === Wavelet Energy ===
    coeffs2 = pywt.dwt2(image, 'db1')  # single-level 2D wavelet transform
    cA, (cH, cV, cD) = coeffs2
    features['wavelet_cA_energy'] = np.sum(cA**2)
    features['wavelet_cH_energy'] = np.sum(cH**2)
    features['wavelet_cV_energy'] = np.sum(cV**2)
    features['wavelet_cD_energy'] = np.sum(cD**2)

    # === Gabor Filters ===
    freqs = [0.1, 0.2, 0.3]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for freq in freqs:
        for theta in thetas:
            filt_real, _ = gabor(image, frequency=freq, theta=theta)
            features[f'gabor_mean_f{freq:.2f}_t{theta:.2f}'] = filt_real.mean()
            features[f'gabor_std_f{freq:.2f}_t{theta:.2f}'] = filt_real.std()

    return features



def process_cell_fourier(exp_path, condition, order=100, n_points=128):

    # list_masks = load_masks(exp_path, condition)
    list_tif_images, list_masks = load_images_and_masks(exp_path, condition)
    unique_bf_idx, bf_counts = np.unique(list_masks[list_masks>0], return_counts=True)

    # Remove bf cell id with very few pixels
    unique_bf_idx = unique_bf_idx[bf_counts>1000]
    mask_shape = (list_masks.shape[1], list_masks.shape[2])
    
    df_final = pd.DataFrame()

    for count_bf, idx_bf in enumerate(unique_bf_idx[:]):

        list_dict, traj, z_pick_list, list_df_textures = [], [], [], []

        # t0 = time.time()
        # To improve accuracy: crop the big mask
        # sub_mask = list_masks == idx_bf
        sub_mask = ne.evaluate("list_masks == idx_bf")
        proj_mask = np.any(sub_mask, axis=0)
        # Collapse time axis using `np.any()` to get where the object exists in space
        proj_mask = np.any(sub_mask, axis=0)  # shape (H, W), True where object appears in any frame
        idx_x, idx_y = np.nonzero(proj_mask)
        x_min, x_max, y_min, y_max = idx_x.min(),idx_x.max()+1, idx_y.min(),idx_y.max()+1
        new_mask = sub_mask[:, x_min:x_max, y_min:y_max]

        # visu_mask = np.zeros_like(new_mask[0], dtype=np.int16)

        for i,mask_idx in enumerate(new_mask):    

            labeled_mask = label(mask_idx)
            unique_labeled = np.unique(labeled_mask[labeled_mask>0])
            unique_labeled.sort()
            if len(unique_labeled)>1: # Meaning there is more than one cell
                # Keep the largest ?
                count_unique = [len(np.where(labeled_mask==x)[0]) for x in unique_labeled]
                # print(idx_bf, i, unique_labeled, count_unique)
                new_label = unique_labeled[np.argmax(count_unique)]
                mask_idx = labeled_mask==new_label

            
            if np.sum(mask_idx)>500:
                # visu_mask += mask_idx.astype(np.int16)

                c_x, c_y = np.where(mask_idx>0)
                centroid = [c_x.mean(), c_y.mean()]
                
                bool_edge = mask_binary_is_on_edge(mask_idx, x_min, y_min, mask_shape) # keep track if at some t, the cell is at the edge of the FOV
                contour = get_contour(mask_idx)
                contour = resample_contour(contour, n_points=n_points)

                coeffs, locus = fourier_descriptor(contour, order=order)
                contour_recon = reconstruct_simple_contour(coeffs, locus=(0,0), number_of_points=n_points)
                contour -= locus

                coeffs_norm = pyefd.normalize_efd(coeffs, size_invariant=True)

                dict_efd_features = extract_efd_features(coeffs_norm)
                dict_other_features = extract_other_cell_features(contour)
                dict_other_features['bool_edge'] = bool_edge

                # fig, ax = plt.subplots()
                # ax.plot(contour[:,0], contour[:,1])
                # ax.plot(contour_recon[:,0], contour_recon[:,1])
                # ax.set_aspect('equal')
                # plt.show()
                # print(dict_efd_features)

                # Get features parameters
                n_x_min, n_x_max, n_y_min, n_y_max = \
                    c_x.min(), c_x.max(), c_y.min(), c_y.max()
                z_tif_image = list_tif_images[i, :, x_min:x_max+1, y_min:y_max+1][:, n_x_min:n_x_max+1, n_y_min:n_y_max+1]
                sub_mask_im = mask_idx[n_x_min:n_x_max+1, n_y_min:n_y_max+1]

                z_pick = pick_z_laplacian(z_tif_image)
                z_pick_list.append(z_pick)

                texture_features = [extract_texture_features(im, sub_mask_im) for im in z_tif_image]
                texture_features = pd.DataFrame(texture_features)
                texture_features['z'] = np.arange(len(z_tif_image))
                # texture_features['t'] = i
                list_df_textures.append(texture_features)

                if i>0:
                    # Get IOU
                    intersection = np.logical_and(mask_idx>0, prev_mask>0).sum()
                    union = np.logical_or(mask_idx>0, prev_mask>0).sum()
                    iou_abs = 1 - intersection / union if union != 0 else 0

                    if prev_contour_recon is not None:
                        try:
                            iou, _, _ = compute_iou(contour_recon, prev_contour_recon, shape=(256, 256))
                            iou = 1 - iou
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
                
                dict_other_features['iou'] = iou
                dict_other_features['iou_abs'] = iou_abs
                dict_other_features['t'] = i

                traj.append(centroid)
                list_dict.append(dict_efd_features|dict_other_features)
                
            else:
                contour_recon = None

            prev_mask = np.copy(mask_idx)
            prev_contour_recon = np.copy(contour_recon)


        # list_coeffs, traj = np.asarray(list_coeffs), np.asarray(traj)
        traj = np.asarray(traj)

        # _, ax = plt.subplots()
        # ax.imshow(visu_mask[:],origin='lower')
        # ax.plot(traj[:,1],traj[:,0],c='black',linewidth=0.5, marker='.')
        # ax.set_aspect('equal')
        # plt.show()

        # name_cols = list(itertools.chain.from_iterable([[f'a_{i}', f'b_{i}', f'c_{i}', f'd_{i}'] for i in range(order)]))
        # df = pd.DataFrame(list_coeffs.reshape((list_coeffs.shape[0],list_coeffs.shape[1]*list_coeffs.shape[2])), \
        #                 columns = name_cols)

        majority_vote_z = np.bincount(z_pick_list).argmax()
        df_cell_texture = pd.concat([x[x['z']==majority_vote_z] for x in list_df_textures])
        df_cell_texture.reset_index(inplace=True)

        df_cell = pd.DataFrame(list_dict)

        # speed, angles = compute_metrics_trajectory_fourier2(traj)
        df_traj = pd.DataFrame({'traj_x':traj[:,0], 'traj_y':traj[:,1]})
                            #'speed': speed, 'angles':angles})
        
        df = pd.concat((df_cell, df_traj, df_cell_texture),axis=1)
        df['Cell bf'] = idx_bf

        # Concatenate with other cells
        df_final = pd.concat((df_final, df))
    
        # Save for safety
        if count_bf%50==0:
            df_final.to_csv(ROOTPATH + exp_path + 'live_cell_features/' + condition + '_fourier2.csv', index=False)

    # except:
    #     pass  
        
    df_final.to_csv(ROOTPATH + exp_path + 'live_cell_features/' + condition + '_fourier2.csv', index=False)

    return(df_final)



def process(pick_exp_path, pick_cond):

    list_tif_images, list_masks = load_images_and_masks(pick_exp_path, pick_cond)

    df = pd.read_csv(save_path_model + 'experiment.csv')

    df = df[(df['Exp']==pick_exp_path)&(df['Condition']==pick_cond)]

    idx_df = 0
    big_features_tab = pd.DataFrame()

    # unique_bf_idx = np.unique(list_masks[list_masks>0])
    unique_bf_idx = df['Cell bf']
    cond_cell, exp_path =  pick_cond, pick_exp_path

    for idx_df in unique_bf_idx:
        # name_cell, cond_cell, exp_path = df['Cell bf'].iloc[idx_df], df['Condition'].iloc[idx_df], df['Exp'].iloc[idx_df]
        # cell_path = [ROOTPATH + exp_path + 'Live/' + cond_cell + '/live_dataset/cell_'+str(name_cell) + '/t' + str(t_).zfill(2) + '.tiff' for t_ in range(21)]
        # alignment_path = ROOTPATH + exp_path + 'alignment/' + cond_cell + '-alignment.csv'
        # alignment = pd.read_csv(alignment_path)
        # bf_center = np.asarray([alignment.loc[alignment['BF']==name_cell, 'BF_center_x'], \
        #                         alignment.loc[alignment['BF']==name_cell, 'BF_center_y']])[:,0]
        name_cell = idx_df
        # Get the bounding box of the cell
        t,x,y = np.where(list_masks==name_cell)
        unique_t = np.unique(t)
        centroids = np.array([[(x[t==t_]).mean(), (y[t==t_]).mean()] for t_ in unique_t]).astype(int)

        xmin,xmax,ymin,ymax = x.min(), x.max(), y.min(), y.max()
        velocity_map = np.zeros((xmax-xmin+1,ymax-ymin+1))
        for t_ in unique_t:
            velocity_map[x[t==t_]-xmin,y[t==t_]-ymin]+=1

        # plt.imshow(velocity_map,origin='lower')
        # plt.plot(centroids[:,1]-ymin,centroids[:,0]-xmin,c='black',marker=".")
        # plt.show()

        features_tab = pd.DataFrame()
        for t_ in unique_t:
            xmin, xmax, ymin, ymax = x[t==t_].min(),x[t==t_].max(),y[t==t_].min(),y[t==t_].max()
            mask = np.zeros((xmax-xmin+1, ymax-ymin+1))
            mask[x[t==t_]-xmin,y[t==t_]-ymin]=1
            im = list_tif_images[t_,:,xmin:xmax+1, ymin:ymax+1]
            plt.imshow(im[0])
            plt.show()
            plt.imshow(list_masks[t_,xmin:xmax+1, ymin:ymax+1]==name_cell)
            plt.show()
            features = extract_cell_features(im, mask)
            features['t'] = t_
            features_tab = pd.concat((features_tab,features), ignore_index=True)
        features_tab['Cell_bf'] = name_cell

        # Now add trajectory information
        traj = np.copy(centroids)
        speed, angles, path_length, net_displacement, tortuosity = compute_metrics_trajectory(traj)
        features_tab['angles'] = None
        features_tab['speed'] = None
        for t_ in range(1,len(angles)+1):
            features_tab.loc[features_tab['t']==t_, 'angles'] = angles[t_-1]
        for t_ in range(len(speed)):
            features_tab.loc[features_tab['t']==t_, 'speed'] = speed[t_]
        features_tab['path_length'] = path_length
        features_tab['net_displacement'] = net_displacement
        features_tab['tortuosity'] = tortuosity
        big_features_tab = pd.concat((big_features_tab,features_tab), ignore_index=True)
        
    big_features_tab.to_csv(ROOTPATH + pick_exp_path + 'live_cell_features/' + pick_cond + '.csv', index=False)



def visu_trajectory(list_masks, idx_bf):
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
            count_unique = [len(np.where(labeled_mask==x)[0]) for x in unique_labeled]
            print(idx_bf, i, unique_labeled, count_unique)
            new_label = unique_labeled[np.argmax(count_unique)]
            mask_idx = labeled_mask==new_label

        
        if np.sum(mask_idx)>500:
            visu_mask += mask_idx.astype(np.int16)
        
    # Final crop again
    idx_x, idx_y = np.nonzero(visu_mask)
    x_min, x_max, y_min, y_max = idx_x.min(),idx_x.max()+1, idx_y.min(),idx_y.max()+1
    visu_mask = visu_mask[x_min:x_max, y_min:y_max]
    return(visu_mask)



def reduce_features_fourier(df_fourier, order=100, filter_error=True):
    df_magnitude = pd.DataFrame({f'magnitude_{i}': \
                np.linalg.norm(df_fourier[[f'a_{i}', f'b_{i}', f'c_{i}', f'd_{i}']], axis=1) for i in range(order)})
    df_elliptic_area = pd.DataFrame({f'elliptic_area_{i}': \
                (df_fourier[f'a_{i}']*df_fourier[f'd_{i}'] - df_fourier[f'b_{i}']*df_fourier[f'c_{i}'])/(i+1) for i in range(order)})
    df_cumsum = pd.DataFrame({f'cumsum_magnitude_{i}': \
                              df_magnitude.cumsum(axis=1)[f'magnitude_{i}'] for i in range(order)})
    df_area = df_cumsum[f'cumsum_magnitude_{order-1}']
    df_area.name = 'area'
    df_magnitude = df_magnitude.div(df_area, axis=0)
    df_fourier.index = df_magnitude.index
    df_elliptic_area.index = df_magnitude.index
    df_fourier = df_fourier.join(df_magnitude).join(df_elliptic_area).join(df_area)
    
    df_group = df_fourier.abs().groupby('Cell bf')
    new_df = df_group.aggregate(['mean', 'std'])
    new_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in new_df.columns]
    new_df = pd.concat((new_df, pd.DataFrame({'is_on_edge':df_group['is_on_edge'].any()})),axis=1)
    new_df = pd.concat((new_df, pd.DataFrame({'count':df_group['t'].count()})),axis=1)

    # Compute trajectory characteristics
    new_df2 = pd.DataFrame()
    for idx, traj in df_group[['traj_x', 'traj_y']]:
        new_dict = compute_metrics_trajectory_fourier(np.asarray(traj))
        new_df2 = pd.concat((new_df2, pd.DataFrame(new_dict, index=[idx])))
    new_df = pd.concat((new_df, new_df2),axis=1)

    # Filter the error
    if filter_error:
        new_df = new_df[new_df['max_speed']<200]
        new_df = new_df[new_df['is_on_edge']==False]

    return(df_fourier, new_df)  


