"""
Sobel texture + Convolutional Autoencoder pipeline
File: sobel_autoencoder.py
Dependencies: numpy, scikit-image, scipy, matplotlib, torch, torchvision, umap-learn (optional)

What this script provides:
- apply_sobel(image): compute Sobel magnitude of a grayscale brightfield FOV
- extract_patches(img, patch_size, stride): patch generator (overlapping)
- hand-crafted feature extraction (global stats, histograms, GLCM)
- PyTorch ConvAutoencoder (encoder -> latent vector -> decoder)
- training loop + helper functions to get latent embeddings for downstream analysis

Adaptable to Streamlit or other pipelines.
"""

# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

import glob
from skimage import io, color, util
from skimage.filters import sobel, sobel_h, sobel_v
from skimage.util import view_as_windows
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label

import umap
import numexpr as ne

# ------------------ Image / Sobel helpers ------------------

def load_image_gray(path):
    img = io.imread(path)
    # if img.ndim == 3:
        # img = color.rgb2gray(img)
    img = util.img_as_float32(img)
    return img


def apply_sobel(img):
    """Compute Sobel magnitude image (texture / edge emphasis).
    Input: img (2D, float in [0,1])
    Returns: sobel_mag (2D, float)
    """
    return sobel(img)


def extract_patches(img, patch_size=64, stride=32, flatten=False):
    """Return patches of shape (N, patch_size, patch_size)
    Uses view_as_windows for efficiency (requires image >= patch_size).
    """
    if img.ndim != 2:
        raise ValueError('expected 2D grayscale image')
    h, w = img.shape
    if h < patch_size or w < patch_size:
        raise ValueError('image smaller than patch_size')
    windows = view_as_windows(img, (patch_size, patch_size), step=stride)
    # windows shape: (n_h, n_w, patch_size, patch_size)
    n_h, n_w = windows.shape[:2]
    patches = windows.reshape(-1, patch_size, patch_size)
    if flatten:
        patches = patches.reshape(patches.shape[0], -1)
    return patches


# ------------------ Hand-crafted Sobel feature extraction ------------------

def sobel_features2D(img, fov_stats=None, n_bins=16, glcm_distances=[1], glcm_angles=[0]):
    """
    Extract hand-crafted features from Sobel edges:
    - Global mean, variance, percentiles of magnitude
    - Histogram of gradient magnitudes
    - Histogram of gradient orientations
    - GLCM/Haralick stats (contrast, homogeneity, energy, correlation)

    Returns: 1D numpy array of features.
    """
    gx = sobel_h(img)
    gy = sobel_v(img)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)

    if fov_stats is not None:
        mean, std = fov_stats   # precomputed from the whole image
        mag = (mag - mean) / (std + 1e-8)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    feats = []
    # Global stats
    feats.append(np.mean(mag))
    feats.append(np.var(mag))
    feats.extend(np.percentile(mag, [25, 50, 75]))

    # Histogram of magnitudes
    hist_mag, _ = np.histogram(mag, bins=n_bins, range=(0, mag.max()), density=True)
    feats.extend(hist_mag)

    # Histogram of orientations (-pi, pi)
    hist_ang, _ = np.histogram(ang, bins=n_bins, range=(-np.pi, np.pi), density=True)
    feats.extend(hist_ang)

    # GLCM features on quantized Sobel magnitude
    mag_q = util.img_as_ubyte(mag / (mag.max() + 1e-8))
    glcm = graycomatrix(mag_q, distances=glcm_distances, angles=glcm_angles, symmetric=True, normed=True)
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        val = graycoprops(glcm, prop)
        feats.extend(val.flatten())

    return np.array(feats, dtype=np.float32)


# def sobel_features(img, n_bins=16, glcm_distances=[1], glcm_angles=[0]):
#     gx = sobel_h(img)
#     gy = sobel_v(img)

#     # Normalize gradients by image contrast
#     gx /= (np.std(img) + 1e-8)
#     gy /= (np.std(img) + 1e-8)

#     mag = np.sqrt(gx**2 + gy**2)
#     ang = np.arctan2(gy, gx)

#     # Normalize magnitude to [0,1]
#     mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

#     feats = []
#     # Global stats (on normalized magnitude)
#     feats.append(np.mean(mag_norm))
#     feats.append(np.var(mag_norm))
#     feats.extend(np.percentile(mag_norm, [25, 50, 75]))

#     # Histogram of magnitudes (fixed range [0,1])
#     hist_mag, _ = np.histogram(mag_norm, bins=n_bins, range=(0,1), density=True)
#     feats.extend(hist_mag)

#     # Histogram of orientations (-pi, pi), independent of contrast
#     hist_ang, _ = np.histogram(ang, bins=n_bins, range=(-np.pi, np.pi), density=True)
#     feats.extend(hist_ang)

#     # GLCM on normalized, quantized magnitudes
#     mag_q = util.img_as_ubyte(mag_norm)
#     glcm = graycomatrix(mag_q, distances=glcm_distances, angles=glcm_angles,
#                         symmetric=True, normed=True)
#     for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
#         val = graycoprops(glcm, prop)
#         feats.extend(val.flatten())

#     return np.array(feats, dtype=np.float32)

def get_fov_stats(list_tif_images):
    n_t, m_z = list_tif_images.shape[0], list_tif_images.shape[1]
    list_stats = np.zeros((n_t, m_z, 2))
    for t, imgs in enumerate(list_tif_images):
        for z, img in enumerate(imgs):
            fov_mag = apply_sobel(img)
            fov_mean, fov_std = fov_mag.mean(), fov_mag.std()
            list_stats[t, z, 0] = fov_mean
            list_stats[t, z, 1] = fov_std
    return(list_stats)


def extract_sobel_features(exp_path, condition):
    list_live_cells = glob.glob(ROOTPATH + exp_path + '/Live/' + condition + '/live_dataset/cell*')

    n_bins = 16
    feats_name = ['mean_sob', 'var_sob', '25_perc_sob', '50_perc_sob', '75_perc_sob']
    feats_name += [f'hist_{b}_grad' for b in range(n_bins)]
    feats_name += [f'hist_{b}_ang' for b in range(n_bins)]
    feats_name += ['contrast', 'homogeneity', 'energy', 'correlation']

    # Whole FOV stats for normalization
    # Get all z and t
    list_tif_images = load_bf_images(ROOTPATH, exp_path, condition)
    list_stats = get_fov_stats(list_tif_images)

    # # Iterate through bf indices
    # unique_bf_idx, bf_counts = np.unique(list_masks[list_masks>0], return_counts=True)
    # # Remove bf cell id with very few pixels
    # unique_bf_idx = unique_bf_idx[bf_counts>1000]
    # mask_shape = (list_masks.shape[1], list_masks.shape[2])

    # for count_bf, idx_bf in enumerate(unique_bf_idx):

    #     df = pd.DataFrame()

    #     # To improve accuracy: crop the big mask
    #     sub_mask = ne.evaluate("list_masks == idx_bf")
    #     proj_mask = np.any(sub_mask, axis=0)
    #     # Collapse time axis using `np.any()` to get where the object exists in space
    #     proj_mask = np.any(sub_mask, axis=0)  # shape (H, W), True where object appears in any frame
    #     idx_x, idx_y = np.nonzero(proj_mask)
    #     x_min, x_max, y_min, y_max = idx_x.min(),idx_x.max()+1, idx_y.min(),idx_y.max()+1
    #     new_mask = sub_mask[:, x_min:x_max, y_min:y_max]
    #     new_image = list_tif_images[:, :, x_min:x_max, y_min:y_max]
        
    #     for i,mask_idx in enumerate(new_mask):

    #         labeled_mask = label(mask_idx)
    #         unique_labeled = np.unique(labeled_mask[labeled_mask>0])
    #         unique_labeled.sort()
    #         if len(unique_labeled) > 1: # Meaning there is more than one cell
    #             # Keep the largest ?
    #             count_unique = np.asarray([len(np.where(labeled_mask==x)[0]) for x in unique_labeled])
    #             ratio = count_unique/count_unique.max()
    #             print(idx_bf, i, unique_labeled, count_unique)
    #             new_label = unique_labeled[np.argmax(count_unique)]
    #             mask_idx = labeled_mask==new_label            
            
    #         if np.sum(mask_idx)>500:
    #             for t,im in enumerate(new_image[i]):

    df = pd.DataFrame()
    for path in list_live_cells[:]:
        cell_id = int(path.split('_')[-1])
        imgs_path = glob.glob(path+'/texture/t*.tiff')
        for img_path in imgs_path:
            # img = load_image_gray(img_path).transpose((2,0,1))
            t = int(img_path.split('/')[-1][1:3])
            img = tifffile.imread(img_path)
            # sob = apply_sobel(img)
            feats = [sobel_features2D(s, fov_stats=list_stats[t, z], n_bins=n_bins) for z,s in enumerate(img)]
            row = pd.DataFrame(feats, columns=feats_name)
            row.index.name = 'z'
            row['cell_bf'] = cell_id
            row['t'] = t
            row = row.set_index(['cell_bf', 't'], append=True)
            df = pd.concat((df, row))
    return(df)

# ------------------ Example usage ------------------


# # Compute Fourier Features from BF masks
pick_cond = CONDITION
pick_exp_path = EXP_PATH
df = extract_sobel_features(pick_exp_path, pick_cond)
df.to_csv(ROOTPATH + pick_exp_path + 'live_cell_features/texture_' + pick_cond + '.csv')

# # Create a single big file
# gene_list = [ "CORO1A", "TNF", "ACTB", "IL2RA", "IRF8", "CD69", "CCR7", "GZMB", "GBP4", "IFNG", "XP01", "TNF_2"] # "CD8", "CORO1A_2"
# list_exp_path, list_condition = ['2025-03-04_Proof-of-concept-Olga/', \
#                                 '2025-03-31_Proof-of-concept-Olga2/', \
#                                  '2025-04-13_Proof-of-concept-Olga3/', \
#                                  '2025-08-03_Proofofconcept-Olga4/'], \
#                                 [['1-NA', '2-4H-Mael','3-4H-Nico', '4-24H-Nico'], \
#                                  ['1-NA', '2-4H','3-24H', '4-MIX'], ['1-NA', '2-4H','3-24H', '4-MIX'], \
#                                     ['1-NA', '2-4H','3-24H', '4-MIX']]


# # Create a single big file
# gene_list = [ "CORO1A", "TNF", "ACTB", "IL2RA", "IRF8", "CD69", "CCR7", "GZMB", "GBP4", "IFNG", "XP01", "TNF_2"] # "CD8", "CORO1A_2"
# list_exp_path, list_condition = ['2025-03-04_Proof-of-concept-Olga/', \
#                                 '2025-03-31_Proof-of-concept-Olga2/', \
#                                  '2025-04-13_Proof-of-concept-Olga3/', \
#                                  '2025-08-03_Proofofconcept-Olga4/'], \
#                                 [['1-NA', '2-4H-Mael','3-4H-Nico', '4-24H-Nico'], \
#                                  ['1-NA', '2-4H','3-24H', '4-MIX'], ['1-NA', '2-4H','3-24H', '4-MIX'], \
#                                     ['1-NA', '2-4H','3-24H', '4-MIX']]
# all_df = pd.DataFrame()
# for exp_path, l_cond in zip(list_exp_path, list_condition):
#     for condition in l_cond:
#         # df = extract_sobel_features(exp_path, condition)
#         # print(exp_path, condition, df.shape, all_df.shape)
#         # df.to_csv(ROOTPATH + exp_path + 'live_cell_features/texture_' + condition + '.csv')
#         try:
#             df = pd.read_csv(ROOTPATH + exp_path + 'live_cell_features/texture_' + condition + '.csv', index_col=['cell_bf', 't', 'z'])
#             df['condition'] = condition
#             df['exp_path'] = exp_path
#             df = df.set_index(['exp_path', 'condition'], append=True)
#             all_df = pd.concat((all_df, df))
#         except:
#             pass
# all_df.to_csv(ROOTPATH + 'learning_activation/texture.csv')
# all_df = all_df.dropna()


# feats_textures_mean = all_df.groupby(['exp_path', 'condition', 'cell_bf']).agg('mean')
# # feats_textures_mean = feats_textures_mean.dropna()

# umap_model = umap.UMAP(n_components=2, random_state=42)
# feats_textures_mean = (feats_textures_mean-feats_textures_mean.mean())/feats_textures_mean.var()
# feats_textures_mean = feats_textures_mean[['contrast', 'homogeneity', 'energy',
#        'correlation',
#        'hist_14_ang', 'hist_15_ang']]
# x_2D = umap_model.fit_transform(feats_textures_mean)
# feats_textures_mean['umap_0'] = x_2D[:,0]
# feats_textures_mean['umap_1'] = x_2D[:,1]

# plt.figure(figsize=(10,10))
# # for condition in ['1-NA', '2-4H-Mael', '3-4H-Nico', '4-24H-Nico']:
# # for condition in ['1-NA', '2-4H', '3-24H']:
#     # plt.scatter(feats_textures_mean.loc[('2025-03-31_Proof-of-concept-Olga2/', condition, slice(None)), 'umap_0'], \
#     #             feats_textures_mean.loc[('2025-03-31_Proof-of-concept-Olga2/', condition, slice(None)), 'umap_1'], s=1)
#     # plt.scatter(feats_textures_mean.loc[(slice(None), condition, slice(None)), 'umap_0'], \
#     #             feats_textures_mean.loc[(slice(None), condition, slice(None)), 'umap_1'], s=1)
# # for exp_path in ['2025-03-04_Proof-of-concept-Olga/', \
# #                                 '2025-03-31_Proof-of-concept-Olga2/']:
# for exp_path in feats_textures_mean.index.get_level_values('exp_path').unique():
#     plt.scatter(feats_textures_mean.loc[(exp_path, slice(None), slice(None)), 'umap_0'], \
#             feats_textures_mean.loc[(exp_path, slice(None), slice(None)), 'umap_1'], s=1, label=exp_path)
# plt.legend()
# plt.show()
# for condition in feats_textures_mean.index.get_level_values('condition').unique():
#     plt.scatter(feats_textures_mean.loc[(slice(None), condition, slice(None)), 'umap_0'], \
#             feats_textures_mean.loc[(slice(None), condition, slice(None)), 'umap_1'], s=1, label=condition)
# plt.legend()
# # plt.aspect('equal')
# plt.show()



# # if __name__ == '__main__':
# # Example: change these to your file and params
# img_path = '/Volumes/projects/pbi_helix/2025-03-31_Proof-of-concept-Olga2/Live/1-NA/live_dataset/cell_14/texture/t04.tiff'  # replace with your file
# patch_size = 64
# stride = 32
# latent_dim = 64
# batch_size = 64
# n_epochs = 25

# # 1) Load image and compute Sobel texture
# img = load_image_gray(img_path).transpose((2,0,1))
# sob = apply_sobel(img)
# # quick display
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1); plt.imshow(img[2], cmap='gray'); plt.title('orig'); plt.axis('off')
# plt.subplot(1,2,2); plt.imshow(sob[2], cmap='gray'); plt.title('sobel'); plt.axis('off')
# plt.show()


# # 2) Extract patches from the SOBEL magnitude image
# patches = extract_patches(sob, patch_size=patch_size, stride=stride)
# print(f'extracted {len(patches)} patches of size {patch_size}x{patch_size}')

# # 3) Compute hand-crafted features on the first few patches
# feats = [sobel_features(p) for p in patches[:10]]
# print('example handcrafted feature vector shape:', np.array(feats).shape)

# # 4) Create dataset and dataloader for autoencoder
# ds = PatchDataset(patches)
# loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

# # 5) Build model and train
# model = ConvAutoencoder(latent_dim=latent_dim)
# model = train_autoencoder(model, loader, n_epochs=n_epochs, lr=1e-3)

# # 6) Visualize reconstructions
# plot_reconstructions(model, ds, n=6)

# # 7) Get latent embeddings and optionally visualize with UMAP
# latents = get_latents(model, loader)
# print('latents shape:', latents.shape)
# if umap is not None:
#     visualize_latents_umap(latents)

# # 8) save model
# torch.save(model.state_dict(), 'conv_autoencoder.pth')
# print('saved conv_autoencoder.pth')

# # End of file
