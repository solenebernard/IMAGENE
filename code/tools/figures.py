'''
Shared tools for the figures (7_figures/)
- parameters of the figures (7_figures/config_figures.yaml)
- loading of the dataset, of the images and of the masks
- saving of the figures
'''

import matplotlib
matplotlib.use('Agg') # The figures are saved, not shown
from matplotlib import rc
from skimage import measure
from sklearn.preprocessing import StandardScaler

# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_fish import show_mip
from tools.tools_bf import load_tracks, load_global_id_map, load_tracked_mask, load_tracked_tile_mask
from tools.matching import passes_ratio_qc_table

rc('font', family='sans-serif')
rc('font', **{'sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans']})
Image.MAX_IMAGE_PIXELS = 2000000000

# Parameters of the figures
with open("7_figures/config_figures.yaml", "r") as f:
    FIG = yaml.safe_load(f)

FIG_CONDITION = FIG['condition']
DATA_PATH = ROOTPATH + EXP_PATH
# Folder of the example images, or None to read the images from the folders of the pipeline
EXAMPLE_DIR = pathlib.Path(FIG['data_dir']) if FIG['data_dir'] is not None else None


def save_figure(fig, name, formats=('pdf', 'png'), **kwargs):
    '''Save the figure in OUTPUT_DIR/name.pdf and .png, and close it'''
    path = pathlib.Path(FIG['output_dir']) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(f'{path}.{fmt}', format=fmt, dpi=300, **kwargs)
    plt.close(fig)
    print(f'Saved {path}')


def save_video(frames, output_path, fps):
    '''
    Save the frames (H, W, 3), in BGR as for OpenCV, as a mp4 video
    Encoded in H.264 with ffmpeg when it is available (read by all the players), else in MPEG-4 with OpenCV
    '''
    import shutil
    import subprocess
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.ascontiguousarray(frame, dtype=np.uint8) for frame in frames]
    height, width = frames[0].shape[:2]

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is not None and b'libx264' in subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True).stdout:
        # The size must be even for H.264
        command = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
                   '-r', str(fps), '-i', '-', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p', '-crf', '18', str(output_path)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f'ffmpeg failed to write {output_path}')
    else:
        import cv2
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()
        print('ffmpeg with libx264 not found: the video is encoded in MPEG-4, which some players do not read')
    print(f'Saved {output_path}')


def plot_contour(ax, binary_mask, transpose=False, **kwargs):
    '''Plot the contour of a binary mask. Returns False if the mask is empty'''
    contours = measure.find_contours(binary_mask)
    if len(contours) == 0:
        return False
    contour = contours[0]
    if transpose:
        ax.plot(contour[:, 0], contour[:, 1], **kwargs)
    else:
        ax.plot(contour[:, 1], contour[:, 0], **kwargs)
    return True


##### DATASET

def load_dataset(ratio_qc=RATIO_QC):
    '''
    Table of features and gene expression of step 5 (6_results/dataset.csv)
    The gene expression is kept only for the cells used by the models: passing the quality control of the
    matching (ratio_qc), not on the edge, present in all the time frames, and with enough RNA (the matched cells
    with almost no RNA in every gene, below learning.min_mean_rna, are removed)
    '''
    from tools.model import mean_rna_per_gene
    df = pd.read_csv('6_results/dataset.csv', index_col=['exp_path', 'condition', 'cell_bf'])
    bool_match = passes_ratio_qc_table(df, ratio_qc) & (df['is_on_edge'] == False) & (df['count'] == N_FRAMES) \
                 & (mean_rna_per_gene(df) >= MIN_MEAN_RNA)
    df.loc[~bool_match, 'Fish'] = None
    for gene in LEARNING_GENES:
        df.loc[~bool_match, f'Gene_{gene}'] = None
    return df


def select_features(df):
    '''The features of the live cells used by the model (same selection as in step 5)'''
    columns_remove = [c for c in df.columns \
                      if c.startswith(('Gene_', 'delta_z_', 'Fish', 'Distance', 'Id_2nd', 'is_on_edge', 'count', \
                                       'BF_center_', 'traj_', 'centroid-', 'moments_', 'inertia_', 'umap'))]
    columns_remove += ['avg_speed', 'max_speed', 'speed_std', 'elongation', 'total_distance']
    return [x for x in df.columns if x not in columns_remove]


def cd8_positive(df):
    '''Index of the CD8 positive cells: at least two RNA of CD8'''
    return df.index[df['Gene_CD8'] > 1]


def add_umap_features(df, features, random_state=42):
    '''UMAP of the features of the live cells: columns umap_0 and umap_1'''
    import umap
    X = df[features].fillna(df[features].median())
    x_transform = umap.UMAP(n_components=2, random_state=random_state).fit_transform(StandardScaler().fit_transform(X))
    return pd.concat((df, pd.DataFrame(x_transform, columns=['umap_0', 'umap_1'], index=X.index)), axis=1)


def add_umap_genes(df, random_state=42):
    '''UMAP of the gene expression of the CD8 positive cells: columns umap2_0 and umap2_1'''
    import umap
    Y = df[[f'Gene_{gene}' for gene in LEARNING_GENES]].dropna(axis=0, how='any')
    Y = Y[Y['Gene_CD8'] > 1]
    x_transform = umap.UMAP(n_components=2, random_state=random_state).fit_transform(StandardScaler().fit_transform(Y))
    return pd.concat((df, pd.DataFrame(x_transform, columns=['umap2_0', 'umap2_1'], index=Y.index)), axis=1)


##### IMAGES AND MASKS
# Read from the example images of the figures (EXAMPLE_DIR), or from the folders of the pipeline if data_dir is null

def load_bf_tiles(condition, s, z=Z_FINETUNE, list_t=None):
    '''BF images of the tile s for the time frames list_t. Dimension: (t,x,y)'''
    if EXAMPLE_DIR is not None:
        folder = EXAMPLE_DIR / 'live_tiles' / condition
    else:
        folder = pathlib.Path(DATA_PATH + 'Live/' + condition + '/non_merged/')
    if list_t is None:
        list_t = range(N_FRAMES)
    paths = [list(folder.glob(f'*_t{str(t).zfill(2)}_s{str(s).zfill(2)}_z{str(z).zfill(2)}_ch00.tif'))[0] for t in list_t]
    return np.asarray([tifffile.imread(p) for p in paths])


def load_tile_masks(condition, s, z=Z_FINETUNE, list_t=None):
    '''Masks of the tile s with the track ID of each cell, for the time frames list_t. Dimension: (t,x,y)'''
    if list_t is None:
        list_t = range(N_FRAMES)
    if EXAMPLE_DIR is not None:
        folder = EXAMPLE_DIR / 'live_tiles' / condition
        paths = [folder / f'renamed_file_t{str(t).zfill(2)}_s{str(s).zfill(2)}_z{str(z).zfill(2)}.png' for t in list_t]
        return np.asarray([np.asarray(Image.open(p)) for p in paths])
    tracks, id_map = load_tracks(ROOTPATH, EXP_PATH, condition), load_global_id_map(ROOTPATH, EXP_PATH, condition)
    return np.asarray([load_tracked_tile_mask(ROOTPATH, EXP_PATH, condition, t, s, tracks, id_map, z=z) for t in list_t])


def load_merged_bf(condition, t, z):
    '''Merged BF image of the time frame t and the z slice z'''
    if EXAMPLE_DIR is not None:
        folder = EXAMPLE_DIR / 'live_merged' / condition
    else:
        folder = pathlib.Path(DATA_PATH + 'Live/' + condition + '/')
    path = list(folder.glob(f'*_t{str(t).zfill(2)}_z{str(z).zfill(2)}_ch00.tif'))[0]
    return tifffile.imread(path)


def load_merged_mask(condition, t):
    '''Global mask of the time frame t, with the track ID of each cell'''
    if EXAMPLE_DIR is not None:
        return np.asarray(Image.open(EXAMPLE_DIR / 'live_merged' / condition / f'seg_movie_merged_t{str(t).zfill(2)}_cp_mask.png'))
    return load_tracked_mask(ROOTPATH, EXP_PATH, condition, t, load_tracks(ROOTPATH, EXP_PATH, condition))


def load_fish_mip(condition, s, t, ch):
    '''Maximum intensity projection of a FISH tile, for the round t and the channel ch'''
    if EXAMPLE_DIR is not None:
        return tifffile.imread(EXAMPLE_DIR / 'fish' / condition / f'mip_s{str(s).zfill(2)}_t{str(t).zfill(2)}_ch{str(ch).zfill(2)}.tif')
    return show_mip(ROOTPATH, EXP_PATH, condition, s, t, ch)


def load_fish_tile(condition, s, t, ch):
    '''Maximum intensity projection and detected spots (z,x,y) of a FISH tile, for the round t and the channel ch'''
    s_, t_, ch_ = str(s).zfill(2), str(t).zfill(2), str(ch).zfill(2)
    mip = load_fish_mip(condition, s, t, ch)
    if EXAMPLE_DIR is not None:
        spots = np.load(EXAMPLE_DIR / 'fish' / condition / f'spots_s{s_}_t{t_}_ch{ch_}.npy')
    else:
        spots = np.load(DATA_PATH + 'Fish/bigfish/' + condition + f'/t{t_}_s{s_}_ch{ch_}.npy')
    return mip, spots.astype(np.int64)


def load_fish_tile_mask(condition, s):
    '''Mask of the cells of a FISH tile'''
    s_ = str(s).zfill(2)
    if EXAMPLE_DIR is not None:
        return np.asarray(Image.open(EXAMPLE_DIR / 'fish' / condition / f'mask_s{s_}.png'))
    return tifffile.imread(DATA_PATH + 'Fish/masks/' + condition + f'/mosaic_mask_reassigned_s{s_}.tiff')


def load_fish_dapi_mip(condition, s):
    '''Maximum intensity projection of the DAPI channel of a FISH tile (first round)'''
    s_ = str(s).zfill(2)
    if EXAMPLE_DIR is not None:
        return tifffile.imread(EXAMPLE_DIR / 'fish' / condition / f'dapi_mip_s{s_}.tif')
    paths = sorted(pathlib.Path(DATA_PATH + 'Fish/' + condition + '/').glob(f'*_t00_s{s_}_z*_ch02.tif'))
    return np.max([tifffile.imread(p) for p in paths], axis=0)


def load_clumps_thresholds(condition):
    '''Thresholds of the clumps selected in step 2.4, indexed by (s, t, ch)'''
    if EXAMPLE_DIR is not None:
        path = EXAMPLE_DIR / 'fish' / condition / 'clumps_thresholds.csv'
    else:
        path = DATA_PATH + f'Fish/clumps_detection/{condition}_thresholds.csv'
    return pd.read_csv(path, index_col=['s', 't', 'ch'])


def load_overlay(condition):
    '''Overlay of step 3: BF mask in the first channel, FISH mask in the second one, aligned'''
    if EXAMPLE_DIR is not None:
        return np.load(EXAMPLE_DIR / 'matching' / f'{condition}-overlay.npz')['overlay']
    return np.load(DATA_PATH + f'alignment/{condition}-overlay.npy')


def load_alignment(condition):
    '''Table of the pairs of BF and FISH cells of step 3'''
    if EXAMPLE_DIR is not None:
        return pd.read_csv(EXAMPLE_DIR / 'matching' / f'{condition}-alignment.csv')
    return pd.read_csv(DATA_PATH + f'alignment/{condition}-alignment.csv')
