# Import local tools
from tools.read_yaml import *
from tools.imports import *


# === PARAMETERS ===
image_folder = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/non_merged/'
save_folder = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/'
rows, cols = 4, 4                # Grid size
overlap = 0.1

print(image_folder, save_folder)

# === Helper: extract tile index from filename ===
def extract_position(fname):
    match = re.search(r'_s(\d+)_', fname)
    return int(match.group(1)) if match else None

# Scan all all t frames 
# Get all possible z
frame_names_z = list(pathlib.Path(image_folder).glob('*_t00_s00_z*_ch00.tif'))
pattern = r"_z(\d{2})_"
match_z = [re.search(pattern, str(filename)) for filename in frame_names_z]
match_z = [m.group(1) for m in match_z if m]
match_z.sort()
max_z_ = str(len(match_z)-1).zfill(2)

# Get all possible t
frame_names_t = list(pathlib.Path(image_folder).glob('*_t*_s00_z{}_ch00.tif'.format(max_z_)))
pattern = r"_t(\d{2})_"
match_t = [re.search(pattern, str(filename)) for filename in frame_names_t]
match_t = [m.group(1) for m in match_t if m]
match_t.sort()

# Get all possible tiles s
frame_names_s = list(pathlib.Path(image_folder).glob('*_t00_s*_z00_ch00.tif'.format(max_z_)))
pattern = r"_s(\d{2})_"
match_s = [re.search(pattern, str(filename)) for filename in frame_names_s]
match_s = [m.group(1) for m in match_s if m]
match_s.sort()

def stitch(t_, z_):

    # Get all tiles
    file_dict = [list(pathlib.Path(image_folder).glob('*_t{}_s{}_z{}_ch00.tif'.format(t_, s_, z_))) for s_ in match_s]

    # === Load sample image to get dimensions ===
    sample = tifffile.imread(file_dict[0])
    H, W = sample.shape
    overlap_x = int(W * overlap)
    overlap_y = int(H * overlap)
    step_x = W - overlap_x
    step_y = H - overlap_y

    # === Create canvas and weight map ===
    stitched = np.zeros((step_y * (rows - 1) + H, step_x * (cols - 1) + W), dtype=np.float32)
    weight_map = np.zeros_like(stitched, dtype=np.float32)

    # === Row/col index with snake pattern ===
    def get_row_col(index, num_cols):
        row = index // num_cols
        col = index % num_cols if row % 2 == 0 else num_cols - 1 - (index % num_cols)
        return row, col

    # === Stitching with blending ===
    for idx in range(rows * cols):
        fname = file_dict[idx]
        img = tifffile.imread(fname).astype(np.float32)
        row, col = get_row_col(idx, cols)

        y = row * step_y
        x = col * step_x

        h, w = img.shape

        # Create blending weights
        wy = np.ones(h)
        wx = np.ones(w)

        if row > 0:
            wy[:overlap_y] = np.linspace(0, 1, overlap_y)
        if row < rows - 1:
            wy[-overlap_y:] = np.linspace(1, 0, overlap_y)

        if col > 0:
            wx[:overlap_x] = np.linspace(0, 1, overlap_x)
        if col < cols - 1:
            wx[-overlap_x:] = np.linspace(1, 0, overlap_x)

        weight = np.outer(wy, wx)

        # Place image and weights into stitched canvas
        stitched[y:y+h, x:x+w] += img * weight
        weight_map[y:y+h, x:x+w] += weight

    # === Normalize by accumulated weights to finish blending ===
    stitched_final = (stitched / np.maximum(weight_map, 1e-6)).astype(np.uint16)

    # === Save and/or show ===
    # imageio.imwrite('stitched_blended.tif', stitched_final)
    # plt.imshow(stitched_final, cmap='gray')
    # plt.title('Blended 4×4 Mosaic')
    # plt.axis('off')
    # plt.show()
    output_path = save_folder +"BF_custom_stitched_merged_t{}_z{}_ch00.tif".format(t_, z_)
    # return(stitched_final)
    tifffile.imwrite(output_path, stitched_final)


for t_ in match_t:
    for z_ in match_z:
        stitch(t_, z_)