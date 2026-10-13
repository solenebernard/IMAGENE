'''
Video of the pipeline for one FISH tile
1. live cell timelapse with the contour of the tracked cells
2. DAPI of the fixed cells with the contour of the FISH cells (same color as the matched live cell)
3. projection of the smFISH signal of a gene
4. matching of the live and fixed cells

Saved in OUTPUT_DIR/video/movie.mp4
Input: merged BF images, global masks and tracks of step 1, FISH images and masks of step 2, overlay and table of step 3
'''

import cv2

from tools.figures import *

params = FIG['video']
condition = FIG_CONDITION
s = params['tile']
n_repeat = 20 # Number of frames of each still image


def normalize_to_uint8(frame, eps=1e-8):
    '''Normalize a frame to uint8 [0,255]'''
    frame = frame.astype(np.float32)
    frame = (frame - frame.min()) / (frame.max() - frame.min() + eps)
    return (255 * frame).clip(0, 255).astype(np.uint8)


def id_to_color(unique_id):
    '''Deterministic color for a cell ID'''
    rng = np.random.default_rng(seed=int(unique_id))
    return tuple(int(c) for c in rng.integers(50, 255, size=3))


def contour_mask_movie(movie_frames, movie_masks, thickness=2):
    '''
    movie_frames: (T, H, W, 3)
    movie_masks:  (T, H, W) masks of the cells
    returns: (T, H, W, 3) frames with the contour of each cell, colored by its ID
    '''
    frames = []
    for frame, mask_t in zip(movie_frames, movie_masks):
        frame = frame.copy()
        for uid in np.unique(mask_t[mask_t>0]):
            contours, _ = cv2.findContours((mask_t == uid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, contourIdx=-1, color=id_to_color(uid), thickness=thickness)
        frames.append(frame)
    return np.asarray(frames)


def centroid_from_mask(mask, obj_id):
    ys, xs = np.where(mask == obj_id)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())  # (x, y)


def draw_connections(frame, bf_mask, fish_mask, match_df, line_color=(0, 255, 0), thickness=5):
    '''Draw a line between the centroids of the matched BF and FISH cells'''
    for id_bf, id_fish in zip(match_df['BF'], match_df['Fish']):
        c_bf = centroid_from_mask(bf_mask, id_bf)
        c_fish = centroid_from_mask(fish_mask, id_fish)
        if c_bf is None or c_fish is None:
            continue
        cv2.line(frame, c_bf, c_fish, color=line_color, thickness=thickness)
        cv2.circle(frame, c_fish, 4, (255, 0, 0), -1)
        cv2.circle(frame, c_bf, 4, (0, 0, 0), -1)
    return frame


def add_transition(movie_frames, next_frames, n_transition=6):
    '''Blend the last frame of the movie into the first frame of the next part, then add the next part'''
    last, first = movie_frames[-1].astype(np.float32), next_frames[0].astype(np.float32)
    transition = [(1 - alpha) * last + alpha * first for alpha in np.linspace(0, 1, n_transition)]
    return np.concatenate([movie_frames, np.stack(transition).astype(np.uint8), next_frames], axis=0)


def add_title(frames, title, font_scale=2, thickness=3):
    '''Write a centered title at the top of all the frames'''
    frames = frames.astype(np.uint8).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(title, font, font_scale, thickness)
    position = ((frames.shape[2] - text_width) // 2, text_height + 20)
    for frame in frames:
        cv2.putText(frame, title, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frames


##### LOAD THE DATA
# Position of the tile in the mosaic of FISH tiles
pattern_snake_path = DATA_PATH + "Fish/pattern_snake/" + condition + ".npy"
if pathlib.Path(pattern_snake_path).exists():
    pattern_snake = np.load(pattern_snake_path, allow_pickle=True)
else:
    pattern_snake = np.load(FISH_DEFAULT_PATTERN)
row, col = [int(x[0]) for x in np.where(pattern_snake == s)]

# FISH images and mask of the tile
dapi = tifffile.imread(DATA_PATH + f'Fish/mip/{condition}_round_0_s{str(s).zfill(2)}_mip.tif')[1]
fish = load_fish_mip(condition, s, params['fish_round'], params['fish_channel'])
step = FISH_TILE_SIZE - FISH_OVERLAP
mask_fish = tifffile.imread(DATA_PATH + f'Fish/masks/{condition}/mosaic_mask_reassigned.tiff')
mask_fish = mask_fish[row*step:row*step+FISH_TILE_SIZE, col*step:col*step+FISH_TILE_SIZE]

# Matching: first channel of the overlay is the BF mask after alignment, the second one the FISH mask
overlay, match = load_overlay(condition), load_alignment(condition)

# BF images and masks
bf_images = np.asarray([load_merged_bf(condition, t, params['z']) for t in range(N_FRAMES)])
list_masks = np.asarray([load_merged_mask(condition, t) for t in range(N_FRAMES)])

def first_pixel(mask, id_cell):
    '''First row and first column of a cell of the mask'''
    return np.array([x.min() for x in np.where(mask == id_cell)])


def offset_between(mask_a, mask_b):
    '''Translation from mask_b to mask_a (position in a = position in b + offset), from a cell present in both'''
    for id_cell in np.unique(mask_b[mask_b>0]):
        if np.any(mask_a == id_cell):
            pos_a, pos_b = first_pixel(mask_a, id_cell), first_pixel(mask_b, id_cell)
            # Skip the cells cut by the border of one of the masks
            if np.all(pos_a > 0) and np.all(pos_b > 0):
                return pos_a - pos_b
    raise ValueError('No cell in common between the two masks')


# The tile at the pixel size of the BF image
size_im = int(np.round(FISH_TILE_SIZE*SCALE))
dapi = cv2.resize(dapi, dsize=(size_im, size_im), interpolation=cv2.INTER_LINEAR)
mask_fish = cv2.resize(mask_fish, dsize=(size_im, size_im), interpolation=cv2.INTER_NEAREST)
fish = cv2.resize(fish, dsize=(size_im, size_im), interpolation=cv2.INTER_NEAREST)

# Position of the tile in the overlay (from its FISH cells), and of the overlay in the BF image (from its BF cells)
tile_x, tile_y = offset_between(overlay[:,:,1], mask_fish)
crop_x, crop_y = offset_between(list_masks[-1], overlay[:,:,0])
tile_overlay = (slice(tile_x, tile_x+size_im), slice(tile_y, tile_y+size_im))
tile_bf = (slice(crop_x+tile_x, crop_x+tile_x+size_im), slice(crop_y+tile_y, crop_y+tile_y+size_im))
movie_frames = bf_images[(slice(None),) + tile_bf]
movie_masks = list_masks[(slice(None),) + tile_bf]

# The reliable pairs of the tile
match_tile = match[match['BF'].isin(np.unique(movie_masks[-1]))]
match_tile = match_tile[passes_ratio_qc_table(match_tile, RATIO_QC)]
# Give the matched FISH cells the ID of their BF cell: same color in the video
rename_fish_mask = np.copy(mask_fish)
for id_bf, id_fish in zip(match_tile['BF'], match_tile['Fish']):
    rename_fish_mask[mask_fish==id_fish] = id_bf


##### THE FOUR PARTS OF THE VIDEO
# 1. Live cell timelapse
movie_gray = np.tile(normalize_to_uint8(movie_frames)[:,:,:,None], (1,1,1,3))
part_bf = contour_mask_movie(movie_gray, movie_masks, thickness=2)

# 2. DAPI with the contour of the FISH cells
dapi_color = np.tile(normalize_to_uint8(dapi)[:,:,None], (1,1,3))
dapi_color[:,:,-1] = 0
part_dapi = np.tile(contour_mask_movie(dapi_color[None], rename_fish_mask[None], thickness=2), (n_repeat,1,1,1))

# 3. Projection of the smFISH signal
fish = normalize_to_uint8(np.clip(fish, a_min=None, a_max=np.quantile(fish, 0.9995)))
part_fish = np.tile(fish[None,:,:,None], (n_repeat,1,1,3))

# 4. Matching: BF cells in gray, FISH cells in color, and a line between the matched cells
overlay_tile = (overlay[tile_overlay]>0)*255
bf_overlay = np.tile(overlay_tile[:,:,:1], (1,1,3))
fish_overlay = np.stack((overlay_tile[:,:,1], np.zeros_like(overlay_tile[:,:,1]), np.zeros_like(overlay_tile[:,:,1])), axis=-1)
part_matching = draw_connections(((fish_overlay+bf_overlay)/2).astype(np.uint8), movie_masks[-1], mask_fish, match_tile)
part_matching = np.tile(part_matching[None], (n_repeat,1,1,1))

full_movie = add_title(part_bf, 'Live cell timelapse')
full_movie = add_transition(full_movie, add_title(part_dapi, 'DAPI stained immobilized and fixed cells'))
full_movie = add_transition(full_movie, add_title(part_fish, f'z-projection of {params["fish_gene"]} smFISH'))
full_movie = add_transition(full_movie, add_title(part_matching, 'Cell matching across live and fixed modalities', font_scale=1.5))

save_video(full_movie, pathlib.Path(FIG['output_dir']) / 'video/movie.mp4', fps=params['fps'])
