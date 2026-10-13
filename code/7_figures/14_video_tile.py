'''
Video of the live cell timelapse of one BF tile, with the mask and the ID of each tracked cell

Saved in OUTPUT_DIR/video/tile_{condition}_s{tile}.mp4
Input: BF tiles and tile masks and tracks of step 1

Run: python3 -m 7_figures.14_video_tile
     python3 -m 7_figures.14_video_tile --condition 1-NA --tile 3
     python3 -m 7_figures.14_video_tile --exp_path /path/to/experiment/ --condition 1-NA   (other experiment than config.yaml)
     python3 -m 7_figures.14_video_tile --data_dir 7_figures/data/                          (example images)
'''

import argparse
import cv2

import tools.figures
from tools.figures import *

params = FIG['video_tile']
parser = argparse.ArgumentParser()
parser.add_argument("--condition", type=str, default=FIG_CONDITION)
parser.add_argument("--tile", type=int, default=params['tile'])
parser.add_argument("--exp_path", type=str, default=None, help='Folder of the experiment, if different from config.yaml')
parser.add_argument("--data_dir", type=str, default=None, help='Folder of the example images, if different from config_figures.yaml')
args = parser.parse_args()
if args.exp_path is not None:
    # Read the images from this experiment instead of the example images or of the experiment of config.yaml
    tools.figures.DATA_PATH = args.exp_path
    tools.figures.EXAMPLE_DIR = None
if args.data_dir is not None:
    tools.figures.EXAMPLE_DIR = pathlib.Path(args.data_dir)
condition, s = args.condition, args.tile


def id_to_color(unique_id):
    '''Deterministic color for a cell ID'''
    rng = np.random.default_rng(seed=int(unique_id))
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def render_frame(image, mask, t, alpha):
    '''BF image with the mask (transparent color and contour) and the ID of each cell'''
    frame = np.tile(image[:,:,None], (1,1,3))
    colored = frame.copy()
    list_id = np.unique(mask[mask>0])
    for id_cell in list_id:
        colored[mask==id_cell] = id_to_color(id_cell)
    frame = cv2.addWeighted(colored, alpha, frame, 1-alpha, 0)
    for id_cell in list_id:
        binary = (mask==id_cell).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, contourIdx=-1, color=id_to_color(id_cell), thickness=2)
    # The ID above each cell, in white with a black outline
    for id_cell in list_id:
        ys, xs = np.where(mask==id_cell)
        (text_width, _), _ = cv2.getTextSize(str(id_cell), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        position = (int(xs.mean()-text_width/2), max(int(ys.min())-8, 20))
        cv2.putText(frame, str(id_cell), position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, str(id_cell), position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # Time frame
    cv2.putText(frame, f't={t}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, f't={t}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    return frame


image_bf = load_bf_tiles(condition, s)
masks = load_tile_masks(condition, s)

# Same contrast for all the time frames
vmin, vmax = np.quantile(image_bf, 0.001), np.quantile(image_bf, 0.999)
image_bf = (np.clip((image_bf-vmin)/(vmax-vmin), 0, 1)*255).astype(np.uint8)

frames = [render_frame(image, mask, t, params['alpha']) for t, (image, mask) in enumerate(zip(image_bf, masks))]
output_path = pathlib.Path(FIG['output_dir']) / f'video/tile_{condition}_s{str(s).zfill(2)}.mp4'
save_video(frames, output_path, fps=params['fps'])
print(f'{len(frames)} frames, {len(np.unique(masks[masks>0]))} cells')
