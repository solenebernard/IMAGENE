# Code based on Video predictor example notebook SAM2 https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb
# RUN SAM 2 for all cells simultansously
# The results are then filtered based on cell segmentations appearing in all frames
# and having a minimum size of 500 pixels (to avoid random noise effects)
'''
Optimize code to run SAM2 propagation on big images.
'''

# Import local tools
from tools.read_yaml import *
from tools.imports import *

## FOR MULTIPROCESSING
parser = argparse.ArgumentParser()
parser.add_argument("--loop_i", type=int)
args = parser.parse_args()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from skimage.measure import regionprops, label
from matplotlib.patches import Rectangle
from sam2.build_sam import build_sam2_video_predictor

# Initialize variables
data_path_live = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/'
video_dir = data_path_live + 'video/'
sam2_checkpoint = MODEL_PATH
path_output = data_path_live + 'results/'
device = DEVICE
backward = BACKWARD # If we take the first or last t frames of brightfield mask

model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Upload cellpose segmentation
mask_path_list = list(pathlib.Path(video_dir).glob('*_cp_masks.png')) # Result given by cell pose
mask_path_list.sort()
# Take the last segmentation 
if backward:
    mask_path = mask_path_list[-1]
else:
    mask_path = mask_path_list[0]
print(mask_path)
cellpose_segmentation = Image.open(mask_path)


if device == "gpu":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    
elif device == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# scan all the JPEG frame names in this directory


# Scan all all t frames 
# Get all possible z
frame_names_z = list(pathlib.Path(data_path_live).glob('*_t00_z*_ch00.tif'))
pattern = r"_z(\d{2})_"
match_z = [re.search(pattern, str(filename)) for filename in frame_names_z]
match_z = [m.group(1) for m in match_z if m]
match_z.sort()
max_z_ = str(len(match_z)-1).zfill(2)

# GET TIF FILE NAMES
frame_names = list(pathlib.Path(data_path_live).glob('*t*_z{}_ch00.tif'.format(max_z_)))
frame_names.sort()

# Build the model
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


# Iterate over each cells
cellpose_segmentation = np.asarray(cellpose_segmentation)
cell_indices = np.unique(cellpose_segmentation[cellpose_segmentation>0])
regions = regionprops(cellpose_segmentation)
centroids = np.array([np.int16(np.floor(x.centroid))[::-1] for x in regions])

im_size = 2048
overlap = im_size//3
crop_center_size = im_size-2*overlap

n_images_0, n_images_1 = np.asarray(cellpose_segmentation).shape[0]//(crop_center_size)+1, \
    np.asarray(cellpose_segmentation).shape[1]//(crop_center_size)+1


def get_bounding_box(mask):
    """Find the bounding box of a binary mask (True or False) using min/max values."""
    # Find where the mask is 1
    y_indices, x_indices = np.where(mask)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None  # No foreground pixels

    # Compute bounding box (x_min, y_min, x_max, y_max)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    return (x_min, y_min, x_max, y_max)


def run_loop(loop_i, loop_j, predictor, frame_names, cellpose_segmentation, im_size, crop_center_size, overlap, video_dir, path_output, backward = False):

    # Take only origins in the center
    xmin, xmax = loop_i*crop_center_size, (loop_i+1)*crop_center_size
    ymin, ymax = loop_j*crop_center_size, (loop_j+1)*crop_center_size

    # The big image
    Xmin, Xmax = xmin-overlap, xmax+overlap
    Ymin, Ymax = ymin-overlap, ymax+overlap

    # New image (size im_size x im_size)
    xmin_, xmax_, ymin_, ymax_ = 0, im_size, 0, im_size
    if Xmin<0:
        xmin_ = overlap
    if Ymin<0:
        ymin_ = overlap
    
    # Handle the border
    Xmin, Xmax = max(Xmin,0), min(Xmax,cellpose_segmentation.shape[0])
    Ymin, Ymax = max(Ymin,0), min(Ymax,cellpose_segmentation.shape[1])
    if Ymax>=cellpose_segmentation.shape[1]:
        ymax_ = Ymax-Ymin
    if Xmax>=cellpose_segmentation.shape[0]:
        xmax_ = Xmax-Xmin
    # Select a big crop of the mask
    mask = np.zeros((im_size,im_size)).astype(np.uint16)
    mask[xmin_:xmax_,ymin_:ymax_] += cellpose_segmentation[Xmin:Xmax, Ymin:Ymax]

    # Get centers only in the crop in the middle
    sub_regions = regionprops(mask)
    sub_centroids = np.array([np.int16(np.floor(x.centroid)) for x in sub_regions])
    bool_pos = np.where((sub_centroids[:,0]>=overlap)&(sub_centroids[:,0]<=crop_center_size+overlap)\
                        &(sub_centroids[:,1]>=overlap)&(sub_centroids[:,1]<=crop_center_size+overlap))
    sub_pos = sub_centroids[bool_pos]
    sub_index = np.asarray([mask[x[0],x[1]] for x in sub_pos])

    # Only if there are some cells detected
    if len(sub_index)>0:
        
        # Keep only cells which centroids are in the middle of the big crop
        sub_mask = np.zeros_like(mask)
        for x in sub_index:
            sub_mask[mask==x] = x

        # Define the path for the new folder
        new_folder_path = video_dir + f'subsampled_{loop_i}_{loop_j}/'
        os.makedirs(new_folder_path, exist_ok=True)

        t_max = len(frame_names)
        
        list_frame_names = frame_names.copy()
        if backward:
            # Go backward! 
            list_frame_names = list_frame_names[::-1]

        for t_,im_file in enumerate(list_frame_names):
            
            # Open the big TIF file
            im = tifffile.imread(im_file)
            im = np.asarray(Image.fromarray(im/255))
            image = np.zeros((im_size,im_size))

            image[xmin_:xmax_,ymin_:ymax_] += im[Xmin:Xmax, Ymin:Ymax]
            image = Image.fromarray(image.astype(np.uint8))
            if backward:
                name_image = f'{str(t_).zfill(2)}_{str(t_max-t_).zfill(2)}_{loop_i}_{loop_j}.jpg'
            else:
                name_image = f'{str(t_).zfill(2)}_{str(t_).zfill(2)}_{loop_i}_{loop_j}.jpg'
            image.save(new_folder_path + name_image, quality=100)

        # Use SAM2 to forward propagade the segmentation
        inference_state = predictor.init_state(video_path=new_folder_path)
        predictor.reset_state(inference_state)

        video_segments = {}  # video_segments contains the per-frame segmentation results
        ann_frame_idx = 0
        labels = np.array([1], np.int32)

        # For all cells: add it to the predictor
        for idx, center in zip(sub_index, sub_pos):
            bbox = get_bounding_box(sub_mask==idx)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=idx,
                    points=center[None],
                    labels=labels,
                    box=bbox
                    )

        # Propagate to SAM2
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Fetch the result and save
        masks_output = []
        for k in video_segments.keys():
            # Filter the too big cells ?
            mask_output = np.zeros((im_size,im_size), dtype=np.uint16)
            for key in video_segments[k].keys():
                area = np.sum(video_segments[k][key])
                if (area>500)&(area<10000): # Minimum and maximum size of cell
                    mask_output += key*video_segments[k][key][0]
            masks_output.append(mask_output)
        masks_output = np.asarray(masks_output, dtype=np.uint16)

        # Monitor evolution of the cell
        # Because cell can disappear from the FOV ! 
        unique_id_cell = np.unique(masks_output[0, masks_output[0]>0]) 
        for unique_id in unique_id_cell:
            # Check the evolution of the area of the cell
            evolution_area = np.mean(masks_output==unique_id, axis=(1,2))
            # If at the time point the cell looses 80% of its area
            t_first = np.where((evolution_area/evolution_area[0])<0.2)[0]
            if len(t_first)>0:
                # Then remove all the pixels for following frames
                for t in range(t_first[0], len(masks_output)):
                    masks_output[t, masks_output[t]==unique_id]=0

        # Store only the position of positive values
        # To save storage
        t,x,y = np.where(masks_output>0)
        id = masks_output[t,x,y]
        tab = np.stack((id, t, x, y))
        tab[2] += xmin - overlap
        tab[3] += ymin - overlap
        np.save(path_output + f'seg_movie_{loop_i}_{loop_j}.npy', tab)

# for loop_i in range(0,1):
for loop_i in range(args.loop_i, args.loop_i+1):
     for loop_j in range(n_images_1):
         run_loop(loop_i, loop_j, predictor, frame_names, cellpose_segmentation, im_size, crop_center_size, overlap, video_dir, path_output, backward=BACKWARD)

# Merge all the segmentations
list_tab = []
for loop_i in range(n_images_0):
    for loop_j in range(n_images_1):
        try:
            tab = np.load(path_output + f'seg_movie_{loop_i}_{loop_j}.npy')
            list_tab.append(tab)
        except:
            pass

# Concatenate all positions
list_tab = np.concatenate(list_tab,axis=1)
# Filter with possible values
list_tab = list_tab[:, list_tab[2]<cellpose_segmentation.shape[0]]
list_tab = list_tab[:, list_tab[3]<cellpose_segmentation.shape[1]]

final_segmentation = np.zeros((len(frame_names), ) + cellpose_segmentation.shape, dtype=np.uint16)
final_segmentation[list_tab[1],list_tab[2],list_tab[3]] = list_tab[0]

            

# Save to png
t_max = len(final_segmentation)
for t,x in enumerate(final_segmentation):
    im_seg = Image.fromarray(x.astype(np.uint16))
    if backward:
        t_ = str(t_max-t-1).zfill(2)
    else:
        t_ = str(t).zfill(2)
    im_seg.save(path_output + f'seg_movie_merged_t{t_}_cp_mask.png')
    