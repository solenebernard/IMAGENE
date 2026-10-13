
import numpy as np
import tifffile
import pathlib
from PIL import Image
import re
import cv2

def load_masks(rootpath, exp_path, condition, clean=False, t_max=None):
    data_path_raw = rootpath + exp_path + 'Live/' + condition + '/'
    if clean:
        path_masks_sam2 = data_path_raw + 'clean_masks/'
    else:
        path_masks_sam2 = data_path_raw + 'results/'

    # Get all frames
    frame_names_t = list(pathlib.Path(data_path_raw).glob('*_t*_z{}_ch00.tif'.format('00')))
    pattern = r"_t(\d{2})_"
    match_t = [re.search(pattern, str(filename)) for filename in frame_names_t]
    match_t = [m.group(1) for m in match_t if m]
    match_t.sort()
    
    if t_max is not None:
        match_t = np.arange(t_max)
        match_t = [str(t).zfill(2) for t in match_t]

    list_masks = []
    for i in range(len(match_t)):
        mask = np.asarray(Image.open(path_masks_sam2 + 'seg_movie_merged_t{}_cp_mask.png'.format(str(i).zfill(2)))) # The mask
        list_masks.append(mask)
    list_masks = np.asarray(list_masks) # Dimension: (t,x,y)
    return(list_masks)


def load_masks_nuclear(rootpath, exp_path, condition):
    data_path = rootpath + exp_path + 'Live/' + condition + '/'
    # Get all frames
    frame_names_t = list(pathlib.Path(data_path).glob('*_t*_z{}_ch00.tif'.format('00')))
    pattern = r"_t(\d{2})_"
    match_t = [re.search(pattern, str(filename)) for filename in frame_names_t]
    match_t = [m.group(1) for m in match_t if m]
    match_t.sort()

    data_path_mask = data_path + 'nuclear_segmentations/'
    list_masks = []
    for i in range(len(match_t)):
        image_path = list(pathlib.Path(data_path_mask).glob('*_t{}_*cp_*.png'.format(str(i).zfill(2))))
        mask = np.asarray(Image.open(image_path[0])) # The mask
        list_masks.append(mask)
    list_masks = np.asarray(list_masks) # Dimension: (t,x,y)
    return(list_masks)

def load_bf_images(rootpath, exp_path, condition, t_max = None):
    data_path_raw = rootpath + exp_path + 'Live/' + condition + '/'

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

    if t_max is not None:
        match_t = np.arange(t_max)
        match_t = [str(t).zfill(2) for t in match_t]
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

def load_images_and_masks(rootpath, exp_path, condition, clean=False, t_max=None):
    list_tif_images = load_bf_images(rootpath, exp_path, condition, t_max=t_max)
    list_masks = load_masks(rootpath, exp_path, condition, clean=clean, t_max=t_max)
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
