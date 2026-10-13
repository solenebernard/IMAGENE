
"""
Create single cell video of each cell. A unique z plane is chose via Laplacian matrix for each cell.
Stored in: ROOPATH/EXP_PATH/Live/CONDITION/live_dataset/cell_*/t*.tiff
"""

# Import local tools and usual libraries
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

# Import additional libraries
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt

def min_max_norm(image):
    """
    Normalize the image to [0, 1]
    """
    image_min = np.min(image)
    image_max = np.max(image)
    norm_image = (image - image_min) / (image_max - image_min)
    return norm_image

def avg_std_norm(image):
    """
    Normalize with mean and std (after you remove 0 values)
    Because there might be a non-meaningful black border (0 values) 
    which influences the average
    """
    image_mean = np.mean(image[image>0])
    image_std = np.std(image[image>0])
    image[image==0] = image_mean
    norm_image = (image - image_mean) / image_std
    return norm_image


def custom_normalize(image):
    """
    Normalize the image (z,h,w) to (h,w)
    """
    im = np.zeros((image.shape[1:])).astype(np.float64)
    for j in range(image.shape[0]):
        sub_im = image[j]
        sub_im = avg_std_norm(sub_im)
        im += sub_im
    im = avg_std_norm(im)
    return(im)

def remove_border(im):
    """
    The `remove_border` function in Python removes the border with 0 values from an image.
    
    :param im: NumPy array representing an image
    :return: `im` with the border cropped out, if any.
    """
    image = np.copy(im)
    # Crop the border with 0 values
    _,x,y = np.where(image>0)
    if len(x)>0:
        xmin, xmax, ymin, ymax = x.min(),x.max(), y.min(), y.max()
        image = image[:, xmin:xmax,ymin:ymax]
    return(image)


def matrix_decay(size, size_radius, decay_factor=-6e-3):
    '''
    Calculates a decay matrix based on the distance from the center of a square matrix.
    :param size: size of the square matrix that will be generated. 
    It determines the dimensions of the matrix (size x size)
    :param size_radius: the radius within which the distance values will be considered. 
    Any distance less than `size_radius` will be set to 0 in the calculation
    :param decay_factor: controls how quickly the values in the matrix decay 
    as the distance from the center increases.
    '''
    distance = np.meshgrid(np.arange(size)-size//2, np.arange(size)-size//2)
    distance = np.sqrt(distance[0]**2 + distance[1]**2)
    distance -= size_radius
    distance[distance<0] = 0
    distance = distance**2 # Make it smooth
    # Make the value exponentially decrease
    decay = np.exp(decay_factor*distance)/np.max(np.exp(decay_factor*distance))
    return(distance, decay)


def mask_decay(mask, decay_factor=-2e-2):
    """
    Compute exponential decay field outside a binary mask contour.
    
    Parameters
    ----------
    mask : 2D ndarray
        Binary mask (1 inside shape, 0 outside).
    
    Returns
    -------
    distance : 2D ndarray
        Distance from the contour (0 inside, increasing outward).
    decay : 2D ndarray
        Normalized exponential decay field (1 at boundary, decreasing outward).
    """
    # Distance transform outside the mask
    distance = distance_transform_edt(mask == 0)

    # Zero inside the mask
    distance[mask > 0] = 0

    # Smooth (square, like your circle case)
    distance = distance**2  

    # Exponential decay
    decay = np.exp(decay_factor * distance)
    decay /= decay.max()

    return distance, decay

def cell_crop(path_folder_save, list_tif_images, list_masks, id_cell, Xmin, Xmax, Ymin, Ymax, size=224):
    """
    Crop the image with a large fixed size bounding box around the cell.
    Get the bounding box by getting the extremes coordinates of the cell 
    in the mask over all time frames. Then remove outside the border by setting to 0
    """
    # Create a folder for each idx
    idx_folder = os.path.join(path_folder_save, f'cell_{id_cell}')
    os.makedirs(idx_folder, exist_ok=True)
    # create folder for the mask
    path_folder_mask =  os.path.join(idx_folder, 'texture')
    os.makedirs(path_folder_mask, exist_ok=True)

    # Get the bounding box of the cell
    t,x,y = np.where(list_masks==id_cell)
    unique_t = np.unique(t)
    centroids = np.array([[(x[t==t_]).mean(), (y[t==t_]).mean()] for t_ in unique_t]).astype(int)
    
    save_path = os.path.join(idx_folder, 'trajectory.npy')
    trajectory = np.concatenate((unique_t[:,None],centroids),axis=1)
    np.save(save_path, trajectory)

    list_z_plane = []

    n_match_z = list_tif_images.shape[1]
    all_images = np.zeros((len(unique_t), n_match_z, size, size)).astype(np.float64)
    all_masks = np.zeros((len(unique_t), size, size)).astype(np.float64)

    # Iterate over all t and remember the best z plan 
    for t,(t_, center) in enumerate(zip(unique_t, centroids)):
        
        xmin, xmax = center[0] - size//2, center[0] + size//2
        ymin, ymax = center[1] - size//2, center[1] + size//2
        # Check the bounds in the big image
        xmin0, ymin0 = max(xmin, Xmin), max(ymin, Ymin)
        xmax0, ymax0  = min(xmax, Xmax - 1), min(ymax, Ymax - 1)
        # Check the bounds in the small image
        xmin1, ymin1 = max(-xmin, 0), max(-ymin, 0)
        xmax1, ymax1  = xmin1 + (xmax0-xmin0), ymin1 + (ymax0-ymin0)
        # Tighter crop for laplacian
        size2 = 100
        xmin, xmax = center[0] - size2//2, center[0] + size2//2
        ymin, ymax = center[1] - size2//2, center[1] + size2//2
        xmin2, ymin2 = max(xmin, Xmin), max(ymin, Ymin)
        xmax2, ymax2  = min(xmax, Xmax - 1), min(ymax, Ymax - 1)

        # Use Laplacian metric to measure blurness
        z = pick_z_laplacian(list_tif_images[t_, :, xmin2:xmax2, ymin2:ymax2])
        im = list_tif_images[t_, :, xmin0:xmax0, ymin0:ymax0]
        list_z_plane.append(z)
        all_images[t, :, xmin1:xmax1, ymin1:ymax1] = im
        all_masks[t, xmin1:xmax1, ymin1:ymax1] = list_masks[t_, xmin0:xmax0, ymin0:ymax0]==id_cell

    # Majority vote: get the most popular z 
    list_z_plane = np.asarray(list_z_plane)
    count_z = np.asarray([np.sum(list_z_plane==i) for i in range(n_match_z)])
    majority_z = np.argmax(count_z)
    
    # Now select each image in the most popular z plan
    for t,t_ in enumerate(unique_t):

        # Pick the good z
        distance, decay = mask_decay(all_masks[t], decay_factor=-2e-2)
        final_image = all_images[t, :]
        # Replace 0 value by background value
        background = np.asarray([f[distance>0] for f in final_image])
        background_mean = np.asarray([b[b>0].mean() for b in background])
        for i in range(len(final_image)):
            (final_image[i])[final_image[i]==0] = background_mean[i]
        image_background = np.ones_like(final_image) * background_mean[:,None,None]
        image_background = gaussian_filter(final_image, sigma=100)

        # Optional: add gradient? To erase the surroundings of the centered cell
        # Make a large circle with radius filled with 1 values in the middle and 0 
        # smoothly oustide
        final_image = decay*final_image + (1-decay)*image_background

        # Save as 16 bits: save the cropped original image
        final_image[final_image>2**16-1]=2**16-1
        final_image[final_image<0]=0
        final_image = np.round(final_image).astype(np.uint16)

        # Save the image in TIFF format
        output_tiff_path = os.path.join(idx_folder, 't{}.tiff'.format(str(t_).zfill(2)))
        # plt.imshow(final_image[majority_z])
        # plt.show()
        tifffile.imwrite(output_tiff_path, final_image[majority_z])
        
        # All z stack
        tifffile.imwrite(os.path.join(path_folder_mask, 't{}.tiff'.format(str(t_).zfill(2))), \
                         final_image)
    
    # And the mask
    tifffile.imwrite(os.path.join(path_folder_mask, 'mask.tiff'), all_masks)
    
    return(majority_z)

def main():
    
    # Initialize variables
    data_path_raw = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/'
    save_path_end = 'live_dataset/'

    # Constants
    size_radius = SIZE_RADIUS

    list_tif_images, list_masks = load_images_and_masks(ROOTPATH, EXP_PATH, CONDITION, clean=True)

    # Get the dimensions of the images
    Xmin, Xmax = 0, list_tif_images.shape[2]
    Ymin, Ymax = 0, list_tif_images.shape[3]

    # Get index of cells
    idx_cells = np.unique(list_masks[list_masks>0])

    # Create the new dataset folder
    path_new_data = os.path.join(data_path_raw, save_path_end)
    os.makedirs(path_new_data, exist_ok=True)


    list_z_choice = []
    for id_cell in idx_cells:
        z = cell_crop(path_new_data, list_tif_images, list_masks, id_cell, \
            Xmin=Xmin, Xmax=Xmax, Ymin=Ymin, Ymax=Ymax)
        list_z_choice.append(z)

    df = pd.DataFrame({'Cell_bf':idx_cells, 'Pick_z':np.asarray(list_z_choice)})
    df.to_csv(ROOTPATH + EXP_PATH + 'live_cell_features/pick_z_'+CONDITION+'.csv' )


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = 2000000000
    main()

