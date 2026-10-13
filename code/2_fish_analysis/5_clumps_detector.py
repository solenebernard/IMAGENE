
# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_fish import *
from scipy import ndimage
import imageio.v2 as imageio

# Version with the merged mask
OVERLAP = 206
IM_SIZE = 2048
MINSIZE = 100

path_clumps = f'{ROOTPATH}{EXP_PATH}Fish/clumps_detection/'
thresh_csv = pd.read_csv(path_clumps + f'{CONDITION}_thresholds.csv', \
    index_col=['s','t','ch'])


pattern_snake_path = ROOTPATH + EXP_PATH + "Fish/pattern_snake/" + CONDITION + ".npy"
if pathlib.Path(pattern_snake_path).exists():
    pattern_snake = np.load(pattern_snake_path, allow_pickle = True)
else: # the default one: 5x5
    pattern_snake = np.load(ROOTPATH + 'olgarithm/config_patterns/5x5.npy')
n_rows, n_cols = pattern_snake.shape
    

def remove_small_regions(mask, min_size):
    """
    Set to 0 any connected region in the mask with fewer than `min_size` pixels.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask (2D array).
    min_size : int
        Minimum number of pixels a region must have to be kept.
    
    Returns
    -------
    np.ndarray
        Cleaned binary mask.
    """
    # Label connected components
    labeled, num = ndimage.label(mask)
    
    # Count pixels in each component
    sizes = np.bincount(labeled.ravel())
    
    # Mask out small components
    remove = sizes < min_size
    remove_mask = remove[labeled]
    
    cleaned = mask.copy()
    cleaned[remove_mask] = 0
    return cleaned

    
# Create all masks
for t in range(8):
    for ch in range(2):
        list_mips = []
        for s in range(25):
            mip = show_mip(ROOTPATH, EXP_PATH, CONDITION, s, t, ch)
            list_mips.append(mip)
        thresh = thresh_csv.loc[:,t,ch,:].max().values[0]
    
        s_, t_, ch_ = str(s).zfill(2), str(t).zfill(2), str(ch).zfill(2)
        # Manually
        
        total_image = np.zeros((IM_SIZE*n_rows-(n_rows-1)*OVERLAP, \
                                IM_SIZE*n_cols-(n_cols-1)*OVERLAP))
        X_shape, Y_shape = total_image.shape[0], total_image.shape[1]

        # Do the snake
        for i in range(n_rows):
            for j in range(n_cols):
                id_tile = pattern_snake[i,j]
                
                if id_tile is not None:
                    im = np.copy(list_mips[id_tile]>thresh)
                    # im = label(im)
                    im[im>0] = im[im>0] + 1000*id_tile
                    # im = remove_cell_border(im)
                    
                    if i==0:
                        xmin, xmax = 0,IM_SIZE
                    else:
                        xmin, xmax = i*IM_SIZE-(i)*OVERLAP, IM_SIZE*(i+1)-i*OVERLAP

                    # Go from left to right
                    if j==0:
                        ymin, ymax = 0, IM_SIZE
                    else:
                        ymin, ymax = j*IM_SIZE-(j)*OVERLAP, IM_SIZE*(j+1)-j*OVERLAP
                    
                    # Fix where the two indices overlay
                    overlap = (total_image[xmin:xmax, ymin:ymax]>0)&(im>0) # get the mask for overlay
                    index_overlap_1 = np.unique(im[overlap]) # get indices in the first image

                    # Priority to the first image: erase indices in the second image
                    for x in index_overlap_1:
                        im[im==x] = 0
                    
                    # Add 
                    total_image[xmin:xmax, ymin:ymax] += im   

        total_image = total_image>0
        total_image = remove_small_regions(total_image, MINSIZE)
        imageio.imwrite(path_clumps + f'{CONDITION}_mask_total_t{t_}_ch{ch_}.tiff', total_image.astype(np.uint8))
