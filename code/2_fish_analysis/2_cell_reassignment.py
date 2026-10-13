import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import tifffile
from skimage.measure import regionprops, label

OVERLAP = 206
IM_SIZE = 2048

# Import local tools
from tools.read_yaml import *
from tools.imports import *


'''
This code is for merge all the individual segmentations of FISH tiles
And changes the ID to have a unique ID for each fish mask
The snake pattern done by the microscope is by default 5x5
But it can change, the pattern is stored in EXP_PATH/Fish/pattern_snake/CONDITION.npy
'''

pattern_snake_path = ROOTPATH + EXP_PATH + "Fish/pattern_snake/" + CONDITION + ".npy"
if pathlib.Path(pattern_snake_path).exists():
    pattern_snake = np.load(pattern_snake_path, allow_pickle = True)
else: # the default one: 5x5
    pattern_snake = np.load(ROOTPATH + 'olgarithm/config_patterns/5x5.npy')
folder_mask = ROOTPATH + EXP_PATH + "Fish/masks/"+ CONDITION +"/"

masks = []

# Merge all the segmentation
# + change the ID
# two first number from 00 to 24
# Three last numbers: from 000 to 999

# Loop through the tiles
n_tiles =int(np.max(pattern_snake[pattern_snake!=None])+1)
for s in range(n_tiles):
    s_ = str(s).zfill(2)
    mask_name = '*_s{}_*.png'.format(s_)
    try:
        mask_path = list(pathlib.Path(folder_mask).glob(mask_name))[0]
        mask = np.asarray(Image.open(mask_path))
    except:
        mask=np.zeros((IM_SIZE,IM_SIZE))
    masks.append(mask)


def remove_cell_border(im):
    # Look where are the cells
    x,y = np.where(im>0)
    # Get the index of the cells
    values_pos = im[im>0]
    # Get the shape of the image
    X_shape, Y_shape = im.shape[0]-1, im.shape[1]-1
    # Check if there are cells at the border of the image
    bool_pos_lim = np.any(((x==0),(y==0),(x==X_shape),(y==Y_shape)),axis=0)
    # Get which index of cells are at the border
    ind_lim = values_pos[bool_pos_lim]
    # Remove the cells with those indices
    bool_im = np.any(np.asarray([im==i for i in ind_lim]),axis=0)
    im[bool_im]=0
    return(im)


# Manually
n_rows, n_cols = pattern_snake.shape
total_image = np.zeros((IM_SIZE*n_rows-(n_rows-1)*OVERLAP, \
                        IM_SIZE*n_cols-(n_cols-1)*OVERLAP))
X_shape, Y_shape = total_image.shape[0], total_image.shape[1]

# Do the snake
for i in range(n_rows):
    
    for j in range(n_cols):
        
        id_tile = pattern_snake[i,j]
        
        if id_tile is not None:
            im = np.copy(masks[id_tile])
            # im = label(im)
            im[im>0] = im[im>0] + 1000*id_tile
            im = remove_cell_border(im)
            
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
            
            print(xmin,xmax, ymin,ymax, i,j,id_tile)

            # Save the submask
            s = id_tile
            s_ = str(s).zfill(2)
            tifffile.imwrite(folder_mask + 'mosaic_mask_reassigned_s{}.tiff'.format(s_), im.astype(np.int16))
            
            # Add 
            total_image[xmin:xmax, ymin:ymax] += im           


tifffile.imwrite(folder_mask + 'mosaic_mask_reassigned.tiff', total_image.astype(np.int16))