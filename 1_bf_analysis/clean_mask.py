'''
Clean the mask after SAM2 algorithm
Do check for each cell id for each masks:
    - if the region is a related area
        - if not: remove the small parts. If the parts are not small: remove id
    - if the centroid falls inside the cell
        - if not: remove the id
    - if the area of the cell is larger than 500 pixels
'''

# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

from skimage.measure import regionprops, label
import argparse

# Create of fast version of extract_centroid_coords
def extract_centroid_coords(mask, unique_values=None):
    '''
    Extracts the centroid coordinates of the labeled regions in a mask
    :param mask: 2D numpy array with labeled regions
    :param unique_values (optional): list/1D array. if None, loop over all strictly positive values of mask. If not, loop on unique_values
    :return: x_coords, y_coords: lists of x and y coordinates of the centroids
    '''
    # Initialize lists to store the x and y coordinates of the centroids
    x_coords = []
    y_coords = []
    areas = []
    # Filter out the background (label 0)
    x_pos, y_pos = np.where(mask > 0)
    # Take the labels
    mask_values = mask[x_pos, y_pos]
    # Loop through the unique labels
    if unique_values is None:
        unique_values = np.unique(mask_values)
    for i in unique_values:
        ind = np.where(mask_values == i)[0]
        y,x = x_pos[ind], y_pos[ind]
        # Compute centroid (= the average of the x and y coordinates)
        x_coords.append(x.mean())
        y_coords.append(y.mean())
        # Compute area
        area=len(x)
        areas.append(area)
    x_coords, y_coords = np.asarray(x_coords), np.asarray(y_coords)
    coords = np.stack((x_coords, y_coords))
    return unique_values, coords.T, np.asarray(areas)


def filter_segmentation(mask, min_size = 500):
    '''
    The mask could have some segmentation issue (SAM2)
    A way to detect error is to check if the centroid falls inside a cell
    And remove small cells
    '''
    # Detect problematic cells
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]
    # Compute the centroids of cells
    unique_values, centroids, areas = extract_centroid_coords(mask)
    centroids = np.asarray(np.floor(centroids),dtype=np.int32) # Cast as an index
    # If the value of the mask is 0 at the centroid position: the centroid is outside
    values = mask[centroids[:,1],centroids[:,0]]
    id_val = np.where(values==0)[0] # If it falls outside a cell
    problematic_id = unique_labels[id_val]
    
    problematic_id_size = unique_values[np.where(areas<min_size)[0]]
    problematic_id = np.concatenate((problematic_id, problematic_id_size))
    
    return(problematic_id)

def detect_non_related_areas(mask):
    unique_id = np.unique(mask[mask>0])
    cleaned_mask = np.copy(mask)
    
    # Keep track of problematic id
    list_remove_id = []
    
    for cell_id in unique_id:
        mask_idx = mask==cell_id
        labeled_mask = label(mask_idx)
        unique_labeled = np.unique(labeled_mask[labeled_mask>0])
        unique_labeled.sort()
        if len(unique_labeled)>1: # Meaning the area is not related
            # Keep the largest ?
            count_unique = np.asarray([len(np.where(labeled_mask==x)[0]) for x in unique_labeled])
            ratio = count_unique/count_unique.max()
            # Check if the largest is the unique big cell: the second largest must be at least 2 times smaller
            list_second_ratio = ratio[ratio<1]
            if len(list_second_ratio)>0:
                second_ratio = list_second_ratio.max()
                if second_ratio>0.2:
                    # If the second largest area is too big
                    # Remove the cell id
                    list_remove_id.append(cell_id)
                else:
                    # Only remove small areas
                    for problematic_id,ratio_cell in zip(unique_labeled, ratio):
                        if ratio_cell<1:
                            # Set to 0 the original mask
                            cleaned_mask[labeled_mask==problematic_id]=0
                            
    return(cleaned_mask, list_remove_id)

def remove_id(mask, problematic_id):
    # Iterate over the problematic id
    # And remove the cells
    cleaned_mask = np.copy(mask)
    for x in problematic_id:
        bool_x = cleaned_mask==x
        cleaned_mask[bool_x] = 0
    return(cleaned_mask)


def main(t, path_save):
    mask = load_masks(ROOTPATH, EXP_PATH, CONDITION)[t]
    cleaned_mask, problematic_id = detect_non_related_areas(mask)
    cleaned_mask = remove_id(cleaned_mask, problematic_id)
    # Remove cells where centroids is not inside the cells
    # And small cells
    problematic_id =  filter_segmentation(cleaned_mask, min_size=500)
    cleaned_mask = remove_id(cleaned_mask, problematic_id)
    t_ = str(t).zfill(2)
    im_seg = Image.fromarray(cleaned_mask.astype(np.uint16))
    im_seg.save(path_save + f'seg_movie_merged_t{t_}_cp_mask.png')


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = 2000000000
    path_save = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/clean_masks/' 
    ## FOR MULTIPROCESSING
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop_t", type=int)
    args = parser.parse_args()
    main(args.loop_t, path_save)