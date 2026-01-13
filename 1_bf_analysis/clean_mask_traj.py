'''
Clean the mask after SAM2 algorithm
Do global trajectory check: 
    - if the step between two time points is too large: remove id
'''

# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

def remove_id(mask, problematic_id):
    # Iterate over the problematic id
    # And remove the cells
    cleaned_mask = np.copy(mask)
    for x in problematic_id:
        bool_x = cleaned_mask==x
        cleaned_mask[bool_x] = 0
    return(cleaned_mask)

def clean_trajectory(list_masks, max_dist=200):
    # Initialize lists to store the x and y coordinates of the centroids
    problematic_id = []
    
    # Filter out the background (label 0)
    t_pos, x_pos, y_pos = np.where(list_masks > 0)
    # Take the labels
    mask_values = list_masks[t_pos, x_pos, y_pos]
    unique_values = np.unique(mask_values)
    
    # hist_dist = np.empty(0)
    
    for id_cell in unique_values:
        ind = np.where(mask_values == id_cell)[0]
        t,y,x = t_pos[ind], x_pos[ind], y_pos[ind]
        
        # Only keep maximum number of frames
        if len(np.unique(t))==list_masks.shape[0]:
            # Compute centroid (= the average of the x and y coordinates)
            coords = np.asarray([[x[t==t_].mean(), y[t==t_].mean()] for t_ in np.unique(t)])
            dist = np.linalg.norm(np.diff(coords,axis=0),axis=1)
            if np.max(dist)>max_dist:
                problematic_id.append(id_cell)
            # hist_dist = np.concatenate((hist_dist, dist))
        else:
            problematic_id.append(id_cell)
    
    return(problematic_id)


def main(path_save, max_dist=100):
    list_cleaned_masks = load_masks(ROOTPATH, EXP_PATH, CONDITION, clean=True)
    
    # Clean mask as a whole: detect problematic trajectories
    problematic_id = clean_trajectory(list_cleaned_masks, max_dist=max_dist)
    cleaned_masks = remove_id(list_cleaned_masks, problematic_id)
    
    # Save
    for t,mask in enumerate(cleaned_masks):
        t_ = str(t).zfill(2)
        im_seg = Image.fromarray(mask.astype(np.uint16))
        im_seg.save(path_save + f'seg_movie_merged_t{t_}_cp_mask.png')


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = 2000000000
    path_save = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/clean_masks/' 
    main(path_save, max_dist=100)




