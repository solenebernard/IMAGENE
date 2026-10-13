# Import local tools
from tools.read_yaml import *
from tools.imports import *

import cv2
from scipy.spatial.distance import cdist
from matplotlib.path import Path

name_exp = EXP_PATH[:-1]

# Constants
scale = SCALE
# max_dist = MAX_DIST_MATCHING 
max_dist = 1000 # Do not pick a custom threshold: make it high, and save the distance for each matching, so we can filter after! (do not filter right now)

# Do you we the postuv mask of brighfield for alignment?
postuv_mask = False
# Do you we the DAPI mask or whole cells fish masks for alignment?
DAPI_mask = False


# Initialize variables
data_path_live = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/'
data_path_raw = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/live_dataset/' 
video_dir = data_path_live + 'video/'
path_output = data_path_live + 'results/'
save_path_file = ROOTPATH + EXP_PATH + "alignment/"+CONDITION+"-alignment.csv"
bf_dir =  ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/clean_masks/' 
device = DEVICE

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

def compute_distance(p0, p1):
    '''
    For two sets of points p0 and p1
    Compute the sum of the distances between each point of p1
    to its closest neighbour in p0    
    '''
    # Troncate fish
    # p0 = np.copy(p0)[(p0[:, 0] >= p1[:, 0].min())\
    #                                 &(p0[:, 0] <= p1[:, 0].max())\
    #                                 &(p0[:, 1] >= p1[:, 1].min())\
    #                                 &(p0[:, 1] <= p1[:, 1].max())]
    mat_dist = cdist(p0, p1)
    arg_min_dist = np.argmin(mat_dist, axis=1)
    min_dist = np.min(mat_dist, axis=1)
    return(np.mean(min_dist), arg_min_dist)

def match_closest_neighbour(p0, p1, max_dist, plot=True):

    # Get closest neighbour
    mat_dist = cdist(p0, p1)

    # Initialize matrix of matching from p0 to p1
    mat_match0 = np.zeros_like(mat_dist)
    arg_min_dist0 = np.argsort(mat_dist,axis=1)[:,0]
    mat_match0[np.arange(len(p0)),arg_min_dist0] = 1
    # Set threshold of maximum distance of max_dist
    min_dist0 = np.min(mat_dist, axis=1)
    mat_match0[np.arange(len(p0))[min_dist0>max_dist],arg_min_dist0[min_dist0>max_dist]] = 0

    # Initialize matrix of matching from p1 to p0
    mat_match1 = np.zeros_like(mat_dist)
    arg_min_dist1 = np.argsort(mat_dist,axis=0)[0]
    mat_match1[arg_min_dist1,np.arange(len(p1))] = 1
    # Set threshold of maximum distance of max_dist
    min_dist1 = np.min(mat_dist, axis=0)
    mat_match1[arg_min_dist1[min_dist1>max_dist], np.arange(len(p1))[min_dist1>max_dist]] = 0

    # Total matching matrix is the multiplication of both: it's reciprocal
    # Too conservative?
    mat_match = mat_match0*mat_match1

    # Now: second closest neighbour
    # Initialize new matrix
    # It's not reciprocal
    mat_match_scd_0 = np.zeros_like(mat_match)
    mat_match_scd_1 = np.zeros_like(mat_match)

    # Each matched point of p0 and p1 is linked to its closest neighbour
    ind_matched0 = np.where(mat_match==1)[0]
    ind_matched1 = np.where(mat_match==1)[1]
    
    arg_2min_dist0 = np.argsort(mat_dist,axis=1)[:,1]
    mat_match_scd_0[ind_matched0,arg_2min_dist0[ind_matched0]] = 2

    arg_2min_dist1 = np.argsort(mat_dist,axis=0)[1]
    mat_match_scd_1[arg_2min_dist1[ind_matched1],ind_matched1] = 2

    if plot:
        # Vizualize matching
        plt.figure(figsize=(20,20))
        plt.scatter(p0[:,0], p0[:,1],s=3,c='green',marker='.')
        plt.scatter(p1[:,0], p1[:,1],s=3,c='red',marker='.')
        ind_p0, ind_p1 = np.where(mat_match==1)
        plt.scatter(p0[ind_p0][:,0], p0[ind_p0][:,1],s=8,c='green')
        plt.scatter(p1[ind_p1][:,0], p1[ind_p1][:,1],s=8,c='red')
        plt.plot([p0[ind_p0][:,0],  p1[ind_p1][:,0]], [p0[ind_p0][:,1],  p1[ind_p1][:,1]],c='blue',linewidth=0.5)

        ind_p0, ind_p1 = np.where(mat_match_scd_0==2)
        plt.plot([p0[ind_p0][:,0],  p1[ind_p1][:,0]], [p0[ind_p0][:,1],  p1[ind_p1][:,1]],c='green',linewidth=0.5,alpha=0.5, linestyle='--')
        ind_p0, ind_p1 = np.where(mat_match_scd_1==2)
        plt.plot([p0[ind_p0][:,0],  p1[ind_p1][:,0]], [p0[ind_p0][:,1],  p1[ind_p1][:,1]],c='red',linewidth=0.5,alpha=0.5, linestyle='--')
        plt.show()

    return(mat_dist, mat_match, mat_match_scd_0, mat_match_scd_1)


def brute_force(p0, p1, dx_trans_min, dy_trans_min, dx_trans_max, dy_trans_max, n):
    '''
    Brute force: test all translations between two extreme translation for each axis
    with a granularity n
    '''
    # Get all possible translation for each axis
    dx_trans = np.linspace(dx_trans_min, dx_trans_max, n)
    dy_trans = np.linspace(dy_trans_min, dy_trans_max, n)

    error_mat = np.zeros([n, n])
    for i, x_trans in enumerate(dx_trans):
        for j,y_trans in enumerate(dy_trans):
            p1_trans = p1 + np.array([x_trans, y_trans])

            error, _ = compute_distance(p0, p1_trans)
            error_mat[i, j] = error

    # Get the translation that minimizes the distance
    i,j = np.argmin(error_mat)//n, np.argmin(error_mat)%n
    x_trans, y_trans = dx_trans[i],dy_trans[j]

    return(x_trans, y_trans, error_mat)

def get_translation(p0, p1, n=40, delta=1500):

    p0_xmin, p0_xmax, p0_ymin, p0_ymax = p0[:,0].min(), p0[:,0].max(), \
                p0[:,1].min(), p0[:,1].max()

    p1_xmin, p1_xmax, p1_ymin, p1_ymax = p1[:,0].min(), p1[:,0].max(), \
        p1[:,1].min(), p1[:,1].max()
    
    print(p0_xmin, p0_xmax, p0_ymin, p0_ymax)
    print(p1_xmin, p1_xmax, p1_ymin, p1_ymax)
    dx_trans_max = (p0_xmax-p0_xmin)-(p1_xmax-p1_xmin) + delta#*2/3
    dy_trans_max = (p0_ymax-p0_ymin)-(p1_ymax-p1_ymin) + delta
    dx_trans_min = - delta#*2/3
    dy_trans_min = - delta
    print(dx_trans_max, dy_trans_max)

    x_trans, y_trans, error_mat = brute_force(p0, p1, dx_trans_min, dy_trans_min, dx_trans_max, dy_trans_max, n)
    plt.imshow((error_mat.T),origin='lower')
    plt.colorbar()
    plt.show()
    print(x_trans, y_trans)
    print(np.argmin(error_mat)//n, np.argmin(error_mat)%n)
    # Add one leval of dichotomy
    dx_trans_max = x_trans + 100
    dy_trans_max = y_trans + 100
    dx_trans_min = x_trans - 100
    dy_trans_min = y_trans - 100

    x_trans, y_trans, error_mat2 = brute_force(p0, p1, dx_trans_min, dy_trans_min, dx_trans_max, dy_trans_max, n)
    plt.imshow((error_mat2.T),origin='lower')
    plt.colorbar()
    plt.show()

    # print(x_trans, y_trans)
    # print(np.argmin(error_mat)//n, np.argmin(error_mat)%n)
    # # Add one leval of dichotomy
    # dx_trans_max = x_trans + 10
    # dy_trans_max = y_trans + 10
    # dx_trans_min = x_trans - 10
    # dy_trans_min = y_trans - 10

    # x_trans, y_trans, error_mat2 = brute_force(p0, p1, dx_trans_min, dy_trans_min, dx_trans_max, dy_trans_max, 10)
    # plt.imshow((error_mat2.T),origin='lower')
    # plt.colorbar()
    # plt.show()

    return(x_trans, y_trans, error_mat)


def filter_segmentation(mask):
    '''
    The mask could have some segmentation issue (SAM2)
    A way to detect error is to check if the centroid falls inside a cell
    If not: remove the segmentation!
    Also remove small cells!
    '''
    labeled_im = mask.copy()
    # Detect problematic cells
    unique_labels = np.unique(labeled_im)
    unique_labels = unique_labels[unique_labels > 0]
    # Compute the centroids of cells
    unique_values, centroids, areas = extract_centroid_coords(mask)
    centroids = np.asarray(np.floor(centroids),dtype=np.int32) # Cast as an index
    # If the value of the mask is 0 at the centroid position: the centroid is outside
    values = labeled_im[centroids[:,1],centroids[:,0]]
    id_val = np.where(values==0)[0] # If it falls outside a cell
    problematic_id = unique_labels[id_val]
    # print(problematic_id)

    # Iterate over the problematic id
    # And remove the cells
    for x in problematic_id:
        bool_x = labeled_im==x
        labeled_im[bool_x] = 0
    
    # Filter size with size smaller than 500
    problematic_id = unique_values[np.where(areas<500)[0]]
    print(problematic_id)
    for x in problematic_id:
        bool_x = labeled_im==x
        labeled_im[bool_x] = 0
    
    return(labeled_im)
            

def compute_area(mask_pos, list_ind):
    list_sum = []
    for x in list_ind:
        sum = np.sum(mask_pos==x)
        list_sum.append(sum)
    return(np.asarray(list_sum))


list_files_bf = list(pathlib.Path(bf_dir).glob('seg_movie_merged_t*_cp_mask.png'))
list_files_bf.sort()
img = Image.open(list_files_bf[-1])

if postuv_mask:
    img_match = list(pathlib.Path(bf_dir).glob('*POSTUV*'))[0]
    img_match = Image.open(img_match)
else:
    img_match = img
    

bf_segmentation_match = np.asarray(img_match)
# bf_segmentation_match = filter_segmentation(bf_segmentation_match)

if postuv_mask:
    bf_segmentation = np.asarray(img)
    # bf_segmentation = filter_segmentation(bf_segmentation)
else:
    bf_segmentation = bf_segmentation_match


# Fish segmentation
FISH_segmentation = np.asarray(tifffile.imread(ROOTPATH + EXP_PATH + 'Fish/masks/' + CONDITION + "/mosaic_mask_reassigned.tiff")) # segmentation of FISH
# Realign according to the scaling factor
FISH_segmentation = cv2.resize(FISH_segmentation, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
FISH_segmentation = filter_segmentation(FISH_segmentation)

if DAPI_mask:
    fish_match = np.asarray(tifffile.imread(ROOTPATH + EXP_PATH + 'Fish/masks/' + CONDITION + "/DAPI_masks/mosaic_mask_reassigned.tiff")) 
    fish_match = cv2.resize(fish_match, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    fish_match = filter_segmentation(fish_match)
else:
    fish_match = FISH_segmentation
# img = img.transpose(Image.ROTATE_270)


# Extract centroids
mask_values_bf, centroids_bf, areas_bf = extract_centroid_coords(bf_segmentation)
mask_values_bf_match, centroids_bf_match, areas_bf_match = extract_centroid_coords(bf_segmentation_match)
mask_values_fish, centroids_fish, areas_fish = extract_centroid_coords(FISH_segmentation)
mask_values_fish_match, centroids_fish_match, areas_fish_match = extract_centroid_coords(fish_match)

# Translate 
x_trans, y_trans, error_mat = get_translation(centroids_fish_match, centroids_bf_match, n=60, delta=2000)
centroids_bf = centroids_bf + np.array([x_trans, y_trans])

# Match with closest neighbour
mat_dist, mat_match, mat_match_scd_fish, mat_match_scd_bf = match_closest_neighbour(centroids_fish, centroids_bf, max_dist)

# Closest neighbour
ind_p0, ind_p1 = np.where(mat_match==1)
matched_fish_id, matched_bf_id = mask_values_fish[ind_p0], mask_values_bf[ind_p1]
matched_dist = mat_dist[ind_p0, ind_p1]
matched_fish_centroid, matched_bf_centroid = centroids_fish[ind_p0], centroids_bf[ind_p1]

# Second closest neigbour from p0 to p1
ind2_p1 = np.asarray([np.where(mat_match_scd_fish[x]==2)[0][0] for x in ind_p0]) # keep the order 
matched_bf_id_scd = mask_values_bf[ind2_p1]
matched_dist_2_fish = mat_dist[ind_p0, ind2_p1]

# Second closest neigbour from p1 to p0 # PROBLEM
ind2_p0 = np.asarray([np.where(mat_match_scd_bf[:,x]==2)[0][0] for x in ind_p1]) # keep the order 
matched_fish_id_scd = mask_values_fish[ind2_p0]
matched_dist_2_bf = mat_dist[ind2_p0, ind_p1]


area_fish = compute_area(FISH_segmentation[FISH_segmentation>0], matched_fish_id)
area_bf = compute_area(bf_segmentation[bf_segmentation>0], matched_bf_id)


# Save dictionnary in csv
save_frame = pd.DataFrame({"BF":matched_bf_id, "Fish":matched_fish_id, \
                        "BF_center_x":matched_bf_centroid[:,0], "BF_center_y":matched_bf_centroid[:,1], \
                        "Fish_center_x":matched_fish_centroid[:,0], "Fish_center_y":matched_fish_centroid[:,1], \
                        "Distance":matched_dist, \
                        'Distance_2nd_bf_to_fish': matched_dist_2_bf, 'Distance_2nd_fish_to_bf': matched_dist_2_fish, \
                        'Id_2nd_bf_to_fish': matched_fish_id_scd, 'Id_2nd_fish_to_bf': matched_bf_id_scd})
save_frame.to_csv(save_path_file, index=False)
