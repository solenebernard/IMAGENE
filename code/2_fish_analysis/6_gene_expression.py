# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_fish import *

import cv2

# Analyze the UFISH result
# Combined with the fish segmentation
# Create gene expression

CONDITION = '4-24H'
# Initialize variables
folder_tif = ROOTPATH + EXP_PATH + 'Fish/' + CONDITION + '/'
folder_mask = ROOTPATH + EXP_PATH + 'Fish/masks/' + CONDITION + '/'
folder_save_loc = ROOTPATH + EXP_PATH + 'Fish/bigfish/' + CONDITION + '/'
device = DEVICE
output_file = ROOTPATH + EXP_PATH + 'Fish/' + CONDITION + '.csv'
CLUMPS_DETECTOR = True

# Load the custom snake pattern made by the microscope
pattern_snake_path = ROOTPATH + EXP_PATH + "Fish/pattern_snake/" + CONDITION + ".npy"
if pathlib.Path(pattern_snake_path).exists():
    pattern_snake = np.load(pattern_snake_path, allow_pickle = True)
else: # the default one: 5x5
    pattern_snake = np.load(ROOTPATH + 'olgarithm/config_patterns/5x5.npy')
n_rows, n_cols = pattern_snake.shape


# For all two genes, in every round
# For each tile s

dict_gene = {}

# Version with the merged mask
OVERLAP = 206
IM_SIZE = 2048

# Load the big mask
big_mask = tifffile.imread(folder_mask + 'mosaic_mask_reassigned.tiff')

######## SPECIFIC CASE IF THERE IS AN ISSUE BETWEEN ROUNDS ########
# Need to translate + rotate the big mask to match new position
## FOR EXAMPLE HIV1
# FOR 3-IL15 and 4-IL7IL15: shift between round0 and round1 

# To modify: found by trial and error

# # 3-IL15 (HIV1)
# dx, dy= 450, 990 # parameters of translation
# angle = 0 # parameter of rotation

# # 2-24H (christian)
# dx, dy= 40, 40 # parameters of translation
# angle = 0 # parameter of rotation

# # Translate the mask and crop
# new_big_mask = np.zeros((dx+big_mask.shape[0],dy+big_mask.shape[1]), dtype=big_mask.dtype)
# new_big_mask[dx:,dy:] += big_mask
# big_mask = np.copy(new_big_mask)

# # Rotation
# (h, w) = new_big_mask.shape[:2]
# center = (w // 2, h // 2)
# # Define rotation matrix (angle in degrees, scale=1.0)
# M = cv2.getRotationMatrix2D(center, angle=angle, scale=1.0)
# # Apply rotation
# big_mask = cv2.warpAffine(new_big_mask, M, (w, h))
#####################################################################


unique_id = np.unique(big_mask[big_mask>0])
unique_values = unique_id

# Loop through the available rounds
available_rounds = list(pathlib.Path(folder_save_loc).glob('t*_s{}_*.npy'.format('00')))
available_rounds = [int((str(r).split('/')[-1]).split('_')[0][1:]) for r in available_rounds]
available_rounds = np.unique(available_rounds)

columns = ['Gene_' + gene for gene in GENES] + \
            ['delta_z_l_' + gene for gene in GENES] + \
            ['delta_z_r_' + gene for gene in GENES] 
df = pd.DataFrame(None, index=unique_id, columns=columns)

# # Init de dictionnary
# for x in unique_id:
#     # Initialize None value for each gene and each cell
#     # Initialize delta z left and right also, for each gene
#     dict_gene[x] = [[None for _ in range(len(GENES))], [None for _ in range(len(GENES))], [None for _ in range(len(GENES))]]

for t in available_rounds:
    
    t_ =  str(t).zfill(2)
    # Loop through the available channel:
    available_channels = list(pathlib.Path(folder_save_loc).glob('t{}_s{}_*.npy'.format(t_, '00')))
    available_channels = [int((str(r).split('/')[-1]).split('_')[-1][2:-4]) for r in available_channels]
    available_channels = np.unique(available_channels)
    # Keep only 0 and 1 values (no DAPI)
    available_channels = available_channels[available_channels<2]

    for ch in available_channels:
        
        ch_ = str(ch).zfill(2)

        all_spots = np.empty((0,3))

        # Loop through the tiles
        # Concatenate all spots and translate it to match the big mask
        
        # Available tiles
        available_tiles = list(pathlib.Path(folder_save_loc).glob('t{}_s*_ch{}.npy'.format(t_, ch_)))
        available_tiles = [int((str(r).split('/')[-1]).split('_')[1][1:3]) for r in available_tiles]
        available_tiles = np.unique(available_tiles)
        
        list_mips = []
        
        for s in available_tiles:
            s_ = str(s).zfill(2)
            # #  Old snake code
            # q, r = (s//5), s%5
            # # Snake!
            # if (s//5)%2==0:
            #     delta_spots_y = r*(IM_SIZE-OVERLAP)
            # else:
            #     delta_spots_y = (4-r)*(IM_SIZE-OVERLAP)
            # delta_spots_x = q*(IM_SIZE-OVERLAP)
            
            q, r = np.where(pattern_snake==s)
            q, r = int(q[0]), int(r[0])
            delta_spots_y = r*(IM_SIZE-OVERLAP)
            delta_spots_x = q*(IM_SIZE-OVERLAP)

            name_file_spots = 't{}_s{}_ch{}.npy'.format(t_, s_, ch_)
            spots = np.load(folder_save_loc+name_file_spots)

            # Remove overlap! To no count spots twice (or thrice!)
            if r<(n_cols-1):
                if pattern_snake[q,r+1] is not None: # Remove only when there is a next tile
                    spots = spots[spots[:,2]<IM_SIZE-OVERLAP]
            if q<(n_rows-1):
                if pattern_snake[q+1,r] is not None: # Remove only when there is a next tile
                    spots = spots[spots[:,1]<IM_SIZE-OVERLAP]

            # Translate spots 
            spots[:,1] += delta_spots_x
            spots[:,2] += delta_spots_y
            all_spots = np.concatenate((all_spots, spots))
            
        if CLUMPS_DETECTOR:
            path_clumps = f'{ROOTPATH}{EXP_PATH}Fish/clumps_detection/'+ f'{CONDITION}_mask_total_t{t_}_ch{ch_}.png'
            mask_clumps =np.asarray(Image.open(path_clumps))
            filtered_cells = np.unique(big_mask[np.where(mask_clumps>0)])
            print(t,ch, filtered_cells)
        else:
            filtered_cells = [] # empty list to remove no cell
            
        spots_int = np.round(all_spots[:,1:]).astype(np.int64) # Cast as integer
        cells_spots = big_mask[spots_int[:,0],spots_int[:,1]] # get value on the mask
        counts = [np.sum(cells_spots==x) for x in unique_values] # generate gene expression         

        # Measure on what z slice were the spots
        name_tif = '*_t{}_s{}_z*_ch{}.tif'.format(t_, s_, ch_)
        frame_names = list(pathlib.Path(folder_tif).glob(name_tif)) ## Get number of available z slices
        z_range = np.arange(len(frame_names)) # z range
        z_hist_cells = [np.asarray([np.sum(all_spots[cells_spots==x,0]==z) for z in z_range]) for x in unique_values] # get on what z slice
        z_detect = [z_range[np.where(x>0)[0]] for x in z_hist_cells]

        deltaz_l_r = []
        for x in z_detect:
            if len(x)>0:
                z_min_detect, z_max_detect = x.min(), x.max()
                delta_z_l, delta_z_r = z_min_detect-z_range.min(), z_range.max()-z_max_detect
            else:
                delta_z_l, delta_z_r = None, None
            deltaz_l_r.append([delta_z_l, delta_z_r])
        
        # Get back to the gene value, depending on t and ch
        index_gene = t*2 + int(ch)
        gene = GENES[index_gene]
        
        for v,c,d in zip(unique_id, counts, deltaz_l_r):
            # replace the none value by the count
            if v not in filtered_cells:
                df.loc[v, f'Gene_{gene}'] = c
                df.loc[v, f'delta_z_l_{gene}'] = d[0]
                df.loc[v, f'delta_z_r_{gene}'] = d[1]

df.index.name = 'Key'
df.to_csv(output_file)