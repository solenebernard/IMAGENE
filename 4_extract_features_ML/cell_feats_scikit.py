# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

from skimage.measure import regionprops_table

def feats(list_masks, rootpath, exp_path, condition):
    '''
    - Load the nucleus and whole cell masks
    - Relabel the nucleus mask to match the whole cell id
    - Compute: whole cell area, nucleus area, and distance between centroids
    '''
    n_t = len(list_masks)
    
    # Compute the features
    # Ratio 
    props_all = []
    for t in range(n_t):
        props_cells = regionprops_table(
            list_masks[t],
            properties=('label', 'area', 'centroid', 'bbox_area', 'perimeter', \
                'eccentricity','equivalent_diameter_area', 'extent', \
                'feret_diameter_max', 'major_axis_length', 'minor_axis_length', \
                'orientation', 'solidity', 'moments_hu', 'moments_central', \
                'moments_normalized', 'inertia_tensor', 'inertia_tensor_eigvals')
        )
        df_cells = pd.DataFrame(props_cells)
        df_cells['t'] = t
        
        props_all.append(df_cells)
    # Concatenate all frames
    df_cells_all = pd.concat([c for c in props_all])
    df_cells_all = df_cells_all.rename(columns={'label':'cell_bf'})
    df_cells_all = df_cells_all.set_index(['cell_bf', 't'])
    # Shift x and y **within each cell_bf**
    df_cells_all['centroid-0_next'] = df_cells_all.groupby(level='cell_bf')['centroid-0'].shift(-1)
    df_cells_all['centroid-1_next'] = df_cells_all.groupby(level='cell_bf')['centroid-1'].shift(-1)
    # Compute Euclidean distance
    df_cells_all['distance'] = np.sqrt((df_cells_all['centroid-0_next'] - df_cells_all['centroid-0'])**2 + \
        (df_cells_all['centroid-1_next'] - df_cells_all['centroid-1'])**2)
    # Optional: drop helper columns
    df_cells_all = df_cells_all.drop(columns=['centroid-0_next', 'centroid-1_next'])
    # df_merged['distance_centroids'] = np.sqrt(
    #     (df_merged['centroid-0_cell'] - df_merged['centroid-0_nucleus'])**2 +
    #     (df_merged['centroid-1_cell'] - df_merged['centroid-1_nucleus'])**2
    # )
    df_cells_all.to_csv(rootpath + exp_path + 'live_cell_features/cell_feats_scikit_' + condition + '.csv', index=True)
    
def main(rootpath, exp_path, condition):
    clean=True
    # masks_nucleus = load_masks_nuclear(ROOTPATH, EXP_PATH, CONDITION)
    list_masks = load_masks(rootpath, exp_path, condition, clean=clean)
    feats(list_masks, rootpath, exp_path, condition)


if __name__ == "__main__":
    main(ROOTPATH, EXP_PATH, CONDITION)
        
    