# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

from skimage.measure import regionprops_table

def feats(inner_body, outer_body, list_masks):
    '''
    - Load the nucleus and whole cell masks
    - Relabel the nucleus mask to match the whole cell id
    - Compute: whole cell area, nucleus area, and distance between centroids
    '''
    n_t = len(list_masks)
    # Relabel the inner and outer body masks with original whole cell id
    new_inner_body, new_outer_body = np.zeros_like(list_masks), np.zeros_like(list_masks)
    # Change ID of the masks of the inner body to match the original ID
    for t in range(len(list_masks)):
        bool_pos = inner_body[t]>0
        id_original = list_masks[t][bool_pos]
        new_inner_body[t][bool_pos] = id_original
    # Change ID of the masks of outer body to match the original ID
    for t in range(len(list_masks)):
        bool_pos = outer_body[t]>0
        id_original = list_masks[t][bool_pos]
        new_outer_body[t][bool_pos] = id_original
    
    # Compute the features
    # Ratio 
    props_all = []
    for t in range(n_t):
        props_cells = regionprops_table(
            list_masks[t],
            properties=('label', 'area', 'centroid')
        )
        props_inner_body = regionprops_table(
            new_inner_body[t],
            properties=('label', 'area', 'centroid', 'bbox_area', 'perimeter', \
                'eccentricity', 'equivalent_diameter_area', 'feret_diameter_max', 'orientation')
        )
        props_outer_body = regionprops_table(
            new_outer_body[t],
            properties=('label', 'area', 'centroid', 'bbox_area', 'perimeter', \
                'eccentricity', 'equivalent_diameter_area', 'feret_diameter_max', 'orientation')
        )
        df_cells = pd.DataFrame(props_cells)
        df_cells['t'] = t
        df_inner = pd.DataFrame(props_inner_body)
        df_inner['t'] = t
        df_outer = pd.DataFrame(props_outer_body)
        df_outer['t'] = t

        props_all.append((df_cells, df_inner, df_outer))
    # Concatenate all frames
    df_cells_all = pd.concat([c for c, _, _ in props_all])
    df_cells_all = df_cells_all.rename(
        columns={col: col + '_cell' for col in df_cells_all.columns if col not in ['label', 't']}
    ) 
    df_inner_all = pd.concat([i for _, i, _ in props_all])
    df_inner_all = df_inner_all.rename(
        columns={col: col + '_inner' for col in df_inner_all.columns if col not in ['label', 't']}
    ) 
    df_outer_all = pd.concat([o for _, _, o in props_all])
    df_outer_all = df_outer_all.rename(
        columns={col: col + '_outer' for col in df_outer_all.columns if col not in ['label', 't']}
    ) 
    df_merged = (
        df_cells_all
        .merge(df_inner_all, on=['label', 't'], suffixes=('_cell', '_inner'), how='inner')
        .merge(df_outer_all, on=['label', 't'], suffixes=('', '_outer'), how='inner')
    )
    
    df_merged = df_merged.rename(columns={'label':'cell_bf'})
    df_merged = df_merged.set_index(['cell_bf', 't'])
    df_merged['distance_centroids_inner_cell'] = np.sqrt(
        (df_merged['centroid-0_cell'] - df_merged['centroid-0_inner'])**2 +
        (df_merged['centroid-1_cell'] - df_merged['centroid-1_inner'])**2
    )
    df_merged['distance_centroids_outer_cell'] = np.sqrt(
        (df_merged['centroid-0_cell'] - df_merged['centroid-0_outer'])**2 +
        (df_merged['centroid-1_cell'] - df_merged['centroid-1_outer'])**2
    )
    df_merged['distance_centroids_outer_inner'] = np.sqrt(
        (df_merged['centroid-0_inner'] - df_merged['centroid-0_outer'])**2 +
        (df_merged['centroid-1_inner'] - df_merged['centroid-1_outer'])**2
    )
    df_merged['ratio_area_inner_outer'] = df_merged['area_inner']/df_merged['area_outer']
    df_merged['ratio_area_inner_cell'] = df_merged['area_inner']/df_merged['area_cell']
    df_merged['ratio_area_cell_outer'] = df_merged['area_cell']/df_merged['area_outer']
    df_merged.to_csv(ROOTPATH + EXP_PATH + 'live_cell_features/lamellipodia_feats_' + CONDITION + '.csv', index=True)


    
def main():
    clean=True
    folder_lamellipodia = ROOTPATH + EXP_PATH + 'Live/'+ CONDITION+ '/lamellipodia_segmentations/'
    mask = load_masks(ROOTPATH, EXP_PATH, CONDITION, clean=clean)[-1] # Last mask

    # # Cropping parameters for olga2 24H
    # inner_body_cropped = np.asarray(Image.open(folder_lamellipodia + 'inner_body_cropped.png'))
    # outer_body_cropped = np.asarray(Image.open(folder_lamellipodia + 'outer_body_cropped.png'))
    
    # height, width = mask.shape
    # crop_left, crop_right = 600, 600
    # crop_top, crop_bottom = 800, 450
    # inner_body, outer_body = np.zeros_like(mask), np.zeros_like(mask)
    # inner_body[crop_top:height-crop_bottom,crop_left:width-crop_right] = inner_body_cropped
    # outer_body[crop_top:height-crop_bottom,crop_left:width-crop_right] = outer_body_cropped
    # Image.fromarray(inner_body).save(folder_lamellipodia + 'inner_body.png')
    # Image.fromarray(outer_body).save(folder_lamellipodia + 'outer_body.png')
    
    # Load images
    inner_body = np.asarray(Image.open(folder_lamellipodia + 'inner_body.png'))
    outer_body = np.asarray(Image.open(folder_lamellipodia + 'outer_body.png'))
    
    feats(inner_body[None], outer_body[None], mask[None])
    

if __name__ == "__main__":
    main()

