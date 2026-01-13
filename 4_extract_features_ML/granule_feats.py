# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

from skimage.measure import regionprops_table

def feats(xy_pos, mask):
    '''
    Counts number of position in each mask
    '''
    xy_pos_int = np.round(xy_pos).astype(np.int64)
    mask_value = np.asarray([mask[y,x] for x,y in zip(xy_pos_int['X'],xy_pos_int['Y'])])
    unique_cell_id = np.unique(mask[mask>0])
    granule_count = [np.sum(mask_value==cell_id) for cell_id in unique_cell_id]
    
    df = pd.DataFrame({'cell_bf': unique_cell_id, \
            'granule_count':granule_count})
    df = df.set_index('cell_bf')
    path_granules = ROOTPATH + EXP_PATH + 'live_cell_features/granule_feats_'+ CONDITION+ '.csv'
    df.to_csv(path_granules)
    

    
def main():
    clean=True
    folder_granule = ROOTPATH + EXP_PATH + 'Live/'+ CONDITION+ '/granule_position/'
    mask = load_masks(ROOTPATH, EXP_PATH, CONDITION, clean=clean)[-1] # Last mask 

    # Load positions
    xy_pos = pd.read_csv(folder_granule + 'xy_pos.csv')[['X', 'Y']]
    
    feats(xy_pos, mask)
    

if __name__ == "__main__":
    main()

