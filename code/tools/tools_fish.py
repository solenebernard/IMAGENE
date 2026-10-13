# Import local tools
import tifffile
import numpy as np
import pathlib
import pandas as pd

def assemble_z_stack(frame_names):
    z_stack = []
    for image_path in frame_names:
        image = tifffile.imread(image_path)
        z_stack.append(np.array(image))
    z_stack_array = np.stack(z_stack, axis=0)
    return z_stack_array


def show_mip(rootpath, exp_path, condition, s, t, ch):
    folder_tif = rootpath + exp_path + 'Fish/' + condition + '/'
    s_, t_, ch_ = str(s).zfill(2), str(t).zfill(2), str(ch).zfill(2)
    # Get the mip
    name_tif = '*_t{}_s{}_z*_ch{}.tif'.format(t_, s_, ch_)
    frame_names = list(pathlib.Path(folder_tif).glob(name_tif))
    images = assemble_z_stack(frame_names)
    mip = np.max(images,axis=0)
    return(mip)

def show_tile(rootpath, exp_path, condition, s, t, ch):
    folder_tif = rootpath + exp_path + 'Fish/' + condition + '/'
    s_, t_, ch_ = str(s).zfill(2), str(t).zfill(2), str(ch).zfill(2)
    # Get the mip
    name_tif = '*_t{}_s{}_z*_ch{}.tif'.format(t_, s_, ch_)
    frame_names = list(pathlib.Path(folder_tif).glob(name_tif))
    images = assemble_z_stack(frame_names)
    return(images)



def show_mip_and_spots(rootpath, exp_path, condition, s, t, ch):
    folder_tif = rootpath + exp_path + 'Fish/' + condition + '/'
    s_, t_, ch_ = str(s).zfill(2), str(t).zfill(2), str(ch).zfill(2)
    # Get the mip
    name_tif = '*_t{}_s{}_z*_ch{}.tif'.format(t_, s_, ch_)
    frame_names = list(pathlib.Path(folder_tif).glob(name_tif))
    images = assemble_z_stack(frame_names)
    mip = np.max(images,axis=0)
    
    # Spots
    folder_spots = rootpath + exp_path + 'Fish/bigfish/' + condition + '/'
    spots = np.load(folder_spots + 't{}_s{}_ch{}.npy'.format(t_, s_, ch_))

    # Mask 
    # gene_expr = pd.read_csv(rootpath + exp_path + 'Fish/' + condition + '.csv', index_col='Key')
    mask = rootpath + exp_path + 'Fish/masks/' + condition + f'/mosaic_mask_reassigned_s{s_}.tiff'
    mask = tifffile.imread(mask)
    # mask = ROOTPATH + exp_path + 'Fish/masks/' + condition + f'/mosaic_mask_reassigned.tiff'
    # mask = tifffile.imread(mask)
    
    # mask_gene = mask.copy()
    # for id_fish in np.unique(mask[(mask>s*1000)&(mask<(s+1)*1000)]):
    #     mask_gene[mask==id_fish] = gene_expr.loc[id_fish, 'Gene_TNF']
    
    return(mip, spots, mask)
