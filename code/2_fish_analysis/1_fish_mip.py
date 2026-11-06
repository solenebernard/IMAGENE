'''
Step 2.1: maximum intensity projection (MIP) of each smFISH tile, for the segmentation of the cells (step M3).

Each output image has two channels: the MIP over z of the FISH channels (ch00 and ch01) of the rounds
fish.mip_rounds, and the MIP over z of the DAPI channel (ch02) of the first round.

Input:  ROOTPATH/EXP_PATH/Fish/CONDITION/*_t{round}_s{s}_z{z}_ch{ch}.tif
Output: ROOTPATH/EXP_PATH/Fish/mip/CONDITION_round_{rounds}_s{s}_mip.tif

Run: python3 -m 2_fish_analysis.1_fish_mip
'''


# Import local tools
import numpy as np
import os
import pathlib
import tifffile
from tools.read_yaml import CONDITION, EXP_PATH, FISH_MIP_ROUNDS, FISH_N_TILES, ROOTPATH


def main():
    '''MIP of every tile'''
    folder_tif = ROOTPATH + EXP_PATH + 'Fish/' + CONDITION + '/'
    path_save = ROOTPATH + EXP_PATH + 'Fish/mip/'

    # Take the MIP on only two rounds (4 genes + DAPI)
    t_rounds = FISH_MIP_ROUNDS

    for idx in range(FISH_N_TILES):

        s_ = str(idx).zfill(2)
        frame_names = []

        # get dapi
        name_tif = '*_t00_s{}_z*_ch02.tif'.format(s_)
        frame_names_dapi = list(pathlib.Path(folder_tif).glob(name_tif))
        # Get all files corresponding to the rounds in t_rounds
        for t in t_rounds:
            t_ = str(t).zfill(2)
            name_tif = '*_t{}_s{}_z*_ch00.tif'.format(t_, s_)
            frame_names += list(pathlib.Path(folder_tif).glob(name_tif))
            name_tif = '*_t{}_s{}_z*_ch01.tif'.format(t_, s_)
            frame_names += list(pathlib.Path(folder_tif).glob(name_tif))

        # Load the images
        images = []
        for i, frame in enumerate(frame_names):
            im = tifffile.imread(frame)
            images.append(im)
        images = np.asarray(images)

        images_dapi = []
        for i, frame in enumerate(frame_names_dapi):
            im = tifffile.imread(frame)
            images_dapi.append(im)
        images_dapi = np.asarray(images_dapi)

        print(idx, images.shape)

        image = np.stack((np.max(images, axis=0), np.max(images_dapi, axis=0)))

        # Save the MIP in TIFF format
        short_name = "_".join(map(str, t_rounds))
        save_file_name = CONDITION + '_round_' + short_name + '_s{}_mip.tif'.format(str(idx).zfill(2))
        output_tiff_path = os.path.join(path_save, save_file_name)
        tifffile.imwrite(output_tiff_path, image)


if __name__ == "__main__":
    main()
