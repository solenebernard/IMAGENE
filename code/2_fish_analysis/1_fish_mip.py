# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_fish import *

CONDITION = '1-NA'
folder_tif = ROOTPATH + EXP_PATH + 'Fish/' + CONDITION + '/'
path_save = ROOTPATH + EXP_PATH + 'Fish/mip/'

# Take the MIP on only two rounds (4 genes + DAPI)
t_rounds = [0, 1, 2, 3, 4, 5, 6, 7]
t_rounds = [0]

for idx in range(25):

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

    image = np.stack((np.max(images,axis=0), np.max(images_dapi,axis=0)))

    # Save the MIP in TIFF format
    short_name = "_".join(map(str, t_rounds))
    save_file_name = CONDITION + '_round_'+short_name+'_s{}_mip.tif'.format(str(idx).zfill(2))
    output_tiff_path = os.path.join(path_save, save_file_name)
    tifffile.imwrite(output_tiff_path, image)
