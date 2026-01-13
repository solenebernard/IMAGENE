'''
- Create folders for initialization of the project
- Save the images in jpeg format in ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/video/'
    for future SAM2 propagation
'''

# Import local tools and libraries
from tools.read_yaml import *
from tools.imports import *

# Get the name of the z slice for which segmentation was done
def extract_z_values(folder_path):
    """
    Search for files in folder_path with pattern like:
    BF_custom_stitched_merged_t20_z01_ch00_cp_masks.png
    and extract the z value.
    """
    z_pattern = re.compile(r"_z(\d+)_")  # matches _z01_, _z15_, etc.
    for file in os.listdir(folder_path):
        if file.endswith(".png"):  # only process .png files
            match = z_pattern.search(file)
            if match:
                z_value = match.group(1)
                return(z_value)

    return z_value


if __name__ == "__main__":
    # Initialize variables
    data_path_live = ROOTPATH + EXP_PATH + 'Live/' + CONDITION + '/'
    data_path_fish = ROOTPATH + EXP_PATH + 'Fish/' + CONDITION + '/'
    video_dir = data_path_live + 'video/'

    # Create all necessary folders for the ongoing preprocessing
    newpaths = [data_path_live + 'live_dataset/',  data_path_live + 'results/', data_path_live + 'clean_masks/']
    newpaths += [data_path_fish + 'masks/', data_path_fish + 'merged/', data_path_fish + 'mip/', data_path_fish + 'bigfish/']
    newpaths += [data_path_fish + 'masks/' + CONDITION + '/', data_path_fish + 'bigfish/' + CONDITION + '/']
    newpaths += [ROOTPATH + EXP_PATH + 'live_cell_features/']

    print(newpaths)

    for newpath in newpaths:
        if not os.path.exists(newpath):
            print(newpath)
            os.makedirs(newpath)

    z_value = extract_z_values(video_dir)

    # EXTRACT JPEG FRAMES FROM TIF FILES
    frame_names = list(pathlib.Path(data_path_live).glob(f'*t*_z{z_value}_ch00.tif'))
    frame_names.sort()
    print(frame_names)
    for i, frame in enumerate(frame_names):
        im = tifffile.imread(frame)
        im = Image.fromarray(im/256)
        im = im.convert("L")
        im.save(video_dir + f'{i}.jpg', quality=100)