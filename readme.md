# IMAGENE

[▶ Watch demo](demo.mp4)

In the following, there are three main variables for data storage
- `ROOTPATH` – main dataset directory (depends on where the data is stored).
- `EXP_PATH` – subfolder for a specific experiment (for example (`2025-10-13_exp/`))
- `CONDITION` – condition name (e.g. `1-NA`, `2-MIX`, `3-24H`)

This repository contains all the steps from raw images processing of live cell images in brightfield (BF) (step 1), raw smFish images (Fish) (step 2) to match (step 3) and features extraction (step 4) and model training with results (step 5). Because raw images are too heavy, only a table with features of each individual cells are stored in folder `6_results/`.

### Algorithm for IMAGENE project

The variables of the experiment are stored in the external file `config.yaml` . 
Depending on which computer the algorithm is running, select the good path of files.
Here the architecture of the files
- Live/
    - Condition A/ # Where are stored the merged BF .tif images : _t*_z*_ch00.tif
        - non_merged/ # Where are stored the non merged BF .tif images : _t*_s*_z*_ch00.tif
        - live_dataset/ # Individual cell folder containing centered cropped images
            - cell_001/
                - t00.tiff
                - t01.tiff
                - ...
            - ...
        - videos/ # Where to save the input images BF and input mask of SAM2
        - results/ # The raw output masks of SAM2 
        - clean_masks/ # Final segmentations masks: filtered from results/
        - lamellipodia_segmentations/ # hand computed masks of lamellipodia
        - nuclear_segmentations/ # hand computed masks of nucleus
    - Condition B/


- Fish/
    - Condition A/
    - Condition B/
    - Masks # Hand annotation helped with cellpose
        - Condition A/
        - Condition B/
    - mip/ # maximum intensity projections
    - bigfish/
        - Condition A/ # localisation of spots detected on each tile and channel
        - Condition B/
    - condition.csv # gene expression profile
- alignment/ # Where are the matched pair of indices between Live and Fish cells
- live_cell_features / # extracted features from live cells

# 1 - Live cell preprocessing

## 1.1 (Optional) Merge the BF images (might already be achieved by the microscope)
In the case where images were exported unmerged, save the images in  `{ROOTPATH}/{EXP_PATH}/Live/{CONDITION}/non_merged/`
Then run `stitch_blend.py`. Save the merged images in `{ROOTPATH}/{EXP_PATH}/Live/{CONDITION}/`.

## Manual steps
Run pretrained cpam cellpose model to segment last frame of the whole BF FOV (merged one) (or first but we recommand last) and do manual correcting of the mask.
If last: set BACKWARD to True, else False.
The segmentation of cellpose can be achieved via (by replacing the good image_path):
cellpose --use_gpu --model cpsam --image_path image_path --verbose
Save the .png mask in `{ROOTPATH}/{EXP_PATH}/Live/{CONDITION}/videos/`

## 1.2 Prepare the live images
Run `prepare_live_images.py`
Convert the tif files by converting them in `.jpg` images because SAM2 analyzes jpeg only.

## 1.3 Run segmentation with SAM2
Run `segment_video.py --loop_i i` with i going from 0 to 13. Requires GPU.

## 1.4 Clean individualy the mask
Corrects output of SAM2.
Run `clean_mask.py --loop_t t` with t going from 0 to 21 (number of t frames)

## 1.5 Clean the masks as a whole
Corrects output of SAM2 by filtering badly tracked cells (the trajectory is suspiciously wrong).
Run `clean_mask_traj.py`

## 1.6 Create individual videos of cells
Run `create_live_dataset.py`. From the segmented cells obtained before, create individual folder in `Condition A/live_dataset/` and tif file for every time point. The tiff are obtained from cropping of the original tif file (not to loose quality). Also compute trajectory.png and saves in the save folder, image that capture the whole trajectory of the cell (overlayed masks).

# 2 - Analyze gene expression

## 2.1 Compute MIP
Run `fish_mip.py`. Computes the Maximum Intensity Projection (MIP) of each tile depending on rounds selected in array t_rounds (overlayed of all gene expressed). Used later for segmentation. Saved in `{ROOTPATH}/{EXP_PATH}/Fish/mip/`.
Saves FISH signal in first channel and DAPI in second channel.

## Manual steps
Choose thresholds for spots detection for each condition and each gene.
Saves as a dictionnary in `{ROOTPATH}/{EXP_PATH}/Fish/bigfish/{CONDITION}/CONDITION_thresh_BIGFISH.npy`
Segment from the MIP of individual tile. Save the masks in `{ROOTPATH}/{EXP_PATH}/Fish/masks/{CONDITION}/*.png`

## 2.2 Merge the segmentations
Make a big FOV from the individual tiles for future matching. Default merging trajectory is following a snake of size 5x5 (default parameters). For some experiments the shape was different. In this case encode the trajectory in a array in `{ROOTPATH}/{EXP_PATH}/Fish/pattern_snake/CONDITION.npy` (see example in `config_patterns/`). 
A unique ID is attributed to each tile when merging (relabeling procedure). Each relabelled tile is save in `{ROOTPATH}/{EXP_PATH}/Fish/masks/mosaic_mask_reassigned_s*.tiff`, and the merged one is in the same folder with name mosaic_mask_reassigned.tiff.

## 2.3 Run bigfish 
Run `bigfish_detection.py --loop_i s`. With s going from 0 to 24 (for each tile). Needs previously set thresholds (manual step). Needs GPU and do parallel computing. Saves 3D positions of spots in `{ROOTPATH}/{EXP_PATH}/Fish/bigfish/{CONDITION}/` in individual npy file for each tile, round and tile.

## 2.4 Compute gene profile
Run `gene_expression.py`. Simply counts the number of spots detected in each cell. Generates a .csv saved in `{ROOTPATH}/{EXP_PATH}/Fish/{CONDITION}.csv.`


# 3 - Match 
Run `live_fish_matching.py`. Matches Live and Fish masks via brute force technique.
Save a csv with two first columns: brightifeld cell id and fish cell id.
Saved in `{ROOTPATH}/{EXP_PATH}/alignment/{CONDITION}-alignment.csv`

# 4 - Extract features
Manually extract features

## 4.1 Texture features
Run `sobel_filer.py`. Save in `{ROOTPATH}/{EXP_PATH}/Live/live_cell_features/texture_{CONDITION}.csv`.

## 4.2 Fourier Features
Run `hand_features_fourier.py`. Save in `{ROOTPATH}/{EXP_PATH}/Live/live_cell_features/fourier_features_{CONDITION}.csv`.

## 4.3 Cell Features with scikit-image
Run `cell_feats_scikit.py`. Save in `{ROOTPATH}/{EXP_PATH}/Live/live_cell_features/cell_feats_scikit_{CONDITION}.csv`.

# 5 - Learning

## 5.1 Create table of features
Run `concatenate_features.py`. Saves in `{ROOTPATH}/learning/learning_features.csv`.

## 5.2 Training/testing and model analysis
Run `model.py`.
