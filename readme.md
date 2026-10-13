# IMAGENE

Code accompanying the IMAGENE study: predicting the gene expression of single T cells from their behaviour in label-free live imaging.

Each cell is filmed in brightfield (BF), then fixed and imaged by sequential smFISH on the same field of view. The pipeline segments the cells in the BF movie with a finetuned Cellpose model and tracks them over time, counts the RNA spots of each gene in the smFISH images, pairs every live cell with its smFISH counterpart, extracts morphology and motility features from the movie, and trains classifiers that predict gene expression from these features.

[▶ Watch demo](code/demo.mp4)

[▶ Watch the control experiment](control/demo_control.mp4): in this experiment each cell carries a colour barcode, read before the live imaging and again after fixation, to check that a live cell and the fixed cell it is paired with are the same cell. The movie follows the cells from the colour barcode to the timelapse, the timelapse after UV and the colour after fixation, with one identifier per cell.

## Overview

The repository has two folders: `code/`, with the scripts, and `exp/`, where the images and tables of the experiment are placed (see [Data layout](#data-layout)). All the commands below are run from `code/`, and the paths of the files and folders of the code are given from `code/`.

`code/` has one folder per step. Folders and scripts are numbered in the order in which they are run.

| Step | Folder | Purpose | Main output |
|---|---|---|---|
| 1 | `1_bf_analysis/` | Segment the cells on every BF tile and time frame, merge the tiles and track the cells over the movie | Label masks of every cell at every time frame |
| 2 | `2_fish_analysis/` | Detect the RNA spots and assign them to the cells | Gene expression table (spot counts per cell and gene) |
| 3 | `3_match_bf_fish/` | Pair each live cell with its smFISH counterpart | Table of matched BF and FISH cell identifiers |
| 4 | `4_extract_features_ML/` | Compute shape, motility and neighbourhood features of each live cell | Feature tables per cell and time frame |
| 5 | `5_POC_activation/` | Assemble the dataset, train and evaluate the classifiers | `6_results/` |
| 6 | `6_results/` | Tables and figure produced by step 5 | — |
| 7 | `7_figures/` | Figures and video of the paper | `7_figures/output/` |

Raw images are too large to be distributed. `6_results/` contains the final per-cell table, from which step 5.3 can be reproduced directly.

## Setup

### Installation

The Python libraries are listed in `requirements.txt`, with the versions used for the results of the paper (Python 3.10). They are installed in the current environment with:

```bash
conda create -n imagene python=3.10
conda activate imagene
bash install.sh
```

`install.sh` installs the libraries, checks that each of them can be imported, and reports whether the two tools below that are not Python libraries are available.

| Tool | Used in | Role |
|---|---|---|
| [Cellpose](https://github.com/MouseLand/cellpose) (`cpsam` model, and the same model finetuned on the BF images of this study) | Steps 1 and 2 | Cell segmentation. Installed by `install.sh` |
| [Big-FISH](https://github.com/fish-quant/big-fish), through `pipeline_smfish-main` cloned next to the folder `code/` | Step 2 | RNA spot detection. Big-FISH is installed by `install.sh`; `pipeline_smfish-main` is cloned separately |
| [ffmpeg](https://ffmpeg.org) with the H.264 encoder | Step 7 | Encoding of the videos (`conda install -c conda-forge ffmpeg`). Without it, the videos are saved in MPEG-4 |

### Configuration

All parameters are in `config.yaml`; no value has to be edited in the scripts. The paths of `config.yaml` are generic. To use your own paths without changing this file, copy it to `config_confidential.yaml` and edit the copy: when this file exists it is read instead of `config.yaml`, and it is not tracked by git. The pipeline processes one condition of one experiment at a time: change `exp_path` and `condition` and run it again for each condition. Distances and sizes are in pixels of the BF image unless stated otherwise.

**`local`, `cluster` — paths and computer.** `cluster` is used if the host name contains `maestro`, `local` otherwise.

| Parameter | Meaning |
|---|---|
| `rootpath` | Root directory of the data. Can be left empty if `exp_path` is a full path |
| `model_finetune_path` | Cellpose model finetuned on the BF images (manual step M1). By default `models/general_exp_retrained_bf_all_conditions`, from the folder `code/`. The model (1.2 GB) is too large for GitHub and is not in the repository: it has to be placed in `models/` |
| `device` | `cpu` or `cuda` |
| `num_workers`, `rootpath_atlas` | Not used by the current scripts |

**`experiment` — the data being processed.**

| Parameter | Meaning |
|---|---|
| `exp_path` | Folder of the experiment, appended to `rootpath` |
| `condition` | Condition processed by steps 1 to 4, e.g. `1-NA`, `3-24H`, `4-24H` |
| `genes` | Genes in acquisition order: two per smFISH round (channel 0 then channel 1). Sets the name of the columns of the gene expression table |
| `rounds` | smFISH rounds |
| `n_features_dim` | Not used by the current scripts |

**`constants`.**

| Parameter | Meaning |
|---|---|
| `scale` | Ratio between the pixel sizes of the smFISH and BF images, used to rescale the smFISH masks to the BF image (step 3) |
| `max_dist` | Maximum distance between the two cells of a pair in step 3. Kept high on purpose: the distances are saved and the pairs are filtered in step 5 |
| `size_radius` | Radius of the crop of the single-cell movies (step 1.4) |

**`live` — step 1.**

| Parameter | Meaning |
|---|---|
| `n_frames` | Number of time frames of the movie |
| `n_rows`, `n_cols` | Grid of BF tiles, acquired in snake order |
| `tile_size` | Size of a tile |
| `overlap` | Fraction of a tile that overlaps its neighbour |
| `z_finetune` | z plane of the tiles that is segmented |
| `border_margin` | A cell closer than this to the border of its tile is considered truncated when the tiles are merged (1.2) |
| `iou_threshold` | Minimum IoU to link two overlapping cells between consecutive frames (1.3) |
| `max_centroid_distance` | Maximum distance to link two cells whose masks do not overlap (1.3) |
| `min_frames` | Tracks present in fewer frames are not saved (1.3). With 1, all the tracks are kept |
| `min_cell_area` | Objects smaller than this are ignored when the Voronoi diagram is computed (4.2) |

**`fish` — step 2.**

| Parameter | Meaning |
|---|---|
| `n_tiles` | Number of smFISH tiles |
| `n_channels` | Number of smFISH channels per round, DAPI excluded |
| `tile_size`, `overlap` | Size of a tile and overlap between two tiles, in pixels of the smFISH image |
| `default_pattern` | Layout of the tiles used when `Fish/pattern_snake/{CONDITION}.npy` does not exist |
| `mip_rounds` | Rounds included in the projection used to segment the cells (2.1) |
| `voxel_size_nm`, `object_radius_nm` | Voxel size and expected radius of a spot (z, y, x), in nanometres, for Big-FISH (2.3) |
| `clumps_detector` | If `True`, a cell overlapping a fluorescent aggregate gets no value for the corresponding gene (2.6) |
| `clumps_min_size` | Aggregates smaller than this, in pixels of the smFISH image, are ignored |

**`matching` — step 3.**

| Parameter | Meaning |
|---|---|
| `dapi_mask` | If `True`, the alignment uses the masks of the nuclei instead of the masks of the cells |
| `translation_n` | Number of translations tested along each axis, for each of the two levels of the search |
| `translation_delta` | First (basic) search: range of the translations tested around the two fields of view, in pixels |
| `translation_delta_refined` | Second (refined) search: the translations are tested within this distance, in pixels, of the best translation of the first search |

**`features` — step 4.**

| Parameter | Meaning |
|---|---|
| `fourier_order` | Number of harmonics of the elliptic Fourier descriptors |
| `fourier_n_points` | Number of points of the reconstructed contour |
| `msd_n_max` | Maximum time lag of the mean squared displacement; shorter tracks get no `alpha` |
| `neighbors_thresholds` | Distances within which the neighbours of a cell are counted |

**`learning` — step 5 and figures.**

| Parameter | Meaning |
|---|---|
| `exp_paths`, `conditions` | Experiments, and conditions of each experiment, gathered in `dataset.csv` (5.1) |
| `conditions_gene_profile` | Conditions of each experiment gathered in `all_gene_profile.csv` (5.2) |
| `filter_error`, `max_speed` | If `filter_error` is `True`, cells touching the border of the field of view, and cells moving by more than `max_speed` between two frames, are left out of the dataset |
| `alpha_max_std_err` | Above this standard error of the fit, the exponent `alpha` is a missing value (replaced by the median in the model) |
| `genes` | Genes predicted by the model |
| `gene_thresholds` | Number of RNA from which the expression of a gene is called high |
| `min_mean_rna` | Minimum number of RNA per gene of a cell used by the models: total number of RNA of the genes of `genes` divided by the number of these genes with a value for the cell. The matched cells below it have almost no RNA in any gene and form a group separate from the other cells |
| `ratio_qc` | Quality control of the matching, between 0 and 1, larger is stricter. A pair is kept if `1 − d / d₂ ≥ ratio_qc` for both cells, `d` being the distance between the two cells and `d₂` the distance to the second nearest candidate |
| `n_splits` | Number of folds of the cross-validation, for both the evaluation (outer folds) and the selection of the hyperparameters (inner folds) |
| `n_iter` | Number of hyperparameter combinations tested by the random search |
| `n_extreme` | Number of cells with the lowest, and with the highest, expression kept for the analysis on extreme values |
| `n_permutations` | Number of shuffles of the labels in the permutation test. The smallest p-value that can be obtained is 1 / (`n_permutations` + 1) |
| `n_bootstrap` | Number of resamplings of the cells for the confidence intervals |
| `random_state` | Seed of the folds, of the random search and of the random forest, so that the results are identical from one run to the next |
| `n_jobs` | Number of cores used to train the models at the same time in steps 5.3 and 5.5 (`-1`: all the cores of the computer). The results do not depend on it |
| `param_dist` | Values of the hyperparameters of the random forest explored by the random search |

The parameters of the figures are in a separate file, `7_figures/config_figures.yaml` (see step 7).

### Running the scripts

Each folder contains one file, `run.sh`, that runs all the steps of the folder in order and stops at the first step that fails:

```bash
bash 1_bf_analysis/run.sh
```

A single script is run from the folder `code/`, as a module and without the `.py` extension:

```bash
python3 -m 1_bf_analysis.1_stitch_blend
```

The shell commands of the sections below are the content of the `run.sh` files. They use variables read from `config.yaml`, which are set in the current shell with:

```bash
eval "$(python3 -m tools.config_to_shell)"
```

| Variable | Value |
|---|---|
| `ROOTPATH`, `EXP_PATH`, `CONDITION` | As in `config.yaml` |
| `LIVE`, `FISH` | `{ROOTPATH}{EXP_PATH}Live/{CONDITION}` and `{ROOTPATH}{EXP_PATH}Fish` |
| `MODEL_FINETUNE` | Finetuned Cellpose model |
| `N_FRAMES` | Number of time frames |
| `Z_FINETUNE` | z plane of the BF tiles that is segmented |
| `FISH_N_TILES` | Number of smFISH tiles |

In step 2, `run.sh` uses the Cellpose masks of the smFISH tiles without correction when it is run from start to end. To correct them by hand, comment out the steps that follow the manual step, run, correct the masks, then run again with the completed steps commented out.

### Data layout

With the default `config.yaml` (`rootpath: '../'`, `exp_path: 'exp/'`), the experiment is the folder `exp/` next to `code/`. Other locations are set in `config_confidential.yaml`.

```
{ROOTPATH}/{EXP_PATH}/                 exp/ by default
├── Live/
│   └── {CONDITION}/                  merged BF images: *_t*_z*_ch00.tif
│       ├── non_merged/               BF tiles: *_t*_s*_z*_ch00.tif
│       │   └── finetune_z2/          Cellpose masks of the tiles: *_cp_masks.png
│       ├── global_masks/             mask of the whole field of view at each frame (global_mask_t*.png) and tile of origin of each cell (global_id_map.csv)
│       ├── track/tracks.csv          tracks: identifier, frame, cell in the global mask, centroid and area
│       └── live_dataset/cell_*/      cropped movie of each cell: t*.tiff
├── Fish/
│   ├── {CONDITION}/                  smFISH tiles: *_t*_s*_z*_ch*.tif (t = round, ch02 = DAPI)
│   ├── mip/                          maximum intensity projection of each tile
│   ├── masks/{CONDITION}/            cell masks of each tile and of the mosaic
│   ├── bigfish/{CONDITION}/          detection thresholds and spot coordinates
│   ├── clumps_detection/             thresholds and masks of the fluorescent aggregates
│   ├── pattern_snake/{CONDITION}.npy tile layout, if different from the default (config_patterns/5x5.npy)
│   └── {CONDITION}.csv               gene expression table
├── alignment/{CONDITION}-alignment.csv   matched BF and FISH cells
└── live_cell_features/               feature tables of step 4
```

---

## Step 1 — Live cell segmentation and tracking (`1_bf_analysis/`)

Every BF tile is segmented at every time frame with a Cellpose model finetuned on these images. The tile masks of a frame are merged into one mask of the whole field of view, and the cells are then linked from frame to frame so that each cell gets one identifier over time. The masks are not modified after segmentation: tracking only produces a table that says which cell of each frame belongs to which track.

| # | Script | Purpose | Input | Output |
|---|---|---|---|---|
| 1.1 | `1_stitch_blend.py` | Stitches the BF tiles (4 × 4 grid in snake order, 10 % overlap) with linear blending. The merged images are used for the single-cell movies (1.4). Only needed if the microscope exported unmerged tiles. | `non_merged/*_t*_s*_z*_ch00.tif` | `BF_custom_stitched_merged_t*_z*_ch00.tif` |
| M1 | *manual* — finetuned Cellpose | Segments every BF tile (z plane `z_finetune`) at every time frame. | `non_merged/*_z{Z_FINETUNE}_ch00.tif` | `non_merged/finetune_z2/*_cp_masks.png` |
| 1.2 | `2_merge_masks.py` | Merges the tile masks of each frame into one mask of the field of view. A cell seen in two overlapping tiles is kept once: the version that is not cut by a tile border is preferred, otherwise the one of the first tile. | Masks of M1 | `global_masks/global_mask_t*.png`, `global_masks/global_id_map.csv` (for each cell of each frame: `global_id`, tile `s` and identifier in the tile `local_id`) |
| 1.3 | `3_track_cells.py` | Links the cells between consecutive frames by optimal one-to-one assignment (Hungarian algorithm). The cost of a link is the overlap of the two masks (IoU) and, when they do not overlap, the distance between centroids (at most `max_centroid_distance`). A cell that is not linked starts a new track. | `global_masks/` | `track/tracks.csv`: one row per track and frame, with `unique_id` (track), `t`, `global_id` (cell in the global mask of the frame), `centroid_x`, `centroid_y`, `area` |
| 1.4 | `4_create_live_dataset.py` | Crops a movie centred on each cell tracked over all the frames, at the sharpest z plane (Laplacian variance). | Merged frames, `global_masks/`, `track/` | `live_dataset/cell_*/t*.tiff`, `trajectory.npy`; `live_cell_features/pick_z_{CONDITION}.csv` |

The mask of a track at a given frame is the cell `global_id` of `global_mask_t*.png`, and its mask in the original tile is the cell `local_id` of the tile `s` (`global_id_map.csv`). The following steps use these two tables to read the tracked cells from the global masks or from the tile masks.

No cell is removed at this stage. Tracks that are not present in all the frames, or that touch the border of the field of view, are excluded when the dataset is used (step 5).

The number of frames, the tile grid, the tile size and the overlap are set in the `live` section of `config.yaml` (21 frames and a 4 × 4 snake of 2048 × 2048 tiles with 10 % overlap in this study). The merging procedure assumes that cells are smaller than the overlap between tiles, so that each cell is entirely visible in at least one tile.

### Commands

**1.1 — Stitching** (optional)

```bash
python3 -m 1_bf_analysis.1_stitch_blend
```

**M1 — Segmentation of the tiles with the finetuned model**

```bash
mkdir -p ${LIVE}/non_merged/finetune_z2
cp ${LIVE}/non_merged/*_z${Z_FINETUNE}_ch00.tif ${LIVE}/non_merged/finetune_z2/
cellpose --use_gpu --pretrained_model ${MODEL_FINETUNE} \
    --dir ${LIVE}/non_merged/finetune_z2 --savedir ${LIVE}/non_merged/finetune_z2 \
    --save_png --no_npy --verbose
```

**1.2 to 1.4 — Merging, tracking and single-cell movies**

```bash
python3 -m 1_bf_analysis.2_merge_masks
python3 -m 1_bf_analysis.3_track_cells
python3 -m 1_bf_analysis.4_create_live_dataset
```

---

## Step 2 — Gene expression from smFISH (`2_fish_analysis/`)

Cells are segmented on each smFISH tile and assembled into one mosaic with unique identifiers. RNA spots are detected in 3D for each round and channel, and counted per cell. Cells overlapping a fluorescent aggregate are excluded for the affected gene.

| # | Script | Purpose | Input | Output | Compute |
|---|---|---|---|---|---|
| 2.1 | `1_fish_mip.py` | Computes the maximum intensity projection of each tile: FISH signal of the selected rounds (channel 1) and DAPI (channel 2). | `Fish/{CONDITION}/*.tif` | `Fish/mip/{CONDITION}_round_*_s*_mip.tif` | CPU |
| M3 | *manual* — Cellpose | Segments the cells on the projection of each tile and corrects the masks by hand. | `Fish/mip/` | `Fish/masks/{CONDITION}/*_s*_*.png` | GPU |
| M4 | *manual* — thresholds | Chooses the spot detection threshold of each round and channel. | smFISH tiles | `Fish/bigfish/{CONDITION}/{CONDITION}_thresh_BIGFISH.npy` | — |
| 2.2 | `2_cell_reassignment.py` | Assembles the tile masks into one mosaic, gives each cell a unique identifier (`1000 × tile + cell`), and removes cells touching a tile border or duplicated in an overlap. | Masks of M3 | `Fish/masks/{CONDITION}/mosaic_mask_reassigned_s*.tiff` (tiles), `mosaic_mask_reassigned.tiff` (mosaic) | CPU |
| 2.3 | `3_bigfish_detection.py --loop_i s` | Detects the RNA spots in 3D with Big-FISH, for one tile and all rounds and channels. | smFISH tiles, thresholds of M4 | `Fish/bigfish/{CONDITION}/t*_s*_ch*.npy` | GPU |
| 2.4 | `4_clumps_detector_streamlit.py` | Interactive selection of the intensity threshold above which a region is a fluorescent aggregate and not single spots. | smFISH tiles, outputs of 2.2 and 2.3 | `Fish/clumps_detection/{CONDITION}_thresholds.csv` | local, interactive |
| 2.5 | `5_clumps_detector.py` | Builds the mosaic mask of the aggregates for each round and channel. | Thresholds of 2.4 | `Fish/clumps_detection/{CONDITION}_mask_total_t*_ch*.png` | CPU |
| 2.6 | `6_gene_expression.py` | Counts the spots of each gene in each cell. Spots in tile overlaps are counted once, and cells overlapping an aggregate get no value for that gene. | Outputs of 2.2, 2.3 and 2.5 | `Fish/{CONDITION}.csv`: `Gene_*` (counts), `delta_z_l_*`, `delta_z_r_*` (distance of the spots to the first and last z plane) | CPU |

The default tile layout is a 5 × 5 snake (`config_patterns/5x5.npy`) of 2048 × 2048 tiles with an overlap of 206 pixels; these values are set in the `fish` section of `config.yaml`. Another layout is given as an array of tile indices in `Fish/pattern_snake/{CONDITION}.npy`.

### Commands

**2.1 — Projections**

```bash
mkdir -p ${FISH}/mip ${FISH}/masks/${CONDITION} ${FISH}/bigfish/${CONDITION} ${FISH}/clumps_detection
python3 -m 2_fish_analysis.1_fish_mip
```

**M3 — Manual segmentation of the smFISH tiles.** One call per tile.

```bash
for s in $(seq -w 0 $((FISH_N_TILES-1))); do
    cellpose --use_gpu --model cpsam --image_path ${FISH}/mip/${CONDITION}_round_*_s${s}_mip.tif --save_png --verbose
done
```

Correct the masks in the Cellpose interface, save them as PNG, then place them in `masks/`:

```bash
mv ${FISH}/mip/${CONDITION}_round_*_mip_cp_masks.png ${FISH}/masks/${CONDITION}/
```

**M4 — Detection thresholds.** They must be saved before step 2.3 is reached, otherwise `run.sh` stops there. One threshold per round `t` and channel `ch`, stored in a dictionary whose keys contain `t{round}_` and `_ch{channel}`. The values below are placeholders.

```bash
python3 - <<EOF
import numpy as np
thresholds = {
    't00_s00_ch00': 100, 't00_s00_ch01': 100,
    't01_s00_ch00': 100, 't01_s00_ch01': 100,
    # ... one entry per round and channel
}
np.save('${FISH}/bigfish/${CONDITION}/${CONDITION}_thresh_BIGFISH.npy', thresholds)
EOF
```

**2.2 — Mosaic of the masks**

```bash
python3 -m 2_fish_analysis.2_cell_reassignment
```

**2.3 — Spot detection.** One call per tile.

```bash
for s in $(seq 0 $((FISH_N_TILES-1))); do
    python3 -m 2_fish_analysis.3_bigfish_detection --loop_i $s
done
```

**2.4 — Thresholds of the aggregates.** Interactive, so it is not run by `run.sh` and must be done before step 2.5. For each tile, round and channel showing an aggregate, adjust the threshold and save it.

```bash
streamlit run 2_fish_analysis/4_clumps_detector_streamlit.py
```

**2.5 and 2.6 — Aggregate masks and gene expression**

```bash
python3 -m 2_fish_analysis.5_clumps_detector
python3 -m 2_fish_analysis.6_gene_expression
```

---

## Step 3 — Matching live and fixed cells (`3_match_bf_fish/`)

| # | Script | Purpose | Input | Output | Compute |
|---|---|---|---|---|---|
| 3.1 | `1_live_fish_matching.py` | Rescales the smFISH mosaic to the BF pixel size, finds the translation between the two images by exhaustive search on the cell centroids, then pairs each smFISH cell with its nearest BF cell in the last frame. The identifier and the centroid of the BF cells are read from the table of the tracks. No pair is discarded at this stage: the distances to the first and second nearest neighbours are saved, and the quality control of the pairs is applied in step 5. | `track/tracks.csv` (last frame), `Fish/masks/{CONDITION}/mosaic_mask_reassigned.tiff` | `alignment/{CONDITION}-alignment.csv`: `BF`, `Fish`, centroids, `Distance`, `Distance_2nd_*`, `Id_2nd_*`; `{CONDITION}-overlay.npy`: the BF mask of the last frame and the smFISH mask aligned on it, on the whole BF field of view; `{CONDITION}-matching.png`: illustration of the alignment and of the pairs, before and after the quality control | CPU |

```bash
mkdir -p ${ROOTPATH}${EXP_PATH}alignment
python3 -m 3_match_bf_fish.1_live_fish_matching
```

---

## Step 4 — Live cell features (`4_extract_features_ML/`)

All tables are written in `{ROOTPATH}/{EXP_PATH}/live_cell_features/`, with one row per cell (`cell_bf`) and time frame (`t`) unless stated otherwise.

| # | Script | Purpose | Input | Output |
|---|---|---|---|---|
| 4.1 | `1_hand_features_fourier_finetune.py` | Features of each tracked cell. The contour is measured on the Cellpose mask of the tile of the cell, and the other features on the global masks; both are linked to the track through the tables of step 1: elliptic Fourier descriptors of the contour (5 harmonics), scikit-image shape descriptors, overlap of the cell with itself between consecutive frames (IoU), total area explored, and cells touching the border of the field of view. | Tile masks of M1, `global_masks/`, `track/` | `fourier_finetune_{CONDITION}.csv`, `finetune_scikit_{CONDITION}.csv`, `finetune_iou_{CONDITION}.csv`, `finetune_area_explored_{CONDITION}.csv` (per cell), `is_on_edge_{CONDITION}.csv`, `finetune_tile_{CONDITION}.csv` |
| 4.2 | `2_dynamics_features.py` | Motility and neighbourhood: exponent `alpha` of the mean squared displacement (per cell), Voronoi cell area, distance to the border, and number of neighbours within several distances (`neighbors_thresholds`). | `track/tracks.csv` | `dynamics_feats_{CONDITION}.csv` (per cell), `neighbors_feats_{CONDITION}.csv` |

```bash
python3 -m 4_extract_features_ML.1_hand_features_fourier_finetune
python3 -m 4_extract_features_ML.2_dynamics_features
```

### Features

The features belong to five families, defined in `tools/model.py` and used by steps 5.3 and 5.5. A feature measured at each time frame is summarised over the movie by four numbers, with the suffixes `_mean`, `_std`, `_max` and `_min`: this is the "per frame" of the column "Measured". The last column gives the number of features of the models that each row represents: 133 in total, with the current `fourier_order` (5) and `neighbors_thresholds` (5 distances). The column "Parameters" gives the parameters of `config.yaml` on which the feature depends, with their current value ("—": none). In addition, all the features depend on the number of time frames (`n_frames` = 21) and on the z plane used for the segmentation (`z_finetune` = 2).

The size of the cell is carried by one feature only, its mean area. All the other lengths and areas of the shape and overlap families are expressed relative to the size of the cell: "÷ diameter" and "÷ area" in the definitions are divisions of the summary numbers by the mean equivalent diameter and by the mean area of the cell over the movie. The Fourier features are divided in step 5.1, so they are already relative to the size in `6_results/dataset.csv`; the others are stored in pixels (40× image, 0.1625 µm per pixel) in this table and divided when the models are trained (`normalize_features` in `tools/model.py`).

**Size** (1 feature)

| Feature | Definition | Measured | Number of features | Parameters |
|---|---|---|---|---|
| `area_mean` | Mean area of the mask of the cell over the movie, in pixels² | per cell | 1 | — |

**Shape** (86 features; scikit-image descriptors of the global mask, and contour of the tile mask)

| Feature | Definition | Measured | Number of features | Parameters |
|---|---|---|---|---|
| `area_std`, `area_max`, `area_min` | Variation of the area of the mask over the movie, ÷ area | per cell | 3 | — |
| `equivalent_diameter_area` | Diameter of the disc with the same area as the mask, ÷ diameter. Its mean is equal to 1 and is not used: only `_std`, `_max` and `_min` | per frame | 3 (3 summaries) | — |
| `bbox_area` | Area of the bounding box of the mask, ÷ area | per frame | 4 (4 summaries) | — |
| `perimeter` | Perimeter of the mask, ÷ diameter | per frame | 4 (4 summaries) | — |
| `feret_diameter_max` | Longest distance between two points of the mask, ÷ diameter | per frame | 4 (4 summaries) | — |
| `major_axis_length`, `minor_axis_length` | Axes of the ellipse with the same second moments as the mask, ÷ diameter | per frame | 8 (2 × 4 summaries) | — |
| `eccentricity` | Eccentricity of this ellipse: 0 for a disc, close to 1 for an elongated cell | per frame | 4 (4 summaries) | — |
| `orientation` | Angle of the major axis of this ellipse in the image, in radians | per frame | 4 (4 summaries) | — |
| `extent` | Area of the mask divided by the area of its bounding box | per frame | 4 (4 summaries) | — |
| `solidity` | Area of the mask divided by the area of its convex hull: 1 for a convex cell | per frame | 4 (4 summaries) | — |
| `circularity` | 4π × area / perimeter²: 1 for a disc, lower for an elongated or irregular cell | per frame | 4 (4 summaries) | — |
| `magnitude_k` | Magnitude of harmonic *k* of the elliptic Fourier descriptors of the contour, √(a² + b² + c² + d²), ÷ diameter. `k` goes from 0 (first harmonic, the overall ellipse of the cell) to `fourier_order` − 1 (finer details of the contour) | per frame | 20 (5 harmonics × 4 summaries) | `fourier_order` = 5 |
| `elliptic_area_k` | Area of the ellipse traced by harmonic *k*, π \|ad − bc\|, ÷ area | per frame | 20 (5 harmonics × 4 summaries) | `fourier_order` = 5 |

The Fourier coefficients are also normalised for the rotation and the starting point of the contour, in step 4.1.

**Motility** (8 features; trajectory of the centroid of the cell; distances in pixels, not divided by the size of the cell)

| Feature | Definition | Measured | Number of features | Parameters |
|---|---|---|---|---|
| `delta_distance` | Displacement of the centroid between two consecutive frames | per frame | 4 (4 summaries) | `max_speed` = 200 (cells with a larger displacement are removed) |
| `net_displacement` | Distance between the first and the last position | per cell | 1 | — |
| `straightness` | `net_displacement` divided by the length of the trajectory (sum of the displacements): 1 for a straight trajectory | per cell | 1 | — |
| `mean_turning_angle` | Mean angle between two consecutive displacements, in radians | per cell | 1 | — |
| `alpha` | Exponent of the mean squared displacement (1: random walk, above 1: directed motion). Missing when the fit is not reliable (`alpha_max_std_err`) | per cell | 1 | `msd_n_max` = 15, `alpha_max_std_err` = 0.03 |

**Overlap** (10 features; overlap of the masks of the cell over time: these features mix the displacement and the changes of shape of the cell)

| Feature | Definition | Measured | Number of features | Parameters |
|---|---|---|---|---|
| `iou` | Intersection over union of the mask with the mask of the same cell at the next frame: 1 if the cell neither moves nor changes | per frame | 4 (4 summaries) | — |
| `iou_abs` | Same after aligning the two masks on their centroids: only the change of shape remains | per frame | 4 (4 summaries) | — |
| `area_explored_total` | Area of the union of the masks of the cell over the movie, divided by the number of frames, ÷ area: from 1 / number of frames (the cell did not move) to 1 (the cell never overlaps with itself) | per cell | 1 | — |
| `convex_hull_area` | Area of the convex hull of the union of the masks of the cell over the movie, ÷ area | per cell | 1 | — |

**Neighbourhood** (28 features; from the centroids of all the cells of the field of view; distances in pixels)

| Feature | Definition | Measured | Number of features | Parameters |
|---|---|---|---|---|
| `vor_area` | Area of the Voronoi cell of the cell: the smaller, the denser its surroundings | per frame | 4 (4 summaries) | `min_cell_area` = 500 |
| `distance_boundary` | Distance from the centroid to the border of its Voronoi cell | per frame | 4 (4 summaries) | `min_cell_area` = 500 |
| `n_neigh_thresh_d` | Number of cells closer than *d* pixels (`neighbors_thresholds`) | per frame | 20 (5 distances × 4 summaries) | `neighbors_thresholds` = 50, 100, 150, 200, 250 |

Other columns of `6_results/dataset.csv` are not used by the models: the position of the cell (`centroid-*`), the image moments and inertia tensor (`moments_*`, `inertia_*`), `avg_speed`, `max_speed` and `speed_std` (same information as `delta_distance`), `total_distance` (length of the trajectory: for a cell present in all the frames, it is `delta_distance_mean` multiplied by the number of displacements, so the same information), `elongation` of the trajectory, and the columns of the matching and of the gene expression.

---

## Step 5 — Dataset and prediction of gene expression (`5_POC_activation/`)

| # | Script | Purpose | Input | Output |
|---|---|---|---|---|
| 5.1 | `1_concatenate_features.py` | Builds the dataset, one row per live cell. Time series are summarised by their mean, standard deviation, maximum and minimum, and trajectory statistics are added (distance, displacement, straightness, turning angle). All the live cells are kept: the features are joined with the matching table and the gene expression when the cell has a smFISH partner, and these columns are empty otherwise, so that the table can also be used for analyses of the live cells alone. Cells on the border of the field of view and cells with a displacement above 200 pixels between two frames are removed. Runs over the experiments and conditions of the `learning` section of `config.yaml` (`3-24H` and `4-24H`). | Tables of steps 2, 3 and 4 | `6_results/dataset.csv` |
| 5.2 | `2_concatenate_gene_expr.py` | Gathers the gene expression tables of all experiments and conditions in one file. | `Fish/{CONDITION}.csv` | `6_results/all_gene_profile.csv` |
| 5.3 | `3_model.py` | Keeps reliably matched cells: pairs passing the quality control of the matching, present in the 21 frames, not on the border, with at least `min_mean_rna` RNA per gene on average, and CD8 positive. The quality control uses a single ratio `ratio_qc` between 0 and 1 (larger is stricter): a pair is kept if `1 − d / d₂ ≥ ratio_qc` on both sides, where `d` is the distance between the two cells and `d₂` the distance to the second nearest candidate, so that a pair is kept only when it is unambiguous. For each gene, binarises expression at a fixed threshold (`gene_thresholds`) and trains a random forest. The model is evaluated by nested stratified 5-fold cross-validation: the hyperparameters (randomised search), the median used for the missing values and the scaling of the features are all determined on the training cells of each fold, and the scores are computed on the held-out cells. Each score is reported with a 95 % confidence interval (bootstrap of the cells), and compared with the scores obtained with shuffled labels (permutation test, p-values corrected for the number of genes). The analysis is repeated on the cells with the lowest and highest expression only. | `6_results/dataset.csv` | `6_results/performances_24H.csv`, `6_results/performances_24H_extreme.csv`, `6_results/prediction_{gene}.csv`, `6_results/feature_importances.csv` |
| 5.4 | `4_figures_performances.py` | Draws the figures of the performance from the tables of 5.3, without retraining: size of the dataset of each gene, balanced accuracy in the nested cross-validation, with a star for the genes predicted significantly better than chance. | `6_results/performances_24H.csv`, `6_results/performances_24H_extreme.csv` | `6_results/dataset_size.png`, `6_results/performance.png` |
| 5.5 | `5_model_feature_groups.py` | Measures what each family of features predicts (families described in step 4). The same prediction as 5.3 (same cells, random forest, nested cross-validation) is trained nine times for each gene: with each family alone (size, shape, motility, overlap, neighbourhood), then with size + motility, size + shape, size + motility + shape, and all the families together. The models are tested on the same cells and folds, and each score is given with a 95 % confidence interval (bootstrap of the cells). Each model is also compared with chance by the permutation test of 5.3, corrected for the number of genes (columns `p_value`, `q_value` and `significant`). | `6_results/dataset.csv` | `6_results/performances_feature_groups.csv` |
| 5.6 | `6_figures_feature_groups.py` | Draws the figures of 5.5, without retraining: balanced accuracy of each set of features, with a star when the model is significantly better than chance. | Table of 5.5 | `6_results/performance_feature_groups.png` |

```bash
python3 -m 5_POC_activation.1_concatenate_features
python3 -m 5_POC_activation.2_concatenate_gene_expr
python3 -m 5_POC_activation.3_model
python3 -m 5_POC_activation.4_figures_performances
python3 -m 5_POC_activation.5_model_feature_groups
python3 -m 5_POC_activation.6_figures_feature_groups
```

`3_model.py` has two options, which can be combined. `--normalize_target` saves its results under a different name, so that the standard results are kept; `4_figures_performances.py` takes the same option to draw the corresponding figures.

| Option | Effect | Suffix of the output files |
|---|---|---|
| `--quick` | 3 hyperparameter combinations, 20 shuffles and 100 resamplings instead of the values of `config.yaml`. To check that everything runs; the p-values are not meaningful | none: overwrites the standard results |
| `--normalize_target` | Predicts the expression adjusted for the size of the cell: the number of RNA is divided by the mean area of the live cell relative to the median area, and the threshold of the gene is applied to this adjusted number | `_normalized_target` |

`5_model_feature_groups.py` takes `--quick` only.

The trainings of steps 5.3 and 5.5 are independent of each other, so they are run at the same time, one per core (`learning.n_jobs` in `config.yaml`). The results are identical whatever the number of cores.


---

## Results (`6_results/`)

The tables are in the repository. The figures (`*.png`) are generated by steps 5.4 and 5.6 and are not tracked by git.

| File | Content |
|---|---|
| `dataset.csv` | One row per live cell: features of step 4 and, for the cells with a smFISH partner, matching information of step 3 and gene expression of step 2. Index: `exp_path`, `condition`, `cell_bf`. |
| `all_gene_profile.csv` | Gene expression of every smFISH cell, for all experiments and conditions. |
| `performances_24H.csv` | Per gene: AUC, balanced accuracy, true positive and true negative rates on the training and test folds, and class sizes, averaged over the folds of the nested cross-validation; 95 % confidence intervals (`*_ci_low`, `*_ci_high`); balanced accuracy with shuffled labels (`Balanced_Accuracy_permuted`, mean, and `_permuted_95`, 95th percentile), `p_value` of the permutation test, `q_value` (p-value corrected for the number of genes, Benjamini–Hochberg) and `significant` (better than chance: `q_value` below 0.05). |
| `performances_24H_extreme.csv` | Same metrics when only the cells with the lowest and highest expression are kept. |
| `prediction_{gene}.csv` | For every live cell: class predicted by the model of the gene (0: low, 1: high), measured expression and true class when the cell is reliably matched. For these cells, the prediction is the one made when the cell was in a test fold. |
| `feature_importances.csv` | Importance of each feature in the random forest of each gene, averaged over the folds. |
| `dataset_size.png` | Number of cells with a low and with a high expression of each gene, in the analysis on all the cells. |
| `performance.png` | Balanced accuracy per gene in the nested cross-validation, for all the cells and for the extreme cells, with its 95 % confidence interval. Star: significantly better than chance (permutation test, corrected for the number of genes). |
| `performances_feature_groups.csv` | Per gene and per set of features (each family alone and their combinations): number of features, class sizes, balanced accuracy and AUC with their 95 % confidence intervals, `p_value` of the permutation test, `q_value` (corrected for the number of genes) and `significant` (`q_value` below 0.05). |
| `performance_feature_groups.png` | Balanced accuracy per gene with each set of features. Star: significantly better than chance. |

---

## Step 7 — Figures (`7_figures/`)

Each script generates the figures of one part of the pipeline from the outputs of the previous steps, and saves them as PDF and PNG in `7_figures/output/`. The scripts only read the data: nothing is recomputed, except the UMAP embeddings and the correlations.

The parameters specific to the figures are in `7_figures/config_figures.yaml`: source of the images, condition and tiles shown, crops, example cells. Cell identifiers left to `null` are chosen automatically, so that the figures can be generated for any experiment. The shared code (loading of the data, saving of the figures) is in `tools/figures.py`.

| # | Script | Figures | Input |
|---|---|---|---|
| 7.1 | `1_segmentation.py` | Contour and identifier of the tracked cells on a BF tile, for all the time frames and for a subset; contours colored by the prediction of the model for one gene | BF tiles, tile masks and tracks of step 1, `6_results/prediction_{gene}.csv` |
| 7.2 | `2_fish_rna.py` | Projection of each gene for a non-activated and an activated tile, without and with the detected spots, the contour of the cells and the spot count; segmentation of the smFISH cells on the projection of all genes and DAPI | smFISH tiles, spots and masks of step 2 |
| 7.3 | `3_clumps.py` | Detection of the fluorescent aggregates and cells excluded for the gene | smFISH tile and mask, thresholds of 2.4 |
| 7.4 | `4_matching.py` | Segmentations of the two modalities and pairs kept after the quality control; alignment map and overlay; number of pairs as a function of `ratio_qc`; number of cells at each step (`cell_counts.csv`) | Overlay and table of step 3, `6_results/dataset.csv` |
| 7.5 | `5_live_features.py` | Voronoi diagram; number of neighbours within several distances; trajectory, path length and explored area of a cell; the cell at each time frame | Merged BF frame and mask, BF tiles, tile masks and tracks of step 1 |
| 7.6 | `6_fourier_features.py` | Reconstruction of the contour of a cell with an increasing number of harmonics; magnitude and elliptic area of each harmonic | Tile masks and tracks of step 1 |
| 7.7 | `7_gene_thresholds.py` | Distribution of the expression of each gene in the non-activated and activated conditions, with the threshold between low and high expression | `6_results/all_gene_profile.csv` |
| 7.8 | `8_umap.py` | UMAP of the live cell features, colored by features and by gene expression, and with a thumbnail of each cell; UMAP of the gene expression, colored by gene expression and by features | `6_results/dataset.csv`, single-cell movies of 1.4 |
| 7.9 | `9_correlations.py` | Clustered matrix of the significant Spearman correlations (false discovery rate below 5 %) between features and genes, for all the cells and for the cells with extreme expression | `6_results/dataset.csv` |
| 7.10 | `10_model_performance.py` | Class sizes, true positive rate, true negative rate and balanced accuracy per gene, for all the cells and for the cells with extreme expression | `6_results/performances_24H.csv`, `performances_24H_extreme.csv` |
| 7.11 | `11_example_cells.py` | BF images over time of a cell with high and a cell with low expression of a gene; their position in the UMAP; their features compared with the distribution over all cells | `6_results/dataset.csv`, merged BF frames, `global_masks/`, `track/` |
| 7.12 | `12_contours_trajectories.py` | Contours and trajectories of cells by condition, and for the cells with the lowest and highest expression of each gene | `6_results/dataset.csv`, tables of 4.1 |
| 7.13 | `13_video.py` | Video of one field of view: live timelapse with tracked contours, DAPI with the contours of the matched fixed cells, smFISH projection, and matching | Merged BF frames, `global_masks/`, `track/`, smFISH images and masks, overlay and table of step 3 |
| 7.14 | `14_video_tile.py` | Video of the timelapse of one BF tile, with the mask and the identifier of each tracked cell. `--condition` and `--tile` select the tile | BF tiles, tile masks and tracks of step 1 |
| 7.15 | `15_cell_flow.py` | Flow (Sankey) diagram of the number of live cells at each step, for each activated condition, with flows proportional to the number of cells and the cells lost at each step shown leaving: tracked over the whole movie, in the field of view of the smFISH image, matched to a smFISH cell, passing the quality control, with at least `min_mean_rna` RNA per gene on average, CD8 positive (`cell_flow.csv`) | `6_results/dataset.csv`, overlay of step 3 |
| 7.16 | `16_size_expression.py` | Scatter plots of the mean area of the live cell against the expression of each gene, for the cells used by the models: number of RNA (`size_expression`), and number of RNA normalized by the size of the cell, as with `3_model.py --normalize_target` (`size_expression_normalized`). Each panel gives the Spearman correlation and the threshold of the gene | `6_results/dataset.csv` |

```bash
bash 7_figures/run.sh
# or one script: python3 -m 7_figures.1_segmentation
```

### Example images

The images needed for the figures of the paper are provided in `7_figures/data/`, so that the figures can be generated without the raw data. They are used when `data_dir` is set in `config_figures.yaml` (default). With `data_dir: null`, the images are read from the folders of the pipeline (`rootpath` and `exp_path` of `config.yaml`) instead.

| Folder | Content | Used by |
|---|---|---|
| `data/live_tiles/{CONDITION}/` | One BF tile (z plane 02) at the 21 time frames, and its masks with the track identifiers; the first frame of a second tile, inside the smFISH field of view, for the figure with the prediction | 7.1, 7.5, 7.6 |
| `data/live_merged/{CONDITION}/` | Part of the merged BF image of the first frame, and its mask | 7.5 |
| `data/fish/{CONDITION}/` | For one smFISH tile: projection and detected spots of each gene, mask of the cells, DAPI projection, thresholds of the aggregates | 7.2, 7.3 |
| `data/matching/` | For each activated condition: overlay of the two segmentations after alignment, and table of the pairs | 7.4, 7.15 |

Scripts 7.7 to 7.10 only read the tables of `6_results/`. Scripts 7.11 to 7.13 need the whole field of view over time or the feature tables of step 4, which are not provided: they are run with `data_dir: null`.

The UMAP embeddings use a fixed random seed, so the figures are identical from one run to the next.
