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

Each step is described in [`code/details.md`](code/details.md): scripts, inputs and outputs, commands, definition of the features, and content of the result files.

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

**`experiment` — the data being processed.**

| Parameter | Meaning |
|---|---|
| `exp_path` | Folder of the experiment, appended to `rootpath` |
| `condition` | Condition processed by steps 1 to 4, e.g. `1-NA`, `3-24H`, `4-24H` |
| `genes` | Genes in acquisition order: two per smFISH round (channel 0 then channel 1). Sets the name of the columns of the gene expression table |
| `rounds` | smFISH rounds |

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

The parameters of the figures are in a separate file, `7_figures/config_figures.yaml` (see step 7 in [`code/details.md`](code/details.md)).

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
