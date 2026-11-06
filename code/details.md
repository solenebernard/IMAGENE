# Details of the steps

This file describes each step of the pipeline: the scripts, their inputs and outputs, the commands, the features and the result files. The installation, the parameters of `config.yaml`, how to run the scripts and the layout of the data are in [`../README.md`](../README.md).

All the commands are run from the folder `code/`, and the paths of the files and folders of the code are given from `code/`.

| Step | Folder |
|---|---|
| [1 — Live cell segmentation and tracking](#step-1--live-cell-segmentation-and-tracking-1_bf_analysis) | `1_bf_analysis/` |
| [2 — Gene expression from smFISH](#step-2--gene-expression-from-smfish-2_fish_analysis) | `2_fish_analysis/` |
| [3 — Matching live and fixed cells](#step-3--matching-live-and-fixed-cells-3_match_bf_fish) | `3_match_bf_fish/` |
| [4 — Live cell features](#step-4--live-cell-features-4_extract_features_ml) | `4_extract_features_ML/` |
| [5 — Dataset and prediction of gene expression](#step-5--dataset-and-prediction-of-gene-expression-5_poc_activation) | `5_POC_activation/` |
| [Results](#results-expresults) | `exp/results/` |
| [6 — Main figure of the prediction](#step-6--main-figure-of-the-prediction-6_exp_figures) | `6_exp_figures/` |
| [7 — Main figure of the control experiment](#step-7--main-figure-of-the-control-experiment-7_control_figures) | `7_control_figures/` |


### Shared code

| File | Content |
|---|---|
| `tools/read_yaml.py` | Reads `config.yaml` (or `config_confidential.yaml`) and exposes its parameters as constants (`ROOTPATH`, `EXP_PATH`, `CONDITION`, ...), with the shell variables `CONDITION` and `CONFIG_SET` |
| `tools/config_to_shell.py` | Prints the parameters as shell variables, for `run.sh`, `run_control.sh` and the manual steps |
| `tools/tools_bf.py` | Brightfield images of step 1: position of the tiles, Cellpose masks, global masks and tracks, masks with the track identifier of each cell, elliptic Fourier descriptors |
| `tools/tools_fish.py` | smFISH images of step 2: z stacks, projections, snake pattern and mosaic of the tiles |
| `tools/matching.py` | Step 3: translation between the two fields of view, pairs of reciprocal closest neighbours, quality control of the pairs |
| `tools/model.py` | Step 5: families and normalisation of the features, random forest, nested cross-validation, permutation test |
| `tools/figures.py` | Steps 6 and 7: loading of the dataset, UMAP, contours and trajectories, saving and assembly of the figures |
| `tools/style.py` | Font and size of the panels of the figures |
| `config_patterns/` | Snake pattern of the smFISH tiles (`5x5.npy`, default of step 2), written by `create_pattern.py` |

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
| 4.1 | `1_hand_features_fourier_finetune.py` | Features of each tracked cell. The contour is measured on the Cellpose mask of the tile of the cell, and the other features on the global masks; both are linked to the track through the tables of step 1: elliptic Fourier descriptors of the contour (5 harmonics; Kuhl and Giardina, parametrised by the arc length of the contour, computed with pyefd), scikit-image shape descriptors, overlap of the cell with itself between consecutive frames (IoU), total area explored, and cells touching the border of the field of view. | Tile masks of M1, `global_masks/`, `track/` | `fourier_finetune_{CONDITION}.csv`, `finetune_scikit_{CONDITION}.csv`, `finetune_iou_{CONDITION}.csv`, `finetune_area_explored_{CONDITION}.csv` (per cell), `is_on_edge_{CONDITION}.csv`, `finetune_tile_{CONDITION}.csv` |
| 4.2 | `2_dynamics_features.py` | Motility and neighbourhood: exponent `alpha` of the mean squared displacement (per cell), Voronoi cell area, distance to the border, and number of neighbours within several distances (`neighbors_thresholds`). | `track/tracks.csv` | `dynamics_feats_{CONDITION}.csv` (per cell), `neighbors_feats_{CONDITION}.csv` |

```bash
python3 -m 4_extract_features_ML.1_hand_features_fourier_finetune
python3 -m 4_extract_features_ML.2_dynamics_features
```

### Features

The features belong to five families, defined in `tools/model.py` and used by steps 5.3 and 5.5. A feature measured at each time frame is summarised over the movie by four numbers, with the suffixes `_mean`, `_std`, `_max` and `_min`: this is the "per frame" of the column "Measured". The last column gives the number of features of the models that each row represents: 133 in total, with the current `fourier_order` (5) and `neighbors_thresholds` (5 distances). The column "Parameters" gives the parameters of `config.yaml` on which the feature depends, with their current value ("—": none). In addition, all the features depend on the number of time frames (`n_frames` = 21) and on the z plane used for the segmentation (`z_finetune` = 2).

The size of the cell is carried by one feature only, its mean area. All the other lengths and areas of the shape and overlap families are expressed relative to the size of the cell: "÷ diameter" and "÷ area" in the definitions are divisions of the summary numbers by the mean equivalent diameter and by the mean area of the cell over the movie. The Fourier features are divided in step 5.1, so they are already relative to the size in `exp/results/dataset.csv`; the others are stored in pixels (40× image, 0.1625 µm per pixel) in this table and divided when the models are trained (`normalize_features` in `tools/model.py`).

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

Other columns of `exp/results/dataset.csv` are not used by the models: the position of the cell (`centroid-*`), the image moments and inertia tensor (`moments_*`, `inertia_*`), `avg_speed`, `max_speed` and `speed_std` (same information as `delta_distance`), `total_distance` (length of the trajectory: for a cell present in all the frames, it is `delta_distance_mean` multiplied by the number of displacements, so the same information), `elongation` of the trajectory, and the columns of the matching and of the gene expression.

---

## Step 5 — Dataset and prediction of gene expression (`5_POC_activation/`)

| # | Script | Purpose | Input | Output |
|---|---|---|---|---|
| 5.1 | `1_concatenate_features.py` | Builds the dataset, one row per live cell. Time series are summarised by their mean, standard deviation, maximum and minimum, and trajectory statistics are added (distance, displacement, straightness, turning angle). All the live cells are kept: the features are joined with the matching table and the gene expression when the cell has a smFISH partner, and these columns are empty otherwise, so that the table can also be used for analyses of the live cells alone. Cells on the border of the field of view and cells with a displacement above 200 pixels between two frames are removed. Runs over the experiments and conditions of the `learning` section of `config.yaml` (`3-24H` and `4-24H`). | Tables of steps 2, 3 and 4 | `exp/results/dataset.csv` |
| 5.2 | `2_concatenate_gene_expr.py` | Gathers the gene expression tables of all experiments and conditions in one file. | `Fish/{CONDITION}.csv` | `exp/results/all_gene_profile.csv` |
| 5.3 | `3_model.py` | Keeps reliably matched cells: pairs passing the quality control of the matching, present in the 21 frames, not on the border, with at least `min_mean_rna` RNA per gene on average, and CD8 positive. The quality control uses a single ratio `ratio_qc` between 0 and 1 (larger is stricter): a pair is kept if `1 − d / d₂ ≥ ratio_qc` on both sides, where `d` is the distance between the two cells and `d₂` the distance to the second nearest candidate, so that a pair is kept only when it is unambiguous. For each gene, binarises expression at a fixed threshold (`gene_thresholds`) and trains a random forest. The model is evaluated by nested stratified 5-fold cross-validation: the hyperparameters (randomised search), the median used for the missing values and the scaling of the features are all determined on the training cells of each fold, and the scores are computed on the held-out cells. Each score is reported with a 95 % confidence interval (bootstrap of the cells), and compared with the scores obtained with shuffled labels (permutation test, p-values corrected for the number of genes). The analysis is repeated on the cells with extreme expression only: the `n_extreme` lowest expressing cells of the low class and the `n_extreme` highest expressing cells of the high class (a class with fewer cells is kept whole). | `exp/results/dataset.csv` | `exp/results/performances_24H.csv`, `exp/results/performances_24H_extreme.csv`, `exp/results/prediction_{gene}.csv`, `exp/results/feature_importances.csv` |
| 5.4 | `4_figures_performances.py` | Draws the figures of the performance from the tables of 5.3, without retraining: size of the dataset of each gene, balanced accuracy in the nested cross-validation, with a star for the genes predicted significantly better than chance. | `exp/results/performances_24H.csv`, `exp/results/performances_24H_extreme.csv` | `exp/results/figures/dataset_size.png`, `exp/results/figures/performance.png` |
| 5.5 | `5_model_feature_groups.py` | Measures what each family of features predicts (families described in step 4). The same prediction as 5.3 (same cells, random forest, nested cross-validation) is trained nine times for each gene: with each family alone (size, shape, motility, overlap, neighbourhood), then with size + motility, size + shape, size + motility + shape, and all the families together. The models are tested on the same cells and folds, and each score is given with a 95 % confidence interval (bootstrap of the cells). Each model is also compared with chance by the permutation test of 5.3, corrected for the number of genes (columns `p_value`, `q_value` and `significant`). | `exp/results/dataset.csv` | `exp/results/performances_feature_groups.csv` |
| 5.6 | `6_figures_feature_groups.py` | Draws the figures of 5.5, without retraining: balanced accuracy of each set of features, with a star when the model is significantly better than chance. | Table of 5.5 | `exp/results/figures/performance_feature_groups.png` |

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
| `--quick` | 3 hyperparameter combinations, 20 shuffles and 100 resamplings instead of the values of `config.yaml`. To check that everything runs; the p-values are not meaningful | none, but saved in `exp/results/quick/`: the standard results are kept |
| `--normalize_target` | Predicts the expression adjusted for the size of the cell: the number of RNA is divided by the mean area of the live cell relative to the median area, and the threshold of the gene is applied to this adjusted number | `_normalized_target` |

`5_model_feature_groups.py` takes `--quick` only.

The trainings of steps 5.3 and 5.5 are independent of each other, so they are run at the same time, one per core (`learning.n_jobs` in `config.yaml`). The results are identical whatever the number of cores.


---

## Results (`exp/results/`)

The repository contains `dataset.csv`, `all_gene_profile.csv` and the three `performances_*.csv` tables: they are the inputs of the figures and the numbers of the paper. `prediction_{gene}.csv` and `feature_importances.csv` are written by step 5.3 when it is run again, and the figures (`figures/*.png`) by steps 5.4 and 5.6; they are not tracked by git.

| File | Content |
|---|---|
| `dataset.csv` | One row per live cell: features of step 4 and, for the cells with a smFISH partner, matching information of step 3 and gene expression of step 2. Index: `exp_path`, `condition`, `cell_bf`. |
| `all_gene_profile.csv` | Gene expression of every smFISH cell, for all experiments and conditions. |
| `performances_24H.csv` | Per gene: AUC, balanced accuracy, true positive and true negative rates on the training and test folds, and class sizes, averaged over the folds of the nested cross-validation; 95 % confidence intervals (`*_ci_low`, `*_ci_high`); balanced accuracy with shuffled labels (`Balanced_Accuracy_permuted`, mean, and `_permuted_95`, 95th percentile), `p_value` of the permutation test, `q_value` (p-value corrected for the number of genes, Benjamini–Hochberg) and `significant` (better than chance: `q_value` below 0.05). |
| `performances_24H_extreme.csv` | Same metrics on the cells with extreme expression: the `n_extreme` cells with the lowest expression among the cells of the low class, and the `n_extreme` cells with the highest expression among the cells of the high class. A class with fewer than `n_extreme` cells is kept whole. |
| `prediction_{gene}.csv` | For every live cell: class predicted by the model of the gene (0: low, 1: high), measured expression and true class when the cell is reliably matched. For these cells, the prediction is the one made when the cell was in a test fold. |
| `feature_importances.csv` | Importance of each feature in the random forest of each gene, averaged over the folds. |
| `figures/dataset_size.png` | Number of cells with a low and with a high expression of each gene, in the analysis on all the cells. |
| `figures/performance.png` | Balanced accuracy per gene in the nested cross-validation, for all the cells and for the extreme cells, with its 95 % confidence interval. Star: significantly better than chance (permutation test, corrected for the number of genes). |
| `performances_feature_groups.csv` | Per gene and per set of features (each family alone and their combinations): number of features, class sizes, balanced accuracy and AUC with their 95 % confidence intervals, `p_value` of the permutation test, `q_value` (corrected for the number of genes) and `significant` (`q_value` below 0.05). |
| `figures/performance_feature_groups.png` | Balanced accuracy per gene with each set of features. Star: significantly better than chance. |

---

## Step 6 — Main figure of the prediction (`6_exp_figures/`)

The main figure of the prediction of the gene expression is drawn by four scripts and two scripts of step 5, and saved as PDF and PNG in `exp/results/figures/`. The scripts only read the tables of `exp/results/` and of step 4.1: nothing is recomputed, except the UMAP embeddings (fixed seed, identical from one run to the next). The parameters are in `6_exp_figures/config_figures.yaml`, and the shared code (loading of the data, saving and assembly of the figures) in `tools/figures.py`.

| # | Script | Figure | Input |
|---|---|---|---|
| 5.4 | `5_POC_activation/4_figures_performances.py` | Panel g: number of cells with a low and a high expression of each gene, and balanced accuracy per gene for all the cells and for the extreme cells | `exp/results/performances_24H.csv`, `performances_24H_extreme.csv` |
| 5.6 | `5_POC_activation/6_figures_feature_groups.py` | Panel h: balanced accuracy per gene with each set of features | `exp/results/performances_feature_groups.csv` |
| 6.1 | `1_gene_thresholds.py` | Panel f: distribution of the expression of each gene in the non-activated and activated conditions, with the threshold between low and high expression | `exp/results/all_gene_profile.csv` |
| 6.2 | `2_umap.py` | Panels a to e: UMAP of the live cell features, coloured by four features, by the expression of each gene, and with a thumbnail of each cell; UMAP of the gene expression, coloured by the genes and by four features | `exp/results/dataset.csv`, `exp/thumbnails/` |
| 6.3 | `3_contours_trajectories_grid.py` | Panel i: trajectories and contours of the cells with the lowest and the highest expression of CORO1A, GZMB and TNF (`contours_trajectories.grid_genes`). The panels of the trajectories, and those of the contours, share the same limits, set by default so that all the cells fit | `exp/results/dataset.csv`, `exp/live_cell_features/fourier_finetune_*.csv` and `finetune_scikit_*.csv` |
| 6.4 | `4_main_figure_prediction.py` | Main figure, assembled from the PNG of the panels above (`exp/results/figures/main/main_figure_prediction`) | Figures of 5.4, 5.6, 6.1, 6.2 and 6.3 |

```bash
bash 6_exp_figures/run.sh               # the six scripts above, in order
python3 -m 6_exp_figures.2_umap         # a single script
```

**Thumbnails.** `exp/thumbnails/{CONDITION}.npz` holds the first frame of the single-cell movie of each cell (step 1.4), with the contrast of the figure, in 8 bits, so that the UMAP with the cell images can be drawn without the movies (about 10 MB instead of 150 MB). They are used when `data_dir` is set in `config_figures.yaml` (default). With `data_dir: null`, the frames are read from the movies of step 1.4 instead, which gives the same figure. The thumbnails are written again from the movies with:

```bash
python3 -m 6_exp_figures.2_umap --export_thumbnails ../exp/thumbnails/   # with data_dir: null
```

---

## Step 7 — Main figure of the control experiment (`7_control_figures/`)

In the control experiment, the cells are stained with two dyes, which gives each cell a colour barcode (positive or negative in each channel). The barcode is read before the movie and again after fixation: a live cell and the fixed cell it is paired with must have the same barcode, which measures the accuracy of the tracking and of the matching. Two densities of cells are imaged; the panels of a single density show the high density (`--canal 2`; `--canal 05` for the low density).

The images of the control experiment were analysed with the scripts of steps 1 and 3 by `run_control.sh` (see below), then the outputs needed by the figures were gathered in `control/` (`control.exp_path` of `config.yaml`). This folder is what the figure scripts read; the raw images are not included:

| Folder | Content |
|---|---|
| `control/figures_input/canal2/`, `canal05/` | For each density (high, low): barcode intensities of each cell before the movie and after fixation (`*_signal_per_object.csv`), tracks and motion of the cells of the movie and of the movie after UV, pairs of cells between the acquisitions (`match_step_*`), and the masks and frame of the movies drawn in the panels |
| `control/alignment/highdensity_barcode_afterUV/` | Centroids and parameters of the alignment of the last frame of the movie on the barcode after fixation |
| `control/BarcodeAfterUV/canal_highdensity/global_masks/` | Mask of the fixed cells, for the size of their field of view |

Scripts 1 to 7 draw one panel each, in the order of the panels (saved in `control/results/figures/`, `canal2/` for the high density), and script 8 assembles them:

| # | Script | Panel |
|---|---|---|
| 7.1 | `1_sticking_fraction_after_uv.py` | b) Fraction of the cells that stay put over time after the photo-immobilization |
| 7.2 | `2_trajectories_contours_after_uv.py` | c) Trajectories and contours of the cells during the movie, right after and 30 min after the immobilization |
| 7.3 | `3_alignment_search.py` | d) Search of the translation aligning the last frame of the movie on the barcode after fixation: mean distance to the nearest cell over a coarse, then a refined grid of translations |
| 7.4 | `4_matching_live_fixed.py` | e) Segmentation of the last frame of the movie (green), of the fixed cells aligned on it (red), and their overlay with the pairs passing the quality control (`--ratio_qc`, 0.6 by default) |
| 7.5 | `5_confusion_matrices.py` | f) Barcode of the live cell against barcode of the fixed cell: confusion matrices without quality control and with `ratio_qc` ≥ 0.3 and ≥ 0.6, one row per density |
| 7.6 | `6_accuracy_vs_qc_ratio.py` | g) Balanced accuracy of the barcodes, and number of pairs, as a function of `ratio_qc`, for both densities |
| 7.7 | `7_flux_diagram_cells.py` | h) Flux diagram: cells tracked over the movie → alive → with a barcode → paired after fixation → passing the quality control → same or different barcode |
| 7.8 | `8_main_figure_control.py` | Main figure, assembled from the panels above (`control/results/figures/main_figure_control`) |

```bash
bash 7_control_figures/run.sh                                          # the eight scripts above, in order
python3 -m 7_control_figures.5_confusion_matrices --qc_values 0.3 0.6  # a single script (-h lists the options)
```

The scripts share `_common.py` (location of the data, options common to every script, tables used by several figures), `_illustration_lib.py` (barcodes of the paired live and fixed cells, panels e to g), `_attachment_lib.py` (motion of the cells after UV, panels b and c), `pipeline/tracking_accuracy_step0_step4.py` (chain of the cells from the barcode before the movie to the barcode after fixation, panel h) and `pipeline/helpers.py` (loading and alignment of the masks). The barcode is called positive for the half of the cells with the highest 90th percentile in each channel (default of the figures).

### Analysis of the control experiment (`run_control.sh`, for information)

`run_control.sh` produced the content of `control/` from the raw images of the control experiment, which are not in this repository: it is provided for information and cannot be run from the repository. It expects the raw images in the folder of the control experiment (`control.exp_path`):

| Folder | Acquisition | Tiles |
|---|---|---|
| `BarcodeBeforeTimelapse/canal_{density}/` | Colour barcode before the movie (ch00 brightfield, ch01 and ch02 the two dyes) | 4 × 4 |
| `Live/canal_{density}/` | Movie, 21 frames, 4 planes | 4 × 4 |
| `BarcodeAfterUV/canal_{density}/` | Colour barcode after fixation, other objective (scale 40/63) | 8 × 8 (high density), 7 × 8 (low density) |
| `LongMovieAfterUV/canal_highdensity/` | Movie after UV, 172 frames | 1 |

`density` is `highdensity` or `lowdensity`. The images are analysed with the scripts of steps 1 and 3, unchanged, in the folder `analysis/` of the experiment; the parameters of each acquisition (number of frames, tiles) replace the ones of `config.yaml` for each run (`CONFIG_SET`, see `tools/read_yaml.py`):

| # | Script | Purpose |
|---|---|---|
| 1 | `7_control_figures/analysis/1_link_images.py` | Links the images in `analysis/Live/{acquisition}/non_merged/` with the names read by step 1 (only the plane z02 of the movies). One acquisition per `CONDITION`: `step0_{density}` (barcode before the movie), `step1_{density}` (movie), `afterUV_{density}` (barcode after fixation), `step3_{density}` (movie after UV). The Cellpose masks already computed for these images are linked too |
| M1 | finetuned Cellpose | Segmentation of every tile (off by default: `SEGMENTATION=false` reuses masks already computed, linked by `1_link_images.py`) |
| 1.2 | `1_bf_analysis/2_merge_masks.py` | Global mask of each acquisition and frame |
| 1.3 | `1_bf_analysis/3_track_cells.py` | Tracking of the movies; and barcode before the movie ↔ first frame of the movie, tracked as a movie of two frames |
| 1.1 | `1_bf_analysis/1_stitch_blend.py` | Stitched images of the movie (high density) |
| 3.1 | `3_match_bf_fish/1_live_fish_matching.py` | Last frame of the movie ↔ barcode after fixation, as for the smFISH (the mask after fixation in place of the smFISH mask); last frame of the movie ↔ first frame of the movie after UV (scale 1, wider translation search) |
| 2 | `7_control_figures/analysis/2_prepare_figure_inputs.py` | Builds `figures_input/` from the outputs above: links to the images and masks, tracks of step 1.3 and pairs of step 3.1 converted to the tables read by the figures |
| — | `7_control_figures/pipeline/barcode_intensities.py` | Barcode of each cell: 90th percentile of ch01 and ch02 in its mask, before the movie and after fixation (`*_signal_per_object.csv`) |
| — | `7_control_figures/pipeline/track_dynamics.py` | Motion and shape of each tracked cell over the movie, and over successive windows of 21 frames of the movie after UV (`*_track_dynamics*.csv`, `*_track_per_frame.csv`) |
| — | `7_control_figures/run.sh` | Main figure of the control experiment |

The tracks and the pairs of cells used by the figures are therefore those of steps 1.3 and 3.1 of the pipeline. The folder `alignment/` (centroids and translation read by `3_alignment_search.py`) is not written by `run_control.sh`.

