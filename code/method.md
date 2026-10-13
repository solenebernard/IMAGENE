# Methods

Passages in square brackets are to be completed.

## Image acquisition

Live cells were imaged in brightfield (BF) with a 40× objective (pixel size 0.1625 µm), as a 4 × 4 grid of 2048 × 2048-pixel tiles (332.8 µm) with 10 % overlap, covering a field of 1231.4 µm, at 21 time points separated by 21 seconds (Δt = 21 s, 7 minutes between the first and the last time point) and several focal planes. The same field was then imaged by sequential smFISH with a 63× objective (pixel size 0.1032 µm), as a 5 × 5 grid of 2048 × 2048-pixel tiles (211.3 µm) with 10 % overlap, covering a field of 972.0 µm, with two RNA channels and one DAPI channel per round, as z-stacks. Each gene corresponds to one round and one channel. Unless stated otherwise, distances and areas below are given in pixels of the BF images (1 pixel = 0.1625 µm).

## Segmentation of live cells

Each BF tile was segmented independently at every time point with Cellpose-SAM (`cpsam`), on the focal plane in which cells appeared sharpest by visual inspection; the same plane was used for all tiles and time points. The model was fine-tuned on eight BF tiles drawn at random from the non-activated and 24 h-activated conditions.

The tile masks of a time point were merged into one label image of the whole field of view. Tiles were placed according to the acquisition (snake) order and processed in increasing tile index. A cell was considered truncated if any of its pixels lay within 3 pixels (0.5 µm) of the tile border. When a cell overlapped a cell already placed from a previous tile, a single instance was kept: a non-truncated instance was preferred over truncated ones, and otherwise the instance from the first tile was retained. The tile of origin of each cell was recorded, so that contour descriptors could be computed on the original tile mask.

## Tracking

Cells were linked between consecutive time points by solving a linear assignment problem (Hungarian algorithm). For a cell *i* at time *t* and a cell *j* at time *t* + 1, the cost was 1 − IoU(*i*, *j*) when their masks overlapped and 1 + *d*(*i*, *j*)/*D* otherwise, where IoU is the intersection over union of the two masks, *d* the distance between their centroids and *D* the diagonal of the field of view. Any pair of overlapping masks is therefore preferred to any pair of non-overlapping ones. An assigned pair was accepted if the masks overlapped or, failing that, if *d* ≤ 100 pixels (16.3 µm). A cell left unlinked initiated a new track. Segmentation masks were not modified by tracking.

For the analysis, we retained tracks present at all 21 time points and never touching the border of the field of view.

## Live-cell features

**Contour.** The outer contour of each cell was extracted from its mask in the original tile. Its elliptic Fourier descriptors (*a*ₖ, *b*ₖ, *c*ₖ, *d*ₖ) were computed for the first 5 harmonics and normalised for rotation and starting point. Each harmonic *k* was summarised by its magnitude, √(*a*ₖ² + *b*ₖ² + *c*ₖ² + *d*ₖ²), and by the area of the ellipse traced by the harmonic, π |*a*ₖ*d*ₖ − *b*ₖ*c*ₖ|. To remove the size of the cell, the summaries of the magnitudes over the movie (see below) were divided by the mean equivalent diameter of the cell, and those of the elliptic areas by its mean area.

**Shape.** Area, bounding-box area, perimeter, eccentricity, equivalent diameter, extent, maximum Feret diameter, major and minor axis lengths, orientation and solidity were measured at each time point, as well as the circularity (4π × area / perimeter², equal to 1 for a disc).

**Dynamics.** At each time point we measured the displacement of the centroid, the intersection over union of the mask with the mask of the same cell at the next time point, and the same quantity after aligning the two masks on their centroids, which isolates the change of shape from the displacement. For each trajectory we computed the total path length, the net displacement, the straightness (net displacement divided by path length), and the mean turning angle. From the union of the masks of the cell over the movie we computed the area covered by the cell, divided by the number of time points, and the area of the convex hull of this union. The exponent α of the time-averaged mean squared displacement was obtained by a linear fit in log–log scale over lags of 1 to 14 time points; it was treated as missing when the standard error of the fit exceeded 0.03.

**Neighbourhood.** At each time point, a Voronoi diagram was built from the centroids of all cells larger than 500 pixels (13.2 µm²) and clipped to the field of view. We recorded the area of the Voronoi cell of each cell, the distance from its centroid to the boundary of that Voronoi cell, and the number of cells within 50, 100, 150, 200 and 250 pixels (8.1 to 40.6 µm).

**Normalisation by cell size.** So that the size of the cell was carried by a single feature, its mean area over the movie, the other features with the dimension of a length or of an area were expressed relative to the size of the cell before training. The summaries of the perimeter, of the Feret diameter, of the axis lengths, of the equivalent diameter and of the Fourier magnitudes were divided by the mean equivalent diameter of the cell; the summaries of the bounding-box area and of the Fourier elliptic areas, the standard deviation, maximum and minimum of the area, the area covered by the cell and the area of the convex hull of its masks were divided by its mean area. The distances travelled by the centroid and the neighbourhood features were not rescaled.

Quantities measured at each time point were summarised per cell by their mean, standard deviation, minimum and maximum over the movie.

## Registration of live and smFISH images

The smFISH mosaic of cell masks was rescaled to the BF pixel size (factor 40/63, the ratio of the two pixel sizes; nearest-neighbour interpolation), and objects smaller than 500 pixels (13.2 µm²) or whose centroid fell outside their mask were discarded. Registration used cell centroids only: those of the smFISH cells, and those of the live cells at the last time point of the movie.

The two fields of view were related by a translation, estimated by exhaustive search. For each candidate translation on a regular 60 × 60 grid, we computed the mean distance from each smFISH centroid to the nearest translated BF centroid. The first search covered translations up to 2000 pixels (325 µm) beyond the difference of size between the two fields of view. The translation minimising this quantity was refined by a second 60 × 60 search within ± 300 pixels (48.8 µm) of the first optimum.

After registration, a live cell and a smFISH cell were paired if each was the nearest neighbour of the other. For every pair we recorded the centroid distance *d*, the distance *d*₂ᴮ from the live cell to its second-nearest smFISH cell, and the distance *d*₂ᶠ from the smFISH cell to its second-nearest live cell.

## Quality control of the pairs

Cells can move, detach or be lost between the end of live imaging and the smFISH acquisition, so a nearest-neighbour pair is not necessarily correct, in particular in dense regions. The ambiguity of each pair was quantified by comparing its distance with that of the best alternative on both sides:

*q* = min(1 − *d*/*d*₂ᴮ, 1 − *d*/*d*₂ᶠ).

*q* ranges from 0, when an alternative partner is as close as the assigned one, to 1, when no other candidate is nearby. Pairs with *q* ≥ 0.3 were retained: the second-nearest candidate had to be at least 1.4 times farther than the assigned partner, for both cells. This criterion is relative and therefore adapts to local cell density.

## Prediction of gene expression from live-cell features

**Cells and labels.** The analysis was restricted to live cells retained after tracking, with a pair passing the quality control and at least two CD8 transcripts. Cells with fewer than 3 transcripts per gene on average (total number of transcripts over the ten genes divided by the number of genes with a value for the cell) were excluded (29 of the 496 matched cells, with at most 2.1 transcripts per gene, against at least 5.3 for the others), with almost no transcript of any gene. For each gene, expression was binarised at a gene-specific threshold.

**Model.** A random forest classifier (100 trees, class weights inversely proportional to class frequencies) was trained to predict the class of a cell from its live-cell features. Missing feature values were replaced by the median of the feature, and features were standardised.

**Nested cross-validation.** Performance was estimated by nested stratified 5-fold cross-validation. The cells of the two activated conditions, each acquired as one field of view, were pooled and split at random into five folds with the same proportion of low and high cells, so that the training and the held-out cells of a fold come from both fields of view. In each outer fold, the hyperparameters were selected using the training cells only, by a randomised search of 30 combinations (maximum depth: unlimited, 5, 10 or 20; minimum samples per split: 2, 5 or 10; minimum samples per leaf: 1, 2 or 4; features considered per split: square root, binary logarithm, or all; bootstrap sampling: with or without) scored by balanced accuracy in an inner stratified 5-fold cross-validation. The median imputation and the standardisation were also fitted on the training cells only. The selected model was refitted on the training cells and evaluated on the held-out cells. We report the balanced accuracy, defined as the mean of the true-positive and true-negative rates, and the area under the ROC curve, averaged over the five outer folds.

**Confidence intervals.** Confidence intervals (95 %) were obtained by bootstrap of the cells: the predictions of the held-out cells, pooled over the five outer folds, were resampled 1,000 times with replacement, and the interval is given by the 2.5th and 97.5th percentiles of the score over the resamples.

**Statistical significance.** A gene was declared predicted significantly better than chance, and marked with a star in the figures, on the basis of a permutation test of the balanced accuracy, computed as follows.

1. *Score expected by chance.* The low and high labels of the cells were shuffled at random, which keeps the number of cells of each class but removes any link between the features and the expression. The 5-fold cross-validation was run again on these shuffled labels and its balanced accuracy recorded. This was repeated 200 times, giving the distribution of the balanced accuracy when the features carry no information on the gene.
2. *Same settings for real and shuffled labels.* To keep the computation tractable, the hyperparameters were not searched again for each shuffle: the combination selected most often across the five outer folds was used. The balanced accuracy with the real labels was recomputed with this same combination, so that real and shuffled labels are compared under identical settings.
3. *p-value.* The p-value of a gene is the proportion of runs, among the 200 shuffles and the real one, in which the balanced accuracy is at least as high as with the real labels: p = (1 + *k*) / 201, where *k* is the number of shuffles reaching the real score. A small p-value means that such a score is rarely obtained by chance; the smallest possible value is 1/201 ≈ 0.005.
4. *Correction for the number of genes.* Because one test is performed per gene, the p-values of the genes were adjusted with the Benjamini–Hochberg procedure, which controls the expected proportion of false positives among the genes declared predictable.
5. *Decision.* A gene was considered significant when its adjusted p-value was below 0.05.

The test was applied separately to the analysis on all the cells and to the analysis on the extreme cells.

**Extreme expression.** The analysis was repeated on the 100 cells with the lowest and the 100 cells with the highest expression of each gene. Because these cells are selected by their label, this analysis gives an upper bound and not a general estimate of performance.

**Contribution of each family of features.** To measure what each family of features predicts, the same procedure (same cells, same outer folds, nested cross-validation) was repeated with the features of one family only (size: mean area of the cell; shape; motility; overlap; neighbourhood), and with the combinations size + motility, size + shape, size + motility + shape, and all the families together. The significance of each of these models was computed with the permutation test described above, the p-values being adjusted for the number of genes separately for each set of features.

All random steps used a fixed seed.

## Association between features and gene expression

The association between each feature and the expression of each gene was measured by the Spearman correlation, on the cells with a pair passing the quality control. P-values were corrected for the number of feature–gene pairs by the Benjamini–Hochberg procedure, and correlations with a false discovery rate above 5 % were set to zero before hierarchical clustering. Two-dimensional embeddings of the cells were computed with UMAP, either from the standardised live-cell features or from the standardised gene expression.

## Performance of the prediction

The dataset contains 1,779 live cells from the two 24 h-activated conditions. 457 of them (240 in 3-24H and 217 in 4-24H) are tracked over the whole movie, matched to a smFISH cell with a pair passing the quality control (`ratio_qc` = 0.3), have at least 3 RNA per gene on average and are CD8 positive; they are the cells used by the models. The evaluations do not use the same number of cells, and are therefore reported separately. In all the tables, a star marks a model significantly better than chance (permutation test, p-value corrected for the number of genes below 0.05).

**Table 1 — Nested cross-validation on all the cells, with all the features.** In each of the five folds, the model is trained on 80 % of the cells below and tested on the remaining 20 %; the cells of both fields of view are mixed.

| Gene | Cells (low / high) | Balanced accuracy [95 % CI] | AUC [95 % CI] | Balanced accuracy with shuffled labels (mean) | Better than chance |
|---|---|---|---|---|---|
| CORO1A | 253 / 183 | 74.7 % [70.5–78.7] | 0.83 [0.78–0.86] | 49.7 % | * |
| ACTB | 177 / 273 | 72.3 % [67.9–76.4] | 0.79 [0.73–0.82] | 49.5 % | * |
| IL2RA | 280 / 172 | 71.9 % [67.9–76.0] | 0.78 [0.73–0.82] | 50.1 % | * |
| CCR7 | 331 / 108 | 65.6 % [60.0–70.9] | 0.77 [0.71–0.81] | 49.8 % | * |
| CD69 | 171 / 256 | 60.4 % [55.6–65.4] | 0.63 [0.57–0.68] | 50.5 % | * |
| GZMB | 366 / 85 | 58.8 % [53.2–64.3] | 0.62 [0.51–0.65] | 49.8 % | * |
| GBP4 | 51 / 384 | 55.4 % [49.1–62.0] | 0.60 [0.45–0.65] | 50.1 % | * |
| IRF8 | 360 / 91 | 54.7 % [49.6–59.7] | 0.58 [0.49–0.63] | 49.9 % | * |
| TNF | 393 / 57 | 54.6 % [49.8–60.1] | 0.58 [0.45–0.62] | 49.7 % | * |

**Table 2 — Nested cross-validation on the extreme cells.** For each gene, only the 100 cells with the lowest and the 100 cells with the highest expression are kept, among the cells of Table 1; in each fold the model is trained on 160 cells and tested on 40. Here the two classes are defined by rank and not by the threshold of the gene: the 100 lowest cells against the 100 highest. For a gene with fewer than 100 cells on one side of its threshold (GZMB, GBP4, IRF8, TNF), some of the cells of that class are therefore on the other side of the threshold used in Table 1.

| Gene | Cells (low / high) | Balanced accuracy [95 % CI] | AUC [95 % CI] | Balanced accuracy with shuffled labels (mean) | Better than chance |
|---|---|---|---|---|---|
| CORO1A | 100 / 100 | 85.0 % [80.3–89.8] | 0.90 [0.85–0.94] | 49.8 % | * |
| ACTB | 100 / 100 | 81.0 % [75.5–86.3] | 0.89 [0.84–0.93] | 50.1 % | * |
| IL2RA | 100 / 100 | 82.5 % [77.0–87.5] | 0.87 [0.82–0.92] | 50.3 % | * |
| CCR7 | 100 / 100 | 81.0 % [75.2–86.1] | 0.88 [0.83–0.92] | 50.1 % | * |
| CD69 | 100 / 100 | 70.0 % [63.2–76.1] | 0.74 [0.66–0.80] | 49.7 % | * |
| GZMB | 100 / 100 | 63.0 % [56.6–69.8] | 0.66 [0.58–0.73] | 49.9 % | * |
| GBP4 | 100 / 100 | 52.0 % [45.0–58.8] | 0.52 [0.42–0.58] | 49.9 % | no |
| IRF8 | 100 / 100 | 44.0 % [37.1–51.2] | 0.45 [0.36–0.53] | 50.3 % | no |
| TNF | 100 / 100 | 55.0 % [48.0–62.5] | 0.52 [0.43–0.61] | 49.9 % | no |

**Table 3 — Balanced accuracy with each family of features.** Same cells, folds and nested cross-validation as Table 1, with the features of one family only, then with combinations of families. Number of features in brackets. The column "All" is the model of Table 1. The 95 % confidence intervals are in `performances_feature_groups.csv`; they are about ± 5 points.

| Gene | Size (1) | Shape (86) | Motility (9) | Overlap (10) | Neighbourhood (28) | Size + motility (10) | Size + shape (87) | Size + motility + shape (96) | All (134) |
|---|---|---|---|---|---|---|---|---|---|
| CORO1A | 73.7 % * | 75.0 % * | 70.1 % * | 68.2 % * | 56.5 % | 77.4 % * | 76.6 % * | 75.8 % * | 74.7 % * |
| ACTB | 68.5 % * | 69.6 % * | 64.5 % * | 68.0 % * | 53.7 % | 71.0 % * | 73.4 % * | 71.7 % * | 72.3 % * |
| IL2RA | 69.3 % * | 71.1 % * | 66.0 % * | 65.1 % * | 52.8 % | 72.0 % * | 71.1 % * | 70.6 % * | 71.9 % * |
| CCR7 | 71.1 % * | 64.6 % * | 63.1 % * | 60.5 % * | 52.6 % | 71.9 % * | 64.7 % * | 64.5 % * | 65.6 % * |
| CD69 | 54.4 % | 58.9 % * | 57.5 % * | 58.5 % * | 50.4 % | 62.2 % * | 58.0 % * | 59.1 % * | 60.4 % * |
| GZMB | 68.6 % * | 56.8 % * | 58.6 % * | 58.1 % * | 50.8 % | 63.7 % * | 60.0 % * | 60.8 % * | 58.8 % * |
| GBP4 | 51.7 % | 53.0 % | 47.7 % | 46.6 % | 46.2 % | 48.6 % | 50.8 % | 57.3 % | 55.4 % * |
| IRF8 | 55.0 % | 48.2 % | 50.7 % | 52.9 % | 46.7 % | 56.7 % * | 49.4 % | 52.3 % | 54.6 % * |
| TNF | 52.3 % | 53.1 % | 53.2 % | 56.4 % * | 53.6 % | 51.5 % | 52.4 % | 55.3 % | 54.6 % * |

Balanced accuracy: mean of the true-positive and true-negative rates on the test cells. CI: confidence interval by bootstrap of the cells. Shuffled labels: mean balanced accuracy over 200 runs with shuffled labels.

**Conclusions.**

- **Three genes are predicted well above chance: CORO1A, ACTB and IL2RA.** Their balanced accuracy is between 72 and 75 %, with confidence intervals far from 50 %. It rises to 81–85 % when only the cells with the most extreme expression are considered.
- **CCR7, CD69 and GZMB are predicted moderately to weakly.** Their balanced accuracy is 66 %, 60 % and 59 %. With the extreme cells only, CCR7 reaches 81 %, CD69 70 % and GZMB 63 %.
- **GBP4, IRF8 and TNF are not predicted in practice.** Their balanced accuracy is 55 %, the lower bound of their confidence interval is at 50 %, and none of them is better than chance in the analysis on the extreme cells. They are marked as significant in Table 1 because the permutation test detects a difference from shuffled labels even when it is very small; a star means that the score is unlikely to be due to chance, not that the prediction is useful. The rare class is poorly recognised: 27 % of the high cells for IRF8, 19 % for TNF, and 28 % of the low cells for GBP4. With one family of features at a time, none of these genes is predicted better than chance, with the exception of TNF with the overlap features (56 %).
- **Most of the prediction is carried by cell size.** The mean area of the cell, used as the only feature, reaches 69 to 74 % for CORO1A, CCR7, IL2RA, GZMB and ACTB (Table 3), close to the model with all the 134 features (59 to 75 %), and it is the most important feature of the full model for these five genes. The number of transcripts of the best predicted genes increases with cell size (Spearman correlation between mean area and count of 0.52 to 0.66 for CORO1A, ACTB, IL2RA and CCR7). For GZMB and CCR7, the area alone is better than all the features together (69 % against 59 %, and 71 % against 66 %), and the area with the 86 shape features is also lower than the area alone (60 % against 69 %, and 65 % against 71 %): with about 450 cells, many weakly informative features cost more than they bring.
- **For a few genes, the best score is obtained with the size and one other family** (Table 3). The size with the motility gives the best model of CORO1A (77 %, against 74 % with the size alone; 10 features) and of CD69 (62 %, against 54 %), and the size with the shape gives the best model of ACTB (73 %, against 69 %). Using the size, the motility and the shape together does not improve on these models.
- **CD69 is the gene whose prediction does not come from size.** The area alone does not predict it (54 %, not better than chance), whereas the shape, the motility and the overlap features, each used alone, do (58 to 59 %), and the size and the motility together reach 62 %, its best score.
- **Motility alone predicts six genes better than chance** (CORO1A 70 %, IL2RA 66 %, ACTB 65 %, CCR7 63 %, GZMB 59 %, CD69 58 %). Because larger cells also move more, this score does not by itself separate the motility from the size; the comparison between the size alone and the size with the motility (Table 3) does.
- **The neighbourhood of the cell carries no information.** Used alone, its features are never better than chance (46 to 57 %).

## Software and code availability

Analyses were performed in Python 3.10 with Cellpose 4.1.1 (PyTorch 2.10), Big-FISH 0.6.2, NumPy 1.26, pandas 2.3, SciPy 1.12, scikit-image 0.25, scikit-learn 1.7, pyefd 1.7 and umap-learn 0.5. The code, the parameters used for the analysis (`config.yaml`) and the table of features and gene expression of each cell are available at https://github.com/solenebernard/IMAGENE.
