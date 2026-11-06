



# IMAGENE

Code and data accompanying the IMAGENE study: predicting the gene expression of single T cells from their behaviour in label-free live imaging.

Watch the demo:

https://github.com/user-attachments/assets/246bf983-aec8-4e2e-9442-9fccf3b8a0b1

One field of view through the method. The live cells are filmed in brightfield and tracked, with one contour per cell. They are then immobilised and fixed, and imaged again: DAPI staining of the nuclei, then smFISH of each gene (here, the projection of the ACTB channel). Finally, each live cell is matched to its fixed counterpart, which gives for each cell both its behaviour in the movie and its gene expression.

[▶ Watch the control experiment (32 s)](demo_control.mp4): the accuracy of this pairing is measured with cells that carry a colour barcode (two fluorescent dyes). The movie follows the same cells, with one identifier and the colour of their barcode each, through the barcode read before the live imaging, the live-cell timelapse, the cells immobilised by photoactivation (up to 55 min later), and the barcode read again after fixation. It ends with the pairs of live and fixed cells passing the quality control: a correct pair has the same barcode on both sides.

Each cell is filmed in brightfield (BF), then fixed and imaged by sequential smFISH on the same field of view. The pipeline segments the cells in the BF movie with a finetuned Cellpose model and tracks them over time, counts the RNA spots of each gene in the smFISH images, pairs every live cell with its smFISH counterpart, extracts shape, motility and neighbourhood features from the movie, and trains random forests that predict the expression of each gene from these features. A control experiment, in which each cell carries a colour barcode read before the movie and after fixation, measures the accuracy of the pairing of live and fixed cells.

The methods and the results are described in [`code/method.md`](code/method.md).

## Repository layout

The code and the data are in separate folders:

```
README.md, demo.mp4, demo_control.mp4
code/                     the code and its parameters only
exp/                      main experiment: inputs of step 5 and results
├── Fish/                 number of RNA of each gene in each fixed cell (step 2): {CONDITION}.csv
├── alignment/            pairs of live and fixed cells, with their distances (step 3): {CONDITION}-alignment.csv
├── live_cell_features/   features of each live cell at each time frame (step 4)
├── thumbnails/           first frame of the movie of each cell, for the UMAP with the cell images
└── results/              tables of step 5 (dataset.csv, all_gene_profile.csv, performances_*.csv) and figures/
control/                  control experiment: inputs of its main figure and results
├── figures_input/        barcodes, tracks, pairs and masks of the cells, for the two densities of cells
├── alignment/            alignment of the movie on the barcode after fixation
├── BarcodeAfterUV/       mask of the fixed cells
└── results/              figures/
```

The experiment contains two conditions activated for 24 hours (`3-24H`, `4-24H`), each one field of view, and a non-activated condition (`1-NA`) used for the thresholds of the genes.

Steps 1 to 4, and the analysis of the control experiment (`code/run_control.sh`), are provided for information and cannot be reproduced yet, as they need the raw images and the finetuned Cellpose model; everything from step 5, and the two main figures, can be reproduced from the tables included in `exp/` and `control/`.

## Installation

Python 3.10, with the versions of the libraries used for the paper (`code/requirements.txt`):

```bash
conda create -n imagene python=3.10
conda activate imagene
bash code/install.sh
```

`install.sh` installs the libraries and checks that each one can be imported.

## Reproducing the results and the figures

All the commands are run from the folder `code/`:

```bash
cd code
bash run.sh
```

`run.sh` runs step 5 (about 1 h 15 min on 16 cores), then the two main figures. Each step can be switched off at the top of the file. The figures alone, from the tables of `exp/results/`, take about 1 min 30 s:

```bash
bash 6_exp_figures/run.sh       # exp/results/figures/main/main_figure_prediction.png and .pdf
bash 7_control_figures/run.sh   # control/results/figures/main_figure_control.png and .pdf
```

Step 5, script by script:

| Command | Output in `exp/results/` | Time (16 cores) |
|---|---|---|
| `python3 -m 5_POC_activation.1_concatenate_features` | `dataset.csv`: one row per live cell, with its features and its gene expression | seconds |
| `python3 -m 5_POC_activation.2_concatenate_gene_expr` | `all_gene_profile.csv`: gene expression of every fixed cell | seconds |
| `python3 -m 5_POC_activation.3_model` | `performances_24H.csv`, `performances_24H_extreme.csv` (Tables 1 and 2 of `method.md`), `prediction_{gene}.csv`, `feature_importances.csv` | 15 min |
| `python3 -m 5_POC_activation.4_figures_performances` | `figures/dataset_size.png`, `figures/performance.png` | seconds |
| `python3 -m 5_POC_activation.5_model_feature_groups` | `performances_feature_groups.csv` (Table 3) | 1 h |
| `python3 -m 5_POC_activation.6_figures_feature_groups` | `figures/performance_feature_groups.png` | seconds |

`--quick` (for `3_model` and `5_model_feature_groups`) runs with fewer combinations of hyperparameters, shuffles and resamplings, to check the installation in a few minutes. Its tables are saved in `exp/results/quick/`, so that the results of the paper are kept, and its p-values are not meaningful.

All the random steps use a fixed seed (`learning.random_state` in `code/config.yaml`), so running step 5 again gives the tables of `exp/results/`.

## Code

```
code/
├── config.yaml                   all the parameters of the pipeline, and the paths of exp/ and control/
├── run.sh, install.sh, requirements.txt
├── run_control.sh                analysis of the control experiment with the scripts of steps 1 and 3 (for information)
├── method.md                     methods and results
├── details.md                    each step in detail: scripts, inputs and outputs, definition of the features, result files
├── 1_bf_analysis/ … 5_POC_activation/   the steps of the pipeline (one run.sh per folder)
├── 6_exp_figures/                main figure of the prediction (config_figures.yaml: its parameters)
├── 7_control_figures/            main figure of the control experiment; analysis/ and pipeline/: preparation of its inputs
├── config_patterns/              layouts of the smFISH tiles
└── tools/                        shared code (parameters, models, figures)
```

The detail of each step, of each feature and of each result file is in [`code/details.md`](code/details.md).

## Configuration

All the parameters are in `code/config.yaml`; no value has to be changed in the scripts. The paths are given from `rootpath` (`../`, the folder above `code/`): `experiment.exp_path` (`exp/`), `learning.results_path` (`exp/results/`) and `control.exp_path` (`control/`). To use other paths without changing this file, copy it to `code/config_confidential.yaml` (not tracked by git): this copy is then read instead. The parameters of the figures are in `code/6_exp_figures/config_figures.yaml`.

Steps 1 to 4 process one condition at a time: the condition is `experiment.condition` of `config.yaml`, or the shell variable `CONDITION` (`CONDITION=4-24H python3 -m 1_bf_analysis.2_merge_masks`). A parameter can be replaced for one run with `CONFIG_SET="section.parameter=value"`. The layout of the raw data that they read, and their manual steps, are described in `code/details.md`.

## Data availability

The raw images and the finetuned Cellpose model are available upon request.

## Citation

If you use this code or these data, please cite:

Gariboldi, M. I., Sturmach, C., Bernard, S., Weber, C., Gourves, M., Umeda, Y., Li, X., Yamahira, S., Fernandes, J., Saez-Cirion, A., Yamaguchi, S., Muller, F. (2026). IMAGENE: Single-cell association of live cell imaging and gene expression profiles of non-adherent cells through photoactivatable adhesives. *bioRxiv*. https://doi.org/10.64898/2026.01.20.700510

```bibtex
@article{gariboldi2026imagene,
  title   = {IMAGENE: Single-cell association of live cell imaging and gene expression profiles of non-adherent cells through photoactivatable adhesives},
  author  = {Gariboldi, M. I. and Sturmach, C. and Bernard, S. and Weber, C. and Gourves, M. and Umeda, Y. and Li, X. and Yamahira, S. and Fernandes, J. and Saez-Cirion, A. and Yamaguchi, S. and Muller, F.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.01.20.700510}
}
```

## Use of a large language model

Claude (Anthropic; model: Claude Opus 5.5, accessed via Claude Code) was used between September and November 2026 to reformat existing code, to add or improve code comments, and to make minor refactorings (handling of the configuration, packaging of the dependencies) that do not change the methods or the results. It was also consulted on the statistical analysis and suggested some of the procedures used, such as the correction for multiple testing (Benjamini–Hochberg) and the bootstrap confidence intervals. The authors take full responsibility for the code, the analyses and the results.
