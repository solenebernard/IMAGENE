'''
Main figure on the prediction of the gene expression, assembled from figures saved as PNG by other scripts:
    a) UMAP of the features of the live cells, coloured by four features            2_umap.py
    b) the same UMAP, coloured by the expression of each gene                         2_umap.py
    c) the same UMAP, each cell shown by its first BF frame                           2_umap.py
    d) UMAP of the gene expression, coloured by the expression of each gene           2_umap.py
    e) the same UMAP, coloured by four features of the live cells                     2_umap.py
    f) thresholds between low and high expression                                     1_gene_thresholds.py
    g) size of the dataset of each gene (top), performance of the models (bottom)     5_POC_activation/4_figures_performances.py
    h) performance with each group of features                                        5_POC_activation/6_figures_feature_groups.py
    i) trajectories and contours of the cells with a low and a high expression        3_contours_trajectories_grid.py
Run the scripts above first. Each panel is drawn by its script with the size of its cells of the grid (PANEL_UNIT of
tools/style.py), so that its text has the same size in the figure (see tools/figures.compose_grid).

Output: exp/results/figures/main/main_figure_prediction.png and .pdf, and a copy of the figure and of its panels in
exp/results/figures/main/FIGURE 1 EXP/
'''

from tools.figures import compose_grid, OUTPUT_DIR
from tools.read_yaml import RESULTS_PATH


def main():
    '''Draw and save the figures'''
    model = OUTPUT_DIR / 'model'
    main = OUTPUT_DIR / 'main'
    # Grid of square cells (tools/style.py, PANEL_UNIT): (letter, images, column, row, width, height)
    #   a b b c
    #   d d e c
    #   f g g i   (g: 1.7 cells, i: 1.3 cells)
    #   h h h h
    panels = [
        ('a', [model / 'umap_color_features.png'], 0, 0, 1, 1),
        ('b', [model / 'umap_color_genes.png'], 1, 0, 2, 1),
        ('c', [model / 'umap_thumbnails.png'], 3, 0, 1, 2, True),  # centred vertically
        ('d', [model / 'umap_genes.png'], 0, 1, 2, 1),
        ('e', [model / 'umap_genes_color_feat.png'], 2, 1, 1, 1),
        ('f', [model / 'thresholds.png'], 0, 2, 1, 1),
        ('g', [RESULTS_PATH + 'figures/dataset_size.png', RESULTS_PATH + 'figures/performance.png'], 1, 2, 1.7, 1),
        ('i', [model / 'contours_trajectories_grid.png'], 2.7, 2, 1.3, 1),
        ('h', [RESULTS_PATH + 'figures/performance_feature_groups.png'], 0, 3, 4, 1),
    ]
    compose_grid(panels, main / 'main_figure_prediction', panels_dir=main / 'FIGURE 1 EXP')


if __name__ == "__main__":
    main()
