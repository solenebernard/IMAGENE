'''
Shared tools for the figures (6_exp_figures/)
- parameters of the figures (6_exp_figures/config_figures.yaml)
- loading of the dataset and of the tables of step 4.1, contours and trajectories of the cells
- assembly of the main figure from the panels saved as PNG
- saving of the figures
'''

import matplotlib
matplotlib.use('Agg')  # The figures are saved, not shown
from sklearn.preprocessing import StandardScaler

# Import local tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import yaml
from PIL import Image
from tools.read_yaml import (
    EXP_PATH, FOURIER_ORDER, LEARNING_GENES, MIN_MEAN_RNA, N_FRAMES, RANDOM_STATE, RATIO_QC, RESULTS_PATH, ROOTPATH
)
from tools.matching import passes_ratio_qc_table

from tools.style import FONTS, PANEL_UNIT, set_style

# Same font on all the figures (tools/style.py)
set_style()
Image.MAX_IMAGE_PIXELS = 2000000000

# Parameters of the figures
with open("6_exp_figures/config_figures.yaml", "r") as f:
    FIG = yaml.safe_load(f)

FIG_CONDITION = FIG['condition']
DATA_PATH = ROOTPATH + EXP_PATH
# Folder with thumbnails/ (images of the cells of 2_umap), or None to read them from the single-cell movies of step 1.4
EXAMPLE_DIR = pathlib.Path(ROOTPATH + FIG['data_dir']) if FIG['data_dir'] is not None else None
# Folder of the figures
OUTPUT_DIR = pathlib.Path(ROOTPATH + FIG['output_dir'])


def save_figure(fig, name, formats=('pdf', 'png'), **kwargs):
    '''Save the figure in OUTPUT_DIR/name.pdf and .png, and close it'''
    path = OUTPUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # The text that goes beyond the figure is kept
    kwargs.setdefault('bbox_inches', 'tight')
    for fmt in formats:
        fig.savefig(f'{path}.{fmt}', format=fmt, dpi=300, **kwargs)
    plt.close(fig)
    print(f'Saved {path}')


##### DATASET

def load_dataset(ratio_qc=RATIO_QC):
    '''
    Table of features and gene expression of step 5 (exp/results/dataset.csv)
    The gene expression is kept only for the cells used by the models: passing the quality control of the
    matching (ratio_qc), not on the edge, present in all the time frames, and with enough RNA (the matched cells
    with almost no RNA in every gene, below learning.min_mean_rna, are removed)
    '''
    from tools.model import mean_rna_per_gene
    df = pd.read_csv(RESULTS_PATH + 'dataset.csv', index_col=['exp_path', 'condition', 'cell_bf'])
    bool_match = passes_ratio_qc_table(df, ratio_qc) & (df['is_on_edge'] == False) & (df['count'] == N_FRAMES) \
    & (mean_rna_per_gene(df) >= MIN_MEAN_RNA)
    df.loc[~bool_match, 'Fish'] = None
    for gene in LEARNING_GENES:
        df.loc[~bool_match, f'Gene_{gene}'] = None
    return df


def select_features(df):
    '''The features of the live cells used by the model (same selection as in step 5)'''
    columns_remove = [c for c in df.columns
                      if c.startswith(('Gene_', 'delta_z_', 'Fish', 'Distance', 'Id_2nd', 'is_on_edge', 'count',
                                       'BF_center_', 'traj_', 'centroid-', 'moments_', 'inertia_', 'umap'))]
    columns_remove += ['avg_speed', 'max_speed', 'speed_std', 'elongation', 'total_distance']
    return [x for x in df.columns if x not in columns_remove]


def cd8_positive(df):
    '''Index of the CD8 positive cells: at least two RNA of CD8'''
    return df.index[df['Gene_CD8'] > 1]


def add_umap_features(df, features, random_state=RANDOM_STATE):
    '''UMAP of the features of the live cells: columns umap_0 and umap_1'''
    import umap
    X = df[features].fillna(df[features].median())
    x_transform = umap.UMAP(n_components=2, random_state=random_state).fit_transform(StandardScaler().fit_transform(X))
    return pd.concat((df, pd.DataFrame(x_transform, columns=['umap_0', 'umap_1'], index=X.index)), axis=1)


def add_umap_genes(df, random_state=RANDOM_STATE):
    '''UMAP of the gene expression of the CD8 positive cells: columns umap2_0 and umap2_1'''
    import umap
    Y = df[[f'Gene_{gene}' for gene in LEARNING_GENES]].dropna(axis=0, how='any')
    Y = Y[Y['Gene_CD8'] > 1]
    x_transform = umap.UMAP(n_components=2, random_state=random_state).fit_transform(StandardScaler().fit_transform(Y))
    return pd.concat((df, pd.DataFrame(x_transform, columns=['umap2_0', 'umap2_1'], index=Y.index)), axis=1)


##### MAIN FIGURES: PANELS ASSEMBLED FROM THE FIGURES SAVED AS PNG

def save_through_local_file(fig, output_path, dpi):
    '''Saves the figure as output_path.png and .pdf, written first in a local temporary folder then copied: writing
    large files directly on a network folder can fail when the file is closed (e.g. when it is open in a viewer)'''
    import os
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        for fmt in ['png', 'pdf']:
            local = pathlib.Path(folder) / f'figure.{fmt}'
            fig.savefig(local, dpi=dpi)
            # Copied next to the destination, then renamed onto it (the destination is never opened for writing)
            destination = pathlib.Path(f'{output_path}.{fmt}')
            temporary = destination.with_name(destination.name + '.tmp')
            shutil.copyfile(local, temporary)
            os.replace(temporary, destination)


def copy_panels(output_path, panel_paths, panels_dir):
    '''Copies the figure output_path (.png and .pdf) and the files of its panels in panels_dir, each panel file
    preceded by its letter. panel_paths: list of (letter, image paths)'''
    import shutil
    panels_dir = pathlib.Path(panels_dir)
    panels_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ['png', 'pdf']:
        shutil.copyfile(f'{output_path}.{fmt}', panels_dir / f'{pathlib.Path(output_path).name}.{fmt}')
    for letter, paths in panel_paths:
        for path in map(pathlib.Path, paths):
            for source in [path, path.with_suffix('.pdf')]:
                if source.exists():
                    shutil.copyfile(source, panels_dir / (f'{letter}_{source.name}' if letter else source.name))
    print(f'Figure and panels copied in {panels_dir}')


def compose_grid(panels, output_path, font_size=7, dpi=600, letter_size=12, letter_space=0.18, panels_dir=None):
    '''
    Assembles figures saved as PNG into one figure on a grid of square cells, with a letter on each panel.
    panels: list of (letter, image paths stacked from top to bottom, column, row, width, height[, centre]), the position
    and the size in cells of the grid; centre=True centres the panel vertically in its cells, its letter just above it. A cell is PANEL_UNIT inches (tools/style.py) at the size the panels are drawn,
    multiplied by font_size / base font (10 pt): a panel drawn with the size of its cells has its text at font_size pt.
    Each panel keeps its proportions and is fitted in its cells; the letter_space inches above each row of cells hold
    the letters. A panel whose images are missing is left blank, with its letter.
    Saved as output_path.png and output_path.pdf; panels_dir: see copy_panels
    '''
    from matplotlib.image import imread
    unit = PANEL_UNIT * font_size / FONTS['base']
    n_cols = max(p[2] + p[4] for p in panels)
    n_rows = max(p[3] + p[5] for p in panels)
    width, height = n_cols * unit, n_rows * (unit + letter_space)
    fig = plt.figure(figsize=(width, height))
    for letter, paths, x, y, w, h, *centre in panels:
        left, top = x * unit, height - y * (unit + letter_space)
        images = []
        for path in paths:
            if pathlib.Path(path).exists():
                images.append((imread(path), Image.open(path).info.get('dpi', (dpi, dpi))[0]))
            else:
                print(f'Missing panel image: {path}')
        if not images:
            fig.text((left + 0.02) / width, (top - 0.01) / height, letter, fontsize=letter_size, fontweight='bold',
                     va='top', ha='left', zorder=10)
            continue
        # Size of the images at the size they were drawn (inches), stacked; fitted in the cells of the panel
        natural_width = max(im.shape[1] / d for im, d in images)
        natural_height = sum(im.shape[0] / d for im, d in images)
        box_width, box_height = w * unit, h * unit + (h - 1) * letter_space
        scale = min(box_width / natural_width, box_height / natural_height)
        print(f'Panel {letter}: scaled by {scale:.2f}, base font of {FONTS["base"] * scale:.1f} pt')
        if centre and centre[0]:
            # Centred vertically: the space left in the cells is shared above and below, the letter moves with it
            top -= (box_height - natural_height * scale) / 2
        fig.text((left + 0.02) / width, (top - 0.01) / height, letter, fontsize=letter_size, fontweight='bold',
                 va='top', ha='left', zorder=10)
        y_image = top - letter_space
        for image, d in images:
            image_width, image_height = image.shape[1] / d * scale, image.shape[0] / d * scale
            ax = fig.add_axes([left / width, (y_image - image_height) / height, image_width / width, image_height / height])
            ax.imshow(image)
            ax.axis('off')
            y_image -= image_height
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_through_local_file(fig, output_path, dpi)
    plt.close(fig)
    print(f'Saved {output_path} ({width:.1f} x {height:.1f} inches)')
    if panels_dir is not None:
        copy_panels(output_path, [(p[0], p[1]) for p in panels], panels_dir)


def compose_figure(rows, output_path, width=10, font_size=7, dpi=600, letter_size=12, letter_space=0.18, gap=0.1,
                   panels_dir=None):
    '''
    Assembles figures saved as PNG into one figure, with a letter on each panel.
    rows: list of rows. Each element of a row is a panel (letter, image paths stacked from top to bottom[, width]), or
    a list of panels stacked from top to bottom in one column.
    - a panel without width keeps its size (PNG pixels / dpi of the PNG) multiplied by a factor that is the same for
      all these panels: their text, drawn with the font sizes of tools/style.py, has the same size in the figure.
      The factor gives font_size pt to the base font (10 pt), or less if a row is wider than the figure.
      width=None: the figure is as wide as its widest row with this factor
    - a panel with a width (in inches of the figure) is scaled to this width: for the panels without text (images)
    - a panel whose list of images is empty, or whose images are missing, is left blank (to be completed by hand),
      with its letter, width inches wide (3 by default)
    The columns of a row are spaced evenly, with at least gap inches between them. The height of a row is the one of
    its tallest column; each panel has letter_space inches above it for its letter.
    Saved as output_path.png and output_path.pdf. panels_dir: folder where the figure and the files of its panels
    (PNG, and PDF when it exists) are also copied, the name of each panel file preceded by its letter
    '''
    from matplotlib.image import imread

    def load(paths):
        '''Images of the panel and their width in inches at their own dpi'''
        images = []
        for path in paths:
            if not pathlib.Path(path).exists():
                print(f'Missing panel image: {path}')
                continue
            dpi_image = Image.open(path).info.get('dpi', (dpi, dpi))[0]
            image = imread(path)
            images.append((image, image.shape[1] / dpi_image))
        return images

    # Each row: list of columns, each column: list of panels (letter, images, width or None)
    columns = [[[(panel[0], load(panel[1]), panel[2] if len(panel) > 2 else None)
                 for panel in (element if isinstance(element, list) else [element])] for element in row] for row in rows]

    def natural_width(column):
        '''Width of the column at its own size (panels without width), and the width it needs in any case'''
        natural = max([max(w for _, w in images) for _, images, fixed in column if images and fixed is None] + [0])
        fixed = max([fixed if fixed is not None else (0 if images else 3) for _, images, fixed in column] + [0])
        return natural, fixed

    # Common factor of the panels without width: font_size, or less to fit the widest row in the figure
    scale = font_size / FONTS['base']
    if width is None:
        width = max(sum(max(n * scale, f) for n, f in map(natural_width, row)) + gap * (len(row) - 1) for row in columns)
    for row in columns:
        widths = [natural_width(column) for column in row]
        natural = sum(n for n, _ in widths)
        if natural > 0:
            scale = min(scale, (width - sum(f for n, f in widths if n == 0) - gap * (len(row) - 1)) / natural)
    print(f'Panels scaled by {scale:.2f}: base font of {FONTS["base"] * scale:.1f} pt in a figure {width:.1f} inches wide')

    def panel_size(images, fixed):
        '''Width and height in inches of the panel in the figure (with its letter), and the width of its images'''
        if not images:
            return fixed if fixed is not None else 3, letter_space + width * 0.25, []
        widths = [fixed if fixed is not None else w * scale for _, w in images]
        return max(widths), letter_space + sum(im.shape[0] / im.shape[1] * w
                                               for (im, _), w in zip(images, widths)), widths

    sizes = [[[panel_size(images, fixed) for _, images, fixed in column] for column in row] for row in columns]
    heights = [max(sum(h for _, h, _ in column) for column in row) for row in sizes]
    total_height = sum(heights)
    fig = plt.figure(figsize=(width, total_height))
    y = total_height
    for row, row_sizes, height in zip(columns, sizes, heights):
        column_widths = [max(w for w, _, _ in column) for column in row_sizes]
        # Space left in the row, shared between the columns
        space = (width - sum(column_widths)) / (len(row) - 1) if len(row) > 1 else 0
        x = 0.0
        for column, column_sizes, column_width in zip(row, row_sizes, column_widths):
            y_panel = y
            for (letter, images, _), (_, panel_height, widths) in zip(column, column_sizes):
                y_image = y_panel - letter_space
                for (image, _), image_width in zip(images, widths):
                    image_height = image.shape[0] / image.shape[1] * image_width
                    ax = fig.add_axes([x / width, (y_image - image_height) / total_height,
                                       image_width / width, image_height / total_height])
                    ax.imshow(image)
                    ax.axis('off')
                    y_image -= image_height
                # Letter in the space above the images, drawn over them
                fig.text((x + 0.02) / width, (y_panel - 0.01) / total_height, letter, fontsize=letter_size,
                         fontweight='bold', va='top', ha='left', zorder=10)
                y_panel -= panel_height
            x += column_width + space
        y -= height
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_through_local_file(fig, output_path, dpi)
    plt.close(fig)
    print(f'Saved {output_path}')
    if panels_dir is not None:
        copy_panels(output_path, [(p[0], p[1]) for row in rows
                                  for element in row for p in (element if isinstance(element, list) else [element])],
                    panels_dir)


##### FEATURES OF STEP 4.1, CONTOURS AND TRAJECTORIES

def load_features(name, conditions):
    '''
    Table of features of step 4.1 (name: file name with {condition}, e.g. 'fourier_finetune_{condition}.csv') for
    the conditions that are available, indexed by (condition, cell_bf, t)
    '''
    tables = {}
    for condition in conditions:
        path = pathlib.Path(DATA_PATH + 'live_cell_features/' + name.format(condition=condition))
        if path.exists():
            tables[condition] = pd.read_csv(path, index_col=['cell_bf', 't'])
    return pd.concat(tables, names=['condition']).sort_index()


def cell_contour(df_fourier, condition, cell_bf, n_points=128):
    '''Contour (n_points, 2) of the cell at its first time frame, centred, reconstructed from its Fourier coefficients'''
    import pyefd
    coeffs = np.asarray(df_fourier.loc[(condition, cell_bf)].iloc[0]).reshape((FOURIER_ORDER, 4))
    return pyefd.reconstruct_contour(coeffs, np.array([0, 0]), n_points)


def cell_trajectory(df_scikit, condition, cell_bf):
    '''Trajectory of the centroid of the cell (columns centroid-0, centroid-1), starting from the origin'''
    traj = df_scikit.loc[(condition, cell_bf), ['centroid-0', 'centroid-1']]
    return traj - traj.iloc[0]
