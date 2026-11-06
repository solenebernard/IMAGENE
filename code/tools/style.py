'''
Font of all the figures, so that the text looks the same on every panel of the main figures.
The panels are drawn with these font sizes (in pt), then tools/figures.compose_figure scales all the panels of a main
figure by the same factor: the text has the same size on all the panels of the main figure.

Usage: from tools.style import FONTS, set_style; set_style() before drawing
'''

import matplotlib

# The first font of the list that is installed is used: install Arial on every computer that draws the panels,
# else the panels drawn on different computers may have different fonts
FONT_FAMILY = ['Arial', 'Helvetica', 'DejaVu Sans']
# base: default text, title: titles, label: axis labels, tick: tick labels, legend: legends,
# annotation: values written on the plots, symbol: stars of significance.
# The main figures scale their panels by 0.7 (7 pt for the base font): the titles are drawn at 14.3 pt, 10 pt in the figures
FONTS = {'base': 10, 'title': 14.3, 'label': 10, 'tick': 9, 'legend': 9, 'annotation': 8, 'symbol': 12}
# Height (inches) of the panels of the main figure of the prediction (6_exp_figures/4_main_figure_prediction.py), as
# drawn by their scripts: each panel is 1 to 4 units wide and 1 unit high (2 units for the UMAP of the thumbnails)
PANEL_UNIT = 4.3


def set_style():
    '''Sets the font family and the font sizes of FONTS as the defaults of matplotlib'''
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': FONT_FAMILY,
                                'font.size': FONTS['base'], 'axes.titlesize': FONTS['title'],
                                'axes.labelsize': FONTS['label'], 'xtick.labelsize': FONTS['tick'],
                                'ytick.labelsize': FONTS['tick'], 'legend.fontsize': FONTS['legend'],
                                'legend.title_fontsize': FONTS['legend'], 'figure.titlesize': FONTS['title'],
                                'figure.labelsize': FONTS['label'],
                                # Text kept as text in the PDF files
                                'pdf.fonttype': 42})
