'''
Illustration of the Fourier features of the contour of a cell (step 4.1)
- fourier_features_cell_{id}: mask of the cell, reconstruction of its contour with an increasing number
  of harmonics, and magnitude and elliptic area of each harmonic

Input: tile masks and tracks of step 1
'''

import cv2
import pyefd

from tools.figures import *

params = FIG['fourier']
plot_order = params['orders']
size_im = params['size']
n_points = 256


def get_contour(mask):
    '''Contour (x,y) of the largest region of the mask, starting at the angle 0 from the centroid'''
    contours, _ = cv2.findContours((mask>0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(float)
    cx, cy = contour.mean(axis=0)
    angles = np.arctan2(contour[:,1] - cy, contour[:,0] - cx)
    return np.roll(contour, -np.argmin(np.abs(angles)), axis=0)


def elliptic_fourier_descriptors_angle(contour, order=10):
    '''Elliptic Fourier coefficients [a_n, b_n, c_n, d_n] for n=1..order, with the FFT of the contour'''
    N = contour.shape[0]
    X = np.fft.fft(contour[:,0]) / N
    Y = np.fft.fft(contour[:,1]) / N
    coeffs = [[2*np.real(X[n]), -2*np.imag(X[n]), 2*np.real(Y[n]), -2*np.imag(Y[n])] for n in range(1, order+1)]
    return np.array(coeffs)


def get_features_efd(efd):
    '''Magnitude and elliptic area of each harmonic, from the coefficients [a_n, b_n, c_n, d_n]'''
    a, b, c, d = efd[:,0], efd[:,1], efd[:,2], efd[:,3]
    term1 = a**2 + b**2 + c**2 + d**2
    term2 = np.sqrt((a**2 + b**2 - c**2 - d**2)**2 + 4*(a*c + b*d)**2)
    # Semi-axes of the ellipse of each harmonic
    lambda1 = np.sqrt(0.5 * (term1 + term2))
    lambda2 = np.sqrt(0.5 * np.clip(term1 - term2, a_min=0, a_max=None))
    return np.sqrt(term1), np.pi * lambda1 * lambda2


def crop_cell(mask, id_cell, size_im):
    '''Binary image of size size_im centered on the cell'''
    x, y = np.where(mask == id_cell)
    sub_mask = np.zeros((size_im, size_im), dtype=bool)
    sub_mask[x - (x.min()+x.max())//2 + size_im//2, y - (y.min()+y.max())//2 + size_im//2] = True
    return sub_mask


mask = load_tile_masks(FIG_CONDITION, params['tile'], list_t=[params['frame']])[0]
list_cells = params['cells']
if list_cells is None:
    list_cells = np.unique(mask[mask>0])[params['first_cell']:params['first_cell']+params['n_cells']]

for id_cell in list_cells:
    sub_mask = crop_cell(mask, id_cell, size_im)
    contour = get_contour(sub_mask)

    # Reconstruction of the contour with an increasing number of harmonics
    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=max(plot_order))
    locus = pyefd.calculate_dc_coefficients(contour)
    contours = [pyefd.reconstruct_contour(coeffs[:order], locus, n_points) for order in plot_order]
    # Magnitude and elliptic area of each harmonic
    M, A = get_features_efd(elliptic_fourier_descriptors_angle(contour, order=max(plot_order)))

    fig, axes = plt.subplots(1, len(plot_order)+2, figsize=((len(plot_order)+3)*2, 3))
    axes[0].imshow(sub_mask, cmap='gray', interpolation=None, origin='lower')
    axes[0].set_title('Cell mask')
    for ax, c, order in zip(axes[1:], contours, plot_order):
        ax.plot(contour[:,0], contour[:,1], color='black', linestyle='--', alpha=0.5)
        ax.plot(c[:,0], c[:,1])
        ax.set_title(f'Reconstruction with k={order}')
    for ax in axes[:-1]:
        ax.set_xlim(0, size_im)
        ax.set_ylim(0, size_im)
        ax.axis('off')
        ax.set_aspect('equal')
    harmonics = np.arange(1, max(plot_order)+1)
    axes[-1].bar(harmonics-0.2, M, width=0.4, label='Magnitude')
    axes[-1].bar(harmonics+0.2, A, width=0.4, label='Elliptic area')
    axes[-1].legend()
    axes[-1].set_xlabel('Harmonic k ')
    axes[-1].set_yscale('log')
    axes[-1].set_xticks(harmonics)
    axes[-1].set_title('Elliptic descriptors')
    fig.tight_layout()
    save_figure(fig, f'fourier/fourier_features_cell_{id_cell}')
