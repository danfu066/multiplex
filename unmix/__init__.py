# Copyright (c) 2026 Dan Fu@UW
"""
unmix — Hyperspectral unmixing, denoising, and classification algorithms.

All functions accept a hypercube of shape (H, W, L) and optional basis/params,
returning a standardised result dict with no side effects or UI dependencies.

Modules
-------
denoising       MPPCA, Wavelet (VisuShrink), Savitzky-Golay, TV3D, BM4D
unmixing        MCR-ALS, NMF, MESMA, NNLS, FCLS
classification  PCA, MNF, SAM, SID, RX
"""

from .denoising import denoise_mppca, denoise_wavelet, denoise_savgol, denoise_tv3d, denoise_bm4d
from .unmixing import run_mcr_als, run_nmf, run_mesma, run_linear_unmix, run_fcls
from .classification import run_pca, run_mnf, run_sam, run_sid, run_rx
from .endmember import run_nfindr, run_vca
from .fitting import fit_spectrum

__all__ = [
    # Denoising
    'denoise_mppca',
    'denoise_wavelet',
    'denoise_savgol',
    'denoise_tv3d',
    'denoise_bm4d',
    # Unmixing (blind source separation)
    'run_mcr_als',
    'run_nmf',
    'run_mesma',
    'run_linear_unmix',
    'run_fcls',
    # Classification / Decomposition
    'run_pca',
    'run_mnf',
    'run_sam',
    'run_sid',
    'run_rx',
    # Endmember extraction
    'run_nfindr',
    'run_vca',
    # Spectral fitting
    'fit_spectrum',
]
