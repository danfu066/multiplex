"""
unmix — Hyperspectral unmixing, denoising, and classification algorithms.

All functions accept a hypercube of shape (H, W, L) and optional basis/params,
returning a standardised result dict with no side effects or UI dependencies.

Modules
-------
denoising       MPPCA, Wavelet (VisuShrink)
unmixing        MCR-ALS, NMF, MESMA
classification  PCA, SAM, SID, SVR
"""

from .denoising import denoise_mppca, denoise_wavelet
from .unmixing import run_mcr_als, run_nmf, run_mesma
from .classification import run_pca, run_sam, run_sid, run_svr
from .endmember import run_nfindr, run_vca
from .fitting import fit_spectrum

__all__ = [
    # Denoising
    'denoise_mppca',
    'denoise_wavelet',
    # Unmixing (blind source separation)
    'run_mcr_als',
    'run_nmf',
    'run_mesma',
    # Classification / Decomposition
    'run_pca',
    'run_sam',
    'run_sid',
    'run_svr',
    # Endmember extraction
    'run_nfindr',
    'run_vca',
    # Spectral fitting
    'fit_spectrum',
]
