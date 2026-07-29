"""
Denoising algorithms for hyperspectral cubes.

All functions accept a hypercube of shape (H, W, L) and return a
denoised hypercube of the same shape.  No side effects, no UI deps.
"""

from typing import Dict, Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Standardised result dict
# ---------------------------------------------------------------------------
#   {
#       'hypercube': np.ndarray,      # denoised cube (H, W, L)
#       'info': Dict[str, Any],        # human-readable summary
#   }
# ---------------------------------------------------------------------------


def denoise_mppca(
    hypercube: np.ndarray,
    variance_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Multiplicative PCA denoising via truncated SVD.

    Unfolds the hypercube to (H*W, L), performs SVD, keeps only
    components explaining >= *variance_threshold* cumulative variance,
    then refolds.

    Parameters
    ----------
    hypercube : (H, W, L) array
    variance_threshold : float in (0, 1)
        Cumulative explained-variance cutoff (default 0.95).

    Returns
    -------
    dict with keys ``hypercube`` (denoised cube) and ``info``.
    """
    H, W, L = hypercube.shape
    data = hypercube.reshape(-1, L).astype(np.float64)

    # Mean-centre
    mean_spec = np.mean(data, axis=0)
    data_centered = data - mean_spec

    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)

    explained = (S ** 2) / np.sum(S ** 2)
    cumvar = np.cumsum(explained)

    n_keep = np.searchsorted(cumvar, variance_threshold) + 1
    n_keep = max(n_keep, 1)

    data_denoised = U[:, :n_keep] @ np.diag(S[:n_keep]) @ Vt[:n_keep, :]
    data_denoised += mean_spec

    return {
        'hypercube': data_denoised.reshape(H, W, L),
        'info': {
            'n_components_total': len(S),
            'n_components_kept': n_keep,
            'cumulative_variance': float(cumvar[n_keep - 1]),
        },
    }


def denoise_wavelet(
    hypercube: np.ndarray,
    wavelet: str = 'db4',
    max_level: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Wavelet denoising via universal (VisuShrink) thresholding.

    Applies a 1-D DWT along the spectral axis, decomposes to *max_level*,
    applies universal threshold (sigma * sqrt(2 * log(N))) to detail
    coefficients, then reconstructs.  Sigma estimated via MAD of
    finest-level details.

    Parameters
    ----------
    hypercube : (H, W, L) array
    wavelet : str
        PyWavelets wavelet name (default 'db4').
    max_level : int or None
        Decomposition level; ``None`` → min(auto, 4).

    Returns
    -------
    dict with keys ``hypercube`` and ``info``.

    Raises
    ------
    ImportError
        If PyWavelets is not installed.
    """
    try:
        import pywt
    except ImportError:
        raise ImportError(
            "Install PyWavelets for wavelet denoising: "
            "pip install PyWavelets")

    H, W, L = hypercube.shape
    if max_level is None:
        max_level = min(
            pywt.dwt_max_level(L, pywt.Wavelet(wavelet).dec_len), 4)

    denoised = np.zeros_like(hypercube, dtype=np.float64)

    for i in range(H):
        for j in range(W):
            coeffs = pywt.wavedec(
                hypercube[i, j, :].astype(np.float64),
                wavelet, level=max_level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(L))
            coeffs_thresh = [coeffs[0]] + [
                pywt.threshold(c, threshold, mode='soft')
                for c in coeffs[1:]]
            rec = pywt.waverec(coeffs_thresh, wavelet)
            denoised[i, j, :] = rec[:L]

    denoised = denoised[:, :, :L]

    return {
        'hypercube': denoised,
        'info': {
            'wavelet': wavelet,
            'level': max_level,
        },
    }
