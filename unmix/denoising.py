# Copyright (c) 2026 Dan Fu@UW
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


def denoise_savgol(
    hypercube: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
) -> Dict[str, Any]:
    """
    Savitzky-Golay Spectral Smoothing Filter.

    Smooths noise along the spectral dimension (Z-axis) while preserving sharp
    spectral peaks and narrow linewidths via local polynomial fitting.

    Parameters
    ----------
    hypercube : (H, W, L) array
    window_length : int (odd, > polyorder)
    polyorder : int

    Returns
    -------
    dict with keys ``hypercube`` and ``info``.
    """
    from scipy.signal import savgol_filter

    H, W, L = hypercube.shape
    wl_len = min(window_length, L if L % 2 != 0 else L - 1)
    if wl_len < polyorder + 2:
        wl_len = polyorder + 2
        if wl_len % 2 == 0:
            wl_len += 1

    denoised = savgol_filter(hypercube.astype(np.float64), window_length=wl_len, polyorder=polyorder, axis=-1)

    return {
        'hypercube': denoised,
        'info': {
            'algorithm': 'Savitzky-Golay Spectral Smoothing',
            'window_length': wl_len,
            'polyorder': polyorder,
        },
    }


def denoise_tv3d(
    hypercube: np.ndarray,
    weight: float = 0.05,
    n_iter_max: int = 30,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    3D Total Variation denoising via Chambolle's dual projection algorithm.

    Solves  min_u  1/(2*weight) ||u - f||^2 + TV(u)
    using the fast dual projection method (Chambolle 2004) with Neumann
    boundary conditions.  Pre-allocates all working arrays so the inner
    loop performs zero heap allocations.

    Parameters
    ----------
    hypercube : (H, W, L) array
    weight : float — regularisation strength (higher = smoother, 0.05–2.0)
    n_iter_max : int — projection iterations (30 usually sufficient)
    progress_callback : callable, optional — called with int percent each iteration

    Returns
    -------
    dict with keys ``hypercube`` and ``info``.
    """
    H, W, L = hypercube.shape
    f = hypercube.astype(np.float64)

    fmin, fmax = f.min(), f.max()
    rng = fmax - fmin
    if rng == 0:
        return {'hypercube': hypercube.copy(),
                'info': {'algorithm': '3D Total Variation (Chambolle)', 'weight': weight}}

    f = (f - fmin) / rng
    f_over_w = f / weight
    tau = 1.0 / 12.0  # step size < 1/(4*ndim) guarantees convergence

    # Pre-allocate dual variables and working arrays (zero allocation inner loop)
    px = np.zeros_like(f)
    py = np.zeros_like(f)
    pz = np.zeros_like(f)
    div = np.empty_like(f)
    gx = np.empty_like(f)
    gy = np.empty_like(f)
    gz = np.empty_like(f)

    for it in range(n_iter_max):
        # div(p) via backward differences with Neumann boundary
        div[0, :, :] = px[0, :, :]
        np.subtract(px[1:, :, :], px[:-1, :, :], out=div[1:, :, :])
        div[:, 0, :] += py[:, 0, :]
        gy[:, 1:, :] = py[:, 1:, :] - py[:, :-1, :]
        div[:, 1:, :] += gy[:, 1:, :]
        div[:, :, 0] += pz[:, :, 0]
        gz[:, :, 1:] = pz[:, :, 1:] - pz[:, :, :-1]
        div[:, :, 1:] += gz[:, :, 1:]

        # div now becomes u = div(p) - f/weight
        div -= f_over_w

        # grad(u) via forward differences with Neumann boundary (zero at boundary)
        gx[:] = 0.0
        np.subtract(div[1:, :, :], div[:-1, :, :], out=gx[:-1, :, :])
        gy[:] = 0.0
        np.subtract(div[:, 1:, :], div[:, :-1, :], out=gy[:, :-1, :])
        gz[:] = 0.0
        np.subtract(div[:, :, 1:], div[:, :, :-1], out=gz[:, :, :-1])

        # p += tau * grad  (scale in place to avoid temporaries)
        gx *= tau; px += gx
        gy *= tau; py += gy
        gz *= tau; pz += gz

        # Project onto unit ball  |p| <= 1  (reuse g arrays for norm)
        np.multiply(px, px, out=gx)
        np.multiply(py, py, out=gy)
        gx += gy
        np.multiply(pz, pz, out=gy)
        gx += gy
        np.sqrt(gx, out=gx)
        np.maximum(gx, 1.0, out=gx)
        px /= gx
        py /= gx
        pz /= gx

        if progress_callback:
            progress_callback(int(100 * (it + 1) / n_iter_max))

    # Reconstruct u = f - weight * div(p*)
    div[0, :, :] = px[0, :, :]
    np.subtract(px[1:, :, :], px[:-1, :, :], out=div[1:, :, :])
    div[:, 0, :] += py[:, 0, :]
    gy[:, 1:, :] = py[:, 1:, :] - py[:, :-1, :]
    div[:, 1:, :] += gy[:, 1:, :]
    div[:, :, 0] += pz[:, :, 0]
    gz[:, :, 1:] = pz[:, :, 1:] - pz[:, :, :-1]
    div[:, :, 1:] += gz[:, :, 1:]

    denoised = (f - weight * div) * rng + fmin

    return {
        'hypercube': denoised,
        'info': {
            'algorithm': '3D Total Variation (Chambolle)',
            'weight': weight,
            'iterations': n_iter_max,
        },
    }


def denoise_bm4d(
    hypercube: np.ndarray,
    sigma: Optional[float] = None,
) -> Dict[str, Any]:
    """
    BM4D volumetric denoising, with spatial-spectral PCA fallback.

    When the ``bm4d`` package is installed, runs true Block-Matching 4D
    filtering.  Otherwise falls back to a PCA-subspace + spatial median
    filter that combines spectral dimensionality reduction with
    edge-preserving spatial smoothing — a proper spatial-spectral
    denoiser, unlike plain PCA truncation.

    Parameters
    ----------
    hypercube : (H, W, L) array
    sigma : float or None — noise standard deviation in data units.
        Auto-estimated from spatial differences when *None* (recommended).

    Returns
    -------
    dict with keys ``hypercube`` and ``info``.
    """
    from scipy.ndimage import median_filter as _median_filter

    H, W, L = hypercube.shape
    f = hypercube.astype(np.float64)
    fmin, fmax = float(f.min()), float(f.max())
    rng = fmax - fmin
    if rng == 0:
        return {'hypercube': hypercube.copy(),
                'info': {'algorithm': 'BM4D', 'sigma': 0.0}}

    # Auto-estimate noise level from horizontal spatial differences (MAD)
    noise_samples = np.diff(f, axis=1).ravel()
    sigma_est = float(np.median(np.abs(noise_samples)) / 0.6745) / np.sqrt(2.0)
    sigma_data = sigma if sigma is not None else sigma_est

    try:
        import bm4d as _bm4d

        cube_norm = (f - fmin) / rng
        sigma_norm = sigma_data / rng
        denoised_norm = _bm4d.bm4d(cube_norm, sigma_psd=sigma_norm)
        denoised = denoised_norm * rng + fmin

        return {
            'hypercube': denoised,
            'info': {
                'algorithm': 'BM4D',
                'sigma': sigma_data,
            },
        }

    except ImportError:
        # Spatial-spectral PCA denoising: PCA subspace + per-component
        # spatial median filter.  The median filter provides edge-preserving
        # spatial smoothing that plain PCA truncation completely lacks.
        data = f.reshape(-1, L)
        mean_spec = np.mean(data, axis=0)
        data_centered = data - mean_spec

        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)

        cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
        n_keep = max(1, int(np.searchsorted(cumvar, 0.99) + 1))
        n_keep = min(n_keep, L)

        scores = (U[:, :n_keep] * S[:n_keep]).reshape(H, W, n_keep)
        for k in range(n_keep):
            scores[:, :, k] = _median_filter(scores[:, :, k], size=3)

        denoised = (scores.reshape(-1, n_keep) @ Vt[:n_keep, :] + mean_spec).reshape(H, W, L)

        return {
            'hypercube': denoised,
            'info': {
                'algorithm': 'Spatial-Spectral PCA Denoising (BM4D fallback)',
                'n_components_kept': n_keep,
                'estimated_noise_std': sigma_est,
                'note': 'Install bm4d for true BM4D: pip install bm4d',
            },
        }


def denoise_wavelet(
    hypercube: np.ndarray,
    wavelet: str = 'db4',
    max_level: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fast Vectorized Wavelet Denoising.
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("Install PyWavelets for wavelet denoising: pip install PyWavelets")

    H, W, L = hypercube.shape
    if max_level is None:
        max_level = min(pywt.dwt_max_level(L, pywt.Wavelet(wavelet).dec_len), 3)

    X = hypercube.reshape(-1, L).astype(np.float64)  # (N, L)
    N = X.shape[0]

    # Vectorized 1D DWT along spectral axis
    coeffs = pywt.wavedec(X, wavelet, axis=1, level=max_level)
    sigma = np.median(np.abs(coeffs[-1]), axis=1, keepdims=True) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(L))

    coeffs_thresh = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode='soft')
        for c in coeffs[1:]
    ]
    rec = pywt.waverec(coeffs_thresh, wavelet, axis=1)
    denoised = rec[:, :L].reshape(H, W, L)

    return {
        'hypercube': denoised,
        'info': {
            'wavelet': wavelet,
            'level': max_level,
        },
    }
