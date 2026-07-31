# Copyright (c) 2026 Dan Fu@UW
"""
Classification, decomposition, and anomaly detection algorithms.

PCA / MNF return decomposition results (scores + loadings).
SAM, SID return classification / abundance results.
RX returns anomaly detection scores.

All functions return a standardised result dict with no side effects.
"""

from typing import Dict, Any, Optional, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _compute_r2_residuals(
    data: np.ndarray,
    reconstructed: np.ndarray,
    shape_2d: tuple,
) -> tuple:
    """
    Compute per-pixel R² and RMS residual.

    Parameters
    ----------
    data : (n_pixels, L)
    reconstructed : (n_pixels, L)
    shape_2d : (H, W)

    Returns
    -------
    (r_squared, residuals) both (H, W)
    """
    L = data.shape[1]
    residuals_flat = data - reconstructed
    ss_res = np.sum(residuals_flat ** 2, axis=1)
    ss_tot = np.sum(
        (data - np.mean(data, axis=1, keepdims=True)) ** 2, axis=1)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    r2 = np.clip(r2, 0.0, 1.0)
    return (
        r2.reshape(shape_2d),
        np.sqrt(ss_res / L).reshape(shape_2d),
    )


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(
    hypercube: np.ndarray,
    n_components: int,
) -> Dict[str, Any]:
    """
    Principal Component Analysis via SVD.

    Not true unmixing — reveals dominant spectral modes and intrinsic
    dimensionality of the data.

    Parameters
    ----------
    hypercube : (H, W, L)
    n_components : int — number of principal components to retain.

    Returns
    -------
    dict with keys ``concentrations`` (PC scores), ``basis_spectra``
    (PC loadings), ``r_squared``, ``residuals``, ``info``.
    """
    H, W, L = hypercube.shape
    n_pixels = H * W

    if n_components >= L:
        raise ValueError(
            f"n_components ({n_components}) must be < n_bands ({L})")

    data = hypercube.reshape(-1, L).astype(np.float64)
    mean_spec = np.mean(data, axis=0)
    data_centered = data - mean_spec

    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)

    C = U[:, :n_components] * S[:n_components]
    loadings = Vt[:n_components, :]

    explained = (S ** 2) / np.sum(S ** 2)
    cumvar = np.cumsum(explained)

    reconstructed = C @ loadings + mean_spec
    r2, resid = _compute_r2_residuals(data, reconstructed, (H, W))

    return {
        'concentrations': C.reshape(H, W, n_components),
        'basis_spectra': loadings,
        'r_squared': r2,
        'residuals': resid,
        'info': {
            'n_components': n_components,
            'n_pixels': n_pixels,
            'explained_variance': explained[:n_components].tolist(),
            'cumulative_variance': cumvar[:n_components].tolist(),
        },
    }


# ---------------------------------------------------------------------------
# MNF — Minimum Noise Fraction
# ---------------------------------------------------------------------------

def run_mnf(
    hypercube: np.ndarray,
    n_components: int,
) -> Dict[str, Any]:
    """
    Minimum Noise Fraction (MNF) transform.

    Orders components by signal-to-noise ratio rather than variance,
    making it superior to PCA for noisy hyperspectral data.  Noise is
    estimated from spatial first-differences.

    Parameters
    ----------
    hypercube : (H, W, L)
    n_components : int — number of MNF components to retain.

    Returns
    -------
    dict with keys ``concentrations`` (MNF scores), ``basis_spectra``
    (MNF loadings), ``r_squared``, ``residuals``, ``info``.
    """
    from scipy.linalg import eigh

    H, W, L = hypercube.shape
    n_pixels = H * W

    if n_components >= L:
        raise ValueError(
            f"n_components ({n_components}) must be < n_bands ({L})")

    data = hypercube.reshape(-1, L).astype(np.float64)
    mean_spec = np.mean(data, axis=0)
    data_centered = data - mean_spec

    # Noise covariance estimated from horizontal spatial differences
    noise_h = np.diff(hypercube, axis=1).reshape(-1, L)
    noise_cov = (noise_h.T @ noise_h) / (2.0 * noise_h.shape[0])

    data_cov = (data_centered.T @ data_centered) / n_pixels

    # Regularize noise covariance for numerical stability
    noise_cov += np.eye(L) * 1e-10

    eigenvalues, eigenvectors = eigh(data_cov, noise_cov)

    # Sort by decreasing eigenvalue (highest SNR first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings = eigenvectors[:, :n_components].T  # (n_components, L)
    scores = data_centered @ eigenvectors[:, :n_components]  # (N, n_components)

    reconstructed = scores @ loadings + mean_spec
    r2, resid = _compute_r2_residuals(data, reconstructed, (H, W))

    return {
        'concentrations': scores.reshape(H, W, n_components),
        'basis_spectra': loadings,
        'r_squared': r2,
        'residuals': resid,
        'info': {
            'algorithm': 'MNF',
            'n_components': n_components,
            'n_pixels': n_pixels,
            'snr_eigenvalues': eigenvalues[:n_components].tolist(),
        },
    }


# ---------------------------------------------------------------------------
# SAM — Spectral Angle Mapper
# ---------------------------------------------------------------------------

def run_sam(
    hypercube: np.ndarray,
    basis_spectra: np.ndarray,
) -> Dict[str, Any]:
    """
    Spectral Angle Mapper.

    Computes the spectral angle between each pixel and all reference
    spectra.  Classifies each pixel to the reference with the smallest
    angle.  Illumination-invariant.

    Parameters
    ----------
    hypercube : (H, W, L)
    basis_spectra : (K, L) — reference endmember spectra.

    Returns
    -------
    dict with keys ``concentrations`` (one-hot classification map),
    ``r_squared`` (normalised inverse angle), ``residuals`` (min angle),
    ``classification`` (integer class map), ``info``.
    """
    H, W, L = hypercube.shape
    n_endmembers = basis_spectra.shape[0]

    data = hypercube.reshape(-1, L).astype(np.float64)
    basis = basis_spectra.astype(np.float64)

    data_norm = np.linalg.norm(data, axis=1, keepdims=True)
    basis_norm = np.linalg.norm(basis, axis=1, keepdims=True)
    data_norm[data_norm == 0] = 1e-10
    basis_norm[basis_norm == 0] = 1e-10

    cos_sim = (data @ basis.T) / (data_norm @ basis_norm.T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angles = np.arccos(cos_sim)

    best_match = np.argmin(angles, axis=1)
    min_angles = np.min(angles, axis=1)

    class_map = best_match.reshape(H, W)

    max_angle = np.max(min_angles)
    if max_angle > 0:
        abundances = 1.0 - min_angles / max_angle
    else:
        abundances = np.ones(H * W)
    abundances = abundances.reshape(H, W)

    C = np.zeros((H, W, n_endmembers))
    for k in range(n_endmembers):
        C[:, :, k] = (class_map == k).astype(float)

    return {
        'concentrations': C,
        'r_squared': abundances,
        'residuals': min_angles.reshape(H, W) / np.pi,
        'classification': class_map,
        'info': {
            'n_endmembers': n_endmembers,
            'n_pixels': H * W,
            'mean_angle_deg': float(np.mean(min_angles) * 180 / np.pi),
            'median_angle_deg': float(np.median(min_angles) * 180 / np.pi),
        },
    }


# ---------------------------------------------------------------------------
# SID — Spectral Information Divergence
# ---------------------------------------------------------------------------

def run_sid(
    hypercube: np.ndarray,
    basis_spectra: np.ndarray,
) -> Dict[str, Any]:
    """
    Spectral Information Divergence.

    Computes the symmetric KL-divergence between each pixel and all
    reference spectra.  Better for non-linear effects than SAM.

    Parameters
    ----------
    hypercube : (H, W, L)
    basis_spectra : (K, L) — reference endmember spectra.

    Returns
    -------
    dict with keys ``concentrations`` (one-hot), ``r_squared``,
    ``residuals`` (min SID), ``classification``, ``info``.
    """
    H, W, L = hypercube.shape
    n_endmembers = basis_spectra.shape[0]

    data = hypercube.reshape(-1, L).astype(np.float64)
    basis = basis_spectra.astype(np.float64)

    data_sum = np.sum(data, axis=1, keepdims=True)
    basis_sum = np.sum(basis, axis=1, keepdims=True)
    data_sum[data_sum == 0] = 1e-10
    basis_sum[basis_sum == 0] = 1e-10

    data_pdf = data / data_sum
    basis_pdf = basis / basis_sum

    eps = 1e-10
    data_pdf = np.maximum(data_pdf, eps)
    basis_pdf = np.maximum(basis_pdf, eps)

    kl_div = np.zeros((H * W, n_endmembers))

    for k in range(n_endmembers):
        p = data_pdf
        q = basis_pdf[k, :]
        d_pq = np.sum(p * np.log(p / q), axis=1)
        d_qp = np.sum(q * np.log(q / p), axis=1)
        kl_div[:, k] = d_pq + d_qp

    best_match = np.argmin(kl_div, axis=1)
    min_sid = np.min(kl_div, axis=1)

    class_map = best_match.reshape(H, W)

    max_sid = np.max(min_sid)
    if max_sid > 0:
        abundances = 1.0 - min_sid / max_sid
    else:
        abundances = np.ones(H * W)
    abundances = abundances.reshape(H, W)

    C = np.zeros((H, W, n_endmembers))
    for k in range(n_endmembers):
        C[:, :, k] = (class_map == k).astype(float)

    return {
        'concentrations': C,
        'r_squared': abundances,
        'residuals': min_sid.reshape(H, W),
        'classification': class_map,
        'info': {
            'n_endmembers': n_endmembers,
            'n_pixels': H * W,
            'mean_sid': float(np.mean(min_sid)),
            'median_sid': float(np.median(min_sid)),
        },
    }


# ---------------------------------------------------------------------------
# RX — Reed-Xiaoli Anomaly Detection
# ---------------------------------------------------------------------------

def run_rx(
    hypercube: np.ndarray,
) -> Dict[str, Any]:
    """
    Reed-Xiaoli (RX) anomaly detection.

    Computes the Mahalanobis distance of each pixel from the global
    spectral mean.  High-scoring pixels have unusual spectra relative
    to the image background — no reference spectra required.

    Parameters
    ----------
    hypercube : (H, W, L) array

    Returns
    -------
    dict with keys ``anomaly_score`` (H, W), ``info``.
    """
    H, W, L = hypercube.shape
    data = hypercube.reshape(-1, L).astype(np.float64)
    n_pixels = data.shape[0]

    mean_spec = np.mean(data, axis=0)
    data_centered = data - mean_spec
    cov = (data_centered.T @ data_centered) / n_pixels
    cov += np.eye(L) * 1e-10
    cov_inv = np.linalg.inv(cov)

    rx_score = np.sum((data_centered @ cov_inv) * data_centered, axis=1)

    return {
        'anomaly_score': rx_score.reshape(H, W),
        'info': {
            'algorithm': 'RX Anomaly Detection',
            'n_pixels': n_pixels,
            'n_bands': L,
            'mean_score': float(np.mean(rx_score)),
            'max_score': float(np.max(rx_score)),
            'threshold_99': float(np.percentile(rx_score, 99)),
        },
    }
