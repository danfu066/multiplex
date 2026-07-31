# Copyright (c) 2026 Dan Fu@UW
"""
Blind-source-separation / unmixing algorithms.

All functions return a standardised result dict with no side effects.

Result dict keys
----------------
concentrations : (H, W, K) — abundance per component per pixel
basis_spectra  : (K, L)    — spectral profiles (components / endmembers)
r_squared      : (H, W)    — coefficient of determination per pixel
residuals      : (H, W)    — RMS residual per pixel
info           : dict       — human-readable summary
"""

from typing import Dict, Any, Optional, List, Callable

import numpy as np
from scipy.optimize import nnls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_n_components(
    hypercube: np.ndarray,
    variance_threshold: float = 0.95,
) -> int:
    """Auto-detect number of components via cumulative explained variance."""
    _, S, _ = np.linalg.svd(
        hypercube.reshape(-1, hypercube.shape[-1]).astype(np.float64),
        full_matrices=False,
    )
    explained = (S ** 2) / np.sum(S ** 2)
    cumvar = np.cumsum(explained)
    n = np.searchsorted(cumvar, variance_threshold) + 1
    return max(n, 1)


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
# MCR-ALS
# ---------------------------------------------------------------------------

def run_mcr_als(
    hypercube: np.ndarray,
    n_components: Optional[int] = None,
    max_iter: int = 100,
    non_neg: bool = True,
    closure: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Multivariate Curve Resolution — Alternating Least Squares.

    Decomposes the hypercube into concentration and spectral profiles
    without a known basis.  SVD initialisation, then ALS iterations
    with optional non-negativity and closure constraints.

    Parameters
    ----------
    hypercube : (H, W, L) array
    n_components : int or None
        ``None`` → auto-detect (95 % cumulative variance).
    max_iter : int
    non_neg : bool — enforce concentrations & spectra >= 0
    closure : bool — enforce concentrations sum to 1 per pixel
    progress_callback : callable(progress_pct, status_text) or None

    Returns
    -------
    dict with keys ``concentrations``, ``basis_spectra``,
    ``r_squared``, ``residuals``, ``info``.
    """
    H, W, L = hypercube.shape
    n_pixels = H * W

    if n_components is None:
        n_components = _auto_n_components(hypercube)

    if n_components >= L:
        raise ValueError(
            f"n_components ({n_components}) must be < n_bands ({L})")

    data = hypercube.reshape(-1, L).astype(np.float64)
    # Initialize initial seed spectra using VCA (Vertex Component Analysis) to ensure distinct initial endmembers
    try:
        vca_res = run_vca(hypercube, n_endmembers=n_components)
        S_mat = vca_res['endmembers'].copy()
    except Exception:
        U, S, Vt = np.linalg.svd(data, full_matrices=False)
        S_mat = np.abs(Vt[:n_components, :].copy())

    C = np.zeros((n_pixels, n_components), dtype=np.float64)

    converged = False
    for iteration in range(max_iter):
        C_old = C.copy()
        S_old = S_mat.copy()

        # Update concentrations: C = data @ pinv(S)
        S_pinv = np.linalg.pinv(S_mat)
        C = data @ S_pinv
        if non_neg:
            C = np.maximum(C, 0)
        if closure:
            row_sums = C.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            C = C / row_sums

        # Update spectra: S = pinv(C) @ data
        C_pinv = np.linalg.pinv(C)
        S_mat = C_pinv @ data
        if non_neg:
            S_mat = np.maximum(S_mat, 0)

        dc = np.linalg.norm(C - C_old) / (np.linalg.norm(C_old) + 1e-12)
        ds = np.linalg.norm(S_mat - S_old) / (
            np.linalg.norm(S_old) + 1e-12)

        if progress_callback:
            pct = int((iteration + 1) / max_iter * 100)
            progress_callback(
                pct,
                f"Iter {iteration + 1}/{max_iter}  "
                f"ΔC={dc:.2e}  ΔS={ds:.2e}",
            )

        if dc < 1e-6 and ds < 1e-6:
            converged = True
            break

    reconstructed = C @ S_mat
    r2, resid = _compute_r2_residuals(data, reconstructed, (H, W))

    return {
        'concentrations': C.reshape(H, W, n_components),
        'basis_spectra': S_mat,
        'r_squared': r2,
        'residuals': resid,
        'info': {
            'n_components': n_components,
            'n_pixels': n_pixels,
            'converged': converged,
            'iterations': iteration + 1,
        },
    }


# ---------------------------------------------------------------------------
# NMF
# ---------------------------------------------------------------------------

def run_nmf(
    hypercube: np.ndarray,
    n_components: Optional[int] = None,
    max_iter: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Non-Negative Matrix Factorization.

    Uses scikit-learn NMF with multiplicative update rules.

    Parameters
    ----------
    hypercube : (H, W, L)
    n_components : int or None — ``None`` → auto-detect.
    max_iter : int
    random_state : int

    Returns
    -------
    dict with keys ``concentrations``, ``basis_spectra``,
    ``r_squared``, ``residuals``, ``info``.
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        raise ImportError(
            "Install scikit-learn for NMF: pip install scikit-learn")

    H, W, L = hypercube.shape
    n_pixels = H * W

    if n_components is None:
        n_components = _auto_n_components(hypercube)

    if n_components >= L:
        raise ValueError(
            f"n_components ({n_components}) must be < n_bands ({L})")

    data = hypercube.reshape(-1, L).astype(np.float64)
    model = NMF(
        n_components=n_components,
        init='nndsvdar',
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )
    C = model.fit_transform(data)
    S_mat = model.components_

    reconstructed = C @ S_mat
    r2, resid = _compute_r2_residuals(data, reconstructed, (H, W))

    return {
        'concentrations': C.reshape(H, W, n_components),
        'basis_spectra': S_mat,
        'r_squared': r2,
        'residuals': resid,
        'info': {
            'n_components': n_components,
            'n_pixels': n_pixels,
            'random_state': random_state,
        },
    }


# ---------------------------------------------------------------------------
# MESMA
# ---------------------------------------------------------------------------

def run_mesma(
    hypercube: np.ndarray,
    basis_spectra: np.ndarray,
    max_endmembers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Fast Vectorized Multiple Endmember Spectral Mixture Analysis (MESMA).

    For each pixel, selects the best subset of *max_endmembers* endmembers from
    *basis_spectra* and fits via fast non-negative projection.

    Parameters
    ----------
    hypercube : (H, W, L) array
    basis_spectra : (K, L) array — calibration endmembers
    max_endmembers : int or None — ``None`` → min(4, K).
    progress_callback : callable(progress_pct, status_text) or None

    Returns
    -------
    dict with keys ``concentrations``, ``basis_spectra``, ``r_squared``, ``residuals``, ``info``.
    """
    H, W, L = hypercube.shape
    n_pixels = H * W
    K = basis_spectra.shape[0]

    if K < 2:
        raise ValueError("Need at least 2 endmembers for MESMA.")

    if max_endmembers is None or max_endmembers > K:
        max_endmembers = min(4, K)

    from itertools import combinations
    subsets = list(combinations(range(K), max_endmembers))

    X = hypercube.reshape(-1, L).astype(np.float64)  # (N, L)
    best_C = np.zeros((n_pixels, K), dtype=np.float64)
    min_residuals = np.full(n_pixels, np.inf, dtype=np.float64)

    # Evaluate each subset vectorized across all pixels
    for sub_idx, sub in enumerate(subsets):
        E_sub = basis_spectra[list(sub)].T  # (L, M)
        inv_GtG = np.linalg.pinv(E_sub.T @ E_sub)  # (M, M)
        W_sub = np.maximum(0.0, X @ (E_sub @ inv_GtG))  # (N, M)

        recon_sub = W_sub @ E_sub.T  # (N, L)
        res_sub = np.sum((X - recon_sub) ** 2, axis=1)  # (N,)

        better_mask = res_sub < min_residuals
        if np.any(better_mask):
            min_residuals[better_mask] = res_sub[better_mask]
            best_C[better_mask, :] = 0.0
            idx_better = np.where(better_mask)[0]
            best_C[idx_better[:, None], np.array(sub)] = W_sub[better_mask]

        if progress_callback:
            pct = int((sub_idx + 1) / len(subsets) * 100)
            progress_callback(pct, f"Evaluating MESMA subset {sub_idx+1}/{len(subsets)}")

    reconstructed = best_C @ basis_spectra
    r2, resid = _compute_r2_residuals(X, reconstructed, (H, W))

    return {
        'concentrations': best_C.reshape(H, W, K),
        'basis_spectra': basis_spectra,
        'r_squared': r2,
        'residuals': resid,
        'info': {
            'n_endmembers': K,
            'max_per_pixel': max_endmembers,
            'n_subsets': len(subsets),
            'n_pixels': n_pixels,
        },
    }


def run_linear_unmix(
    hypercube: np.ndarray,
    basis_spectra: np.ndarray,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Per-pixel Non-Negative Least Squares (NNLS) Linear Spectral Unmixing.

    Solves ``min ||x - A c||_2  s.t. c >= 0`` for every pixel using
    ``scipy.optimize.nnls``, guaranteeing the true optimal non-negative
    solution (not a clipped OLS approximation).

    Parameters
    ----------
    hypercube : (H, W, L) array
    basis_spectra : (K, L) array — reference endmember spectra
    progress_callback : callable(progress_pct, status_text) or None

    Returns
    -------
    dict with keys ``concentrations``, ``basis_spectra``, ``r_squared``,
    ``residuals``, ``info``.
    """
    H, W, L = hypercube.shape
    K = basis_spectra.shape[0]
    n_pixels = H * W
    X = hypercube.reshape(-1, L).astype(np.float64)
    A = basis_spectra.T.astype(np.float64)  # (L, K)

    C = np.zeros((n_pixels, K), dtype=np.float64)
    report_every = max(1, n_pixels // 100)
    for i in range(n_pixels):
        C[i], _ = nnls(A, X[i])
        if progress_callback and i % report_every == 0:
            progress_callback(int(i / n_pixels * 100),
                              f"NNLS pixel {i}/{n_pixels}")

    if progress_callback:
        progress_callback(100, "NNLS complete")

    reconstructed = C @ basis_spectra
    r2, resid = _compute_r2_residuals(X, reconstructed, (H, W))

    return {
        'concentrations': C.reshape(H, W, K),
        'basis_spectra': basis_spectra,
        'r_squared': r2,
        'residuals': resid,
        'info': {'n_endmembers': K, 'algorithm': 'NNLS Linear Unmixing',
                 'n_pixels': n_pixels},
    }


def run_fcls(
    hypercube: np.ndarray,
    basis_spectra: np.ndarray,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Fully Constrained Least Squares (FCLS) Linear Spectral Unmixing.

    Enforces both non-negativity (c >= 0) and sum-to-one (sum(c) = 1)
    per pixel by augmenting the endmember matrix with a unity row and
    solving the augmented NNLS problem (Heinz & Chang 2001).

    Parameters
    ----------
    hypercube : (H, W, L) array
    basis_spectra : (K, L) array — reference endmember spectra
    progress_callback : callable(progress_pct, status_text) or None

    Returns
    -------
    dict with keys ``concentrations``, ``basis_spectra``, ``r_squared``,
    ``residuals``, ``info``.
    """
    H, W, L = hypercube.shape
    K = basis_spectra.shape[0]
    n_pixels = H * W
    X = hypercube.reshape(-1, L).astype(np.float64)

    delta = np.max(np.abs(basis_spectra)) * K
    A_aug = np.vstack([basis_spectra.T, delta * np.ones((1, K))])  # (L+1, K)

    C = np.zeros((n_pixels, K), dtype=np.float64)
    report_every = max(1, n_pixels // 100)
    for i in range(n_pixels):
        b_aug = np.append(X[i], delta)
        C[i], _ = nnls(A_aug, b_aug)
        if progress_callback and i % report_every == 0:
            progress_callback(int(i / n_pixels * 100),
                              f"FCLS pixel {i}/{n_pixels}")

    if progress_callback:
        progress_callback(100, "FCLS complete")

    reconstructed = C @ basis_spectra
    r2, resid = _compute_r2_residuals(X, reconstructed, (H, W))

    return {
        'concentrations': C.reshape(H, W, K),
        'basis_spectra': basis_spectra,
        'r_squared': r2,
        'residuals': resid,
        'info': {'n_endmembers': K, 'algorithm': 'FCLS Linear Unmixing',
                 'n_pixels': n_pixels},
    }
