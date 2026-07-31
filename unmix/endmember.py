# Copyright (c) 2026 Dan Fu@UW
"""
Automatic endmember extraction algorithms.

All functions accept a hypercube of shape (H, W, L) and return
extracted endmembers plus spatial indices.

Result dict keys
----------------
endmembers    : (K, L) — extracted endmember spectra
indices       : (K,)   — pixel index (row-major) of each endmember
positions     : (K, 2) — (row, col) spatial coordinates
info          : dict   — algorithm-specific metadata
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# N-FINDR (N-FindR)
# ---------------------------------------------------------------------------

def run_nfindr(
    hypercube: np.ndarray,
    n_endmembers: int,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    N-FINDR endmember extraction.

    Reduces the data to (n_endmembers - 1) dimensions via PCA, then
    iteratively finds the simplex of maximum volume by replacing each
    vertex with the pixel that maximises the simplex volume.

    Parameters
    ----------
    hypercube : (H, W, L) array
    n_endmembers : int — number of endmembers to extract.
    random_state : int — seed for initial vertex selection.

    Returns
    -------
    dict with keys ``endmembers``, ``indices``, ``positions``, ``info``.
    """
    H, W, L = hypercube.shape
    n_pixels = H * W
    K = n_endmembers

    if K > n_pixels:
        raise ValueError(
            f"n_endmembers ({K}) must be <= n_pixels ({n_pixels})")
    if K < 2:
        raise ValueError("n_endmembers must be >= 2")

    data = hypercube.reshape(-1, L).astype(np.float64)

    # PCA reduction to (K-1) dimensions for volume computation
    mean_spec = np.mean(data, axis=0)
    data_centered = data - mean_spec
    _, _, Vt = np.linalg.svd(data_centered, full_matrices=False)
    proj_matrix = Vt[:K - 1, :]  # (K-1, L)
    data_reduced = data_centered @ proj_matrix.T  # (N, K-1)

    rng = np.random.RandomState(random_state)
    vertices = rng.choice(n_pixels, size=K, replace=False)

    def _simplex_volume(verts):
        """Signed volume of simplex formed by vertex indices in reduced space."""
        coords = data_reduced[verts]  # (K, K-1)
        augmented = np.ones((K, K), dtype=np.float64)
        augmented[:, 1:] = coords
        return np.abs(np.linalg.det(augmented))

    max_iter = 50
    for iteration in range(max_iter):
        changed = False
        for i in range(K):
            current_vol = _simplex_volume(vertices)
            best_vol = current_vol
            best_pixel = vertices[i]

            test_verts = vertices.copy()
            for p in range(n_pixels):
                if p in vertices:
                    continue
                test_verts[i] = p
                vol = _simplex_volume(test_verts)
                if vol > best_vol:
                    best_vol = vol
                    best_pixel = p

            if best_pixel != vertices[i]:
                vertices[i] = best_pixel
                changed = True

        if not changed:
            break

    endmembers = data[vertices]
    rows = vertices // W
    cols = vertices % W

    return {
        'endmembers': endmembers,
        'indices': vertices,
        'positions': np.column_stack([rows, cols]),
        'info': {
            'algorithm': 'N-FINDR',
            'n_endmembers': K,
            'n_bands': L,
            'n_pixels': n_pixels,
            'iterations': iteration + 1,
            'converged': not changed,
            'final_volume': float(_simplex_volume(vertices)),
        },
    }


# ---------------------------------------------------------------------------
# VCA (Vertex Component Analysis)
# ---------------------------------------------------------------------------

def run_vca(
    hypercube: np.ndarray,
    n_endmembers: int,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Vertex Component Analysis endmember extraction.

    Projects data onto a (p)-dimensional PCA subspace, then iteratively
    identifies simplex vertices by projecting onto directions orthogonal
    to the subspace spanned by already-selected endmembers.

    Based on: Nascimento & Bioucas-Dias (2005), "Vertex Component
    Analysis: A Fast Algorithm to Unmix Hyperspectral Data", IEEE TGRS.

    Parameters
    ----------
    hypercube : (H, W, L) array
    n_endmembers : int — number of endmembers to extract.
    random_state : int — seed for initial direction.

    Returns
    -------
    dict with keys ``endmembers``, ``indices``, ``positions``, ``info``.
    """
    H, W, L = hypercube.shape
    n_pixels = H * W
    K = n_endmembers

    if K < 2:
        raise ValueError("n_endmembers must be >= 2")

    data = hypercube.reshape(-1, L).astype(np.float64)

    # PCA projection to K dimensions
    mean_spec = np.mean(data, axis=0)
    data_centered = data - mean_spec
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    proj = Vt[:K, :]  # (K, L)
    Y = data_centered @ proj.T  # (N, K) — data in reduced space

    rng = np.random.RandomState(random_state)
    vertices = []
    A = np.zeros((K, 0), dtype=np.float64)  # selected endmember columns

    for k in range(K):
        if A.shape[1] == 0:
            # First endmember: random direction
            w = rng.randn(K)
        else:
            # Orthogonal projector onto complement of span(A)
            # f = (I - A @ pinv(A)) @ w
            w = rng.randn(K)
            proj_A = A @ np.linalg.pinv(A)  # projection onto span(A)
            w = w - proj_A @ w

        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            w = rng.randn(K)
            w_norm = np.linalg.norm(w)
        w = w / w_norm

        projections = np.abs(Y @ w)
        idx = np.argmax(projections)
        vertices.append(idx)

        A = np.column_stack([A, Y[idx]])

    vertices = np.array(vertices)
    endmembers = data[vertices]
    rows = vertices // W
    cols = vertices % W

    return {
        'endmembers': endmembers,
        'indices': vertices,
        'positions': np.column_stack([rows, cols]),
        'info': {
            'algorithm': 'VCA',
            'n_endmembers': K,
            'n_bands': L,
            'n_pixels': n_pixels,
        },
    }
