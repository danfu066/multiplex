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

    Finds the simplex of maximum volume in spectral space.  Iteratively
    replaces each vertex with the pixel farthest from the hyperplane
    spanned by the remaining vertices until convergence.

    Parameters
    ----------
    hypercube : (H, W, L) array
    n_endmembers : int — number of endmembers to extract.
    random_state : int — seed for initial vertex selection.

    Returns
    -------
    dict with keys ``endmembers``, ``indices``, ``positions``, ``info``.

    Notes
    -----
    Crispin, J. T. et al. (2000). "MINLR, N-FINDR, and 2D-SFSCARD:
    Overview of three multichannel AVIRIS endmember-purity index algorithms."
    """
    H, W, L = hypercube.shape
    n_pixels = H * W

    if n_endmembers >= L:
        raise ValueError(
            f"n_endmembers ({n_endmembers}) must be < n_bands ({L})")
    if n_endmembers > n_pixels:
        raise ValueError(
            f"n_endmembers ({n_endmembers}) must be <= n_pixels ({n_pixels})")

    data = hypercube.reshape(-1, L).astype(np.float64)

    # Centroid-based initialization (more stable than random)
    centroid = np.mean(data, axis=0)
    dists = np.linalg.norm(data - centroid, axis=1)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_pixels)
    vertices = perm[:n_endmembers]

    # N-FINDR iterations
    max_iter = 50
    for iteration in range(max_iter):
        old_vertices = vertices.copy()

        for i in range(n_endmembers):
            # Hyperplane spanned by all other vertices
            others = np.delete(vertices, i)
            A = data[others]  # (K-1, L)

            # Project all pixels onto the normal of this hyperplane
            # Normal = null space of A → last column of Vt from SVD
            _, _, Vt = np.linalg.svd(A, full_matrices=False)
            normal = Vt[-1, :]  # direction orthogonal to hyperplane

            # Signed distance of each pixel from hyperplane through origin
            # Offset by one vertex to anchor the plane
            diff = data - data[others[0]]
            projections = np.abs(diff @ normal)

            # Find pixel farthest from hyperplane
            best = np.argmax(projections)
            vertices[i] = best

        # Convergence: vertex set unchanged
        if np.array_equal(vertices, old_vertices):
            break

    # Extract endmember spectra and positions
    endmembers = data[vertices]
    rows = vertices // W
    cols = vertices % W

    return {
        'endmembers': endmembers,
        'indices': vertices,
        'positions': np.column_stack([rows, cols]),
        'info': {
            'algorithm': 'N-FINDR',
            'n_endmembers': n_endmembers,
            'n_bands': L,
            'n_pixels': n_pixels,
            'iterations': iteration + 1,
            'converged': np.array_equal(vertices, old_vertices),
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

    Fast NIPALS-style algorithm that exploits the minimum volume
    property of the spectral simplex.  Projects data onto orthogonal
    directions to successively isolate vertices.

    Parameters
    ----------
    hypercube : (H, W, L) array
    n_endmembers : int — number of endmembers to extract.
    random_state : int — seed for initial direction.

    Returns
    -------
    dict with keys ``endmembers``, ``indices``, ``positions``, ``info``.

    Notes
    -----
    Nascimento, J. M. & Plaza, A. (2002). "Dimensionality reduction and
    classification of hyperspectral data with histograms of angles."
    IEEE TGRS, 40(1), 253-266.
    """
    H, W, L = hypercube.shape
    n_pixels = H * W

    if n_endmembers >= L:
        raise ValueError(
            f"n_endmembers ({n_endmembers}) must be < n_bands ({L})")

    data = hypercube.reshape(-1, L).astype(np.float64)

    # Mean-centre the data
    mean = np.mean(data, axis=0)
    data_centered = data - mean

    rng = np.random.RandomState(random_state)
    vertices = []

    # Random initial unit vector
    p = rng.randn(L)
    p /= np.linalg.norm(p)

    for k in range(n_endmembers):
        # Project all pixels onto the orthogonal direction p
        projections = np.abs(data_centered @ p)

        # Vertex = pixel with maximum projection
        idx = np.argmax(projections)
        vertices.append(idx)

        if k < n_endmembers - 1:
            # Compute new orthogonal direction via NIPALS
            endmember = data_centered[idx]
            # Project all data onto direction of found endmember
            scores = data_centered @ endmember
            scores /= np.linalg.norm(endmember) ** 2
            # Deflate: remove contribution of found endmember
            data_centered -= np.outer(scores, endmember)

            # New random direction in deflated space
            p = rng.randn(L)
            p /= np.linalg.norm(p)

    # Extract endmember spectra from original (non-centred) data
    endmembers = data[vertices]
    rows = np.array(vertices) // W
    cols = np.array(vertices) % W

    return {
        'endmembers': endmembers,
        'indices': np.array(vertices),
        'positions': np.column_stack([rows, cols]),
        'info': {
            'algorithm': 'VCA',
            'n_endmembers': n_endmembers,
            'n_bands': L,
            'n_pixels': n_pixels,
        },
    }
