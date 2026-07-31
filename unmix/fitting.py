# Copyright (c) 2026 Dan Fu@UW
"""
Spectral peak fitting utilities.

Fit Gaussian, Lorentzian, or pseudo-Voigt profiles to a spectrum
using scipy.optimize.curve_fit.

Result dict keys
----------------
fitted        : (L,) — fitted spectrum values
residuals     : (L,) — measured - fitted
peaks         : list of dicts — one per peak found
              Each dict: {'center', 'amplitude', 'width', 'area', 'model'}
info          : dict — algorithm metadata, chi2, n_peaks
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def _gaussian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Single Gaussian peak."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))


def _lorentzian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Single Lorentzian (Cauchy) peak."""
    return amplitude * (width ** 2) / ((x - center) ** 2 + width ** 2)


def _multi_gaussian(x: np.ndarray, *params: float) -> np.ndarray:
    """Sum of Gaussians. params = (amp0, cen0, wid0, amp1, cen1, wid1, ...)."""
    n_peaks = len(params) // 3
    result = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        amp, cen, wid = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        result += _gaussian(x, amp, cen, wid)
    return result


def _multi_lorentzian(x: np.ndarray, *params: float) -> np.ndarray:
    """Sum of Lorentzians. params = (amp0, cen0, wid0, amp1, cen1, wid1, ...)."""
    n_peaks = len(params) // 3
    result = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        amp, cen, wid = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        result += _lorentzian(x, amp, cen, wid)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_spectrum(
    spectrum: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    model: str = 'gaussian',
    n_peaks: Optional[int] = None,
    prominence: float = 0.1,
    min_distance: int = 5,
    baseline: Optional[float] = None,
    max_iter: int = 1000,
) -> Dict[str, Any]:
    """
    Fit one or more peak profiles to a spectrum.

    Parameters
    ----------
    spectrum : (L,) array
        Measured spectrum to fit.
    wavelengths : (L,) array, optional
        Wavelength / band values. Defaults to ``np.arange(L)``.
    model : {'gaussian', 'lorentzian'}
        Line shape for each peak.
    n_peaks : int, optional
        Number of peaks to fit. Auto-detected via ``scipy.signal.find_peaks``
        if not specified.
    prominence : float
        Minimum peak prominence for auto-detection (fraction of
        ``max(spectrum) - min(spectrum)``).
    min_distance : int
        Minimum number of bands between detected peaks.
    baseline : float, optional
        Baseline offset to subtract before fitting. Defaults to
        ``min(spectrum)``.
    max_iter : int
        Maximum iterations for ``curve_fit``.

    Returns
    -------
    dict with keys ``fitted``, ``residuals``, ``peaks``, ``info``.

    Raises
    ------
    ValueError
        If the spectrum is empty or no peaks are detected.
    RuntimeError
        If the fit fails to converge.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    L = len(spectrum)

    if L == 0:
        raise ValueError("Spectrum is empty")

    if wavelengths is None:
        wavelengths = np.arange(L, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)

    if len(wavelengths) != L:
        raise ValueError(
            f"wavelengths length ({len(wavelengths)}) != spectrum length ({L})")

    # Baseline subtraction
    if baseline is None:
        baseline = np.min(spectrum)
    spec_clean = spectrum - baseline

    # Auto-detect peaks
    if n_peaks is None:
        peak_range = np.max(spec_clean) - np.min(spec_clean)
        prom = prominence * peak_range if peak_range > 0 else 1e-6
        peak_indices, properties = find_peaks(
            spec_clean, prominence=prom, distance=min_distance)
        n_peaks = len(peak_indices)

    if n_peaks == 0:
        raise ValueError("No peaks detected — try lowering prominence")
    if n_peaks > 10:
        n_peaks = 10  # sanity cap — curve_fit becomes unstable with many peaks

    # Initial parameter guesses: (amp, center, width) × n_peaks
    peak_indices, properties = find_peaks(
        spec_clean, prominence=0.01 * (np.max(spec_clean) - np.min(spec_clean)),
        distance=min_distance)
    peak_indices = peak_indices[:n_peaks]

    p0 = []
    for idx in peak_indices:
        amp = spec_clean[idx]
        cen = wavelengths[idx]
        half_max = amp / 2.0
        # Find left half-maximum crossing
        left_idx = idx
        for j in range(idx - 1, -1, -1):
            if spec_clean[j] <= half_max:
                left_idx = j
                break
        # Find right half-maximum crossing
        right_idx = idx
        for j in range(idx + 1, L):
            if spec_clean[j] <= half_max:
                right_idx = j
                break
        fwhm = abs(wavelengths[right_idx] - wavelengths[left_idx])
        if fwhm > 0:
            if model == 'gaussian':
                wid = fwhm / 2.355  # FWHM = 2*sqrt(2*ln2)*sigma
            else:
                wid = fwhm / 2.0    # Lorentzian FWHM = 2*gamma
        else:
            wid = (wavelengths[-1] - wavelengths[0]) / (10.0 * n_peaks)
        wid = max(0.1, wid)
        p0.extend([amp, cen, wid])

    # Select model function
    if model == 'gaussian':
        func = _multi_gaussian
    elif model == 'lorentzian':
        func = _multi_lorentzian
    else:
        raise ValueError(f"Unknown model: {model!r}")

    # Bounds: amplitude >= 0, width >= 0.1
    lower_bounds = [0.0] * (3 * n_peaks)
    for i in range(2, 3 * n_peaks, 3):
        lower_bounds[i] = 0.1
    bounds = (lower_bounds, [np.inf] * (3 * n_peaks))

    try:
        popt, pcov = curve_fit(
            func, wavelengths, spec_clean,
            p0=p0, bounds=bounds, maxfev=max_iter * 3 * n_peaks)
    except RuntimeError as e:
        raise RuntimeError(f"Fit failed to converge: {e}")

    # Build fitted spectrum
    fitted = func(wavelengths, *popt) + baseline
    residuals = spectrum - fitted

    # Extract peak parameters
    peaks = []
    for i in range(n_peaks):
        amp = popt[3 * i]
        cen = popt[3 * i + 1]
        wid = popt[3 * i + 2]
        # Area approximation
        if model == 'gaussian':
            area = amp * wid * np.sqrt(2 * np.pi)
        else:
            area = amp * wid * np.pi
        peaks.append({
            'center': float(cen),
            'amplitude': float(amp),
            'width': float(wid),
            'area': float(area),
            'model': model,
        })

    chi2 = np.sum(residuals ** 2) / (L - 3 * n_peaks) if L > 3 * n_peaks else np.inf

    return {
        'fitted': fitted,
        'residuals': residuals,
        'peaks': peaks,
        'info': {
            'algorithm': f'{model.title()} Fitting',
            'n_peaks': n_peaks,
            'n_bands': L,
            'baseline': float(baseline),
            'chi2': float(chi2),
            'model': model,
        },
    }
