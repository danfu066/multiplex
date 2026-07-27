# Hyperspectral Unmixer

Interactive desktop GUI for hyperspectral image analysis — unmixing, denoising, classification, endmember extraction, spectral fitting, and visualization.

Built with **PyQt5** (UI), **Matplotlib** (plots), **NumPy/SciPy** (numerics), **scikit-learn** (ML), and **PyWavelets** (wavelet denoising).

---

## Setup

```powershell
# 1. Activate virtual environment (from Multiplexing/)
.venv\Scripts\activate

# 2. Install dependencies (first time only)
pip install -r unmix\requirements.txt

# 3. Run
python unmix\Unmixer.py
```

VSCode: **Ctrl+Shift+P → Python: Select Interpreter → `.venv`**. New terminals auto-activate.

---

## Interface

The window is split into three panels:

| Panel | Purpose |
|---|---|
| **Spatial View** (top-left) | Band image with selection overlays, colorbar |
| **Spectrum / Residual** (bottom-left) | Selected pixel spectrum + fit residual |
| **Controls** (right) | Band selector, component combo, selection tools, progress, status |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `P` | Point selection tool |
| `C` | Circle selection tool |
| `S` | Square selection tool |
| `Escape` | Clear selection |

---

## Features

### File I/O

| Menu | Description |
|---|---|
| **File → Open ENVI** | Load `.hdr`/`.dat` ENVI hyperspectral cubes (BIL, BSQ, BIP interleaves) |
| **File → Open NPY** | Load `.npy` arrays of shape `(H, W, L)` |
| **File → Open CSV Spectrum** | Load reference spectra as calibration basis |
| **File → Export Results** | Save concentration maps and R² as CSV |

### Denoising

| Menu | Algorithm | What it does |
|---|---|---|
| **Analysis → Denoise → MPPCA** | Multiplicative PCA | Truncated SVD on unfolded cube. Keeps components explaining ≥95% variance. Discards noise-dominated components. |
| **Analysis → Denoise → Wavelet** | VisuShrink (universal threshold) | Wavelet decomposition per band, soft-thresholding, reconstruction. Uses Daubechies-4 wavelet. Requires `PyWavelets`. |

> Both replace the loaded hypercube in-place. Denoise before unmixing for cleaner results.

### Unmixing (Blind Source Separation)

These algorithms decompose the hypercube into basis spectra × concentration maps **without** needing reference spectra.

| Menu | Algorithm | What it does |
|---|---|---|
| **Analysis → Unmixing → MCR-ALS** | Multivariate Curve Resolution — Alternating Least Squares | Alternates between estimating spectra and concentrations with non-negativity and optional closure (sum-to-one) constraints. Can auto-detect components via SVD elbow. |
| **Analysis → Unmixing → NMF** | Non-Negative Matrix Factorization | Multiplicative-update NMF. Guarantees non-negative basis and concentrations. Specify number of components. Requires `scikit-learn`. |
| **Analysis → Unmixing → MESMA** | Multiple Endmember Spectral Mixture Analysis | For each pixel, tries all combinations of endmembers up to a max count, picks the best fit. Requires a loaded basis. |

### Decomposition

| Menu | Algorithm | What it does |
|---|---|---|
| **Analysis → Decomposition → ICA** | Independent Component Analysis | FastICA (NIPALS). Finds statistically independent components. Good for separating mixed signals. Requires `scikit-learn`. |

### Classification

These require a loaded calibration basis (CSV or extracted endmembers).

| Menu | Algorithm | What it does |
|---|---|---|
| **Analysis → Classification → PCA** | Principal Component Analysis | SVD-based. Returns loadings as basis spectra, scores as concentrations. Shows explained variance per component. |
| **Analysis → Classification → SAM** | Spectral Angle Mapper | Classifies each pixel by minimum spectral angle to a basis spectrum. Angle-based, illumination-invariant. |
| **Analysis → Classification → SID** | Spectral Information Divergence | KL-divergence between pixel and each basis spectrum. Sensitive to spectral shape differences. |
| **Analysis → Classification → SVR** | Support Vector Regression | Non-linear regression to estimate abundances. Trains on synthetic mixtures of basis spectra. Requires `scikit-learn`. |

### Endmember Extraction

Automatically find pure-pixel endmembers from the data.

| Menu | Algorithm | What it does |
|---|---|---|
| **Analysis → Extract Endmembers → N-FINDR** | N-FINDR | Finds endmembers that maximize the volume of the simplex in spectral space. Iterative, robust. |
| **Analysis → Extract Endmembers → VCA** | Vertex Component Analysis | Fast NIPALS-style method. Uses orthogonal null-space projection to find simplex vertices. |
| **Analysis → Extract Endmembers → Interactive Picker** | Manual selection | Toggle picker mode (Ctrl+I), click pixels on the spatial view. Marked with magenta circles. Click **Apply Picked Endmembers** to load them as basis. |

Extracted endmembers are automatically loaded as the calibration basis and displayed in the spectrum panel.

### Spectral Fitting

Fit parametric peak models to a selected pixel or ROI mean spectrum.

| Menu | Model | What it does |
|---|---|---|
| **Analysis → Fit Peaks → Gaussian Fit** | Sum of Gaussians | Fits `N` Gaussian peaks (auto-detected via peak finding) to the spectrum. Returns peak centers, widths, amplitudes, χ². |
| **Analysis → Fit Peaks → Lorentzian Fit** | Sum of Lorentzians | Same as above but with Lorentzian line shapes. Better for resonant/narrow features. |

> Requires a point, circle, or square selection first. Results shown in the Spectrum/Residual panels.

### Visualization

| Menu | Description |
|---|---|
| **Analysis → Visualization → RGB Composite** | Select 3 bands as R/G/B channels. False-color image with per-channel percentile normalization. |
| **Analysis → Visualization → PC-RGB** | First 3 PCA components as R/G/B. Fast false-color showing dominant variance directions. |

---

## Data Convention

Hypercube shape: **`(height, width, bands)`** — access is `hypercube[y, x, band]`.

Basis spectra shape: **`(n_endmembers, bands)`**.

---

## Architecture

```
Multiplexing/
├── Unmixer.py              # PyQt5 GUI (thin UI layer)
├── requirements.txt        # Dependencies
├── .vscode/settings.json   # VSCode interpreter config
└── unmix/                  # Algorithm package (pure functions)
    ├── __init__.py         # Public API
    ├── denoising.py        # MPPCA, Wavelet
    ├── unmixing.py         # MCR-ALS, NMF, MESMA
    ├── classification.py   # PCA, SAM, SID, SVR
    ├── endmember.py        # N-FINDR, VCA
    ├── fitting.py          # Gaussian, Lorentzian
    └── README.md           # Package API reference
```

All algorithms in `unmix/` are **pure functions** — no UI dependencies, no side effects. They accept `(H, W, L)` arrays and return standardized result dicts. See `unmix/README.md` for API details.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `scipy` | SVD, optimization, signal processing |
| `matplotlib` | Plots, embedded canvases |
| `PyQt5` | GUI framework |
| `scikit-learn` | NMF, SVR, ICA, PCA |
| `PyWavelets` | Wavelet denoising |

---

## Author

Multiplexing Lab, University of Washington
