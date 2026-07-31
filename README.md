# Multiplexing & Hyperspectral Analysis Suite

An interactive, high-performance Python/PyQt5 software suite for hyperspectral image viewing, spectral unmixing, signal processing, and liquid-crystal optical path difference (OPD) retardance measurement.

---

## 1. Overview & Applications

This suite consists of three core applications tailored for biological imaging, material science, and optical physics:

### 1. `HyperViewer.py`
A lightweight, high-speed viewer for 3D hyperspectral data cubes $(H \times W \times \lambda)$.
- **Interactive Spatial View**: Step through spectral frames using a smooth slider.
- **Synchronized Side Views**: Real-time **X-Z** (horizontal slice vs wavelength) and **Y-Z** (vertical slice vs wavelength) cross-sections linked to spatial crosshairs.
- **ROI Tools**: Select Point ($+$), Circle ($\bigcirc$), and Square ($\square$) regions of interest to extract and average spectra.
- **Multi-Spectrum Comparison**: Overlay multiple spectra with automatic color assignment and 5% margin autoscaling.

### 2. `unmix/Unmixer.py`
An advanced hyperspectral analysis and linear/non-linear unmixing workbench inheriting full viewer capabilities.
- **Spectral Reference Library**: Load, transform, and manage reference dye/fluorophore spectra (CSV, XLSX, MAT, NPY, TXT).
- **Unmixing & Decomposition Algorithms**:
  - **NNLS & FCLS**: Non-Negative Least Squares and Fully Constrained Least Squares ($\sum C_k = 1$).
  - **MCR-ALS**: Multivariate Curve Resolution - Alternating Least Squares (with non-negativity and sum-to-one constraints).
  - **NMF**: Non-Negative Matrix Factorization (scikit-learn).
  - **MESMA**: Multiple Endmember Spectral Mixture Analysis.
  - **SAM & SID**: Spectral Angle Mapper and Spectral Information Divergence classification.
  - **SVR**: Support Vector Regression unmixing.
  - **RX Anomaly Detection**: Spatial Mahalanobis distance anomaly finder.
  - **PCA, MNF & ICA**: Principal Component Analysis, Minimum Noise Fraction (SNR-ordered), and Independent Component Analysis.
- **Denoising**: 3D Total Variation (3D-TV), MPPCA (Marchenko-Pastur PCA), Savitzky-Golay 1D spectral polynomial filter, and Wavelet (VisuShrink) filtering.
- **Spectral Normalization**: Real-time switching between Raw Intensity, Total Area, Peak Height, L2 Vector Norm, and Standard Normal Variate (SNV Z-Score).
- **Endmember Extraction**: PPI (Pixel Purity Index), VCA (Vertex Component Analysis), N-FINDR, and Interactive Endmember Picker.
- **Peak Fitting**: Gaussian and Lorentzian spectral curve fitting.
- **Composites**: RGB color synthesis and PC-RGB (first 3 Principal Components) pseudo-color mapping.
- **Interactive Guide & Standardized Display**: F1 User Guide detailing all algorithms, clean 3-column dual-list UI (`ROI Spectra` / `Basis Spectra`), and synchronized **`↕ Autoscale Y`** canvas toolbars across all suite applications.

### 3. `OPDviewer/OPDViewer.py`
Extends `HyperViewer` for liquid-crystal retardance measurement, voltage-sweep conversion, and forward physics simulations.
- **Voltage Metadata Extraction**: Automatically parses per-frame drive voltages ($V$) from TIFF ImageJ metadata (`Labels`) or sidecar files (`_voltages.npy` / `.csv`).
- **Voltage X-Axis**: Displays **Drive Voltage (V)** on the spectrum plot x-axis instead of arbitrary frame index.
- **Convert to OPD**: Converts intensity modulation $I(x, y, V)$ into absolute Optical Path Difference $\text{OPD}(x, y, V)$ in nanometers via phase-unwrapping, fold counting, and spatial continuity constraints.
- **Load OPD**: Opens previously saved `.npy`, `.tif`, `.tiff`, or `.mat` OPD stacks.
- **Revert / Simulate Intensity**: Converts any OPD map stack back to an intensity map stack $I(x, y, V) = \sum_\lambda S(\lambda) \sin^2(\pi \cdot \text{OPD} / \lambda)$ for a monochromatic wavelength $\lambda_0$ or polychromatic illumination spectrum $S(\lambda)$ (crossed or parallel PBS configuration).
- **Save OPD**: Exports processed OPD cubes as TIFF stacks with embedded per-frame drive voltage metadata (`Labels`) or NumPy arrays with sidecar `_voltages.npy` files.

### 4. `knifedge.py`
3D Focus Depth & Knife-Edge Sharpness Analysis Application based on `HyperViewer`.
- **Dual 3D Dataset Loading**: Loads Dataset A (Primary) and Dataset B (Comparison) for focus separation $\Delta Z$ analysis.
- **Spatial View Switcher**: Toggles 3D spatial rendering between Dataset A and Dataset B (`Spatial 3D: Dataset A | Dataset B`).
- **Interactive 1D Line Profiles & Rectangle ROIs**: Extracts 1D raw intensity profiles $I(x)$ or $I(y)$ along active crosshair row/column or rectangular ribbon ROIs with automatic ribbon averaging.
- **Physical Spatial Calibration ($\mu\text{m}$ Scale)**: Inputs spatial pixel sizes ($X, Y, Z$ in $\mu\text{m}$) directly on the image navigation toolbar to scale 1D line profile axes, fitted edge centers $x_0$, FWHM measurements, and $Z$ focus depth curves into physical units ($\mu\text{m}$).
- **Automatic Parameter Memory (`QSettings`)**: Persistent session memory that automatically saves and reloads **Fit Window**, **Min Edge Distance**, **Peak Height Threshold**, **Pixel Sizes ($X, Y, Z$)**, **Line Width**, **Smooth Sigma**, and **Fit Model** across application restarts.
- **Theoretical Knife-Edge & High-NA Optics Calculator**: Action button launching a theoretical diffraction pop-up supporting:
  - *Airy Disk / Circular Aperture (Realistic High NA)*: $w_0 = 0.514 \lambda / \text{NA}$, $\text{FWHM}_0 = 0.510 \lambda / \text{NA}$.
  - *Richards-Wolf Vectorial Diffraction (High NA $\ge 0.70$)*: Includes depolarization scaling factor $\sqrt{1 - 0.25 (\text{NA}/n)^2}$.
  - *Paraxial Untruncated Gaussian (Idealized)*: $w_0 = \lambda / (\pi \cdot \text{NA})$.
- **Multi-Edge FWHM vs Z Focus Curves**: Plots edge blurring ($\text{FWHM}$) across all $Z$ focus frames using high-contrast colors (`#D32F2F`, `#1976D2`, `#F57C00`) and clean circle markers (`o-` / `o--`), automatically detecting sharpest focus position $Z_{\text{focus}}$ and focus separation $\Delta Z$.
- **Plot & Data Export (PNG, SVG, CSV)**: Dedicated export buttons for theoretical diffraction curves and experimental focus depth curves, saving high-resolution PNG/SVG figures or raw CSV numerical data tables.
- **Full-Grid 2D Focus Map**: Automatically detects grid edges across the image and renders an interpolated 2D heatmap $Z_{\text{min}}(x, y)$.

---

## 2. User Interface & Controls

### Spatial Image & Side Views
- **Spatial View (Center)**: Displays the 2D image at the selected spectral frame or voltage. Axis ticks display pixel coordinates ($X, Y$).
- **XZ View (Top)**: Shows the horizontal cross-section at the red crosshair $Y$ position versus spectral wavelength / voltage. Its width is locked 100% to the spatial image.
- **YZ View (Left)**: Shows the vertical cross-section at the red crosshair $X$ position versus spectral wavelength / voltage.
- **Colorbar (Right)**: Shows current intensity / OPD mapping.

### ROI Selection Tools
Select an ROI tool from the left panel:
- **Point ($+$)**: Single-pixel extraction. Displays crosshair lines at click position.
- **Circle ($\bigcirc$)**: Click and drag to create a circular ROI outline. Extracts mean spectrum and standard deviation envelope within the circle.
- **Square ($\square$)**: Click and drag to create a square ROI outline. Extracts mean spectrum and standard deviation envelope within the square.
- **Add Current Selection**: Saves the active ROI to the spectrum list and renders its open outline marker permanently on the spatial image.
- **Delete / Visibility**: Check/uncheck boxes to toggle individual spectrum visibility, click `×` to delete single spectra, or click **Delete Selected** to clear multiple selections. All spatial markers update in real time.

### Frame Navigation & Spatial Calibration Toolbar (Bottom Bar)
- **Step Arrow Buttons (◀ / ▶)**: Step forward or backward by single frames/Z-planes using mouse or arrow buttons.
- **Horizontal Slider & Direct Spinbox**: Drag the slider or type an exact frame index into the numeric spinbox to jump directly to any frame.
- **Spatial Calibration Inputs ($X, Y, Z$ in $\mu\text{m}$)**: Enter physical pixel sizes ($X$, $Y$ in $\mu\text{m}/\text{px}$) and $Z$ step size ($\mu\text{m}/\text{frame}$) to automatically convert all line profile positions, FWHM edge blur widths, and focus depth curves to physical micrometers ($\mu\text{m}$).
- **Built-in Matplotlib Zoom & Pan**: Integrated navigation toolbar directly under the spatial canvas providing rubber-band **Zoom 🔍** and hand **Pan ✋** tools for interactive spatial exploration.
- **Unified Views**: Changing the colormap (e.g. `gray`, `viridis`, `plasma`, `jet`, `inferno`) applies the colormap and exact intensity limits simultaneously across **XY spatial view**, **XZ top side view**, and **YZ left side view**.
- **Editable Color Limits (`C-Min` / `C-Max`)**: Type or adjust values in `C-Min` and `C-Max` spinboxes at the bottom of the spatial view to manually set contrast and color mapping limits across all 3 views and the colorbar.
- **Auto Scale**: Click `Auto Scale` to reset color limits to the current frame's automatic minimum and maximum intensity.

### Spatial Resample (`Resample ▾`)
- **Compact Dropdown Button**: Single `Resample ▾` dropdown button on the top toolbar for rapid spatial resolution scaling.
- **Downscale (1/2, 1/3, 1/4)**: Uses $k \times k$ block-averaging to reduce spatial dimensions, shrinking pixel count by up to 16x and accelerating unmixing and OPD calculations by up to 1600%.
- **Upscale (2x, 3x, 4x)**: Uses spatial bilinear interpolation to enlarge the image grid.

---

## 3. File Formats & Compatibility

Supported input files:
- **TIFF Stacks** (`.tif`, `.tiff`): 3D multi-page TIFF images.
- **NumPy Binary** (`.npy`): 3D arrays of shape $(H, W, \text{Bands})$ or $(N, H, W)$.
- **MATLAB Data** (`.mat`): 3D arrays containing spectral hypercubes.
- **Reference Spectra Files** (`.csv`, `.xlsx`, `.mat`, `.npy`, `.txt`): Spectral library files formatted with wavelengths in first row/column and fluorophore/dye names.

---

## 4. Launch Commands

Run any viewer directly from PowerShell or Command Prompt:

```powershell
# 1. Launch Hyperspectral Unmixing Workbench
python unmix\Unmixer.py

# 2. Launch 3D Knife-Edge Focus Sharpness Analysis
python knifedge.py

# 3. Launch Hyperspectral Image Viewer
python HyperViewer.py

# 4. Launch Liquid Crystal OPD Retardance Viewer
python OPDviewer\OPDViewer.py
```

---

## 5. Keyboard Shortcuts

| Shortcut | Function | Application |
| :--- | :--- | :--- |
| `Ctrl + O` | Open File / Dataset | All |
| `Ctrl + T` | Open TIFF Stack | Unmixer |
| `Ctrl + E` | Export Unmixing Results (CSV) | Unmixer |
| `Ctrl + M` | Run MCR-ALS Unmixing | Unmixer |
| `Ctrl + N` | Run NMF Unmixing | Unmixer |
| `Ctrl + I` | Toggle Interactive Endmember Picker | Unmixer |
| `Ctrl + G` | Run Gaussian Peak Fit | Unmixer |
| `Ctrl + L` | Run Lorentzian Peak Fit | Unmixer |
| `F1` | Open User Guide / Documentation | All |
| `Ctrl + Q` | Exit Application | All |

---

*© 2026 Multiplexing Research Suite*
