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
- **Unmixing Algorithms**:
  - **MCR-ALS**: Multivariate Curve Resolution - Alternating Least Squares (with non-negativity constraints).
  - **NMF**: Non-Negative Matrix Factorization (scikit-learn).
  - **MESMA**: Multiple Endmember Spectral Mixture Analysis.
  - **SAM & SID**: Spectral Angle Mapper and Spectral Information Divergence classification.
  - **SVR**: Support Vector Regression unmixing.
- **Denoising**: MPPCA (Marchenko-Pastur PCA) and Wavelet (VisuShrink) filtering.
- **Endmember Extraction**: PPI (Pixel Purity Index), VCA (Vertex Component Analysis), N-FINDR, and Interactive Endmember Picker.
- **Peak Fitting**: Gaussian and Lorentzian spectral curve fitting.
- **Composites**: RGB color synthesis and PC-RGB (first 3 Principal Components) pseudo-color mapping.

### 3. `OPDviewer/OPDViewer.py`
Extends `HyperViewer` for liquid-crystal retardance measurement and voltage-sweep conversion.
- **Voltage Metadata Extraction**: Automatically parses per-frame drive voltages ($V$) from TIFF ImageJ metadata (`Labels`) or sidecar files (`_voltages.npy` / `.csv`).
- **Voltage X-Axis**: Displays **Drive Voltage (V)** on the spectrum plot x-axis instead of arbitrary frame index.
- **Convert to OPD**: Converts intensity modulation $I(x, y, V)$ into absolute Optical Path Difference $\text{OPD}(x, y, V)$ in nanometers via phase-unwrapping, fold counting, and spatial continuity constraints.
- **Revert & Save**: Toggle back to raw intensity at any time and export processed OPD cubes as NumPy arrays or TIFF stacks.

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

### Color Scale & Contrast Editing
- **Unified Views**: Changing the colormap (e.g. `gray`, `viridis`, `plasma`, `jet`, `inferno`) applies the colormap and exact intensity limits simultaneously across **XY spatial view**, **XZ top side view**, and **YZ left side view**.
- **Editable Color Limits (`C-Min` / `C-Max`)**: Type or adjust values in the `C-Min` and `C-Max` spinboxes to manually set contrast and color mapping limits across all 3 views and the colorbar.
- **Auto Scale**: Click `Auto Scale` to reset the color limits to the current frame's automatic minimum and maximum intensity.

### Spatial Resample (Downscale / Upscale)
- **Downscale (1/2, 1/3, 1/4)**: Uses $k \times k$ block-averaging to reduce spatial dimensions, shrinking pixel count by up to 16x and accelerating unmixing and OPD calculations by up to 1600%.
- **Upscale (2x, 3x, 4x)**: Uses spatial bilinear interpolation to enlarge the image grid.

### Spectrum Plot Controls
- **🔒 Lock Axes**: Freezes current $X$ and $Y$ axis limits during frame switching.
- **⤡ Autoscale**: Automatically scales spectrum plot axes to fit all visible spectra with a 5% margin.
- **🔍 Zoom**: Toggles interactive rectangle zoom on the spectrum plot.

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

# 2. Launch Hyperspectral Image Viewer
python HyperViewer.py

# 3. Launch Liquid Crystal OPD Retardance Viewer
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
