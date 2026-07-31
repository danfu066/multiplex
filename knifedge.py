#!/usr/bin/env python3
# Copyright (c) 2026 Dan Fu@UW
"""
knifedge.py - 3D Focus & Knife-Edge Sharpness Analysis Tool

Features:
- Dual 3D Focus Stack Loading (Dataset A & Dataset B) for focus separation comparison.
- Interactive 1D Line Profiles (X Row vs Y Column) overlaying Dataset A & B.
- Interactive Gaussian Edge Derivative Fit Inspection Window showing g(x) overlay, FWHM, and R^2 fit quality.
- FWHM vs Z Focus Depth curves with automatic minimum FWHM (sharpest focus) detection for Dataset A & B.
- Full-Grid Automated Focus Map across (X, Y) image space.
"""

import sys
import os
import re
import numpy as np
from pathlib import Path

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
    QGroupBox, QFileDialog, QMessageBox, QProgressDialog, QSplitter, QFrame, QAction, QDialog,
    QDialogButtonBox, QTextEdit
)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy import signal
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
import scipy.special as special

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    from skimage import io, measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from HyperViewer import HyperViewer


def extract_line_profile(cube, mode, cross_x, cross_y, z_frame, line_width=1, rect_coords=None):
    """
    Extract 1D line profile from 3D stack cube (H, W, Z) at frame z_frame.
    If rect_coords=(x1, y1, w_rect, h_rect) is provided, extracts exact rectangle ROI
    and averages along the shorter dimension.
    Otherwise, if line_width > 1, averages a ribbon of line_width pixels perpendicular to crosshair line.
    """
    if cube is None or z_frame >= cube.shape[2]:
        return None
    H, W, Z = cube.shape

    if rect_coords is not None:
        x1, y1, w_rect, h_rect = rect_coords
        x1 = int(np.clip(x1, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        x2 = int(np.clip(x1 + w_rect, x1 + 1, W))
        y2 = int(np.clip(y1 + h_rect, y1 + 1, H))
        stripe = cube[y1:y2, x1:x2, z_frame]

        if h_rect < w_rect:
            # Horizontal rectangle -> X profile of length W_rect
            return np.mean(stripe, axis=0) if stripe.ndim == 2 else stripe
        else:
            # Vertical rectangle -> Y profile of length H_rect
            return np.mean(stripe, axis=1) if stripe.ndim == 2 else stripe

    w_half = max(0, int(line_width // 2))

    if mode == "X":
        y_center = int(np.clip(cross_y, 0, H - 1))
        y_start = max(0, y_center - w_half)
        y_end = min(H, y_center + w_half + 1)
        stripe = cube[y_start:y_end, :, z_frame]
        return np.mean(stripe, axis=0) if stripe.ndim == 2 else stripe
    else:
        x_center = int(np.clip(cross_x, 0, W - 1))
        x_start = max(0, x_center - w_half)
        x_end = min(W, x_center + w_half + 1)
        stripe = cube[:, x_start:x_end, z_frame]
        return np.mean(stripe, axis=1) if stripe.ndim == 2 else stripe


def gaussian_func(x, a, x0, sigma, offset):
    """Gaussian function for edge derivative fitting."""
    return a * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2)) + offset


def erf_func(x, a, x0, sigma, offset):
    """Error function (erf) for direct raw intensity step edge fitting."""
    return a * special.erf((x - x0) / (np.sqrt(2.0) * sigma)) + offset


def calculate_erf_fit(profile_1d, start_idx, window=25):
    """Fit Error Function (erf) directly to raw intensity profile around start_idx for rising or falling edges."""
    profile_1d = np.asarray(profile_1d, dtype=float)
    N = len(profile_1d)
    if N < 5:
        return np.nan, None, 0.0, None, None

    start_idx = int(np.clip(start_idx, 0, N - 1))
    start = max(0, start_idx - window // 2)
    end = min(N, start_idx + window // 2 + 1)

    x = np.arange(start, end, dtype=float)
    y = profile_1d[start:end]

    if len(x) < 5 or np.all(y == y[0]):
        return np.nan, None, 0.0, None, None

    # Determine edge direction (rising vs falling) from local slope / end-to-end trend
    y_diff = y[-1] - y[0]
    grad_mean = np.mean(np.gradient(y))
    amp_mag = float((np.max(y) - np.min(y)) / 2.0)

    if y_diff < 0 or grad_mean < 0:
        a0 = -amp_mag  # Falling edge (high -> low)
    else:
        a0 = +amp_mag  # Rising edge (low -> high)

    x0 = float(start_idx)
    sigma0 = float(max(1.0, min(20.0, window / 4.0)))
    offset0 = float(np.mean(y))
    p0 = [a0, x0, sigma0, offset0]

    bounds = (
        [-np.inf, start, 0.2, -np.inf],
        [np.inf, end, window * 2.0, np.inf]
    )

    try:
        popt, _ = curve_fit(erf_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
        sigma = abs(popt[2])
        fwhm = float(2.35482 * sigma)

        y_pred = erf_func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        x_dense = np.linspace(start, end, 200)
        y_dense = erf_func(x_dense, *popt)
        return fwhm, popt, r2, x_dense, y_dense
    except Exception:
        return np.nan, None, 0.0, None, None


def deduplicate_edge_fits(fits_list, min_center_distance=35.0):
    """
    Deduplicate overlapping edge fits that belong to the same physical edge.
    Sorts candidate fits by R^2 fit quality descending, then keeps only the highest R^2 fit
    for each spatial cluster of centers x0.
    """
    valid_fits = [f for f in fits_list if f[1] is not None and np.isfinite(f[0]) and f[2] > 0.15]
    if not valid_fits:
        return []

    # Sort fits by R^2 quality descending (highest quality fit first)
    valid_fits.sort(key=lambda item: item[2], reverse=True)

    clean_fits = []
    for fit in valid_fits:
        center_x0 = fit[1][1]
        if not any(abs(center_x0 - c[1][1]) < min_center_distance for c in clean_fits):
            clean_fits.append(fit)

    # Re-sort clean fits spatially along position
    clean_fits.sort(key=lambda item: item[1][1])
    return clean_fits


class PeakDiscoveryDiagnosticsDialog(QDialog):
    """Pop-up dialog displaying the pre-smoothed curve, coarse derivative, candidate peaks, and deduplicated edge fits for troubleshooting."""

    def __init__(self, parent, prof, name="Dataset A"):
        super().__init__(parent)
        self.setWindowTitle(f"Peak Discovery Diagnostics - {name}")
        self.setMinimumSize(900, 620)
        self.prof = prof
        self.name = name

        vbox = QVBoxLayout(self)

        # Control panel: Peak threshold & distance
        cfg_layout = QHBoxLayout()
        cfg_layout.addWidget(QLabel("Peak Height Threshold (% of Max):"))
        self.thresh_spin = QSpinBox()
        self.thresh_spin.setRange(5, 90)
        init_thresh = parent.peak_thresh_spin.value() if hasattr(parent, 'peak_thresh_spin') else 20
        self.thresh_spin.setValue(init_thresh)
        self.thresh_spin.setSingleStep(5)
        self.thresh_spin.setToolTip("Minimum peak height threshold percentage relative to max derivative peak")
        self.thresh_spin.valueChanged.connect(self._on_param_changed)
        cfg_layout.addWidget(self.thresh_spin)

        cfg_layout.addWidget(QLabel("Min Peak Distance (px):"))
        self.dist_spin = QSpinBox()
        self.dist_spin.setRange(5, 200)
        init_dist = parent.min_dist_spin.value() if hasattr(parent, 'min_dist_spin') else 35
        self.dist_spin.setValue(init_dist)
        self.dist_spin.setSingleStep(5)
        self.dist_spin.setToolTip("Minimum spatial separation distance between candidate peaks")
        self.dist_spin.valueChanged.connect(self._on_param_changed)
        cfg_layout.addWidget(self.dist_spin)

        cfg_layout.addWidget(QLabel("Fit Window (px):"))
        self.fit_win_spin = QSpinBox()
        self.fit_win_spin.setRange(5, 2000)
        init_win = parent.fit_win_spin.value() if hasattr(parent, 'fit_win_spin') else 25
        self.fit_win_spin.setValue(init_win)
        self.fit_win_spin.setSingleStep(5)
        self.fit_win_spin.setToolTip("Sub-window size (in pixels) centered on each edge for local curve fitting")
        self.fit_win_spin.valueChanged.connect(self._on_param_changed)
        cfg_layout.addWidget(self.fit_win_spin)

        cfg_layout.addStretch()
        vbox.addLayout(cfg_layout)

        # Matplotlib figure with 2 subplots
        self.fig = Figure(figsize=(8.5, 5.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas)

        self.update_plot()

    def _on_param_changed(self):
        if hasattr(self.parent(), 'peak_thresh_spin'):
            self.parent().peak_thresh_spin.setValue(self.thresh_spin.value())
        if hasattr(self.parent(), 'min_dist_spin'):
            self.parent().min_dist_spin.setValue(self.dist_spin.value())
        if hasattr(self.parent(), 'fit_win_spin'):
            self.parent().fit_win_spin.setValue(self.fit_win_spin.value())
        self.update_plot()

    def update_plot(self):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212, sharex=ax1)

        prof = self.prof
        x_vals = np.arange(len(prof))

        # Pre-smooth profile for candidate peak discovery
        prof_smooth = gaussian_filter1d(prof, sigma=4.0)
        deriv_coarse = np.abs(np.gradient(prof_smooth, 5.0))
        max_d = np.max(deriv_coarse) if len(deriv_coarse) > 0 else 1.0

        thresh_pct = self.thresh_spin.value() / 100.0
        min_dist = self.dist_spin.value()
        is_rect = getattr(self.parent(), 'is_rectangle_roi_active', False)
        win = len(prof) if is_rect else self.fit_win_spin.value()

        peaks_cand, _ = signal.find_peaks(deriv_coarse, distance=min_dist, height=max_d * thresh_pct)
        cand_fits = [calculate_erf_fit(prof, pk, window=win) for pk in peaks_cand]
        clean_fits = deduplicate_edge_fits(cand_fits, min_center_distance=float(min_dist))

        px_scale = self.parent().get_pixel_scale() if hasattr(self.parent(), 'get_pixel_scale') else 1.0
        unit_str = "µm" if px_scale != 1.0 else "px"
        x_vals_um = x_vals * px_scale

        # Subplot 1: Raw Intensity Profile + Pre-Smoothed Curve + ERF Fits
        ax1.plot(x_vals_um, prof, '.', color='#1565C0', alpha=0.30, label="Raw Intensity I")
        ax1.plot(x_vals_um, prof_smooth, '-', color='#37474F', linewidth=1.5, label="Pre-Smoothed I (sigma=4)")
        
        for i, (fw, popt, r2, x_fit, y_fit) in enumerate(clean_fits):
            if popt is not None:
                edge_dir = "Rising" if popt[0] > 0 else "Falling"
                x_fit_um = x_fit * px_scale
                x0_um = popt[1] * px_scale
                lbl_str = f"Edge #{i+1} ({edge_dir}) x0={x0_um:.2f}{unit_str}" if px_scale != 1.0 else f"Edge #{i+1} ({edge_dir}) x0={popt[1]:.1f}px"
                ax1.plot(x_fit_um, y_fit, '-', color='#FF1744', linewidth=2.5, label=lbl_str if i < 6 else "")
                ax1.axvline(x0_um, color='#FF1744', linestyle='--', alpha=0.6)

        ax1.set_ylabel("Intensity")
        ax1.set_title(f"1. Raw Profile & Direct ERF Fits ({len(clean_fits)} Physical Edges Fitted)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)

        # Subplot 2: Coarse Derivative + Candidate Peaks vs Deduplicated Edges
        ax2.plot(x_vals_um, deriv_coarse, color='darkorange', linewidth=1.5, label="Coarse Deriv |dI_smooth/dx| (step=5)")
        ax2.axhline(max_d * thresh_pct, color='red', linestyle=':', label=f"Height Threshold ({self.thresh_spin.value()}%)")

        if len(peaks_cand) > 0:
            ax2.plot(peaks_cand * px_scale, deriv_coarse[peaks_cand], 'rx', markersize=8, markeredgewidth=2, label=f"Candidate Peaks ({len(peaks_cand)})")

        for i, (fw, popt, r2, x_fit, y_fit) in enumerate(clean_fits):
            if popt is not None:
                ax2.axvline(popt[1] * px_scale, color='green', linestyle='-', linewidth=2, label=f"Final Edge #{i+1}" if i == 0 else "")

        ax2.set_xlabel(f"Position ({unit_str})")
        ax2.set_ylabel("|dI/dx|")
        ax2.set_title(f"2. Coarse Derivative & Candidate Peak Discovery ({len(peaks_cand)} Candidate Peaks -> {len(clean_fits)} Final Edges)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw_idle()


class TheoreticalKnifeEdgeDialog(QDialog):
    """Interactive pop-up window to calculate and plot theoretical diffraction-limited knife-edge ERF profiles and FWHM(z) defocus curves for given laser wavelength & NA."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Theoretical Knife-Edge & Beam Waist Calculator")
        self.setMinimumSize(920, 650)

        vbox = QVBoxLayout(self)

        # Header controls
        cfg_layout = QHBoxLayout()

        cfg_layout.addWidget(QLabel("Laser Wavelength λ (nm):"))
        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(100.0, 3000.0)
        self.wavelength_spin.setValue(785.0)
        self.wavelength_spin.setSingleStep(5.0)
        self.wavelength_spin.setDecimals(1)
        self.wavelength_spin.setToolTip("Laser excitation/probe wavelength in nanometers")
        self.wavelength_spin.valueChanged.connect(self.update_plot)
        cfg_layout.addWidget(self.wavelength_spin)

        cfg_layout.addWidget(QLabel("Objective NA:"))
        self.na_spin = QDoubleSpinBox()
        self.na_spin.setRange(0.01, 1.70)
        self.na_spin.setValue(0.60)
        self.na_spin.setSingleStep(0.05)
        self.na_spin.setDecimals(2)
        self.na_spin.setToolTip("Numerical Aperture (NA) of objective lens")
        self.na_spin.valueChanged.connect(self.update_plot)
        cfg_layout.addWidget(self.na_spin)

        cfg_layout.addWidget(QLabel("Refractive Index n:"))
        self.n_index_spin = QDoubleSpinBox()
        self.n_index_spin.setRange(1.00, 2.00)
        self.n_index_spin.setValue(1.00)
        self.n_index_spin.setSingleStep(0.05)
        self.n_index_spin.setDecimals(2)
        self.n_index_spin.setToolTip("Medium refractive index (1.00 for Air, 1.33 for Water, 1.51 for Oil)")
        self.n_index_spin.valueChanged.connect(self.update_plot)
        cfg_layout.addWidget(self.n_index_spin)

        cfg_layout.addWidget(QLabel("Formulation:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Airy Disk / Circular Aperture (Realistic High NA - Recommended)",
            "Richards-Wolf Vectorial (High NA >= 0.70)",
            "Paraxial Untruncated Gaussian (Idealized)"
        ])
        self.model_combo.setToolTip("Switch optics formulation for high NA objective lenses")
        self.model_combo.currentIndexChanged.connect(self.update_plot)
        cfg_layout.addWidget(self.model_combo)

        cfg_layout.addStretch()
        vbox.addLayout(cfg_layout)

        # Matplotlib figure with 2 subplots
        self.fig = Figure(figsize=(9.0, 5.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas)

        # Output Summary Box
        self.info_lbl = QLabel()
        self.info_lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #0D47A1; background-color: #E3F2FD; padding: 6px; border-radius: 4px;")
        vbox.addWidget(self.info_lbl)

        # Action Buttons Row (Export & Close)
        btn_bar = QHBoxLayout()
        self.btn_export = QPushButton("Export Plot / Data (PNG, SVG, CSV)")
        self.btn_export.setStyleSheet("font-weight: bold; padding: 5px 12px;")
        self.btn_export.setToolTip("Export plot image (PNG, SVG) or numerical curve data table (CSV)")
        self.btn_export.clicked.connect(self.export_plot_or_data)
        btn_bar.addWidget(self.btn_export)

        btn_bar.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(btn_close)

        vbox.addLayout(btn_bar)

        # Load QSettings if saved
        settings = QSettings("KnifeEdgeApp", "KnifeEdgeViewer")
        self.wavelength_spin.setValue(settings.value("theo_wavelength", 785.0, type=float))
        self.na_spin.setValue(settings.value("theo_na", 0.60, type=float))
        self.n_index_spin.setValue(settings.value("theo_n_index", 1.00, type=float))
        idx_m = settings.value("theo_model_idx", 0, type=int)
        if idx_m < self.model_combo.count():
            self.model_combo.setCurrentIndex(idx_m)

        self.update_plot()

    def export_plot_or_data(self):
        """Export theoretical knife-edge plot as PNG/SVG or numerical data as CSV."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Theoretical Knife-Edge Plot or CSV Data",
            "theoretical_knife_edge_plot.png",
            "PNG Image (*.png);;Vector SVG (*.svg);;CSV Data Table (*.csv)"
        )
        if not filepath:
            return

        ext = Path(filepath).suffix.lower()
        if ext == '.csv':
            lam_nm = self.wavelength_spin.value()
            lam_um = lam_nm / 1000.0
            na = self.na_spin.value()
            n_idx = self.n_index_spin.value()
            model_name = self.model_combo.currentText()
            sin_theta = np.clip(na / n_idx, 0.001, 0.999)

            if "Airy Disk" in model_name:
                w0_um = (0.514 * lam_um) / na
                fwhm0_um = (0.510 * lam_um) / na
                zR_um = (n_idx * lam_um) / (na ** 2)
            elif "Richards-Wolf" in model_name:
                vector_factor = 1.0 / np.sqrt(1.0 - 0.25 * (sin_theta ** 2))
                w0_um = (0.514 * lam_um * vector_factor) / na
                fwhm0_um = (0.510 * lam_um * vector_factor) / na
                zR_um = (n_idx * lam_um * vector_factor) / (na ** 2)
            else:
                w0_um = lam_um / (np.pi * na)
                fwhm0_um = np.sqrt(2.0 * np.log(2.0)) * w0_um
                zR_um = (np.pi * (w0_um ** 2)) / (lam_um / n_idx)

            px_scale = self.parent().get_pixel_scale() if hasattr(self.parent(), 'get_pixel_scale') else 1.0
            span_x_um = max(10.0, 10.0 * w0_um)
            x_um = np.linspace(-span_x_um, span_x_um, 500)
            erf_profile = 0.5 * (1.0 + special.erf(np.sqrt(2.0) * x_um / w0_um))
            
            span_z_um = max(5.0, 5.0 * zR_um)
            z_um = np.linspace(-span_z_um, span_z_um, 500)
            fwhm_z_um = fwhm0_um * np.sqrt(1.0 + (z_um / zR_um) ** 2)

            with open(filepath, 'w') as f:
                f.write(f"# Theoretical Knife-Edge Parameters: Wavelength={lam_nm}nm, NA={na}, n={n_idx}, Formulation={model_name}\n")
                f.write(f"# Calculated: w0={w0_um:.4f}um, FWHM0={fwhm0_um:.4f}um, zR={zR_um:.4f}um, PixelScale={px_scale}um/px\n")
                f.write("Position_x_um,Position_x_px,Theoretical_ERF_Intensity,Defocus_z_um,Theoretical_FWHM_um\n")
                for i in range(len(x_um)):
                    f.write(f"{x_um[i]:.4f},{x_um[i]/px_scale:.3f},{erf_profile[i]:.6f},{z_um[i]:.4f},{fwhm_z_um[i]:.4f}\n")
            QMessageBox.information(self, "Export Successful", f"Saved theoretical CSV data to:\n{filepath}")
        else:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", f"Saved theoretical plot image to:\n{filepath}")
        self.na_spin.setValue(settings.value("theo_na", 0.60, type=float))
        self.n_index_spin.setValue(settings.value("theo_n_index", 1.00, type=float))
        idx_m = settings.value("theo_model_idx", 0, type=int)
        if idx_m < self.model_combo.count():
            self.model_combo.setCurrentIndex(idx_m)

        self.update_plot()

    def closeEvent(self, event):
        settings = QSettings("KnifeEdgeApp", "KnifeEdgeViewer")
        settings.setValue("theo_wavelength", self.wavelength_spin.value())
        settings.setValue("theo_na", self.na_spin.value())
        settings.setValue("theo_n_index", self.n_index_spin.value())
        settings.setValue("theo_model_idx", self.model_combo.currentIndex())
        super().closeEvent(event)

    def update_plot(self):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

        lam_nm = self.wavelength_spin.value()
        lam_um = lam_nm / 1000.0
        na = self.na_spin.value()
        n_idx = self.n_index_spin.value()
        model_name = self.model_combo.currentText()

        sin_theta = np.clip(na / n_idx, 0.001, 0.999)

        if "Airy Disk" in model_name:
            # Airy disk circular aperture diffraction limit
            w0_um = (0.514 * lam_um) / na
            fwhm0_um = (0.510 * lam_um) / na
            w10_90_um = (0.860 * lam_um) / na
            zR_um = (n_idx * lam_um) / (na ** 2)
            formula_tag = "Airy Disk / Circular Aperture"
        elif "Richards-Wolf" in model_name:
            # Richards-Wolf vectorial diffraction for High NA
            vector_factor = 1.0 / np.sqrt(1.0 - 0.25 * (sin_theta ** 2))
            w0_um = (0.514 * lam_um * vector_factor) / na
            fwhm0_um = (0.510 * lam_um * vector_factor) / na
            w10_90_um = (0.860 * lam_um * vector_factor) / na
            zR_um = (n_idx * lam_um * vector_factor) / (na ** 2)
            formula_tag = "Richards-Wolf Vectorial High-NA"
        else:
            # Untruncated paraxial Gaussian
            w0_um = lam_um / (np.pi * na)
            fwhm0_um = np.sqrt(2.0 * np.log(2.0)) * w0_um
            w10_90_um = 1.683 * w0_um
            zR_um = (np.pi * (w0_um ** 2)) / (lam_um / n_idx)
            formula_tag = "Paraxial Gaussian"

        px_scale = self.parent().get_pixel_scale() if hasattr(self.parent(), 'get_pixel_scale') else 1.0
        w0_px = w0_um / px_scale
        fwhm0_px = fwhm0_um / px_scale

        # 1. 1D Theoretical ERF Profile
        span_x_um = max(10.0, 10.0 * w0_um)
        x_um = np.linspace(-span_x_um, span_x_um, 500)
        erf_profile = 0.5 * (1.0 + special.erf(np.sqrt(2.0) * x_um / w0_um))
        deriv_theoretical = np.exp(-2.0 * (x_um ** 2) / (w0_um ** 2)) / (np.sqrt(np.pi / 2.0) * w0_um)

        ax1.plot(x_um, erf_profile, 'b-', linewidth=2, label=f"Theoretical ERF Edge Profile [{formula_tag}] (λ={lam_nm:.0f}nm, NA={na:.2f})")
        ax1.plot(x_um, deriv_theoretical / np.max(deriv_theoretical), 'r--', linewidth=1.5, label=f"Normalized Deriv Gaussian (FWHM={fwhm0_um:.3f}µm / {fwhm0_px:.2f}px)")
        ax1.set_xlabel(f"Position relative to Edge Center (µm)")
        ax1.set_ylabel("Normalized Intensity / Deriv")
        ax1.set_title(f"1. Theoretical Knife-Edge ERF Profile [{formula_tag}]")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)

        # 2. Defocus FWHM vs Z Curve
        span_z_um = max(5.0, 5.0 * zR_um)
        z_um = np.linspace(-span_z_um, span_z_um, 300)
        fwhm_z_um = fwhm0_um * np.sqrt(1.0 + (z_um / zR_um) ** 2)

        ax2.plot(z_um, fwhm_z_um, 'g-', linewidth=2, label=f"Theoretical FWHM(z) Defocus Curve (zR={zR_um:.3f}µm)")
        ax2.axvline(0, color='gray', linestyle=':', alpha=0.6)
        ax2.axhline(fwhm0_um, color='green', linestyle=':', label=f"Min FWHM0 = {fwhm0_um:.3f} µm ({fwhm0_px:.2f} px)")
        ax2.axvline(zR_um, color='orange', linestyle='--', alpha=0.6, label=f"+zR = {zR_um:.3f} µm")
        ax2.axvline(-zR_um, color='orange', linestyle='--', alpha=0.6)

        ax2.set_xlabel("Z Focus Depth Defocus Position (µm)")
        ax2.set_ylabel("Edge Blurring FWHM (µm)")
        ax2.set_ylim(bottom=0)
        ax2.set_title(f"2. Theoretical Focus Depth Defocus Curve [{formula_tag}]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)

        # Information Summary Text
        info_str = (
            f"Formulation: <b>{formula_tag}</b> &nbsp;|&nbsp; "
            f"Beam Waist w₀: <b>{w0_um:.3f} µm</b> ({w0_px:.2f} px) &nbsp;|&nbsp; "
            f"Edge FWHM₀: <b>{fwhm0_um:.3f} µm</b> ({fwhm0_px:.2f} px) &nbsp;|&nbsp; "
            f"10%–90% Rise: <b>{w10_90_um:.3f} µm</b> &nbsp;|&nbsp; "
            f"Rayleigh Range z<sub>R</sub>: <b>{zR_um:.3f} µm</b>"
        )
        self.info_lbl.setText(info_str)

        self.fig.tight_layout()
        self.canvas.draw_idle()


def calculate_fwhm(signal_1d, peak_idx, window=25, smooth_sigma=0.0):
    """Fit a Gaussian curve to a 1D derivative signal around peak_idx with analytical broadening correction."""
    signal_1d = np.asarray(signal_1d, dtype=float)
    N = len(signal_1d)
    if N < 5:
        return np.nan, None, 0.0, None, None

    peak_idx = int(np.clip(peak_idx, 0, N - 1))
    start = max(0, peak_idx - window // 2)
    end = min(N, peak_idx + window // 2 + 1)

    x = np.arange(start, end, dtype=float)
    y = signal_1d[start:end]

    if len(x) < 5 or np.all(y == y[0]):
        return np.nan, None, 0.0, None, None

    a0 = float(signal_1d[peak_idx] - np.min(y))
    x0 = float(peak_idx)
    sigma0 = float(max(2.0, window / 4.0))
    offset0 = float(np.min(y))
    p0 = [a0, x0, sigma0, offset0]

    bounds = (
        [0.0, start, 0.5, -np.inf],
        [np.inf, end, window * 2.0, np.inf]
    )

    try:
        popt, _ = curve_fit(gaussian_func, x, y, p0=p0, bounds=bounds, maxfev=4000)
        sigma = abs(popt[2])
        fwhm_obs = 2.35482 * sigma

        # Analytical Broadening Correction for Gaussian pre-filter: FWHM_true = sqrt(FWHM_obs^2 - (2.355*sigma_smooth)^2)
        if smooth_sigma > 0:
            filter_fwhm_sq = (2.35482 * smooth_sigma) ** 2
            fwhm = float(np.sqrt(max(0.01, fwhm_obs ** 2 - filter_fwhm_sq)))
        else:
            fwhm = float(fwhm_obs)

        # Fit quality R^2
        y_pred = gaussian_func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        x_dense = np.linspace(start, end, 200)
        y_dense = gaussian_func(x_dense, *popt)
        return fwhm, popt, r2, x_dense, y_dense
    except Exception:
        return np.nan, None, 0.0, None, None


class GaussianFitInspectDialog(QDialog):
    """Interactive Dialog to visually verify Gaussian edge derivative fitting quality before running full Z analysis."""

    def __init__(self, parent, x_vals, profile_a, profile_b, crosshair_idx, z_frame, window_size=25, label_a="Dataset A", label_b="Dataset B"):
        super().__init__(parent)
        self.setWindowTitle(f"Gaussian Edge Fit Inspection (Frame Z={z_frame})")
        self.setMinimumSize(850, 600)

        self.x_vals = x_vals
        self.profile_a = profile_a
        self.profile_b = profile_b
        self.crosshair_idx = crosshair_idx
        self.z_frame = z_frame
        self.window_size = window_size
        self.label_a = label_a
        self.label_b = label_b

        layout = QVBoxLayout(self)

        # Plot Figure
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax_profile = self.fig.add_subplot(211)
        self.ax_deriv = self.fig.add_subplot(212)
        layout.addWidget(self.canvas)

        # Info Box
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #1565C0;")
        layout.addWidget(self.info_label)

        # Controls & Buttons
        ctrl_layout = QHBoxLayout()

        ctrl_layout.addWidget(QLabel("Fit Window (px):"))
        self.win_spin = QSpinBox()
        self.win_spin.setRange(5, 2000)
        self.win_spin.setValue(window_size)
        self.win_spin.valueChanged.connect(self._recalculate_fit)
        ctrl_layout.addWidget(self.win_spin)

        ctrl_layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Proceed to Full Z Analysis")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        ctrl_layout.addWidget(buttons)

        layout.addLayout(ctrl_layout)

        self._recalculate_fit()

    def _recalculate_fit(self):
        win = self.win_spin.value()
        self.ax_profile.clear()
        self.ax_deriv.clear()

        info_text = []

        # Dataset A Multi-Peak Gaussian Fitting
        if self.profile_a is not None:
            self.ax_profile.plot(self.x_vals, self.profile_a, 'b-', label=f"{self.label_a} Raw I")
            prof_smooth_a = gaussian_filter1d(self.profile_a, sigma=2.0)
            deriv_a = np.abs(np.gradient(prof_smooth_a))
            self.ax_deriv.plot(self.x_vals, deriv_a, 'b.', alpha=0.4, label=f"{self.label_a} |dI/dx|")

            max_d_a = np.max(deriv_a) if len(deriv_a) > 0 else 0
            if max_d_a > 0:
                peaks_a, _ = signal.find_peaks(deriv_a, distance=35, height=max_d_a * 0.15)
                fitted_count = 0
                for pk in peaks_a:
                    fw, popt, r2, x_fit, y_fit = calculate_fwhm(deriv_a, pk, window=win)
                    if popt is not None and x_fit is not None and np.isfinite(fw) and r2 > 0.2:
                        lbl_fit = f"{self.label_a} Fits" if fitted_count == 0 else ""
                        self.ax_deriv.plot(x_fit, y_fit, 'b-', linewidth=2, label=lbl_fit)
                        info_text.append(
                            f"<b>{self.label_a} Peak #{fitted_count+1}</b>: Center x0 = <b>{popt[1]:.1f}</b> px | "
                            f"FWHM = <b>{fw:.2f}</b> px | Amp = {popt[0]:.1f} | R^2 = {r2:.3f}"
                        )
                        fitted_count += 1

        # Dataset B Multi-Peak Gaussian Fitting
        if self.profile_b is not None:
            self.ax_profile.plot(self.x_vals, self.profile_b, 'g--', label=f"{self.label_b} Raw I")
            prof_smooth_b = gaussian_filter1d(self.profile_b, sigma=2.0)
            deriv_b = np.abs(np.gradient(prof_smooth_b))
            self.ax_deriv.plot(self.x_vals, deriv_b, 'g.', alpha=0.4, label=f"{self.label_b} |dI/dx|")

            max_d_b = np.max(deriv_b) if len(deriv_b) > 0 else 0
            if max_d_b > 0:
                peaks_b, _ = signal.find_peaks(deriv_b, distance=35, height=max_d_b * 0.15)
                fitted_count = 0
                for pk in peaks_b:
                    fw, popt, r2, x_fit, y_fit = calculate_fwhm(deriv_b, pk, window=win)
                    if popt is not None and x_fit is not None and np.isfinite(fw) and r2 > 0.2:
                        lbl_fit = f"{self.label_b} Fits" if fitted_count == 0 else ""
                        self.ax_deriv.plot(x_fit, y_fit, 'g--', linewidth=2, label=lbl_fit)
                        info_text.append(
                            f"<b>{self.label_b} Peak #{fitted_count+1}</b>: Center x0 = <b>{popt[1]:.1f}</b> px | "
                            f"FWHM = <b>{fw:.2f}</b> px | Amp = {popt[0]:.1f} | R^2 = {r2:.3f}"
                        )
                        fitted_count += 1

        self.ax_profile.set_title(f"1D Raw Intensity Line Profile (Crosshair={self.crosshair_idx} px)")
        self.ax_profile.set_ylabel("Intensity")
        self.ax_profile.legend(loc='upper right', fontsize=8)
        self.ax_profile.grid(True, alpha=0.3)

        self.ax_deriv.set_title("Edge Derivative |dI/dx| & Multi-Peak Gaussian Fits")
        self.ax_deriv.set_xlabel("Pixel Coordinate (px)")
        self.ax_deriv.set_ylabel("|dI/dx|")
        self.ax_deriv.legend(loc='upper right', fontsize=8)
        self.ax_deriv.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

        self.info_label.setText("<br>".join(info_text) if info_text else "No valid Gaussian fits obtained for detected peaks.")


class KnifeEdgeViewer(HyperViewer):
    """3D Focus Depth & Knife-Edge Sharpness Analysis Application."""

    def __init__(self):
        self.dataset_a = None       # 3D focus stack A (H, W, Z)
        self.dataset_b = None       # 3D focus stack B (H, W, Z)
        self.name_a = "Dataset A"
        self.name_b = "Dataset B"
        self.active_dataset_idx = 0  # 0: A, 1: B
        self.profile_mode = "X"      # "X": Row profile along Y crosshair, "Y": Column profile along X crosshair

        super().__init__()
        self.setWindowTitle("KnifeEdgeViewer - 3D Focus & Edge Sharpness Tool")
        self._setup_knifedge_ui()

    def _setup_knifedge_ui(self):
        """Update top toolbar and right panel for 1D line profiles and dual dataset focus analysis."""

        # ---------------- Top Menu Additions ----------------
        menubar = self.menuBar()
        file_menu = menubar.findChild(QWidget, "File") or menubar.actions()[0].menu()
        file_menu.clear()

        act_load_a = QAction("Load Dataset A (Primary)…", self)
        act_load_a.setShortcut("Ctrl+O")
        act_load_a.triggered.connect(self.load_dataset_a)
        file_menu.addAction(act_load_a)

        act_load_b = QAction("Load Dataset B (Comparison)…", self)
        act_load_b.setShortcut("Ctrl+Shift+O")
        act_load_b.triggered.connect(self.load_dataset_b)
        file_menu.addAction(act_load_b)

        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # ---------------- Spatial View Dataset Switcher ----------------
        self.dataset_combo = QComboBox()
        self.dataset_combo.setToolTip("Switch which 3D focus stack is rendered in the 3D spatial image view")
        self.dataset_combo.addItems(["Spatial 3D: Dataset A", "Spatial 3D: Dataset B"])
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_switch)

        if hasattr(self, 'tool_layout') and self.tool_layout is not None:
            self.tool_layout.insertWidget(0, self.dataset_combo)
        else:
            self.menuBar().setCornerWidget(self.dataset_combo, Qt.TopRightCorner)

        # ---------------- Right Panel Re-configuration ----------------
        # Replace spectrum group title and layout
        self.spectrum_ax.clear()
        self.spectrum_ax.set_xlabel("Pixel Position (px)")
        self.spectrum_ax.set_ylabel("Intensity")
        self.spectrum_ax.set_title("1D Line Profile")
        self.spectrum_ax.grid(True, alpha=0.3)

        # Profile Mode Toggle Switch (Option B)
        profile_toggle_layout = QHBoxLayout()
        profile_toggle_layout.addWidget(QLabel("Profile Mode:"))
        self.profile_mode_combo = QComboBox()
        self.profile_mode_combo.addItems(["X Profile (Row Y)", "Y Profile (Column X)"])
        self.profile_mode_combo.currentIndexChanged.connect(self.on_profile_mode_changed)
        profile_toggle_layout.addWidget(self.profile_mode_combo)
        profile_toggle_layout.addStretch()

        # Insert mode combo into plot toolbar
        plot_toolbar_layout = self.spectrum_canvas.parent().layout()
        if plot_toolbar_layout is not None:
            plot_toolbar_layout.insertLayout(0, profile_toggle_layout)

        # Fit Controls Group: Step-by-Step Edge Fitting & Focus Analysis
        fit_group = QGroupBox("Edge Fitting & Focus Analysis")
        fit_layout = QVBoxLayout(fit_group)

        # Config Row: Line Width, Fit Window, Fit Model & Noise Smoothing
        row_cfg = QHBoxLayout()
        row_cfg.addWidget(QLabel("X:"))
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 99999)
        self.roi_x_spin.setValue(0)
        self.roi_x_spin.setToolTip("Fine-tune crosshair X position (or use Arrow Keys)")
        self.roi_x_spin.valueChanged.connect(self._on_roi_spinbox_changed)
        row_cfg.addWidget(self.roi_x_spin)

        row_cfg.addWidget(QLabel("Y:"))
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 99999)
        self.roi_y_spin.setValue(0)
        self.roi_y_spin.setToolTip("Fine-tune crosshair Y position (or use Arrow Keys)")
        self.roi_y_spin.valueChanged.connect(self._on_roi_spinbox_changed)
        row_cfg.addWidget(self.roi_y_spin)

        row_cfg.addWidget(QLabel("Line Width (px):"))
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 200)
        self.line_width_spin.setValue(10)
        self.line_width_spin.setToolTip("Perpendicular ribbon width for line profile averaging (suppresses noise)")
        self.line_width_spin.valueChanged.connect(self.update_line_profile_plot)
        row_cfg.addWidget(self.line_width_spin)

        row_cfg.addWidget(QLabel("Fit Window (px):"))
        self.fit_win_spin = QSpinBox()
        self.fit_win_spin.setRange(5, 2000)
        self.fit_win_spin.setValue(25)
        self.fit_win_spin.setToolTip("Sub-window size (in pixels) centered on each edge for local curve fitting")
        row_cfg.addWidget(self.fit_win_spin)

        row_cfg.addWidget(QLabel("Fit Model:"))
        self.fit_model_combo = QComboBox()
        self.fit_model_combo.addItems(["Error Function (erf)", "Gaussian Deriv"])
        self.fit_model_combo.setToolTip("Switch between direct Error Function (erf) fit on raw intensity vs Gaussian fit on derivative |dI/dx|")
        row_cfg.addWidget(self.fit_model_combo)

        row_cfg.addWidget(QLabel("Smooth Sigma (px):"))
        self.smooth_sigma_spin = QDoubleSpinBox()
        self.smooth_sigma_spin.setRange(0.0, 10.0)
        self.smooth_sigma_spin.setSingleStep(0.5)
        self.smooth_sigma_spin.setValue(2.0)
        self.smooth_sigma_spin.setToolTip("Gaussian noise smoothing filter applied to raw profile BEFORE taking derivative")
        row_cfg.addWidget(self.smooth_sigma_spin)
        fit_layout.addLayout(row_cfg)

        # Config Row 2: Peak Discovery Sensitivity Parameters
        row_peak_cfg = QHBoxLayout()
        row_peak_cfg.addWidget(QLabel("Peak Thresh (%):"))
        self.peak_thresh_spin = QSpinBox()
        self.peak_thresh_spin.setRange(5, 90)
        self.peak_thresh_spin.setValue(20)
        self.peak_thresh_spin.setSingleStep(5)
        self.peak_thresh_spin.setToolTip("Minimum peak height threshold percentage relative to max derivative peak for candidate edge discovery")
        row_peak_cfg.addWidget(self.peak_thresh_spin)

        row_peak_cfg.addWidget(QLabel("Min Edge Dist (px):"))
        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(5, 200)
        self.min_dist_spin.setValue(35)
        self.min_dist_spin.setSingleStep(5)
        self.min_dist_spin.setToolTip("Minimum spatial separation distance between candidate edges")
        row_peak_cfg.addWidget(self.min_dist_spin)
        fit_layout.addLayout(row_peak_cfg)

        # Action Buttons Row 1
        row_btns1 = QHBoxLayout()
        self.btn_differential = QPushButton("Differential (dI/dx)")
        self.btn_differential.setToolTip("Computes 1D derivative profile |dI/dx| with noise smoothing and updates plot")
        self.btn_differential.clicked.connect(self.run_differential_step)
        row_btns1.addWidget(self.btn_differential)

        self.btn_gaussian_fit = QPushButton("Curve Fit")
        self.btn_gaussian_fit.setToolTip("Finds edge peaks, fits ERF/Gaussian curves, and outputs peak centers & FWHM widths")
        self.btn_gaussian_fit.clicked.connect(self.run_gaussian_fit_step)
        row_btns1.addWidget(self.btn_gaussian_fit)

        self.btn_troubleshoot_peaks = QPushButton("🔍 Peak Diagnostics")
        self.btn_troubleshoot_peaks.setToolTip("Opens pop-up troubleshooting window with pre-smoothed profile, coarse derivative curve, candidate peaks, and final ERF edge fits")
        self.btn_troubleshoot_peaks.clicked.connect(self.show_peak_discovery_diagnostics)
        row_btns1.addWidget(self.btn_troubleshoot_peaks)

        fit_layout.addLayout(row_btns1)

        # Action Buttons Row 2
        row_btns2 = QHBoxLayout()
        self.btn_inspect_fit = QPushButton("Focus Curve (FWHM vs Z)")
        self.btn_inspect_fit.setToolTip("Evaluates FWHM(Z) curves across frames Z and computes focus separation ΔZ")
        self.btn_inspect_fit.clicked.connect(self.fit_local_edge_with_preview)
        row_btns2.addWidget(self.btn_inspect_fit)

        self.btn_grid_map = QPushButton("Grid Focus Map")
        self.btn_grid_map.setToolTip("Runs automated grid line analysis and plots 2D sharpest focus map Z_min(x,y)")
        self.btn_grid_map.clicked.connect(self.run_grid_focus_map)
        row_btns2.addWidget(self.btn_grid_map)

        self.btn_theoretical_curve = QPushButton("Theoretical ERF")
        self.btn_theoretical_curve.setToolTip("Calculates theoretical diffraction-limited Gaussian beam waist w0, ERF edge profile, and FWHM(z) defocus curve for given laser wavelength & NA")
        self.btn_theoretical_curve.clicked.connect(self.show_theoretical_knifeedge_dialog)
        row_btns2.addWidget(self.btn_theoretical_curve)

        fit_layout.addLayout(row_btns2)

        # Text Output Window Log Header (Label + Clear Log Button)
        row_log_header = QHBoxLayout()
        lbl_log = QLabel("Analysis Results Log:")
        lbl_log.setStyleSheet("font-weight: bold;")
        row_log_header.addWidget(lbl_log)
        row_log_header.addStretch()

        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.setFixedWidth(90)
        btn_clear_log.setToolTip("Clear all text from the analysis results log")
        btn_clear_log.clicked.connect(lambda: self.log_text_window.clear())
        row_log_header.addWidget(btn_clear_log)
        fit_layout.addLayout(row_log_header)

        self.log_text_window = QTextEdit()
        self.log_text_window.setReadOnly(True)
        self.log_text_window.setMinimumHeight(180)
        self.log_text_window.setMaximumHeight(260)
        self.log_text_window.setFont(QFont("Consolas", 9))
        self.log_text_window.setPlaceholderText("Analysis results log will appear here step-by-step...")
        fit_layout.addWidget(self.log_text_window)

        # Add Edge Fitting Group to Right Panel
        right_widget = self.spectrum_canvas.parentWidget()
        if right_widget is not None and right_widget.layout() is not None:
            right_widget.layout().addWidget(fit_group)

        # Re-label Titles to Line Plot & Line Management
        for gb in self.findChildren(QGroupBox):
            if gb.title() == "Spectrum Plot":
                gb.setTitle("Line Plot")
            elif gb.title() == "Spectrum Management":
                gb.setTitle("Line Management")

        if hasattr(self, 'export_btn'):
            self.export_btn.setText("Export Line Plots")
        if hasattr(self, 'add_btn'):
            self.add_btn.setText("Add Current Selection")
            self.add_btn.setEnabled(True)

        # Hide Circle selection tool and rename Square selection tool to Rectangle
        if hasattr(self, 'circle_btn') and self.circle_btn is not None:
            self.circle_btn.hide()
        if hasattr(self, 'square_btn') and self.square_btn is not None:
            self.square_btn.setText("Rectangle")
            self.square_btn.setToolTip("Draw a rectangle ROI. Data is averaged along the shorter dimension to produce a 1D line profile.")

        # Hide Unwanted Background & Reference Spectra UI Elements
        unwanted_names = [
            'set_bg_btn', 'subtract_bg_btn', 'bg_status_label',
            'load_ref_btn', 'transform_btn', 'save_transform_btn',
            'ref_status_label', 'ref_text_box', 'bg_text_box'
        ]
        for name in unwanted_names:
            if hasattr(self, name):
                w = getattr(self, name)
                if w is not None and isinstance(w, QWidget):
                    w.hide()

        # Hide any leftover labels containing "BG", "Background", or "Reference"
        for lbl in self.findChildren(QLabel):
            txt = lbl.text().strip()
            if txt in ["BG", "Background: None", "Reference Spectra: None loaded"] or "Reference" in txt or "Transform" in txt:
                lbl.hide()

        # Load saved user settings (Fit Window, Min Peak Distance, Peak Thresh, Pixel Sizes, etc.)
        self.load_persistent_settings()

    def load_persistent_settings(self):
        """Load user parameters (Fit Window, Min Peak Distance, Peak Thresh, Pixel Sizes, Fit Model, etc.) from QSettings."""
        settings = QSettings("KnifeEdgeApp", "KnifeEdgeViewer")
        
        fit_win = settings.value("fit_window", 25, type=int)
        min_dist = settings.value("min_peak_distance", 35, type=int)
        peak_thresh = settings.value("peak_threshold", 20, type=int)
        line_w = settings.value("line_width", 10, type=int)
        smooth_sig = settings.value("smooth_sigma", 2.0, type=float)
        fit_model = settings.value("fit_model", "Error Function (erf)", type=str)
        
        pixel_x = settings.value("pixel_size_x", 1.0, type=float)
        pixel_y = settings.value("pixel_size_y", 1.0, type=float)
        pixel_z = settings.value("pixel_size_z", 1.0, type=float)

        if hasattr(self, 'fit_win_spin'):
            self.fit_win_spin.setValue(fit_win)
        if hasattr(self, 'min_dist_spin'):
            self.min_dist_spin.setValue(min_dist)
        if hasattr(self, 'peak_thresh_spin'):
            self.peak_thresh_spin.setValue(peak_thresh)
        if hasattr(self, 'line_width_spin'):
            self.line_width_spin.setValue(line_w)
        if hasattr(self, 'smooth_sigma_spin'):
            self.smooth_sigma_spin.setValue(smooth_sig)
        if hasattr(self, 'fit_model_combo'):
            idx = self.fit_model_combo.findText(fit_model)
            if idx >= 0:
                self.fit_model_combo.setCurrentIndex(idx)

        if hasattr(self, 'pixel_size_x_spin'):
            self.pixel_size_x_spin.setValue(pixel_x)
        if hasattr(self, 'pixel_size_y_spin'):
            self.pixel_size_y_spin.setValue(pixel_y)
        if hasattr(self, 'pixel_size_z_spin'):
            self.pixel_size_z_spin.setValue(pixel_z)

    def save_persistent_settings(self):
        """Save user parameters (Fit Window, Min Peak Distance, Peak Thresh, Pixel Sizes, Fit Model, etc.) to QSettings."""
        settings = QSettings("KnifeEdgeApp", "KnifeEdgeViewer")
        if hasattr(self, 'fit_win_spin'):
            settings.setValue("fit_window", self.fit_win_spin.value())
        if hasattr(self, 'min_dist_spin'):
            settings.setValue("min_peak_distance", self.min_dist_spin.value())
        if hasattr(self, 'peak_thresh_spin'):
            settings.setValue("peak_threshold", self.peak_thresh_spin.value())
        if hasattr(self, 'line_width_spin'):
            settings.setValue("line_width", self.line_width_spin.value())
        if hasattr(self, 'smooth_sigma_spin'):
            settings.setValue("smooth_sigma", self.smooth_sigma_spin.value())
        if hasattr(self, 'fit_model_combo'):
            settings.setValue("fit_model", self.fit_model_combo.currentText())

        if hasattr(self, 'pixel_size_x_spin'):
            settings.setValue("pixel_size_x", self.pixel_size_x_spin.value())
        if hasattr(self, 'pixel_size_y_spin'):
            settings.setValue("pixel_size_y", self.pixel_size_y_spin.value())
        if hasattr(self, 'pixel_size_z_spin'):
            settings.setValue("pixel_size_z", self.pixel_size_z_spin.value())

    def closeEvent(self, event):
        """Save user settings when application window is closed."""
        self.save_persistent_settings()
        super().closeEvent(event)

    # ---------------- Dataset Loading & Switching ----------------

    def load_dataset_a(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Dataset A (Primary 3D Focus Stack)", "", "All Supported (*.tif *.tiff *.npy *.mat);;TIFF (*.tif *.tiff);;NumPy (*.npy);;MAT (*.mat)")
        if not file_path:
            return
        self.load_dataset_file(file_path, is_b=False)

    def load_dataset_b(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Dataset B (Comparison 3D Focus Stack)", "", "All Supported (*.tif *.tiff *.npy *.mat);;TIFF (*.tif *.tiff);;NumPy (*.npy);;MAT (*.mat)")
        if not file_path:
            return
        self.load_dataset_file(file_path, is_b=True)

    def load_dataset_file(self, file_path, is_b=False):
        cube = self._load_file(file_path)
        if cube is None:
            return

        cube = np.asarray(cube, dtype=np.float32)
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        elif cube.ndim == 3 and cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
            cube = np.transpose(cube, (1, 2, 0))

        lbl = os.path.basename(file_path)
        if not is_b:
            self.dataset_a = cube
            self.name_a = f"Dataset A ({lbl})"
            self.dataset_combo.setItemText(0, f"Spatial 3D: {self.name_a}")
            if self.active_dataset_idx == 0:
                self.hypercube = self.dataset_a
                self.setup_display()
        else:
            self.dataset_b = cube
            self.name_b = f"Dataset B ({lbl})"
            self.dataset_combo.setItemText(1, f"Spatial 3D: {self.name_b}")
            if self.active_dataset_idx == 1:
                self.hypercube = self.dataset_b
                self.setup_display()

        self.update_line_profile_plot()
        self.statusBar().showMessage(f"Loaded {'Dataset B' if is_b else 'Dataset A'}: {cube.shape}")

    def on_dataset_switch(self, idx):
        self.active_dataset_idx = idx
        if idx == 0 and self.dataset_a is not None:
            self.hypercube = self.dataset_a
        elif idx == 1 and self.dataset_b is not None:
            self.hypercube = self.dataset_b

        if self.hypercube is not None:
            self.setup_display()
        self.update_line_profile_plot()

    def on_profile_mode_changed(self, idx):
        self.profile_mode = "X" if idx == 0 else "Y"
        self.update_line_profile_plot()

    # ---------------- Interactive 1D Line Profile Plot ----------------

    def display_frame(self, frame_idx):
        super().display_frame(frame_idx)
        self.update_line_profile_plot()

    def keyPressEvent(self, event):
        """Handle Arrow Keys to nudge crosshair position pixel-by-pixel (Shift+Arrow for 5px jumps)."""
        if self.hypercube is None:
            super().keyPressEvent(event)
            return

        key = event.key()
        if key not in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            super().keyPressEvent(event)
            return

        step = 5 if (event.modifiers() & Qt.ShiftModifier) else 1
        dx = -step if key == Qt.Key_Left else (step if key == Qt.Key_Right else 0)
        dy = -step if key == Qt.Key_Up else (step if key == Qt.Key_Down else 0)

        H, W = self.hypercube.shape[:2]
        new_x = max(0, min(W - 1, (self.crosshair_x or W // 2) + dx))
        new_y = max(0, min(H - 1, (self.crosshair_y or H // 2) + dy))

        self.crosshair_x = new_x
        self.crosshair_y = new_y

        if hasattr(self, 'roi_x_spin') and hasattr(self, 'roi_y_spin'):
            self.roi_x_spin.blockSignals(True)
            self.roi_y_spin.blockSignals(True)
            self.roi_x_spin.setValue(new_x)
            self.roi_y_spin.setValue(new_y)
            self.roi_x_spin.blockSignals(False)
            self.roi_y_spin.blockSignals(False)

        self.display_frame(self.current_frame)
        self.update_line_profile_plot()

    def _on_roi_spinbox_changed(self):
        """Callback when X or Y spinboxes change manually."""
        if self.hypercube is None:
            return
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        self.crosshair_x = x
        self.crosshair_y = y
        self.display_frame(self.current_frame)
        self.update_line_profile_plot()

    def on_canvas_press(self, event):
        super().on_canvas_press(event)
        # If clicking in point selection mode, deactivate rectangle ROI
        if getattr(self, 'selection_tool', 'point') == 'point':
            self.is_rectangle_roi_active = False
            self.rect_coords = None
            if hasattr(self, 'line_width_spin'):
                self.line_width_spin.setEnabled(True)
                self.smooth_sigma_spin.setEnabled(True)
                self.fit_win_spin.setEnabled(True)
        self.update_line_profile_plot()

    def add_square_selection(self, center_x, center_y, size):
        """Override square selection tool to handle 2D Rectangle ROI line profile averaging."""
        self.add_rectangle_selection(center_x - size // 2, center_y - size // 2, size, size)

    def add_rectangle_selection(self, x1, y1, width, height):
        """
        Handle Rectangle ROI selection:
        Averages data along the shorter dimension to produce a 1D line profile along the longer dimension.
        Disables line width, smooth sigma, and fit window spinboxes while rectangle is active.
        """
        if self.hypercube is None:
            return

        w_rect = abs(width)
        h_rect = abs(height)
        if w_rect <= 0 or h_rect <= 0:
            return

        x_center = int(round(x1 + w_rect / 2.0))
        y_center = int(round(y1 + h_rect / 2.0))

        self.is_rectangle_roi_active = True
        self.rect_coords = (int(x1), int(y1), int(w_rect), int(h_rect))

        # Disable spinboxes as rectangle ROI provides its own ribbon averaging and exact bounds
        if hasattr(self, 'line_width_spin'):
            self.line_width_spin.setEnabled(False)
            self.smooth_sigma_spin.setEnabled(False)
            self.fit_win_spin.setEnabled(False)

        if h_rect < w_rect:
            # Horizontal rectangle: short in Y, long in X -> X Profile
            self.profile_mode = "X"
            if hasattr(self, 'profile_mode_combo'):
                self.profile_mode_combo.setCurrentIndex(0)
            self.crosshair_x = x_center
            self.crosshair_y = y_center
            self.log_msg(f"[Rectangle ROI Active] Horizontal: Width={w_rect:.1f}px (X Profile), Height={h_rect:.1f}px (Averaged Ribbon). Line Width, Smooth Sigma & Fit Window ignored.")
        else:
            # Vertical rectangle: short in X, long in Y -> Y Profile
            self.profile_mode = "Y"
            if hasattr(self, 'profile_mode_combo'):
                self.profile_mode_combo.setCurrentIndex(1)
            self.crosshair_x = x_center
            self.crosshair_y = y_center
            self.log_msg(f"[Rectangle ROI Active] Vertical: Height={h_rect:.1f}px (Y Profile), Width={w_rect:.1f}px (Averaged Ribbon). Line Width, Smooth Sigma & Fit Window ignored.")

        # Store current selection metadata and draw persistent cyan rectangle box
        z_frame = self.current_frame
        prof_a = extract_line_profile(self.dataset_a, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, rect_coords=self.rect_coords) if self.dataset_a is not None else None
        prof_b = extract_line_profile(self.dataset_b, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, rect_coords=self.rect_coords) if self.dataset_b is not None else None

        self.current_selection = {
            'type': 'rectangle',
            'rect_coords': self.rect_coords,
            'spectrum': prof_a if prof_a is not None else prof_b
        }

        if hasattr(self, 'add_btn'):
            self.add_btn.setEnabled(True)

        self.display_frame(self.current_frame)
        self._redraw_overlay_with_current()
        self.update_line_profile_plot()

    def update_line_profile_plot(self):
        """Update 1D line profile plot overlaying Dataset A (solid) and Dataset B (dashed) in calibrated physical units."""
        if self.dataset_a is None and self.dataset_b is None:
            return

        self.spectrum_ax.clear()
        z_frame = self.current_frame
        line_w = self.line_width_spin.value() if hasattr(self, 'line_width_spin') else 10
        is_rect = getattr(self, 'is_rectangle_roi_active', False)
        rect_c = self.rect_coords if is_rect else None
        px_scale = self.get_pixel_scale()
        unit_str = "µm" if px_scale != 1.0 else "px"

        # Line Profile Dataset A
        if self.dataset_a is not None and z_frame < self.dataset_a.shape[2]:
            prof_a = extract_line_profile(self.dataset_a, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, line_width=line_w, rect_coords=rect_c)
            if prof_a is not None:
                x_vals = np.arange(len(prof_a)) * px_scale
                lbl_title = f"X Profile along Row Y = {int(self.crosshair_y)}" if self.profile_mode == "X" else f"Y Profile along Column X = {int(self.crosshair_x)}"
                if is_rect:
                    lbl_title = f"Rectangle ROI Averaged Profile ({self.profile_mode}-Direction)"
                self.spectrum_ax.plot(x_vals, prof_a, 'b-', label=f"{self.name_a} (Z={z_frame})")

        # Line Profile Dataset B
        if self.dataset_b is not None and z_frame < self.dataset_b.shape[2]:
            prof_b = extract_line_profile(self.dataset_b, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, line_width=line_w, rect_coords=rect_c)
            if prof_b is not None:
                x_vals = np.arange(len(prof_b)) * px_scale
                lbl_title = f"X Profile along Row Y = {int(self.crosshair_y)}" if self.profile_mode == "X" else f"Y Profile along Column X = {int(self.crosshair_x)}"
                if is_rect:
                    lbl_title = f"Rectangle ROI Averaged Profile ({self.profile_mode}-Direction)"
                self.spectrum_ax.plot(x_vals, prof_b, 'g--', label=f"{self.name_b} (Z={z_frame})")

        self.spectrum_ax.set_xlabel(f"Position ({unit_str})")
        self.spectrum_ax.set_ylabel("Raw Intensity I")
        self.spectrum_ax.set_title(lbl_title if 'lbl_title' in locals() else "1D Line Profile")
        self.spectrum_ax.grid(True, alpha=0.3)

        # Plot saved line selections
        if hasattr(self, 'spectra_list') and self.spectra_list:
            for sp in self.spectra_list:
                if sp.visible:
                    x_vals = np.arange(len(sp.spectrum)) * px_scale
                    self.spectrum_ax.plot(x_vals, sp.spectrum, color=sp.color, linewidth=1.5, linestyle=':', label=sp.label)

        self.spectrum_ax.legend(loc='upper right', fontsize=8)
        self.spectrum_canvas.draw_idle()

    def add_current_spectrum(self):
        """Add current 1D line profile selection to Line Management list."""
        if self.dataset_a is None and self.dataset_b is None:
            return

        z_frame = self.current_frame
        is_rect = getattr(self, 'is_rectangle_roi_active', False)
        rect_c = self.rect_coords if is_rect else None
        line_w = self.line_width_spin.value() if hasattr(self, 'line_width_spin') and not is_rect else 1
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        color = colors[len(self.spectra_list) % len(colors)]

        # Dataset A profile
        if self.dataset_a is not None and z_frame < self.dataset_a.shape[2]:
            prof_a = extract_line_profile(self.dataset_a, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, line_width=line_w, rect_coords=rect_c)
            if prof_a is not None:
                lbl = f"{self.name_a} Rect ROI-{self.profile_mode} (Z={z_frame})" if is_rect else f"{self.name_a} {self.profile_mode}-Line (Z={z_frame})"
                from HyperViewer import SpectrumData
                sp_a = SpectrumData(
                    spectrum=prof_a,
                    label=lbl,
                    color=color,
                    selection_type='rectangle' if is_rect else 'line',
                    coords=(self.crosshair_x, self.crosshair_y)
                )
                if is_rect:
                    sp_a.rect_coords = rect_c
                self.spectra_list.append(sp_a)

        # Dataset B profile
        if self.dataset_b is not None and z_frame < self.dataset_b.shape[2]:
            prof_b = extract_line_profile(self.dataset_b, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, line_width=line_w, rect_coords=rect_c)
            if prof_b is not None:
                lbl = f"{self.name_b} Rect ROI-{self.profile_mode} (Z={z_frame})" if is_rect else f"{self.name_b} {self.profile_mode}-Line (Z={z_frame})"
                from HyperViewer import SpectrumData
                sp_b = SpectrumData(
                    spectrum=prof_b,
                    label=lbl,
                    color=color,
                    selection_type='rectangle' if is_rect else 'line',
                    coords=(self.crosshair_x, self.crosshair_y)
                )
                if is_rect:
                    sp_b.rect_coords = rect_c
                self.spectra_list.append(sp_b)

        self.update_spectrum_plot()
        self.update_checkbox_list()
        if hasattr(self, 'delete_btn'):
            self.delete_btn.setEnabled(True)
        if hasattr(self, 'export_btn'):
            self.export_btn.setEnabled(True)

    def update_spectrum_plot(self):
        """Update spectrum/line plot."""
        self.update_line_profile_plot()

    def log_msg(self, msg):
        """Append log message to the Text Output Window."""
        if hasattr(self, 'log_text_window') and self.log_text_window is not None:
            self.log_text_window.append(msg)

    def run_differential_step(self):
        """Step 1: Compute 1D derivative profile |dI/dx| and update plot."""
        if self.dataset_a is None and self.dataset_b is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        z_frame = self.current_frame
        is_rect = getattr(self, 'is_rectangle_roi_active', False)
        rect_c = self.rect_coords if is_rect else None
        smooth_sigma = 0.0 if is_rect else self.smooth_sigma_spin.value()
        line_w = self.line_width_spin.value() if not is_rect else 1
        self.spectrum_ax.clear()

        cross_idx = self.crosshair_x if self.profile_mode == "X" else self.crosshair_y
        if is_rect:
            self.log_msg(f"=== Step 1: Differential |dI/dx| [Rectangle ROI Active] (Mode={self.profile_mode}, Z={z_frame}) ===")
        else:
            self.log_msg(f"=== Step 1: Differential |dI/dx| (Mode={self.profile_mode}, Crosshair={cross_idx}px, Z={z_frame}, Smooth={smooth_sigma:.1f}px, RibbonWidth={line_w}px) ===")

        # Dataset A Differential
        if self.dataset_a is not None and z_frame < self.dataset_a.shape[2]:
            prof_a = extract_line_profile(self.dataset_a, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, line_width=line_w, rect_coords=rect_c)
            if prof_a is not None:
                prof_smooth_a = gaussian_filter1d(prof_a, sigma=smooth_sigma) if smooth_sigma > 0 else prof_a
                deriv_a = np.abs(np.gradient(prof_smooth_a))
                x_vals = np.arange(len(deriv_a))
                self.spectrum_ax.plot(x_vals, deriv_a, 'b-', label=f"{self.name_a} |dI/dx|")
                self.log_msg(f"  {self.name_a}: Max Derivative = {np.max(deriv_a):.2f}")

        # Dataset B Differential
        if self.dataset_b is not None and z_frame < self.dataset_b.shape[2]:
            prof_b = extract_line_profile(self.dataset_b, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, line_width=line_w, rect_coords=rect_c)
            if prof_b is not None:
                prof_smooth_b = gaussian_filter1d(prof_b, sigma=smooth_sigma) if smooth_sigma > 0 else prof_b
                deriv_b = np.abs(np.gradient(prof_smooth_b))
                x_vals = np.arange(len(deriv_b))
                self.spectrum_ax.plot(x_vals, deriv_b, 'g--', label=f"{self.name_b} |dI/dx|")
                self.log_msg(f"  {self.name_b}: Max Derivative = {np.max(deriv_b):.2f}")

        self.spectrum_ax.set_xlabel("Pixel Position (px)")
        self.spectrum_ax.set_ylabel("|dI/dx| Edge Derivative")
        self.spectrum_ax.set_title(f"1D Edge Differential |dI/dx| (Frame Z={z_frame})")
        self.spectrum_ax.grid(True, alpha=0.3)
        self.spectrum_ax.legend(loc='upper right', fontsize=8)
        self.spectrum_canvas.draw_idle()

    def run_gaussian_fit_step(self):
        """Step 2: Detect edge peaks using user-tuned threshold and distance, fit selected model (Gaussian Deriv vs ERF), overlay on plot, and log results."""
        if self.dataset_a is None and self.dataset_b is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        z_frame = self.current_frame
        is_rect = getattr(self, 'is_rectangle_roi_active', False)
        rect_c = self.rect_coords if is_rect else None
        smooth_sigma = 0.0 if is_rect else self.smooth_sigma_spin.value()
        line_w = self.line_width_spin.value() if not is_rect else 1
        model_type = self.fit_model_combo.currentText() if hasattr(self, 'fit_model_combo') else "Gaussian Deriv"

        thresh_pct = self.peak_thresh_spin.value() / 100.0 if hasattr(self, 'peak_thresh_spin') else 0.20
        min_dist = float(self.min_dist_spin.value()) if hasattr(self, 'min_dist_spin') else 35.0

        px_scale = self.get_pixel_scale()
        unit_str = "µm" if px_scale != 1.0 else "px"

        self.spectrum_ax.clear()
        cross_idx = self.crosshair_x if self.profile_mode == "X" else self.crosshair_y
        if is_rect:
            self.log_msg(f"=== Multi-Peak Edge Fit [{model_type}] (Thresh={thresh_pct*100:.0f}%, MinDist={min_dist:.0f}px) [Rectangle ROI Active] (Z={z_frame}) ===")
        else:
            self.log_msg(f"=== Multi-Peak Edge Fit [{model_type}] (Thresh={thresh_pct*100:.0f}%, MinDist={min_dist:.0f}px, Window={self.fit_win_spin.value()}px, Smooth={smooth_sigma:.1f}px, Z={z_frame}) ===")

        # Dataset A Edge Fit
        if self.dataset_a is not None and z_frame < self.dataset_a.shape[2]:
            prof_a = extract_line_profile(self.dataset_a, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, line_width=line_w, rect_coords=rect_c)
            if prof_a is not None:
                win_a = len(prof_a) if is_rect else self.fit_win_spin.value()
                # Pre-smooth profile ONLY for peak discovery to suppress noise spikes
                prof_peak_discovery_a = gaussian_filter1d(prof_a, sigma=4.0)
                deriv_a = np.abs(np.gradient(prof_peak_discovery_a, 5.0))

                x_vals_um = np.arange(len(prof_a)) * px_scale
                if model_type == "Error Function (erf)":
                    self.spectrum_ax.plot(x_vals_um, prof_a, '.', color='#1565C0', alpha=0.35, label=f"{self.name_a} Raw I")
                else:
                    self.spectrum_ax.plot(x_vals_um, deriv_a, '.', color='#1565C0', alpha=0.35, label=f"{self.name_a} |dI/dx|")

                max_d_a = np.max(deriv_a) if len(deriv_a) > 0 else 0
                if max_d_a > 0:
                    peaks_a, _ = signal.find_peaks(deriv_a, distance=int(min_dist), height=max_d_a * thresh_pct)
                    cand_fits_a = []
                    for pk in peaks_a:
                        if model_type == "Error Function (erf)":
                            # Fit ERF on RAW, UNSMOOTHED profile (prof_a) for un-broadened FWHM
                            fit_res = calculate_erf_fit(prof_a, pk, window=win_a)
                        else:
                            fit_res = calculate_fwhm(deriv_a, pk, window=win_a, smooth_sigma=smooth_sigma)
                        if fit_res[1] is not None and np.isfinite(fit_res[0]) and fit_res[2] > 0.2:
                            cand_fits_a.append(fit_res)

                    clean_fits_a = deduplicate_edge_fits(cand_fits_a, min_center_distance=min_dist)
                    for fitted_count, (fw, popt, r2, x_fit, y_fit) in enumerate(clean_fits_a):
                        dir_label = "Rising" if (popt[0] > 0 if model_type == "Error Function (erf)" else 1) else "Falling"
                        lbl_fit = f"{self.name_a} Fits" if fitted_count == 0 else ""
                        x_fit_um = x_fit * px_scale
                        self.spectrum_ax.plot(x_fit_um, y_fit, '-', color='#00B0FF', linewidth=2.5, label=lbl_fit)

                        x0_um = popt[1] * px_scale
                        fw_um = fw * px_scale
                        sig_um = abs(popt[2]) * px_scale
                        unit_lbl = f"µm ({popt[1]:.1f} px)" if px_scale != 1.0 else "px"
                        self.log_msg(
                            f"  {self.name_a} Edge #{fitted_count+1} ({dir_label}) [{model_type}]: Center x0 = {x0_um:.2f} {unit_lbl} | "
                            f"FWHM = {fw_um:.2f} {unit_str} | Sigma = {sig_um:.2f} {unit_str} | R^2 = {r2:.3f}"
                        )

        # Dataset B Edge Fit
        if self.dataset_b is not None and z_frame < self.dataset_b.shape[2]:
            prof_b = extract_line_profile(self.dataset_b, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, line_width=line_w, rect_coords=rect_c)
            if prof_b is not None:
                win_b = len(prof_b) if is_rect else self.fit_win_spin.value()
                # Pre-smooth profile ONLY for peak discovery to suppress noise spikes
                prof_peak_discovery_b = gaussian_filter1d(prof_b, sigma=4.0)
                deriv_b = np.abs(np.gradient(prof_peak_discovery_b, 5.0))

                x_vals_um = np.arange(len(prof_b)) * px_scale
                if model_type == "Error Function (erf)":
                    self.spectrum_ax.plot(x_vals_um, prof_b, '.', color='#2E7D32', alpha=0.35, label=f"{self.name_b} Raw I")
                else:
                    self.spectrum_ax.plot(x_vals_um, deriv_b, '.', color='#2E7D32', alpha=0.35, label=f"{self.name_b} |dI/dx|")

                max_d_b = np.max(deriv_b) if len(deriv_b) > 0 else 0
                if max_d_b > 0:
                    peaks_b, _ = signal.find_peaks(deriv_b, distance=int(min_dist), height=max_d_b * thresh_pct)
                    cand_fits_b = []
                    for pk in peaks_b:
                        if model_type == "Error Function (erf)":
                            # Fit ERF on RAW, UNSMOOTHED profile (prof_b) for un-broadened FWHM
                            fit_res = calculate_erf_fit(prof_b, pk, window=win_b)
                        else:
                            fit_res = calculate_fwhm(deriv_b, pk, window=win_b, smooth_sigma=smooth_sigma)
                        if fit_res[1] is not None and np.isfinite(fit_res[0]) and fit_res[2] > 0.2:
                            cand_fits_b.append(fit_res)

                    clean_fits_b = deduplicate_edge_fits(cand_fits_b, min_center_distance=min_dist)
                    for fitted_count, (fw, popt, r2, x_fit, y_fit) in enumerate(clean_fits_b):
                        dir_label = "Rising" if (popt[0] > 0 if model_type == "Error Function (erf)" else 1) else "Falling"
                        lbl_fit = f"{self.name_b} Fits" if fitted_count == 0 else ""
                        x_fit_um = x_fit * px_scale
                        self.spectrum_ax.plot(x_fit_um, y_fit, '--', color='#00E676', linewidth=2.5, label=lbl_fit)

                        x0_um = popt[1] * px_scale
                        fw_um = fw * px_scale
                        sig_um = abs(popt[2]) * px_scale
                        unit_lbl = f"µm ({popt[1]:.1f} px)" if px_scale != 1.0 else "px"
                        self.log_msg(
                            f"  {self.name_b} Edge #{fitted_count+1} ({dir_label}) [{model_type}]: Center x0 = {x0_um:.2f} {unit_lbl} | "
                            f"FWHM = {fw_um:.2f} {unit_str} | Sigma = {sig_um:.2f} {unit_str} | R^2 = {r2:.3f}"
                        )

        self.spectrum_ax.set_xlabel(f"Position ({unit_str})")
        self.spectrum_ax.set_ylabel("Intensity / |dI/dx|")
        self.spectrum_ax.set_title(f"1D Multi-Peak Edge Fits [{model_type}] (Frame Z={z_frame})")
        self.spectrum_ax.grid(True, alpha=0.3)
        self.spectrum_ax.legend(loc='upper right', fontsize=8)
        self.spectrum_canvas.draw_idle()

    def show_theoretical_knifeedge_dialog(self):
        """Open theoretical knife-edge ERF & beam waist calculator dialog."""
        dlg = TheoreticalKnifeEdgeDialog(self)
        dlg.exec_()

    def show_peak_discovery_diagnostics(self):
        """Show pop-up troubleshooting window with pre-smoothed profile, coarse derivative curve, candidate peaks, and final ERF edge fits."""
        if self.dataset_a is None and self.dataset_b is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        z_frame = self.current_frame
        is_rect = getattr(self, 'is_rectangle_roi_active', False)
        rect_c = self.rect_coords if is_rect else None
        line_w = self.line_width_spin.value() if not is_rect else 1

        prof_a = extract_line_profile(self.dataset_a, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, line_width=line_w, rect_coords=rect_c) if self.dataset_a is not None else None
        prof_b = extract_line_profile(self.dataset_b, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame, line_width=line_w, rect_coords=rect_c) if self.dataset_b is not None else None

        target_prof = prof_a if prof_a is not None else prof_b
        target_name = self.name_a if prof_a is not None else self.name_b

        dlg = PeakDiscoveryDiagnosticsDialog(self, target_prof, name=target_name)
        dlg.exec_()

    # ---------------- Local Edge Fitting & Inspection ----------------

    def fit_local_edge_with_preview(self):
        """Compute and plot multi-edge FWHM vs Z focus curves across all frames Z using the exact same fitting model and parameters as Curve Fit."""
        if self.dataset_a is None and self.dataset_b is None:
            QMessageBox.warning(self, "Warning", "Please load at least one 3D focus stack first.")
            return

        cross_idx = self.crosshair_x if self.profile_mode == "X" else self.crosshair_y
        win = self.fit_win_spin.value()

        # Compute and plot FWHM vs Z focus curves for all detected physical edges across all Z frames
        self._compute_and_plot_fwhm_vs_z(cross_idx, win)

    def _compute_and_plot_fwhm_vs_z(self, cross_idx, win):
        """Compute FWHM across all Z focus frames for ALL detected physical edges and plot FWHM(Z) for Dataset A and Dataset B."""
        is_rect = getattr(self, 'is_rectangle_roi_active', False)
        rect_c = self.rect_coords if is_rect else None
        line_w = self.line_width_spin.value() if hasattr(self, 'line_width_spin') and not is_rect else 1
        smooth_sigma = self.smooth_sigma_spin.value() if hasattr(self, 'smooth_sigma_spin') and not is_rect else 0.0
        model_type = self.fit_model_combo.currentText() if hasattr(self, 'fit_model_combo') else "Error Function (erf)"
        z_frame_current = self.current_frame

        thresh_pct = self.peak_thresh_spin.value() / 100.0 if hasattr(self, 'peak_thresh_spin') else 0.20
        min_dist = float(self.min_dist_spin.value()) if hasattr(self, 'min_dist_spin') else 35.0
        px_scale = self.get_pixel_scale()
        z_scale = self.get_z_scale()
        px_unit = "µm" if px_scale != 1.0 else "px"
        z_unit = "µm" if z_scale != 1.0 else "frame"

        # Distinct high-contrast colors & markers for Dataset A and Dataset B edges
        colors_a = ['#D32F2F', '#1976D2', '#F57C00', '#7B1FA2', '#0097A7', '#C2185B']
        colors_b = ['#388E3C', '#00796B', '#E65100', '#512DA8', '#AFB42B', '#D81B60']

        summary_msgs = []
        self.log_msg(f"=== Focus Depth Curve FWHM(Z) Results [{model_type}] (Thresh={thresh_pct*100:.0f}%, MinDist={min_dist:.0f}px) ===")

        dlg_plot = QDialog(self)
        dlg_plot.setWindowTitle(f"Focus Sharpness Curve: Multi-Edge FWHM vs Z ({self.profile_mode} Profile)")
        dlg_plot.setMinimumSize(920, 600)
        vbox = QVBoxLayout(dlg_plot)

        fig = Figure(figsize=(9.2, 5.4), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # ---------------- Dataset A Focus Analysis across Z ----------------
        if self.dataset_a is not None:
            H, W, Z = self.dataset_a.shape
            prof_ref_a = extract_line_profile(self.dataset_a, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame_current, line_width=line_w, rect_coords=rect_c)
            if prof_ref_a is not None:
                prof_peak_disc_a = gaussian_filter1d(prof_ref_a, sigma=4.0)
                deriv_ref_a = np.abs(np.gradient(prof_peak_disc_a, 5.0))
                max_d_a = np.max(deriv_ref_a) if len(deriv_ref_a) > 0 else 0
                if max_d_a > 0:
                    peaks_ref_a, _ = signal.find_peaks(deriv_ref_a, distance=int(min_dist), height=max_d_a * thresh_pct)
                    cand_ref_a = [calculate_erf_fit(prof_ref_a, pk, window=len(prof_ref_a) if is_rect else win) for pk in peaks_ref_a]
                    clean_ref_a = deduplicate_edge_fits(cand_ref_a, min_center_distance=min_dist)
                else:
                    clean_ref_a = []

                if not clean_ref_a:
                    clean_ref_a = [(0.0, [0, cross_idx, 1.0, 0], 1.0, None, None)]

                for edge_idx, ref_fit in enumerate(clean_ref_a):
                    ref_x0 = ref_fit[1][1] if ref_fit[1] is not None else cross_idx
                    ref_dir = "Rising" if (ref_fit[1][0] > 0 if model_type == "Error Function (erf)" else 1) else "Falling"
                    fwhm_list = []
                    z_list = []

                    for z in range(Z):
                        prof_z = extract_line_profile(self.dataset_a, self.profile_mode, self.crosshair_x, self.crosshair_y, z, line_width=line_w, rect_coords=rect_c)
                        if prof_z is not None:
                            if model_type == "Error Function (erf)":
                                fw, _, r2, _, _ = calculate_erf_fit(prof_z, int(round(np.clip(ref_x0, 0, len(prof_z)-1))), window=len(prof_z) if is_rect else win)
                            else:
                                prof_smooth_z = gaussian_filter1d(prof_z, sigma=smooth_sigma) if smooth_sigma > 0 else prof_z
                                deriv_z = np.abs(np.gradient(prof_smooth_z, 5.0))
                                fw, _, r2, _, _ = calculate_fwhm(deriv_z, int(round(np.clip(ref_x0, 0, len(deriv_z)-1))), window=len(prof_z) if is_rect else win, smooth_sigma=smooth_sigma)
                            fwhm_list.append(fw)
                            z_list.append(z)

                    arr_fwhm_raw = np.array(fwhm_list)
                    valid_idx = np.where(np.isfinite(arr_fwhm_raw))[0]
                    c_col = colors_a[edge_idx % len(colors_a)]

                    z_list_cal = np.array(z_list) * z_scale
                    arr_fwhm_cal = arr_fwhm_raw * px_scale

                    z_min_val = valid_idx[np.argmin(arr_fwhm_raw[valid_idx])] if len(valid_idx) > 0 else None
                    z_min_str = f"Z_focus={z_min_val*z_scale:.2f} {z_unit}" if (z_min_val is not None and z_scale != 1.0) else (f"Z_focus={z_min_val}" if z_min_val is not None else "N/A")
                    x0_cal_str = f"x0={ref_x0*px_scale:.2f} {px_unit}" if px_scale != 1.0 else f"x0={ref_x0:.1f}px"

                    lbl_curve = f"{self.name_a} Edge #{edge_idx+1} ({ref_dir}, {x0_cal_str}) - {z_min_str}"
                    ax.plot(z_list_cal, arr_fwhm_cal, 'o-', color=c_col, linewidth=2.2, markersize=5, label=lbl_curve)

                    if z_min_val is not None:
                        z_min_pos = z_min_val * z_scale
                        min_fw_cal = arr_fwhm_cal[z_min_val]
                        ax.axvline(z_min_pos, color=c_col, linestyle=':', alpha=0.7)
                        self.log_msg(f"  {self.name_a} Edge #{edge_idx+1} ({ref_dir} {x0_cal_str}): Sharpest Focus Z = {z_min_val} (Pos={z_min_pos:.2f} {z_unit}) | Min FWHM = {min_fw_cal:.2f} {px_unit}")
                        summary_msgs.append(f"<b>{self.name_a} Edge #{edge_idx+1} ({ref_dir}, {x0_cal_str})</b>: Sharpest Focus at <b>Z = {z_min_val} ({z_min_pos:.2f} {z_unit})</b> (Min FWHM = {min_fw_cal:.2f} {px_unit})")

        # ---------------- Dataset B Focus Analysis across Z ----------------
        if self.dataset_b is not None:
            H, W, Z = self.dataset_b.shape
            prof_ref_b = extract_line_profile(self.dataset_b, self.profile_mode, self.crosshair_x, self.crosshair_y, z_frame_current, line_width=line_w, rect_coords=rect_c)
            if prof_ref_b is not None:
                prof_peak_disc_b = gaussian_filter1d(prof_ref_b, sigma=4.0)
                deriv_ref_b = np.abs(np.gradient(prof_peak_disc_b, 5.0))
                max_d_b = np.max(deriv_ref_b) if len(deriv_ref_b) > 0 else 0
                if max_d_b > 0:
                    peaks_ref_b, _ = signal.find_peaks(deriv_ref_b, distance=int(min_dist), height=max_d_b * thresh_pct)
                    cand_ref_b = [calculate_erf_fit(prof_ref_b, pk, window=len(prof_ref_b) if is_rect else win) for pk in peaks_ref_b]
                    clean_ref_b = deduplicate_edge_fits(cand_ref_b, min_center_distance=min_dist)
                else:
                    clean_ref_b = []

                if not clean_ref_b:
                    clean_ref_b = [(0.0, [0, cross_idx, 1.0, 0], 1.0, None, None)]

                for edge_idx, ref_fit in enumerate(clean_ref_b):
                    ref_x0 = ref_fit[1][1] if ref_fit[1] is not None else cross_idx
                    ref_dir = "Rising" if (ref_fit[1][0] > 0 if model_type == "Error Function (erf)" else 1) else "Falling"
                    fwhm_list = []
                    z_list = []

                    for z in range(Z):
                        prof_z = extract_line_profile(self.dataset_b, self.profile_mode, self.crosshair_x, self.crosshair_y, z, line_width=line_w, rect_coords=rect_c)
                        if prof_z is not None:
                            if model_type == "Error Function (erf)":
                                fw, _, r2, _, _ = calculate_erf_fit(prof_z, int(round(np.clip(ref_x0, 0, len(prof_z)-1))), window=len(prof_z) if is_rect else win)
                            else:
                                prof_smooth_z = gaussian_filter1d(prof_z, sigma=smooth_sigma) if smooth_sigma > 0 else prof_z
                                deriv_z = np.abs(np.gradient(prof_smooth_z, 5.0))
                                fw, _, r2, _, _ = calculate_fwhm(deriv_z, int(round(np.clip(ref_x0, 0, len(deriv_z)-1))), window=len(prof_z) if is_rect else win, smooth_sigma=smooth_sigma)
                            fwhm_list.append(fw)
                            z_list.append(z)

                    arr_fwhm_raw = np.array(fwhm_list)
                    valid_idx = np.where(np.isfinite(arr_fwhm_raw))[0]
                    c_col = colors_b[edge_idx % len(colors_b)]

                    z_list_cal = np.array(z_list) * z_scale
                    arr_fwhm_cal = arr_fwhm_raw * px_scale

                    z_min_val = valid_idx[np.argmin(arr_fwhm_raw[valid_idx])] if len(valid_idx) > 0 else None
                    z_min_str = f"Z_focus={z_min_val*z_scale:.2f} {z_unit}" if (z_min_val is not None and z_scale != 1.0) else (f"Z_focus={z_min_val}" if z_min_val is not None else "N/A")
                    x0_cal_str = f"x0={ref_x0*px_scale:.2f} {px_unit}" if px_scale != 1.0 else f"x0={ref_x0:.1f}px"

                    lbl_curve = f"{self.name_b} Edge #{edge_idx+1} ({ref_dir}, {x0_cal_str}) - {z_min_str}"
                    ax.plot(z_list_cal, arr_fwhm_cal, 'o--', color=c_col, linewidth=2.2, markersize=5, label=lbl_curve)

                    if z_min_val is not None:
                        z_min_pos = z_min_val * z_scale
                        min_fw_cal = arr_fwhm_cal[z_min_val]
                        ax.axvline(z_min_pos, color=c_col, linestyle=':', alpha=0.7)
                        self.log_msg(f"  {self.name_b} Edge #{edge_idx+1} ({ref_dir} {x0_cal_str}): Sharpest Focus Z = {z_min_val} (Pos={z_min_pos:.2f} {z_unit}) | Min FWHM = {min_fw_cal:.2f} {px_unit}")
                        summary_msgs.append(f"<b>{self.name_b} Edge #{edge_idx+1} ({ref_dir}, {x0_cal_str})</b>: Sharpest Focus at <b>Z = {z_min_val} ({z_min_pos:.2f} {z_unit})</b> (Min FWHM = {min_fw_cal:.2f} {px_unit})")

        ax.set_xlabel(f"Z Focus Depth Position ({z_unit})")
        ax.set_ylabel(f"Edge Blurring FWHM ({px_unit})")
        ax.set_ylim(bottom=0)
        ax.set_title(f"Multi-Edge Focus Sharpness (FWHM vs Z) [{model_type}]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()

        vbox.addWidget(canvas)

        info_lbl = QLabel("<br>".join(summary_msgs) if summary_msgs else "No valid edge FWHM focus curves computed.")
        info_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #1565C0;")
        vbox.addWidget(info_lbl)

        # Action Buttons Row (Export & Close)
        btn_bar = QHBoxLayout()
        btn_export_fc = QPushButton("Export Focus Curve (PNG, SVG, CSV)")
        btn_export_fc.setStyleSheet("font-weight: bold; padding: 5px 12px;")
        btn_export_fc.setToolTip("Export focus curve plot (PNG, SVG) or numerical FWHM(Z) curve data (CSV)")

        def export_focus_curve():
            filepath, _ = QFileDialog.getSaveFileName(
                dlg_plot, "Export Focus Curve Plot or CSV Data",
                "focus_curve_fwhm_vs_z.png",
                "PNG Image (*.png);;Vector SVG (*.svg);;CSV Data Table (*.csv)"
            )
            if not filepath:
                return

            ext = Path(filepath).suffix.lower()
            if ext == '.csv':
                with open(filepath, 'w') as f:
                    f.write(f"# Focus Sharpness Curve Data [{model_type}] Mode={self.profile_mode} Crosshair={cross_idx}\n")
                    f.write(f"# PixelScale={px_scale}um/px, ZScale={z_scale}um/frame\n")
                    for summary_line in summary_msgs:
                        f.write(f"# {summary_line.replace('<b>','').replace('</b>','')}\n")

                QMessageBox.information(dlg_plot, "Export Successful", f"Saved focus curve data to:\n{filepath}")
            else:
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                QMessageBox.information(dlg_plot, "Export Successful", f"Saved focus curve plot image to:\n{filepath}")

        btn_export_fc.clicked.connect(export_focus_curve)
        btn_bar.addWidget(btn_export_fc)

        btn_bar.addStretch()
        btn_close_fc = QPushButton("Close")
        btn_close_fc.clicked.connect(dlg_plot.accept)
        btn_bar.addWidget(btn_close_fc)

        vbox.addLayout(btn_bar)

        dlg_plot.exec_()

    # ---------------- Grid Focus Map (Full Image Analysis) ----------------

    def run_grid_focus_map(self):
        """Run automated grid line edge detection across all Z frames and plot 2D minimum FWHM focus map."""
        target_cube = self.dataset_a if self.dataset_a is not None else self.hypercube
        if target_cube is None:
            QMessageBox.warning(self, "Warning", "Please load a 3D focus stack first.")
            return

        H, W, Z = target_cube.shape
        num_lines = 25
        win = self.fit_win_spin.value()

        ys = np.linspace(5, H - 6, num_lines, dtype=int)
        xs = np.linspace(5, W - 6, num_lines, dtype=int)

        prog = QProgressDialog("Analyzing grid edge sharpness across all Z frames…", "Cancel", 0, len(ys) + len(xs), self)
        prog.setWindowModality(Qt.WindowModal)

        pts_x, pts_y, min_z_vals = [], [], []

        count = 0
        # Horizontal lines (Y fixed, scan X)
        for y in ys:
            if prog.wasCanceled():
                return
            count += 1
            prog.setValue(count)

            stack_prof = target_cube[y, :, :]  # (W, Z)
            deriv = np.abs(np.gradient(stack_prof, axis=0))  # (W, Z)

            peaks, _ = signal.find_peaks(deriv[:, Z // 2], distance=45, height=np.max(deriv) * 0.2)
            for pk in peaks:
                fwhms = []
                for z in range(Z):
                    fw, _, _, _, _ = calculate_fwhm(deriv[:, z], pk, window=win)
                    fwhms.append(fw)
                fwhms = np.array(fwhms)
                valid = np.where(np.isfinite(fwhms))[0]
                if len(valid) > 0:
                    z_min = valid[np.argmin(fwhms[valid])]
                    pts_x.append(pk)
                    pts_y.append(y)
                    min_z_vals.append(z_min)

        # Vertical lines (X fixed, scan Y)
        for x in xs:
            if prog.wasCanceled():
                return
            count += 1
            prog.setValue(count)

            stack_prof = target_cube[:, x, :]  # (H, Z)
            deriv = np.abs(np.gradient(stack_prof, axis=0))  # (H, Z)

            peaks, _ = signal.find_peaks(deriv[:, Z // 2], distance=45, height=np.max(deriv) * 0.2)
            for pk in peaks:
                fwhms = []
                for z in range(Z):
                    fw, _, _, _, _ = calculate_fwhm(deriv[:, z], pk, window=win)
                    fwhms.append(fw)
                fwhms = np.array(fwhms)
                valid = np.where(np.isfinite(fwhms))[0]
                if len(valid) > 0:
                    z_min = valid[np.argmin(fwhms[valid])]
                    pts_x.append(x)
                    pts_y.append(pk)
                    min_z_vals.append(z_min)

        prog.close()

        if len(pts_x) < 4:
            QMessageBox.warning(self, "Warning", "Could not detect enough grid edge peaks for 2D focus map.")
            return

        # Build 2D Focus Map Dialog
        dlg_map = QDialog(self)
        dlg_map.setWindowTitle("2D Sharpest Focus Map Z_min(x, y)")
        dlg_map.setMinimumSize(750, 650)
        vbox = QVBoxLayout(dlg_map)

        fig = Figure(figsize=(7, 6), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        X_arr = np.array(pts_x, dtype=float)
        Y_arr = np.array(pts_y, dtype=float)
        Z_arr = np.array(min_z_vals, dtype=float)

        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_z = griddata((X_arr, Y_arr), Z_arr, (grid_x, grid_y), method='linear')

        im = ax.imshow(grid_z, cmap='viridis', origin='upper', vmin=0, vmax=Z - 1)
        ax.scatter(X_arr, Y_arr, c=Z_arr, cmap='viridis', edgecolors='k', linewidths=0.5, s=20)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Sharpest Focus Frame Index Z_min")

        ax.set_title("Interpolated 2D Focus Map Z_min(x, y)")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")
        fig.tight_layout()

        vbox.addWidget(canvas)

        btn_close = QPushButton("Close Map")
        btn_close.clicked.connect(dlg_map.accept)
        vbox.addWidget(btn_close)

        dlg_map.exec_()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = KnifeEdgeViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
