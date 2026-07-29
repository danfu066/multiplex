#!/usr/bin/env python3
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

from PyQt5.QtCore import Qt
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


def gaussian_func(x, a, x0, sigma, offset):
    """Gaussian function for edge derivative fitting."""
    return a * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2)) + offset


def calculate_fwhm(signal_1d, peak_idx, window=25, smooth_sigma=0.0):
    """Fit a Gaussian curve to a 1D derivative signal around peak_idx with optional noise smoothing."""
    signal_1d = np.asarray(signal_1d, dtype=float)
    if smooth_sigma > 0:
        signal_1d = gaussian_filter1d(signal_1d, sigma=smooth_sigma)
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
        fwhm = 2.35482 * sigma

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
        self.win_spin.setRange(5, 100)
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
            deriv_a = np.abs(np.gradient(self.profile_a))
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
            deriv_b = np.abs(np.gradient(self.profile_b))
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

        # Config Row: Fit Window & Noise Smoothing
        row_cfg = QHBoxLayout()
        row_cfg.addWidget(QLabel("Fit Window (px):"))
        self.fit_win_spin = QSpinBox()
        self.fit_win_spin.setRange(5, 150)
        self.fit_win_spin.setValue(25)
        row_cfg.addWidget(self.fit_win_spin)

        row_cfg.addWidget(QLabel("Smooth Sigma (px):"))
        self.smooth_sigma_spin = QDoubleSpinBox()
        self.smooth_sigma_spin.setRange(0.0, 10.0)
        self.smooth_sigma_spin.setSingleStep(0.5)
        self.smooth_sigma_spin.setValue(2.0)
        self.smooth_sigma_spin.setToolTip("Gaussian noise smoothing filter for high-noise real data")
        row_cfg.addWidget(self.smooth_sigma_spin)
        fit_layout.addLayout(row_cfg)

        # Step 1 & 2 Action Buttons
        row_btns1 = QHBoxLayout()
        self.btn_differential = QPushButton("1. Differential (dI/dx)")
        self.btn_differential.setToolTip("Computes 1D derivative profile |dI/dx| with noise smoothing and updates plot")
        self.btn_differential.clicked.connect(self.run_differential_step)
        row_btns1.addWidget(self.btn_differential)

        self.btn_gaussian_fit = QPushButton("2. Gaussian Fit")
        self.btn_gaussian_fit.setToolTip("Finds edge peaks, fits Gaussian curves, and outputs peak centers & FWHM widths")
        self.btn_gaussian_fit.clicked.connect(self.run_gaussian_fit_step)
        row_btns1.addWidget(self.btn_gaussian_fit)
        fit_layout.addLayout(row_btns1)

        # Step 3 & 4 Action Buttons
        row_btns2 = QHBoxLayout()
        self.btn_inspect_fit = QPushButton("3. Focus Curve (FWHM vs Z)")
        self.btn_inspect_fit.setToolTip("Evaluates FWHM(Z) curves across frames Z and computes focus separation ΔZ")
        self.btn_inspect_fit.clicked.connect(self.fit_local_edge_with_preview)
        row_btns2.addWidget(self.btn_inspect_fit)

        self.btn_grid_map = QPushButton("4. Grid Focus Map")
        self.btn_grid_map.setToolTip("Runs automated grid line analysis and plots 2D sharpest focus map Z_min(x,y)")
        self.btn_grid_map.clicked.connect(self.run_grid_focus_map)
        row_btns2.addWidget(self.btn_grid_map)
        fit_layout.addLayout(row_btns2)

        # Text Output Window Log
        fit_layout.addWidget(QLabel("Analysis Results Log:"))
        self.log_text_window = QTextEdit()
        self.log_text_window.setReadOnly(True)
        self.log_text_window.setMaximumHeight(150)
        self.log_text_window.setFont(QFont("Consolas", 9))
        self.log_text_window.setPlaceholderText("Analysis results log will appear here step-by-step...")
        fit_layout.addWidget(self.log_text_window)

        row_log_ctrl = QHBoxLayout()
        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.clicked.connect(self.log_text_window.clear)
        row_log_ctrl.addStretch()
        row_log_ctrl.addWidget(btn_clear_log)
        fit_layout.addLayout(row_log_ctrl)

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

        # Hide Unwanted Background & Reference Spectra UI Elements
        unwanted_names = [
            'set_bg_btn', 'subtract_bg_btn', 'bg_status_label',
            'load_ref_btn', 'transform_ref_btn', 'save_ref_btn',
            'ref_status_label', 'ref_text_box', 'bg_text_box'
        ]
        for name in unwanted_names:
            if hasattr(self, name):
                w = getattr(self, name)
                if w is not None and isinstance(w, QWidget):
                    w.hide()

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

    def on_canvas_press(self, event):
        super().on_canvas_press(event)
        self.update_line_profile_plot()

    def update_line_profile_plot(self):
        """Update 1D line profile plot overlaying Dataset A (solid) and Dataset B (dashed)."""
        if self.dataset_a is None and self.dataset_b is None:
            return

        self.spectrum_ax.clear()
        z_frame = self.current_frame

        # Line Profile Dataset A
        if self.dataset_a is not None and z_frame < self.dataset_a.shape[2]:
            H, W, Z = self.dataset_a.shape
            if self.profile_mode == "X":
                y = int(np.clip(self.crosshair_y, 0, H - 1))
                prof_a = self.dataset_a[y, :, z_frame]
                x_vals = np.arange(W)
                lbl_title = f"X Profile along Row Y = {y}"
            else:
                x = int(np.clip(self.crosshair_x, 0, W - 1))
                prof_a = self.dataset_a[:, x, z_frame]
                x_vals = np.arange(H)
                lbl_title = f"Y Profile along Column X = {x}"

            self.spectrum_ax.plot(x_vals, prof_a, 'b-', label=f"{self.name_a} (Z={z_frame})")

        # Line Profile Dataset B
        if self.dataset_b is not None and z_frame < self.dataset_b.shape[2]:
            H, W, Z = self.dataset_b.shape
            if self.profile_mode == "X":
                y = int(np.clip(self.crosshair_y, 0, H - 1))
                prof_b = self.dataset_b[y, :, z_frame]
                x_vals = np.arange(W)
                lbl_title = f"X Profile along Row Y = {y}"
            else:
                x = int(np.clip(self.crosshair_x, 0, W - 1))
                prof_b = self.dataset_b[:, x, z_frame]
                x_vals = np.arange(H)
                lbl_title = f"Y Profile along Column X = {x}"

            self.spectrum_ax.plot(x_vals, prof_b, 'g--', label=f"{self.name_b} (Z={z_frame})")

        self.spectrum_ax.set_xlabel("Pixel Position (px)")
        self.spectrum_ax.set_ylabel("Raw Intensity I")
        self.spectrum_ax.set_title(lbl_title if 'lbl_title' in locals() else "1D Line Profile")
        self.spectrum_ax.grid(True, alpha=0.3)

        # Plot saved line selections
        if hasattr(self, 'spectra_list') and self.spectra_list:
            for sp in self.spectra_list:
                if sp.visible:
                    x_vals = np.arange(len(sp.spectrum))
                    self.spectrum_ax.plot(x_vals, sp.spectrum, color=sp.color, linewidth=1.5, linestyle=':', label=sp.label)

        self.spectrum_ax.legend(loc='upper right', fontsize=8)
        self.spectrum_canvas.draw_idle()

    def add_current_spectrum(self):
        """Add current 1D line profile selection to Line Management list."""
        if self.dataset_a is None and self.dataset_b is None:
            return

        z_frame = self.current_frame
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        color = colors[len(self.spectra_list) % len(colors)]

        # Dataset A profile
        if self.dataset_a is not None and z_frame < self.dataset_a.shape[2]:
            H, W, Z = self.dataset_a.shape
            if self.profile_mode == "X":
                y = int(np.clip(self.crosshair_y, 0, H - 1))
                prof = self.dataset_a[y, :, z_frame]
                lbl = f"{self.name_a} X-Line (Y={y}, Z={z_frame})"
            else:
                x = int(np.clip(self.crosshair_x, 0, W - 1))
                prof = self.dataset_a[:, x, z_frame]
                lbl = f"{self.name_a} Y-Line (X={x}, Z={z_frame})"

            from HyperViewer import SpectrumData
            sp_a = SpectrumData(
                spectrum=prof,
                label=lbl,
                color=color,
                selection_type='line',
                coords=(self.crosshair_x, self.crosshair_y)
            )
            self.spectra_list.append(sp_a)

        # Dataset B profile
        if self.dataset_b is not None and z_frame < self.dataset_b.shape[2]:
            H, W, Z = self.dataset_b.shape
            if self.profile_mode == "X":
                y = int(np.clip(self.crosshair_y, 0, H - 1))
                prof = self.dataset_b[y, :, z_frame]
                lbl = f"{self.name_b} X-Line (Y={y}, Z={z_frame})"
            else:
                x = int(np.clip(self.crosshair_x, 0, W - 1))
                prof = self.dataset_b[:, x, z_frame]
                lbl = f"{self.name_b} Y-Line (X={x}, Z={z_frame})"

            from HyperViewer import SpectrumData
            sp_b = SpectrumData(
                spectrum=prof,
                label=lbl,
                color=color,
                selection_type='line',
                coords=(self.crosshair_x, self.crosshair_y)
            )
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
        """Step 1: Compute 1D derivative profile |dI/dx| with noise smoothing and update plot."""
        if self.dataset_a is None and self.dataset_b is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        z_frame = self.current_frame
        smooth_sigma = self.smooth_sigma_spin.value()
        self.spectrum_ax.clear()

        cross_idx = self.crosshair_x if self.profile_mode == "X" else self.crosshair_y
        self.log_msg(f"=== Step 1: Differential |dI/dx| (Mode={self.profile_mode}, Crosshair={cross_idx}px, Z={z_frame}, Smooth={smooth_sigma:.1f}px) ===")

        # Dataset A Differential
        if self.dataset_a is not None and z_frame < self.dataset_a.shape[2]:
            H, W, Z = self.dataset_a.shape
            prof_a = self.dataset_a[int(np.clip(self.crosshair_y, 0, H-1)), :, z_frame] if self.profile_mode == "X" else self.dataset_a[:, int(np.clip(self.crosshair_x, 0, W-1)), z_frame]
            deriv_a = np.abs(np.gradient(prof_a))
            if smooth_sigma > 0:
                deriv_a = gaussian_filter1d(deriv_a, sigma=smooth_sigma)
            x_vals = np.arange(len(deriv_a))
            self.spectrum_ax.plot(x_vals, deriv_a, 'b-', label=f"{self.name_a} |dI/dx|")
            self.log_msg(f"  {self.name_a}: Max Derivative = {np.max(deriv_a):.2f}")

        # Dataset B Differential
        if self.dataset_b is not None and z_frame < self.dataset_b.shape[2]:
            H, W, Z = self.dataset_b.shape
            prof_b = self.dataset_b[int(np.clip(self.crosshair_y, 0, H-1)), :, z_frame] if self.profile_mode == "X" else self.dataset_b[:, int(np.clip(self.crosshair_x, 0, W-1)), z_frame]
            deriv_b = np.abs(np.gradient(prof_b))
            if smooth_sigma > 0:
                deriv_b = gaussian_filter1d(deriv_b, sigma=smooth_sigma)
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
        """Step 2: Detect derivative peaks, fit Gaussian curves, overlay on plot, and log results."""
        if self.dataset_a is None and self.dataset_b is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        z_frame = self.current_frame
        win = self.fit_win_spin.value()
        smooth_sigma = self.smooth_sigma_spin.value()

        self.spectrum_ax.clear()
        cross_idx = self.crosshair_x if self.profile_mode == "X" else self.crosshair_y
        self.log_msg(f"=== Step 2: Multi-Peak Gaussian Fit (Window={win}px, Smooth={smooth_sigma:.1f}px, Z={z_frame}) ===")

        # Dataset A Multi-Peak Gaussian Fit
        if self.dataset_a is not None and z_frame < self.dataset_a.shape[2]:
            H, W, Z = self.dataset_a.shape
            prof_a = self.dataset_a[int(np.clip(self.crosshair_y, 0, H-1)), :, z_frame] if self.profile_mode == "X" else self.dataset_a[:, int(np.clip(self.crosshair_x, 0, W-1)), z_frame]
            deriv_a = np.abs(np.gradient(prof_a))
            if smooth_sigma > 0:
                deriv_a = gaussian_filter1d(deriv_a, sigma=smooth_sigma)

            x_vals = np.arange(len(deriv_a))
            self.spectrum_ax.plot(x_vals, deriv_a, 'b.', alpha=0.4, label=f"{self.name_a} |dI/dx|")

            max_d_a = np.max(deriv_a) if len(deriv_a) > 0 else 0
            if max_d_a > 0:
                peaks_a, _ = signal.find_peaks(deriv_a, distance=35, height=max_d_a * 0.15)
                fitted_count = 0
                for pk in peaks_a:
                    fw, popt, r2, x_fit, y_fit = calculate_fwhm(deriv_a, pk, window=win)
                    if popt is not None and x_fit is not None and np.isfinite(fw) and r2 > 0.2:
                        lbl_fit = f"{self.name_a} Fits" if fitted_count == 0 else ""
                        self.spectrum_ax.plot(x_fit, y_fit, 'b-', linewidth=2, label=lbl_fit)
                        self.log_msg(
                            f"  {self.name_a} Peak #{fitted_count+1}: Center x0 = {popt[1]:.1f} px | "
                            f"FWHM = {fw:.2f} px | Amp = {popt[0]:.1f} | R^2 = {r2:.3f}"
                        )
                        fitted_count += 1

        # Dataset B Multi-Peak Gaussian Fit
        if self.dataset_b is not None and z_frame < self.dataset_b.shape[2]:
            H, W, Z = self.dataset_b.shape
            prof_b = self.dataset_b[int(np.clip(self.crosshair_y, 0, H-1)), :, z_frame] if self.profile_mode == "X" else self.dataset_b[:, int(np.clip(self.crosshair_x, 0, W-1)), z_frame]
            deriv_b = np.abs(np.gradient(prof_b))
            if smooth_sigma > 0:
                deriv_b = gaussian_filter1d(deriv_b, sigma=smooth_sigma)

            x_vals = np.arange(len(deriv_b))
            self.spectrum_ax.plot(x_vals, deriv_b, 'g.', alpha=0.4, label=f"{self.name_b} |dI/dx|")

            max_d_b = np.max(deriv_b) if len(deriv_b) > 0 else 0
            if max_d_b > 0:
                peaks_b, _ = signal.find_peaks(deriv_b, distance=35, height=max_d_b * 0.15)
                fitted_count = 0
                for pk in peaks_b:
                    fw, popt, r2, x_fit, y_fit = calculate_fwhm(deriv_b, pk, window=win)
                    if popt is not None and x_fit is not None and np.isfinite(fw) and r2 > 0.2:
                        lbl_fit = f"{self.name_b} Fits" if fitted_count == 0 else ""
                        self.spectrum_ax.plot(x_fit, y_fit, 'g--', linewidth=2, label=lbl_fit)
                        self.log_msg(
                            f"  {self.name_b} Peak #{fitted_count+1}: Center x0 = {popt[1]:.1f} px | "
                            f"FWHM = {fw:.2f} px | Amp = {popt[0]:.1f} | R^2 = {r2:.3f}"
                        )
                        fitted_count += 1

        self.spectrum_ax.set_xlabel("Pixel Position (px)")
        self.spectrum_ax.set_ylabel("|dI/dx| Edge Derivative")
        self.spectrum_ax.set_title(f"1D Multi-Peak Gaussian Edge Fits (Frame Z={z_frame})")
        self.spectrum_ax.grid(True, alpha=0.3)
        self.spectrum_ax.legend(loc='upper right', fontsize=8)
        self.spectrum_canvas.draw_idle()

    # ---------------- Local Edge Fitting & Inspection ----------------

    def fit_local_edge_with_preview(self):
        """Perform interactive Gaussian fit preview, then compute FWHM vs Z focus curves for Dataset A & B."""
        if self.dataset_a is None and self.dataset_b is None:
            QMessageBox.warning(self, "Warning", "Please load at least one 3D focus stack first.")
            return

        z_frame = self.current_frame
        win = self.fit_win_spin.value()

        # Extract profiles at crosshair
        prof_a = None
        prof_b = None
        cross_idx = self.crosshair_x if self.profile_mode == "X" else self.crosshair_y

        if self.dataset_a is not None:
            H, W, Z = self.dataset_a.shape
            if self.profile_mode == "X":
                prof_a = self.dataset_a[int(np.clip(self.crosshair_y, 0, H - 1)), :, z_frame]
                x_vals = np.arange(W)
            else:
                prof_a = self.dataset_a[:, int(np.clip(self.crosshair_x, 0, W - 1)), z_frame]
                x_vals = np.arange(H)

        if self.dataset_b is not None:
            H, W, Z = self.dataset_b.shape
            if self.profile_mode == "X":
                prof_b = self.dataset_b[int(np.clip(self.crosshair_y, 0, H - 1)), :, z_frame]
                x_vals = np.arange(W)
            else:
                prof_b = self.dataset_b[:, int(np.clip(self.crosshair_x, 0, W - 1)), z_frame]
                x_vals = np.arange(H)

        # Show interactive fit inspection window
        dlg = GaussianFitInspectDialog(
            self, x_vals, prof_a, prof_b, cross_idx, z_frame,
            window_size=win, label_a=self.name_a, label_b=self.name_b
        )

        if dlg.exec_() != QDialog.Accepted:
            return

        win = dlg.win_spin.value()

        # Compute FWHM vs Z curves across all frames
        self._compute_and_plot_fwhm_vs_z(cross_idx, win)

    def _compute_and_plot_fwhm_vs_z(self, cross_idx, win):
        """Compute FWHM across all Z focus frames and plot FWHM(Z) for Dataset A and Dataset B."""
        fwhm_list_a, z_list_a = [], []
        fwhm_list_b, z_list_b = [], []

        # Dataset A FWHM vs Z
        if self.dataset_a is not None:
            H, W, Z = self.dataset_a.shape
            for z in range(Z):
                prof = self.dataset_a[int(np.clip(self.crosshair_y, 0, H - 1)), :, z] if self.profile_mode == "X" else self.dataset_a[:, int(np.clip(self.crosshair_x, 0, W - 1)), z]
                deriv = np.abs(np.gradient(prof))
                peaks, _ = signal.find_peaks(deriv, distance=20)
                pk = peaks[np.argmin(np.abs(peaks - cross_idx))] if len(peaks) > 0 else cross_idx
                fw, _, r2, _, _ = calculate_fwhm(deriv, pk, window=win)
                fwhm_list_a.append(fw)
                z_list_a.append(z)

        # Dataset B FWHM vs Z
        if self.dataset_b is not None:
            H, W, Z = self.dataset_b.shape
            for z in range(Z):
                prof = self.dataset_b[int(np.clip(self.crosshair_y, 0, H - 1)), :, z] if self.profile_mode == "X" else self.dataset_b[:, int(np.clip(self.crosshair_x, 0, W - 1)), z]
                deriv = np.abs(np.gradient(prof))
                peaks, _ = signal.find_peaks(deriv, distance=20)
                pk = peaks[np.argmin(np.abs(peaks - cross_idx))] if len(peaks) > 0 else cross_idx
                fw, _, r2, _, _ = calculate_fwhm(deriv, pk, window=win)
                fwhm_list_b.append(fw)
                z_list_b.append(z)

        # Plot FWHM vs Z Dialog
        dlg_plot = QDialog(self)
        dlg_plot.setWindowTitle(f"Focus Sharpness Curve: FWHM vs Z ({self.profile_mode} Profile)")
        dlg_plot.setMinimumSize(700, 500)
        vbox = QVBoxLayout(dlg_plot)

        fig = Figure(figsize=(7, 4.5), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        summary_msgs = []
        self.log_msg(f"=== Step 3: Focus Depth Curve FWHM(Z) Results ===")

        if fwhm_list_a:
            arr_a = np.array(fwhm_list_a)
            ax.plot(z_list_a, arr_a, 'bo-', linewidth=2, label=f"{self.name_a} FWHM(Z)")
            valid_a = np.where(np.isfinite(arr_a))[0]
            if len(valid_a) > 0:
                z_min_a = valid_a[np.argmin(arr_a[valid_a])]
                min_fwhm_a = arr_a[z_min_a]
                ax.axvline(z_min_a, color='b', linestyle=':', label=f"{self.name_a} Sharpest Focus Z={z_min_a}")
                summary_msgs.append(f"<b>{self.name_a}</b>: Sharpest Focus at Frame <b>Z = {z_min_a}</b> (Min FWHM = {min_fwhm_a:.2f} px)")
                self.log_msg(f"  {self.name_a}: Sharpest Focus at Frame Z = {z_min_a} (Min FWHM = {min_fwhm_a:.2f} px)")

        if fwhm_list_b:
            arr_b = np.array(fwhm_list_b)
            ax.plot(z_list_b, arr_b, 'gs--', linewidth=2, label=f"{self.name_b} FWHM(Z)")
            valid_b = np.where(np.isfinite(arr_b))[0]
            if len(valid_b) > 0:
                z_min_b = valid_b[np.argmin(arr_b[valid_b])]
                min_fwhm_b = arr_b[z_min_b]
                ax.axvline(z_min_b, color='g', linestyle=':', label=f"{self.name_b} Sharpest Focus Z={z_min_b}")
                summary_msgs.append(f"<b>{self.name_b}</b>: Sharpest Focus at Frame <b>Z = {z_min_b}</b> (Min FWHM = {min_fwhm_b:.2f} px)")
                self.log_msg(f"  {self.name_b}: Sharpest Focus at Frame Z = {z_min_b} (Min FWHM = {min_fwhm_b:.2f} px)")

        if 'z_min_a' in locals() and 'z_min_b' in locals():
            delta_z = abs(z_min_a - z_min_b)
            summary_msgs.append(f"<b>Focus Separation ΔZ</b> = <b>{delta_z} frames</b> (Z_a={z_min_a} vs Z_b={z_min_b})")
            self.log_msg(f"  Focus Separation ΔZ = {delta_z} frames (Z_a={z_min_a} vs Z_b={z_min_b})")

        ax.set_xlabel("Z Focus Depth Frame Index")
        ax.set_ylabel("Edge Blurring FWHM (px)")
        ax.set_title("Edge Blurring (FWHM) vs Focus Depth Z")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()

        vbox.addWidget(canvas)

        lbl_summary = QLabel("<br>".join(summary_msgs))
        lbl_summary.setStyleSheet("font-size: 13px; color: #0D47A1;")
        vbox.addWidget(lbl_summary)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg_plot.accept)
        vbox.addWidget(btn_close)

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
