# Copyright (c) 2026 Dan Fu@UW
"""
Hyperspectral Unmixer GUI
===========================
Interactive tool for hyperspectral image viewing, unmixing, denoising, classification,
endmember extraction, spectral fitting, and visualization.

Inherits the complete interactive desktop GUI layout from HyperViewer (spectral frame slider,
XZ/YZ cross-sections, large spectrum canvas, spectrum management, reference spectra library)
and integrates all pure numerical algorithms from the ``unmix`` package.

Usage
-----
    python unmix/Unmixer.py

Author  : Multiplexing Lab, University of Washington
Date    : 2026-07-27
Version : 2.1
"""

import sys
import os
import struct
import numpy as np

# Allow running directly or as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFileDialog, QMessageBox, QLabel, QPushButton, QSlider,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton,
    QButtonGroup, QGroupBox, QProgressBar, QInputDialog, QDialog,
    QFormLayout, QScrollArea, QAction, QTabWidget, QLineEdit, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QKeySequence

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Circle

# Optional imports with fallbacks
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from unmix import (
    run_mcr_als, run_nmf, run_mesma, run_linear_unmix, run_fcls,
    run_pca, run_mnf, run_sam, run_sid, run_rx,
    denoise_mppca, denoise_wavelet, denoise_savgol, denoise_tv3d, denoise_bm4d,
    run_nfindr, run_vca, fit_spectrum,
)


def clean_navigation_toolbar(toolbar, autoscale_callback=None):
    """
    Remove Back (<-), Forward (->), Pan (4-way double arrow), and Subplots configuration buttons
    from Matplotlib NavigationToolbar across both Image and Plot canvases.
    Optionally add an '↕ Autoscale Y' button for plot canvases.
    """
    for action in list(toolbar.actions()):
        txt = (action.text() or "").lower()
        ttip = (action.toolTip() or "").lower()
        if any(k in txt or k in ttip for k in ['back', 'forward', 'pan', 'subplots', 'configure']):
            toolbar.removeAction(action)

    if autoscale_callback is not None:
        autoscale_action = QAction("↕ Autoscale Y", toolbar)
        autoscale_action.setToolTip("Autoscale Y-axis limits to fit all visible curves")
        autoscale_action.triggered.connect(autoscale_callback)
        toolbar.addAction(autoscale_action)


class SpectrumData:
    """Store spectrum data with metadata."""
    def __init__(self, spectrum, std=None, label="", color="blue", selection_type="point",
                 coords=None, mask=None, linestyle="-"):
        self.spectrum = np.array(spectrum, dtype=float)
        self.std = np.array(std, dtype=float) if std is not None else None
        self.label = label
        self.color = color
        self.selection_type = selection_type
        self.coords = coords
        self.mask = mask
        self.visible = True
        self.linestyle = linestyle


class SpectraDisplayWindow(QMainWindow):
    """Separate window to display reference / transformed spectra library."""
    def __init__(self, title, wavelengths, spectra, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 600)
        self.wavelengths = wavelengths
        self.spectra = spectra
        self.labels = labels
        self._init_ui()
        self._plot_spectra()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.fig = Figure(figsize=(9, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True, alpha=0.3)
        layout.addWidget(self.canvas)

    def update_data(self, wavelengths, spectra, labels):
        self.wavelengths = wavelengths
        self.spectra = spectra
        self.labels = labels
        self._plot_spectra()

    def _plot_spectra(self):
        self.ax.clear()
        if self.spectra is not None and len(self.spectra) > 0:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.spectra)))
            wl = self.wavelengths if self.wavelengths is not None else np.arange(1, self.spectra.shape[1] + 1, dtype=float)
            for i, (spec, lbl) in enumerate(zip(self.spectra, self.labels)):
                self.ax.plot(wl, spec, color=colors[i % len(colors)], label=lbl, linewidth=1.2)
            self.ax.set_xlabel("Wavelength / Band")
            self.ax.set_ylabel("Intensity")
            self.ax.set_title("Reference Spectra Library")
            self.ax.legend(loc='best', fontsize=8)
            self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()


class UnmixerWindow(QMainWindow):
    """Hyperspectral Unmixer main window inheriting full HyperViewer layout."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperspectral Unmixer v2.1")
        self.setMinimumSize(1600, 1000)
        self.resize(1800, 1100)

        # ---- Data state ----
        self.hypercube = None          # (H, W, L) ndarray
        self.metadata = {}             # metadata dict
        self.current_frame = 0         # Current spectral frame / band index
        self.crosshair_x = None        # X position for Y-spectral side view
        self.crosshair_y = None        # Y position for X-spectral side view

        # ---- Unmixing / Basis state ----
        self.basis_spectra = None      # (N, L) calibration / endmember spectra
        self.basis_labels = None       # list[str]
        self.basis_wavelengths = None  # (L,) wavelengths
        self.concentrations = None     # (H, W, N) unmixing result
        self.r_squared = None          # (H, W) per-pixel R²
        self.residuals = None          # (H, W) per-pixel residual RMS
        self.unmixing_done = False

        # ---- Reference spectra library ----
        self.reference_spectra = None
        self.reference_labels = []
        self.reference_wavelengths = None
        self.reference_window = None

        # ---- Transformed spectra ----
        self.transformed_spectra = None
        self.transformed_labels = []
        self.transformed_wavelengths = None

        # ---- Selection state ----
        self.selection_tool = None     # 'point' | 'circle' | 'square' | None
        self.selection_active = False
        self.selection_start = None
        self.current_selection = None
        self.spectra_list = []         # list of ROI SpectrumData objects
        self.basis_spectra_list = []   # list of Basis / Reference SpectrumData objects
        self.background_spectrum = None
        self.bg_radio_group = QButtonGroup(self)
        self._temp_rect = None
        self._temp_circle = None
        self.current_patch = None

        # ---- Endmember picker state ----
        self.endmember_mode = False
        self.picked_endmembers = []
        self.picked_endmember_patches = []

        # ---- RGB composite state ----
        self.rgb_mode = False
        self.rgb_image = None

        # ---- Display settings ----
        self.current_colormap = 'gray'
        self.selection_size = 10
        self.axes_locked = True
        self.zoom_mode = False
        self.saved_xlim = None
        self.saved_ylim = None

        self._build_ui()
        self._build_menu()
        self.installEventFilter(self)
        self.statusBar().showMessage("Ready")

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        """Build the HyperViewer layout with 2D spatial view, XZ/YZ side views, frame slider, and large spectrum panel."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # ==================== Left Panel: Spatial View & Frame Slider ====================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)

        # Selection Toolbar
        tool_group = QGroupBox("Selection Tools")
        tool_layout = QHBoxLayout(tool_group)
        tool_layout.setContentsMargins(4, 4, 4, 4)

        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.setExclusive(False)

        self.tool_point = QPushButton("• Point")
        self.tool_point.setCheckable(True)
        self.tool_point.setChecked(True)
        self.selection_tool = 'point'
        self.tool_point.clicked.connect(lambda: self._set_selection_tool('point'))
        self.tool_button_group.addButton(self.tool_point)
        tool_layout.addWidget(self.tool_point)

        self.tool_circle = QPushButton("◯ Circle")
        self.tool_circle.setCheckable(True)
        self.tool_circle.clicked.connect(lambda: self._set_selection_tool('circle'))
        self.tool_button_group.addButton(self.tool_circle)
        tool_layout.addWidget(self.tool_circle)

        self.tool_square = QPushButton("□ Square")
        self.tool_square.setCheckable(True)
        self.tool_square.clicked.connect(lambda: self._set_selection_tool('square'))
        self.tool_button_group.addButton(self.tool_square)
        tool_layout.addWidget(self.tool_square)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_selection)
        tool_layout.addWidget(self.clear_btn)

        tool_layout.addSpacing(10)
        tool_layout.addWidget(QLabel("X:"))
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 99999)
        self.roi_x_spin.setValue(0)
        self.roi_x_spin.setToolTip("Fine-tune ROI X position (or use Arrow Keys)")
        self.roi_x_spin.valueChanged.connect(self._on_roi_spinbox_changed)
        tool_layout.addWidget(self.roi_x_spin)

        tool_layout.addWidget(QLabel("Y:"))
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 99999)
        self.roi_y_spin.setValue(0)
        self.roi_y_spin.setToolTip("Fine-tune ROI Y position (or use Arrow Keys)")
        self.roi_y_spin.valueChanged.connect(self._on_roi_spinbox_changed)
        tool_layout.addWidget(self.roi_y_spin)

        tool_layout.addWidget(QLabel("Size:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 200)
        self.size_spin.setValue(10)
        self.size_spin.setSuffix(" px")
        self.size_spin.valueChanged.connect(self._on_roi_spinbox_changed)
        tool_layout.addWidget(self.size_spin)

        tool_layout.addSpacing(15)
        # Single compact Resample dropdown button (1/2, 1/3, 1/4, 2x, 3x, 4x)
        self.resample_combo = QComboBox()
        self.resample_combo.setToolTip("Resample spatial resolution (Downscale: 1/2, 1/3, 1/4 | Upscale: 2x, 3x, 4x)")
        self.resample_combo.addItems([
            "Resample ▾",
            "1/2",
            "1/3",
            "1/4",
            "2x",
            "3x",
            "4x"
        ])
        self.resample_combo.activated[str].connect(self.apply_spatial_resample)
        tool_layout.addWidget(self.resample_combo)

        tool_layout.addSpacing(10)
        self.reset_data_btn = QPushButton("Reset Data")
        self.reset_data_btn.setStyleSheet("font-weight: bold; color: #D32F2F;")
        self.reset_data_btn.setToolTip("Restore original raw intensity hypercube (reverses spatial resampling, background subtraction, and denoising)")
        self.reset_data_btn.clicked.connect(self.restart_analysis)
        tool_layout.addWidget(self.reset_data_btn)

        tool_layout.addStretch()
        left_layout.addWidget(tool_group)

        # Spatial View Figure with XZ and YZ side views
        image_group = QGroupBox("Spatial View")
        image_layout = QVBoxLayout(image_group)
        image_layout.setContentsMargins(4, 4, 4, 4)

        self.image_fig = Figure(figsize=(9, 9), dpi=100)
        self.image_canvas = FigureCanvas(self.image_fig)
        self.image_canvas.mpl_connect('button_press_event', self._on_canvas_press)
        self.image_canvas.mpl_connect('button_release_event', self._on_canvas_release)
        self.image_canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        self.ax_main = self.image_fig.add_subplot(111)
        divider = make_axes_locatable(self.ax_main)
        self.ax_x_spectral = divider.append_axes("top", size="18%", pad=0.1, sharex=self.ax_main)
        self.ax_y_spectral = divider.append_axes("left", size="18%", pad=0.1, sharey=self.ax_main)
        self.ax_cbar = divider.append_axes("right", size="3%", pad=0.1)

        self.ax_x_spectral.tick_params(bottom=False, labelbottom=False)
        self.ax_y_spectral.tick_params(left=False, labelleft=False)
        self.ax_cbar.set_xticks([])
        self.ax_cbar.set_yticks([])

        image_layout.addWidget(self.image_canvas)

        # Built-in Matplotlib Zoom & Pan navigation toolbar for spatial view
        self.nav_toolbar = NavigationToolbar(self.image_canvas, self)
        clean_navigation_toolbar(self.nav_toolbar)
        image_layout.addWidget(self.nav_toolbar)

        # Bottom Color & Scale Controls bar directly under spatial image canvas
        color_bar_layout = QHBoxLayout()
        color_bar_layout.setContentsMargins(4, 2, 4, 2)
        color_bar_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            'gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
            'bone', 'copper', 'red', 'green', 'blue'
        ])
        self.colormap_combo.setCurrentText('gray')
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        color_bar_layout.addWidget(self.colormap_combo)

        color_bar_layout.addSpacing(15)
        color_bar_layout.addWidget(QLabel("C-Min:"))
        self.vmin_spinbox = QDoubleSpinBox()
        self.vmin_spinbox.setRange(-1e9, 1e9)
        self.vmin_spinbox.setDecimals(2)
        self.vmin_spinbox.setFixedWidth(75)
        self.vmin_spinbox.valueChanged.connect(self._on_clim_changed)
        color_bar_layout.addWidget(self.vmin_spinbox)

        color_bar_layout.addWidget(QLabel("C-Max:"))
        self.vmax_spinbox = QDoubleSpinBox()
        self.vmax_spinbox.setRange(-1e9, 1e9)
        self.vmax_spinbox.setDecimals(2)
        self.vmax_spinbox.setFixedWidth(75)
        self.vmax_spinbox.valueChanged.connect(self._on_clim_changed)
        color_bar_layout.addWidget(self.vmax_spinbox)

        self.auto_clim_btn = QPushButton("Auto Scale")
        self.auto_clim_btn.setToolTip("Reset color scale to auto min/max of current frame")
        self.auto_clim_btn.clicked.connect(self._reset_clim_auto)
        color_bar_layout.addWidget(self.auto_clim_btn)

        color_bar_layout.addStretch()
        image_layout.addLayout(color_bar_layout)
        left_layout.addWidget(image_group)

        # Spectral Frame / Focus Z Slider Group
        slider_group = QGroupBox("Spectral Frame / Focus Z")
        slider_vlayout = QVBoxLayout(slider_group)
        slider_vlayout.setContentsMargins(6, 4, 6, 4)

        slider_hlayout = QHBoxLayout()

        self.btn_prev_frame = QPushButton("◀")
        self.btn_prev_frame.setToolTip("Previous Frame (Left Arrow)")
        self.btn_prev_frame.setFixedWidth(35)
        self.btn_prev_frame.clicked.connect(self.prev_frame)
        slider_hlayout.addWidget(self.btn_prev_frame)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(1)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        slider_hlayout.addWidget(self.frame_slider)

        self.btn_next_frame = QPushButton("▶")
        self.btn_next_frame.setToolTip("Next Frame (Right Arrow)")
        self.btn_next_frame.setFixedWidth(35)
        self.btn_next_frame.clicked.connect(self.next_frame)
        slider_hlayout.addWidget(self.btn_next_frame)

        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setRange(0, 0)
        self.frame_spinbox.setFixedWidth(65)
        self.frame_spinbox.setToolTip("Type exact frame / Z index")
        self.frame_spinbox.valueChanged.connect(self.on_frame_spinbox_changed)
        slider_hlayout.addWidget(self.frame_spinbox)

        # Pixel Size Spatial Calibration Boxes (X, Y, Z in µm)
        slider_hlayout.addSpacing(12)
        lbl_cal = QLabel("<b>Pixel Size (µm):</b>")
        lbl_cal.setToolTip("Spatial calibration pixel sizes in micrometers (µm)")
        slider_hlayout.addWidget(lbl_cal)

        slider_hlayout.addWidget(QLabel("X:"))
        self.pixel_size_x_spin = QDoubleSpinBox()
        self.pixel_size_x_spin.setRange(0.0001, 10000.0)
        self.pixel_size_x_spin.setValue(1.0)
        self.pixel_size_x_spin.setSingleStep(0.1)
        self.pixel_size_x_spin.setDecimals(3)
        self.pixel_size_x_spin.setFixedWidth(65)
        self.pixel_size_x_spin.setToolTip("X pixel size in micrometers (µm/px)")
        slider_hlayout.addWidget(self.pixel_size_x_spin)

        slider_hlayout.addWidget(QLabel("Y:"))
        self.pixel_size_y_spin = QDoubleSpinBox()
        self.pixel_size_y_spin.setRange(0.0001, 10000.0)
        self.pixel_size_y_spin.setValue(1.0)
        self.pixel_size_y_spin.setSingleStep(0.1)
        self.pixel_size_y_spin.setDecimals(3)
        self.pixel_size_y_spin.setFixedWidth(65)
        self.pixel_size_y_spin.setToolTip("Y pixel size in micrometers (µm/px)")
        slider_hlayout.addWidget(self.pixel_size_y_spin)

        slider_hlayout.addWidget(QLabel("Z:"))
        self.pixel_size_z_spin = QDoubleSpinBox()
        self.pixel_size_z_spin.setRange(0.0001, 10000.0)
        self.pixel_size_z_spin.setValue(1.0)
        self.pixel_size_z_spin.setSingleStep(0.1)
        self.pixel_size_z_spin.setDecimals(3)
        self.pixel_size_z_spin.setFixedWidth(65)
        self.pixel_size_z_spin.setToolTip("Z step size in micrometers (µm/frame)")
        slider_hlayout.addWidget(self.pixel_size_z_spin)

        slider_vlayout.addLayout(slider_hlayout)

        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        slider_vlayout.addWidget(self.frame_label)

        left_layout.addWidget(slider_group)
        main_layout.addWidget(left_panel, stretch=3)

        # ==================== Right Panel: Plot Windows & ROI Manager ====================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)

        # Tabbed Plot Container (Tab 1: Extracted Spectra | Tab 2: Analysis Results & Fitting)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setStyleSheet("QTabBar::tab { font-weight: bold; padding: 6px 12px; }")

        # --- Tab 1: Extracted Spectra ---
        tab_spec = QWidget()
        layout_spec = QVBoxLayout(tab_spec)
        layout_spec.setContentsMargins(2, 2, 2, 2)

        self.spectrum_fig = Figure(figsize=(7, 6), dpi=100)
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_canvas.mpl_connect('scroll_event', self.on_spectrum_scroll)
        self.spectrum_nav_toolbar = NavigationToolbar(self.spectrum_canvas, self)
        clean_navigation_toolbar(self.spectrum_nav_toolbar, self.autoscale_y_axis)
        layout_spec.addWidget(self.spectrum_nav_toolbar)

        self.ax_spectrum = self.spectrum_fig.add_subplot(111)
        self.ax_spectrum.set_xlabel('Spectral Band / Wavelength')
        self.ax_spectrum.set_ylabel('Intensity')
        self.ax_spectrum.set_title('Extracted Spectra (ROIs & Points)')
        self.ax_spectrum.grid(True, alpha=0.3)
        layout_spec.addWidget(self.spectrum_canvas)
        self.plot_tabs.addTab(tab_spec, "📈 Extracted Spectra")

        # --- Tab 2: Analysis Results & Fitting ---
        tab_analysis = QWidget()
        layout_analysis = QVBoxLayout(tab_analysis)
        layout_analysis.setContentsMargins(2, 2, 2, 2)

        self.analysis_fig = Figure(figsize=(7, 6), dpi=100)
        self.analysis_canvas = FigureCanvas(self.analysis_fig)
        self.analysis_nav_toolbar = NavigationToolbar(self.analysis_canvas, self)
        clean_navigation_toolbar(self.analysis_nav_toolbar, self.autoscale_y_axis)
        layout_analysis.addWidget(self.analysis_nav_toolbar)

        self.ax_analysis = self.analysis_fig.add_subplot(111)
        self.ax_analysis.set_xlabel('Spectral Band / Wavelength')
        self.ax_analysis.set_ylabel('Component Amplitude / Fitting Value')
        self.ax_analysis.set_title('Analysis Results & Fitting Error')
        self.ax_analysis.grid(True, alpha=0.3)
        layout_analysis.addWidget(self.analysis_canvas)
        self.plot_tabs.addTab(tab_analysis, "📊 Analysis Results & Fitting")
        self.plot_tabs.currentChanged.connect(self._on_plot_tab_changed)

        right_layout.addWidget(self.plot_tabs, stretch=4)

        # --- Spectrum Management & ROI Manager Panel ---
        control_group = QGroupBox("Spectrum & ROI Manager")
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(6, 6, 6, 6)

        # Header Row: Left Half = ROI Manager Label | Right Half = Export All & Load Ref Buttons
        header_row = QHBoxLayout()
        roi_lbl = QLabel("<b>ROI Manager & Saved Spectra</b>")
        roi_lbl.setStyleSheet("font-size: 12px; color: #1565C0;")
        header_row.addWidget(roi_lbl)

        self.bg_status_label = QLabel("Background: None")
        self.bg_status_label.setStyleSheet("font-size: 11px; color: #757575;")
        header_row.addWidget(self.bg_status_label)

        header_row.addStretch()

        self.export_btn = QPushButton("Export All Spectra")
        self.export_btn.setToolTip("Export all saved spectra to CSV file")
        self.export_btn.clicked.connect(self._export_all_spectra)
        header_row.addWidget(self.export_btn)

        self.load_ref_btn = QPushButton("Load Reference Spectra")
        self.load_ref_btn.setToolTip("Load reference dye/fluorophore spectra from CSV, XLSX, MAT file")
        self.load_ref_btn.clicked.connect(self.load_reference_spectra)
        header_row.addWidget(self.load_ref_btn)

        control_layout.addLayout(header_row)

        # 3-Column Dual List Layout: Left = ROI Spectra | Center = Actions & ➡️ Transfer | Right = Basis Spectra
        side_by_side_layout = QHBoxLayout()

        # Column 1 (Left): ROI Spectra List Box (Extracted from Canvas Image)
        roi_container = QVBoxLayout()
        roi_container.setSpacing(1)
        roi_header = QHBoxLayout()
        roi_header.setContentsMargins(4, 0, 4, 0)
        roi_header.addWidget(QLabel("<b>✓</b>"), 0)
        roi_header.addWidget(QLabel("<b>ROI Spectra</b>"), 1)
        roi_container.addLayout(roi_header)

        self.spectrum_scroll = QScrollArea()
        self.spectrum_scroll.setWidgetResizable(True)
        self.spectrum_scroll.setMinimumHeight(180)
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_widget)
        self.checkbox_layout.setAlignment(Qt.AlignTop)
        self.checkbox_layout.setContentsMargins(2, 2, 2, 2)
        self.spectrum_scroll.setWidget(self.checkbox_widget)
        roi_container.addWidget(self.spectrum_scroll)

        side_by_side_layout.addLayout(roi_container, stretch=2)

        # Column 2 (Center): Action & Transfer Buttons Stack
        btn_vbox = QVBoxLayout()
        btn_vbox.setContentsMargins(4, 0, 4, 0)

        self.add_btn = QPushButton("Add Selection")
        self.add_btn.setToolTip("Add selected canvas point/region spectrum to ROI list")
        self.add_btn.clicked.connect(self.add_current_spectrum)
        btn_vbox.addWidget(self.add_btn)

        self.copy_to_basis_btn = QPushButton("➡️ Add to Basis")
        self.copy_to_basis_btn.setStyleSheet("font-weight: bold; color: #1565C0; font-size: 12px;")
        self.copy_to_basis_btn.setToolTip("Copy checked ROI spectra to Basis fitting library")
        self.copy_to_basis_btn.clicked.connect(self.copy_roi_to_basis)
        btn_vbox.addWidget(self.copy_to_basis_btn)

        self.fit_roi_btn = QPushButton("Fit ROI to Basis")
        self.fit_roi_btn.setStyleSheet("font-weight: bold; color: #2E7D32; font-size: 12px;")
        self.fit_roi_btn.setToolTip("Fit selected ROI spectrum against Basis spectra using NNLS")
        self.fit_roi_btn.clicked.connect(self.fit_selected_roi_to_basis)
        btn_vbox.addWidget(self.fit_roi_btn)

        self.set_bg_btn = QPushButton("Set as Background")
        self.set_bg_btn.clicked.connect(self.set_background_spectrum)
        btn_vbox.addWidget(self.set_bg_btn)

        self.subtract_bg_btn = QPushButton("Subtract Background")
        self.subtract_bg_btn.setCheckable(True)
        self.subtract_bg_btn.clicked.connect(self.toggle_background_subtraction)
        btn_vbox.addWidget(self.subtract_bg_btn)

        norm_lbl = QLabel("Normalization:")
        norm_lbl.setStyleSheet("font-size: 11px; font-weight: bold; margin-top: 4px;")
        btn_vbox.addWidget(norm_lbl)

        self.norm_mode_combo = QComboBox()
        self.norm_mode_combo.addItem("Original (Raw)", 'raw')
        self.norm_mode_combo.addItem("Total Intensity (spec / sum)", 'total')
        self.norm_mode_combo.addItem("Peak Intensity (spec / max)", 'peak')
        self.norm_mode_combo.addItem("L2 Vector Norm (spec / ||spec||)", 'l2')
        self.norm_mode_combo.addItem("SNV Z-Score ((spec - μ) / σ)", 'snv')
        self.norm_mode_combo.setToolTip("Select spectral normalization & chemometric standardisation mode")
        self.norm_mode_combo.currentIndexChanged.connect(self.on_norm_mode_changed)
        btn_vbox.addWidget(self.norm_mode_combo)

        side_by_side_layout.addLayout(btn_vbox, stretch=1)

        # Column 3 (Right): Basis Spectra List Box (Used for Fitting & Unmixing)
        basis_container = QVBoxLayout()
        basis_container.setSpacing(1)
        basis_header = QHBoxLayout()
        basis_header.setContentsMargins(4, 0, 4, 0)
        basis_header.addWidget(QLabel("<b>✓</b>"), 0)
        basis_header.addWidget(QLabel("<b>Basis Spectra</b>"), 1)
        basis_container.addLayout(basis_header)

        self.basis_scroll = QScrollArea()
        self.basis_scroll.setWidgetResizable(True)
        self.basis_scroll.setMinimumHeight(180)
        self.basis_widget = QWidget()
        self.basis_layout = QVBoxLayout(self.basis_widget)
        self.basis_layout.setAlignment(Qt.AlignTop)
        self.basis_layout.setContentsMargins(2, 2, 2, 2)
        self.basis_scroll.setWidget(self.basis_widget)
        basis_container.addWidget(self.basis_scroll)

        side_by_side_layout.addLayout(basis_container, stretch=2)

        control_layout.addLayout(side_by_side_layout)

        # Unmixing controls & Progress bar
        unmix_control = QHBoxLayout()
        self.component_combo = QComboBox()
        self.component_combo.addItem("Total Intensity", 'total')
        self.component_combo.currentTextChanged.connect(self._on_component_changed)
        unmix_control.addWidget(QLabel("Display Component:"))
        unmix_control.addWidget(self.component_combo, stretch=1)
        control_layout.addLayout(unmix_control)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)

        self.unmix_status = QLabel("")
        self.unmix_status.setWordWrap(True)
        control_layout.addWidget(self.unmix_status)

        right_layout.addWidget(control_group, stretch=2)
        main_layout.addWidget(right_panel, stretch=2)

    # -----------------------------------------------------------------------
    # Menu Construction
    # -----------------------------------------------------------------------

    def _build_menu(self):
        """Build full menu bar with File, Analysis, and Help submenus."""
        menubar = self.menuBar()

        # ---- File menu ----
        file_menu = menubar.addMenu("File")

        open_envi_action = QAction("Open ENVI (.hdr/.dat)", self)
        open_envi_action.setShortcut("Ctrl+O")
        open_envi_action.setToolTip("Load ENVI format hyperspectral image")
        open_envi_action.triggered.connect(self.open_envi)
        file_menu.addAction(open_envi_action)

        open_npy_action = QAction("Open NPY", self)
        open_npy_action.setShortcut("Ctrl+Shift+O")
        open_npy_action.setToolTip("Load .npy hyperspectral cube")
        open_npy_action.triggered.connect(self.open_npy)
        file_menu.addAction(open_npy_action)

        open_tiff_action = QAction("Open TIFF Stack (.tif/.tiff)", self)
        open_tiff_action.setShortcut("Ctrl+T")
        open_tiff_action.setToolTip("Load multipage TIFF stack hyperspectral image")
        open_tiff_action.triggered.connect(self.open_tiff)
        file_menu.addAction(open_tiff_action)

        file_menu.addSeparator()

        self.export_action = QAction("Export Results", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.setToolTip("Export unmixing results to CSV")
        self.export_action.triggered.connect(self.export_results)
        self.export_action.setEnabled(False)
        file_menu.addAction(self.export_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ---- Analysis menu ----
        analysis_menu = menubar.addMenu("Analysis")

        # Denoising
        denoise_menu = analysis_menu.addMenu("Denoise")
        savgol_action = QAction("Savitzky-Golay Spectral Filter", self)
        savgol_action.setToolTip("Polynomial spectral smoothing filter along Z-axis")
        savgol_action.triggered.connect(self.denoise_savgol)
        denoise_menu.addAction(savgol_action)

        tv3d_action = QAction("3D Total Variation (3D-TV)", self)
        tv3d_action.setToolTip("Spatial-spectral Chambolle total variation noise reduction")
        tv3d_action.triggered.connect(self.denoise_tv3d)
        denoise_menu.addAction(tv3d_action)

        bm4d_action = QAction("BM4D / FastHyDe Subspace Filter", self)
        bm4d_action.setToolTip("Non-local 4D patch matching and subspace noise filtering")
        bm4d_action.triggered.connect(self.denoise_bm4d)
        denoise_menu.addAction(bm4d_action)

        mppca_action = QAction("MPPCA (Multiplicative PCA)", self)
        mppca_action.triggered.connect(self.denoise_mppca)
        denoise_menu.addAction(mppca_action)

        wavelet_action = QAction("Wavelet (VisuShrink)", self)
        wavelet_action.triggered.connect(self.denoise_wavelet)
        denoise_menu.addAction(wavelet_action)

        # Unmixing
        unmix_menu = analysis_menu.addMenu("Unmixing")
        nnls_action = QAction("NNLS Linear Unmixing (via ROIs / Ref Spectra)", self)
        nnls_action.setToolTip("Per-pixel Non-Negative Least Squares against saved ROIs or reference endmembers")
        nnls_action.triggered.connect(self.run_linear_unmix)
        unmix_menu.addAction(nnls_action)

        fcls_action = QAction("FCLS (Fully Constrained Least Squares)", self)
        fcls_action.setShortcut("Ctrl+L")
        fcls_action.setToolTip("NNLS with sum-to-one constraint — gold standard for abundance estimation")
        fcls_action.triggered.connect(self.run_fcls)
        unmix_menu.addAction(fcls_action)

        mcr_action = QAction("MCR-ALS", self)
        mcr_action.setShortcut("Ctrl+M")
        mcr_action.triggered.connect(self.run_mcr_als)
        unmix_menu.addAction(mcr_action)

        nmf_action = QAction("NMF (Non-Negative Matrix Factorization)", self)
        nmf_action.setShortcut("Ctrl+N")
        nmf_action.triggered.connect(self.run_nmf)
        unmix_menu.addAction(nmf_action)

        mesma_action = QAction("MESMA (Multiple Endmember Spectral Mixture)", self)
        mesma_action.triggered.connect(self.run_mesma)
        unmix_menu.addAction(mesma_action)

        # Matrix Decomposition / Factor Analysis
        decomp_menu = analysis_menu.addMenu("Decomposition")
        pca_action = QAction("PCA (Principal Component Analysis)", self)
        pca_action.triggered.connect(self.run_pca)
        decomp_menu.addAction(pca_action)

        mnf_action = QAction("MNF (Minimum Noise Fraction)", self)
        mnf_action.setToolTip("Orders components by SNR instead of variance — better than PCA for noisy data")
        mnf_action.triggered.connect(self.run_mnf)
        decomp_menu.addAction(mnf_action)

        ica_action = QAction("ICA (Independent Component Analysis)", self)
        ica_action.triggered.connect(self.run_ica)
        decomp_menu.addAction(ica_action)

        # Classification / Matching
        class_menu = analysis_menu.addMenu("Classification")
        sam_action = QAction("SAM (Spectral Angle Mapper)", self)
        sam_action.triggered.connect(self.run_sam)
        class_menu.addAction(sam_action)

        sid_action = QAction("SID (Spectral Information Divergence)", self)
        sid_action.triggered.connect(self.run_sid)
        class_menu.addAction(sid_action)

        rx_action = QAction("RX Anomaly Detection", self)
        rx_action.setToolTip("Reed-Xiaoli Mahalanobis distance anomaly detection (no reference needed)")
        rx_action.triggered.connect(self.run_rx)
        class_menu.addAction(rx_action)

        analysis_menu.addSeparator()

        # Endmember extraction
        em_menu = analysis_menu.addMenu("Extract Endmembers")
        nfindr_action = QAction("N-FINDR", self)
        nfindr_action.setShortcut("Ctrl+F")
        nfindr_action.triggered.connect(self.run_nfindr_dialog)
        em_menu.addAction(nfindr_action)

        vca_action = QAction("VCA (Vertex Component Analysis)", self)
        vca_action.setShortcut("Ctrl+V")
        vca_action.triggered.connect(self.run_vca_dialog)
        em_menu.addAction(vca_action)

        picker_action = QAction("Interactive Picker", self)
        picker_action.setShortcut("Ctrl+I")
        picker_action.triggered.connect(self.toggle_endmember_picker)
        em_menu.addAction(picker_action)

        analysis_menu.addSeparator()

        # Peak fitting
        fit_menu = analysis_menu.addMenu("Fit Peaks")
        gauss_action = QAction("Gaussian Fit", self)
        gauss_action.setShortcut("Ctrl+G")
        gauss_action.triggered.connect(lambda: self.run_fit_spectrum('gaussian'))
        fit_menu.addAction(gauss_action)

        lorentz_action = QAction("Lorentzian Fit", self)
        lorentz_action.setShortcut("Ctrl+L")
        lorentz_action.triggered.connect(lambda: self.run_fit_spectrum('lorentzian'))
        fit_menu.addAction(lorentz_action)

        analysis_menu.addSeparator()

        # Visualization
        viz_menu = analysis_menu.addMenu("Visualization")
        rgb_action = QAction("RGB Composite", self)
        rgb_action.triggered.connect(self.show_rgb_composite)
        viz_menu.addAction(rgb_action)

        pcrgb_action = QAction("PC-RGB (First 3 PCA Components)", self)
        pcrgb_action.triggered.connect(self.show_pcrgb_composite)
        viz_menu.addAction(pcrgb_action)

        # ---- Help menu ----
        help_menu = menubar.addMenu("Help")

        help_action = QAction("User Guide (README)", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # -----------------------------------------------------------------------
    # Data Loading & Frame Handling
    # -----------------------------------------------------------------------

    def open_envi(self):
        """Load an ENVI hyperspectral image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open ENVI Header", "", "Header Files (*.hdr);;All Files (*)")
        if not file_path:
            return
        try:
            self._load_envi(file_path)
        except Exception as e:
            QMessageBox.critical(self, "ENVI Load Error", f"Failed to load ENVI file:\n{e}")

    def _load_envi(self, hdr_path):
        params = {}
        with open(hdr_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    params[key.strip().lower()] = val.strip().strip('{}')

        samples = int(params['samples'])
        lines = int(params['lines'])
        bands = int(params['bands'])
        data_type_str = params.get('data type', '4')
        data_type_map = {
            '1': np.int16, '2': np.float32, '3': np.int32,
            '4': np.float64, '5': np.complex64, '6': np.complex128,
            '9': np.uint16, '12': np.uint32, '13': np.uint8, '14': np.int8,
        }
        dtype = data_type_map.get(data_type_str, np.float64)
        interleave = params.get('interleave', 'bil')[:3].lower()

        dat_path = hdr_path.replace('.hdr', '.dat')
        if not os.path.exists(dat_path):
            dat_path = hdr_path.replace('.hdr', '.img')
        if not os.path.exists(dat_path):
            dat_path = hdr_path + '.dat'
        if not os.path.exists(dat_path):
            dat_path = hdr_path + '.img'

        wavelengths = None
        if 'wavelength' in params:
            wl_str = params['wavelength'].strip('{}')
            wavelengths = np.array([float(x) for x in wl_str.split(',')])

        data = np.fromfile(dat_path, dtype=dtype)
        if interleave == 'bsq':
            data = data.reshape((bands, lines, samples)).transpose(1, 2, 0)
        elif interleave == 'bil':
            data = data.reshape((lines, bands, samples)).transpose(0, 2, 1)
        elif interleave == 'bip':
            data = data.reshape((lines, samples, bands))

        self.metadata = params
        self.basis_wavelengths = wavelengths
        self.set_data(data)

    def open_npy(self):
        """Load a .npy hyperspectral cube."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open NPY", "", "NPY Files (*.npy)")
        if not file_path:
            return
        try:
            data = np.load(file_path).astype(np.float64)
            self.set_data(data)
        except Exception as e:
            QMessageBox.critical(self, "NPY Load Error", f"Failed to load .npy file:\n{e}")

    def open_tiff(self):
        """Load a multipage TIFF stack hyperspectral image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF Stack", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
        if not file_path:
            return
        try:
            self._load_tiff(file_path)
        except Exception as e:
            QMessageBox.critical(self, "TIFF Load Error", f"Failed to load TIFF stack:\n{e}")

    def _load_tiff(self, file_path):
        try:
            import tifffile
            data = tifffile.imread(file_path)
        except ImportError:
            try:
                from skimage import io
                data = io.imread(file_path)
            except ImportError:
                raise ImportError("Install tifffile for TIFF support: pip install tifffile")

        data = np.array(data, dtype=np.float64)
        if data.ndim == 3:
            data = np.moveaxis(data, 0, -1)
        elif data.ndim == 2:
            data = data[:, :, np.newaxis]
        else:
            raise ValueError(f"Unsupported array shape in TIFF: {data.shape}")

        self.set_data(data)

    def set_data(self, hypercube, metadata=None):
        """Initialize hypercube data and setup slider/crosshairs."""
        self.hypercube = np.asarray(hypercube, dtype=float)
        self.raw_hypercube = self.hypercube.copy()
        if metadata:
            self.metadata = metadata
        height, width, bands = self.hypercube.shape

        self.crosshair_x = width // 2
        self.crosshair_y = height // 2

        self.frame_slider.setRange(0, bands - 1)
        if hasattr(self, 'frame_spinbox'):
            self.frame_spinbox.setRange(0, bands - 1)
            self.frame_spinbox.setValue(0)
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"Frame: 0 / {bands - 1}")

        self._on_data_loaded()

    def restart_analysis(self):
        """Reset hypercube data back to original raw intensity and restore initial state."""
        if hasattr(self, 'raw_hypercube') and self.raw_hypercube is not None:
            self.hypercube = self.raw_hypercube.copy()
            self.background_spectrum = None
            if hasattr(self, 'subtract_bg_btn'):
                self.subtract_bg_btn.setChecked(False)
            if hasattr(self, 'bg_status_label'):
                self.bg_status_label.setText("Background: None")
                self.bg_status_label.setStyleSheet("color: gray;")
            if hasattr(self, 'norm_mode_combo'):
                self.norm_mode_combo.blockSignals(True)
                self.norm_mode_combo.setCurrentIndex(0) # Original (Raw)
                self.norm_mode_combo.blockSignals(False)
            self.display_frame(self.current_frame)
            self.update_spectrum_plot()
            self.statusBar().showMessage("Restarted: Restored hypercube to original raw intensity.")
            QMessageBox.information(self, "Restart Complete", "Restored hypercube to original raw intensity.\nBackground subtraction and denoising filters have been reset.")
        else:
            QMessageBox.warning(self, "No Original Data", "No original hypercube data found to restore.")

    def prev_frame(self):
        """Step backward 1 frame."""
        val = self.frame_slider.value()
        if val > 0:
            self.frame_slider.setValue(val - 1)

    def next_frame(self):
        """Step forward 1 frame."""
        val = self.frame_slider.value()
        if val < self.frame_slider.maximum():
            self.frame_slider.setValue(val + 1)

    def on_frame_spinbox_changed(self, val):
        """Handle direct frame spinbox input."""
        if self.frame_slider.value() != val:
            self.frame_slider.setValue(val)

    def on_frame_changed(self, value):
        """Handle spectral frame slider change."""
        self.current_frame = value
        if hasattr(self, 'frame_spinbox'):
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(value)
            self.frame_spinbox.blockSignals(False)
        if self.hypercube is not None:
            bands = self.hypercube.shape[2]
            self.frame_label.setText(f"Frame: {value} / {bands - 1}")
            self.display_frame(value)

    def display_frame(self, frame_idx):
        """Display a specific spectral frame with XZ and YZ cross-sections sharing colormap & scale."""
        if self.hypercube is None:
            return

        height, width, bands = self.hypercube.shape
        frame_idx = max(0, min(frame_idx, bands - 1))
        self.current_frame = frame_idx

        # Determine spatial matrix to display on main view (XY)
        active_comp = self.component_combo.currentData() if hasattr(self, 'component_combo') else 'total'
        if active_comp == 'total' or active_comp is None or self.concentrations is None:
            disp_image = self.hypercube[:, :, frame_idx]
        elif active_comp == 'r2' and self.r_squared is not None:
            disp_image = self.r_squared
        elif active_comp == 'residual' and self.residuals is not None:
            disp_image = self.residuals
        elif isinstance(active_comp, int) and self.concentrations is not None and active_comp < self.concentrations.shape[2]:
            disp_image = self.concentrations[:, :, active_comp]
        else:
            disp_image = self.hypercube[:, :, frame_idx]

        # Determine color limits (vmin, vmax)
        if hasattr(self, 'clim') and self.clim is not None:
            vmin, vmax = self.clim
        else:
            vmin = float(np.nanmin(disp_image))
            vmax = float(np.nanmax(disp_image))
            if vmin == vmax:
                vmax = vmin + 1.0

        # Update min/max spinboxes without triggering signals
        if hasattr(self, 'vmin_spinbox') and hasattr(self, 'vmax_spinbox'):
            self.vmin_spinbox.blockSignals(True)
            self.vmax_spinbox.blockSignals(True)
            self.vmin_spinbox.setValue(vmin)
            self.vmax_spinbox.setValue(vmax)
            self.vmin_spinbox.blockSignals(False)
            self.vmax_spinbox.blockSignals(False)

        # Main Spatial View (XY)
        self.ax_main.clear()
        im = self.ax_main.imshow(disp_image, cmap=self.current_colormap,
                                vmin=vmin, vmax=vmax,
                                interpolation='nearest', extent=[-0.5, width - 0.5, height - 0.5, -0.5])
        self.ax_main.set_xlim(-0.5, width - 0.5)
        self.ax_main.set_ylim(height - 0.5, -0.5)
        self.im_handle = im

        # Draw crosshairs only for point or default navigation
        cy = max(0, min(self.crosshair_y or height // 2, height - 1))
        cx = max(0, min(self.crosshair_x or width // 2, width - 1))
        if self.selection_tool in (None, 'point'):
            self.ax_main.axhline(cy, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            self.ax_main.axvline(cx, color='red', linestyle='--', linewidth=0.8, alpha=0.7)

        # Colorbar
        self.ax_cbar.clear()
        cbar = self.image_fig.colorbar(im, cax=self.ax_cbar, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)

        # Update XZ Spectral View (Top): row at cy -> shape (width, bands)
        self.ax_x_spectral.clear()
        x_spectral = self.hypercube[cy, :, :]
        self.ax_x_spectral.imshow(x_spectral.T, cmap=self.current_colormap,
                                   vmin=vmin, vmax=vmax, aspect='auto',
                                   extent=[-0.5, width - 0.5, bands - 0.5, -0.5])
        self.ax_x_spectral.set_xlim(-0.5, width - 0.5)
        self.ax_x_spectral.set_ylim(bands - 0.5, -0.5)
        self.ax_x_spectral.tick_params(bottom=False, labelbottom=False)

        # Update YZ Spectral View (Left): col at cx -> shape (height, bands)
        self.ax_y_spectral.clear()
        y_spectral = self.hypercube[:, cx, :]
        self.ax_y_spectral.imshow(y_spectral, cmap=self.current_colormap,
                                   vmin=vmin, vmax=vmax, aspect='auto',
                                   extent=[-0.5, bands - 0.5, height - 0.5, -0.5])
        self.ax_y_spectral.set_xlim(-0.5, bands - 0.5)
        self.ax_y_spectral.set_ylim(height - 0.5, -0.5)
        self.ax_y_spectral.tick_params(left=False, labelleft=False)

        self._redraw_selections()
        self.image_canvas.draw_idle()

    def _on_data_loaded(self):
        """Callback when new hypercube data is loaded."""
        H, W, L = self.hypercube.shape
        self.unmixing_done = False
        self.concentrations = None
        self.r_squared = None
        self.residuals = None
        self.saved_xlim = None
        self.saved_ylim = None
        if hasattr(self, 'export_action'):
            self.export_action.setEnabled(False)

        if hasattr(self, 'roi_x_spin') and hasattr(self, 'roi_y_spin'):
            self.roi_x_spin.blockSignals(True)
            self.roi_y_spin.blockSignals(True)
            self.roi_x_spin.setRange(0, W - 1)
            self.roi_y_spin.setRange(0, H - 1)
            self.roi_x_spin.setValue(self.crosshair_x)
            self.roi_y_spin.setValue(self.crosshair_y)
            self.roi_x_spin.blockSignals(False)
            self.roi_y_spin.blockSignals(False)

        self.display_frame(self.current_frame)
        self._update_roi_from_position(self.crosshair_x, self.crosshair_y)
        self.autoscale_axes()
        self.statusBar().showMessage(f"Loaded Hypercube: {H}x{W}x{L}")

    # -----------------------------------------------------------------------
    # Spatial View Interactions & Selections
    # -----------------------------------------------------------------------

    def _set_selection_tool(self, tool):
        self.selection_tool = tool
        self.tool_point.setChecked(tool == 'point')
        self.tool_circle.setChecked(tool == 'circle')
        self.tool_square.setChecked(tool == 'square')
        self.statusBar().showMessage(f"Selection Tool: {tool.title()}")

    def _on_colormap_changed(self, cmap_name):
        self.current_colormap = cmap_name
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

    def _on_clim_changed(self):
        """Handle manual color limit (Vmin, Vmax) spinbox changes."""
        vmin = self.vmin_spinbox.value()
        vmax = self.vmax_spinbox.value()
        if vmin >= vmax:
            return
        self.clim = (vmin, vmax)
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

    def _reset_clim_auto(self):
        """Reset color scale limits to auto frame min/max."""
        self.clim = None
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

    def apply_spatial_resample(self, factor_str=None):
        """Downscale (1/2, 1/3, 1/4 via block averaging) or Upscale (2x, 3x, 4x via bilinear interpolation)."""
        if self.hypercube is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            if hasattr(self, 'resample_combo'):
                self.resample_combo.setCurrentIndex(0)
            return

        if not factor_str or factor_str == "Resample ▾":
            factor_str = self.resample_combo.currentText()

        if factor_str == "Resample ▾":
            return

        # Reset combo box index back to 'Resample ▾' title without infinite signals
        self.resample_combo.blockSignals(True)
        self.resample_combo.setCurrentIndex(0)
        self.resample_combo.blockSignals(False)

        H, W, B = self.hypercube.shape

        if "1/2" in factor_str:
            k = 2
            mode = "downscale"
        elif "1/3" in factor_str:
            k = 3
            mode = "downscale"
        elif "1/4" in factor_str:
            k = 4
            mode = "downscale"
        elif "2x" in factor_str:
            k = 2
            mode = "upscale"
        elif "3x" in factor_str:
            k = 3
            mode = "upscale"
        elif "4x" in factor_str:
            k = 4
            mode = "upscale"
        else:
            return

        try:
            if mode == "downscale":
                H_new = H // k
                W_new = W // k
                if H_new < 2 or W_new < 2:
                    QMessageBox.warning(self, "Warning", f"Image dimensions ({H}x{W}) are too small to downscale by 1/{k}.")
                    return
                cube_cropped = self.hypercube[:H_new * k, :W_new * k, :]
                new_cube = cube_cropped.reshape(H_new, k, W_new, k, B).mean(axis=(1, 3))
                msg = f"Downscaled image by 1/{k} ({H}x{W} -> {H_new}x{W_new})"

            else:  # upscale
                try:
                    from scipy.ndimage import zoom
                    new_cube = zoom(self.hypercube, (k, k, 1), order=1)
                except ImportError:
                    new_cube = np.repeat(np.repeat(self.hypercube, k, axis=0), k, axis=1)
                H_new, W_new = new_cube.shape[:2]
                msg = f"Upscaled image by {k}x ({H}x{W} -> {H_new}x{W_new})"

            self.hypercube = new_cube.astype(np.float32)
            self.clear_selection()
            self.spectra_list = []
            self.update_checkbox_list()
            self._on_data_loaded()
            self.statusBar().showMessage(msg)
            QMessageBox.information(self, "Spatial Resample", msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to resample image:\n{e}")

    def _on_canvas_press(self, event):
        if event.inaxes != self.ax_main or self.hypercube is None:
            return

        x, y = int(round(event.xdata)), int(round(event.ydata))
        H, W, L = self.hypercube.shape

        if 0 <= x < W and 0 <= y < H:
            self.crosshair_x = x
            self.crosshair_y = y

        if event.button == 1 and self.selection_tool == 'point':
            if 0 <= x < W and 0 <= y < H:
                spectrum = self.hypercube[y, x, :]
                mask = np.zeros((H, W), dtype=bool)
                mask[y, x] = True
                self.current_selection = {
                    'type': 'point', 'x': x, 'y': y, 'coords': (float(x), float(y)),
                    'spectrum': spectrum, 'mask': mask,
                }
                self._plot_spectrum(spectrum)
                self.display_frame(self.current_frame)
                self.add_btn.setEnabled(True)

        elif event.button == 1 and self.selection_tool in ('circle', 'square'):
            self.selection_active = True
            self.selection_start = (event.xdata, event.ydata)

    def _on_canvas_motion(self, event):
        if not self.selection_active or event.inaxes != self.ax_main or self.selection_start is None:
            return
        x0, y0 = self.selection_start
        x1, y1 = event.xdata, event.ydata

        # Clear previous temporary patch
        if hasattr(self, 'current_patch') and self.current_patch:
            try:
                self.current_patch.remove()
            except Exception:
                pass
            self.current_patch = None

        if self.selection_tool == 'square':
            size = max(abs(x1 - x0), abs(y1 - y0))
            half_size = size / 2.0
            self.current_patch = Rectangle((x0 - half_size, y0 - half_size), size, size,
                                          fill=False, color='blue', linewidth=2, linestyle='--')
            self.ax_main.add_patch(self.current_patch)
        elif self.selection_tool == 'circle':
            r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            self.current_patch = Circle((x0, y0), r, fill=False, color='green', linewidth=2, linestyle='--')
            self.ax_main.add_patch(self.current_patch)
        self.image_canvas.draw_idle()

    def _on_canvas_release(self, event):
        if not self.selection_active:
            return
        self.selection_active = False

        if self.selection_start is None or self.hypercube is None:
            return
        x0, y0 = self.selection_start
        x1, y1 = event.xdata, event.ydata
        H, W, L = self.hypercube.shape
        mask = np.zeros((H, W), dtype=bool)

        radius_val = None
        size_val = None

        if self.selection_tool == 'square':
            size_val = float(max(1.0, max(abs(x1 - x0), abs(y1 - y0))))
            half_size = size_val / 2.0
            x_min, x_max = max(0, int(round(x0 - half_size))), min(W - 1, int(round(x0 + half_size)))
            y_min, y_max = max(0, int(round(y0 - half_size))), min(H - 1, int(round(y0 + half_size)))
            mask[y_min:y_max+1, x_min:x_max+1] = True
        elif self.selection_tool == 'circle':
            radius_val = float(max(1.0, np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)))
            yy, xx = np.ogrid[:H, :W]
            mask[(xx - x0) ** 2 + (yy - y0) ** 2 <= radius_val ** 2] = True

        if mask.any():
            spectra = self.hypercube[mask]
            mean_spectrum = spectra.mean(axis=0)
            std = spectra.std(axis=0)
            self.current_selection = {
                'type': self.selection_tool,
                'coords': (float(x0), float(y0)),
                'x': int(round(x0)),
                'y': int(round(y0)),
                'radius': radius_val,
                'size': size_val,
                'spectrum': mean_spectrum,
                'std': std,
                'mask': mask,
                'n_pixels': int(mask.sum()),
            }
            self.update_spectrum_plot()
            self.add_btn.setEnabled(True)

            # Update X/Y spinboxes
            if hasattr(self, 'roi_x_spin') and hasattr(self, 'roi_y_spin'):
                self.roi_x_spin.blockSignals(True)
                self.roi_y_spin.blockSignals(True)
                self.roi_x_spin.setValue(int(round(x0)))
                self.roi_y_spin.setValue(int(round(y0)))
                self.roi_x_spin.blockSignals(False)
                self.roi_y_spin.blockSignals(False)

        if hasattr(self, 'current_patch') and self.current_patch:
            try:
                self.current_patch.remove()
            except Exception:
                pass
            self.current_patch = None
        self.display_frame(self.current_frame)

    def eventFilter(self, obj, event):
        """Global Event Filter: Intercept Arrow Keys and Enter globally for ROI position nudging."""
        if event.type() == QEvent.KeyPress and self.hypercube is not None:
            key = event.key()
            if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down, Qt.Key_Return, Qt.Key_Enter):
                focus_widget = QApplication.focusWidget()
                if isinstance(focus_widget, (QLineEdit, QTextEdit)):
                    return super().eventFilter(obj, event)

                self.keyPressEvent(event)
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """Handle Arrow Keys to nudge ROI position pixel-by-pixel (Shift+Arrow for 5px jumps)."""
        if self.hypercube is None:
            super().keyPressEvent(event)
            return

        key = event.key()
        if key not in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down, Qt.Key_Return, Qt.Key_Enter):
            super().keyPressEvent(event)
            return

        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.add_current_spectrum()
            return

        step = 5 if (event.modifiers() & Qt.ShiftModifier) else 1
        dx = -step if key == Qt.Key_Left else (step if key == Qt.Key_Right else 0)
        dy = -step if key == Qt.Key_Up else (step if key == Qt.Key_Down else 0)

        H, W, L = self.hypercube.shape
        new_x = max(0, min(W - 1, self.crosshair_x + dx))
        new_y = max(0, min(H - 1, self.crosshair_y + dy))

        self.crosshair_x = new_x
        self.crosshair_y = new_y

        self.roi_x_spin.blockSignals(True)
        self.roi_y_spin.blockSignals(True)
        self.roi_x_spin.setValue(new_x)
        self.roi_y_spin.setValue(new_y)
        self.roi_x_spin.blockSignals(False)
        self.roi_y_spin.blockSignals(False)

        self._update_roi_from_position(new_x, new_y)

    def _on_roi_spinbox_changed(self):
        """Callback when X, Y, or Size spinboxes change manually."""
        if self.hypercube is None:
            return
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        self.crosshair_x = x
        self.crosshair_y = y
        self._update_roi_from_position(x, y)

    def _update_roi_from_position(self, x, y):
        """Re-extract ROI spectrum at (x, y) and redraw visual selection."""
        if self.hypercube is None:
            return

        H, W, L = self.hypercube.shape
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))

        stype = self.selection_tool or 'point'
        mask = np.zeros((H, W), dtype=bool)

        if stype == 'point':
            mask[y, x] = True
            spectrum = self.hypercube[y, x, :]
            self.current_selection = {
                'type': 'point', 'x': x, 'y': y, 'coords': (float(x), float(y)),
                'spectrum': spectrum, 'mask': mask,
            }
        elif stype in ('circle', 'square'):
            size_px = self.size_spin.value()
            r = size_px / 2.0
            if stype == 'circle':
                Y_grid, X_grid = np.ogrid[:H, :W]
                mask[(X_grid - x)**2 + (Y_grid - y)**2 <= r**2] = True
            else: # square
                x0 = max(0, int(round(x - r)))
                x1 = min(W - 1, int(round(x + r)))
                y0 = max(0, int(round(y - r)))
                y1 = min(H - 1, int(round(y + r)))
                mask[y0:y1+1, x0:x1+1] = True

            if not np.any(mask):
                mask[y, x] = True

            spectra = self.hypercube[mask]
            spectrum = spectra.mean(axis=0)
            self.current_selection = {
                'type': stype, 'coords': (float(x), float(y)), 'x': x, 'y': y,
                'radius': r, 'size': size_px,
                'spectrum': spectrum, 'mask': mask, 'n_pixels': int(mask.sum()),
            }

        self._plot_spectrum(spectrum)
        self.display_frame(self.current_frame)
        self.add_btn.setEnabled(True)

    def _align_side_views(self, event=None):
        if not hasattr(self, 'ax_main') or self.ax_main is None or self.ax_x_spectral is None or self.ax_y_spectral is None:
            return
        pos_main = self.ax_main.get_position()
        pos_x = self.ax_x_spectral.get_position()
        pos_y = self.ax_y_spectral.get_position()

        # Align XZ view left edge and width to match main spatial image
        self.ax_x_spectral.set_position([pos_main.x0, pos_x.y0, pos_main.width, pos_x.height])
        # Align YZ view bottom edge and height to match main spatial image
        self.ax_y_spectral.set_position([pos_y.x0, pos_main.y0, pos_y.width, pos_main.height])

    def _redraw_selections(self):
        """Redraw open selection markers (+ for point, O for circle, □ for square) for all saved and active selections."""
        # 1. Draw saved selections from self.spectra_list
        for sdata in self.spectra_list:
            if not sdata.visible or sdata.coords is None:
                continue
            cx, cy = sdata.coords
            c = sdata.color
            if sdata.selection_type == 'point':
                self.ax_main.plot(cx, cy, '+', color=c, markersize=10, markeredgewidth=2)
            elif sdata.selection_type == 'circle' and hasattr(sdata, 'radius') and sdata.radius:
                circle = Circle((cx, cy), sdata.radius, fill=False, edgecolor=c, linewidth=2)
                self.ax_main.add_patch(circle)
            elif sdata.selection_type == 'square' and hasattr(sdata, 'size') and sdata.size:
                half = sdata.size / 2.0
                rect = Rectangle((cx - half, cy - half), sdata.size, sdata.size,
                                 fill=False, edgecolor=c, linewidth=2)
                self.ax_main.add_patch(rect)

        # 2. Draw active current selection
        if self.current_selection and 'coords' in self.current_selection:
            sel = self.current_selection
            stype = sel.get('type', 'point')
            cx, cy = sel['coords']
            if stype == 'point':
                self.ax_main.plot(cx, cy, '+', color='red', markersize=10, markeredgewidth=2)
            elif stype == 'circle' and 'radius' in sel:
                circle = Circle((cx, cy), sel['radius'], fill=False, edgecolor='green', linewidth=2, linestyle='--')
                self.ax_main.add_patch(circle)
            elif stype == 'square' and 'size' in sel:
                half = sel['size'] / 2.0
                rect = Rectangle((cx - half, cy - half), sel['size'], sel['size'],
                                 fill=False, edgecolor='blue', linewidth=2, linestyle='--')
                self.ax_main.add_patch(rect)

    def clear_selection(self):
        self.current_selection = None
        self.selection_tool = None
        self.tool_point.setChecked(False)
        self.tool_circle.setChecked(False)
        self.tool_square.setChecked(False)
        self.add_btn.setEnabled(False)
        if self.hypercube is not None:
            self.display_frame(self.current_frame)
        self.statusBar().showMessage("Selection cleared")

    # -----------------------------------------------------------------------
    # Spectrum Plotting & Management
    # -----------------------------------------------------------------------

    def toggle_lock_axes(self):
        self.axes_locked = self.lock_axes_btn.isChecked()
        self.lock_axes_btn.setText("🔒 Lock" if self.axes_locked else "🔓 Lock")
        if self.axes_locked:
            self.saved_xlim = self.ax_spectrum.get_xlim()
            self.saved_ylim = self.ax_spectrum.get_ylim()

    def autoscale_axes(self):
        """Autoscale spectrum plot axes to fit all visible data (matching HyperViewer)."""
        all_xmin, all_xmax = [], []
        all_ymin, all_ymax = [], []
        for line in self.ax_spectrum.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) > 0:
                all_xmin.append(np.min(xdata))
                all_xmax.append(np.max(xdata))
            if len(ydata) > 0:
                all_ymin.append(np.min(ydata))
                all_ymax.append(np.max(ydata))

        if all_xmin and all_ymax:
            x_min_val, x_max_val = min(all_xmin), max(all_xmax)
            y_min_val, y_max_val = min(all_ymin), max(all_ymax)
            x_margin = (x_max_val - x_min_val) * 0.05 if x_max_val != x_min_val else 1.0
            y_margin = (y_max_val - y_min_val) * 0.05 if y_max_val != y_min_val else 0.1
            self.ax_spectrum.set_xlim(x_min_val - x_margin, x_max_val + x_margin)
            self.ax_spectrum.set_ylim(y_min_val - y_margin, y_max_val + y_margin)

        if self.axes_locked and all_xmin and all_ymax:
            self.saved_xlim = self.ax_spectrum.get_xlim()
            self.saved_ylim = self.ax_spectrum.get_ylim()
        self.spectrum_canvas.draw_idle()

    def toggle_zoom_mode(self):
        self.zoom_mode = self.zoom_btn.isChecked()
        if self.zoom_mode:
            self.zoom_btn.setText("✋ Zoom")
            self.spectrum_canvas.setCursor(Qt.CrossCursor)
        else:
            self.zoom_btn.setText("🔍 Zoom")
            self.spectrum_canvas.setCursor(Qt.ArrowCursor)

    def on_spectrum_scroll(self, event):
        if not self.zoom_mode or event.xdata is None or event.ydata is None:
            return
        xlim = self.ax_spectrum.get_xlim()
        ylim = self.ax_spectrum.get_ylim()
        factor = 0.9 if event.button == 'up' else 1.1

        x_center, y_center = event.xdata, event.ydata
        new_xmin = x_center + (xlim[0] - x_center) * factor
        new_xmax = x_center + (xlim[1] - x_center) * factor
        new_ymin = y_center + (ylim[0] - y_center) * factor
        new_ymax = y_center + (ylim[1] - y_center) * factor

        self.ax_spectrum.set_xlim(new_xmin, new_xmax)
        self.ax_spectrum.set_ylim(new_ymin, new_ymax)
        self.spectrum_canvas.draw_idle()

    def _apply_spectral_normalization(self, spec, mode):
        """Helper to apply chosen spectral normalization / chemometric standardisation."""
        spec = np.asarray(spec, dtype=np.float64).copy()
        if mode == 'total':
            tot = np.sum(np.abs(spec))
            return spec / tot if tot > 0 else spec
        elif mode == 'peak':
            pk = np.max(np.abs(spec))
            return spec / pk if pk > 0 else spec
        elif mode == 'l2':
            norm = np.linalg.norm(spec)
            return spec / norm if norm > 0 else spec
        elif mode == 'snv':
            std = np.std(spec)
            return (spec - np.mean(spec)) / std if std > 0 else (spec - np.mean(spec))
        return spec

    def _on_plot_tab_changed(self, index):
        """Callback when user switches between Tab 1 (Extracted) and Tab 2 (Analysis)."""
        if index == 0:
            self.update_spectrum_plot()
        elif index == 1:
            self.update_basis_plot()

    def on_norm_mode_changed(self):
        """Callback when user selects a different normalization dropdown mode."""
        self.update_spectrum_plot()
        self.update_basis_plot()
        if hasattr(self, 'unmixing_done') and self.unmixing_done and self.basis_spectra is not None:
            mode_lbl = self.norm_mode_combo.currentText()
            self._apply_unmixing_result({'concentrations': self.concentrations, 'basis_spectra': self.basis_spectra, 'r_squared': self.r_squared, 'residuals': self.residuals}, f"Analysis ({mode_lbl})")

    def update_spectrum_plot(self):
        """Plot all saved spectra plus current active selection (matching HyperViewer)."""
        self.ax_spectrum.clear()
        sub_bg = hasattr(self, 'subtract_bg_btn') and self.subtract_bg_btn.isChecked() and self.background_spectrum is not None
        mode = self.norm_mode_combo.currentData() if hasattr(self, 'norm_mode_combo') else 'raw'

        has_data = False
        # 1. Plot all saved/added spectra
        for sdata in self.spectra_list:
            if not sdata.visible:
                continue
            spec = (sdata.spectrum - self.background_spectrum) if sub_bg else sdata.spectrum.copy()
            spec = self._apply_spectral_normalization(spec, mode)

            wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, len(spec) + 1, dtype=float)
            ls = getattr(sdata, 'linestyle', '-')
            self.ax_spectrum.plot(wl, spec, color=sdata.color, linestyle=ls, label=sdata.label, linewidth=1.8)

            if sdata.std is not None:
                std = self._apply_spectral_normalization(sdata.std, mode) if mode != 'raw' else sdata.std
                self.ax_spectrum.fill_between(
                    wl, spec - std, spec + std,
                    alpha=0.2, color=sdata.color)
            has_data = True

        # 2. Plot active selection on top as 'Current' (black dashed line)
        if self.current_selection is not None and 'spectrum' in self.current_selection:
            spec = self.current_selection['spectrum'].copy()
            if sub_bg and self.background_spectrum is not None:
                spec = spec - self.background_spectrum
            spec = self._apply_spectral_normalization(spec, mode)

            wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, len(spec) + 1, dtype=float)
            self.ax_spectrum.plot(wl, spec, 'k--', linewidth=1.5, label='Current')

            if self.current_selection.get('std') is not None:
                std = self._apply_spectral_normalization(self.current_selection['std'], mode) if mode != 'raw' else self.current_selection['std']
                self.ax_spectrum.fill_between(
                    wl, spec - std, spec + std,
                    alpha=0.25, color='black')
            has_data = True

        self.ax_spectrum.set_xlabel("Wavelength / Band")
        mode_titles = {
            'raw': "Intensity",
            'total': "Total Normalized Intensity (spec / sum)",
            'peak': "Peak Normalized Intensity (spec / max)",
            'l2': "L2 Normalized Intensity (unit vector)",
            'snv': "SNV Z-Score ((spec - μ) / σ)"
        }
        self.ax_spectrum.set_ylabel(mode_titles.get(mode, "Intensity"))
        self.ax_spectrum.set_title(f"Extracted Spectra ({mode_titles.get(mode, 'Original')})")

        if has_data:
            self.ax_spectrum.legend(loc='upper right', fontsize=8)
            self.ax_spectrum.relim()
            d = self.ax_spectrum.dataLim
            if np.isfinite(d.xmin) and np.isfinite(d.xmax) and np.isfinite(d.ymin) and np.isfinite(d.ymax):
                x_margin = (d.xmax - d.xmin) * 0.05 if d.xmax != d.xmin else 1.0
                y_margin = (d.ymax - d.ymin) * 0.05 if d.ymax != d.ymin else 0.1
                computed_xlim = (d.xmin - x_margin, d.xmax + x_margin)
                computed_ylim = (d.ymin - y_margin, d.ymax + y_margin)
                if not self.axes_locked or self.saved_xlim is None or self.saved_ylim is None:
                    self.ax_spectrum.set_xlim(computed_xlim)
                    self.ax_spectrum.set_ylim(computed_ylim)
                    self.saved_xlim = computed_xlim
                    self.saved_ylim = computed_ylim
                else:
                    self.ax_spectrum.set_xlim(self.saved_xlim)
                    self.ax_spectrum.set_ylim(self.saved_ylim)

        self.ax_spectrum.grid(True, alpha=0.3)

        self.spectrum_canvas.draw_idle()

    def _plot_spectrum(self, spectrum=None):
        self.update_spectrum_plot()

    def _plot_residual(self, residual=None):
        self.update_spectrum_plot()

    def autoscale_y_axis(self):
        """Autoscale Y-axis limits to fit all visible spectra on active tab while keeping X range."""
        active_tab = self.plot_tabs.currentIndex() if hasattr(self, 'plot_tabs') else 0
        if active_tab == 0:
            ax = self.ax_spectrum
            canvas = self.spectrum_canvas
            mode = self.norm_mode_combo.currentData() if hasattr(self, 'norm_mode_combo') else 'raw'
            sub_bg = hasattr(self, 'subtract_bg_btn') and self.subtract_bg_btn.isChecked() and self.background_spectrum is not None
            xlim = ax.get_xlim()

            all_spectra = []
            for sdata in self.spectra_list:
                if not sdata.visible:
                    continue
                spec = (sdata.spectrum - self.background_spectrum) if sub_bg else sdata.spectrum.copy()
                all_spectra.append(self._apply_spectral_normalization(spec, mode))

            if self.current_selection is not None and 'spectrum' in self.current_selection:
                spec = self.current_selection['spectrum'].copy()
                if sub_bg and self.background_spectrum is not None:
                    spec = spec - self.background_spectrum
                all_spectra.append(self._apply_spectral_normalization(spec, mode))

            y_vals = []
            for spec in all_spectra:
                wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, len(spec) + 1, dtype=float)
                mask = (wl >= xlim[0]) & (wl <= xlim[1])
                if np.any(mask):
                    y_vals.extend(spec[mask])

            if y_vals:
                ymin, ymax = float(np.min(y_vals)), float(np.max(y_vals))
                margin = (ymax - ymin) * 0.05 if ymax != ymin else 1.0
                ax.set_ylim(ymin - margin, ymax + margin)
                canvas.draw_idle()
        else:
            ax = self.ax_analysis
            canvas = self.analysis_canvas
            mode = self.norm_mode_combo.currentData() if hasattr(self, 'norm_mode_combo') else 'raw'
            xlim = ax.get_xlim()
            y_vals = []
            if hasattr(self, 'basis_spectra_list') and self.basis_spectra_list:
                for sdata in self.basis_spectra_list:
                    if getattr(sdata, 'visible', True):
                        spec = self._apply_spectral_normalization(sdata.spectrum, mode)
                        wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, len(spec) + 1, dtype=float)
                        mask = (wl >= xlim[0]) & (wl <= xlim[1])
                        if np.any(mask):
                            y_vals.extend(spec[mask])
            if y_vals:
                ymin, ymax = float(np.min(y_vals)), float(np.max(y_vals))
                margin = (ymax - ymin) * 0.05 if ymax != ymin else 1.0
                ax.set_ylim(ymin - margin, ymax + margin)
                canvas.draw_idle()

    def add_current_spectrum(self):
        if self.current_selection is None or 'spectrum' not in self.current_selection:
            return
        sel = self.current_selection
        spec = sel['spectrum']
        std = sel.get('std', None)
        stype = sel.get('type', 'point')
        lbl = f"ROI {len(self.spectra_list) + 1}"
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
        color = colors[len(self.spectra_list) % len(colors)]

        sdata = SpectrumData(
            spec, std=std, label=lbl, color=color,
            selection_type=stype, coords=sel.get('coords'), mask=sel.get('mask')
        )
        sdata.radius = sel.get('radius')
        sdata.size = sel.get('size')
        self.spectra_list.append(sdata)
        self.update_checkbox_list()
        self.update_spectrum_plot()
        self.autoscale_axes()
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

    def update_checkbox_list(self):
        # Clear existing ROI list
        while self.checkbox_layout.count():
            child = self.checkbox_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                while child.layout().count():
                    sub = child.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        for idx, sdata in enumerate(self.spectra_list):
            row = QHBoxLayout()
            chk = QCheckBox()
            chk.setChecked(sdata.visible)
            chk.stateChanged.connect(lambda state, i=idx: self.toggle_spectrum_visibility(i))
            row.addWidget(chk)

            lbl = QLabel(sdata.label)
            lbl.setStyleSheet(f"color: {sdata.color}; font-size: 11px; font-weight: normal;")
            row.addWidget(lbl)
            row.addStretch()

            del_btn = QPushButton("×")
            del_btn.setFixedSize(20, 20)
            del_btn.setToolTip("Delete this ROI spectrum")
            del_btn.clicked.connect(lambda checked, i=idx: self.delete_single_spectrum(i))
            row.addWidget(del_btn)

            self.checkbox_layout.addLayout(row)

        if hasattr(self, 'delete_btn'):
            self.delete_btn.setEnabled(len(self.spectra_list) > 0)
        self.set_bg_btn.setEnabled(len(self.spectra_list) > 0)

    def copy_roi_to_basis(self):
        """➡️ Copy checked ROI spectra from Left list to Right Basis list."""
        if not hasattr(self, 'basis_spectra_list'):
            self.basis_spectra_list = []

        active_rois = [s for s in self.spectra_list if getattr(s, 'visible', True)]
        if not active_rois:
            QMessageBox.warning(self, "No ROI Selected", "Check at least 1 ROI spectrum in the left list to copy to Basis.")
            return

        copied_count = 0
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for sdata in active_rois:
            b_label = sdata.label
            new_sdata = SpectrumData(
                sdata.spectrum.copy(), std=None, label=b_label, color=colors[len(self.basis_spectra_list) % len(colors)],
                selection_type='basis', coords=None, mask=None
            )
            self.basis_spectra_list.append(new_sdata)
            copied_count += 1

        self.update_basis_checkbox_list()
        self._update_basis_matrix_from_list()
        self.statusBar().showMessage(f"Copied {copied_count} ROI spectra to Basis Fitting Library.")

    def update_basis_checkbox_list(self):
        """Update Right List Box (Basis / Reference Spectra)."""
        if not hasattr(self, 'basis_spectra_list'):
            self.basis_spectra_list = []

        while self.basis_layout.count():
            child = self.basis_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                while child.layout().count():
                    sub = child.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        for idx, sdata in enumerate(self.basis_spectra_list):
            row = QHBoxLayout()
            chk = QCheckBox()
            chk.setChecked(getattr(sdata, 'visible', True))
            chk.stateChanged.connect(lambda state, i=idx: self.toggle_basis_visibility(i))
            row.addWidget(chk)

            lbl = QLabel(sdata.label)
            lbl.setStyleSheet(f"color: {sdata.color}; font-size: 11px; font-weight: normal;")
            row.addWidget(lbl)
            row.addStretch()

            del_btn = QPushButton("×")
            del_btn.setFixedSize(20, 20)
            del_btn.setToolTip("Delete this basis spectrum")
            del_btn.clicked.connect(lambda checked, i=idx: self.delete_single_basis_spectrum(i))
            row.addWidget(del_btn)

            self.basis_layout.addLayout(row)

    def toggle_basis_visibility(self, index):
        if hasattr(self, 'basis_spectra_list') and 0 <= index < len(self.basis_spectra_list):
            self.basis_spectra_list[index].visible = not getattr(self.basis_spectra_list[index], 'visible', True)
            self._update_basis_matrix_from_list()

    def delete_single_basis_spectrum(self, index):
        if hasattr(self, 'basis_spectra_list') and 0 <= index < len(self.basis_spectra_list):
            self.basis_spectra_list.pop(index)
            self.update_basis_checkbox_list()
            self._update_basis_matrix_from_list()

    def _update_basis_matrix_from_list(self):
        if not hasattr(self, 'basis_spectra_list') or not self.basis_spectra_list:
            self.basis_spectra = None
            self.basis_labels = []
            self.update_basis_plot()
            return

        active_basis = [s for s in self.basis_spectra_list if getattr(s, 'visible', True)]
        if active_basis:
            self.basis_spectra = np.array([s.spectrum for s in active_basis])
            self.basis_labels = [s.label for s in active_basis]
            first_len = len(active_basis[0].spectrum)
            if self.basis_wavelengths is None or len(self.basis_wavelengths) != first_len:
                self.basis_wavelengths = np.arange(1, first_len + 1, dtype=float)
        else:
            self.basis_spectra = None
            self.basis_labels = []

        self.update_basis_plot()

    def update_basis_plot(self):
        """Plot all active basis/reference spectra on Tab 2 (Analysis Results & Fitting)."""
        if not hasattr(self, 'analysis_fig') or self.analysis_fig is None:
            return
        self.analysis_fig.clear()
        mode = self.norm_mode_combo.currentData() if hasattr(self, 'norm_mode_combo') else 'raw'
        if hasattr(self, 'basis_spectra_list') and self.basis_spectra_list:
            active_basis = [s for s in self.basis_spectra_list if getattr(s, 'visible', True)]
            if active_basis:
                ax = self.analysis_fig.add_subplot(111)
                first_len = len(active_basis[0].spectrum)
                wl = self.basis_wavelengths if self.basis_wavelengths is not None and len(self.basis_wavelengths) == first_len else np.arange(1, first_len + 1, dtype=float)
                for sdata in active_basis:
                    spec = self._apply_spectral_normalization(sdata.spectrum, mode)
                    ax.plot(wl, spec, color=sdata.color, linewidth=2.0, label=sdata.label)

                mode_titles = {
                    'raw': "Amplitude / Intensity",
                    'total': "Normalized Amplitude (spec / sum)",
                    'peak': "Peak Normalized Amplitude (spec / max)",
                    'l2': "L2 Vector Norm (spec / ||spec||)",
                    'snv': "SNV Z-Score ((spec - μ) / σ)"
                }
                ax.set_xlabel("Spectral Band / Wavelength")
                ax.set_ylabel(mode_titles.get(mode, "Amplitude"))
                ax.set_title(f"Basis & Reference Spectra Library (K={len(active_basis)}) [{mode_titles.get(mode, 'Original')}]")
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=8)

        self.analysis_fig.tight_layout()
        self.analysis_canvas.draw_idle()

    def fit_selected_roi_to_basis(self):
        """Fit checked/selected ROI spectrum against Basis spectra using NNLS and plot fit + residual."""
        active_rois = [s for s in self.spectra_list if getattr(s, 'visible', True)]
        if not active_rois:
            QMessageBox.warning(self, "No ROI Selected", "Select or check at least 1 ROI spectrum in the left list to fit.")
            return

        if not hasattr(self, 'basis_spectra_list') or not self.basis_spectra_list:
            QMessageBox.warning(self, "No Basis Spectra", "No basis spectra in the right list! Click '➡️ Add to Basis' or load reference spectra first.")
            return

        active_basis = [s for s in self.basis_spectra_list if getattr(s, 'visible', True)]
        if not active_basis:
            QMessageBox.warning(self, "No Basis Spectra Checked", "Check at least 1 basis spectrum in the right list.")
            return

        mode = self.norm_mode_combo.currentData() if hasattr(self, 'norm_mode_combo') else 'raw'

        target_sdata = active_rois[0]
        measured_spec = self._apply_spectral_normalization(target_sdata.spectrum, mode)
        basis_matrix = np.array([self._apply_spectral_normalization(b.spectrum, mode) for b in active_basis])
        basis_labels = [b.label for b in active_basis]
        basis_colors = [b.color for b in active_basis]

        first_len = len(measured_spec)
        wl = self.basis_wavelengths if self.basis_wavelengths is not None and len(self.basis_wavelengths) == first_len else np.arange(1, first_len + 1, dtype=float)

        try:
            from scipy.optimize import nnls
            coeffs, _ = nnls(basis_matrix.T, measured_spec)
        except Exception:
            coeffs, _, _, _ = np.linalg.lstsq(basis_matrix.T, measured_spec, rcond=None)
            coeffs = np.maximum(coeffs, 0)

        fitted_spec = basis_matrix.T @ coeffs
        residuals = measured_spec - fitted_spec
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((measured_spec - np.mean(measured_spec)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rms_err = np.sqrt(np.mean(residuals ** 2))

        tot_c = np.sum(coeffs)
        percents = (coeffs / tot_c * 100.0) if tot_c > 0 else np.zeros_like(coeffs)

        # Add Fitted Spectrum (dashed red line) and Residual Spectrum (dotted magenta line) to ROI list (Left side)
        fit_label = f"[Fit] {target_sdata.label}"
        sdata_fit = SpectrumData(
            fitted_spec, std=None, label=fit_label, color='red',
            selection_type='fit', coords=None, mask=None, linestyle='--'
        )
        self.spectra_list.append(sdata_fit)

        res_label = f"[Res] {target_sdata.label}"
        sdata_res = SpectrumData(
            residuals, std=None, label=res_label, color='magenta',
            selection_type='fit', coords=None, mask=None, linestyle=':'
        )
        self.spectra_list.append(sdata_res)

        self.update_checkbox_list()
        self.update_spectrum_plot()

        if hasattr(self, 'plot_tabs'):
            self.plot_tabs.setCurrentIndex(0) # Switch to Tab 1 (Extracted Spectra)

        msg = f"NNLS Fit for {target_sdata.label}:\nR² = {r2:.4f}, RMS Error = {rms_err:.4e}\n\nComponents:\n"
        for k in range(len(active_basis)):
            msg += f"  • {basis_labels[k]}: coeff = {coeffs[k]:.4f} ({percents[k]:.1f}%)\n"
        QMessageBox.information(self, "ROI Fit Complete", msg)

    def toggle_spectrum_visibility(self, index):
        if 0 <= index < len(self.spectra_list):
            self.spectra_list[index].visible = not self.spectra_list[index].visible
            self.update_spectrum_plot()
            if self.hypercube is not None:
                self.display_frame(self.current_frame)

    def delete_single_spectrum(self, index):
        if 0 <= index < len(self.spectra_list):
            self.spectra_list.pop(index)
            self.update_checkbox_list()
            self.update_spectrum_plot()
            self.autoscale_axes()
            if self.hypercube is not None:
                self.display_frame(self.current_frame)

    def delete_selected_spectrum(self):
        """Delete checked spectra, or the last spectrum if none checked."""
        if not self.spectra_list:
            return

        indices_to_delete = []
        for i in range(self.checkbox_layout.count()):
            item = self.checkbox_layout.itemAt(i)
            if item and item.layout():
                check_item = item.layout().itemAt(0)
                if check_item and check_item.widget() and isinstance(check_item.widget(), QCheckBox):
                    if check_item.widget().isChecked():
                        indices_to_delete.append(i)

        if not indices_to_delete:
            indices_to_delete = [len(self.spectra_list) - 1]

        for idx in sorted(indices_to_delete, reverse=True):
            if 0 <= idx < len(self.spectra_list):
                self.spectra_list.pop(idx)

        self.update_checkbox_list()
        self.update_spectrum_plot()
        self.autoscale_axes()
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

    def set_background_spectrum(self):
        bg_id = self.bg_radio_group.checkedId()
        if bg_id < 0 and self.spectra_list:
            bg_id = len(self.spectra_list) - 1
        if 0 <= bg_id < len(self.spectra_list):
            self.background_spectrum = self.spectra_list[bg_id].spectrum.copy()
            self.subtract_bg_btn.setEnabled(True)
            lbl_text = f"Background: {self.spectra_list[bg_id].label}"
            if hasattr(self, 'bg_status_label') and self.bg_status_label is not None:
                self.bg_status_label.setText(lbl_text)
                self.bg_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.statusBar().showMessage(f"Set background spectrum to: {self.spectra_list[bg_id].label}")
        else:
            QMessageBox.warning(self, "No Spectrum Selected", "Add a spectrum to the list first or select a radio button to set as background.")

    def toggle_background_subtraction(self):
        if self.background_spectrum is None:
            self.subtract_bg_btn.setChecked(False)
            return
        self._plot_all_spectra()

    def _plot_all_spectra(self):
        self.update_spectrum_plot()

    def _export_all_spectra(self):
        """
        Smart Export Spectra:
        Exports whatever spectra are currently displayed in the active tab/window:
        - Tab 1 (Extracted Spectra): Exports saved ROI & point spectra.
        - Tab 2 (Analysis Results & Fitting): Exports unmixed component spectra / endmembers.
        """
        is_analysis_tab = hasattr(self, 'plot_tabs') and self.plot_tabs.currentIndex() == 1

        if is_analysis_tab and self.basis_spectra is not None:
            # Export Component / Endmember Spectra from Analysis Window
            file_path, _ = QFileDialog.getSaveFileName(self, "Export Analysis Component Spectra", "", "CSV Files (*.csv);;NumPy Files (*.npy)")
            if not file_path:
                return
            try:
                K, L = self.basis_spectra.shape
                wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, L + 1, dtype=float)
                labels = self.basis_labels or [f"Component {i+1}" for i in range(K)]

                if file_path.endswith('.npy'):
                    np.save(file_path, {'spectra': self.basis_spectra, 'labels': labels, 'wavelengths': wl})
                else:
                    with open(file_path, 'w') as f:
                        f.write("Wavelength," + ",".join(labels) + "\n")
                        for i, w in enumerate(wl):
                            row = [str(w)] + [str(self.basis_spectra[j, i]) for j in range(K)]
                            f.write(",".join(row) + "\n")
                self.statusBar().showMessage(f"Exported {K} component spectra to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

        elif self.spectra_list:
            # Export Saved ROI Spectra from ROI Manager Window
            file_path, _ = QFileDialog.getSaveFileName(self, "Export ROI Spectra", "", "CSV Files (*.csv);;NumPy Files (*.npy)")
            if not file_path:
                return
            try:
                first_len = len(self.spectra_list[0].spectrum)
                wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, first_len + 1, dtype=float)
                if file_path.endswith('.npy'):
                    np.save(file_path, {'spectra': np.array([s.spectrum for s in self.spectra_list]),
                                        'labels': [s.label for s in self.spectra_list],
                                        'wavelengths': wl})
                else:
                    with open(file_path, 'w') as f:
                        f.write("Wavelength," + ",".join(s.label for s in self.spectra_list) + "\n")
                        for i, w in enumerate(wl):
                            row = [str(w)] + [str(s.spectrum[i]) if i < len(s.spectrum) else "" for s in self.spectra_list]
                            f.write(",".join(row) + "\n")
                self.statusBar().showMessage(f"Exported {len(self.spectra_list)} ROI spectra to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
        else:
            QMessageBox.warning(self, "No Spectra", "No spectra are currently displayed to export.")

    # -----------------------------------------------------------------------
    # Reference Spectra Library Loading & Transforming
    # -----------------------------------------------------------------------

    def load_reference_spectra(self):
        """Load reference spectra from file (CSV, XLSX, MAT, NPY, TXT)."""
        file_types = "All Supported (*.csv *.xlsx *.mat *.npy *.txt);;CSV Files (*.csv);;Excel Files (*.xlsx);;MAT Files (*.mat);;NumPy Files (*.npy);;Text Files (*.txt);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Reference Spectra", "", file_types)
        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()
            spectra, labels, wavelengths = None, [], None

            if ext == '.csv':
                wavelengths, spectra, labels = self._load_csv_spectra(file_path)
            elif ext == '.txt':
                wavelengths, spectra, labels = self._load_txt_spectra(file_path)
            elif ext == '.xlsx':
                import pandas as pd
                df = pd.read_excel(file_path)
                if all(self._is_numeric(v) for v in df.iloc[0].values):
                    wavelengths = df.iloc[0, 1:].values.astype(float)
                    labels = [str(df.iloc[i, 0]) for i in range(1, df.shape[0])]
                    spectra = df.iloc[1:, 1:].values.astype(float)
                else:
                    wavelengths = df.iloc[:, 0].values.astype(float)
                    labels = [str(c) for c in df.columns[1:]]
                    spectra = df.iloc[:, 1:].values.T.astype(float)
            elif ext == '.mat':
                if not HAS_SCIPY:
                    QMessageBox.critical(self, "Error", "Install scipy for MAT file support: pip install scipy")
                    return
                mat_data = sio.loadmat(file_path)
                for key, val in mat_data.items():
                    if not key.startswith('_') and isinstance(val, np.ndarray) and val.ndim == 2:
                        spectra = val
                        labels = [f"Spectrum {i+1}" for i in range(spectra.shape[0])]
                        break
            elif ext == '.npy':
                data = np.load(file_path, allow_pickle=True)
                if isinstance(data, np.ndarray) and data.ndim == 0:
                    data = data.item()
                if isinstance(data, dict):
                    spectra = np.array(data.get('spectra', []))
                    wavelengths = np.array(data.get('wavelengths', []), dtype=float) if 'wavelengths' in data else None
                    labels = [str(l) for l in data.get('labels', [])]
                elif isinstance(data, np.ndarray) and data.ndim == 2:
                    spectra = data
                    labels = [f"Spectrum {i+1}" for i in range(spectra.shape[0])]

            if spectra is None or spectra.ndim != 2:
                QMessageBox.critical(self, "Parse Error", "Could not parse 2D spectra array from file.")
                return

            if wavelengths is None or len(wavelengths) != spectra.shape[1]:
                wavelengths = np.arange(1, spectra.shape[1] + 1, dtype=float)
            if not labels or len(labels) != spectra.shape[0]:
                labels = [f"Spectrum {i+1}" for i in range(spectra.shape[0])]

            self.reference_spectra = spectra
            self.reference_labels = labels
            self.reference_wavelengths = wavelengths

            self.basis_spectra = spectra
            self.basis_labels = labels
            self.basis_wavelengths = wavelengths

            # Add each reference spectrum directly into Right List Box (Basis / Reference Spectra)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            if not hasattr(self, 'basis_spectra_list'):
                self.basis_spectra_list = []
            for i in range(spectra.shape[0]):
                lbl = labels[i]
                sdata = SpectrumData(
                    spectra[i], label=lbl, color=colors[i % len(colors)],
                    selection_type='reference', coords=None
                )
                self.basis_spectra_list.append(sdata)

            self.update_basis_checkbox_list()
            self._update_basis_matrix_from_list()

            n_ref = len(labels)
            wl_range = f"{wavelengths[0]:.1f}–{wavelengths[-1]:.1f}"
            if hasattr(self, 'ref_status_label'):
                self.ref_status_label.setText(f"Reference Spectra: {n_ref} loaded (λ: {wl_range})")
                self.ref_status_label.setStyleSheet("color: green; font-weight: bold;")
            if hasattr(self, 'transform_btn'):
                self.transform_btn.setEnabled(True)
            self.statusBar().showMessage(f"Loaded {n_ref} reference spectra into Basis library (λ: {wl_range})")

            if hasattr(self, 'plot_tabs'):
                self.plot_tabs.setCurrentIndex(1) # Switch to Tab 2 (Analysis Results & Fitting)
            self.statusBar().showMessage(f"Loaded {n_ref} reference spectra into Basis library.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load reference spectra:\n{e}")

    def _is_numeric(self, val):
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    def _load_csv_spectra(self, file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 2:
            return None, None, []
        header = lines[0].split(',')
        first_col_numeric = all(self._is_numeric(lines[i].split(',')[0].strip()) for i in range(1, len(lines)))
        first_row_numeric = all(self._is_numeric(v.strip()) for v in header)

        if first_col_numeric and len(header) > 1:
            wavelengths = np.array([float(lines[i].split(',')[0]) for i in range(1, len(lines))])
            labels = [h.strip() for h in header[1:]]
            spectra = np.array([[float(v) for v in lines[i].split(',')[1:]] for i in range(1, len(lines))]).T
        elif first_row_numeric:
            wavelengths = np.array([float(v.strip()) for v in header])
            labels, data_rows = [], []
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) > 1:
                    labels.append(parts[0].strip())
                    data_rows.append([float(v) for v in parts[1:]])
            spectra = np.array(data_rows)
        else:
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            spectra = data.T if data.shape[0] < data.shape[1] else data
            wavelengths = np.arange(1, spectra.shape[1] + 1, dtype=float)
            labels = [f"Spectrum {i+1}" for i in range(spectra.shape[0])]

        return wavelengths, spectra, labels

    def _load_txt_spectra(self, file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 2:
            return None, None, []
        header = lines[0].split()
        first_col_numeric = all(self._is_numeric(lines[i].split()[0].strip()) for i in range(1, len(lines)))
        if first_col_numeric and len(header) > 1:
            wavelengths = np.array([float(lines[i].split()[0]) for i in range(1, len(lines))])
            labels = [h.strip() for h in header[1:]]
            spectra = np.array([[float(v) for v in lines[i].split()[1:]] for i in range(1, len(lines))]).T
        else:
            wavelengths = np.array([float(v.strip()) for v in header])
            labels, data_rows = [], []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) > 1:
                    labels.append(parts[0].strip())
                    data_rows.append([float(v) for v in parts[1:]])
            spectra = np.array(data_rows)

        return wavelengths, spectra, labels

    def _show_reference_window(self):
        if self.reference_spectra is None:
            return
        if self.reference_window is None:
            self.reference_window = SpectraDisplayWindow(
                "Reference Spectra Library", self.reference_wavelengths,
                self.reference_spectra, self.reference_labels, parent=self)
        else:
            self.reference_window.update_data(self.reference_wavelengths, self.reference_spectra, self.reference_labels)
        self.reference_window.show()

    def transform_spectra(self):
        if self.reference_spectra is None:
            QMessageBox.warning(self, "Warning", "No reference spectra loaded!")
            return
        if self.current_selection is None:
            QMessageBox.warning(self, "Warning", "Make a selection on the canvas first!")
            return

        I = self.current_selection['spectrum']
        wavelengths = self.reference_wavelengths
        if len(I) != len(wavelengths):
            QMessageBox.warning(self, "Warning", f"Dimension mismatch: selection has {len(I)} bands, reference has {len(wavelengths)} bands.")
            return

        I_expanded = I[:, np.newaxis]
        lambda_expanded = wavelengths[np.newaxis, :]
        cos_squared = np.cos(np.pi * I_expanded / lambda_expanded) ** 2
        transformed = (cos_squared @ self.reference_spectra.T).T

        self.transformed_spectra = transformed
        self.transformed_labels = [f"Transformed: {lbl}" for lbl in self.reference_labels]
        self.transformed_wavelengths = wavelengths
        self.save_transform_btn.setEnabled(True)

        self.basis_spectra = transformed
        self.basis_labels = self.transformed_labels

        QMessageBox.information(self, "Transform Complete", f"Transformed {len(transformed)} reference spectra.")

    def save_transformed_spectra(self):
        if self.transformed_spectra is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Transformed Spectra", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            with open(file_path, 'w') as f:
                f.write("Wavelength," + ",".join(self.transformed_labels) + "\n")
                for i, w in enumerate(self.transformed_wavelengths):
                    row = [str(w)] + [str(self.transformed_spectra[j, i]) for j in range(self.transformed_spectra.shape[0])]
                    f.write(",".join(row) + "\n")
            self.statusBar().showMessage(f"Saved transformed spectra to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # -----------------------------------------------------------------------
    # Unmixing & Analysis Algorithm Runners
    # -----------------------------------------------------------------------

    def denoise_savgol(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        result = denoise_savgol(self.hypercube, window_length=7, polyorder=2)
        self.hypercube = result["hypercube"]
        self.display_frame(self.current_frame)
        QMessageBox.information(self, "Savitzky-Golay Denoising Complete",
                                f"Applied Savitzky-Golay Spectral Filter:\n"
                                f"Window Length: {result['info']['window_length']} bands, Poly order: {result['info']['polyorder']}")

    def denoise_tv3d(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        w, ok = QInputDialog.getDouble(self, "3D-TV Denoising", "Denoising Strength / Weight λ (higher = smoother):", 0.05, 0.001, 2.0, 3)
        if not ok:
            return
        try:
            result = denoise_tv3d(self.hypercube, weight=w, n_iter_max=30)
            self.hypercube = result["hypercube"]
            self.display_frame(self.current_frame)
            self.update_spectrum_plot()
            QMessageBox.information(self, "3D-TV Denoising Complete", f"3D Total Variation (TV) applied with smoothing weight λ = {w}.")
        except Exception as e:
            QMessageBox.critical(self, "3D-TV Denoising Error", str(e))

    def denoise_bm4d(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        try:
            result = denoise_bm4d(self.hypercube)
            self.hypercube = result["hypercube"]
            self.display_frame(self.current_frame)
            self.update_spectrum_plot()
            info = result['info']
            msg = f"Algorithm: {info['algorithm']}"
            if 'note' in info:
                msg += f"\n\n{info['note']}"
            QMessageBox.information(self, "BM4D Denoising Complete", msg)
        except Exception as e:
            QMessageBox.critical(self, "BM4D Denoising Error", str(e))

    def denoise_mppca(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        result = denoise_mppca(self.hypercube, variance_threshold=0.95)
        self.hypercube = result["hypercube"]
        self.display_frame(self.current_frame)
        info = result["info"]
        QMessageBox.information(self, "MPPCA Denoising Complete",
                                f"Retained {info['n_components_kept']} of {info['n_components_total']} components\n"
                                f"({info['cumulative_variance'] * 100:.1f}% cumulative variance)")

    def denoise_wavelet(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        try:
            result = denoise_wavelet(self.hypercube, wavelet='db4')
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", "Install PyWavelets for wavelet denoising:\n  pip install PyWavelets")
            return
        self.hypercube = result["hypercube"]
        self.display_frame(self.current_frame)
        QMessageBox.information(self, "Wavelet Denoising Complete", f"Wavelet: db4, Level: {result['info']['level']}")

    def run_mcr_als(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n_comp, ok = QInputDialog.getInt(self, "MCR-ALS Unmixing", "Number of components / endmembers:", 3, 2, min(10, L - 1))
        if not ok:
            return
        self.progress_bar.setValue(10)
        self.unmix_status.setText("MCR-ALS in progress…")
        try:
            result = run_mcr_als(self.hypercube, n_components=n_comp, max_iter=150, non_neg=True)
            self.basis_labels = [f"MCR Component {i+1}" for i in range(n_comp)]
            self._apply_unmixing_result(result, f"MCR-ALS (K={n_comp})")
        except Exception as e:
            QMessageBox.critical(self, "MCR-ALS Error", f"MCR-ALS failed:\n{e}")

    def run_nmf(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n_comp, ok = QInputDialog.getInt(self, "NMF Unmixing", "Number of components / endmembers:", 3, 2, min(10, L - 1))
        if not ok:
            return
        self.progress_bar.setValue(10)
        self.unmix_status.setText("NMF in progress…")
        try:
            result = run_nmf(self.hypercube, n_components=n_comp, max_iter=200)
            self.basis_labels = [f"NMF Component {i+1}" for i in range(n_comp)]
            self._apply_unmixing_result(result, f"NMF (K={n_comp})")
        except Exception as e:
            QMessageBox.critical(self, "NMF Error", f"NMF failed:\n{e}")

    def _get_normalized_hypercube(self):
        """Return hypercube normalized according to active norm_mode_combo selection."""
        if self.hypercube is None:
            return None
        mode = self.norm_mode_combo.currentData() if hasattr(self, 'norm_mode_combo') else 'raw'
        if mode == 'raw':
            return self.hypercube

        H, W, L = self.hypercube.shape
        cube = self.hypercube.astype(np.float64)
        if mode == 'total':
            tot = np.sum(np.abs(cube), axis=-1, keepdims=True)
            return np.where(tot > 0, cube / tot, cube)
        elif mode == 'peak':
            pk = np.max(np.abs(cube), axis=-1, keepdims=True)
            return np.where(pk > 0, cube / pk, cube)
        elif mode == 'l2':
            norm = np.linalg.norm(cube, axis=-1, keepdims=True)
            return np.where(norm > 0, cube / norm, cube)
        elif mode == 'snv':
            std = np.std(cube, axis=-1, keepdims=True)
            mean = np.mean(cube, axis=-1, keepdims=True)
            return np.where(std > 0, (cube - mean) / std, cube - mean)
        return cube

    def _get_or_extract_basis_spectra(self, default_k=3):
        """
        Smart Basis Extractor pulling directly from Right List Box (Basis / Reference Spectra).
        """
        mode = self.norm_mode_combo.currentData() if hasattr(self, 'norm_mode_combo') else 'raw'
        basis = None
        labels = []

        if hasattr(self, 'basis_spectra_list') and self.basis_spectra_list:
            active_basis = [s for s in self.basis_spectra_list if getattr(s, 'visible', True)]
            if active_basis:
                basis = np.array([s.spectrum for s in active_basis])
                labels = [s.label for s in active_basis]
                self.basis_spectra = basis
                self.basis_labels = labels

        if basis is None and self.basis_spectra is not None:
            basis = self.basis_spectra
            labels = self.basis_labels or [f"Basis {i+1}" for i in range(basis.shape[0])]

        if basis is None and self.hypercube is not None:
            # Fallback: Auto-extract VCA endmembers if Basis list is completely empty
            H, W, L = self.hypercube.shape
            k = min(default_k, L - 1)
            res = run_vca(self._get_normalized_hypercube(), n_endmembers=k)
            basis = res['endmembers']
            labels = [f"VCA Endmember {i+1}" for i in range(k)]
            self.basis_spectra = basis
            self.basis_labels = labels
            if not hasattr(self, 'basis_spectra_list'):
                self.basis_spectra_list = []
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i in range(k):
                sdata = SpectrumData(basis[i], label=labels[i], color=colors[i % len(colors)], selection_type='vca')
                self.basis_spectra_list.append(sdata)
            self.update_basis_checkbox_list()

        if basis is not None and mode != 'raw':
            norm_basis = np.array([self._apply_spectral_normalization(b, mode) for b in basis])
            return norm_basis, labels

        return basis, labels

    def run_linear_unmix(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        basis, labels = self._get_or_extract_basis_spectra()
        if basis is None:
            return
        cube = self._get_normalized_hypercube()
        self.progress_bar.setValue(10)
        mode_txt = self.norm_mode_combo.currentText()
        self.unmix_status.setText(f"NNLS Linear Unmixing [{mode_txt}] in progress…")
        try:
            result = run_linear_unmix(cube, basis)
            self._apply_unmixing_result(result, f"NNLS Linear Unmixing [{mode_txt}]")
        except Exception as e:
            QMessageBox.critical(self, "Linear Unmixing Error", f"NNLS Linear Unmixing failed:\n{e}")

    def run_mesma(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        basis, labels = self._get_or_extract_basis_spectra()
        if basis is None:
            return
        K = basis.shape[0]
        max_end, ok = QInputDialog.getInt(self, "MESMA Parameters", f"Max endmembers allowed per pixel (Library size={K}):", min(3, K), 2, K)
        if not ok:
            return
        cube = self._get_normalized_hypercube()
        self.progress_bar.setValue(10)
        mode_txt = self.norm_mode_combo.currentText()
        self.unmix_status.setText(f"MESMA [{mode_txt}] in progress…")
        try:
            result = run_mesma(cube, basis, max_endmembers=max_end)
            self._apply_unmixing_result(result, f"MESMA [{mode_txt}] (Max={max_end})")
        except Exception as e:
            QMessageBox.critical(self, "MESMA Error", f"MESMA failed:\n{e}")

    def run_pca(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n_comp, ok = QInputDialog.getInt(self, "PCA Decomposition", "Number of Principal Components:", 3, 2, min(10, L - 1))
        if not ok:
            return
        try:
            cube = self._get_normalized_hypercube()
            result = run_pca(cube, n_components=n_comp)
            self.basis_labels = [f"PC-{i+1}" for i in range(n_comp)]
            self._apply_unmixing_result(result, f"PCA (K={n_comp})")
        except Exception as e:
            QMessageBox.critical(self, "PCA Error", f"PCA failed:\n{e}")

    def run_sam(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        basis, labels = self._get_or_extract_basis_spectra()
        if basis is None:
            return
        try:
            result = run_sam(self.hypercube, basis)
            self._apply_unmixing_result(result, "SAM Classification")
        except Exception as e:
            QMessageBox.critical(self, "SAM Error", f"SAM failed:\n{e}")

    def run_sid(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        basis, labels = self._get_or_extract_basis_spectra()
        if basis is None:
            return
        try:
            result = run_sid(self.hypercube, basis)
            self._apply_unmixing_result(result, "SID Classification")
        except Exception as e:
            QMessageBox.critical(self, "SID Error", f"SID failed:\n{e}")

    def run_fcls(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        basis, labels = self._get_or_extract_basis_spectra()
        if basis is None:
            return
        cube = self._get_normalized_hypercube()
        self.progress_bar.setValue(10)
        mode_txt = self.norm_mode_combo.currentText()
        self.unmix_status.setText(f"FCLS [{mode_txt}] in progress…")
        try:
            result = run_fcls(cube, basis)
            self._apply_unmixing_result(result, f"FCLS [{mode_txt}]")
        except Exception as e:
            QMessageBox.critical(self, "FCLS Error", f"FCLS failed:\n{e}")

    def run_mnf(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n_comp, ok = QInputDialog.getInt(self, "MNF Decomposition", "Number of MNF Components:", 3, 2, min(10, L - 1))
        if not ok:
            return
        try:
            cube = self._get_normalized_hypercube()
            result = run_mnf(cube, n_components=n_comp)
            self.basis_labels = [f"MNF-{i+1}" for i in range(n_comp)]
            self._apply_unmixing_result(result, f"MNF (K={n_comp})")
        except Exception as e:
            QMessageBox.critical(self, "MNF Error", f"MNF failed:\n{e}")

    def run_rx(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        try:
            result = run_rx(self.hypercube)
            score_map = result['anomaly_score']
            info = result['info']

            self.ax_main.clear()
            im = self.ax_main.imshow(score_map, cmap='hot', interpolation='nearest')
            self.ax_main.set_title("RX Anomaly Detection")
            self.ax_cbar.clear()
            self.image_fig.colorbar(im, cax=self.ax_cbar, orientation='vertical')
            self.image_canvas.draw_idle()

            self.unmix_status.setText(
                f"RX Anomaly Detection — Mean: {info['mean_score']:.2f}, "
                f"Max: {info['max_score']:.2f}, 99th pct: {info['threshold_99']:.2f}")
            self.unmix_status.setStyleSheet("color: green; font-weight: bold;")
            self.progress_bar.setValue(100)
        except Exception as e:
            QMessageBox.critical(self, "RX Error", f"RX anomaly detection failed:\n{e}")

    def run_ica(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n_comp, ok = QInputDialog.getInt(self, "ICA Decomposition", "Number of Independent Components:", 3, 2, min(10, L - 1))
        if not ok:
            return
        try:
            from sklearn.decomposition import FastICA
            cube = self._get_normalized_hypercube()
            data = cube.reshape(-1, L).astype(float)
            ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
            C = ica.fit_transform(data)
            S_mat = ica.components_
            reconstructed = C @ S_mat
            ss_res = np.sum((data - reconstructed) ** 2, axis=1)
            ss_tot = np.sum((data - np.mean(data, axis=0)) ** 2, axis=1)
            r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)

            result = {
                'concentrations': C.reshape(H, W, -1),
                'basis_spectra': S_mat,
                'r_squared': r2.reshape(H, W),
                'residuals': np.sqrt(ss_res / L).reshape(H, W),
                'info': {'n_components': S_mat.shape[0]}
            }
            self.basis_labels = [f"IC-{i+1}" for i in range(n_comp)]
            self._apply_unmixing_result(result, f"ICA (K={n_comp})")
        except Exception as e:
            QMessageBox.critical(self, "ICA Error", f"ICA failed:\n{e}")

    def run_nfindr_dialog(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n_end, ok = QInputDialog.getInt(self, "N-FINDR Endmember Extraction", "Number of endmembers:", 3, 2, min(10, L-1))
        if ok:
            try:
                res = run_nfindr(self.hypercube, n_endmembers=n_end)
                self.basis_spectra = res['endmembers']
                self.basis_labels = [f"N-FINDR Endmember {i+1}" for i in range(n_end)]
                result = run_linear_unmix(self.hypercube, self.basis_spectra)
                self._apply_unmixing_result(result, f"N-FINDR Unmixing (K={n_end})")
            except Exception as e:
                QMessageBox.critical(self, "N-FINDR Error", str(e))

    def run_vca_dialog(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n_end, ok = QInputDialog.getInt(self, "VCA Endmember Extraction", "Number of endmembers:", 3, 2, min(10, L-1))
        if ok:
            try:
                res = run_vca(self.hypercube, n_endmembers=n_end)
                self.basis_spectra = res['endmembers']
                self.basis_labels = [f"VCA Endmember {i+1}" for i in range(n_end)]
                result = run_linear_unmix(self.hypercube, self.basis_spectra)
                self._apply_unmixing_result(result, f"VCA Unmixing (K={n_end})")
            except Exception as e:
                QMessageBox.critical(self, "VCA Error", str(e))

    def toggle_endmember_picker(self):
        self.endmember_mode = not self.endmember_mode
        if self.endmember_mode:
            self.statusBar().showMessage("Endmember Picker Active: Click pixels on Spatial View")
        else:
            self.statusBar().showMessage("Endmember Picker Deactivated")

    def run_fit_spectrum(self, model_type='gaussian'):
        if self.current_selection is None or 'spectrum' not in self.current_selection:
            QMessageBox.warning(self, "No Spectrum", "Select a pixel or region first.")
            return
        spectrum = self.current_selection['spectrum']
        try:
            res = fit_spectrum(spectrum, wavelengths=self.basis_wavelengths, model=model_type)
            wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, len(spectrum) + 1, dtype=float)

            self.analysis_fig.clear()
            gs = self.analysis_fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)
            ax1 = self.analysis_fig.add_subplot(gs[0])
            ax2 = self.analysis_fig.add_subplot(gs[1], sharex=ax1)

            # Subplot 1: Measured Spectrum + Total Fitted Curve
            ax1.plot(wl, spectrum, 'k.', markersize=6, label='Measured Data')
            ax1.plot(wl, res['fitted'], 'r-', linewidth=2.0, label=f"{model_type.title()} Fit (Total)")

            # Draw individual peak components
            for pk_i, pk in enumerate(res.get('peaks', [])):
                if 'component' in pk:
                    ax1.plot(wl, pk['component'], '--', alpha=0.7, label=f"Peak #{pk_i+1} @ {pk.get('center', 0):.1f}")

            ax1.set_ylabel("Intensity")
            ax1.set_title(f"Peak Fitting — {len(res.get('peaks', []))} Peaks Found ({model_type.title()})")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right', fontsize=8)

            # Subplot 2: Residual Fit Errors
            residuals = spectrum - res['fitted']
            ax2.plot(wl, residuals, 'b-', linewidth=1.5, label='Residual Error')
            ax2.axhline(0, color='gray', linestyle=':', alpha=0.7)
            ax2.set_xlabel("Wavelength / Band")
            ax2.set_ylabel("Residual")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right', fontsize=8)

            self.analysis_fig.tight_layout()
            self.analysis_canvas.draw_idle()

            if hasattr(self, 'plot_tabs'):
                self.plot_tabs.setCurrentIndex(1)
            self.statusBar().showMessage(f"Peak Fit Complete: {len(res.get('peaks', []))} peaks fitted.")
        except Exception as e:
            QMessageBox.critical(self, "Fit Error", str(e))

    def show_rgb_composite(self):
        if self.hypercube is None:
            return
        H, W, L = self.hypercube.shape
        if L < 3:
            return
        r_idx, ok1 = QInputDialog.getInt(self, "RGB Composite", "Red Band:", 0, 0, L - 1)
        if not ok1: return
        g_idx, ok2 = QInputDialog.getInt(self, "RGB Composite", "Green Band:", min(L//2, L-1), 0, L - 1)
        if not ok2: return
        b_idx, ok3 = QInputDialog.getInt(self, "RGB Composite", "Blue Band:", L - 1, 0, L - 1)
        if not ok3: return

        rgb = np.zeros((H, W, 3), dtype=float)
        for c, idx in enumerate([r_idx, g_idx, b_idx]):
            band = self.hypercube[:, :, idx]
            p2, p98 = np.percentile(band, (2, 98))
            rgb[:, :, c] = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)

        self.ax_main.clear()
        self.ax_main.imshow(rgb)
        self.ax_main.set_title(f"RGB Composite (R:{r_idx}, G:{g_idx}, B:{b_idx})")
        self.image_canvas.draw_idle()

    def show_pcrgb_composite(self):
        if self.hypercube is None:
            return
        H, W, L = self.hypercube.shape
        res = run_pca(self.hypercube, n_components=min(3, L-1))
        scores = res['concentrations']
        rgb = np.zeros((H, W, 3), dtype=float)
        for c in range(min(3, scores.shape[2])):
            band = scores[:, :, c]
            p2, p98 = np.percentile(band, (2, 98))
            rgb[:, :, c] = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)

        self.ax_main.clear()
        self.ax_main.imshow(rgb)
        self.ax_main.set_title("PC-RGB Composite (First 3 PCA Components)")
        self.image_canvas.draw_idle()

    # -----------------------------------------------------------------------
    # Helper: Apply Unmixing / Classification Results
    # -----------------------------------------------------------------------

    def _apply_unmixing_result(self, result, algo_name):
        self.concentrations = result['concentrations']
        self.basis_spectra = result.get('basis_spectra')
        self.r_squared = result.get('r_squared')
        self.residuals = result.get('residuals')
        self.unmixing_done = True

        self.progress_bar.setValue(100)
        mean_r2 = float(np.mean(self.r_squared)) if self.r_squared is not None else 0.0
        self.unmix_status.setText(f"{algo_name} Complete — Mean R²: {mean_r2:.4f}")
        self.unmix_status.setStyleSheet("color: green; font-weight: bold;")
        self.export_action.setEnabled(True)

        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        self.component_combo.addItem("Total Intensity / Concentration", 'total')
        if self.r_squared is not None:
            self.component_combo.addItem("R² Fit Map", 'r2')
        if self.residuals is not None:
            self.component_combo.addItem("Residual RMS Map", 'residual')

        K = self.concentrations.shape[2]
        if self.basis_spectra is not None and self.basis_spectra.shape[0] == K:
            if self.basis_labels is None or len(self.basis_labels) != K:
                labels = [f"{algo_name} Component {i+1}" for i in range(K)]
                self.basis_labels = labels
            else:
                labels = self.basis_labels
        else:
            labels = [f"{algo_name} Component {i+1}" for i in range(K)]
            self.basis_labels = labels

        for i in range(K):
            lbl = labels[i] if i < len(labels) else f"Component {i+1}"
            self.component_combo.addItem(lbl, i)

        self.component_combo.blockSignals(False)
        self._on_component_changed(0)

        # Automatically populate generated members into the Right List Box (Basis / Reference Spectra)
        if self.basis_spectra is not None and self.basis_spectra.shape[0] == K:
            self.basis_spectra_list = []
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            for i in range(K):
                lbl = labels[i] if i < len(labels) else f"{algo_name} Component {i+1}"
                sdata = SpectrumData(
                    self.basis_spectra[i], label=lbl, color=colors[i % len(colors)],
                    selection_type='basis', coords=None
                )
                self.basis_spectra_list.append(sdata)

            self.update_basis_checkbox_list()

        # Clear old analysis plots completely and render new component spectra
        self.analysis_fig.clear()
        mode = self.norm_mode_combo.currentData() if hasattr(self, 'norm_mode_combo') else 'raw'
        if self.basis_spectra is not None:
            ax = self.analysis_fig.add_subplot(111)
            wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, self.basis_spectra.shape[1] + 1, dtype=float)
            n_curves = min(15, self.basis_spectra.shape[0])
            for idx in range(n_curves):
                lbl = labels[idx] if idx < len(labels) else f"Component {idx+1}"
                spec = self._apply_spectral_normalization(self.basis_spectra[idx], mode)
                ax.plot(wl, spec, linewidth=2.0, label=lbl)

            ax.set_xlabel("Spectral Band / Wavelength")
            mode_titles = {
                'raw': "Component Amplitude / Endmember Intensity",
                'total': "Normalized Amplitude (spec / sum)",
                'peak': "Peak Normalized Amplitude (spec / max)",
                'l2': "L2 Vector Norm (spec / ||spec||)",
                'snv': "SNV Z-Score ((spec - μ) / σ)"
            }
            ax.set_ylabel(mode_titles.get(mode, "Component Amplitude"))
            ax.set_title(f"{algo_name} — Component Spectra [{mode_titles.get(mode, 'Original')}] (K={self.basis_spectra.shape[0]}) [Mean R²: {mean_r2:.4f}]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        self.analysis_fig.tight_layout()
        self.analysis_canvas.draw_idle()

        if hasattr(self, 'plot_tabs'):
            self.plot_tabs.setCurrentIndex(1)

    def _on_component_changed(self, index):
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

    def export_results(self):
        if self.concentrations is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            H, W, K = self.concentrations.shape
            with open(file_path, 'w') as f:
                header = ["Y", "X"] + [f"Comp_{k+1}" for k in range(K)] + ["R2", "Residual"]
                f.write(",".join(header) + "\n")
                for y in range(H):
                    for x in range(W):
                        comps = [str(self.concentrations[y, x, k]) for k in range(K)]
                        r2 = str(self.r_squared[y, x]) if self.r_squared is not None else ""
                        res = str(self.residuals[y, x]) if self.residuals is not None else ""
                        f.write(f"{y},{x}," + ",".join(comps) + f",{r2},{res}\n")
            self.statusBar().showMessage(f"Exported results to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export results: {e}")

    def show_about(self):
        """Show about / help dialog displaying Comprehensive Analysis Guide."""
        self.show_help()

    def show_help(self):
        """Show comprehensive interactive user guide and algorithm documentation window."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Hyperspectral Analysis & Unmixing Method Guide")
        dialog.resize(950, 750)
        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit(dialog)
        text_edit.setReadOnly(True)

        guide_html = r"""
        <h2>🔬 Hyperspectral Analysis & Unmixing Method Guide</h2>
        <p>This guide explains each numerical algorithm, what inputs are required, optimal use cases, and failure modes.</p>
        <hr>

        <h3>1. Linear & Supervised Unmixing (Requires Basis Library)</h3>
        
        <h4>🟢 Non-Negative Least Squares (NNLS)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + active Basis spectra library in the Right List Box (<code>Basis / Reference Spectra</code>).</li>
            <li><b>How it Works:</b> Solves \(\min_{C \ge 0} \|Y - C B\|_2^2\) for non-negative concentration maps \(C\).</li>
            <li><b>When it Works Well:</b> When all physical chemical components in the sample are present in the basis library and light mixing is linear.</li>
            <li><b>When it Fails:</b> Fails if an unknown component is missing from the basis library (forces other components to compensate, inflating residuals) or if basis spectra are highly collinear/identical.</li>
        </ul>

        <h4>🟢 Multiple Endmember Spectral Mixture Analysis (MESMA)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + reference endmember library (e.g., 4 to 20 reference spectra).</li>
            <li><b>How it Works:</b> Evaluates all subsets of endmembers (e.g. best 2, 3, or 4) for <i>each pixel independently</i> and selects the combination yielding the minimum residual error.</li>
            <li><b>When it Works Well:</b> Ideal for complex samples where not every pixel contains all endmembers (e.g., spatially heterogeneous samples).</li>
            <li><b>When it Fails:</b> Combinatorial explosion if library size \(K > 30\) (causes slowdown). Fails if endmembers lack distinct spectral features.</li>
        </ul>

        <h4>🟢 Spectral Angle Mapper (SAM)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + reference spectra library.</li>
            <li><b>How it Works:</b> Computes spectral angle \(\theta = \arccos\left(\frac{y \cdot b}{\|y\| \|b\|}\right)\) between pixel spectrum and basis vectors.</li>
            <li><b>When it Works Well:</b> Insensitive to overall illumination intensity variations, shading, and throughput fluctuations. Excellent for shape-based classification.</li>
            <li><b>When it Fails:</b> Fails when chemical discrimination relies on absolute intensity magnitude rather than spectral shape.</li>
        </ul>

        <h4>🟢 Spectral Information Divergence (SID)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + reference spectra library.</li>
            <li><b>How it Works:</b> Measures Kullback-Leibler information divergence between normalized spectral probability distributions.</li>
            <li><b>When it Works Well:</b> Highly sensitive to subtle spectral shape differences and slope changes.</li>
            <li><b>When it Fails:</b> Sensitive to zero/negative values (requires positive non-zero spectra).</li>
        </ul>

        <h4>🟢 Fully Constrained Least Squares (FCLS)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + active Basis spectra library.</li>
            <li><b>How it Works:</b> Solves NNLS with an additional sum-to-one constraint (\(\sum C_k = 1\)) per pixel via augmented matrix formulation (Heinz &amp; Chang 2001).</li>
            <li><b>When it Works Well:</b> Gold standard for physical abundance estimation — concentrations represent true fractional abundances summing to 100%.</li>
            <li><b>When it Fails:</b> Fails if some physical components are missing from basis (the sum-to-one constraint forces overestimation of remaining components).</li>
        </ul>

        <h4>🟢 RX Anomaly Detection</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube only (no reference spectra).</li>
            <li><b>How it Works:</b> Computes Mahalanobis distance of each pixel from the global spectral mean using the inverse covariance matrix.</li>
            <li><b>When it Works Well:</b> Detecting spectrally unusual pixels (contaminants, defects, rare materials) without prior knowledge of what to look for.</li>
            <li><b>When it Fails:</b> Fails when anomalies are frequent enough to dominate the covariance estimate.</li>
        </ul>
        <hr>

        <h3>2. Blind Source Separation & Decomposition (No Basis Needed)</h3>

        <h4>🔵 Multivariate Curve Resolution - Alternating Least Squares (MCR-ALS)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + component count \(K\) (Prompted from 2 to 10).</li>
            <li><b>How it Works:</b> Iteratively updates concentrations \(C\) and spectra \(S\) via alternating least squares with VCA seed initialization.</li>
            <li><b>When it Works Well:</b> Outstanding for completely blind samples where reference spectra are unknown. Enforces non-negativity and optional closure (\(\sum C_k = 1\)).</li>
            <li><b>When it Fails:</b> Requires distinct initial endmembers; can collapse into duplicate components if component count \(K\) is set too high for the dataset's true rank.</li>
        </ul>

        <h4>🔵 Non-Negative Matrix Factorization (NMF)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + component count \(K\).</li>
            <li><b>How it Works:</b> Multiplicative non-negative matrix update \(Y \approx W H\).</li>
            <li><b>When it Works Well:</b> Extracting additive, non-negative parts-based components.</li>
            <li><b>When it Fails:</b> Non-convex optimization; sensitive to initial random seed.</li>
        </ul>

        <h4>🔵 Vertex Component Analysis (VCA)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + endmember count \(K\).</li>
            <li><b>How it Works:</b> Simplex geometry projection to identify extreme vertices (pure endmembers).</li>
            <li><b>When it Works Well:</b> Fast and robust endmember extraction when pure or near-pure pixels exist in the scene.</li>
            <li><b>When it Fails:</b> Fails if the sample is extremely intimately mixed (no pure pixels present).</li>
        </ul>

        <h4>🔵 Principal Component Analysis (PCA)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + component count \(K\).</li>
            <li><b>How it Works:</b> Orthogonal SVD decomposition maximizing variance.</li>
            <li><b>When it Works Well:</b> Dimensionality reduction, decorrelation, and noise reduction.</li>
            <li><b>When it Fails:</b> Yields negative values and orthogonal axes that do <b>NOT</b> correspond to real physical spectra.</li>
        </ul>

        <h4>🔵 Minimum Noise Fraction (MNF)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + component count \(K\).</li>
            <li><b>How it Works:</b> Generalised eigenvalue decomposition ordering components by signal-to-noise ratio (SNR) rather than variance. Noise estimated from spatial first-differences.</li>
            <li><b>When it Works Well:</b> Superior to PCA for noisy hyperspectral data — first components have highest SNR. Standard preprocessing step in remote sensing.</li>
            <li><b>When it Fails:</b> Noise model assumes spatially uncorrelated noise; systematic spatial artifacts may violate this assumption.</li>
        </ul>

        <h4>🔵 Independent Component Analysis (ICA)</h4>
        <ul>
            <li><b>What it Needs:</b> Target image cube + component count \(K\).</li>
            <li><b>How it Works:</b> FastICA maximizing statistical independence / non-Gaussianity.</li>
            <li><b>When it Works Well:</b> Separating independent noise sources, artifacts, or distinct spatial patterns.</li>
            <li><b>When it Fails:</b> Does not enforce non-negativity; sign and order of components are arbitrary.</li>
        </ul>
        <hr>

        <h3>3. Hyperspectral Denoising Algorithms</h3>
        <ul>
            <li><b>3D Total Variation (3D-TV):</b> PDE primal gradient descent minimizing spatial-spectral Total Variation. Smooths noise while preserving sharp spatial edges and spectral transitions. Prompt asks for weight \(\lambda\) (e.g. 0.2 to 1.0).</li>
            <li><b>Savitzky-Golay:</b> 1D spectral polynomial smoothing. Preserves peak heights and spectral band positions.</li>
            <li><b>MPPCA:</b> Marchenko-Pastur PCA denoising based on random matrix theory. Automatically separates signal subspace from noise.</li>
            <li><b>BM4D / Wavelet:</b> Advanced 4D block-matching & 1D wavelet shrinkage filters.</li>
        </ul>
        <hr>

        <h3>4. Spectral Normalization & Standardisation Modes</h3>
        <ul>
            <li><b>Original (Raw):</b> Raw intensity detector counts.</li>
            <li><b>Total Intensity:</b> Normalized by total spectral area (\(I / \sum I\)). Eliminates overall thickness/concentration variations.</li>
            <li><b>Peak Intensity:</b> Normalized by maximum peak (\(I / I_{\max}\)). Rescales peak height to 1.0.</li>
            <li><b>L2 Vector Norm:</b> Unit vector length normalization (\(I / \|I\|\)).</li>
            <li><b>SNV Z-Score:</b> Standard Normal Variate (\((I - \mu) / \sigma\)). Corrects scattering baseline slopes and offsets.</li>
        </ul>
        """

        text_edit.setHtml(guide_html)
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close", dialog)
        close_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec_()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QDialog { font-size: 11px; }
        QDialog QLabel { font-size: 11px; }
        QDialog QLineEdit, QDialog QSpinBox, QDialog QDoubleSpinBox { font-size: 11px; min-height: 24px; padding: 2px; }
        QDialog QTabWidget::tab { font-size: 11px; padding: 4px 8px; }
        QDialog QPushButton { font-size: 11px; padding: 4px 10px; min-height: 24px; }
    """)
    window = UnmixerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()