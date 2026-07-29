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
    QFormLayout, QScrollArea, QAction
)
from PyQt5.QtCore import Qt, QTimer
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
    run_mcr_als, run_nmf, run_mesma,
    run_pca, run_sam, run_sid, run_svr,
    denoise_mppca, denoise_wavelet,
    run_nfindr, run_vca, fit_spectrum,
)


class SpectrumData:
    """Store spectrum data with metadata."""
    def __init__(self, spectrum, std=None, label="", color="blue", selection_type="point",
                 coords=None, mask=None):
        self.spectrum = np.array(spectrum, dtype=float)
        self.std = np.array(std, dtype=float) if std is not None else None
        self.label = label
        self.color = color
        self.selection_type = selection_type
        self.coords = coords
        self.mask = mask
        self.visible = True


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
        self.spectra_list = []         # list of SpectrumData objects
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
        tool_layout.addWidget(QLabel("Size:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 100)
        self.size_spin.setValue(10)
        self.size_spin.setSuffix(" px")
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

        # ==================== Right Panel: Spectrum Plot & Management ====================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)

        # Spectrum Plot Box (generous height)
        spectrum_group = QGroupBox("Spectrum Plot")
        spectrum_layout = QVBoxLayout(spectrum_group)
        spectrum_layout.setContentsMargins(4, 4, 4, 4)

        plot_toolbar_layout = QHBoxLayout()

        self.lock_axes_btn = QPushButton("🔒 Lock")
        self.lock_axes_btn.setToolTip("Lock axes (prevents auto-rescaling)")
        self.lock_axes_btn.setCheckable(True)
        self.lock_axes_btn.setChecked(True)
        self.lock_axes_btn.clicked.connect(self.toggle_lock_axes)
        self.lock_axes_btn.setFixedWidth(85)
        plot_toolbar_layout.addWidget(self.lock_axes_btn)

        self.autoscale_btn = QPushButton("⤡ Autoscale")
        self.autoscale_btn.setToolTip("Autoscale axes to fit all data")
        self.autoscale_btn.clicked.connect(self.autoscale_axes)
        self.autoscale_btn.setFixedWidth(95)
        plot_toolbar_layout.addWidget(self.autoscale_btn)

        self.zoom_btn = QPushButton("🔍 Zoom")
        self.zoom_btn.setToolTip("Zoom mode (mouse wheel to zoom in/out)")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.setChecked(False)
        self.zoom_btn.clicked.connect(self.toggle_zoom_mode)
        self.zoom_btn.setFixedWidth(85)
        plot_toolbar_layout.addWidget(self.zoom_btn)

        plot_toolbar_layout.addStretch()
        spectrum_layout.addLayout(plot_toolbar_layout)

        # Large Spectrum Canvas
        self.spectrum_fig = Figure(figsize=(7, 7), dpi=100)
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_canvas.mpl_connect('scroll_event', self.on_spectrum_scroll)

        self.ax_spectrum = self.spectrum_fig.add_subplot(111)
        self.ax_spectrum.set_xlabel('Spectral Band')
        self.ax_spectrum.set_ylabel('Intensity')
        self.ax_spectrum.set_title('Extracted Spectra')
        self.ax_spectrum.grid(True, alpha=0.3)

        spectrum_layout.addWidget(self.spectrum_canvas)
        right_layout.addWidget(spectrum_group, stretch=3)

        # Spectrum Management & Reference Spectra Group
        control_group = QGroupBox("Spectrum Management")
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(4, 4, 4, 4)

        btn_row1 = QHBoxLayout()
        self.add_btn = QPushButton("Add Current Selection")
        self.add_btn.clicked.connect(self.add_current_spectrum)
        btn_row1.addWidget(self.add_btn)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected_spectrum)
        btn_row1.addWidget(self.delete_btn)
        control_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.set_bg_btn = QPushButton("Set as Background")
        self.set_bg_btn.clicked.connect(self.set_background_spectrum)
        btn_row2.addWidget(self.set_bg_btn)

        self.subtract_bg_btn = QPushButton("Subtract Background")
        self.subtract_bg_btn.setCheckable(True)
        self.subtract_bg_btn.clicked.connect(self.toggle_background_subtraction)
        btn_row2.addWidget(self.subtract_bg_btn)
        control_layout.addLayout(btn_row2)

        self.bg_status_label = QLabel("Background: None")
        self.bg_status_label.setStyleSheet("color: gray; font-style: italic;")
        control_layout.addWidget(self.bg_status_label)

        self.export_btn = QPushButton("Export All Spectra")
        self.export_btn.clicked.connect(self._export_all_spectra)
        control_layout.addWidget(self.export_btn)

        control_layout.addSpacing(4)

        # Reference Spectra Buttons
        ref_group = QHBoxLayout()
        self.load_ref_btn = QPushButton("Load Reference Spectra")
        self.load_ref_btn.clicked.connect(self.load_reference_spectra)
        ref_group.addWidget(self.load_ref_btn)

        self.transform_btn = QPushButton("Transform Spectra")
        self.transform_btn.setEnabled(False)
        self.transform_btn.clicked.connect(self.transform_spectra)
        ref_group.addWidget(self.transform_btn)

        self.save_transform_btn = QPushButton("Save Transformed")
        self.save_transform_btn.setEnabled(False)
        self.save_transform_btn.clicked.connect(self.save_transformed_spectra)
        ref_group.addWidget(self.save_transform_btn)
        control_layout.addLayout(ref_group)

        self.ref_status_label = QLabel("Reference Spectra: None loaded")
        self.ref_status_label.setStyleSheet("color: gray; font-style: italic;")
        self.ref_status_label.setWordWrap(True)
        control_layout.addWidget(self.ref_status_label)

        # Scrollable table list for spectra
        self.spectrum_scroll = QScrollArea()
        self.spectrum_scroll.setWidgetResizable(True)
        self.spectrum_scroll.setMaximumHeight(160)
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_widget)
        self.checkbox_layout.setAlignment(Qt.AlignTop)
        self.spectrum_scroll.setWidget(self.checkbox_widget)

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("✓"), 0)
        header_layout.addWidget(QLabel("BG"), 0)
        header_layout.addWidget(QLabel("Spectrum"), 1)
        header_layout.addWidget(QLabel("Del"), 0)
        control_layout.addLayout(header_layout)
        control_layout.addWidget(self.spectrum_scroll)

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
        mppca_action = QAction("MPPCA (Multiplicative PCA)", self)
        mppca_action.triggered.connect(self.denoise_mppca)
        denoise_menu.addAction(mppca_action)

        wavelet_action = QAction("Wavelet (VisuShrink)", self)
        wavelet_action.triggered.connect(self.denoise_wavelet)
        denoise_menu.addAction(wavelet_action)

        # Unmixing
        unmix_menu = analysis_menu.addMenu("Unmixing")
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

        # Decomposition
        decomp_menu = analysis_menu.addMenu("Decomposition")
        ica_action = QAction("ICA (Independent Component Analysis)", self)
        ica_action.triggered.connect(self.run_ica)
        decomp_menu.addAction(ica_action)

        # Classification
        class_menu = analysis_menu.addMenu("Classification")
        pca_action = QAction("PCA (Principal Component Analysis)", self)
        pca_action.triggered.connect(self.run_pca)
        class_menu.addAction(pca_action)

        sam_action = QAction("SAM (Spectral Angle Mapper)", self)
        sam_action.triggered.connect(self.run_sam)
        class_menu.addAction(sam_action)

        sid_action = QAction("SID (Spectral Information Divergence)", self)
        sid_action.triggered.connect(self.run_sid)
        class_menu.addAction(sid_action)

        svr_action = QAction("SVR (Support Vector Regression)", self)
        svr_action.triggered.connect(self.run_svr)
        class_menu.addAction(svr_action)

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
        if interleave in ('bil', 'bsq'):
            data = data.reshape((bands, lines, samples)).transpose(1, 2, 0)
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

        # Determine color limits (vmin, vmax)
        if hasattr(self, 'clim') and self.clim is not None:
            vmin, vmax = self.clim
        else:
            frame_data = self.hypercube[:, :, frame_idx]
            vmin = float(np.nanmin(frame_data))
            vmax = float(np.nanmax(frame_data))
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
        im = self.ax_main.imshow(self.hypercube[:, :, frame_idx], cmap=self.current_colormap,
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
        self.export_action.setEnabled(False)

        self.display_frame(self.current_frame)
        self._plot_spectrum(self.hypercube[H // 2, W // 2, :])
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
                'radius': radius_val,
                'size': size_val,
                'spectrum': mean_spectrum,
                'std': std,
                'mask': mask,
                'n_pixels': int(mask.sum()),
            }
            self.update_spectrum_plot()
            self.add_btn.setEnabled(True)

        if hasattr(self, 'current_patch') and self.current_patch:
            try:
                self.current_patch.remove()
            except Exception:
                pass
            self.current_patch = None
        self.display_frame(self.current_frame)

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

        if self.axes_locked:
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

    def update_spectrum_plot(self):
        """Plot all saved spectra plus current active selection (matching HyperViewer)."""
        self.ax_spectrum.clear()
        sub_bg = self.subtract_bg_btn.isChecked() and self.background_spectrum is not None

        has_data = False
        # 1. Plot all saved/added spectra
        for sdata in self.spectra_list:
            if not sdata.visible:
                continue
            spec = sdata.spectrum - self.background_spectrum if sub_bg else sdata.spectrum
            wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, len(spec) + 1, dtype=float)
            self.ax_spectrum.plot(wl, spec, color=sdata.color, label=sdata.label, linewidth=1.5)

            if sdata.std is not None:
                self.ax_spectrum.fill_between(
                    wl, spec - sdata.std, spec + sdata.std,
                    alpha=0.2, color=sdata.color)
            has_data = True

        # 2. Plot active selection on top as 'Current' (black dashed line)
        if self.current_selection is not None and 'spectrum' in self.current_selection:
            spec = self.current_selection['spectrum']
            if sub_bg and self.background_spectrum is not None:
                spec = spec - self.background_spectrum
            wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, len(spec) + 1, dtype=float)
            self.ax_spectrum.plot(wl, spec, 'k--', linewidth=1.5, label='Current')

            if self.current_selection.get('std') is not None:
                std = self.current_selection['std']
                self.ax_spectrum.fill_between(
                    wl, spec - std, spec + std,
                    alpha=0.25, color='black')
            has_data = True

        self.ax_spectrum.set_xlabel("Wavelength / Band")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.set_title("Extracted Spectra")

        if has_data:
            self.ax_spectrum.legend(loc='upper right', fontsize=8)

        self.ax_spectrum.grid(True, alpha=0.3)

        if self.axes_locked:
            if self.saved_xlim is None or self.saved_ylim is None:
                self.saved_xlim = self.ax_spectrum.get_xlim()
                self.saved_ylim = self.ax_spectrum.get_ylim()
            else:
                self.ax_spectrum.set_xlim(self.saved_xlim)
                self.ax_spectrum.set_ylim(self.saved_ylim)

        self.spectrum_canvas.draw_idle()

    def _plot_spectrum(self, spectrum=None):
        self.update_spectrum_plot()

    def _plot_residual(self, residual=None):
        self.update_spectrum_plot()

    def add_current_spectrum(self):
        if self.current_selection is None or 'spectrum' not in self.current_selection:
            return
        sel = self.current_selection
        spec = sel['spectrum']
        std = sel.get('std', None)
        stype = sel.get('type', 'point')
        lbl = f"Spectrum {len(self.spectra_list) + 1} ({stype})"
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
        # Clear existing list
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

            radio = QRadioButton()
            self.bg_radio_group.addButton(radio, idx)
            row.addWidget(radio)

            lbl = QLabel(sdata.label)
            lbl.setStyleSheet(f"color: {sdata.color};")
            row.addWidget(lbl)
            row.addStretch()

            del_btn = QPushButton("×")
            del_btn.setFixedSize(20, 20)
            del_btn.clicked.connect(lambda checked, i=idx: self.delete_single_spectrum(i))
            row.addWidget(del_btn)

            self.checkbox_layout.addLayout(row)

        self.delete_btn.setEnabled(len(self.spectra_list) > 0)
        self.set_bg_btn.setEnabled(len(self.spectra_list) > 0)

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
            self.bg_status_label.setText(f"Background: {self.spectra_list[bg_id].label}")
            self.bg_status_label.setStyleSheet("color: green; font-weight: bold;")

    def toggle_background_subtraction(self):
        if self.background_spectrum is None:
            self.subtract_bg_btn.setChecked(False)
            return
        self._plot_all_spectra()

    def _plot_all_spectra(self):
        self.update_spectrum_plot()

    def _export_all_spectra(self):
        if not self.spectra_list:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Spectra", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            first_len = len(self.spectra_list[0].spectrum)
            wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, first_len + 1, dtype=float)
            with open(file_path, 'w') as f:
                f.write("Wavelength," + ",".join(s.label for s in self.spectra_list) + "\n")
                for i, w in enumerate(wl):
                    row = [str(w)] + [str(s.spectrum[i]) if i < len(s.spectrum) else "" for s in self.spectra_list]
                    f.write(",".join(row) + "\n")
            self.statusBar().showMessage(f"Exported spectra to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

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

            # Set as current basis spectra for unmixing & classification
            self.basis_spectra = spectra
            self.basis_labels = labels
            self.basis_wavelengths = wavelengths

            n_ref = len(labels)
            wl_range = f"{wavelengths[0]:.1f}–{wavelengths[-1]:.1f}"
            self.ref_status_label.setText(f"Reference Spectra: {n_ref} loaded (λ: {wl_range})")
            self.ref_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.transform_btn.setEnabled(True)

            self._show_reference_window()
            QMessageBox.information(self, "Reference Spectra Loaded",
                                   f"Loaded {n_ref} reference spectra as calibration basis:\n" +
                                   "\n".join(f"  • {lbl}" for lbl in labels[:10]))
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
        self.progress_bar.setValue(10)
        self.unmix_status.setText("MCR-ALS in progress…")
        try:
            result = run_mcr_als(self.hypercube, n_components=min(5, L-1), max_iter=100, non_neg=True)
            self._apply_unmixing_result(result, "MCR-ALS")
        except Exception as e:
            QMessageBox.critical(self, "MCR-ALS Error", f"MCR-ALS failed:\n{e}")

    def run_nmf(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        self.progress_bar.setValue(10)
        self.unmix_status.setText("NMF in progress…")
        try:
            result = run_nmf(self.hypercube, n_components=min(5, L-1), max_iter=200)
            self._apply_unmixing_result(result, "NMF")
        except Exception as e:
            QMessageBox.critical(self, "NMF Error", f"NMF failed:\n{e}")

    def run_mesma(self):
        if self.hypercube is None or self.basis_spectra is None:
            QMessageBox.warning(self, "No Basis Data", "Load reference spectra or extract endmembers first.")
            return
        self.progress_bar.setValue(10)
        self.unmix_status.setText("MESMA in progress…")
        try:
            result = run_mesma(self.hypercube, self.basis_spectra)
            self._apply_unmixing_result(result, "MESMA")
        except Exception as e:
            QMessageBox.critical(self, "MESMA Error", f"MESMA failed:\n{e}")

    def run_pca(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        try:
            result = run_pca(self.hypercube, n_components=min(5, L-1))
            self._apply_unmixing_result(result, "PCA")
        except Exception as e:
            QMessageBox.critical(self, "PCA Error", f"PCA failed:\n{e}")

    def run_sam(self):
        if self.hypercube is None or self.basis_spectra is None:
            QMessageBox.warning(self, "No Basis Data", "Load reference spectra or extract endmembers first.")
            return
        try:
            result = run_sam(self.hypercube, self.basis_spectra)
            self._apply_unmixing_result(result, "SAM Classification")
        except Exception as e:
            QMessageBox.critical(self, "SAM Error", f"SAM failed:\n{e}")

    def run_sid(self):
        if self.hypercube is None or self.basis_spectra is None:
            QMessageBox.warning(self, "No Basis Data", "Load reference spectra or extract endmembers first.")
            return
        try:
            result = run_sid(self.hypercube, self.basis_spectra)
            self._apply_unmixing_result(result, "SID Classification")
        except Exception as e:
            QMessageBox.critical(self, "SID Error", f"SID failed:\n{e}")

    def run_svr(self):
        if self.hypercube is None or self.basis_spectra is None:
            QMessageBox.warning(self, "No Basis Data", "Load reference spectra or extract endmembers first.")
            return
        self.progress_bar.setValue(10)
        self.unmix_status.setText("Training SVR models…")
        try:
            result = run_svr(self.hypercube, self.basis_spectra, n_train=1000)
            self._apply_unmixing_result(result, "SVR Unmixing")
        except Exception as e:
            QMessageBox.critical(self, "SVR Error", f"SVR failed:\n{e}")

    def run_ica(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        try:
            from sklearn.decomposition import FastICA
            data = self.hypercube.reshape(-1, L).astype(float)
            ica = FastICA(n_components=min(5, L-1), random_state=42)
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
            self._apply_unmixing_result(result, "ICA Decomposition")
        except Exception as e:
            QMessageBox.critical(self, "ICA Error", f"ICA failed:\n{e}")

    def run_nfindr_dialog(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n_end, ok = QInputDialog.getInt(self, "N-FINDR", "Number of endmembers:", 3, 2, min(10, L-1))
        if ok:
            try:
                res = run_nfindr(self.hypercube, n_endmembers=n_end)
                self.basis_spectra = res['endmembers']
                self.basis_labels = [f"Endmember {i+1}" for i in range(n_end)]
                self._show_reference_window()
                QMessageBox.information(self, "N-FINDR Success", f"Extracted {n_end} endmembers via N-FINDR.")
            except Exception as e:
                QMessageBox.critical(self, "N-FINDR Error", str(e))

    def run_vca_dialog(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data", "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n_end, ok = QInputDialog.getInt(self, "VCA", "Number of endmembers:", 3, 2, min(10, L-1))
        if ok:
            try:
                res = run_vca(self.hypercube, n_endmembers=n_end)
                self.basis_spectra = res['endmembers']
                self.basis_labels = [f"Endmember {i+1}" for i in range(n_end)]
                self._show_reference_window()
                QMessageBox.information(self, "VCA Success", f"Extracted {n_end} endmembers via VCA.")
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
            fig, ax = plt.subplots(figsize=(8, 4))
            wl = self.basis_wavelengths if self.basis_wavelengths is not None else np.arange(1, len(spectrum) + 1, dtype=float)
            ax.plot(wl, spectrum, 'k.', label='Measured')
            ax.plot(wl, res['fitted'], 'r-', label=f"{model_type.title()} Fit")
            ax.set_xlabel("Wavelength / Band")
            ax.set_ylabel("Intensity")
            ax.set_title(f"Peak Fitting — {len(res['peaks'])} Peaks Found (χ²={res['info']['chi2']:.4e})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.show()
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
        self.basis_spectra = result.get('basis_spectra', self.basis_spectra)
        self.r_squared = result.get('r_squared')
        self.residuals = result.get('residuals')
        self.unmixing_done = True

        self.progress_bar.setValue(100)
        mean_r2 = float(np.mean(self.r_squared)) if self.r_squared is not None else 0.0
        self.unmix_status.setText(f"{algo_name} Complete — Mean R²: {mean_r2:.4f}")
        self.unmix_status.setStyleSheet("color: green; font-weight: bold;")
        self.export_action.setEnabled(True)

        self.component_combo.clear()
        self.component_combo.addItem("Total Concentration", 'total')
        if self.r_squared is not None:
            self.component_combo.addItem("R² Map", 'r2')
        if self.residuals is not None:
            self.component_combo.addItem("Residual RMS", 'residual')

        K = self.concentrations.shape[2]
        labels = self.basis_labels or [f"Component {i+1}" for i in range(K)]
        for i in range(K):
            lbl = labels[i] if i < len(labels) else f"Component {i+1}"
            self.component_combo.addItem(lbl, i)

        self._on_component_changed(0)

    def _on_component_changed(self, index):
        if self.concentrations is None:
            return
        data_data = self.component_combo.currentData()
        if data_data == 'total':
            disp = self.concentrations.sum(axis=2)
            title = "Total Concentration"
        elif data_data == 'r2':
            disp = self.r_squared
            title = "R² Map"
        elif data_data == 'residual':
            disp = self.residuals
            title = "Residual RMS Map"
        elif isinstance(data_data, int):
            disp = self.concentrations[:, :, data_data]
            lbl = self.basis_labels[data_data] if self.basis_labels and data_data < len(self.basis_labels) else f"Comp {data_data+1}"
            title = f"Abundance: {lbl}"
        else:
            disp = self.concentrations[:, :, 0]
            title = "Component 1"

        H, W = disp.shape
        self.ax_main.clear()
        im = self.ax_main.imshow(disp, cmap=self.current_colormap)
        self.ax_main.set_title(title)
        self.image_canvas.draw_idle()

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
        """Show about / help dialog displaying README.md content."""
        self.show_help()

    def show_help(self):
        """Show user guide and documentation from README.md in a scrollable dialog."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md"),
            os.path.join(os.getcwd(), "README.md"),
        ]
        readme_path = None
        for p in possible_paths:
            if os.path.exists(p):
                readme_path = p
                break

        content = ""
        if readme_path:
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                content = f"Error reading README.md: {e}"
        else:
            content = "README.md file not found."

        dialog = QDialog(self)
        dialog.setWindowTitle("User Guide & Documentation - README.md")
        dialog.resize(850, 650)
        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit(dialog)
        text_edit.setReadOnly(True)
        if hasattr(text_edit, 'setMarkdown'):
            text_edit.setMarkdown(content)
        else:
            text_edit.setPlainText(content)

        layout.addWidget(text_edit)
        close_btn = QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec_()


def main():
    app = QApplication(sys.argv)
    window = UnmixerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()