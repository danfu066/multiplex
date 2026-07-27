"""
Hyperspectral Unmixing GUI
===========================
Interactive tool for hyperspectral image unmixing, denoising, classification,
endmember extraction, spectral fitting, and visualization.

Algorithms are delegated to the ``unmix`` package — this file is a thin
PyQt5 UI layer with dialogs, progress feedback, and Matplotlib canvases.

Usage
-----
    python Unmixer.py

    Open File → Open ENVI / Open NPY to load a hypercube, then use the
    Analysis menu for unmixing, denoising, classification, endmember
    extraction, and spectral fitting.

Author  : Multiplexing Lab, University of Washington
Date    : 2026-07-26
Version : 2.0
"""

import sys
import os
# Allow running from inside the unmix/ package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFileDialog, QMessageBox, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton,
    QButtonGroup, QGroupBox, QProgressBar, QInputDialog, QDialog,
    QFormLayout, QComboBox, QPushButton as QDialogButton,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QAction
from PyQt5.QtGui import QKeySequence
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Circle
from scipy import ndimage

from unmix import (
    run_mcr_als, run_nmf, run_mesma,
    run_pca, run_sam, run_sid, run_svr,
    denoise_mppca, denoise_wavelet,
    run_nfindr, run_vca, fit_spectrum,
)


# ===================================================================
# Main Window
# ===================================================================

class UnmixerWindow(QMainWindow):
    """Hyperspectral unmixing main window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperspectral Unmixer v2.0")
        self.resize(1400, 900)

        # ---- Data state ----
        self.hypercube = None          # (H, W, L) ndarray
        self.metadata = {}             # wavelength, spatial info, etc.
        self.basis_spectra = None      # (N, L) calibration / endmember spectra
        self.basis_labels = None       # list[str]
        self.basis_wavelengths = None  # (L,) wavelengths
        self.concentrations = None     # (H, W, N) unmixing result
        self.r_squared = None          # (H*W,) per-pixel R²
        self.residuals = None          # (H*W,) per-pixel residual RMS
        self.unmixing_done = False

        # ---- Selection state ----
        self.current_selection = None  # dict with 'spectrum', 'pixels', etc.
        self.selection_tool = None     # 'point' | 'circle' | 'square' | None
        self._temp_rect = None         # Rectangle patch for drag
        self._temp_circle = None       # Circle patch for drag
        self._drag_start = None        # (x, y) in data coords

        # ---- Endmember picker state ----
        self.endmember_mode = False
        self.picked_endmembers = []
        self.picked_endmember_patches = []

        # ---- RGB composite state ----
        self.rgb_mode = False
        self.rgb_image = None

        # ---- Spectrum management state ----
        self.extracted_spectra = []    # list of (spectrum, label) tuples
        self.background_spectrum = None

        # ---- UI references (set by _build_ui) ----
        self.image_canvas = None
        self.spectrum_canvas = None
        self.ax_main = None
        self.ax_x_avg = None
        self.ax_y_avg = None
        self.ax_spectrum = None
        self.ax_cbar = None
        self.im_handle = None
        self.norm = None
        self.current_band = 0
        self.colormap = 'gray'
        self.selection_size = 10
        self.progress_bar = None
        self.unmix_status = None
        self.export_action = None

        self._build_ui()
        self._build_menu()
        self.statusBar().showMessage("Ready")

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        """Build the main layout with splitter, canvases, and controls."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # --- Left: selection toolbar + spatial view ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Selection toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(2, 2, 2, 2)

        self.tool_point = QPushButton("Point")
        self.tool_point.setCheckable(True)
        self.tool_point.clicked.connect(lambda: self._set_selection_tool('point'))
        toolbar.addWidget(self.tool_point)

        self.tool_circle = QPushButton("Circle")
        self.tool_circle.setCheckable(True)
        self.tool_circle.clicked.connect(lambda: self._set_selection_tool('circle'))
        toolbar.addWidget(self.tool_circle)

        self.tool_square = QPushButton("Square")
        self.tool_square.setCheckable(True)
        self.tool_square.clicked.connect(lambda: self._set_selection_tool('square'))
        toolbar.addWidget(self.tool_square)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_selection)
        toolbar.addWidget(self.clear_btn)

        toolbar.addSpacing(10)

        toolbar.addWidget(QLabel("Size:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 100)
        self.size_spin.setValue(10)
        self.size_spin.setSuffix(" px")
        toolbar.addWidget(self.size_spin)

        toolbar.addSpacing(10)

        toolbar.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])
        self.colormap_combo.setCurrentText('gray')
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        toolbar.addWidget(self.colormap_combo)

        toolbar.addStretch()
        left_layout.addLayout(toolbar)

        # Spatial view with linked marginals
        self.fig_main = Figure(figsize=(7, 6), dpi=100)
        self.image_canvas = FigureCanvas(self.fig_main)

        # GridSpec: 3x3 for marginals
        self.gs_main = self.fig_main.add_gridspec(
            3, 3,
            left=0.10, right=0.92, top=0.92, bottom=0.08,
            width_ratios=[0.08, 1.0, 0.03],
            height_ratios=[0.08, 1.0, 0.0],
            hspace=0.15, wspace=0.15)

        # X spectral average (top marginal)
        self.ax_x_avg = self.fig_main.add_subplot(self.gs_main[0, 1])
        self.ax_x_avg.set_title("X Spectral Average")
        self.ax_x_avg.set_xticks([])
        self.ax_x_avg.set_yticks([])

        # Y spectral average (left marginal)
        self.ax_y_avg = self.fig_main.add_subplot(self.gs_main[1, 0])
        self.ax_y_avg.set_ylabel("Y Spectral Average", rotation=0, labelpad=10)
        self.ax_y_avg.yaxis.set_label_position("left")
        self.ax_y_avg.set_xticks([])

        # Main spatial view
        self.ax_main = self.fig_main.add_subplot(self.gs_main[1, 1])
        self.ax_main.set_title("Spatial View")
        self.ax_main.set_xlabel("X (pixels)")
        self.ax_main.set_ylabel("Y (pixels)")
        self.ax_main.set_aspect("equal")

        # Colorbar axis
        self.ax_cbar = self.fig_main.add_subplot(self.gs_main[1, 2])

        left_layout.addWidget(self.image_canvas)

        # --- Right: spectrum plot + spectrum management ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Spectrum plot with toolbar
        spec_toolbar = QHBoxLayout()
        spec_toolbar.setContentsMargins(2, 2, 2, 2)

        self.spec_lock_btn = QPushButton("Lock")
        self.spec_lock_btn.setCheckable(True)
        spec_toolbar.addWidget(self.spec_lock_btn)

        self.spec_autoscale_btn = QPushButton("Autoscale")
        self.spec_autoscale_btn.clicked.connect(self._autoscale_spectrum)
        spec_toolbar.addWidget(self.spec_autoscale_btn)

        self.spec_zoom_btn = QPushButton("Zoom")
        self.spec_zoom_btn.setCheckable(True)
        spec_toolbar.addWidget(self.spec_zoom_btn)

        spec_toolbar.addStretch()
        right_layout.addLayout(spec_toolbar)

        # Spectrum canvas
        self.fig_spec = Figure(figsize=(4, 3), dpi=100)
        self.spectrum_canvas = FigureCanvas(self.fig_spec)
        gs_spec = self.fig_spec.add_gridspec(
            1, 1, left=0.15, right=0.92, top=0.88, bottom=0.12)
        self.ax_spectrum = self.fig_spec.add_subplot(gs_spec[0, 0])
        self.ax_spectrum.set_title("Extracted Spectra")
        self.ax_spectrum.set_xlabel("Spectral Band")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.grid(True, alpha=0.3)
        right_layout.addWidget(self.spectrum_canvas)

        # Spectrum management panel
        mgmt_group = QGroupBox("Spectrum Management")
        mgmt_layout = QVBoxLayout(mgmt_group)

        self.add_spec_btn = QPushButton("Add Current Selection")
        self.add_spec_btn.clicked.connect(self._add_current_spectrum)
        mgmt_layout.addWidget(self.add_spec_btn)

        self.del_spec_btn = QPushButton("Delete Selected")
        self.del_spec_btn.clicked.connect(self._delete_selected_spectrum)
        mgmt_layout.addWidget(self.del_spec_btn)

        self.set_bg_btn = QPushButton("Set as Background")
        self.set_bg_btn.clicked.connect(self._set_background)
        mgmt_layout.addWidget(self.set_bg_btn)

        self.sub_bg_btn = QPushButton("Subtract Background")
        self.sub_bg_btn.clicked.connect(self._subtract_background)
        mgmt_layout.addWidget(self.sub_bg_btn)

        self.bg_status = QLabel("Background: None")
        self.bg_status.setStyleSheet("color: gray;")
        mgmt_layout.addWidget(self.bg_status)

        mgmt_layout.addStretch()

        self.export_all_btn = QPushButton("Export All Spectra")
        self.export_all_btn.clicked.connect(self._export_all_spectra)
        mgmt_layout.addWidget(self.export_all_btn)

        right_layout.addWidget(mgmt_group)

        # Progress + status (hidden, used by algorithm runners)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        self.unmix_status = QLabel("")
        self.unmix_status.setWordWrap(True)
        right_layout.addWidget(self.unmix_status)

        # --- Splitter ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Canvas interaction        # Canvas interaction
        self.image_canvas.mpl_connect('button_press_event', self._on_canvas_press)
        self.image_canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.image_canvas.mpl_connect('button_release_event', self._on_canvas_release)
        self.image_canvas.mpl_connect('key_press_event', self._on_canvas_key)

    # -----------------------------------------------------------------------
    # Menu bar
    # -----------------------------------------------------------------------

    def _build_menu(self):
        """Build the menu bar with all analysis, file, and view actions."""
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

        # — Denoise submenu —
        denoise_menu = analysis_menu.addMenu("Denoise")

        mppca_action = QAction("MPPCA", self)
        mppca_action.setToolTip("Multiplicative PCA denoising")
        mppca_action.triggered.connect(self.denoise_mppca)
        denoise_menu.addAction(mppca_action)

        wavelet_action = QAction("Wavelet", self)
        wavelet_action.setToolTip("Wavelet denoising (VisuShrink)")
        wavelet_action.triggered.connect(self.denoise_wavelet)
        denoise_menu.addAction(wavelet_action)

        analysis_menu.addSeparator()

        # — Unmixing submenu —
        unmix_menu = analysis_menu.addMenu("Unmixing")

        mcr_action = QAction("MCR-ALS Decomposition", self)
        mcr_action.setShortcut("Ctrl+M")
        mcr_action.setToolTip("Multivariate Curve Resolution - Alternating Least Squares")
        mcr_action.triggered.connect(self.run_mcr_als)
        unmix_menu.addAction(mcr_action)

        nmf_action = QAction("NMF", self)
        nmf_action.setShortcut("Ctrl+N")
        nmf_action.setToolTip("Non-Negative Matrix Factorization")
        nmf_action.triggered.connect(self.run_nmf)
        unmix_menu.addAction(nmf_action)

        mesma_action = QAction("MESMA", self)
        mesma_action.setToolTip("Multiple Endmember Spectral Mixture Analysis")
        mesma_action.triggered.connect(self.run_mesma)
        unmix_menu.addAction(mesma_action)

        analysis_menu.addSeparator()

        # — Decomposition submenu —
        decomp_menu = analysis_menu.addMenu("Decomposition")

        ica_action = QAction("ICA", self)
        ica_action.setToolTip("Independent Component Analysis")
        ica_action.triggered.connect(self.run_ica)
        decomp_menu.addAction(ica_action)

        analysis_menu.addSeparator()

        # — Classification submenu —
        class_menu = analysis_menu.addMenu("Classification")

        pca_action = QAction("PCA", self)
        pca_action.setShortcut("Ctrl+P")
        pca_action.setToolTip("Principal Component Analysis")
        pca_action.triggered.connect(self.run_pca)
        class_menu.addAction(pca_action)

        svr_action = QAction("SVR", self)
        svr_action.setToolTip("Support Vector Regression unmixing")
        svr_action.triggered.connect(self.run_svr)
        class_menu.addAction(svr_action)

        sam_action = QAction("SAM", self)
        sam_action.setToolTip("Spectral Angle Mapper classification")
        sam_action.triggered.connect(self.run_sam)
        class_menu.addAction(sam_action)

        sid_action = QAction("SID", self)
        sid_action.setToolTip("Spectral Information Divergence classification")
        sid_action.triggered.connect(self.run_sid)
        class_menu.addAction(sid_action)

        # — Endmember extraction submenu —
        em_menu = analysis_menu.addMenu("Extract Endmembers")

        nfindr_action = QAction("N-FINDR", self)
        nfindr_action.setShortcut("Ctrl+F")
        nfindr_action.setToolTip(
            "N-FINDR — maximum volume simplex endmember extraction\n"
            "Automatically finds pure-pixel endmembers")
        nfindr_action.triggered.connect(self.run_nfindr_dialog)
        em_menu.addAction(nfindr_action)

        vca_action = QAction("VCA (Vertex Component Analysis)", self)
        vca_action.setShortcut("Ctrl+V")
        vca_action.setToolTip(
            "VCA — fast NIPALS-style vertex component analysis\n"
            "Orthogonal projection to find simplex vertices")
        vca_action.triggered.connect(self.run_vca_dialog)
        em_menu.addAction(vca_action)

        picker_action = QAction("Interactive Picker", self)
        picker_action.setShortcut("Ctrl+I")
        picker_action.setToolTip(
            "Click on the spatial view to pick endmembers manually\n"
            "Toggle off with same shortcut or Clear button")
        picker_action.triggered.connect(self.toggle_endmember_picker)
        em_menu.addAction(picker_action)

        apply_picked_action = QAction("Apply Picked Endmembers", self)
        apply_picked_action.setToolTip(
            "Use picked pixels as endmember basis for unmixing")
        apply_picked_action.triggered.connect(self._apply_picked_endmembers)
        em_menu.addAction(apply_picked_action)

        analysis_menu.addSeparator()

        # — Spectral fitting submenu —
        fit_menu = analysis_menu.addMenu("Fit Peaks")

        gauss_action = QAction("Gaussian Fit", self)
        gauss_action.setShortcut("Ctrl+G")
        gauss_action.setToolTip(
            "Fit Gaussian peaks to the selected pixel/ROI spectrum\n"
            "Requires a point or region selection first")
        gauss_action.triggered.connect(lambda: self.run_fit_spectrum('gaussian'))
        fit_menu.addAction(gauss_action)

        lorentz_action = QAction("Lorentzian Fit", self)
        lorentz_action.setShortcut("Ctrl+L")
        lorentz_action.setToolTip(
            "Fit Lorentzian peaks to the selected pixel/ROI spectrum\n"
            "Requires a point or region selection first")
        lorentz_action.triggered.connect(lambda: self.run_fit_spectrum('lorentzian'))
        fit_menu.addAction(lorentz_action)

        analysis_menu.addSeparator()

        # — Visualization submenu —
        viz_menu = analysis_menu.addMenu("Visualization")

        rgb_action = QAction("RGB Composite", self)
        rgb_action.setToolTip(
            "Select 3 bands to create a false-color RGB image")
        rgb_action.triggered.connect(self.show_rgb_composite)
        viz_menu.addAction(rgb_action)

        pcrgb_action = QAction("PC-RGB (First 3 PCA Components)", self)
        pcrgb_action.setToolTip(
            "Use first 3 PCA components as RGB channels\n"
            "Fast false-color visualization of dominant variance")
        pcrgb_action.triggered.connect(self.show_pcrgb_composite)
        viz_menu.addAction(pcrgb_action)

        # ---- Help menu ----
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # -----------------------------------------------------------------------
    # File I/O
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
            QMessageBox.critical(self, "ENVI Load Error",
                                 f"Failed to load ENVI file:\n{e}")

    def _load_envi(self, hdr_path):
        """Parse ENVI .hdr and load associated .dat as (H, W, L) cube."""
        import struct

        # Parse header
        params = {}
        with open(hdr_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    params[key.strip().lower()] = val.strip().strip('{}')

        # Determine file format
        interleave = params.get('interleave', 'bil').lower()
        samples = int(params['samples'])
        lines = int(params['lines'])
        bands = int(params['bands'])
        data_type_str = params.get('data type', '4')
        data_type_map = {
            '1': np.int16, '2': np.float32, '3': np.int32,
            '4': np.float64, '5': np.complex64, '6': np.complex128,
            '9': np.uint16, '12': np.uint32, '13': np.uint8,
            '14': np.int8,
        }
        dtype = data_type_map.get(data_type_str, np.float64)
        interleave = interleave[:3].lower()  # 'bil', 'bsq', 'bip'

        # Find data file
        if 'header offset' in params:
            dat_path = hdr_path.replace('.hdr', '.dat')
        else:
            dat_path = hdr_path.replace('.hdr', '.img')
        if not os.path.exists(dat_path):
            dat_path = hdr_path + '.dat'
        if not os.path.exists(dat_path):
            dat_path = hdr_path + '.img'

        # Read wavelength info
        wavelengths = None
        if 'wavelength' in params:
            wl_str = params['wavelength'].strip('{}')
            wavelengths = np.array([float(x) for x in wl_str.split(',')])

        # Load data
        if interleave == 'bil':
            data = np.fromfile(dat_path, dtype=dtype)
            data = data.reshape((bands, lines, samples)).transpose(1, 2, 0)
        elif interleave == 'bsq':
            data = np.fromfile(dat_path, dtype=dtype)
            data = data.reshape((bands, lines, samples)).transpose(1, 2, 0)
        elif interleave == 'bip':
            data = np.fromfile(dat_path, dtype=dtype)
            data = data.reshape((lines, samples, bands))
        else:
            raise ValueError(f"Unknown interleave: {interleave}")

        self.hypercube = data.astype(np.float64)
        self.basis_wavelengths = wavelengths
        self.metadata = params
        self._on_data_loaded()
        self.statusBar().showMessage(f"Loaded ENVI: {lines}x{samples}x{bands}")

    def open_npy(self):
        """Load a .npy hyperspectral cube."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open NPY", "", "NPY Files (*.npy)")
        if not file_path:
            return
        try:
            self.hypercube = np.load(file_path).astype(np.float64)
            self._on_data_loaded()
            H, W, L = self.hypercube.shape
            self.statusBar().showMessage(f"Loaded NPY: {H}x{W}x{L}")
        except Exception as e:
            QMessageBox.critical(self, "NPY Load Error",
                                 f"Failed to load .npy file:\n{e}")

    def open_tiff(self):
        """Load a multipage TIFF stack hyperspectral image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF Stack", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
        if not file_path:
            return
        try:
            self._load_tiff(file_path)
        except Exception as e:
            QMessageBox.critical(self, "TIFF Load Error",
                                 f"Failed to load TIFF stack:\n{e}")

    def _load_tiff(self, file_path):
        """Read multipage TIFF stack and format as (H, W, L) hypercube."""
        try:
            import tifffile
            data = tifffile.imread(file_path)
        except ImportError:
            try:
                from skimage import io
                data = io.imread(file_path)
            except ImportError:
                raise ImportError(
                    "Install tifffile for TIFF support: pip install tifffile")

        data = np.array(data, dtype=np.float64)

        if data.ndim == 3:
            # tifffile.imread returns (pages/bands, H, W) for multi-page TIFF stacks.
            # Transpose to (H, W, bands) for Unmixer hypercube convention.
            data = np.moveaxis(data, 0, -1)
        elif data.ndim == 2:
            data = data[:, :, np.newaxis]
        else:
            raise ValueError(f"Unsupported array shape in TIFF: {data.shape}")

        self.hypercube = data
        self._on_data_loaded()
        H, W, L = self.hypercube.shape
        self.statusBar().showMessage(f"Loaded TIFF: {H}x{W}x{L}")

    def open_csv_basis(self):
        """Load a CSV spectrum file as calibration basis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV Spectrum", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        try:
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            if data.ndim == 1:
                self.basis_spectra = data.reshape(1, -1)
                self.basis_labels = ["Spectrum 1"]
            elif data.ndim == 2:
                if data.shape[0] < data.shape[1]:
                    self.basis_spectra = data.T  # (N, L)
                else:
                    self.basis_spectra = data    # (N, L)
                self.basis_labels = [f"Spectrum {i+1}"
                                     for i in range(self.basis_spectra.shape[0])]
            else:
                raise ValueError("CSV must be 1D or 2D")

            # Check if first column is wavelength
            if self.basis_spectra.shape[1] >= 2:
                # Try to detect wavelength column
                self.basis_wavelengths = np.arange(1, self.basis_spectra.shape[1] + 1, dtype=float)

            self._plot_basis_spectra()
            self._update_run_button()
            self.statusBar().showMessage(
                f"Loaded basis: {self.basis_spectra.shape[0]} spectra, "
                f"{self.basis_spectra.shape[1]} bands")
        except Exception as e:
            QMessageBox.critical(self, "CSV Load Error",
                                 f"Failed to load CSV:\n{e}")

    def export_results(self):
        """Export unmixing results to CSV."""
        if not self.unmixing_done or self.concentrations is None:
            QMessageBox.information(self, "No Results",
                                    "Run an unmixing algorithm first.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            H, W, N = self.concentrations.shape
            for n in range(N):
                comp = self.concentrations[:, :, n]
                fname = f"{file_path.rsplit('.', 1)[0]}_comp{n+1}.csv"
                np.savetxt(fname, comp, delimiter=',')
            if self.r_squared is not None:
                r2 = self.r_squared.reshape(H, W)
                fname = f"{file_path.rsplit('.', 1)[0]}_r2.csv"
                np.savetxt(fname, r2, delimiter=',')
            self.statusBar().showMessage(f"Exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # -----------------------------------------------------------------------
    # Data loaded callback
    # -----------------------------------------------------------------------

    def _on_data_loaded(self):
        """Called after loading a hypercube — set up views."""
        H, W, L = self.hypercube.shape
        self.current_band = 0
        self.basis_wavelengths = np.arange(1, L + 1, dtype=float)

        self._refresh_spatial_view()
        self._plot_spectrum(np.zeros(L))

    def _on_colormap_changed(self, cmap_name):
        """Update colormap when selection changes."""
        self.colormap = cmap_name
        self._refresh_spatial_view()

    def _refresh_spatial_view(self):
        """Update the spatial view with the current band."""
        if self.hypercube is None:
            return
        H, W, L = self.hypercube.shape
        band_data = self.hypercube[:, :, self.current_band]

        # Percentile clipping for better contrast
        p_low, p_high = np.percentile(band_data, (1, 99))
        self.norm = Normalize(vmin=p_low, vmax=p_high)

        # Main view
        self.ax_main.clear()
        self.im_handle = self.ax_main.imshow(
            band_data, cmap=self.colormap, norm=self.norm, interpolation='nearest')
        self.ax_main.set_title(f"Band {self.current_band + 1} / {L}")
        self.ax_main.set_xlabel("X (pixels)")
        self.ax_main.set_ylabel("Y (pixels)")
        self.ax_main.set_xlim(-0.5, W - 0.5)
        self.ax_main.set_ylim(H - 0.5, -0.5)
        self.ax_main.set_aspect("equal")

        # X spectral average (top marginal)
        self.ax_x_avg.clear()
        x_avg = band_data.mean(axis=0)
        self.ax_x_avg.plot(x_avg, 'k-', linewidth=0.8)
        self.ax_x_avg.set_xlim(-0.5, W - 0.5)
        self.ax_x_avg.set_xticks([])
        self.ax_x_avg.set_yticks([])

        # Y spectral average (left marginal)
        self.ax_y_avg.clear()
        y_avg = band_data.mean(axis=1)
        self.ax_y_avg.plot(y_avg, 'k-', linewidth=0.8)
        self.ax_y_avg.set_ylim(H - 0.5, -0.5)
        self.ax_y_avg.set_xticks([])

        # Colorbar
        self.fig_main.del_axes(self.ax_cbar)
        self.ax_cbar = self.fig_main.add_subplot(self.gs_main[1, 2])
        self.fig_main.colorbar(self.im_handle, cax=self.ax_cbar)

        self._redraw_selections()
        self.image_canvas.draw_idle()

    def _redraw_selections(self):
        """Redraw selection overlays on spatial view."""
        # Remove old selection overlays
        for child in self.ax_main.get_children():
            if isinstance(child, (Rectangle, Circle)) and hasattr(child, '_sel_overlay'):
                child.remove()

        if self.current_selection and 'mask' in self.current_selection:
            mask = self.current_selection['mask']
            y_indices, x_indices = np.where(mask)
            if len(x_indices) > 0:
                self.ax_main.scatter(x_indices, y_indices,
                                     c='yellow', s=5, alpha=0.5,
                                     edgecolors='red', linewidths=0.5,
                                     label='Selection', zorder=5)
                legend = self.ax_main.legend(loc='upper right', fontsize=8)
                for txt in legend.get_texts():
                    txt.set_color('yellow')

        # Redraw picked endmember markers
        for patch in self.picked_endmember_patches:
            self.ax_main.add_patch(patch)

    # -----------------------------------------------------------------------
    # Canvas interaction
    # -----------------------------------------------------------------------

    def _set_selection_tool(self, tool):
        """Set the active selection tool."""
        self.selection_tool = tool
        self.tool_point.setChecked(tool == 'point')
        self.tool_circle.setChecked(tool == 'circle')
        self.tool_square.setChecked(tool == 'square')
        self.statusBar().showMessage(f"Selection tool: {tool.title()}")

    def _on_canvas_press(self, event):
        """Handle mouse press on spatial canvas."""
        if self.hypercube is None:
            return

        # Endmember picker mode — intercept clicks
        if self.endmember_mode and event.inaxes == self.ax_main and event.button == 1:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            H, W = self.hypercube.shape[:2]
            if 0 <= x < W and 0 <= y < H:
                self.picked_endmembers.append((y, x))
                marker = Circle((x, y), 3, fill=False, color='magenta',
                                linewidth=2)
                self.ax_main.add_patch(marker)
                self.picked_endmember_patches.append(marker)
                self.image_canvas.draw_idle()
                n = len(self.picked_endmembers)
                self.statusBar().showMessage(
                    f'Endmember picker: {n} picked — click more or use Extract Endmembers menu')
            return

        if event.inaxes != self.ax_main:
            return

        if event.button == 1 and self.selection_tool is None:
            # Point selection
            x, y = int(round(event.xdata)), int(round(event.ydata))
            H, W, L = self.hypercube.shape
            if 0 <= x < W and 0 <= y < H:
                spectrum = self.hypercube[y, x, :]
                self.current_selection = {
                    'type': 'point',
                    'x': x, 'y': y,
                    'spectrum': spectrum,
                    'mask': np.zeros((H, W), dtype=bool),
                }
                self.current_selection['mask'][y, x] = True
                self._plot_spectrum(spectrum)
                self._plot_residual(np.zeros(L))
                self._redraw_selections()
                self.image_canvas.draw_idle()
                self.statusBar().showMessage(f"Pixel ({x}, {y}) selected")

        elif event.button == 1 and self.selection_tool in ('circle', 'square'):
            self._drag_start = (event.xdata, event.ydata)
            if self.selection_tool == 'square':
                self._temp_rect = Rectangle(
                    (event.xdata, event.ydata), 0, 0,
                    fill=False, edgecolor='red', linewidth=2, linestyle='--',
                    _sel_overlay=True)
                self.ax_main.add_patch(self._temp_rect)
            elif self.selection_tool == 'circle':
                self._temp_circle = Circle(
                    (event.xdata, event.ydata), 0,
                    fill=False, edgecolor='red', linewidth=2, linestyle='--',
                    _sel_overlay=True)
                self.ax_main.add_patch(self._temp_circle)

    def _on_canvas_motion(self, event):
        """Handle mouse motion on spatial canvas."""
        if self._drag_start is None or self._temp_rect is None and self._temp_circle is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.xdata, event.ydata

        if self.selection_tool == 'square' and self._temp_rect is not None:
            self._temp_rect.set_xy((min(x0, x1), min(y0, y1)))
            self._temp_rect.set_width(abs(x1 - x0))
            self._temp_rect.set_height(abs(y1 - y0))
        elif self.selection_tool == 'circle' and self._temp_circle is not None:
            r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            self._temp_circle.set_center((x0, y0))
            self._temp_circle.set_radius(r)
        self.image_canvas.draw_idle()

    def _on_canvas_release(self, event):
        """Handle mouse release on spatial canvas."""
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.xdata, event.ydata
        H, W, L = self.hypercube.shape

        mask = np.zeros((H, W), dtype=bool)

        if self.selection_tool == 'square':
            x_min, x_max = sorted([int(round(x0)), int(round(x1))])
            y_min, y_max = sorted([int(round(y0)), int(round(y1))])
            x_min, x_max = max(0, x_min), min(W - 1, x_max)
            y_min, y_max = max(0, y_min), min(H - 1, y_max)
            mask[y_min:y_max+1, x_min:x_max+1] = True

        elif self.selection_tool == 'circle':
            r = int(round(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)))
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
            mask[dist <= r] = True

        if mask.any():
            spectra = self.hypercube[mask]
            mean_spectrum = spectra.mean(axis=0)
            self.current_selection = {
                'type': self.selection_tool,
                'spectrum': mean_spectrum,
                'mask': mask,
                'n_pixels': int(mask.sum()),
            }
            self._plot_spectrum(mean_spectrum)
            std = spectra.std(axis=0)
            self._plot_residual(std)
            self._redraw_selections()
            self.image_canvas.draw_idle()
            self.statusBar().showMessage(
                f"{self.selection_tool.title()}: {mask.sum()} pixels selected")

        # Clean up temp overlays
        if self._temp_rect is not None:
            self._temp_rect.remove()
            self._temp_rect = None
        if self._temp_circle is not None:
            self._temp_circle.remove()
            self._temp_circle = None
        self._drag_start = None

    def _on_canvas_key(self, event):
        """Handle key press on spatial canvas."""
        if event.key == 'escape':
            self.clear_selection()
        elif event.key == 'p':
            self._set_selection_tool('point')
        elif event.key == 'c':
            self._set_selection_tool('circle')
        elif event.key == 's':
            self._set_selection_tool('square')

    def _display_component(self, key):
        """Display a component as a concentration map."""
        if self.concentrations is None:
            return
        H, W, N = self.concentrations.shape

        if key == 'total':
            data = self.concentrations.sum(axis=2)
        elif key == 'residual':
            data = self.residuals.reshape(H, W) if self.residuals is not None else np.zeros((H, W))
        elif key == 'r2':
            data = self.r_squared.reshape(H, W) if self.r_squared is not None else np.zeros((H, W))
        else:
            idx = self.basis_labels.index(key) if self.basis_labels else 0
            data = self.concentrations[:, :, idx]

        self.ax_main.clear()
        p_low, p_high = np.percentile(data, (1, 99))
        self.norm = Normalize(vmin=p_low, vmax=p_high)
        self.im_handle = self.ax_main.imshow(
            data, cmap=self.colormap, norm=self.norm, interpolation='nearest')
        self.ax_main.set_title(f"Component: {key}")
        self.ax_main.set_xlabel("X (pixels)")
        self.ax_main.set_ylabel("Y (pixels)")
        self.ax_main.set_xlim(-0.5, W - 0.5)
        self.ax_main.set_ylim(H - 0.5, -0.5)
        self.ax_main.set_aspect("equal")

        self.fig_main.del_axes(self.ax_cbar)
        self.ax_cbar = self.fig_main.add_subplot(self.gs_main[1, 2])
        self.fig_main.colorbar(self.im_handle, cax=self.ax_cbar)

        self._redraw_selections()
        self.image_canvas.draw_idle()

    def _plot_spectrum(self, spectrum):
        """Plot a spectrum in the spectrum panel."""
        self.ax_spectrum.clear()
        wavelengths = self.basis_wavelengths or np.arange(1, len(spectrum) + 1, dtype=float)
        self.ax_spectrum.plot(wavelengths, spectrum, 'b-', linewidth=1)
        self.ax_spectrum.set_xlabel("Spectral Band")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.set_title("Extracted Spectra")
        self.ax_spectrum.grid(True, alpha=0.3)
        self.spectrum_canvas.draw_idle()

    def _plot_basis_spectra(self):
        """Plot basis/endmember spectra in the spectrum panel."""
        if self.basis_spectra is None:
            return
        self.ax_spectrum.clear()
        wavelengths = self.basis_wavelengths or np.arange(1, self.basis_spectra.shape[1] + 1, dtype=float)
        colors = plt.cm.tab10(np.linspace(0, 1, self.basis_spectra.shape[0]))
        for i, (spec, label) in enumerate(zip(self.basis_spectra, self.basis_labels or [])):
            self.ax_spectrum.plot(wavelengths, spec, color=colors[i],
                                  label=label, linewidth=1.2)
        self.ax_spectrum.set_xlabel("Wavelength / Band")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.set_title("Basis Spectra")
        self.ax_spectrum.legend(loc='best', fontsize=8)
        self.ax_spectrum.grid(True, alpha=0.3)
        self.spectrum_canvas.draw_idle()

    def clear_selection(self):
        """Clear the current selection and overlays."""
        self.current_selection = None
        self.selection_tool = None
        self.tool_point.setChecked(False)
        self.tool_circle.setChecked(False)
        self.tool_square.setChecked(False)

        # Clear endmember picker state
        self.endmember_mode = False
        self.picked_endmembers = []
        for patch in self.picked_endmember_patches:
            try:
                patch.remove()
            except (NotImplementedError, AttributeError):
                pass
        self.picked_endmember_patches = []
        self.rgb_mode = False
        self.rgb_image = None
        self.statusBar().showMessage('Selection cleared')

        if self.hypercube is not None:
            self._refresh_spatial_view()

    def _add_current_spectrum(self):
        """Add the currently selected spectrum to the extracted list."""
        if self.current_selection is None or 'spectrum' not in self.current_selection:
            return
        spectrum = self.current_selection['spectrum']
        label = f"Spec {len(self.extracted_spectra) + 1}"
        self.extracted_spectra.append((spectrum, label))
        self._plot_extracted_spectra()

    def _delete_selected_spectrum(self):
        """Remove the last extracted spectrum."""
        if self.extracted_spectra:
            self.extracted_spectra.pop()
            self._plot_extracted_spectra()

    def _set_background(self):
        """Set the current selection as background spectrum."""
        if self.current_selection is None or 'spectrum' not in self.current_selection:
            return
        self.background_spectrum = self.current_selection['spectrum'].copy()
        self.bg_status.setText("Background: Set")
        self.bg_status.setStyleSheet("color: green;")

    def _subtract_background(self):
        """Subtract background from current selection and add result."""
        if self.background_spectrum is None:
            return
        if self.current_selection is None or 'spectrum' not in self.current_selection:
            return
        result = self.current_selection['spectrum'] - self.background_spectrum
        label = f"BG-Subtracted {len(self.extracted_spectra) + 1}"
        self.extracted_spectra.append((result, label))
        self._plot_extracted_spectra()

    def _plot_extracted_spectra(self):
        """Plot all extracted spectra."""
        self.ax_spectrum.clear()
        wavelengths = self.basis_wavelengths or np.arange(1, 100, dtype=float)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.extracted_spectra), 1)))
        for i, (spec, label) in enumerate(self.extracted_spectra):
            wl = wavelengths[:len(spec)] if len(wavelengths) >= len(spec) else np.arange(1, len(spec) + 1, dtype=float)
            self.ax_spectrum.plot(wl, spec, color=colors[i % len(colors)],
                                  label=label, linewidth=1)
        self.ax_spectrum.set_xlabel("Spectral Band")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.set_title("Extracted Spectra")
        self.ax_spectrum.legend(loc='best', fontsize=8)
        self.ax_spectrum.grid(True, alpha=0.3)
        self.spectrum_canvas.draw_idle()

    def _autoscale_spectrum(self):
        """Autoscale the spectrum view."""
        self.ax_spectrum.relim()
        self.ax_spectrum.autoscale_view()
        self.spectrum_canvas.draw_idle()

    def _export_all_spectra(self):
        """Export all extracted spectra to CSV."""
        if not self.extracted_spectra:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Spectra", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            wavelengths = self.basis_wavelengths or np.arange(1, self.extracted_spectra[0][0].shape[0] + 1, dtype=float)
            with open(file_path, 'w') as f:
                f.write("Wavelength," + ",".join(label for _, label in self.extracted_spectra) + "\n")
                for i, wl in enumerate(wavelengths):
                    row = [str(wl)]
                    for spec, _ in self.extracted_spectra:
                        row.append(str(spec[i]) if i < len(spec) else "")
                    f.write(",".join(row) + "\n")
            self.statusBar().showMessage(f"Exported {len(self.extracted_spectra)} spectra to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _set_ui_interactive(self, enabled):
        """Enable/disable UI elements during processing."""
        self.image_canvas.set_cursor(
            Qt.WaitCursor if not enabled else Qt.ArrowCursor)
        self.update()

    def _update_run_button(self):
        """Update button states based on data availability."""
        pass

    # -----------------------------------------------------------------------
    # Algorithm runners  (delegate to unmix package)
    # -----------------------------------------------------------------------

    def denoise_mppca(self):
        """Multiplicative PCA denoising via truncated SVD."""
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data",
                                "Load a hyperspectral image first.")
            return

        H, W, L = self.hypercube.shape
        result = denoise_mppca(self.hypercube, variance_threshold=0.95)
        self.hypercube = result["hypercube"]
        self._refresh_spatial_view()

        info = result["info"]
        QMessageBox.information(
            self, "MPPCA Denoising Complete",
            f"Retained {info['n_components_kept']} of {info['n_components_total']} components\n"
            f"({info['cumulative_variance'] * 100:.1f}% cumulative variance)\n\n"
            f"Original hypercube replaced.")

    def denoise_wavelet(self):
        """Wavelet denoising via universal (VisuShrink) thresholding."""
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data",
                                "Load a hyperspectral image first.")
            return

        try:
            result = denoise_wavelet(self.hypercube, wavelet='db4')
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "Install PyWavelets for wavelet denoising:\n"
                "  pip install PyWavelets")
            return

        self.hypercube = result["hypercube"]
        self._refresh_spatial_view()

        info = result["info"]
        QMessageBox.information(
            self, "Wavelet Denoising Complete",
            f"Wavelet: {info['wavelet']}, Level: {info['level']}\n"
            f"Original hypercube replaced.")

    def run_mcr_als(self):
        """Run MCR-ALS decomposition via parameter dialog."""
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data",
                                "Load a hyperspectral image first.")
            return

        H, W, L = self.hypercube.shape
        dialog = MCRALSDialog(self, L)
        if not dialog.exec_():
            return

        auto_detect = dialog.auto_check.isChecked()
        n_comp = dialog.n_comp_spin.value() if not auto_detect else None
        max_iter = dialog.max_iter_spin.value()
        non_neg = dialog.nn_check.isChecked()
        closure = dialog.closure_check.isChecked()

        self._set_ui_interactive(False)
        self.progress_bar.setValue(0)
        self.unmix_status.setText("MCR-ALS in progress…")
        self.unmix_status.setStyleSheet("color: blue; font-weight: bold;")

        try:
            result = run_mcr_als(
                self.hypercube,
                n_components=n_comp,
                max_iter=max_iter,
                non_neg=non_neg,
                closure=closure,
            )
        except Exception as e:
            self._set_ui_interactive(True)
            QMessageBox.critical(self, "MCR-ALS Error",
                                 f"Decomposition failed:\n{e}")
            return

        S = result["basis_spectra"]
        C = result["concentrations"]
        n_comp_actual = S.shape[0]

        self.basis_spectra = S
        self.basis_labels = [f"Component {k + 1}" for k in range(n_comp_actual)]
        self.basis_wavelengths = np.arange(1, L + 1, dtype=float)
        self.concentrations = C.reshape(H, W, n_comp_actual)
        self.r_squared = result["r_squared"]
        self.residuals = result["residuals"]

        self.unmixing_done = True
        self.progress_bar.setValue(100)

        self.export_action.setEnabled(True)
        self._set_ui_interactive(True)

        mean_r2 = float(np.mean(self.r_squared))
        median_r2 = float(np.median(self.r_squared))
        mean_res = float(np.mean(self.residuals))

        self.unmix_status.setText(
            f"Done — {n_comp_actual} components, {H * W} pixels\n"
            f"R²: mean={mean_r2:.4f}, median={median_r2:.4f}\n"
            f"Residual RMS: {mean_res:.4f}")
        self.unmix_status.setStyleSheet("color: green; font-weight: bold;")

        QMessageBox.information(
            self, "MCR-ALS Complete",
            f"MCR-ALS decomposition finished.\n\n"
            f"Components : {n_comp_actual}\n"
            f"Pixels     : {H * W} ({H}×{W})\n\n"
            f"R² statistics:\n"
            f"  Mean   : {mean_r2:.4f}\n"
            f"  Median : {median_r2:.4f}\n\n"
            f"Residual RMS: {mean_res:.4f}")

        self._plot_basis_spectra()

    def run_nmf(self):
        """Run Non-Negative Matrix Factorization."""
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data",
                                "Load a hyperspectral image first.")
            return

        H, W, L = self.hypercube.shape

        n_comp, ok = QInputDialog.getInt(
            self, "NMF Parameters",
            "Number of components:",
            min(10, L - 1), 1, L - 1, 1)
        if not ok:
            return

        self._set_ui_interactive(False)
        self.progress_bar.setValue(0)
        self.unmix_status.setText("NMF in progress…")
        self.unmix_status.setStyleSheet("color: blue; font-weight: bold;")

        try:
            result = run_nmf(self.hypercube, n_components=n_comp)
        except Exception as e:
            self._set_ui_interactive(True)
            QMessageBox.critical(self, "NMF Error",
                                 f"Decomposition failed:\n{e}")
            return

        S = result["basis_spectra"]
        C = result["concentrations"]

        self.basis_spectra = S
        self.basis_labels = [f"Component {k + 1}" for k in range(n_comp)]
        self.basis_wavelengths = np.arange(1, L + 1, dtype=float)
        self.concentrations = C.reshape(H, W, n_comp)
        self.r_squared = result["r_squared"]
        self.residuals = result["residuals"]

        self.unmixing_done = True
        self.progress_bar.setValue(100)

        self.export_action.setEnabled(True)
        self._set_ui_interactive(True)

        mean_r2 = float(np.mean(self.r_squared))
        median_r2 = float(np.median(self.r_squared))
        mean_res = float(np.mean(self.residuals))

        self.unmix_status.setText(
            f"Done — {n_comp} components, {H * W} pixels\n"
            f"R²: mean={mean_r2:.4f}, median={median_r2:.4f}\n"
            f"Residual RMS: {mean_res:.4f}")
        self.unmix_status.setStyleSheet("color: green; font-weight: bold;")

        QMessageBox.information(
            self, "NMF Complete",
            f"Non-Negative Matrix Factorization finished.\n\n"
            f"Components : {n_comp}\n"
            f"Pixels     : {H * W} ({H}×{W})\n\n"
            f"R² statistics:\n"
            f"  Mean   : {mean_r2:.4f}\n"
            f"  Median : {median_r2:.4f}\n\n"
            f"Residual RMS: {mean_res:.4f}")

        self._plot_basis_spectra()

    def run_mesma(self):
        """Run Multiple Endmember Spectral Mixture Analysis."""
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data",
                                "Load a hyperspectral image first.")
            return
        if self.basis_spectra is None:
            QMessageBox.warning(self, "No Basis",
                                "Load a calibration basis first.")
            return

        H, W, L = self.hypercube.shape
        n_endmembers = self.basis_spectra.shape[0]

        if n_endmembers < 2:
            QMessageBox.warning(self, "Insufficient Endmembers",
                                "Need at least 2 endmembers for MESMA.")
            return

        max_em, ok = QInputDialog.getInt(
            self, "MESMA Parameters",
            "Maximum endmembers per pixel:",
            min(5, n_endmembers), 1, n_endmembers, 1)
        if not ok:
            return

        self._set_ui_interactive(False)
        self.progress_bar.setValue(0)
        self.unmix_status.setText("MESMA in progress…")
        self.unmix_status.setStyleSheet("color: blue; font-weight: bold;")

        try:
            result = run_mesma(
                self.hypercube, self.basis_spectra, max_endmembers=max_em)
        except Exception as e:
            self._set_ui_interactive(True)
            QMessageBox.critical(self, "MESMA Error",
                                 f"Analysis failed:\n{e}")
            return

        C = result["concentrations"]

        self.concentrations = C.reshape(H, W, n_endmembers)
        self.r_squared = result["r_squared"]
        self.residuals = result["residuals"]

        self.unmixing_done = True
        self.progress_bar.setValue(100)

        self.export_action.setEnabled(True)
        self._set_ui_interactive(True)

        mean_r2 = float(np.mean(self.r_squared))
        median_r2 = float(np.median(self.r_squared))
        mean_res = float(np.mean(self.residuals))

        self.unmix_status.setText(
            f"Done — {n_endmembers} endmembers, {H * W} pixels\n"
            f"R²: mean={mean_r2:.4f}, median={median_r2:.4f}\n"
            f"Residual RMS: {mean_res:.4f}")
        self.unmix_status.setStyleSheet("color: green; font-weight: bold;")

        QMessageBox.information(
            self, "MESMA Complete",
            f"Multiple Endmember Spectral Mixture Analysis finished.\n\n"
            f"Endmembers : {n_endmembers}\n"
            f"Max per pixel : {max_em}\n"
            f"Pixels     : {H * W} ({H}×{W})\n\n"
            f"R² statistics:\n"
            f"  Mean   : {mean_r2:.4f}\n"
            f"  Median : {median_r2:.4f}\n\n"
            f"Residual RMS: {mean_res:.4f}")

    def run_pca(self):
        """Run Principal Component Analysis."""
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data",
                                "Load a hyperspectral image first.")
            return

        H, W, L = self.hypercube.shape

        n_comp, ok = QInputDialog.getInt(
            self, "PCA Parameters",
            "Number of principal components:",
            min(10, L - 1), 1, L - 1, 1)
        if not ok:
            return

        self._set_ui_interactive(False)
        self.progress_bar.setValue(0)
        self.unmix_status.setText("PCA in progress…")
        self.unmix_status.setStyleSheet("color: blue; font-weight: bold;")

        try:
            result = run_pca(self.hypercube, n_components=n_comp)
        except Exception as e:
            self._set_ui_interactive(True)
            QMessageBox.critical(self, "PCA Error",
                                 f"Decomposition failed:\n{e}")
            return

        loadings = result["basis_spectra"]
        C = result["concentrations"]
        explained = result["info"]["explained_variance"]
        cumvar = result["info"]["cumulative_variance"]

        self.basis_spectra = loadings
        self.basis_labels = [f"PC {k + 1}" for k in range(n_comp)]
        self.basis_wavelengths = np.arange(1, L + 1, dtype=float)
        self.concentrations = C.reshape(H, W, n_comp)
        self.r_squared = result["r_squared"]
        self.residuals = result["residuals"]

        self.unmixing_done = True
        self.progress_bar.setValue(100)

        self.export_action.setEnabled(True)
        self._set_ui_interactive(True)

        mean_r2 = float(np.mean(self.r_squared))
        median_r2 = float(np.median(self.r_squared))
        mean_res = float(np.mean(self.residuals))

        self.unmix_status.setText(
            f"Done — {n_comp} components, {H * W} pixels\n"
            f"R²: mean={mean_r2:.4f}, median={median_r2:.4f}\n"
            f"Residual RMS: {mean_res:.4f}")
        self.unmix_status.setStyleSheet("color: green; font-weight: bold;")

        expl_lines = '\n'.join(
            f"  PC {k + 1}: {explained[k] * 100:.2f}% "
            f"(cumulative: {cumvar[k] * 100:.2f}%)"
            for k in range(n_comp))

        QMessageBox.information(
            self, "PCA Complete",
            f"Principal Component Analysis finished.\n\n"
            f"Components : {n_comp}\n"
            f"Pixels     : {H * W} ({H}×{W})\n\n"
            f"Explained variance:\n"
            f"{expl_lines}\n\n"
            f"R² statistics:\n"
            f"  Mean   : {mean_r2:.4f}\n"
            f"  Median : {median_r2:.4f}\n\n"
            f"Residual RMS: {mean_res:.4f}")

        self._plot_basis_spectra()

    def run_svr(self):
        """Run Support Vector Regression."""
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data",
                                "Load a hyperspectral image first.")
            return
        if self.basis_spectra is None:
            QMessageBox.warning(self, "No Basis",
                                "Load a calibration basis first.")
            return

        try:
            import sklearn  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "Install scikit-learn for SVR:\n  pip install scikit-learn")
            return

        H, W, L = self.hypercube.shape
        n_endmembers = self.basis_spectra.shape[0]

        self._set_ui_interactive(False)
        self.progress_bar.setValue(0)
        self.unmix_status.setText("SVR training in progress…")
        self.unmix_status.setStyleSheet("color: blue; font-weight: bold;")

        try:
            result = run_svr(self.hypercube, self.basis_spectra)
        except Exception as e:
            self._set_ui_interactive(True)
            QMessageBox.critical(self, "SVR Error",
                                 f"Analysis failed:\n{e}")
            return

        C = result["concentrations"]

        self.concentrations = C.reshape(H, W, n_endmembers)
        self.r_squared = result["r_squared"]
        self.residuals = result["residuals"]

        self.unmixing_done = True
        self.progress_bar.setValue(100)

        self.export_action.setEnabled(True)
        self._set_ui_interactive(True)

        mean_r2 = float(np.mean(self.r_squared))
        median_r2 = float(np.median(self.r_squared))
        mean_res = float(np.mean(self.residuals))

        self.unmix_status.setText(
            f"Done — {n_endmembers} endmembers, {H * W} pixels\n"
            f"R²: mean={mean_r2:.4f}, median={median_r2:.4f}\n"
            f"Residual RMS: {mean_res:.4f}")
        self.unmix_status.setStyleSheet("color: green; font-weight: bold;")

        n_train = result["info"]["n_train"]
        QMessageBox.information(
            self, "SVR Complete",
            f"Support Vector Regression finished.\n\n"
            f"Endmembers : {n_endmembers}\n"
            f"Training samples : {n_train}\n"
            f"Pixels     : {H * W} ({H}×{W})\n\n"
            f"R² statistics:\n"
            f"  Mean   : {mean_r2:.4f}\n"
            f"  Median : {median_r2:.4f}\n\n"
            f"Residual RMS: {mean_res:.4f}")

    def run_sam(self):
        """Run Spectral Angle Mapper classification."""
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data",
                                "Load a hyperspectral image first.")
            return
        if self.basis_spectra is None:
            QMessageBox.warning(self, "No Basis",
                                "Load a calibration basis first.")
            return

        H, W, L = self.hypercube.shape
        n_endmembers = self.basis_spectra.shape[0]

        self._set_ui_interactive(False)
        self.progress_bar.setValue(0)
        self.unmix_status.setText("SAM in progress…")
        self.unmix_status.setStyleSheet("color: blue; font-weight: bold;")

        try:
            result = run_sam(self.hypercube, self.basis_spectra)
        except Exception as e:
            self._set_ui_interactive(True)
            QMessageBox.critical(self, "SAM Error",
                                 f"Analysis failed:\n{e}")
            return

        C = result["concentrations"]
        min_angles = result["residuals"] * np.pi

        self.concentrations = C.reshape(H, W, n_endmembers)
        self.r_squared = result["r_squared"]
        self.residuals = result["residuals"]

        self.unmixing_done = True
        self.progress_bar.setValue(100)

        self.export_action.setEnabled(True)
        self._set_ui_interactive(True)

        mean_r2 = float(np.mean(self.r_squared))
        median_r2 = float(np.median(self.r_squared))
        mean_res = float(np.mean(self.residuals))

        self.unmix_status.setText(
            f"Done — {n_endmembers} endmembers, {H * W} pixels\n"
            f"R²: mean={mean_r2:.4f}, median={median_r2:.4f}\n"
            f"Residual RMS: {mean_res:.4f}")
        self.unmix_status.setStyleSheet("color: green; font-weight: bold;")

        QMessageBox.information(
            self, "SAM Complete",
            f"Spectral Angle Mapper finished.\n\n"
            f"Endmembers : {n_endmembers}\n"
            f"Pixels     : {H * W} ({H}×{W})\n\n"
            f"R² statistics:\n"
            f"  Mean   : {mean_r2:.4f}\n"
            f"  Median : {median_r2:.4f}\n\n"
            f"Residual RMS: {mean_res:.4f}")

    def run_sid(self):
        """Run Spectral Information Divergence classification."""
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data",
                                "Load a hyperspectral image first.")
            return
        if self.basis_spectra is None:
            QMessageBox.warning(self, "No Basis",
                                "Load a calibration basis first.")
            return

        H, W, L = self.hypercube.shape
        n_endmembers = self.basis_spectra.shape[0]

        self._set_ui_interactive(False)
        self.progress_bar.setValue(0)
        self.unmix_status.setText("SID in progress…")
        self.unmix_status.setStyleSheet("color: blue; font-weight: bold;")

        try:
            result = run_sid(self.hypercube, self.basis_spectra)
        except Exception as e:
            self._set_ui_interactive(True)
            QMessageBox.critical(self, "SID Error",
                                 f"Analysis failed:\n{e}")
            return

        C = result["concentrations"]

        self.concentrations = C.reshape(H, W, n_endmembers)
        self.r_squared = result["r_squared"]
        self.residuals = result["residuals"]

        self.unmixing_done = True
        self.progress_bar.setValue(100)

        self.export_action.setEnabled(True)
        self._set_ui_interactive(True)

        mean_r2 = float(np.mean(self.r_squared))
        median_r2 = float(np.median(self.r_squared))
        mean_res = float(np.mean(self.residuals))

        self.unmix_status.setText(
            f"Done — {n_endmembers} endmembers, {H * W} pixels\n"
            f"R²: mean={mean_r2:.4f}, median={median_r2:.4f}\n"
            f"Residual RMS: {mean_res:.4f}")
        self.unmix_status.setStyleSheet("color: green; font-weight: bold;")

        QMessageBox.information(
            self, "SID Complete",
            f"Spectral Information Divergence finished.\n\n"
            f"Endmembers : {n_endmembers}\n"
            f"Pixels     : {H * W} ({H}×{W})\n\n"
            f"R² statistics:\n"
            f"  Mean   : {mean_r2:.4f}\n"
            f"  Median : {median_r2:.4f}\n\n"
            f"Residual RMS: {mean_res:.4f}")

    def run_ica(self):
        """Run Independent Component Analysis."""
        if self.hypercube is None:
            QMessageBox.warning(self, "No Data",
                                "Load a hyperspectral image first.")
            return

        H, W, L = self.hypercube.shape

        n_comp, ok = QInputDialog.getInt(
            self, "ICA Parameters",
            "Number of independent components:",
            min(10, L - 1), 1, L - 1, 1)
        if not ok:
            return

        self._set_ui_interactive(False)
        self.progress_bar.setValue(0)
        self.unmix_status.setText("ICA in progress…")
        self.unmix_status.setStyleSheet("color: blue; font-weight: bold;")

        try:
            from sklearn.decomposition import FastICA
        except ImportError:
            self._set_ui_interactive(True)
            QMessageBox.critical(
                self, "Missing Dependency",
                "Install scikit-learn for ICA:\n  pip install scikit-learn")
            return

        try:
            X = self.hypercube.reshape(-1, L)
            X_mean = X.mean(axis=0)
            X_centered = X - X_mean

            ica = FastICA(n_components=n_comp, random_state=42)
            S_ica = ica.fit_transform(X_centered)

            # Reconstruct to get R²
            A = ica.mixing_  # (n_comp, n_comp)
            # Pad to full L for reconstruction
            A_full = np.zeros((n_comp, L))
            A_full[:, :n_comp] = A
            X_rec = S_ica @ A_full + X_mean

            residuals = np.sqrt(np.mean((X - X_rec) ** 2, axis=1))
            ss_res = np.sum((X - X_rec) ** 2, axis=1)
            ss_tot = np.sum((X - X.mean(axis=0)) ** 2, axis=1)
            r_squared = 1 - ss_res / (ss_tot + 1e-10)

            self.basis_spectra = ica.mixing_
            self.basis_labels = [f"IC {k + 1}" for k in range(n_comp)]
            self.basis_wavelengths = np.arange(1, n_comp + 1, dtype=float)
            self.concentrations = S_ica.reshape(H, W, n_comp)
            self.r_squared = r_squared
            self.residuals = residuals

            self.unmixing_done = True
            self.progress_bar.setValue(100)

            self.component_combo.clear()
            self.component_combo.addItem("Total Concentration", 'total')
            self.component_combo.addItem("Residual (RMS)", 'residual')
            self.component_combo.addItem("R² (Goodness of Fit)", 'r2')
            for lbl in self.basis_labels:
                self.component_combo.addItem(lbl, lbl)

            self.concentration_radio.setEnabled(True)
            self.export_action.setEnabled(True)
            self._set_ui_interactive(True)

            mean_r2 = float(np.mean(self.r_squared))
            median_r2 = float(np.median(self.r_squared))
            mean_res = float(np.mean(self.residuals))

            self.unmix_status.setText(
                f"Done — {n_comp} components, {H * W} pixels\n"
                f"R²: mean={mean_r2:.4f}, median={median_r2:.4f}\n"
                f"Residual RMS: {mean_res:.4f}")
            self.unmix_status.setStyleSheet("color: green; font-weight: bold;")

            QMessageBox.information(
                self, "ICA Complete",
                f"Independent Component Analysis finished.\n\n"
                f"Components : {n_comp}\n"
                f"Pixels     : {H * W} ({H}×{W})\n\n"
                f"R² statistics:\n"
                f"  Mean   : {mean_r2:.4f}\n"
                f"  Median : {median_r2:.4f}\n\n"
                f"Residual RMS: {mean_res:.4f}")

            self._plot_basis_spectra()

        except Exception as e:
            self._set_ui_interactive(True)
            QMessageBox.critical(self, "ICA Error",
                                 f"Decomposition failed:\n{e}")

    # -----------------------------------------------------------------------
    # Endmember extraction
    # -----------------------------------------------------------------------

    def run_nfindr_dialog(self):
        """Dialog to run N-FINDR endmember extraction."""
        if self.hypercube is None:
            QMessageBox.information(self, "No Data",
                                    "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n, ok = QInputDialog.getInt(
            self, "N-FINDR Endmembers",
            f"Number of endmembers to extract (max {min(L-1, H*W-1)}):",
            min(5, L - 2), 2, min(L - 1, H * W - 1), 1)
        if not ok:
            return
        try:
            result = run_nfindr(self.hypercube, n_endmembers=n)
            self._apply_endmember_result(result, "N-FINDR")
        except Exception as e:
            QMessageBox.critical(self, "N-FINDR Error", str(e))

    def run_vca_dialog(self):
        """Dialog to run VCA endmember extraction."""
        if self.hypercube is None:
            QMessageBox.information(self, "No Data",
                                    "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        n, ok = QInputDialog.getInt(
            self, "VCA Endmembers",
            f"Number of endmembers to extract (max {L-1}):",
            min(5, L - 2), 2, L - 1, 1)
        if not ok:
            return
        try:
            result = run_vca(self.hypercube, n_endmembers=n)
            self._apply_endmember_result(result, "VCA")
        except Exception as e:
            QMessageBox.critical(self, "VCA Error", str(e))

    def _apply_endmember_result(self, result, algorithm):
        """Apply extracted endmembers to the basis and mark pixels."""
        endmembers = result['endmembers']
        positions = result['positions']
        info = result['info']

        # Load as basis
        self.basis_spectra = endmembers
        self.basis_labels = [f"{algorithm} #{i+1}" for i in range(len(endmembers))]
        self.basis_wavelengths = np.arange(1, endmembers.shape[1] + 1, dtype=float)

        # Mark pixels on spatial view
        for row, col in positions:
            marker = Circle((col, row), 4, fill=False, color='magenta',
                            linewidth=2)
            self.ax_main.add_patch(marker)
            self.picked_endmember_patches.append(marker)

        self._plot_basis_spectra()
        self._update_run_button()
        self.image_canvas.draw_idle()
        self.statusBar().showMessage(
            f'{algorithm}: extracted {len(endmembers)} endmembers '
            f'({info.get("iterations", "N/A")})')

    def toggle_endmember_picker(self):
        """Toggle endmember picker mode on/off."""
        self.endmember_mode = not self.endmember_mode
        if self.endmember_mode:
            self.statusBar().showMessage(
                'Endmember picker ON — click pixels on spatial view')
        else:
            n = len(self.picked_endmembers)
            self.statusBar().showMessage(
                f'Endmember picker OFF — {n} pixels picked')

    def _apply_picked_endmembers(self):
        """Use picked pixels as endmember basis."""
        if not self.picked_endmembers:
            QMessageBox.information(self, "No Picks",
                                    "Pick at least one pixel first.")
            return
        endmembers = []
        for row, col in self.picked_endmembers:
            endmembers.append(self.hypercube[row, col, :])
        endmembers = np.array(endmembers)
        self.basis_spectra = endmembers
        self.basis_labels = [f"Picked #{i+1}" for i in range(len(endmembers))]
        self.basis_wavelengths = np.arange(1, endmembers.shape[1] + 1, dtype=float)
        self._plot_basis_spectra()
        self._update_run_button()
        self.statusBar().showMessage(
            f'Loaded {len(endmembers)} picked endmembers as basis')

    # -----------------------------------------------------------------------
    # Spectral fitting
    # -----------------------------------------------------------------------

    def run_fit_spectrum(self, model='gaussian'):
        """Fit peaks to the currently selected spectrum."""
        if self.hypercube is None:
            QMessageBox.information(self, "No Data",
                                    "Load a hyperspectral image first.")
            return
        if self.current_selection is None or 'spectrum' not in self.current_selection:
            QMessageBox.information(self, "No Selection",
                                    "Select a pixel or region first (Point/Circle/Square tool).")
            return

        spectrum = self.current_selection['spectrum']
        wavelengths = self.basis_wavelengths
        if wavelengths is None or len(wavelengths) != len(spectrum):
            wavelengths = np.arange(1, len(spectrum) + 1, dtype=float)

        try:
            result = fit_spectrum(
                spectrum, wavelengths=wavelengths, model=model)
            self._show_fit_result(result, wavelengths, spectrum)
        except ValueError as e:
            QMessageBox.information(self, "Fit Info", str(e))
        except RuntimeError as e:
            QMessageBox.critical(self, "Fit Error", str(e))

    def _show_fit_result(self, result, wavelengths, measured):
        """Display spectrum fit in the comparison panel."""
        self.ax_spectrum.clear()
        self.ax_residual.clear()

        self.ax_spectrum.plot(wavelengths, measured, 'k-', linewidth=1,
                              label='Measured', alpha=0.7)
        self.ax_spectrum.plot(wavelengths, result['fitted'], 'r-',
                              linewidth=1.5, label='Fitted')
        self.ax_spectrum.set_xlabel('Wavelength / Band')
        self.ax_spectrum.set_ylabel('Intensity')
        self.ax_spectrum.set_title(
            f'{result["info"]["model"].title()} Fit — '
            f'{result["info"]["n_peaks"]} peaks, χ²={result["info"]["chi2"]:.2e}')
        self.ax_spectrum.legend(loc='best', fontsize=8)
        self.ax_spectrum.grid(True, alpha=0.3)

        self.ax_residual.plot(wavelengths, result['residuals'], 'b-',
                              linewidth=0.8, alpha=0.7)
        self.ax_residual.axhline(0, color='gray', linewidth=0.5)
        self.ax_residual.set_xlabel('Wavelength / Band')
        self.ax_residual.set_ylabel('Residual')
        self.ax_residual.set_title('Fit Residual')
        self.ax_residual.grid(True, alpha=0.3)

        self.spectrum_canvas.draw_idle()

        # Status bar with peak summary
        peaks = result['peaks']
        summary = '; '.join(
            f'Peak {i+1}: {p["center"]:.1f} (w={p["width"]:.1f})'
            for i, p in enumerate(peaks))
        self.statusBar().showMessage(summary)

    # -----------------------------------------------------------------------
    # RGB / PC-RGB visualization
    # -----------------------------------------------------------------------

    def show_rgb_composite(self):
        """Dialog to select 3 bands and display as RGB composite."""
        if self.hypercube is None:
            QMessageBox.information(self, "No Data",
                                    "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        if L < 3:
            QMessageBox.information(self, "Not Enough Bands",
                                    f"Need at least 3 bands (have {L}).")
            return

        # Build a simple band selection dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("RGB Composite — Select 3 Bands")
        layout = QFormLayout(dlg)

        channels = {'Red': None, 'Green': None, 'Blue': None}
        for ch_name in ('Red', 'Green', 'Blue'):
            combo = QComboBox()
            for b in range(L):
                combo.addItem(f"Band {b} (index {b})", b)
            # Default to evenly spaced bands
            if ch_name == 'Red':
                combo.setCurrentIndex(min(L - 1, L * 2 // 3))
            elif ch_name == 'Green':
                combo.setCurrentIndex(min(L - 1, L // 2))
            else:
                combo.setCurrentIndex(0)
            channels[ch_name] = combo
            layout.addRow(f"{ch_name} channel:", combo)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dlg.accept)
        layout.addRow(ok_btn)
        dlg.setLayout(layout)
        dlg.resize(350, 150)

        if dlg.exec_() != QDialog.Accepted:
            return

        bands = {ch: channels[ch].currentData() for ch in channels}

        # Build RGB image — normalize each channel to 0-1
        rgb = np.zeros((H, W, 3), dtype=np.float64)
        for idx, ch in enumerate(('Red', 'Green', 'Blue')):
            band = self.hypercube[:, :, bands[ch]]
            bmin, bmax = band.min(), band.max()
            if bmax > bmin:
                rgb[:, :, idx] = (band - bmin) / (bmax - bmin)
            else:
                rgb[:, :, idx] = 0.0

        self.rgb_image = rgb
        self.rgb_mode = True

        # Display
        self.ax_main.clear()
        self.ax_main.imshow(rgb, interpolation='nearest')
        self.ax_main.set_title(
            f"RGB Composite — R:Band {bands['Red']}, "
            f"G:Band {bands['Green']}, B:Band {bands['Blue']}")
        self.ax_main.set_xlabel("X (pixels)")
        self.ax_main.set_ylabel("Y (pixels)")
        self.ax_main.set_xlim(-0.5, W - 0.5)
        self.ax_main.set_ylim(H - 0.5, -0.5)
        # Hide colorbar for RGB
        self.ax_cbar.set_visible(False)
        self._redraw_selections()
        self.image_canvas.draw_idle()
        self.statusBar().showMessage(
            f"RGB: R={bands['Red']}, G={bands['Green']}, B={bands['Blue']}")

    def show_pcrgb_composite(self):
        """Compute PCA and display first 3 components as RGB."""
        if self.hypercube is None:
            QMessageBox.information(self, "No Data",
                                    "Load a hyperspectral image first.")
            return
        H, W, L = self.hypercube.shape
        if L < 3:
            QMessageBox.information(self, "Not Enough Bands",
                                    f"Need at least 3 bands (have {L}).")
            return

        try:
            # Use PCA from unmix package
            pca_result = run_pca(self.hypercube, n_components=3)
            scores = pca_result['scores']  # (H*W, 3)
            scores = scores.reshape(H, W, 3)

            # Normalize each PC to 0-1
            rgb = np.zeros((H, W, 3), dtype=np.float64)
            for i in range(3):
                pc = scores[:, :, i]
                pmin, pmax = pc.min(), pc.max()
                if pmax > pmin:
                    rgb[:, :, i] = (pc - pmin) / (pmax - pmin)

            self.rgb_image = rgb
            self.rgb_mode = True

            self.ax_main.clear()
            self.ax_main.imshow(rgb, interpolation='nearest')
            self.ax_main.set_title(
                "PC-RGB — PCA Components 1-3"
                + f" (variance: {pca_result['info']['explained_variance_ratio']})")
            self.ax_main.set_xlabel("X (pixels)")
            self.ax_main.set_ylabel("Y (pixels)")
            self.ax_main.set_xlim(-0.5, W - 0.5)
            self.ax_main.set_ylim(H - 0.5, -0.5)
            self.ax_cbar.set_visible(False)
            self._redraw_selections()
            self.image_canvas.draw_idle()
            self.statusBar().showMessage(
                "PC-RGB displayed (first 3 PCA components)")

        except Exception as e:
            QMessageBox.critical(self, "PC-RGB Error", str(e))

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Hyperspectral Unmixer",
            "<h3>Hyperspectral Unmixer v2.0</h3>"
            "<p>Interactive tool for hyperspectral image analysis.</p>"
            "<p>Algorithms: MCR-ALS, NMF, MESMA, PCA, ICA, SVR, SAM, SID, "
            "N-FINDR, VCA, Gaussian/Lorentzian fitting</p>"
            "<p>Visualization: RGB/PC-RGB composites, concentration maps, "
            "basis spectra</p>"
            "<p>Multiplexing Lab, University of Washington</p>")

    # -----------------------------------------------------------------------
    # Helper: show pixel comparison
    # -----------------------------------------------------------------------

    def _show_pixel_comparison(self, hypercube_orig, hypercube_proc,
                                x, y, title="Pixel Comparison"):
        """Show original vs processed spectrum for a single pixel."""
        fig, ax = plt.subplots(figsize=(8, 4))
        L = hypercube_orig.shape[2]
        wavelengths = self.basis_wavelengths or np.arange(1, L + 1, dtype=float)
        ax.plot(wavelengths, hypercube_orig[y, x, :], 'b-',
                linewidth=1, label='Original', alpha=0.7)
        ax.plot(wavelengths, hypercube_proc[y, x, :], 'r-',
                linewidth=1.5, label='Processed')
        ax.set_xlabel("Wavelength / Band")
        ax.set_ylabel("Intensity")
        ax.set_title(f"{title} — Pixel ({x}, {y})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.show()


# ===================================================================
# Dialogs
# ===================================================================

class MCRALSDialog(QDialog):
    """Parameter dialog for MCR-ALS decomposition."""

    def __init__(self, parent=None, n_bands=0):
        super().__init__(parent)
        self.setWindowTitle("MCR-ALS Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        self.n_comp_spin = QSpinBox()
        self.n_comp_spin.setRange(2, max(2, n_bands - 1))
        self.n_comp_spin.setValue(min(10, n_bands - 1))
        layout.addRow("Number of components:", self.n_comp_spin)

        self.auto_check = QCheckBox("Auto-detect components (SVD elbow)")
        self.auto_check.setChecked(True)
        layout.addRow("", self.auto_check)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 1000)
        self.max_iter_spin.setValue(200)
        layout.addRow("Max iterations:", self.max_iter_spin)

        self.nn_check = QCheckBox("Non-negativity constraint")
        self.nn_check.setChecked(True)
        layout.addRow("", self.nn_check)

        self.closure_check = QCheckBox("Closure (sum-to-one)")
        layout.addRow("", self.closure_check)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow("", QWidget())
        layout.addRow("", btn_layout)

        self.setLayout(layout)
        self.resize(400, 250)


# ===================================================================
# Entry point
# ===================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = UnmixerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()