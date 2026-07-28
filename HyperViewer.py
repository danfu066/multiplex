"""
HyperViewer: Hyperspectral Image Viewer
========================================
A PyQt5-based application for viewing and analyzing hyperspectral data (2D spatial + 1D spectral).

Features:
- Load hyperspectral images from TIFF stacks, NPY, and MAT files
- Interactive 2D spatial view with spectral frame slider
- X and Y spectral average side views
- Selection tools: point, circle, square
- Spectrum plotting with color-coded curves
- Spectrum management: add, toggle, delete, export
- Standard deviation visualization for area selections
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QGroupBox, QCheckBox,
    QScrollArea, QButtonGroup, QMessageBox, QToolBar, QAction, QSpinBox,
    QDoubleSpinBox, QRadioButton, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPen, QBrush, QColor

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import Normalize

# Optional imports with fallbacks
try:
    from skimage import io
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not installed. TIFF stack loading may be limited.")

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. MAT file loading not available.")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class SpectrumData:
    """Store spectrum data with metadata"""
    def __init__(self, spectrum, std=None, label="", color="blue", selection_type="point",
                 coords=None, mask=None):
        self.spectrum = np.array(spectrum)
        self.std = np.array(std) if std is not None else None
        self.label = label
        self.color = color
        self.selection_type = selection_type
        self.coords = coords  # (x, y) for point, or center for circle/square
        self.mask = mask  # boolean mask for area selections
        self.visible = True
        self.radius = None  # For circle selections
        self.size = None  # For square selections


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for PyQt5"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class SelectionOverlay:
    """Manage selection overlays on the main image"""
    def __init__(self, ax):
        self.ax = ax
        self.patches = []
    
    def clear(self):
        # Clear by removing all children that we added
        for patch in self.patches[:]:  # Copy list to avoid modification during iteration
            try:
                patch.remove()
            except (NotImplementedError, AttributeError):
                pass  # Some artists can't be removed
        self.patches = []
        # Alternative: just clear axes overlays
        self.ax.figure.canvas.draw_idle()
    
    def add_point(self, x, y, color='red', marker='+'):
        point = self.ax.plot(x, y, marker, color=color, markersize=10, markeredgewidth=2)[0]
        self.patches.append(point)
        self.ax.figure.canvas.draw_idle()
    
    def add_circle(self, center_x, center_y, radius, color='green'):
        circle = Circle((center_x, center_y), radius, fill=False, color=color, linewidth=2)
        self.ax.add_patch(circle)
        self.patches.append(circle)
        self.ax.figure.canvas.draw_idle()
    
    def add_square(self, center_x, center_y, size, color='blue'):
        half_size = size / 2
        square = Rectangle((center_x - half_size, center_y - half_size), size, size,
                          fill=False, color=color, linewidth=2)
        self.ax.add_patch(square)
        self.patches.append(square)
        self.ax.figure.canvas.draw_idle()


class HyperViewer(QMainWindow):
    """Main HyperViewer Application"""

    def __init__(self):
        super().__init__()

        # Data storage
        self.hypercube = None  # Shape: (height, width, spectral_bands)
        self.hypercube_original = None  # Store original for background subtraction
        self.current_frame = 0
        self.spectra_list = []
        self.spectrum_id_counter = 0
        self.color_cycle = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange',
                           'purple', 'brown', 'pink', 'gray', 'lime', 'olive', 'navy', 'teal']
        
        # Background subtraction
        self.background_spectrum = None
        self.background_subtracted = False

        # Color limits for contrast/colorbar editing
        self.clim = None

        # Cross-section position for X and Y spectral views
        self.crosshair_x = None  # X position for Y-spectral view (horizontal line)
        self.crosshair_y = None  # Y position for X-spectral view (vertical line)

        # Selection state
        self.selection_tool = None  # 'point', 'circle', 'square'
        self.selection_active = False
        self.selection_start = None
        self.current_patch = None
        
        # Button group for background selection radio buttons
        self.bg_radio_group = None

        # Current colormap
        self.current_colormap = 'gray'

        # Colorbar axis
        self.ax_cbar = None

        # Spectrum plot state
        self.axes_locked = True
        self.zoom_mode = False
        self.saved_xlim = None
        self.saved_ylim = None

        # Reference spectra library
        self.reference_spectra = None  # 2D array: (n_spectra, n_bands)
        self.reference_labels = []  # Labels for each spectrum (dye names)
        self.reference_wavelengths = None  # Wavelength array (n_bands,)
        self.reference_window = None  # Reference spectra display window

        # Transformed spectra
        self.transformed_spectra = None  # 2D array: (n_spectra, n_bands)
        self.transformed_labels = []  # Labels for transformed spectra
        self.transformed_wavelengths = None  # Wavelength array
        self.transform_window = None  # Transformed spectra display window

        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('HyperViewer - Hyperspectral Image Viewer')
        self.setMinimumSize(1800, 1200)
        self.resize(2000, 1300)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Image display and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Toolbar for selection tools
        self.create_toolbar(left_layout)
        
        # Image display group
        image_group = QGroupBox("Spatial View")
        image_layout = QVBoxLayout(image_group)
        
        # Create figure with proper layout for main image, side views, and colorbar
        self.image_fig = Figure(figsize=(11, 11))
        self.image_canvas = FigureCanvas(self.image_fig)

        # Main axes for spatial view
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        self.ax_main = self.image_fig.add_subplot(111)
        divider = make_axes_locatable(self.ax_main)
        self.ax_x_spectral = divider.append_axes("top", size="18%", pad=0.1, sharex=self.ax_main)
        self.ax_y_spectral = divider.append_axes("left", size="18%", pad=0.1, sharey=self.ax_main)
        self.ax_cbar = divider.append_axes("right", size="3%", pad=0.1)

        # Hide tick labels for side views
        plt.setp(self.ax_x_spectral.get_xticklabels(), visible=False)
        plt.setp(self.ax_y_spectral.get_yticklabels(), visible=False)
        self.ax_x_spectral.tick_params(bottom=False, labelbottom=False)
        self.ax_y_spectral.tick_params(left=False, labelleft=False)

        # Hide colorbar axis ticks
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
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            'gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
            'bone', 'copper', 'red', 'green', 'blue'
        ])
        self.cmap_combo.setCurrentText('gray')
        self.cmap_combo.currentTextChanged.connect(self.on_colormap_changed)
        color_bar_layout.addWidget(self.cmap_combo)

        color_bar_layout.addSpacing(15)
        color_bar_layout.addWidget(QLabel("C-Min:"))
        self.vmin_spinbox = QDoubleSpinBox()
        self.vmin_spinbox.setRange(-1e9, 1e9)
        self.vmin_spinbox.setDecimals(2)
        self.vmin_spinbox.setFixedWidth(75)
        self.vmin_spinbox.valueChanged.connect(self.on_clim_changed)
        color_bar_layout.addWidget(self.vmin_spinbox)

        color_bar_layout.addWidget(QLabel("C-Max:"))
        self.vmax_spinbox = QDoubleSpinBox()
        self.vmax_spinbox.setRange(-1e9, 1e9)
        self.vmax_spinbox.setDecimals(2)
        self.vmax_spinbox.setFixedWidth(75)
        self.vmax_spinbox.valueChanged.connect(self.on_clim_changed)
        color_bar_layout.addWidget(self.vmax_spinbox)

        self.auto_clim_btn = QPushButton("Auto Scale")
        self.auto_clim_btn.setToolTip("Reset color scale to auto min/max of current frame")
        self.auto_clim_btn.clicked.connect(self.reset_clim_auto)
        color_bar_layout.addWidget(self.auto_clim_btn)

        color_bar_layout.addStretch()
        image_layout.addLayout(color_bar_layout)

        left_layout.addWidget(image_group)
        
        # Slider for spectral frames
        slider_group = QGroupBox("Spectral Frame")
        slider_layout = QVBoxLayout(slider_group)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        slider_layout.addWidget(self.frame_label)
        
        left_layout.addWidget(slider_group)
        
        main_layout.addWidget(left_panel, stretch=3)
        
        # Right panel: Spectrum display and controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Spectrum plot
        spectrum_group = QGroupBox("Spectrum Plot")
        spectrum_layout = QVBoxLayout(spectrum_group)

        # Plot toolbar
        plot_toolbar_layout = QHBoxLayout()

        self.lock_axes_btn = QPushButton("🔒 Lock")
        self.lock_axes_btn.setToolTip("Lock axes (prevents auto-rescaling)")
        self.lock_axes_btn.setCheckable(True)
        self.lock_axes_btn.setChecked(True)
        self.lock_axes_btn.clicked.connect(self.toggle_lock_axes)
        self.lock_axes_btn.setFixedWidth(80)
        plot_toolbar_layout.addWidget(self.lock_axes_btn)

        self.autoscale_btn = QPushButton("⤡ Autoscale")
        self.autoscale_btn.setToolTip("Autoscale axes to fit all data")
        self.autoscale_btn.clicked.connect(self.autoscale_axes)
        self.autoscale_btn.setFixedWidth(90)
        plot_toolbar_layout.addWidget(self.autoscale_btn)

        self.zoom_btn = QPushButton("🔍 Zoom")
        self.zoom_btn.setToolTip("Zoom mode (mouse wheel to zoom in/out)")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.setChecked(False)
        self.zoom_btn.clicked.connect(self.toggle_zoom_mode)
        self.zoom_btn.setFixedWidth(80)
        plot_toolbar_layout.addWidget(self.zoom_btn)

        plot_toolbar_layout.addStretch()

        spectrum_layout.addLayout(plot_toolbar_layout)

        self.spectrum_fig = Figure(figsize=(7, 8))
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_canvas.mpl_connect('scroll_event', self.on_spectrum_scroll)
        self.spectrum_ax = self.spectrum_fig.add_subplot(111)
        self.spectrum_ax.set_xlabel('Spectral Band')
        self.spectrum_ax.set_ylabel('Intensity')
        self.spectrum_ax.set_title('Extracted Spectra')
        self.spectrum_ax.grid(True, alpha=0.3)

        spectrum_layout.addWidget(self.spectrum_canvas)
        right_layout.addWidget(spectrum_group)
        
        # Spectrum management controls
        control_group = QGroupBox("Spectrum Management")
        control_layout = QVBoxLayout(control_group)
        
        # Add button
        self.add_btn = QPushButton("Add Current Selection")
        self.add_btn.clicked.connect(self.add_current_spectrum)
        self.add_btn.setEnabled(False)
        control_layout.addWidget(self.add_btn)
        
        # Delete button
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected_spectrum)
        self.delete_btn.setEnabled(False)
        control_layout.addWidget(self.delete_btn)

        # Background subtraction buttons
        self.set_bg_btn = QPushButton("Set as Background")
        self.set_bg_btn.clicked.connect(self.set_background_spectrum)
        self.set_bg_btn.setEnabled(False)
        control_layout.addWidget(self.set_bg_btn)

        self.subtract_bg_btn = QPushButton("Subtract Background")
        self.subtract_bg_btn.clicked.connect(self.toggle_background_subtraction)
        self.subtract_bg_btn.setEnabled(False)
        self.subtract_bg_btn.setCheckable(True)
        control_layout.addWidget(self.subtract_bg_btn)

        # Background status label
        self.bg_status_label = QLabel("Background: None")
        self.bg_status_label.setStyleSheet("color: gray; font-style: italic;")
        control_layout.addWidget(self.bg_status_label)

        # Export button
        self.export_btn = QPushButton("Export All Spectra")
        self.export_btn.clicked.connect(self.export_spectra)
        control_layout.addWidget(self.export_btn)

        # Separator
        control_layout.addSpacing(5)

        # Reference spectra buttons
        ref_group = QHBoxLayout()

        self.load_ref_btn = QPushButton("Load Reference Spectra")
        self.load_ref_btn.clicked.connect(self.load_reference_spectra)
        ref_group.addWidget(self.load_ref_btn)

        self.transform_btn = QPushButton("Transform Spectra")
        self.transform_btn.clicked.connect(self.transform_spectra)
        self.transform_btn.setEnabled(False)
        ref_group.addWidget(self.transform_btn)

        self.save_transform_btn = QPushButton("Save Transformed")
        self.save_transform_btn.clicked.connect(self.save_transformed_spectra)
        self.save_transform_btn.setEnabled(False)
        ref_group.addWidget(self.save_transform_btn)

        control_layout.addLayout(ref_group)

        # Reference spectra status
        self.ref_status_label = QLabel("Reference Spectra: None loaded")
        self.ref_status_label.setStyleSheet("color: gray; font-style: italic;")
        self.ref_status_label.setWordWrap(True)
        control_layout.addWidget(self.ref_status_label)

        # Scrollable area for checkboxes
        self.spectrum_scroll = QScrollArea()
        self.spectrum_scroll.setWidgetResizable(True)
        self.spectrum_scroll.setMaximumHeight(200)
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_widget)
        self.checkbox_layout.setAlignment(Qt.AlignTop)
        self.spectrum_scroll.setWidget(self.checkbox_widget)
        
        # Column headers
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("✓"), 0)  # Visibility checkbox
        header_layout.addWidget(QLabel("BG"), 0)  # Background radio
        header_layout.addWidget(QLabel("Spectrum"), 1)  # Name
        header_layout.addWidget(QLabel("Del"), 0)  # Delete button
        control_layout.addLayout(header_layout)
        control_layout.addWidget(self.spectrum_scroll)
        
        right_layout.addWidget(control_group)
        
        main_layout.addWidget(right_panel, stretch=2)
        
        # Create menus
        self.create_menu_bar()
        
        # Initialize selection overlay manager
        self.selection_overlay = SelectionOverlay(self.ax_main)
        
        # Current selection data
        self.current_selection = None
        
    def create_toolbar(self, layout):
        """Create selection tool toolbar"""
        tool_group = QGroupBox("Selection Tools")
        tool_layout = QHBoxLayout(tool_group)
        self.tool_layout = tool_layout
        
        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.setExclusive(True)
        
        # Point tool
        self.point_btn = QPushButton("• Point")
        self.point_btn.setCheckable(True)
        self.point_btn.clicked.connect(lambda: self.set_selection_tool('point'))
        tool_layout.addWidget(self.point_btn)
        self.tool_button_group.addButton(self.point_btn)
        
        # Circle tool
        self.circle_btn = QPushButton("○ Circle")
        self.circle_btn.setCheckable(True)
        self.circle_btn.clicked.connect(lambda: self.set_selection_tool('circle'))
        tool_layout.addWidget(self.circle_btn)
        self.tool_button_group.addButton(self.circle_btn)
        
        # Square tool
        self.square_btn = QPushButton("□ Square")
        self.square_btn.setCheckable(True)
        self.square_btn.clicked.connect(lambda: self.set_selection_tool('square'))
        tool_layout.addWidget(self.square_btn)
        self.tool_button_group.addButton(self.square_btn)
        
        # Clear selections button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_selection)
        tool_layout.addWidget(self.clear_btn)
        
        # Circle radius / Square size
        self.size_label = QLabel("Size:")
        tool_layout.addWidget(self.size_label)
        
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(1, 100)
        self.size_spinbox.setValue(10)
        self.size_spinbox.setSuffix(" px")
        tool_layout.addWidget(self.size_spinbox)
        
        # Separator
        tool_layout.addSpacing(20)

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

        layout.addWidget(tool_group)
        
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Image", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")

        help_action = QAction("User Guide (README)", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def set_selection_tool(self, tool):
        """Set the current selection tool"""
        self.selection_tool = tool
        
        # Update button states
        self.point_btn.setChecked(tool == 'point')
        self.circle_btn.setChecked(tool == 'circle')
        self.square_btn.setChecked(tool == 'square')
        
        # Enable canvas interaction
        if self.hypercube is not None:
            self.ax_main.figure.canvas.mpl_connect('button_press_event', self.on_canvas_press)
            self.ax_main.figure.canvas.mpl_connect('button_release_event', self.on_canvas_release)
            self.ax_main.figure.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        
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
            if hasattr(self, 'hypercube_original') and self.hypercube_original is not None:
                self.hypercube_original = self.hypercube.copy()
            self.clear_selection()
            self.spectra_list = []
            self.update_checkbox_list()
            self.setup_display()
            self.statusBar().showMessage(msg)
            QMessageBox.information(self, "Spatial Resample", msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to resample image:\n{e}")

    def load_image(self):
        """Open file dialog and load hyperspectral image"""
        file_types = "All Supported (*.tif *.tiff *.npy *.mat);;TIFF Files (*.tif *.tiff);;NumPy Files (*.npy);;MAT Files (*.mat);;All Files (*.*)"
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Hyperspectral Image", "", file_types)
        
        if file_path:
            try:
                self.hypercube = self._load_file(file_path)
                
                if self.hypercube is not None:
                    # Ensure correct shape: (height, width, bands)
                    if self.hypercube.ndim == 2:
                        self.hypercube = self.hypercube[:, :, np.newaxis]
                    elif self.hypercube.ndim == 3:
                        # Check if bands are in first dimension
                        if self.hypercube.shape[0] < self.hypercube.shape[1] and self.hypercube.shape[0] < self.hypercube.shape[2]:
                            self.hypercube = np.transpose(self.hypercube, (1, 2, 0))
                    
                    self.setup_display()
                    self.setWindowTitle(f'HyperViewer - {os.path.basename(file_path)}')
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
    
    def _load_file(self, file_path):
        """Load file based on extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.tif', '.tiff']:
            return self._load_tiff(file_path)
        elif ext == '.npy':
            return self._load_npy(file_path)
        elif ext == '.mat':
            return self._load_mat(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _load_tiff(self, file_path):
        """Load TIFF stack"""
        if HAS_TIFFFILE:
            data = tifffile.imread(file_path)
        elif HAS_SKIMAGE:
            data = io.imread(file_path)
        else:
            raise ImportError("Please install tifffile or scikit-image for TIFF support:\n"
                            "pip install tifffile scikit-image")
        
        return np.array(data)
    
    def _load_npy(self, file_path):
        """Load NumPy file"""
        return np.load(file_path)
    
    def _load_mat(self, file_path):
        """Load MATLAB file"""
        if not HAS_SCIPY:
            raise ImportError("Please install scipy for MAT file support:\n"
                            "pip install scipy")
        
        mat_data = sio.loadmat(file_path)
        
        # Find the hyperspectral data array
        for key, value in mat_data.items():
            if not key.startswith('_') and isinstance(value, np.ndarray):
                if value.ndim >= 2:
                    return value
        
        raise ValueError("No suitable array found in MAT file")
    
    def setup_display(self):
        """Setup display after loading image"""
        if self.hypercube is None:
            return

        height, width, bands = self.hypercube.shape
        
        # Store original hypercube for background subtraction
        self.hypercube_original = self.hypercube.copy()
        self.background_spectrum = None
        self.background_subtracted = False
        self.subtract_bg_btn.setEnabled(False)
        self.bg_status_label.setText("Background: None")
        self.bg_status_label.setStyleSheet("color: gray; font-style: italic;")
        
        # Initialize crosshair positions to center of image
        self.crosshair_x = width // 2
        self.crosshair_y = height // 2

        # Setup slider
        self.frame_slider.setRange(0, bands - 1)
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"Frame: 0 / {bands - 1}")

        # Clear previous content
        self.ax_main.clear()
        self.ax_x_spectral.clear()
        self.ax_y_spectral.clear()

        # Display initial frame
        self.display_frame(0)
        
        # Reset selection
        self.clear_selection()
        
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

    def display_frame(self, frame_idx):
        """Display a specific spectral frame with unified colormap and color limits across XY, XZ, YZ views."""
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

        # 1. Main spatial view (XY)
        self.ax_main.clear()
        im = self.ax_main.imshow(self.hypercube[:, :, frame_idx], cmap=self.current_colormap,
                                vmin=vmin, vmax=vmax,
                                interpolation='nearest', extent=[-0.5, width - 0.5, height - 0.5, -0.5])
        self.ax_main.set_xlim(-0.5, width - 0.5)
        self.ax_main.set_ylim(height - 0.5, -0.5)

        # Update colorbar
        self._update_colorbar(im)

        # 2. X spectral view (XZ): horizontal cross-section at crosshair_y
        self.ax_x_spectral.clear()
        cy = max(0, min(self.crosshair_y if self.crosshair_y is not None else height // 2, height - 1))
        x_spectral = self.hypercube[cy, :, :]
        self.ax_x_spectral.imshow(x_spectral.T, cmap=self.current_colormap,
                                   vmin=vmin, vmax=vmax, aspect='auto',
                                   extent=[-0.5, width - 0.5, bands - 0.5, -0.5])
        self.ax_x_spectral.set_xlim(-0.5, width - 0.5)
        self.ax_x_spectral.set_ylim(bands - 0.5, -0.5)
        plt.setp(self.ax_x_spectral.get_xticklabels(), visible=False)

        # 3. Y spectral view (YZ): vertical cross-section at crosshair_x
        self.ax_y_spectral.clear()
        cx = max(0, min(self.crosshair_x if self.crosshair_x is not None else width // 2, width - 1))
        y_spectral = self.hypercube[:, cx, :]
        self.ax_y_spectral.imshow(y_spectral, cmap=self.current_colormap,
                                   vmin=vmin, vmax=vmax, aspect='auto',
                                   extent=[-0.5, bands - 0.5, height - 0.5, -0.5])
        self.ax_y_spectral.set_xlim(-0.5, bands - 0.5)
        self.ax_y_spectral.set_ylim(height - 0.5, -0.5)
        plt.setp(self.ax_y_spectral.get_yticklabels(), visible=False)

        # Redraw all selections - reset overlay patches list since axes was cleared
        self.selection_overlay.patches = []
        self.redraw_all_selections()

        self.image_canvas.draw_idle()

    def _update_colorbar(self, mappable):
        """Update the colorbar with new mappable and clean labels"""
        self.ax_cbar.clear()
        cbar = self.image_fig.colorbar(mappable, cax=self.ax_cbar, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)

    def on_colormap_changed(self, cmap_name):
        """Handle colormap change"""
        self.current_colormap = cmap_name
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

    def on_clim_changed(self):
        """Handle manual color limit (Vmin, Vmax) spinbox changes."""
        vmin = self.vmin_spinbox.value()
        vmax = self.vmax_spinbox.value()
        if vmin >= vmax:
            return
        self.clim = (vmin, vmax)
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

    def reset_clim_auto(self):
        """Reset color scale limits to auto frame min/max."""
        self.clim = None
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

    def toggle_lock_axes(self):
        """Toggle lock/unlock for spectrum plot axes"""
        self.axes_locked = self.lock_axes_btn.isChecked()
        if self.axes_locked:
            self.lock_axes_btn.setText("🔒 Lock")
            self.zoom_btn.setChecked(False)
            self.zoom_mode = False
            self.zoom_btn.setText("🔍 Zoom")
            self.spectrum_canvas.setCursor(Qt.ArrowCursor)
            # Save current limits
            self.saved_xlim = self.spectrum_ax.get_xlim()
            self.saved_ylim = self.spectrum_ax.get_ylim()
        else:
            self.lock_axes_btn.setText("🔓 Lock")

    def autoscale_axes(self):
        """Autoscale spectrum plot axes to fit all visible data"""
        # Collect all data extents from lines
        all_xmin, all_xmax = [], []
        all_ymin, all_ymax = [], []
        for line in self.spectrum_ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) > 0:
                all_xmin.append(np.min(xdata))
                all_xmax.append(np.max(xdata))
            if len(ydata) > 0:
                all_ymin.append(np.min(ydata))
                all_ymax.append(np.max(ydata))

        if all_xmin and all_ymax:
            x_margin = (max(all_xmax) - min(all_xmin)) * 0.05
            y_margin = (max(all_ymax) - min(all_ymin)) * 0.05
            self.spectrum_ax.set_xlim(min(all_xmin) - x_margin, max(all_xmax) + x_margin)
            self.spectrum_ax.set_ylim(min(all_ymin) - y_margin, max(all_ymax) + y_margin)

        # If locked, update saved limits
        if self.axes_locked:
            self.saved_xlim = self.spectrum_ax.get_xlim()
            self.saved_ylim = self.spectrum_ax.get_ylim()
        self.spectrum_canvas.draw_idle()

    def toggle_zoom_mode(self):
        """Toggle zoom mode for spectrum plot"""
        self.zoom_mode = self.zoom_btn.isChecked()
        if self.zoom_mode:
            self.zoom_btn.setText("✋ Zoom")
            self.lock_axes_btn.setChecked(False)
            self.axes_locked = False
            self.lock_axes_btn.setText("🔓 Lock")
            self.spectrum_canvas.setCursor(Qt.CrossCursor)
        else:
            self.zoom_btn.setText("🔍 Zoom")
            self.lock_axes_btn.setChecked(True)
            self.axes_locked = True
            self.lock_axes_btn.setText("🔒 Lock")
            # Save current zoomed limits as the locked view
            self.saved_xlim = self.spectrum_ax.get_xlim()
            self.saved_ylim = self.spectrum_ax.get_ylim()
            self.spectrum_canvas.setCursor(Qt.ArrowCursor)

    def on_spectrum_scroll(self, event):
        """Handle mouse wheel scroll on spectrum plot for zooming"""
        if not self.zoom_mode:
            return

        # Accept scroll anywhere on the canvas (not just axes)
        if event.xdata is None or event.ydata is None:
            return

        # Get current limits
        xlim = self.spectrum_ax.get_xlim()
        ylim = self.spectrum_ax.get_ylim()

        # Zoom factor
        factor = 0.9 if event.button == 'up' else 1.1

        # Calculate new limits centered on mouse position
        x_center = event.xdata
        y_center = event.ydata

        new_xmin = x_center + (xlim[0] - x_center) * factor
        new_xmax = x_center + (xlim[1] - x_center) * factor
        self.spectrum_ax.set_xlim(new_xmin, new_xmax)

        new_ymin = y_center + (ylim[0] - y_center) * factor
        new_ymax = y_center + (ylim[1] - y_center) * factor
        self.spectrum_ax.set_ylim(new_ymin, new_ymax)

        self.spectrum_canvas.draw_idle()

    def on_frame_changed(self, value):
        """Handle spectral frame slider change"""
        self.current_frame = value
        self.frame_label.setText(f"Frame: {value} / {self.frame_slider.maximum()}")
        self.display_frame(value)
        
    def on_canvas_press(self, event):
        """Handle mouse press on canvas"""
        if self.selection_tool is None or event.inaxes != self.ax_main:
            return
        
        if event.button == 1:  # Left click
            self.selection_active = True
            self.selection_start = (event.xdata, event.ydata)
            
            if self.selection_tool == 'point':
                # For point, just add immediately
                self.add_point_selection(event.xdata, event.ydata)
                self.selection_active = False
                
    def on_canvas_release(self, event):
        """Handle mouse release on canvas"""
        if not self.selection_active or event.inaxes != self.ax_main:
            return
        
        self.selection_active = False
        
        if self.selection_start is None:
            return
        
        x1, y1 = self.selection_start
        x2, y2 = event.xdata, event.ydata
        
        if self.selection_tool == 'circle':
            radius = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            self.add_circle_selection(x1, y1, radius)
            
        elif self.selection_tool == 'square':
            size = max(abs(x2 - x1), abs(y2 - y1))
            self.add_square_selection(x1, y1, size)

        # Clear temporary patch
        if self.current_patch:
            try:
                self.current_patch.remove()
            except (NotImplementedError, AttributeError):
                pass
            self.current_patch = None
            self.image_canvas.draw_idle()
    
    def on_canvas_motion(self, event):
        """Handle mouse motion on canvas"""
        if not self.selection_active or event.inaxes != self.ax_main:
            return
        
        if self.selection_start is None:
            return

        x1, y1 = self.selection_start
        x2, y2 = event.xdata, event.ydata

        # Clear previous temporary patch
        if self.current_patch:
            try:
                self.current_patch.remove()
            except (NotImplementedError, AttributeError):
                pass
            self.current_patch = None

        # Draw temporary patch
        if self.selection_tool == 'circle':
            radius = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            self.current_patch = Circle((x1, y1), radius, fill=False, color='green', linewidth=2, alpha=0.7)
            self.ax_main.add_patch(self.current_patch)

        elif self.selection_tool == 'square':
            size = max(abs(x2 - x1), abs(y2 - y1))
            half_size = size / 2
            self.current_patch = Rectangle((x1 - half_size, y1 - half_size), size, size,
                                          fill=False, color='blue', linewidth=2, alpha=0.7)
            self.ax_main.add_patch(self.current_patch)

        self.image_canvas.draw_idle()
    
    def add_point_selection(self, x, y):
        """Add point selection"""
        x, y = int(round(x)), int(round(y))
        height, width = self.hypercube.shape[:2]

        if 0 <= x < width and 0 <= y < height:
            self.current_selection = {
                'type': 'point',
                'coords': (x, y),
                'spectrum': self.hypercube[y, x, :]
            }
            # Update crosshair positions
            self.crosshair_x = x
            self.crosshair_y = y
            # Update display and redraw overlays (saved + current)
            self.display_frame(self.current_frame)
            self._redraw_overlay_with_current()
            self.add_btn.setEnabled(True)
            self.plot_current_spectrum()

    def _redraw_overlay_with_current(self):
        """Redraw overlay with saved selections plus current selection"""
        self.selection_overlay.patches = []
        # First draw all saved selections
        self.redraw_all_selections()
        # Then draw current selection
        if self.current_selection:
            sel = self.current_selection
            if sel['type'] == 'point':
                self.selection_overlay.add_point(*sel['coords'])
            elif sel['type'] == 'circle':
                self.selection_overlay.add_circle(*sel['coords'], sel['radius'])
            elif sel['type'] == 'square':
                self.selection_overlay.add_square(*sel['coords'], sel['size'])
    
    def add_circle_selection(self, center_x, center_y, radius):
        """Add circle selection"""
        center_x, center_y = int(round(center_x)), int(round(center_y))
        radius = max(1, int(round(radius)))
        height, width = self.hypercube.shape[:2]

        # Create mask
        y, x = np.ogrid[:height, :width]
        mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)

        # Extract spectra
        spectra_in_region = self.hypercube[mask]
        mean_spectrum = np.mean(spectra_in_region, axis=0)
        std_spectrum = np.std(spectra_in_region, axis=0)

        self.current_selection = {
            'type': 'circle',
            'coords': (center_x, center_y),
            'radius': radius,
            'mask': mask,
            'spectrum': mean_spectrum,
            'std': std_spectrum
        }

        # Update crosshair positions
        self.crosshair_x = center_x
        self.crosshair_y = center_y

        # Update display and redraw overlays (saved + current)
        self.display_frame(self.current_frame)
        self._redraw_overlay_with_current()
        self.add_btn.setEnabled(True)
        self.plot_current_spectrum()

    def add_square_selection(self, center_x, center_y, size):
        """Add square selection"""
        center_x, center_y = int(round(center_x)), int(round(center_y))
        size = max(1, int(round(size)))
        half_size = size // 2
        height, width = self.hypercube.shape[:2]

        # Create mask
        x1 = max(0, center_x - half_size)
        x2 = min(width, center_x + half_size + 1)
        y1 = max(0, center_y - half_size)
        y2 = min(height, center_y + half_size + 1)

        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = True

        # Extract spectra
        spectra_in_region = self.hypercube[mask]
        mean_spectrum = np.mean(spectra_in_region, axis=0)
        std_spectrum = np.std(spectra_in_region, axis=0)

        self.current_selection = {
            'type': 'square',
            'coords': (center_x, center_y),
            'size': size,
            'mask': mask,
            'spectrum': mean_spectrum,
            'std': std_spectrum
        }

        # Update crosshair positions
        self.crosshair_x = center_x
        self.crosshair_y = center_y

        # Update display and redraw overlays (saved + current)
        self.display_frame(self.current_frame)
        self._redraw_overlay_with_current()
        self.add_btn.setEnabled(True)
        self.plot_current_spectrum()

    def redraw_selection(self):
        """Redraw current selection overlay"""
        self.selection_overlay.patches = []  # Clear list since axes was cleared

        if self.current_selection:
            sel = self.current_selection
            if sel['type'] == 'point':
                self.selection_overlay.add_point(*sel['coords'])
            elif sel['type'] == 'circle':
                self.selection_overlay.add_circle(*sel['coords'], sel['radius'])
            elif sel['type'] == 'square':
                self.selection_overlay.add_square(*sel['coords'], sel['size'])

    def redraw_all_selections(self):
        """Redraw all saved selections from spectra_list"""
        for spectrum_data in self.spectra_list:
            if spectrum_data.visible and spectrum_data.coords is not None:
                if spectrum_data.selection_type == 'point':
                    self.selection_overlay.add_point(*spectrum_data.coords, color=spectrum_data.color, marker='+')
                elif spectrum_data.selection_type == 'circle':
                    self.selection_overlay.add_circle(*spectrum_data.coords, spectrum_data.radius, color=spectrum_data.color)
                elif spectrum_data.selection_type == 'square':
                    self.selection_overlay.add_square(*spectrum_data.coords, spectrum_data.size, color=spectrum_data.color)

    def plot_current_spectrum(self):
        """Plot all saved spectra plus current selection"""
        self.spectrum_ax.clear()
        self.spectrum_ax.set_xlabel('Spectral Band')
        self.spectrum_ax.set_ylabel('Intensity')
        self.spectrum_ax.set_title('Extracted Spectra')
        self.spectrum_ax.grid(True, alpha=0.3)

        # Plot all saved spectra
        has_data = False
        for spectrum_data in self.spectra_list:
            if spectrum_data.visible:
                bands = len(spectrum_data.spectrum)
                self.spectrum_ax.plot(range(bands), spectrum_data.spectrum,
                                     color=spectrum_data.color, linewidth=2,
                                     label=spectrum_data.label)

                # Plot std if available
                if spectrum_data.std is not None:
                    self.spectrum_ax.fill_between(range(bands),
                                                 spectrum_data.spectrum - spectrum_data.std,
                                                 spectrum_data.spectrum + spectrum_data.std,
                                                 alpha=0.2, color=spectrum_data.color)
                has_data = True

        # Plot current selection on top
        if self.current_selection is not None:
            spectrum = self.current_selection['spectrum']
            bands = len(spectrum)
            self.spectrum_ax.plot(range(bands), spectrum, 'k--', linewidth=2, label='Current')

            # Plot std if available
            if self.current_selection.get('std') is not None:
                std = self.current_selection['std']
                self.spectrum_ax.fill_between(range(bands), spectrum - std, spectrum + std,
                                             alpha=0.3, color='black', label='±1σ')

        if has_data or self.current_selection is not None:
            self.spectrum_ax.legend(loc='upper right', fontsize=8)

        # Enforce or initialize locked limits
        if self.axes_locked:
            if self.saved_xlim is None or self.saved_ylim is None:
                # First draw: save the autoscaled limits
                self.saved_xlim = self.spectrum_ax.get_xlim()
                self.saved_ylim = self.spectrum_ax.get_ylim()
            else:
                self.spectrum_ax.set_xlim(self.saved_xlim)
                self.spectrum_ax.set_ylim(self.saved_ylim)

        self.spectrum_canvas.draw_idle()

    def clear_selection(self):
        """Clear current selection (not saved spectra)"""
        self.current_selection = None
        self.selection_overlay.patches = []  # Clear list
        self.add_btn.setEnabled(False)
        
        # Redraw all saved selections
        self.redraw_all_selections()

        # Clear spectrum plot
        self.spectrum_ax.clear()
        self.spectrum_ax.set_xlabel('Spectral Band')
        self.spectrum_ax.set_ylabel('Intensity')
        self.spectrum_ax.set_title('Extracted Spectra')
        self.spectrum_ax.grid(True, alpha=0.3)
        self.spectrum_canvas.draw_idle()
    
    def add_current_spectrum(self):
        """Add current selection to spectra list"""
        if self.current_selection is None:
            return

        # Get color
        color = self.color_cycle[self.spectrum_id_counter % len(self.color_cycle)]
        self.spectrum_id_counter += 1

        # Create spectrum data
        spectrum_data = SpectrumData(
            spectrum=self.current_selection['spectrum'],
            std=self.current_selection.get('std'),
            label=f"Spectrum {len(self.spectra_list) + 1} ({self.current_selection['type']})",
            color=color,
            selection_type=self.current_selection['type'],
            coords=self.current_selection.get('coords'),
            mask=self.current_selection.get('mask')
        )

        # Store additional selection parameters
        if self.current_selection['type'] == 'circle':
            spectrum_data.radius = self.current_selection.get('radius', 10)
        elif self.current_selection['type'] == 'square':
            spectrum_data.size = self.current_selection.get('size', 10)

        self.spectra_list.append(spectrum_data)
        self.update_spectrum_plot()
        self.update_checkbox_list()
        self.set_bg_btn.setEnabled(True)  # Enable background button when there's at least one spectrum
        self.delete_btn.setEnabled(True)  # Enable delete button when there's at least one spectrum

        # Clear current selection after adding
        self.current_selection = None
        self.selection_overlay.patches = []
        self.redraw_all_selections()
        self.add_btn.setEnabled(False)
        
    def update_spectrum_plot(self):
        """Update the spectrum plot with all added spectra"""
        self.spectrum_ax.clear()
        self.spectrum_ax.set_xlabel('Spectral Band')
        self.spectrum_ax.set_ylabel('Intensity')
        self.spectrum_ax.set_title('Extracted Spectra')
        self.spectrum_ax.grid(True, alpha=0.3)
        
        has_data = False
        for spectrum_data in self.spectra_list:
            if spectrum_data.visible:
                bands = len(spectrum_data.spectrum)
                self.spectrum_ax.plot(range(bands), spectrum_data.spectrum,
                                     color=spectrum_data.color, linewidth=2,
                                     label=spectrum_data.label)
                
                # Plot std if available
                if spectrum_data.std is not None:
                    self.spectrum_ax.fill_between(range(bands),
                                                 spectrum_data.spectrum - spectrum_data.std,
                                                 spectrum_data.spectrum + spectrum_data.std,
                                                 alpha=0.2, color=spectrum_data.color)
                has_data = True
        
        if has_data:
            self.spectrum_ax.legend(loc='upper right', fontsize=8)

        # Enforce or initialize locked limits
        if self.axes_locked:
            if self.saved_xlim is None or self.saved_ylim is None:
                self.saved_xlim = self.spectrum_ax.get_xlim()
                self.saved_ylim = self.spectrum_ax.get_ylim()
            else:
                self.spectrum_ax.set_xlim(self.saved_xlim)
                self.spectrum_ax.set_ylim(self.saved_ylim)

        self.spectrum_canvas.draw_idle()

    def update_checkbox_list(self):
        """Update checkbox list for spectrum visibility"""
        # Clear existing checkboxes
        while self.checkbox_layout.count():
            item = self.checkbox_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create new button group for radio buttons (ensures mutual exclusivity)
        self.bg_radio_group = QButtonGroup(self)
        
        # Determine which spectrum is currently set as background
        bg_spectrum_id = None
        if self.background_spectrum is not None:
            for idx, spectrum_data in enumerate(self.spectra_list):
                if np.array_equal(self.background_spectrum, spectrum_data.spectrum):
                    bg_spectrum_id = idx
                    break

        # Add checkboxes and radio buttons for each spectrum
        for idx, spectrum_data in enumerate(self.spectra_list):
            # Create a horizontal layout for each spectrum
            item_layout = QHBoxLayout()
            
            # Checkbox for visibility
            checkbox = QCheckBox()
            checkbox.setChecked(spectrum_data.visible)
            checkbox.setToolTip("Toggle visibility")
            checkbox.stateChanged.connect(lambda state, i=idx: self.toggle_spectrum_visibility(i))
            item_layout.addWidget(checkbox)
            
            # Radio button for background selection
            radio = QRadioButton()
            radio.setToolTip("Select as background")
            radio.setChecked(bg_spectrum_id == idx)
            self.bg_radio_group.addButton(radio, idx)  # Add to button group with index
            item_layout.addWidget(radio)
            
            # Label with spectrum name
            label = QLabel(spectrum_data.label)
            label.setStyleSheet(f"color: {spectrum_data.color};")
            item_layout.addWidget(label)
            
            # Stretch to push delete button to right
            item_layout.addStretch()
            
            # Small delete button for this spectrum
            del_btn = QPushButton("×")
            del_btn.setFixedSize(20, 20)
            del_btn.setToolTip("Delete this spectrum")
            del_btn.clicked.connect(lambda checked, i=idx: self.delete_single_spectrum(i))
            item_layout.addWidget(del_btn)
            
            self.checkbox_layout.addLayout(item_layout)
    
    def toggle_spectrum_visibility(self, index):
        """Toggle visibility of a spectrum"""
        if 0 <= index < len(self.spectra_list):
            self.spectra_list[index].visible = not self.spectra_list[index].visible
            self.update_spectrum_plot()
            # Redraw selections on image
            self.selection_overlay.patches = []
            self.redraw_all_selections()
            self.image_canvas.draw_idle()
    
    def delete_selected_spectrum(self):
        """Delete selected spectra (delete checked ones, or last if none checked)"""
        if not self.spectra_list:
            return

        # Find checked checkboxes and delete those spectra
        indices_to_delete = []
        for i in range(self.checkbox_layout.count()):
            item = self.checkbox_layout.itemAt(i)
            if item.layout():
                # Get the checkbox (first widget in the layout)
                check_item = item.layout().itemAt(0)
                if check_item and check_item.widget() and isinstance(check_item.widget(), QCheckBox):
                    if check_item.widget().isChecked():
                        indices_to_delete.append(i)

        if not indices_to_delete:
            # If nothing checked, delete the last one
            indices_to_delete = [len(self.spectra_list) - 1]

        # Delete in reverse order to maintain indices
        for idx in sorted(indices_to_delete, reverse=True):
            if 0 <= idx < len(self.spectra_list):
                self.spectra_list.pop(idx)

        self.update_spectrum_plot()
        self.update_checkbox_list()
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

        # Update button states
        if len(self.spectra_list) == 0:
            self.set_bg_btn.setEnabled(False)
            self.subtract_bg_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.background_spectrum = None
            self.background_subtracted = False
            self.subtract_bg_btn.setChecked(False)
            self.bg_status_label.setText("Background: None")
            self.bg_status_label.setStyleSheet("color: gray; font-style: italic;")
        else:
            self.set_bg_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)

    def delete_single_spectrum(self, index):
        """Delete a single spectrum by index"""
        if not self.spectra_list or index < 0 or index >= len(self.spectra_list):
            return

        self.spectra_list.pop(index)
        self.update_spectrum_plot()
        self.update_checkbox_list()
        if self.hypercube is not None:
            self.display_frame(self.current_frame)

        # Update button states
        if len(self.spectra_list) == 0:
            self.set_bg_btn.setEnabled(False)
            self.subtract_bg_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.background_spectrum = None
            self.background_subtracted = False
            self.subtract_bg_btn.setChecked(False)
            self.bg_status_label.setText("Background: None")
            self.bg_status_label.setStyleSheet("color: gray; font-style: italic;")
        else:
            self.set_bg_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)

    def set_background_spectrum(self):
        """Set the selected spectrum as background using radio button"""
        if not self.spectra_list:
            return

        # Get checked radio button ID from button group
        bg_index = self.bg_radio_group.checkedId()
        
        # If nothing checked, use the last one
        if bg_index < 0:
            bg_index = len(self.spectra_list) - 1

        # Use the selected spectrum as background
        self.background_spectrum = self.spectra_list[bg_index].spectrum.copy()
        self.subtract_bg_btn.setEnabled(True)
        self.bg_status_label.setText(f"Background: {self.spectra_list[bg_index].label}")
        self.bg_status_label.setStyleSheet("color: green; font-weight: bold;")
        
        # Update checkbox list to show correct radio button state
        self.update_checkbox_list()
        
        QMessageBox.information(self, "Background Set",
                               f"Background set from: {self.spectra_list[bg_index].label}\n"
                               "Click 'Subtract Background' to apply.")

    def toggle_background_subtraction(self):
        """Toggle background subtraction on/off"""
        if self.background_spectrum is None:
            self.subtract_bg_btn.setChecked(False)
            QMessageBox.warning(self, "Warning", "No background spectrum set!")
            return

        self.background_subtracted = self.subtract_bg_btn.isChecked()

        if self.background_subtracted:
            # Apply background subtraction to hypercube
            self.hypercube = self.hypercube_original - self.background_spectrum[np.newaxis, np.newaxis, :]
            self.bg_status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            # Restore original
            self.hypercube = self.hypercube_original.copy()
            self.bg_status_label.setStyleSheet("color: green; font-weight: bold;")
        
        # Update all spectra
        self._update_all_spectra()
        
        # Redisplay current frame
        self.display_frame(self.current_frame)
        
        # Update spectrum plot
        self.update_spectrum_plot()

    def _update_all_spectra(self):
        """Update all stored spectra after background subtraction"""
        if self.background_spectrum is None:
            return
        
        for spectrum_data in self.spectra_list:
            if spectrum_data.mask is not None:
                # Area selection - recalculate from hypercube
                spectra_in_region = self.hypercube[spectrum_data.mask]
                spectrum_data.spectrum = np.mean(spectra_in_region, axis=0)
                if spectrum_data.std is not None:
                    spectrum_data.std = np.std(spectra_in_region, axis=0)
            elif spectrum_data.coords is not None:
                # Point selection - get from hypercube
                x, y = spectrum_data.coords
                spectrum_data.spectrum = self.hypercube[y, x, :]

    def export_spectra(self):
        """Export all spectra to file"""
        if not self.spectra_list:
            QMessageBox.warning(self, "Warning", "No spectra to export!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Spectra", "spectra.npy",
                                                   "NumPy Files (*.npy);;CSV Files (*.csv);;All Files (*.*)")
        
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.npy':
                    # Save as numpy array
                    spectra_array = np.array([s.spectrum for s in self.spectra_list])
                    np.save(file_path, spectra_array)
                    
                elif ext == '.csv':
                    # Save as CSV
                    spectra_array = np.array([s.spectrum for s in self.spectra_list])
                    np.savetxt(file_path, spectra_array, delimiter=',',
                              header=','.join([f'Band_{i}' for i in range(spectra_array.shape[1])]))
                
                else:
                    # Default to npy
                    spectra_array = np.array([s.spectrum for s in self.spectra_list])
                    np.save(file_path, spectra_array)
                
                QMessageBox.information(self, "Success", f"Exported {len(self.spectra_list)} spectra to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export spectra:\n{str(e)}")
    
    def show_about(self):
        """Show about dialog"""
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

    def load_reference_spectra(self):
        """Load reference spectra from file (CSV, XLSX, MAT, NPY, TXT)
        
        Expected format:
        - First row = wavelengths, remaining rows = spectra (each row starts with dye name)
          OR
        - First column = wavelengths, first row = dye names, rest = intensity data
        """
        file_types = "All Supported (*.csv *.xlsx *.mat *.npy *.txt);;CSV Files (*.csv);;Excel Files (*.xlsx);;MAT Files (*.mat);;NumPy Files (*.npy);;Text Files (*.txt);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Reference Spectra", "", file_types)

        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()
            spectra = None
            labels = []
            wavelengths = None

            if ext == '.csv':
                wavelengths, spectra, labels = self._load_csv_spectra(file_path)

            elif ext == '.txt':
                wavelengths, spectra, labels = self._load_txt_spectra(file_path)

            elif ext == '.xlsx':
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path)
                    # Check if first row looks like wavelengths (all numeric)
                    first_row_numeric = all(self._is_numeric(v) for v in df.iloc[0].values)
                    if first_row_numeric:
                        # First row = wavelengths, first column = dye names
                        wavelengths = df.iloc[0, 1:].values.astype(float)
                        labels = [str(df.iloc[i, 0]) for i in range(1, df.shape[0])]
                        spectra = df.iloc[1:, 1:].values.astype(float)
                    else:
                        # First column = wavelengths, first row = dye names
                        wavelengths = df.iloc[:, 0].values.astype(float)
                        labels = [str(c) for c in df.columns[1:]]
                        spectra = df.iloc[:, 1:].values.T.astype(float)
                except ImportError:
                    QMessageBox.critical(self, "Error", "Please install pandas for Excel file support:\npip install pandas")
                    return

            elif ext == '.mat':
                if not HAS_SCIPY:
                    QMessageBox.critical(self, "Error", "Please install scipy for MAT file support:\npip install scipy")
                    return
                mat_data = sio.loadmat(file_path)
                # Try to find wavelengths and spectra
                wl_key = None
                spec_key = None
                for key in mat_data:
                    if key.startswith('_'):
                        continue
                    val = mat_data[key]
                    if not isinstance(val, np.ndarray):
                        continue
                    if val.ndim == 1 and wl_key is None:
                        wl_key = key
                    elif val.ndim == 2 and spec_key is None:
                        spec_key = key
                if spec_key:
                    spectra = mat_data[spec_key]
                    if wl_key:
                        wavelengths = mat_data[wl_key].flatten().astype(float)
                    else:
                        wavelengths = np.arange(1, spectra.shape[1] + 1).astype(float)
                    labels = [f"Spectrum {i+1}" for i in range(spectra.shape[0])]
                # Look for label array
                for key in mat_data:
                    if key.startswith('_'):
                        continue
                    val = mat_data[key]
                    if isinstance(val, np.ndarray) and val.dtype.kind in ['U', 'S', 'O']:
                        labels = [str(v) for v in val.flatten()]
                        break

            elif ext == '.npy':
                data = np.load(file_path, allow_pickle=True)
                if isinstance(data, dict):
                    # Dictionary with 'wavelengths', 'spectra', 'labels'
                    if 'spectra' in data:
                        spectra = np.array(data['spectra'])
                    if 'wavelengths' in data:
                        wavelengths = np.array(data['wavelengths'], dtype=float)
                    if 'labels' in data:
                        labels = [str(l) for l in data['labels']]
                elif isinstance(data, np.ndarray) and data.ndim == 2:
                    # If first row is numeric (wavelengths), rest are spectra
                    first_row = data[0, :]
                    if np.issubdtype(first_row.dtype, np.number):
                        wavelengths = first_row
                        spectra = data[1:, :]
                        labels = [f"Spectrum {i+1}" for i in range(spectra.shape[0])]
                    else:
                        spectra = data
                        wavelengths = np.arange(1, spectra.shape[1] + 1).astype(float)
                        labels = [f"Spectrum {i+1}" for i in range(spectra.shape[0])]
                else:
                    QMessageBox.critical(self, "Error", "NPY file must contain a 2D array or a dict with 'spectra', 'wavelengths', and 'labels'.")
                    return
            else:
                QMessageBox.critical(self, "Error", f"Unsupported file format: {ext}")
                return

            if spectra is None or spectra.ndim != 2:
                QMessageBox.critical(self, "Error", "Could not parse spectra from file. Ensure it's a 2D array.")
                return

            # Ensure wavelengths match
            if wavelengths is None or len(wavelengths) != spectra.shape[1]:
                wavelengths = np.arange(1, spectra.shape[1] + 1).astype(float)

            if not labels or len(labels) != spectra.shape[0]:
                labels = [f"Spectrum {i+1}" for i in range(spectra.shape[0])]

            self.reference_spectra = spectra
            self.reference_labels = labels
            self.reference_wavelengths = wavelengths

            # Reset transformed data when loading new reference spectra
            self.transformed_spectra = None
            self.transformed_labels = []
            self.transformed_wavelengths = None
            self.save_transform_btn.setEnabled(False)

            # Update UI
            n_spectra = len(self.reference_labels)
            wl_range = f"{wavelengths[0]:.1f}–{wavelengths[-1]:.1f}"
            self.ref_status_label.setText(
                f"Reference Spectra: {n_spectra} loaded (λ: {wl_range})"
            )
            self.ref_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.transform_btn.setEnabled(True)

            # Show reference spectra window
            self._show_reference_window()

            QMessageBox.information(self, "Success",
                                   f"Loaded {n_spectra} reference spectra:\n" +
                                   "\n".join(f"  • {label}" for label in self.reference_labels))

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error",
                               f"Failed to load reference spectra:\n{str(e)}\n\n{traceback.format_exc()}")

    def _is_numeric(self, val):
        """Check if a value can be converted to float"""
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    def _load_csv_spectra(self, file_path):
        """Load spectra from CSV file.
        
        Format A: First row = wavelengths, first column = dye names
        Format B: First column = wavelengths, first row = dye names
        """
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 2:
            return None, None, []

        # Parse header
        header = lines[0].split(',')
        header_numeric = [self._is_numeric(v.strip()) for v in header]
        
        # Check if first row is wavelengths (all except first column are numeric)
        first_row_all_numeric = all(self._is_numeric(v.strip()) for v in lines[0].split(','))
        
        # Check if first column is wavelengths (all except header are numeric)
        first_col_all_numeric = all(
            self._is_numeric(line.split(',')[0].strip()) 
            for line in lines[1:]
        )

        wavelengths = None
        spectra = None
        labels = []

        if first_col_all_numeric and len(header) > 1:
            # Format B: First column = wavelengths, first row = dye names
            wavelengths = np.array([float(lines[i].split(',')[0]) for i in range(1, len(lines))])
            labels = [h.strip() for h in header[1:]]
            spectra = np.array([
                [float(v) for v in lines[i].split(',')[1:]]
                for i in range(1, len(lines))
            ]).T  # Transpose to get (n_spectra, n_bands)
        elif first_row_all_numeric:
            # Format A: First row = wavelengths, first column = dye names
            wavelengths = np.array([float(v.strip()) for v in lines[0].split(',')])
            labels = []
            data_rows = []
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) > 1:
                    labels.append(parts[0].strip())
                    data_rows.append([float(v) for v in parts[1:]])
            spectra = np.array(data_rows)
        else:
            # Fallback: no wavelength column, treat all as spectra
            raise ValueError("CSV format not recognized. First row should be wavelengths or first column should be wavelengths.")

        return wavelengths, spectra, labels

    def _load_txt_spectra(self, file_path):
        """Load spectra from TXT file (whitespace separated).
        
        Same logic as CSV but with whitespace delimiter.
        """
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 2:
            return None, None, []

        header = lines[0].split()
        
        # Check if first column is wavelengths
        first_col_all_numeric = all(
            self._is_numeric(line.split()[0].strip()) 
            for line in lines[1:]
        )

        first_row_all_numeric = all(self._is_numeric(v.strip()) for v in lines[0].split())

        wavelengths = None
        spectra = None
        labels = []

        if first_col_all_numeric and len(header) > 1:
            # First column = wavelengths, first row = dye names
            wavelengths = np.array([float(lines[i].split()[0]) for i in range(1, len(lines))])
            labels = [h.strip() for h in header[1:]]
            spectra = np.array([
                [float(v) for v in lines[i].split()[1:]]
                for i in range(1, len(lines))
            ]).T
        elif first_row_all_numeric:
            # First row = wavelengths, first column = dye names
            wavelengths = np.array([float(v.strip()) for v in lines[0].split()])
            labels = []
            data_rows = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) > 1:
                    labels.append(parts[0].strip())
                    data_rows.append([float(v) for v in parts[1:]])
            spectra = np.array(data_rows)
        else:
            raise ValueError("TXT format not recognized. First row should be wavelengths or first column should be wavelengths.")

        return wavelengths, spectra, labels

    def _show_reference_window(self):
        """Show reference spectra in a separate window"""
        if self.reference_spectra is None:
            return

        if self.reference_window is None:
            self.reference_window = SpectraDisplayWindow(
                "Reference Spectra Library",
                self.reference_wavelengths,
                self.reference_spectra,
                self.reference_labels
            )
            self.reference_window.show()
        else:
            self.reference_window.update_data(
                self.reference_wavelengths,
                self.reference_spectra,
                self.reference_labels
            )
            self.reference_window.show()
            self.reference_window.raise_()
            self.reference_window.activateWindow()

    def transform_spectra(self):
        """Transform reference spectra using current selection's average spectrum"""
        if self.reference_spectra is None:
            QMessageBox.warning(self, "Warning", "No reference spectra loaded!")
            return

        if self.current_selection is None:
            QMessageBox.warning(self, "Warning",
                               "Make a selection on the canvas first (point, circle, or square).\n"
                               "The average spectrum from this selection will be used for transformation.")
            return

        # Get average spectrum I from current selection
        I = self.current_selection['spectrum']  # 1D array of length n_bands_hyperspectral
        wavelengths = self.reference_wavelengths  # 1D array of length n_bands_reference

        # Verify dimensions match
        if len(I) != len(wavelengths):
            QMessageBox.warning(self, "Warning",
                               f"Wavelength mismatch: Selection has {len(I)} bands, "
                               f"reference spectra have {len(wavelengths)} bands.\n"
                               f"Transformation requires matching dimensions.")
            return

        # Transform each reference spectrum J
        n_ref = self.reference_spectra.shape[0]
        transformed = np.zeros((n_ref, len(I)))

        # Precompute cos^2 matrix: (n_ref, n_bands, n_bands)
        # T[i, k] = sum_j cos(pi * I[k] / lambda_j)^2 * J[i, j]
        # Vectorized: for each reference spectrum i, for each output position k:
        # T[i, k] = sum_j [cos(pi * I[k] / lambda_j)^2 * J[i, j]]

        # Compute the cos^2 weight matrix: (n_bands_output, n_bands_input)
        # I[k] for each k, lambda_j for each j
        # cos_squared[k, j] = cos(pi * I[k] / lambda_j)^2
        I_expanded = I[:, np.newaxis]  # (n_bands, 1)
        lambda_expanded = wavelengths[np.newaxis, :]  # (1, n_bands)
        cos_squared = np.cos(np.pi * I_expanded / lambda_expanded) ** 2  # (n_bands, n_bands)

        # T[i, :] = cos_squared @ J[i, :]
        # (n_ref, n_bands) = (n_bands, n_bands) @ (n_bands, n_ref).T -> (n_bands, n_ref)
        transformed = (cos_squared @ self.reference_spectra.T).T  # (n_ref, n_bands)

        # Build transformed labels
        transformed_labels = [f"Transformed: {self.reference_labels[i]}" for i in range(n_ref)]

        # Store transformed data
        self.transformed_spectra = transformed
        self.transformed_labels = transformed_labels
        self.transformed_wavelengths = wavelengths.copy()

        # Add transformed spectra to the main plot
        transform_colors = ['magenta', 'cyan', 'orange', 'lime', 'pink', 'brown', 'navy', 'teal']
        for i in range(n_ref):
            spectrum_data = SpectrumData(
                spectrum=transformed[i],
                label=transformed_labels[i],
                color=transform_colors[i % len(transform_colors)],
                selection_type="transformed",
                coords=None
            )
            self.spectra_list.append(spectrum_data)

        self.update_spectrum_plot()
        self.update_checkbox_list()
        self.delete_btn.setEnabled(True)

        # Enable save button
        self.save_transform_btn.setEnabled(True)

        # Show transformed spectra in a separate window
        self._show_transformed_window(wavelengths, transformed, transformed_labels)

        QMessageBox.information(self, "Transform Complete",
                               f"Transformed {n_ref} reference spectra using current selection.")

    def _show_transformed_window(self, wavelengths, transformed, labels):
        """Show transformed spectra in a separate window"""
        self.transform_window = SpectraDisplayWindow(
            "Transformed Spectra",
            wavelengths,
            transformed,
            labels
        )
        self.transform_window.show()
        self.transform_window.raise_()
        self.transform_window.activateWindow()

    def save_transformed_spectra(self):
        """Save transformed spectra to file"""
        if self.transformed_spectra is None:
            QMessageBox.warning(self, "Warning", "No transformed spectra to save!\nPerform a transformation first.")
            return

        file_types = "CSV Files (*.csv);;NumPy Files (*.npy);;Excel Files (*.xlsx);;Text Files (*.txt);;MAT Files (*.mat)"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Transformed Spectra", "transformed_spectra.csv", file_types)

        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == '.csv':
                # CSV: first column = dye names, rest = spectral bands
                # Prepend wavelength as first row if available
                with open(file_path, 'w') as f:
                    if self.transformed_wavelengths is not None:
                        wl_str = ','.join([f"{v:.4f}" for v in self.transformed_wavelengths])
                        f.write(f",{wl_str}\n")
                    for i, (spectrum, label) in enumerate(zip(self.transformed_spectra, self.transformed_labels)):
                        vals = ','.join([f"{v:.6f}" for v in spectrum])
                        f.write(f"{label},{vals}\n")

            elif ext == '.txt':
                # TXT: whitespace separated, first column = dye names
                with open(file_path, 'w') as f:
                    if self.transformed_wavelengths is not None:
                        wl_str = '  '.join([f"{v:.4f}" for v in self.transformed_wavelengths])
                        f.write(f"  {wl_str}\n")
                    for i, (spectrum, label) in enumerate(zip(self.transformed_spectra, self.transformed_labels)):
                        vals = '  '.join([f"{v:.6f}" for v in spectrum])
                        f.write(f"{label}  {vals}\n")

            elif ext == '.npy':
                # Save as dict with wavelengths, spectra, labels
                save_dict = {
                    'spectra': self.transformed_spectra,
                    'labels': np.array(self.transformed_labels),
                }
                if self.transformed_wavelengths is not None:
                    save_dict['wavelengths'] = self.transformed_wavelengths
                np.save(file_path, save_dict)

            elif ext == '.xlsx':
                try:
                    import pandas as pd
                    # Build DataFrame: first column = dye names, rest = spectral bands
                    data = {'Dye': self.transformed_labels}
                    n_bands = self.transformed_spectra.shape[1]
                    for j in range(n_bands):
                        col_name = f"{self.transformed_wavelengths[j]:.2f}" if self.transformed_wavelengths is not None else f"Band_{j}"
                        data[col_name] = self.transformed_spectra[:, j]
                    df = pd.DataFrame(data)
                    df.to_excel(file_path, index=False)
                except ImportError:
                    QMessageBox.critical(self, "Error", "Please install pandas for Excel file support:\npip install pandas")
                    return

            elif ext == '.mat':
                if not HAS_SCIPY:
                    QMessageBox.critical(self, "Error", "Please install scipy for MAT file support:\npip install scipy")
                    return
                save_dict = {
                    'spectra': self.transformed_spectra,
                    'labels': np.array(self.transformed_labels),
                }
                if self.transformed_wavelengths is not None:
                    save_dict['wavelengths'] = self.transformed_wavelengths
                sio.savemat(file_path, save_dict)

            else:
                QMessageBox.critical(self, "Error", f"Unsupported file format: {ext}")
                return

            QMessageBox.information(self, "Success",
                                   f"Saved {len(self.transformed_labels)} transformed spectra to:\n{file_path}")

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error",
                               f"Failed to save transformed spectra:\n{str(e)}\n\n{traceback.format_exc()}")


class SpectraDisplayWindow(QMainWindow):
    """Separate window to display spectra with wavelength axis"""
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

        # Figure
        self.fig = Figure(figsize=(9, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True, alpha=0.3)

        layout.addWidget(self.canvas)

        # Toolbar buttons
        btn_layout = QHBoxLayout()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _plot_spectra(self):
        """Plot all spectra with different colors using wavelength axis"""
        self.ax.clear()

        color_cycle = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange',
                       'purple', 'brown', 'pink', 'lime', 'teal', 'navy',
                       'olive', 'coral', 'gold', 'crimson', 'indigo']

        for i, (spectrum, label) in enumerate(zip(self.spectra, self.labels)):
            color = color_cycle[i % len(color_cycle)]
            self.ax.plot(self.wavelengths, spectrum, color=color, linewidth=2, label=label)

        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel('Intensity')
        self.ax.set_title(self.windowTitle())
        self.ax.legend(loc='best', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def update_data(self, wavelengths, spectra, labels):
        """Update with new spectra data"""
        self.setWindowTitle(self.windowTitle())  # Keep existing title
        self.wavelengths = wavelengths
        self.spectra = spectra
        self.labels = labels
        self._plot_spectra()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    viewer = HyperViewer()
    viewer.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
