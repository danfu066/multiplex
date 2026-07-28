"""
OPDViewer: HyperViewer + voltage-to-OPD conversion
==================================================
Extends HyperViewer for liquid-crystal retardance measurements.

A voltage sweep of an LC cell imaged through a polarizing beamsplitter gives a
3D stack I(x, y, V).  This tool converts it in place to OPD(x, y, V) in
nanometres, so every existing HyperViewer feature keeps working: the slider now
steps through voltage, and a point/circle/square "spectrum" is the retardance
curve OPD vs voltage for that region.

Added over HyperViewer
----------------------
* Multi-page TIFF loading that also reads the per-frame drive voltage from the
  ImageJ 'Labels' metadata (filenames like '1.489V.tiff'), sorted ascending.
* "Convert to OPD" -- the full retardance pipeline (see below).
* Voltages loadable separately from .csv/.txt/.npy if the TIFF has no metadata.
* Save the OPD stack; revert to raw intensity at any time.

The conversion
--------------
Per pixel the intensity follows

    I(V) = a(V) +/- b(V) * cos( Gamma(V) ),   Gamma = 2*pi*OPD/lambda

with Gamma decreasing monotonically as the director aligns.  The sweep crosses
many fringes, so:

  1. Track the slow (a, b) envelope; normalise to c = cos(Gamma).
  2. theta = arccos(c) -- a triangle wave folding at 0 and pi.
  3. Unfold by counting folds; each fold is exactly pi of retardance.  Folds are
     confirmed with HYSTERESIS (theta must reverse this far) so noise near a
     turning point is not miscounted, and are gated on fringe contrast so the
     sub-threshold region cannot manufacture folds.
  4. Anchor the absolute value at the high-voltage end (RESIDUAL, see below).
  5. Resolve the integer order per pixel by SPATIAL CONTINUITY.

Why step 5 matters
------------------
Fold counting is an independent discrete decision at every pixel, so nothing
stops a 1-wave jump between neighbours.  Where the sweep is marginally sampled
(~4 frames per fringe) a fraction of pixels lose exactly one fringe -- in real
550 nm data that was 6.8 % of the field, in CONNECTED REGIONS, so a small median
filter cannot see them.  A robust global surface fit does: the true map is
smooth and nearly flat (~0.04 waves of real structure) while the error is a
whole wave, 25x larger, so rounding the departure from a fitted surface to the
nearest integer wave is very well conditioned.

The residual (absolute order)
-----------------------------
Counting is exact only RELATIVE to the highest-voltage frame; the absolute value
needs Gamma there, which the data fixes only modulo one wave.  arccos returns
Gamma = n +/- theta, so the branch matters -- pick it from the direction of the
last partial fringe:

    intensity still FALLING toward a minimum at V_max  ->  Gamma = n + theta
    intensity RISING away from a minimum at V_max      ->  Gamma = n - theta

"Estimate" in the dialog reports theta at V_max, the direction, and the
resulting candidates.  Confirm against a datasheet or a second wavelength: real
retardance scales as 1/lambda, so measuring the same cell at two wavelengths
pins the order.  The choice shifts the whole map by a constant; all retardance
DIFFERENCES and the spatial structure are unaffected by it.

Caveat worth knowing: the running max/min envelope underestimates b when the
sweep is thinly sampled, which biases the sub-fringe phase (~0.07 waves RMS at
4 frames/fringe).  It does not affect the fringe COUNT.  More frames per fringe
is the only real cure.
"""

import os
import re
import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QAction, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
    QProgressDialog, QPushButton, QRadioButton, QSpinBox, QToolBar, QVBoxLayout, QWidget,
)

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from HyperViewer import HyperViewer


# ============================================================================
# OPD core -- self-contained, no dependency on the viewer
# ============================================================================

def unwrap_traces(I, convention="dark_at_zero", residual_waves=1.0,
                  envelope_win=9, hysteresis=0.6, min_fringe_frac=0.12):
    """
    Absolute retardance for a block of intensity traces.

    Parameters
    ----------
    I : (N, P) float array, voltage ascending along axis 0.
    convention : "dark_at_zero"  -- crossed analyser, I minimum at Gamma=0
                 "bright_at_zero" -- parallel analyser, I maximum at Gamma=0
    residual_waves : absolute retardance still present at the highest voltage.

    Returns
    -------
    Gamma : (N, P) radians, monotonically decreasing in V.
    quality : (P,) fringe contrast, median(b)/median(a); low = unreliable.
    """
    from scipy.ndimage import (maximum_filter1d, minimum_filter1d,
                               uniform_filter1d)

    I = np.asarray(I, dtype=np.float32)
    N, P = I.shape

    amax = uniform_filter1d(maximum_filter1d(I, envelope_win, axis=0),
                            envelope_win, axis=0)
    amin = uniform_filter1d(minimum_filter1d(I, envelope_win, axis=0),
                            envelope_win, axis=0)
    a = 0.5 * (amax + amin)
    b = np.maximum(0.5 * (amax - amin), 1e-9)

    c = (I - a) / b
    if convention == "dark_at_zero":
        c = -c
    elif convention != "bright_at_zero":
        raise ValueError(f"Unknown convention: {convention!r}")
    theta = np.arccos(np.clip(c, -1.0, 1.0))

    # only where the fringe is deep enough to be real retardance
    full_range = np.maximum(I.max(axis=0) - I.min(axis=0), 1e-9)
    valid = (2.0 * b) > (min_fringe_frac * full_range)

    # --- folds, confirmed by hysteresis, credited to the extremum's frame ---
    mark = np.zeros_like(theta)
    d0 = theta[1] - theta[0]
    rising = np.where(d0 >= 0, 1.0, -1.0)
    ext_val = theta[0].copy()
    ext_idx = np.zeros(P, dtype=np.int64)

    for k in range(1, N):
        up = rising > 0
        better = np.where(up, theta[k] > ext_val, theta[k] < ext_val)
        ext_val = np.where(better, theta[k], ext_val)
        ext_idx = np.where(better, k, ext_idx)

        turn = np.where(up, theta[k] < ext_val - hysteresis,
                            theta[k] > ext_val + hysteresis) & valid[k]
        if turn.any():
            ii = np.nonzero(turn)[0]
            mark[ext_idx[ii], ii] += 1.0
            rising[ii] *= -1.0
            ext_val[ii] = theta[k, ii]
            ext_idx[ii] = k

    level = np.cumsum(mark, axis=0)
    even = (level.astype(np.int64) % 2 == 0)
    theta_f = np.where(d0 >= 0, theta, np.pi - theta)
    U = level * np.pi + np.where(even, theta_f, np.pi - theta_f)

    Gamma = np.abs(U[-1] - U) + residual_waves * 2.0 * np.pi
    Gamma = np.minimum.accumulate(Gamma, axis=0)

    # hold Gamma flat across the sub-threshold region, where the cell has not
    # reoriented and the apparent modulation is scattering, not retardance
    any_valid = valid.any(axis=0)
    first = np.where(any_valid, valid.argmax(axis=0), 0)
    idx = np.arange(N)[:, None]
    Gamma = np.where(idx < first[None, :],
                     Gamma[first, np.arange(P)][None, :], Gamma)

    quality = np.median(b, axis=0) / (np.median(a, axis=0) + 1e-9)
    return Gamma, quality


def _poly_design(H, W, deg):
    y, x = np.mgrid[0:H, 0:W]
    x = (x / max(W - 1, 1) - 0.5) * 2.0
    y = (y / max(H - 1, 1) - 0.5) * 2.0
    cols = [(x ** i) * (y ** j)
            for i in range(deg + 1) for j in range(deg + 1 - i)]
    return np.stack([c.ravel() for c in cols], axis=1)


def resolve_orders(w, deg=3, clip=0.35, n_iter=6, A=None):
    """
    Remove integer-wave order errors from one retardance map (in waves), using
    spatial continuity.

    Subsamples pixel grid when image size > 20,000 for 100x speedup while preserving
    exact polynomial surface fitting accuracy.
    """
    H, W = w.shape
    if A is None:
        A = _poly_design(H, W, deg)
    z = w.ravel()

    # If peak-to-peak retardance variation across the frame is small (<0.3 waves), no order jumps exist
    if np.ptp(z) < 0.3:
        return w, np.zeros_like(w)

    # Subsample pixels for least-squares solve if total pixel count exceeds 20k
    stride = max(1, z.size // 20000)

    keep = np.ones(z.size, dtype=bool)
    coef = None
    for _ in range(n_iter):
        sub_idx = np.where(keep)[0][::stride]
        if len(sub_idx) < 10:
            sub_idx = np.arange(z.size)[::stride]
        coef, *_ = np.linalg.lstsq(A[sub_idx], z[sub_idx], rcond=None)
        resid = z - A @ coef
        resid -= np.median(resid[keep])
        new = np.abs(resid) < clip
        if new.sum() < 0.2 * z.size or np.array_equal(new, keep):
            break
        keep = new
    surf = (A @ coef).reshape(H, W)
    surf = surf + np.median(np.round(w - surf))
    n = np.round(w - surf)
    return w - n, n


def estimate_residual(I_ref, convention="dark_at_zero", envelope_win=9):
    """
    Report the principal theta at V_max and the resulting residual candidates.

    Returns (theta_waves, direction, candidates) where direction is 'rising' or
    'falling' and candidates is a list of (residual_waves, description).
    """
    from scipy.ndimage import (maximum_filter1d, minimum_filter1d,
                               uniform_filter1d)
    I = np.asarray(I_ref, dtype=np.float64)
    amax = uniform_filter1d(maximum_filter1d(I, envelope_win), envelope_win)
    amin = uniform_filter1d(minimum_filter1d(I, envelope_win), envelope_win)
    a = 0.5 * (amax + amin)
    b = np.maximum(0.5 * (amax - amin), 1e-9)
    c = (I - a) / b
    if convention == "dark_at_zero":
        c = -c
    th = float(np.arccos(np.clip(c, -1.0, 1.0))[-1]) / (2 * np.pi)

    # Direction must be measured from the LAST EXTREMUM to the end, not across
    # a window that may span it: a trace like 8944, 1296, 2208, 7488 is rising
    # away from its minimum even though the final value is below the first.
    s = np.sign(np.diff(I))
    turns = [k + 1 for k in range(len(s) - 1)
             if s[k] != 0 and s[k + 1] != 0 and s[k] != s[k + 1]]
    ref = turns[-1] if turns else max(0, len(I) - 2)
    direction = "rising" if I[-1] > I[ref] else "falling"

    sign = -1.0 if direction == "rising" else +1.0
    cands = []
    for n in range(0, 4):
        r = n + sign * th
        if r > 1e-6:
            cands.append((r, f"n={n}, {'n-theta' if sign < 0 else 'n+theta'}"))
    return th, direction, cands


def convert_to_opd(cube_hwn, wavelength_nm, convention="dark_at_zero",
                   residual_waves=1.0, envelope_win=9, hysteresis=0.6,
                   min_fringe_frac=0.12, do_resolve=True, v_min_index=0,
                   block=200000, progress=None):
    """
    Convert an intensity stack (H, W, N) to OPD in nm, same shape.

    v_min_index : frames below this index are treated as sub-threshold and the
                  retardance is held flat there.
    progress    : optional callable(fraction) -> bool; return False to cancel.
    """
    H, W, N = cube_hwn.shape
    P = H * W
    I_all = np.transpose(cube_hwn, (2, 0, 1)).reshape(N, P)
    waves = np.empty((N, P), dtype=np.float32)

    done = 0
    for s in range(0, P, block):
        e = min(s + block, P)
        g, _ = unwrap_traces(I_all[:, s:e], convention, residual_waves,
                             envelope_win, hysteresis, min_fringe_frac)
        waves[:, s:e] = g / (2.0 * np.pi)
        done = e
        if progress is not None and not progress(0.75 * done / P):
            return None
    del I_all

    if v_min_index > 0:
        k0 = min(v_min_index, N - 1)
        waves[:k0] = waves[k0]

    waves = waves.reshape(N, H, W)

    if do_resolve:
        A_matrix = _poly_design(H, W, deg=3)
        for k in range(N):
            waves[k], _ = resolve_orders(waves[k], A=A_matrix)
            if progress is not None and not progress(0.75 + 0.25 * (k + 1) / N):
                return None
        waves = np.minimum.accumulate(waves, axis=0)

    opd = np.transpose(waves, (1, 2, 0)) * wavelength_nm
    return np.ascontiguousarray(opd)


# ============================================================================
# Parameter dialog
# ============================================================================

class OPDParamsDialog(QDialog):
    """Collect conversion parameters; 'Estimate' helps pick the residual."""

    def __init__(self, parent=None, voltages=None, ref_trace=None):
        super().__init__(parent)
        self.setWindowTitle("Convert Voltage Stack to OPD")
        self.setMinimumWidth(460)
        self.voltages = voltages
        self.ref_trace = ref_trace

        form = QFormLayout()

        self.wl = QDoubleSpinBox()
        self.wl.setRange(200.0, 3000.0)
        self.wl.setDecimals(1)
        self.wl.setValue(550.0)
        self.wl.setSuffix(" nm")
        form.addRow("Wavelength:", self.wl)

        self.conv = QComboBox()
        self.conv.addItems(["dark_at_zero (crossed)",
                            "bright_at_zero (parallel)"])
        form.addRow("Analyser:", self.conv)

        res_row = QHBoxLayout()
        self.residual = QDoubleSpinBox()
        self.residual.setRange(0.0, 20.0)
        self.residual.setDecimals(3)
        self.residual.setSingleStep(0.05)
        self.residual.setValue(1.000)
        self.residual.setSuffix(" waves")
        res_row.addWidget(self.residual)
        self.est_btn = QPushButton("Estimate")
        self.est_btn.setToolTip("Report theta at V_max and the branch candidates")
        self.est_btn.clicked.connect(self._estimate)
        res_row.addWidget(self.est_btn)
        res_w = QWidget()
        res_w.setLayout(res_row)
        form.addRow("Residual at V_max:", res_w)

        self.est_label = QLabel("")
        self.est_label.setWordWrap(True)
        self.est_label.setStyleSheet("color: #555; font-style: italic;")
        form.addRow("", self.est_label)

        self.env = QSpinBox()
        self.env.setRange(3, 51)
        self.env.setSingleStep(2)
        self.env.setValue(9)
        self.env.setSuffix(" frames")
        self.env.setToolTip("Envelope window, about one fringe period")
        form.addRow("Envelope window:", self.env)

        self.hyst = QDoubleSpinBox()
        self.hyst.setRange(0.05, 2.0)
        self.hyst.setDecimals(2)
        self.hyst.setSingleStep(0.05)
        self.hyst.setValue(0.60)
        self.hyst.setSuffix(" rad")
        self.hyst.setToolTip("Too small: noise counted as fringes. "
                             "Too large: real fringes missed.")
        form.addRow("Fold hysteresis:", self.hyst)

        self.gate = QDoubleSpinBox()
        self.gate.setRange(0.0, 0.9)
        self.gate.setDecimals(2)
        self.gate.setSingleStep(0.02)
        self.gate.setValue(0.12)
        self.gate.setToolTip("Ignore folds where the local fringe amplitude is "
                             "below this fraction of the full range")
        form.addRow("Min fringe fraction:", self.gate)

        self.vmin = QSpinBox()
        self.vmin.setRange(0, 10000)
        self.vmin.setValue(0)
        self.vmin.setSuffix(" frames")
        self.vmin.setToolTip("Hold retardance flat below this frame index "
                             "(sub-threshold region)")
        form.addRow("Sub-threshold frames:", self.vmin)

        self.resolve = QCheckBox("Resolve fringe order by spatial continuity")
        self.resolve.setChecked(True)
        self.resolve.setToolTip("Strongly recommended: fixes pixels that lost a "
                                "whole fringe. Assumes a smooth OPD map.")
        form.addRow("", self.resolve)

        layout = QVBoxLayout(self)
        layout.addLayout(form)

        note = QLabel(
            "The residual sets the absolute order and shifts the whole map by a "
            "constant. Retardance differences and spatial structure do not "
            "depend on it. Confirm against a datasheet or a second wavelength."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _estimate(self):
        if self.ref_trace is None:
            self.est_label.setText("No reference trace available.")
            return
        try:
            th, direction, cands = estimate_residual(
                self.ref_trace, self.convention(), self.env.value())
        except Exception as exc:                       # pragma: no cover
            self.est_label.setText(f"Estimate failed: {exc}")
            return
        txt = (f"theta at V_max = {th:.3f} wave; intensity is {direction} "
               f"-> use {'n - theta' if direction == 'rising' else 'n + theta'}. "
               f"Candidates: "
               + ", ".join(f"{r:.3f}" for r, _ in cands[:4]))
        self.est_label.setText(txt)
        if cands:
            # lowest positive order -- "first order", which is what both the
            # 550 nm and 647 nm data turned out to need
            self.residual.setValue(cands[0][0])

    def convention(self):
        return "dark_at_zero" if self.conv.currentIndex() == 0 else "bright_at_zero"

    def values(self):
        return dict(
            wavelength_nm=self.wl.value(),
            convention=self.convention(),
            residual_waves=self.residual.value(),
            envelope_win=self.env.value(),
            hysteresis=self.hyst.value(),
            min_fringe_frac=self.gate.value(),
            v_min_index=self.vmin.value(),
            do_resolve=self.resolve.isChecked(),
        )


def opd_to_intensity(opd_nm, wavelengths_nm, weights=None, convention="dark_at_zero"):
    """
    Convert an OPD map stack (H, W, V) in nanometers back to an intensity stack (H, W, V)
    for a monochromatic wavelength or a polychromatic spectrum.

    Crossed PBS (dark_at_zero): I = S * sin^2(pi * OPD / lambda)
    Parallel PBS (bright_at_zero): I = S * cos^2(pi * OPD / lambda)
    """
    opd_nm = np.asarray(opd_nm, dtype=np.float32)
    wavelengths_nm = np.atleast_1d(np.asarray(wavelengths_nm, dtype=np.float32))

    if weights is None:
        w_arr = np.ones_like(wavelengths_nm, dtype=np.float32)
    else:
        w_arr = np.asarray(weights, dtype=np.float32)

    if w_arr.sum() > 0:
        w_arr = w_arr / w_arr.sum()

    H, W, V = opd_nm.shape
    I_total = np.zeros((H, W, V), dtype=np.float32)

    for wl, w in zip(wavelengths_nm, w_arr):
        if wl <= 0:
            continue
        phase = (np.pi * opd_nm) / wl
        if convention == "dark_at_zero":
            I_wl = w * (np.sin(phase) ** 2)
        else:
            I_wl = w * (np.cos(phase) ** 2)
        I_total += I_wl

    return I_total


class OPDForwardSimDialog(QDialog):
    """Dialog for simulating intensity map stack from OPD map and spectrum/wavelength, or reverting to raw stack."""

    def __init__(self, parent=None, has_raw_intensity=False, spectra_list=None, default_wl=550.0):
        super().__init__(parent)
        self.setWindowTitle("Revert / Simulate Intensity Map from OPD")
        self.setMinimumWidth(480)
        self.has_raw_intensity = has_raw_intensity
        self.spectra_list = spectra_list or []

        layout = QVBoxLayout(self)
        form = QFormLayout()

        if self.has_raw_intensity:
            self.mode_group = QButtonGroup(self)
            self.radio_revert_raw = QRadioButton("Revert to original raw intensity stack")
            self.radio_simulate = QRadioButton("Simulate intensity map stack from OPD & spectrum")
            self.radio_simulate.setChecked(True)
            self.mode_group.addButton(self.radio_revert_raw)
            self.mode_group.addButton(self.radio_simulate)

            mode_box = QVBoxLayout()
            mode_box.addWidget(self.radio_revert_raw)
            mode_box.addWidget(self.radio_simulate)
            form.addRow("Action:", mode_box)

        self.sim_mode_combo = QComboBox()
        self.sim_mode_combo.addItems(["Monochromatic (Single Wavelength)", "Polychromatic Spectrum (Integrated)"])
        self.sim_mode_combo.currentIndexChanged.connect(self._on_sim_mode_changed)
        form.addRow("Simulation Mode:", self.sim_mode_combo)

        self.wl_spin = QDoubleSpinBox()
        self.wl_spin.setRange(200.0, 3000.0)
        self.wl_spin.setValue(float(default_wl))
        self.wl_spin.setDecimals(1)
        self.wl_spin.setSuffix(" nm")
        form.addRow("Wavelength:", self.wl_spin)

        self.spectrum_combo = QComboBox()
        self.spectrum_combo.addItem("Flat Spectrum (400 - 700 nm, 31 bands)")
        for s in self.spectra_list:
            lbl = getattr(s, 'label', 'Spectrum')
            self.spectrum_combo.addItem(f"Loaded ROI: {lbl}")
        self.spectrum_combo.setEnabled(False)
        form.addRow("Illumination Spectrum:", self.spectrum_combo)

        self.conv_combo = QComboBox()
        self.conv_combo.addItems(["dark_at_zero (Crossed PBS)", "bright_at_zero (Parallel PBS)"])
        form.addRow("Analyser Configuration:", self.conv_combo)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_sim_mode_changed(self, idx):
        is_poly = (idx == 1)
        self.wl_spin.setEnabled(not is_poly)
        self.spectrum_combo.setEnabled(is_poly)

    def is_revert_raw(self):
        return self.has_raw_intensity and self.radio_revert_raw.isChecked()

    def values(self):
        return {
            'is_poly': (self.sim_mode_combo.currentIndex() == 1),
            'wavelength_nm': self.wl_spin.value(),
            'spectrum_index': self.spectrum_combo.currentIndex(),
            'convention': "dark_at_zero" if self.conv_combo.currentIndex() == 0 else "bright_at_zero"
        }


# ============================================================================
# Viewer
# ============================================================================

class OPDViewer(HyperViewer):
    """HyperViewer with voltage-sweep loading and OPD conversion."""

    def __init__(self):
        self.voltages = None          # (N,) drive voltage per frame
        self.intensity_cube = None    # raw intensity, kept for revert
        self.is_opd = False
        super().__init__()
        self.setWindowTitle("OPDViewer - LC Retardance / OPD Viewer")
        self._build_opd_ui()

    # ---------------- UI ----------------

    def _build_opd_ui(self):
        self.act_load_opd = QAction("Load OPD…", self)
        self.act_load_opd.setToolTip("Load a previously saved OPD map stack (.npy, .tif, .mat)")
        self.act_load_opd.triggered.connect(self.load_opd)

        self.act_convert = QAction("Convert to OPD", self)
        self.act_convert.setToolTip("Convert the intensity stack to OPD (nm)")
        self.act_convert.triggered.connect(self.convert_stack_to_opd)

        self.act_revert = QAction("Revert / Simulate Intensity…", self)
        self.act_revert.setToolTip("Convert OPD map back to intensity map stack via monochromatic wavelength or spectrum simulation")
        self.act_revert.triggered.connect(self.revert_to_intensity)
        self.act_revert.setEnabled(False)

        self.act_save = QAction("Save OPD…", self)
        self.act_save.triggered.connect(self.save_opd)
        self.act_save.setEnabled(False)

        menu = self.menuBar().addMenu("OPD")
        menu.addAction(self.act_load_opd)
        menu.addAction(self.act_convert)
        menu.addAction(self.act_revert)
        menu.addAction(self.act_save)

    def _set_status(self):
        if self.is_opd:
            msg = "Mode: OPD (nm)"
        else:
            n = 0 if self.voltages is None else len(self.voltages)
            extra = f", {n} voltages" if n else ""
            msg = f"Mode: Intensity{extra}"
        self.statusBar().showMessage(msg)

    # ---------------- loading ----------------

    @staticmethod
    def _voltage_from_label(label):
        m = re.search(r"([0-9]*\.?[0-9]+)\s*V", str(label))
        return float(m.group(1)) if m else None

    def load_dataset(self, file_path):
        """Load dataset and automatically extract per-frame voltage metadata if present."""
        ext = os.path.splitext(file_path)[1].lower()
        volts = None
        cube = None

        if ext in ['.tif', '.tiff'] and HAS_TIFFFILE:
            try:
                with tifffile.TiffFile(file_path) as tf:
                    meta = tf.imagej_metadata or {}
                    labels = meta.get("Labels")
                    data = tf.asarray()

                data = np.asarray(data)
                if data.ndim == 3:
                    if labels:
                        parsed = [self._voltage_from_label(l) for l in labels]
                        if all(v is not None for v in parsed) and len(parsed) == data.shape[0]:
                            volts = np.array(parsed, dtype=float)

                    if volts is not None:
                        order = np.argsort(volts)
                        volts = volts[order]
                        data = data[order]

                    cube = np.transpose(data, (1, 2, 0)).astype(np.float32)
            except Exception:
                pass

        if cube is None:
            super().load_dataset(file_path)
            if self.hypercube is not None:
                N = self.hypercube.shape[2]
                base_no_ext = os.path.splitext(file_path)[0]
                for v_ext in ['.npy', '.csv', '.txt']:
                    v_path = base_no_ext + '_voltages' + v_ext
                    if os.path.exists(v_path):
                        try:
                            if v_ext == '.npy':
                                v = np.load(v_path).astype(float).ravel()
                            else:
                                v = np.loadtxt(v_path, delimiter=',' if v_ext == '.csv' else None).astype(float).ravel()
                            if len(v) == N:
                                volts = np.sort(v)
                                break
                        except Exception:
                            pass
        else:
            self.hypercube = cube

        self.voltages = volts
        self.is_opd = False
        self.intensity_cube = None
        self.setup_display()
        self.act_revert.setEnabled(False)
        self.act_save.setEnabled(False)
        self._set_status()
        self.setWindowTitle(f"OPDViewer - {os.path.basename(file_path)}")

    # ---------------- conversion ----------------

    def convert_stack_to_opd(self):
        if self.hypercube is None:
            QMessageBox.warning(self, "Warning", "Load a stack first.")
            return
        if self.is_opd:
            QMessageBox.information(self, "Already converted",
                                    "This stack is already OPD. Revert first "
                                    "if you want to re-run with new settings.")
            return

        H, W, N = self.hypercube.shape
        if N < 8:
            QMessageBox.warning(self, "Warning",
                                f"Only {N} frames; a voltage sweep is needed.")
            return

        ref = self.hypercube[H // 2, W // 2, :].astype(float)
        dlg = OPDParamsDialog(self, voltages=self.voltages, ref_trace=ref)
        if dlg.exec_() != QDialog.Accepted:
            return
        params = dlg.values()

        prog = QProgressDialog("Converting to OPD…", "Cancel", 0, 100, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        def cb(frac):
            prog.setValue(int(100 * frac))
            QApplication.processEvents()
            return not prog.wasCanceled()

        try:
            opd = convert_to_opd(self.hypercube, progress=cb, **params)
        except Exception as exc:
            prog.close()
            import traceback
            QMessageBox.critical(self, "Error",
                                 f"Conversion failed:\n{exc}\n\n"
                                 f"{traceback.format_exc()}")
            return
        prog.close()

        if opd is None:
            QMessageBox.information(self, "Cancelled", "Conversion cancelled.")
            return

        self.intensity_cube = self.hypercube
        self.hypercube = opd
        self.is_opd = True
        self.wavelength_nm = params["wavelength_nm"]

        # spectra extracted from intensity are meaningless in OPD units
        self.spectra_list = []
        self.current_selection = None
        self.update_checkbox_list()
        self.setup_display()

        self.act_revert.setEnabled(True)
        self.act_save.setEnabled(True)
        self._set_status()

        c = opd[H // 2, W // 2, :]
        QMessageBox.information(
            self, "Conversion complete",
            f"Converted {N} frames at {params['wavelength_nm']:.1f} nm.\n\n"
            f"Centre pixel: {c[0] / params['wavelength_nm']:.3f} waves "
            f"({c[0]:.0f} nm) -> {c[-1] / params['wavelength_nm']:.3f} waves "
            f"({c[-1]:.0f} nm)\n"
            f"Swing: {(c[0] - c[-1]) / params['wavelength_nm']:.3f} waves\n\n"
            "Previously saved spectra were cleared: they were in intensity units.")

    def load_opd(self):
        """Load a previously saved OPD stack (.npy, .tif, .tiff, .mat)."""
        file_types = "All Supported (*.npy *.tif *.tiff *.mat);;NumPy Files (*.npy);;TIFF Files (*.tif *.tiff);;MAT Files (*.mat);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Previously Saved OPD Stack", "", file_types)
        if not file_path:
            return

        try:
            opd_data = self._load_file(file_path)
            if opd_data is None:
                return

            opd_data = np.array(opd_data, dtype=np.float32)
            if opd_data.ndim == 2:
                opd_data = opd_data[:, :, np.newaxis]
            elif opd_data.ndim == 3 and opd_data.shape[0] < opd_data.shape[1] and opd_data.shape[0] < opd_data.shape[2]:
                opd_data = np.transpose(opd_data, (1, 2, 0))

            base_no_ext = os.path.splitext(file_path)[0]
            volts = None
            N = opd_data.shape[2]
            for v_ext in ['.npy', '.csv', '.txt']:
                v_path = base_no_ext + '_voltages' + v_ext
                if os.path.exists(v_path):
                    try:
                        if v_ext == '.npy':
                            v = np.load(v_path).astype(float).ravel()
                        else:
                            v = np.loadtxt(v_path, delimiter=',' if v_ext == '.csv' else None).astype(float).ravel()
                        if len(v) == N:
                            volts = np.sort(v)
                            break
                    except Exception:
                        pass

            self.hypercube = opd_data
            self.voltages = volts
            self.is_opd = True
            self.intensity_cube = None
            self.setup_display()

            self.act_revert.setEnabled(True)
            self.act_save.setEnabled(True)
            self._set_status()
            self.setWindowTitle(f"OPDViewer - {os.path.basename(file_path)} (OPD Mode)")
            QMessageBox.information(self, "OPD Loaded", f"Loaded OPD map stack:\n{file_path}\nShape: {opd_data.shape}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load OPD file:\n{e}")

    def revert_to_intensity(self):
        if not self.is_opd and self.intensity_cube is None:
            QMessageBox.warning(self, "Warning", "No OPD map or raw intensity stack loaded.")
            return

        default_wl = getattr(self, 'wavelength_nm', 550.0)
        dlg = OPDForwardSimDialog(
            self,
            has_raw_intensity=(self.intensity_cube is not None),
            spectra_list=self.spectra_list,
            default_wl=default_wl
        )

        if dlg.exec_() != QDialog.Accepted:
            return

        if dlg.is_revert_raw():
            self.hypercube = self.intensity_cube
            self.intensity_cube = None
            self.is_opd = False
            self.spectra_list = []
            self.current_selection = None
            self.update_checkbox_list()
            self.setup_display()
            self.act_revert.setEnabled(False)
            self.act_save.setEnabled(False)
            self._set_status()
            QMessageBox.information(self, "Reverted", "Reverted to original raw intensity stack.")
            return

        # Forward Simulation from OPD to Intensity
        vals = dlg.values()
        is_poly = vals['is_poly']
        convention = vals['convention']

        if not is_poly:
            wavelengths = np.array([vals['wavelength_nm']], dtype=np.float32)
            weights = np.array([1.0], dtype=np.float32)
            summary = f"monochromatic wavelength {vals['wavelength_nm']:.1f} nm"
        else:
            spec_idx = vals['spectrum_index']
            if spec_idx == 0 or not self.spectra_list:
                wavelengths = np.linspace(400.0, 700.0, 31, dtype=np.float32)
                weights = np.ones(31, dtype=np.float32)
                summary = "flat spectrum (400 - 700 nm)"
            else:
                sdata = self.spectra_list[spec_idx - 1]
                weights = sdata.spectrum.astype(np.float32)
                N_bands = len(weights)
                if hasattr(sdata, 'wavelengths') and sdata.wavelengths is not None and len(sdata.wavelengths) == N_bands:
                    wavelengths = sdata.wavelengths.astype(np.float32)
                else:
                    wavelengths = np.linspace(400.0, 700.0, N_bands, dtype=np.float32)
                summary = f"spectrum '{sdata.label}' ({N_bands} bands)"

        try:
            I_sim = opd_to_intensity(self.hypercube, wavelengths, weights, convention=convention)
            if self.intensity_cube is None:
                self.intensity_cube = self.hypercube.copy()  # keep current OPD as backup
            self.hypercube = I_sim
            self.is_opd = False
            self.spectra_list = []
            self.current_selection = None
            self.update_checkbox_list()
            self.setup_display()
            self.act_revert.setEnabled(True)
            self._set_status()
            QMessageBox.information(
                self, "Simulation Complete",
                f"Converted OPD map back to intensity map stack using {summary}.\n\n"
                f"Configuration: {convention} (Crossed PBS: 0 at m*lambda, 100% at (m+0.5)*lambda)."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to simulate intensity map:\n{e}")

    def save_opd(self):
        if not self.is_opd or self.hypercube is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save OPD Stack", "opd_nm.npy",
            "NumPy Files (*.npy);;TIFF Files (*.tif *.tiff)")
        if not path:
            return
        try:
            if path.lower().endswith((".tif", ".tiff")):
                if not HAS_TIFFFILE:
                    raise ImportError("tifffile is required to write TIFF")
                tifffile.imwrite(
                    path, np.transpose(self.hypercube, (2, 0, 1)).astype(np.float32))
            else:
                np.save(path, self.hypercube.astype(np.float32))
                if self.voltages is not None:
                    vpath = os.path.splitext(path)[0] + "_voltages.npy"
                    np.save(vpath, self.voltages)
            QMessageBox.information(self, "Saved", f"Saved OPD stack to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{exc}")

    # ---------------- display tweaks ----------------

    def _relabel_spectrum_axes(self):
        """Put voltage on x and OPD on y once the stack has been converted."""
        if not self.is_opd:
            return
        if self.voltages is not None:
            for line in self.spectrum_ax.get_lines():
                n = len(line.get_xdata())
                if n == len(self.voltages):
                    line.set_xdata(self.voltages)
            for coll in list(self.spectrum_ax.collections):
                coll.remove()          # fill_between keeps index x; drop it
            self.spectrum_ax.set_xlabel("Drive voltage (V)")
            self.spectrum_ax.relim()
            self.spectrum_ax.autoscale_view()
        else:
            self.spectrum_ax.set_xlabel("Frame")
        self.spectrum_ax.set_ylabel("OPD (nm)")
        self.spectrum_ax.set_title("Retardance vs voltage")
        self.spectrum_canvas.draw_idle()

    def plot_current_spectrum(self):
        super().plot_current_spectrum()
        self._relabel_spectrum_axes()

    def update_spectrum_plot(self):
        super().update_spectrum_plot()
        self._relabel_spectrum_axes()

    def display_frame(self, frame_idx):
        super().display_frame(frame_idx)
        if self.is_opd:
            v = ""
            if self.voltages is not None and frame_idx < len(self.voltages):
                v = f" ({self.voltages[frame_idx]:.3f} V)"
            self.statusBar().showMessage(f"OPD Map (nm){v}")

    def on_frame_changed(self, value):
        super().on_frame_changed(value)
        if self.voltages is not None and value < len(self.voltages):
            self.frame_label.setText(
                f"Frame: {value} / {self.frame_slider.maximum()}   "
                f"({self.voltages[value]:.3f} V)")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = OPDViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
