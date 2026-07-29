# Copyright (c) 2026 Dan Fu@UW
"""
Unit tests for unmix package algorithms.

Tests all unmixing, denoising, classification, endmember extraction,
and peak fitting algorithms on synthetic hyperspectral cubes.
"""

import sys
import os
import unittest
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unmix import (
    denoise_mppca, denoise_wavelet,
    run_mcr_als, run_nmf, run_mesma,
    run_pca, run_sam, run_sid, run_svr,
    run_nfindr, run_vca, fit_spectrum,
)


class TestUnmixPackage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a synthetic hyperspectral cube (H=20, W=20, L=30) with 3 endmembers."""
        np.random.seed(42)
        cls.H, cls.W, cls.L = 20, 20, 30
        cls.K = 3  # endmembers

        # Wavelength axis
        cls.bands = np.linspace(400, 700, cls.L)

        # 3 synthetic Gaussian basis spectra
        s1 = np.exp(-((cls.bands - 480) ** 2) / (2 * 20 ** 2))
        s2 = np.exp(-((cls.bands - 550) ** 2) / (2 * 25 ** 2))
        s3 = np.exp(-((cls.bands - 640) ** 2) / (2 * 20 ** 2))
        cls.basis = np.vstack([s1, s2, s3])  # shape (3, 30)

        # Synthetic Dirichlet concentrations
        concentrations_flat = np.random.dirichlet([1, 1, 1], size=cls.H * cls.W)
        cls.true_concentrations = concentrations_flat.reshape(cls.H, cls.W, cls.K)

        # Clean cube
        clean_cube = (concentrations_flat @ cls.basis).reshape(cls.H, cls.W, cls.L)

        # Add Gaussian noise
        noise = np.random.normal(0, 0.02, clean_cube.shape)
        cls.cube = np.clip(clean_cube + noise, 0, None)

    # -----------------------------------------------------------------------
    # Denoising
    # -----------------------------------------------------------------------

    def test_denoise_mppca(self):
        result = denoise_mppca(self.cube, variance_threshold=0.95)
        self.assertEqual(result['hypercube'].shape, (self.H, self.W, self.L))
        self.assertIn('n_components_kept', result['info'])
        self.assertGreaterEqual(result['info']['n_components_kept'], 1)

    def test_denoise_wavelet(self):
        result = denoise_wavelet(self.cube, wavelet='db4')
        self.assertEqual(result['hypercube'].shape, (self.H, self.W, self.L))
        self.assertEqual(result['info']['wavelet'], 'db4')

    def test_tiff_stack_loading(self):
        import tifffile
        import tempfile
        # Create temporary multipage TIFF stack (L, H, W)
        stack = np.moveaxis(self.cube, -1, 0).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if hasattr(tifffile, 'imwrite'):
                tifffile.imwrite(tmp_path, stack)
            else:
                tifffile.imsave(tmp_path, stack)
            loaded = tifffile.imread(tmp_path)
            loaded = np.moveaxis(loaded, 0, -1)
            self.assertEqual(loaded.shape, (self.H, self.W, self.L))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # -----------------------------------------------------------------------
    # Blind Unmixing
    # -----------------------------------------------------------------------

    def test_run_mcr_als(self):
        result = run_mcr_als(self.cube, n_components=3, max_iter=20, non_neg=True)
        self.assertEqual(result['concentrations'].shape, (self.H, self.W, 3))
        self.assertEqual(result['basis_spectra'].shape, (3, self.L))
        self.assertEqual(result['r_squared'].shape, (self.H, self.W))
        self.assertEqual(result['residuals'].shape, (self.H, self.W))

    def test_run_nmf(self):
        result = run_nmf(self.cube, n_components=3, max_iter=50, random_state=42)
        self.assertEqual(result['concentrations'].shape, (self.H, self.W, 3))
        self.assertEqual(result['basis_spectra'].shape, (3, self.L))

    def test_run_mesma(self):
        result = run_mesma(self.cube, self.basis, max_endmembers=2)
        self.assertEqual(result['concentrations'].shape, (self.H, self.W, 3))
        self.assertEqual(result['basis_spectra'].shape, (3, self.L))

    # -----------------------------------------------------------------------
    # Classification / Decomposition
    # -----------------------------------------------------------------------

    def test_run_pca(self):
        result = run_pca(self.cube, n_components=3)
        self.assertEqual(result['concentrations'].shape, (self.H, self.W, 3))
        self.assertEqual(result['basis_spectra'].shape, (3, self.L))
        self.assertEqual(len(result['info']['explained_variance']), 3)

    def test_run_sam(self):
        result = run_sam(self.cube, self.basis)
        self.assertEqual(result['concentrations'].shape, (self.H, self.W, 3))
        self.assertEqual(result['classification'].shape, (self.H, self.W))

    def test_run_sid(self):
        result = run_sid(self.cube, self.basis)
        self.assertEqual(result['concentrations'].shape, (self.H, self.W, 3))
        self.assertEqual(result['classification'].shape, (self.H, self.W))

    def test_run_svr(self):
        result = run_svr(self.cube, self.basis, n_train=200, random_state=42)
        self.assertEqual(result['concentrations'].shape, (self.H, self.W, 3))

    # -----------------------------------------------------------------------
    # Endmember Extraction
    # -----------------------------------------------------------------------

    def test_run_nfindr(self):
        result = run_nfindr(self.cube, n_endmembers=3, random_state=42)
        self.assertEqual(result['endmembers'].shape, (3, self.L))
        self.assertEqual(result['positions'].shape, (3, 2))

    def test_run_vca(self):
        result = run_vca(self.cube, n_endmembers=3, random_state=42)
        self.assertEqual(result['endmembers'].shape, (3, self.L))
        self.assertEqual(result['positions'].shape, (3, 2))

    # -----------------------------------------------------------------------
    # Peak Fitting
    # -----------------------------------------------------------------------

    def test_fit_spectrum_gaussian(self):
        spectrum = self.basis[0]  # Single Gaussian peak at band index ~8 (480nm)
        result = fit_spectrum(spectrum, wavelengths=self.bands, model='gaussian', n_peaks=1)
        self.assertEqual(len(result['fitted']), self.L)
        self.assertEqual(len(result['peaks']), 1)
        self.assertAlmostEqual(result['peaks'][0]['center'], 480.0, delta=5.0)

    def test_fit_spectrum_lorentzian(self):
        spectrum = self.basis[1]
        result = fit_spectrum(spectrum, wavelengths=self.bands, model='lorentzian', n_peaks=1)
        self.assertEqual(len(result['fitted']), self.L)
        self.assertEqual(len(result['peaks']), 1)


if __name__ == '__main__':
    unittest.main()
