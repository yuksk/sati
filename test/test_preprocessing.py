import unittest

import numpy as np

import sati.preprocessing


class TestPreprocessing(unittest.TestCase):
    """Test class of preprocessing.py"""

    def test_scale_dimension(self):
        """Test that the input array must be 1D or 2D."""
        a = np.arange(8).reshape(2, 2, 2)
        with self.assertRaises(ValueError):
            _, _ = sati.preprocessing.standardize(a)

    def test_scale(self):
        # 1D
        a = np.arange(5.0)
        mean, std = 2.0, np.sqrt(2.0)
        stats_expected = (mean, std)
        a_std_expected = (a - 2.0) / np.sqrt(2.0)
        a_std, stats = sati.preprocessing.standardize(a)
        np.testing.assert_allclose(a_std, a_std_expected, rtol=1e-14)
        np.testing.assert_allclose(stats, stats_expected, rtol=1e-14)

        # 2D
        a = np.vstack((a, a * 2 + 1, a * 3 + 2))
        mean = np.array([mean, mean * 2 + 1, mean * 3 + 2])
        std = np.array([std, std * 2, std * 3])
        stats_expected = (mean, std)
        a_std_expected = np.vstack((a_std_expected,) * 3)
        a_std, stats = sati.preprocessing.standardize(a)
        np.testing.assert_allclose(a_std, a_std_expected, rtol=1e-14)
        np.testing.assert_allclose(stats, stats_expected, rtol=1e-14)

    def test_transform(self):
        poly = sati.preprocessing.PolynomialFeatures(2)
        pos = np.arange(8).reshape(2, 4)
        x, y = pos[0, :], pos[1, :]
        phi = poly.transform(pos)
        phi_expected = np.vstack((x, y, x * x, x * y, y * y))
        np.testing.assert_allclose(phi, phi_expected, rtol=1e-14)

    # --- standardize() with explicit stats ---

    def test_scale_with_explicit_stats_1d(self):
        """Test standardize() uses provided stats instead of computing them."""
        a = np.arange(5.0)
        mean, std = 3.0, 2.0
        a_std, stats_out = sati.preprocessing.standardize(a, (mean, std))
        np.testing.assert_allclose(a_std, (a - 3.0) / 2.0, rtol=1e-14)
        self.assertEqual(stats_out[0], 3.0)
        self.assertEqual(stats_out[1], 2.0)

    def test_scale_with_explicit_stats_2d(self):
        """Test standardize() with explicit stats on 2D array."""
        a = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
        mean = np.array([2.0, 3.0])
        std = np.array([1.0, 2.0])
        a_std, stats_out = sati.preprocessing.standardize(a, (mean, std))
        expected = (a - mean.reshape(-1, 1)) / std.reshape(-1, 1)
        np.testing.assert_allclose(a_std, expected, rtol=1e-14)
        np.testing.assert_allclose(stats_out[0], mean, rtol=1e-14)
        np.testing.assert_allclose(stats_out[1], std, rtol=1e-14)

    # --- PolynomialFeatures ---

    def test_transform_degree1(self):
        """Test PolynomialFeatures degree=1 returns (x, y) only."""
        poly = sati.preprocessing.PolynomialFeatures(1)
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        phi = poly.transform(pos)
        phi_expected = np.vstack((pos[0], pos[1]))
        np.testing.assert_allclose(phi, phi_expected, rtol=1e-14)

    def test_transform_degree3(self):
        """Test PolynomialFeatures degree=3 produces correct number of terms."""
        # degree d → (d+2)(d+1)/2 - 1 features
        poly = sati.preprocessing.PolynomialFeatures(3)
        pos = np.arange(8).reshape(2, 4).astype(float)
        phi = poly.transform(pos)
        # degree=3: 9 terms (x,y,x2,xy,y2,x3,x2y,xy2,y3)
        self.assertEqual(phi.shape[0], 9)
        self.assertEqual(phi.shape[1], 4)
