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
        x, y = pos[0,:], pos[1,:]
        phi = poly.transform(pos)
        phi_expected = np.vstack((x, y, x * x, x * y, y * y))
        np.testing.assert_allclose(phi, phi_expected, rtol=1e-14)

    def test_guess(self):
        image = np.zeros((128, 128))
        image[:64, 64:] += 1
        image[64:, :64] += 2
        image[64:, 64:] += 3
        rsp_expected = np.zeros((4, 128, 128), dtype=bool)
        rsp_expected[0,:64,:64] = True
        rsp_expected[1,:64, 64:] = True
        rsp_expected[2, 64:,:64] = True
        rsp_expected[3, 64:, 64:] = True

        seeds = ((0, 0), (0, 100), (100, 0))
        rsp = sati.preprocessing.GuessInitRsp(image, 4, threshold=0.1,
                                              seeds=seeds)

        np.testing.assert_equal(rsp.seeds[:3], seeds)
        np.testing.assert_equal(rsp.guess, rsp_expected)
        np.testing.assert_equal(rsp.image, image)

        rsp._GuessInitRsp__test = True
        rsp.show()
