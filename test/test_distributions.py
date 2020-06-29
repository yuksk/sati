import sys
import unittest

import numpy as np
import scipy.stats
import scipy.special

import sati.distributions


class TestDistribution(unittest.TestCase):
    """Test class of distribution.py"""

    def test_norm_pdf(self):
        """Test probability distribution function of normal distribution."""
        loc, scale = 0.51, 1.12
        x = np.linspace(scipy.stats.norm.ppf(0.01, loc=loc, scale=scale),
                        scipy.stats.norm.ppf(0.99, loc=loc, scale=scale), 128)
        f = sati.distributions.Norm(loc=loc, scale=scale)
        np.testing.assert_allclose(
            f.pdf(x).reshape(-1),
            scipy.stats.norm.pdf(x, loc=loc, scale=scale), rtol=1e-14)

    def test_norm_grad(self):
        loc, scale = 0.51, 1.12
        x = np.array([loc])
        f = sati.distributions.Norm(loc=loc, scale=scale)
        g = np.concatenate(f.grad(x, 1), axis=None) * f.pdf(x).reshape(-1)
        g_expected = [0, -1 / (np.sqrt(2 * np.pi) * scale * scale)]
        np.testing.assert_allclose(g, g_expected, rtol=1e-14)

    def test_cauchy_pdf(self):
        """Test probability distribution function of Cauchy distribution."""
        loc, scale = 0.51, 1.12
        x = np.linspace(scipy.stats.cauchy.ppf(0.01, loc=loc, scale=scale),
                        scipy.stats.cauchy.ppf(0.99, loc=loc, scale=scale),
                        128)
        f = sati.distributions.Cauchy(loc=loc, scale=scale)
        np.testing.assert_allclose(
            f.pdf(x).reshape(-1),
            scipy.stats.cauchy.pdf(x, loc=loc, scale=scale), rtol=1e-14)

    def test_cauchy_grad(self):
        loc, scale = 0.51, 1.12
        x = np.array([loc])
        f = sati.distributions.Cauchy(loc=loc, scale=scale)
        g = np.concatenate(f.grad(x, 1), axis=None) * f.pdf(x).reshape(-1)
        g_expected = [0, -1 / (np.pi * scale * scale)]
        np.testing.assert_allclose(g, g_expected, rtol=1e-14)

    def test_t_pdf(self):
        """Test probability distribution function of Student's t
        distribution."""
        df, loc, scale = 2.74, 0.51, 1.12
        x = np.linspace(scipy.stats.t.ppf(0.01, df, loc=loc, scale=scale),
                        scipy.stats.t.ppf(0.99, df, loc=loc, scale=scale), 128)
        f = sati.distributions.T(loc=loc, scale=scale, df=df)
        np.testing.assert_allclose(
            f.pdf(x).reshape(-1),
            scipy.stats.t.pdf(x, df, loc=loc, scale=scale), rtol=1e-14)

    def test_t_grad(self):
        df, loc, scale = 2.74, 0.51, 1.12
        x = np.array([loc])
        f = sati.distributions.T(loc=loc, scale=scale, df=df)
        g = np.concatenate(f.grad(x, 1), axis=None) * f.pdf(x).reshape(-1)
        # values by Mathematica
        g_expected = [0, -0.290819072103647, 0.0102554148775136]
        np.testing.assert_allclose(g, g_expected, rtol=1e-14)

    def test_vonmises_pdf(self):
        """Test probability distribution function of von Mises distribution."""
        kappa, loc, scale = 1.07, 0.51, 1.12
        x = np.linspace(scipy.stats.vonmises.ppf(0.01, kappa, loc=loc,
                                                 scale=scale),
                        scipy.stats.vonmises.ppf(0.99, kappa, loc=loc,
                                                 scale=scale),
                        128)
        f = sati.distributions.VonMises(loc=loc, scale=scale, kappa=kappa)
        np.testing.assert_allclose(
            f.pdf(x).reshape(-1),
            scipy.stats.vonmises.pdf(x*2*np.pi, kappa, loc=loc*scale,
                                     scale=scale) * scale,
            rtol=1e-14)

    def test_vonmises_grad(self):
        x = np.array([2.12])

        kappa, loc, scale = 1.07, 0.51, 1.12
        f = sati.distributions.VonMises(loc=loc, scale=scale, kappa=kappa)
        g = np.concatenate(f.grad(x, 1), axis=None)
        # values by Mathematica
        g_expected = [5.55740456209862, -0.990627015637780, -10.5193729211152]
        np.testing.assert_allclose(g, g_expected, rtol=1e-14)

    def test_vonmises_ll(self):
        x = np.array([2.12])
        kappa, loc, scale = 1.07, 0.51, 1.12
        f = sati.distributions.VonMises(loc=loc, scale=scale, kappa=kappa)
        self.assertAlmostEqual(f.loglikelihood_at_pixel(x),
                               np.log(f.pdf(x))[0], places=14)

    def test_logi0(self):
        x = np.linspace(100., 700., 7)
        a = sati.distributions.VonMises._logi0(x)
        b = np.log(2 * np.pi * scipy.special.i0(x))
        np.testing.assert_allclose(a, b, rtol=1e-14)

