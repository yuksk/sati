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
        x = np.linspace(
            scipy.stats.norm.ppf(0.01, loc=loc, scale=scale),
            scipy.stats.norm.ppf(0.99, loc=loc, scale=scale),
            128,
        )
        f = sati.distributions.Norm(loc=loc, scale=scale)
        np.testing.assert_allclose(
            f.pdf(x).reshape(-1),
            scipy.stats.norm.pdf(x, loc=loc, scale=scale),
            rtol=1e-14,
        )

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
        x = np.linspace(
            scipy.stats.cauchy.ppf(0.01, loc=loc, scale=scale),
            scipy.stats.cauchy.ppf(0.99, loc=loc, scale=scale),
            128,
        )
        f = sati.distributions.Cauchy(loc=loc, scale=scale)
        np.testing.assert_allclose(
            f.pdf(x).reshape(-1),
            scipy.stats.cauchy.pdf(x, loc=loc, scale=scale),
            rtol=1e-14,
        )

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
        x = np.linspace(
            scipy.stats.t.ppf(0.01, df, loc=loc, scale=scale),
            scipy.stats.t.ppf(0.99, df, loc=loc, scale=scale),
            128,
        )
        f = sati.distributions.T(loc=loc, scale=scale, df=df)
        np.testing.assert_allclose(
            f.pdf(x).reshape(-1),
            scipy.stats.t.pdf(x, df, loc=loc, scale=scale),
            rtol=1e-14,
        )

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
        x = np.linspace(
            scipy.stats.vonmises.ppf(0.01, kappa, loc=loc, scale=scale),
            scipy.stats.vonmises.ppf(0.99, kappa, loc=loc, scale=scale),
            128,
        )
        f = sati.distributions.VonMises(loc=loc, scale=scale, kappa=kappa)
        np.testing.assert_allclose(
            f.pdf(x).reshape(-1),
            scipy.stats.vonmises.pdf(x * 2 * np.pi, kappa, loc=loc * scale, scale=scale)
            * scale,
            rtol=1e-14,
        )

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
        self.assertAlmostEqual(
            f.loglikelihood_at_pixel(x), np.log(f.pdf(x))[0], places=14
        )

    def test_logi0(self):
        x = np.linspace(100.0, 700.0, 7)
        a = sati.distributions.VonMises._logi0(x)
        b = np.log(2 * np.pi * scipy.special.i0(x))
        np.testing.assert_allclose(a, b, rtol=1e-14)

    # --- scaler() ---

    def test_scaler_forward(self):
        """Test scaler() forward transform rescales loc and scale."""
        loc = np.array([1.0, 3.0])
        scale = np.array([0.5, 1.5])
        f = sati.distributions.Norm(loc=loc.copy(), scale=scale.copy())
        mean, std = 2.0, 4.0
        f.scaler((mean, std))
        np.testing.assert_allclose(f.loc, (loc - mean) / std, rtol=1e-14)
        np.testing.assert_allclose(f.scale, scale / std, rtol=1e-14)

    def test_scaler_backward(self):
        """Test scaler() backward transform restores original scale."""
        loc = np.array([-0.25, 0.25])
        scale = np.array([0.125, 0.375])
        f = sati.distributions.Norm(loc=loc.copy(), scale=scale.copy())
        mean, std = 2.0, 4.0
        f.scaler((mean, std), backward=True)
        np.testing.assert_allclose(f.loc, loc * std + mean, rtol=1e-14)
        np.testing.assert_allclose(f.scale, scale * std, rtol=1e-14)

    def test_scaler_loc_none(self):
        """Test scaler() skips loc when it is None."""
        f = sati.distributions.Norm(scale=np.array([1.0, 2.0]))
        f.scaler((1.0, 2.0))
        self.assertIsNone(f.loc)
        np.testing.assert_allclose(f.scale, np.array([0.5, 1.0]), rtol=1e-14)

    def test_scaler_roundtrip(self):
        """Test that forward then backward scaler restores original values."""
        loc0 = np.array([1.0, 2.0, 3.0])
        scale0 = np.array([0.5, 0.4, 0.3])
        f = sati.distributions.Norm(loc=loc0.copy(), scale=scale0.copy())
        image_scale = (1.5, 2.5)
        f.scaler(image_scale)
        f.scaler(image_scale, backward=True)
        np.testing.assert_allclose(f.loc, loc0, rtol=1e-14)
        np.testing.assert_allclose(f.scale, scale0, rtol=1e-14)

    def test_vonmises_scaler(self):
        """Test VonMises scaler transforms only scale, not loc."""
        loc0, scale0 = 0.3, 3.0
        f = sati.distributions.VonMises(loc=loc0, scale=scale0, kappa=1.0)
        mean, std = 0.5, 2.0
        f.scaler((mean, std))
        self.assertAlmostEqual(f.scale, scale0 / std, places=14)
        self.assertAlmostEqual(f.loc, loc0, places=14)

    def test_vonmises_scaler_backward(self):
        """Test VonMises backward scaler restores scale."""
        f = sati.distributions.VonMises(loc=0.3, scale=1.5, kappa=1.0)
        f.scaler((0.5, 2.0), backward=True)
        self.assertAlmostEqual(f.scale, 3.0, places=14)

    # --- center(), std(), mixing_coef(), fullpdf() ---

    def test_center(self):
        """Test Distribution.center() computes weighted mean per class."""
        t = np.array([1.0, 2.0, 3.0, 4.0])
        gamma = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        f = sati.distributions.Norm(loc=np.zeros(2), scale=np.ones(2))
        centers = f.center(t, gamma)
        np.testing.assert_allclose(centers, [1.0, 4.0], rtol=1e-14)

    def test_center_uniform_weights(self):
        """Test center() equals arithmetic mean with uniform weights."""
        t = np.array([1.0, 2.0, 3.0, 4.0])
        gamma = np.ones((1, 4)) * 0.25
        f = sati.distributions.Norm(loc=np.array([0.0]), scale=np.ones(1))
        centers = f.center(t, gamma)
        np.testing.assert_allclose(centers, [2.5], rtol=1e-14)

    def test_std(self):
        """Test Distribution.std() computes weighted standard deviation."""
        t = np.array([0.0, 2.0])
        gamma = np.array([[0.5, 0.5]])
        f = sati.distributions.Norm(loc=np.array([1.0]), scale=np.array([1.0]))
        std = f.std(t, gamma, np.array([1.0]))
        np.testing.assert_allclose(std, [1.0], rtol=1e-14)

    def test_mixing_coef(self):
        """Test Distribution.mixing_coef() sums to 1 and matches proportion."""
        gamma = np.array([[3.0, 1.0], [1.0, 3.0]])
        f = sati.distributions.Norm(loc=np.zeros(2), scale=np.ones(2))
        pi = f.mixing_coef(gamma)
        np.testing.assert_allclose(pi, [0.5, 0.5], rtol=1e-14)
        self.assertAlmostEqual(pi.sum(), 1.0, places=14)

    def test_mixing_coef_unequal(self):
        """Test mixing_coef() with unequal class totals."""
        gamma = np.array([[3.0, 0.0], [1.0, 0.0]])
        f = sati.distributions.Norm(loc=np.zeros(2), scale=np.ones(2))
        pi = f.mixing_coef(gamma)
        np.testing.assert_allclose(pi, [0.75, 0.25], rtol=1e-14)

    def test_fullpdf(self):
        """Test fullpdf() equals pi * pdf()."""
        loc = np.array([0.0, 1.0])
        scale = np.array([1.0, 1.0])
        f = sati.distributions.Norm(loc=loc, scale=scale)
        f.pi = np.array([0.3, 0.7])
        t = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            f.fullpdf(t), f.pi.reshape(-1, 1) * f.pdf(t), rtol=1e-14
        )

    # --- retain() / revert() ---

    def test_retain_revert(self):
        """Test retain() and revert() restore loc and scale."""
        f = sati.distributions.Norm(
            loc=np.array([1.0, 2.0]), scale=np.array([0.5, 0.3])
        )
        f.retain()
        f.loc = np.array([99.0, 99.0])
        f.scale = np.array([99.0, 99.0])
        f.revert()
        np.testing.assert_allclose(f.loc, [1.0, 2.0], rtol=1e-14)
        np.testing.assert_allclose(f.scale, [0.5, 0.3], rtol=1e-14)

    def test_t_retain_revert_df(self):
        """Test T.retain() and revert() also restores df."""
        f = sati.distributions.T(
            loc=np.array([0.0, 1.0]),
            scale=np.array([1.0, 0.5]),
            df=np.array([2.0, 3.0]),
        )
        f.retain()
        f.df = np.array([99.0, 99.0])
        f.revert()
        np.testing.assert_allclose(f.df, [2.0, 3.0], rtol=1e-14)

    # --- T.initial_step() ---

    def test_t_initial_step_df_none(self):
        """Test T.initial_step() sets df to ones when df is None."""
        t = np.array([0.0, 0.5, 1.0, 1.5])
        gamma = np.ones((2, 4)) * 0.5
        f = sati.distributions.T(loc=np.array([0.5, 1.0]), scale=np.array([1.0, 1.0]))
        f.initial_step(t, gamma)
        np.testing.assert_allclose(f.df, np.ones(2), rtol=1e-14)

    def test_t_initial_step_df_retained(self):
        """Test T.initial_step() keeps pre-specified df unchanged."""
        t = np.array([0.0, 0.5, 1.0, 1.5])
        gamma = np.ones((2, 4)) * 0.5
        df_given = np.array([2.5, 3.5])
        f = sati.distributions.T(
            loc=np.array([0.5, 1.0]), scale=np.array([1.0, 1.0]), df=df_given.copy()
        )
        f.initial_step(t, gamma)
        np.testing.assert_allclose(f.df, df_given, rtol=1e-14)
