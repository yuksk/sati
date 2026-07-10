import unittest

import numpy as np

import sati.planes


class TestPlanes(unittest.TestCase):
    """Test class of planes.py"""

    def test_coef(self):
        """Test if the coef is correctly expaned or truncated
        when it is given."""
        shape = (4, 4)
        given = np.array([1.0, 2.0, 3.0])

        plane = sati.planes.Poly(degree=1, coef=given)
        plane.setup(shape, None, None)
        expected = np.copy(given)
        np.testing.assert_allclose(plane.coef, expected, rtol=1e-14)

        plane = sati.planes.Poly(degree=2, coef=given)
        plane.setup(shape, None, None)
        expected = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(plane.coef, expected, rtol=1e-14)

        given = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        plane = sati.planes.Poly(degree=1, coef=given)
        plane.setup(shape, None, None)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(plane.coef, expected, rtol=1e-14)

    def test_decay_arguments(self):
        with self.assertRaises(ValueError):
            _ = sati.planes.Decay(tau=-100.0, coef=1.0)
        with self.assertRaises(ValueError):
            _ = sati.planes.Decay(tau=100.0, coef=1.0, kind="notexisting")
        with self.assertRaises(ValueError):
            _ = sati.planes.Decay(tau=100.0, coef=1.0, orgdrct="xlu")

        # These should not raise an error.
        _ = sati.planes.Decay(tau=100.0, coef=1.0, orgdrct="xlb")
        # case insensitive
        _ = sati.planes.Decay(tau=100.0, coef=1.0, orgdrct="XLB")
        # no particular order
        _ = sati.planes.Decay(tau=100.0, coef=1.0, orgdrct="lbx")
        # no need to specify all
        _ = sati.planes.Decay(tau=100.0, coef=1.0, orgdrct="xl")
        # duplication is allowed
        _ = sati.planes.Decay(tau=100.0, coef=1.0, orgdrct="xbx")

    # --- Poly: grad(), retain()/revert() ---

    def test_poly_grad_shape(self):
        """Test that Poly.grad() returns a vector with length equal to
        the number of polynomial coefficients."""
        shape = (8, 8)
        n_pixels = shape[0] * shape[1]
        poly = sati.planes.Poly(degree=1)
        poly.setup(shape, None, None)
        # degree=1 has 3 terms: bias + x + y
        u = np.ones((3, n_pixels))
        grad = poly.grad(u)
        self.assertEqual(grad.ndim, 1)
        self.assertEqual(grad.size, 3)

    def test_poly_grad_degree2_shape(self):
        """Test Poly.grad() shape for degree=2 (6 terms)."""
        shape = (8, 8)
        n_pixels = shape[0] * shape[1]
        poly = sati.planes.Poly(degree=2)
        poly.setup(shape, None, None)
        # degree=2: bias, x, y, x^2, xy, y^2 = 6 terms
        u = np.ones((2, n_pixels))
        grad = poly.grad(u)
        self.assertEqual(grad.size, 6)

    def test_poly_grad_zero_input(self):
        """Test that Poly.grad() returns zero when u is zero."""
        shape = (8, 8)
        n_pixels = shape[0] * shape[1]
        poly = sati.planes.Poly(degree=1)
        poly.setup(shape, None, None)
        u = np.zeros((2, n_pixels))
        grad = poly.grad(u)
        np.testing.assert_allclose(grad, np.zeros(3), atol=1e-14)

    def test_poly_retain_revert(self):
        """Test Poly retain() and revert() restore coef and plane."""
        shape = (8, 8)
        n_pixels = shape[0] * shape[1]
        coef0 = np.array([1.0, 0.5, -0.3])
        poly = sati.planes.Poly(degree=1, coef=coef0.copy())
        poly.setup(shape, None, None)
        # Initialize plane
        t = np.zeros(n_pixels)
        gamma = np.ones((2, n_pixels)) * 0.5
        mu = np.array([0.0, 1.0])
        poly.initial_step(t, gamma, mu)
        poly.retain()
        saved_coef = poly.coef.copy()
        saved_plane = poly.plane.copy()
        poly.coef = np.array([99.0, 99.0, 99.0])
        poly.revert()
        np.testing.assert_allclose(poly.coef, saved_coef, rtol=1e-14)
        np.testing.assert_allclose(poly.plane, saved_plane, rtol=1e-14)

    # --- Decay: plane values and finalize() ---

    def test_decay_exp_plane(self):
        """Test Decay (exp) plane values match manual calculation."""
        shape = (4, 4)
        n = np.prod(shape)
        tau, coef = (10.0,), (2.0,)
        decay = sati.planes.Decay(tau=tau, coef=coef, kind="exp", orgdrct="lbx")
        decay.setup(shape, (0.0, 1.0), roi=None)
        decay.update_ga(decay.x0())

        index = np.linspace(0, 1, num=n, endpoint=False)
        beta = -n / np.array(tau[0])
        expected = np.array(coef[0]) * np.exp(beta * index)
        np.testing.assert_allclose(decay.plane, expected, rtol=1e-14)

    def test_decay_log_plane(self):
        """Test Decay (log) plane values match manual calculation."""
        shape = (4, 4)
        n = np.prod(shape)
        tau, coef = (100.0,), (1.5,)
        decay = sati.planes.Decay(tau=tau, coef=coef, kind="log", orgdrct="lbx")
        decay.setup(shape, (0.0, 1.0), roi=None)
        decay.update_ga(decay.x0())

        index = np.linspace(0, 1, num=n, endpoint=False)
        beta = np.sqrt(np.array(tau[0]) / n)
        expected = np.array(coef[0]) * np.log(index + beta**2)
        np.testing.assert_allclose(decay.plane, expected, rtol=1e-14)

    def test_decay_finalize_tau_exp(self):
        """Test Decay.finalize() (exp) recovers tau from internal beta."""
        shape = (8, 8)
        tau_in, coef_in = (500.0,), (-2.0,)
        decay = sati.planes.Decay(tau=tau_in, coef=coef_in, kind="exp", orgdrct="lbx")
        decay.setup(shape, (0.0, 1.0), roi=None)
        decay.update_ga(decay.x0())
        decay.finalize(std=1.0)
        np.testing.assert_allclose(decay.tau, np.array(tau_in), rtol=1e-14)

    def test_decay_finalize_tau_log(self):
        """Test Decay.finalize() (log) recovers tau from internal beta."""
        shape = (8, 8)
        tau_in, coef_in = (2500.0,), (1.5,)
        decay = sati.planes.Decay(tau=tau_in, coef=coef_in, kind="log", orgdrct="lbx")
        decay.setup(shape, (0.0, 1.0), roi=None)
        decay.update_ga(decay.x0())
        decay.finalize(std=1.0)
        np.testing.assert_allclose(decay.tau, np.array(tau_in), rtol=1e-12)

    def test_decay_finalize_coef_rescaled(self):
        """Test Decay.finalize() multiplies coef back by image std."""
        shape = (4, 4)
        std_image = 3.0
        tau_in, coef_in = (50.0,), (2.0,)
        decay = sati.planes.Decay(tau=tau_in, coef=coef_in, kind="exp", orgdrct="lbx")
        decay.setup(shape, (0.0, std_image), roi=None)
        decay.update_ga(decay.x0())
        decay.finalize(std=std_image)
        # coef was divided by std_image in setup, then multiplied back by finalize
        np.testing.assert_allclose(decay.coef, np.array(coef_in), rtol=1e-14)
