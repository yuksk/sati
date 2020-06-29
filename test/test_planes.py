import unittest

import numpy as np

import sati.planes


class TestPlanes(unittest.TestCase):
    """Test class of planes.py"""

    def test_coef(self):
        """Test if the coef is correctly expaned or truncated
        when it is given."""
        shape = (4, 4)
        given = np.array([1., 2., 3.])

        plane = sati.planes.Poly(degree=1, coef=given)
        plane.setup(shape, None)
        expected = np.copy(given)
        np.testing.assert_allclose(plane.coef, expected, rtol=1e-14)

        plane = sati.planes.Poly(degree=2, coef=given)
        plane.setup(shape, None)
        expected = np.array([1., 2., 3., 0., 0., 0.])
        np.testing.assert_allclose(plane.coef, expected, rtol=1e-14)

        given = np.array([1., 2., 3., 0., 0., 0.])
        plane = sati.planes.Poly(degree=1, coef=given)
        plane.setup(shape, None)
        expected = np.array([1., 2., 3.])
        np.testing.assert_allclose(plane.coef, expected, rtol=1e-14)

    def test_decay_arguments(self):
        with self.assertRaises(ValueError):
            _ = sati.planes.Decay(tau=-100., coef=1.)
        with self.assertRaises(ValueError):
            _ = sati.planes.Decay(tau=100., coef=1., kind='notexisting')
        with self.assertRaises(ValueError):
            _ = sati.planes.Decay(tau=100., coef=1., orgdrct='xlu')

        # These should not raise an error.
        _ = sati.planes.Decay(tau=100., coef=1., orgdrct='xlb')
        # case insensitive
        _ = sati.planes.Decay(tau=100., coef=1., orgdrct='XLB')
        # no particular order
        _ = sati.planes.Decay(tau=100., coef=1., orgdrct='lbx')
        # no need to specify all
        _ = sati.planes.Decay(tau=100., coef=1., orgdrct='xl')
        # duplication is allowed
        _ = sati.planes.Decay(tau=100., coef=1., orgdrct='xbx')

