import itertools
import os
import pickle
import unittest
import warnings
from copy import deepcopy

import numpy as np
import scipy.stats

import sati.distributions
import sati.planes
import sati.model


class TestModel(unittest.TestCase):
    """Test class of model.py"""

    def setUp(self):
        self.n = 128
        self.d = 2.6
        self.loc = [0.4 - self.d, 0.4, 0.4 + self.d * 2]
        self.loc_expected = self.loc - np.mean(self.loc)
        self.scale_expected = [5e-2, 2e-2, 1e-2]
        self.df_expected = [1.2, 1.6, 0.9]
        self.slope = [1e-3, -5e-3]
        self.plane = np.array(
            [[i*self.slope[0] + j*self.slope[1]
              for i in range(self.n)] for j in range(self.n)],)

        self.rsp = np.zeros((len(self.scale_expected), self.n, self.n))
        self.rsp[0,:50, 50:] = 1
        self.rsp[1,:,:] = 1
        self.rsp[1,:50, 50:] = 0
        self.rsp[1,75:,:100] = 0
        self.rsp[2, 75:,:100] = 1

        self.methods = ('l-bfgs-b', 'adam')
        self.options = {
                'l-bfgs-b' : {'maxcor': 20, 'maxls': 40,
                              'ftol': np.finfo(float).eps**(2/3),
                              'gtol': np.finfo(float).eps**(1/3)},
                'adam' : None
                }

    def create_data(self, dist, plane=True):
        f = {'norm': scipy.stats.norm.rvs,
             'cauchy': scipy.stats.cauchy.rvs,
             't': scipy.stats.t.rvs}
        if dist == 't':
            x = [np.array(f[dist](size=self.n**2, loc=self.loc[i],
                                  scale=self.scale_expected[i],
                                  df=self.df_expected[i], random_state=i+3))
                 .reshape((self.n, self.n))
                 for i in range(len(self.scale_expected))]
        else:
            x = [np.array(f[dist](size=self.n**2, loc=self.loc[i],
                                  scale=self.scale_expected[i],
                                  random_state=i+3))
                 .reshape((self.n, self.n))
                 for i in range(len(self.scale_expected))]
        d = x[1]
        d[:50, 50:] = x[0][:50, 50:]
        d[75:,:100] = x[2][75:,:100]
        if plane:
            d += self.plane
        else:
            # Used for assertion
            d -= np.mean(d)
        return d

    def test_missing_attribute(self):
        """Make sure failure if one of rsp, poly, and dist is missing,
        and success if the missing one is given."""
        image = self.create_data('cauchy')
        poly = sati.planes.Poly()
        dist = sati.distributions.Cauchy()

        m = sati.Model(image, poly=poly, dist=dist)
        self.assertRaisesRegex(sati.model.NoAttributeError, 'rsp', m.optimize)
        m.rsp = self.rsp
        self.assertIsNone(m.optimize(method='quick', verbosity=0))

        m = sati.Model(image, rsp=self.rsp, dist=dist)
        self.assertRaisesRegex(sati.model.NoAttributeError, 'poly', m.optimize)
        m.poly = poly
        self.assertIsNone(m.optimize(method='quick', verbosity=0))

        m = sati.Model(image, rsp=self.rsp, poly=poly)
        self.assertRaisesRegex(sati.model.NoAttributeError, 'dist', m.optimize)
        m.dist = dist
        self.assertIsNone(m.optimize(method='quick', verbosity=0))

    def test_pickle(self):
        image = self.create_data('cauchy')
        image += np.array(
            [[i*i*1e-4 + j*j*2e-4 for i in range(self.n)]
             for j in range(self.n)])

        m = sati.Model(image, rsp=self.rsp, poly=sati.planes.Poly(),
                       dist=sati.distributions.Cauchy())
        m.optimize(method='quick', verbosity=0)
        m.pickle('tmp.pickle')

        m.poly = sati.planes.Poly(degree=2, coef=m.poly.coef)
        m.optimize(method='quick', verbosity=0)

        m1 = sati.Model.unpickle('tmp.pickle')
        m1.poly = sati.planes.Poly(degree=2, coef=m1.poly.coef)
        m1.optimize(method='quick', verbosity=0)

        with self.subTest(parameter='loc'):
            np.testing.assert_allclose(m.dist.loc, m1.dist.loc, rtol=1e-14)
        with self.subTest(parameter='scale'):
            np.testing.assert_allclose(m.dist.scale, m1.dist.scale, rtol=1e-14)
        with self.subTest(parameter='fullplane'):
            np.testing.assert_allclose(m.poly.plane, m1.poly.plane, rtol=1e-14)

        os.remove('tmp.pickle')

    def test_deepcopy(self):
        image = self.create_data('cauchy')
        image += np.array(
            [[i*i*1e-4 + j*j*2e-4 for i in range(self.n)]
             for j in range(self.n)])

        m = sati.Model(image, rsp=self.rsp, poly=sati.planes.Poly(),
                       dist=sati.distributions.Cauchy())
        m.optimize(method='quick', verbosity=0)
        m1 = deepcopy(m)
        self.assertNotEqual(id(m.rsp), id(m1.rsp))
        self.assertNotEqual(id(m.poly), id(m1.poly))
        self.assertNotEqual(id(m.dist), id(m1.dist))

        m.poly = sati.planes.Poly(degree=2, coef=m.poly.coef)
        m.optimize(method='quick', verbosity=0)

        m1.poly = sati.planes.Poly(degree=2, coef=m1.poly.coef)
        m1.optimize(method='quick', verbosity=0)

        with self.subTest(parameter='loc'):
            np.testing.assert_allclose(m.dist.loc, m1.dist.loc, rtol=1e-14)
        with self.subTest(parameter='scale'):
            np.testing.assert_allclose(m.dist.scale, m1.dist.scale, rtol=1e-14)
        with self.subTest(parameter='fullplane'):
            np.testing.assert_allclose(m.poly.plane, m1.poly.plane, rtol=1e-14)

    def test_not_converged(self):
        """Test a case of not-converged."""
        m = sati.Model(self.create_data('cauchy'), rsp=self.rsp,
                       poly=sati.planes.Poly(),
                       dist=sati.distributions.Cauchy())
        with self.assertWarns(sati.model.NotConvergedWarning):
            m.optimize(method='quick', maxiter=4, verbosity=0)

    def test_verbosity(self):
        """Use verbosity=2"""
        warnings.simplefilter('ignore', sati.model.NotConvergedWarning)
        m = sati.Model(self.create_data('cauchy'), rsp=self.rsp,
                       poly=sati.planes.Poly(),
                       dist=sati.distributions.Cauchy())
        m.optimize(method='l-bfgs-b', maxiter=3, verbosity=2)
        warnings.resetwarnings()

    def test_no_further_optimization(self):
        """Test too small learning rate"""
        m = sati.Model(self.create_data('cauchy'), rsp=self.rsp,
                       poly=sati.planes.Poly(),
                       dist=sati.distributions.Cauchy())
        with self.assertWarns(sati.model.MstepWarning):
            m.optimize(method='adam', tol=1e-7, verbosity=0,
                       options={'ftol': 1e-5})

    def test_unknown_method(self):
        m = sati.Model(self.create_data('cauchy'), rsp=self.rsp,
                       poly=sati.planes.Poly(),
                       dist=sati.distributions.Cauchy())
        with self.assertRaisesRegex(ValueError, 'unknown'):
            m.optimize(method='not-existing')

    def test_quick_not_available(self):
        m = sati.Model(self.create_data('cauchy'), rsp=self.rsp,
                       poly=sati.planes.Poly(),
                       dist=sati.distributions.T())
        with self.assertRaisesRegex(ValueError, 'quick'):
            m.optimize(method='quick')

    def test_method_auto(self):
        warnings.simplefilter('ignore', sati.model.SatiWarning)

        m = sati.Model(self.create_data('cauchy'), rsp=self.rsp,
                       poly=sati.planes.Poly(),
                       dist=sati.distributions.Cauchy())
        m.optimize(method='auto', maxiter=2, verbosity=0)
        self.assertEqual(m._Model__method, 'quick')

        m = sati.Model(self.create_data('cauchy'), rsp=self.rsp,
                       poly=sati.planes.Poly(),
                       dist=sati.distributions.T())
        m.optimize(method='auto', maxiter=2, verbosity=0)
        self.assertEqual(m._Model__method, 'l-bfgs-b')

        warnings.resetwarnings()

    def test_quick(self):
        """Optimize with the quick method."""
        dists = {'cauchy': sati.distributions.Cauchy(),
                 'norm': sati.distributions.Norm()}
        plane = sati.planes.Poly()
        for d in dists:
            image = self.create_data(d)
            m = sati.Model(image, rsp=self.rsp, poly=plane, dist=dists[d])
            m.optimize(method='quick', verbosity=0)

            with self.subTest(distribution=d, parameter='loc'):
                np.testing.assert_allclose(m.dist.loc - np.mean(m.dist.loc),
                                           self.loc_expected, rtol=7e-4)
            with self.subTest(distribution=d, parameter='scale'):
                np.testing.assert_allclose(m.dist.scale, self.scale_expected,
                                           rtol=2e-2)
            with self.subTest(distribution=d, parameter='subtracted'):
                np.testing.assert_allclose(m.subtracted - np.mean(m.subtracted),
                                           self.create_data(d, plane=False),
                                           rtol=4e-3)

    def test_ga(self):
        """Optimize with gradient ascent."""
        dists = {'cauchy': sati.distributions.Cauchy(),
                 'norm': sati.distributions.Norm()}
        plane = sati.planes.Poly()

        for d, mtd in itertools.product(dists, self.methods):
            image = self.create_data(d)
            m = sati.Model(image, rsp=self.rsp, poly=plane, dist=dists[d])
            m.optimize(method=mtd, verbosity=0, options=self.options[mtd])

            with self.subTest(method=mtd, distribution=d, parameter='loc'):
                np.testing.assert_allclose(m.dist.loc - np.mean(m.dist.loc),
                                           self.loc_expected, rtol=6e-4)
            with self.subTest(method=mtd, distribution=d, parameter='scale'):
                np.testing.assert_allclose(m.dist.scale, self.scale_expected,
                                           rtol=2e-2)
            with self.subTest(method=mtd, distribution=d, parameter='subtracted'):
                np.testing.assert_allclose(m.subtracted - np.mean(m.subtracted),
                                           self.create_data(d, plane=False),
                                           rtol=5e-3)

    def test_ga_t(self):
        """Test estimated parameters (t)"""
        image = self.create_data('t')

        for mtd in self.methods:
            m = sati.Model(image, rsp=self.rsp, poly=sati.planes.Poly(),
                           dist=sati.distributions.Cauchy())
            m.optimize(method=mtd, verbosity=0, options=self.options[mtd])

            m.dist = sati.distributions.T(loc=m.dist.loc, scale=m.dist.scale)
            m.optimize(method=mtd, verbosity=0, options=self.options[mtd])

            with self.subTest(method=mtd, parameter='loc'):
                np.testing.assert_allclose(m.dist.loc - np.mean(m.dist.loc),
                                           self.loc_expected, rtol=1e-3)
            with self.subTest(method=mtd, parameter='scale'):
                np.testing.assert_allclose(m.dist.scale, self.scale_expected,
                                           rtol=4e-2)
            with self.subTest(method=mtd, parameter='df'):
                np.testing.assert_allclose(m.dist.df, self.df_expected, rtol=3e-2)
            with self.subTest(method=mtd, parameter='subtracted'):
                np.testing.assert_allclose(m.subtracted - np.mean(m.subtracted),
                                           self.create_data('t', plane=False),
                                           rtol=1e-3)

    def test_decay_exp(self):
        """Test exp decay term"""
        image_xlb = self.create_data('cauchy')
        image_yrt = np.copy(image_xlb)
        image_expected = self.create_data('cauchy', plane=False)

        tau  = (500, 2500)
        coef = (-2.0, -0.4)
        shape = image_xlb.shape
        index = np.linspace(0, -image_xlb.size, image_xlb.size, endpoint=False)
        for t, c in zip(tau, coef):
            image_xlb += (c * np.exp(index / t)).reshape(shape)
            image_yrt += (c * np.exp(np.flip(index) / t)).reshape(shape).T

        orgdrcts = ('xlb', 'yrt')
        images = {'xlb': image_xlb, 'yrt': image_yrt}

        rtols = {
            'l-bfgs-b': {'loc': 4.3e-4, 'scale': 0.018, 't': 0.011,
                         'tau': 0.0067, 'coef': 0.0057},
            'adam': {'loc': 4.4e-4, 'scale': 0.018, 't': 0.011,
                     'tau': 0.012, 'coef': 0.0037},
        }

        for mtd, o in itertools.product(self.methods, orgdrcts):
            m = sati.Model(images[o], rsp=self.rsp, poly=sati.planes.Poly(),
                           decay=sati.planes.Decay(tau=tau, coef=coef,
                                                   kind='exp',orgdrct=o),
                           dist=sati.distributions.Cauchy())
            m.optimize(method=mtd, verbosity=0, options=self.options[mtd])

            with self.subTest(method=mtd, orgdrct=o, parameter='loc'):
                np.testing.assert_allclose(m.dist.loc - np.mean(m.dist.loc),
                                           self.loc_expected,
                                           rtol=rtols[mtd]['loc'])
            with self.subTest(method=mtd, orgdrct=o, parameter='scale'):
                np.testing.assert_allclose(m.dist.scale, self.scale_expected,
                                           rtol=rtols[mtd]['scale'])
            with self.subTest(method=mtd, orgdrct=o, parameter='subtracted'):
                np.testing.assert_allclose(m.subtracted - np.mean(m.subtracted),
                                           image_expected,
                                           rtol=rtols[mtd]['t'])
            with self.subTest(method=mtd, orgdrct=o, parameter='decay (tau)'):
                np.testing.assert_allclose(m.decay.tau, tau,
                                           rtol=rtols[mtd]['tau'])
            with self.subTest(method=mtd, orgdrct=o, parameter='decay (coef)'):
                np.testing.assert_allclose(m.decay.coef, coef,
                                           rtol=rtols[mtd]['coef'])

    def test_decay_log(self):
        """Test log decay term"""
        image = self.create_data('cauchy')
        image_expected = self.create_data('cauchy', plane=False)

        tau, coef, orgdrct = (2500, ), (1.5, ), 'yrt'
        index = np.linspace(0, image.size, image.size, endpoint=False)
        for t, c, in zip(tau, coef):
            image += (c * np.log(np.flip(index) + t)).reshape(image.shape).T

        rtols = {
            'l-bfgs-b': {'loc': 3.4e-4, 'scale': 0.018, 't': 0.025,
                         'tau': 8.4e-3, 'coef': 5.4e-3},
            'adam': {'loc': 3.4e-4, 'scale': 0.018, 't': 0.029,
                     'tau': 1.1e-2, 'coef': 6.8e-3},
        }

        for mtd in self.methods:
            m = sati.Model(image, rsp=self.rsp, poly=sati.planes.Poly(),
                           decay=sati.planes.Decay(tau=tau, coef=coef,
                                                   kind='log',orgdrct=orgdrct),
                           dist=sati.distributions.Cauchy())
            m.optimize(method=mtd, verbosity=0, options=self.options[mtd])

            with self.subTest(method=mtd, parameter='loc'):
                np.testing.assert_allclose(m.dist.loc - np.mean(m.dist.loc),
                                           self.loc_expected,
                                           rtol=rtols[mtd]['loc'])
            with self.subTest(method=mtd, parameter='scale'):
                np.testing.assert_allclose(m.dist.scale, self.scale_expected,
                                           rtol=rtols[mtd]['scale'])
            with self.subTest(method=mtd, parameter='subtracted'):
                np.testing.assert_allclose(m.subtracted - np.mean(m.subtracted),
                                           image_expected,
                                           rtol=rtols[mtd]['t'])
            with self.subTest(method=mtd, parameter='decay (tau)'):
                np.testing.assert_allclose(m.decay.tau, tau,
                                           rtol=rtols[mtd]['tau'])
            with self.subTest(method=mtd, parameter='decay (coef)'):
                np.testing.assert_allclose(m.decay.coef, coef ,
                                           rtol=rtols[mtd]['coef'])

    def test_prior_vonmises(self):
        kappas = {'l-bfgs-b': 0.04, 'adam': 0.1}

        for mtd in self.methods:
            prior = sati.distributions.VonMises(
                    scale=2.5,
                    kappa=np.ones(self.rsp.shape[0])*kappas[mtd])
            m = sati.Model(self.create_data('cauchy'), rsp=self.rsp,
                           poly=sati.planes.Poly(),
                           dist=sati.distributions.Cauchy(), prior=prior)
            m.optimize(method=mtd, verbosity=0, options=self.options[mtd])

            with self.subTest(method=mtd, parameter='loc'):
                np.testing.assert_allclose(m.dist.loc - np.mean(m.dist.loc),
                                           self.loc_expected, rtol=6e-4)
            with self.subTest(method=mtd, parameter='scale'):
                np.testing.assert_allclose(m.dist.scale, self.scale_expected,
                                           rtol=4e-2)
            with self.subTest(method=mtd, parameter='spacing'):
                self.assertAlmostEqual(m.prior.scale, self.d, places=3)

    def test_simple_roi(self):
        """Test a simple case with an ROI."""
        data = self.create_data('cauchy')
        roi = np.ones_like(data, dtype='?')
        roi[:,:10] = False
        roi[:, 120:] = False
        poly = sati.planes.Poly()
        dist = sati.distributions.Cauchy()

        for mtd in self.methods:
            m_roi = sati.Model(data, rsp=self.rsp, poly=poly, dist=dist, roi=roi)
            m_view = sati.Model(data[:, 10:120], rsp=self.rsp[:,:, 10:120],
                                poly=poly, dist=dist)
            m_roi.optimize(method=mtd, verbosity=0, options=self.options[mtd])
            m_view.optimize(method=mtd, verbosity=0, options=self.options[mtd])

            loc_roi = m_roi.dist.loc - np.mean(m_roi.dist.loc)
            loc_view = m_view.dist.loc - np.mean(m_view.dist.loc)

            with self.subTest(method=mtd, parameter='loc'):
                np.testing.assert_allclose(loc_roi, loc_view, rtol=1e-14)
            with self.subTest(method=mtd, parameter='scale'):
                np.testing.assert_allclose(m_roi.dist.scale, m_view.dist.scale,
                                           rtol=1e-14)

    def test_decay_exp_roi(self):
        image = self.create_data('cauchy')
        tau, coef = 500, -2.0
        index = np.linspace(0, -image.size, image.size, endpoint=False)
        image += (coef * np.exp(index / tau)).reshape(image.shape)

        poly = sati.planes.Poly()
        dist = sati.distributions.Cauchy()
        decay = sati.planes.Decay(tau=tau, coef=coef, orgdrct='lbx', kind='exp')
        roi = np.ones_like(image, dtype='?')
        roi[120:,:] = False

        for mtd in self.methods:
            m_roi = sati.Model(image, rsp=self.rsp, poly=poly, dist=dist,
                               decay=decay, roi=roi)
            m_view = sati.Model(image[:120,:], rsp=self.rsp[:,:120,:],
                                poly=poly, dist=dist, decay=decay)
            m_roi.optimize(method=mtd, verbosity=0, options=self.options[mtd])
            m_view.optimize(method=mtd, verbosity=0, options=self.options[mtd])

            with self.subTest(method=mtd, parameter='loc'):
                np.testing.assert_allclose(
                    m_roi.dist.loc - np.mean(m_roi.dist.loc),
                    m_view.dist.loc - np.mean(m_view.dist.loc), rtol=5e-6)
            with self.subTest(method=mtd, parameter='scale'):
                np.testing.assert_allclose(m_roi.dist.scale, m_view.dist.scale,
                                           rtol=5e-4)
            with self.subTest(method=mtd, parameter='tau'):
                np.testing.assert_allclose(m_roi.decay.tau, m_view.decay.tau,
                                           rtol=1e-4)
            with self.subTest(method=mtd, parameter='coef'):
                np.testing.assert_allclose(m_roi.decay.coef, m_view.decay.coef,
                                           rtol=3e-5)

    def test_decay_log_roi(self):
        image = self.create_data('cauchy')
        tau, coef, orgdrct = 2500, 1.5, 'yrt'
        index = np.linspace(0, image.size, image.size, endpoint=False)
        image += (coef * np.log(np.flip(index) + tau)).reshape(image.shape).T

        poly = sati.planes.Poly()
        dist = sati.distributions.Cauchy()
        decay = sati.planes.Decay(tau=tau, coef=coef, orgdrct=orgdrct,
                                  kind='log')
        roi = np.ones_like(image, dtype='?')
        roi[:,:8] = False

        for mtd in self.methods:
            m_roi = sati.Model(image, rsp=self.rsp, poly=poly, dist=dist,
                               decay=decay, roi=roi)
            m_view = sati.Model(image[:,8:], rsp=self.rsp[:,:,8:], poly=poly,
                                dist=dist, decay=decay)
            m_roi.optimize(method=mtd, verbosity=0, options=self.options[mtd])
            m_view.optimize(method=mtd, verbosity=0, options=self.options[mtd])

            with self.subTest(method=mtd, parameter='loc'):
                np.testing.assert_allclose(
                    m_roi.dist.loc - np.mean(m_roi.dist.loc),
                    m_view.dist.loc - np.mean(m_view.dist.loc), rtol=1.6e-5)
            with self.subTest(method=mtd, parameter='scale'):
                np.testing.assert_allclose(m_roi.dist.scale, m_view.dist.scale,
                                           rtol=2e-3)
            with self.subTest(method=mtd, parameter='tau'):
                np.testing.assert_allclose(m_roi.decay.tau, m_view.decay.tau,
                                           rtol=3e-3)
            with self.subTest(method=mtd, parameter='coef'):
                np.testing.assert_allclose(m_roi.decay.coef, m_view.decay.coef,
                                           rtol=3e-3)


class TestModel2(unittest.TestCase):
    """Test class of model.py, tests without optimizing calculations."""

    def test_invalid_arguments(self):
        """Test cases invalid arguments are given to sati.Model()."""
        image = [[0, 1], [2, 3]]
        with self.assertRaisesRegex(sati.model.ArgumentTypeError,
                                    'numpy.ndarray'):
            sati.Model(image)

        image = np.array(image)
        poly = sati.planes.Poly()
        decay = sati.planes.Decay(tau=1)

        with self.assertRaisesRegex(sati.model.ArgumentTypeError,
                                    'sati.distributions.Distribution'):
            sati.Model(image, dist=1)

        with self.assertRaisesRegex(sati.model.ArgumentTypeError,
                                    'sati.planes.Poly'):
            sati.Model(image, poly=decay)

        with self.assertRaisesRegex(sati.model.ArgumentTypeError,
                                    'sati.planes.Decay'):
            sati.Model(image, decay=poly)

        with self.assertRaisesRegex(sati.model.ArgumentTypeError,
                                    'sati.distributions.Distribution'):
            sati.Model(image, prior=1)

        with self.assertRaisesRegex(sati.model.ArgumentTypeError,
                                    'numpy.ndarray'):
            sati.Model(image, roi=1)

    def test_swap_axes(self):
        xlb = np.arange(16).reshape(4, 4)
        ylb = np.array([[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]])
        xrb = np.array([[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8], [15, 14, 13, 12]])
        yrb = np.array([[12, 8, 4, 0], [13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3]])
        xlt = np.array([[12, 13, 14, 15], [8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]])
        ylt = np.array([[3, 7, 11, 15], [2, 6, 10, 14], [1, 5, 9, 13], [0, 4, 8, 12]])
        xrt = np.array([[15, 14, 13, 12], [11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]])
        yrt = np.array([[15, 11, 7, 3], [14, 10, 6, 2], [13, 9, 5, 1], [12, 8, 4, 0]])

        def test(array, orgdrct):
            t = sati.model._swap_axes(array, orgdrct)
            with self.subTest(parameter=orgdrct+'_forward'):
                np.testing.assert_equal(t, xlb)
            t = sati.model._swap_axes(t, orgdrct, backward=True)
            with self.subTest(parameter=orgdrct+'_backward'):
                np.testing.assert_equal(t, array)

        def test3D(array, orgdrct):
            shape = (2, 4, 4)
            t = sati.model._swap_axes(np.broadcast_to(array, shape), orgdrct)
            with self.subTest(parameter=orgdrct + '_forward'):
                np.testing.assert_equal(t, np.broadcast_to(xlb, shape))
            t = sati.model._swap_axes(t, orgdrct, backward=True)
            with self.subTest(parameter=orgdrct+'_backward'):
                np.testing.assert_equal(t, np.broadcast_to(array, shape))

        d = {'ylb': ylb, 'yrb': yrb, 'xrb': xrb, 'ylt': ylt,
             'xlt': xlt, 'yrt': yrt, 'xrt': xrt,}
        for orgdrct in d:
            test(d[orgdrct], orgdrct)
            test3D(d[orgdrct], orgdrct)

    def test_elapsed(self):
        self.assertEqual(sati.model._elapsed(3600 * 2 + 60 * 3 + 4.9),
                         'elapsed: 2 h 3 m 4 s')
        self.assertEqual(sati.model._elapsed(3600),
                         'elapsed: 1 h 0 m 0 s')
        self.assertEqual(sati.model._elapsed(60 * 48 + 19.6),
                         'elapsed: 48 m 19 s')
        self.assertEqual(sati.model._elapsed(120),
                         'elapsed: 2 m 0 s')
        self.assertEqual(sati.model._elapsed(32.999),
                         'elapsed: 32.999 s')
