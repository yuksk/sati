__all__ = ['Poly', 'Decay']

from abc import ABCMeta, abstractmethod

import numexpr as ne
import numpy as np
from scipy import linalg

import sati.preprocessing


class Plane(metaclass=ABCMeta):
    """A base class to construct specific plane classes."""
    @abstractmethod
    def setup(self, shape, roi):
        """Set up variables of a model plane.

        Parameters
        ----------
        shape : tuple of int
            Shape of an input image.
        roi : ndarray
            A 2D array indicating an region of interst.
        """

    @abstractmethod
    def update_ga(self, step):
        """Update the decay coefficients and a model plane."""

    @abstractmethod
    def x0(self):
        """Create a part of x0 used in scipy.optimize.minimize"""

    @abstractmethod
    def lb(self):
        """Create a part of lb used in scipy.optimize.Bounds"""

    @abstractmethod
    def retain(self):
        """Retain the present coefficients and plane so that they can be
        reverted later.
        """

    @abstractmethod
    def revert(self):
        """Revert the coefficients and plane to the retained ones."""

    @abstractmethod
    def grad(self, u):
        """Calculate derivative of log likelihood with respect to plane
        coefficients.

        Parameters
        ----------
        u : numpy.ndarray
            A part common to the derivative of log likelihood with respect
            to mu (center). The first and second dimensions are class and
            position, responsibility.

        Returns
        -------
        numpy.ndarray
            Derivative of log likelihood with respect to plane coefficients.
        """

    @abstractmethod
    def finalize(self):
        """Calculate a full plane."""


class Poly(Plane):
    r"""Class for a polynomial plane.

    .. math::
       f(n) = w_0 + w_1x_n + w_2y_n + w_3x_n^2 + w_4x_ny_n + w_5y_n^2 + \dots

    Parameters
    ----------
    degree : ``int``, default 1
        Degree of the polynomial plane.
    coef : ``numpy.ndarray``, default ``None``
        Initial guess of :math:`w_i`. By default, this is automatically
        caluculated from an input image and an initial guess of the
        responsibility.

    Attributes
    ----------
    coef, plane : ``numpy.ndarray``
        Optimized :math:`w_i` and :math:`f(n)`, responsibility.
    """
    def __init__(self, degree=1, coef=None):
        self.__degree = degree
        self.coef = coef

    def setup(self, shape, roi):
        self.__shape = shape
        self.__roi = roi

        # One-dimensionalized positions.
        x = np.tile(range(shape[1]), shape[0])
        y = np.arange(shape[0]).repeat(shape[1])

        # Separate the bias term from the others for standardization later.
        poly = sati.preprocessing.PolynomialFeatures(self.__degree)
        phi_poly = poly.transform(np.vstack((x, y)))
        phi_bias = np.ones(phi_poly.shape[1])

        # Standardize the polynomial terms and add the bias term after
        # the standardization. The first and second dimension of self.__phi
        # is degree and position, respectively. When an ROI is used, it is
        # applied before standardization.
        if roi is None:
            phi_poly_std, _ = sati.preprocessing.standardize(phi_poly)
            self.__phi = np.vstack((phi_bias, phi_poly_std))
        else:
            phi_poly_roi = np.compress(roi.reshape(-1) > 0, phi_poly, axis=1)
            phi_bias_roi = np.ones(phi_poly_roi.shape[1])
            phi_poly_roi_std, scaler = \
                sati.preprocessing.standardize(phi_poly_roi)
            phi_poly_std, _ = sati.preprocessing.standardize(phi_poly, scaler)
            self.__phi = np.vstack((phi_bias_roi, phi_poly_roi_std))
            self.__phi_full = np.vstack((phi_bias, phi_poly_std))

        if self.coef is not None:
            # Expand or truncate if necessary
            ndegree = self.__phi.shape[0]
            if self.coef.size < ndegree:
                self.coef = np.pad(self.coef, [0, ndegree - self.coef.size],
                                   'constant')
            else:
                self.coef = self.coef[:ndegree]

    def initial_step(self, t, gamma, mu):
        """Calculate initial values required in the initial step of the
        main loop.

        Unless plane coefficients are given, calculate those for each
        class from the initial responsibility and the initial center
        values, and use the average weighted by the initial responsibility
        as the initial coefficients.
        """
        if self.coef is None:
            nclass = gamma.shape[0]
            A = [(gamma[k].reshape(1, -1) * self.__phi) @ self.__phi.T
                    for k in range(nclass)]
            b = [self.__phi @ (gamma[k] * (t - mu[k])) for k in range(nclass)]
            x = [linalg.solve(A[k], b[k]) for k in range(nclass)]
            self.coef = np.average(x, axis=0, weights=gamma.sum(axis=1))
        self.plane = self.__calc_plane()

    def update_quick(self, t, weight, mu):
        """Calculate the plane coefficients and update a model plane with
        the coefficients. This is used in the quick method.
        """
        A = (weight.sum(axis=0).reshape(1, -1) * self.__phi) @ self.__phi.T
        b = self.__phi @ (t * weight.sum(axis=0) - mu @ weight)
        self.retain()
        self.coef = linalg.solve(A, b)
        self.plane = self.__calc_plane()

    def update_ga(self, new):
        self.coef = new
        self.plane = self.__calc_plane()

    def x0(self):
        return self.coef

    def lb(self):
        return np.full_like(self.coef, -np.inf)

    def retain(self):
        self.__coef, self.__plane = self.coef, self.plane

    def revert(self):
        self.coef, self.plane = self.__coef, self.__plane

    def grad(self, u):
        return self.__phi @ u.sum(axis=0)

    def finalize(self, std):
        if self.__roi is None:
            self.plane = self.plane.reshape(self.__shape) * std
        else:
            self.plane = np.dot(self.__phi_full.T, self.coef * std) \
                           .reshape(self.__shape)

    def __calc_plane(self):
        return self.coef @ self.__phi


class Decay(Plane):
    r"""Class for a decay plane.

    .. math::
       f(n) = \sum_i A_i\exp(-n/\tau_i)
       \quad\mathrm{or}\quad
       f(n) = \sum_i A_i\ln(n+\tau_i)

    Parameters
    ----------
    tau : ``float`` or array-like
        Initial guess of :math:`\tau_i` in the unit of pixels.
    coef : ``float`` or array-like, default 1
        Initial guess of :math:`A_i`
    kind : {'exp', 'log'}, default 'log'
        Decay function.
    orgdrct : ``str``, default 'lbx'
        Origin and direction of decay specifying chronological order of
        pixels in an image. The origin is the starting point of scan and
        the direction is that of fast scan. The origin is described by
        either 'lb', 'rb', 'lt', or 'rt', denoting left bottom, right
        bottom, left top, and right bottom, respectively. The direction
        is described by either 'x' or 'y'.

        .. image:: orgdrct.svg

        The characters are case-insensitive and have no particular order.

    Attributes
    ----------
    tau, coef, plane : ``numpy.ndarray``
        Optimized :math:`\tau_i`, :math:`A_i` and :math:`f(n)`,
        responsibility.
    orgdrct : ``str``
        Origin and direction of decay.
    """
    def __init__(self, *, tau, coef=None, kind='log', orgdrct='lbx'):
        self.tau = np.array(tau)
        if np.any(self.tau <= 0):
            raise ValueError("The 'tau' must be positive.")

        self.coef = None if coef is None else np.array(coef)

        if kind not in ('exp', 'log'):
            raise ValueError("The 'kind' argument must be 'exp' or 'log'.")
        self.__kind = kind

        if any([(c not in 'lrtbxy') for c in orgdrct.lower()]):
            raise ValueError('Unknown flag is included in orgdrct')
        # self.orgdrct is used in not this class but sati.Model to reorder
        # image, responsibility, and roi (if any)
        self.orgdrct = orgdrct.lower()

    def __str__(self):
        np.set_printoptions(precision=3)
        return f'{self.__kind} decay (tau): {self.tau} (pixel)\n' \
               f'          (A): {self.coef}'

    def setup(self, shape, scale, roi):
        self.__shape = shape
        self.__scale = scale
        self.__roi = roi

        self.__index = np.linspace(0, 1, num=np.prod(shape), endpoint=False)
        if roi is not None:
            self.__index = np.compress(roi.reshape(-1) > 0, self.__index)
        self.plane = np.empty_like(self.__index)

        if self.__kind == 'exp':
            self.__beta = -np.prod(shape) / np.array(self.tau)
        else:
            # use beta**2 instead of tau to guarantee that log takes
            # a positive number
            self.__beta = np.sqrt(np.array(self.tau) / np.prod(shape))

        self.coef = np.ones_like(self.__beta) if self.coef is None \
                else self.coef / self.__scale[1]

        self.grad = {'exp': self._grad_exp,
                     'log': self._grad_log}[self.__kind]
        self.__calc_plane = {'exp': self._calc_plane_exp,
                             'log': self._calc_plane_log}[self.__kind]

    def update_ga(self, new):
        self.__beta, self.coef = new[:self.__beta.size], new[self.__beta.size:]
        self.plane = self.__calc_plane()

    def x0(self):
        return np.concatenate([self.__beta, self.coef], axis=None)

    def lb(self):
        inf = np.full_like(self.coef, -np.inf)
        return np.concatenate([inf, inf], axis=None)

    def retain(self):
        self.__beta_ = self.__beta
        self.__coef, self.__plane = self.coef, self.plane

    def revert(self):
        self.__beta = self.__beta_
        self.coef, self.plane = self.__coef, self.__plane

    def grad(self):
        """This function is assigned to either _grad_exp() or _grad_log()
        in setup(). The names of functions to be assigned start with not __
        but _ for pickling."""

    def _grad_exp(self, u):
        i, b = self.__index.reshape(1, -1), self.__beta.reshape(-1, 1)
        v = u.sum(axis=0).reshape(1, -1) * ne.evaluate('exp(b * i)')
        grad__beta = ne.evaluate('sum(i * v, axis=1)') * self.coef
        grad_coef = ne.evaluate('sum(v, axis=1)')
        return np.concatenate([grad__beta, grad_coef])

    def _grad_log(self, u):
        i, b = self.__index.reshape(1, -1), self.__beta.reshape(-1, 1)
        v, w = u.sum(axis=0).reshape(1, -1), ne.evaluate('i + b**2')
        grad__beta = ne.evaluate('sum(v / w, axis=1)') * 2 * self.coef * self.__beta
        grad_coef = ne.evaluate('sum(v * log(w), axis=1)')
        return np.concatenate([grad__beta, grad_coef])

    def finalize(self, std):
        if self.__roi is None:
            self.plane = self.plane.reshape(self.__shape) * std
        else:
            index = np.linspace(0, 1, num=np.prod(self.__shape),
                                endpoint=False)
            if self.__kind == 'exp':
                self.plane = np.dot(np.exp(self.__beta.reshape(1, -1)
                                           * index.reshape(-1, 1)),
                                    self.coef * std).reshape(self.__shape)
            else:
                self.plane = np.dot(np.log(index.reshape(-1, 1)
                                           + (self.__beta**2).reshape(1, -1)),
                                    self.coef * std).reshape(self.__shape)

        if self.__kind == 'exp':
            self.tau = -np.prod(self.__shape) / self.__beta
        else:
            self.tau = self.__beta**2 * np.prod(self.__shape)
        self.coef *= self.__scale[1]

    def __calc_plane(self):
        """This function is assigned to either _calc_plane_exp() or
        _calc_plane_log() in setup(). The names of functions to be
        assigned start with not __ but _ for pickling."""

    def _calc_plane_exp(self):
        i, b = self.__index.reshape(-1, 1), self.__beta.reshape(1, -1)
        return np.dot(ne.evaluate('exp(b * i)'), self.coef)

    def _calc_plane_log(self):
        i, b = self.__index.reshape(-1, 1), self.__beta.reshape(1, -1)
        return np.dot(ne.evaluate('log(i + b**2)'), self.coef)
