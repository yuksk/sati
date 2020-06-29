__all__ = ['Norm', 'Cauchy', 'T', 'VonMises']

from abc import ABCMeta, abstractmethod

import numpy as np
import numexpr as ne
import scipy.special


class Distribution(metaclass=ABCMeta):
    """A base class to construct specific distribution classes.

    All distributions take ``loc`` and ``scale`` as keyword parameters to
    adjust the location and scale of the distribution, just as
    `scipy.stat
    <https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html#shifting-and-scaling>`_,
    unless otherwise noted.

    Optimized parameters are stored in attributes of the same name.
    """
    def __init__(self, loc=None, scale=None):
        """
        Parameters
        ----------
        loc : array-like, default None
            Initial guess of the location parameter.
        scale : array-like, default None
            Initial guess of the scale parameter.
        """
        self.loc = None if loc is None else np.array(loc)
        self.scale = None if scale is None else np.array(scale)

    def __str__(self):
        np.set_printoptions(precision=3)
        return f'centers: {self.loc}\nscales:  {self.scale}'

    def scaler(self, image_scale, backward=False):
        """Apply scaling used to standardize an input image to
        distribution parameters.

        Parameters
        ----------
        image_scale : tuple of float
            Values used to standardize an input image, mean and std.
        backward : bool
            If true, apply the reverse process to distribution parameter
            in order to restore the original scale of an input image.
        """
        (mean, std) = image_scale
        if backward:
            self.loc = self.loc * std + mean
            self.scale = self.scale * std
            return
        if self.loc is not None:
            self.loc = (self.loc - mean) / std
        if self.scale is not None:
            self.scale = self.scale / std

    @abstractmethod
    def pdf(self, t_subtracted):
        """Probability distribution function.

        Parameters
        ----------
        t_subtracted : numpy.ndarray
            One-dimensionalized image subtracted by a model plane.

        Returns
        -------
        numpy.ndarray, 2D
            The first dimension is class, and the second dimension is position.
        """

    def initial_step(self, t_subtracted, gamma):
        """Calculate distribution parameters in the initial step.
        """
        self.pi = self.mixing_coef(gamma)
        if self.loc is None:
            self.loc = self.center(t_subtracted, gamma)
        if self.scale is None:
            self.scale = self.std(t_subtracted, gamma, self.loc)

    def retain(self):
        """Retain the present distribution parameters so that they can be
        reverted later."""
        self.__loc, self.__scale = self.loc, self.scale

    def revert(self):
        """Revert the distribution parameters to those retained beforehand.
        """
        self.loc, self.scale = self.__loc, self.__scale

    @abstractmethod
    def grad(self, t_subtracted, gamma):
        """Calculate derivative of log likelihood with respect to distribution
        parameters.

        Parameters
        ----------
        t_subtracted : numpy.ndarray, 1D
            One-dimensionalized image subtracted by a model plane.
        gamma : numpy.ndarray, 2D
            Two-dimensionalized responsibility. The first and second
            dimensions are class and position, responsibility.

        Returns
        -------
        tuple of numpy.ndarray
            By summing an ndarray of this list along axis 1, derivative
            of the log likelihood is obtained. The ndarrays are returned
            without being summed because the first element (grad_loc_like)
            is also used to calculate derivative with respect to the plane
            coefficients.
        """

    def update_ga(self, new):
        """Update distribution parameters in the M-step.

        Parameters
        ----------
        new : numpy.ndarray
            The size is product of numbers of classes and parameters.
        """
        if isinstance(self.loc, float):
            self.loc, self.scale = new[0], new[1]
        else:
            K = self.loc.size
            self.loc, self.scale = new[:K], new[K:K * 2]

    def x0(self):
        """Create a part of x0 used in scipy.optimize.minimize"""
        return np.concatenate([self.loc, self.scale], axis=None)

    def lb(self):
        """Create a part of lb used in scipy.optimize.Bounds"""
        return np.concatenate([np.full_like(self.loc, -np.inf),
                               np.full_like(self.scale, np.finfo(float).eps)],
                              axis=None)

    def center(self, t, gamma):
        """Calculate center values of a distribution.

        Parameters
        ----------
        t : numpy.ndarray, 1D
            One-dimensionalized image, or one subtracted by a model plane.
        gamma : numpy.ndarray, 2D
            Two-dimensionalized responsibility. The first and second
            dimensions are class and position, responsibility.

        Returns
        -------
        numpy.ndarray
            The center values of each class.
        """
        return np.array([np.average(t, weights=gamma[k])
                         for k in range(gamma.shape[0])])

    def std(self, t, gamma, loc):
        """Calculate standard deviation of a distribution.

        Parameters
        ----------
        t : numpy.ndarray
            One-dimensionalized image, or one subtracted by a model plane.
        gamma : numpy.ndarray
            Two-dimensionalized responsibility.
        loc : numpy.ndarray
            Center values.

        Returns
        -------
        numpy.ndarray
            Standard deviation of each class.
        """
        return np.array([
            0 if (n := np.sum(gamma[k])) == 0
            else np.sqrt(np.sum(np.square(t - loc[k]) * gamma[k]) / n)
            for k in range(gamma.shape[0])])

    def mixing_coef(self, gamma):
        """Calculate mixing coefficients.

        Parameters
        ----------
        gamma : numpy.ndarray
            Two-dimensionalized responsibility.

        Returns
        -------
        numpy.ndarray
            The mixing coefficients of each class.
        """
        return (N_k := gamma.sum(axis=1)) / N_k.sum()

    def fullpdf(self, t_subtracted):
        """Probability density function multiplied by mixing coefficient.

        Parameters
        ----------
        t_subtracted : numpy.ndarray
            One-dimensionalized image subtracted by a model plane.

        Returns
        -------
        numpy.ndarray
            The first dimension is class, and the second dimension is
            position.
        """
        return self.pi.reshape(-1, 1) * self.pdf(t_subtracted)


class Norm(Distribution):
    r"""Class for normal distribution,

    .. math::
       p(x|\mu, \sigma) = \dfrac{1}{\sqrt{2\pi\sigma^2}}
           \exp\left\{-\dfrac{(x-\mu)^2}{2\sigma^2}\right\}

    where :math:`\mu` and :math:`\sigma` are the location and scale
    parameter, respectively.

    Parameters
    ----------
    loc : array-like, default ``None``
        Initial guess of the location parameter. By default, this is
        automatically calculated.
    scale : array-like, default ``None``
        Initial guess of the scale parameter. By default, this is
        automatically calculated.

    Attributes
    ----------
    loc, scale : ``numpy.ndarray``
        Optimized parameters.
    """
    def pdf(self, t_subtracted):
        p = np.pi * 2
        t = t_subtracted.reshape(1, -1)
        s2, l = (self.scale**2).reshape(-1, 1), self.loc.reshape(-1, 1)
        return ne.evaluate('exp(-(t - l)**2 / (2 * s2)) / sqrt(p * s2)')

    def update_quick(self, t_subtracted, gamma):
        """Calculate distribution parameters in the M-step using the
        quick method.

        Parameters
        ----------
        t_subtracted : numpy.ndarray, 1D
            One-dimensionalized image subtracted by a model plane.
        gamma : numpy.ndarray, 2D
            Two-dimensionalized responsibility. The first and second
            dimensions are class and position, responsibility.
        """
        self.weight = gamma
        self.retain()
        self.loc = self.center(t_subtracted, gamma)
        self.scale = self.std(t_subtracted, gamma, self.loc)
        self.pi = self.mixing_coef(gamma)

    def grad(self, t_subtracted, gamma):
        # Derivative is obtained by summing the following arrays in axis 1
        # (position). But an array before summation (grad_loc_like) is used
        # for calculating derivative with respect to plane coefficients.
        # Therefore, arrays are returned without the summation.
        p = np.pi * 2
        pdf = self.pdf(t_subtracted)
        t = t_subtracted.reshape(1, -1)
        s, l = self.scale.reshape(-1, 1), self.loc.reshape(-1, 1)
        grad_loc_like = ne.evaluate('gamma * (t - l) / s**2')
        grad_scale_like = ne.evaluate('gamma * (t**2 - 2 * t * l + l**2 - s**2) / s**3')
        return (grad_loc_like, grad_scale_like)


class Cauchy(Distribution):
    r"""Class for Cauchy distribution.

    .. math::
       p(x|\mu, \sigma) = \dfrac{1}{\pi}
                          \dfrac{\sigma}{(x-\mu)^2+\sigma^2}

    where :math:`\mu` and :math:`\sigma` are the location and scale
    parameter, respectively.

    Parameters
    ----------
    loc : array-like, default ``None``
        Initial guess of the location parameter. By default, this is
        automatically calculated.
    scale : array-like, default ``None``
        Initial guess of the scale parameter. By default, this is
        automatically calculated.

    Attributes
    ----------
    loc, scale : ``numpy.ndarray``
        Optimized parameters.
    """
    def pdf(self, t_subtracted):
        p = np.pi
        t = t_subtracted.reshape(1, -1)
        s, l = self.scale.reshape(-1, 1), self.loc.reshape(-1, 1)
        return ne.evaluate('s / (((t-l)**2 + s**2) * p)')

    def update_quick(self, t_subtracted, gamma):
        """Calculate distribution parameters in the M-step using the quick
        method.

        Parameters
        ----------
        t_subtracted : numpy.ndarray, 1D
            One-dimensionalized image subtracted by a model plane.
        gamma : numpy.ndarray, 2D
            Two-dimensionalized responsibility. The first and second
            dimensions are class and position, responsibility.
        """
        p = np.pi
        t = t_subtracted.reshape(1, -1)
        s, l = self.scale.reshape(-1, 1), self.loc.reshape(-1, 1)
        self.weight = ne.evaluate('gamma / (((t-l)**2 + s**2) * p)')
        gamma_tilde = self.weight * self.scale.reshape(-1, 1)
        self.retain()
        self.loc = self.center(t_subtracted, gamma_tilde)
        self.scale = self.__scaling_param(gamma, gamma_tilde)
        self.pi = self.mixing_coef(gamma)

    def __scaling_param(self, gamma, gamma_tilde):
        numerator = gamma.sum(axis=1)
        denominator = gamma_tilde.sum(axis=1) * 2 * np.pi
        return np.divide(numerator, denominator, where=denominator!=0)

    def grad(self, t_subtracted, gamma):
        p = np.pi * 2
        pdf = self.pdf(t_subtracted)
        t = t_subtracted.reshape(1, -1)
        s, l = self.scale.reshape(-1, 1), self.loc.reshape(-1, 1)
        # Derivative is obtained by summing the following arrays in axis 1
        # (position). But an array before summation (grad_loc_like) is used
        # for calculating derivative with respect to plane coefficients.
        # Therefore, arrays are returned without the summation.
        grad_loc_like = ne.evaluate('gamma * pdf * p / s * (t - l)')
        grad_scale_like = ne.evaluate('gamma * (1 / s - pdf * p)')
        return (grad_loc_like, grad_scale_like)


class T(Distribution):
    r"""Class for Student's t distribution,

    .. math::
       p(x|\mu, \sigma, \mu) =
            \dfrac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sigma\sqrt{\pi\nu}}
            \left\{1+\dfrac{1}{\nu}
                   \left(\dfrac{x-\mu}{\sigma}\right)^2\right\}
            ^{-(\nu+1)/2}

    where :math:`\mu`, :math:`\sigma`, and :math:`\nu` are the location,
    scale, and degree of freedom parameter, respectively.

    Parameters
    ----------
    loc : array-like, default ``None``
        Initial guess of the location parameter. By default, this is
        automatically calculated.
    scale : array-like, default ``None``
        Initial guess of the scale parameter. By default, this is
        automatically calculated.
    df : array-like, default ``None``
        Initial guess of the degree of freedom. By default, this is
        ``numpy.ones_like(loc)``.

    Attributes
    ----------
    loc, scale, df : ``numpy.ndarray``
        Optimized parameters.
    """
    def __init__(self, loc=None, scale=None, df=None):
        super().__init__(loc=loc, scale=scale)
        self.df = None if df is None else np.array(df)

    def __str__(self):
        return super().__str__() + f'\ndegree of freedom: {self.df}'

    def pdf(self, t_subtracted):
        u = (1 / (self.scale**2 * self.df)).reshape(-1, 1)
        v = self.df.reshape(-1, 1)
        return np.sqrt(u) / scipy.special.beta(v * 0.5, 0.5) \
            * (1 + u * (t_subtracted.reshape(1, -1) \
            - self.loc.reshape(-1, 1))**2)**(-0.5*v - 0.5)

    def initial_step(self, t_subtracted, gamma):
        super().initial_step(t_subtracted, gamma)
        if self.df is None:
            self.df = np.ones_like(self.loc)

    def retain(self):
        super().retain()
        self.__df = self.df

    def revert(self):
        super().revert()
        self.df = self.__df

    def grad(self, t_subtracted, gamma):
        x = t_subtracted.reshape(1, -1) - self.loc.reshape(-1, 1)
        y = 1 / self.scale
        u = (x * y.reshape(-1, 1))**2 + self.df.reshape(-1, 1)
        v = (self.df.reshape(-1, 1) + 1) * np.reciprocal(u)
        w = 1 + scipy.special.psi((self.df + 1) * 0.5) \
            - scipy.special.psi(self.df * 0.5) + np.log(self.df)
        # Derivative is obtained by summing the following arrays in axis 1
        # (position). But an array before summation (grad_loc_like) is used
        # for calculating derivative with respect to plane coefficients.
        # Therefore, arrays are returned without the summation.
        grad_loc_like = gamma * x * v * (y ** 2).reshape(-1, 1)
        grad_scale_like = gamma * (1 - v) * (self.df * y).reshape(-1, 1)
        grad_df_like = gamma * 0.5 * (w.reshape(-1, 1) - np.log(u) - v)
        return(grad_loc_like, grad_scale_like, grad_df_like)

    def update_ga(self, new):
        super().update_ga(new)
        self.df = new[self.df.size*2:]

    def x0(self):
        return np.concatenate([self.loc, self.scale, self.df], axis=None)

    def lb(self):
        inf = np.full_like(self.loc, -np.inf)
        epsilon = np.full_like(self.scale, np.finfo(float).eps)
        return np.concatenate([inf, epsilon, epsilon], axis=None)


class VonMises(Distribution):
    r"""Class for von Mises distribution.

    .. note::

        This class is intended as a prior distribution for the location
        parameter :math:`\mu` of either :class:`sati.distributions.Norm`,
        :class:`sati.distributions.Cauchy`, or
        :class:`sati.distributions.T`.

    .. math::
       p(\mu|c_0, \varphi_0, \kappa) = \dfrac{1}{2\pi I_0(\kappa)}
           \exp\left\{
               \kappa\cos\left(\dfrac{2\pi\mu}{c_0}-\varphi_0\right)
           \right\}

    where :math:`\varphi_0`, :math:`c_0`, and :math:`\kappa` are the
    location, scale, and concentration parameters, respectively.
    :math:`I_0` is the modified Bessel function of the first kind
    at order 0.

    This is slightly different from the standard form of von Mises
    distribution to estimate a unit height of steps.

    Parameters
    ----------
    loc : ``float``, default 0
        Initial guess of the location parameter :math:`\varphi_0`.
    scale : ``float``
        Initial guess of the scale parameter :math:`c_0`. This takes
        an initial guess of a unit height of steps.
    kappa : array-like, default 1.0
        The concentration parameter. The length of ``kappa`` must be the
        same as number of terraces. To exclude a terrace from the MAP
        estimation, set the corresponding element of ``kappa`` to 0.

    Attributes
    ----------
    loc, scale  : ``float``
        Optimized parameters.
    """
    def __init__(self, *, loc=0., scale, kappa=[1.]):
        self.loc, self.scale, self.__kappa = loc, scale, np.asarray(kappa)

    def __str__(self):
        return f'spacing: {self.scale:.3f}'

    def pdf(self, mu):
        """Probability distribution function.

        Parameters
        ----------
        mu : ``numpy.ndarray``
            The ``loc`` attribute of either :class:`sati.distributions.Norm`,
            :class:`sati.distributions.Cauchy`, or
            :class:`sati.distributions.T` is expected.

        Returns
        -------
        ``numpy.ndarray``
            Values at each class (terrace)
        """
        return np.exp(self.__kappa * np.cos(2 * np.pi * mu / self.scale
                                            - self.loc)
                     ) / (2 * np.pi * scipy.special.i0(self.__kappa))

    def loglikelihood_at_pixel(self, mu):
        """
        Parameters
        ----------
        mu : ``numpy.ndarray``
            The ``loc`` attribute of either :class:`sati.distributions.Norm`,
            :class:`sati.distributions.Cauchy`, or
            :class:`sati.distributions.T` is expected.
        """
        return (self.__kappa * np.cos(2 * np.pi * mu / self.scale - self.loc)
                - self._logi0(self.__kappa)).sum()

    def grad(self, mu, N):
        """
        Parameters
        ----------
        mu : ``numpy.ndarray``
            The ``loc`` attribute of either :class:`sati.distributions.Norm`,
            :class:`sati.distributions.Cauchy`, or
            :class:`sati.distributions.T` is expected.
        N : int
            Size in position.
        """
        u = self.__kappa * np.sin(2 * np.pi * mu / self.scale - self.loc)
        grad_mu = -2 * np.pi * N / self.scale * u
        grad_loc = N * u.sum()
        grad_scale = 2 * np.pi * N / (self.scale * self.scale) * (mu * u).sum()
        return (grad_mu, grad_loc, grad_scale)

    def scaler(self, image_scale, backward=False):
        (_, std) = image_scale
        self.scale = self.scale * std if backward else self.scale / std

    @classmethod
    def _logi0(cls, x):
        """This function returns a value of np.log(2*np.pi*np.i0(x)).
        When x >~ 800, scipy.special.i0(x) exceeds the limit of float64
        although np.log(scipy.special.i0(x)) is still not so large.
        To avoid this issue, np.log(np.i0(x)) is directly calculated
        without calculating np.i0(x). This is a logarithmic version of
        np.lib.function_base._i0_2(x), which is
        exp(x) * _chbevl(32.0/x - 2.0, _i0B) / sqrt(x)
        """
        return np.piecewise(x, [x < 700.], [cls.__logi0_1, cls.__logi0_2])

    @staticmethod
    def __logi0_1(x):
        return np.log(2 * np.pi * scipy.special.i0(x))

    @classmethod
    def __logi0_2(cls, x):
        return np.log(2 * np.pi) + x \
                + np.log(cls.__chbevl(32.0 / x - 2.0, cls.__i0B)) \
                - 0.5 * np.log(x)

    @staticmethod
    def __chbevl(x, vals):
        """This is a copy of np.lib.function_base._chbevl"""
        b0, b1 = vals[0], 0.0

        for i in range(1, len(vals)):
            b2 = b1
            b1 = b0
            b0 = x*b1 - b2 + vals[i]

        return 0.5 * (b0 - b2)

    # This is a copy of np.lib.function_base._i0B
    __i0B = (
        -7.23318048787475395456E-18,
        -4.83050448594418207126E-18,
        4.46562142029675999901E-17,
        3.46122286769746109310E-17,
        -2.82762398051658348494E-16,
        -3.42548561967721913462E-16,
        1.77256013305652638360E-15,
        3.81168066935262242075E-15,
        -9.55484669882830764870E-15,
        -4.15056934728722208663E-14,
        1.54008621752140982691E-14,
        3.85277838274214270114E-13,
        7.18012445138366623367E-13,
        -1.79417853150680611778E-12,
        -1.32158118404477131188E-11,
        -3.14991652796324136454E-11,
        1.18891471078464383424E-11,
        4.94060238822496958910E-10,
        3.39623202570838634515E-9,
        2.26666899049817806459E-8,
        2.04891858946906374183E-7,
        2.89137052083475648297E-6,
        6.88975834691682398426E-5,
        3.36911647825569408990E-3,
        8.04490411014108831608E-1
    )
