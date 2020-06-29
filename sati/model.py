import pickle
import sys
import time
import warnings
from copy import deepcopy

import numpy as np
import scipy.optimize

import sati.objective
import sati.preprocessing
import sati.optimize

class Model:
    """Class for describing a model.

    Parameters
    ----------
    image : ``numpy.ndarray``
        An input image.
    rsp : ``numpy.ndarray``
        An initial guess of the responsibility. This takes a 3D array.
        The first dimension is terrace, and the second and third
        dimensions are positions.
    dist : :class:`sati.distributions.Distribution`
        An initial guess of a distribution.
    poly : :class:`sati.planes.Poly`
        An initial guess of a polynomial plane.
    decay : :class:`sati.planes.Decay`
        An initial guess of a decay plane.
    prior : :class:`sati.distributions.Distribution`, default ``None``
        A prior distribution for ``dist.loc``.
    roi : ``numpy.ndarray``, default ``None``
        A 2D array that has the same shape as the image array and
        specifies an region of interest. Set pixels to be calculated
        to 1.  By default, the whole area is used for the calculation.

    Attributes
    ----------
    objective : ``list`` of :class:`sati.objective.Objective`
        The first element is for the main EM loop, and the second
        element is for the M step. If the quick method is used,
        the second one is not used.
    image : ``numpy.ndarray``
        A copy of the input image.
    subtracted : ``numpy.ndarray``
        An image subtracted by an estimated model plane.
    rsp : ``numpy.ndarray``
        An estimated responsibility.
    dist : :class:`sati.distributions.Distribution`
        An estimated distribution
    poly : :class:`sati.planes.Poly`
        An estimated polynomial plane
    decay : :class:`sati.planes.Decay`
        An estimated decay plane
    prior : :class:`sati.distributions.Distribution`
        An estimated distribution
    n_iter : ``int``
        A number of iterations run by the main EM loop to reach
        the specified tolerance.
    """
    def __init__(self, image, rsp=None, dist=None, poly=None, decay=None,
            prior=None, roi=None):
        self.image, self.rsp, self.dist, self.poly, self.decay, self.prior, \
            self.roi =  image, rsp, dist, poly, decay, prior, roi

    def __validate(self, name, value):
        type_ = self.__ATTRIBUTES[name][0]
        isNoneAllowed = self.__ATTRIBUTES[name][1]
        if isNoneAllowed and (value is None):
            pass
        elif not isinstance(value, type_):
            raise ArgumentTypeError(name, type_)

    @property
    def __ATTRIBUTES(self):
        # The second element of the tuple is if None is allowed.
        return {'image': (np.ndarray, False),
                'rsp': (np.ndarray, True),
                'dist': (sati.distributions.Distribution, True),
                'poly': (sati.planes.Poly, True),
                'decay': (sati.planes.Decay, True),
                'prior': (sati.distributions.Distribution, True),
                'roi': (np.ndarray, True),}

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, value):
        self.__validate('image', value)
        self.__image = np.copy(value)

    @property
    def rsp(self):
        return self.__rsp

    @rsp.setter
    def rsp(self, value):
        self.__validate('rsp', value)
        self.__rsp = np.copy(value) if value is not None else None

    @property
    def dist(self):
        return self.__dist

    @dist.setter
    def dist(self, value):
        self.__validate('dist', value)
        self.__dist = deepcopy(value)

    @property
    def poly(self):
        return self.__poly

    @poly.setter
    def poly(self, value):
        self.__validate('poly', value)
        self.__poly = deepcopy(value)

    @property
    def decay(self):
        return self.__decay

    @decay.setter
    def decay(self, value):
        self.__validate('decay', value)
        self.__decay = deepcopy(value)

    @property
    def prior(self):
        return self.__prior

    @prior.setter
    def prior(self, value):
        self.__validate('prior', value)
        self.__prior = deepcopy(value)

    @property
    def roi(self):
        return self.__roi

    @roi.setter
    def roi(self, value):
        self.__validate('roi', value)
        self.__roi = np.copy(value) if value is not None else None

    def __getstate__(self):
        d = {}
        for key in self.__ATTRIBUTES:
            d.update({key: getattr(self, f'_{self.__class__.__name__}__{key}')})
        return d

    def __setstate__(self, state):
        for key in state:
            setattr(self, f'_{self.__class__.__name__}__{key}', state[key])

    def optimize(self, method='auto', maxiter=256, tol=1e-7, verbosity=1,
                 options=None):
        """Optimize model parameters using the expectation-maximization
        (EM) algorithm.

        Parameters
        ----------
        quick : ``bool``, default ``True``
            Use the quick method when possible, that is, Normal or Cauchy
            distribution is used, ``decay`` is ``None``, and ``prior``
            is ``None``.
        method : {'auto', 'l-bfgs-b', 'adam', 'quick'}, default ``auto``
            The method used for the optimization in the M-step.
            When ``auto``, ``quick`` is used if ``dist`` is the normal or
            Cauchy distribution, otherwise ``l-bfgs-b`` is used.
        maxiter: ``int``, default 256
            The maximum number of iterations of the main EM loop.
        tol : ``float``, default 1e-7
            The fractional tolerance to stop the iteration of the main EM loop.
        verbosity : {0, 1, 2}, default 1
            Set 1 to show a summary of result, 2 to show values of
            objective function in the console during calculation, and
            0 to show nothing.

            .. note::

               If you use the Win32 console, 1 (default) or 0 is recommended
               because the values are not shown neatly. (The Win32 console
               does not support the ANSI escape code.)

        options : ``dict``, default ``None``
            A dictionary of optimizer options. If ``None``, the default value
            different for each optimizer is used.

            * For ``method=='l-bfgs-b'``, see `scipy.optimize.minimize(method='L-BFGS-B') <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_
            * For ``method=='adam'``, see :func:`sati.optimize.minimize`.

        """
        self.__maxiter = maxiter
        self.__verbosity = verbosity
        self.__options = options
        self.objective = sati.objective.Objective(maxiter, tol, verbosity==2)
        self.__method = method.lower()
        self.__m_step = self.__select_m_step()

        self.__setup()
        self.__run_em()
        self.__finalize()

    def __setup(self):
        """Reduce dimension, apply roi (if any), standardize."""
        if self.decay is not None:
            self.__swap_axes()

        self.__t = self.image.reshape(-1)
        if self.rsp is None:
            raise NoAttributeError('rsp')
        self.__gamma = self.rsp.reshape((self.rsp.shape[0], -1))

        if self.roi is not None:
            self.roi = (self.roi > 0).reshape(-1)
            self.__t = np.compress(self.roi, self.__t)
            self.__gamma = np.compress(self.roi, self.__gamma, axis=1)

        self.__t, self.__image_scale = sati.preprocessing.standardize(self.__t)

        if self.poly is None:
            raise NoAttributeError('poly')
        self.poly.setup(self.image.shape, self.roi)

        if self.dist is None:
            raise NoAttributeError('dist')
        self.dist.scaler(self.__image_scale)

        self.__classes = [self.poly, self.dist]

        if self.decay is not None:
            self.decay.setup(self.image.shape, self.__image_scale, self.roi)
            self.__classes.append(self.decay)

        if self.prior is not None:
            self.prior.scaler(self.__image_scale)
            self.__classes.append(self.prior)

    def __swap_axes(self, backward=False):
        self.image = _swap_axes(self.image, self.decay.orgdrct, backward)
        self.rsp = _swap_axes(self.rsp, self.decay.orgdrct, backward)
        if self.roi is not None:
            self.roi = _swap_axes(self.roi, self.decay.orgdrct, backward)
        if backward:
            self.subtracted = _swap_axes(self.subtracted, self.decay.orgdrct,
                                         backward)

    def __select_m_step(self):
        if self.__method not in ['auto', 'l-bfgs-b', 'adam', 'quick']:
            raise ValueError('unknown method.')

        is_quick_available = all([
            hasattr(self.dist, 'update_quick'),
            self.decay is None, self.prior is None])

        if self.__method == 'quick':
            if is_quick_available:
                return self.__m_step_quick
            else:
                raise ValueError("'quick' is not available.")
        if self.__method == 'l-bfgs-b':
            return self.__m_step_ga_bfgs
        if self.__method == 'adam':
            return self.__m_step_ga_adam

        # The following is for 'auto'
        if is_quick_available:
            self.__method = 'quick'
            return self.__m_step_quick
        else:
            self.__method = 'l-bfgs-b'
            return self.__m_step_ga_bfgs

    def __run_em(self):
        start = time.time()
        self.__initial_step()
        try:
            for self.n_iter in range(1, self.__maxiter):
                self.__e_step()
                self.__m_step()
                if (self.objective.isconverged()):
                    break
        except KeyboardInterrupt:
            warnings.simplefilter('ignore', SatiWarning)
            print('\b\bCaught KeyboardInterrupt.')
            pass
        self.__elapsed = time.time() - start

    def __initial_step(self):
        """Do the initial step of the EM loop.

        The initial values of polynomial plane coefficients are determined,
        followed by those of the distribution parameters.
        """
        loc = self.dist.center(self.__t, self.__gamma)
        self.poly.initial_step(self.__t, self.__gamma, loc)

        self.__t_subtracted = self.__t - self.poly.plane
        self.dist.initial_step(self.__t_subtracted, self.__gamma)

        self.objective.append(
            sati.objective.calc(self.dist, self.__t_subtracted,
                                prior=self.prior))

    def __e_step(self):
        """Do the E-step of the EM algorithm and update the gamma."""
        numerator = self.dist.fullpdf(self.__t_subtracted)
        denominator = numerator.sum(axis=0)
        self.__gamma = np.divide(numerator, denominator, where=denominator!=0)

    def __m_step_quick(self):
        """Do the M-step with the quick algorithm."""
        self.dist.update_quick(self.__t_subtracted, self.__gamma)
        obj0 = sati.objective.calc(self.dist, self.__t_subtracted)
        if obj0 < self.objective.value[-1]:
            self.dist.revert()
            obj0 = self.objective.value[-1]

        self.poly.update_quick(self.__t, self.dist.weight, self.dist.loc)
        self.__t_subtracted = self.__t - self.poly.plane
        obj1 = sati.objective.calc(self.dist, self.__t_subtracted)
        if obj1 < obj0:
            self.poly.revert()
            self.__t_subtracted = self.__t - self.poly.plane
            obj1 = obj0

        self.objective.append(obj1)

    def __m_step_ga_bfgs(self):
        """Do the M-step using the L-BFGS-B algorithm."""
        x0 = [c.x0() for c in self.__classes]
        self.__m_step_sizes = [x.size for x in x0]
        bounds = scipy.optimize.Bounds(
                np.concatenate([c.lb() for c in self.__classes], axis=None),
                np.inf)
        result = scipy.optimize.minimize(
                self.__m_step_ga_fun, np.concatenate(x0, axis=None),
                method='L-BFGS-B', jac=self.__m_step_ga_jac, bounds=bounds,
                options=self.__options)

        if not result.success:
            warnings.warn(MstepWarning(result.message))

        self.dist.pi = self.dist.mixing_coef(self.__gamma)
        self.objective.append(result.fun*(-1))

    def __m_step_ga_adam(self):
        """Do the M-step using the ADAM algorithm."""
        x0 = [c.x0() for c in self.__classes]
        self.__m_step_sizes = [x.size for x in x0]
        for c in self.__classes:
            c.retain()

        options = dict(sati.optimize.options_default.items())
        if self.__options is not None:
            options.update(self.__options)

        while True:
            result = sati.optimize.minimize(
                    self.__m_step_ga_fun, np.concatenate(x0, axis=None),
                    method='adam', jac=self.__m_step_ga_jac, options=options)
            # Normal exit
            if result.fun*(-1) > self.objective.value[-1]:
                break
            # Redo the M-step after reducing the learning rate
            options['lr'] *= options['reduction']
            if options['lr'] < np.finfo(float).eps:
                warnings.warn(MstepWarning('Small learning rate'))
                break
            for c in self.__classes:
                c.revert()
            self.__t_subtracted = self.__t - self.poly.plane

        self.dist.pi = self.dist.mixing_coef(self.__gamma)
        self.objective.append(result.fun*(-1))

    def __m_step_ga_update_values(self, x):
        n = self.__m_step_sizes  # shortcut
        self.poly.update_ga(x[:(i:=n[0])])
        self.__t_subtracted = self.__t - self.poly.plane
        self.dist.update_ga(x[i:(i:=i+n[1])])
        if self.decay is not None:
            self.decay.update_ga(x[i:(i:=i+n[2])])
            self.__t_subtracted -= self.decay.plane
        if self.prior is not None:
            self.prior.update_ga(x[i:])

    def __m_step_ga_fun(self, x):
        self.__m_step_ga_update_values(x)
        return sati.objective.calc(self.dist, self.__t_subtracted,
                                   prior=self.prior) * (-1)

    def __m_step_ga_jac(self, x):
        self.__m_step_ga_update_values(x)
        gd0 = self.dist.grad(self.__t_subtracted, self.__gamma)
        gpoly = self.poly.grad(gd0[0])
        gdist = np.stack([x.sum(axis=1) for x in gd0])
        glist = [gpoly, gdist]
        if self.decay is not None:
            gdecay = self.decay.grad(gd0[0])
            glist.append(gdecay)
        if self.prior is not None:
            gprior = self.prior.grad(self.dist.loc, self.__t.size)
            gdist[0] += gprior[0]
            glist.append(gprior[1:])
        return np.concatenate(glist, axis=None) * (-1)

    def __finalize(self):
        n_class = self.__gamma.shape[0]
        if self.roi is not None:
            self.__rsp = np.zeros((n_class, self.image.size))
            np.place(self.__rsp,
                     np.broadcast_to(self.roi, self.__rsp.shape),
                     self.__gamma)
        else:
            self.__rsp = self.__gamma
        self.__rsp.resize((n_class,) + self.image.shape)

        self.dist.scaler(self.__image_scale, backward=True)
        self.poly.finalize(self.__image_scale[1])
        self.subtracted = self.image - self.poly.plane

        if self.decay is not None:
            self.decay.finalize(self.__image_scale[1])
            self.subtracted -= self.decay.plane
            self.__swap_axes(backward=True)

        if self.prior is not None:
            self.prior.scaler(self.__image_scale, backward=True)

        # show messages
        if not self.objective.isconverged():
            warnings.warn(NotConvergedWarning(self.__maxiter))

        if self.__verbosity > 0:
            print(f'method: {self.__method}')
            print(self.objective)
            print(self.dist)
            if self.decay is not None:
                print(self.decay)
            if self.prior is not None:
                print(self.prior)
            print(_elapsed(self.__elapsed))

    def pickle(self, file):
        """Pickle a model.

        This is a wrapper of the following.

        .. code-block::

            with open(file, 'wb') as f:
                pickle.dump(self, f)

        Parameters
        ----------
        file : str
            A path-like object of the file to be saved.
        """
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def unpickle(file):
        """Unpickle a model.

        This is a wrapper of the following.

        .. code-block::

            with open(file, 'rb') as f:
                return pickle.load(f)

        Parameters
        ----------
        file : str
            A path-like object of the file to be opened.
        """
        with open(file, 'rb') as f:
            return pickle.load(f)


def _swap_axes(array, orgdrct, backward=False):
    """By flipping and swapping axes, reorder data array to the
    standard order.

    Forward (backward==False) is from real to standardized,
    backward (backward==True) is from standardized to real.
    Since flip and transpose are not commutable, swap must come at
    the last for forward, and at the beginning for backward. """
    if 'y' in orgdrct and backward:
        array = np.swapaxes(array, array.ndim - 2, array.ndim - 1)
    if 'r' in orgdrct:
        array = np.flip(array, array.ndim - 1)
    if 't' in orgdrct:
        array = np.flip(array, array.ndim - 2)
    if 'y' in orgdrct and not backward:
        array = np.swapaxes(array, array.ndim - 2, array.ndim - 1)
    return array


def _elapsed(elapsed):
    h, m, s = int(elapsed // 3600), int(elapsed // 60 % 60), int(elapsed % 60)
    if elapsed < 120:
        return f'elapsed: {elapsed:.3f} s'
    elif elapsed < 3600:
        return f'elapsed: {m} m {s} s'
    else:
        return f'elapsed: {h} h {m} m {s} s'


class SatiError(Exception):
    """A base class of exceptions thrown by sati.Model."""
    pass


class ArgumentTypeError(SatiError):
    """An exception thrown when type of an argument is wrong."""
    def __init__(self, name, type_):
        self.__name = name
        self.__type = type_

    def __str__(self):
        return f"'{self.__name}' must be an instance of " \
               f'{self.__type.__module__}.{self.__type.__name__}'


class NoAttributeError(SatiError):
    """An exception thrown when an attribute is None."""
    def __init__(self, attr):
        self.__attr = attr

    def __str__(self):
        return f"'{self.__attr}' must be given."


class SatiWarning(Warning):
    """A base class of warning thrown by sati.Model."""
    pass


class NotConvergedWarning(SatiWarning):
    """A warning thrown when the optimization did not reach to the
    specfied tolerance. """
    def __init__(self, n_iter):
        self.__n_iter = n_iter

    def __str__(self):
        return f'Not converged after {self.__n_iter} iterations.'


class MstepWarning(SatiWarning):
    """A warning thrown when the optimizer of M-step fails."""
    def __init__(self, msg):
        self.__msg = msg

    def __str__(self):
        return f'The optimizer of M-step exited unsuccessfully.\n' \
               f'{self.__msg}'

