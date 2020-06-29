import numpy as np
import scipy.optimize


"""The default of optimizer options. ``reduction`` is used not in
``mimimize()`` but in the loop calling ``minimize()``. It's included
here for the documentation purpose."""
options_default = {
        'ftol': 1e-7, 'maxiter': 256, 'lr': 1e-3, 'betas': [0.9, 0.999],
        'epsilon': 1e-8, 'reduction': 0.5}


def minimize(fun, x0, *, method='adam', jac,
        options={'ftol': 1e-7, 'maxiter': 256, 'lr': 1e-3,
                 'betas': [0.9, 0.999], 'epsilon': 1e-8, 'reduction': 0.5}):
    """Minimization of scalar function of one or more variables.

    The parameters are similar to those of `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized: ``fun(x) -> float``
        where ``x`` is an array with shape (n,), where ``n`` is the number
        of independent variables.
    x0 : ``numpy.ndarray``
        Initial guess. Array of real elements of size (n,).
    method : {'adam'}, default 'adam'
        Optimizer.
    jac : callable
        A function that returns the gradient vector:
        ``jac(x) -> array_like, shape (n,)``
        where ``x`` is an array with shape (n,).
    options : ``dict``
        Optimizer options.

        * betas : ``tuple`` of ``float``
        * epsilon : ``float``

        These are parameters defined in the reference [1]_.

        * lr : ``float``, the learning rate.
        * ftol : ``float``, the fractional tolerance to stop the iteration.
        * maxiter : ``int``, the maximum number of iterations.
        * reduction : ``float``, this value is multiplied to ``lr`` when the objective function is smaller than that of the last step in the main EM loop.


    Returns
    -------
    scipy.optimize.OptimizeResult
        The optimization result. The results are stored in the attributes of
        ``x``, ``success``, ``fun``, ``jac``, and ``nit``. See
        `scipy.optimize.OptimizeResult <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult>`_ for more details.

    References
    ---=------
    .. [1] Diederik P. Kingma, Jimmy Ba, "Adam: A Method for Stochastic
       Optimization", arXiv:1412.6980
    """
    if type(options) == dict and options != options_default:
        options_ = options
        options = dict(options_default.items())
        options.update(options_)

    x = np.copy(x0)
    obj0 = fun(x)
    optimizer = _Adam(alpha=options['lr'], betas=options['betas'],
                      epsilon=options['epsilon'])

    for i in range(1, options['maxiter']):
        g = jac(x)
        x -= optimizer.step(g, i)
        obj1 = fun(x)
        if np.fabs((obj1 - obj0) / obj1) < options['ftol']:
            break
        obj0 = obj1

    result = {'x': x, 'success': True, 'fun': obj1, 'jac': g, 'nit': i}
    return scipy.optimize.OptimizeResult(result)


class _Adam():
    """Class for Adam.

    Parameters
    ----------
    alpha : float, default 1e-3
    betas : list of float, default [0.9, 0.999]
    epsilon : float, default 1e-8
        There parameters are as defined in the reference [1]_.

    References
    ----------
    .. [1] Diederik P. Kingma, Jimmy Ba, "Adam: A Method for Stochastic
       Optimization", arXiv:1412.6980
    """
    def __init__(self, alpha=1e-3, betas=[0.9, 0.999], epsilon=1e-8):
        self.alpha = alpha
        self.betas = betas
        self.epsilon = epsilon

    def step(self, g, i):
        """Calculate a step of gradient ascent.

        Parameters
        ----------
        g : numpy.ndarray
            Gradient of parameters.
        i : int
            Iteration index.

        Returns
        -------
        numpy.nudarray
            A step at i-th iteration.
        """
        if i == 1:
            self.__m, self.__v = np.zeros_like(g), np.zeros_like(g)

        self.__m = self.betas[0] * self.__m + (1 - self.betas[0]) * g
        self.__v = self.betas[1] * self.__v + (1 - self.betas[1]) * g * g

        hatm = self.__m / (1 - self.betas[0]**i)
        hatv = self.__v / (1 - self.betas[1]**i)
        return self.alpha * hatm / (np.sqrt(hatv) + self.epsilon)

