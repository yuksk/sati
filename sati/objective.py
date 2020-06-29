import sys

import numpy as np


def calc(dist, t_subtracted, prior=None):
    """Calculate a value of objective function.

    Parameters
    ----------
    dist : a subclass of :class:`sati.distributions.Distribution`
        A distribution used in a model.
    t_subtracted : ``numpy.ndarray``
        Subtracted one-dimensionalized image.
    prior : :class:`sati.distributions.VonMises`
        A prior distribution.

    Returns
    -------
    float :
        A value of objective function.
    """
    likelihood_at_pixel = dist.fullpdf(t_subtracted).sum(axis=0)
    np.place(likelihood_at_pixel, likelihood_at_pixel==0, sys.float_info.min)
    if prior is None:
        return np.log(likelihood_at_pixel).sum()
    else:
        return np.log(likelihood_at_pixel).sum() \
                + prior.loglikelihood_at_pixel(dist.loc) * t_subtracted.size


class Objective():
    """Class for an objective function.

    Parameters
    ----------
    maxiter : ``int``
        The maximum number of iterations of the main EM loop.
    tol : ``float``
        The fractional tolerance to stop the iteration.
    verbose : ``bool``
        If ``True``, show values of objective function in the console
        during calculation.

    Attributes
    ----------
    value, diff : ``list`` of ``float``
        Objective function and its relative difference. Values are
        appended at each cycle of iterations.
    """
    def __init__(self, maxiter, tol, verbose):
        self.__tol, self.__verbose = tol, verbose
        self.value = []
        self.diff = []

        # used to show value and diff.
        self.__precision = (6, 4)

        # values used for the console update.
        self.__lines = 7
        self.__index_width = int(np.trunc(np.log10(maxiter))) + 1

    def __str__(self):
        return f'iterations: {len(self.value)-1}\n' \
               f'objective: {self.value[-1]:.{self.__precision[0]}e}'

    def append(self, value):
        """Append a value to the list of values and update list of differences.

        Parameters
        ----------
        value : float
            A value of an objective function.
        """
        if len(self.diff) == 0 or value == 0:
            self.diff.append(np.nan)
        else:
            self.diff.append(np.fabs((value - self.value[-1]) / value))
        self.value.append(value)

        if self.__verbose:
            self.__update_console()

    def isconverged(self):
        return self.diff[-1] < self.__tol

    def __update_console(self):
        def oneline(i):
            """Construct a line."""
            linestr = ''
            # If the objective decreases, change the character color.
            if i > 0 and self.value[i] < self.value[i - 1]:
                linestr += '\033[32m'  # green
            # Content of line
            linestr += ' '.join([
                f"{i:<{self.__index_width}}",
                f"{self.value[i]:+.{self.__precision[0]}e}",
                f"({self.diff[i]:.{self.__precision[1]}e})"])
            # Revert the character color.
            linestr += '\033[39m'
            return linestr

        if len(self.value) == 1:
            # Prepare lines in case the prompt is close to the bottom.
            print('\n' * (self.__lines- 1))

        print(f"\033[{self.__lines}F", end='')
        L = len(self.value)
        nlines = min(L, self.__lines)
        print('\n'.join([oneline(i) for i in range(L - nlines, L)]))
        print('\n' * (self.__lines - nlines), end='')

