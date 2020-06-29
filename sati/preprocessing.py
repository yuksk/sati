import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

def standardize(arr, stats=None):
    """Standardize the input.

    Parameters
    ----------
    arr : numpy.ndarray
        An array to be standardized.
    stats : tuple of float, default None
        Average and standard deviation used to standardize the arr.
        If None, those of the arr are used.

    Returns
    -------
    arr_std : numpy.ndarray
        A standardized array
    stats : tuple of float
        Average and standard deviation used to standardize the input
        array.
    """
    if arr.ndim > 2:
        raise ValueError("'arr' must be 1D or 2D.")

    if stats is None:
        if arr.ndim == 1:
            stats = (arr.mean(), arr.std())
        else:
            stats = (arr.mean(axis=1), arr.std(axis=1))

    if arr.ndim == 1:
        arr_std = (arr - stats[0]) / stats[1]
    else:
        arr_std = (arr - stats[0].reshape(-1, 1)) / stats[1].reshape(-1, 1)

    return arr_std, stats


class PolynomialFeatures():
    """A class like sklearn.preprocessing.PolynomialFeatures."""
    def __init__(self, degree):
        """
        Parameters
        ----------
        degree : int
            degree of polynomial plane
        """
        self.__degree = degree

    def transform(self, r):
        """This is similar to fit_transform, but the returned array is
        transported."""
        n = (self.__degree + 2) * (self.__degree + 1) // 2 - 1
        arr = np.empty((n, r.shape[1]))

        k = 0
        for i in range(1, self.__degree+1):
            for j in range(i + 1):
                arr[k,:] = r[0,:]**(i - j) * r[1,:]**j
                k += 1
        return arr


class GuessInitRsp():
    """Class for making a guess of initial responsibility.

    Parameters
    ----------
    array : array-like
        image array
    n : ``int``
        Number of terraces
    threshold : ``float``
        When difference of values between adjacent pixels is below
        this threshold, those pixels are regared in the same terrace.
    seeds : array-like of ``tuple``
        Use this as initial locations of clustering, if given. Each tuple
        denotes an initial location. If number of tuples is less than ``n``,
        random initial locations are used for the rest.
    min_size : ``int``, default 10
        Minimum number of pixels of a terrace.

    Attributes
    ----------
    guess : ``numpy.ndarray``
        An array of suggested initial responsibility
    image : ``numpy.ndarray``
        An array of an image shown by show(). A value at each pixel denotes
        a terrace in which the pixel is assigned. -1 means unassigned.

    """
    def __init__(self, array, n, threshold, seeds=None, min_size=10):
        self.__array = np.asarray(array)
        self.seeds = []
        self.guess = self.__make_guess(n, threshold, seeds, min_size)
        self.image = self.__make_image()
        print(self)

    def __str__(self):
        msg = 'terrace  seeds  pixels\n' \
              '----------------------\n'
        for i in range(self.guess.shape[0]):
            msg += f'{i}: '\
                   f'({self.seeds[i][0]:.0f}, {self.seeds[i][1]:.0f}) '\
                   f'{self.guess[i,:,:].sum()}\n'
        msg += f'coverage {self.guess.sum()/self.__array.size*100:.0f}% ' \
               f'({self.guess.sum()}/{self.__array.size})'
        return msg

    def __make_guess(self, n, thresh, seeds, min_size):
        """
        1. select a point from self.__seeds or a random point that is not yet
           included in any clusters
        2. grow a cluster from it
        3. if the cluster has a point that is already included in an existing
           cluster, merge them
        4. repeat 1-3 until reaching the specified number of clusters
        """
        _seeds = list(reversed(seeds)) if seeds is not None else[]
        guess = np.zeros((n,) + self.__array.shape, dtype=bool)
        i = 0
        while i < n:
            seed = _seeds.pop() if _seeds else self.__random_pickup(guess)
            cluster = self.__grow_cluster(seed, thresh)
            if cluster.sum() < min_size:
                continue
            guess[i,:,:] = cluster
            self.seeds.append(seed)
            i += 1
        return guess

    def __grow_cluster(self, init_loc, thresh):
        """Grow a cluster from a point specified by ``init_loc``. Two
        points are regarded in the same cluster if difference of values
        of the two points is below ``self.__thresh``."""
        cluster = np.zeros_like(self.__array, dtype=bool)
        cluster[init_loc[0], init_loc[1]] = True
        pocket = [init_loc]
        adjacent = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        m, n = self.__array.shape
        while pocket:
            pt = pocket.pop(0)
            neighbors_in_cluster = [
                    (pt[0] - i, pt[1] - j) for (i, j) in adjacent
                    if 0 <= pt[0] - i < m and 0 <= pt[1] - j < n and
                    not cluster[pt[0] - i, pt[1] - j] and
                    np.absolute(self.__array[pt[0], pt[1]]
                                - self.__array[pt[0] - i, pt[1] - j])
                    < thresh]
            for nbr in neighbors_in_cluster:
                pocket.append(nbr)
                cluster[nbr[0], nbr[1]] = True
        return cluster

    def __random_pickup(self, guess):
        """Return a pixel that is not yet included in any clusters."""
        already_clustered = guess.sum(axis=0)
        while True:
            p1 = random.randint(0, guess.shape[1] - 1)
            p2 = random.randint(0, guess.shape[2] - 1)
            if not already_clustered[p1, p2]:
                return (p1, p2)

    def __make_image(self):
        image = np.full((self.guess.shape[1], self.guess.shape[2]), -1)
        for i in range(self.guess.shape[0]):
            image += self.guess[i,:,:] * (i + 1)
        return image

    def show(self, **kwargs):
        """Show an image of suggested initial responsibility.

        Parameters
        ----------
        **kwargs :
            keyword parameters to be passed to matplotlib.pyplot.imshow()
        """
        n = self.guess.shape[0]
        cmap_name = 'tab20' if n > 10 else 'tab10'
        clrs = ((1., 1., 1.),) + get_cmap(cmap_name).colors
        cmap = ListedColormap(np.delete(clrs, slice(n+1, None), axis=0))
        self.image = self.__make_image()
        plt.imshow(self.image, cmap = cmap, origin = 'lower', interpolation = None,
                   resample=False, vmin=-1.5, vmax=n-0.5, **kwargs)
        plt.colorbar(label='index', ticks=list(range(-1, n)))

        # The __test attribute is used not to block test
        block = not hasattr(self, f'_{self.__class__.__name__}__test')
        plt.show(block=block)


