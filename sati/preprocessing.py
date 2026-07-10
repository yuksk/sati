import numpy as np


def standardize(
    arr: np.ndarray, stats: tuple[float, float] | None = None
) -> tuple[np.ndarray, tuple[float, float]]:
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


class PolynomialFeatures:
    """A class like sklearn.preprocessing.PolynomialFeatures."""

    def __init__(self, degree: int) -> None:
        """
        Parameters
        ----------
        degree : int
            degree of polynomial plane
        """
        self.__degree = degree

    def transform(self, r: np.ndarray) -> np.ndarray:
        """Similar to ``fit_transform``, but returns the transposed array."""
        n = (self.__degree + 2) * (self.__degree + 1) // 2 - 1
        arr = np.empty((n, r.shape[1]))

        k = 0
        for i in range(1, self.__degree + 1):
            for j in range(i + 1):
                arr[k, :] = r[0, :] ** (i - j) * r[1, :] ** j
                k += 1
        return arr
