Sati
====
Statistical Analysis of Topographic Images

.. image:: ./docs/source/raw_subtracted.png

.. common

This package enables you to simultaneously

* subtract background (a polynomial surface, logarithmic decays, exponential decays)
* label terraces
* estimate terrace heights
* estimate the unit height of steps

even in the presence of steps.

Install
-------
You can install the package from the git repogitory using ``pip``
::

  $ pip install sati

Requirements
------------
Sati requires the following dependencies:

* python 3.8 or later
* matplotlib
* numexpr
* numpy
* scipy

Reference
---------
If you find this package is useful for your analysis, please refer `Y. Kohsaka, Rev. Sci. Instrum. 92, 033702 (2021) <https://doi.org/10.1063/5.0038852>`_.

Documents
---------
The usage examples and the API documentation are available at https://yuksk.github.io/sati/index.html


