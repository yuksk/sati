Examples
==============

Guess initial responsibility
----------------------------
The first step is to make an initial guess of responsibility.
`sati.GuessInitRsp <api.html#sati.preprocessing.GuessInitRsp>`_ helps
this process.
::

  # importing packages
  import numpy as np
  import sati

  # Image to be analyzed needs to be given as numpy.ndarray.
  image = np.loadtxt('yourdata.csv')

  # Suppose the image has 5 terraces of different heights.
  # When difference of values between adjacent pixels is below
  # a threshold, the pixels are regared in the same terrace.
  # The unit of threshold is that of the height of the image (e.g., nm).
  initrsp = sati.GuessInitRsp(image, n=5, threshold=5e-3)

  # Show a guess of initial responsibility.
  initrsp.show()

If it works, each terrace is assigned to a class. Since the seeds (starting
points) of terrace search are randomly chosen, results are different every
time.  Once you get a nice initial responsibility, you can reproduce it from
the seeds used for the guess that is saved in the attribute ``seeds``.

.. note::

    If there are two (or more) terraces that are supposed to be the same
    height in an image, all of them should be assigned to the same class, or
    only one of them should be assigned to a class and the other(s) should
    be left unassigned.


Subtract a linear plane
-----------------------
Using the initial responsibility above, analysis with a linear plane and
Cauchy distribution can be done as follows.
::


  # Polynomial plane of degree 1
  poly = sati.planes.Poly()

  # Cauchy distribution
  dist = sati.distributions.Cauchy()

  # Create an model and optimize it.
  # initrsp.guess is the initial responsibility above.
  m = sati.Model(image, rsp=initrsp.guess, dist=dist, poly=poly)
  m.optimize()

Optimized parameters are given in the attributes. In the
above example,

* ``m.subtracted`` a plane-subtracted image
* ``m.rsp`` estimated responsibility, *i.e.*, soft clustering of terraces
* ``m.dist.loc`` estimated location parameters of the distribution,
  *i.e.*, heights of terraces
* ``m.dist.scale`` estimated scale parameters of the distribution.

See :ref:`api` for more details.


Subtract decays
---------------
Logarithmic / exponential decays are included by setting the ``decay``.
::

  m = sati.Model(image, rsp=initrsp.guess, poly=sati.planes.Poly(),
                 decay=sati.planes.Decay(tau=500, coef=-2e-11),
                 dist=sati.distributions.Cauchy())
  m.optimize()

The unit of ``tau`` and ``coef`` are pixel and the unit of the height
(*e.g.*, nm), respectively. Depending on scan direction of an image, you need
to set the ``orgdrct`` arguments to specify chronological order of pixels.
See `API documentation <api.html#sati.planes.Decay>`_ for more details.

.. note::

   When ``kind=='exp'``, the sign of ``coef`` is opposite to the *z*-direction
   to which the decay goes. For example, when the *z*-direction is positive
   (the tip height gets larger as time goes on), ``coef`` is negative.


Use a previous result as an initial value
-----------------------------------------
For example, when you subtract a polynomial surface of ``degree`` > 1, you
may want to use coefficients obtained for a model with a lower degree as
initial values to prevent the optimization from being trapped in a small
local maximum. Here is such an example.
::

  # Linear plane
  m = sati.Model(image, rsp=initrsp.guess, poly=sati.planes.Poly(),
                 dist=sati.distributions.Cauchy())
  m.optimize()

  # Use the result as an initial guess and optimize again.
  m.poly = sati.planes.Poly(degree=2, coef=m.poly.coef)
  m.optimize()


.. _estimating-unitheight:

Estimate a unit height of steps
-------------------------------
Estimating a unit height of steps is achieved by adopting von Mises-Fisher
distribution as a prior distribution for the location parameter of a model
distribution.
::

  # Optimize a model to get a set of initial values for the next step
  m = sati.Model(image, rsp=initrsp.guess, poly=sati.planes.Poly(),
                 dist=sati.distributions.Cauchy())
  m.optimize()

  # Set von Mises-Fisher distribution as a prior probability.
  m.prior = sati.distributions.VonMises(scale=2., kappa=[.1]*rsp.shape[0]))
  m.optimize()

.. note::

   ``len(kappa)`` must be the same as the number of terraces.

The ``scale`` parameter takes an initial guess of the lattice constant.
The estimated lattice constant is given in the ``scale`` attribute
(``m.prior.scale`` in the above example).


Save interim result
-------------------
You can save a result so that you can reuse it as a set of initial parameters
for successive calculations. This is useful, for example, when you change
parameters at the end of a series of long calculations.

For example, when you `estimate an unit height of steps <#estimating-unitheight>`_,
you would repeat calculations with different ``kappa``. If calculations
before applying a prior distribution take long time, you may want to save
the result to reuse it.
::

  m = sati.Model(image, rsp=initrsp.guess, poly=sati.planes.Poly(),
                 dist=sati.distributions.Cauchy())
  m.optimize()

  # Save the result for subsequent calculations
  m.pickle('tmp.pickle')

  # kappa = 0.01
  m.prior = sati.distributions.VonMises(scale=2., kappa=[.01]*rsp.shape[0]))
  m.optimize()

Suppose you decide to calculate with larger ``kappa`` after seeing the result
of ``kappa=0.01``. You can directly calculate the final step as follows.
::

  m = sati.Model.unpickle('tmp.pickle')
  m.prior = sati.distributions.VonMises(scale=2., kappa=[.1]*rsp.shape[0]))
  m.optimize()

  m = sati.Model.unpickle('tmp.pickle')
  m.prior = sati.distributions.VonMises(scale=2., kappa=[1.]*rsp.shape[0]))
  m.optimize()

If you reuse a result without saving it, you can also use ``copy.deepcopy``
of the `copy <https://docs.python.org/3/library/copy.html>`_ module.
