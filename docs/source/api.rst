.. _api:

API documentation
=================

sati
----

.. autoclass:: sati.Model
   :members: optimize, pickle, unpickle

.. autoclass:: sati.GuessInitRsp
   :members: show

sati.distributions
------------------

.. py:currentmodule:: sati.distributions

.. autoclass:: Distribution

.. autoclass:: Norm
   :show-inheritance:

.. autoclass:: Cauchy
   :show-inheritance:

.. autoclass:: T
   :show-inheritance:

.. autoclass:: VonMises
   :show-inheritance:

sati.planes
-----------

.. py:currentmodule:: sati.planes

.. autoclass:: Poly

.. autoclass:: Decay

sati.objective
--------------

.. autofunction:: sati.objective.calc
.. autoclass:: sati.objective.Objective

sati.optimize
-------------

.. autofunction:: sati.optimize.minimize
