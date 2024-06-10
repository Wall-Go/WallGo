======================================
First example
======================================

Defining a model in WallGo requires a few different ingredients: a scalar potential, a list of the particles in the model together with their properties, and the matrix elements for interactions between these particles. The matrix elements are used to compute the collision integrals in the C++ part of WallGo. The collision integrals are then loaded into the Python part of WallGo.

Concretely, let's consider a simple model of a real scalar field coupled to a Dirac fermion via a Yukawa coupling,

.. math::
	\mathscr{L} = 
	-\frac{1}{2}\partial_\mu \phi \partial^\mu \phi - \sigma \phi - \frac{m^2}{2}\phi^2 - \frac{g}{3!} \phi^3 - \frac{\lambda}{4!} \phi^4
	-i\bar{\psi}\gamma^\mu \partial_\mu \psi - m \bar{\psi}\psi
	-y \phi \bar{\psi}\psi.

In this case the scalar field may undergo a phase transition, with the fermion field contributing to the friction for the bubble wall growth.

The definition of the Model starts by inheriting from the :py:data:`WallGo.GenericModel` class. This class holds the features of a model which enter directly in the Python side of WallGo. This includes the list of particles (:py:data:`WallGo.Particles` objects) and a reference to a definition of the effective potential.

.. literalinclude:: ../../Models/Yukawa/Yukawa.py
   :language: py
   :lines: 4-69

The effective potential itself is defined separately, by inheriting from the :py:data:`WallGo.EffectivePotential` class.

.. literalinclude:: ../../Models/Yukawa/Yukawa.py
   :language: py
   :lines: 72-109

**********
References
**********

.. footbibliography::
