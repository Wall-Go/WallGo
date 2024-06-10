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

**********
References
**********

.. footbibliography::
