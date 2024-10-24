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

In this case the scalar field may undergo a phase transition, with the fermion field contributing to the friction for the bubble wall growth. This model has been used as a toy model for the study of bubble nucleation in cosmological phase transitions. :footcite:p:`Gould:2021ccf`

The definition of the Model starts by inheriting from the :py:data:`WallGo.GenericModel` class. This class holds the features of a model which enter directly in the Python side of WallGo. This includes the list of particles (:py:data:`WallGo.Particle` objects) and a reference to a definition of the effective potential.

.. literalinclude:: ../../Models/Yukawa/yukawa.py
   :language: py
   :lines: 6-90

The scalar potential is used both for determining the free energy of homogeneous phases and for the shape and width of the bubble wall. In principle the potentials determining these two phenomena are different, as the former is coarse grained all the way to infinite length scales, while the latter can only consistenly be coarse grained on length scales shorter than the bubble wall width. :footcite:p:`Langer:1974cpa` Nervertheless, at high temperatures and to leading order in powers of the coupling, these two potentials agree.

At high temperatures, the leading order effective potential of our simple model is

.. math::
	V^\text{eff}(\phi, T) =
	- \frac{\pi^2}{20} T^4 + 
	\sigma_\text{eff}\phi
	+ \frac{1}{2}m^2_\text{eff}\phi^2
	+ \frac{1}{3!}g \phi^3
	+ \frac{1}{4!}\lambda \phi^4,

where we have defined the effective tadpole coefficient and effective mass as 

.. math::
	\sigma_\text{eff} =
	\sigma + \frac{1}{24}(g + 4y m_f)T^2,

	m^2_\text{eff} =
	m^2 + \frac{1}{24}(\lambda + 4y^2)T^2.

The implementation in WallGo is as follows: one defines a class, here called :py:data:`WallGo.EffectivePotentialYukawa` which inherits from the base class :py:data:`WallGo.EffectivePotential`. This definition must contain a member function called :py:data:`evaluate` which evaluates the potential as a function of the scalar fields and temperature.

.. literalinclude:: ../../Models/Yukawa/yukawa.py
   :language: py
   :lines: 93-152

The initialisation of an :py:data:`WallGo.EffectivePotential` object takes the model parameters and the number of background scalar fields as arguments and stores them for use in evaluating the potential. It is possible to override other member functions when defining :py:data:`WallGo.EffectivePotentialYukawa`, such as the initialisation function, or to add additional member functions and variables, though we haven't done so in this simple example.

Once these two classes have been defined, we can now run WallGo to compute the bubble wall speed.

.. literalinclude:: ../../Models/Yukawa/yukawa.py
   :language: py
   :lines: 155-215

**********
References
**********

.. footbibliography::
