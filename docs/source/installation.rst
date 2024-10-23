===========================================
Installation
===========================================


Installing WallGo with pip
===========================================

WallGo can be installed as a Python package using pip:

.. code-block:: bash
    
    pip install WallGo


.. Installing WallGo with conan
.. ===========================================


WallGoCollision and WallGoMatrix
===========================================

The main WallGo package is accompanied by two subsidiary software packages, which are installed separately. For details, see the links below.

- `WallGoMatrix <https://github.com/Wall-Go/WallGoMatrix>`_ computes the relevant matrix elements for the out-of-equilibrium particles, and is written in Mathematica. It builds on existing Mathematica packages [DRalgo](https://github.com/DR-algo/DRalgo) and [GroupMath](https://renatofonseca.net/groupmath).
- `WallGoCollision <https://github.com/Wall-Go/WallGoCollision>`_ performs the higher-dimensional integrals to obtain the collision terms in the Boltzmann equations, and is written in C++. It also has Python bindings so that it can be called directly from Python, but still benefits from the speedup from compiled C++ code.
