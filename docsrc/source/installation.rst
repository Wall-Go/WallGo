===========================================
Installation
===========================================

WallGo comes in two parts: there is a C++ code for computing the collision integrals
and a Python package for computing the bubble wall speed, which requires the collision
integrals as input.

The instructions below assume that you have first cloned the
`repository <https://github.com/Wall-Go/WallGo>`_.

Collision integrals (C++)
===========================================

To install the C++ part of WallGo, the easiest way of handling the dependencies is
with the Conan package manager (version > 2.0). The build proceeds as::

    cd Collision
    conan install . --output-folder=build --build=missing
    cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
    cmake --build build
    cmake --install build

**Hint:** Conan can be installed with pip. 

Python package
===========================================

The Python part of WallGo be installed as a package (in developer mode) with pip,
or pip3. From the base directory of the repository, use::

    pip install -e .

