![WallGo](docs/source/figures/wallgo.svg)

# WallGo

WallGo is open-source scientific software for computing the bubble wall speed for cosmological phase transitions.


## Status

Alpha v0.1.0 
![WallGo tests](https://github.com/Wall-Go/WallGo/actions/workflows/main.yml/badge.svg)

## About this project

**WallGo** is an open source code for the computation of the bubble wall velocity and bubble wall width in first-order cosmological phase transitions.
The main WallGo Python package determines the wall velocity and width by solving the scalar field(s) equation of motion, the Boltzmann equations and energy-momentum conservation for the fluid velocity and temperature.

WallGo is accompanied by two subsidiary software packages:
- **WallGoMatrix** computes the relevant matrix elements for the out-of-equilibrium particles, and is written in Mathematica. It builds on existing Mathematica packages DRalgo and GroupMath.
- **WallGoCollision** performs the higher-dimensional integrals to obtain the collision terms in the Boltzmann equations, and is written in C++. It also has Python bindings so that it can be called directly from Python, but still benefits from the speedup from compiled C++ code.

Users can implement their own models by specifying an effective potential and a list of out-of-equilibrium particles and their corresponding interactions.

## Installation

WallGo can be installed as a Python package (in developer mode) with pip, using:

    python -m pip install -e .

(or equivalent with python3) from the base directory of the repository.

To also install the requirements for tests, linting and the documentation
instead run

    python -m pip install -e ".[docs,lint,tests]""


### Requirements

WallGo is based on Python 3. Necessary requirements are installed automatically with
pip. They can be found in the file `pyproject.toml`.


### Tests

Tests can be run from the base directory with:

    pytest -v


## Examples

A number of example models are collected in the directory `Models/`, including the following:

- Standard Model with light Higgs
- Inert Doublet Model
- Real singlet scalar extension of the Standard Model
- Many scalar extension of the Standard Model
- Yukawa model

After installing the package, these examples can be run directly with Python, as
in:

    python3 Models/SingletStandardModel_Z2/singletStandardModelZ2.py


## License

Copyright (c) 2024 Andreas Ekstedt, Oliver Gould, Joonas Hirvonen,
Benoit Laurent, Lauri Niemi, Philipp Schicho, and Jorinde van de Vis.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
