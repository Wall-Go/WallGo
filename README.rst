===========================================
WallGo
===========================================

Computes the bubble wall speed for cosmological phase transitions.

|

Requirements
===========================================

Based on Python 3. Necessary requirements are installed automatically with
pip. They can be found in pyproject.toml.

|


Installation
===========================================

Can be installed as a package (in developer mode) with pip, using::

    python -m pip install -e .

(or equivalent with python3) from the base directory of the repository.

To also install the requirements for tests, linting and the documentation
instead run

    python -m pip install -e ".[docs,lint,tests]""

|

Tests
===========================================

Tests can be run with::

    pytest -v

|

Examples
===========================================

A number of example models are collected in the directory `Models/`, include the
Standard Model with light Higgs, singlet and doublet scalar extensions of the
Standard Model and a simple Yukawa model. After installing the package, these can
be run directly with Python, as
in::

    python3 Models/SingletStandardModel_Z2/SingletStandardModel_Z2.py


License
===========================================

Copyright (c) 2024 Andreas Ekstedt, Oliver Gould, Joonas Hirvonen,
Benoit Laurent, Lauri Niemi, Philipp Schicho, and Jorinde van de Vis

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

|