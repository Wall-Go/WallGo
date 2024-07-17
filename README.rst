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

Can be installed as a package (in developer mode) with pip, or pip3, using::

    pip install -e .

from the base directory of the repository.

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
