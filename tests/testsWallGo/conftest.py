"""
A collection of fixtures for tests not using real model
"""

import pytest
import numpy as np
import WallGo

## should clean these imports...
from .Benchmarks.SingletSM_Z2.Benchmarks_singlet import singletBenchmarks
from .BenchmarkPoint import BenchmarkPoint


# This is a parametrized fixture so argument name needs to be 'request'
@pytest.fixture
def boltzmannTestBackground(spatialGridSize: int) -> WallGo.BoltzmannBackground:
    """
    BoltzmannBackground with simple analytic v(z), T(z) and field(z) profiles
    """

    v = -np.ones(spatialGridSize + 1) / np.sqrt(3)
    v += 0.01 * np.sin(10 * 2 * np.pi * np.arange(spatialGridSize + 1))
    velocityMid = 0.5 * (v[0] + v[-1])

    # Test background field in WallGo.Fields format. Need to give it a list of field-space points,
    # but a 1D list is interpreted as one such point (with many independent background fields).
    # So give a 2D list.

    field = np.ones((spatialGridSize + 1,))
    field[spatialGridSize // 2 :] = 0
    field += 0.1 * np.sin(7 * 2 * np.pi * np.arange(spatialGridSize + 1) + 6)
    field = WallGo.Fields(field[:, np.newaxis])
    temperature = 100 * np.ones(spatialGridSize + 1)

    return WallGo.BoltzmannBackground(
        velocityMid=velocityMid,
        velocityProfile=v,
        fieldProfiles=field,
        temperatureProfile=temperature,
        polynomialBasis="Cardinal",
    )


@pytest.fixture
def particle() -> WallGo.Particle:
    """
    A "top" example Particle object for tests
    """
    return WallGo.Particle(
        name="top",
        index=0,
        msqVacuum=lambda phi: 0.5 * phi.getField(0) ** 2,
        msqDerivative=lambda fields: np.transpose(
            [fields.getField(0), 0 * fields.getField(1)]
        ),
        msqThermal=lambda T: 0.1 * T**2,
        statistics="Fermion",
        totalDOFs=12,
    )


##------- Old stuff, for newer fixtures see conftest.py in Benchmarks/SingletSM_Z2


""" Below are some fixtures for testing stuff in SM + singlet, Z2 symmetric.
For defining common fixtures I use the 'params = [...]' keyword; tests that call these fixtures 
are automatically repeated with all parameters in the list. Note though that this makes it difficult
to assert different numbers for different parameters, unless the expected results are somehow passed
as params too; for example as otherData dict in BenchmarkPoint class.

In most tests we probably want to prefer the @pytest.mark.parametrize pattern and pass BenchmarkPoint objects
along with just the expected results that the test in question needs.
 
TODO should we use autouse=True for the benchmark fixtures?
"""


## These benchmark points will automatically be run for tests that ask for this fixture
@pytest.fixture(scope="module", params=singletBenchmarks)
def singletModelBenchmarkPoint(request) -> BenchmarkPoint:
    yield request.param


## NB: fixture argument name needs to be 'request'. This is due to magic


## Fixture model objects for benchmarks for tests that would rather start from a model than from the inputs.
@pytest.fixture(scope="module", params=singletBenchmarks)
def singletModelZ2_fixture(request: BenchmarkPoint):
    """Gives a model object for Standard Model + singlet with Z2 symmetry.
    Also returns the expected results for that benchmark.
    Note that our model contains an effective potential object, so no need to have separate fixtures for the Veff.
    """

    yield SingletSM_Z2(request.param.inputParams), request.param.expectedResults
