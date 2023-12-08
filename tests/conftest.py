import pytest
import numpy as np

import WallSpeed

## should clean these imports...

from Models.SingletStandardModel_Z2.SingletStandardModel_Z2 import SingletSM_Z2 # Benoit benchmark model

from .BenchmarkPoint import BenchmarkPoint

# Ugly directory structure:
from tests.Benchmarks.SingletSM_Z2.Benchmarks_singlet import singletBenchmarks

## Call this to load Jb, Jf interpolations. Not sure if we actually want this though
#WallSpeed.initialize()


def background(M):
    vw = 0#1 / np.sqrt(3)
    v = - np.ones(M - 1) / np.sqrt(3)
    v += 0.01 * np.sin(10 * 2 * np.pi * np.arange(M - 1))
    velocityMid = 0.5 * (v[0] + v[-1])
    field = np.ones((M - 1,))
    field[M // 2:]  = 0
    field += 0.1 * np.sin(7 * 2 * np.pi * np.arange(M - 1) + 6)
    T = 100 * np.ones(M - 1)
    T += 1 * np.sin(11 * 2 * np.pi * np.arange(M - 1) + 6)
    return WallSpeed.BoltzmannBackground(
        velocityMid=velocityMid,
        velocityProfile=v,
        fieldProfile=field,
        temperatureProfile=T,
        polynomialBasis="Cardinal",
    )


@pytest.fixture
def particle():
    return WallSpeed.Particle(
        name="top",
        msqVacuum=lambda phi: 0.5 * phi**2,
        msqThermal=lambda T: 0.1 * T**2,
        statistics="Fermion",
        inEquilibrium=False,
        ultrarelativistic=False,
        multiplicity=1,
    )


""" Below are some boilerplate classes and fixtures for testing stuff in SM + singlet, Z2 symmetric.
For defining common fixtures I use the 'params = [...]' keyword; tests that call these fixtures 
are automatically repeated with all parameters in the list. Note though that this makes it difficult
to assert different numbers for different parameters, unless the expected results are somehow passed
as params too; for example as otherData dict in BenchmarkPoint class.

In most tests we probably want to prefer the @pytest.mark.parametrize pattern and pass BenchmarkPoint objects
along with just the expected results that the test in question needs.
 
TODO should we use autouse=True for the benchmark fixtures?
"""



## NB: fixture argument name needs to be 'request'. This is due to magic

## These benchmark points will automatically be run for tests that ask for this fixture
@pytest.fixture(scope="module", params=singletBenchmarks)
def singletModelBenchmarkPoint(request) -> BenchmarkPoint:
    yield request.param


## Fixture model objects for benchmarks for tests that would rather start from a model than from the inputs.  
@pytest.fixture(scope="module", params=singletBenchmarks)
def singletModelZ2_fixture(request: BenchmarkPoint):
    """Gives a model object for Standard Model + singlet with Z2 symmetry.
    Also returns the expected results for that benchmark. 
    Note that our model contains an effective potential object, so no need to have separate fixtures for the Veff. 
    """

    yield SingletSM_Z2(request.param.inputParams), request.param.expectedResults
