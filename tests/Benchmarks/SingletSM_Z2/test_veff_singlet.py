import pytest
import numpy as np
import WallSpeed

from Models.SingletStandardModel_Z2.SingletStandardModel_Z2 import SingletSM_Z2 # Benoit benchmark model

## List of BM points
from tests.Benchmarks.SingletSM_Z2.Benchmarks_singlet import singletBenchmarks

from tests.BenchmarkPoint import BenchmarkPoint, BenchmarkModel


@pytest.mark.parametrize("fields, temperature, expectedVeffValue", [
        ([110, 130], 100, -1.19018205e+09),
])
def test_singletModelVeffValue(singletBenchmarkModel: BenchmarkModel, fields: list, temperature: float, expectedVeffValue: float):

    v, x = fields
    relativeTolerance = 1e-6

    ## Could also take model objects as inputs instead of BM. But doesn't really matter as long as the model is fast to construct

    model = singletBenchmarkModel.model

    ## This tests real part only!!

    res = model.effectivePotential.evaluate([v, x], temperature)
    assert res == pytest.approx( expectedVeffValue, rel=relativeTolerance)
    
    ## Let's test field input in different list/array forms
    res = model.effectivePotential.evaluate([[v], [x]], temperature)
    assert res == pytest.approx( [expectedVeffValue], rel=relativeTolerance) ## Looks like this passes even if the list structure is not exactly the same
    assert res.shape == (1,)


## Same as test_singletModelVeffValue but gives the Veff list of field-space points
@pytest.mark.parametrize("fields, temperature, expectedVeffValue", [
        ([[100,110],[130,130]], 100, [-1.194963e+09, -1.190182e+09]),
])
def test_singletModelVeffValue_manyFieldPoints(singletBenchmarkModel: BenchmarkModel, fields: list, temperature: float, expectedVeffValue: list[float]):

    relativeTolerance = 1e-6

    model = singletBenchmarkModel.model

    ## This tests real part only!!
    res = model.effectivePotential.evaluate(fields, temperature)
    np.testing.assert_allclose(res, expectedVeffValue, rtol=relativeTolerance)



@pytest.mark.parametrize("initialGuess, temperature, expectedMinimum, expectedVeffValue", [
    ([ 0.0, 200.0 ], 100, [0.0, 104.86914171], -1.223482e+09),
    ([ 246.0, 0.0 ], 100, [195.03215146, 0.0], -1.231926e+09)
])
def test_singletModelVeffMinimization(singletBenchmarkModel: BenchmarkModel, initialGuess: list[float], temperature: float, expectedMinimum: list[float], expectedVeffValue: float):

    relativeTolerance = 1e-3
    model = singletBenchmarkModel.model

    resMinimum, resValue = model.effectivePotential.findLocalMinimum(initialGuess, temperature)

    np.testing.assert_allclose(resMinimum, expectedMinimum, rtol=relativeTolerance)
    np.testing.assert_allclose(resValue, expectedVeffValue, rtol=relativeTolerance)



## Test fincTc(). This is relatively slow

@pytest.mark.slow
@pytest.mark.parametrize("minimum1, minimum2, expectedTc", [
    ([ 0.0, 200.0 ], [ 246.0, 0.0 ], 108.22)
])
def test_singletModelFindCriticalTemperature(singletBenchmarkModel: BenchmarkModel, minimum1: list[float], minimum2: list[float], expectedTc: float):

    relativeTolerance = 1e-3

    ## limit search to this T range
    TMin = 100
    TMax = 150

    model = singletBenchmarkModel.model
    Tc = model.effectivePotential.findCriticalTemperature(minimum1, minimum2, TMin, TMax)

    assert Tc == pytest.approx(expectedTc, rel=relativeTolerance)
