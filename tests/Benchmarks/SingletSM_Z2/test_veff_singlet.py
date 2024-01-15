import pytest
import numpy as np
import WallGo
import WallGo.GenericModel
import WallGo.EffectivePotential
import WallGo.Fields

from Models.SingletStandardModel_Z2.SingletStandardModel_Z2 import SingletSM_Z2 # Benoit benchmark model

## List of BM points
from tests.Benchmarks.SingletSM_Z2.Benchmarks_singlet import singletBenchmarks

from tests.BenchmarkPoint import BenchmarkPoint, BenchmarkModel


@pytest.mark.parametrize("fields, temperature, expectedVeffValue", [
        (WallGo.Fields([110, 130]), 100, -1.19018205e+09),
        (WallGo.Fields([130, 130]), 100, -1.17839699e+09),
])
def test_singletModelVeffValue(singletBenchmarkModel: BenchmarkModel, fields: WallGo.Fields, temperature: float, expectedVeffValue: WallGo.Fields):

    relativeTolerance = 1e-6

    ## Could also take model objects as inputs instead of BM. But doesn't really matter as long as the model is fast to construct

    model = singletBenchmarkModel.model

    ## This tests real part only!!

    res = model.effectivePotential.evaluate(fields, temperature)
    assert res == pytest.approx( expectedVeffValue, rel=relativeTolerance)
    


## Same as test_singletModelVeffValue but gives the Veff list of field-space points
@pytest.mark.parametrize("fields, temperature, expectedVeffValue", [
        (WallGo.Fields([[110,130],[130,130]]), 100, [-1.19018205e+09, -1.17839699e+09]),
])
def test_singletModelVeffValue_manyFieldPoints(singletBenchmarkModel: BenchmarkModel, fields: WallGo.Fields, temperature: float, expectedVeffValue: WallGo.Fields):

    relativeTolerance = 1e-6

    model = singletBenchmarkModel.model

    ## This tests real part only!!
    res = model.effectivePotential.evaluate(fields, temperature)
    np.testing.assert_allclose(res, expectedVeffValue, rtol=relativeTolerance)



@pytest.mark.parametrize("initialGuess, temperature, expectedMinimum, expectedVeffValue", [
    (WallGo.Fields([ 0.0, 200.0 ]), 100, WallGo.Fields([0.0, 104.86914171]), -1.223482e+09),
    (WallGo.Fields([ 246.0, 0.0 ]), 100, WallGo.Fields([195.03215146, 0.0]), -1.231926e+09)
])
def test_singletModelVeffMinimization(singletBenchmarkModel: BenchmarkModel, initialGuess: WallGo.Fields, temperature: float, expectedMinimum: WallGo.Fields, expectedVeffValue: float):

    relativeTolerance = 1e-3
    model = singletBenchmarkModel.model

    resMinimum, resValue = model.effectivePotential.findLocalMinimum(initialGuess, temperature)

    ## The expected value is for full V(phi) + constants(T) so include that
    resValue = model.effectivePotential.evaluate(resMinimum, temperature)

    np.testing.assert_allclose(resMinimum, expectedMinimum, rtol=relativeTolerance)
    np.testing.assert_allclose(resValue, expectedVeffValue, rtol=relativeTolerance)



## Test fincTc(). This is relatively slow

@pytest.mark.slow
@pytest.mark.parametrize("minimum1, minimum2, expectedTc", [
    (WallGo.Fields([ 0.0, 200.0 ]), WallGo.Fields([ 246.0, 0.0 ]), 108.22)
])
def test_singletModelFindCriticalTemperature(singletBenchmarkModel: BenchmarkModel, minimum1: WallGo.Fields, minimum2: WallGo.Fields, expectedTc: float):

    relativeTolerance = 1e-3

    ## limit search to this T range
    TMin = 100
    TMax = 150

    model = singletBenchmarkModel.model
    Tc = model.effectivePotential.findCriticalTemperature(minimum1, minimum2, TMin, TMax)

    assert Tc == pytest.approx(expectedTc, rel=relativeTolerance)

@pytest.mark.parametrize("fields, temperature, expectedVeffValue", [
        (WallGo.Fields([110, 130]), 100, WallGo.Fields([512754.5552253, 1437167.06776619])),
        (WallGo.Fields([130, 130]), 100, WallGo.Fields([670916.4147377, 1712203.95803452]))
])
def test_singletModelDerivField(singletBenchmarkModel: BenchmarkModel, fields: WallGo.Fields, temperature: float, expectedVeffValue: WallGo.Fields):

    relativeTolerance = 1e-6

    ## Could also take model objects as inputs instead of BM. But doesn't really matter as long as the model is fast to construct

    model = singletBenchmarkModel.model

    ## This tests real part only!!

    res = model.effectivePotential.derivField(fields, temperature)
    assert res == pytest.approx( expectedVeffValue, rel=relativeTolerance)