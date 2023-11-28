import pytest
import numpy as np
import WallSpeed

from Models.SingletStandardModel_Z2.SingletStandardModel_Z2 import SingletSM_Z2 # Benoit benchmark model

from tests.Benchmarks.singletSM_Z2 import singletBenchmarks ## Gives a list of benchmark points for testing numbers

## These are all model dependent tests that focus on checking that effective potential gives right numbers for given benchmark points.
## Ideally we'd have model-independent tests that check Veff functionality,
## and model-dependent tests would be done elsewhere, possibly as a part of a larger test that does a full wall speed calculation.


@pytest.mark.parametrize("BM, fields, temperature, expectedVeffValue", [
        (singletBenchmarks[0], [110, 130], 100, -1.19018205e+09),
        (singletBenchmarks[1], [110, 130], 100, -1.15446813e+09),
        (singletBenchmarks[2], [110, 130], 100, -1.23602025e+09)
])
def test_singletModelVeffValue(BM, fields: list, temperature: float, expectedVeffValue: float):

    v, x = fields
    relativeTolerance = 1e-6

    ## Could also take model objects as inputs instead of BM. But doesn't really matter as long as the model is fast to construct

    model = SingletSM_Z2(BM.inputParams)

    ## This tests real part only!!

    res = model.effectivePotential.evaluate([v, x], temperature)
    assert res == pytest.approx( expectedVeffValue, rel=relativeTolerance)
    
    ## Let's test field input in different list/array forms
    res = model.effectivePotential.evaluate([[v], [x]], temperature)
    assert res == pytest.approx( [expectedVeffValue], rel=relativeTolerance) ## Looks like this passes even if the list structure is not exactly the same
    assert res.shape == (1,)


## Same as test_singletModelVeffValue but gives the Veff list of field-space points
@pytest.mark.parametrize("BM, fields, temperature, expectedVeffValue", [
        (singletBenchmarks[0], [[100,110],[130,130]], 100, [-1.194963e+09, -1.190182e+09]),
        (singletBenchmarks[1], [[100,110],[130,130]], 100, [-1.16182579e+09, -1.15446813e+09]),
        (singletBenchmarks[2], [[100,110],[130,130]], 100, [-1.246694e+09, -1.236020e+09])
])
def test_singletModelVeffValue_manyFieldPoints(BM, fields: list, temperature: float, expectedVeffValue: list[float]):

    relativeTolerance = 1e-6

    model = SingletSM_Z2(BM.inputParams)

    ## This tests real part only!!
    res = model.effectivePotential.evaluate(fields, temperature)
    np.testing.assert_allclose(res, expectedVeffValue, rtol=relativeTolerance)



### Test Veff minimization

@pytest.mark.parametrize("BM, initialGuess, temperature, expectedMinimum, expectedVeffValue", [
    (singletBenchmarks[0], [ 0.0, 200.0 ], 100, [0.0, 104.86914171], -1.223482e+09),
    (singletBenchmarks[0], [ 246.0, 0.0 ], 100, [195.03215146, 0.0], -1.231926e+09)
])
def test_singletModelVeffMinimization(BM, initialGuess: list[float], temperature: float, expectedMinimum: list[float], expectedVeffValue: float):

    relativeTolerance = 1e-3
    model = SingletSM_Z2(BM.inputParams)

    resMinimum, resValue = model.effectivePotential.findLocalMinimum(initialGuess, temperature)

    np.testing.assert_allclose(resMinimum, expectedMinimum, rtol=relativeTolerance)
    np.testing.assert_allclose(resValue, expectedVeffValue, rtol=relativeTolerance)



@pytest.mark.parametrize("BM, minimum1, minimum2, expectedTc", [
    (singletBenchmarks[0], [ 0.0, 200.0 ], [ 246.0, 0.0 ], 108.22)
])
def test_singletModelFindCriticalTemperature(BM, minimum1: list[float], minimum2: list[float], expectedTc: float):

    relativeTolerance = 1e-3

    ## limit search to this T range
    TMin = 100
    TMax = 150

    model = SingletSM_Z2(BM.inputParams)
    Tc = model.effectivePotential.findCriticalTemperature(minimum1, minimum2, TMin, TMax)

    assert Tc == pytest.approx(expectedTc, rel=relativeTolerance)


## TODO this is too big for a unit test, and not even a test on Veff! This is also slow - hence the marker
@pytest.mark.slow
@pytest.mark.parametrize("BM, minimum1, minimum2, Tc, Tn", [
    (singletBenchmarks[0], [ 0.0, 200.0 ], [ 246.0, 0.0 ], 108.22, 100)
])
def test_singletModelHydroBoundaries(BM, minimum1, minimum2, Tc, Tn):

    model = SingletSM_Z2(BM.inputParams)

    phaseLocation1, VeffValue1 = model.effectivePotential.findLocalMinimum(minimum1, Tn)
    phaseLocation2, VeffValue2 = model.effectivePotential.findLocalMinimum(minimum2, Tn)

    thermodynamics = WallSpeed.Thermodynamics(model.effectivePotential, Tc, Tn, phaseLocation2, phaseLocation1)
    hydro = WallSpeed.Hydro(thermodynamics)

    ## Jouguet velocity
    vJ_expected = 0.6444
    res = hydro.vJ
    assert res == pytest.approx(vJ_expected, rel=1e-2)

    vw_in = 0.5229
    res = hydro.findHydroBoundaries(vw_in)

    ## Goal values for hydro boundaries. These are the first 4 return values from findHydroBoundaries so check those only
    c1 = -3331587978
    c2 = 2976953742
    Tplus = 103.1
    Tminus = 100.1
    res = res[:4]

    np.testing.assert_allclose(res[:4], (c1, c2, Tplus, Tminus), rtol=1e-2)
