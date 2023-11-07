import pytest
import numpy as np
import WallSpeed
from Models.SingletStandardModel_Z2.SingletStandardModel_Z2 import SingletSM_Z2

def test_BM1():
    ## Create control class
    Tc = 108.22
    Tn = 100
    vJ = 0.6444
    vw = 0.5229
    c1 = -3331587978
    c2 = 2976953742
    Tplus = 103.1
    Tminus = 100.1
    inputParameters = {
        #"RGScale" : 91.1876,
        "RGScale" : 125., # <- Benoit benchmark
        "v0" : 246.0,
        "MW" : 80.379,
        "MZ" : 91.1876,
        "Mt" : 173.0,
        "g3" : 1.2279920495357861,
        # scalar specific, choose Benoit benchmark values
        "mh1" : 125.0,
        "mh2" : 120.0,
        "a2" : 0.9,
        "b4" : 1.0
    }
    model = SingletSM_Z2(inputParameters)
    userInput = {
        "Tn" : Tn,
        "phaseLocation1" : [ 0.0, 200.0 ],
        "phaseLocation2" : [ 246.0, 0.0 ]
    }
    res = model.effectivePotential.evaluate([[110],[130]], 100)
    assert res.shape == (1,)
    assert res == pytest.approx([-1.19018205e09], rel=1e-3)

    phaseLocation1, VeffValue1 = model.effectivePotential.findLocalMinimum(userInput["phaseLocation1"], 100)
    phaseLocation2, VeffValue2 = model.effectivePotential.findLocalMinimum(userInput["phaseLocation2"], 100)
    np.testing.assert_allclose(
         [phaseLocation1,phaseLocation2], [[0.0, 104.86914171],[195.03215146, 0.0]], rtol=1e-4
    )

    res = model.effectivePotential.findCriticalTemperature(phaseLocation1, phaseLocation2, TMin = Tn, TMax = 500)
    assert res == pytest.approx(Tc, rel=1e-2)
    # free.interpolateMinima(0, 1.2 * Tc, 1)
    phaseLocation1, VeffValue1 = model.effectivePotential.findLocalMinimum(userInput["phaseLocation1"], Tn)
    phaseLocation2, VeffValue2 = model.effectivePotential.findLocalMinimum(userInput["phaseLocation2"], Tn)
    thermodynamics = WallSpeed.Thermodynamics(model.effectivePotential, Tc, Tn, phaseLocation2, phaseLocation1)
    # hydro = WallSpeed.Hydro(thermodynamics)
    # res = hydro.vJ
    # assert res == pytest.approx(vJ, rel=1e-2)
    # res = hydro.findHydroBoundaries(vw)
    # np.testing.assert_allclose(res[:4], (c1, c2, Tplus, Tminus), rtol=1e-2)


def test_BM2():
    ## Create control class
    inputParameters = {
        #"RGScale" : 91.1876,
        "RGScale" : 125., # <- Benoit benchmark
        "v0" : 246.0,
        "MW" : 80.379,
        "MZ" : 91.1876,
        "Mt" : 173.0,
        "g3" : 1.2279920495357861,
        # scalar specific, choose Benoit benchmark values
        "mh1" : 125.0,
        "mh2" : 160.0,
        "a2" : 1.2,
        "b4" : 1.0
    }
    model = SingletSM_Z2(inputParameters)

    res = model.effectivePotential.evaluate([[100,110],[130,130]], 100)
    assert res.shape == (2,)
    np.testing.assert_allclose(
        res, [-1.16182579e+09, -1.15446813e+09], rtol=1e-3
    )

def test_BM3():
    ## Create control class
    inputParameters = {
        #"RGScale" : 91.1876,
        "RGScale" : 125., # <- Benoit benchmark
        "v0" : 246.0,
        "MW" : 80.379,
        "MZ" : 91.1876,
        "Mt" : 173.0,
        "g3" : 1.2279920495357861,
        # scalar specific, choose Benoit benchmark values
        "mh1" : 125.0,
        "mh2" : 160.0,
        "a2" : 1.6,
        "b4" : 1.0
    }
    model = SingletSM_Z2(inputParameters)

    res = model.effectivePotential.evaluate([[110],[130]], 100)
    assert res.shape == (1,)
    assert res == pytest.approx([-1.23684861e09], rel=1e-2)
    res = model.effectivePotential.evaluate([110,130], 100)
    assert res.shape == ()
    assert res == pytest.approx(-1.23684861e09, rel=1e-2)

def test_BM4():
    ## Create control class
    inputParameters = {
        #"RGScale" : 91.1876,
        "RGScale" : 125., # <- Benoit benchmark
        "v0" : 246.0,
        "MW" : 80.379,
        "MZ" : 91.1876,
        "Mt" : 173.0,
        "g3" : 1.2279920495357861,
        # scalar specific, choose Benoit benchmark values
        "mh1" : 125.0,
        "mh2" : 80.0,
        "a2" : 0.5,
        "b4" : 1.0
    }
    model = SingletSM_Z2(inputParameters)

    res = model.effectivePotential.evaluate([[100, 110],[100, 130]], 100)
    np.testing.assert_allclose(
        res, [-1210419844, -1.180062e+09], rtol=1e-2
    )