import pytest
import numpy as np
from WallSpeed import Particle, FreeEnergy, Model
from WallSpeed.Thermodynamics import Thermodynamics
from WallSpeed.Hydro import Hydro

#model1 = TestModel.TestModel(0.2,0.1,0.4)

#print(model1.eBrok(0.3))

#print(findJouguetVelocity(model1,0.7))

def test_BM1():
    Tc = 108.22
    Tn = 100
    vJ = 0.6444
    vw = 0.5229
    c1 = 3331587978
    c2 = 2976953742
    Tplus = 103.1
    Tminus = 100.1
    mod = Model(125,120,1.0,0.9)
    params = mod.params
    res = mod.Vtot([[110,130]],100)
    assert res == pytest.approx(-1.19018205e+09,rel=1e-2)
    free = FreeEnergy(mod.Vtot, Tc, Tnucl=Tn, params=params)
    res = free.findPhases(100)
    np.testing.assert_allclose(res,[[195.03215146, 0.],[0.,104.86914171]],rtol=1e-2)
    # res = mod.find_Tc()
    res = 108.22
    assert res == pytest.approx(Tc,rel=1e-2)
    free.interpolateMinima(0,1.2*Tc,1)
    thermo = Thermodynamics(free)
    hydro = Hydro(thermo)
    res = hydro.vJ
    assert res == pytest.approx(vJ,rel=1e-2)
    res = hydro.findHydroBoundaries(vw)
    np.testing.assert_allclose(res,(c1,c2,Tplus,Tminus),rtol=1e-2)


def test_BM2():
    mod = Model(125,160.,1.0,1.2)
    res = mod.Vtot([[110,130]],100)
    assert res == pytest.approx(-1.15450678e+09,rel=1e-2)

def test_BM3():
    mod = Model(125,160,1.0,1.6)
    res = mod.Vtot([[110,130]],100)
    assert res == pytest.approx(-1.23684861e+09,rel=1e-2)

def test_BM4():
    mod = Model(125,80,1.0,0.5)
    res = mod.Vtot([[100,100]],100)
    assert res == pytest.approx(-1210419844,rel=1e-2)