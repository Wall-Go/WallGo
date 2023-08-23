import pytest
import numpy as np
from WallSpeed import Particle, FreeEnergy, Model

#model1 = TestModel.TestModel(0.2,0.1,0.4)

#print(model1.eBrok(0.3))

#print(findJouguetVelocity(model1,0.7))

def test_BM1():
    Tc = 132.58
    Tn = 129.61
    mod = Model(125,103.79,1.0,0.7152)
    params = mod.params
    res = mod.Vtot([[110,130]],100)
    assert res == pytest.approx(-1.1851e+09,rel=0.01)
    fxSM = FreeEnergy(mod.Vtot, Tc, Tn, params=params)
    res = fxSM.findPhases(100)
    np.testing.assert_allclose(res,[[195.35990073, 0.],[0.,96.45836659]],rtol=0.01)
    # res = mod.find_Tc()
    res = 132.58
    assert res == pytest.approx(Tc,rel=0.01)

def test_BM2():
    mod = Model(125,160.,1.0,1.2)
    res = mod.Vtot([[110,130]],100)
    assert res == pytest.approx(-1.15450678e+09,rel=0.01)

def test_BM3():
    mod = Model(125,160,1.0,1.6)
    res = mod.Vtot([[110,130]],100)
    assert res == pytest.approx(-1.23684861e+09,rel=0.01)