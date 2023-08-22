import pytest
from WallSpeed import Particle, FreeEnergy, Model

#model1 = TestModel.TestModel(0.2,0.1,0.4)

#print(model1.eBrok(0.3))

#print(findJouguetVelocity(model1,0.7))

def test_BM1():
    mod = Model(125,103.79,1.0,0.7152)
    res = mod.Vtot([[110,130]],100)
    assert res == pytest.approx(-1.1851e+09,rel=0.01)
    # res = mod.find_Tc()
    res = 132.58
    assert res == pytest.approx(132.58,rel=0.01)

def test_BM2():
    mod = Model(125,160.,1.0,1.2)
    res = mod.Vtot([[110,130]],100)
    assert res == pytest.approx(-1.15450678e+09,rel=0.01)

def test_BM3():
    mod = Model(125,160,1.0,1.6)
    res = mod.Vtot([[110,130]],100)
    assert res == pytest.approx(-1.23684861e+09,rel=0.01)