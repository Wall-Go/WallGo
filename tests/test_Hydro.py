import pytest
import numpy as np
import TestModel
from scipy.integrate import odeint
from WallSpeed.Hydro import *

#model1 = TestModel.TestModel(0.2,0.1,0.4)

#print(model1.eBrok(0.3))

#print(findJouguetVelocity(model1,0.7))

model1 = TestModel.TestModel(0.2,0.1,0.4)

def test_JouguetVelocity():
    res = np.zeros(5)
    for i in range(5):
        res[i] = findJouguetVelocity(model1,0.5+i*0.1)
#    assert res == pytest.approx([0.840948,0.776119,0.7240,0.6836,0.651791],rel=0.01*np.ones(5))
    np.testing.assert_allclose(res,[0.840948,0.776119,0.7240,0.6836,0.651791],rtol = 10**-3,atol = 0)

def test_matchDeton():
    res = matchDeton(model1,1.1*findJouguetVelocity(model1,0.5),0.5)
    np.testing.assert_allclose(res,(0.925043,0.848164,0.5,0.614381),rtol = 10**-3,atol = 0)
    res = matchDeton(model1,1.1*findJouguetVelocity(model1,0.6),0.6)
    np.testing.assert_allclose(res,(0.853731,0.777282,0.6,0.685916),rtol = 10**-3,atol = 0)
    res = matchDeton(model1,1.1*findJouguetVelocity(model1,0.7),0.7)
    np.testing.assert_allclose(res,(0.796415,0.737286,0.7,0.763685),rtol = 10**-3,atol = 0)
    res = matchDeton(model1,1.1*findJouguetVelocity(model1,0.8),0.8)
    np.testing.assert_allclose(res,(0.751924,0.710458,0.8,0.846123),rtol = 10**-3,atol = 0)
    res = matchDeton(model1,1.1*findJouguetVelocity(model1,0.9),0.9)
    np.testing.assert_allclose(res,(0.71697,0.690044,0.9,0.931932),rtol = 10**-3,atol = 0)

def test_matchDeflagOrHyb():
    res = matchDeflagOrHyb(model1,0.5,0.4)
    np.testing.assert_allclose(res,(0.4,0.5,0.825993,0.771703),rtol = 10**-3,atol = 0)
    res = matchDeflagOrHyb(model1,0.6, 0.3)
    np.testing.assert_allclose(res,(0.3,0.530156,0.698846,0.593875),rtol = 10**-3,atol = 0)
    res = matchDeflagOrHyb(model1,0.3, 0.2)
    np.testing.assert_allclose(res,(0.2,0.3,0.667112,0.614376),rtol = 10**-3,atol = 0)
    res = matchDeflagOrHyb(model1,0.7, 0.4)
    np.testing.assert_allclose(res,(0.4,0.547745,0.814862,0.734061),rtol = 10**-3,atol = 0)

def test_solveHydroShock():
    res = solveHydroShock(model1, 0.5, 0.4,0.825993)
    assert res == pytest.approx(0.77525, rel=0.01)
    res = solveHydroShock(model1, 0.6, 0.3,0.698846)
    assert res == pytest.approx(0.576319, rel=0.01)
    res = solveHydroShock(model1, 0.3, 0.2,0.6671123)
    assert res == pytest.approx(0.642264, rel=0.01)
    res = solveHydroShock(model1, 0.7, 0.4,0.73406141)
    assert res == pytest.approx(0.576516, rel=0.01)

def test_findMatching():
    res = findMatching(model1,0.3,0.5)
    np.testing.assert_allclose(res,(0.0308804,0.3,0.5419,0.361743),rtol = 10**-2,atol = 0)
    res = findMatching(model1,0.3,0.8)
    np.testing.assert_allclose(res,(0.265521,0.3,0.811487,0.793731),rtol = 10**-2,atol = 0)
    res = findMatching(model1,0.3,0.9)
    np.testing.assert_allclose(res,(0.28306,0.3,0.90647,0.898604),rtol = 10**-2,atol = 0)
    res = findMatching(model1,0.6,0.5)
    np.testing.assert_allclose(res,(0.208003, 0.508124,0.628915,0.503117),rtol = 10**-2,atol = 0)
    res = findMatching(model1,0.6,0.8)
    np.testing.assert_allclose(res,(0.447702, 0.554666,0.897803,0.831459),rtol = 10**-2,atol = 0)
    res = findMatching(model1,0.6,0.9)
    np.testing.assert_allclose(res,(0.485733, 0.559572,0.98525,0.933473),rtol = 10**-2,atol = 0)
    res = findMatching(model1,0.9,0.5)
    np.testing.assert_allclose(res,(0.9, 0.789344,0.5,0.62322),rtol = 10**-2,atol = 0)
    res = findMatching(model1,0.9,0.8)
    np.testing.assert_allclose(res,(0.9, 0.889579,0.8,0.829928),rtol = 10**-2,atol = 0)
    res = findMatching(model1,0.9,0.9)
    np.testing.assert_allclose(res,(0.9, 0.894957,0.9,0.918446),rtol = 10**-2,atol = 0)
