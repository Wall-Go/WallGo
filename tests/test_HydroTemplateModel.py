import pytest
import numpy as np
import TestModel
from scipy.integrate import odeint
from WallSpeed.Hydro import *
from WallSpeed.HydroTemplateModel import *


#These tests are all based on a comparison between the classes HydroTemplateModel and Hydro used with TestTemplateModel
N = 100
rng = np.random.default_rng(1)

def test_JouguetVelocity():
    res1,res2 = np.zeros(N),np.zeros(N)
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N)
    cb2 = 1/4+(1/3-1/4)*rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    for i in range(N):
        model = TestModel.TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i])
        hydro = Hydro(model,1)
        hydroTemplate = HydroTemplateModel(model,1)
        res1[i] = hydro.findJouguetVelocity()
        res2[i] = hydroTemplate.findJouguetVelocity()
    np.testing.assert_allclose(res1,res2,rtol = 10**-6,atol = 0)
    
def test_findMatching():
    res1,res2 = np.zeros((N,4)),np.zeros((N,4))
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N)
    cb2 = 1/4+(1/3-1/4)*rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    vw = rng.random(N)
    for i in range(N):
        model = TestModel.TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i])
        hydro = Hydro(model,1)
        hydroTemplate = HydroTemplateModel(model,1)
        res1[i] = hydro.findMatching(vw[i])
        res2[i] = hydroTemplate.findMatching(vw[i])
        if np.isnan(res1[i,0]):
            res1[i] = [0,0,0,0]
        if np.isnan(res2[i,0]):
            res2[i] = [0,0,0,0]
    np.testing.assert_allclose(res1,res2,rtol = 10**-3,atol = 5e-3)
    
def test_findvwLTE():
    res1,res2 = np.zeros(N),np.zeros(N)
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N)
    cb2 = 1/4+(1/3-1/4)*rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    for i in range(N):
        model = TestModel.TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i])
        hydro = Hydro(model,1)
        hydroTemplate = HydroTemplateModel(model,1)
        res1[i] = hydro.findvwLTE2()
        res2[i] = hydroTemplate.findvwLTE()
    np.testing.assert_allclose(res1,res2,rtol = 10**-3,atol = 0)

# def test_matchDeton():
#     hydro = Hydro(model1,0.5)
#     res = hydro.matchDeton(1.1*hydro.findJouguetVelocity())
#     np.testing.assert_allclose(res,(0.925043,0.848164,0.5,0.614381),rtol = 10**-3,atol = 0)
#     hydro = Hydro(model1,0.6)
#     res = hydro.matchDeton(1.1*hydro.findJouguetVelocity())
#     np.testing.assert_allclose(res,(0.853731,0.777282,0.6,0.685916),rtol = 10**-3,atol = 0)
#     hydro = Hydro(model1,0.7)
#     res = hydro.matchDeton(1.1*hydro.findJouguetVelocity())
#     np.testing.assert_allclose(res,(0.796415,0.737286,0.7,0.763685),rtol = 10**-3,atol = 0)
#     hydro = Hydro(model1,0.8)
#     res = hydro.matchDeton(1.1*hydro.findJouguetVelocity())
#     np.testing.assert_allclose(res,(0.751924,0.710458,0.8,0.846123),rtol = 10**-3,atol = 0)
#     hydro = Hydro(model1,0.9)
#     res = hydro.matchDeton(1.1*hydro.findJouguetVelocity())
#     np.testing.assert_allclose(res,(0.71697,0.690044,0.9,0.931932),rtol = 10**-3,atol = 0)

# def test_matchDeflagOrHyb():
#     hydro = Hydro(model1,1)
#     res = hydro.matchDeflagOrHyb(0.5,0.4)
#     np.testing.assert_allclose(res,(0.4,0.5,0.825993,0.771703),rtol = 10**-3,atol = 0)
#     res = hydro.matchDeflagOrHyb(0.6, 0.3)
#     np.testing.assert_allclose(res,(0.3,0.530156,0.698846,0.593875),rtol = 10**-3,atol = 0)
#     res = hydro.matchDeflagOrHyb(0.3, 0.2)
#     np.testing.assert_allclose(res,(0.2,0.3,0.667112,0.614376),rtol = 10**-3,atol = 0)
#     res = hydro.matchDeflagOrHyb(0.7, 0.4)
#     np.testing.assert_allclose(res,(0.4,0.547745,0.814862,0.734061),rtol = 10**-3,atol = 0)

# def test_solveHydroShock():
#     res = hydro.solveHydroShock(0.5, 0.4,0.825993)
#     assert res == pytest.approx(0.77525, rel=0.01)
#     res = hydro.solveHydroShock(0.6, 0.3,0.698846)
#     assert res == pytest.approx(0.576319, rel=0.01)
#     res = hydro.solveHydroShock(0.3, 0.2,0.6671123)
#     assert res == pytest.approx(0.642264, rel=0.01)
#     res = hydro.solveHydroShock(0.7, 0.4,0.73406141)
#     assert res == pytest.approx(0.576516, rel=0.01)

# def test_strongestShock():
#     res = hydro.strongestShock(0.2)
#     assert res == pytest.approx(0.509786, rel=0.01)
#     res = hydro.strongestShock(0.3)
#     assert res == pytest.approx(0.488307, rel=0.01)
#     res = hydro.strongestShock(0.4)
#     assert res == pytest.approx(0.462405, rel=0.01)
#     res = hydro.strongestShock(0.5)
#     assert res == pytest.approx(0.433052, rel=0.01)
#     res = hydro.strongestShock(0.6)
#     assert res == pytest.approx(0.401013, rel=0.01)
#     res = hydro.strongestShock(0.7)
#     assert res == pytest.approx(0.366219, rel=0.01)
#     res = hydro.strongestShock(0.8)
#     assert res == pytest.approx(0.327039, rel=0.01)
#     res = hydro.strongestShock(0.9)
#     assert res == pytest.approx(0.278722, rel=0.01)

# def test_findMatching():
#     hydro = Hydro(model1,0.5)
#     hydro.vJ = hydro.findJouguetVelocity()
#     res = hydro.findMatching(0.3)
#     np.testing.assert_allclose(res,(0.0308804,0.3,0.5419,0.361743),rtol = 10**-2,atol = 0)
#     res = hydro.findMatching(0.6)
#     np.testing.assert_allclose(res,(0.208003, 0.508124,0.628915,0.503117),rtol = 10**-2,atol = 0)
#     res = hydro.findMatching(0.9)
#     np.testing.assert_allclose(res,(0.9, 0.789344,0.5,0.62322),rtol = 10**-2,atol = 0)
#     hydro = Hydro(model1,0.8)
#     hydro.vJ = hydro.findJouguetVelocity()
#     res = hydro.findMatching(0.3)
#     np.testing.assert_allclose(res,(0.265521,0.3,0.811487,0.793731),rtol = 10**-2,atol = 0)
#     res = hydro.findMatching(0.6)
#     np.testing.assert_allclose(res,(0.447702, 0.554666,0.897803,0.831459),rtol = 10**-2,atol = 0)
#     res = hydro.findMatching(0.9)
#     np.testing.assert_allclose(res,(0.9, 0.889579,0.8,0.829928),rtol = 10**-2,atol = 0)
#     hydro = Hydro(model1,0.9)
#     hydro.vJ = hydro.findJouguetVelocity()
#     res = hydro.findMatching(0.3)
#     np.testing.assert_allclose(res,(0.28306,0.3,0.90647,0.898604),rtol = 10**-2,atol = 0)
#     res = hydro.findMatching(0.6)
#     np.testing.assert_allclose(res,(0.485733, 0.559572,0.98525,0.933473),rtol = 10**-2,atol = 0)
#     res = hydro.findMatching(0.9)
#     np.testing.assert_allclose(res,(0.9, 0.894957,0.9,0.918446),rtol = 10**-2,atol = 0)

# # Test local thermal equilibrium solution in bag model

# def test_LTE():
#     model2 = TestModel.TestModelBag(0.9)
#     hydro2 = Hydro(model2,1)
#     res = np.zeros(5)
#     for i in range(5):
#         hydro2 = Hydro(model2,0.5+i*0.1)
#         res[i] = hydro2.findvwLTE2()
#     np.testing.assert_allclose(res,[1.,1.,1.,0.714738,0.6018],rtol = 10**-3,atol = 0)

#     model2 = TestModel.TestModelBag(0.8)
#     hydro2 = Hydro(model2,1)
#     res2 = np.zeros(4)
#     for i in range(4):
#         hydro2 = Hydro(model2,0.6+i*0.1)
#         res2[i] = hydro2.findvwLTE2()
#     np.testing.assert_allclose(res2,[0.87429,0.7902,0.6856,0.5619],rtol = 10**-3,atol = 0)

