import pytest
from dataclasses import dataclass
import numpy as np
from scipy.integrate import odeint
import WallGo

# defines the toy xSM model, used in 2004.06995 and 2010.09744
# critical temperature is at T=1

@dataclass
class FreeEnergyHack:
    minPossibleTemperature: [float, bool]
    maxPossibleTemperature: [float, bool] 


class TestModel2Step(WallGo.Thermodynamics):
    __test__ = False
    abrok = 0.2
    asym = 0.1
    musqq = 0.4

    def __init__(self, abrok, asym, musqq, Tn):
        self.aLowT = abrok
        self.aHighT = asym
        self.musq = musqq
        self.Tnucl = Tn
        self.Tc = 1

        self.freeEnergyHigh=FreeEnergyHack(minPossibleTemperature=[0.01, False], maxPossibleTemperature=[5., False])
        self.freeEnergyLow =FreeEnergyHack(minPossibleTemperature=[0.01, False], maxPossibleTemperature=[5., False])

        self.TMinLowT = 0.01
        self.TMaxLowT = 5.
        self.TMinHighT = 0.01
        self.TMaxHighT = 5.

    #Pressure in high T phase
    def pHighT(self, T):
        return T**4. + (self.aLowT - self.aHighT + self.aHighT*T**2 - self.musq)**2-self.musq**2

    #T-derivative of the pressure in the high T phase
    def dpHighT(self, T):
        return 4*T**3. + 4. * self.aHighT * T *(self.aLowT - self.aHighT + self.aHighT * T**2-self.musq)

    #Second T-derivative of the pressure in the high T phase
    def ddpHighT(self, T):
        return 12.*T**2. +8 * self.aHighT**2. * T**2. + 4.*self.aHighT*(self.aLowT-self.aHighT +self.aHighT*T**2-self.musq)

    #Pressure in the low T phase
    def pLowT(self, T):
        return T**4. + (self.aLowT*T**2. - self.musq)**2. - self.musq**2.

    #T-derivative of the pressure in the low T phase
    def dpLowT(self, T):
        return 4.*T**3. + 4. * self.aLowT * T *(self.aLowT*T**2 - self.musq)

    #Second T-derivative of the pressure in the low T phase
    def ddpLowT(self, T):
        return 12.*T**2. +8. * self.aLowT**2. * T**2. + 4.*self.aLowT*(self.aLowT*T**2.-self.musq)


#Defines the bag equation of state
#Note that a factor 1/3 a_+ Tc**4 has been scaled out
#The critical temperature is at Tc=1, which relates psi and the (rescaled) bag constant epsilon: eps = 1-psi
#The phase transition strength at temperature t is given by: \alpha(t) = 1/3.*(1-psi)(1/t)**4

class TestModelBag(WallGo.Thermodynamics):
    __test__ = False

    def __init__(self, psi, Tn):
        self.psi = psi #number of degrees of freedom of the low T phase divided by the number of degrees of freedom in the high T phase
        self.eps = 1. - psi #this is the bag constant times 3 and divided by (the number of degrees of freedom of the high T phase times Tc^4)
        self.Tnucl = Tn
        self.Tc = 1

        self.freeEnergyHigh=FreeEnergyHack(minPossibleTemperature=[0.1, False], maxPossibleTemperature=[500.,False])
        self.freeEnergyLow =FreeEnergyHack(minPossibleTemperature=[0.1, False], maxPossibleTemperature=[500., False])

        self.TMinLowT = 0.01
        self.TMaxLowT = 5.
        self.TMinHighT = 0.01
        self.TMaxHighT = 5.

    #Pressure in high T phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pHighT(self, T):
        return T**4. - self.eps

    #T-derivative of the pressure in the high T phase
    def dpHighT(self, T):
        return 4.*T**3.

    #Second T-derivative of the pressure in the high T phase
    def ddpHighT(self, T):
        return 12.*T**2.


    #Pressure in the low T phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pLowT(self, T):
        return self.psi*T**4.

    #T-derivative of the pressure in the low T phase
    def dpLowT(self, T):
        return 4.*self.psi*T**3.

    #Second T-derivative of the pressure in the low T phase
    def ddpLowT(self, T):
        return 12.*self.psi*T**2.


#These tests are all based on a comparison to external code which computes the same quantities


model1 = TestModel2Step(0.2,0.1,0.4,1)
# Maximum and minimum temperature used in Hydrodynamics, in units of Tnucl
tmax = 10
tmin = 0.01
rtol = 1e-6
atol = 1e-6
hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)

def test_JouguetVelocity():
    res = np.zeros(5)
    for i in range(5):
        model1.Tnucl = 0.5+i*0.1
        hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)
        res[i] = hydrodynamics.findJouguetVelocity()
    np.testing.assert_allclose(res,[0.840948,0.776119,0.7240,0.6836,0.651791],rtol = 10**-3,atol = 0)

def test_matchDeton():
    model1 = TestModel2Step(0.2,0.1,0.4,0.5)
    hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)
    res = hydrodynamics.matchDeton(1.1*hydrodynamics.vJ)
    np.testing.assert_allclose(res,(0.925043,0.848164,0.5,0.614381),rtol = 10**-3,atol = 0)
    model1 = TestModel2Step(0.2,0.1,0.4,0.6)
    hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)
    res = hydrodynamics.matchDeton(1.1*hydrodynamics.vJ)
    np.testing.assert_allclose(res,(0.853731,0.777282,0.6,0.685916),rtol = 10**-3,atol = 0)
    model1 = TestModel2Step(0.2,0.1,0.4,0.7)
    hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)
    res = hydrodynamics.matchDeton(1.1*hydrodynamics.vJ)
    np.testing.assert_allclose(res,(0.796415,0.737286,0.7,0.763685),rtol = 10**-3,atol = 0)
    model1 = TestModel2Step(0.2,0.1,0.4,0.8)
    hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)
    res = hydrodynamics.matchDeton(1.1*hydrodynamics.vJ)
    np.testing.assert_allclose(res,(0.751924,0.710458,0.8,0.846123),rtol = 10**-3,atol = 0)
    model1 = TestModel2Step(0.2,0.1,0.4,0.9)
    hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)
    res = hydrodynamics.matchDeton(1.1*hydrodynamics.vJ)
    np.testing.assert_allclose(res,(0.71697,0.690044,0.9,0.931932),rtol = 10**-3,atol = 0)

def test_matchDeflagOrHyb():
    #This does not depend on the nucleation temperature, so no need to reinitialize model1
    hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)
    res = hydrodynamics.matchDeflagOrHyb(0.5,0.4)
    np.testing.assert_allclose(res,(0.4,0.5,0.825993,0.771703),rtol = 10**-3,atol = 0)
    res = hydrodynamics.matchDeflagOrHyb(0.6, 0.3)
    np.testing.assert_allclose(res,(0.3,0.530156,0.698846,0.593875),rtol = 10**-3,atol = 0)
    res = hydrodynamics.matchDeflagOrHyb(0.3, 0.2)
    np.testing.assert_allclose(res,(0.2,0.3,0.667112,0.614376),rtol = 10**-3,atol = 0)
    res = hydrodynamics.matchDeflagOrHyb(0.7, 0.4)
    np.testing.assert_allclose(res,(0.4,0.547745,0.814862,0.734061),rtol = 10**-3,atol = 0)

def test_solveHydroShock():
    res = hydrodynamics.solveHydroShock(0.5, 0.4,0.825993)
    assert res == pytest.approx(0.77525, rel=0.01)
    res = hydrodynamics.solveHydroShock(0.6, 0.3,0.698846)
    assert res == pytest.approx(0.576319, rel=0.01)
    res = hydrodynamics.solveHydroShock(0.3, 0.2,0.6671123)
    assert res == pytest.approx(0.642264, rel=0.01)
    res = hydrodynamics.solveHydroShock(0.7, 0.4,0.73406141)
    assert res == pytest.approx(0.576516, rel=0.01)

#Commented out because corresponding function is commented out, since it is never used.
# def test_strongestShock():
#     res = hydrodynamics.strongestShock(0.2)
#     assert res == pytest.approx(0.509786, rel=0.01)
#     res = hydrodynamics.strongestShock(0.3)
#     assert res == pytest.approx(0.488307, rel=0.01)
#     res = hydrodynamics.strongestShock(0.4)
#     assert res == pytest.approx(0.462405, rel=0.01)
#     res = hydrodynamics.strongestShock(0.5)
#     assert res == pytest.approx(0.433052, rel=0.01)
#     res = hydrodynamics.strongestShock(0.6)
#     assert res == pytest.approx(0.401013, rel=0.01)
#     res = hydrodynamics.strongestShock(0.7)
#     assert res == pytest.approx(0.366219, rel=0.01)
#     res = hydrodynamics.strongestShock(0.8)
#     assert res == pytest.approx(0.327039, rel=0.01)
#     res = hydrodynamics.strongestShock(0.9)
#     assert res == pytest.approx(0.278722, rel=0.01)

def test_findMatching():
    model1 = TestModel2Step(0.2,0.1,0.4,0.5)
    hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)
    hydrodynamics.vJ = hydrodynamics.findJouguetVelocity()
    res = hydrodynamics.findMatching(0.3)
    np.testing.assert_allclose(res,(0.0308804,0.3,0.5419,0.361743),rtol = 10**-2,atol = 0)
    res = hydrodynamics.findMatching(0.6)
    np.testing.assert_allclose(res,(0.208003, 0.508124,0.628915,0.503117),rtol = 10**-2,atol = 0)
    res = hydrodynamics.findMatching(0.9)
    np.testing.assert_allclose(res,(0.9, 0.789344,0.5,0.62322),rtol = 10**-2,atol = 0)
    model1 = TestModel2Step(0.2,0.1,0.4,0.8)
    hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)
    hydrodynamics.vJ = hydrodynamics.findJouguetVelocity()
    res = hydrodynamics.findMatching(0.3)
    np.testing.assert_allclose(res,(0.265521,0.3,0.811487,0.793731),rtol = 10**-2,atol = 0)
    res = hydrodynamics.findMatching(0.6)
    np.testing.assert_allclose(res,(0.447702, 0.554666,0.897803,0.831459),rtol = 10**-2,atol = 0)
    res = hydrodynamics.findMatching(0.9)
    np.testing.assert_allclose(res,(0.9, 0.889579,0.8,0.829928),rtol = 10**-2,atol = 0)
    model1 = TestModel2Step(0.2,0.1,0.4,0.9)
    hydrodynamics = WallGo.Hydrodynamics(model1, tmax, tmin, rtol, atol)
    hydrodynamics.vJ = hydrodynamics.findJouguetVelocity()
    res = hydrodynamics.findMatching(0.3)
    np.testing.assert_allclose(res,(0.28306,0.3,0.90647,0.898604),rtol = 10**-2,atol = 0)
    res = hydrodynamics.findMatching(0.6)
    np.testing.assert_allclose(res,(0.485733, 0.559572,0.98525,0.933473),rtol = 10**-2,atol = 0)
    res = hydrodynamics.findMatching(0.9)
    np.testing.assert_allclose(res,(0.9, 0.894957,0.9,0.918446),rtol = 10**-2,atol = 0)

### Test local thermal equilibrium solution in bag model


def test_LTE():
    res = np.zeros(5)
    for i in range(5):
        model2 = TestModelBag(0.9,0.5+i*0.1)
        hydrodynamics2 = WallGo.Hydrodynamics(model2, tmax, tmin, rtol, atol)
        res[i] = hydrodynamics2.findvwLTE()
    np.testing.assert_allclose(res,[1.,1.,1.,0.714738,0.6018],rtol = 10**-3,atol = 0)

    res2 = np.zeros(4)
    for i in range(4):
        model2 =     model2 = TestModelBag(0.8,0.6+i*0.1)
        hydrodynamics2 = WallGo.Hydrodynamics(model2, tmax, tmin, rtol, atol)
        res2[i] = hydrodynamics2.findvwLTE()
    np.testing.assert_allclose(res2,[0.87429,0.7902,0.6856,0.5619],rtol = 10**-3,atol = 0)