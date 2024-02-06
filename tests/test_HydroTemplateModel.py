import pytest
import numpy as np
from scipy.integrate import odeint
import WallGo

class TestModelTemplate(WallGo.Thermodynamics):
    __test__ = False

    def __init__(self, alN, psiN, cb2, cs2, Tn, Tc, wn=1):
        self.alN = alN # Strength parameter alpha_n of the phase transition at the nucleation temperature
        self.psiN = psiN # Enthalpy in the low T phase divided by the enthalpy in the high T phase (both evaluated at the nucleation temperature)
        self.cb2 = cb2
        self.cs2 = cs2
        self.nu = 1+1/self.cb2
        self.mu = 1+1/self.cs2

        self.Tnucl = Tn # Nucleation temperature
        self.Tc = Tc
        self.wn = wn # Enthalpy in the high T phase at the nucleation temperature
        self.ap = 3*wn/(self.mu*Tn**self.mu)
        self.am = 3*wn*psiN/(self.nu*Tn**self.nu)
        self.eps = 0
        self.eps = (self.pHighT(Tn)-self.pLowT(Tn)-cb2*(self.eHighT(Tn)-self.eLowT(Tn)-3*wn*alN))/(1+cb2)

    #Pressure in high T phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pHighT(self, T):
        return self.ap*T**self.mu/3 - self.eps

    #T-derivative of the pressure in the high T phase
    def dpHighT(self, T):
        return self.mu*self.ap*T**(self.mu-1)/3

    #Second T-derivative of the pressure in the high T phase
    def ddpHighT(self, T):
        return self.mu*(self.mu-1)*self.ap*T**(self.mu-2)/3


    #Pressure in the low T phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pLowT(self, T):
        return self.am*T**self.nu/3

    #T-derivative of the pressure in the low T phase
    def dpLowT(self, T):
        return self.nu*self.am*T**(self.nu-1)/3

    #Second T-derivative of the pressure in the low T phase
    def ddpLowT(self, T):
        return self.nu*(self.nu-1)*self.am*T**(self.nu-2)/3


#These tests are all based on a comparison between the classes HydroTemplateModel and Hydro used with TestTemplateModel
N = 20
rng = np.random.default_rng(1)

def test_JouguetVelocity():
    res1,res2 = np.zeros(N),np.zeros(N)
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    for i in range(N):
        model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],1,1)
        hydro = WallGo.Hydro(model)
        hydroTemplate = WallGo.HydroTemplateModel(model)
        res1[i] = hydro.findJouguetVelocity()
        res2[i] = hydroTemplate.findJouguetVelocity()
    np.testing.assert_allclose(res1,res2,rtol = 10**-6,atol = 0)

def test_findMatching():
    res1,res2 = np.zeros((N,4)),np.zeros((N,4))
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    vw = rng.random(N)
    for i in range(N):
        model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],1,1)
        hydro = WallGo.Hydro(model,1e-6,500,1e-6,1e-6)
        hydroTemplate = WallGo.HydroTemplateModel(model,1e-6,1e-6)
        if vw[i] < hydro.minVelocity():
            res1[i] = [0,0,0,0]
        else:    
            res1[i] = hydro.findMatching(vw[i])
        if vw[i] < hydroTemplate.minVelocity():
            res2[i] = [0,0,0,0]
        else:
            res2[i] = hydroTemplate.findMatching(vw[i])
    np.testing.assert_allclose(res1,res2,rtol = 10**-2,atol = 0)

def test_findvwLTE():
    res1,res2 = np.zeros(N),np.zeros(N)
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N) # I put a 0.1 here - otherwise this test gets stuck. Need to fix that obviously
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    for i in range(N):
        model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],1,1)
        hydro = WallGo.Hydro(model,1e-6,50,1e-6,1e-6)
        hydroTemplate = WallGo.HydroTemplateModel(model)
        print(f"{psiN[i]=} {alN[i]=} {cs2[i]=} {cb2[i]=} {hydro.vMin=}")
        res1[i] = hydro.findvwLTE()
        print(res1[i])
        res2[i] = hydroTemplate.findvwLTE()
        print(res2[i])
    np.testing.assert_allclose(res1,res2,rtol = 10**-4,atol = 0)

def test_findHydroBoundaries():
    res1,res2 = np.zeros((N,5)),np.zeros((N,5))
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N)   
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    vw = rng.random(N)
    for i in range(N):
        print(f"{alN[i]=} {psiN[i]=} {cb2[i]=} {cs2[i]=}")
        model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],1,1)
        hydro = WallGo.Hydro(model,1e-6,10,1e-10,1e-10)
        hydroTemplate = WallGo.HydroTemplateModel(model,1e-6,1e-6)
        res1[i] = hydro.findHydroBoundaries(vw[i])
        print(f"{res1[i]=}")
        res2[i] = hydroTemplate.findHydroBoundaries(vw[i])
        print(f"{res2[i]=}")
        if np.isnan(res1[i,0]):
            res1[i] = [0,0,0,0,0]
        if np.isnan(res2[i,0]):
            res2[i] = [0,0,0,0,0]
    np.testing.assert_allclose(res1,res2,rtol = 10**-3,atol = 0)