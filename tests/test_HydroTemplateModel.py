import pytest
import numpy as np
from scipy.integrate import odeint
import WallGo

class TestModelTemplate(WallGo.Thermodynamics):
    __test__ = False

    def __init__(self, alN, psiN, cb2, cs2, Tn):
        self.alN = alN # Strength parameter alpha_n of the phase transition at the nucleation temperature
        self.psiN = psiN # Enthalpy in the low T phase divided by the enthalpy in the high T phase (both evaluated at the nucleation temperature)
        self.cb2 = cb2
        self.cs2 = cs2
        self.nu = 1+1/self.cb2
        self.mu = 1+1/self.cs2

        self.Tc = 1.
        self.Tnucl = Tn # Nucleation temperature

#        self.wn = wn # Enthalpy in the high T phase at the nucleation temperature
#        self.ap = 3*wn/(self.mu*Tn**self.mu)
#        self.am = 3*wn*psiN/(self.nu*Tn**self.nu)
        self.amoap = self.mu/self.nu*self.Tnucl**(self.mu-self.nu)*self.psiN #ratio of degrees of freedom between the high-energy and low-energy phase
        self.eps = 1-self.amoap  # epsilon like in eq. 22 of 2303.10171, but rescaled by a factor ap*Tc^mu/3

    #Pressure in high T phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pHighT(self, T):
        return T**self.mu - self.eps

    #T-derivative of the pressure in the high T phase
    def dpHighT(self, T):
        return self.mu*T**(self.mu-1)

    #Second T-derivative of the pressure in the high T phase
    def ddpHighT(self, T):
        return self.mu*(self.mu-1)*T**(self.mu-2)


    #Pressure in the low T phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pLowT(self, T):
        return self.amoap*T**self.nu

    #T-derivative of the pressure in the low T phase
    def dpLowT(self, T):
        return self.nu*self.amoap*T**(self.nu-1)

    #Second T-derivative of the pressure in the low T phase
    def ddpLowT(self, T):
        return self.nu*(self.nu-1)*self.amoap*T**(self.nu-2)


#These tests are all based on a comparison between the classes HydroTemplateModel and Hydro used with TestTemplateModel
N = 20
rng = np.random.default_rng(1)

def test_JouguetVelocity():
    res1,res2 = np.zeros(N),np.zeros(N)
    Tn = 1-0.5*rng.random(N)
    psiN = 1-0.5*rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    nu = 1+1/cb2
    mu = 1+1/cs2    
    alN = 1/3*(1 - psiN*Tn**(-nu)+ nu/mu*(Tn**(-mu)-1))
    for i in range(N):
        if alN[i]>0.001:
            model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],Tn[i])
            hydro = WallGo.Hydro(model,1e-6,50,1e-6,1e-6)
            hydroTemplate = WallGo.HydroTemplateModel(model)
            res1[i] = hydro.findJouguetVelocity()
            res2[i] = hydroTemplate.findJouguetVelocity()
        else:
            res1[i] = 0
            res2[i] = 0
        print(f"{alN[i]=} {psiN[i]=} {cb2[i]=} {cs2[i]=} {Tn[i]=}")
        print(res1[i],res2[i])
    np.testing.assert_allclose(res1,res2,rtol = 10**-4,atol = 0)

def test_findMatching():
    res1,res2 = np.zeros((N,4)),np.zeros((N,4))
    Tn = 1-0.5*rng.random(N)
    psiN = 1-0.5*rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    nu = 1+1/cb2
    mu = 1+1/cs2    
    alN = 1/3*(1 - psiN*Tn**(-nu)+ nu/mu*(Tn**(-mu)-1))
    vw = rng.random(N)
    for i in range(N):
        if alN[i]>0.001:
            model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],Tn[i])
            hydro = WallGo.Hydro(model,1e-6,50,1e-6,1e-6)
            hydroTemplate = WallGo.HydroTemplateModel(model,1e-6,1e-6)
            res1[i] = hydro.findMatching(vw[i])
            res2[i] = hydroTemplate.findMatching(vw[i])
            print(f"{vw[i]=} {alN[i]=} {psiN[i]=} {cb2[i]=} {cs2[i]=} {Tn[i]=}")
            print(res1[i],res2[i])
            if np.isnan(res1[i,0]):
                res1[i] = [0,0,0,0]
            if np.isnan(res2[i,0]):
                res2[i] = [0,0,0,0]
        else:
            res1[i] = [0,0,0,0]
            res2[i] = [0,0,0,0]
    np.testing.assert_allclose(res1,res2,rtol = 10**-3,atol = 0)

def test_findvwLTE():
    res1,res2 = np.zeros(N),np.zeros(N)
    Tn = 1-0.5*rng.random(N)
    psiN = 1-0.5*rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    nu = 1+1/cb2
    mu = 1+1/cs2    
    alN = 1/3*(1 - psiN*Tn**(-nu)+ nu/mu*(Tn**(-mu)-1))
    for i in range(N):
        if alN[i]>0.001:
            model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],Tn[i])
            hydro = WallGo.Hydro(model,1e-6,50,1e-6,1e-6)
            hydroTemplate = WallGo.HydroTemplateModel(model)
            print(f"{alN[i]=} {psiN[i]=} {cb2[i]=} {cs2[i]=} {Tn[i]=} {hydroTemplate.max_al()=}")
            res1[i] = hydro.findvwLTE()
            res2[i] = hydroTemplate.findvwLTE()
            print(res1[i])
            print(res2[i])
        else:
            res1[i]=0
            res2[i]=0
    np.testing.assert_allclose(res1,res2,rtol = 10**-4,atol = 0)

def test_findHydroBoundaries():
    res1,res2 = np.zeros((N,5)),np.zeros((N,5))
    Tn = 1-0.5*rng.random(N)
    psiN = 1-0.5*rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    nu = 1+1/cb2
    mu = 1+1/cs2    
    alN = 1/3*(1 - psiN*Tn**(-nu)+ nu/mu*(Tn**(-mu)-1))
    vw = rng.random(N)
    for i in range(N):
        model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],Tn[i])
        #print('Define hydro and hydroTemplate')
        hydro = WallGo.Hydro(model,1e-6,50,1e-6,1e-6)
        hydroTemplate = WallGo.HydroTemplateModel(model,1e-6,1e-6)
        print(f"{vw[i]=} {alN[i]=} {psiN[i]=} {cb2[i]=} {cs2[i]=} {Tn[i]=}")
        res1[i] = hydro.findHydroBoundaries(vw[i])
        print(f"{hydro.findMatching(vw[i])=}")
        print(res1[i])
        res2[i] = hydroTemplate.findHydroBoundaries(vw[i])
        print(f"{hydroTemplate.findMatching(vw[i])=}")
        print(res2[i])
        if np.isnan(res1[i,0]):
            res1[i] = [0,0,0,0,0]
        if np.isnan(res2[i,0]):
            res2[i] = [0,0,0,0,0]
    np.testing.assert_allclose(res1,res2,rtol = 10**-3,atol = 0)
