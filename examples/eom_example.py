"""
An attempt to run EOM.py 
"""
import numpy as np # arrays, maths and stuff
from pprint import pprint # pretty printing of dicts
from WallSpeed.Grid import Grid
from WallSpeed.Polynomial import Polynomial
from WallSpeed.Boltzmann import BoltzmannBackground, BoltzmannSolver
from WallSpeed.Thermodynamics import Thermodynamics
from WallSpeed.Hydro import Hydro
from WallSpeed import Particle, FreeEnergy, Model
from WallSpeed.EOM import findWallVelocityLoop

"""
Grid
"""
M = 20
N = 20
grid = Grid(M, N, 1, 1)
poly = Polynomial(grid)

"""
Model definition
"""
class xSM(Model):
    def __init__(self):
        self.v0 = 246.
        self.muhsq = 7825.
        self.lamh = 0.129074
        self.mussq = 10774.6
        self.lams = 1
        self.lamm = 1.2
        self.g1 = 0.349791
        self.g2 = 0.652905
        self.yt = 0.992283

        # adding as model parameters for convenience
        self.muhT = 1/48.*(9*self.g2**2+3*self.g1**2+2*(6*self.yt**2 + 12*self.lamh+ self.lamm))
        self.musT = 1/12.*(2*self.lamm + 3*self.lams)
        self.b = 107.75 * np.pi**2 / 90

        self.params = { # putting params together into dict for WallGo
            "v0" : self.v0,
            "muhsq" : self.muhsq,
            "lamh" : self.lamh,
            "mussq" : self.mussq,
            "lams" : self.lams,
            "lamm" : self.lamm,
            "g2" : self.g2,
            "g1" : self.g1,
            "yt" : self.yt,
            "muhT" : self.muhT,
            "musT" : self.musT,
            "b" : self.b,
        }

        self.Tc = np.sqrt(
        (
            - self.muhT*self.lams*self.muhsq
            + self.musT*self.lamh*self.mussq
            - np.sqrt(self.lamh*self.lams)*(self.musT*self.muhsq-self.muhT*self.mussq)
        )/ (self.musT**2*self.lamh - self.muhT**2*self.lams)
        )

    def Vtot(self, field, T, include_radiation = True):
        # The user defines their effective free energy
        field = np.asanyarray(field)
        h, s = field[...,0], field[...,1]
        V0 = (
            -1/2.*self.muhsq*h**2 + 1/4.*self.lamh*h**4
            -1/2.*self.mussq*s**2 + 1/4.*self.lams*s**4
            + 1/4.*self.lamm*s**2*h**2
            + 1/4.*self.lamh*self.v0**4
        )
        VT = 1/2.*(self.muhT*h**2 + self.musT*s**2)*T**2
        V = V0 + VT
        if include_radiation:
            fsymT = - self.b*T**4
            V += fsymT
        return V

def dfdT(field, T, v0, muhsq, lamh, mussq, lams, lamm, g2, g1, yt, muhT, musT, b):
    # The user may or may not define this
    field = np.asanyarray(field)
    h, s = field[...,0], field[...,1]
    muhT = 1/48.*(9*g2**2+3*g1**2+2*(6*yt**2 + 12*lamh+ lamm))
    musT = 1/12.*(2*lamm + 3*lams)
    return (muhT*h**2 + musT*s**2)*T - 4*b*T**3


def dfdPhi(field, T, v0, muhsq, lamh, mussq, lams, lamm, g2, g1, yt, muhT, musT, b):
    # The user may or may not define this
    field = np.asanyarray(field)
    h, s = field[...,0], field[...,1]
    dV0dh = -muhsq*h + lamh*h**3 + 1/2.*lamm*s**2*h
    dVTdh = muhT*h*T**2
    dV0ds = -mussq*s + lams*s**3 + 1/2.*lamm*s*h**2
    dVTds = musT*s*T**2
    return_val = np.empty_like(field)
    return_val[..., 0] = dV0dh + dVTdh
    return_val[..., 1] = dV0ds + dVTds
    return return_val


# defining the free energy for WallGo
mod = xSM()
params=mod.params
pprint(params)

Tc = mod.Tc
Tn = 100 # only Tn is strictly necessary
print(f"{Tc=}, {Tn=}")


# overriding whole class is porably not so ideal
# class FreeEnergy(FreeEnergy):

#     def findPhases(self, T):
#         """Finds all phases at a given temperature T (hard coded version)

#         Parameters
#         ----------
#         T : float
#             The temperature for which to find the phases.

#         Returns
#         -------
#         phases : array_like
#             A list of phases

#         """
#         p = self.params
#         hsq = (-p["muhT"]*T**2+p["muhsq"])/p["lamh"]
#         ssq = (-p["musT"]*T**2+p["mussq"])/p["lams"]
#         return np.array([[np.sqrt(hsq),0],[0,np.sqrt(ssq)]])


fxSM = FreeEnergy(mod.Vtot, Tc, Tn, params=params, dfdPhi=dfdPhi)
fxSM.interpolateMinima(0,1.2*Tc,1)
print("\nFree energy:", fxSM)
print(f"{fxSM([0, 1], 100)=}")
print(f"{fxSM.derivT([0, 1], 100)=}")
print(f"{fxSM.derivField([0, 1], 100)=}")


"""
Particle
"""
top = Particle(
    "top",
    msqVacuum=lambda X: params["yt"]**2 * np.asanyarray(X)[0]**2,
    msqThermal=lambda T: params["yt"]**2 * T**2,
    statistics="Fermion",
    inEquilibrium=False,
    ultrarelativistic=False,
    collisionPrefactors=[params["g2"]**4, params["g2"]**4, params["g2"]**4],
)

"""
Compute the wall velocity in local thermal equilibrium
"""
thermo = Thermodynamics(fxSM)
hydro = Hydro(thermo)
vwLTE = hydro.findvwLTE()
print("The wall velocity in local thermal equilibrium is")
print(vwLTE)

"""
Compute the wall velocity with out-of-equilibrium effects
"""
print(findWallVelocityLoop(top,fxSM,None,1,grid))
