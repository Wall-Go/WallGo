"""
A first example.
"""
import numpy as np # arrays, maths and stuff
from pprint import pprint # pretty printing of dicts
from WallSpeed.Grid import Grid
from WallSpeed.Polynomial import Polynomial
from WallSpeed.Boltzmann import BoltzmannSolver
from WallSpeed.Thermodynamics import Thermodynamics
#from WallSpeed.eomHydro import findWallVelocityLoop
from WallSpeed import Particle, FreeEnergy

"""
Model definition
"""
print("Model: xSM\n")
v0 = 246.22
muhsq = 7825.
lamh = 0.129074
mussq = 10774.6
lams = 1
lamm = 1.2
gp = 0.349791
g = 0.652905
yt = 0.992283

# adding as model parameters for convenience
th = 1/48.*(9*g**2+3*gp**2+2*(6*yt**2 + 12*lamh+ lamm))
ts = 1/12.*(2*lamm + 3*lams)
b = 107.75 * np.pi**2 / 90

def f(field, T, v0, muhsq, lamh, mussq, lams, lamm, g, gp, yt, th, ts, b):
    # The user defines their effective free energy
    field = np.asanyarray(field)
    h, s = field[...,0], field[...,1]
    V0 = (
        -1/2.*muhsq*h**2 + 1/4.*lamh*h**4
        -1/2.*mussq*s**2 + 1/4.*lams*s**4
        + 1/4.*lamm*s**2*h**2
        + 1/4.*lamh*v0**4
    )
    VT = 1/2.*(th*h**2 + ts*s**2)*T**2
    fsymT = - b*T**4
    return V0 + VT + fsymT


def dfdT(field, T, v0, muhsq, lamh, mussq, lams, lamm, g, gp, yt, th, ts, b):
    # The user may or may not define this
    field = np.asanyarray(field)
    h, s = field[...,0], field[...,1]
    th = 1/48.*(9*g**2+3*gp**2+2*(6*yt**2 + 12*lamh+ lamm))
    ts = 1/12.*(2*lamm + 3*lams)
    return (th*h**2 + ts*s**2)*T - 4*b*T**3


def dfdPhi(field, T, v0, muhsq, lamh, mussq, lams, lamm, g, gp, yt, th, ts, b):
    # The user may or may not define this
    field = np.asanyarray(field)
    h, s = field[...,0], field[...,1]
    dV0dh = -muhsq*h + lamh*h**3 + 1/2.*lamm*s**2*h
    dVTdh = th*h*T**2
    dV0ds = -mussq*s + lams*s**3 + 1/2.*lamm*s*h**2
    dVTds = ts*s*T**2
    return_val = np.empty_like(field)
    return_val[..., 0] = dV0dh + dVTdh
    return_val[..., 1] = dV0ds + dVTds
    return return_val


Tc = np.sqrt(
    (
        -th*lams*muhsq + ts*lamh*mussq
        - np.sqrt(lamh*lams)*(ts*muhsq-th*mussq)
    )
    / (ts**2*lamh - th**2*lams)
)
Tn = 100 # only Tn is strictly necessary
print(f"{Tc=}, {Tn=}")

# defining the free energy for WallGo
params = { # putting params together into dict for WallGo
    "v0" : v0,
    "muhsq" : muhsq,
    "lamh" : lamh,
    "mussq" : mussq,
    "lams" : lams,
    "lamm" : lamm,
    "g" : g,
    "gp" : gp,
    "yt" : yt,
    "th" : th,
    "ts" : ts,
    "b" : b,
}
pprint(params)
fxSM = FreeEnergy(f, Tc, Tn, params=params, dfdPhi=dfdPhi)
print("\nFree energy:", fxSM)
print(f"{fxSM([0, 1], 100)=}")
print(f"{fxSM.derivT([0, 1], 100)=}")
print(f"{fxSM.derivField([0, 1], 100)=}")

# looking at thermodynamics
thermo = Thermodynamics(fxSM)
print("\nThermodynamics:", thermo)
print(f"{thermo.pSym(100)=}")
print(f"{thermo.pBrok(100)=}")
print(f"{thermo.ddpBrok(100)=}")


# defining particles which are out of equilibrium for WallGo
top = Particle(
    "top",
    msqVacuum=lambda X: yt**2 * np.asanyarray(X)[..., 0]**2,
    msqThermal=lambda T: yt**2 * T**2,
    statistics="Fermion",
    inEquilibrium=False,
    ultrarelativistic=False,
    collisionPrefactors=[g**4, g**4, g**4],
)
particles = [top]
print("\ntop quark:", top)

# grid size
M = 20
N = 20

# now compute the bubble wall speed
# findWallVelocityLoop
