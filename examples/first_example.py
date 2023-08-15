"""
A first example.
"""
import numpy as np # arrays, maths and stuff
from pprint import pprint # pretty printing of dicts
from WallSpeed.Grid import Grid
from WallSpeed.Polynomial import Polynomial
from WallSpeed.Boltzmann import BoltzmannSolver
#from WallSpeed.eomHydro import findWallVelocityLoop
from WallSpeed import Particle, FreeEnergy

"""
Model definition
"""
print("Model: xSM")
p = {
    "v0" : 246.22,
    "muhsq" : 7825.,
    "lamh" : 0.129074,
    "mussq" : 10774.6,
    "lams" : 1.,
    "lamm" : 1.2,
    "g" : 0.349791,
    "gp" : 0.652905,
    "yt" : 0.992283,
}
pprint(p)

th = 1/48.*(9*p["g"]**2+3*p["gp"]**2+2*(6*p["yt"]**2 + 12*p["lamh"]+ p["lamm"]))
ts = 1/12.*(2*p["lamm"] + 3*p["lams"])
p["th"] = th # not necessary for model definition, but simplifies things
p["ts"] = ts


def f(X, T, p):
    # The user defines their effective free energy
    X = np.asanyarray(X)
    h, s = X[...,0], X[...,1]
    V0 = (
        -1/2.*p["muhsq"]*h**2 + 1/4.*p["lamh"]*h**4
        -1/2.*p["mussq"]*s**2 + 1/4.*p["lams"]*s**4
        + 1/4.*p["lamm"]*s**2*h**2
        + 1/4.*p["lamh"]*p["v0"]**4
    )
    VT = 1/2.*(p["th"]*h**2 + p["ts"]*s**2)*T**2
    fsymT = - 107.75*np.pi**2/90*T**4
    return V0 + VT + fsymT


Tc = np.sqrt(
    (
        -th*p["lams"]*p["muhsq"] + ts*p["lamh"]*p["mussq"]
        - np.sqrt(p["lamh"]*p["lams"])*(ts*p["muhsq"]-p["th"]*p["mussq"])
    )
    / (p["ts"]**2*p["lamh"] - p["th"]**2*p["lams"])
)
Tn = 100 # only Tn is strictly necessary
print(f"{Tc=}, {Tn=}")

# defining the free energy for WallGo
fxSM = FreeEnergy(f, Tn, params=p)
print("Free energy:", fxSM)

# defining particles which are out of equilibrium for WallGo
top = Particle(
    msqVacuum=lambda X: p["yt"]**2 * np.asanyarray(X)[..., 0]**2,
    msqThermal=lambda T: p["yt"]**2 * T**2,
    statistics="Fermion",
    collisionPrefactors=[p["g"]**4, p["g"]**4, p["g"]**4],
)
particles = [top]
print("top quark:", top)

# grid size
M = 20
N = 20

# now compute the bubble wall speed
# findWallVelocityLoop
